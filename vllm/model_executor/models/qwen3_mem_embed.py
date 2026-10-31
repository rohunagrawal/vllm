# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 with a ReLay memory layer (shared-bank, ungated, ``after_attention``).

Inference-only port of the memory-layers ``qwen3_mem_embed`` reader. At each layer in
``config.mem_layers`` the residual stream is read by a memory block between self-attention and
the MLP::

    h = x + attn(x)
    h = h + mem(h)         # mem(h) is the write: mem_layernorm -> mem_q_proj -> per-head mem_q_norm
                           #   -> top-k over the bank (softmax of dots / sqrt(k_dim))
                           #   -> value read in fp32 -> mem_o_proj (fp32)
    out = h + mlp(norm(h))

The bank (keys already normalized with ``mem_k_norm``, raw values) is built outside vLLM, e.g. by a
PyTorch encoder, and copied in place into fixed-capacity buffers with ``set_memory_bank`` so
torch.compile and CUDA graph replay keep the same tensor addresses.

Config fields (``hf_overrides``): ``mem_layers`` (default [18]), ``mem_num_heads`` (4),
``mem_k_dim`` (1024), ``mem_v_dim`` (1024), ``mem_top_k`` (64), ``mem_capacity`` (bank slots),
``mem_retrieval`` (``"exact"`` dense GEMM + top-k, or ``"cagra"``: cuVS FP16 CAGRA candidates
reranked with the bf16 keys, see ``qwen3_mem_cagra``), ``mem_cagra_itopk`` (1024) and
``mem_cagra_candidates`` (512).

Bank hand-off: CUDA graphs are captured at engine start, so a CAGRA index must exist before that.
Call ``set_pending_memory_bank(keys, values)`` before constructing an in-process ``LLM``; the
model copies it in (and builds the index) at the end of weight loading. ``set_memory_bank`` still
refills the buffers of an exact-retrieval model afterwards, but cannot rebuild a CAGRA index.

Memory projections are not in the base Qwen3 checkpoint: they are randomly initialised
(sigma 0.02, seeded by ``mem_init_seed``) unless the checkpoint provides the trained tensors
(``model.layers.N.mem_layernorm.weight`` and ``model.layers.N.memory.{mem_q_proj,mem_q_norm,mem_o_proj,
mem_k_norm}.weight``, as written by memory-layers ``vllm_export.py`` / ``export_relay.py``); then
``mem_weights_loaded`` is True and ``MemoryBlock.mem_k_norm`` holds the key-norm weight that banks built
outside vLLM are normalised with.

``mem_masked_bank=True`` (exact retrieval only): ``mem_capacity`` is a maximum; the first ``bank_n`` slots are
valid and ``set_memory_bank`` may be called again with fewer slots (in place, CUDA-graph safe), e.g. one bank
per query. Unset (default) keeps the fixed-size path used by the evidence-size sweep unchanged.
"""

import math
import re
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.utils.torch_utils import direct_register_custom_op

from .qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model
from .qwen3_mem_cagra import _BANKS, CagraBank
from .utils import PPMissingLayer, extract_layer_index, maybe_prefix

_QUERY_CHUNK = 1024  # max query rows scored against the bank at a time
_LOGITS_BUDGET = 3 << 29  # ~1.5 GiB of fp32 logits per chunk: rows shrink as the bank grows
_EPS = 1e-6
_MEM_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.(?:memory\.)?(?P<leaf>mem_[a-z_]+)(?:\.weight)?$"
)
_MEM_LOADED_LEAVES = ("mem_q_proj", "mem_q_norm", "mem_o_proj", "mem_k_norm", "mem_layernorm")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    # fp32 arithmetic then cast back, matching the JAX/torch reference reader.
    xf = x.float()
    return (weight.float() * xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    )


def _mem_read(
    q: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor, top_k: int
) -> torch.Tensor:
    """q [R, k_dim] -> fp32 read [R, v_dim]: exact top-k softmax read over the whole bank."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    chunk = max(1, min(_QUERY_CHUNK, _LOGITS_BUDGET // (4 * bank_k.shape[0])))
    outs = []
    for part in q.split(chunk):
        logits = (part @ bank_k.T).float() * scale
        scores, idx = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("qk,qkv->qv", probs, bank_v[idx].float()))
    return torch.cat(outs)


def _mem_read_masked(
    q: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor, top_k: int, bank_n: torch.Tensor
) -> torch.Tensor:
    """As ``_mem_read`` over a bank whose first ``bank_n[0]`` slots are valid (the rest never win)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    chunk = max(1, min(_QUERY_CHUNK, _LOGITS_BUDGET // (4 * bank_k.shape[0])))
    invalid = torch.arange(bank_k.shape[0], device=q.device) >= bank_n
    outs = []
    for part in q.split(chunk):
        logits = (part @ bank_k.T).float() * scale
        scores, idx = torch.topk(logits.masked_fill(invalid, float("-inf")), top_k, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("qk,qkv->qv", probs, bank_v[idx].float()))
    return torch.cat(outs)


def _mem_read_masked_fake(
    q: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor, top_k: int, bank_n: torch.Tensor
) -> torch.Tensor:
    return torch.empty((q.shape[0], bank_v.shape[-1]), dtype=torch.float32, device=q.device)


direct_register_custom_op(
    op_name="mem_read_masked",
    op_func=_mem_read_masked,
    mutates_args=[],
    fake_impl=_mem_read_masked_fake,
)


def _mem_read_fake(
    q: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor, top_k: int
) -> torch.Tensor:
    return torch.empty((q.shape[0], bank_v.shape[-1]), dtype=torch.float32, device=q.device)


direct_register_custom_op(
    op_name="mem_read",
    op_func=_mem_read,
    mutates_args=[],
    fake_impl=_mem_read_fake,
)

_RERANK_CHUNK = 512  # rows per CAGRA search + rerank (bounds the [rows, candidates, dim] gather)


def _mem_read_cagra(
    q: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor, top_k: int, handle: int
) -> torch.Tensor:
    """CAGRA candidates -> exact bf16 rerank -> top-k softmax read. Same output as ``_mem_read``."""
    bank = _BANKS[handle]
    scale = 1.0 / math.sqrt(q.shape[-1])
    outs = []
    for part in q.split(_RERANK_CHUNK):
        local = bank.search(part)  # [r, candidates] int32, -1 invalid
        valid = local >= 0
        ids = local.clamp_min(0).long()
        candidate_keys = bank_k[ids]
        if bank_k.dtype == torch.int8:
            candidate_keys = (candidate_keys.float() * bank.key_scale[ids]).to(part.dtype)
        dots = torch.bmm(candidate_keys, part.unsqueeze(-1)).squeeze(-1).float() * scale
        scores, which = torch.topk(dots.masked_fill(~valid, float("-inf")), top_k, dim=-1)
        ids = ids.gather(-1, which)
        group = getattr(bank, "group", None)
        if group is not None and group.world_size > 1:
            # Exchange only local top-k scores. The gathered position identifies
            # the owning rank and its local top-k slot; IDs stay on their owner.
            all_scores = group.all_gather(scores, dim=-1)
            global_scores, positions = all_scores.topk(top_k, dim=-1)
            probabilities = torch.softmax(global_scores, dim=-1)
            weights = torch.zeros_like(all_scores).scatter(-1, positions, probabilities)
            probs = weights[:, group.rank_in_group * top_k:(group.rank_in_group + 1) * top_k]
        else:
            probs = torch.softmax(scores, dim=-1)
        selected_values = bank_v[ids].float()
        if bank_v.dtype == torch.int8:
            selected_values = selected_values * bank.value_scale[ids]
        read = torch.einsum("qk,qkv->qv", probs, selected_values)
        if group is not None and group.world_size > 1:
            read = group.all_reduce(read)
        outs.append(read)
    return torch.cat(outs)


def _mem_read_cagra_fake(
    q: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor, top_k: int, handle: int
) -> torch.Tensor:
    return torch.empty((q.shape[0], bank_v.shape[-1]), dtype=torch.float32, device=q.device)


direct_register_custom_op(
    op_name="mem_read_cagra",
    op_func=_mem_read_cagra,
    mutates_args=[],
    fake_impl=_mem_read_cagra_fake,
)

_PENDING_BANK: tuple[torch.Tensor, torch.Tensor] | None = None


def set_pending_memory_bank(keys: torch.Tensor, values: torch.Tensor) -> None:
    """Stage a bank (keys already k-normalised) for the next in-process model load."""
    global _PENDING_BANK
    _PENDING_BANK = (keys, values)


class MemoryBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        k_dim: int,
        v_dim: int,
        top_k: int,
        capacity: int,
        dtype: torch.dtype,
        retrieval: str = "exact",
        cagra_itopk: int = 1024,
        cagra_candidates: int = 512,
        layer_idx: int = -1,
        masked: bool = False,
        bank_int8: bool = False,
        bank_shard: bool = False,
    ) -> None:
        super().__init__()
        assert retrieval in ("exact", "cagra"), retrieval
        self.bank_group = None
        self.global_capacity = capacity
        self.bank_start = 0
        if bank_shard:
            from vllm.distributed import get_tp_group
            if retrieval != "cagra" or masked:
                raise ValueError("Bank sharding requires unmasked CAGRA retrieval")
            self.bank_group = get_tp_group()
            rank, world = self.bank_group.rank_in_group, self.bank_group.world_size
            self.bank_start = capacity * rank // world
            end = capacity * (rank + 1) // world
            capacity = end - self.bank_start
            if capacity < top_k or cagra_candidates < top_k:
                raise ValueError("Each shard and candidate budget must contain at least top_k rows")
        self.bank_int8 = bank_int8
        if bank_int8 and (retrieval != "cagra" or masked):
            raise ValueError("INT8 memory banks require unmasked CAGRA retrieval")
        self.layer_idx = layer_idx
        assert not (masked and retrieval == "cagra"), "a CAGRA bank is built once at full capacity"
        self.masked = masked  # exact only: bank_n valid slots of `capacity`, refillable per query
        self.num_heads, self.k_dim, self.v_dim, self.top_k = num_heads, k_dim, v_dim, top_k
        self.retrieval, self.cagra_itopk, self.cagra_candidates = (
            retrieval,
            cagra_itopk,
            cagra_candidates,
        )
        self.cagra: CagraBank | None = None
        self.cagra_handle = -1  # set by build_cagra before the first (tracing) forward
        self.mem_layernorm = nn.Parameter(torch.ones(hidden_size, dtype=dtype), requires_grad=False)
        self.mem_q_proj = nn.Parameter(
            torch.empty(num_heads * k_dim, hidden_size, dtype=dtype), requires_grad=False
        )
        self.mem_q_norm = nn.Parameter(torch.ones(k_dim, dtype=dtype), requires_grad=False)
        # Key norm weight, used only to normalise banks built outside vLLM (see load_trained).
        self.register_buffer("mem_k_norm", torch.ones(k_dim, dtype=dtype), persistent=False)
        # fp32 output projection [num_heads * v_dim, hidden]: the reference reads in fp32.
        self.mem_o_proj_f32 = nn.Parameter(
            torch.empty(num_heads * v_dim, hidden_size, dtype=torch.float32), requires_grad=False
        )
        # Fixed-capacity bank; contents are replaced in place by set_memory_bank.
        self.capacity = capacity
        self.register_buffer("bank_n", torch.full((1,), capacity, dtype=torch.int64), persistent=False)
        self.register_buffer("bank_k", torch.zeros(0 if bank_int8 else capacity, k_dim, dtype=torch.int8 if bank_int8 else dtype), persistent=False)
        # CAGRA builds its index from the keys only, and cuVS's NN-descent needs a second copy of the
        # dataset while building, so at millions of slots the value bank is allocated *after* the index
        # exists (load_bank). Exact retrieval has no such peak and keeps the buffer from the start.
        self.register_buffer(
            "bank_v",
            torch.zeros(capacity if retrieval == "exact" else 0, v_dim, dtype=dtype),
            persistent=False,
        )

    def init_random(self, seed: int) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        for p in (self.mem_q_proj, self.mem_o_proj_f32):
            p.data.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).mul_(0.02))

    def load_trained(self, t: dict[str, torch.Tensor]) -> None:
        """Trained export tensors: q_proj [H*k, D], q_norm [k], o_proj [D, H*v], layernorm [D], k_norm [k]."""
        self.mem_q_proj.data.copy_(t["mem_q_proj"])
        self.mem_q_norm.data.copy_(t["mem_q_norm"])
        self.mem_layernorm.data.copy_(t["mem_layernorm"])
        self.mem_o_proj_f32.data.copy_(t["mem_o_proj"].float().T)  # [H*v, D], fp32 like the reference
        if "mem_k_norm" in t:
            self.mem_k_norm.copy_(t["mem_k_norm"])

    def load_bank(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Fill the bank in place (keys already k-normalised); CAGRA blocks build their index here."""
        if self.bank_group is not None:
            if keys.shape[0] == self.global_capacity:
                sl = slice(self.bank_start, self.bank_start + self.capacity)
                keys, values = keys[sl], values[sl]
            elif keys.shape[0] != self.capacity:
                raise ValueError("Expected a full bank or this rank's local shard")
        n = keys.shape[0]
        assert n == values.shape[0], "keys/values slot counts differ"
        if self.masked:
            assert 0 < n <= self.capacity, f"bank has {n} slots, mem_capacity is {self.capacity}"
            self.bank_k[:n].copy_(keys)
            self.bank_v[:n].copy_(values)
            self.bank_n.fill_(n)  # in place: captured CUDA graphs read the new count
            return
        assert n == self.capacity, f"bank has {n} slots, mem_capacity is {self.capacity}"
        if self.bank_int8:
            if self.cagra is not None:
                raise ValueError("An INT8 CAGRA bank cannot be refilled")
            for name, source in (("k", keys), ("v", values)):
                target = torch.empty(source.shape, dtype=torch.int8, device=self.bank_k.device)
                scales = torch.empty((n, 1), dtype=torch.float32, device=target.device)
                for start in range(0, n, 8192):
                    rows = source[start:start + 8192].to(device=target.device, dtype=torch.float32)
                    scale = (rows.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
                    target[start:start + 8192] = (rows / scale).round().clamp(-127, 127).to(torch.int8)
                    scales[start:start + 8192] = scale
                setattr(self, "bank_" + name, target)
                self.register_buffer("bank_" + name + "_scale", scales, persistent=False)
                del rows, scale
                if name == "k":
                    self.build_cagra()
            self.cagra.key_scale = self.bank_k_scale
            self.cagra.value_scale = self.bank_v_scale
            return
        self.bank_k.copy_(keys)  # direct cross-device copy: no full-size temporary on the GPU
        if self.retrieval == "cagra":
            assert self.cagra is None, "a built CAGRA index cannot be refilled in place"
            torch.cuda.empty_cache()  # give the cached blocks back: the build needs the headroom
            self.build_cagra()
            torch.cuda.empty_cache()
        if self.bank_v.shape[0] != self.capacity:
            self.bank_v = torch.empty(
                self.capacity, self.v_dim, dtype=self.bank_k.dtype, device=self.bank_k.device
            )
        self.bank_v.copy_(values)

    def build_cagra(self) -> None:
        self.cagra = CagraBank(
            self.bank_k, itopk=self.cagra_itopk, candidates=self.cagra_candidates,
            key_scale=getattr(self, "bank_k_scale", None), group=self.bank_group
        )
        self.cagra.group = self.bank_group
        self.cagra_handle = self.cagra.handle

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [T, D] (full residual stream) -> the memory write (added to h by the caller)."""
        t = h.shape[0]
        q = _rms_norm(h, self.mem_layernorm) @ self.mem_q_proj.T
        q = _rms_norm(q.view(t * self.num_heads, self.k_dim), self.mem_q_norm)
        if self.retrieval == "cagra":
            y = torch.ops.vllm.mem_read_cagra(
                q, self.bank_k, self.bank_v, self.top_k, self.cagra_handle
            )
        elif self.masked:
            y = torch.ops.vllm.mem_read_masked(q, self.bank_k, self.bank_v, self.top_k, self.bank_n)
        else:
            y = torch.ops.vllm.mem_read(q, self.bank_k, self.bank_v, self.top_k)
        out = y.view(t, self.num_heads * self.v_dim) @ self.mem_o_proj_f32
        return out.to(h.dtype)


class Qwen3MemEmbedDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(config, cache_config, quant_config, prefix, **kwargs)
        self.memory = None
        if extract_layer_index(prefix) in getattr(config, "mem_layers", [18]):
            self.memory = MemoryBlock(
                hidden_size=config.hidden_size,
                num_heads=getattr(config, "mem_num_heads", 4),
                k_dim=getattr(config, "mem_k_dim", 1024),
                v_dim=getattr(config, "mem_v_dim", 1024),
                top_k=getattr(config, "mem_top_k", 64),
                capacity=getattr(config, "mem_capacity", 131072),
                dtype=torch.get_default_dtype(),
                retrieval=getattr(config, "mem_retrieval", "exact"),
                cagra_itopk=getattr(config, "mem_cagra_itopk", 1024),
                cagra_candidates=getattr(config, "mem_cagra_candidates", 512),
                layer_idx=extract_layer_index(prefix),
                masked=getattr(config, "mem_masked_bank", False),
                bank_int8=getattr(config, "mem_bank_int8", False),
                bank_shard=getattr(config, "mem_bank_shard", False),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        if self.memory is not None:
            # Memory reads the true residual stream h = attn_out + residual; the fused
            # post-attention norm then adds the write once: residual = write + h.
            h = hidden_states + residual
            hidden_states, residual = self.post_attention_layernorm(self.memory(h), h)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3MemEmbedModel(Qwen3Model):
    decoder_layer_type = Qwen3MemEmbedDecoderLayer


class Qwen3MemEmbedForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen3MemEmbedModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if not self.memory_blocks():
            raise RuntimeError(
                "No memory blocks were constructed. Check the Qwen3 decoder-layer "
                "factory and mem_layers configuration before benchmarking."
            )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def memory_blocks(self) -> list[MemoryBlock]:
        return [m for m in self.modules() if isinstance(m, MemoryBlock)]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Trained exports carry the memory tensors next to the Qwen3 ones (names as written by the
        # memory-layers ``vllm_export.py``): ``model.layers.N.mem_layernorm.weight`` and
        # ``model.layers.N.memory.{mem_q_proj,mem_q_norm,mem_o_proj,mem_k_norm}.weight``. The bank
        # tensors (``memory.mem_k`` / ``memory.mem_v``) and the unused ``mem_layer_scale`` /
        # ``mem_o_norm`` are skipped: banks are handed over with set_pending_memory_bank.
        mem_tensors: dict[tuple[int, str], torch.Tensor] = {}

        def split(it):
            for name, w in it:
                m = _MEM_WEIGHT_RE.match(name)
                if m is None:
                    yield name, w
                    continue
                leaf = m.group("leaf")
                if leaf in _MEM_LOADED_LEAVES:
                    mem_tensors[(int(m.group("layer")), leaf)] = w

        loaded = super().load_weights(split(weights))
        seed = getattr(self.config, "mem_init_seed", 0)
        self.mem_weights_loaded = False
        for i, block in enumerate(self.memory_blocks()):
            layer = block.layer_idx
            have = {leaf: mem_tensors[(layer, leaf)] for (l, leaf) in mem_tensors if l == layer}
            if {"mem_q_proj", "mem_q_norm", "mem_o_proj", "mem_layernorm"} <= set(have):
                block.load_trained(have)
                self.mem_weights_loaded = True
            else:
                assert not have, f"layer {layer}: incomplete memory tensors {sorted(have)}"
                block.init_random(seed + i)
        for name, _ in self.named_parameters():
            if ".memory." in name:
                loaded.add(name)
        global _PENDING_BANK
        bank_path = getattr(self.config, "mem_bank_path", None)
        if bank_path is not None:
            # mmap keeps each worker's CPU/GPU loading bounded to its own rows.
            payload = torch.load(bank_path, map_location="cpu", mmap=True, weights_only=True)
            self.set_memory_bank(payload["keys"], payload["values"])
        elif _PENDING_BANK is not None:
            self.set_memory_bank(*_PENDING_BANK)
            _PENDING_BANK = None
        elif any(b.retrieval == "cagra" for b in self.memory_blocks()):
            raise RuntimeError(
                "mem_retrieval='cagra' needs the bank before CUDA graph capture: call "
                "set_pending_memory_bank(keys, values) before constructing the LLM"
            )
        return loaded

    def set_memory_bank(self, keys: torch.Tensor, values: torch.Tensor) -> int:
        """Copy a bank (keys already k-normalised) into every memory block, in place.

        The bank must have exactly ``mem_capacity`` slots. For ``mem_retrieval="cagra"`` this also
        builds the index, so it can be called once (before CUDA graph capture), not to refill.
        """
        assert keys.shape[0] == values.shape[0] > 0
        for block in self.memory_blocks():
            block.load_bank(keys, values)
        return keys.shape[0]

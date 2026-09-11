# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Qwen3 model with memory-embedding layers."""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    maybe_prefix,
)


def chunked_memory_lookup(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    top_k: int,
    chunk_size: int,
    temperature: float = 1.0,
    activation: str = "softmax",
    phantom_log_n: float = 0.0,
) -> torch.Tensor:
    """Read a shared memory bank without materializing all retrieval scores."""
    query_shape = query.shape
    retrieval_dim = query_shape[-1]
    flat_query = query.reshape(-1, retrieval_dim)
    top_k = min(top_k, keys.shape[0])

    best_scores = flat_query.new_empty((flat_query.shape[0], 0))
    best_indices = torch.empty(
        (flat_query.shape[0], 0), dtype=torch.long, device=query.device
    )
    scale = retrieval_dim**-0.5

    for start in range(0, keys.shape[0], chunk_size):
        chunk = keys[start : start + chunk_size]
        scores = torch.matmul(flat_query, chunk.t()) * scale
        chunk_k = min(top_k, chunk.shape[0])
        scores, indices = torch.topk(scores, chunk_k, dim=-1, sorted=False)
        indices = indices + start

        scores = torch.cat((best_scores, scores), dim=-1)
        indices = torch.cat((best_indices, indices), dim=-1)
        merge_k = min(top_k, scores.shape[-1])
        best_scores, selected = torch.topk(scores, merge_k, dim=-1, sorted=False)
        best_indices = torch.gather(indices, -1, selected)

    scaled_scores = best_scores / temperature
    if activation == "relu":
        weights = torch.relu(scaled_scores)
    elif activation == "sigmoid":
        weights = torch.sigmoid(scaled_scores)
    elif phantom_log_n > 0.0:
        maximum = scaled_scores.amax(dim=-1, keepdim=True).detach()
        exponentials = torch.exp(scaled_scores - maximum)
        background = torch.exp(scaled_scores.new_tensor(phantom_log_n) - maximum)
        weights = exponentials / (exponentials.sum(dim=-1, keepdim=True) + background)
    else:
        weights = torch.softmax(scaled_scores, dim=-1)

    selected_values = values[best_indices]
    output = torch.sum(weights.unsqueeze(-1) * selected_values, dim=-2)
    return output.reshape(*query_shape[:-1], values.shape[-1])


class Qwen3MemoryLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.mem_num_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError("mem_num_heads must be divisible by tensor_parallel_size")
        self.num_heads = self.total_num_heads // tp_size
        self.key_dim = config.mem_k_dim
        self.value_dim = config.mem_v_dim
        self.top_k = config.mem_top_k
        self.chunk_size = getattr(config, "mem_lookup_chunk_size", None) or 65536
        self.temperature = getattr(config, "mem_softmax_temp", 1.0)
        self.activation = getattr(config, "mem_score_activation", "softmax")
        self.phantom_log_n = getattr(config, "mem_phantom_log_n", 0.0)
        self.keys_are_normalized = getattr(config, "mem_k_prenormed", False)
        self.rms_norm_eps = config.rms_norm_eps

        if self.activation not in ("softmax", "relu", "sigmoid"):
            raise ValueError(f"Unsupported memory score activation: {self.activation}")

        self.mem_q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mem_q_proj",
        )
        self.mem_o_proj = RowParallelLinear(
            self.total_num_heads * self.value_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mem_o_proj",
        )
        self.mem_q_norm = RMSNorm(self.key_dim, eps=config.rms_norm_eps)
        self.mem_k_norm = RMSNorm(self.key_dim, eps=config.rms_norm_eps)
        self.mem_o_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mem_layer_scale = nn.Parameter(torch.empty(()), requires_grad=False)
        self.mem_k = nn.Parameter(
            torch.empty(config.mem_size, self.key_dim), requires_grad=False
        )
        self.mem_v = nn.Parameter(
            torch.empty(config.mem_size, self.value_dim), requires_grad=False
        )

    def normalize_keys(self) -> None:
        if self.keys_are_normalized:
            return
        with torch.no_grad():
            key_float = self.mem_k.float()
            inverse_rms = torch.rsqrt(
                key_float.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
            )
            normalized = key_float * inverse_rms
            normalized = normalized * self.mem_k_norm.weight.float()
            self.mem_k.copy_(normalized.to(self.mem_k.dtype))
        self.keys_are_normalized = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, _ = self.mem_q_proj(hidden_states)
        query = query.view(-1, self.num_heads, self.key_dim)
        query = self.mem_q_norm(query)
        memory = chunked_memory_lookup(
            query,
            self.mem_k,
            self.mem_v,
            self.top_k,
            self.chunk_size,
            self.temperature,
            self.activation,
            self.phantom_log_n,
        )
        output, _ = self.mem_o_proj(memory.flatten(-2))
        return output


class Qwen3MemEmbedDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__(
            config,
            cache_config,
            quant_config,
            prefix,
            per_layer_sliding_window,
        )
        layer_index = extract_layer_index(prefix)
        self.has_memory = layer_index in config.mem_layers
        if self.has_memory:
            self.mem_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.memory = Qwen3MemoryLayer(config, quant_config, prefix)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_memory:
            return super().forward(positions, hidden_states, residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        memory_input, residual = self.mem_layernorm(hidden_states, residual)
        hidden_states = self.memory(memory_input)
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
class Qwen3MemEmbedModel(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        if getattr(config, "mem_placement", "after_attention") != "after_attention":
            raise ValueError("Only mem_placement='after_attention' is supported")
        unsupported = {
            "mem_use_gating": getattr(config, "mem_use_gating", False),
            "mem_use_product_keys": getattr(config, "mem_use_product_keys", False),
            "span_window": getattr(config, "span_window", 0),
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(f"Unsupported memory options: {', '.join(enabled)}")
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MemEmbedDecoderLayer,
        )


class Qwen3MemEmbedForCausalLM(Qwen3ForCausalLM):
    hf_to_vllm_mapper = Qwen3MemEmbedModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3MemEmbedModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        for layer in self.model.layers:
            if isinstance(layer, Qwen3MemEmbedDecoderLayer) and layer.has_memory:
                layer.memory.normalize_keys()
        return loaded

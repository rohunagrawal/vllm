# ReLay-LM compiled hybrid support on vLLM 0.11.0

This fork is based on upstream v0.11.0 (`b8b302cde434df8c9289a2b465406b47ebab1c2d`).
It ports the existing Qwen3 memory embedding architecture from the
`rohunagrawal/vllm` `qwen3-mem-embed` branch (`8fde3f99915489404d1a19740af668863a66dd96`)
while making per-question bank selection valid under torch.compile and CUDA graphs.

The exported full corpus is immutable. Each memory layer owns fixed-capacity key,
value and validity tensors allocated before compilation. `set_memory_active_indices`
updates their contents in place. The forward pass reads those buffers unconditionally;
it never branches on a changing Python attribute or captures a changing tensor address.
Invalid slots are masked before top-k. Fewer-than-k selections work, and zero slots
produce exactly zero memory output. `mem_active_capacity` must cover the largest
selection in the run, and changing it requires a new engine. This intentionally supports
serialized offline evaluation, not requests with different banks batched together.

Requirements: `max_num_seqs=1`, prefix caching disabled (cached hidden states depend on
the bank), explicit active indices before every generation, and single-GPU TP=1 for the
paper experiment. Exact top-k is used, unlike the JAX checkpoint's approximate top-k;
this is an explicit numerical recipe difference, not a change to learned weights.

The MLP LoRA adapters must be merged by the checkpoint exporter. Memory `mem_o_norm`
and `mem_layer_scale` parameters are loaded for checkpoint compatibility but are not
applied, matching the current JAX `models/memory.py::memory_layer` implementation.
Unsupported placement/gating/product keys/span read options raise errors.

No C++ or CUDA kernels were modified. The parent repository's
`scripts/gpu_vllm/compiled_bind_wheel.py` binds the matching vLLM 0.11.0 wheel's binary
extensions and bundled FlashAttention files into this source checkout. Set PYTHONPATH
to this checkout only for hybrid processes, leaving the baseline runtime unchanged.
Use Triton's bundled CUDA 12 assembler; the node's global CUDA 13 assembler is incompatible.

Validation (H100, 2026-10-07): parent repository
`tests/test_compiled_hybrid_memory.py --cuda` passes exact top-k reference, masks,
empty bank, fullgraph compilation and CUDA graph state-update checks.
`scripts/gpu_vllm/compiled_smoke.py` passes full vLLM eager and compiled engines with
banks A/B/empty/one/A: repeated A gives identical within-mode logprobs; changed banks
change logprobs; all five greedy token sequences agree between modes. Across 287 shared
logprobs the largest compiled/eager absolute difference was 0.003976, consistent with
BF16 fused-operation rounding. This synthetic smoke does not establish trained-checkpoint
parity; a trained-model probe is additionally required before paper runs.

Additional trained validation (same H100): actual step900 reader A/B/empty/one/A
passes in eager and compiled engines; all5 greedy token sequences match, with max
absolute difference0.135401 across287 shared logprobs. A direct learned-memory
projection/read/write test on2048 actual corpus slots against CPUJAX gives write
relativeL2=0.009430 and cosine=0.999956. BF16 projection/RMS arithmetic differs
slightly between frameworks. Weighted BF16 products now accumulate in FP32 before
casting output, matching JAX einsum rather than rounding individual products.
Worker telemetry exposes corpus/active buffers, reader parameters, KV cache and
allocator peaks so a shared GPU reservation budget does not obscure actual footprint.

# Qwen3 memory-embedding inference

`Qwen3MemEmbedForCausalLM` keeps the standard vLLM Qwen3 attention, KV cache,
continuous batching, and tensor-parallel linears. At configured decoder layers it
adds a shared-bank top-k memory read after self-attention and before the MLP.

The Hugging Face checkpoint must include these extra Qwen3 config fields:
`mem_layers`, `mem_size`, `mem_num_heads`, `mem_k_dim`, `mem_v_dim`, and
`mem_top_k`. Each configured layer contains its learned projections and its static
corpus bank under `model.layers.<n>.memory.*`. Keys are normalized once after
loading unless `mem_k_prenormed` is true. Retrieval scans the bank in
`mem_lookup_chunk_size` chunks, so prefill does not materialize the full score
matrix.

For a loading and generation smoke test, create a checkpoint whose zero memory
output projection leaves the base model unchanged:

```bash
.venv/bin/python \
  examples/offline_inference/qwen3_mem_embed/make_smoke_checkpoint.py \
  --base-model ~/weights/huggingface/Qwen/Qwen3-4B \
  --output-dir ~/weights/qwen3-mem-embed-smoke

.venv/bin/python -c '
from vllm import LLM, SamplingParams
llm = LLM(model="~/weights/qwen3-mem-embed-smoke", tensor_parallel_size=2)
print(llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=8))[0].outputs[0].text)
'
```

The memory-layers repository's `VLLM_EXPORT_DIR` integration exports a trained
checkpoint and its corpus bank from the native JAX large-memory evaluator. Use
exact top-k (`MEM_APPROX_TOPK=0`) and greedy decoding for the closest numerical
comparison.

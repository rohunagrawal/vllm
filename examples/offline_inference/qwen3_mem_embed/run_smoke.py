"""Run a two-GPU smoke test for a Qwen3 memory-embedding export.

On machines without a system CUDA toolkit, point ``CUDA_HOME`` at the CUDA
toolkit bundled in the vLLM environment before running this script.
"""

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="/mnt/home/ragrawal/weights/qwen3-mem-embed-smoke",
        tensor_parallel_size=2,
        max_model_len=512,
        max_num_seqs=8,
        gpu_memory_utilization=0.25,
    )
    outputs = llm.generate(
        ["The capital of France is"], SamplingParams(temperature=0, max_tokens=8)
    )
    print("COMPILED_SMOKE_OUTPUT", repr(outputs[0].outputs[0].text), flush=True)


if __name__ == "__main__":
    main()

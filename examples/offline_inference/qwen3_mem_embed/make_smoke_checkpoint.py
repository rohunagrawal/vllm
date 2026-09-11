"""Create a Qwen3MemEmbed loading smoke test from a base Qwen3 checkpoint."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.base_model.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text())
    layer = 18
    num_heads = 4
    key_dim = 1024
    value_dim = 1024
    bank_size = 64
    config.update(
        {
            "architectures": ["Qwen3MemEmbedForCausalLM"],
            "mem_layers": [layer],
            "mem_size": bank_size,
            "mem_num_heads": num_heads,
            "mem_k_dim": key_dim,
            "mem_v_dim": value_dim,
            "mem_top_k": 8,
            "mem_lookup_chunk_size": 32,
            "mem_k_prenormed": True,
            "mem_placement": "after_attention",
            "mem_score_activation": "softmax",
            "mem_softmax_temp": 1.0,
            "mem_phantom_log_n": 0.0,
            "mem_use_gating": False,
            "mem_use_product_keys": False,
            "span_window": 0,
        }
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for path in source.iterdir():
        if not path.is_file() or path.name in {
            "config.json",
            "model.safetensors.index.json",
        }:
            continue
        destination = output / path.name
        if path.suffix == ".safetensors":
            if not destination.exists():
                os.link(path, destination)
        else:
            shutil.copy2(path, destination)

    hidden_size = config["hidden_size"]
    generator = torch.Generator().manual_seed(42)
    prefix = f"model.layers.{layer}"
    memory = {
        f"{prefix}.mem_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        f"{prefix}.memory.mem_q_proj.weight": torch.randn(
            num_heads * key_dim,
            hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02,
        f"{prefix}.memory.mem_o_proj.weight": torch.zeros(
            hidden_size, num_heads * value_dim, dtype=torch.bfloat16
        ),
        f"{prefix}.memory.mem_q_norm.weight": torch.ones(key_dim, dtype=torch.bfloat16),
        f"{prefix}.memory.mem_k_norm.weight": torch.ones(key_dim, dtype=torch.bfloat16),
        f"{prefix}.memory.mem_o_norm.weight": torch.ones(
            hidden_size, dtype=torch.bfloat16
        ),
        f"{prefix}.memory.mem_layer_scale": torch.tensor(0.1, dtype=torch.bfloat16),
        f"{prefix}.memory.mem_k": torch.randn(
            bank_size, key_dim, dtype=torch.bfloat16, generator=generator
        ),
        f"{prefix}.memory.mem_v": torch.randn(
            bank_size, value_dim, dtype=torch.bfloat16, generator=generator
        ),
    }
    memory_filename = "memory.safetensors"
    save_file(memory, output / memory_filename)

    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"].update({name: memory_filename for name in memory})
    index["metadata"]["total_size"] += sum(
        tensor.numel() * tensor.element_size() for tensor in memory.values()
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()

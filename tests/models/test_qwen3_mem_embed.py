# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.qwen3_mem_embed import chunked_memory_lookup


@pytest.mark.parametrize("chunk_size", [3, 7, 32])
def test_chunked_memory_lookup_matches_dense(chunk_size: int) -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(5, 2, 8, generator=generator)
    keys = torch.randn(19, 8, generator=generator)
    values = torch.randn(19, 6, generator=generator)
    top_k = 4

    scores = torch.einsum("thd,md->thm", query, keys) * query.shape[-1] ** -0.5
    top_scores, top_indices = torch.topk(scores, top_k, dim=-1)
    expected = torch.sum(
        torch.softmax(top_scores, dim=-1).unsqueeze(-1) * values[top_indices],
        dim=-2,
    )

    actual = chunked_memory_lookup(
        query, keys, values, top_k=top_k, chunk_size=chunk_size
    )

    torch.testing.assert_close(actual, expected)


def test_chunked_memory_lookup_caps_top_k_at_bank_size() -> None:
    query = torch.tensor([[[1.0, 0.0]]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[3.0], [7.0]])

    actual = chunked_memory_lookup(query, keys, values, top_k=8, chunk_size=1)
    scores = torch.tensor([1.0, 0.0]) / (2.0**0.5)
    expected = torch.sum(torch.softmax(scores, dim=0) * values[:, 0])

    torch.testing.assert_close(actual.squeeze(), expected)

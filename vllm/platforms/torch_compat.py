# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator compatibility for the ReLay evaluation runtime on PyTorch 2.8."""

import torch


def empty_cache() -> None:
    """Release unused allocator blocks on current and older PyTorch versions."""
    if hasattr(torch.accelerator, "empty_cache"):
        torch.accelerator.empty_cache()
    else:
        torch.cuda.empty_cache()

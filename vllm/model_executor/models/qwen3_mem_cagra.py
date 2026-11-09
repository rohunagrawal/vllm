# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuVS CAGRA retrieval backend for the Qwen3MemEmbed memory bank.

``CagraBank`` indexes the (already k-normalised) bank keys and returns candidate ids for
a batch
of queries. The memory op then reranks the candidates with the original bf16 keys, takes
the exact
top-k of those and does the usual softmax value read (see
``qwen3_mem_embed._mem_read_cagra``).

Index: FP32 MIPS-to-L2 augmentation (one extra coordinate making all keys equal-norm,
plus 128
zero padding dims) cast to FP16, NN-descent build, graph degree 64 / intermediate 128,
``multi_cta`` search, search width 1, itopk 1024, automatic iteration count.

CUDA graph safety (this is what makes it usable inside vLLM's captured decode steps):

* cuVS (26.x) initialises a dataset descriptor on the first stream that searches and
then issues
  ``cudaStreamWaitEvent`` on a ready-event whenever a *different* stream searches, which
  is illegal
  during capture. vLLM captures on a fresh stream, so every search runs on one private
  side
  stream bound to the cuVS resources once. It is forked from / joined to the caller's
  stream with
  stream waits, which is legal inside a capture.
* Output buffers are torch tensors (allocated in the graph's pool during capture); CuPy
only sees
  raw-pointer views of them. No DLPack call or CuPy allocation happens during capture,
  since both
  insert stream waits on uncaptured events.
* cuVS scratch comes from RMM's stream-ordered ``cudaMallocAsync`` resource with
``release_threshold=0``:
  capturable (allocations become graph memory nodes) and returned to the driver as soon
  as the build/search
  frees it (RMM's default threshold is the whole device, which keeps the scratch
  resident). (A pool
  would also work for capture, but it never shrinks: at 8M keys it kept 26 GiB of NN-
  descent scratch.)

Neighbors are written as uint32 and read as int32: ids are below 2**31 and the invalid-
neighbor
sentinel 0xFFFFFFFF reads as -1.
"""

import ctypes
import glob
import math
import os
import time
from pathlib import Path

import torch

_BANKS: dict[
    int, "CagraBank"
] = {}  # handle -> bank; the custom op receives the integer handle


def _cview(t: torch.Tensor, dtype):
    """Zero-copy CuPy view over a torch tensor's memory (no DLPack, no stream sync)."""
    import cupy as cp

    mem = cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * t.element_size(), t)
    return cp.ndarray(tuple(t.shape), dtype, cp.cuda.MemoryPointer(mem, 0))


def _stream_setter():
    import cuvs

    paths = glob.glob(
        str(Path(cuvs.__file__).parent.parent / "**/libcuvs_c.so"), recursive=True
    )
    setter = ctypes.CDLL(paths[0]).cuvsStreamSet
    setter.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
    setter.restype = ctypes.c_int
    return setter


class CagraBank:
    key_scale: torch.Tensor
    value_scale: torch.Tensor
    group: object
    softmax_temperature: float

    def __init__(
        self,
        keys: torch.Tensor,
        *,
        itopk: int = 1024,
        candidates: int = 512,
        graph_degree: int = 64,
        intermediate_graph_degree: int = 128,
        max_iterations: int = 0,
        key_scale: torch.Tensor | None = None,
        group=None,
        valid_rows: int | None = None,
    ) -> None:
        if os.environ.get("RELAY_MEMORY_SEARCH") == "dense_exact":
            from scripts.benchmarks.paper_rerun.ruler_exact_search import initialize

            initialize(self, keys, key_scale, candidates)
            self.handle = len(_BANKS)
            _BANKS[self.handle] = self
            return
        self.backend = "cagra"
        import cupy as cp
        import rmm
        from cuvs.common import Resources
        from cuvs.neighbors import cagra

        assert keys.is_cuda and keys.ndim == 2
        self.n, self.dim = keys.shape
        if valid_rows is not None:
            assert 0 < valid_rows <= self.n
            self.n = valid_rows
        self.candidates = min(candidates, self.n)
        self._cagra = cagra
        self._cp = cp
        # release_threshold=0: RMM's default keeps freed memory in the driver
        # pool (threshold = device size),
        # which retained ~20 GiB of NN-descent scratch after the build at 8M keys.
        rmm.mr.set_current_device_resource(
            rmm.mr.CudaAsyncMemoryResource(release_threshold=0)
        )
        self.stream = torch.cuda.Stream()  # the only stream cuVS ever runs on
        self.resources = Resources()
        if _stream_setter()(self.resources.get_c_obj(), self.stream.cuda_stream) != 1:
            raise RuntimeError("cuVS failed to bind the private CUDA stream")

        # FP16 augmented dataset [n, dim + 128] = [key | sqrt(radius -
        # |key|^2) | 127 zeros], built in
        # chunks so no full-size FP32 copy of the bank is ever materialised
        # (matters at millions of keys).
        self.quantized = (
            key_scale is not None
            and os.environ.get("RELAY_CAGRA_DATASET_DTYPE", "int8") != "fp16"
        )
        self.norm_dims = int(os.environ.get("RELAY_CAGRA_NORM_DIMS", "1"))
        assert self.norm_dims in (1, 128)
        step = 8192

        def rows(i):
            end = min(i + step, self.n)
            result = keys[i:end].float()
            return result if key_scale is None else result * key_scale[i:end]

        max_norm = torch.zeros((), device=keys.device)
        max_abs = torch.zeros((), device=keys.device)
        for i in range(0, self.n, step):
            current_rows = rows(i)
            max_norm = torch.maximum(max_norm, current_rows.square().sum(-1).max())
            max_abs = torch.maximum(max_abs, current_rows.abs().max())
        if group is not None and group.world_size > 1:
            torch.distributed.all_reduce(
                max_norm, op=torch.distributed.ReduceOp.MAX, group=group.device_group
            )
        if group is not None and group.world_size > 1:
            torch.distributed.all_reduce(
                max_abs, op=torch.distributed.ReduceOp.MAX, group=group.device_group
            )
        radius = max_norm + 1e-3
        # One shared scale preserves the augmented L2 geometry. Query clipping
        # is bounded by this same radius and must be evaluated for quality later.
        self.search_scale = radius.sqrt().clamp_min(1e-12) / 127
        if self.norm_dims == 128:
            self.search_scale = (
                torch.maximum(max_abs, radius.sqrt() / math.sqrt(128)).clamp_min(1e-12)
                / 127
            )
        dtype = torch.int8 if self.quantized else torch.float16
        self.dataset = torch.zeros(
            (self.n, self.dim + 128), dtype=dtype, device=keys.device
        )
        for i in range(0, self.n, step):
            kf = rows(i)
            aug = torch.zeros((len(kf), self.dim + 128), device=keys.device)
            aug[:, : self.dim] = kf
            norm_coordinate = (radius - kf.square().sum(-1)).clamp_min(0).sqrt()
            if self.norm_dims == 1:
                aug[:, self.dim] = norm_coordinate
            else:
                aug[:, self.dim :] = norm_coordinate[:, None] / math.sqrt(128)
            if self.quantized:
                aug = (aug / self.search_scale).round().clamp(-127, 127)
            self.dataset[i : i + step] = aug.to(dtype)
        del kf, aug

        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        with (
            torch.cuda.stream(self.stream),
            cp.cuda.ExternalStream(self.stream.cuda_stream),
        ):

            def build_index():
                return cagra.build(
                    cagra.IndexParams(
                        metric="sqeuclidean",
                        graph_degree=graph_degree,
                        intermediate_graph_degree=intermediate_graph_degree,
                        build_algo="nn_descent",
                    ),
                    _cview(self.dataset, cp.int8 if self.quantized else cp.float16),
                    resources=self.resources,
                )

            if os.environ.get("RELAY_CAGRA_BUILD_OFFLOAD_KEYS") == "1":
                from scripts.benchmarks.paper_rerun.ruler_cagra_build import (
                    build_with_offloaded_keys,
                )

                self.index = build_with_offloaded_keys(self, keys, build_index)
            else:
                self.index = build_index()
        self.resources.sync()
        torch.accelerator.synchronize()
        self.build_s = time.perf_counter() - t0
        self.params = cagra.SearchParams(
            algo="multi_cta",
            itopk_size=itopk,
            search_width=1,
            max_iterations=max_iterations,
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handle = len(_BANKS)
        _BANKS[self.handle] = self

    def search(self, q: torch.Tensor) -> torch.Tensor:
        """Q [R, dim] (any float dtype) -> int32 candidate ids [R, candidates]; -1 marks
        invalid.
        """
        if getattr(self, "exact_search", False):
            from scripts.benchmarks.paper_rerun.ruler_exact_search import search

            return search(self, q)
        cp = self._cp
        rows = q.shape[0]
        if (
            rows not in self._buffers
        ):  # torch-owned: lands in the graph pool during capture
            self._buffers[rows] = (
                torch.empty(
                    rows, self.candidates, device=q.device, dtype=torch.float32
                ),
                torch.empty(rows, self.candidates, device=q.device, dtype=torch.int32),
            )
        dist_t, nbr_t = self._buffers[rows]
        aug = torch.cat([q.float(), torch.zeros((rows, 128), device=q.device)], -1)
        if self.quantized:
            aug = (aug / self.search_scale).round().clamp(-127, 127).to(torch.int8)
        else:
            aug = aug.half()
        aug = aug.contiguous()
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)  # fork
        aug.record_stream(self.stream)
        with (
            torch.cuda.stream(self.stream),
            cp.cuda.ExternalStream(self.stream.cuda_stream),
        ):
            self._cagra.search(
                self.params,
                self.index,
                _cview(aug, cp.int8 if self.quantized else cp.float16),
                self.candidates,
                distances=_cview(dist_t, cp.float32),
                neighbors=_cview(nbr_t, cp.uint32),
                resources=self.resources,
            )
        cur.wait_stream(self.stream)  # join
        return nbr_t

    def memory_bytes(self) -> dict[str, int]:
        if getattr(self, "exact_search", False):
            return {
                "dataset_bf16_exact": self.dataset.numel() * self.dataset.element_size()
            }
        return {
            "dataset_int8" if self.quantized else "dataset_fp16": self.dataset.numel()
            * self.dataset.element_size()
        }

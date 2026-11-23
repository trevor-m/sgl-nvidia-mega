# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Engram prefetch on all the WKV columns, on CPU: a whole layer is SGLang's own
weight and scale tensors, it never runs the mailbox exchange (whose mailboxes hold
column quarters), and the startup summary says why build chose it."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import runtime
from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

N, K = 4 * ep.SCALE_TILE_ROWS, 64


def _engram():
    wkv = SimpleNamespace(
        weight=torch.zeros((N, K), dtype=torch.float8_e4m3fn),
        weight_scale_inv_swizzled=torch.arange(N * 4, dtype=torch.int32).to(
            torch.uint8
        ),
    )
    return SimpleNamespace(wkv=wkv)


@pytest.mark.parametrize("rank", [0, 3])
def test_a_whole_layer_is_sglangs_tensors(rank):
    e = _engram()
    layer = ep._Layer(e, rank, 4, whole=True)
    assert layer.weight.data_ptr() == e.wkv.weight.data_ptr()
    assert layer.weight.shape == (N, K)
    assert layer.scale is e.wkv.weight_scale_inv_swizzled
    assert layer.cols == slice(0, N) and layer.n == N


def test_a_quarter_is_still_one_ranks_scale_tiles():
    e = _engram()
    layer = ep._Layer(e, 1, 4)
    assert layer.cols == slice(N // 4, N // 2)
    assert torch.equal(layer.scale, e.wkv.weight_scale_inv_swizzled[N : 2 * N])


def test_the_whole_columns_never_run_the_exchange():
    with pytest.raises(ValueError, match="column quarters"):
        ep.EngramPrefetch({}, None, 1, deadline_s=1.0, whole=True)


def test_the_summary_names_the_whole_columns_and_why():
    assert runtime._engram_built(SimpleNamespace()) == "built, checked bit for bit"
    note = runtime._engram_built(
        SimpleNamespace(whole_reason="a WKV column quarter is not SGLang's GEMM")
    )
    assert note.startswith("built on all WKV columns (a WKV column quarter")
    assert note.endswith("checked bit for bit")

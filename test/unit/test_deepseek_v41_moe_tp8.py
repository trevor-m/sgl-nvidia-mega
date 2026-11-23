# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE TP8 on Rubin on CPU: the pair's expert shapes and the counted rewrites of the
TP4 GEMM sources for the 288-wide slice padded to 384. The kernels are checked against
FlashInfer's routed MoE in test/kernels/test_deepseek_v41_moe.py (sm107)."""

import re

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm107.moe import routed_expert_gemm as reg


def _gemm_source(name: str) -> str:
    return (reg.SOURCE_DIR / f"{name}.cu").read_text()


def test_the_expert_shapes_are_moe_tp4_and_tp8():
    intermediate = 2304  # DeepSeek-V4.1-Flash's moe_intermediate_size
    for tp, shape in ((4, reg.EXPERT_SHAPES[640]), (8, reg.EXPERT_SHAPES[384])):
        assert shape.slice == intermediate // tp
        assert shape.padded == -(-shape.slice // 128) * 128
        # w13's 128-row tiles (gate and up interleaved by 64) that hold real rows
        assert shape.gemm1_units == -(-2 * shape.slice // 128)


def test_the_tp8_rewrites_leave_no_tp4_width():
    gemm1 = reg._tp8_gemm1_source(_gemm_source("gemm1"))
    gemm2 = reg._tp8_gemm2_source(_gemm_source("gemm2"))
    for source in (gemm1, gemm2):
        assert not re.search(r"\* 640\b", source)  # the H row stride
        assert not re.search(r"n_tile(_\d)? \* 160\b", source)  # H scale blocks a tile
    assert "if (m_tile == 4) {" in gemm1 and "if (m_tile == 8) {" not in gemm1
    # each GEMM2 role keeps two of its three stage blocks (K 256 + the half stage 128)
    for counter in ("stage", "stage_1", "stage_6"):
        assert gemm2.count(f"if ({counter} == 5) {{") == 2


def test_a_rewrite_that_no_longer_matches_raises():
    with pytest.raises(RuntimeError, match="TP8 rewrite"):
        reg._tp8_gemm1_source(_gemm_source("gemm1").replace("* 640 + 576 +", "* 640 +"))
    with pytest.raises(RuntimeError, match="stage blocks"):
        reg._tp8_gemm2_source("")


def _weights(padded: int, experts: int = 32):
    h = reg.HIDDEN
    return (
        torch.zeros((experts, 2 * padded, h // 2), dtype=torch.uint8),
        torch.zeros((experts, 2 * padded, h // 32), dtype=torch.uint8),
        torch.zeros((experts, h, padded // 2), dtype=torch.uint8),
        torch.zeros((experts, h, padded // 32), dtype=torch.uint8),
        torch.full((experts,), 10.0),
    )


@pytest.mark.parametrize("padded", [640, 384])
def test_check_weights_reads_the_shape_from_w2(padded):
    assert reg.check_weights(*_weights(padded), top_k=1) == reg.EXPERT_SHAPES[padded]


def test_check_weights_refuses_other_widths_and_nonzero_padding():
    with pytest.raises(ValueError, match="padded intermediate"):
        reg.check_weights(*_weights(512), top_k=1)
    w13, s13, w2, s2, clamp = _weights(384)
    w13[0, 128 * 5] = 1  # the sixth w13 tile is padding at MoE TP8
    with pytest.raises(ValueError, match="padding"):
        reg.check_weights(w13, s13, w2, s2, clamp, top_k=1)
    w13[0, 128 * 5] = 0
    w2[1, 0, 288 // 2] = 1  # K past the 288-wide slice
    with pytest.raises(ValueError, match="padding"):
        reg.check_weights(w13, s13, w2, s2, clamp, top_k=1)

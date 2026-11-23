# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The kernel directory each GPU architecture reads (deepseek_v41/arch.py), and that the
two architecture directories never import each other."""

import ast
from pathlib import Path

import pytest

from sgl_nvidia_mega.deepseek_v41 import arch

PACKAGE = Path(arch.__file__).resolve().parent


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(arch, "_chosen", None)
    return arch


@pytest.mark.parametrize(
    "isa, directory",
    [("sm_103", "sm103"), ("sm_107", "sm107"), (None, "sm103")],
)
def test_the_device_isa_picks_the_directory(fresh, monkeypatch, isa, directory):
    monkeypatch.setattr(fresh, "_device_isa", lambda: isa)
    assert fresh.directory() == directory
    # fixed once picked: a later ISA read does not move it
    monkeypatch.setattr(fresh, "_device_isa", lambda: "sm_103")
    assert fresh.directory() == directory


def test_a_handle_resolves_in_the_picked_directory(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "_device_isa", lambda: "sm_107")
    handle = fresh.kernels("attention.wo_a")
    assert handle.__name__ == "sgl_nvidia_mega.deepseek_v41.sm107.attention.wo_a"


def test_writes_through_a_handle_reach_the_module(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "_device_isa", lambda: None)
    handle = fresh.kernels("mxfp8_ll")
    module = fresh.kernel_module("mxfp8_ll")
    monkeypatch.setattr(handle, "MAX_ROWS", 3)
    assert module.MAX_ROWS == 3


def test_use_pins_and_refuses_a_change(fresh):
    fresh.use("sm107")
    assert fresh.directory() == "sm107"
    with pytest.raises(RuntimeError):
        fresh.use("sm103")
    with pytest.raises(ValueError):
        fresh.use("sm120")


@pytest.mark.parametrize("own, other", [("sm103", "sm107"), ("sm107", "sm103")])
def test_the_architecture_directories_never_import_each_other(own, other):
    needle = f"sgl_nvidia_mega.deepseek_v41.{other}"
    for path in (PACKAGE / own).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(needle), (
                    f"{path} imports {node.module}"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(needle), (
                        f"{path} imports {alias.name}"
                    )


def test_the_step_modules_read_kernels_through_arch():
    """Outside sm103/ and sm107/, no module imports an architecture directory."""
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE).parts
        if rel[0] in ("sm103", "sm107"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    (
                        "sgl_nvidia_mega.deepseek_v41.sm103",
                        "sgl_nvidia_mega.deepseek_v41.sm107",
                    )
                ), f"{path} imports {node.module}"

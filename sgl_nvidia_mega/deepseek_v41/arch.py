# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The kernel directory of this GPU: ``sm103`` (B300 / GB300) or ``sm107`` (Rubin).

The step packages (``attention/``, ``moe/``, ``boundary/``, ...) are shared by both
architectures; the kernels and their launchers are not. A step module names a kernel
module by its path inside the architecture directory (``kernels("attention.wo_a")``)
and reads it through :func:`module`, which resolves the directory from the device's
architecture the first time a kernel is used (at build) and never changes it after.

Without a visible GPU (the CPU unit tests, the launcher) the directory is ``sm103``
unless :func:`use` picked one.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py": ("device_isa",),
}

# device ISA -> kernel directory
DIRECTORIES = {"sm_103": "sm103", "sm_107": "sm107"}
DEFAULT = "sm103"
_PACKAGE = "sgl_nvidia_mega.deepseek_v41"

_chosen: str | None = None


def _device_isa() -> str | None:
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import device_isa
    except ImportError:
        return None
    return device_isa()


def directory() -> str:
    """This process's kernel directory, fixed on first call."""
    global _chosen
    if _chosen is None:
        isa = _device_isa()
        _chosen = DIRECTORIES.get(isa, DEFAULT) if isa else DEFAULT
    return _chosen


def use(name: str) -> None:
    """Pin the directory (tests); refused once a kernel module was resolved."""
    global _chosen
    if name not in DIRECTORIES.values():
        raise ValueError(f"{name!r} is not one of {sorted(DIRECTORIES.values())}")
    if _chosen is not None and _chosen != name:
        raise RuntimeError(f"the kernel directory is already {_chosen!r}")
    _chosen = name


def kernel_module(path: str) -> ModuleType:
    """``path`` (e.g. ``"attention.wo_a"``) in this GPU's kernel directory."""
    return importlib.import_module(f"{_PACKAGE}.{directory()}.{path}")


class _Lazy:
    """A kernel module resolved on first attribute access, so step modules can name
    it at import time (before any GPU is visible)."""

    __slots__ = ("_path",)

    def __init__(self, path: str) -> None:
        object.__setattr__(self, "_path", path)

    def __getattr__(self, attr: str):
        return getattr(kernel_module(object.__getattribute__(self, "_path")), attr)

    # writes go to the module too (tests patch kernel functions through a step module)
    def __setattr__(self, attr: str, value) -> None:
        setattr(kernel_module(object.__getattribute__(self, "_path")), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(kernel_module(object.__getattribute__(self, "_path")), attr)

    def __repr__(self) -> str:
        return f"<kernel module {object.__getattribute__(self, '_path')!r} of {directory()}>"


def kernels(path: str) -> _Lazy:
    """A handle on ``path`` in this GPU's kernel directory, resolved when first read."""
    return _Lazy(path)

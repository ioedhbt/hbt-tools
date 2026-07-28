"""
models/tuning/_cuda_env.py — Shared CUDA detection (runtime, zero-cost when
CuPy is absent), imported by every tuning/ module that needs to pick a CPU
vs. GPU compute path.

Split out of models/base_ui.py's module preamble, which every tuning
sub-module used to share by living in the same file.  Kept as one module so
the detection logic (and its result) has a single source of truth instead
of being copy-pasted into every file that needs it.
"""
from __future__ import annotations

try:
    import cupy as _cp
    _HAS_CUDA = True
    _v = _cp.cuda.runtime.runtimeGetVersion()      # e.g. 13000
    _CUDA_VER = f"{_v // 1000}.{(_v % 1000) // 10}"
except Exception:
    _cp = None          # type: ignore[assignment]
    _HAS_CUDA = False
    _CUDA_VER = ""

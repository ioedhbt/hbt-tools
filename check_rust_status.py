#!/usr/bin/env python3
"""
check_rust_status.py — One-liner status check for the Rust acceleration.

Run from the repo root:
    python check_rust_status.py

Reports:
  • Whether the committed binary exists for the current OS+arch.
  • Whether the wrapper can actually import it.
  • Whether the env-var `HBT_DISABLE_RUST=1` is forcing the NumPy path.
  • A timing comparison on a representative tuning-sweep payload
    (so you can see the speedup ratio on YOUR machine).

Exits 0 if everything is wired up, 1 if the binary is missing or the
import failed.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _arch_tag() -> str:
    m = (platform.machine() or "unknown").lower()
    if sys.platform == "win32":
        return f"win_{m}"
    if sys.platform == "darwin":
        return f"macosx_{m}"
    return f"linux_{m}"


def main() -> int:
    # Force UTF-8 stdout on Windows so the emoji status icons don't crash
    # on cp1252 default encoding.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print(" HBT Rust acceleration status")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]} on {sys.platform}")
    print(f"Machine arch: {_arch_tag()}")

    bin_dir = ROOT / "tools" / "SSM" / "rust_kernels" / "bin" / _arch_tag()
    print(f"Bin dir     : {bin_dir}")
    if bin_dir.is_dir():
        bins = (list(bin_dir.glob("hbt_rust_kernels*.pyd"))
                + list(bin_dir.glob("hbt_rust_kernels*.so"))
                + list(bin_dir.glob("hbt_rust_kernels*.dylib")))
        # Filter out the `.old-<ts>.<ext>` files that the build script
        # leaves behind when it couldn't overwrite a loaded binary in
        # place.  Those aren't active — they're just retired copies.
        bins = [b for b in bins if ".old-" not in b.name]
        if bins:
            for b in bins:
                size_kb = b.stat().st_size / 1024
                print(f"  committed binary: {b.name}  ({size_kb:.0f} KB)")
        else:
            print("  (no compiled binary in this folder yet)")
    else:
        print("  (folder doesn't exist yet — run "
              "`python build_rust_kernels.py` once)")

    # Force the wrapper to re-load with current env state.
    sys.path.insert(0, str(ROOT))
    if "tools.SSM.helpers.rust_kernels" in sys.modules:
        del sys.modules["tools.SSM.helpers.rust_kernels"]
    from tools.SSM.helpers.rust_kernels import HAS_RUST  # noqa: E402

    forced_off = os.environ.get("HBT_DISABLE_RUST", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    print()
    print(f"HBT_DISABLE_RUST env: {'set (forcing NumPy path)' if forced_off else 'not set'}")
    if HAS_RUST:
        print("Wrapper status: 🦀 Rust kernels ACTIVE")
    elif forced_off:
        print("Wrapper status: 🐢 NumPy fallback "
              "(forced off via HBT_DISABLE_RUST)")
    elif bin_dir.is_dir() and any(bin_dir.iterdir()):
        print("Wrapper status: 🐢 NumPy fallback "
              "(binary exists but failed to import — see error below)")
        # Pull the captured error from the wrapper module first; if it
        # didn't capture anything, retry the import here to surface the
        # exception type and message.
        from tools.SSM.helpers.rust_kernels import _RUST_IMPORT_ERROR  # noqa: E402
        if _RUST_IMPORT_ERROR:
            print(f"  import error: {_RUST_IMPORT_ERROR}")
        else:
            try:
                import hbt_rust_kernels  # noqa: F401
            except Exception as e:                      # pragma: no cover
                print(f"  import error: {type(e).__name__}: {e}")
    else:
        print("Wrapper status: 🐢 NumPy fallback "
              "(no binary committed for this OS/arch)")
        print("  To build: `python build_rust_kernels.py`  "
              "(needs the Rust toolchain — https://rustup.rs)")

    # ── Quick timing check on the primary hot kernel ─────────────────────────
    try:
        import numpy as np
    except ImportError:
        print("\nnumpy missing — skipping timing check.")
        return 0 if HAS_RUST else 1

    from tools.SSM.helpers.rust_kernels import (    # noqa: E402
        y_to_s_4d, _np_y_to_s_4d,
    )

    B, N = 4096, 201
    rng = np.random.default_rng(2026)
    Y = (rng.standard_normal((B, N, 2, 2))
         + 1j * rng.standard_normal((B, N, 2, 2))) * 0.01
    Y = Y.astype(np.complex128)

    def _bench(fn, *a, n=3):
        fn(*a)  # warm
        t0 = time.perf_counter()
        for _ in range(n):
            fn(*a)
        return (time.perf_counter() - t0) / n

    print()
    print(f"Quick benchmark — y_to_s_4d at B={B}, N={N}:")
    t_active = _bench(y_to_s_4d, Y, 50.0)
    t_numpy  = _bench(_np_y_to_s_4d, Y, 50.0)
    print(f"  active path : {t_active*1e3:7.2f} ms"
          f"  ({'Rust' if HAS_RUST else 'NumPy'})")
    print(f"  numpy ref   : {t_numpy*1e3:7.2f} ms")
    if HAS_RUST:
        print(f"  speedup     : {t_numpy/t_active:5.2f}×")
    print()

    return 0 if HAS_RUST else 1


if __name__ == "__main__":
    raise SystemExit(main())

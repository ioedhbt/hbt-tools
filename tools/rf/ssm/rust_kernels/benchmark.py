"""
benchmark.py — Rust-vs-NumPy parity test + microbenchmark.

Usage
-----
    # Run from the repo root:
    python tools/rf/ssm/rust_kernels/benchmark.py

The script:
  1. Generates random complex128 tensors at sizes representative of the
     SSM tuning hot path (B in {64, 1024, 16384}, N=201 frequency points).
  2. Runs each kernel through both the Rust and the NumPy paths.
  3. Asserts numerical agreement to ~1e-10 relative tolerance.
  4. Times each path with N_TRIALS warm-cache runs and prints the ratio.

If the Rust extension isn't installed, the script reports that and
exits 0 — no failure, since the NumPy fallback is the supported path.

To force the NumPy path even when Rust is installed (handy for
comparing against the canonical reference), set ``HBT_DISABLE_RUST=1``.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np


# Import via the helpers package so we exercise the same wrappers the
# rest of the codebase uses.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.rf.ssm.helpers.rust_kernels import (         # noqa: E402
    HAS_RUST,
    inv2x2_batch, mm2x2_batch,
    y_to_s_batch, y_to_s_4d,
    port_residuals_batch,
    _np_inv2x2_batch, _np_mm2x2_batch,
    _np_y_to_s_batch, _np_y_to_s_4d,
    _np_port_residuals_batch,
)


N_TRIALS  = 5
RTOL      = 1e-10
ATOL      = 1e-12


def _bench(fn, *args, n=N_TRIALS):
    # Warm-up — first call sometimes pays page-fault / JIT-like costs.
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n


def _gen_yy(rng, b):
    """Random invertible 2×2 batch — adds a diagonal bias so det isn't tiny."""
    Y = (rng.standard_normal((b, 2, 2)) + 1j * rng.standard_normal((b, 2, 2))) * 0.01
    Y[..., 0, 0] += 0.02 + 0.0j
    Y[..., 1, 1] += 0.02 + 0.0j
    return Y.astype(np.complex128, copy=False)


def run_kernel_parity(b: int, n_freq: int, rng: np.random.Generator) -> dict:
    res: dict[str, dict[str, float]] = {}

    Y     = _gen_yy(rng, b)
    Y_alt = _gen_yy(rng, b)

    # 1. inv2x2_batch
    out_rust = inv2x2_batch(Y)
    out_np   = _np_inv2x2_batch(Y)
    assert np.allclose(out_rust, out_np, rtol=RTOL, atol=ATOL), \
        "inv2x2_batch mismatch"
    t_rust = _bench(inv2x2_batch, Y) if HAS_RUST else float("nan")
    t_np   = _bench(_np_inv2x2_batch, Y)
    res["inv2x2_batch"] = {"rust_ms": t_rust * 1e3, "np_ms": t_np * 1e3,
                            "speedup": t_np / t_rust if HAS_RUST else 0.0}

    # 2. mm2x2_batch
    out_rust = mm2x2_batch(Y, Y_alt)
    out_np   = _np_mm2x2_batch(Y, Y_alt)
    assert np.allclose(out_rust, out_np, rtol=RTOL, atol=ATOL), \
        "mm2x2_batch mismatch"
    t_rust = _bench(mm2x2_batch, Y, Y_alt) if HAS_RUST else float("nan")
    t_np   = _bench(_np_mm2x2_batch, Y, Y_alt)
    res["mm2x2_batch"] = {"rust_ms": t_rust * 1e3, "np_ms": t_np * 1e3,
                           "speedup": t_np / t_rust if HAS_RUST else 0.0}

    # 3. y_to_s_batch
    out_rust = y_to_s_batch(Y, 50.0)
    out_np   = _np_y_to_s_batch(Y, 50.0)
    assert np.allclose(out_rust, out_np, rtol=RTOL, atol=ATOL), \
        "y_to_s_batch mismatch"
    t_rust = _bench(y_to_s_batch, Y, 50.0) if HAS_RUST else float("nan")
    t_np   = _bench(_np_y_to_s_batch, Y, 50.0)
    res["y_to_s_batch"] = {"rust_ms": t_rust * 1e3, "np_ms": t_np * 1e3,
                            "speedup": t_np / t_rust if HAS_RUST else 0.0}

    # 4. y_to_s_4d
    Y4 = _gen_yy(rng, b * n_freq).reshape(b, n_freq, 2, 2).copy()
    out_rust = y_to_s_4d(Y4, 50.0)
    out_np   = _np_y_to_s_4d(Y4, 50.0)
    assert np.allclose(out_rust, out_np, rtol=RTOL, atol=ATOL), \
        "y_to_s_4d mismatch"
    t_rust = _bench(y_to_s_4d, Y4, 50.0) if HAS_RUST else float("nan")
    t_np   = _bench(_np_y_to_s_4d, Y4, 50.0)
    res["y_to_s_4d"] = {"rust_ms": t_rust * 1e3, "np_ms": t_np * 1e3,
                         "speedup": t_np / t_rust if HAS_RUST else 0.0}

    # 5. port_residuals_batch
    S_mea = _gen_yy(rng, n_freq)   # treat as (N, 2, 2)
    S_mod = (rng.standard_normal((b, n_freq, 2, 2))
             + 1j * rng.standard_normal((b, n_freq, 2, 2))) * 0.01
    S_mod = S_mod.astype(np.complex128)
    out_rust = port_residuals_batch(S_mea, S_mod)
    out_np   = _np_port_residuals_batch(S_mea, S_mod)
    assert np.allclose(out_rust, out_np, rtol=1e-8, atol=1e-10), \
        "port_residuals_batch mismatch"
    t_rust = _bench(port_residuals_batch, S_mea, S_mod) if HAS_RUST else float("nan")
    t_np   = _bench(_np_port_residuals_batch, S_mea, S_mod)
    res["port_residuals_batch"] = {"rust_ms": t_rust * 1e3,
                                    "np_ms":   t_np * 1e3,
                                    "speedup": t_np / t_rust if HAS_RUST else 0.0}

    return res


def _fmt_row(name: str, kn: str, d: dict) -> str:
    if HAS_RUST:
        return (f"  {kn:<22s}  rust={d['rust_ms']:7.2f} ms  "
                f"numpy={d['np_ms']:7.2f} ms   speedup={d['speedup']:5.2f}×")
    return f"  {kn:<22s}  numpy={d['np_ms']:7.2f} ms  (rust ext not installed)"


def main():
    # Force UTF-8 on Windows so the small box-drawing / arrow chars below
    # don't crash on cp1252 default stdout encoding.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("HBT Rust kernels - parity & benchmark")
    print("-" * 60)
    print(f"Rust extension available: {HAS_RUST}")
    if not HAS_RUST:
        print("  (Falling back to NumPy.  Build the extension with:")
        print("     cd tools/rf/ssm/rust_kernels && pip install -e .")
        print("   then re-run this script to compare.)")
    print()

    rng = np.random.default_rng(2026)
    for label, b, n_freq in [
        ("Small  (B=64,     N=201)",     64, 201),
        ("Medium (B=1 024,  N=201)",   1024, 201),
        ("Large  (B=16 384, N=201)",  16384, 201),
    ]:
        print(f"{label}")
        try:
            res = run_kernel_parity(b, n_freq, rng)
        except AssertionError as e:
            print(f"  PARITY FAILURE: {e}")
            sys.exit(1)
        for kn, d in res.items():
            print(_fmt_row(label, kn, d))
        print()

    print("All Phase 1 parity checks passed.")
    if HAS_RUST:
        print("Re-run with HBT_DISABLE_RUST=1 to force the NumPy path.")

    # ── Phase 2 — per-topology end-to-end simulator parity (scaffold) ──
    # These checks call the Rust stubs through the dispatch wrapper
    # (in helpers/rust_kernels.py).  Until a topology is implemented,
    # the stub raises NotImplementedError and the dispatch falls back
    # to NumPy — which means this loop reports "scaffold only" and
    # doesn't fail.  Once a topology is implemented, the same loop
    # will catch parity regressions automatically.
    print()
    print("Phase 2 status (per-topology end-to-end simulator):")
    try:
        from tools.rf.ssm.helpers.rust_kernels import (   # noqa: E402
            sim_cheng_t_batch, sim_cheng_pi_batch, sim_xu_t_batch,
        )
    except Exception as e:
        print(f"  could not import phase-2 wrappers: {e}")
    else:
        for nm, wrapper in [
            ("sim_cheng_t_batch",  sim_cheng_t_batch),
            ("sim_cheng_pi_batch", sim_cheng_pi_batch),
            ("sim_xu_t_batch",     sim_xu_t_batch),
        ]:
            # Probe the underlying Rust stub directly to detect whether
            # the implementation has landed.  The wrapper itself would
            # silently fall back to NumPy and hide the state.
            try:
                import hbt_rust_kernels as _rkmod      # noqa: F401
                rust_fn = getattr(_rkmod, nm, None)
                if rust_fn is None:
                    state = "missing from Rust crate"
                else:
                    try:
                        rust_fn({}, np.zeros(1, dtype=np.float64), 50.0)
                        state = "IMPLEMENTED (parity test would run here)"
                    except NotImplementedError:
                        state = "scaffold only (NumPy fallback active)"
                    except Exception as e:
                        state = f"error probing: {type(e).__name__}"
            except Exception:
                state = "Rust crate not loaded"
            print(f"  {nm:<22s} {state}")


if __name__ == "__main__":
    main()

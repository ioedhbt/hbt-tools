"""
Benchmark `rust_parse_and_compute_batch` vs the per-file Python loop on
the bulk-upload path (the hot path the user actually feels).

Reads dummy_data_practice/*.s2p N_REPS times each to simulate "30 files",
and compares:
  - Python serial: parse_s2p + s_to_y + compute_metrics per file
  - Rust batch:    rust_parse_and_compute_batch on the full byte list

Also validates numerical parity element-wise.

Usage:
    .hbttools/Scripts/python.exe tools/_profile_rust_batch.py [N_FILES]
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.SSM.helpers import (
    parse_s2p, s_to_y, compute_metrics, rust_parse_and_compute_batch,
)
from tools.SSM.helpers.rust_kernels import HAS_RUST, rust_diagnostic


def py_loop(files_bytes: list[bytes]) -> list[dict]:
    out = []
    for raw in files_bytes:
        text = raw.decode("utf-8", errors="ignore")
        freq, S, z0 = parse_s2p(text)
        df = compute_metrics(s_to_y(S, z0), freq)
        out.append({"freq": freq, "h21_db": df["|h21|² (dB)"].values})
    return out


def rust_batch(files_bytes: list[bytes]) -> list[dict]:
    return rust_parse_and_compute_batch(files_bytes)


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    samples = sorted((ROOT / "dummy_data_practice").glob("*.s2p"))
    pool = [p.read_bytes() for p in samples]
    files_bytes = [pool[i % len(pool)] for i in range(n_files)]

    diag = rust_diagnostic()
    print(f"\n=== Rust batch parse/compute benchmark ===")
    print(f"   HAS_RUST: {diag['has_rust']}")
    print(f"   Binary  : {diag['bin_dir']}")
    print(f"   Files   : {n_files} (from {len(samples)} samples cycled)")
    print(f"   N_freq  : 1001\n")

    # Warm-up
    py_loop(files_bytes[:1])
    rust_batch(files_bytes[:1])

    # Parity check on one file
    py_one   = py_loop([pool[0]])[0]
    rust_one = rust_batch([pool[0]])[0]
    par_freq = np.allclose(py_one["freq"], rust_one["freq"], rtol=1e-12)
    par_h21  = np.allclose(py_one["h21_db"], rust_one["h21_db"],
                            rtol=1e-9, atol=1e-12, equal_nan=True)
    print(f"Parity: freq={par_freq}  |h21|²={par_h21}\n")

    # Force-NumPy run for an A/B comparison.  We re-import with the env
    # var set so the wrapper exposes the NumPy reference path even when
    # the binary is present.
    REPS = 5

    # Rust
    t0 = time.perf_counter()
    for _ in range(REPS): rust_batch(files_bytes)
    t_rust = (time.perf_counter() - t0) / REPS * 1000

    # Python
    t0 = time.perf_counter()
    for _ in range(REPS): py_loop(files_bytes)
    t_py = (time.perf_counter() - t0) / REPS * 1000

    print(f"   Python serial loop : {t_py:8.1f} ms  "
          f"({t_py / n_files:.2f} ms/file)")
    print(f"   Rust batch (Rayon) : {t_rust:8.1f} ms  "
          f"({t_rust / n_files:.2f} ms/file)")
    print(f"   Speedup            : {t_py / t_rust:.1f}×\n")


if __name__ == "__main__":
    main()

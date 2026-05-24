"""
Standalone profiler for the bulk-s2p-upload hot path.

Runs the same per-file pipeline that `process_dut` does inside the Streamlit
app, but with `perf_counter` brackets around each stage so we can see where
the wall-clock actually goes.  No Streamlit, no Plotly — just the math.

Also simulates a 30-file "Streamlit rerun loop" with vs without the
session-state cache added in IOED_HBT_RF_extract.py — confirms the cache
turns a multi-second rerun into a sub-millisecond one.

Usage (from repo root):
    .hbttools/Scripts/python.exe tools/_profile_bulk_upload.py [REPS]

REPS defaults to 5 (each file processed REPS times; medians reported).
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path
from statistics import median

# Force UTF-8 on stdout so the bar / box-drawing chars render under cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.SSM.helpers.s2p_io import parse_s2p
from tools.SSM.helpers.rf_math import s_to_y, y_to_s_vec
from tools.SSM.helpers.metrics import compute_metrics, extract_limit


SAMPLES = sorted((ROOT / "dummy_data_practice").glob("*.s2p"))


def profile_file(path: Path) -> dict[str, float]:
    """Return per-stage timings (ms) for a single file."""
    raw = path.read_bytes()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    text = raw.decode("utf-8", errors="ignore")
    timings["decode"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    freq, S_raw, z0 = parse_s2p(text)
    timings["parse_s2p"] = (time.perf_counter() - t0) * 1000.0
    timings["_n_freq"] = len(freq)

    t0 = time.perf_counter()
    Y_raw = s_to_y(S_raw, z0)
    timings["s_to_y"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    df_raw = compute_metrics(Y_raw, freq)
    timings["compute_metrics"] = (time.perf_counter() - t0) * 1000.0

    f_arr = df_raw["Freq (GHz)"].values
    t0 = time.perf_counter()
    extract_limit(f_arr, df_raw["|h21|² (dB)"].values,
                  df_raw["fT Plateau (GHz)"].values, 2, 0.01, 50.0)
    extract_limit(f_arr, df_raw["Mason U (dB)"].values,
                  df_raw["fmax U Plateau (GHz)"].values, 2, 0.01, 50.0)
    extract_limit(f_arr, df_raw["MAG/MSG (dB)"].values,
                  df_raw["fmax MAG Plateau (GHz)"].values, 2, 0.01, 50.0)
    timings["extract_limit_x3"] = (time.perf_counter() - t0) * 1000.0

    return timings


def _full_process(content_bytes: bytes):
    """Mimic process_dut's math content (no Streamlit-only bits)."""
    text = content_bytes.decode("utf-8", errors="ignore")
    freq, S_raw, z0 = parse_s2p(text)
    Y_raw = s_to_y(S_raw, z0)
    df_raw = compute_metrics(Y_raw, freq)
    f_arr = df_raw["Freq (GHz)"].values
    extract_limit(f_arr, df_raw["|h21|² (dB)"].values,
                  df_raw["fT Plateau (GHz)"].values, 2, 0.01, 50.0)
    extract_limit(f_arr, df_raw["Mason U (dB)"].values,
                  df_raw["fmax U Plateau (GHz)"].values, 2, 0.01, 50.0)
    extract_limit(f_arr, df_raw["MAG/MSG (dB)"].values,
                  df_raw["fmax MAG Plateau (GHz)"].values, 2, 0.01, 50.0)
    return {"freq": freq, "S": S_raw, "Y": Y_raw, "df": df_raw}


def _cache_key(content: bytes, *params) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(content)
    h.update(repr(params).encode())
    return h.digest()


def simulate_streamlit_loop(n_files: int, n_reruns: int, use_cache: bool) -> float:
    """Simulate Streamlit's bulk-upload loop over `n_files` (cycled from
    SAMPLES) across `n_reruns` reruns.  Returns total wall-clock (ms)."""
    # Pre-load file bytes once so we don't measure disk I/O (Streamlit's
    # file_uploader keeps the upload in memory already).
    pool = [p.read_bytes() for p in SAMPLES]
    files = [(f"file_{i:03d}.s2p", pool[i % len(pool)]) for i in range(n_files)]

    cache: dict[bytes, dict] = {}
    chart_params = (2, 0.01, 50.0)

    t0 = time.perf_counter()
    for _ in range(n_reruns):
        for name, content in files:
            if use_cache:
                key = _cache_key(content, *chart_params)
                hit = cache.get(key)
                if hit is None:
                    cache[key] = _full_process(content)
            else:
                _full_process(content)
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"\n=== Per-file pipeline breakdown ===")
    print(f"Sample files: {len(SAMPLES)} (each timed {reps}× — medians shown)\n")

    if not SAMPLES:
        print(f"No .s2p files found in {ROOT / 'dummy_data_practice'}")
        return

    for p in SAMPLES:
        profile_file(p)

    all_runs = [profile_file(p) for p in SAMPLES for _ in range(reps)]
    keys = [k for k in all_runs[0] if not k.startswith("_")]
    agg = {k: median(r[k] for r in all_runs) for k in keys}
    per_file_total = sum(agg.values())

    print("Median across ALL samples:")
    for k in keys:
        pct = 100.0 * agg[k] / per_file_total if per_file_total > 0 else 0.0
        print(f"   {k:<28s}  {agg[k]:8.3f} ms  {pct:5.1f}%")
    print(f"   {'PER-FILE TOTAL':<28s}  {per_file_total:8.3f} ms\n")

    # ── Cache vs no-cache simulation ────────────────────────────────────────
    print("=== Streamlit rerun simulation (30 files, 5 reruns) ===")
    print("    [models: every slider drag / checkbox toggle = 1 rerun]\n")
    n_files, n_reruns = 30, 5

    t_nocache = simulate_streamlit_loop(n_files, n_reruns, use_cache=False)
    t_cache = simulate_streamlit_loop(n_files, n_reruns, use_cache=True)

    print(f"   No cache (current main):  {t_nocache:9.1f} ms  "
          f"({t_nocache / n_reruns:.1f} ms/rerun)")
    print(f"   With cache (this PR):     {t_cache:9.1f} ms  "
          f"({t_cache / n_reruns:.1f} ms/rerun avg, "
          f"first rerun cold, rest warm)")
    print(f"   Speedup over {n_reruns} reruns: {t_nocache / t_cache:.1f}×\n")


if __name__ == "__main__":
    main()

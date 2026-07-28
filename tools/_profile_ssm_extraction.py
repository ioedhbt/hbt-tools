"""
Standalone profiler for the SSM extraction tab's math pipeline.

Mirrors render_ssm_tab's per-DUT work without Streamlit/Plotly:
    parse → s_to_y → step_open → step_short → peel_parasitics
          → _step2_T/_pi → _step3_T/_pi → simulate_vec
          → render comparator math (compute_h21_U, find_ft_fmax, etc.)

Synthesises Open + Short dummies from the sample DUT file so we can
exercise the full de-embed → extract → simulate chain.  Numerical results
are meaningless; we're just timing the code paths the user actually hits
when opening the SSM Extraction tab.

Usage (from repo root):
    .hbttools/Scripts/python.exe tools/_profile_ssm_extraction.py [REPS]
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from statistics import median

# Guarded — see _profile_bulk_upload.py: rewrapping at import time closes the
# interpreter's real stdout for anything that imports the `tools` package.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.rf.ssm.helpers.s2p_io import parse_s2p
from tools.rf.ssm.helpers.rf_math import s_to_y, y_to_s_vec
from tools.rf.ssm.helpers.metrics import compute_h21_U, find_ft_fmax, extrap_20dbdec
from tools.rf.ssm.helpers.deembed_math import (
    step_open, step_short, peel_parasitics,
    build_Y_pad_vec, build_Z_ser_vec,
)
from tools.rf.ssm.models.cheng import (
    _step2_T, _step2_pi, _step3_T, _step3_pi,
    _Y_int_T_vec, _Y_int_Pi_vec, _sim_wrap_vec,
)


SAMPLES = sorted((ROOT / "dummy_data_practice").glob("*.s2p"))


def _bracket(timings: dict, label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    timings[label] = timings.get(label, 0.0) + (time.perf_counter() - t0) * 1000.0
    return out


def _synth_dummies(freq, z0):
    """Build a (Open, Short) dummy pair with small, realistic pad/lead values
    so the de-embed math has something non-trivial to chew on.  Numerical
    results are not validated — purpose is just to exercise the code path."""
    omega = 2.0 * np.pi * freq
    p_open = dict(Cpbe=8e-15, Cpce=6e-15, Cpbc=2e-15)
    Y_pad = build_Y_pad_vec(p_open, omega, np)
    S_open = y_to_s_vec(Y_pad, z0, np)

    p_short = dict(**p_open,
                   Lb=40e-12, Lc=40e-12, Le=20e-12,
                   Rpb=0.5, Rpc=0.5, Rpe=0.2,
                   Cpar_Lb=0.0, Cpar_Lc=0.0, Cpar_Le=0.0)
    Z_ser = build_Z_ser_vec(p_short, omega, np)
    # Y_short = Y_pad + inv(Z_ser); inline 2×2 inverse for the ser branch
    a = Z_ser[..., 0, 0]; b = Z_ser[..., 0, 1]
    c = Z_ser[..., 1, 0]; d = Z_ser[..., 1, 1]
    inv_det = 1.0 / (a * d - b * c)
    Y_ser = np.empty_like(Z_ser)
    Y_ser[..., 0, 0] =  d * inv_det
    Y_ser[..., 0, 1] = -b * inv_det
    Y_ser[..., 1, 0] = -c * inv_det
    Y_ser[..., 1, 1] =  a * inv_det
    S_short = y_to_s_vec(Y_pad + Y_ser, z0, np)
    return (freq, S_open, z0), (freq, S_short, z0), p_short


def profile_extraction(path: Path) -> dict[str, float]:
    """Return per-stage timings (ms) for one full SSM extraction."""
    timings: dict[str, float] = {}
    raw = path.read_bytes()

    # ── 0. Decode + parse ──────────────────────────────────────────────────
    text = _bracket(timings, "0a. decode", raw.decode, "utf-8", errors="ignore")
    freq, S_raw, z0 = _bracket(timings, "0b. parse_s2p", parse_s2p, text)
    n_low = max(2, len(freq) // 8)

    # ── 0c. S→Y ────────────────────────────────────────────────────────────
    Y_raw = _bracket(timings, "0c. s_to_y", s_to_y, S_raw, z0)

    # ── 1. Synthesise dummy O/S so we have a realistic input ───────────────
    open_data, short_data, p_short = _synth_dummies(freq, z0)

    # ── Step 1 / 1b: pad + lead extraction (the very first thing the tab does)
    para_open, _ = _bracket(timings, "1. step_open (pad caps)",
                             step_open, open_data, n0=0, n1=len(freq) // 4)
    para_short, _ = _bracket(timings, "1b. step_short (lead L/R)",
                              step_short, short_data, freq,
                              para_open["Cpbe"], para_open["Cpce"],
                              para_open["Cpbc"], open_data=open_data,
                              n0=0, n1=max(3, int(len(freq) * 0.20)))

    para_eff = {**para_open, **para_short,
                "Cpbe_mode": "None", "Cpce_mode": "None", "Cpbc_mode": "None",
                "Cpbe_extra": 0.0, "Cpce_extra": 0.0, "Cpbc_extra": 0.0,
                "Cpar_Lb": 0.0, "Cpar_Lc": 0.0, "Cpar_Le": 0.0}

    # ── Step 2: de-embed pad + lead ────────────────────────────────────────
    Y_ex1 = _bracket(timings, "2. peel_parasitics", peel_parasitics,
                      S_raw, freq, z0, para_eff)

    # ── Step 3 (T-topology): Cbex, Cbcx ────────────────────────────────────
    res2_T, arr2_T = _bracket(timings, "3T. _step2_T (Cbex/Cbcx)",
                               _step2_T, Y_ex1, freq, n_low)

    # ── Step 4 (T-topology): intrinsic Rbi/Rbe/Cbe/Rbc/Cbc/α0/τB/τC ────────
    res3_T, _ = _bracket(timings, "4T. _step3_T (intrinsic)",
                          _step3_T, arr2_T["Y_ex2"], freq, res2_T["Cbcx"], n_low)

    # ── Step 3 (π-topology): Cbex, Cbcx ────────────────────────────────────
    res2_pi, arr2_pi = _bracket(timings, "3π. _step2_pi (Cbex/Cbcx)",
                                 _step2_pi, Y_ex1, freq, n_low)

    # ── Step 4 (π-topology): intrinsic Rbi/Rbe/Cbe/Cbc/Gm0/τ ───────────────
    res3_pi, _ = _bracket(timings, "4π. _step3_pi (intrinsic)",
                           _step3_pi, arr2_pi["Y_ex2"], freq,
                           res2_pi["Cbcx"], n_low)

    # ── Step 5: forward-simulate T ─────────────────────────────────────────
    p_T = {**para_eff, **res2_T, **res3_T}
    S_T = _bracket(timings, "5T. _sim_wrap_vec (T)",
                    _sim_wrap_vec, _Y_int_T_vec, p_T, freq, z0, np)

    p_pi = {**para_eff, **res2_pi, **res3_pi}
    S_pi = _bracket(timings, "5π. _sim_wrap_vec (π)",
                     _sim_wrap_vec, _Y_int_Pi_vec, p_pi, freq, z0, np)

    # ── Bode comparator math (per-tab render path) ─────────────────────────
    f_ghz = freq * 1e-9
    h21_m, U_m = _bracket(timings, "6a. compute_h21_U (meas)",
                           compute_h21_U, S_raw)
    h21_t, U_t = _bracket(timings, "6b. compute_h21_U (T model)",
                           compute_h21_U, S_T)
    h21_p, U_p = _bracket(timings, "6c. compute_h21_U (π model)",
                           compute_h21_U, S_pi)
    _bracket(timings, "6d. find_ft_fmax ×3",
              lambda: [find_ft_fmax(f_ghz, h21_m, U_m),
                       find_ft_fmax(f_ghz, h21_t, U_t),
                       find_ft_fmax(f_ghz, h21_p, U_p)])
    _bracket(timings, "6e. extrap_20dbdec ×6",
              lambda: [extrap_20dbdec(f_ghz, h21_m),
                       extrap_20dbdec(f_ghz, U_m),
                       extrap_20dbdec(f_ghz, h21_t),
                       extrap_20dbdec(f_ghz, U_t),
                       extrap_20dbdec(f_ghz, h21_p),
                       extrap_20dbdec(f_ghz, U_p)])

    timings["_n_freq"] = len(freq)
    return timings


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not SAMPLES:
        print(f"No .s2p files found in {ROOT / 'dummy_data_practice'}")
        return

    print(f"\n=== SSM extraction pipeline profiler ===")
    print(f"Sample files: {len(SAMPLES)} (each timed {reps}× — medians shown)")
    print(f"Pipeline: parse → s_to_y → step_open → step_short → peel → "
          f"_step2/_step3 (T+π) → _sim_wrap_vec → metrics\n")

    # Warm-up to seed disk cache + Streamlit @cache_data wrappers.
    for p in SAMPLES:
        profile_extraction(p)

    all_runs = [profile_extraction(p) for p in SAMPLES for _ in range(reps)]
    n_freq = all_runs[0]["_n_freq"]

    keys = [k for k in all_runs[0] if not k.startswith("_")]
    med = {k: median(r[k] for r in all_runs) for k in keys}
    total = sum(med.values())

    print(f"Per-DUT extraction (N_freq={n_freq}):\n")
    width = max(len(k) for k in keys)
    for k in keys:
        pct = 100.0 * med[k] / total if total > 0 else 0.0
        bar = "█" * int(pct / 2)
        print(f"   {k:<{width}s}  {med[k]:8.3f} ms  {pct:5.1f}%  {bar}")
    print(f"   {'TOTAL':<{width}s}  {total:8.3f} ms\n")

    # Group summary
    groups = [
        ("Parse + S→Y      (stage 0)",      ["0a. decode", "0b. parse_s2p", "0c. s_to_y"]),
        ("Pad/lead extract (stage 1+1b)",   ["1. step_open (pad caps)", "1b. step_short (lead L/R)"]),
        ("De-embed         (stage 2)",       ["2. peel_parasitics"]),
        ("Cbex/Cbcx        (stage 3)",       ["3T. _step2_T (Cbex/Cbcx)", "3π. _step2_pi (Cbex/Cbcx)"]),
        ("Intrinsic        (stage 4)",       ["4T. _step3_T (intrinsic)", "4π. _step3_pi (intrinsic)"]),
        ("Forward simulate (stage 5)",       ["5T. _sim_wrap_vec (T)", "5π. _sim_wrap_vec (π)"]),
        ("Bode/fT/fmax     (stage 6)",       ["6a. compute_h21_U (meas)",
                                              "6b. compute_h21_U (T model)",
                                              "6c. compute_h21_U (π model)",
                                              "6d. find_ft_fmax ×3",
                                              "6e. extrap_20dbdec ×6"]),
    ]
    print("Grouped totals:")
    for label, ks in groups:
        g = sum(med[k] for k in ks if k in med)
        pct = 100.0 * g / total if total > 0 else 0.0
        print(f"   {label:<35s}  {g:7.3f} ms  {pct:5.1f}%")
    print(f"   {'PER-DUT TOTAL':<35s}  {total:7.3f} ms")


if __name__ == "__main__":
    main()

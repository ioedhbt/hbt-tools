"""
models/base_ui.py — Shared Streamlit UI helpers used by all model classes.

Kept separate from the abstract base so model files import one thing,
not a chain of ssm_* modules.
"""
from __future__ import annotations
import gc
import numpy as np
import threading
import streamlit as st
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from itertools import product as iterproduct
from io import BytesIO

from ..helpers import (extended_smith_grid, params_hash, s_to_y,
                        build_Y_pad_batch, build_Z_ser_batch,
                        plotly_with_dl,
                        quickset_buttons, apply_pending)

# Streamlit's "rerun current script" exception — raised when any st.* call
# happens after the user has clicked a widget that triggers a re-run (e.g.
# the Stop button below).  We catch it so the long-running tuning loop can
# break out cleanly without losing the persisted top-K state.
#
# We also catch `StopException`, which Streamlit raises through a *different*
# code path (e.g. `st.stop()` or some Stop-button paths in newer versions).
# Both indicate "the user wants the current run to end now"; we treat them
# identically — persist state, free GPU buffers, and re-raise so Streamlit
# can finish the rerun cleanly.
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException as _RerunException
except Exception:
    try:
        from streamlit.runtime.scriptrunner import RerunException as _RerunException
    except Exception:
        _RerunException = None  # type: ignore[assignment]
try:
    from streamlit.runtime.scriptrunner.script_runner import StopException as _StopException
except Exception:
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException as _StopException
    except Exception:
        _StopException = None  # type: ignore[assignment]

# ── CUDA detection (runtime, zero-cost when CuPy is absent) ─────────────────

try:
    import cupy as _cp
    _HAS_CUDA = True
    _v = _cp.cuda.runtime.runtimeGetVersion()      # e.g. 13000
    _CUDA_VER = f"{_v // 1000}.{(_v % 1000) // 10}"
except Exception:
    _cp = None          # type: ignore[assignment]
    _HAS_CUDA = False
    _CUDA_VER = ""


# ── Pad parameter specs (shared across all models) ────────────────────────────
# (key, display_label, SI_scale, unit_string, format_string, step)

PAD_SPECS = [
    ("Cpbe","Cpbe", 1e15,"fF","%.4f",0.1),
    ("Cpce","Cpce", 1e15,"fF","%.4f",0.1),
    ("Cpbc","Cpbc", 1e15,"fF","%.4f",0.01),
    ("Lb",  "Lb",   1e12,"pH","%.3f",0.1),
    ("Lc",  "Lc",   1e12,"pH","%.3f",0.1),
    ("Le",  "Le",   1e12,"pH","%.3f",0.01),
    ("Rpb", "Rb",   1.0, "Ω", "%.4f",0.01),
    ("Rpc", "Rc",   1.0, "Ω", "%.4f",0.01),
    ("Rpe", "Re",   1.0, "Ω", "%.4f",0.01),
]
_PAD_KEYS = [k for k, *_ in PAD_SPECS]


# ── Residual ──────────────────────────────────────────────────────────────────

def ssm_residual(S_mea: np.ndarray, S_mod: np.ndarray) -> float:
    """RMS relative S-parameter residual across all four ports (%)."""
    total = 0.0
    for r in range(2):
        for c in range(2):
            sm = S_mea[:,r,c]; sk = S_mod[:,r,c]
            denom = np.sum(np.abs(sm)**2)
            if denom > 0:
                total += np.sqrt(np.sum(np.abs(sm-sk)**2) / denom)
    return total / 4.0 * 100.0


# ── Per-port residuals ────────────────────────────────────────────────────────

def _port_residuals(S_mea, S_mod):
    """Return dict with Total, S11, S12, S21, S22 residuals in %."""
    res = {}
    total = 0.0
    for name, (r, c) in [("S11",(0,0)),("S12",(0,1)),("S21",(1,0)),("S22",(1,1))]:
        sm = S_mea[:, r, c]; sk = S_mod[:, r, c]
        denom = np.sum(np.abs(sm)**2)
        if denom > 0:
            val = np.sqrt(np.sum(np.abs(sm - sk)**2) / denom) * 100.0
        else:
            val = 0.0
        res[name] = val
        total += val
    res["Total"] = total / 4.0
    return res


def _port_residuals_batch(S_mea, S_mod_batch, xp):
    """Batched residuals — *fused* across all 4 ports for minimum kernel launches.

    S_mea       : (N, 2, 2)              — measured (already on device)
    S_mod_batch : (B, N, 2, 2)           — model
    Returns dict whose values are (B,) arrays on the *xp* device.

    Old impl ran ~5 ops per port × 4 ports + summation = ~22 kernel launches.
    This impl does ~7 launches total — significant Python/launch overhead saved
    in the GPU hot loop.
    """
    diff   = S_mea[None, :, :, :] - S_mod_batch        # (B, N, 2, 2)
    num    = xp.sum(xp.abs(diff)  ** 2, axis=1)        # (B, 2, 2)
    den    = xp.sum(xp.abs(S_mea) ** 2, axis=0)        # (2, 2)
    den_s  = xp.where(den > 0, den, 1.0)               # avoid div-by-0
    val    = xp.sqrt(num / den_s[None, :, :]) * 100.0  # (B, 2, 2)
    val    = xp.where(den[None, :, :] > 0, val, 0.0)

    s11 = val[:, 0, 0]
    s12 = val[:, 0, 1]
    s21 = val[:, 1, 0]
    s22 = val[:, 1, 1]
    total = (s11 + s12 + s21 + s22) * 0.25
    return {"S11": s11, "S12": s12, "S21": s21, "S22": s22, "Total": total}


# ── Smith chart ───────────────────────────────────────────────────────────────

_SMITH_COLORS = {"S11":"#1f77b4","S22":"#ff7f0e","S21":"#2ca02c","S12":"#d62728"}

def render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith",
                       show_title=True, meas_label="Meas.", sim_label="Model",
                       *, compact: bool = False, height: int | None = None,
                       extra_download: tuple | None = None):
    """Render a Plotly Smith chart with measured (markers) + modeled (dashed).

    Parameters
    ----------
    compact : bool, default False
        When True, the layout is tuned for narrow containers (e.g. the
        side-by-side Smith+Bode in the Visual Tuning preview):
          • legend repositioned to the BOTTOM (instead of right side).
          • height shrunk to 380 unless ``height`` overrides it.
          • margins tightened.
        Default False preserves the original full-width layout.
    height : int, optional
        Explicit pixel height; overrides the compact / default values.
    """
    if scales is None:
        scales = {"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0}
    fig = go.Figure()
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr)
    for name, (r, c) in [("S11",(0,0)),("S22",(1,1)),("S21",(1,0)),("S12",(0,1))]:
        col = _SMITH_COLORS[name]; sc = scales.get(name, 1.0)
        sm = S_mea[:,r,c]*sc; sk = S_sim[:,r,c]*sc
        sc_lbl = "" if abs(sc-1.0)<1e-9 else (f" ×{sc:.2g}" if sc>=1 else f" ÷{1/sc:.2g}")
        fig.add_trace(go.Scattergl(x=sm.real, y=sm.imag, mode="markers",
                                  name=f"{name}{sc_lbl} {meas_label}",
                                  marker=dict(color=col, size=5, symbol="circle"),
                                  hovertemplate=f"{name} {meas_label}<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scattergl(x=sk.real, y=sk.imag, mode="lines",
                                  name=f"{name}{sc_lbl} {sim_label}",
                                  line=dict(color=col, width=2.0, dash="dash"),
                                  hovertemplate=f"{name} {sim_label}<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
    title_cfg = (dict(text=f"Smith Chart - {model_name}", font=dict(size=12))
                 if show_title else None)
    if compact:
        # The bottom-legend layout needs ~140 px of bottom margin to
        # keep the X-axis title "Re(Γ)" clear of the legend chips,
        # plus another ~30 px below for the "● Meas. — — Model"
        # annotation.  At height=540 the plot area is still ≈ 390 px
        # which keeps the unit-circle squareness visually reasonable.
        legend_cfg = dict(orientation="h", x=0.5, y=-0.22,
                          xanchor="center", yanchor="top",
                          font=dict(size=9))
        margin_cfg = dict(l=40, r=15, t=40 if show_title else 10, b=140)
        eff_height = height if height is not None else 540
        annotation_y = -0.42  # below the legend chips
    else:
        legend_cfg = dict(x=1.02, y=1.0, xanchor="left")
        margin_cfg = dict(l=50, r=30, t=70, b=50)
        eff_height = height if height is not None else 560
        annotation_y = -0.08
    fig.update_layout(
        title=title_cfg,
        xaxis=dict(title="Re(Γ)", range=[-1.1,1.1], scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1,1.1], showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=eff_height,
        margin=margin_cfg,
        legend=legend_cfg,
        hovermode="closest",
        annotations=[dict(x=0.5, y=annotation_y, xref="paper", yref="paper",
                          showarrow=False,
                          text=f"● {meas_label} (markers)  |  - - {sim_label} (dashed)",
                          font=dict(size=10, color="gray"), align="center")])
    plotly_with_dl(fig, key=key, filename=key, extra_download=extra_download)


def render_smith_with_ftfmax(S_raw, S_sim, freq, model_name: str,
                             model_short: str, fname: str, scales=None,
                             *, s2p_bytes: bytes | None = None,
                             s2p_filename: str | None = None):
    """
    Two-column layout: Smith chart (left) + fT/fmax mini-card (right).

    The Smith chart still owns the residual line in its title; the mini-card
    on the right shows |h21|² and Mason U for both measured and modeled with
    20 dB/dec extrapolation when needed.  Use this in place of the bare
    `render_smith_chart()` call inside each model's `render_override_and_smith`.

    When ``s2p_bytes`` is supplied, the Smith chart's download row gains a
    second button (📥 S2P) right next to the standard xlsx — this replaced
    the standalone "Download Modeled DUT S2P" section that used to live at
    the bottom of the SSM extraction tab.
    """
    # Local import — ssm_plots imports back from base_ui at module load time,
    # so a top-level import here would create a circular dependency.
    from ..ssm_plots import render_ft_fmax_card

    port_res = _port_residuals(S_raw, S_sim)
    err = ssm_residual(S_raw, S_sim)
    title_line1 = f"**{model_name} Measured vs Modeled** — **Total Residual**: {err:.2f}%"
    title_line2 = (f"**per-trace residuals**: **S11**: {port_res['S11']:.2f}%  , **S12**: {port_res['S12']:.2f}%  , "
                   f"**S21**: {port_res['S21']:.2f}%  , **S22**: {port_res['S22']:.2f}%")
    st.markdown(f"{title_line1}; {title_line2}")
    extra_dl = None
    if s2p_bytes is not None and s2p_filename is not None:
        extra_dl = ("📥 modeled S2P", s2p_bytes, s2p_filename, "text/plain")
    col_l, col_r = st.columns([1.05, 1])
    with col_l:
        render_smith_chart(S_raw, S_sim, model_name, err, scales,
                           key=f"smith_{model_short}_{fname}",
                           extra_download=extra_dl)
    with col_r:
        render_ft_fmax_card(S_raw, S_sim, freq,
                            model_name=model_name,
                            key=f"ftfmax_card_{model_short}_{fname}",
                            height=560)


def smith_scale_controls(fname, topo_key) -> dict:
    st.markdown("<small>**S-param display scale** — multiply before plotting "
                "(does not affect residual)</small>", unsafe_allow_html=True)
    sc = {}
    for col_w, name, default in zip(st.columns(4),
                                    ["S11","S12","S21","S22"],
                                    [1.0,   1.0,   1.0,   1.0]):
        sk = f"smith_scale_{topo_key}_{name}_{fname}"
        if sk not in st.session_state:
            st.session_state[sk] = default
        # NOTE: do NOT pass `value=` alongside `key=` when the key is
        # already initialized in session_state — Streamlit's
        # check_session_state_rules logs a warning ("created via
        # st.session_state.X and value parameter").  The widget reads
        # the current value from session_state via the key alone.
        sc[name] = col_w.number_input(f"{name} ×",
                                       min_value=0.01, max_value=1000.0,
                                       step=0.5, format="%.2f", key=sk)
    return sc


# ── Pad sync (call once per model's render_override_and_smith) ────────────────

def sync_pad_from_preov(fname: str, topo_key: str, para_eff: dict):
    """
    Copy current pre-extraction pad values into the per-topology session state
    only when the pre-extraction values have changed (detected via MD5 hash).
    Prevents stale pad values in the Smith chart override form.
    """
    preov_hash  = params_hash({k: para_eff.get(k, 0.0) for k in _PAD_KEYS})
    sync_key    = f"smith_pad_synced_{topo_key}_{fname}"
    if st.session_state.get(sync_key) != preov_hash:
        for key, _, scale, *_ in PAD_SPECS:
            st.session_state[f"sim_{topo_key}_{key}_{fname}"] = float(para_eff.get(key, 0.0)) * scale
        st.session_state[sync_key] = preov_hash

def _render_cbex_sweep_tool(*, cbex_arr, freq, f_ghz, f_min_v, f_max_v,
                            cbex_scale, cbex_unit, cbex_param_key,
                            model_short, fname, g_idx, rng_tag,
                            cbex_sweep_fn, param_groups):
    """Render the Min / Step / Max inputs + Calculate button for the Cbex
    sweep.  On button click: generate candidate Cbex values, call
    `cbex_sweep_fn(cbex_SI_array, mask)` to get per-candidate std(Cbcx_arr),
    pick the argmin, and write the best value into the Cbex number_input's
    session-state key so the widget below renders the new value on the next
    natural rerun.

    The std window (mask) comes from the Cbcx group's slider session state
    if the user has already moved it, otherwise the full frequency range.
    """
    # Defaults: min = min|Cbex_arr| (fF), max = max|Cbex_arr| (fF), step = 10 fF
    fin = cbex_arr[np.isfinite(cbex_arr)]
    if len(fin) == 0:
        st.caption("Cbex sweep unavailable — no finite Cbex samples.")
        return
    abs_disp = np.abs(fin) * cbex_scale
    default_min = float(np.min(abs_disp))
    default_max = float(np.max(abs_disp))
    if default_max <= default_min:
        default_max = default_min + 1.0
    default_step = 0.1

    # Per-group state keys — tied to rng_tag so defaults refresh on slider move
    sweep_key_base = f"cbex_sweep_{model_short}_{fname}_{rng_tag}"
    k_min  = f"{sweep_key_base}_min"
    k_step = f"{sweep_key_base}_step"
    k_max  = f"{sweep_key_base}_max"
    k_res  = f"cbex_sweep_result_{model_short}_{fname}"  # persists across reruns

    st.markdown("---")
    st.markdown("**🔍 Cbex sweep — minimise std(Cbcx)**")

    c_min, c_step, c_max = st.columns(3)
    sweep_min = c_min.number_input(
        f"Min ({cbex_unit})",
        value=float(st.session_state.get(k_min, default_min)),
        min_value=0.0, format="%.4f", key=k_min)
    sweep_step = c_step.number_input(
        f"Step ({cbex_unit})",
        value=float(st.session_state.get(k_step, default_step)),
        min_value=1e-6, format="%.4f", key=k_step)
    sweep_max = c_max.number_input(
        f"Max ({cbex_unit})",
        value=float(st.session_state.get(k_max, default_max)),
        min_value=0.0, format="%.4f", key=k_max)
    run_sweep = st.button(
        "Calculate",
        key=f"{sweep_key_base}_btn",
        width="stretch")

    if run_sweep:
        if sweep_max < sweep_min or sweep_step <= 0:
            st.error("Sweep range is invalid: need max ≥ min and step > 0.")
        else:
            # Build candidate array in display units, then convert to SI
            n_pts = int(np.floor((sweep_max - sweep_min) / sweep_step)) + 1
            n_pts = max(2, min(n_pts, 10000))  # sanity cap
            cand_disp = sweep_min + np.arange(n_pts) * sweep_step
            cand_disp = cand_disp[cand_disp <= sweep_max + 1e-12]
            cand_SI = cand_disp / float(cbex_scale)

            # Mask: use the Cbcx group's slider range if the user has set it,
            # otherwise the full freq range.  Find the Cbcx group by searching
            # param_groups for one whose params reference "Cbcx_arr" (was
            # `g_idx + 1` but Ccex now sits between Cbex and Cbcx for ChengT).
            cbcx_g_idx = next(
                (i for i, g in enumerate(param_groups)
                 if any(spec[0] == "Cbcx_arr" for spec in g.get("params", []))),
                g_idx + 1,
            )
            cbcx_sl_key = f"pfp_sl_{model_short}_{cbcx_g_idx}_{fname}"
            cbcx_range = st.session_state.get(cbcx_sl_key, (f_min_v, f_max_v))
            try:
                f_lo_cbcx, f_hi_cbcx = float(cbcx_range[0]), float(cbcx_range[1])
            except Exception:
                f_lo_cbcx, f_hi_cbcx = f_min_v, f_max_v
            mask = (f_ghz >= f_lo_cbcx) & (f_ghz <= f_hi_cbcx)
            if not mask.any():
                mask = np.ones_like(f_ghz, dtype=bool)

            try:
                stds = cbex_sweep_fn(cand_SI, mask)
            except Exception as exc:
                st.error(f"Sweep failed: {exc!r}")
                return
            stds = np.asarray(stds, dtype=float)
            if not np.any(np.isfinite(stds)):
                st.error("All candidates produced non-finite std(Cbcx) — "
                         "try a different range.")
                return
            best_k = int(np.nanargmin(stds))
            best_cbex_SI = float(cand_SI[best_k])
            best_cbex_disp = best_cbex_SI * float(cbex_scale)
            best_std = float(stds[best_k])

            # Stage the result in a "pending" key.  We CAN'T write directly
            # to the number_input's session-state key here because that
            # widget has already been instantiated earlier in this same run
            # (it lives in cols[0] of the same row).  On the next rerun the
            # plot loop will see this pending value and copy it into the
            # widget key BEFORE the widget is created — see the
            # `pfp_pending_*` lookup just above the number_input render.
            pending_key = (f"pfp_pending_{model_short}_{cbex_param_key}_"
                           f"{fname}")
            st.session_state[pending_key] = float(best_cbex_disp)

            # Persist the message so it survives the rerun
            st.session_state[k_res] = {
                "best_disp": best_cbex_disp,
                "best_std":  best_std,
                "n_pts":     len(cand_SI),
                "f_lo":      f_lo_cbcx,
                "f_hi":      f_hi_cbcx,
                "rng_tag":   rng_tag,
            }
            st.rerun()

    # Show the last sweep result (if any) — scoped to this rng_tag so it
    # clears when the Cbex slider is moved.
    last = st.session_state.get(k_res)
    if last and last.get("rng_tag") == rng_tag:
        st.success(
            f"Best Cbex = **{last['best_disp']:.4f} {cbex_unit}**  "
            f"(std(Cbcx) = {last['best_std']:.3e}, "
            f"{last['n_pts']} candidates, "
            f"Cbcx window {last['f_lo']:.2f}–{last['f_hi']:.2f} GHz)")


def _render_tau_total_fit_section(*, all_data, fname, model_short,
                                  params, para_eff):
    """
    Multi-file 1/(2πfT) vs 1/IC linear fit (T-model reference only).

    For each bias file in ``all_data``, computes τ_total = 1/(2π f_T) from
    the de-embedded |h21|² 0-dB crossing.  Plots τ_total (ps) versus 1/IC
    (1/mA), linear-fits, and reports:
      - Cje (from slope):  slope = (η kT/q) · CJE  →  CJE = slope / (η · Vt).
        (Cbc contribution to slope is neglected per Cheng et al., paper Eq. 1.)
      - τB + τC (from intercept): intercept − (RC + REE) · CBC.
      - Per-file τCC = (rE + REE + RC) · CBC and τE = rE · CJE,
        with rE = η kT / (q IC).

    References:
      - Equation (1) of K. Y. D. Cheng et al., "Hot electron injection
        effect on the microwave performance of type-I/II AlInP/GaAsSb/InP
        DHBTs" — supplies the total-delay formula.  Paper notation:
        REE/RC/rE → code: Rpe/Rpc/(ηkT/qIC).
      - H. G. Liu, N. Tao, S. P. Watkins, C. R. Bolognesi, "Extraction
        of the Average Collector Velocity in High-Speed Type-II
        InP–GaAsSb–InP DHBTs," IEEE EDL 25(12), 2004 — supplies the
        v_c = W_C/(2 τ_C) split with default v_c = 4×10⁷ cm/s (peak
        across a 2000 Å InP collector).

    Hidden when ``all_data`` has < 2 files (the fit needs ≥ 2 bias points).
    The extracted Cje is for reference only — it does NOT feed back into
    the model's own extracted parameters.
    """
    import pandas as pd
    from pathlib import Path
    from ..helpers import peel_parasitics, compute_metrics, extract_limit

    if not all_data or len(all_data) < 2:
        return False

    with st.expander("📐 Cje / τB+τC / τCC / τE from 1/(2πfT) vs 1/IC fit  "
                     "(T-model reference)",
                     expanded=False):
        st.caption(
            "Reference extraction (extracted values do NOT feed back into "
            "the model fit). Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12), 2004 'Extraction of the average collector velocity in high-speed Type-II InP-GaAsSb-InP_DHBTs.pdf'")
        st.latex(
            r"\frac{1}{2\pi f_T}=\tau_B+\tau_C+\frac{\eta k T}{q I_C}\,C_{JE}"
            r"+\left(R_C+R_{EE}+\frac{\eta k T}{q I_C}\right)C_{BC}")
        st.caption(
            "Notation: REE → emitter access resistance (Rpe here); "
            "RC → collector access (Rpc here); rE = ηkT/(qIC) → intrinsic "
            "base-emitter resistance.")

        # ── fT per file (Open+Short de-embedded, access R RETAINED) ─────────
        # The Cheng formula's RC/REE refer to the access resistance that the
        # device sees at the fT-measurement plane.  If we used the same
        # `para_eff` that the model extraction uses, ``peel_parasitics`` would
        # also strip Rpb/Rpc/Rpe (when sourced from Z-param / open-collector
        # / Cold-HBT) — that puts the fT plane past the access R and the
        # (RC+REE)·CBC intercept correction over-subtracts.
        #
        # Smart behavior: zero out Rpb/Rpc/Rpe before peeling.  When the user
        # has NOT entered any pad caps / lead L in the previous section
        # (because the files are already pre-de-embedded), every C and L in
        # ``para_eff`` is zero — and peel_parasitics becomes an algebraic
        # no-op (Y_pad=0, Z_ser=0 → returns Y_dut unchanged).  Otherwise it
        # peels only the caps and leads, exactly as requested.
        _para_pad_lead_only = dict(para_eff)
        _para_pad_lead_only["Rpb"] = 0.0
        _para_pad_lead_only["Rpc"] = 0.0
        _para_pad_lead_only["Rpe"] = 0.0

        recs = []
        for fn, d in all_data.items():
            try:
                Y     = peel_parasitics(d["S_raw"], d["freq"], d["z0"],
                                        _para_pad_lead_only)
                dfm   = compute_metrics(Y, d["freq"])
                f_ghz = dfm["Freq (GHz)"].to_numpy()
                fT_v, _, _ = extract_limit(
                    f_ghz, dfm["|h21|² (dB)"].to_numpy(),
                    dfm["fT Plateau (GHz)"].to_numpy(),
                    n_pts=2,
                    f_min=float(f_ghz[0]), f_max=float(f_ghz[-1]))
                fT_GHz   = float(fT_v) if np.isfinite(fT_v) else np.nan
                tau_tot  = (1.0 / (2.0 * np.pi * fT_GHz * 1e9)
                            if (np.isfinite(fT_GHz) and fT_GHz > 0) else np.nan)
                recs.append({"fn": fn, "stem": Path(fn).stem,
                             "fT_GHz": fT_GHz, "tau_s": tau_tot})
            except Exception as ex:
                recs.append({"fn": fn, "stem": Path(fn).stem,
                             "fT_GHz": np.nan, "tau_s": np.nan,
                             "err": str(ex)})

        # Sort by fT descending (matches Z-param method's "sort by extracted-
        # quantity desc" convention — higher fT files appear first).
        recs.sort(key=lambda r: (r["fT_GHz"] if np.isfinite(r["fT_GHz"])
                                  else -np.inf),
                  reverse=True)

        # st.markdown(
        #     "**Files (fT measured after Open+Short pad/lead de-embedding "
        #     "— access R RETAINED so RC/REE in the formula remain meaningful; "
        #     "no-op when the file is already pre-de-embedded):**")
        hcols = st.columns([0.3, 2.0, 1.2, 1.4, 1.4])
        for h, t in zip(hcols, ["", "File", "fT (GHz)",
                                  "τ_total (ps)", "IC (mA)"]):
            h.markdown(f"<small><b>{t}</b></small>", unsafe_allow_html=True)

        # Per-file checkbox + IC input.  Seed IC from rz12_Ie_{fn} (Z-param's
        # IE input) since IE ≈ IC in normal HBT operation; the user can refine.
        points = []   # list of (1/IC[1/mA], τ_total[ps], stem, IC_mA)
        for r in recs:
            c0, c1, c2, c3, c4 = st.columns([0.3, 2.0, 1.2, 1.4, 1.4])
            use_key = f"taut_use_{r['fn']}__{fname}__{model_short}"
            if use_key not in st.session_state:
                st.session_state[use_key] = True
            use = c0.checkbox(f"Use {r['stem']}",
                              key=use_key + "_w",
                              value=st.session_state[use_key],
                              label_visibility="collapsed")
            st.session_state[use_key] = use

            c1.markdown(f"<small>{r['stem']}</small>", unsafe_allow_html=True)
            c2.markdown(
                (f"<small>{r['fT_GHz']:.3f}</small>"
                 if np.isfinite(r["fT_GHz"]) else "<small>—</small>"),
                unsafe_allow_html=True)
            c3.markdown(
                (f"<small>{r['tau_s']*1e12:.4f}</small>"
                 if np.isfinite(r["tau_s"]) else "<small>—</small>"),
                unsafe_allow_html=True)

            ic_key = f"taut_Ic_{r['fn']}__{fname}__{model_short}"
            if ic_key not in st.session_state:
                # Default to Z-param Ie (≈ Ic in normal mode), else 0.
                st.session_state[ic_key] = float(
                    st.session_state.get(f"rz12_Ie_{r['fn']}", 0.0))
            ic_mA = c4.number_input(
                f"IC for {r['stem']}",
                min_value=0.0, step=0.1, format="%.3f",
                value=float(st.session_state[ic_key]),
                key=ic_key + "_w",
                label_visibility="collapsed")
            st.session_state[ic_key] = ic_mA

            if use and ic_mA > 0 and np.isfinite(r["tau_s"]):
                points.append((1.0 / ic_mA,
                                r["tau_s"] * 1e12,
                                r["stem"], ic_mA))

        if len(points) < 2:
            st.info("Enter IC for at least two enabled files to fit.")
            return

        x = np.array([p[0] for p in points])     # 1/IC (1/mA)
        y = np.array([p[1] for p in points])     # τ_total (ps)
        lbls = [p[2] for p in points]
        try:
            slope, intercept = np.polyfit(x, y, 1)    # slope: ps·mA, int: ps
        except Exception as ex:
            st.error(f"Linear fit failed: {ex}")
            return

        # ── Plot ──────────────────────────────────────────────────────────────
        x_fit = np.linspace(0.0, float(x.max() * 1.08), 200)
        y_fit = slope * x_fit + intercept
        fig = go.Figure()
        # Trace names become Excel sheet names in the xlsx download, so they
        # must avoid characters Excel forbids in sheet titles: / \ ? * [ ]
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers+text", text=lbls,
            textposition="top center", name="τ_total",
            marker=dict(size=11, color="#1f77b4",
                        line=dict(color="#0d4a7a", width=1.5))))
        fig.add_trace(go.Scattergl(
            x=x_fit, y=y_fit, mode="lines",
            name=f"Fit  slope={slope:.4g} ps·mA   int={intercept:.4g} ps",
            line=dict(color="#d62728", width=2, dash="dash")))
        fig.add_trace(go.Scattergl(
            x=[0.0], y=[intercept], mode="markers",
            name=f"Intercept = {intercept:.4f} ps",
            marker=dict(size=14, symbol="star", color="#d62728")))
        fig.update_layout(
            title=f"1/(2π f_T) vs 1/I_C  —  {model_short} model (reference)",
            xaxis=dict(title="1/I_C (1/mA)", rangemode="tozero",
                       showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(title="τ_total = 1/(2π f_T) (ps)",
                       showgrid=True, gridcolor="#ebebeb"),
            plot_bgcolor="white", paper_bgcolor="white", height=380,
            legend=dict(x=0.45, y=0.05,
                        xanchor="left", yanchor="bottom",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#ccc", borderwidth=1,
                        font=dict(size=10)),
            margin=dict(l=55, r=20, t=50, b=50))
        plotly_with_dl(fig,
                       key=f"taut_fit_{model_short}_{fname}",
                       filename=f"taut_fit_{model_short}_{fname}")

        # ── Inputs (defaults from current file's extraction) ─────────────────
        st.markdown("**Inputs (defaults from this file's extraction):**")
        Re_def  = float(para_eff.get("Rpe", 0.0))
        Rc_def  = float(para_eff.get("Rpc", 0.0))
        Cbc_def = (float(params.get("Cbc",  0.0))
                   + float(params.get("Cbcx", 0.0)))   # total = intrinsic + extrinsic

        ci1, ci2, ci3, ci4, ci5 = st.columns(5)
        Re_val = ci1.number_input(
            "Re — REE (Ω)", min_value=0.0, value=Re_def, format="%.4f",
            key=f"taut_Re_{model_short}_{fname}",
            help="Emitter access resistance.  Default = Rpe used in extraction.")
        Rc_val = ci2.number_input(
            "Rc (Ω)", min_value=0.0, value=Rc_def, format="%.4f",
            key=f"taut_Rc_{model_short}_{fname}",
            help="Collector access resistance.  Default = Rpc used in extraction.")
        Cbc_val_fF = ci3.number_input(
            "Cbc total (fF)", min_value=0.0,
            value=Cbc_def * 1e15, format="%.4f",
            key=f"taut_Cbc_{model_short}_{fname}",
            help="Total base-collector cap.  Default = Cbc + Cbcx (intrinsic + extrinsic).")
        eta_val = ci4.number_input(
            "η (ideality)", min_value=0.5, max_value=3.0,
            value=1.0, step=0.05, format="%.3f",
            key=f"taut_eta_{model_short}_{fname}",
            help="Ideality factor for r_E = η kT/(q IC).  Set this from a "
                 "Gummel-plot fit of your device (typical InP HBT: 1.0–1.2).")
        T_K = ci5.number_input(
            "T (K)", min_value=1.0, value=300.0, step=5.0, format="%.1f",
            key=f"taut_T_{model_short}_{fname}",
            help="Temperature for kT/q.")

        Cbc_val = Cbc_val_fF * 1e-15
        Vt = 1.380649e-23 * T_K / 1.602176634e-19         # kT/q  (V)

        # ── Derived (slope → Cje; intercept → τB+τC) ─────────────────────────
        # Units conversion: slope is in ps·mA = (s·1e-12)·(A·1e-3) = s·A · 1e-15.
        # In SI, slope_SI = (η · Vt) · Cje  with  [V · F] = [s · A].
        # ⇒ Cje[F] = slope_SI / (η · Vt) = slope[ps·mA] · 1e-15 / (η · Vt).
        # ⇒ Cje[fF] = slope[ps·mA] / (η · Vt[V]).
        Cje_fF = (slope / (eta_val * Vt)) if (eta_val * Vt) > 0 else 0.0
        Cje_F  = Cje_fF * 1e-15

        tau_BC_ps = intercept - (Re_val + Rc_val) * Cbc_val * 1e12

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Slope", f"{slope:.4g} ps·mA",
                  help="d(τ_total)/d(1/I_C) — drives Cje.")
        m2.metric("Intercept", f"{intercept:.4f} ps",
                  help="τ_total extrapolated to 1/I_C → 0.")
        m3.metric("Cje  (ref.)", f"{Cje_fF:.4f} fF",
                  help="Cje = slope / (η · kT/q).  Reference only.")
        m4.metric("τB + τC  (ref.)", f"{tau_BC_ps:.4f} ps",
                  help="τB+τC = intercept − (Rc + Re) · Cbc.")

        # ── Split τB / τC using assumed collector velocity v_c ───────────────
        # Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12), 2004 — "Extraction
        # of the Average Collector Velocity in High-Speed Type-II
        # InP–GaAsSb–InP DHBTs" — found v_c peaks at 4×10⁷ cm/s across a
        # 2000 Å InP collector at V_CB ≈ 0.4 V.  Using the same definition
        # v_c = W_C / (2 τ_C):
        #     τ_C = W_C / (2 v_c)
        #     τ_B = (τ_B + τ_C)_intercept − τ_C
        # Default v_c is the Liu peak for InP collectors; user can edit for
        # other collector materials / thicknesses / biases.  Publishes to
        # session state so the τB / τC number_inputs farther down offer a
        # "v_c = …" quickset button.
        st.markdown("**Split τB / τC using assumed average collector velocity "
                    "(Liu et al. 2004 — default for InP collector):**")
        cv1, cv2, cv3, cv4 = st.columns(4)
        Wc_nm = cv1.number_input(
            "W_C (nm)", min_value=1.0, value=120.0, step=10.0, format="%.2f",
            key=f"taut_Wc_{model_short}_{fname}",
            help="Collector depletion width.")
        v_c_cms = cv2.number_input(
            "v_c (cm/s)", min_value=1.0e5, value=4.0e7,
            step=1.0e6, format="%.3e",
            key=f"taut_vc_{model_short}_{fname}",
            help="Average collector velocity.  Default 4×10⁷ cm/s — peak "
                 "value extracted for 2000 Å InP collectors in Liu, Tao, "
                 "Watkins, Bolognesi, IEEE EDL 25(12), 2004.  Adjust for "
                 "other collector materials / thicknesses / biases.")
        v_c_ms     = v_c_cms * 1e-2                      # cm/s → m/s
        Wc_m       = Wc_nm * 1e-9
        tauC_vc_s  = Wc_m / (2.0 * v_c_ms)               # seconds
        tauC_vc_ps = tauC_vc_s * 1e12
        tauB_vc_ps = tau_BC_ps - tauC_vc_ps              # ps
        tauB_vc_s  = tauB_vc_ps * 1e-12

        cv3.metric("τC  (from v_c)", f"{tauC_vc_ps:.4f} ps",
                   help="τ_C = W_C / (2 v_c).")
        cv4.metric("τB  (from v_c)", f"{tauB_vc_ps:.4f} ps",
                   help="τ_B = (τ_B+τ_C) − τ_C.")

        # Publish v_c-derived values (SI seconds) so τB / τC number_inputs
        # can read them via the "v_c = …" quickset button.  Stored only when
        # finite; deleted otherwise so the button auto-hides when the fit
        # degrades.
        for pub_key, val in (
            (f"taut_pub_tauB_{model_short}_{fname}", tauB_vc_s),
            (f"taut_pub_tauC_{model_short}_{fname}", tauC_vc_s),
        ):
            if np.isfinite(val) and abs(val) > 0:
                st.session_state[pub_key] = float(val)
            else:
                st.session_state.pop(pub_key, None)

        # ── Per-file τCC and τE ─────────────────────────────────────────────
        rows = []
        for _, tau_total_ps, stem, ic_mA in points:
            ic_A   = ic_mA * 1e-3
            rE_i   = (eta_val * Vt) / ic_A if ic_A > 0 else np.nan
            tau_cc = (rE_i + Re_val + Rc_val) * Cbc_val * 1e12  # ps
            tau_E  = rE_i * Cje_F * 1e12                         # ps
            rows.append({
                "File":              stem,
                "IC (mA)":           f"{ic_mA:.4f}",
                "rE = ηkT/(qIC) (Ω)": f"{rE_i:.4f}",
                "τCC = (rE+Re+Rc)·Cbc (ps)": f"{tau_cc:.4f}",
                "τE = rE·Cje (ps)":  f"{tau_E:.4f}",
                "τ_total measured (ps)": f"{tau_total_ps:.4f}",
            })
        st.markdown("**Per-file derived delays (using inputs above):**")
        st.dataframe(pd.DataFrame(rows),
                     width="stretch", hide_index=True)
    return True


# Component grouping for the diagram-mode fine-tune editor (the "✏️ Fine-tune"
# override expander).  Keys are matched against each model's spec list; any spec
# key not named here lands in a trailing "Other" group, so nothing is hidden.
_FINETUNE_DIAGRAM_GROUPS = [
    ("Pad parasitics",    ["Cpbe", "Cpbc", "Cpce", "Cgsp", "Cdsp", "Cgdp",
                           "Rsub1", "Rsub2"]),
    ("Lead inductance",   ["Lb", "Lc", "Le"]),
    ("Access resistance", ["Rpb", "Rpc", "Rpe"]),
    ("Extrinsic C",       ["Cbex", "Cbcx", "Rbcx"]),
    ("Delay",             ["tauB", "tauC", "tau", "R_delay", "C_delay"]),
    ("Intrinsic",         ["Rbi", "Rbe", "Cbe", "Cbc", "Rbc", "alpha0", "Gm0"]),
]


def render_finetune_diagram(*, all_specs, state_key_for, active_state,
                            calc_vals, render_illustration):
    """Diagram-mode alternative for a model's "✏️ Fine-tune" override expander.

    Two columns:
      • Left  — the model topology schematic, with the component the user last
        edited ringed in red (via ``render_illustration(preview, highlight)``).
      • Right — every override value, grouped (parasitics / lead L / access R /
        extrinsic C / delay / intrinsic).  Each ``number_input`` writes the SAME
        session-state key the list view uses, so the two modes stay in lock-step
        and the caller's downstream parameter assembly is unchanged.  Editing a
        field also moves the highlight to that component.

    Parameters
    ----------
    all_specs        : list of ``(key, label, scale, unit, fmt, step)``.
    state_key_for    : ``f(param_key) -> session_state key`` — lets each caller
                       map a param to its own widget-key scheme (SSM extraction
                       uses ``sim_{topo}_{key}_{fname}``; the RF simulator uses
                       ``rfsim_{prefix}_{pad|ext|int}_{key}``).
    active_state     : session_state key holding the highlighted param.
    calc_vals        : SI defaults used to seed an unset widget (``{}`` ⇒ 0).
    render_illustration : ``f(preview_all_p_SI, highlight_key) -> None``.
    """
    spec_lookup = {s[0]: s for s in all_specs}
    ordered: list[str] = []
    for _lbl, keys in _FINETUNE_DIAGRAM_GROUPS:
        ordered += [k for k in keys if k in spec_lookup]
    groups = [(lbl, [k for k in keys if k in spec_lookup])
              for lbl, keys in _FINETUNE_DIAGRAM_GROUPS]
    other = [s[0] for s in all_specs if s[0] not in ordered]
    if other:
        groups.append(("Other", other))

    col_diag, col_inp = st.columns([1.1, 1], gap="medium")

    with col_inp:
        st.caption("Edit any value — the diagram highlights the component you "
                   "last changed.")
        for g_label, keys in groups:
            if not keys:
                continue
            st.markdown(f"**{g_label}**")
            for i in range(0, len(keys), 2):
                cols = st.columns(2)
                for col_w, key in zip(cols, keys[i:i + 2]):
                    _k, lbl, sc, unit, fmt, step = spec_lookup[key]
                    skey = state_key_for(key)
                    if skey not in st.session_state:
                        st.session_state[skey] = float(calc_vals.get(key, 0.0)) * sc

                    def _mk(_key=key):
                        def _cb():
                            st.session_state[active_state] = _key
                        return _cb

                    col_w.number_input(
                        f"{lbl} ({unit})" if unit else lbl,
                        key=skey, format=fmt, step=step, on_change=_mk())

    with col_diag:
        active = st.session_state.get(active_state)
        preview = {
            s[0]: float(st.session_state.get(
                state_key_for(s[0]),
                float(calc_vals.get(s[0], 0.0)) * s[2])) / s[2]
            for s in all_specs}
        render_illustration(preview, active)
        if active:
            st.caption(f"Editing **{active}**")


def render_interactive_param_groups(params, arrays, freq, fname, model_short, param_groups,
                                    cold_res=None, cold_param_map=None, reextract_fn=None,
                                    cbex_sweep_fn=None,
                                    all_data=None, para_eff=None):

    """
    For each parameter group, render:
      - A labelled section heading with dependency info
      - A "same range as previous" button (for dependent groups)
      - A frequency range slider
      - Per-frequency line plots with a dashed line at the current median
      - A number_input per parameter for manual override
      - (Groups flagged `cbex_sweep_group` only) a Cbex sweep tool that
        searches for the Cbex value that minimises std(Cbcx_arr) across
        the Cbcx-group's selected frequency window.
      - (Groups flagged `tau_total_fit_group` only) the multi-file
        1/(2πfT) vs 1/IC reference fit — auto-hidden when only one s2p
        file is loaded.  Needs ``all_data`` and ``para_eff`` to be passed
        through; otherwise the group is silently skipped.
    Returns a copy of params with all overrides applied (SI units).
    """
    
    f_ghz   = freq * 1e-9
    f_min_v = float(f_ghz[0])
    f_max_v = float(f_ghz[-1])
    step_v  = max(round((f_max_v - f_min_v) / 100, 3), 0.001)

    params_out = dict(params)
    live_arrays = dict(arrays)   # updated mid-loop after re-extraction
    live_params = dict(params)   # updated mid-loop; used for change detection
    prev_range = (f_min_v, f_max_v)

    with st.expander("📊 Interactive Parameter Extraction", expanded=False):
        # Thick outline on the large per-group boxes so each parameter group
        # reads as a bold card, set apart from the thin per-parameter
        # sub-containers inside it.  Streamlit puts the user `key` class on the
        # inner `stVerticalBlock` element (NOT the border wrapper), so we draw
        # the border directly on that keyed element and disable Streamlit's own
        # `border=True` wrapper (below) to avoid a double frame.  Scoped to the
        # `st-key-pfp_groupbox_*` prefix, so no other container is affected.
        st.markdown(
            """<style>
            div[class*="st-key-pfp_groupbox_"] {
                border: 3px solid #9aa0a6 !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem !important;
            }
            </style>""",
            unsafe_allow_html=True)
        for g_idx, group in enumerate(param_groups):
            g_label  = group["label"]
            g_params = group["params"]
            g_deps   = group.get("depends_on", [])

            # ── tau_total_fit_group: multi-file 1/(2πfT) vs 1/IC reference ──
            # Rendered as its own nested expander, so no leading heading.
            # Silently skips when called from a single-file context or
            # when the caller didn't pass all_data/para_eff.
            if group.get("tau_total_fit_group"):
                _tau_rendered = False
                if all_data is not None and para_eff is not None:
                    _tau_rendered = bool(_render_tau_total_fit_section(
                        all_data=all_data, fname=fname,
                        model_short=model_short,
                        params=live_params, para_eff=para_eff))
                # Only emit the trailing separator when the section actually
                # rendered.  With a single s2p file the fit is hidden, so
                # skipping the rule avoids a double "---" (the previous group
                # already drew one) showing as two empty lines.
                if _tau_rendered and g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # Heading: special fit groups (Z-plots / Fbi / F1) print it here;
            # normal parameter groups print it inside their bordered box below.
            _is_special_group = any(group.get(_k) for _k in
                                    ("z_plots_group", "fbi_fit_group",
                                     "f1_fit_group"))
            if _is_special_group:
                st.markdown(f"**{g_label}**")

            # ── z_plots_group: Z1, Z3, Z4 Re/Im plots ───────────────────────────
            if group.get("z_plots_group"):
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = (f_min_v, f_max_v)
                f_lo, f_hi = st.slider(
                    "Frequency range (GHz)",
                    min_value=f_min_v, max_value=f_max_v,
                    value=st.session_state[sl_key],
                    step=step_v, format="%.2f", key=sl_key)
                mask   = (f_ghz >= f_lo) & (f_ghz <= f_hi)
                f_plot = f_ghz[mask]

                for zlabel, zarr in [("Z1", live_arrays.get("Z1")),
                                      ("Z3", live_arrays.get("Z3")),
                                      ("Z4", live_arrays.get("Z4"))]:
                    if zarr is None:
                        continue
                    c1, c2 = st.columns(2)
                    for col_w, part_fn, part_lbl in [(c1, np.real, "Re"),
                                                      (c2, np.imag, "Im")]:
                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(
                            x=f_plot, y=part_fn(zarr[mask]), mode="lines",
                            line=dict(color="#1f77b4", width=2)))
                        fig.update_layout(
                            title=dict(text=f"{part_lbl}({zlabel})", font=dict(size=12)),
                            xaxis_title="Frequency (GHz)",
                            yaxis_title=f"{part_lbl}({zlabel}) (Ω)",
                            plot_bgcolor="white", paper_bgcolor="white", height=220,
                            margin=dict(l=50, r=20, t=35, b=40), showlegend=False)
                        fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                        fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                        plotly_with_dl(fig,
                                       key=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}",
                                       filename=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}",
                                       container=col_w)

                prev_range = (f_lo, f_hi)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # ── fbi_fit_group: Fbi vs ω², single fit-window slider ───────────────
            if group.get("fbi_fit_group"):
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_fbi_sl_{model_short}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = f_max_v
                f_hi_fbi = st.slider(
                    "Fbi linear fit upper frequency (GHz)",
                    min_value=f_min_v, max_value=f_max_v,
                    value=float(st.session_state[sl_key]),
                    step=step_v, format="%.2f", key=sl_key)

                n_fit_new  = int(np.sum(f_ghz <= f_hi_fbi))
                n_fit_key  = f"pfp_nfit_{model_short}_{fname}"
                n_fit_prev = st.session_state.get(n_fit_key, n_fit_new)
                st.session_state[n_fit_key] = n_fit_new

                if n_fit_new != n_fit_prev and reextract_fn is not None:
                    params_out["_n_fit"] = n_fit_new
                    try:
                        _new_p, _new_a = reextract_fn(params_out, 1, live_arrays)
                        live_params.update(_new_p)
                        params_out.update(_new_p)
                        live_arrays.update(_new_a)
                    except Exception:
                        pass
                    params_out.pop("_n_fit", None)

                omega2  = live_arrays.get("omega2")
                Fbi_arr = live_arrays.get("Fbi")
                A0      = live_params.get("A0", 0.0)
                B0      = live_params.get("B0", 0.0)

                if Fbi_arr is not None and omega2 is not None:
                    valid_mask = (np.isfinite(Fbi_arr) & (np.abs(Fbi_arr) < 1e15)
                                  & (Fbi_arr > 0))
                    win_mask   = valid_mask & (f_ghz <= f_hi_fbi)
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=omega2[valid_mask], y=Fbi_arr[valid_mask], mode="markers",
                        name="All data", marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scattergl(
                        x=omega2[win_mask], y=Fbi_arr[win_mask], mode="markers",
                        name="Fit window", marker=dict(size=6, color="#1f77b4")))
                    if A0 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scattergl(
                            x=xf, y=A0 + B0 * xf, mode="lines",
                            name=f"Fit  A₀={A0:.3e}  B₀={B0:.3e}",
                            line=dict(color="#d62728", dash="dash", width=2)))
                    fig.update_layout(
                        title=dict(text="Fbi vs ω²  [Eq. 8]", font=dict(size=12)),
                        xaxis_title="ω² (rad²/s²)", yaxis_title="Fbi (rad/s)",
                        plot_bgcolor="white", paper_bgcolor="white", height=300,
                        margin=dict(l=55, r=10, t=40, b=42), showlegend=True,
                        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                    font=dict(size=9)))
                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                    plotly_with_dl(fig, key=f"pfp_fbi_{model_short}_{fname}",
                                   filename=f"pfp_fbi_{model_short}_{fname}")

                    Tbi_fit = float(np.sqrt(max(B0 / A0, 0.0))) if A0 > 1e-30 else 0.0
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("A₀", f"{A0:.4e}", help="Intercept of Fbi vs ω²")
                    mc2.metric("B₀", f"{B0:.4e}", help="Slope of Fbi vs ω²")
                    mc3.metric("Tbi = √(B₀/A₀)", f"{Tbi_fit*1e12:.4f} ps",
                               help="Intrinsic base time constant from fit")

                prev_range = (f_hi_fbi, f_hi_fbi)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # ── f1_fit_group: F1 vs ω² fit + Tbe number_input ───────────────────
            if group.get("f1_fit_group"):
                if g_deps:
                    st.caption(f"Depends on: {', '.join(g_deps)}")
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_f1_sl_{model_short}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = f_max_v
                f_hi_f1 = st.slider(
                    "F1 linear fit upper frequency (GHz)",
                    min_value=f_min_v, max_value=f_max_v,
                    value=float(st.session_state[sl_key]),
                    step=step_v, format="%.2f", key=sl_key)

                n_fit_f1_new = int(np.sum(f_ghz <= f_hi_f1))
                nf1_key      = f"pfp_nfit_f1_{model_short}_{fname}"
                nf1_prev     = st.session_state.get(nf1_key, n_fit_f1_new)
                st.session_state[nf1_key] = n_fit_f1_new

                if n_fit_f1_new != nf1_prev and reextract_fn is not None:
                    params_out["_n_fit_f1"] = n_fit_f1_new
                    try:
                        _new_p, _new_a = reextract_fn(params_out, 4, live_arrays)
                        live_params.update(_new_p)
                        params_out.update(_new_p)
                        live_arrays.update(_new_a)
                    except Exception:
                        pass
                    params_out.pop("_n_fit_f1", None)

                omega2 = live_arrays.get("omega2")
                F1_arr = live_arrays.get("F1")
                A1     = live_params.get("A1", 0.0)
                B1     = live_params.get("B1", 0.0)

                if F1_arr is not None and omega2 is not None:
                    valid_mask = (np.isfinite(F1_arr) & (np.abs(F1_arr) < 1e15)
                                  & (F1_arr > 0))
                    win_mask   = valid_mask & (f_ghz <= f_hi_f1)
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=omega2[valid_mask], y=F1_arr[valid_mask], mode="markers",
                        name="All data", marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scattergl(
                        x=omega2[win_mask], y=F1_arr[win_mask], mode="markers",
                        name="Fit window", marker=dict(size=6, color="#1f77b4")))
                    if A1 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scattergl(
                            x=xf, y=A1 + B1 * xf, mode="lines",
                            name=f"Fit  A={A1:.3e}  B={B1:.3e}",
                            line=dict(color="#d62728", dash="dash", width=2)))
                    fig.update_layout(
                        title=dict(text="F1 vs ω²  [Eq. 19]", font=dict(size=12)),
                        xaxis_title="ω² (rad²/s²)", yaxis_title="F1 (rad/s)",
                        plot_bgcolor="white", paper_bgcolor="white", height=300,
                        margin=dict(l=55, r=10, t=40, b=42), showlegend=True,
                        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                    font=dict(size=9)))
                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                    plotly_with_dl(fig, key=f"pfp_f1_{model_short}_{fname}",
                                   filename=f"pfp_f1_{model_short}_{fname}")

                    alpha   = 1.0 / A1 if A1 > 1e-30 else 0.0
                    Tbe_fit = float(np.sqrt(max(B1 / A1, 0.0))) if A1 > 1e-30 else 0.0
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("A", f"{A1:.4e}", help="Intercept of F1 vs ω²")
                    mc2.metric("B", f"{B1:.4e}", help="Slope of F1 vs ω²")
                    mc3.metric("α = 1/A", f"{alpha:.4e}", help="α = R(T − Tbe)")
                    mc4.metric("Tbe = √(B/A)", f"{Tbe_fit*1e12:.4f} ps",
                               help="Emitter time constant from fit")

                # Tbe number_input (user-overridable)
                _upstream_vals = tuple(
                    round(params_out.get(spec[1], 0.0) * 1e15)
                    for gi in range(g_idx)
                    for spec in param_groups[gi]["params"]
                )
                upstream_tag = str(hash(_upstream_vals) % (10 ** 9))
                rng_tag      = f"{f_hi_f1:.3f}"

                for arr_key, param_key, label, scale, unit in group.get("params", []):
                    if arr_key not in live_arrays:
                        continue
                    raw     = live_arrays[arr_key]
                    arr_num = np.abs(raw) if np.iscomplexobj(raw) else np.real(raw)
                    fin     = arr_num[np.isfinite(arr_num)]
                    auto_SI   = float(np.median(fin)) if len(fin) > 0 \
                                else float(params_out.get(param_key, 0.0))
                    auto_disp = auto_SI * scale
                    inp_key   = (f"pfp_inp_{model_short}_{param_key}_{fname}"
                                 f"_{rng_tag}_{upstream_tag}")
                    actual_val = st.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp), format="%.5g", key=inp_key)
                    params_out[param_key] = actual_val / scale

                # Re-extract downstream if Tbe changed
                if reextract_fn is not None:
                    _grp_pk  = {spec[1] for spec in group.get("params", [])}
                    _grp_ak  = {spec[0] for spec in group.get("params", [])}
                    _changed = any(
                        abs(params_out.get(pk, 0.0) - live_params.get(pk, 0.0))
                        > 1e-9 * (abs(live_params.get(pk, 0.0)) + 1e-30)
                        for pk in _grp_pk
                    )
                    if _changed:
                        try:
                            _new_p, _new_a = reextract_fn(params_out, g_idx, live_arrays)
                            _processed_pk = {
                                spec[1]
                                for gi in range(g_idx + 1)
                                for spec in param_groups[gi]["params"]
                            }
                            _processed_ak = {
                                spec[0]
                                for gi in range(g_idx + 1)
                                for spec in param_groups[gi]["params"]
                            }
                            for _k, _v in _new_p.items():
                                if _k not in _processed_pk:
                                    live_params[_k] = _v
                                    params_out[_k]  = _v
                            for _k, _v in _new_a.items():
                                if _k not in _processed_ak:
                                    live_arrays[_k] = _v
                        except Exception:
                            pass

                prev_range = (f_hi_f1, f_hi_f1)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # Wrap the whole parameter group (title + slider + per-parameter
            # plots) in one large card, so each group (Cbex, Cbcx, intrinsic,
            # τB, τC, …) reads as distinct.  `border=False` here — the thick
            # outline is drawn by the scoped `st-key-pfp_groupbox_*` CSS above
            # (Streamlit's own border wrapper would otherwise add a second,
            # thin frame).  All of this group's top-level widgets render into
            # `box`; nested per-parameter plots inherit it via `box.columns()`.
            box = st.container(
                border=False,
                key=f"pfp_groupbox_{model_short}_{g_idx}_{fname}")
            box.markdown(f"**{g_label}**")
            if g_deps:
                box.caption(f"Depends on: {', '.join(g_deps)}")

            slider_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"

            # "Use previous range" button for dependent groups
            if g_deps:
                if box.button(f"↩ Same range as previous group",
                             key=f"pfp_useprev_{model_short}_{g_idx}_{fname}"):
                    st.session_state[slider_key] = prev_range
                    st.rerun()

            # Render per-group formulas before the slider
            for formula_type, formula_content in group.get("formulas", []):
                if formula_type == "markdown":
                    box.markdown(formula_content)
                elif formula_type == "latex":
                    box.latex(formula_content)


            # Initialize slider — pre-seed session_state and DO NOT
            # pass `value=` alongside `key=`, otherwise Streamlit logs
            # `check_session_state_rules` warnings about a widget being
            # created with both a default value and a Session-State entry.
            # Also clamp any pre-existing value to the current [min, max]
            # bounds in case a previous file had a different freq range.
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (f_min_v, f_max_v)
            else:
                _cur = st.session_state[slider_key]
                try:
                    _lo = max(f_min_v, min(f_max_v, float(_cur[0])))
                    _hi = max(f_min_v, min(f_max_v, float(_cur[1])))
                    if _lo > _hi:
                        _lo, _hi = f_min_v, f_max_v
                    st.session_state[slider_key] = (_lo, _hi)
                except (TypeError, ValueError, IndexError):
                    st.session_state[slider_key] = (f_min_v, f_max_v)

            f_lo, f_hi = box.slider(
                "Frequency range (GHz)",
                min_value=f_min_v, max_value=f_max_v,
                step=step_v, format="%.2f",
                key=slider_key)

            mask   = (f_ghz >= f_lo) & (f_ghz <= f_hi)
            f_plot = f_ghz[mask]
            # Tag used in widget keys — changing it recreates inputs fresh on slider move
            rng_tag = f"{f_lo:.3f}_{f_hi:.3f}"

            valid_specs = [
                s for s in g_params
                if s[0] in live_arrays and isinstance(live_arrays[s[0]], np.ndarray)
            ]


            _is_cbex_sweep_group = (group.get("cbex_sweep_group")
                                    and cbex_sweep_fn is not None
                                    and "Cbex_arr" in live_arrays)
            _cbex_sweep_rendered = False

            for row_start in range(0, len(valid_specs), 2):
                row  = valid_specs[row_start:row_start + 2]
                cols = box.columns(2)
                for _col_outer, (arr_key, param_key, label, scale, unit) in zip(cols, row):
                    # Wrap each parameter's plot + input + quickset buttons in
                    # a bordered container so individual extracted parameters
                    # are visually distinct from each other within a group.
                    col_w = _col_outer.container(border=True)
                    raw_masked = live_arrays[arr_key][mask]
                    arr_plot   = (np.abs(raw_masked) if np.iscomplexobj(raw_masked)
                                  else np.real(raw_masked)) * scale

                    # Recompute median from current slider range
                    arr_num = np.abs(raw_masked) if np.iscomplexobj(raw_masked) else np.real(raw_masked)
                    fin     = arr_num[np.isfinite(arr_num)]
                    _use_first = param_key in group.get("use_first_params", set())
                    if _use_first:
                        auto_SI = float(fin[0]) if len(fin) > 0 else float(params.get(param_key, 0.0))
                    else:
                        auto_SI = float(np.median(fin)) if len(fin) > 0 else float(params.get(param_key, 0.0))

                    auto_disp = auto_SI * scale

                    # Pre-read session state so plot can use it before number_input renders
                    # Hash of all upstream groups' current values — changes when any
                    # upstream param is overridden, forcing downstream inputs to reset.
                    _upstream_vals = tuple(
                        round(params_out.get(spec[1], 0.0) * 1e15)
                        for gi in range(g_idx)
                        for spec in param_groups[gi]["params"]
                    )
                    upstream_tag = str(hash(_upstream_vals) % (10 ** 9))
                    inp_key   = f"pfp_inp_{model_short}_{param_key}_{fname}_{rng_tag}_{upstream_tag}"

                    # Transfer any pending override (e.g. from the Cbex
                    # sweep tool) into the widget key.  Must happen BEFORE
                    # the number_input is instantiated, or Streamlit raises
                    # "session_state ... cannot be modified after widget".
                    pending_key = f"pfp_pending_{model_short}_{param_key}_{fname}"
                    if pending_key in st.session_state:
                        st.session_state[inp_key] = st.session_state.pop(pending_key)
                    # Same for quickset-button writes
                    apply_pending(inp_key)

                    user_disp = float(st.session_state.get(inp_key, auto_disp))
                    user_SI   = user_disp / scale

                    ylabel = f"{label} ({unit})" if unit else label
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=f_plot, y=arr_plot, mode="lines", name=label,
                        line=dict(color="#1f77b4", width=2)))
                    if np.isfinite(user_disp):
                        fig.add_hline(
                            y=user_disp,
                            line=dict(color="#d62728", width=1.8, dash="dash"),
                            annotation_text=f"{user_disp:.4g} {unit}",
                            annotation_position="right",
                            annotation_font=dict(size=9, color="#d62728"))
                    # Cold-section value (computed once — used for both the
                    # plot annotation and the "cold = …" quickset button).
                    _cold_disp = None
                    if cold_res is not None and cold_param_map is not None:
                        _ck = cold_param_map.get(param_key)
                        if _ck and _ck in cold_res:
                            _v = float(cold_res[_ck]) * scale
                            if np.isfinite(_v):
                                _cold_disp = _v
                    if _cold_disp is not None:
                        fig.add_hline(
                            y=_cold_disp,
                            line=dict(color="#2ca02c", width=1.5, dash="dot"),
                            annotation_text=f"Cold: {_cold_disp:.4g} {unit}",
                            annotation_position="left",
                            annotation_font=dict(size=9, color="#2ca02c"))
                    fig.update_layout(
                        title=dict(text=label, font=dict(size=12)),
                        xaxis_title="Frequency (GHz)", yaxis_title=ylabel,
                        plot_bgcolor="white", paper_bgcolor="white", height=240,
                        margin=dict(l=50, r=60, t=35, b=40),
                        showlegend=False, hovermode="x unified")

                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    _y_range = None
                    if np.isfinite(user_disp) and abs(user_disp) > 1e-30:
                        _v5  = 5.0 * abs(user_disp)
                        _fin = arr_plot[np.isfinite(arr_plot)]
                        if len(_fin) > 0 and (_fin.max() > _v5 or _fin.min() < -_v5):
                            _y_range = [-_v5, _v5]
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb",
                                     **({"range": _y_range} if _y_range is not None else {}))

                    plotly_with_dl(fig,
                                   key=f"pfp_{model_short}_{arr_key}_{fname}",
                                   filename=f"pfp_{model_short}_{arr_key}_{fname}",
                                   container=col_w)

                    # Number input — key includes rng_tag so it resets to new median on slider move
                    actual_val = col_w.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp),
                        format="%.5g",
                        key=inp_key)

                    # Quickset buttons row beneath the input (Step 3 layout).
                    # cold_disp surfaces beside "default" when the parameter
                    # has a value extracted in the Cold-HBT section.
                    quickset_buttons(container=col_w,
                                      key_prefix=inp_key,
                                      target_key=inp_key,
                                      arr_disp=arr_plot,
                                      default_disp=auto_disp,
                                      cold_disp=_cold_disp,
                                      unit=unit,
                                      fmt="%.4g", layout="below")

                    # Extra quickset buttons sourced from elsewhere in the UI:
                    #   - τB / τC : v_c-method values published by the
                    #     tau_total_fit_group section (T-models, ≥2 files).
                    #   - Rbe     : Z-parameter method result (when run and
                    #     this DUT is part of the fit).
                    _extra_qs = []
                    if param_key in ("tauB", "tauC"):
                        _pub = st.session_state.get(
                            f"taut_pub_{param_key}_{model_short}_{fname}")
                        if _pub is not None and np.isfinite(_pub):
                            _extra_qs.append(("v_c", float(_pub) * scale))
                    elif param_key == "Rbe":
                        _rz_rbe = st.session_state.get(f"rz12_Rbe_{fname}")
                        if _rz_rbe is not None and np.isfinite(_rz_rbe) \
                                and abs(_rz_rbe) > 0:
                            _extra_qs.append(("Z-param", float(_rz_rbe) * scale))

                    if _extra_qs:
                        _bcols = col_w.columns(len(_extra_qs))
                        for _bc, (_lbl, _val) in zip(_bcols, _extra_qs):
                            _btn_text = (f"{_lbl} = {_val:.4g} {unit}"
                                         if unit else f"{_lbl} = {_val:.4g}")
                            if _bc.button(_btn_text,
                                          key=f"{inp_key}_qs_extra_{_lbl}",
                                          width="stretch",
                                          help=f"Set {label} to the {_lbl} reference value"):
                                st.session_state[inp_key + "_pending"] = float(_val)
                                st.rerun()

                    params_out[param_key] = actual_val / scale

                # Render the Cbex sweep tool in the unused 2nd column,
                # immediately beside the Cbex plot.
                if (_is_cbex_sweep_group
                        and not _cbex_sweep_rendered
                        and len(row) < 2):
                    with cols[1]:
                        _render_cbex_sweep_tool(
                            cbex_arr=live_arrays["Cbex_arr"],
                            freq=freq,
                            f_ghz=f_ghz,
                            f_min_v=f_min_v,
                            f_max_v=f_max_v,
                            cbex_scale=g_params[0][3],   # 1e15 (fF)
                            cbex_unit=g_params[0][4],    # "fF"
                            cbex_param_key=g_params[0][1],  # "Cbex"
                            model_short=model_short,
                            fname=fname,
                            g_idx=g_idx,
                            rng_tag=rng_tag,
                            cbex_sweep_fn=cbex_sweep_fn,
                            param_groups=param_groups,
                        )
                    _cbex_sweep_rendered = True

            # ── Re-extract downstream groups if any param in this group changed ──
            if reextract_fn is not None and g_idx < len(param_groups) - 1:
                _grp_param_keys = {spec[1] for spec in g_params}
                _grp_arr_keys   = {spec[0] for spec in g_params}
                _any_changed = any(
                    abs(params_out.get(pk, 0.0) - live_params.get(pk, 0.0))
                    > 1e-9 * (abs(live_params.get(pk, 0.0)) + 1e-30)
                    for pk in _grp_param_keys
                    if pk in params_out
                )
                if _any_changed:
                    try:
                        _new_p, _new_a = reextract_fn(params_out, g_idx, live_arrays)
                        # Only overwrite params/arrays from groups not yet processed.
                        # Protecting all groups 0..g_idx prevents a downstream
                        # re-extraction (e.g. tauB change) from clobbering user
                        # overrides set in earlier groups (e.g. Rbi, Rbe, …).
                        _processed_param_keys = {
                            spec[1]
                            for gi in range(g_idx + 1)
                            for spec in param_groups[gi]["params"]
                        }
                        _processed_arr_keys = {
                            spec[0]
                            for gi in range(g_idx + 1)
                            for spec in param_groups[gi]["params"]
                        }
                        for _k, _v in _new_p.items():
                            if _k not in _processed_param_keys:
                                live_params[_k] = _v
                                params_out[_k]  = _v
                        for _k, _v in _new_a.items():
                            if _k not in _processed_arr_keys:
                                live_arrays[_k] = _v
                    except Exception:
                        pass  # silently ignore re-extraction failures

            prev_range = (f_lo, f_hi)
            if g_idx < len(param_groups) - 1:
                st.markdown("---")

    return params_out


# ── Streamlit fragment decorator (1.36 → st.experimental_fragment;
#    1.37+ → st.fragment).  Wrapping the slider-preview render functions
#    in a fragment confines slider-drag reruns to JUST the fragment —
#    the rest of the SSM script (Sections 1-5, all other models) does
#    NOT re-execute.  That's the order-of-magnitude speedup for Live
#    mode (1-2 s per drag → ~150 ms).
_FRAGMENT = (getattr(st, "fragment", None)
             or getattr(st, "experimental_fragment", None)
             or (lambda f: f))


# ── Tuning (parameter sweep + residual table) ───────────────────────────────

def _make_sweep_values(min_val, max_val, step):
    """Generate sweep values, always including max_val as the last point."""
    if step <= 0 or abs(max_val - min_val) < 1e-15:
        return np.array([min_val])
    if max_val < min_val:
        return np.array([min_val])
    values = np.arange(min_val, max_val + step * 0.5, step)
    if len(values) == 0:
        return np.array([min_val])
    # Ensure max is included
    if abs(values[-1] - max_val) > 1e-12:
        values = np.append(values, max_val)
    return values


def _fmt_eta(seconds) -> str:
    """Human-readable ETA: seconds under a minute, ``Xm Ys`` under an hour,
    ``Xh Ym Zs`` beyond an hour."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    total = int(round(s))
    if total < 3600:
        m, sec = divmod(total, 60)
        return f"{m}m {sec}s"
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"


def _fmt_eval_time(seconds) -> str:
    """Total run time as ``xx s (xx h: xx m: xx s)`` — raw seconds plus an
    hours:minutes:seconds breakdown."""
    s = max(0.0, float(seconds))
    total = int(s)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{s:.1f} s ({h} h: {m:02d} m: {sec:02d} s)"


def _render_slider_preview(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key):
    """Sandbox-style slider preview at the top of the Tuning expander.

    Two flavors selectable via the mode radio:

      🐢 Live (Streamlit)    — drag any number of sliders; every drag-tick
                                triggers a Streamlit rerun + a full sim.
                                Slow with many params or many freq points,
                                but supports multi-param sliding.

      ⚡ Plotly slider       — click ``🧮 Build animation`` once, then the
                                embedded Plotly figure scrubs through
                                pre-computed frames entirely client-side
                                (no Streamlit rerun per drag-tick).
                                Limited to one sweep parameter.

    Sliders in either mode write into ``slpreview_*`` session keys.
    ✅ "Use these values" (live mode) copies them into the main ``sim_*``
    keys; the auto-save gate in ``render_override_and_smith`` then picks
    that change up and persists it to the fit cache.
    """
    mode_key = f"slpreview_mode_{topo_key}_{fname}"
    mode = st.radio(
        "Preview mode",
        options=["🐢 Live (Streamlit rerun per drag)",
                 "⚡ Plotly slider (pre-computed frames)"],
        index=0, horizontal=True,
        key=mode_key,
        help="Live: drag any number of sliders; every tick reruns Streamlit "
             "and re-simulates.  Plotly: click Build once, then scrub through "
             "pre-computed frames entirely client-side (one sweep param at a "
             "time, but instant per drag).")
    if mode.startswith("⚡"):
        _render_plotly_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                       tuning_specs, fname, topo_key)
    else:
        _render_live_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                     tuning_specs, fname, topo_key)


def _slider_default_range(current_disp):
    """Sane default (min, max, step) for one slider given the current value."""
    if abs(current_disp) < 1e-30:
        return -1.0, 1.0, 0.01
    lo = current_disp * 0.1 if current_disp > 0 else current_disp * 10
    hi = current_disp * 10  if current_disp > 0 else current_disp * 0.1
    d_min, d_max = min(lo, hi), max(lo, hi)
    d_step = max((d_max - d_min) / 100, 1e-9)
    return d_min, d_max, d_step


def _multi_metric_top_n(arr, per_metric: int = 10):
    """Return the union of top-``per_metric`` rows by each of the first five
    columns (Total, S11, S12, S21, S22), deduped, sorted by Total Residual.

    ``arr`` is an (N, n_cols) float ndarray.  Rows whose Total Residual is
    non-finite or >= 1e30 (the BIG sentinel used to mark filtered combos)
    are dropped first.  Result row count: 10 (all five metrics' bests are
    the same row) ≤ R ≤ 50 (all distinct).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or arr.shape[0] == 0:
        return arr
    finite_mask = np.isfinite(arr[:, 0]) & (arr[:, 0] < 1e30)
    arr = arr[finite_mask]
    if arr.shape[0] == 0:
        return arr
    keep: set[int] = set()
    for c in range(min(5, arr.shape[1])):
        col_vals = arr[:, c]
        col_safe = np.where(np.isfinite(col_vals), col_vals, np.inf)
        n_take   = min(per_metric, arr.shape[0])
        idx      = np.argpartition(col_safe, n_take - 1)[:n_take]
        keep.update(idx.tolist())
    sub = arr[sorted(keep)]
    unique = np.unique(sub, axis=0)
    return unique[np.argsort(unique[:, 0])]


@_FRAGMENT
def _render_live_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                tuning_specs, fname, topo_key):
    """Streamlit-rerun-per-drag preview.

    Layout (v4 — fixed-height scroll containers, same primitive
    Streamlit's own sidebar uses for independent scrolling)
    --------------------------------------------------------
    Two top-level columns, each wrapped in ``st.container(height=…)``
    so they get their own scrollbar.  The page itself does not grow
    when many variable cards are added — the LEFT container scrolls
    internally, the RIGHT container stays put.

      ┌─ Left scroll-box ──────────┐ ┌─ Right scroll-box ─────────────┐
      │ [Parameters to slide]      │ │   Smith         |     Bode     │
      │  (narrow multiselect)      │ │   (compact)     |   (compact)   │
      │ ─────────                  │ │   legend below  |  legend below │
      │ variable card 1            │ │                                │
      │ variable card 2            │ │                                │
      │ variable card 3            │ │                                │
      │ ... (1 per row, scrolls)   │ │                                │
      └────────────────────────────┘ └────────────────────────────────┘
      ┌─ Below the columns (always visible) ─────────────────────────┐
      │  ✅ Use these values     ↩️ Reset preview                     │
      └───────────────────────────────────────────────────────────────┘

    Buttons live OUTSIDE the scroll boxes so the user doesn't have to
    scroll the left panel to find them.
    """
    from ..ssm_plots import render_ft_fmax_card

    # Fixed height for the two scroll boxes — picks a value comfortable
    # on a typical 1080p laptop, deliberately taller than the plots so
    # the right box never shows its own scrollbar (the plots fit) while
    # the left box gets scrollbars when the user adds many variables.
    _BOX_HEIGHT = 680

    label_for = {s[0]: s[1] for s in tuning_specs}

    # ── Top-level split — controls slightly narrower so the plots have
    #    room to display Smith + Bode side-by-side.
    controls_col, plots_col = st.columns([0.85, 1.15])

    # Pre-declare BOTH scroll containers so we can append into them in
    # any order (simulation happens after slider values are read).
    with controls_col:
        sliders_box = st.container(height=_BOX_HEIGHT, border=False)
    with plots_col:
        plots_box = st.container(height=_BOX_HEIGHT, border=False)

    # ── Controls box: multiselect (narrowed) + variable cards ─────────
    with sliders_box:
        # Constrain multiselect width with a sub-column so the chips
        # don't wrap awkwardly across the full panel.
        sel_key  = f"slpreview_sel_{topo_key}_{fname}"
        ms_col, _ms_pad = st.columns([3, 1])
        with ms_col:
            selected = st.multiselect(
                "Parameters to slide",
                options=[s[0] for s in tuning_specs],
                default=st.session_state.get(sel_key, []),
                format_func=lambda k: label_for.get(k, k),
                key=sel_key,
                help="Pick parameter(s) to drag.  Plots on the right "
                     "show the slider-substituted model in real time.  "
                     "The main Smith / fT-fmax plots above stay frozen "
                     "until you click ✅ Use these values.")

        selected_specs = [s for s in tuning_specs if s[0] in selected]
        preview_overrides: dict[str, float] = {}

        # Initialize ranges + sliders
        for spec in selected_specs:
            key, _, scale = spec[0], spec[1], spec[2]
            current_disp  = float(all_p.get(key, 0.0)) * scale
            kp = f"slpreview_{topo_key}_{key}_{fname}"
            if f"{kp}_min" not in st.session_state:
                d_min, d_max, d_step = _slider_default_range(current_disp)
                st.session_state[f"{kp}_min"]  = float(d_min)
                st.session_state[f"{kp}_max"]  = float(d_max)
                st.session_state[f"{kp}_step"] = float(d_step)
            if kp not in st.session_state:
                st.session_state[kp] = float(current_disp)

        # Variable cards — ONE per row.  The fixed-height scroll box
        # handles overflow internally.
        if not selected_specs:
            st.caption("Select one or more parameters above to begin.")
        else:
            for spec in selected_specs:
                key, label, scale = spec[0], spec[1], spec[2]
                unit = spec[3] if len(spec) > 3 else ""
                fmt  = spec[4] if len(spec) > 4 else "%.4g"
                current_disp = float(all_p.get(key, 0.0)) * scale
                kp = f"slpreview_{topo_key}_{key}_{fname}"
                mn = float(st.session_state[f"{kp}_min"])
                mx = float(st.session_state[f"{kp}_max"])
                sp = float(st.session_state[f"{kp}_step"])
                if mx <= mn:
                    mx = mn + max(sp, abs(mn) * 1e-6 + 1e-9)
                sp_safe = sp if sp > 0 else max((mx - mn) / 100, 1e-12)
                # Clamp the persisted slider value into the CURRENT
                # min/max bounds and write it back to session_state
                # BEFORE the widget renders.  Passing `value=` to a
                # widget that also has `key=` (where the key is in
                # session_state) triggers Streamlit's
                # check_session_state_rules warning every render —
                # the supported pattern is "set the key in
                # session_state, then omit value=".
                cur_v = min(max(float(st.session_state.get(kp, current_disp)),
                                mn), mx)
                st.session_state[kp] = cur_v
                label_unit = f"{label} ({unit})" if unit else label
                with st.container(border=True):
                    head = st.columns([1.4, 1, 1, 1])
                    head[0].markdown(
                        f"<div style='padding-top:1.6em;font-weight:600'>"
                        f"{label_unit}</div>",
                        unsafe_allow_html=True)
                    head[1].number_input("Min", format=fmt,
                                         key=f"{kp}_min")
                    head[2].number_input("Step", format=fmt,
                                         key=f"{kp}_step",
                                         min_value=0.0)
                    head[3].number_input("Max", format=fmt,
                                         key=f"{kp}_max")
                    v = st.slider(label_unit, min_value=mn, max_value=mx,
                                  step=sp_safe, format=fmt,
                                  key=kp, label_visibility="collapsed")
                    st.caption(f"main: **{current_disp:.4g}**  →  "
                               f"preview: **{v:.4g}** {unit}".rstrip())
                preview_overrides[key] = float(v) / scale

    # ── Preview simulation (same logic as before, narrowed plot heights
    #    because each plot now occupies a half-width column).
    if preview_overrides:
        all_p_prev = dict(all_p)
        all_p_prev.update(preview_overrides)
        swept_keys = list(preview_overrides.keys())
        xp_live = _cp if _HAS_CUDA else np

        static_cache_key  = f"slpreview_static_cache_{topo_key}_{fname}"
        static_hash_input = {k: float(v) for k, v in all_p.items()
                              if k not in swept_keys
                              and isinstance(v, (int, float))}
        static_hash_input["__swept"] = tuple(sorted(swept_keys))
        static_hash_input["__nf"]    = int(len(freq))
        static_hash_input["__xp"]    = "cuda" if xp_live is not np else "cpu"
        static_hash = params_hash({k: str(v) for k, v in static_hash_input.items()})
        cached_static = st.session_state.get(static_cache_key)
        if cached_static is None or cached_static.get("hash") != static_hash:
            try:
                static_cache = model_cls.build_static_cache(
                    all_p, freq, xp=xp_live, swept_keys=swept_keys)
            except Exception:
                static_cache = None
            st.session_state[static_cache_key] = {"hash": static_hash,
                                                  "cache": static_cache}
            cached_static = st.session_state[static_cache_key]
        static_cache = cached_static.get("cache")

        p_batch = dict(all_p)
        for k, v in preview_overrides.items():
            p_batch[k] = xp_live.asarray([v], dtype=float)
        S_prev = None
        try:
            S_b = model_cls.simulate_batch(p_batch, freq, z0,
                                            xp=xp_live, cache=static_cache)
            if xp_live is not np:
                S_b = _cp.asnumpy(S_b)
            S_prev = np.asarray(S_b)[0]
        except Exception:
            try:
                S_prev = model_cls.simulate_vec(all_p_prev, freq, z0)
            except Exception as e:
                st.error(f"Preview simulation failed: {e}")
                S_prev = None
        if S_prev is not None and not np.all(np.isfinite(S_prev)):
            st.warning("Preview S-parameters contain non-finite values — "
                       "adjust slider ranges to avoid singular combinations.")
            S_prev = None
        if S_prev is not None:
            # Render INTO the right scroll box.  Smith + Bode go in two
            # sub-columns so they sit side-by-side, with legends below
            # each plot (compact mode on the smith chart).
            with plots_box:
                st.markdown(
                    f"<div style='font-size:0.85em;color:#555;"
                    f"margin-bottom:4px'>Preview — {model_cls.NAME} "
                    f"(residual {ssm_residual(S_raw, S_prev):.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True)
                smith_col, bode_col = st.columns(2)
                with smith_col:
                    render_smith_chart(
                        S_raw, S_prev, model_cls.NAME,
                        ssm_residual(S_raw, S_prev),
                        scales=None,
                        key=f"slpreview_smith_{topo_key}_{fname}",
                        show_title=False,
                        compact=True, height=540)
                with bode_col:
                    render_ft_fmax_card(
                        S_raw, S_prev, freq,
                        model_name=model_cls.NAME,
                        key=f"slpreview_bode_{topo_key}_{fname}",
                        height=540, compact=True)

    # ── Commit / Reset buttons — appended to the left column BELOW the
    #    sliders_box scroll container, so they're always visible
    #    without scrolling the slider list.
    with controls_col:
        bc1, bc2 = st.columns(2)
        commit_clicked = bc1.button(
            "✅ Use these values",
            key=f"slpreview_commit_{topo_key}_{fname}",
            disabled=(len(preview_overrides) == 0),
            help="Copy slider values into the fine-tune Smith-chart override "
                 "fields above.  Does NOT auto-save to the persistent fit "
                 "cache — only direct edits in the fine-tune number_inputs do.",
            width="stretch")
        reset_clicked = bc2.button(
            "↩️ Reset preview",
            key=f"slpreview_reset_{topo_key}_{fname}",
            help="Discard slider drags and clear remembered min/step/max.",
            width="stretch")

    if commit_clicked:
        for k, v_si in preview_overrides.items():
            sc = next(s[2] for s in tuning_specs if s[0] == k)
            st.session_state[f"sim_{topo_key}_{k}_{fname}"] = float(v_si) * sc
        st.rerun()

    if reset_clicked:
        for s in tuning_specs:
            kp = f"slpreview_{topo_key}_{s[0]}_{fname}"
            for suf in ("", "_min", "_step", "_max"):
                st.session_state.pop(kp + suf, None)
        st.rerun()


def _quantize_S_batch_int16(S_batch):
    """Per-element int16 quantization, scaled by |element|.max() across all
    frames so each of the four S-elements (S11/S12/S21/S22) keeps its own
    dynamic range.  Returns a dict with int16 re/im arrays + their scale
    factors; ``_dequantize_S_frame`` reconstructs a single frame back to
    complex64 on slider lookup.

    Memory: 2 × int16 per complex (4 bytes) vs. 16 bytes per complex128
    or 8 per complex64 — i.e. 4× compression over complex128 / 2× over
    complex64.  For S-params on the unit circle (|Γ| ≤ 1) the round-trip
    error is ~3e-5, indistinguishable on a Smith / Bode plot."""
    S = np.asarray(S_batch)
    # shape (B, N, 2, 2) — compute per-element global max over (B, N).
    re = S.real
    im = S.imag
    # axes 0 and 1 are batch + freq; axes 2-3 are the 2x2 indices.
    re_max = np.maximum(np.abs(re).max(axis=(0, 1)), 1e-30)
    im_max = np.maximum(np.abs(im).max(axis=(0, 1)), 1e-30)
    re_q = np.round(re / re_max[None, None, :, :] * 32767).astype(np.int16)
    im_q = np.round(im / im_max[None, None, :, :] * 32767).astype(np.int16)
    return {"re_q":   re_q,
            "im_q":   im_q,
            "re_max": re_max.astype(np.float32),
            "im_max": im_max.astype(np.float32)}


def _dequantize_S_frame(quant, joint):
    """Reconstruct a single frame's S-matrix (shape (N, 2, 2) complex64)
    from the int16-quantized store built by ``_quantize_S_batch_int16``."""
    re = quant["re_q"][joint].astype(np.float32) * (
        quant["re_max"][None, :, :] / 32767.0)
    im = quant["im_q"][joint].astype(np.float32) * (
        quant["im_max"][None, :, :] / 32767.0)
    return (re + 1j * im).astype(np.complex64)


def _chunked_simulate_batch_to_host(model_cls, p_batch, freq, z0, *,
                                     xp, chunk_size: int = 5000,
                                     dtype=np.complex64):
    """Run ``simulate_batch`` in chunks of ``chunk_size`` so OOM doesn't bite
    on million-frame sweeps.  Returns one host-side ``np.ndarray`` of the
    requested complex ``dtype``.

    The default storage dtype is **complex64** (2× memory savings vs.
    complex128 with imperceptible visual difference on Smith + Bode).
    Pass ``dtype=np.complex128`` for full fp64 storage if you need it
    for downstream residual calculations.
    """
    sweep_keys = []
    B = None
    for k, v in p_batch.items():
        if isinstance(v, np.ndarray) or (_HAS_CUDA and isinstance(v, _cp.ndarray)):
            sweep_keys.append(k)
            if B is None:
                B = int(v.shape[0])
    if B is None or B <= chunk_size:
        out = model_cls.simulate_batch(p_batch, freq, z0, xp=xp)
        if xp is not np:
            out = _cp.asnumpy(out)
        return np.asarray(out).astype(dtype, copy=False)

    chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        sub_p = {k: (v[start:end] if k in sweep_keys else v)
                 for k, v in p_batch.items()}
        S_c = model_cls.simulate_batch(sub_p, freq, z0, xp=xp)
        if xp is not np:
            S_c = _cp.asnumpy(S_c)
            try:
                _cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        chunks.append(np.asarray(S_c).astype(dtype, copy=False))
    return np.concatenate(chunks, axis=0)


@_FRAGMENT
def _render_plotly_server_cached_view(state, S_raw, freq, model_cls,
                                       fname, topo_key, decim_n_max):
    """Server-cached rendering: Streamlit slider per axis + Plotly figure
    with the SINGLE current frame.  No browser-side bulk transfer — the
    pre-computed ``S_batch`` lives in session_state on the server, and
    each slider tick triggers a Streamlit rerun that looks up exactly
    one frame.  ~100-300 ms per drag tick but supports arbitrarily large
    sweeps (only bounded by server RAM)."""
    from ..ssm_plots import render_ft_fmax_card

    slider_specs = state["slider_specs"]
    dims = [len(sp["values_disp"]) for sp in slider_specs]
    S_batch = state.get("S_batch")
    S_quant = state.get("S_quant")
    if S_batch is None and S_quant is None:
        st.error("Build state is missing both S_batch and S_quant — "
                 "click 🧮 Build animation to refresh.")
        return

    # ── Streamlit sliders, 2 per row ────────────────────────────────────
    pos: list[int] = []
    for row_start in range(0, len(slider_specs), 2):
        row = slider_specs[row_start:row_start + 2]
        row_cols = st.columns(len(row))
        for col_w, sp in zip(row_cols, row):
            idx = slider_specs.index(sp)
            n = len(sp["values_disp"])
            key = f"slprev_pl_sc_pos_{topo_key}_{fname}_{idx}"
            if key not in st.session_state:
                st.session_state[key] = n // 2
            with col_w:
                v_idx = st.slider(
                    f"{sp['label']} ({sp['unit']})" if sp["unit"]
                    else sp["label"],
                    min_value=0, max_value=n - 1, step=1, key=key)
                v_disp = sp["values_disp"][v_idx]
                fmt = sp.get("fmt", "%.4g").lstrip("%")
                try:
                    v_str = format(float(v_disp), fmt)
                except (ValueError, TypeError):
                    v_str = str(v_disp)
                st.caption(f"= **{v_str} {sp['unit']}**".rstrip())
            pos.append(v_idx)

    # Joint frame index (row-major over slider_specs)
    joint = 0
    for k, p in enumerate(pos):
        joint = joint * dims[k] + p

    # Pull single frame (decode from int16 if quantized) and decimate freq.
    if S_batch is not None:
        S_frame = S_batch[joint]
    else:
        S_frame = _dequantize_S_frame(S_quant, joint)
    if decim_n_max < S_frame.shape[0]:
        stride = max(1, int(np.ceil(S_frame.shape[0] / decim_n_max)))
        S_frame_d = S_frame[::stride]
        freq_d    = np.asarray(freq)[::stride]
    else:
        S_frame_d = S_frame
        freq_d    = np.asarray(freq)
    if S_raw is not None:
        if S_raw.shape[0] != len(freq_d):
            r = max(1, S_raw.shape[0] // len(freq_d))
            S_raw_d = S_raw[::r][: len(freq_d)]
        else:
            S_raw_d = S_raw
    else:
        S_raw_d = None

    # Render Smith + Bode side by side using existing helpers
    col_smith, col_bode = st.columns([1.05, 1])
    with col_smith:
        st.markdown("**Smith chart**")
        if S_raw_d is not None:
            render_smith_chart(S_raw_d, S_frame_d, model_cls.NAME,
                               ssm_residual(S_raw_d, S_frame_d),
                               scales=None,
                               key=f"slprev_pl_sc_smith_{topo_key}_{fname}",
                               show_title=False)
        else:
            # Model-only Smith chart (RF simulator path)
            from ..helpers.plotly_plots import make_smith as _make_smith
            import pandas as _pd
            df = _pd.DataFrame({
                "Freq (GHz)": freq_d * 1e-9,
                "S11_meas":   S_frame_d[:, 0, 0],
                "S12_meas":   S_frame_d[:, 0, 1],
                "S21_meas":   S_frame_d[:, 1, 0],
                "S22_meas":   S_frame_d[:, 1, 1],
            })
            fig = _make_smith(df, freq_d * 1e-9,
                              float(freq_d[0] * 1e-9),
                              float(freq_d[-1] * 1e-9),
                              (True, True, True, True),
                              (1.0, 1.0, 1.0, 1.0),
                              model_cls.NAME)
            st.plotly_chart(fig, width="stretch",
                            key=f"slprev_pl_sc_smith_{topo_key}_{fname}")

    with col_bode:
        st.markdown("**fT / fmax**")
        if S_raw_d is not None:
            render_ft_fmax_card(S_raw_d, S_frame_d, freq_d,
                                model_name=model_cls.NAME,
                                key=f"slprev_pl_sc_bode_{topo_key}_{fname}",
                                height=560)
        else:
            # Model-only bode — quick build
            from ..helpers.metrics import compute_h21_U
            h21_db, U_db = compute_h21_U(S_frame_d)
            import plotly.graph_objects as _go
            f_g = freq_d * 1e-9
            fig = _go.Figure()
            fig.add_trace(_go.Scatter(x=f_g, y=h21_db, mode="lines",
                                      name="|h21|² model",
                                      line=dict(color="#1f77b4", width=2)))
            fig.add_trace(_go.Scatter(x=f_g, y=U_db, mode="lines",
                                      name="Mason U model",
                                      line=dict(color="#d62728", width=2,
                                                dash="dash")))
            fig.update_layout(
                height=560, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Frequency (GHz)", type="log",
                           showgrid=True, gridcolor="#ebebeb"),
                yaxis=dict(title="Gain (dB)", range=[0, 50],
                           showgrid=True, gridcolor="#ebebeb"),
                margin=dict(l=55, r=20, t=20, b=50),
                legend=dict(orientation="h", x=0.5, y=-0.18,
                            xanchor="center", yanchor="top"))
            st.plotly_chart(fig, width="stretch",
                            key=f"slprev_pl_sc_bode_{topo_key}_{fname}")


@_FRAGMENT
def _render_plotly_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                  tuning_specs, fname, topo_key):
    """Pre-computed Plotly slider — joint (cartesian) multi-param scan.

    Each selected param becomes one Plotly slider.  Frames are the *full
    cartesian product* of all sliders' sweep values, so dragging slider
    B reflects the model at the *current position of every other slider*
    (true M × N × K joint behaviour).  Coordination between sliders is
    done by injected JS listening to ``plotly_sliderchange``.

    Performance notes
    -----------------
    • All traces use ``Scattergl`` (WebGL).
    • Frequency axis is decimated to ≤ 200 points per trace.
    • Single ``simulate_batch`` call runs the full cartesian product —
      one GPU pass when CUDA is available.

    Cache
    -----
    Built batch + slider specs stash in session_state until the user
    changes the selection / ranges and clicks ``🧮 Build`` again.
    """
    from ..helpers.plotly_plots import make_smith_bode_joint_slider_html

    label_for = {s[0]: s[1] for s in tuning_specs}
    options   = [s[0] for s in tuning_specs]

    sel_key = f"slprev_pl_sel_{topo_key}_{fname}"
    selected = st.multiselect(
        "Sweep parameters",
        options=options,
        default=st.session_state.get(sel_key, [options[0]] if options else []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        help="Each selected param gets its own Plotly slider in the figure. "
             "Frames are the FULL cartesian product, so dragging slider B "
             "reflects the model at the current position of every other "
             "slider (true joint scan).  Watch the total frame count below "
             "— it grows multiplicatively.")

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    if not selected_specs:
        st.caption("Select one or more parameters above and click "
                   "**🧮 Build animation**.")
        return

    # ── Initialize per-param ranges ────────────────────────────────────
    for spec in selected_specs:
        key, _, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"slprev_pl_{topo_key}_{key}_{fname}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, _ = _slider_default_range(current_disp)
            st.session_state[f"{kp}_min"]    = float(d_min)
            st.session_state[f"{kp}_max"]    = float(d_max)
            st.session_state[f"{kp}_frames"] = 11

    # ── 2-column variable-card grid (min / max / frames per card) ──────
    def _render_one_range_card(spec):
        key, label, scale = spec[0], spec[1], spec[2]
        unit = spec[3] if len(spec) > 3 else ""
        fmt  = spec[4] if len(spec) > 4 else "%.4g"
        kp   = f"slprev_pl_{topo_key}_{key}_{fname}"
        label_unit = f"{label} ({unit})" if unit else label
        with st.container(border=True):
            head = st.columns([1.4, 1, 1, 1])
            # Top-pad the variable name so it sits at the same vertical
            # level as the input boxes (whose own "Min"/"Max"/"Frames"
            # labels add ~1.6 em of header height above them).
            head[0].markdown(
                f"<div style='padding-top:1.6em;font-weight:600'>"
                f"{label_unit}</div>",
                unsafe_allow_html=True)
            head[1].number_input("Min", format=fmt, key=f"{kp}_min")
            head[2].number_input("Max", format=fmt, key=f"{kp}_max")
            head[3].number_input("Frames", min_value=2, max_value=100, step=1,
                                 key=f"{kp}_frames",
                                 help="Frames per axis (2–100). "
                                      "Total = product across params.")

    for row_start in range(0, len(selected_specs), 2):
        row_specs = selected_specs[row_start:row_start + 2]
        l_col, r_col = st.columns(2)
        with l_col:
            _render_one_range_card(row_specs[0])
        with r_col:
            if len(row_specs) > 1:
                _render_one_range_card(row_specs[1])
            else:
                st.empty()

    # ── Decimation (fidelity) control + payload estimate ───────────────
    n_freq_full = int(len(freq))
    decim_key   = f"slprev_pl_decim_{topo_key}_{fname}"
    if decim_key not in st.session_state:
        st.session_state[decim_key] = min(120, n_freq_full)

    dims_preview = []
    for spec in selected_specs:
        kp = f"slprev_pl_{topo_key}_{spec[0]}_{fname}"
        dims_preview.append(int(st.session_state.get(f"{kp}_frames", 11)))
    total_frames = int(np.prod(dims_preview)) if dims_preview else 0

    # Estimated payload: (5-sig-fig ≈ 7 chars / number) × 10 numbers / sample.
    decim_n = int(st.session_state.get(decim_key, min(120, n_freq_full)))
    decim_n = min(decim_n, n_freq_full)
    est_mb  = total_frames * decim_n * 10 * 7 / 1024 / 1024

    fd_col1, fd_col2 = st.columns([1, 2])
    with fd_col1:
        st.number_input(
            f"Freq points (max: {n_freq_full})",
            min_value=20, max_value=n_freq_full, step=10,
            key=decim_key,
            help=f"Frequency points kept per trace (max = {n_freq_full} = "
                 "full fidelity).  Smith and Bode look smooth from ~120 "
                 "onward; raising it just makes the browser payload bigger.")
    with fd_col2:
        st.caption(
            "Cartesian sweep: "
            + " × ".join(str(d) for d in dims_preview)
            + f" = **{total_frames}** frames · {decim_n} freq pts · "
            f"estimated payload ≈ **{est_mb:.0f} MB**")

    # ── Server-cached toggle.  When ON: pre-compute stays in server RAM
    #    and only the current frame is shipped per slider tick (Streamlit
    #    reruns drive the slider).  Bypasses Streamlit's 200 MB browser
    #    message limit at the cost of ~100-300 ms per drag.
    sc_key = f"slprev_pl_servercached_{topo_key}_{fname}"
    server_cached = st.checkbox(
        "📡 Server-cached mode (Streamlit sliders, supports unlimited "
        "sweep size, slower drag)",
        value=st.session_state.get(sc_key, False), key=sc_key,
        help="OFF (default): pre-computed frames are embedded in the "
             "browser and scrubbing is instant — but the total payload is "
             "capped at Streamlit's 200 MB message size.  ON: frames stay "
             "in server RAM and only the current frame is sent per slider "
             "drag.  Each drag triggers a Streamlit rerun (~100-300 ms) "
             "but the sweep can be arbitrarily large.")
    int16_key = f"slprev_pl_int16_{topo_key}_{fname}"
    if server_cached:
        # Memory estimate — complex64 storage = 8 bytes per S element,
        # int16 quant = 4 bytes per element (2× compression over c64).
        bytes_per_elem = 4 if st.session_state.get(int16_key, False) else 8
        ram_mb = total_frames * n_freq_full * bytes_per_elem * 4 / 1024 / 1024
        col_lbl, col_q = st.columns([2, 1])
        col_lbl.caption(
            f"Server RAM estimate ("
            f"{'int16 quant' if bytes_per_elem == 4 else 'complex64'}, "
            f"full-fidelity batch): ≈ **{ram_mb:.0f} MB** in session_state.")
        col_q.checkbox(
            "🗜️ int16 quantize",
            value=st.session_state.get(int16_key, False),
            key=int16_key,
            help="Per-trace int16 quantization (rescaled by element-wise "
                 "max) — 2× memory savings vs. complex64 with ~3e-5 "
                 "round-trip error.  Negligible visual difference on "
                 "Smith / Bode plots, recommended for very large batches.")
        if ram_mb > 8000:
            st.warning(f"⚠️ ~{ram_mb/1024:.1f} GB server-side may OOM on "
                       "modest machines.  Reduce per-axis frame counts if "
                       "Build runs out of memory.")
    else:
        if est_mb > 180:
            st.error(
                f"❌ Estimated payload ≈ {est_mb:.0f} MB will exceed "
                "Streamlit's 200 MB browser-message limit.  Enable "
                "**📡 Server-cached mode** above, lower the **Freq points** "
                "value, reduce per-axis frame counts, or raise the limit "
                "via `.streamlit/config.toml` → `[server] maxMessageSize "
                "= 500`.")
        elif est_mb > 120:
            st.warning(f"⚠️ Estimated payload ≈ {est_mb:.0f} MB is close "
                       "to Streamlit's 200 MB limit.")
        elif total_frames > 2000:
            st.warning(f"⚠️ {total_frames} frames may stutter on "
                       "slider drag.")

    # ── CUDA checkbox + Build button (button next to checkbox when CUDA available)
    cuda_toggle_key = f"slprev_pl_cuda_{topo_key}_{fname}"
    if _HAS_CUDA:
        cuda_col, btn_col = st.columns([1.6, 1])
        with cuda_col:
            use_cuda = st.checkbox(f"⚡ Use CUDA (cupy {_CUDA_VER}) for "
                                    "batched simulation",
                                    value=st.session_state.get(cuda_toggle_key, True),
                                    key=cuda_toggle_key,
                                    help="Off-load the joint cartesian "
                                         "batched simulation to the GPU.  "
                                         "Result is brought back to host as "
                                         "fp64 for Plotly embedding.")
        with btn_col:
            build_clicked = st.button("🧮 Build animation",
                                      key=f"slprev_pl_build_{topo_key}_{fname}",
                                      width="stretch",
                                      help="Pre-compute the cartesian joint "
                                           "sweep and embed with JS-"
                                           "coordinated multi-sliders.")
    else:
        use_cuda = False
        build_clicked = st.button("🧮 Build animation",
                                  key=f"slprev_pl_build_{topo_key}_{fname}",
                                  width="stretch",
                                  help="Pre-compute the cartesian joint "
                                       "sweep and embed with JS-coordinated "
                                       "multi-sliders.")

    state_key = f"slprev_pl_state_{topo_key}_{fname}"

    if build_clicked:
        import time as _time
        xp = _cp if (use_cuda and _HAS_CUDA) else np
        device_label = (f"GPU (cupy {_CUDA_VER})"
                        if xp is not np else "CPU (numpy)")
        # Build per-axis sweeps then meshgrid → cartesian product
        sweep_disps  : list[np.ndarray] = []
        sweep_sis    : list[np.ndarray] = []
        slider_specs_out: list[dict] = []
        for spec in selected_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            kp   = f"slprev_pl_{topo_key}_{key}_{fname}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            nf = int(st.session_state[f"{kp}_frames"])
            if mx <= mn:
                mx = mn + abs(mn) * 1e-6 + 1e-9
            sd = np.linspace(mn, mx, nf)
            sweep_disps.append(sd)
            sweep_sis.append(sd / scale)
            slider_specs_out.append(dict(
                label=label, unit=unit, fmt=fmt,
                values_disp=sd.tolist()))
        meshes = np.meshgrid(*sweep_sis, indexing="ij")
        flats  = [m.ravel() for m in meshes]
        n_total = int(flats[0].size) if flats else 0

        # Spinner so the user sees that compute is happening (the Plotly
        # figure below stays "stale" — the prior build — until the new
        # batch finishes and we replace it).  For very large sweeps the
        # chunked path avoids GPU OOM by simulating in slabs of 5 000.
        t0 = _time.perf_counter()
        with st.spinner(f"Computing {n_total} frames on {device_label}…"):
            p_batch = dict(all_p)
            for spec, flat in zip(selected_specs, flats):
                p_batch[spec[0]] = xp.asarray(flat, dtype=float)
            try:
                S_b = _chunked_simulate_batch_to_host(
                    model_cls, p_batch, freq, z0, xp=xp)
            except Exception as e:
                st.error(f"Batched preview simulation failed: {e}")
                return
        elapsed = _time.perf_counter() - t0

        if not np.all(np.isfinite(S_b)):
            st.warning("Some frames contain non-finite S-parameters — "
                       "narrow the ranges to avoid singular combinations.")

        state_dict = {
            "slider_specs":  slider_specs_out,
            "selected_keys": [s[0] for s in selected_specs],
            "elapsed_s":     elapsed,
            "device":        device_label,
            "n_total":       n_total,
        }
        if server_cached and st.session_state.get(int16_key, False):
            # Pack the batch as int16 (per-element rescaled).  4× smaller
            # than complex128, 2× smaller than complex64.  Slider lookup
            # decodes one frame at a time via _dequantize_S_frame.
            state_dict["S_quant"] = _quantize_S_batch_int16(S_b)
            state_dict["S_batch"] = None  # free the big complex array
        else:
            state_dict["S_batch"] = S_b
        st.session_state[state_key] = state_dict

    state = st.session_state.get(state_key)
    if state is None:
        st.info("Click **🧮 Build animation** to compute frames for the "
                "Plotly slider(s).")
        return

    cached_keys  = state.get("selected_keys", [])
    current_keys = [s[0] for s in selected_specs]
    if cached_keys != current_keys:
        st.warning("Selection changed since last build "
                   f"(cached: {cached_keys}, current: {current_keys}).  "
                   "Click **🧮 Build animation** to refresh.")
        return

    # Device + elapsed banner so the user can confirm GPU vs CPU.
    elapsed = float(state.get("elapsed_s", 0.0))
    if state.get("S_batch") is not None:
        n_total = int(state.get("n_total", 0)) or len(state["S_batch"])
    elif state.get("S_quant") is not None:
        n_total = int(state.get("n_total", 0)) or state["S_quant"]["re_q"].shape[0]
    else:
        n_total = 0
    device  = str(state.get("device", "?"))
    ms_each = (elapsed / max(1, n_total)) * 1000.0
    storage = "int16-quant" if state.get("S_quant") is not None else "complex64"
    st.caption(f"✅ Built **{n_total}** frames on **{device}** in "
               f"**{elapsed:.2f} s** ({ms_each:.1f} ms/frame, "
               f"storage: {storage}).  Drag any slider below to scrub.")

    if server_cached:
        _render_plotly_server_cached_view(
            state, S_raw, freq, model_cls, fname, topo_key,
            decim_n_max=int(st.session_state.get(decim_key, 120)))
    else:
        if state.get("S_batch") is None:
            st.warning("Last build used int16 quantization (server-cached "
                       "only).  Rebuild without **🗜️ int16 quantize** to "
                       "embed frames in the browser.")
            return
        html = make_smith_bode_joint_slider_html(
            S_batch_joint=state["S_batch"],
            freq=freq,
            slider_specs=state["slider_specs"],
            model_name=model_cls.NAME,
            S_meas=S_raw,
            decimate_points=int(st.session_state.get(decim_key, 120)),
            # Mirror the Plotly Smith chart's per-trace display scale (set via
            # smith_scale_controls) so the slider's Smith view matches it.
            smith_mults={
                nm: float(st.session_state.get(
                    f"smith_scale_{topo_key}_{nm}_{fname}", 1.0))
                for nm in ("S11", "S12", "S21", "S22")
            },
        )
        # Plotly figure (height=500) + HTML slider rows below.  Each row
        # is ~36 px tall; container has ~26 px padding.  +40 px buffer.
        n_sl = len(state["slider_specs"])
        iframe_height = 500 + 26 + 36 * n_sl + 30
        # st.iframe replaced components.v1.html (deprecated 2026-06-01).
        # When src is a raw HTML string (no http(s) / file / Path prefix)
        # Streamlit embeds it directly in an iframe — same behaviour as
        # the old components.html call.  No `scrolling` parameter; the
        # `height=` integer is interpreted in pixels just like before.
        st.iframe(html, height=iframe_height)


def render_visual_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                                   tuning_specs, fname, topo_key):
    """🎚️ Visual Tuning expander — drag sliders to see the model react.

    Strictly UI-only: writes into the same fine-tune ``sim_*`` session
    keys via the ✅ commit button.  Auto-save is suppressed for slider
    commits (see ``render_override_and_smith``).
    """
    with st.expander("🎚️ Visual Tuning", expanded=False):
        # ── Backend status badges — show ALL active accelerators.
        #    When both CUDA and Rust are available, the Visual Tuning
        #    Live mode picks CUDA via `xp=cupy`, but the Plotly slider
        #    "Build animation" path (cache-less batched sim) can route
        #    through Rust.  Showing both lets the user know what's
        #    available, not just which one wins per click.
        from ..helpers.rust_kernels import (
            HAS_RUST as _HAS_RUST_BACKEND,
            rust_diagnostic as _rust_diag,
        )
        _chips = []
        if _HAS_CUDA:
            _chips.append(
                "<span style='background:#e3f2fd;color:#0d47a1;"
                f"padding:2px 8px;border-radius:4px;font-size:0.8em;"
                f"font-weight:600'>⚡ CUDA (cupy {_CUDA_VER})</span>")
        if _HAS_RUST_BACKEND:
            _chips.append(
                "<span style='background:#fff3e0;color:#e65100;"
                "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                "font-weight:600'>🦀 Rust kernels</span>")
        if not _chips:
            _chips.append(
                "<span style='background:#eceff1;color:#37474f;"
                "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                "font-weight:600'>🐢 NumPy fallback</span>"
                "  <span style='font-size:0.78em;color:#666'>"
                "(build the Rust crate for ~10× speedup — see "
                "tools/SSM/rust_kernels/README.md)</span>")
        st.markdown(
            "Compute backends available: " + "  ".join(_chips),
            unsafe_allow_html=True)

        # ── Diagnostic: when the binary exists on disk but Rust didn't
        #    load, surface the actual import error inside an expander so
        #    the user doesn't have to dig through the terminal.
        if not _HAS_RUST_BACKEND:
            _d = _rust_diag()
            if _d["binary_files"] and not _d["force_numpy"]:
                with st.expander("🛠️ Why is Rust not active?", expanded=False):
                    st.code(
                        f"arch_tag       : {_d['arch_tag']}\n"
                        f"bin_dir        : {_d['bin_dir']}\n"
                        f"bin_dir_exists : {_d['bin_dir_exists']}\n"
                        f"binary_files   : {_d['binary_files']}\n"
                        f"import_error   : {_d['import_error']}\n"
                        f"force_numpy    : {_d['force_numpy']}\n",
                        language="text")
                    st.caption(
                        "A binary exists on disk but couldn't be imported. "
                        "Most common cause: the Streamlit process was started "
                        "*before* the binary was placed in this folder — "
                        "restart the launcher (`python LAUNCH_Tool.py`) to "
                        "pick it up.  If the import error persists after a "
                        "fresh restart, the .pyd may be from a different ABI; "
                        "delete it and run `python build_rust_kernels.py`.")
        _render_slider_preview(model_cls, all_p, S_raw, freq, z0,
                               tuning_specs, fname, topo_key)


def render_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key):
    """🔧 Auto Tuning for Minimum Residuals — grid sweep + residual ranking.

    Renders the full sweep-grid / brute-force / optimized / prioritize /
    minimize-deviation toolset.  Kept as ``render_tuning_expander`` for
    call-site stability (still imported under that name); the visible
    label is "Auto Tuning for Minimum Residuals".

    Parameters
    ----------
    model_cls : AbstractSSMModel subclass with .simulate(params, freq, z0)
    all_p     : dict — current parameter values in SI units
    S_raw     : np.ndarray — measured S-parameters (N, 2, 2)
    freq      : np.ndarray — frequency array
    z0        : float
    tuning_specs : list of (key, label, scale, unit, ...) tuples
                   scale converts SI → display unit (e.g. 1e15 for fF)
    fname     : str — filename for unique widget keys
    topo_key  : str — topology short name for unique widget keys
    """
    import pandas as pd

    with st.expander("🔧 Auto Tuning for Minimum Residuals", expanded=False):

        # ── Compute backend badge — cached in session_state so the
        #    label stays stable across reruns.  Without the cache the
        #    badge sometimes flipped Rust→NumPy after a Stop/cancel
        #    rerun (re-import races on `hbt_rust_kernels` after the
        #    sweep loop unwinds); caching guarantees the badge reflects
        #    a one-time, definitive detection.
        _BADGE_KEY = f"_rust_active_cached_{model_cls.SHORT}_{fname}"

        def _detect_rust_active() -> bool:
            try:
                from ..helpers.rust_kernels import (
                    HAS_RUST as _HR,
                    _phase2_dispatch_enabled as _phase2_on,
                    SIM_FOR_TOPOLOGY as _SIM_TOPO,
                )
            except Exception:                              # pragma: no cover
                return False
            return bool(
                _HR and _phase2_on() and
                _SIM_TOPO.get(model_cls.SHORT) is not None
            )

        # Sticky-True caching: re-evaluate every render, but never flip
        # an active=True back to False within the session.  This keeps
        # the original "no Rust→NumPy flicker after Stop" guarantee
        # *and* allows the badge to flip from stale-False → True the
        # moment the Rust binary becomes importable (e.g. user just
        # finished building it without restarting Streamlit).
        _now_active = _detect_rust_active()
        _prev = bool(st.session_state.get(_BADGE_KEY, False))
        st.session_state[_BADGE_KEY] = bool(_now_active or _prev)
        _rust_active_here = st.session_state[_BADGE_KEY]
        if _HAS_CUDA and _rust_active_here:
            _bk = ("<span style='background:#e3f2fd;color:#0d47a1;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'>⚡ CUDA available + 🦀 Rust active</span>"
                   "  <span style='font-size:0.78em;color:#666'>"
                   "(CUDA buttons → cupy; CPU buttons → Rust)</span>")
        elif _HAS_CUDA:
            _bk = ("<span style='background:#e3f2fd;color:#0d47a1;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'>⚡ CUDA available</span>"
                   "  <span style='font-size:0.78em;color:#666'>"
                   "(CUDA buttons → cupy; CPU buttons → NumPy.  "
                   "Set <code>HBT_USE_RUST_SIM_BATCH=1</code> to enable "
                   "Rust on the CPU path for ~25× faster sweeps.)</span>")
        elif _rust_active_here:
            _bk = ("<span style='background:#fff3e0;color:#e65100;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'>🦀 Rust active</span>"
                   "  <span style='font-size:0.78em;color:#666'>"
                   "(CPU buttons → Rust kernel)</span>")
        else:
            _bk = ("<span style='background:#eceff1;color:#37474f;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'>🐢 NumPy</span>"
                   "  <span style='font-size:0.78em;color:#666'>"
                   "(CPU buttons → NumPy.  Build the Rust crate "
                   "(<code>python build_rust_kernels.py</code>) and set "
                   "<code>HBT_USE_RUST_SIM_BATCH=1</code> for ~25× speedup.)"
                   "</span>")
        st.markdown(f"Compute backend: {_bk}", unsafe_allow_html=True)

        # ── Toolbar: Use default values + Select all + De-select all ──
        # All three buttons live above the table so users hit the bulk
        # actions before scanning per-row.  Each writes to session_state
        # and reruns so the table picks up the new values on the next
        # render pass.
        _tb1, _tb2, _tb3, _ = st.columns([1.0, 1.0, 1.0, 3.0])
        if _tb1.button("↩️ Use default values",
                        key=f"tune_defaults_{topo_key}_{fname}"):
            for spec in tuning_specs:
                key, scale = spec[0], spec[2]
                current_si = float(all_p.get(key, 0.0))
                current_disp = current_si * scale
                kp = f"tune_{topo_key}_{key}_{fname}"
                if abs(current_si) < 1e-30:
                    st.session_state[f"{kp}_min"] = 0.0
                    st.session_state[f"{kp}_step"] = 0.0
                    st.session_state[f"{kp}_max"] = 0.0
                else:
                    st.session_state[f"{kp}_min"] = current_disp - 1.0
                    st.session_state[f"{kp}_step"] = 1.0
                    st.session_state[f"{kp}_max"] = current_disp + 1.0
            st.rerun()
        if _tb2.button("✅ Select all",
                        key=f"tune_select_all_{topo_key}_{fname}"):
            for spec in tuning_specs:
                key = spec[0]
                st.session_state[f"tune_{topo_key}_{key}_{fname}_chk"] = True
            st.rerun()
        if _tb3.button("❌ De-select all",
                        key=f"tune_deselect_all_{topo_key}_{fname}"):
            for spec in tuning_specs:
                key = spec[0]
                st.session_state[f"tune_{topo_key}_{key}_{fname}_chk"] = False
            st.rerun()

        # Column headers
        hdr = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
        hdr[0].markdown("**Sweep**")
        hdr[1].markdown("**Parameter**")
        hdr[2].markdown("**Min**")
        hdr[3].markdown("**Step**")
        hdr[4].markdown("**Max**")
        hdr[5].markdown("**# Calc**")

        # Pre-initialize session state defaults (only on first render of each key)
        for spec in tuning_specs:
            key, scale = spec[0], spec[2]
            current_si = float(all_p.get(key, 0.0))
            current_disp = current_si * scale
            is_zero = abs(current_si) < 1e-30
            kp = f"tune_{topo_key}_{key}_{fname}"
            if f"{kp}_chk" not in st.session_state:
                st.session_state[f"{kp}_chk"]  = not is_zero
                st.session_state[f"{kp}_min"]  = 0.0 if is_zero else current_disp - 1.0
                st.session_state[f"{kp}_step"] = 0.0 if is_zero else 1.0
                st.session_state[f"{kp}_max"]  = 0.0 if is_zero else current_disp + 1.0

        # Build per-parameter rows
        param_rows = []
        for spec in tuning_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            current_si = float(all_p.get(key, 0.0))
            current_disp = current_si * scale

            kp = f"tune_{topo_key}_{key}_{fname}"

            cols = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
            enabled = cols[0].checkbox(f"sweep_{key}", value=False, key=f"{kp}_chk",
                                       label_visibility="collapsed")
            cur_str = f"{current_disp:.2f} {unit}".strip()
            cols[1].markdown(f"**{label}** ({unit}) ({cur_str})" if unit else f"**{label}** ({cur_str})")

            if enabled:
                min_val = cols[2].number_input("Min", format="%.5g", key=f"{kp}_min",
                                               label_visibility="collapsed")
                step_val = cols[3].number_input("Step", format="%.5g", key=f"{kp}_step",
                                                label_visibility="collapsed")
                max_val = cols[4].number_input("Max", format="%.5g", key=f"{kp}_max",
                                               label_visibility="collapsed")
                sweep = _make_sweep_values(min_val, max_val, step_val)
                n_calc = len(sweep)
            else:
                min_val = current_disp
                step_val = 0.0
                max_val = current_disp
                cols[2].markdown(f"{current_disp:.2f}")
                cols[3].markdown("0")
                cols[4].markdown(f"{current_disp:.2f}")
                sweep = np.array([current_disp])
                n_calc = 1
            cols[5].markdown(f"**{n_calc}**")

            param_rows.append({
                "key": key, "label": label, "scale": scale, "unit": unit,
                "enabled": enabled, "sweep": sweep, "n_calc": n_calc,
            })

        # Header row (shown above first data row)
        # We display it once at the top with column labels
        # (Streamlit renders top-to-bottom, so we insert it via a caption)

        # Total calculation count
        total_calcs = 1
        for row in param_rows:
            if row["enabled"]:
                total_calcs *= row["n_calc"]
            # unchecked params contribute 1 (single current value)

        st.markdown(f"**Total calculations: {total_calcs:,}**")

        if total_calcs > 500_000:
            st.warning("More than 500,000 combinations — this may take a long time.")

        # ── CUDA availability note ──────────────────────────────────────
        if _HAS_CUDA:
            st.caption(f"CUDA {_CUDA_VER} detected — GPU acceleration available")

        # ── Calculate buttons — table-style layout, one bordered card
        #    per "mode" with title, description, and (CPU, CUDA) button
        #    pair.  Visually separates the four orthogonal strategies
        #    so the user doesn't have to read button labels to know
        #    what's what.
        _cpu_tag = "🦀 Rust" if _rust_active_here else "🐢 NumPy"
        # The backend (Rust vs NumPy) is no longer shown on the CPU button
        # face — it's surfaced in the hover tooltip instead.
        _cpu_help_suffix = f"\n\nCPU backend: {_cpu_tag}."

        # Modes share a 2-column inner grid (CPU left, CUDA right).
        # When no CUDA is available the right column is dropped.
        def _action_cols():
            return st.columns(2) if _HAS_CUDA else (st.columns(1)[0],
                                                     st.empty())

        # ── 🧮 Brute-force all combos ───────────────────────────────────
        _calc_all_help = (
            "Evaluates EVERY combination in the sweep grid above and ranks "
            "the top-100 results by lowest TOTAL residual.")
        with st.container(border=True):
            st.markdown(
                "**🧮 Brute force — all combos**  "
                "<span style='color:#666;font-size:0.85em'>"
                "Evaluate every combination in the grid, rank by total "
                "residual.</span>",
                unsafe_allow_html=True)
            _c_cpu, _c_cuda = _action_cols()
            cpu_clicked = _c_cpu.button(
                "Brute force with CPU",
                key=f"tune_calc_{topo_key}_{fname}",
                help=_calc_all_help + _cpu_help_suffix,
                width="stretch")
            cuda_clicked = (
                _HAS_CUDA
                and _c_cuda.button(
                    "⚡ Brute force with CUDA",
                    key=f"tune_calc_cuda_{topo_key}_{fname}",
                    help=_calc_all_help,
                    width="stretch"))

        # ── 🎯 Optimized recursive bisection ────────────────────────────
        _opt_help = (
            "Repeatedly subsamples 5 evenly-spaced values per swept parameter "
            "(e.g. min=1, step=1, max=100 → 1, 25, 50, 75, 100), picks the 2 "
            "combos with the lowest total residual, then narrows the search "
            "box to those two values and recurses.  Iteration stops once a "
            "brute-force sweep at the user's chosen step would fit ≤ 5,000,000 "
            "combos, and that final refinement is run as the closing pass.")
        with st.container(border=True):
            st.markdown(
                "**🎯 Optimized — recursive bisection**  "
                "<span style='color:#666;font-size:0.85em'>"
                "Iteratively narrow the search box: 5-point subsample → "
                "keep top-2 → recurse until the final pass fits ≤5M combos."
                "</span>",
                unsafe_allow_html=True)
            _c_cpu, _c_cuda = _action_cols()
            opt_cpu_clicked = _c_cpu.button(
                "Optimized with CPU",
                key=f"tune_calc_opt_{topo_key}_{fname}",
                help=_opt_help + _cpu_help_suffix,
                width="stretch")
            opt_cuda_clicked = (
                _HAS_CUDA
                and _c_cuda.button(
                    "⚡ Optimized with CUDA",
                    key=f"tune_calc_opt_cuda_{topo_key}_{fname}",
                    help=_opt_help,
                    width="stretch"))

        # ── 🥇 Prioritized by single S-parameter ────────────────────────
        _prio_help = (
            "Runs the same full-grid sweep but ranks results by the residual "
            "of a single chosen S-parameter instead of the total.")
        with st.container(border=True):
            st.markdown(
                "**🥇 Prioritized — rank by one S-parameter**  "
                "<span style='color:#666;font-size:0.85em'>"
                "Same full-grid sweep, but the top-100 selection is sorted "
                "by a single S-param's residual rather than the total."
                "</span>",
                unsafe_allow_html=True)
            _prio_row = st.columns([0.6, 2])
            _prio_row[0].markdown(
                "<div style='padding-top:0.4em'>"
                "<small><b>Sort by</b></small></div>",
                unsafe_allow_html=True)
            prio_metric = _prio_row[1].radio(
                "Prioritize sort metric",
                ["S11", "S12", "S21", "S22"],
                horizontal=True,
                label_visibility="collapsed",
                key=f"tune_prio_metric_{topo_key}_{fname}")
            _c_cpu, _c_cuda = _action_cols()
            prio_cpu_clicked = _c_cpu.button(
                "Prioritized with CPU",
                key=f"tune_calc_prio_{topo_key}_{fname}",
                help=_prio_help + _cpu_help_suffix,
                width="stretch")
            prio_cuda_clicked = (
                _HAS_CUDA
                and _c_cuda.button(
                    "⚡ Prioritized with CUDA",
                    key=f"tune_calc_prio_cuda_{topo_key}_{fname}",
                    help=_prio_help,
                    width="stretch"))

        # ── 🎚️  Minimize deviation (HIDDEN for now — to re-enable, flip
        #    `_SHOW_MIN_DEV` to True.  All the underlying logic at the
        #    `bal_cpu_clicked or bal_cuda_clicked` dispatch site below
        #    stays wired up; the only thing the toggle gates is the
        #    UI section.).
        _SHOW_MIN_DEV = False
        if _SHOW_MIN_DEV:
            _bal_help = (
                "Runs the full-grid sweep but discards combos whose per-port "
                "residuals are unbalanced (peak-to-peak spread > threshold).  "
                "Surviving balanced combos are then ranked by lowest total "
                "residual.  Optionally also drop combos where any single port "
                "residual exceeds a quality floor.")
            with st.container(border=True):
                st.markdown(
                    "**🎚️ Minimize deviation — peak-to-peak filter**  "
                    "<span style='color:#666;font-size:0.85em'>"
                    "Discard unbalanced combos, then rank survivors by "
                    "total residual.</span>",
                    unsafe_allow_html=True)
                _bal_inputs = st.columns([1, 1, 2])
                bal_dev_threshold = _bal_inputs[0].number_input(
                    "Max per-port deviation (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(st.session_state.get(
                        f"tune_bal_dev_{topo_key}_{fname}", 3.0)),
                    step=0.5, format="%.2f",
                    key=f"tune_bal_dev_{topo_key}_{fname}",
                    help="Max allowed |max(S11,S12,S21,S22) − min(...)| residual "
                         "spread (in %).  Smaller = more balanced.")
                _bal_use_res = _bal_inputs[1].checkbox(
                    "Use residual cap",
                    value=False,
                    key=f"tune_bal_use_res_{topo_key}_{fname}")
                bal_res_threshold = _bal_inputs[2].number_input(
                    "Max per-port residual (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(st.session_state.get(
                        f"tune_bal_res_{topo_key}_{fname}", 5.0)),
                    step=0.5, format="%.2f",
                    key=f"tune_bal_res_{topo_key}_{fname}",
                    disabled=(not _bal_use_res),
                    help="Quality floor — drop combos where any port's residual "
                         "exceeds this.")
                _c_cpu, _c_cuda = _action_cols()
                bal_cpu_clicked = _c_cpu.button(
                    f"CPU — {_cpu_tag}",
                    key=f"tune_calc_bal_{topo_key}_{fname}",
                    help=_bal_help,
                    width="stretch")
                bal_cuda_clicked = (
                    _HAS_CUDA
                    and _c_cuda.button(
                        "⚡ CUDA",
                        key=f"tune_calc_bal_cuda_{topo_key}_{fname}",
                        help=_bal_help,
                        width="stretch"))
        else:
            # Section hidden — neutralise the flags + defaults so the
            # downstream dispatch (`elif bal_cpu_clicked or
            # bal_cuda_clicked:` further below) is a no-op.
            bal_cpu_clicked = False
            bal_cuda_clicked = False
            bal_dev_threshold = 3.0
            _bal_use_res = False
            bal_res_threshold = 5.0

        # # ── Auto (Nelder-Mead) ──────────────────────────────────────────
        # st.caption(
        #     "🤖 **Auto (Nelder-Mead)** seeds the simplex from the current "
        #     "parameter values, expands/contracts adaptively (large moves when "
        #     "residual is high, small moves when low), and converges to a "
        #     "local minimum within the **Min/Max** boxes above.  Cheap, "
        #     "gradient-free, and respects the Stop button.  Most useful as a "
        #     "polish step after a coarse grid sweep.")
        # _auto_inputs = st.columns([1, 1, 2])
        # auto_max_iter = int(_auto_inputs[0].number_input(
        #     "Max iterations",
        #     min_value=10, max_value=10_000,
        #     value=int(st.session_state.get(
        #         f"tune_auto_iter_{topo_key}_{fname}", 200)),
        #     step=10,
        #     key=f"tune_auto_iter_{topo_key}_{fname}",
        #     help="Nelder-Mead iteration cap (function evals ≈ iters × (n+1))."))
        # auto_restart = _auto_inputs[1].checkbox(
        #     "Random restart on collapse",
        #     value=True,
        #     key=f"tune_auto_restart_{topo_key}_{fname}",
        #     help="If the simplex shrinks below tol before convergence, "
        #          "perturb x0 and run a second pass.")
        # _auto_btn = _auto_inputs[2].columns([1, 1] if _HAS_CUDA else [1])
        # auto_cpu_clicked = _auto_btn[0].button(
        #     "🤖 Auto (Nelder-Mead)",
        #     key=f"tune_calc_auto_{topo_key}_{fname}")
        # auto_cuda_clicked = (
        #     _HAS_CUDA
        #     and _auto_btn[1].button(
        #         "⚡ Auto with CUDA",
        #         key=f"tune_calc_auto_cuda_{topo_key}_{fname}"))

        # ── Helper: format a "best result" markdown line including all params ──
        #
        # • Bold the parameter name AND value for any param that is currently
        #   ticked for sweep (its `_chk` session key is True).
        # • Hide all-zero pad caps (Cpbe/Cpce/Cpbc) and lead inductances
        #   (Lb/Lc/Le) since they're often left at 0 and just add noise.
        _HIDE_IF_ZERO = {"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le"}
        def _best_summary_md(best_row, label="Best so far"):
            head = (f"**{label} — Total: {best_row['Total Residual (%)']:.2f}%  |  "
                    f"S11: {best_row['S11 (%)']:.2f}%  S12: {best_row['S12 (%)']:.2f}%  "
                    f"S21: {best_row['S21 (%)']:.2f}%  S22: {best_row['S22 (%)']:.2f}%**")
            parts = []
            for spec in tuning_specs:
                key = spec[0]; label_p = spec[1]
                unit = spec[3] if len(spec) > 3 else ""
                fmt = spec[4] if len(spec) > 4 else "%.4g"
                col = f"{label_p} ({unit})" if unit else label_p
                if col not in best_row.index:
                    continue
                v = float(best_row[col])
                if key in _HIDE_IF_ZERO and np.isfinite(v) and abs(v) < 1e-30:
                    continue
                is_swept = bool(st.session_state.get(
                    f"tune_{topo_key}_{key}_{fname}_chk", False))
                sval = (fmt % v) if np.isfinite(v) else "NaN"
                name_md = f"<b>{label_p}</b>" if is_swept else label_p
                val_md  = (f"<b>{sval} {unit}</b>".rstrip()
                           if is_swept else f"{sval} {unit}".rstrip())
                parts.append(f"{name_md}: {val_md}")
            if parts:
                head += "  \n<small>" + ", ".join(parts) + "</small>"
            return head

        def _run_one_sweep(*, use_cuda: bool,
                            sweep_lists_override: dict | None = None,
                            sort_metric: str = "Total",
                            phase_label: str = "",
                            phase_suffix: str = "",
                            dev_threshold: float | None = None,
                            res_threshold: float | None = None):
            """Run one full parameter-sweep pass.

            sweep_lists_override : dict[str, np.ndarray] | None
                When provided, overrides the per-row sweep arrays for the listed
                keys. Other rows fall back to their checkbox/min/step/max state.
            sort_metric : "Total" | "S11" | "S12" | "S21" | "S22"
                Selects the residual metric used to rank the top-K results.
                Total residual / per-port residuals are still recorded for
                every kept combo regardless of choice.
            phase_label / phase_suffix
                phase_label is shown in the progress text; phase_suffix is
                appended to the Stop-button key so multi-pass runs (e.g.
                Optimized) don't clash on a duplicate Streamlit widget key.
            dev_threshold : float | None
                Max allowed peak-to-peak (max-min) per-port residual deviation
                in %.  Combos exceeding this are pushed to the bottom of the
                ranking — surviving combos are ranked normally by sort_metric.
                Used by the "Minimize deviation" button to pick the lowest
                total residual *among balanced* combinations.
            res_threshold : float | None
                Max allowed per-port residual in %.  Combos with any port
                exceeding this are pushed to the bottom of the ranking.
            """
            xp = _cp if use_cuda else np
            has_batch = hasattr(model_cls, "simulate_batch")
            _mode_label = "CUDA" if use_cuda else "CPU"

            # Only the top-K combos (lowest residuals) are kept in memory.
            # Saves >99% RAM on huge sweeps and side-steps MemoryError.
            TOP_K = 100

            # Build sweep lists (display units) — short host arrays of unique
            # values per parameter (length 1 for unswept params).
            sweep_keys, sweep_lists = [], []
            sweep_scales, sweep_labels, sweep_units = [], [], []
            for row in param_rows:
                sweep_keys.append(row["key"])
                sweep_scales.append(row["scale"])
                sweep_labels.append(row["label"])
                sweep_units.append(row["unit"])
                if sweep_lists_override and row["key"] in sweep_lists_override:
                    sweep_lists.append(np.asarray(
                        sweep_lists_override[row["key"]], dtype=np.float64))
                elif row["enabled"]:
                    sweep_lists.append(np.asarray(row["sweep"], dtype=np.float64))
                else:
                    sweep_lists.append(np.array(
                        [float(all_p.get(row["key"], 0.0)) * row["scale"]],
                        dtype=np.float64))

            n_params = len(sweep_keys)
            L_list = [int(len(sl)) for sl in sweep_lists]
            n_total = 1
            for L in L_list:
                n_total *= L

            # Strides for "ij"-order linear→multi-index decomposition:
            #   combo k → (i_0, i_1, …, i_{n-1})
            #   where i_p = (k // strides[p]) % L_list[p]
            strides = [1] * n_params
            for i in range(n_params - 2, -1, -1):
                strides[i] = strides[i + 1] * L_list[i + 1]

            # Indices of params that actually vary — only these need to be
            # carried per-row in the top-K state. Constants get filled in at
            # display time from sweep_lists.
            swept_idx_full = [i for i in range(n_params) if L_list[i] > 1]
            n_swept = len(swept_idx_full)

            # ── N-D parameter-sweep layout ──────────────────────────────────
            # Each *swept* parameter gets its own broadcast axis.  Sub-networks
            # whose inputs are not all swept are computed at their *native*
            # dimensionality and broadcast across the rest of the grid — so
            # `h/i` in `g·h/i` is built once with shape `(1,Lh,Li,1)` and
            # reused for every value of `g`, with no recomputation.
            #
            # Layout convention:
            #   axis 0 .. n_swept_dims-1  → one length-L_i axis per swept param
            #   axis n_swept_dims         → frequency (length N_freq)
            # Constants stay as Python scalars (broadcast to anything).
            #
            # Output S has shape  (L_0, L_1, ..., L_{n-1}, N_freq, 2, 2)
            # — flattened to (B_inner, N_freq, 2, 2) for residual scoring.

            swept_pos_to_param_idx = [i for i in range(n_params) if L_list[i] > 1]
            n_swept_dims = len(swept_pos_to_param_idx)
            inner_lens = [int(L_list[i]) for i in swept_pos_to_param_idx]

            # Lookup table of *display* values for index→display mapping
            L_max = max(max(L_list), 1)
            sweep_table_h = np.zeros((n_params, L_max), dtype=np.float64)
            for i, sl in enumerate(sweep_lists):
                sweep_table_h[i, :L_list[i]] = sl
            sweep_table_dev = xp.asarray(sweep_table_h)                            # (n_params, L_max)
            if n_swept > 0:
                swept_indices_dev = xp.asarray(np.asarray(swept_idx_full, dtype=np.int64))
            else:
                swept_indices_dev = None

            # Constant scalars passed straight into simulate_batch
            const_si = {sweep_keys[i]: float(sweep_lists[i][0]) / float(sweep_scales[i])
                        for i in range(n_params) if L_list[i] == 1}
            varies_mask = [L_list[i] > 1 for i in range(n_params)]
            swept_set = {sweep_keys[i] for i in range(n_params) if varies_mask[i]}

            # ── FP32 sweep mode (GPU only, model-opt-in) ─────────────────────
            # Cheng T/Pi's batched intrinsic-Y kernels honour cache["_cdtype"]
            # so the main loop can run entirely in complex64.  fp64 rerank
            # below replaces the residuals on the surviving top-K combos so
            # the final ranking and the persisted residuals are double-precision.
            #
            # CPU mode is left at fp64: scalar AVX is the same width for
            # both, and reduced precision saves no wall time on this code.
            use_fp32_sweep = bool(use_cuda and getattr(
                model_cls, "SUPPORTS_FP32_SWEEP", False))
            cdtype_main = np.complex64 if use_fp32_sweep else np.complex128
            rdtype_main = np.float32   if use_fp32_sweep else np.float64

            # Move S_mea onto the compute device once.  Two copies for fp32:
            # the cast (complex64) version drives the main loop, the original
            # complex128 stays alive for the fp64 rerank.
            S_mea_dev_fp64 = xp.asarray(S_raw)
            if use_fp32_sweep:
                S_mea_dev = S_mea_dev_fp64.astype(np.complex64)
            else:
                S_mea_dev = S_mea_dev_fp64

            # ── N-D omega: (1,)*n_swept_dims + (N_freq,) ────────────────────
            # Number of leading 1s = number of swept axes, so omega broadcasts
            # cleanly against any swept-param tensor regardless of which axes
            # it occupies.
            N_freq = len(freq)
            omega_shape = (1,) * n_swept_dims + (N_freq,)
            omega_dev = xp.asarray(
                2.0 * np.pi * np.asarray(freq, dtype=np.float64)
            ).reshape(omega_shape)

            # ── Pre-bake fully-constant sub-networks (lifted out of the loop)
            # If a sub-network has *zero* swept inputs, its tensor is built
            # once here at shape (1,...,1,N_freq) and reused every iteration.
            # If a sub-network has *some* swept inputs, the model code will
            # build it inside simulate_batch — but at its own native dims
            # (smaller than the full inner block), thanks to N-D broadcasting.
            _PAD_CAP_KEYS = ("Cpbe", "Cpce", "Cpbc")
            _SER_LEAD_KEYS = ("Rpb", "Rpc", "Rpe", "Lb", "Lc", "Le")
            _CHENG_EXTR_KEYS = ("Cbex", "Cbcx")
            # Cheng-T intrinsic sub-expression groups: each is pre-bakeable
            # whenever *none* of its inputs are swept.
            _T_ZBE_KEYS   = ("Rbe", "Cbe")
            _T_ZBC_KEYS   = ("Rbc", "Cbc")
            _T_ALPHA_KEYS = ("alpha0", "tauB", "tauC")
            _T_INT_ALL    = ("Rbi",) + _T_ZBE_KEYS + _T_ZBC_KEYS + _T_ALPHA_KEYS
            # Cheng-π intrinsic sub-expression groups
            _PI_YBE_KEYS  = ("Rbe", "Cbe")
            _PI_YBC_KEYS  = ("Rbc", "Cbc")
            _PI_GM_KEYS   = ("Gm0", "tau")
            _PI_INT_ALL   = ("Rbi",) + _PI_YBE_KEYS + _PI_YBC_KEYS + _PI_GM_KEYS

            static_p = dict(all_p)
            for _k, _v in const_si.items():
                static_p[_k] = _v

            # The fp64 cache is the canonical one — built first, then
            # cast to a parallel fp32 cache for the main loop if eligible.
            # The fp64 cache is also kept alive for the fp64 rerank below.
            static_cache_fp64 = {"omega": omega_dev}
            _cached_msgs = []
            if not (set(_PAD_CAP_KEYS) & swept_set):
                static_cache_fp64["Y_pad"] = build_Y_pad_batch(
                    static_p, omega_dev, 1, N_freq, xp)
                _cached_msgs.append("Y_pad")
            if not (set(_SER_LEAD_KEYS) & swept_set):
                static_cache_fp64["Z_ser"] = build_Z_ser_batch(
                    static_p, omega_dev, 1, N_freq, xp)
                _cached_msgs.append("Z_ser")
            if not (set(_CHENG_EXTR_KEYS) & swept_set):
                _Cbex_c = float(static_p.get("Cbex", 0.0))
                _Cbcx_c = float(static_p.get("Cbcx", 0.0))
                static_cache_fp64["Y_extr"] = (1j * omega_dev * _Cbex_c,
                                               1j * omega_dev * _Cbcx_c)
                _cached_msgs.append("Y_extr")

            # ── Cheng-specific intrinsic sub-expression caches ──────────────
            # These mirror what _Y_int_T_batch / _Y_int_Pi_batch will look up:
            # if a (Rbe,Cbe) / (Rbc,Cbc) / (alpha0,tauB,tauC) pair has all
            # constant inputs we can pre-build the result once and reuse it
            # every chunk.  When *every* intrinsic param is constant we go
            # one step further and pre-build the four intrinsic-Y planes
            # outright, skipping the entire per-chunk inv() of Z_in.
            _model_short = getattr(model_cls, "SHORT", "")
            if _model_short == "T":
                if not (set(_T_ZBE_KEYS) & swept_set):
                    _Rbe_c = float(static_p.get("Rbe", 1.0))
                    _Cbe_c = float(static_p.get("Cbe", 0.0))
                    static_cache_fp64["Zbe"] = (
                        _Rbe_c / (1.0 + 1j * omega_dev * _Rbe_c * _Cbe_c))
                    _cached_msgs.append("Zbe")
                if not (set(_T_ZBC_KEYS) & swept_set):
                    _Rbc_c = float(static_p.get("Rbc", 1.0))
                    _Cbc_c = float(static_p.get("Cbc", 0.0))
                    static_cache_fp64["Zbc"] = (
                        _Rbc_c / (1.0 + 1j * omega_dev * _Rbc_c * _Cbc_c))
                    _cached_msgs.append("Zbc")
                if not (set(_T_ALPHA_KEYS) & swept_set):
                    _a0 = float(static_p.get("alpha0", 0.0))
                    _tC = float(static_p.get("tauC",   0.0))
                    _tB = float(static_p.get("tauB",   0.0))
                    static_cache_fp64["alpha"] = (
                        _a0 * xp.exp(-1j * omega_dev * _tC)
                        / (1.0 + 1j * omega_dev * _tB))
                    _cached_msgs.append("alpha")
                if not (set(_T_INT_ALL) & swept_set):
                    # Inline what _Y_int_T_batch would compute, once.
                    _Rbi_c = float(static_p.get("Rbi", 0.0))
                    _Zbe_c = static_cache_fp64["Zbe"]
                    _Zbc_c = static_cache_fp64["Zbc"]
                    _alpha_c = static_cache_fp64["alpha"]
                    _z00 = _Rbi_c + _Zbe_c
                    _z01 = _Zbe_c
                    _z10 = _Zbe_c - _alpha_c * _Zbc_c
                    _z11 = (1.0 - _alpha_c) * _Zbc_c + _Zbe_c
                    _det = _z00 * _z11 - _z01 * _z10
                    _inv_det = 1.0 / _det
                    static_cache_fp64["T_int_planes"] = (
                         _z11 * _inv_det,
                        -_z01 * _inv_det,
                        -_z10 * _inv_det,
                         _z00 * _inv_det,
                    )
                    _cached_msgs.append("T_int_planes")
            elif _model_short == "pi":
                if not (set(_PI_YBE_KEYS) & swept_set):
                    _Rbe_c = float(static_p.get("Rbe", 1.0))
                    _Cbe_c = float(static_p.get("Cbe", 0.0))
                    static_cache_fp64["Ybe"] = (
                        1.0 / _Rbe_c + 1j * omega_dev * _Cbe_c)
                    _cached_msgs.append("Ybe")
                if not (set(_PI_YBC_KEYS) & swept_set):
                    _Rbc_c = float(static_p.get("Rbc", 1e9))
                    _Cbc_c = float(static_p.get("Cbc", 0.0))
                    static_cache_fp64["Ybc"] = (
                        1.0 / _Rbc_c + 1j * omega_dev * _Cbc_c)
                    _cached_msgs.append("Ybc")
                if not (set(_PI_GM_KEYS) & swept_set):
                    _Gm0_c = float(static_p.get("Gm0", 0.0))
                    _tau_c = float(static_p.get("tau", 0.0))
                    static_cache_fp64["gm"] = (
                        _Gm0_c * xp.exp(-1j * omega_dev * _tau_c))
                    _cached_msgs.append("gm")
                if not (set(_PI_INT_ALL) & swept_set):
                    # Inline what _Y_int_Pi_batch would compute, once.
                    _Rbi_c   = float(static_p.get("Rbi", 0.0))
                    _Ybe_c   = static_cache_fp64["Ybe"]
                    _Ybc_c   = static_cache_fp64["Ybc"]
                    _gm_c    = static_cache_fp64["gm"]
                    _yc00 = _Ybe_c + _Ybc_c
                    _yc01 = -_Ybc_c
                    _yc10 = _gm_c  - _Ybc_c
                    _yc11 = _Ybc_c
                    _idc  = 1.0 / (_yc00 * _yc11 - _yc01 * _yc10)
                    _zc00 =  _yc11 * _idc + _Rbi_c
                    _zc01 = -_yc01 * _idc
                    _zc10 = -_yc10 * _idc
                    _zc11 =  _yc00 * _idc
                    _idi  = 1.0 / (_zc00 * _zc11 - _zc01 * _zc10)
                    static_cache_fp64["Pi_int_planes"] = (
                         _zc11 * _idi,
                        -_zc01 * _idi,
                        -_zc10 * _idi,
                         _zc00 * _idi,
                    )
                    _cached_msgs.append("Pi_int_planes")

            # ── Build the fp32 cache (cast of fp64 cache) for main loop ─────
            # Everything except `omega` and the dtype tag is a complex tensor;
            # cast each in-place to complex64.  `omega` is real — keep its
            # own copy at float32 so multiply-by-J stays in c64.
            if use_fp32_sweep:
                omega_dev_fp32 = omega_dev.astype(np.float32)
                static_cache = {"omega": omega_dev_fp32, "_cdtype": np.complex64}
                for _k, _v in static_cache_fp64.items():
                    if _k in ("omega", "_cdtype"):
                        continue
                    if isinstance(_v, tuple):
                        static_cache[_k] = tuple(
                            _p.astype(np.complex64) if hasattr(_p, "astype") else _p
                            for _p in _v
                        )
                    elif hasattr(_v, "astype"):
                        static_cache[_k] = _v.astype(np.complex64)
                    else:
                        static_cache[_k] = _v
            else:
                static_cache = static_cache_fp64
                static_cache["_cdtype"] = np.complex128

            if _cached_msgs:
                _prec_lbl = "fp32" if use_fp32_sweep else "fp64"
                st.caption(f"Pre-baked constant networks ({_prec_lbl}): "
                           + ", ".join(_cached_msgs))
                print(f"[tune] pre-baked ({_prec_lbl}): "
                      f"{', '.join(_cached_msgs)}", flush=True)

            col_names = ["Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
            for lbl, u in zip(sweep_labels, sweep_units):
                col_names.append(f"{lbl} ({u})" if u else lbl)

            sess_key = f"tune_df_{topo_key}_{fname}"

            def _topk_to_df(top_arr_host):
                """Display DataFrame is the top-10 per metric across
                {Total, S11, S12, S21, S22}, deduped — 10 ≤ R ≤ 50 rows
                sorted by Total Residual.  See ``_multi_metric_top_n``."""
                return pd.DataFrame(
                    _multi_metric_top_n(top_arr_host, per_metric=10),
                    columns=col_names)

            # ── Auto slab sizing from device free memory ────────────────────
            # Per-combo working-set estimate.
            #
            # The model uses *scalar plane* representation (4 separate (B,N)
            # complex128 planes per 2×2 matrix, not full (B,N,2,2) tensors)
            # — see _sim_wrap_batch in models/cheng.py.  Realistic peak live
            # planes per combo:
            #   • Y_int (4 planes) + Y_extr (2) + Y_ex (4)       = 10 planes
            #   • Z_ex (4) + Z_ser (4) + Z_tot (4)               = 12 planes
            #   • Y_tot (4) + Y_pad (4) + Y_total (4)            = 12 planes
            #   • Y_norm/M/M_inv/N/S working set                 = 12 planes
            #   • Final stacked (B,N,2,2)                        =  4 planes
            #   • Residual diff/num/val                          =  6 planes
            # Each plane = 16 bytes/element × N_freq elements per combo
            # in complex128, or 8 bytes/element in complex64 (fp32 sweep).
            # Initial guess: ~60 planes × 16 = 960 B/combo per N_freq, with a
            # 1.25× safety margin → 12× N_freq complex128 tensor-equivalents.
            # In fp32 mode the same 60 planes are 8 B each, so the bytes
            # estimate halves.  Replaced after iter 1 by an empirical
            # measurement (see `_calibrated` below) — the initial guess only
            # governs the *first* block size before we have real data.
            _bytes_per_complex = 8 if use_fp32_sweep else 16
            per_combo_bytes = (_bytes_per_complex * 4) * N_freq * 12
            _calibrated = False
            free_label = ""
            if use_cuda:
                try:
                    # Free up any cached blocks first so the query reflects
                    # what we can *actually* allocate now (not what's been
                    # pinned by previous calculations).
                    try:
                        _cp.get_default_memory_pool().free_all_blocks()
                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                    free_b, total_b = _cp.cuda.runtime.memGetInfo()
                    dev = _cp.cuda.Device(0)
                    sm_count = dev.attributes.get("MultiProcessorCount", 0)
                    free_label = (f"GPU{dev.id}: {free_b/1024**3:.2f}/"
                                  f"{total_b/1024**3:.2f} GiB free  ·  {sm_count} SMs")
                    # 0.55 keeps ~45% of free VRAM as headroom for pool
                    # 0.95 uses more VRAM
                    # fragmentation, top-K scratch, persistent buffers, and
                    # the measurement S_mea_dev tensor.
                    budget = int(free_b * 0.95)
                except Exception:
                    budget = 1 * 1024**3
            else:
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                    free_label = f"CPU RAM: {avail/1024**3:.2f} GiB free"
                except Exception:
                    avail = 4 * 1024**3
                budget = int(avail * 0.25)

            max_inner = max(1, budget // max(per_combo_bytes, 1))
            if not use_cuda:
                # CPU mode: cap inner block size more aggressively
                max_inner = min(max_inner, 65_536)
            else:
                # GPU mode: clamp to a sane upper bound to bound output size
                max_inner = min(max_inner, 16_777_216)  # 16M combos per slab

            # ── Multi-axis block sizing ─────────────────────────────────────
            # Pick a per-axis block_shape (one length per swept axis) such
            # that prod(block_shape) <= max_inner.  We start with the full
            # inner_lens and greedily halve the *largest* axis until the
            # product fits.  This generalises the old "slab one axis" logic
            # to handle the case where multiple axes need slabbing — i.e.
            # when the product of all "other" axes alone exceeds VRAM.
            def _shrink_to_budget(shape, budget):
                """Greedy: halve the largest axis until prod(shape) <= budget."""
                bs = list(shape)
                if not bs:
                    return bs
                while True:
                    p = 1
                    for v in bs:
                        p *= v
                    if p <= budget or budget < 1:
                        return bs
                    k_max = 0
                    for k in range(1, len(bs)):
                        if bs[k] > bs[k_max]:
                            k_max = k
                    if bs[k_max] <= 1:
                        return bs  # cannot shrink further
                    bs[k_max] = max(1, bs[k_max] // 2)

            def _shrink_one_step(shape):
                """Halve the largest axis once. Returns (new_shape, did_shrink)."""
                bs = list(shape)
                if not bs:
                    return bs, False
                k_max = 0
                for k in range(1, len(bs)):
                    if bs[k] > bs[k_max]:
                        k_max = k
                if bs[k_max] <= 1:
                    return bs, False
                bs[k_max] = max(1, bs[k_max] // 2)
                return bs, True

            def _grow_to_budget(shape, full_lens, budget):
                """Greedy: double the smallest still-growable axis until either
                doubling again would exceed `budget`, or every axis is at its
                maximum (full_lens[k]).  Returns the new shape (a list)."""
                bs = list(shape)
                if not bs:
                    return bs
                while True:
                    p = 1
                    for v in bs:
                        p *= v
                    if p * 2 > budget:
                        return bs
                    # Pick smallest axis that can still grow
                    cand_k = -1
                    cand_v = None
                    for k in range(len(bs)):
                        if bs[k] < full_lens[k]:
                            if cand_k < 0 or bs[k] < cand_v:
                                cand_k = k
                                cand_v = bs[k]
                    if cand_k < 0:
                        return bs  # nothing left to grow
                    bs[cand_k] = min(full_lens[cand_k], max(2, bs[cand_k] * 2))

            if n_swept_dims == 0:
                block_shape = []
                inner_block = 1
            else:
                block_shape = _shrink_to_budget(inner_lens, max_inner)
                inner_block = 1
                for v in block_shape:
                    inner_block *= v

            CHUNK_MAX = inner_block  # upper bound on per-iter combo count

            import sys as _sys, time as _time
            _t_start = _time.time()
            if free_label:
                st.caption(free_label)
            if n_swept_dims == 0:
                _slab_label = "single combo"
            elif inner_block >= n_total:
                _slab_label = (f"single N-D block, "
                               f"{inner_block:,} combos / iter")
            else:
                # Shape summary: e.g. "block=(8,32,7), 1,792 combos / iter"
                _shape_str = ",".join(str(v) for v in block_shape)
                _full_str  = ",".join(str(v) for v in inner_lens)
                # Conservative iter count from current block (may rise if
                # shrunk later, fall if grown — we don't grow).
                _n_iters_pre = 1
                for k, v in enumerate(block_shape):
                    _n_iters_pre *= (inner_lens[k] + v - 1) // v
                _slab_label = (f"block=({_shape_str})/({_full_str}), "
                               f"{inner_block:,} combos / iter, "
                               f"≥{_n_iters_pre} iters")
            print(f"\n[tune] start  mode={_mode_label}  total={n_total:,}  "
                  f"{_slab_label}  batched={has_batch}  {free_label}", flush=True)

            # ── UI placeholders ────────────────────────────────────────────
            ui_cols = st.columns([5, 1])
            with ui_cols[0]:
                _prog_lbl = f"Tuning {phase_label} ({_mode_label})…" if phase_label \
                            else f"Tuning ({_mode_label})…"
                progress = st.progress(0, text=_prog_lbl)
            with ui_cols[1]:
                stop_box = st.empty()
            best_box = st.empty()      # live "best so far" line

            # The Stop button works by triggering a Streamlit re-run on click;
            # the next st.* call inside the loop raises RerunException, which
            # we catch and turn into a clean cancellation.  No on_click needed.
            stop_key = f"tune_stop_{topo_key}_{fname}{phase_suffix}"
            stop_box.button(
                "⏹ Stop",
                key=stop_key,
                help="Stop the calculation. The best results found so far are kept.",
                type="secondary",
            )

            cancelled = False

            # ── Persistent device buffers for the top-K accumulator ─────────
            # Inf placeholders ensure new finite values always displace them.
            # top_4 must also start at inf — Prioritize sorts by view_4[:, _smap[sort_metric]],
            # so zero placeholders would otherwise out-rank every real residual and the
            # "Best so far" panel would stay empty for the entire sweep.
            top_res    = xp.full(TOP_K, xp.inf, dtype=xp.float64)
            top_4      = xp.full((TOP_K, 4), xp.inf, dtype=xp.float64)
            top_swept  = xp.zeros((TOP_K, max(n_swept, 1)), dtype=xp.float64)

            # ── Pre-allocated scratch buffers for the merge step ────────────
            # Sized for TOP_K + the *maximum* chunk we'd ever submit.  Reused
            # every iteration → zero per-chunk allocation churn for top-K.
            SCRATCH = TOP_K + CHUNK_MAX
            scratch_res   = xp.empty(SCRATCH, dtype=xp.float64)
            scratch_4     = xp.empty((SCRATCH, 4), dtype=xp.float64)
            scratch_swept = xp.empty((SCRATCH, max(n_swept, 1)), dtype=xp.float64)

            # Threshold used to distinguish "real residual" from the BIG=1e308
            # clamp the inner loop assigns to combos that fail the
            # deviation / residual filters or produced NaN/inf simulations.
            # Real residuals are percentages (typically 0.01–1000); 1e100 is a
            # comfortable separator below BIG and above any plausible value.
            _VALID_RES_MAX = 1.0e100

            def _sync_topk_host():
                """Pull the top-K state to host as a (n, 5+n_params) numpy array.
                Drops inf placeholders and BIG-clamped (filter-rejected) rows.
                Constants are filled from sweep_lists.
                """
                if use_cuda:
                    tr = _cp.asnumpy(top_res)
                    t4 = _cp.asnumpy(top_4)
                    ts = _cp.asnumpy(top_swept) if n_swept > 0 else \
                         np.zeros((TOP_K, 0), dtype=np.float64)
                else:
                    tr = np.asarray(top_res)
                    t4 = np.asarray(top_4)
                    ts = np.asarray(top_swept) if n_swept > 0 else \
                         np.zeros((TOP_K, 0), dtype=np.float64)
                valid = np.isfinite(tr) & (tr < _VALID_RES_MAX)
                n = int(valid.sum())
                if n == 0:
                    return None
                out = np.empty((n, 5 + n_params), dtype=np.float64)
                out[:, 0]   = tr[valid]
                out[:, 1:5] = t4[valid]
                j = 0
                for i in range(n_params):
                    if L_list[i] == 1:
                        out[:, 5 + i] = float(sweep_lists[i][0])
                    else:
                        out[:, 5 + i] = ts[valid, j]
                        j += 1
                return out

            def _persist_topk():
                """Best-effort persist to session_state. Safe to call from
                anywhere (including the finally clause)."""
                try:
                    h = _sync_topk_host()
                    if h is not None:
                        st.session_state[sess_key] = _topk_to_df(h)
                except Exception:
                    pass

            def _release_gpu_memory():
                """Best-effort: synchronize the device, run gc twice (to break
                cycles), then return all idle pool blocks to the driver.

                Order matters:
                  1. Synchronize first — async kernels may still be holding
                     tensor inputs alive on the stream.  Without sync,
                     `free_all_blocks()` would skip those blocks.
                  2. gc.collect() twice — CuPy ndarrays often participate in
                     reference cycles via residual/topk dicts; one pass may
                     not break them all.
                  3. Free both device and pinned-host memory pools.
                """
                if not use_cuda:
                    gc.collect()
                    gc.collect()
                    return
                try:
                    _cp.cuda.runtime.deviceSynchronize()
                except Exception:
                    pass
                gc.collect()
                gc.collect()
                try:
                    _cp.get_default_memory_pool().free_all_blocks()
                    _cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            try:
                if not has_batch:
                    raise RuntimeError(
                        f"Model {model_cls.__name__} has no simulate_batch — "
                        f"cannot run batched tuning. Implement simulate_batch.")

                processed = 0
                last_ui = 0.0
                chunks_done = 0
                last_chunk_ms = 0.0
                # axis_offsets[k] = current row index along swept axis k.
                # Advances by block_shape[k] after each successful iteration,
                # carrying over to the next-outer axis at L_k.
                axis_offsets = [0] * n_swept_dims
                # Resize gate — never re-poll the driver more than once per
                # this many seconds.  Polling memGetInfo() is cheap (~µs) but
                # the *real* cost we're avoiding is fragmenting the allocator
                # by trying to resize too aggressively.
                _last_resize_t = 0.0
                _RESIZE_INTERVAL = 2.0
                # Consecutive-OOM counter — drives the escalating recovery
                # strategy: 1st OOM just shrinks (cheap), 2nd consecutive OOM
                # also flushes the pool (expensive but reclaims everything).
                # Reset to 0 after any successful iteration.
                _oom_streak = 0

                while processed < n_total:
                    # ── Adaptive multi-axis block sizing (gated). ──────────
                    # We re-query free VRAM at most once every _RESIZE_INTERVAL
                    # seconds.  Crucially we do NOT call free_all_blocks() —
                    # flushing the pool every iteration kills allocator
                    # amortization and forces every alloc to fall through to
                    # cudaMalloc, which on Windows WDDM saturates the Copy
                    # engine with page-table updates and starves Compute.
                    #
                    # Instead we use the pool's own bookkeeping (free_bytes
                    # = bytes already cached and re-allocatable for free) to
                    # build an "effective free" estimate without disturbing
                    # the pool.  Once `_calibrated`, the budget is also used
                    # to decide whether the block can grow.
                    _now_t = _time.time()
                    if (use_cuda and n_swept_dims > 0
                            and (_now_t - _last_resize_t) >= _RESIZE_INTERVAL):
                        _last_resize_t = _now_t
                        try:
                            _mp = _cp.get_default_memory_pool()
                            _pool_free = _mp.free_bytes()    # cached, re-allocatable
                            _drv_free, _ = _cp.cuda.runtime.memGetInfo()
                            # Effective free = what driver reports + what's
                            # already cached in the pool (the pool's cached
                            # blocks count against driver-reported free, but
                            # are available to us without a malloc).
                            _eff_free = _drv_free + _pool_free
                            _budget_now = int(_eff_free * 0.55)
                            _max_inner_now = max(
                                1, _budget_now // max(per_combo_bytes, 1))
                            _cur_prod = 1
                            for v in block_shape:
                                _cur_prod *= v
                            # Shrink if we're now over budget
                            _shrunk = _shrink_to_budget(block_shape, _max_inner_now)
                            _shrunk_prod = 1
                            for v in _shrunk:
                                _shrunk_prod *= v
                            if _shrunk_prod < _cur_prod:
                                block_shape = _shrunk
                                print(f"\n[tune] free VRAM dropped → block shrunk to "
                                      f"({','.join(str(v) for v in block_shape)}) "
                                      f"= {_shrunk_prod:,} combos / iter",
                                      flush=True)
                            elif _calibrated:
                                # Try to grow.  Only after calibration — using
                                # the pessimistic initial estimate to compute
                                # a grow target would let us grow into an OOM.
                                _grown = _grow_to_budget(
                                    block_shape, inner_lens, _max_inner_now)
                                _grown_prod = 1
                                for v in _grown:
                                    _grown_prod *= v
                                # Only act if growth is meaningful (≥ +50%)
                                if _grown_prod >= int(_cur_prod * 1.5):
                                    block_shape = _grown
                                    print(f"\n[tune] free VRAM ample → block grown to "
                                          f"({','.join(str(v) for v in block_shape)}) "
                                          f"= {_grown_prod:,} combos / iter",
                                          flush=True)
                        except Exception:
                            pass

                    # ── Compute this iteration's per-axis extent ────────────
                    # cur_shape[k] = how many rows of axis k this iteration
                    # covers, capped at the remainder of the axis.
                    if n_swept_dims == 0:
                        cur_shape = ()
                        inner_shape = ()
                    else:
                        cur_shape = tuple(
                            min(block_shape[k], inner_lens[k] - axis_offsets[k])
                            for k in range(n_swept_dims)
                        )
                        inner_shape = cur_shape
                    B_inner = 1
                    for L in inner_shape:
                        B_inner *= L
                    if B_inner == 0:
                        break
                    _t_chunk = _time.time()

                    # ── Build N-D parameter tensors for this slab ───────────
                    # Each swept param gets shape (1,..,L,..,1,1) — its own
                    # length-L axis at its position, 1s elsewhere, and a
                    # trailing 1 for the freq slot.  Constants stay scalar.
                    # On host the per-param values are tiny: only the *device*
                    # tensors matter for VRAM, and they're (1,..,L,..,1,1).
                    p_nd = None
                    S_batch_nd = None
                    S_flat = None
                    res = None
                    cur_total = None
                    cur_4 = None
                    cur_swept = None
                    lin = None
                    idx_2d = None
                    view_res = None
                    idx = None
                    try:
                        p_nd = dict(all_p)
                        for j, key in enumerate(sweep_keys):
                            if not varies_mask[j]:
                                p_nd[key] = const_si[key]
                                continue
                            swept_pos = swept_pos_to_param_idx.index(j)
                            o = axis_offsets[swept_pos]
                            ln = cur_shape[swept_pos]
                            vals_disp = sweep_lists[j][o:o + ln]
                            vals_si = vals_disp / float(sweep_scales[j])
                            nd_shape = [1] * (n_swept_dims + 1)
                            nd_shape[swept_pos] = len(vals_si)
                            # rdtype_main is float32 in fp32 sweep mode, so
                            # the swept tensor doesn't get promoted back to
                            # float64 inside the model kernels.
                            p_nd[key] = xp.asarray(
                                vals_si, dtype=rdtype_main).reshape(nd_shape)
                    except Exception as exc:
                        is_oom = (isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() or
                                  "OutOfMemoryError" in type(exc).__name__)
                        # Drop any partially-built tensors so the retry has room
                        p_nd = None
                        if is_oom:
                            new_block, did = _shrink_one_step(block_shape)
                            if did:
                                block_shape = new_block
                                _oom_streak += 1
                                print(f"\n[tune] OOM in param-gen → block "
                                      f"({','.join(str(v) for v in block_shape)}) "
                                      f"[streak={_oom_streak}]",
                                      flush=True)
                                gc.collect()
                                # Only flush the pool on the *second* OOM in a
                                # row.  A single OOM is usually solved by the
                                # shrink alone — flushing every time would
                                # destroy allocator amortization (the same
                                # bug we're fixing in this commit).
                                if use_cuda and _oom_streak >= 2:
                                    try:
                                        _cp.get_default_memory_pool().free_all_blocks()
                                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                                        print("[tune]   pool flushed (last resort)",
                                              flush=True)
                                    except Exception:
                                        pass
                                continue
                        raise

                    # ── Run simulate + residuals + top-K merge on device ────
                    try:
                        # Snapshot pool size BEFORE simulate so we can
                        # measure the actual per-combo allocation footprint
                        # of this iteration and replace the static estimate
                        # with an empirical one (only on the first call,
                        # gated by `_calibrated`).
                        if use_cuda and not _calibrated:
                            try:
                                _bytes_before = _cp.get_default_memory_pool().total_bytes()
                            except Exception:
                                _bytes_before = 0
                        S_batch_nd = model_cls.simulate_batch(
                            p_nd, freq, z0, xp=xp, cache=static_cache)
                        # Shape: inner_shape + (N_freq, 2, 2)
                        S_flat = S_batch_nd.reshape(B_inner, N_freq, 2, 2)
                        res = _port_residuals_batch(S_mea_dev, S_flat, xp)
                        cur_total = res["Total"]                                  # (B_inner,)
                        cur_4 = xp.stack(
                            [res["S11"], res["S12"], res["S21"], res["S22"]],
                            axis=1)                                                # (B_inner, 4)

                        # NaN/inf protection — sentinel value sorts to bottom
                        # without forcing a host-side .all() check (no sync).
                        BIG = 1.0e308
                        cur_total = xp.where(xp.isfinite(cur_total), cur_total, BIG)
                        # Per-port residuals are also the sort key in
                        # "Prioritize" mode, so clamp NaN/inf there too.
                        cur_4 = xp.where(xp.isfinite(cur_4), cur_4, BIG)

                        # ── Balance / quality filters ───────────────────────
                        # Push unbalanced or poor-residual combos to the
                        # bottom of the ranking by clamping their sort key
                        # to BIG.  Balanced survivors are then sorted normally
                        # → "lowest total among balanced".
                        if dev_threshold is not None:
                            _dev = cur_4.max(axis=1) - cur_4.min(axis=1)
                            cur_total = xp.where(_dev > dev_threshold, BIG, cur_total)
                            cur_4 = xp.where(
                                _dev[:, None] > dev_threshold, BIG, cur_4)
                        if res_threshold is not None:
                            _peak = cur_4.max(axis=1)
                            cur_total = xp.where(_peak > res_threshold, BIG, cur_total)
                            cur_4 = xp.where(
                                _peak[:, None] > res_threshold, BIG, cur_4)

                        # ── Build display values for swept params ───────────
                        # Decompose flat index → multi-axis index using
                        # row-major strides over `inner_shape`, then look up
                        # via sweep_table_dev[swept_indices, axis_index].
                        if n_swept_dims > 0:
                            local_strides_h = np.ones(n_swept_dims, dtype=np.int64)
                            for k in range(n_swept_dims - 2, -1, -1):
                                local_strides_h[k] = local_strides_h[k + 1] * inner_shape[k + 1]
                            local_strides_dev = xp.asarray(local_strides_h)
                            local_lens_dev    = xp.asarray(np.asarray(inner_shape, dtype=np.int64))

                            lin = xp.arange(B_inner, dtype=xp.int64)
                            idx_2d = (lin[:, None] // local_strides_dev[None, :]) \
                                      % local_lens_dev[None, :]                    # (B_inner, n_swept)

                            if any(o > 0 for o in axis_offsets):
                                offset_h = np.asarray(axis_offsets, dtype=np.int64)
                                idx_2d = idx_2d + xp.asarray(offset_h)[None, :]

                            cur_swept = sweep_table_dev[swept_indices_dev[None, :], idx_2d]

                        # ── Top-K merge into pre-allocated scratch ──────────
                        total_in = TOP_K + B_inner
                        scratch_res[:TOP_K]              = top_res
                        scratch_res[TOP_K:total_in]      = cur_total
                        scratch_4[:TOP_K]                = top_4
                        scratch_4[TOP_K:total_in]        = cur_4
                        if n_swept_dims > 0:
                            scratch_swept[:TOP_K]         = top_swept
                            scratch_swept[TOP_K:total_in] = cur_swept

                        view_res = scratch_res[:total_in]
                        view_4   = scratch_4[:total_in]
                        # When prioritizing one S-parameter, sort by that
                        # column instead of the total residual. We still keep
                        # `top_res` = total so the displayed "Total Residual"
                        # column stays meaningful.
                        if sort_metric == "Total":
                            sort_view = view_res
                        else:
                            _smap = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
                            sort_view = view_4[:, _smap[sort_metric]]
                        idx = xp.argpartition(sort_view, TOP_K)[:TOP_K]
                        idx = idx[xp.argsort(sort_view[idx])]

                        top_res[:] = view_res[idx]
                        top_4[:]   = view_4[idx]
                        if n_swept_dims > 0:
                            top_swept[:] = scratch_swept[:total_in][idx]

                        # ── Empirical calibration (first iteration only) ─
                        # Measure how many bytes the pool actually grew by
                        # during this iteration, divide by B_inner, and use
                        # that as the ground-truth per-combo footprint.
                        # Apply a 1.3× safety to absorb spike differences
                        # between iterations.  This replaces the (rough)
                        # initial estimate so subsequent shrink/grow
                        # decisions are made on real data.
                        if use_cuda and not _calibrated and B_inner > 0:
                            try:
                                _bytes_after = _cp.get_default_memory_pool().total_bytes()
                                _delta = max(0, _bytes_after - _bytes_before)
                                if _delta > 0:
                                    _measured = _delta // B_inner
                                    _new_pcb = max(1, int(_measured * 1.3))
                                    print(f"\n[tune] calibrated per_combo_bytes "
                                          f"= {_new_pcb:,}  ({_measured:,} "
                                          f"measured × 1.3 safety; was "
                                          f"{per_combo_bytes:,})",
                                          flush=True)
                                    per_combo_bytes = _new_pcb
                                _calibrated = True
                                # Force a resize check on the *next* iter so
                                # the new estimate can immediately grow the
                                # block if there's headroom.
                                _last_resize_t = 0.0
                            except Exception:
                                _calibrated = True   # don't keep retrying
                    except Exception as exc:
                        is_oom = (isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() or
                                  "OutOfMemoryError" in type(exc).__name__)
                        if is_oom:
                            new_block, did = _shrink_one_step(block_shape)
                            if did:
                                old_str = ",".join(str(v) for v in block_shape)
                                new_str = ",".join(str(v) for v in new_block)
                                _oom_streak += 1
                                print(f"\n[tune] OOM at block=({old_str}) → "
                                      f"retry with ({new_str}) "
                                      f"[streak={_oom_streak}]",
                                      flush=True)
                                block_shape = new_block
                                # Drop intermediates before retrying — finally
                                # block will null these out, but we run gc.collect
                                # immediately to release memory before continue.
                                p_nd = None
                                S_batch_nd = None
                                S_flat = None
                                res = None
                                cur_total = None
                                cur_4 = None
                                cur_swept = None
                                lin = None
                                idx_2d = None
                                view_res = None
                                idx = None
                                gc.collect()
                                # Only flush the pool on the *second*
                                # consecutive OOM.  A single OOM is usually
                                # solved by the shrink alone.  Flushing every
                                # OOM destroys allocator amortization.
                                if use_cuda and _oom_streak >= 2:
                                    try:
                                        _cp.get_default_memory_pool().free_all_blocks()
                                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                                        print("[tune]   pool flushed (last resort)",
                                              flush=True)
                                    except Exception:
                                        pass
                                continue   # do NOT advance counters
                            # Already at all-1s floor — bail out cleanly with
                            # whatever top-K we have, instead of spinning
                            # forever or crashing the Streamlit script.
                            print(f"\n[tune] OOM at block=(1,1,...) — single combo "
                                  f"won't fit in VRAM, giving up",
                                  flush=True)
                            st.error(
                                f"GPU ran out of memory even at block=(1,1,...): "
                                f"a single combination's working set "
                                f"(~{per_combo_bytes/1024**2:.1f} MiB for {N_freq} freq pts) "
                                f"won't fit in available VRAM. Reduce the number "
                                f"of frequency points or free GPU memory.")
                            cancelled = True
                            break
                        off_str = ",".join(str(o) for o in axis_offsets)
                        print(f"\n[tune] block @offsets ({off_str}) failed: {exc!r}",
                              flush=True)
                        # Skip this block on non-OOM errors — advance below
                        # via the post-loop counters by treating it as done.
                        # Fall through to normal advance.
                    finally:
                        # Drop per-iteration intermediates so OOM retry has room
                        # *before* re-allocating next iteration.  We don't
                        # touch `top_*` / `scratch_*` (persistent).
                        p_nd = None
                        S_batch_nd = None
                        S_flat = None
                        res = None
                        cur_total = None
                        cur_4 = None
                        cur_swept = None
                        lin = None
                        idx_2d = None
                        view_res = None
                        idx = None

                    # Successful (or skipped) iteration — advance counters.
                    # Reset OOM streak: we got through a full iteration, so
                    # any past OOMs are no longer "consecutive".
                    _oom_streak = 0
                    # Multi-axis carry: advance the *innermost* axis by its
                    # current block step; if it overflows L_k, reset to 0
                    # and carry into the next-outer axis.
                    if n_swept_dims == 0:
                        processed = 1
                    else:
                        processed += B_inner
                        k = n_swept_dims - 1
                        while k >= 0:
                            axis_offsets[k] += block_shape[k]
                            if axis_offsets[k] < inner_lens[k]:
                                break
                            axis_offsets[k] = 0
                            k -= 1
                        # k < 0  →  every axis wrapped, we're done.  The
                        # while-loop guard `processed < n_total` will exit.
                    chunks_done += 1
                    last_chunk_ms = (_time.time() - _t_chunk) * 1000.0

                    # ── Throttled UI tick: ~2 Hz, the only host sync point ──
                    now = _time.time()
                    if now - last_ui > 0.5 or processed >= n_total:
                        elapsed = now - _t_start
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        eta = (n_total - processed) / rate if rate > 0 else 0.0
                        _phase_str = f" {phase_label}" if phase_label else ""
                        progress.progress(
                            min(1.0, processed / max(n_total, 1)),
                            text=(f"Tuning{_phase_str} ({_mode_label})… "
                                  f"{processed:,}/{n_total:,} combos  "
                                  f"({rate:,.0f}/s, ETA {_fmt_eta(eta)})  "
                                  f"slab={B_inner:,}  ({last_chunk_ms:.1f} ms/iter)"))
                        top_arr_host = _sync_topk_host()
                        if top_arr_host is not None:
                            best_series = pd.Series(top_arr_host[0], index=col_names)
                            best_box.markdown(
                                _best_summary_md(best_series, label="Best so far"),
                                unsafe_allow_html=True,
                            )
                            # Persist every tick so a later crash leaves a result
                            st.session_state[sess_key] = _topk_to_df(top_arr_host)
                        elif dev_threshold is not None or res_threshold is not None:
                            # All combos so far failed the deviation/residual
                            # filters — surface this so the user can widen
                            # thresholds instead of staring at a blank panel.
                            best_box.markdown(
                                "*No combos have passed the deviation/residual "
                                "filters yet — consider widening the thresholds "
                                "if this persists.*")
                        last_ui = now

                    _sys.stdout.write(
                        f"\r[tune] {processed:>11,}/{n_total:,}  "
                        f"({100.0*processed/max(n_total,1):5.1f}%)  "
                        f"{(processed / max(_time.time()-_t_start, 1e-9)):>10,.0f} calc/s  "
                        f"slab={B_inner:,}  {last_chunk_ms:6.1f} ms/iter")
                    _sys.stdout.flush()
            except (KeyboardInterrupt, SystemExit):
                cancelled = True
                st.warning("Computation cancelled — keeping the best results found so far.")
                print("\n[tune] cancelled by user (KeyboardInterrupt)", flush=True)
            except MemoryError as me:
                st.error(f"Out of memory: {me}. Keeping the best results found so far.")
                print(f"\n[tune] MemoryError: {me}", flush=True)
            except BaseException as exc:
                # Streamlit raises RerunException OR StopException when the
                # user clicks any widget (incl. our Stop button) — which one
                # depends on Streamlit version and the exact widget path.
                # Treat both identically: persist state, free GPU buffers,
                # then re-raise so Streamlit can finish the rerun cleanly.
                _is_rerun = (_RerunException is not None
                             and isinstance(exc, _RerunException))
                _is_stop  = (_StopException  is not None
                             and isinstance(exc, _StopException))
                # Belt-and-suspenders: also match by class name in case the
                # exception class moved between Streamlit versions and our
                # imports above silently fell through to None.
                _name = type(exc).__name__
                _is_streamlit_stop = _is_rerun or _is_stop or (
                    _name in ("RerunException", "StopException"))
                if _is_streamlit_stop:
                    cancelled = True
                    print(f"\n[tune] cancelled (Streamlit {_name}, "
                          f"e.g. Stop button)", flush=True)
                    _persist_topk()
                    # Free GPU buffers before re-raising so the rerun starts clean
                    try:
                        S_mea_dev = None
                        S_mea_dev_fp64 = None
                        top_res = None; top_4 = None; top_swept = None
                        scratch_res = None; scratch_4 = None; scratch_swept = None
                        sweep_table_dev = None
                        swept_indices_dev = None
                        omega_dev = None
                        # Per-iteration intermediates that may still be alive
                        # if the exception was raised mid-iteration.  The
                        # inner finally usually nulls these, but we
                        # belt-and-suspenders here.
                        try:
                            del p_nd, S_batch_nd, S_flat, res, cur_total, cur_4
                            del cur_swept, lin, idx_2d, view_res, idx
                        except (NameError, UnboundLocalError):
                            pass
                        static_cache.clear()
                        if static_cache_fp64 is not static_cache:
                            static_cache_fp64.clear()
                    except Exception:
                        pass
                    _release_gpu_memory()
                    raise
                # Anything else: log, persist, re-raise
                print(f"\n[tune] unexpected exception: {exc!r}", flush=True)
                _persist_topk()
                raise
            finally:
                # Always persist whatever we have so a crash never wipes results.
                _persist_topk()

            progress.empty()
            stop_box.empty()

            n_kept = (0 if top_res is None
                      else int(xp.sum(xp.isfinite(top_res)
                                       & (top_res < _VALID_RES_MAX)).item()))

            # Keep a warning visible if a deviation/residual sweep rejected
            # everything; otherwise clear the live "best so far" line because
            # the persistent results table will render the final ranking.
            if (n_kept == 0
                    and (dev_threshold is not None or res_threshold is not None)):
                best_box.warning(
                    "No combos passed the deviation/residual filters. "
                    "Widen the thresholds and re-run.")
                # Clear any stale prior-sweep results so the table below
                # doesn't misleadingly show data from a different setting.
                st.session_state.pop(sess_key, None)
                st.session_state.pop(f"tune_elapsed_{topo_key}_{fname}", None)
            else:
                best_box.empty()
            print(f"\n[tune] done   processed={'?' if cancelled else f'{n_total:,}'}  "
                  f"top={n_kept}  in {_time.time()-_t_start:.2f}s", flush=True)

            # ── FP64 rerank of the surviving top-K ──────────────────────────
            # The main loop ran in complex64 (cache_fp32) so the residuals
            # are fp32-accurate.  Re-evaluate every finite top-K combo at
            # complex128 using `static_cache_fp64`, replace the residuals,
            # and re-sort.  TOP_K is small (~100) so this is one batched
            # simulate_batch call — negligible cost vs the main sweep.
            if (use_fp32_sweep and not cancelled
                    and top_res is not None and n_kept > 0):
                try:
                    _top_h = _sync_topk_host()
                    if _top_h is not None and len(_top_h) > 0:
                        K_rk = int(len(_top_h))
                        # Build (K,) SI arrays for swept params, scalars for
                        # constants — same param dict shape `simulate_batch`
                        # already understands.
                        _p_rk = dict(all_p)
                        for j, key in enumerate(sweep_keys):
                            col_disp = _top_h[:, 5 + j]
                            if L_list[j] == 1:
                                _p_rk[key] = (float(col_disp[0])
                                              / float(sweep_scales[j]))
                            else:
                                _p_rk[key] = xp.asarray(
                                    col_disp / float(sweep_scales[j]),
                                    dtype=np.float64)
                        # Force fp64 cdtype on the rerank cache.
                        static_cache_fp64["_cdtype"] = np.complex128
                        S_rk = model_cls.simulate_batch(
                            _p_rk, freq, z0, xp=xp, cache=static_cache_fp64)
                        S_rk_flat = S_rk.reshape(K_rk, N_freq, 2, 2)
                        res_rk = _port_residuals_batch(
                            S_mea_dev_fp64, S_rk_flat, xp)
                        tot_rk = xp.where(
                            xp.isfinite(res_rk["Total"]),
                            res_rk["Total"], 1.0e308)
                        s4_rk  = xp.stack(
                            [res_rk["S11"], res_rk["S12"],
                             res_rk["S21"], res_rk["S22"]], axis=1)

                        # Re-sort by fp64 residuals — match sort_metric chosen
                        # for the main loop so Prioritize keeps its ranking.
                        if sort_metric == "Total":
                            sort_rk = tot_rk
                        else:
                            _smap = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
                            _col  = _smap[sort_metric]
                            sort_rk = xp.where(xp.isfinite(s4_rk[:, _col]),
                                                s4_rk[:, _col], 1.0e308)
                        order_dev = xp.argsort(sort_rk)
                        tot_rk_s  = tot_rk[order_dev]
                        s4_rk_s   = s4_rk[order_dev]

                        order_h = (_cp.asnumpy(order_dev) if use_cuda
                                   else np.asarray(order_dev))
                        _top_h_s = _top_h[order_h]
                        _top_h_s[:, 0]   = (_cp.asnumpy(tot_rk_s) if use_cuda
                                            else np.asarray(tot_rk_s))
                        _top_h_s[:, 1:5] = (_cp.asnumpy(s4_rk_s) if use_cuda
                                            else np.asarray(s4_rk_s))

                        # Push back into the device top-K accumulators so
                        # display and persist see fp64 values.
                        top_res[:K_rk] = xp.asarray(_top_h_s[:, 0])
                        if K_rk < TOP_K:
                            top_res[K_rk:] = xp.inf
                        top_4[:K_rk] = xp.asarray(_top_h_s[:, 1:5])
                        if n_swept > 0:
                            _ts = np.empty((K_rk, n_swept), dtype=np.float64)
                            _jswept = 0
                            for i in range(n_params):
                                if L_list[i] > 1:
                                    _ts[:, _jswept] = _top_h_s[:, 5 + i]
                                    _jswept += 1
                            top_swept[:K_rk] = xp.asarray(_ts)
                        _persist_topk()
                        print(f"[tune] fp64 rerank: top-{K_rk} "
                              f"re-evaluated and re-sorted", flush=True)
                except Exception as _rk_exc:
                    print(f"[tune] fp64 rerank failed: {_rk_exc!r} — "
                          f"keeping fp32 ranking", flush=True)

            # ── Aggressive cleanup: free everything except the persisted
            # top-100 dataframe (already in st.session_state).  Drop refs
            # first so the GC can collect, then return memory pools to the
            # device / OS.  See _release_gpu_memory() for the sync+gc+flush
            # sequence — without it, async kernels in flight would prevent
            # the pool from actually returning blocks to the driver.
            try:
                S_mea_dev = None
                S_mea_dev_fp64 = None
                top_res = None; top_4 = None; top_swept = None
                scratch_res = None; scratch_4 = None; scratch_swept = None
                sweep_table_dev = None
                swept_indices_dev = None
                omega_dev = None
                static_cache.clear()
                if static_cache_fp64 is not static_cache:
                    static_cache_fp64.clear()
            except Exception:
                pass
            _release_gpu_memory()

            # Persist total wall-clock run time so the results panel can show
            # "Evaluated in …" above the best-residual line (survives reruns).
            st.session_state[f"tune_elapsed_{topo_key}_{fname}"] = (
                _time.time() - _t_start)

        # ── Nelder-Mead "Auto" tuning ──────────────────────────────────
        def _run_nelder_mead(*, max_iter: int, restart: bool,
                             use_cuda: bool = False):
            """Local-search optimisation using scipy's adaptive Nelder-Mead.

            x is in *display units* so the simplex moves at comparable
            magnitudes across mixed parameters (fF / pH / Ω).  Bounds come
            from the per-row Min/Max boxes; the seed is the current
            parameter value clipped into bounds.

            When ``use_cuda`` is True and CuPy is available, each objective
            evaluation uses ``simulate_vec(xp=cupy)`` and computes residuals
            on the GPU, with a single host sync per eval to read the scalar
            total back.  Helps when N_freq is large enough that the GPU
            outruns CPU even at batch=1.

            Top-K accumulator and session-state layout match _run_one_sweep
            so the existing display / "Use best values" button work
            unchanged.
            """
            try:
                from scipy.optimize import minimize as _nm_minimize
            except ImportError:
                st.error(
                    "scipy is required for Nelder-Mead Auto tuning. "
                    "Install with `pip install scipy`.")
                return

            import pandas as pd

            # Only enabled rows participate in optimisation; constants stay put.
            swept_rows = [r for r in param_rows if r["enabled"]]
            if not swept_rows:
                st.warning(
                    "Auto needs at least one parameter with the **Sweep** "
                    "checkbox enabled.")
                return

            sweep_keys_nm   = [r["key"]   for r in swept_rows]
            sweep_scales_nm = [r["scale"] for r in swept_rows]

            # Seed + bounds in display units.
            x0_disp = []
            bounds  = []
            for r in swept_rows:
                cur  = float(all_p.get(r["key"], 0.0)) * r["scale"]
                arr  = np.asarray(r["sweep"], dtype=np.float64)
                lo   = float(np.min(arr))
                hi   = float(np.max(arr))
                if hi <= lo:
                    hi = lo + max(abs(lo), 1.0) * 0.5  # avoid degenerate box
                cur  = max(lo, min(hi, cur))
                x0_disp.append(cur)
                bounds.append((lo, hi))
            x0      = np.asarray(x0_disp, dtype=np.float64)
            lo_arr  = np.asarray([b[0] for b in bounds], dtype=np.float64)
            hi_arr  = np.asarray([b[1] for b in bounds], dtype=np.float64)

            TOP_K = 100
            col_names = [
                "Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
            for r in param_rows:
                u   = r["unit"]
                lbl = r["label"]
                col_names.append(f"{lbl} ({u})" if u else lbl)
            sess_key = f"tune_df_{topo_key}_{fname}"

            # Top-K kept on host as a small max-heap-like sorted list.
            top_rows: list[list[float]] = []
            n_eval   = [0]

            def _push_topk(row: list[float]):
                # Keep top-K by total residual.  Linear insert is fine — TOP_K
                # is 100 and Nelder-Mead does <~10k evals total.
                if len(top_rows) < TOP_K:
                    top_rows.append(row)
                    top_rows.sort(key=lambda r: r[0])
                    return
                if row[0] < top_rows[-1][0]:
                    top_rows[-1] = row
                    top_rows.sort(key=lambda r: r[0])

            use_vec = hasattr(model_cls, "simulate_vec")

            # ── CUDA hot path setup ───────────────────────────────────
            # When use_cuda is requested we move S_raw to device once and
            # run residuals there, doing a single .item() per eval to pull
            # the scalar total back to host for scipy.
            xp_dev = _cp if (use_cuda and _HAS_CUDA and use_vec) else np
            if use_cuda and not _HAS_CUDA:
                st.warning("CUDA not available — falling back to CPU.")
            elif use_cuda and not use_vec:
                st.warning(
                    f"{model_cls.__name__} has no simulate_vec — "
                    "CUDA path needs it; falling back to CPU.")
            S_mea_dev = xp_dev.asarray(S_raw)
            _den_dev  = xp_dev.sum(xp_dev.abs(S_mea_dev) ** 2, axis=0)  # (2,2)

            def _residuals_dev(S_sim_dev):
                """Return (total, s11, s12, s21, s22) as Python floats.
                One host sync per call when xp_dev is cupy."""
                diff = S_mea_dev - S_sim_dev                            # (N,2,2)
                num  = xp_dev.sum(xp_dev.abs(diff) ** 2, axis=0)        # (2,2)
                den_s = xp_dev.where(_den_dev > 0, _den_dev, 1.0)
                val   = xp_dev.sqrt(num / den_s) * 100.0
                val   = xp_dev.where(_den_dev > 0, val, 0.0)
                s11 = val[0, 0]; s12 = val[0, 1]
                s21 = val[1, 0]; s22 = val[1, 1]
                tot = (s11 + s12 + s21 + s22) * 0.25
                if xp_dev is np:
                    return (float(tot), float(s11), float(s12),
                            float(s21), float(s22))
                # Single fused host sync — pulls all five scalars in one shot.
                arr = xp_dev.stack([tot, s11, s12, s21, s22])
                arr_h = _cp.asnumpy(arr)
                return (float(arr_h[0]), float(arr_h[1]), float(arr_h[2]),
                        float(arr_h[3]), float(arr_h[4]))

            def _objective(x_disp):
                # Hard-clip into bounds — scipy NM with bounds usually stays
                # inside but the simplex initialisation can poke out by a hair.
                x_disp = np.minimum(np.maximum(x_disp, lo_arr), hi_arr)
                p = dict(all_p)
                for k, sc, v in zip(sweep_keys_nm, sweep_scales_nm, x_disp):
                    p[k] = float(v) / float(sc)
                try:
                    if use_vec:
                        S_sim = model_cls.simulate_vec(p, freq, z0, xp=xp_dev)
                    else:
                        S_sim = model_cls.simulate(p, freq, z0)
                except Exception:
                    return 1.0e8
                if S_sim is None:
                    return 1.0e8
                if xp_dev is np:
                    r = _port_residuals(S_raw, S_sim)
                    r_tot, r_s11, r_s12, r_s21, r_s22 = (
                        r["Total"], r["S11"], r["S12"], r["S21"], r["S22"])
                else:
                    r_tot, r_s11, r_s12, r_s21, r_s22 = _residuals_dev(S_sim)
                if not np.isfinite(r_tot):
                    return 1.0e8
                # Repack to dict shape used downstream (top-K row build)
                r = {"Total": r_tot, "S11": r_s11, "S12": r_s12,
                     "S21": r_s21, "S22": r_s22}
                # Build the (5 + n_params) row in the same column order as the
                # full-sweep dataframe.
                row = [r["Total"], r["S11"], r["S12"], r["S21"], r["S22"]]
                x_dict = dict(zip(sweep_keys_nm, x_disp))
                for pr in param_rows:
                    k = pr["key"]
                    if k in x_dict:
                        row.append(float(x_dict[k]))
                    else:
                        row.append(float(all_p.get(k, 0.0)) * pr["scale"])
                _push_topk(row)
                n_eval[0] += 1
                return float(r["Total"])

            # ── UI placeholders (mirror the sweep UI) ─────────────────
            ui_cols = st.columns([5, 1])
            with ui_cols[0]:
                progress = st.progress(0, text="Auto (Nelder-Mead)…")
            with ui_cols[1]:
                stop_box = st.empty()
            best_box = st.empty()
            stop_key = f"tune_stop_{topo_key}_{fname}_nm"
            stop_box.button(
                "⏹ Stop", key=stop_key, type="secondary",
                help="Stop the calculation. Best results so far are kept.")

            # NM does roughly maxiter * (n+1) evals worst case.
            ev_budget = max(1, max_iter * (len(x0) + 1))

            def _persist_now():
                if not top_rows:
                    return
                df = pd.DataFrame(
                    _multi_metric_top_n(np.asarray(top_rows, dtype=float),
                                         per_metric=10),
                    columns=col_names)
                st.session_state[sess_key] = df

            def _callback(xk, *args, **kwargs):
                # Periodic progress + best-so-far card.  Streamlit's Stop
                # button works by raising RerunException on any st.* call,
                # so calling progress.progress() here also gives us a clean
                # cancellation point.
                if top_rows:
                    best_series = pd.Series(top_rows[0], index=col_names)
                    best_box.markdown(
                        _best_summary_md(best_series, label="Best so far"),
                        unsafe_allow_html=True,
                    )
                if top_rows:
                    _txt = (f"Auto (Nelder-Mead)… {n_eval[0]} evals  "
                            f"(best Total: {top_rows[0][0]:.3f}%)")
                else:
                    _txt = f"Auto (Nelder-Mead)… {n_eval[0]} evals"
                progress.progress(min(1.0, n_eval[0] / ev_budget), text=_txt)
                _persist_now()

            cancelled = False
            options = {
                "maxiter":  int(max_iter),
                "maxfev":   int(max_iter) * (len(x0) + 1),
                "xatol":    1e-6,
                "fatol":    1e-4,
                "adaptive": True,
            }

            try:
                _nm_minimize(
                    _objective, x0,
                    method="Nelder-Mead",
                    bounds=bounds,
                    callback=_callback,
                    options=options,
                )
                if restart and len(top_rows) > 0:
                    # Random perturbation around current best (10% of box width
                    # per axis) for a single second pass — catches premature
                    # simplex collapse without doubling cost.
                    best_disp = np.asarray(top_rows[0][5:5 + len(param_rows)],
                                           dtype=np.float64)
                    # Pull just the swept components out of best_disp
                    swept_idx = [i for i, pr in enumerate(param_rows)
                                 if pr["enabled"]]
                    x_seed = np.asarray(
                        [best_disp[i] for i in swept_idx], dtype=np.float64)
                    rng = np.random.default_rng(0)
                    span = hi_arr - lo_arr
                    x_seed = np.clip(
                        x_seed + rng.uniform(-0.1, 0.1, size=x_seed.shape) * span,
                        lo_arr, hi_arr)
                    _nm_minimize(
                        _objective, x_seed,
                        method="Nelder-Mead",
                        bounds=bounds,
                        callback=_callback,
                        options=options,
                    )
            except BaseException as exc:
                _is_rerun = (_RerunException is not None
                             and isinstance(exc, _RerunException))
                _is_stop  = (_StopException is not None
                             and isinstance(exc, _StopException))
                _name = type(exc).__name__
                if _is_rerun or _is_stop or _name in (
                        "RerunException", "StopException"):
                    cancelled = True
                    _persist_now()
                    # Drop device buffers before re-raising so the next
                    # rerun starts from a clean pool.
                    if xp_dev is not np:
                        try:
                            S_mea_dev = None
                            _cp.cuda.runtime.deviceSynchronize()
                            gc.collect()
                            _cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
                    raise
                st.error(f"Nelder-Mead failed: {exc!r}")
            finally:
                _persist_now()
                if xp_dev is not np:
                    try:
                        S_mea_dev = None
                        gc.collect()
                        _cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

            progress.empty()
            best_box.empty()
            stop_box.empty()
            if not cancelled and top_rows:
                _mode = "CUDA" if xp_dev is not np else "CPU"
                st.success(
                    f"Auto tuning done ({_mode}) — {n_eval[0]} evals, "
                    f"best Total = {top_rows[0][0]:.3f}%")

        # ── Helper: column-name lookup for a tuning_specs row ──────────
        def _col_name(row):
            return f"{row['label']} ({row['unit']})" if row["unit"] else row["label"]

        # ── Dispatch ────────────────────────────────────────────────────
        if cpu_clicked or cuda_clicked:
            _run_one_sweep(use_cuda=bool(cuda_clicked), sort_metric="Total")
        elif opt_cpu_clicked or opt_cuda_clicked:
            # Recursive bisection: each iteration subsamples N_SUB evenly-
            # spaced points per swept parameter inside the current range, runs
            # the sweep, picks the top-2, and narrows the range to between
            # those two values.  Recursion stops once a final refinement at
            # the user's chosen step fits inside MAX_FINAL combos — that
            # final refinement is then run as the closing pass.
            #
            # N_SUB adapts to the dimensionality so per-iter cost stays
            # tractable: 5^13 ≈ 1.2 B combos is unreasonable, but 3^13 ≈
            # 1.6 M is fine and the box still narrows (just by ½ per iter
            # instead of ¼).  We pick the largest N_SUB ∈ {3, 4, 5} whose
            # full grid fits MAX_PER_ITER.
            _use_cuda      = bool(opt_cuda_clicked)
            MAX_FINAL      = 5_000_000
            # Per-iter cap: GPU can absorb ~100M combos in seconds, CPU
            # struggles past ~10M.  Both still drop to N_SUB=3 for very
            # high-dim sweeps — that's expected and still cheap.
            MAX_PER_ITER   = 100_000_000 if _use_cuda else 10_000_000
            # 3-point bisection halves the box per iter; 5-point quarters
            # it.  Going from a width of ~1 to ~1e-4 takes 14 iters at 3
            # points, 7 iters at 5 points — bumped to 30 to leave headroom.
            MAX_ITERS      = 30

            current_ranges = {}   # key -> (lo, hi) — search box this iter
            user_steps     = {}   # key -> float    — step from the row UI
            rows_by_key    = {}
            for row in param_rows:
                if not row["enabled"]:
                    continue
                full = np.asarray(row["sweep"], dtype=np.float64)
                if len(full) == 0:
                    continue
                current_ranges[row["key"]] = (float(full[0]), float(full[-1]))
                kp = f"tune_{topo_key}_{row['key']}_{fname}"
                user_steps[row["key"]] = (
                    float(st.session_state.get(f"{kp}_step", 0.0)) or 0.0)
                rows_by_key[row["key"]] = row

            if not current_ranges:
                st.warning("Optimized mode needs at least one parameter with "
                           "the **Sweep** checkbox enabled.")
            else:
                # Adaptive N_SUB — largest in {3,4,5} that fits MAX_PER_ITER.
                # If even 3^N exceeds the budget we still use 3 and warn.
                _n_swept = len(current_ranges)
                N_SUB = 3
                for _cand in (5, 4, 3):
                    if (_cand ** _n_swept) <= MAX_PER_ITER:
                        N_SUB = _cand
                        break
                _per_iter = N_SUB ** _n_swept
                if N_SUB == 3 and _per_iter > MAX_PER_ITER:
                    st.warning(
                        f"3-point subsample of {_n_swept} swept params is "
                        f"{_per_iter:,} combos — over the {MAX_PER_ITER:,} "
                        "per-iter target. The run will still proceed but "
                        "each iteration will be slow. Consider sweeping "
                        "fewer parameters at once.")
                else:
                    st.caption(
                        f"🎯 N_SUB = **{N_SUB}** points/param "
                        f"({_per_iter:,} combos per iter for {_n_swept} "
                        "swept params)")
                def _estimate_final_combos(ranges, steps):
                    """How many combos would a brute-force sweep of these
                    ranges at the user's steps generate?"""
                    n = 1
                    for k, (lo, hi) in ranges.items():
                        step = steps.get(k, 0.0)
                        if step <= 0 or abs(hi - lo) < 1e-15:
                            pts = 1
                        else:
                            pts = max(1, int(round(abs(hi - lo) / step)) + 1)
                        n *= pts
                    return n

                # Short-circuit: if the user's full grid already fits the
                # final-refine budget, skip the subsample iterations and run
                # the brute force directly. No point doing 5^N subsampling
                # when we could just sweep everything.
                _initial_full = _estimate_final_combos(current_ranges, user_steps)
                _stop_recursion = False
                _final_iter_idx = None
                _skip_recursion = (_initial_full <= MAX_FINAL)
                if _skip_recursion:
                    st.caption(
                        f"🎯 Full sweep is already **{_initial_full:,}** "
                        f"combos ≤ {MAX_FINAL:,} — skipping subsample, "
                        "running brute force at the user step directly.")
                    _final_iter_idx = 0
                    iter_idx = 0
                # The recursion loop only runs when we actually need it.
                # Wrapped in `if not _skip_recursion` rather than `else:` so
                # the existing for-else block keeps working at its original
                # indentation.
                _do_loop = not _skip_recursion
                _loop_converged = False  # set True on any clean break
                for iter_idx in range(1, MAX_ITERS + 1) if _do_loop else range(0):
                    # Build 5-sample lists within the current ranges
                    sub = {}
                    for k, (lo, hi) in current_ranges.items():
                        if abs(hi - lo) < 1e-15:
                            sub[k] = np.array([lo], dtype=np.float64)
                        else:
                            sub[k] = np.linspace(lo, hi, N_SUB)

                    est_final = _estimate_final_combos(current_ranges, user_steps)
                    st.caption(
                        f"🎯 Optimized iter {iter_idx} — current final-refine "
                        f"estimate: **{est_final:,}** combos "
                        f"(target ≤ {MAX_FINAL:,})")

                    _run_one_sweep(
                        use_cuda=_use_cuda, sweep_lists_override=sub,
                        sort_metric="Total",
                        phase_label=f"Iter {iter_idx} (subsample {N_SUB})",
                        phase_suffix=f"_optP{iter_idx}")

                    _df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
                    if _df is None or len(_df) == 0:
                        # No finite residuals — abandon bisection but
                        # still run the closing pass on current_ranges so
                        # the user gets a brute-force result.
                        st.warning(
                            f"Iteration {iter_idx} produced no finite "
                            "residuals — running the final refinement on "
                            "the current range without further narrowing.")
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break

                    # Narrow to the box between the top-2 combos.  With
                    # only one survivor, centre a quarter-width box on it
                    # so the optimised pass keeps making progress.
                    single = len(_df) < 2
                    new_ranges = {}
                    for k, (lo_old, hi_old) in current_ranges.items():
                        col = _col_name(rows_by_key[k])
                        if col not in _df.columns:
                            new_ranges[k] = (lo_old, hi_old)
                            continue
                        v0 = float(_df.iloc[0][col])
                        if single:
                            half_w = abs(hi_old - lo_old) * 0.25
                            new_ranges[k] = (max(lo_old, v0 - half_w),
                                             min(hi_old, v0 + half_w))
                        else:
                            v1 = float(_df.iloc[1][col])
                            new_ranges[k] = (min(v0, v1), max(v0, v1))

                    # Bail if the box can't shrink anymore (guards against
                    # the corner case where the top-2 are identical).
                    if all(abs(new_ranges[k][1] - new_ranges[k][0]) < 1e-15
                           for k in new_ranges):
                        current_ranges = new_ranges
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break

                    current_ranges = new_ranges

                    # Done once the final brute-force fits the budget
                    if _estimate_final_combos(current_ranges, user_steps) \
                            <= MAX_FINAL:
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break
                # for-else replaced by explicit flag so the warning only
                # fires when the loop actually ran *and* didn't converge.
                if _do_loop and not _loop_converged:
                    st.warning(
                        f"Reached MAX_ITERS={MAX_ITERS} without converging "
                        f"under {MAX_FINAL:,} combos — running the final "
                        "refinement on the last narrowed range anyway.")

                # ── Closing pass: brute-force at the user's step ─────────
                if not _stop_recursion:
                    final_lists = {}
                    for k, (lo, hi) in current_ranges.items():
                        step = user_steps.get(k, 0.0)
                        if step <= 0 or abs(hi - lo) < 1e-15:
                            final_lists[k] = np.array([lo], dtype=np.float64)
                        else:
                            final_lists[k] = _make_sweep_values(lo, hi, step)
                    final_total = 1
                    for arr in final_lists.values():
                        final_total *= len(arr)
                    st.caption(
                        f"🎯 Final refine — running **{final_total:,}** "
                        "combos at user-defined step.")
                    _run_one_sweep(
                        use_cuda=_use_cuda, sweep_lists_override=final_lists,
                        sort_metric="Total",
                        phase_label=f"Final refine ({final_total:,} combos)",
                        phase_suffix="_optFinal")
        elif prio_cpu_clicked or prio_cuda_clicked:
            _run_one_sweep(use_cuda=bool(prio_cuda_clicked),
                            sort_metric=str(prio_metric),
                            phase_label=f"Prioritize {prio_metric}",
                            phase_suffix=f"_prio{prio_metric}")
        elif bal_cpu_clicked or bal_cuda_clicked:
            _res_thr = float(bal_res_threshold) if _bal_use_res else None
            _run_one_sweep(
                use_cuda=bool(bal_cuda_clicked),
                sort_metric="Total",
                phase_label=f"Balance ≤{bal_dev_threshold:.2f}%",
                phase_suffix="_bal",
                dev_threshold=float(bal_dev_threshold),
                res_threshold=_res_thr,
            )
        # elif auto_cpu_clicked or auto_cuda_clicked:
        #     _run_nelder_mead(
        #         max_iter=int(auto_max_iter),
        #         restart=bool(auto_restart),
        #         use_cuda=bool(auto_cuda_clicked),
        #     )

        # Display results if available
        df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
        if df is not None:
            best = df.iloc[0]
            _elapsed = st.session_state.get(f"tune_elapsed_{topo_key}_{fname}")
            if _elapsed is not None:
                st.markdown(f"**Evaluated in {_fmt_eval_time(_elapsed)}**")
            st.markdown(
                _best_summary_md(best, label="Best residual"),
                unsafe_allow_html=True,
            )

            # "Use best values" — push best-row params into the fine-tune widgets.
            # Uses on_click callback so writes happen *before* widgets re-render.
            def _apply_best(best_row, specs, topo, fn):
                for spec in specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    col_name = f"{label} ({unit})" if unit else label
                    disp_val = float(best_row[col_name])
                    st.session_state[f"sim_{topo}_{key}_{fn}"] = disp_val
                    kp = f"tune_{topo}_{key}_{fn}"
                    si_val = disp_val / scale
                    if abs(si_val) < 1e-30:
                        st.session_state[f"{kp}_min"] = 0.0
                        st.session_state[f"{kp}_step"] = 0.0
                        st.session_state[f"{kp}_max"] = 0.0
                    else:
                        st.session_state[f"{kp}_min"] = disp_val - 1.0
                        st.session_state[f"{kp}_step"] = 1.0
                        st.session_state[f"{kp}_max"] = disp_val + 1.0

            st.button("🏆 Use best values", key=f"tune_best_{topo_key}_{fname}",
                      on_click=_apply_best,
                      args=(best, tuning_specs, topo_key, fname))

            st.dataframe(df, width="stretch", hide_index=True)

            # Excel download
            buf = BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            st.download_button(
                "📥 Download as Excel",
                data=buf,
                file_name=f"tuning_{topo_key}_{fname}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"tune_dl_{topo_key}_{fname}",
            )

            # ── Per-parameter sensitivity graphs ─────────────────────���───
            # For each ticked parameter, sweep it while holding others at
            # the best-row values.  Plot S11/S12/S21/S22 residuals.
            ticked_specs = [
                spec for spec in tuning_specs
                if st.session_state.get(
                    f"tune_{topo_key}_{spec[0]}_{fname}_chk", False)
            ]
            if ticked_specs:
                st.markdown("---")
                st.markdown("**Parameter sensitivity** *(other params held at best values)*")

                # Build baseline param dict from best row (in SI)
                best_p = dict(all_p)
                for spec in tuning_specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    col_name = f"{label} ({unit})" if unit else label
                    best_p[key] = float(best[col_name]) / scale

                _sp_colors = {"S11": "#1f77b4", "S12": "#d62728",
                              "S21": "#2ca02c", "S22": "#ff7f0e"}

                # Lay sensitivity charts out in a 2-column grid.  Even
                # index → left column, odd index → right column.  An
                # odd total leaves the final chart alone in the left
                # column (right column stays empty).
                _sens_col_pair = None  # current (left_col, right_col)
                _sens_rendered = 0

                for spec in ticked_specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    kp = f"tune_{topo_key}_{key}_{fname}"
                    sweep_min = float(st.session_state.get(f"{kp}_min", 0))
                    sweep_step = float(st.session_state.get(f"{kp}_step", 0))
                    sweep_max = float(st.session_state.get(f"{kp}_max", 0))
                    sweep_vals = _make_sweep_values(sweep_min, sweep_max, sweep_step)

                    # Cap the sensitivity chart at 50 points — when the
                    # user-defined sweep is denser, drop to 50 evenly-spaced
                    # samples (always keeping the endpoints).
                    SENS_MAX_PTS = 50
                    if len(sweep_vals) > SENS_MAX_PTS:
                        idxs = np.linspace(0, len(sweep_vals) - 1,
                                            SENS_MAX_PTS).round().astype(int)
                        idxs = np.unique(idxs)
                        sweep_vals = sweep_vals[idxs]

                    if len(sweep_vals) < 2:
                        continue  # nothing to plot for a single point

                    # Compute residuals for each sweep value
                    res_s11 = np.zeros(len(sweep_vals))
                    res_s12 = np.zeros(len(sweep_vals))
                    res_s21 = np.zeros(len(sweep_vals))
                    res_s22 = np.zeros(len(sweep_vals))
                    _sens_has_vec = hasattr(model_cls, "simulate_vec")
                    for vi, sv in enumerate(sweep_vals):
                        p = dict(best_p)
                        p[key] = sv / scale  # convert display -> SI
                        try:
                            if _sens_has_vec:
                                S_sim = model_cls.simulate_vec(p, freq, z0, xp=np)
                            else:
                                S_sim = model_cls.simulate(p, freq, z0)
                        except Exception:
                            S_sim = None
                        if S_sim is None:
                            res_s11[vi] = res_s12[vi] = float("inf")
                            res_s21[vi] = res_s22[vi] = float("inf")
                        else:
                            r = _port_residuals(S_raw, S_sim)
                            res_s11[vi] = r["S11"]
                            res_s12[vi] = r["S12"]
                            res_s21[vi] = r["S21"]
                            res_s22[vi] = r["S22"]

                    x_label = f"{label} ({unit})" if unit else label
                    fig = go.Figure()
                    for sp_name, y_data in [("S11", res_s11), ("S12", res_s12),
                                            ("S21", res_s21), ("S22", res_s22)]:
                        fig.add_trace(go.Scattergl(
                            x=sweep_vals, y=y_data, mode="lines+markers",
                            name=sp_name,
                            line=dict(color=_sp_colors[sp_name], width=2),
                            marker=dict(size=4),
                        ))
                    fig.update_layout(
                        title=f"Sensitivity -- {x_label}",
                        xaxis_title=x_label,
                        yaxis_title="Residual (%)",
                        height=350,
                        margin=dict(l=50, r=30, t=40, b=50),
                        legend=dict(orientation="h", y=1.12),
                        hovermode="x unified",
                    )
                    # Open a fresh 2-column pair every even-indexed chart.
                    if _sens_rendered % 2 == 0:
                        _sens_col_pair = st.columns(2)
                    _target = _sens_col_pair[_sens_rendered % 2]
                    with _target:
                        plotly_with_dl(
                            fig,
                            key=f"tune_sens_{topo_key}_{key}_{fname}",
                            filename=f"tune_sens_{topo_key}_{key}_{fname}")
                    _sens_rendered += 1


# ════════════════════════════════════════════════════════════════════════════════
# SSMModelTemplate — shared scaffold for concrete model classes
# ════════════════════════════════════════════════════════════════════════════════
#
# This is a *mixin-style* parent: it provides default implementations of
# simulate_vec, simulate_batch, render_override_and_smith, and
# render_results_table that work for any model whose forward simulation
# follows the standard pad→lead→intrinsic chain.
#
# Concrete model classes inherit from BOTH this template AND AbstractSSMModel
# (the ABC contract in models/__init__.py).  Python's MRO combines them:
#
#     class MyModel(SSMModelTemplate, AbstractSSMModel):
#         NAME, SHORT, TOPOLOGY_CHAR = "...", "...", "..."
#         _INT_SPECS = [...]   # internal param specs (key, label, scale, unit, ...)
#         _EXT_SPECS = [...]   # external param specs
#         _Y_INT_VEC_FN     = staticmethod(_Y_int_xxx_vec)
#         _Y_INT_BATCH_FN   = staticmethod(_Y_int_xxx_batch)
#         _SIM_WRAP_VEC_FN  = staticmethod(_sim_wrap_vec)   # model's wrap
#         _SIM_WRAP_BATCH_FN= staticmethod(_sim_wrap_batch) # model's wrap
#
#         @classmethod
#         def _do_override_ui(cls, fname, calc_vals): ...   # call model's _override_ui
#         @classmethod
#         def _render_topology(cls, all_p, fname): ...      # render topology illus
#         @classmethod
#         def _results_rows(cls, params): ...               # list[(sym, val, unit)]
#         @classmethod
#         def _render_results_trace(cls): pass              # optional formula trace
#
#         # Still required by AbstractSSMModel (model-specific math):
#         extract, simulate, reextract
#
# The template intentionally does NOT inherit AbstractSSMModel so it can live
# in base_ui.py without a circular import — concrete classes inherit both.

class SSMModelTemplate:
    """
    Shared scaffold for HBT small-signal model classes.

    Provides default bodies for the UI-side and forward-sim wrapper methods;
    concrete subclasses supply the math kernels and per-model UI bits via
    class attributes and a handful of classmethod hooks.  See the module
    docstring above for the contract.
    """

    # Subclasses must set these:
    _INT_SPECS: list      = []
    _EXT_SPECS: list      = []
    _Y_INT_VEC_FN         = None    # staticmethod or plain function
    _Y_INT_BATCH_FN       = None
    _SIM_WRAP_VEC_FN      = None
    _SIM_WRAP_BATCH_FN    = None
    # Pad-spec list used by the tuning expander.  Default is the shared
    # PAD_SPECS; override to customise displayed pad-parameter labels
    # (e.g. XuModel relabels Cpce → "Cpce / Cpad").
    _TUNING_PAD_SPECS     = None    # falls back to PAD_SPECS in render_override_and_smith

    # Default: no pre-bakeable sub-networks.  Concrete subclasses override
    # ``STATIC_SUBNETWORKS`` with a dict of {subnet_name: frozenset(deps)}.
    STATIC_SUBNETWORKS: dict = {}

    # ── Pre-bake truth table ──────────────────────────────────────────────────

    @classmethod
    def prebake_static_keys(cls, swept_keys):
        """Given the iterable of *swept* (i.e. changing) param keys, return
        the list of pre-bakeable sub-network names for this model.  A sub-
        network is pre-bakeable iff none of the keys it depends on are
        swept.  Caller can use this to decide which entries to populate
        in the ``cache`` argument to ``simulate_batch``."""
        if not cls.STATIC_SUBNETWORKS:
            return []
        swept_set = set(swept_keys)
        return [name for name, deps in cls.STATIC_SUBNETWORKS.items()
                if not (deps & swept_set)]

    @classmethod
    def build_static_cache(cls, all_p, freq, *, xp=None, swept_keys=()):
        """Build the ``cache`` dict (matching what ``_sim_wrap_batch`` reads)
        for the given fixed-param baseline and the set of params being
        swept.  Concrete subclasses override to fill in model-specific
        sub-networks (Zbe / Zbc / alpha / Y_extr / etc.).

        Default implementation only handles the shared sub-networks
        (``Y_pad``, ``Z_ser``) — enough to give the live mode a meaningful
        speed-up even on models without intrinsic pre-bake support.
        """
        from ..helpers.deembed_math import build_Y_pad_batch, build_Z_ser_batch
        if xp is None:
            xp = np
        omega = xp.asarray(2 * np.pi * np.asarray(freq), dtype=float)
        N = int(omega.shape[0])
        cache: dict = {"omega": omega, "_cdtype": np.complex128}
        static_p = dict(all_p)
        pbk = set(cls.prebake_static_keys(swept_keys))
        if "Y_pad" in pbk and not (cls.STATIC_SUBNETWORKS.get("Y_pad", set()) & set(swept_keys)):
            try:
                cache["Y_pad"] = build_Y_pad_batch(static_p, omega, 1, N, xp)
            except Exception:
                pass
        if "Z_ser" in pbk and not (cls.STATIC_SUBNETWORKS.get("Z_ser", set()) & set(swept_keys)):
            try:
                cache["Z_ser"] = build_Z_ser_batch(static_p, omega, 1, N, xp)
            except Exception:
                pass
        # Concrete subclasses extend via _build_intrinsic_static_cache().
        try:
            cls._build_intrinsic_static_cache(static_p, omega, cache, xp, pbk)
        except (AttributeError, NotImplementedError):
            pass
        return cache

    @classmethod
    def _build_intrinsic_static_cache(cls, p, omega, cache, xp, prebakeable):
        """Override in concrete classes to populate intrinsic sub-networks
        (Y_extr / Zbe / Zbc / alpha / Ybe / Ybc / gm / T_int_planes / etc.).
        Default is no-op."""
        return

    # ── Forward simulation ────────────────────────────────────────────────────

    @classmethod
    def simulate_vec(cls, params, freq, z0=50.0, xp=None):
        """Vectorised simulate — no per-freq loop.  Pass xp=cupy for GPU."""
        if xp is None:
            xp = np
        return cls._SIM_WRAP_VEC_FN(cls._Y_INT_VEC_FN, params, freq, z0, xp)

    @classmethod
    def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
        """Batched simulate over (param_combo × freq).  Pass xp=cupy for GPU.

        params dict values may be scalars or (B,) arrays.
        Returns (B, N_freq, 2, 2) on the *xp* device (no host transfer).
        Optional ``cache`` carries pre-computed constant sub-networks.

        Phase 2 opt-in
        --------------
        When the env var ``HBT_USE_RUST_SIM_BATCH=1`` is set AND the
        Rust crate is loaded AND a Rust kernel exists for this model's
        ``SHORT`` identifier, the CPU path (xp is numpy) routes through
        the Rust end-to-end batched simulator instead of the NumPy
        composition chain.  The CUDA path (xp is cupy) stays on cupy.

        Cache + Rust
        ------------
        When a non-empty ``cache`` is provided, Rust still runs (the
        whole composition is so much faster than NumPy that the cache
        savings can't beat it).  The cache is preserved for the NumPy
        FALLBACK path so a Rust failure (e.g. exotic pad mode raises
        ``NotImplementedError``) still gets the pre-bake speedup.

        The default behaviour (env var unset) is identical to before:
        CPU goes through ``cls._SIM_WRAP_BATCH_FN`` exactly as it does
        today, with no measurable overhead from the opt-in check.
        """
        if xp is None:
            xp = np

        # Phase 2 dispatch — opt-in, CPU-only.  The lazy import keeps
        # this branch zero-cost when Phase 2 isn't activated.
        if xp is np:
            try:
                from ..helpers.rust_kernels import (
                    HAS_RUST as _HAS_RUST,
                    SIM_FOR_TOPOLOGY as _SIM_FOR_TOPOLOGY,
                    _phase2_dispatch_enabled as _phase2_on,
                )
            except Exception:                              # pragma: no cover
                _HAS_RUST = False
                _SIM_FOR_TOPOLOGY = {}

                def _phase2_on() -> bool:
                    return False

            if _HAS_RUST and _phase2_on():
                rust_wrapper = _SIM_FOR_TOPOLOGY.get(cls.SHORT)
                if rust_wrapper is not None:
                    # NumPy fallback closure — preserves the ORIGINAL
                    # cache so the pre-bake speedup isn't lost if Rust
                    # raises (NotImplementedError for exotic pad modes
                    # is the common case).
                    _saved_cache = cache
                    def _np_fallback(_p, _f, _z0):
                        return cls._SIM_WRAP_BATCH_FN(
                            cls._Y_INT_BATCH_FN, _p, _f, _z0, np,
                            _saved_cache)
                    return rust_wrapper(params, freq, z0,
                                         np_fallback=_np_fallback)

        return cls._SIM_WRAP_BATCH_FN(cls._Y_INT_BATCH_FN, params, freq, z0, xp, cache)

    # ── Cached simulate-vec (used by render_override_and_smith) ───────────────

    @classmethod
    def _cached_simulate_vec(cls, all_p, freq, z0, fname):
        """Hash-cached wrapper around ``simulate_vec``.

        Skips re-simulation when the parameter dict + freq length match the
        last invocation.  Errors are caught and reported in-place; the cache
        is populated with a NaN array so the rest of the UI keeps rendering.
        """
        cache_key = f"sim_result_{cls.SHORT}_{fname}"
        hash_key  = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash  = params_hash({k: str(v) for k, v in
                                  {**all_p, "__nf": len(freq)}.items()})

        def _run():
            with st.spinner(f"Simulating {cls.NAME}…"):
                try:
                    return cls.simulate_vec(all_p, freq, z0)
                except Exception as e:
                    st.error(f"Simulation error ({cls.NAME}): {e}")
                    return np.full((len(freq), 2, 2), np.nan + 0j)

        if st.session_state.get(hash_key) != cur_hash:
            S_sim = _run()
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                S_sim = _run()
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash
        return S_sim

    # ── UI scaffolding ────────────────────────────────────────────────────────

    @classmethod
    def render_results_table(cls, params):
        """Render JUST the scalar-parameter dataframe.

        The formula-trace expander is rendered separately via
        :meth:`render_formula_trace` so the call site can place both
        side-by-side in their own columns / expanders.
        """
        import pandas as pd
        rows = cls._results_rows(params)
        st.dataframe(pd.DataFrame(rows, columns=["Symbol", "Value", "Unit"]),
                     width="stretch", hide_index=True)

    @classmethod
    def render_formula_trace(cls):
        """Render the 📐 Full formula trace expander, if the subclass
        provides one.  Default dispatches to the private template hook
        ``_render_results_trace`` — subclasses (Cheng T / π, Xu T)
        override that hook to supply their own LaTeX dependency chain.
        Models without a trace (e.g. Degachi) silently render nothing.
        """
        cls._render_results_trace()

    @classmethod
    def _render_results_trace(cls):
        """Optional: render an expander with the full extraction/sim formula chain.
        Default is a no-op; override to add a 📐 trace expander."""
        return

    @classmethod
    def has_formula_trace(cls) -> bool:
        """True iff the subclass overrides ``_render_results_trace``.

        Used by callers that want to drop the side-by-side layout when
        no formula trace is available (so the parameter table can use
        full width instead of leaving an empty right column).
        """
        return (cls._render_results_trace.__func__
                is not SSMModelTemplate._render_results_trace.__func__)

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        """
        Standard override-UI → cached sim → Smith chart → topology illustration
        → matplotlib Smith expander → tuning expander.

        Concrete subclasses supply the per-model UI pieces via:
          cls._do_override_ui(fname, calc_vals)  → all_p
          cls._render_topology(all_p, fname)     → render illustration
          cls._INT_SPECS / _EXT_SPECS            → tuning specs

        Also performs persistent fit-cache restore (on first render after
        upload) and auto-save (when the user has fine-tuned vs. extraction
        defaults).  See helpers/fit_cache.py.
        """
        from ..helpers.fit_cache import (get_fit, get_fit_timestamp,
                                          save_fit, delete_fit, differs_from)

        params, _arrays = extract_result
        calc_vals = {**para_eff, **params}

        pad_specs   = cls._TUNING_PAD_SPECS if cls._TUNING_PAD_SPECS is not None else PAD_SPECS
        all_specs   = pad_specs + cls._EXT_SPECS + cls._INT_SPECS
        int_ext_keys = [k for k, *_ in cls._EXT_SPECS + cls._INT_SPECS]
        scale_for    = {k: sc for k, _, sc, *_ in all_specs}

        # ── Cache restore — runs on the first call after Run SSM is clicked
        #    (applied_key resides in session_state, which IOED's "Clear SSM
        #    results" button wipes for the file; so a fresh Run SSM cycle
        #    re-applies the cache).
        cached        = get_fit(fname, cls.SHORT)
        cached_ts     = get_fit_timestamp(fname, cls.SHORT)
        applied_key   = f"cache_applied_{cls.SHORT}_{fname}"
        dismissed_key = f"cache_dismissed_{cls.SHORT}_{fname}"

        # Helper closure — write cached values (including pad) into the
        # fine-tune session_state and align the sync hashes so subsequent
        # renders don't overwrite us.
        #
        # Pad strategy: prefer cached pad if the cache has it (preserves
        # the user's fine-tuned pad, e.g. Rpb/Rpc/Rpe from Cold-HBT that
        # don't get re-extracted from Step 1b alone).  Fall back to
        # `para_eff` only when the cache has no value for that key
        # (older cache files written with the no-pad filter, or never
        # touched).  The matching `main_ssm_extraction` change ensures
        # `calc_vals[pad] = para_eff[pad]` (live Step 1), so the pad
        # sync hash below stays aligned with para_eff and
        # `sync_pad_from_preov` doesn't clobber the cached pad on the
        # next render.
        def _apply_cached_to_simstate(cached_dict):
            if not isinstance(cached_dict, dict):
                return
            # Pad: cached value first, else live para_eff.
            for key, _, sc, *_ in pad_specs:
                cv = cached_dict.get(key)
                if not isinstance(cv, (int, float)):
                    cv = para_eff.get(key, 0.0)
                st.session_state[f"sim_{cls.SHORT}_{key}_{fname}"] = float(cv) * sc
            # Int/ext: from cache.
            for k, v in cached_dict.items():
                if (k in scale_for and k not in _PAD_KEYS
                        and isinstance(v, (int, float))):
                    st.session_state[f"sim_{cls.SHORT}_{k}_{fname}"] = (
                        float(v) * scale_for[k])
            # Intrinsic sync hash: hash calc_vals[int_ext].  Since
            # main_ssm filters pad out of `params`, calc_vals[int_ext]
            # equals cached[int_ext] — so _override_ui's intrinsic sync
            # is a no-op.
            st.session_state[f"sim_synchash_{cls.SHORT}_{fname}"] = params_hash(
                {k: str(round(float(calc_vals.get(k, 0.0)), 15))
                 for k in int_ext_keys})
            # Pad sync hash matches live para_eff (NOT cached pad), so
            # sync_pad_from_preov stays a no-op until Step 1 actually
            # changes.  This is what protects the just-loaded cached pad
            # values from being overwritten by para_eff on the next
            # render.
            st.session_state[f"smith_pad_synced_{cls.SHORT}_{fname}"] = params_hash(
                {k: para_eff.get(k, 0.0) for k in _PAD_KEYS})

        # ── Cache restore — runs on the first call after Run SSM is clicked
        #    (applied_key resides in session_state, which IOED's "Clear SSM
        #    results" button wipes for the file; so a fresh Run SSM cycle
        #    re-applies the cache).
        #
        #    `applied_key` is set on this first render *whether or not* a cache
        #    existed.  Critically, when NO cache exists yet, the user's first
        #    fine-tune edit triggers the auto-save below, which creates a cache
        #    file.  If `applied_key` were still unset on the next render, this
        #    branch would see the freshly-written cache and re-apply it —
        #    clobbering the user's in-progress edit with the previous value
        #    (the "have to type every value twice" bug).  Setting the flag now
        #    guarantees the override widgets own session_state from here on.
        if (not st.session_state.get(applied_key)
                and not st.session_state.get(dismissed_key)):
            if cached:
                _apply_cached_to_simstate(cached)
            st.session_state[applied_key] = True

        # Banner whenever a cache entry exists for this (file, model)
        if cached_ts and not st.session_state.get(dismissed_key):
            bc1, bc2 = st.columns([5, 1])
            bc1.info(f"📌 Loaded cached fit for **{cls.NAME}** "
                     f"(saved {cached_ts}).")
            if bc2.button("↩️ Use saved",
                          key=f"cache_reset_{cls.SHORT}_{fname}",
                          help="Re-apply the cached intrinsic/extrinsic values "
                               "into the fine-tune fields below — handy after "
                               "experimenting if you want to revert to the last "
                               "saved snapshot.  Pad fields are always sourced "
                               "from Step 1 / pre-extraction override."):
                _apply_cached_to_simstate(cached)
                # Mark applied so the auto-restore branch above no-ops, then
                # rerun so the fine-tune widgets read the freshly-set values.
                st.session_state[applied_key] = True
                st.rerun()

        all_p = cls._do_override_ui(fname, calc_vals)

        S_sim = cls._cached_simulate_vec(all_p, freq, z0, fname)

        # Build the modeled S2P bytes once here so render_smith_with_ftfmax
        # can wire a "📥 modeled S2P" button next to the xlsx download under
        # the Smith chart.  This replaces the standalone "Download Modeled
        # DUT S2P" section that used to live at the bottom of the SSM tab.
        from ..helpers import write_s2p
        from pathlib import Path as _Path
        s2p_params = {}
        for _k in ("Cpbe", "Cpce", "Cpbc"):
            s2p_params[_k] = f"{para_eff.get(_k, 0.0) * 1e15:.4f} fF"
        for _k_raw, _label in (("Rpb", "Rb"), ("Rpc", "Rc"), ("Rpe", "Re")):
            s2p_params[_label] = f"{para_eff.get(_k_raw, 0.0):.4f} Ω"
        for _k in ("Lb", "Lc", "Le"):
            s2p_params[_k] = f"{para_eff.get(_k, 0.0) * 1e12:.4f} pH"
        s2p_bytes = write_s2p(
            freq, S_sim,
            title=f"DUT {cls.NAME} — {_Path(fname).stem}",
            params=s2p_params,
        )
        s2p_filename = f"model_dut_{_Path(fname).stem}_{cls.SHORT}.s2p"

        sc = smith_scale_controls(fname, cls.SHORT)
        render_smith_with_ftfmax(S_raw, S_sim, freq,
                                 model_name=cls.NAME, model_short=cls.SHORT,
                                 fname=fname, scales=sc,
                                 s2p_bytes=s2p_bytes,
                                 s2p_filename=s2p_filename)

        # τ_total + calculated fmax expander (HBT T/π models only — Kun-Yang
        # HEMT has no Cbcx/Cbc/Rbi/Rb to evaluate the fmax formula).
        if cls.SHORT in ("T", "pi", "XuT"):
            from ..ssm_plots import render_tau_fmax_expander
            _CBC = float(all_p.get("Cbcx", 0.0)) + float(all_p.get("Cbc", 0.0))
            _Rbb = float(all_p.get("Rbi", 0.0)) + float(all_p.get("Rpb", 0.0))
            if cls.SHORT == "pi":
                _tau_sum = float(all_p.get("tau", 0.0))
                _tau_lbl, _tau_tex = "τ", r"\tau"
            else:
                _tau_sum = float(all_p.get("tauB", 0.0)) + float(all_p.get("tauC", 0.0))
                _tau_lbl, _tau_tex = "τB + τC", r"\tau_B+\tau_C"
            render_tau_fmax_expander(key=f"taufmax_{cls.SHORT}_{fname}",
                                     freq=freq, S_meas=S_raw, S_model=S_sim,
                                     CBC=_CBC, Rbb=_Rbb,
                                     tau_sum=_tau_sum, tau_sum_label=_tau_lbl,
                                     tau_sum_tex=_tau_tex,
                                     extrap_key=f"ftfmax_card_{cls.SHORT}_{fname}")

        # Persist the *current* (post-override) param dict so the Complete
        # Parameter Summary can read live values instead of extraction-time ones.
        st.session_state[f"current_p_{cls.SHORT}_{fname}"] = dict(all_p)

        # ── Auto-save — fires whenever the fine-tune section's `all_p`
        # differs from the current "baseline" snapshot.  The baseline is
        # `calc_vals` (extraction defaults) overlaid with any cached
        # values — so after a cache restore, `all_p == baseline` and no
        # redundant save fires; only genuine user edits beyond the
        # cached state trigger a write.  Pad is included so users can
        # fine-tune Rpb/Rpc/Rpe (etc.) and have those values persist.
        baseline = dict(calc_vals)
        if isinstance(cached, dict):
            for _k, _v in cached.items():
                if _k in scale_for and isinstance(_v, (int, float)):
                    baseline[_k] = float(_v)
        check_keys = _PAD_KEYS + int_ext_keys
        if (not st.session_state.get(dismissed_key)
                and differs_from(all_p, baseline, keys=check_keys)):
            save_fit(fname, cls.SHORT, dict(all_p))

        # Topology illustration in its own expander (collapsed by default).
        # For no-parasitics models we also pass the user's *customized* Smith
        # chart (rebuilt from session_state via phase="chart", return_png=True —
        # creates no widgets) so the topology view can overlay it bottom-right.
        from ..ssm_plots import render_matplotlib_smith
        _smith_png = None
        if S_sim is not None:
            try:
                _smith_png = render_matplotlib_smith(
                    S_raw, S_sim, fname, cls.SHORT,
                    default_multiplier=sc,
                    phase="chart", freq_hz=freq, return_png=True)
            except Exception:                            # noqa: BLE001
                _smith_png = None
        with st.expander("🖼️ Topology illustration", expanded=False):
            cls._render_topology(all_p, fname, smith_png=_smith_png)

        # Smith chart (matplotlib) + its controls live in a single
        # expander, rendered side-by-side — matches the RF simulator
        # layout (RF_simulator.py "🍩 Smith Chart (Matplotlib)").  The
        # split-call pattern (controls in the right column, chart in the
        # left) preserves the order-of-operations requirement that
        # widgets render BEFORE the chart so session_state is fresh when
        # the chart half reads it.
        with st.expander("🍩 Smith Chart (Matplotlib)", expanded=False):
            col_mpl_left, col_mpl_right = st.columns([1.2, 1])
            with col_mpl_right:
                render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT,
                                         default_multiplier=sc,
                                         phase="controls", freq_hz=freq)
            with col_mpl_left:
                render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT,
                                         default_multiplier=sc,
                                         phase="chart", freq_hz=freq)

        render_visual_tuning_expander(cls, all_p, S_raw, freq, z0,
                                       all_specs,
                                       fname, cls.SHORT)
        render_tuning_expander(cls, all_p, S_raw, freq, z0,
                               all_specs,
                               fname, cls.SHORT)
        return S_sim

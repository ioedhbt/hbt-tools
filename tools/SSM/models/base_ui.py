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

from ..ssm_core import extended_smith_grid, params_hash, s_to_y
from ..ssm_deembedding import build_Y_pad_batch, build_Z_ser_batch

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

def render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith"):
    if scales is None:
        scales = {"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0}
    fig = go.Figure()
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr)
    for name, (r, c) in [("S11",(0,0)),("S22",(1,1)),("S21",(1,0)),("S12",(0,1))]:
        col = _SMITH_COLORS[name]; sc = scales.get(name, 1.0)
        sm = S_mea[:,r,c]*sc; sk = S_sim[:,r,c]*sc
        sc_lbl = "" if abs(sc-1.0)<1e-9 else (f" ×{sc:.2g}" if sc>=1 else f" ÷{1/sc:.2g}")
        fig.add_trace(go.Scatter(x=sm.real, y=sm.imag, mode="markers",
                                  name=f"{name}{sc_lbl} Meas.",
                                  marker=dict(color=col, size=5, symbol="circle"),
                                  hovertemplate=f"{name} Meas.<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sk.real, y=sk.imag, mode="lines",
                                  name=f"{name}{sc_lbl} Model",
                                  line=dict(color=col, width=2.0, dash="dash"),
                                  hovertemplate=f"{name} Model<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
    port_res = _port_residuals(S_mea, S_sim)
    # title_line1 = f"Measured vs Modeled — {model_name}   (Total Residual: {error_pct:.2f}%)"
    # title_line2 = (f"S11: {port_res['S11']:.2f}%   S12: {port_res['S12']:.2f}%   "
    #                f"S21: {port_res['S21']:.2f}%   S22: {port_res['S22']:.2f}%")
    # st.markdown(title_line1)
    # st.markdown(title_line2)
    fig.update_layout(
        # title=f"{title_line1}<br><sup>{title_line2}</sup>",
        title=f"Smith Chart - {model_name}",
        xaxis=dict(title="Re(Γ)", range=[-1.1,1.1], scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1,1.1], showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        margin=dict(l=50,r=30,t=70,b=50),
        legend=dict(x=1.02, y=1.0, xanchor="left"),
        hovermode="closest",
        annotations=[dict(x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                          text="● Measured (markers)  |  - - Modeled (dashed)",
                          font=dict(size=10, color="gray"), align="center")])
    st.plotly_chart(fig, width="stretch", key=key)


def render_smith_with_ftfmax(S_raw, S_sim, freq, model_name: str,
                             model_short: str, fname: str, scales=None):
    """
    Two-column layout: Smith chart (left) + fT/fmax mini-card (right).

    The Smith chart still owns the residual line in its title; the mini-card
    on the right shows |h21|² and Mason U for both measured and modeled with
    20 dB/dec extrapolation when needed.  Use this in place of the bare
    `render_smith_chart()` call inside each model's `render_override_and_smith`.
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
    # st.markdown(title_line2)
    col_l, col_r = st.columns([1.05, 1])
    with col_l:
        render_smith_chart(S_raw, S_sim, model_name, err, scales,
                           key=f"smith_{model_short}_{fname}")
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
        sc[name] = col_w.number_input(f"{name} ×", min_value=0.01, max_value=1000.0,
                                       value=float(st.session_state[sk]),
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
    st.caption(
        "Sweeps candidate Cbex values in the range below (inclusive), "
        "recomputes Cbcx_arr for each, and picks the Cbex that yields the "
        "lowest standard deviation of Cbcx over the Cbcx group's frequency "
        "window.  Click Calculate to run the sweep and push the best value "
        "into the Cbex input below.")

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
            # otherwise the full freq range.  The Cbcx group is the NEXT group
            # after the Cbex group (g_idx + 1), so its slider key mirrors ours.
            cbcx_sl_key = f"pfp_sl_{model_short}_{g_idx + 1}_{fname}"
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


def render_interactive_param_groups(params, arrays, freq, fname, model_short, param_groups,
                                    cold_res=None, cold_param_map=None, reextract_fn=None,
                                    cbex_sweep_fn=None):

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

    with st.expander("📊 Extracted Parameters vs Frequency — Interactive", expanded=False):
        for g_idx, group in enumerate(param_groups):
            g_label  = group["label"]
            g_params = group["params"]
            g_deps   = group.get("depends_on", [])

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
                        fig.add_trace(go.Scatter(
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
                        col_w.plotly_chart(fig, width="stretch",
                                           key=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}")

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
                    fig.add_trace(go.Scatter(
                        x=omega2[valid_mask], y=Fbi_arr[valid_mask], mode="markers",
                        name="All data", marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scatter(
                        x=omega2[win_mask], y=Fbi_arr[win_mask], mode="markers",
                        name="Fit window", marker=dict(size=6, color="#1f77b4")))
                    if A0 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scatter(
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
                    st.plotly_chart(fig, width="stretch",
                                    key=f"pfp_fbi_{model_short}_{fname}")

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
                    fig.add_trace(go.Scatter(
                        x=omega2[valid_mask], y=F1_arr[valid_mask], mode="markers",
                        name="All data", marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scatter(
                        x=omega2[win_mask], y=F1_arr[win_mask], mode="markers",
                        name="Fit window", marker=dict(size=6, color="#1f77b4")))
                    if A1 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scatter(
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
                    st.plotly_chart(fig, width="stretch",
                                    key=f"pfp_f1_{model_short}_{fname}")

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
                            for _k, _v in _new_p.items():
                                if _k not in _grp_pk:
                                    live_params[_k] = _v
                                    params_out[_k]  = _v
                            for _k, _v in _new_a.items():
                                if _k not in _grp_ak:
                                    live_arrays[_k] = _v
                        except Exception:
                            pass

                prev_range = (f_hi_f1, f_hi_f1)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            if g_deps:
                st.caption(f"Depends on: {', '.join(g_deps)}")
            
            # for kind, content in group.get("formulas", []):
            #     if kind == "latex":
            #         st.latex(content)
            #     else:
            #         st.markdown(content)

            slider_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"

            # "Use previous range" button for dependent groups
            if g_deps:
                if st.button(f"↩ Same range as previous group",
                             key=f"pfp_useprev_{model_short}_{g_idx}_{fname}"):
                    st.session_state[slider_key] = prev_range
                    st.rerun()

            # Render per-group formulas before the slider
            for formula_type, formula_content in group.get("formulas", []):
                if formula_type == "markdown":
                    st.markdown(formula_content)
                elif formula_type == "latex":
                    st.latex(formula_content)


            # Initialize slider
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (f_min_v, f_max_v)

            f_lo, f_hi = st.slider(
                "Frequency range (GHz)",
                min_value=f_min_v, max_value=f_max_v,
                value=st.session_state[slider_key],
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
                cols = st.columns(2)
                for col_w, (arr_key, param_key, label, scale, unit) in zip(cols, row):
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

                    user_disp = float(st.session_state.get(inp_key, auto_disp))
                    user_SI   = user_disp / scale

                    ylabel = f"{label} ({unit})" if unit else label
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=f_plot, y=arr_plot, mode="lines", name=label,
                        line=dict(color="#1f77b4", width=2)))
                    if np.isfinite(user_disp):
                        fig.add_hline(
                            y=user_disp,
                            line=dict(color="#d62728", width=1.8, dash="dash"),
                            annotation_text=f"{user_disp:.4g} {unit}",
                            annotation_position="right",
                            annotation_font=dict(size=9, color="#d62728"))
                    if cold_res is not None and cold_param_map is not None:
                        _ck = cold_param_map.get(param_key)
                        if _ck and _ck in cold_res:
                            _cold_disp = float(cold_res[_ck]) * scale
                            if np.isfinite(_cold_disp):
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

                    col_w.plotly_chart(fig, width="stretch",
                                       key=f"pfp_{model_short}_{arr_key}_{fname}")

                    # Number input — key includes rng_tag so it resets to new median on slider move
                    actual_val = col_w.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp),
                        format="%.5g",
                        key=inp_key)

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
                        # Update live state for downstream groups only
                        for _k, _v in _new_p.items():
                            if _k not in _grp_param_keys:
                                live_params[_k] = _v
                                params_out[_k]  = _v
                        for _k, _v in _new_a.items():
                            if _k not in _grp_arr_keys:
                                live_arrays[_k] = _v
                    except Exception:
                        pass  # silently ignore re-extraction failures

            prev_range = (f_lo, f_hi)
            if g_idx < len(param_groups) - 1:
                st.markdown("---")

    return params_out


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


def render_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key):
    """
    Render the 🔧 Tuning expander below the Smith chart.

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

    with st.expander("🔧 Tuning", expanded=False):
        # Column headers
        hdr = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
        hdr[0].markdown("**Sweep**")
        hdr[1].markdown("**Parameter**")
        hdr[2].markdown("**Min**")
        hdr[3].markdown("**Step**")
        hdr[4].markdown("**Max**")
        hdr[5].markdown("**# Calc**")

        # "Use default" button — resets all min/step/max to current ± 1 (or 0/0/0 for zero params)
        if st.button("↩️ Use default values", key=f"tune_defaults_{topo_key}_{fname}"):
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

        # ── Calculate buttons ───────────────────────────────────────────
        _btn_cols = st.columns([1, 1] if _HAS_CUDA else [1])
        cpu_clicked = _btn_cols[0].button(
            "🧮 Calculate Residuals",
            key=f"tune_calc_{topo_key}_{fname}")
        cuda_clicked = (
            _HAS_CUDA
            and _btn_cols[1].button(
                "⚡ Calculate with CUDA",
                key=f"tune_calc_cuda_{topo_key}_{fname}"))

        # ── Helper: format a "best result" markdown line including all params ──
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
                if col in best_row.index:
                    v = float(best_row[col])
                    sval = (fmt % v) if np.isfinite(v) else "NaN"
                    parts.append(f"{label_p}: {sval} {unit}".strip())
            if parts:
                head += "  \n<small>" + ", ".join(parts) + "</small>"
            return head

        if cpu_clicked or cuda_clicked:
            use_cuda = bool(cuda_clicked)
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
                if row["enabled"]:
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
                return pd.DataFrame(top_arr_host, columns=col_names)

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
                progress = st.progress(0, text=f"Tuning ({_mode_label})…")
            with ui_cols[1]:
                stop_box = st.empty()
            best_box = st.empty()      # live "best so far" line

            # The Stop button works by triggering a Streamlit re-run on click;
            # the next st.* call inside the loop raises RerunException, which
            # we catch and turn into a clean cancellation.  No on_click needed.
            stop_key = f"tune_stop_{topo_key}_{fname}"
            stop_box.button(
                "⏹ Stop",
                key=stop_key,
                help="Stop the calculation. The best results found so far are kept.",
                type="secondary",
            )

            cancelled = False

            # ── Persistent device buffers for the top-K accumulator ─────────
            # Inf placeholders ensure new finite values always displace them.
            top_res    = xp.full(TOP_K, xp.inf, dtype=xp.float64)
            top_4      = xp.zeros((TOP_K, 4),       dtype=xp.float64)
            top_swept  = xp.zeros((TOP_K, max(n_swept, 1)), dtype=xp.float64)

            # ── Pre-allocated scratch buffers for the merge step ────────────
            # Sized for TOP_K + the *maximum* chunk we'd ever submit.  Reused
            # every iteration → zero per-chunk allocation churn for top-K.
            SCRATCH = TOP_K + CHUNK_MAX
            scratch_res   = xp.empty(SCRATCH, dtype=xp.float64)
            scratch_4     = xp.empty((SCRATCH, 4), dtype=xp.float64)
            scratch_swept = xp.empty((SCRATCH, max(n_swept, 1)), dtype=xp.float64)

            def _sync_topk_host():
                """Pull the top-K state to host as a (n, 5+n_params) numpy array.
                Drops inf placeholders. Constants are filled from sweep_lists.
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
                valid = np.isfinite(tr)
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
                        idx = xp.argpartition(view_res, TOP_K)[:TOP_K]
                        idx = idx[xp.argsort(view_res[idx])]

                        top_res[:] = view_res[idx]
                        top_4[:]   = scratch_4[:total_in][idx]
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
                        progress.progress(
                            min(1.0, processed / max(n_total, 1)),
                            text=(f"Tuning ({_mode_label})… "
                                  f"{processed:,}/{n_total:,} combos  "
                                  f"({rate:,.0f}/s, ETA {eta:.1f}s)  "
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
            best_box.empty()
            stop_box.empty()

            n_kept = 0 if top_res is None else int(xp.sum(xp.isfinite(top_res)).item())
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

                        # Re-sort by fp64 residuals
                        order_dev = xp.argsort(tot_rk)
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

        # Display results if available
        df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
        if df is not None:
            best = df.iloc[0]
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

                for spec in ticked_specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    kp = f"tune_{topo_key}_{key}_{fname}"
                    sweep_min = float(st.session_state.get(f"{kp}_min", 0))
                    sweep_step = float(st.session_state.get(f"{kp}_step", 0))
                    sweep_max = float(st.session_state.get(f"{kp}_max", 0))
                    sweep_vals = _make_sweep_values(sweep_min, sweep_max, sweep_step)

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
                        fig.add_trace(go.Scatter(
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
                    st.plotly_chart(fig, width="stretch",
                                   key=f"tune_sens_{topo_key}_{key}_{fname}")

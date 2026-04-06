"""
models/base_ui.py — Shared Streamlit UI helpers used by all model classes.

Kept separate from the abstract base so model files import one thing,
not a chain of ssm_* modules.
"""
from __future__ import annotations
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
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException as _RerunException
except Exception:
    try:
        from streamlit.runtime.scriptrunner import RerunException as _RerunException
    except Exception:
        _RerunException = None  # type: ignore[assignment]

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
    title_line1 = f"Measured vs Modeled — {model_name}   (Total Residual: {error_pct:.2f}%)"
    title_line2 = (f"S11: {port_res['S11']:.2f}%   S12: {port_res['S12']:.2f}%   "
                   f"S21: {port_res['S21']:.2f}%   S22: {port_res['S22']:.2f}%")
    fig.update_layout(
        title=f"{title_line1}<br><sup>{title_line2}</sup>",
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

def render_interactive_param_groups(params, arrays, freq, fname, model_short, param_groups,
                                    cold_res=None, cold_param_map=None, reextract_fn=None):

    """
    For each parameter group, render:
      - A labelled section heading with dependency info
      - A "same range as previous" button (for dependent groups)
      - A frequency range slider
      - Per-frequency line plots with a dashed line at the current median
      - A number_input per parameter for manual override
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

            # ── Pack per-param data into 2D device tables for vectorised
            #     lookup.  Doing the index math in ONE kernel launch instead
            #     of one launch per parameter is a major Python-overhead win.
            L_max = max(max(L_list), 1)
            sweep_table_h = np.zeros((n_params, L_max), dtype=np.float64)
            for i, sl in enumerate(sweep_lists):
                sweep_table_h[i, :L_list[i]] = sl
            sweep_table_dev    = xp.asarray(sweep_table_h)                    # (n_params, L_max)
            strides_dev        = xp.asarray(np.asarray(strides, dtype=np.int64))   # (n_params,)
            L_dev              = xp.asarray(np.asarray(L_list,  dtype=np.int64))   # (n_params,)
            sweep_scales_dev   = xp.asarray(np.asarray(sweep_scales, dtype=np.float64))  # (n_params,)
            param_axis_dev     = xp.arange(n_params, dtype=xp.int64)
            if n_swept > 0:
                swept_indices_dev = xp.asarray(np.asarray(swept_idx_full, dtype=np.int64))
            else:
                swept_indices_dev = None

            # Constant scalars passed straight into simulate_batch
            const_si = {sweep_keys[i]: float(sweep_lists[i][0]) / float(sweep_scales[i])
                        for i in range(n_params) if L_list[i] == 1}
            varies_mask = [L_list[i] > 1 for i in range(n_params)]
            swept_set = {sweep_keys[i] for i in range(n_params) if varies_mask[i]}

            # Move S_mea onto the compute device once
            S_mea_dev = xp.asarray(S_raw)

            # ── Pre-bake constant sub-networks (lifted out of the chunk loop) ──
            # If the user isn't sweeping any of a sub-network's input params,
            # build that network's planes ONCE here and reuse them every chunk.
            # Eliminates dozens of kernel launches per chunk in the common
            # "tune intrinsic params only" case.
            _PAD_CAP_KEYS = ("Cpbe", "Cpce", "Cpbc")
            _SER_LEAD_KEYS = ("Rpb", "Rpc", "Rpe", "Lb", "Lc", "Le")
            _CHENG_EXTR_KEYS = ("Cbex", "Cbcx")

            static_p = dict(all_p)
            for _k, _v in const_si.items():
                static_p[_k] = _v
            omega_dev = xp.asarray(2.0 * np.pi * np.asarray(freq, dtype=np.float64)
                                   ).reshape(1, len(freq))

            static_cache = {"omega": omega_dev}
            _cached_msgs = []
            if not (set(_PAD_CAP_KEYS) & swept_set):
                static_cache["Y_pad"] = build_Y_pad_batch(static_p, omega_dev, 1, len(freq), xp)
                _cached_msgs.append("Y_pad")
            if not (set(_SER_LEAD_KEYS) & swept_set):
                static_cache["Z_ser"] = build_Z_ser_batch(static_p, omega_dev, 1, len(freq), xp)
                _cached_msgs.append("Z_ser")
            if not (set(_CHENG_EXTR_KEYS) & swept_set):
                _Cbex_c = float(static_p.get("Cbex", 0.0))
                _Cbcx_c = float(static_p.get("Cbcx", 0.0))
                static_cache["Y_extr"] = (1j * omega_dev * _Cbex_c,
                                          1j * omega_dev * _Cbcx_c)
                _cached_msgs.append("Y_extr")
            if _cached_msgs:
                st.caption("Pre-baked constant networks: " + ", ".join(_cached_msgs))
                print(f"[tune] pre-baked: {', '.join(_cached_msgs)}", flush=True)

            col_names = ["Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
            for lbl, u in zip(sweep_labels, sweep_units):
                col_names.append(f"{lbl} ({u})" if u else lbl)

            sess_key = f"tune_df_{topo_key}_{fname}"

            def _topk_to_df(top_arr_host):
                return pd.DataFrame(top_arr_host, columns=col_names)

            # ── Auto chunk sizing from device free memory ───────────────────
            # Per-combo working-set estimate:
            #   simulate_batch holds ~8 (B, N, 2, 2) complex128 tensors
            #   simultaneously (Y_pad, Z_ser, Y_int, intermediates, S_out, plus
            #   linalg.inv scratch).  64 bytes per (2, 2) cell × N_freq × ~8.
            N_freq = len(freq)
            per_combo_bytes = 64 * N_freq * 8
            free_label = ""
            if use_cuda:
                try:
                    free_b, total_b = _cp.cuda.runtime.memGetInfo()
                    dev = _cp.cuda.Device(0)
                    sm_count = dev.attributes.get("MultiProcessorCount", 0)
                    free_label = (f"GPU{dev.id}: {free_b/1024**3:.2f}/"
                                  f"{total_b/1024**3:.2f} GiB free  ·  {sm_count} SMs")
                    # Aggressive: use most of the free VRAM as working set.
                    budget = free_b * 0.85
                except Exception:
                    budget = 2 * 1024**3
                CHUNK = max(1024, int(budget // max(per_combo_bytes, 1)))
                CHUNK = min(CHUNK, n_total, 4_000_000)
            else:
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                    free_label = f"CPU RAM: {avail/1024**3:.2f} GiB free"
                except Exception:
                    avail = 4 * 1024**3
                budget = avail * 0.30
                CHUNK = max(64, int(budget // max(per_combo_bytes, 1)))
                CHUNK = min(CHUNK, n_total, 16_384)
            CHUNK_MIN = 16
            CHUNK_MAX = CHUNK   # remember the initial cap for scratch sizing

            import sys as _sys, time as _time
            _t_start = _time.time()
            if free_label:
                st.caption(free_label)
            print(f"\n[tune] start  mode={_mode_label}  total={n_total:,}  "
                  f"chunk={CHUNK:,}  batched={has_batch}  {free_label}", flush=True)

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

            try:
                if not has_batch:
                    raise RuntimeError(
                        f"Model {model_cls.__name__} has no simulate_batch — "
                        f"cannot run batched tuning. Implement simulate_batch.")

                processed = 0
                last_ui = 0.0
                chunks_done = 0
                last_chunk_ms = 0.0
                while processed < n_total:
                    s = processed
                    e = min(s + CHUNK, n_total)
                    B = e - s
                    _t_chunk = _time.time()

                    # ── Generate this chunk's combos directly on the device ──
                    # Vectorised: ONE indexing op covers all parameters.
                    try:
                        lin = xp.arange(s, e, dtype=xp.int64)                       # (B,)
                        idx_2d = (lin[:, None] // strides_dev[None, :]) \
                                  % L_dev[None, :]                                  # (B, n_params)
                        chunk_disp = sweep_table_dev[param_axis_dev[None, :], idx_2d]  # (B, n_params)
                        chunk_si   = chunk_disp / sweep_scales_dev[None, :]            # (B, n_params)

                        p_chunk = dict(all_p)
                        for i, key in enumerate(sweep_keys):
                            if varies_mask[i]:
                                p_chunk[key] = chunk_si[:, i]   # device view
                            else:
                                p_chunk[key] = const_si[key]    # scalar

                        if n_swept > 0:
                            cur_swept = chunk_disp[:, swept_indices_dev]  # (B, n_swept)
                        else:
                            cur_swept = None
                    except Exception as exc:
                        is_oom = (isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() or
                                  "OutOfMemoryError" in type(exc).__name__)
                        if is_oom and CHUNK > CHUNK_MIN:
                            CHUNK = max(CHUNK_MIN, CHUNK // 2)
                            print(f"\n[tune] OOM in param-gen → chunk {CHUNK:,}", flush=True)
                            if use_cuda:
                                try: _cp.get_default_memory_pool().free_all_blocks()
                                except Exception: pass
                            continue
                        raise

                    # ── Run simulate + residuals + top-K merge entirely on
                    #     the device.  No host sync inside this hot block.   ──
                    try:
                        S_batch = model_cls.simulate_batch(
                            p_chunk, freq, z0, xp=xp, cache=static_cache)
                        res = _port_residuals_batch(S_mea_dev, S_batch, xp)
                        cur_total = res["Total"]                              # (B,)
                        cur_4 = xp.stack(
                            [res["S11"], res["S12"], res["S21"], res["S22"]],
                            axis=1)                                            # (B, 4)

                        # NaN/inf protection — replace with a huge finite value
                        # so they sort to the bottom of argpartition without
                        # forcing a host-side .all() check (no sync).
                        BIG = 1.0e308
                        cur_total = xp.where(xp.isfinite(cur_total), cur_total, BIG)

                        # ── Top-K merge into pre-allocated scratch ──────────
                        total_in = TOP_K + B
                        scratch_res[:TOP_K]              = top_res
                        scratch_res[TOP_K:total_in]      = cur_total
                        scratch_4[:TOP_K]                = top_4
                        scratch_4[TOP_K:total_in]        = cur_4
                        if n_swept > 0:
                            scratch_swept[:TOP_K]            = top_swept
                            scratch_swept[TOP_K:total_in]    = cur_swept

                        view_res = scratch_res[:total_in]
                        idx = xp.argpartition(view_res, TOP_K)[:TOP_K]
                        idx = idx[xp.argsort(view_res[idx])]

                        # Write the K winners back into the persistent buffers
                        top_res[:] = view_res[idx]
                        top_4[:]   = scratch_4[:total_in][idx]
                        if n_swept > 0:
                            top_swept[:] = scratch_swept[:total_in][idx]

                        del S_batch, res, cur_total, cur_4, idx, view_res
                        del lin, idx_2d, chunk_disp, chunk_si
                        if cur_swept is not None:
                            del cur_swept
                    except Exception as exc:
                        is_oom = (isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() or
                                  "OutOfMemoryError" in type(exc).__name__)
                        if is_oom and CHUNK > CHUNK_MIN:
                            new_chunk = max(CHUNK_MIN, CHUNK // 2)
                            print(f"\n[tune] OOM at chunk {CHUNK:,} → retry with {new_chunk:,}",
                                  flush=True)
                            CHUNK = new_chunk
                            if use_cuda:
                                try: _cp.get_default_memory_pool().free_all_blocks()
                                except Exception: pass
                            continue   # do NOT advance processed
                        print(f"\n[tune] chunk {s}-{e} failed: {exc!r}", flush=True)
                        processed = e
                        continue

                    processed = e
                    chunks_done += 1
                    last_chunk_ms = (_time.time() - _t_chunk) * 1000.0

                    # ── Throttled UI tick: ~2 Hz, the only host sync point ──
                    now = _time.time()
                    if now - last_ui > 0.5 or processed >= n_total:
                        elapsed = now - _t_start
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        eta = (n_total - processed) / rate if rate > 0 else 0.0
                        progress.progress(
                            processed / n_total,
                            text=(f"Tuning ({_mode_label})… "
                                  f"{processed:,}/{n_total:,} combos  "
                                  f"({rate:,.0f}/s, ETA {eta:.1f}s)  "
                                  f"chunk={CHUNK:,}  ({last_chunk_ms:.1f} ms/chunk)"))
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
                        f"({100.0*processed/n_total:5.1f}%)  "
                        f"{(processed / max(_time.time()-_t_start, 1e-9)):>10,.0f} calc/s  "
                        f"chunk={CHUNK:,}  {last_chunk_ms:6.1f} ms/chunk")
                    _sys.stdout.flush()
            except (KeyboardInterrupt, SystemExit):
                cancelled = True
                st.warning("Computation cancelled — keeping the best results found so far.")
                print("\n[tune] cancelled by user (KeyboardInterrupt)", flush=True)
            except MemoryError as me:
                st.error(f"Out of memory: {me}. Keeping the best results found so far.")
                print(f"\n[tune] MemoryError: {me}", flush=True)
            except BaseException as exc:
                # Streamlit raises RerunException when the user clicks any
                # widget (incl. our Stop button).  Persist and re-raise so
                # Streamlit can finish the rerun cleanly.
                if _RerunException is not None and isinstance(exc, _RerunException):
                    cancelled = True
                    print("\n[tune] cancelled (Streamlit rerun, e.g. Stop button)",
                          flush=True)
                    _persist_topk()
                    # Free GPU buffers before re-raising so the rerun starts clean
                    if use_cuda:
                        try:
                            del S_mea_dev, top_res, top_4, top_swept
                            del scratch_res, scratch_4, scratch_swept
                            del sweep_table_dev, strides_dev, L_dev
                            del sweep_scales_dev, param_axis_dev
                            if swept_indices_dev is not None:
                                del swept_indices_dev
                            static_cache.clear()
                            _cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
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

            # Free GPU buffers
            if use_cuda:
                try:
                    del S_mea_dev, top_res, top_4, top_swept
                    del scratch_res, scratch_4, scratch_swept
                    del sweep_table_dev, strides_dev, L_dev
                    del sweep_scales_dev, param_axis_dev
                    if swept_indices_dev is not None:
                        del swept_indices_dev
                    static_cache.clear()
                    _cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

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

"""
models/smith_ui.py — Smith chart rendering + pad-parameter sync helpers,
shared by every model's render_override_and_smith.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ..helpers import extended_smith_grid, params_hash, plotly_with_dl
from tools.common.i18n import tr

from .residuals import _port_residuals
from .tuning.ranges import PAD_SPECS, _PAD_KEYS


# ── Smith chart ───────────────────────────────────────────────────────────────

_SMITH_COLORS = {"S11":"#1f77b4","S22":"#ff7f0e","S21":"#2ca02c","S12":"#d62728"}

def render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith",
                       show_title=True, meas_label="Meas.", sim_label="Model",
                       *, compact: bool = False, height: int | None = None,
                       extra_download: tuple | None = None,
                       inline_labels: bool = False):
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
    for _grid_tr in extended_smith_grid(1.0):
        fig.add_trace(_grid_tr)
    inline_annos = []     # populated when inline_labels=True (Sxx near trace)
    for name, (r, c) in [("S11",(0,0)),("S22",(1,1)),("S21",(1,0)),("S12",(0,1))]:
        col = _SMITH_COLORS[name]; sc = scales.get(name, 1.0)
        sm = S_mea[:,r,c]*sc; sk = S_sim[:,r,c]*sc
        sc_lbl = "" if abs(sc-1.0)<1e-9 else (f" ×{sc:.2g}" if sc>=1 else f" ÷{1/sc:.2g}")
        fig.add_trace(go.Scattergl(x=sm.real, y=sm.imag, mode="markers",
                                  name=f"{name}{sc_lbl} {meas_label}",
                                  marker=dict(color=col, size=5, symbol="circle"),
                                  showlegend=not inline_labels,
                                  hovertemplate=f"{name} {meas_label}<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scattergl(x=sk.real, y=sk.imag, mode="lines",
                                  name=f"{name}{sc_lbl} {sim_label}",
                                  line=dict(color=col, width=2.0, dash="dash"),
                                  showlegend=not inline_labels,
                                  hovertemplate=f"{name} {sim_label}<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        if inline_labels:
            # Label each S-param near its (model) trace centroid, nudged
            # radially outward — mirrors the matplotlib Smith chart's on-trace
            # Sxx labels and frees the space a legend box would eat.
            with np.errstate(invalid="ignore"):
                cx = float(np.nanmean(sk.real)); cy = float(np.nanmean(sk.imag))
            if np.isfinite(cx) and np.isfinite(cy):
                rr = (cx*cx + cy*cy) ** 0.5
                if rr > 1e-6:
                    f = (rr + 0.13) / rr
                    cx *= f; cy *= f
                inline_annos.append(dict(x=cx, y=cy, xref="x", yref="y",
                                         text=f"{name}{sc_lbl}", showarrow=False,
                                         font=dict(size=13, color=col)))
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
    if inline_labels:
        # Inline-label mode: no legend box; Sxx sit on the traces, plus one
        # small meas/model hint pinned bottom-left inside the plot.
        annotations = list(inline_annos)
        annotations.append(dict(
            x=-1.06, y=-1.06, xref="x", yref="y", xanchor="left", yanchor="bottom",
            showarrow=False,
            text=f"● {meas_label}   - - {sim_label}",
            font=dict(size=9, color="gray"), align="left"))
        if compact:
            margin_cfg = dict(l=40, r=15, t=40 if show_title else 10, b=40)
    else:
        annotations = [dict(x=0.5, y=annotation_y, xref="paper", yref="paper",
                            showarrow=False,
                            text=f"● {meas_label} (markers)  |  - - {sim_label} (dashed)",
                            font=dict(size=10, color="gray"), align="center")]
    fig.update_layout(
        title=title_cfg,
        xaxis=dict(title="Re(Γ)", range=[-1.1,1.1], scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1,1.1], showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=eff_height,
        margin=margin_cfg,
        legend=legend_cfg,
        showlegend=not inline_labels,
        hovermode="closest",
        annotations=annotations)
    plotly_with_dl(fig, key=key, filename=key, extra_download=extra_download)


def render_smith_with_ftfmax(S_raw, S_sim, freq, model_name: str,
                             model_short: str, fname: str, scales=None,
                             *, s2p_bytes: bytes | None = None,
                             s2p_filename: str | None = None):
    """
    Two-column layout: Smith chart (left) + fT/fmax mini-card (right).

    A sticky st.metric row (Total + per-port S11/S12/S21/S22 residuals, keyed
    container `hbt_resrow_<model_short>` pinned via ui_theme.py CSS) renders
    above the Smith chart in place of the old title-line residual sentence;
    the mini-card on the right shows |h21|² and Mason U for both measured and
    modeled with 20 dB/dec extrapolation when needed.  Use this in place of
    the bare `render_smith_chart()` call inside each model's
    `render_override_and_smith`.

    When ``s2p_bytes`` is supplied, the Smith chart's download row gains a
    second button (📥 S2P) right next to the standard xlsx — this replaced
    the standalone "Download Modeled DUT S2P" section that used to live at
    the bottom of the SSM extraction tab.
    """
    # Local import — ssm_plots imports back from base_ui at module load time,
    # so a top-level import here would create a circular dependency.
    from ..ssm_plots import render_ft_fmax_card

    port_res = _port_residuals(S_raw, S_sim)
    err = float(port_res["Total"])
    # Sticky residual strip — keyed container so ui_theme.py can pin it
    # (position:sticky) while the user scrolls the tuning expanders.  The key
    # deliberately excludes fname (dots/commas would break the st-key-* CSS
    # class); model_short is unique per rendered page.
    _prev_key = f"hbt_res_prev_{model_short}_{fname}"
    _prev = st.session_state.get(_prev_key)
    with st.container(key=f"hbt_resrow_{model_short}"):
        mc = st.columns(5)
        for i, pname in enumerate(("Total", "S11", "S12", "S21", "S22")):
            cur = float(port_res[pname])
            delta = None
            if isinstance(_prev, dict) and pname in _prev:
                d = cur - float(_prev[pname])
                if abs(d) >= 0.005:
                    delta = f"{d:+.2f}%"
            mc[i].metric(
                tr("Total residual", "總殘差") if pname == "Total" else pname,
                f"{cur:.2f}%", delta=delta, delta_color="inverse",
                help=tr("RMS relative S-parameter error across all four ports —"
                        " the number to minimise.  Deltas compare against the"
                        " previous simulation of this model.",
                        "四個埠上的均方根相對 S 參數誤差 — 需最小化的數值。"
                        "差值與此模型上一次模擬結果比較。")
                     if pname == "Total" else None)
    st.session_state[_prev_key] = {k: float(v) for k, v in port_res.items()}
    extra_dl = None
    if s2p_bytes is not None and s2p_filename is not None:
        extra_dl = (tr("📥 modeled S2P", "📥 模型 S2P"),
                    s2p_bytes, s2p_filename, "text/plain")
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
    st.markdown(
        f"<b>{tr('S display scale', 'S 顯示縮放')}</b>"
        f" <span class='hbt-help' title='{tr(
            'Multiply each trace before plotting.'
            ' Display only - does not affect the residual.',
            '繪圖前乘上此係數，僅影響顯示，不影響殘差。')}'>?</span>",
        unsafe_allow_html=True)
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
    # Also reseed when a widget key was GC'd by a page switch — Streamlit
    # drops session_state for widgets that skip a run, while this sync hash
    # (a plain value, not a widget) survives and would otherwise block the
    # reseed, leaving the pad inputs to recreate at 0.
    _keys_missing = any(f"sim_{topo_key}_{key}_{fname}" not in st.session_state
                        for key, *_ in PAD_SPECS)
    if _keys_missing or st.session_state.get(sync_key) != preov_hash:
        for key, _, scale, *_ in PAD_SPECS:
            st.session_state[f"sim_{topo_key}_{key}_{fname}"] = float(para_eff.get(key, 0.0)) * scale
        st.session_state[sync_key] = preov_hash

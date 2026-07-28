"""
ui_use.py — "Use" workflow for the custom SSM builder.

Load a saved topology (library dropdown or upload), type a value for every
named component, forward-simulate S-parameters with the generic nodal solver,
and inspect the results.  The result panel now mirrors the built-in models'
forward-simulation view:

* Smith chart (Plotly) + fT/fmax Bode plot with −20 dB/dec & single-pole
  extrapolation (segmented method selector, same as the other models);
* "🔣 Calculated τ_total and fmax" expander;
* "🖼️ Topology illustration" expander — the live schematic (the created
  picture) with PNG / SVG download + copy-image;
* "🍩 Smith Chart (Matplotlib)" publication chart (chart left, controls right);
* "🎚️ Tuning — interactive slider preview" (model-only live sliders).

S2P / Excel downloads mirror the other portal pages.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from .core import simulate_custom_model
from .schematic import (render_schematic, svg_png_buttons,
                        svg_pixel_height)
from ..helpers import (extended_smith_grid, write_s2p, copy_button, fig_to_tsv)
from ..ssm_plots import (render_forward_bode_block, render_tau_fmax_expander,
                        render_matplotlib_smith)
from ..models.base_ui import (smith_scale_controls, render_visual_tuning_expander,
                              render_smith_with_ftfmax, render_tuning_expander)
# Reuse the Fit view's model loader / value inputs / tuning-state clearing so the
# combined view shares one model source (cmf_model) and one value-key scheme.
from .ui_fit import (_make_adapter, _load_model, _value_inputs,
                     _clear_tuning_state, _disp_spec, _kohm_ids,
                     _TOPO as _FTOPO)
from ._i18n import tr

_SMITH_COLORS = {"S11": "#1f77b4", "S22": "#ff7f0e",
                 "S21": "#2ca02c", "S12": "#d62728"}

# Stable "fname" for the no-device (forward) case; with a device attached the
# device label is used instead, giving per-device tuning/value state.
_FNAME = "forward"


def _smith_fig(S: np.ndarray, freq: np.ndarray, title: str,
               scales: dict | None = None) -> go.Figure:
    if scales is None:
        scales = {"S11": 1.0, "S12": 1.0, "S21": 1.0, "S22": 1.0}
    fig = go.Figure()
    for _grid_tr in extended_smith_grid(1.0):
        fig.add_trace(_grid_tr)
    f_ghz = freq * 1e-9
    for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                         ("S21", (1, 0)), ("S12", (0, 1))]:
        sv = S[:, r, c] * scales.get(name, 1.0)        # per-trace display scale
        hov = [f"f={fv:.3f} GHz<br>Re={rv:.4f}<br>Im={iv:.4f}"
               for fv, rv, iv in zip(f_ghz, sv.real, sv.imag)]
        fig.add_trace(go.Scatter(x=sv.real, y=sv.imag, mode="lines", name=name,
                                 line=dict(color=_SMITH_COLORS[name], width=2),
                                 text=hov, hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"Smith Chart — {title}", font=dict(size=12)),
        xaxis=dict(title="Re(Γ)", range=[-1.1, 1.1], scaleanchor="y",
                   scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1, 1.1], showgrid=False,
                   zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(x=1.02, y=1.0, xanchor="left"), hovermode="closest")
    return fig


def render_use_ui(measured: dict | None = None) -> None:
    """Unified custom-model workbench: load a model, set component values, and
    forward-simulate — and, when a measured device is supplied, overlay it on the
    same Smith/Bode with the residual + grid-sweep Auto Tuning.  Replaces the old
    separate *Load / simulate* and *Fit to measurement* modes."""
    has_dev = measured is not None
    st.subheader(tr("🔬 Custom model — simulate", "🔬 自訂模型 — 模擬")
                 + (tr(" & fit to the measured device", " 並對量測元件擬合")
                    if has_dev else ""))
    from .ui_build import fire_pending_download
    fire_pending_download()           # download a model just "sent" from build

    fname = str(measured["label"]) if has_dev else _FNAME
    model = _load_model(fname)        # shared cmf_model source + build handoff
    if model is None:
        st.info(tr("⬆️ Upload a custom model `.json` (or send one from the "
                   "**Build model** view) to simulate", "⬆️ 上傳自訂模型 `.json`"
                   "（或從**建立模型**頁傳送一個）以進行模擬")
                + (tr(" and overlay it on this device.", " 並疊加於此元件上。")
                   if has_dev else "。"))
        return
    # Drop the previous model's stale tuning / value state when it changes.
    tok = st.session_state.get("cmf_model_token", 0)
    if st.session_state.get(f"cmf_seen_{fname}") != tok:
        _clear_tuning_state(fname)
        st.session_state[f"cmf_seen_{fname}"] = tok
    _dev_zh = {"Bipolar": "雙極性", "Unipolar": "單極性"}
    _itype = "π" if model.intrinsic_type == "Pi" else "T"
    st.success(
        tr(f"Loaded **{model.name}** · {model.device} · intrinsic {_itype}",
           f"已載入 **{model.name}** · {_dev_zh.get(model.device, model.device)}"
           f" · 本質 {_itype}")
        + (tr(f"  —  overlaying **{fname}**", f"  —  疊加 **{fname}**")
           if has_dev else ""))

    # ── Frequency: locked to the device grid when fitting, else user-set ──────
    if has_dev:
        freq = np.asarray(measured["freq"], dtype=float)
        z0 = float(measured.get("z0", 50.0))
        S_meas = np.asarray(measured["S"])
    else:
        cf = st.columns(3)
        # min 1 MHz — a 0 GHz start puts a DC point in the sweep, where series
        # C branches / the α source have no meaningful small-signal admittance.
        f0 = cf[0].number_input(tr("Start (GHz)", "起始 (GHz)"), min_value=0.001,
                                value=0.01, format="%.4f", step=0.01, key="cmu_f0")
        npts = cf[1].number_input(tr("Points", "點數"), min_value=2, value=801,
                                  step=1, key="cmu_npts")
        f1 = cf[2].number_input(tr("Stop (GHz)", "結束 (GHz)"), min_value=0.001,
                                value=50.0, format="%.4f", step=1.0, key="cmu_f1")
        if f1 <= f0:
            st.error(tr("Stop frequency must exceed start.",
                        "結束頻率必須大於起始頻率。"))
            return
        freq = np.linspace(float(f0) * 1e9, float(f1) * 1e9, int(npts))
        z0, S_meas = 50.0, None

    st.divider()
    with st.expander(tr("⚙️ Component values", "⚙️ 元件數值"), expanded=True):
        values = _value_inputs(model, fname)

    st.divider()
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S_sim = simulate_custom_model(model, freq, values, z0)
    except Exception as exc:  # degenerate topology / singular matrix
        st.error(tr(f"Simulation failed: {exc}", f"模擬失敗：{exc}"))
        return
    if not np.all(np.isfinite(S_sim)):
        st.warning(tr("Some S-parameter points are non-finite — check for missing "
                      "values (e.g. a series branch left at 0).",
                      "部分 S 參數點為非有限值 — 請檢查是否有缺漏的數值"
                      "（例如某串聯分支仍為 0）。"))

    sc = smith_scale_controls(fname, _FTOPO)   # per-trace Smith × multipliers
    s2p = write_s2p(freq, S_sim, title=f"Custom model {model.name}",
                    params={"intrinsic": model.intrinsic_type})

    # ── Smith + fT/fmax — overlay the device when present, else model-only ────
    if has_dev:
        render_smith_with_ftfmax(
            S_meas, S_sim, freq,
            model_name=f"Custom · {model.name}", model_short=_FTOPO, fname=fname,
            scales=sc, s2p_bytes=s2p,
            s2p_filename=f"{model.name or 'custom'}_sim.s2p")
        render_tau_fmax_expander(
            key=f"cmf_taufmax_{fname}", freq=freq, S_meas=S_meas, S_model=S_sim,
            CBC=0.0, Rbb=0.0, tau_sum=None, tau_sum_label="τ",
            tau_sum_tex=r"\tau", extrap_key=f"ftfmax_card_{_FTOPO}_{fname}")
    else:
        left, right = st.columns([3, 2])
        with left:
            smith = _smith_fig(S_sim, freq, model.name, scales=sc)
            st.plotly_chart(smith, width="stretch")
            dl = st.columns(2)
            dl[0].download_button(tr("📥 .s2p", "📥 .s2p"), data=s2p,
                                  file_name=f"{model.name or 'custom'}.s2p",
                                  mime="text/plain", key="cmu_dl_s2p",
                                  width="stretch")
            copy_button(fig_to_tsv(smith) or "", "cmu_copy_smith",
                        container=dl[1], label=tr("📋 copy", "📋 複製"))
        with right:
            render_forward_bode_block(S_sim, freq, model.name, key="cmu_bode")
        render_tau_fmax_expander(
            key="cmu_taufmax", freq=freq, S_meas=S_sim, CBC=0.0, Rbb=0.0,
            tau_sum=None, tau_sum_label="τ", tau_sum_tex=r"\tau",
            extrap_key="cmu_bode")

    # ── Topology illustration ─────────────────────────────────────────────────
    svg = render_schematic(model, values)
    with st.expander(tr("🖼️ Topology illustration", "🖼️ 拓樸示意圖"),
                     expanded=not has_dev):
        st.iframe(svg, height=svg_pixel_height(svg) + 12)
        scc = st.columns([2, 1])
        svg_png_buttons(svg, filename=f"{model.name or 'custom'}.png",
                        container=scc[0], zoom=2,
                        dl_label=tr("🖼️ Download PNG", "🖼️ 下載 PNG"),
                        copy_label=tr("📋 copy image", "📋 複製圖片"))
        scc[1].download_button(tr("⬇ Download SVG", "⬇ 下載 SVG"), data=svg,
                               file_name=f"{model.name or 'custom'}.svg",
                               mime="image/svg+xml", key="cmu_dl_svg",
                               width="stretch")

    # ── Publication matplotlib Smith chart (chart left, controls right) ──────
    with st.expander(tr("🍩 Smith Chart (Matplotlib)", "🍩 Smith 圖（Matplotlib）"),
                     expanded=False):
        c_left, c_right = st.columns([1.2, 1])
        if has_dev:
            with c_right:
                render_matplotlib_smith(S_meas, S_sim, fname, _FTOPO,
                                        default_multiplier=sc,
                                        phase="controls", freq_hz=freq)
            with c_left:
                render_matplotlib_smith(S_meas, S_sim, fname, _FTOPO,
                                        default_multiplier=sc,
                                        phase="chart", freq_hz=freq)
        else:
            _sets = [{"S": S_sim, "label": "Simulated",
                      "kind": "line", "style": "solid"}]
            with c_right:
                render_matplotlib_smith(fname=fname, topo_key="use", sets=_sets,
                                        default_multiplier=sc,
                                        phase="controls", freq_hz=freq)
            with c_left:
                render_matplotlib_smith(fname=fname, topo_key="use", sets=_sets,
                                        default_multiplier=sc,
                                        phase="chart", freq_hz=freq)

    # ── Tuning — Visual Tuning (🎯 Live tweak / ⚡ Wide sweep) always; Auto
    #    Tuning (grid sweep, ranks vs the device residual) only with a device.
    #    Reference trace: the measured device when present, else the current sim.
    # Same display scale/unit the value inputs used (kΩ on the b–c junction R),
    # so slider drags and "use best values" round-trip through one convention.
    _kohm = _kohm_ids(model)
    tuning_specs = []
    for _title, items in model.grouped_value_specs():
        for cid, kind, name in items:
            _scale, _start, _unit = _disp_spec(cid, kind, _kohm)
            tuning_specs.append((cid, name, _scale, _unit))
    adapter = _make_adapter(model)
    S_ref = S_meas if has_dev else S_sim
    render_visual_tuning_expander(adapter, values, S_ref, freq, z0,
                                  tuning_specs, fname, _FTOPO)
    if has_dev:
        # Progressive auto-fit default scope: everything except the parasitic
        # pad caps + lead inductances (the outermost de-embedded shells).
        _prog_default = [
            cid
            for _t, items in model.grouped_value_specs()
            if _t not in ("Parasitic pad capacitances", "Lead inductance")
            for cid, _k, _n in items]
        render_tuning_expander(adapter, values, S_meas, freq, z0,
                               tuning_specs, fname, _FTOPO,
                               default_fit_keys=_prog_default)

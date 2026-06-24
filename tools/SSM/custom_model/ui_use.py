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

from .core import CustomModel, simulate_custom_model, load_model
from .schematic import (render_schematic, svg_to_png, copy_image_button,
                        svg_pixel_height)
from ..helpers import (extended_smith_grid, write_s2p, copy_button, fig_to_tsv,
                      apply_pending)
from ..ssm_plots import (render_forward_bode_block, render_tau_fmax_expander,
                        render_matplotlib_smith)

# kind → (display-scale, unit, number-format)
_UNITS = {
    "C": (1e15, "fF", "%.4f"),
    "R": (1.0, "Ω", "%.4f"),
    "L": (1e12, "pH", "%.4f"),
    "gm": (1e3, "mS", "%.4f"),
    "tau": (1e12, "ps", "%.4f"),
    "alpha": (1.0, "", "%.5f"),
}
_SMITH_COLORS = {"S11": "#1f77b4", "S22": "#ff7f0e",
                 "S21": "#2ca02c", "S12": "#d62728"}

_MODEL_KEY = "cmu_model"
_TOPO = "cmu"


def _load_source() -> CustomModel | None:
    """Upload a custom-model .json — it loads automatically (no button)."""
    up = st.file_uploader("Upload a custom model .json", type=["json"],
                          key="cmu_upload")
    if up is not None:
        # Auto-load on upload; re-load only when the file's bytes change so we
        # don't rebuild on every rerun.  Guards against a corrupt/old file.
        data = up.getvalue()
        sig = (up.name, len(data), hash(data))
        if st.session_state.get("cmu_upload_sig") != sig:
            try:
                st.session_state[_MODEL_KEY] = load_model(data)
                st.session_state["cmu_upload_sig"] = sig
                # fresh model → drop the cached schematic so it rebuilds
                st.session_state.pop("cmu_base", None)
            except Exception as exc:                 # noqa: BLE001
                st.error(f"Could not read that .json: {exc}")
                return None
    return st.session_state.get(_MODEL_KEY)


def _value_inputs(model: CustomModel) -> dict:
    """Render value inputs grouped outside→inside (parasitic → lead L → access R
    → extrinsic → port/delay → intrinsic), clean names; return {id: SI value}."""
    values: dict = {}
    st.markdown("##### Component values")
    cols_per_row = 4
    for title, items in model.grouped_value_specs():
        st.markdown(f"**{title}**")
        for row_start in range(0, len(items), cols_per_row):
            row = items[row_start:row_start + cols_per_row]
            cs = st.columns(len(row))
            for col, (cid, kind, name) in zip(cs, row):
                scale, unit, fmt = _UNITS.get(kind, (1.0, "", "%.4f"))
                sk = f"cmu_val_{cid}"
                # Promote any value committed by the tuning preview BEFORE the
                # number_input is created (writing it afterwards is illegal).
                apply_pending(sk)
                if sk not in st.session_state:
                    st.session_state[sk] = 0.0
                elif st.session_state[sk] < 0:    # values are non-negative
                    st.session_state[sk] = 0.0
                lbl = f"{name} ({unit})" if unit else name
                disp = col.number_input(lbl, key=sk,
                                        format=fmt, step=0.0, min_value=0.0)
                values[cid] = float(disp) / scale
    return values


def _smith_fig(S: np.ndarray, freq: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr)
    f_ghz = freq * 1e-9
    for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                         ("S21", (1, 0)), ("S12", (0, 1))]:
        sv = S[:, r, c]
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


def _slider_default_range(current_disp):
    """Sane default (min, max, step) for one slider given the current value."""
    if abs(current_disp) < 1e-30:
        return 0.0, 1.0, 0.01
    lo = current_disp * 0.1 if current_disp > 0 else current_disp * 10
    hi = current_disp * 10 if current_disp > 0 else current_disp * 0.1
    d_min, d_max = min(lo, hi), max(lo, hi)
    d_step = max((d_max - d_min) / 100, 1e-9)
    return d_min, d_max, d_step


def _render_tuning_preview(model: CustomModel, freq: np.ndarray,
                           values: dict) -> None:
    """Model-only live slider preview: drag any component value and watch the
    Smith + fT/fmax plots react, then commit the values back into the inputs.
    Mirrors the built-in models' interactive tuning, minus the residual ranking
    (there is no measured device to compare against in forward simulation)."""
    specs = [(cid, kind, name)
             for _title, items in model.grouped_value_specs()
             for cid, kind, name in items]
    label_for = {cid: name for cid, _k, name in specs}

    selected = st.multiselect(
        "Parameters to slide", options=[s[0] for s in specs],
        default=st.session_state.get("cmu_slprev_sel", []),
        format_func=lambda k: label_for.get(k, k), key="cmu_slprev_sel",
        help="Pick component(s) to drag.  The plots below update live; click "
             "✅ Use these values to copy the slider values into the inputs.")
    sel_specs = [s for s in specs if s[0] in selected]
    overrides: dict[str, float] = {}

    for cid, kind, _name in sel_specs:
        scale = _UNITS.get(kind, (1.0, "", "%.4f"))[0]
        cur_disp = float(values.get(cid, 0.0)) * scale
        kp = f"cmu_slprev_{cid}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, d_step = _slider_default_range(cur_disp)
            st.session_state[f"{kp}_min"] = float(d_min)
            st.session_state[f"{kp}_max"] = float(d_max)
            st.session_state[f"{kp}_step"] = float(d_step)

    if sel_specs:
        with st.expander("📏 Slider ranges (min / step / max)", expanded=False):
            for row_start in range(0, len(sel_specs), 2):
                row = sel_specs[row_start:row_start + 2]
                rc = st.columns(len(row))
                for col_w, (cid, kind, name) in zip(rc, row):
                    scale, unit, fmt = _UNITS.get(kind, (1.0, "", "%.4f"))
                    kp = f"cmu_slprev_{cid}"
                    with col_w:
                        st.markdown(f"**{name}** ({unit})" if unit
                                    else f"**{name}**")
                        mc = st.columns(3)
                        mc[0].number_input(f"Min ({unit})" if unit else "Min",
                                           format=fmt, key=f"{kp}_min")
                        mc[1].number_input("Step", format=fmt,
                                           key=f"{kp}_step", min_value=0.0)
                        mc[2].number_input(f"Max ({unit})" if unit else "Max",
                                           format=fmt, key=f"{kp}_max")

    if not sel_specs:
        st.caption("Select one or more parameters above to begin.")
    else:
        slider_cols = st.columns(len(sel_specs))
        for col, (cid, kind, name) in zip(slider_cols, sel_specs):
            scale, unit, fmt = _UNITS.get(kind, (1.0, "", "%.4f"))
            cur_disp = float(values.get(cid, 0.0)) * scale
            kp = f"cmu_slprev_{cid}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            sp = float(st.session_state[f"{kp}_step"])
            if mx <= mn:
                mx = mn + max(sp, abs(mn) * 1e-6 + 1e-9)
            sp_safe = sp if sp > 0 else max((mx - mn) / 100, 1e-12)
            cur_v = min(max(float(st.session_state.get(kp, cur_disp)), mn), mx)
            with col:
                v = st.slider(f"{name} ({unit})" if unit else name,
                              min_value=mn, max_value=mx, step=sp_safe,
                              value=cur_v, format=fmt, key=kp)
                st.caption(f"main: **{cur_disp:.4g}** → preview: "
                           f"**{v:.4g}** {unit}".rstrip())
            overrides[cid] = float(v) / scale

    vals_prev = dict(values)
    vals_prev.update(overrides)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S_prev = simulate_custom_model(model, freq, vals_prev, 50.0)
    except Exception as exc:                                   # noqa: BLE001
        st.error(f"Preview simulation failed: {exc}")
        S_prev = None
    if S_prev is not None and not np.all(np.isfinite(S_prev)):
        st.warning("Preview S-parameters contain non-finite values — adjust "
                   "slider ranges.")
        S_prev = None

    if S_prev is not None:
        col_s, col_b = st.columns([1.05, 1])
        with col_s:
            st.markdown("**Preview Smith chart**")
            st.plotly_chart(_smith_fig(S_prev, freq, model.name),
                            width="stretch", key="cmu_slprev_smith")
        with col_b:
            st.markdown("**Preview fT / fmax**")
            render_forward_bode_block(S_prev, freq, model.name,
                                      key="cmu_slprev_bode")

    if st.button("✅ Use these values", key="cmu_slprev_commit",
                 disabled=(len(overrides) == 0), width="stretch",
                 help="Copy the slider values into the component inputs above."):
        # The cmu_val_* number_inputs were already instantiated this run, so we
        # stage the new values as "_pending" and apply_pending() promotes them
        # before the inputs are built on the next rerun.
        for cid, kind, _name in sel_specs:
            scale = _UNITS.get(kind, (1.0, "", "%.4f"))[0]
            st.session_state[f"cmu_val_{cid}_pending"] = float(overrides[cid]) * scale
        st.rerun()


def render_use_ui() -> None:
    st.subheader("🔬 Use a custom model — forward simulation")
    from .ui_build import fire_pending_download
    fire_pending_download()           # download a model just "sent" from build
    model = _load_source()
    if model is None:
        return

    st.success(f"Loaded **{model.name}** · {model.device} · intrinsic "
               f"{'π' if model.intrinsic_type == 'Pi' else 'T'}")

    cf = st.columns(3)
    f0 = cf[0].number_input("Start (GHz)", min_value=0.0, value=0.01,
                            format="%.4f", step=0.01, key="cmu_f0")
    npts = cf[1].number_input("Points", min_value=2, value=801, step=1,
                              key="cmu_npts")
    f1 = cf[2].number_input("Stop (GHz)", min_value=0.001, value=50.0,
                            format="%.4f", step=1.0, key="cmu_f1")
    if f1 <= f0:
        st.error("Stop frequency must exceed start.")
        return
    freq = np.linspace(float(f0) * 1e9, float(f1) * 1e9, int(npts))

    st.divider()
    values = _value_inputs(model)

    st.divider()
    try:
        S = simulate_custom_model(model, freq, values, z0=50.0)
    except Exception as exc:  # degenerate topology / singular matrix
        st.error(f"Simulation failed: {exc}")
        return

    if not np.all(np.isfinite(S)):
        st.warning("Some S-parameter points are non-finite — check for missing "
                   "values (e.g. a series branch left at 0).")

    # ── Smith (left) + fT/fmax Bode with extrapolation (right) ───────────────
    left, right = st.columns([3, 2])
    with left:
        smith = _smith_fig(S, freq, model.name)
        st.plotly_chart(smith, width="stretch")
        s2p = write_s2p(freq, S, title=f"Custom model {model.name}",
                        params={"intrinsic": model.intrinsic_type})
        sc = st.columns(2)
        sc[0].download_button("📥 .s2p", data=s2p,
                              file_name=f"{model.name or 'custom'}.s2p",
                              mime="text/plain", key="cmu_dl_s2p",
                              width="stretch")
        copy_button(fig_to_tsv(smith) or "", "cmu_copy_smith",
                    container=sc[1], label="📋 copy")
    with right:
        render_forward_bode_block(S, freq, model.name, key="cmu_bode")

    # ── Calculated τ_total and fmax (custom: enter C_BC / R_bb manually) ──────
    render_tau_fmax_expander(
        key="cmu_taufmax", freq=freq, S_meas=S, CBC=0.0, Rbb=0.0,
        tau_sum=None, tau_sum_label="τ", tau_sum_tex=r"\tau",
        extrap_key="cmu_bode")

    # ── Topology illustration — the live schematic (the created picture) ──────
    svg = render_schematic(model, values)
    with st.expander("🖼️ Topology illustration", expanded=True):
        st.iframe(svg, height=svg_pixel_height(svg) + 12)
        png = svg_to_png(svg, zoom=2)
        scc = st.columns(3)
        if png is not None:
            scc[0].download_button("🖼️ Download PNG", data=png,
                                   file_name=f"{model.name or 'custom'}.png",
                                   mime="image/png", key="cmu_dl_png",
                                   width="stretch")
            copy_image_button(png, container=scc[1], label="📋 copy image")
        else:
            scc[0].caption("PNG export needs `rsvg-convert`/`cairosvg`.")
        scc[2].download_button("⬇ Download SVG", data=svg,
                               file_name=f"{model.name or 'custom'}.svg",
                               mime="image/svg+xml", key="cmu_dl_svg",
                               width="stretch")

    # ── Publication matplotlib Smith chart (chart left, controls right) ──────
    with st.expander("🍩 Smith Chart (Matplotlib)", expanded=False):
        c_left, c_right = st.columns([1.2, 1])
        with c_right:
            render_matplotlib_smith(
                fname=_TOPO, topo_key="use",
                sets=[{"S": S, "label": "Simulated",
                       "kind": "line", "style": "solid"}],
                phase="controls", freq_hz=freq)
        with c_left:
            render_matplotlib_smith(
                fname=_TOPO, topo_key="use",
                sets=[{"S": S, "label": "Simulated",
                       "kind": "line", "style": "solid"}],
                phase="chart", freq_hz=freq)

    # ── Interactive tuning (model-only slider preview) ───────────────────────
    with st.expander("🎚️ Tuning — interactive slider preview", expanded=False):
        _render_tuning_preview(model, freq, values)

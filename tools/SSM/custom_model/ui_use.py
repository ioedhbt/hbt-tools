"""
ui_use.py — "Use" workflow for the custom SSM builder.

Load a saved topology (library dropdown or upload), type a value for every
named component, forward-simulate S-parameters with the generic nodal solver,
and inspect the Smith chart + fT/fmax Bode plot.  S2P / Excel downloads mirror
the other portal pages.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from .core import CustomModel, simulate_custom_model, load_model
from .schematic import (render_schematic, svg_to_png, copy_image_button,
                        svg_pixel_height)
from ..helpers import (extended_smith_grid, compute_h21_U, find_ft_fmax,
                       write_s2p, copy_button, fig_to_tsv, fig_to_excel_bytes)

_EXCEL_MIME = ("application/vnd.openxmlformats-officedocument."
               "spreadsheetml.sheet")

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


def _bode_fig(S: np.ndarray, freq: np.ndarray):
    f_ghz = freq * 1e-9
    h21_db, U_db = compute_h21_U(S)
    fT, fmax = find_ft_fmax(f_ghz, h21_db, U_db)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_ghz, y=h21_db, mode="lines",
                             name="|h21|² (dB)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=f_ghz, y=U_db, mode="lines",
                             name="Mason U (dB)", line=dict(color="#d62728")))
    fig.add_hline(y=0, line=dict(color="#888", dash="dot"))
    fig.update_layout(
        title=dict(text="Gain — fT / fmax", font=dict(size=12)),
        xaxis=dict(title="Frequency (GHz)", type="log"),
        yaxis=dict(title="Gain (dB)"),
        plot_bgcolor="white", paper_bgcolor="white", height=420,
        legend=dict(x=0.02, y=0.02), margin=dict(l=50, r=30, t=50, b=50))
    return fig, fT, fmax


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

    # Schematic — value-aware layout so component values get reserved space and
    # never overlap (re-rendered as values change).
    st.markdown("##### Schematic")
    svg = render_schematic(model, values)
    st.iframe(svg, height=svg_pixel_height(svg) + 12)
    png = svg_to_png(svg, zoom=2)
    sc = st.columns(3)
    if png is not None:
        sc[0].download_button("🖼️ Download PNG", data=png,
                              file_name=f"{model.name or 'custom'}.png",
                              mime="image/png", key="cmu_dl_png", width="stretch")
        copy_image_button(png, container=sc[1], label="📋 copy image")
    else:
        sc[0].caption("PNG export needs `rsvg-convert`/`cairosvg` — SVG below.")
    sc[2].download_button("⬇ Download SVG", data=svg,
                          file_name=f"{model.name or 'custom'}.svg",
                          mime="image/svg+xml", key="cmu_dl_svg", width="stretch")

    st.divider()
    try:
        S = simulate_custom_model(model, freq, values, z0=50.0)
    except Exception as exc:  # degenerate topology / singular matrix
        st.error(f"Simulation failed: {exc}")
        return

    if not np.all(np.isfinite(S)):
        st.warning("Some S-parameter points are non-finite — check for missing "
                   "values (e.g. a series branch left at 0).")

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
        bode, fT, fmax = _bode_fig(S, freq)
        st.plotly_chart(bode, width="stretch")
        bc = st.columns(2)
        xlsx = fig_to_excel_bytes(bode)
        if xlsx:
            bc[0].download_button("⬇ xlsx", data=xlsx,
                                  file_name=f"{model.name or 'custom'}_bode.xlsx",
                                  mime=_EXCEL_MIME, key="cmu_dl_bode_xlsx",
                                  width="stretch")
        copy_button(fig_to_tsv(bode) or "", "cmu_copy_bode",
                    container=bc[1], label="📋 copy")
        m = st.columns(2)
        m[0].metric("fT", f"{fT:.2f} GHz" if fT else "—")
        m[1].metric("fmax", f"{fmax:.2f} GHz" if fmax else "—")

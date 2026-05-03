"""
RF_simulator.py — Forward RF S-parameter simulator.

Allows the user to pick an SSM model (Cheng's T / π) or "Open and Short Pad",
key in all extrinsic / intrinsic parameters from scratch, and inspect the
resulting Smith chart and (for SSM models) fT/fmax bode plot.  S2P and Excel
exports are provided for every chart.
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from tools.SSM.models.cheng    import (ChengT, ChengPi,
                                        _render_topology_illustration,
                                        _EXT_T_SPECS, _INT_T_SPECS,
                                        _EXT_PI_SPECS, _INT_PI_SPECS)
from tools.SSM.models.base_ui  import PAD_SPECS
from tools.SSM.ssm_plots       import render_matplotlib_smith
from tools.SSM.helpers         import (extended_smith_grid,
                                        write_s2p, simulate_open, simulate_short,
                                        compute_h21_U, find_ft_fmax,
                                        extrap_20dbdec,
                                        plotly_with_dl, fig_to_excel_bytes)

_EXCEL_MIME = ("application/vnd.openxmlformats-officedocument."
               "spreadsheetml.sheet")


# ─────────────────────────────────────────────────────────────────────────────

st.title("📡 RF Forward Simulator")
st.caption("Forward-simulate S-parameters from a small-signal model "
           "(Cheng T, Cheng π) or from open/short pad parasitics.")


# ─── Frequency axis ──────────────────────────────────────────────────────────

c_f1, c_f2, c_f3 = st.columns(3)
f_start = c_f1.number_input("Start Frequency (GHz)",
                            min_value=0.0, value=0.01,
                            format="%.4f", step=0.01)
n_pts   = c_f2.number_input("Data Points",
                            min_value=2, value=1001, step=1)
f_end   = c_f3.number_input("Final Frequency (GHz)",
                            min_value=0.001, value=50.0,
                            format="%.4f", step=1.0)

if f_end <= f_start:
    st.error("Final frequency must be greater than start frequency.")
    st.stop()

freq  = np.linspace(float(f_start) * 1e9, float(f_end) * 1e9, int(n_pts))
f_ghz = freq * 1e-9


# ─── Model selector ──────────────────────────────────────────────────────────

MODEL_OPTIONS = ["Cheng's T", "Cheng's π", "Open and Short Pad"]
model_choice  = st.radio("Model", MODEL_OPTIONS, horizontal=True, index=0)


# ─── Helpers to render and collect spec lists ────────────────────────────────

def _render_spec_inputs(specs, prefix: str, label: str, cols_per_row: int = 4):
    """Render number_input widgets in rows of ``cols_per_row``.

    Each spec is (key, lbl, sc, unit, fmt, step). Defaults to 0.
    """
    st.markdown(f"**{label}**")
    for row_start in range(0, len(specs), cols_per_row):
        row = specs[row_start:row_start + cols_per_row]
        cs  = st.columns(len(row))
        for col_w, (key, lbl, sc, unit, fmt, step) in zip(cs, row):
            sk = f"rfsim_{prefix}_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                               key=sk, format=fmt, step=step)


def _collect_specs(specs, prefix: str) -> dict:
    """Collect inputs from session_state and convert back to SI units."""
    out = {}
    for key, lbl, sc, unit, fmt, step in specs:
        sk = f"rfsim_{prefix}_{key}"
        out[key] = float(st.session_state.get(sk, 0.0)) / sc
    return out


# ─── Plotly Smith / Bode helpers ─────────────────────────────────────────────

_SMITH_COLORS = {"S11": "#1f77b4", "S22": "#ff7f0e",
                 "S21": "#2ca02c", "S12": "#d62728"}


def _build_smith(S, freq_hz, mults: dict, title: str):
    """``mults`` is a dict ``{"S11":..., "S12":..., "S21":..., "S22":...}``."""
    fig = go.Figure()
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr)
    f_ghz_local = freq_hz * 1e-9
    for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                         ("S21", (1, 0)), ("S12", (0, 1))]:
        col  = _SMITH_COLORS[name]
        m    = float(mults.get(name, 1.0))
        sv   = S[:, r, c] * m
        sc_lbl = "" if abs(m - 1.0) < 1e-9 else (
            f"  ×{m:.3g}" if m >= 1 else f"  ÷{1.0 / m:.3g}")
        hov  = [f"f={fv:.3f} GHz<br>Re={rv:.4f}<br>Im={iv:.4f}"
                for fv, rv, iv in zip(f_ghz_local, sv.real, sv.imag)]
        fig.add_trace(go.Scatter(x=sv.real, y=sv.imag, mode="lines",
                                 name=f"{name}{sc_lbl}",
                                 line=dict(color=col, width=2.0),
                                 text=hov, hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"Smith Chart — {title}", font=dict(size=12)),
        xaxis=dict(title="Re(Γ)", range=[-1.1, 1.1], scaleanchor="y",
                   scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1, 1.1],
                   showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(x=1.02, y=1.0, xanchor="left"),
        hovermode="closest",
    )
    return fig


def _smith_chart_with_dl(fig, key: str, filename: str,
                         s2p_data: bytes, s2p_filename: str):
    """Render a smith chart and put xlsx + s2p download buttons side by side."""
    st.plotly_chart(fig, width="stretch", key=key)
    xl = fig_to_excel_bytes(fig)
    col_xl, col_s2p = st.columns(2)
    if xl is not None:
        col_xl.download_button(
            "⬇ xlsx",
            data=xl,
            file_name=f"{filename}.xlsx",
            mime=_EXCEL_MIME,
            key=f"dl_xl_{key}",
            width="stretch",
        )
    col_s2p.download_button(
        "📥 .s2p",
        data=s2p_data,
        file_name=s2p_filename,
        mime="text/plain",
        key=f"dl_s2p_{key}",
        width="stretch",
    )


def _smith_multiplier_inputs(prefix: str, label: str = "Smith multipliers") -> dict:
    """Render 4 per-trace multipliers (S11/S12/S21/S22) and return a dict."""
    st.markdown(f"**{label}** — × when ≥ 1, ÷ when < 1, per trace")
    cols = st.columns(4)
    out = {}
    for col_w, sp in zip(cols, ("S11", "S12", "S21", "S22")):
        sk = f"rfsim_{prefix}_smithmult_{sp}"
        if sk not in st.session_state:
            st.session_state[sk] = 1.0
        out[sp] = col_w.number_input(f"{sp} ×",
                                     min_value=0.001,
                                     step=0.1, format="%.3f",
                                     key=sk)
    return out


def _build_bode(S, freq_hz, title: str):
    f_ghz_local = freq_hz * 1e-9
    h21_db, U_db = compute_h21_U(S)
    fT, fmax = find_ft_fmax(f_ghz_local, h21_db, U_db)

    fig = go.Figure()
    f_high_track = float(f_ghz_local[-1])
    extrap_used  = False

    def _meas_lbl(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.2f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.2f} GHz (extrap)"
        return f"{name}=n/a"

    def _add_trace(y, base_name, color, dash, kind, in_val):
        """Plot trace + extrapolation; bake the (in-band or extrap) value
        into the legend entry so fT/fmax always show.
        """
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz_local, y)
        ext_val = f0 if f_ext is not None else None
        legend_name = f"{base_name}  [{_meas_lbl(kind, in_val, ext_val)}]"
        fig.add_trace(go.Scatter(x=f_ghz_local, y=y, mode="lines",
                                 name=legend_name,
                                 line=dict(color=color, width=2, dash=dash)))
        if f_ext is not None:
            extrap_used  = True
            f_high_track = max(f_high_track, f0)
            fig.add_trace(go.Scatter(x=f_ext, y=g_ext, mode="lines",
                                     name=f"{legend_name} extrap",
                                     line=dict(color=color, width=2,
                                               dash="dot"),
                                     showlegend=False))

    _add_trace(h21_db, "|h21|²",   "#1f77b4", "solid", "fT",   fT)
    _add_trace(U_db,   "Mason U",  "#d62728", "dash",  "fmax", fmax)

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz_local[0]), 1e-2)
    x_max = (float(f_high_track) * 1.25 if extrap_used
             else float(f_ghz_local[-1]))
    fig.update_layout(
        title=dict(text=f"fT / fmax — {title}", font=dict(size=12)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        legend=dict(orientation="h", x=0.5, y=-0.18,
                    xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=9)),
        hovermode="x unified", margin=dict(l=55, r=20, t=40, b=120),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Branch by model choice
# ═════════════════════════════════════════════════════════════════════════════

# Open/Short use a small subset of the pad/lead specs
_OPEN_SPECS = [
    ("Cpbe", "Cpbe", 1e15, "fF", "%.4f", 0.1),
    ("Cpbc", "Cpbc", 1e15, "fF", "%.4f", 0.01),
    ("Cpce", "Cpce", 1e15, "fF", "%.4f", 0.1),
]
_SHORT_SPECS = [
    ("Lb", "Lb", 1e12, "pH", "%.3f", 0.1),
    ("Lc", "Lc", 1e12, "pH", "%.3f", 0.1),
    ("Le", "Le", 1e12, "pH", "%.3f", 0.01),
]


if model_choice == "Open and Short Pad":
    st.markdown("### Inputs")
    col_in_o, col_in_s = st.columns(2)
    with col_in_o:
        _render_spec_inputs(_OPEN_SPECS, "open",  "Open Pad Capacitances",
                            cols_per_row=3)
    with col_in_s:
        _render_spec_inputs(_SHORT_SPECS, "short", "Short Pad Inductances",
                            cols_per_row=3)

    p_open  = _collect_specs(_OPEN_SPECS, "open")
    p_short = {**_collect_specs(_OPEN_SPECS, "open"),
               **_collect_specs(_SHORT_SPECS, "short"),
               "Rpb": 0.0, "Rpc": 0.0, "Rpe": 0.0}

    try:
        S_open = simulate_open(p_open, freq)
    except Exception as e:
        st.error(f"Open simulation failed: {e}")
        S_open = np.full((len(freq), 2, 2), np.nan + 0j)
    try:
        S_short = simulate_short(p_short, freq)
    except Exception as e:
        st.error(f"Short simulation failed: {e}")
        S_short = np.full((len(freq), 2, 2), np.nan + 0j)

    mults = _smith_multiplier_inputs(
        "os", label="Smith multipliers (apply to both charts)")

    col_chart_o, col_chart_s = st.columns(2)
    with col_chart_o:
        st.markdown("**Open Pad**")
        _smith_chart_with_dl(
            _build_smith(S_open, freq, mults, "Open Pad"),
            key="rfsim_smith_open",
            filename="rfsim_open_smith",
            s2p_data=write_s2p(freq, S_open,
                               title="RF simulator — Open pad",
                               params={k: f"{v:g}" for k, v in p_open.items()
                                       if isinstance(v, (int, float))}),
            s2p_filename="rf_sim_open.s2p",
        )
    with col_chart_s:
        st.markdown("**Short Pad**")
        _smith_chart_with_dl(
            _build_smith(S_short, freq, mults, "Short Pad"),
            key="rfsim_smith_short",
            filename="rfsim_short_smith",
            s2p_data=write_s2p(freq, S_short,
                               title="RF simulator — Short pad",
                               params={k: f"{v:g}" for k, v in p_short.items()
                                       if isinstance(v, (int, float))}),
            s2p_filename="rf_sim_short.s2p",
        )

    with st.expander("📐 Plot Smith chart with matplotlib — Open",
                     expanded=False):
        render_matplotlib_smith(
            fname="rfsim", topo_key="open",
            sets=[{"S": S_open, "label": "Open",
                   "kind": "line", "style": "solid"}],
            default_multiplier=mults,
        )
    with st.expander("📐 Plot Smith chart with matplotlib — Short",
                     expanded=False):
        render_matplotlib_smith(
            fname="rfsim", topo_key="short",
            sets=[{"S": S_short, "label": "Short",
                   "kind": "line", "style": "solid"}],
            default_multiplier=mults,
        )

else:
    # ─── Cheng T / Pi ─────────────────────────────────────────────────────
    if model_choice == "Cheng's T":
        model_cls = ChengT
        ext_specs = _EXT_T_SPECS
        int_specs = _INT_T_SPECS
        topo_char = "T"
        prefix    = "ssm_T"
    else:
        model_cls = ChengPi
        ext_specs = _EXT_PI_SPECS
        int_specs = _INT_PI_SPECS
        topo_char = "pi"
        prefix    = "ssm_pi"

    # Split pad specs by group for the requested layout
    _pad_open_keys  = {"Cpbe", "Cpce", "Cpbc"}
    _pad_short_keys = {"Lb", "Lc", "Le"}
    _pad_r_order    = ["Rpe", "Rpb", "Rpc"]    # Re, Rb, Rc

    pad_open_specs  = [s for s in PAD_SPECS if s[0] in _pad_open_keys]
    pad_short_specs = [s for s in PAD_SPECS if s[0] in _pad_short_keys]
    pad_r_specs     = sorted(
        [s for s in PAD_SPECS if s[0] in _pad_r_order],
        key=lambda s: _pad_r_order.index(s[0]))

    st.markdown("### Inputs")
    with st.expander(f"✏️ {model_cls.NAME} parameters", expanded=True):
        st.markdown("**Pad Parasitics**")
        for col_w, spec in zip(st.columns(3), pad_open_specs):
            key, lbl, sc, unit, fmt, step = spec
            sk = f"rfsim_{prefix}_pad_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})", key=sk,
                               format=fmt, step=step)
        for col_w, spec in zip(st.columns(3), pad_short_specs):
            key, lbl, sc, unit, fmt, step = spec
            sk = f"rfsim_{prefix}_pad_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})", key=sk,
                               format=fmt, step=step)

        st.markdown("**Access Resistance**")
        for col_w, spec in zip(st.columns(3), pad_r_specs):
            key, lbl, sc, unit, fmt, step = spec
            sk = f"rfsim_{prefix}_pad_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})", key=sk,
                               format=fmt, step=step)

        _render_spec_inputs(ext_specs, prefix + "_ext", "Extrinsic Caps")
        _render_spec_inputs(int_specs, prefix + "_int", "Intrinsic")

    p = {**_collect_specs(PAD_SPECS, prefix + "_pad"),
         **_collect_specs(ext_specs, prefix + "_ext"),
         **_collect_specs(int_specs, prefix + "_int")}

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S_sim = model_cls.simulate(p, freq)
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        S_sim = np.full((len(freq), 2, 2), np.nan + 0j)

    if not np.all(np.isfinite(S_sim)):
        st.warning("Simulated S-parameters contain non-finite values "
                   "(some intrinsic parameters are zero or singular). "
                   "Adjust inputs above to see a meaningful trace.")

    mults = _smith_multiplier_inputs(prefix)

    col_smith, col_bode = st.columns(2)
    with col_smith:
        st.markdown(f"**Smith Chart — {model_cls.NAME}**")
        _smith_chart_with_dl(
            _build_smith(S_sim, freq, mults, model_cls.NAME),
            key=f"rfsim_smith_{prefix}",
            filename=f"rfsim_smith_{prefix}",
            s2p_data=write_s2p(freq, S_sim,
                               title=f"RF simulator — {model_cls.NAME}",
                               params={k: f"{v:g}" for k, v in p.items()
                                       if isinstance(v, (int, float))}),
            s2p_filename=f"rf_sim_{prefix}.s2p",
        )
    with col_bode:
        st.markdown(f"**fT / fmax — {model_cls.NAME}**")
        plotly_with_dl(_build_bode(S_sim, freq, model_cls.NAME),
                       key=f"rfsim_bode_{prefix}",
                       filename=f"rfsim_bode_{prefix}")

    with st.expander("🖼️ Topology Illustration", expanded=False):
        try:
            _render_topology_illustration(p, topo_char, f"rfsim_{prefix}")
        except Exception as e:
            st.warning(f"Topology illustration unavailable: {e}")

    with st.expander("📐 Plot Smith chart with matplotlib", expanded=False):
        render_matplotlib_smith(
            fname=f"rfsim_{prefix}", topo_key=topo_char,
            sets=[{"S": S_sim, "label": "Simulated",
                   "kind": "line", "style": "solid"}],
            default_multiplier=mults,
        )

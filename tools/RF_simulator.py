"""
RF_simulator.py — Forward RF S-parameter simulator.

Allows the user to pick an SSM model (Cheng's T / π) or "Open and Short Pad",
key in all extrinsic / intrinsic parameters from scratch, and inspect the
resulting Smith chart and (for SSM models) fT/fmax bode plot.  S2P and Excel
exports are provided for every chart.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "1.1"

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from tools.SSM.models.cheng    import (ChengT, ChengPi,
                                        _render_topology_illustration,
                                        _EXT_T_SPECS, _INT_T_SPECS,
                                        _EXT_PI_SPECS, _INT_PI_SPECS)
from tools.SSM.models.xu       import (XuModel,
                                        _render_topology_illustration as _render_xu_illustration,
                                        _EXT_T_SPECS as _XU_EXT_SPECS,
                                        _INT_T_SPECS as _XU_INT_SPECS,
                                        _XU_PAD_SPECS)
from tools.SSM.models.kunyang  import (KunYangHEMT,
                                        _render_topology_illustration as _render_ky_illustration,
                                        _EXT_KY_SPECS, _INT_KY_SPECS,
                                        _KY_PAD_SPECS, _DEFAULT_PARAMS as _KY_DEFAULT_PARAMS)
from tools.SSM.models.base_ui  import PAD_SPECS
from tools.SSM.ssm_plots       import (render_matplotlib_smith,
                                        render_tau_fmax_expander)
from tools.SSM.helpers         import (extended_smith_grid,
                                        write_s2p, simulate_open, simulate_short,
                                        compute_h21_U, find_ft_fmax,
                                        extrap_20dbdec, single_pole_extrap,
                                        FT_FMAX_SYMBOLS, FT_FMAX_COLORS,
                                        plotly_with_dl, fig_to_excel_bytes,
                                        bode_excel_bytes,
                                        fig_to_tsv, copy_button,
                                        make_smith_bode_slider_fig)

try:
    import cupy as _cp
    _HAS_CUDA = True
    _v = _cp.cuda.runtime.runtimeGetVersion()
    _CUDA_VER = f"{_v // 1000}.{(_v % 1000) // 10}"
except Exception:
    _cp = None  # type: ignore[assignment]
    _HAS_CUDA = False
    _CUDA_VER = ""

_EXCEL_MIME = ("application/vnd.openxmlformats-officedocument."
               "spreadsheetml.sheet")


# ─────────────────────────────────────────────────────────────────────────────

st.title(f"📡 RF Forward Simulator (v{__version__})")
st.caption("Forward-simulate S-parameters from a small-signal model "
           "(Cheng T, Cheng π, Xu T, Kun-Yang HEMT) or from open/short pad parasitics.")


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

MODEL_OPTIONS = ["Cheng's T", "Cheng's π", "Xu T", "Kun-Yang HEMT", "Open and Short Pad"]
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
    tsv = fig_to_tsv(fig)
    col_xl, col_copy, col_s2p = st.columns(3)
    if xl is not None:
        col_xl.download_button(
            "⬇ xlsx",
            data=xl,
            file_name=f"{filename}.xlsx",
            mime=_EXCEL_MIME,
            key=f"dl_xl_{key}",
            width="stretch",
        )
    if tsv:
        copy_button(tsv, key=key, container=col_copy)
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


def _render_slider_preview(model_cls, all_p, freq, mults, prefix: str,
                           pad_specs, ext_specs, int_specs):
    """RF simulator slider preview block — mode-toggled.

    🐢 Live: drag any number of sliders, every tick reruns Streamlit + sim.
    ⚡ Plotly: pre-compute N frames for one swept param, scrub client-side.
    """
    mode_key = f"rfsim_slpreview_mode_{prefix}"
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
        _render_rfsim_plotly_slider_preview(model_cls, all_p, freq, prefix,
                                             pad_specs, ext_specs, int_specs)
    else:
        _render_rfsim_live_slider_preview(model_cls, all_p, freq, mults, prefix,
                                           pad_specs, ext_specs, int_specs)


_FRAGMENT = (getattr(st, "fragment", None)
             or getattr(st, "experimental_fragment", None)
             or (lambda f: f))


def _slider_default_range(current_disp):
    if abs(current_disp) < 1e-30:
        return -1.0, 1.0, 0.01
    lo = current_disp * 0.1 if current_disp > 0 else current_disp * 10
    hi = current_disp * 10  if current_disp > 0 else current_disp * 0.1
    d_min, d_max = min(lo, hi), max(lo, hi)
    d_step = max((d_max - d_min) / 100, 1e-9)
    return d_min, d_max, d_step


@_FRAGMENT
def _render_rfsim_live_slider_preview(model_cls, all_p, freq, mults, prefix: str,
                                       pad_specs, ext_specs, int_specs):
    """Streamlit-rerun-per-drag implementation.

    Layout
    ------
    [ multiselect of params                                                  ]
    [ one column per selected param — slider + delta caption                 ]
    [ ▼ Slider ranges (min / step / max) expander, each row has 3 columns   ]
    [ Smith chart  |  fT/fmax bode  (side by side)                          ]
    [ ✅ Use these values | ↩️ Reset preview                                 ]
    """
    cat_of: dict[str, str] = {}
    for s in pad_specs: cat_of[s[0]] = "pad"
    for s in ext_specs: cat_of[s[0]] = "ext"
    for s in int_specs: cat_of[s[0]] = "int"
    tuning_specs = list(pad_specs) + list(ext_specs) + list(int_specs)
    label_for = {s[0]: s[1] for s in tuning_specs}

    sel_key  = f"rfsim_slpreview_sel_{prefix}"
    selected = st.multiselect(
        "Parameters to slide", options=[s[0] for s in tuning_specs],
        default=st.session_state.get(sel_key, []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        help="Pick parameter(s) to drag.  Plots below show the slider-"
             "substituted model in real time; the main Smith / fT-fmax "
             "plots above stay frozen until you click ✅ Use these values.")

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    preview_overrides: dict[str, float] = {}

    for spec in selected_specs:
        key, _, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"rfsim_slpreview_{prefix}_{key}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, d_step = _slider_default_range(current_disp)
            st.session_state[f"{kp}_min"]  = float(d_min)
            st.session_state[f"{kp}_max"]  = float(d_max)
            st.session_state[f"{kp}_step"] = float(d_step)
        if kp not in st.session_state:
            st.session_state[kp] = float(current_disp)

    if selected_specs:
        with st.expander("📏 Slider ranges (min / step / max)", expanded=False):
            for row_start in range(0, len(selected_specs), 2):
                row_specs = selected_specs[row_start:row_start + 2]
                row_cols  = st.columns(len(row_specs))
                for col_w, spec in zip(row_cols, row_specs):
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    fmt  = spec[4] if len(spec) > 4 else "%.4g"
                    kp   = f"rfsim_slpreview_{prefix}_{key}"
                    with col_w:
                        st.markdown(f"**{label}** ({unit})" if unit
                                    else f"**{label}**")
                        mc = st.columns(3)
                        mc[0].number_input(f"Min ({unit})" if unit else "Min",
                                           format=fmt, key=f"{kp}_min")
                        mc[1].number_input("Step", format=fmt,
                                           key=f"{kp}_step", min_value=0.0)
                        mc[2].number_input(f"Max ({unit})" if unit else "Max",
                                           format=fmt, key=f"{kp}_max")

    if not selected_specs:
        st.caption("Select one or more parameters above to begin.")
    else:
        slider_cols = st.columns(len(selected_specs))
        for col, spec in zip(slider_cols, selected_specs):
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            current_disp = float(all_p.get(key, 0.0)) * scale
            kp = f"rfsim_slpreview_{prefix}_{key}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            sp = float(st.session_state[f"{kp}_step"])
            if mx <= mn:
                mx = mn + max(sp, abs(mn) * 1e-6 + 1e-9)
            sp_safe = sp if sp > 0 else max((mx - mn) / 100, 1e-12)
            cur_v = min(max(float(st.session_state.get(kp, current_disp)),
                            mn), mx)
            with col:
                v = st.slider(f"{label} ({unit})" if unit else label,
                              min_value=mn, max_value=mx, step=sp_safe,
                              value=cur_v, format=fmt, key=kp)
                st.caption(f"main: **{current_disp:.4g}**  →  "
                           f"preview: **{v:.4g}** {unit}".rstrip())
            preview_overrides[key] = float(v) / scale

    # ── Preview plots: Smith | Bode side by side ──────────────────────
    all_p_prev = dict(all_p)
    all_p_prev.update(preview_overrides)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S_prev = model_cls.simulate(all_p_prev, freq)
    except Exception as e:
        st.error(f"Preview simulation failed: {e}")
        S_prev = None
    if S_prev is not None and not np.all(np.isfinite(S_prev)):
        st.warning("Preview S-parameters contain non-finite values — "
                   "adjust slider ranges.")
        S_prev = None
    if S_prev is not None:
        col_s, col_b = st.columns([1.05, 1])
        with col_s:
            st.markdown("**Preview Smith chart**")
            st.plotly_chart(_build_smith(S_prev, freq, mults, model_cls.NAME),
                            width="stretch",
                            key=f"rfsim_slpreview_smith_{prefix}")
        with col_b:
            st.markdown("**Preview fT / fmax**")
            st.plotly_chart(_build_bode(S_prev, freq, model_cls.NAME)[0],
                            width="stretch",
                            key=f"rfsim_slpreview_bode_{prefix}")

    bc1, bc2 = st.columns(2)
    commit_clicked = bc1.button(
        "✅ Use these values",
        key=f"rfsim_slpreview_commit_{prefix}",
        disabled=(len(preview_overrides) == 0),
        help="Copy slider values into the fine-tune number_inputs above.",
        width="stretch")
    reset_clicked = bc2.button(
        "↩️ Reset preview",
        key=f"rfsim_slpreview_reset_{prefix}",
        help="Discard slider drags and clear remembered min/step/max.",
        width="stretch")

    if commit_clicked:
        for k, v_si in preview_overrides.items():
            sc  = next(s[2] for s in tuning_specs if s[0] == k)
            cat = cat_of.get(k)
            if cat is None:
                continue
            st.session_state[f"rfsim_{prefix}_{cat}_{k}"] = float(v_si) * sc
        st.rerun()

    if reset_clicked:
        for s in tuning_specs:
            kp = f"rfsim_slpreview_{prefix}_{s[0]}"
            for suf in ("", "_min", "_step", "_max"):
                st.session_state.pop(kp + suf, None)
        st.rerun()


@_FRAGMENT
def _render_rfsim_plotly_slider_preview(model_cls, all_p, freq, prefix: str,
                                         pad_specs, ext_specs, int_specs):
    """Pre-computed Plotly slider — joint (cartesian) multi-param scan.

    Same UX as the SSM version: each selected param gets its own slider,
    frames are the full cartesian product, JS coordinates the sliders so
    dragging one reflects the *current position* of all the others.
    """
    from tools.SSM.helpers import make_smith_bode_joint_slider_html

    tuning_specs = list(pad_specs) + list(ext_specs) + list(int_specs)
    label_for = {s[0]: s[1] for s in tuning_specs}
    options   = [s[0] for s in tuning_specs]

    sel_key  = f"rfsim_slprev_pl_sel_{prefix}"
    selected = st.multiselect(
        "Sweep parameters",
        options=options,
        default=st.session_state.get(sel_key, [options[0]] if options else []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        help="Each selected param gets its own Plotly slider in the figure. "
             "Frames are the FULL cartesian product — dragging one slider "
             "reflects the model at the current position of every other "
             "slider.")

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    if not selected_specs:
        st.caption("Select one or more parameters above and click "
                   "**🧮 Build animation**.")
        return

    for spec in selected_specs:
        key, _, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"rfsim_slprev_pl_{prefix}_{key}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, _ = _slider_default_range(current_disp)
            st.session_state[f"{kp}_min"]    = float(d_min)
            st.session_state[f"{kp}_max"]    = float(d_max)
            st.session_state[f"{kp}_frames"] = 11

    with st.expander("📏 Slider ranges (min / max / frames)", expanded=False):
        for row_start in range(0, len(selected_specs), 2):
            row_specs = selected_specs[row_start:row_start + 2]
            row_cols  = st.columns(len(row_specs))
            for col_w, spec in zip(row_cols, row_specs):
                key, label, scale = spec[0], spec[1], spec[2]
                unit = spec[3] if len(spec) > 3 else ""
                fmt  = spec[4] if len(spec) > 4 else "%.4g"
                kp   = f"rfsim_slprev_pl_{prefix}_{key}"
                with col_w:
                    st.markdown(f"**{label}** ({unit})" if unit
                                else f"**{label}**")
                    mmf = st.columns(3)
                    mmf[0].number_input(f"Min ({unit})" if unit else "Min",
                                        format=fmt, key=f"{kp}_min")
                    mmf[1].number_input(f"Max ({unit})" if unit else "Max",
                                        format=fmt, key=f"{kp}_max")
                    mmf[2].number_input("Frames", min_value=2, max_value=100,
                                        step=1, key=f"{kp}_frames",
                                        help="Frames per axis (2–100). "
                                             "Total = product across params.")

    n_freq_full = int(len(freq))
    decim_key   = f"rfsim_slprev_pl_decim_{prefix}"
    if decim_key not in st.session_state:
        st.session_state[decim_key] = min(120, n_freq_full)

    dims_preview = []
    for spec in selected_specs:
        kp = f"rfsim_slprev_pl_{prefix}_{spec[0]}"
        dims_preview.append(int(st.session_state.get(f"{kp}_frames", 11)))
    total_frames = int(np.prod(dims_preview)) if dims_preview else 0

    decim_n = int(st.session_state.get(decim_key, min(120, n_freq_full)))
    decim_n = min(decim_n, n_freq_full)
    est_mb  = total_frames * decim_n * 10 * 7 / 1024 / 1024

    fd_col1, fd_col2 = st.columns([1, 2])
    with fd_col1:
        st.number_input(f"Freq points (max: {n_freq_full})",
                        min_value=20, max_value=n_freq_full, step=10,
                        key=decim_key,
                        help=f"Frequency points kept per trace "
                             f"(max = {n_freq_full} = full fidelity).")
    with fd_col2:
        st.caption("Cartesian sweep: "
                   + " × ".join(str(d) for d in dims_preview)
                   + f" = **{total_frames}** frames · {decim_n} freq pts · "
                   f"estimated payload ≈ **{est_mb:.0f} MB**")
    sc_key = f"rfsim_slprev_pl_servercached_{prefix}"
    server_cached = st.checkbox(
        "📡 Server-cached mode (Streamlit sliders, unlimited sweep size, "
        "slower drag)",
        value=st.session_state.get(sc_key, False), key=sc_key,
        help="OFF: embed all frames in the browser (fast scrub, capped by "
             "Streamlit's 200 MB message limit).  ON: pre-computed frames "
             "stay in server RAM and only the current frame is sent per "
             "slider tick (~100-300 ms per drag, no size limit).")
    if server_cached:
        ram_mb = total_frames * n_freq_full * 8 * 4 / 1024 / 1024
        st.caption(f"Server RAM estimate (complex64, full-fidelity batch): "
                   f"≈ **{ram_mb:.0f} MB** in session_state.")
        if ram_mb > 8000:
            st.warning(f"⚠️ ~{ram_mb/1024:.1f} GB server-side may OOM on "
                       "modest machines.  Reduce per-axis frame counts.")
    else:
        if est_mb > 180:
            st.error(
                f"❌ Estimated payload ≈ {est_mb:.0f} MB will exceed "
                "Streamlit's 200 MB browser-message limit.  Enable "
                "**📡 Server-cached mode**, lower the **Freq points** "
                "value, reduce per-axis frame counts, or raise the limit "
                "via `.streamlit/config.toml` → "
                "`[server] maxMessageSize = 500`.")
        elif est_mb > 120:
            st.warning(f"⚠️ Estimated payload ≈ {est_mb:.0f} MB is close "
                       "to Streamlit's 200 MB limit.")
        elif total_frames > 2000:
            st.warning(f"⚠️ {total_frames} frames may stutter on "
                       "slider drag.")

    cuda_toggle_key = f"rfsim_slprev_pl_cuda_{prefix}"
    if _HAS_CUDA:
        cuda_col, btn_col = st.columns([1.6, 1])
        with cuda_col:
            use_cuda = st.checkbox(f"⚡ Use CUDA (cupy {_CUDA_VER}) for "
                                    "batched simulation",
                                    value=st.session_state.get(cuda_toggle_key, True),
                                    key=cuda_toggle_key,
                                    help="Off-load joint cartesian batched "
                                         "simulation to GPU.  Result is "
                                         "brought back to host as fp64.")
        with btn_col:
            build_clicked = st.button("🧮 Build animation",
                                      key=f"rfsim_slprev_pl_build_{prefix}",
                                      width="stretch",
                                      help="Pre-compute the cartesian joint "
                                           "sweep and embed with JS-"
                                           "coordinated multi-sliders.")
    else:
        use_cuda = False
        build_clicked = st.button("🧮 Build animation",
                                  key=f"rfsim_slprev_pl_build_{prefix}",
                                  width="stretch",
                                  help="Pre-compute the cartesian joint "
                                       "sweep and embed with JS-coordinated "
                                       "multi-sliders.")

    state_key = f"rfsim_slprev_pl_state_{prefix}"

    if build_clicked:
        import time as _time
        xp = _cp if (use_cuda and _HAS_CUDA) else np
        device_label = (f"GPU (cupy {_CUDA_VER})"
                        if xp is not np else "CPU (numpy)")
        sweep_disps : list[np.ndarray] = []
        sweep_sis   : list[np.ndarray] = []
        slider_specs_out: list[dict] = []
        for spec in selected_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            kp   = f"rfsim_slprev_pl_{prefix}_{key}"
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

        from tools.SSM.models.base_ui import _chunked_simulate_batch_to_host
        t0 = _time.perf_counter()
        with st.spinner(f"Computing {n_total} frames on {device_label}…"):
            p_batch = dict(all_p)
            for spec, flat in zip(selected_specs, flats):
                p_batch[spec[0]] = xp.asarray(flat, dtype=float)
            try:
                S_b = _chunked_simulate_batch_to_host(
                    model_cls, p_batch, freq, 50.0, xp=xp)
            except Exception as e:
                st.error(f"Batched preview simulation failed: {e}")
                return
        elapsed = _time.perf_counter() - t0

        if not np.all(np.isfinite(S_b)):
            st.warning("Some frames contain non-finite S-parameters — "
                       "narrow the ranges to avoid singular combinations.")

        st.session_state[state_key] = {
            "slider_specs":  slider_specs_out,
            "S_batch":       S_b,
            "selected_keys": [s[0] for s in selected_specs],
            "elapsed_s":     elapsed,
            "device":        device_label,
            "n_total":       n_total,
        }

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

    elapsed = float(state.get("elapsed_s", 0.0))
    n_total = int(state.get("n_total", 0)) or len(state["S_batch"])
    device  = str(state.get("device", "?"))
    ms_each = (elapsed / max(1, n_total)) * 1000.0
    st.caption(f"✅ Built **{n_total}** frames on **{device}** in "
               f"**{elapsed:.2f} s** ({ms_each:.1f} ms/frame).  "
               "Drag any slider below to scrub the joint sweep.")

    if server_cached:
        from tools.SSM.models.base_ui import _render_plotly_server_cached_view
        _render_plotly_server_cached_view(
            state, None, freq, model_cls, "rfsim", prefix,
            decim_n_max=int(st.session_state.get(decim_key, 120)))
    else:
        html = make_smith_bode_joint_slider_html(
            S_batch_joint=state["S_batch"],
            freq=freq,
            slider_specs=state["slider_specs"],
            model_name=model_cls.NAME,
            S_meas=None,
            decimate_points=int(st.session_state.get(decim_key, 120)),
        )
        n_sl = len(state["slider_specs"])
        iframe_height = 500 + 26 + 36 * n_sl + 30
        # st.iframe replaced components.v1.html (deprecated 2026-06-01).
        # When src is a raw HTML string (no http(s) / file / Path prefix)
        # Streamlit embeds it directly in an iframe — same behaviour as
        # the old components.html call.  No `scrolling` parameter; the
        # `height=` integer is interpreted in pixels just like before.
        st.iframe(html, height=iframe_height)


def _build_bode(S, freq_hz, title: str, *,
                extrap_method: str = "−20 dB/dec", sp_window=None):
    """Build the fT/fmax Bode figure.

    Returns ``(fig, any_needs)`` where ``any_needs`` is True when at least one
    trace still has positive gain at the top of the band (i.e. extrapolation is
    required to project a 0-dB crossing).  ``extrap_method`` selects the
    projection: ``"−20 dB/dec"`` (slope-locked) or ``"Single-pole"`` (log-linear
    least-squares fit over ``sp_window=(f_lo, f_hi)`` in GHz).
    """
    f_ghz_local = freq_hz * 1e-9
    h21_db, U_db = compute_h21_U(S)
    fT, fmax = find_ft_fmax(f_ghz_local, h21_db, U_db)

    def _needs(in_val, gain):
        if in_val is not None:
            return False
        with np.errstate(invalid="ignore"):
            return bool(np.nanmax(gain) > 0)
    any_needs = _needs(fT, h21_db) or _needs(fmax, U_db)

    def _extrap(y):
        if (extrap_method == "Single-pole" and sp_window is not None
                and len(f_ghz_local) >= 4):
            il = int(np.searchsorted(f_ghz_local, sp_window[0], side="left"))
            ih = int(np.searchsorted(f_ghz_local, sp_window[1], side="right")) - 1
            il = max(0, min(il, len(f_ghz_local) - 2))
            ih = max(il + 1, min(ih, len(f_ghz_local) - 1))
            r = single_pole_extrap(f_ghz_local, y, il, ih)
            return r[0], r[1], r[2]
        return extrap_20dbdec(f_ghz_local, y)

    fig = go.Figure()
    f_high_track = float(f_ghz_local[-1])
    extrap_used  = False
    # Collected for the standardised fT/fmax xlsx export (simulated +
    # extrapolated columns).  Labels match across the two lists so the
    # workbook reads "<trace>" / "<trace> (extrap)".
    sim_traces:    list[tuple[str, np.ndarray]] = []
    extrap_traces: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _meas_lbl(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.2f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.2f} GHz (extrap)"
        return f"{name}=n/a"

    def _add_trace(y, base_name, color, dash, kind, in_val, symbol):
        """Plot trace + extrapolation; bake the (in-band or extrap) value
        into the legend entry so fT/fmax always show.
        """
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = _extrap(y)
        ext_val = f0 if f_ext is not None else None
        legend_name = f"{base_name}  [{_meas_lbl(kind, in_val, ext_val)}]"
        fig.add_trace(go.Scatter(x=f_ghz_local, y=y, mode="lines+markers",
                                 name=legend_name,
                                 line=dict(color=color, width=2, dash=dash),
                                 marker=dict(symbol=symbol, size=6, color=color)))
        sim_traces.append((f"{base_name} (dB)", np.asarray(y)))
        if f_ext is not None:
            extrap_used  = True
            f_high_track = max(f_high_track, f0)
            extrap_traces.append((f"{base_name} (dB)", f_ext, g_ext))
            fig.add_trace(go.Scatter(x=f_ext, y=g_ext, mode="lines",
                                     name=f"{legend_name} extrap",
                                     line=dict(color=color, width=2,
                                               dash="dot"),
                                     showlegend=False))

    # Standard colour scheme (FT_FMAX_COLORS): fT in blue, fmax in red.
    # Both traces are "measured" sims (no model comparison in this view),
    # so both are solid; extrap fall-throughs in `_add_trace` are dotted.
    _add_trace(h21_db, "|h21|²",  FT_FMAX_COLORS["fT"],   "solid", "fT",   fT,   FT_FMAX_SYMBOLS["h21"])
    _add_trace(U_db,   "Mason U", FT_FMAX_COLORS["fmax"], "solid", "fmax", fmax, FT_FMAX_SYMBOLS["U"])

    bode_xl = bode_excel_bytes(f_ghz_local, sim_traces, extrap_traces)

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
        # Legend pinned bottom-left INSIDE the plot area (paper coords,
        # anchored bottom-left) instead of below the chart.
        legend=dict(orientation="v", x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=13)),
        hovermode="x unified", margin=dict(l=55, r=20, t=40, b=50),
    )
    return fig, any_needs, bode_xl


def _render_bode_block(S, freq_hz, title: str, key: str):
    """Render the fT/fmax Bode plot, then (if extrapolation is needed) an
    extrapolation-method radio (left) and single-pole window slider (right)
    UNDERNEATH the chart.  The figure reads the current selection from
    session_state, so a Streamlit rerun on widget change feeds it back here.
    """
    f_ghz_local = freq_hz * 1e-9
    method = st.session_state.get(f"{key}_extrap_method", "−20 dB/dec")
    sp_window = None
    if method == "Single-pole" and len(f_ghz_local) >= 4:
        f_lo, f_hi = float(f_ghz_local[0]), float(f_ghz_local[-1])
        sp_window = st.session_state.get(f"{key}_sp_window",
                                         (max(f_lo, f_hi - 5.0), f_hi))
    fig, any_needs, bode_xl = _build_bode(S, freq_hz, title,
                                          extrap_method=method, sp_window=sp_window)
    plotly_with_dl(fig, key=key, filename=key, excel_bytes=bode_xl)

    if any_needs and len(f_ghz_local) >= 2:
        ec1, ec2 = st.columns([1, 2])
        if f"{key}_extrap_method" not in st.session_state:
            st.session_state[f"{key}_extrap_method"] = "−20 dB/dec"
        ec1.radio(
            "Extrap. method", ["−20 dB/dec", "Single-pole"],
            key=f"{key}_extrap_method", horizontal=True,
            help="−20 dB/dec anchors a slope-locked line at the last data "
                 "point.  Single-pole fits a log-linear line over the chosen "
                 "window (default = final 5 GHz).")
        if (st.session_state[f"{key}_extrap_method"] == "Single-pole"
                and len(f_ghz_local) >= 4):
            f_lo, f_hi = float(f_ghz_local[0]), float(f_ghz_local[-1])
            sp_default = (max(f_lo, f_hi - 5.0), f_hi)
            ec2.slider(
                "Single-pole fit window (GHz)",
                min_value=f_lo, max_value=f_hi,
                value=st.session_state.get(f"{key}_sp_window", sp_default),
                step=max((f_hi - f_lo) / 400.0, 1e-3),
                key=f"{key}_sp_window")


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
    # ─── Cheng T / Pi / Xu T ──────────────────────────────────────────────
    if model_choice == "Cheng's T":
        model_cls = ChengT
        ext_specs = _EXT_T_SPECS
        int_specs = _INT_T_SPECS
        topo_char = "T"
        prefix    = "ssm_T"
    elif model_choice == "Xu T":
        model_cls = XuModel
        ext_specs = _XU_EXT_SPECS
        int_specs = _XU_INT_SPECS
        topo_char = "T"
        prefix    = "ssm_XuT"
        # Rbcx defaults to 285 kΩ — pre-init so default sim doesn't see Rbcx=0 → Ybcx=∞
        _rbcx_sk = f"rfsim_{prefix}_ext_Rbcx"
        if _rbcx_sk not in st.session_state:
            st.session_state[_rbcx_sk] = 285.0
    elif model_choice == "Kun-Yang HEMT":
        model_cls = KunYangHEMT
        ext_specs = _EXT_KY_SPECS
        int_specs = _INT_KY_SPECS
        topo_char = "pi"
        prefix    = "ssm_KY"
        # Pre-init the intrinsic + custom-pad inputs with the model defaults so
        # the first render produces a finite Smith chart (Rds=0 would explode).
        for _key, _, _sc, *_ in _EXT_KY_SPECS + _INT_KY_SPECS:
            _sk = (f"rfsim_{prefix}_ext_{_key}"
                   if _key in {k for k, *_ in _EXT_KY_SPECS}
                   else f"rfsim_{prefix}_int_{_key}")
            if _sk not in st.session_state:
                st.session_state[_sk] = float(
                    _KY_DEFAULT_PARAMS.get(_key, 0.0)) * float(_sc)
    else:
        model_cls = ChengPi
        ext_specs = _EXT_PI_SPECS
        int_specs = _INT_PI_SPECS
        topo_char = "pi"
        prefix    = "ssm_pi"

    # Split pad specs by group for the requested layout.  Xu uses its own
    # pad-label aliases (Rb→Rbx, Re→Rex, Cpce→Cpad) but identical keys.
    _pad_open_keys  = {"Cpbe", "Cpce", "Cpbc"}
    _pad_short_keys = {"Lb", "Lc", "Le"}
    _pad_r_order    = ["Rpe", "Rpb", "Rpc"]    # Re, Rb, Rc

    if model_cls is XuModel:
        _pad_specs_for_model = _XU_PAD_SPECS
    elif model_cls is KunYangHEMT:
        _pad_specs_for_model = _KY_PAD_SPECS
    else:
        _pad_specs_for_model = PAD_SPECS
    pad_open_specs  = [s for s in _pad_specs_for_model if s[0] in _pad_open_keys]
    pad_short_specs = [s for s in _pad_specs_for_model if s[0] in _pad_short_keys]
    pad_r_specs     = sorted(
        [s for s in _pad_specs_for_model if s[0] in _pad_r_order],
        key=lambda s: _pad_r_order.index(s[0]))

    def _render_pad_row(specs):
        for col_w, spec in zip(st.columns(3), specs):
            key, lbl, sc, unit, fmt, step = spec
            sk = f"rfsim_{prefix}_pad_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})", key=sk,
                               format=fmt, step=step)

    st.markdown("### Inputs")
    with st.expander(f"✏️ {model_cls.NAME} parameters", expanded=True):
        if model_cls is KunYangHEMT:
            # KY: substrate / custom pad FIRST (these caps ARE the pad layer),
            # then access resistance + lead inductances.  No Cpg / Cpd / Cpgd
            # row — those entries are not used by the Kun-Yang model.
            _render_spec_inputs(ext_specs, prefix + "_ext",
                                 "Kun-Yang Custom Pad / Substrate Network")
            st.markdown("**Access Resistance & Lead Inductance**")
            _render_pad_row(pad_short_specs)
            _render_pad_row(pad_r_specs)
            _render_spec_inputs(int_specs, prefix + "_int", "Intrinsic Pi-Model")
        else:
            st.markdown("**Pad Parasitics**")
            _render_pad_row(pad_open_specs)
            _render_pad_row(pad_short_specs)
            st.markdown("**Access Resistance**")
            _render_pad_row(pad_r_specs)
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
        _render_bode_block(S_sim, freq, model_cls.NAME,
                           key=f"rfsim_bode_{prefix}")

    # τ_total + calculated fmax expander — every SSM model except Kun-Yang
    # HEMT (which lacks Cbcx/Cbc/Rbi/Rb for the fmax formula).
    if model_cls is not KunYangHEMT:
        _CBC = float(p.get("Cbcx", 0.0)) + float(p.get("Cbc", 0.0))
        _Rbb = float(p.get("Rbi", 0.0)) + float(p.get("Rpb", 0.0))
        if model_cls is ChengPi:
            _tau_sum = float(p.get("tau", 0.0))
            _tau_lbl, _tau_tex = "τ", r"\tau"
        else:
            _tau_sum = float(p.get("tauB", 0.0)) + float(p.get("tauC", 0.0))
            _tau_lbl, _tau_tex = "τB + τC", r"\tau_B+\tau_C"
        render_tau_fmax_expander(key=f"rfsim_taufmax_{prefix}", freq=freq,
                                 S_meas=S_sim, CBC=_CBC, Rbb=_Rbb,
                                 tau_sum=_tau_sum, tau_sum_label=_tau_lbl,
                                 tau_sum_tex=_tau_tex,
                                 extrap_key=f"rfsim_bode_{prefix}")

    from tools.SSM.ssm_plots import render_matplotlib_smith

    # Topology illustration and matplotlib Smith chart go in their own
    # expanders so users can collapse each independently — mirrors how
    # the SSM extraction tab keeps these on separate axes.
    with st.expander("🖼️ Topology Illustration", expanded=False):
        try:
            if model_cls is XuModel:
                _render_xu_illustration(p, f"rfsim_{prefix}")
            elif model_cls is KunYangHEMT:
                _render_ky_illustration(p, f"rfsim_{prefix}")
            else:
                _render_topology_illustration(p, topo_char,
                                               f"rfsim_{prefix}")
        except Exception as e:
            st.warning(f"Topology illustration unavailable: {e}")

    with st.expander("🍩 Smith Chart (Matplotlib)", expanded=False):
        # Controls on the right column, chart on the left — same
        # split-call pattern the SSM tab uses inside its expander.
        col_mpl_left, col_mpl_right = st.columns([1.2, 1])
        with col_mpl_right:
            render_matplotlib_smith(
                fname=f"rfsim_{prefix}", topo_key=topo_char,
                sets=[{"S": S_sim, "label": "Simulated",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults,
                phase="controls", freq_hz=freq,
            )
        with col_mpl_left:
            render_matplotlib_smith(
                fname=f"rfsim_{prefix}", topo_key=topo_char,
                sets=[{"S": S_sim, "label": "Simulated",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults,
                phase="chart", freq_hz=freq,
            )

    with st.expander("🔧 Tuning — Interactive slider preview", expanded=False):
        _render_slider_preview(model_cls, p, freq, mults, prefix,
                                _pad_specs_for_model, ext_specs, int_specs)

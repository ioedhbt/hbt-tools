"""
ssm_plots.py — Step 1 diagnostic plots and helper visualisations.

All functions render directly into Streamlit and return any UI-state values
needed by the caller (e.g. extended element modes, Cpar values).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from .helpers          import (open_elem_Y, s_to_y, y_to_z,
                                peel_parasitics, simulate_open,
                                plotly_with_dl, bode_excel_bytes, info_icon_html,
                                compute_h21_U, find_ft_fmax, extrap_20dbdec,
                                single_pole_extrap, segmented_radio,
                                FT_FMAX_SYMBOLS, FT_FMAX_COLORS)
from .models.base_ui   import render_smith_chart, ssm_residual
from ..i18n             import tr


# ── Open element mode constants ───────────────────────────────────────────────

OPEN_MODES      = ["None", "Parallel L", "Series L", "Series R"]
_MODE_UNIT      = {"None": None, "Parallel L": "pH", "Series L": "pH", "Series R": "Ω"}
_MODE_SCALE     = {"None": 1,    "Parallel L": 1e12, "Series L": 1e12, "Series R": 1.0}
# Display-only translations for OPEN_MODES — the option VALUE stored in
# session_state / compared elsewhere (`mode != "None"`, `"L" in mode`, the
# _MODE_UNIT / _MODE_SCALE lookups) must stay the English id, so this is
# only ever used through `format_func`.
_OPEN_MODE_ZH   = {"None": "無", "Parallel L": "並聯電感", "Series L": "串聯電感",
                   "Series R": "串聯電阻"}

# Display-only translation for the "−20 dB/dec" / "Single-pole" extrapolation
# method radio, which appears in several places below and whose VALUE is
# compared against the literal English string (`extrap_method == "Single-pole"`)
# well after the widget is drawn — so only the label goes through tr(), never
# the stored value.  "−20 dB/dec" is left as-is (a slope spec, not prose).
_EXTRAP_METHOD_ZH = {"Single-pole": "單極擬合"}


def _extrap_method_label(v: str) -> str:
    return tr(v, _EXTRAP_METHOD_ZH.get(v, v))


# ════════════════════════════════════════════════════════════════════════════════
# Open dummy plots
# ════════════════════════════════════════════════════════════════════════════════

def render_open_plots(open_data, para_caps, open_arr, fname=""):
    """
    Render four expanders for Open dummy diagnostics:
      1. Extra-element model controls (Parallel L / Series L / Series R per cap)
      2. Capacitance Im(Y)/ω vs frequency  [0–50 fF, fixed]
      3. Conductance Re(Y) vs frequency    [series R indicator]
      4. Im(Y)/ω vs 1/ω²                  [Parallel L linearisation]
      5. Smith chart: measured vs forward-simulated Open

    Returns
    -------
    dict  {cap: (mode_str, extra_SI)}  — propagated into para_eff and forward sims.
    """
    f_o, S_o, z0_o = open_data
    f_ghz = f_o * 1e-9
    omega  = 2.0*np.pi*f_o

    # ── 1. Per-cap extra element controls ─────────────────────────────────────
    with st.expander(tr("🔧 Open Pad Element Model — extra parasitic options",
                         "🔧 Open Pad 元件模型 — 額外寄生選項"), expanded=False):
        st.markdown(
            tr("Each pad capacitor can include one secondary parasitic element.  \n"
               "**Parallel L** = inductor ∥ C → resonance at 1/√LC, affects Im(Y)/ω.  \n"
               "**Series L**   = inductor in series with C → increases effective C near resonance.  \n"
               "**Series R**   = resistor in series with C → adds Re(Y) that rises then saturates.  \n"
               "These choices propagate into de-embedding, Cold-HBT correction, and S2P downloads.",
               "每個 pad 電容都可加入一個次要寄生元件。  \n"
               "**並聯電感** = 電感 ∥ C → 於 1/√LC 處產生共振，影響 Im(Y)/ω。  \n"
               "**串聯電感** = 電感與 C 串聯 → 使共振附近的等效 C 增大。  \n"
               "**串聯電阻** = 電阻與 C 串聯 → 使 Re(Y) 先上升後飽和。  \n"
               "這些選擇會傳遞至去嵌入、冷 HBT 修正與 S2P 下載。")
        )
        for cap, cap_lbl in [("Cpbe","Cpbe (B-E)"), ("Cpce","Cpce (C-E)"), ("Cpbc","Cpbc (B-C)")]:
            st.markdown(f"**{cap_lbl}**")
            c1, c2 = st.columns([2, 1])
            mode_sk  = f"open_mode_{cap}_{fname}"
            extra_sk = f"open_extra_{cap}_{fname}"
            if mode_sk  not in st.session_state: st.session_state[mode_sk]  = "None"
            if extra_sk not in st.session_state: st.session_state[extra_sk] = 0.0
            c1.radio(tr(f"Open-element mode for {cap}", f"{cap} 的 Open 元件模式"),
                     OPEN_MODES,
                     horizontal=True, key=mode_sk,
                     format_func=lambda m: tr(m, _OPEN_MODE_ZH.get(m, m)),
                     label_visibility="collapsed")
            mode = st.session_state[mode_sk]
            if mode != "None":
                unit = _MODE_UNIT[mode]
                c2.number_input(tr(f"Extra {unit}", f"額外 {unit}"), min_value=0.0,
                                step=0.1 if unit == "pH" else 0.01,
                                format="%.3f" if unit == "pH" else "%.4f",
                                key=extra_sk)

    def _get_mode_extra(cap):
        """Return (mode_str, extra_in_SI) for one cap."""
        mode  = st.session_state.get(f"open_mode_{cap}_{fname}", "None")
        extra_disp = float(st.session_state.get(f"open_extra_{cap}_{fname}", 0.0))
        sc    = _MODE_SCALE.get(mode, 1)
        return mode, extra_disp / sc

    # ── 2. Capacitance plot ───────────────────────────────────────────────────
    with st.expander(tr("📊 Open — Pad Capacitances vs Frequency",
                         "📊 Open — Pad 電容 vs 頻率"), expanded=True):
        fig_cap = go.Figure()
        for key, lbl, col in [("Cpbe","Cpbe","#1f77b4"),
                               ("Cpce","Cpce","#ff7f0e"),
                               ("Cpbc","Cpbc","#2ca02c")]:
            arr_fF = open_arr[key] * 1e15
            val_fF = para_caps[key] * 1e15
            mode, extra = _get_mode_extra(key)
            # Measured trace
            fig_cap.add_trace(go.Scattergl(x=f_ghz, y=arr_fF,
                name=f"{lbl} (meas.)", line=dict(color=col, width=2), mode="lines"))
            # Median dashed line
            fig_cap.add_trace(go.Scattergl(x=[f_ghz[0], f_ghz[-1]], y=[val_fF, val_fF],
                name=f"{lbl}={val_fF:.3f} fF", line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
            # Modelled effective C overlay (when extra element chosen)
            if mode != "None":
                # See ssm_core.open_elem_Y for the formula
                Y_mod_arr = np.array([open_elem_Y(para_caps[key], mode, extra, w) for w in omega])
                Ceff_fF   = np.imag(Y_mod_arr) / omega * 1e15
                fig_cap.add_trace(go.Scattergl(x=f_ghz, y=Ceff_fF,
                    name=f"{lbl} model ({mode})", line=dict(color=col, width=2, dash="dot"), mode="lines"))
        fig_cap.update_layout(title="Pad Capacitances — Im(Y)/ω",
            xaxis_title="Frequency (GHz)", yaxis_title="Cap (fF)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=18)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_cap.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_cap.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 50])
        plotly_with_dl(fig_cap, key=f"step1_cap_{fname}", filename=f"open_cap_{fname}")
        st.caption(tr("Flat line = pure C. Slope/resonance = inductive effect. Range fixed 0–50 fF.",
                      "水平線 = 純電容。斜率／共振 = 電感效應。範圍固定為 0–50 fF。"))

    # ── 3. Conductance plot (Series R indicator) ──────────────────────────────
    with st.expander(tr("📊 Open — Pad Conductance vs Frequency (Re(Y) — series R indicator)",
                         "📊 Open — Pad 電導 vs 頻率（Re(Y) — 串聯電阻指標）"),
                     expanded=False):
        fig_g = go.Figure()
        for key, lbl, col in [("Cpbe","Gpbe","#1f77b4"),
                               ("Cpce","Gpce","#ff7f0e"),
                               ("Cpbc","Gpbc","#2ca02c")]:
            arr_mS = open_arr[f"G{key[1:]}"] * 1e3   # Gpbe/Gpce/Gpbc keys
            mode, extra = _get_mode_extra(key)
            fig_g.add_trace(go.Scattergl(x=f_ghz, y=arr_mS,
                name=f"{lbl} (meas.)", line=dict(color=col, width=2), mode="lines"))
            if mode == "Series R" and extra > 0:
                # Re[Y_series_R] = ω²RC² / (1+ω²R²C²)  → see ssm_core.open_elem_Y
                G_mod = np.array([np.real(open_elem_Y(para_caps[key], mode, extra, w))*1e3
                                  for w in omega])
                fig_g.add_trace(go.Scattergl(x=f_ghz, y=G_mod,
                    name=f"{lbl} model (R={extra:.3f} Ω)",
                    line=dict(color=col, width=2, dash="dot"), mode="lines"))
        fig_g.add_hline(y=0, line_color="#aaa", line_width=1)
        fig_g.update_layout(
            title="Pad Conductance Re(Y) — nonzero = series R or parallel G loss",
            xaxis_title="Frequency (GHz)", yaxis_title="Conductance (mS)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=18)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_g.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_g.update_yaxes(showgrid=True, gridcolor="#ebebeb")
        plotly_with_dl(fig_g, key=f"step1_cond_{fname}", filename=f"open_conductance_{fname}")
        st.caption(tr(
            "Pure C → Re(Y)=0.  "
            "Series R → Re(Y) = ω²RC² / (1+ω²R²C²) — rises then saturates.  \n"
            "Select 'Series R' above to overlay the modelled curve.",
            "純電容 → Re(Y)=0。  "
            "串聯電阻 → Re(Y) = ω²RC² / (1+ω²R²C²) — 先上升後飽和。  \n"
            "於上方選擇「串聯電阻」以疊加模型曲線。"))

    # ── 4. Im(Y)/ω vs 1/ω² (Parallel L linearisation) ───────────────────────
    with st.expander(tr("📊 Open — Im(Y)/ω vs 1/ω²  (Parallel L linearisation)",
                         "📊 Open — Im(Y)/ω vs 1/ω²（並聯電感線性化）"), expanded=False):
        fig_l = go.Figure()
        one_over_omega2 = 1.0 / (omega**2 + 1e-60)
        for key, lbl, col in [("Cpbe","Cpbe","#1f77b4"),
                               ("Cpce","Cpce","#ff7f0e"),
                               ("Cpbc","Cpbc","#2ca02c")]:
            Ceff = open_arr[key]  # Im(Y)/ω already stored per-frequency
            fig_l.add_trace(go.Scattergl(
                x=one_over_omega2*1e-18, y=Ceff*1e15, name=lbl,
                line=dict(color=col, width=2), mode="lines",
                hovertemplate="1/ω²=%{x:.4f}×10¹⁸<br>Im(Y)/ω=%{y:.3f} fF<extra></extra>"))
        fig_l.update_layout(
            title="Im(Y)/ω vs 1/ω² — slope = −1/L if Parallel L present",
            xaxis_title="1/ω² (× 10¹⁸ rad⁻²s²)", yaxis_title="Im(Y)/ω  (fF equivalent)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=18)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_l.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_l.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 50])
        plotly_with_dl(fig_l, key=f"step1_lind_{fname}", filename=f"open_lind_{fname}")
        st.caption(tr(
            "Parallel L model: Im(Y)/ω = C − 1/(ω²L).  "
            "A straight line with negative slope → L = −1/slope (SI).",
            "並聯電感模型：Im(Y)/ω = C − 1/(ω²L)。  "
            "斜率為負的直線 → L = −1/斜率（SI 單位）。"))

    # ── 5. Smith chart: measured vs modelled Open ─────────────────────────────
    with st.expander(tr("📡 Open — Measured vs Modelled Smith Chart",
                         "📡 Open — 量測 vs 模型 Smith 圖"), expanded=False):
        _p_open = {}
        for cap in ["Cpbe", "Cpce", "Cpbc"]:
            mode, extra = _get_mode_extra(cap)
            _p_open[cap]            = para_caps[cap]
            _p_open[f"{cap}_mode"]  = mode
            _p_open[f"{cap}_extra"] = extra
        # Forward-simulate open (ssm_s2p.simulate_open)
        S_open_sim = simulate_open(_p_open, f_o, z0_o)
        err_open   = ssm_residual(S_o, S_open_sim)
        render_smith_chart(S_o, S_open_sim,
                           f"Open dummy",
                           err_open,
                           scales={"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0},
                           key=f"smith_open_{fname}")
        st.caption(tr("Adjust the extra element controls above — the modelled curve updates live.",
                      "調整上方的額外元件控制項 — 模型曲線會即時更新。"))

    return {cap: _get_mode_extra(cap) for cap in ["Cpbe", "Cpce", "Cpbc"]}


# ════════════════════════════════════════════════════════════════════════════════
# Short dummy plots
# ════════════════════════════════════════════════════════════════════════════════

def render_short_plots(short_arr, para_short, fname="", freq=None):
    """
    Render Lead inductances vs frequency [0–150 pH, fixed] for Short dummy diagnostics.

    Parameters
    ----------
    short_arr  : dict from step_short — must contain Lb, Lc, Le arrays.
    para_short : dict of scalar fitted values (used as horizontal reference lines).
    fname      : Streamlit widget-key suffix.
    freq       : Hz array matching the Lb/Lc/Le sample positions. When provided
                 the x-axis is plotted in GHz (downloads as L vs freq, not point
                 index). When None the legacy point-index axis is used.
    """

    with st.expander(tr("📊 Short — Lead Inductances vs Frequency",
                         "📊 Short — 引線電感 vs 頻率"), expanded=True):
        fig_ind = go.Figure(); any_neg = False
        for key, lbl, col in [("Lb","Lb","#8e44ad"),
                               ("Lc","Lc","#e67e22"),
                               ("Le","Le","#16a085")]:
            arr_pH = short_arr[key] * 1e12
            val_pH = para_short[key] * 1e12
            if val_pH < 0: any_neg = True
            if freq is not None:
                xv = np.asarray(freq) * 1e-9
                hov = (f"{lbl}=%{{y:.4f}} pH<br>"
                       f"f=%{{x:.3f}} GHz<extra></extra>")
            else:
                xv = np.arange(len(arr_pH))
                hov = f"{lbl}=%{{y:.4f}} pH<extra></extra>"
            fig_ind.add_trace(go.Scattergl(x=xv, y=arr_pH,
                name=f"{lbl} (per-freq)",
                line=dict(color=col, width=2), mode="lines",
                hovertemplate=hov))
            fig_ind.add_trace(go.Scattergl(x=[xv[0], xv[-1]], y=[val_pH, val_pH],
                name=f"{lbl}={val_pH:.2f} pH",
                line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
        fig_ind.add_hline(y=0, line_color="#333", line_width=1.2,
                           annotation_text="0 pH", annotation_position="left",
                           annotation_font=dict(size=9, color="#333"))
        x_title = "Frequency (GHz)" if freq is not None else "Point index"
        fig_ind.update_layout(title="Lead Inductances",
            xaxis_title=x_title, yaxis_title="Inductance (pH)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=18)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_ind.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_ind.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 150])
        plotly_with_dl(fig_ind, key=f"step1_ind_{fname}", filename=f"short_inductances_{fname}")
        if any_neg:
            st.warning(tr("One or more lead inductances are negative. Use Short Override to correct.",
                          "一個或多個引線電感為負值。請使用 Short Override 修正。"))
        st.caption(tr("Range fixed 0–150 pH.", "範圍固定為 0–150 pH。"))



# ════════════════════════════════════════════════════════════════════════════════
# Helper: S-parameter comparison plot
# ════════════════════════════════════════════════════════════════════════════════

def _rlc_params_summary(p, fname):
    """Build header param dict for a de-embedded S2P file."""
    d = {"DUT_file": Path(fname).stem}
    for k in ["Cpbe","Cpce","Cpbc"]:
        d[k] = f"{p.get(k,0)*1e15:.4f} fF"
        mode = p.get(f"{k}_mode","None")
        if mode != "None":
            extra = p.get(f"{k}_extra",0.0)
            unit_e = "pH" if "L" in mode else "Ω"
            sc_e   = 1e12 if "L" in mode else 1.0
            d[f"{k}_extra"] = f"{mode}: {extra*sc_e:.4f} {unit_e}"
    for k, lbl, sc_v, unit in [("Lb","Lb",1e12,"pH"),("Lc","Lc",1e12,"pH"),
                                ("Le","Le",1e12,"pH"),("Rpb","Rb",1,"Ω"),
                                ("Rpc","Rc",1,"Ω"),("Rpe","Re",1,"Ω")]:
        d[lbl] = f"{p.get(k,0)*sc_v:.4f} {unit}"
    for ck in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        v = p.get(ck,0.0)
        if v > 0: d[ck] = f"{v*1e15:.4f} fF"
    return d


def _compare_bode_smith(*, S_a, S_b, freq, fname, key_suffix,
                        label_a, label_b, color_a, color_b,
                        smith_meas_label, smith_sim_label, gain_title,
                        extra_download_fn=None):
    """Render a side-by-side bode (h21² + Mason U) and Smith chart comparing
    two S-parameter datasets.  Used by both the OS-deembedded and intrinsic
    preview sections.

    ``extra_download_fn`` (optional) is called as ``fn(fT_b, fmax_b)`` once the
    right-hand trace's fT/fmax are known, and must return a
    ``(label, data_bytes, file_name, mime)`` tuple.  It is forwarded to the
    Smith chart's ``plotly_with_dl`` so the download sits beside that chart's
    ⬇ xlsx button (used to expose the S2P download next to the Smith chart)."""
    from .models.base_ui import smith_scale_controls

    f_ghz = freq * 1e-9
    h21_a, U_a = compute_h21_U(S_a)
    h21_b, U_b = compute_h21_U(S_b)

    fT_a, fmax_a = find_ft_fmax(f_ghz, h21_a, U_a)
    fT_b, fmax_b = find_ft_fmax(f_ghz, h21_b, U_b)

    def _meas_lbl(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.2f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.2f} GHz (extrap)"
        return f"{name}=n/a"

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used = False
    # Collected for the standardised fT/fmax xlsx export (simulated +
    # extrapolated columns).  Labels match across the two lists.
    sim_traces:    list[tuple[str, np.ndarray]] = []
    extrap_traces: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Standardised colours (FT_FMAX_COLORS):  fT trace (h21) = blue, fmax
    # trace (U) = red.  The two datasets A vs B differ by **line dash**
    # rather than colour, so the eye can compare the same metric across
    # transformations (Raw vs De-embedded, OS-deembedded vs Intrinsic).
    #   A : solid          (treated as the baseline / "before" trace)
    #   B : dashed         (the transformed / "after" trace)
    #   extrap : dotted    (in the metric's colour, no legend)
    # ``color_a`` / ``color_b`` kwargs are kept in the signature for
    # backwards compatibility but are now ignored.
    _c_fT, _c_fmax = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]
    del color_a, color_b  # explicitly drop — they no longer drive trace colour

    def _add(y_arr, name_fmt, color, symbol, kind, in_val, dl_label, dash=None):
        """Plot one trace and append the (in-band or extrap) value to its
        legend name.  ``name_fmt`` must include a ``{lbl}`` placeholder.
        ``dl_label`` is the stable column header used in the xlsx export.
        """
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, y_arr)
        ext_val = f0 if f_ext is not None else None
        legend_name = name_fmt.format(lbl=_meas_lbl(kind, in_val, ext_val))
        line_kw = dict(color=color, width=1.4)
        if dash:
            line_kw["dash"] = dash
        fig.add_trace(go.Scattergl(
            x=f_ghz, y=y_arr, mode="lines+markers", name=legend_name,
            line=line_kw,
            marker=dict(symbol=symbol, size=6, color=color)))
        sim_traces.append((dl_label, y_arr))
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            extrap_traces.append((dl_label, f_ext, g_ext))
            fig.add_trace(go.Scattergl(
                x=f_ext, y=g_ext, mode="lines", name=f"{legend_name} extrap",
                line=dict(color=color, width=1.6, dash="dot"),
                showlegend=False))

    _add(h21_a, f"|h21|² {label_a}  [{{lbl}}]", _c_fT,   FT_FMAX_SYMBOLS["h21"], "fT",   fT_a,   f"|h21|² {label_a} (dB)")
    _add(U_a,   f"Mason U {label_a}  [{{lbl}}]", _c_fmax, FT_FMAX_SYMBOLS["U"],   "fmax", fmax_a, f"Mason U {label_a} (dB)")
    _add(h21_b, f"|h21|² {label_b}  [{{lbl}}]", _c_fT,   FT_FMAX_SYMBOLS["h21"], "fT",   fT_b,   f"|h21|² {label_b} (dB)", dash="dash")
    _add(U_b,   f"Mason U {label_b}  [{{lbl}}]", _c_fmax, FT_FMAX_SYMBOLS["U"],   "fmax", fmax_b, f"Mason U {label_b} (dB)", dash="dash")

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    # fT/fmax values are reported in trace legend entries above; vertical
    # markers on the bode plot are intentionally omitted.

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1])
    fig.update_layout(
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        legend=dict(orientation="h", x=0.5, y=-0.22,
                    xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=18)),
        hovermode="x unified", margin=dict(l=55, r=20, t=40, b=220))

    extrap_note = (tr("   Dotted = 20 dB/dec extrapolation past the measured band.",
                      "　點線 = 超出量測頻段的 20 dB/dec 外插。")
                   if extrap_used else "")
    st.markdown(
        tr(f"○ = |h21|² (→ fT).   □ = Mason U (→ fmax).   Y-axis fixed 0–50 dB.{extrap_note}",
           f"○ = |h21|²（→ fT）。   □ = Mason U（→ fmax）。   Y 軸固定為 0–50 dB。{extrap_note}"))

    sc = smith_scale_controls(fname, key_suffix)
    col_gain, col_smith = st.columns([1, 1])
    with col_gain:
        st.markdown(f"**{gain_title}**")
        plotly_with_dl(fig, key=f"bode_{key_suffix}_{fname}",
                       filename=f"bode_{key_suffix}_{fname}",
                       excel_bytes=bode_excel_bytes(f_ghz, sim_traces, extrap_traces))
    with col_smith:
        st.markdown(tr(f"**S-Parameters: {smith_meas_label} vs {smith_sim_label}**",
                       f"**S 參數：{smith_meas_label} vs {smith_sim_label}**"))
        err = ssm_residual(S_a, S_b)
        _extra_dl = (extra_download_fn(fT_b, fmax_b)
                     if extra_download_fn is not None else None)
        render_smith_chart(S_a, S_b, f"{smith_meas_label} vs {smith_sim_label}",
                           err, sc, key=f"smith_{key_suffix}_{fname}",
                           show_title=False,
                           meas_label=smith_meas_label,
                           sim_label=smith_sim_label,
                           extra_download=_extra_dl)

    return fT_b, fmax_b   # return the right-hand trace's fT/fmax for download header


def render_os_deemb_preview(S_raw, freq, z0, para_step1, fname):
    """
    Step 2 — De-embedded Preview.

    Shows raw vs Open/Short de-embedded (pad caps + lead L,R from Steps 1a/1b
    only; no access-resistance correction).  Provides bode+smith comparison
    and a download of the OS de-embedded S2P file.
    """
    from .helpers import y_to_s_batch, write_s2p

    # st.markdown(
    #     "<div style='background:linear-gradient(90deg,#e8f5e9 0%,transparent 100%);"
    #     "border-left:4px solid #2e7d32;padding:8px 14px;border-radius:0 6px 6px 0;"
    #     "margin-bottom:2px'><strong>📊 Raw vs Open/Short De-embedded</strong></div>",
    #     unsafe_allow_html=True)

    Y_step1 = peel_parasitics(S_raw, freq, z0, para_step1)
    S_step1 = y_to_s_batch(Y_step1, z0)

    with st.expander(tr("📐 Open/Short de-embedding formulas",
                         "📐 Open/Short 去嵌入公式"), expanded=False):
        st.markdown(
            tr("**Open + Short de-embedding chain** *(Gao §4.2)*  \n"
               "Applied at every frequency point independently.",
               "**Open + Short 去嵌入鏈** *(Gao §4.2)*  \n"
               "各頻率點皆獨立套用。"))
        st.markdown(tr("**Step 1 — Open (parallel pad subtraction):**  \n"
                       "S → Y,  then Y − Y_open:",
                       "**步驟 1 — Open（並聯 pad 扣除）：**  \n"
                       "S → Y，再 Y − Y_open："))
        st.latex(
            r"Y_1 = Y_{DUT} - Y_{pad},\quad "
            r"Y_{pad}=\begin{bmatrix}Y_{pbe}+Y_{pbc}&-Y_{pbc}\\-Y_{pbc}&Y_{pce}+Y_{pbc}\end{bmatrix}")
        st.markdown(tr("where $Y_{pXX} = j\\omega C_{pXX}$ (extended: Parallel L, Series L, or Series R).",
                       "其中 $Y_{pXX} = j\\omega C_{pXX}$（擴充：並聯電感、串聯電感或串聯電阻）。"))
        st.markdown(tr("**Step 2 — Short (series lead subtraction):**  \n"
                       "Y → Z,  then Z − Z_short,  then Z → Y:",
                       "**步驟 2 — Short（串聯引線扣除）：**  \n"
                       "Y → Z，再 Z − Z_short，再 Z → Y："))
        st.latex(
            r"Z_2 = Z_1 - Z_{ser},\quad "
            r"Z_{ser}=\begin{bmatrix}Z_b+Z_e&Z_e\\Z_e&Z_c+Z_e\end{bmatrix}")
        st.markdown(tr(
            r"where $Z_b = R_b + j\omega L_b$, etc. (Rb/Rc/Re here come from the Short dummy only — "
            r"access-resistance correction is applied later in Step 3).",
            r"其中 $Z_b = R_b + j\omega L_b$，其餘依此類推（此處的 Rb/Rc/Re 僅來自 Short dummy — "
            r"存取電阻修正將於步驟 3 套用）。"))

    with st.expander(tr("📊 Plots", "📊 圖表"), expanded=False):
        st.markdown(
            tr("**Raw** = measured DUT (no de-embedding).  \n"
               "**OS de-embedded** = Open+Short parasitics removed using Step 1a/1b "
               "extracted values (Cpbe/Cpce/Cpbc + Lb/Lc/Le + short-dummy Rs).",
               "**Raw** = 量測 DUT（未去嵌入）。  \n"
               "**OS de-embedded** = 以步驟 1a/1b 萃取值（Cpbe/Cpce/Cpbc + Lb/Lc/Le + "
               "short dummy Rs）移除 Open+Short 寄生。"))

        # OS de-embedded S2P download — sits beside the Smith chart's
        # ⬇ xlsx button (built from the right-hand trace's fT/fmax).
        def _os_extra_dl(fT_b, fmax_b):
            p_hdr = _rlc_params_summary(para_step1, fname)
            p_hdr["fT"]   = f"{fT_b:.3f} GHz"   if fT_b   is not None else "n/a"
            p_hdr["fmax"] = f"{fmax_b:.3f} GHz" if fmax_b is not None else "n/a"
            return ("📥 OSdeembedded_*.s2p",
                    write_s2p(freq, S_step1,
                              title=f"OS de-embedded — {Path(fname).stem}",
                              params=p_hdr),
                    f"OSdeembedded_{Path(fname).stem}.s2p",
                    "text/plain")

        fT_os, fmax_os = _compare_bode_smith(
            S_a=S_raw, S_b=S_step1, freq=freq, fname=fname, key_suffix="osdeemb",
            label_a="Raw", label_b="OS de-embedded",
            color_a="#2ca02c", color_b="#1f77b4",
            smith_meas_label="Raw", smith_sim_label="OS de-embedded",
            gain_title=tr("Gain vs Frequency — Raw vs OS De-embedded",
                          "增益 vs 頻率 — Raw vs OS De-embedded"),
            extra_download_fn=_os_extra_dl)

    return S_step1


def render_intrinsic_preview(S_raw, freq, z0, para_step1, para_eff, fname,
                              S_step1=None):
    """
    Step 3 footer — Intrinsic preview.

    Shows Open/Short de-embedded (Step 2) vs Intrinsic (Open/Short + access
    resistance Rb/Rc/Re removed via para_eff).  Provides bode+smith
    comparison and a download of the intrinsic S2P file.
    """
    from .helpers import y_to_s_batch, write_s2p

    # st.markdown(
    #     "<div style='background:linear-gradient(90deg,#0d737722 0%,transparent 100%);"
    #     "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
    #     "margin-bottom:2px'><strong>📊 OS De-embedded vs Intrinsic</strong>"
    #     f"{info_icon_html('OS de-embedded = Open/Short calibration only (Step 2). Intrinsic = OS calibration + access-resistance Rb/Rc/Re removal using the values selected in Pre-Extraction Review above.')}"
    #     "</div>",
    #     unsafe_allow_html=True)

    if S_step1 is None:
        Y_step1 = peel_parasitics(S_raw, freq, z0, para_step1)
        S_step1 = y_to_s_batch(Y_step1, z0)

    Y_pareff = peel_parasitics(S_raw, freq, z0, para_eff)
    S_pareff = y_to_s_batch(Y_pareff, z0)

    with st.expander(tr("📊 Plots", "📊 圖表"), expanded=False):
        # Intrinsic S2P download — sits beside the Smith chart's ⬇ xlsx button
        # (built from the right-hand trace's fT/fmax).
        def _intrinsic_extra_dl(fT_b, fmax_b):
            p_hdr = _rlc_params_summary(para_eff, fname)
            p_hdr["fT"]   = f"{fT_b:.3f} GHz"   if fT_b   is not None else "n/a"
            p_hdr["fmax"] = f"{fmax_b:.3f} GHz" if fmax_b is not None else "n/a"
            return ("📥 intrinsic_*.s2p",
                    write_s2p(freq, S_pareff,
                              title=f"Intrinsic — {Path(fname).stem}",
                              params=p_hdr),
                    f"intrinsic_{Path(fname).stem}.s2p",
                    "text/plain")

        fT_in, fmax_in = _compare_bode_smith(
            S_a=S_step1, S_b=S_pareff, freq=freq, fname=fname, key_suffix="intrinsic",
            label_a="OS de-embedded", label_b="Intrinsic",
            color_a="#1f77b4", color_b="#e67e22",
            smith_meas_label="OS de-embedded", smith_sim_label="Intrinsic",
            gain_title=tr("Gain vs Frequency — OS De-embedded vs Intrinsic",
                          "增益 vs 頻率 — OS De-embedded vs Intrinsic"),
            extra_download_fn=_intrinsic_extra_dl)


# ════════════════════════════════════════════════════════════════════════════════
# Helper: fT / fmax Bode plot
# ════════════════════════════════════════════════════════════════════════════════

# compute_h21_U, find_ft_fmax, extrap_20dbdec moved to tools/SSM/helpers/metrics.py
# and imported at the top of this file via `from .helpers import …`.


def render_ft_fmax_card(S_mea, S_sim, freq, *, model_name: str,
                        key: str, height: int = 560,
                        compact: bool = False):
    """
    Compact four-trace fT/fmax mini-plot (|h21|² and Mason U, measured + modeled).

    Standardised colour / dash convention
    -------------------------------------
      |h21|²  → fT   → blue   (`FT_FMAX_COLORS["fT"]`)
      Mason U → fmax → red    (`FT_FMAX_COLORS["fmax"]`)
      measured     : solid + markers (circle for h21, square for U)
      modeled      : dashed (no markers)
      extrap (20dB
        / single-pole) : dotted (no markers)

    Legend layout
    -------------
    Vertical legend pinned to the bottom-left INSIDE the plot area (rather
    than below the chart).  A vertical stack is used instead of a 2x2 grid so
    the long "Meas. (fT=… GHz)" entries don't overlap in the narrow in-plot box.

    Extrapolation
    -------------
    When at least one of the four traces still has gain > 0 dB at the
    highest measured frequency, a Streamlit radio UNDERNEATH the chart lets
    the user pick the extrapolation method:
      • "−20 dB/dec"  — slope-locked line anchored at the last data point.
      • "Single-pole" — log-linear (least-squares) fit over a user-chosen
                        frequency window (slider, default = final 5 GHz),
                        placed to the right of the radio.
    Both render as dotted lines in the metric's colour.

    ``compact=True`` is retained for API compatibility (callers still pass it)
    but no longer changes the margin now that the legend sits inside the plot.
    """
    f_ghz = np.asarray(freq) * 1e-9
    h21_m, U_m = compute_h21_U(S_mea)
    h21_s, U_s = compute_h21_U(S_sim)

    # In-band 0-dB crossings (None if the trace doesn't cross within the band).
    fT_m_in,  fmax_m_in  = find_ft_fmax(f_ghz, h21_m, U_m)
    fT_s_in,  fmax_s_in  = find_ft_fmax(f_ghz, h21_s, U_s)

    # Does any trace need extrapolation?  Per-trace: only true when the
    # in-band crossing is None AND the trace's max gain is positive.
    def _needs_extrap(in_val, gain_arr) -> bool:
        if in_val is not None:
            return False
        with np.errstate(invalid="ignore"):
            return bool(np.nanmax(gain_arr) > 0)
    any_needs = any(_needs_extrap(v, g) for v, g in (
        (fT_m_in, h21_m), (fmax_m_in, U_m),
        (fT_s_in, h21_s), (fmax_s_in, U_s)))

    # ── Extrapolation method — READ the current selection from session_state
    #    so the figure can be built BEFORE the radio/slider are drawn.  Those
    #    widgets live UNDERNEATH the chart (rendered after plotly_with_dl), so
    #    a Streamlit rerun on change feeds the new value back up here.
    #    Defaults: −20 dB/dec; single-pole window = final 5 GHz.
    extrap_method = "−20 dB/dec"
    sp_window: tuple[float, float] | None = None
    if any_needs and len(f_ghz) >= 2:
        extrap_method = st.session_state.get(f"{key}_extrap_method",
                                             "−20 dB/dec")
        if extrap_method == "Single-pole" and len(f_ghz) >= 4:
            f_lo, f_hi = float(f_ghz[0]), float(f_ghz[-1])
            sp_default = (max(f_lo, f_hi - 5.0), f_hi)
            sp_window = st.session_state.get(f"{key}_sp_window", sp_default)

    def _do_extrap(gain_db):
        """Return ``(f_ext, g_ext, f0)`` per the selected method, or
        ``(None, None, None)`` when the trace already crosses in band."""
        if extrap_method == "Single-pole" and sp_window is not None and len(f_ghz) >= 4:
            il = int(np.searchsorted(f_ghz, sp_window[0], side="left"))
            ih = int(np.searchsorted(f_ghz, sp_window[1], side="right")) - 1
            il = max(0, min(il, len(f_ghz) - 2))
            ih = max(il + 1, min(ih, len(f_ghz) - 1))
            r = single_pole_extrap(f_ghz, gain_db, il, ih)
            return r[0], r[1], r[2]
        return extrap_20dbdec(f_ghz, gain_db)

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used  = False
    # Collected for the standardised fT/fmax xlsx export (simulated +
    # extrapolated columns).  Labels match across the two lists.
    sim_traces:    list[tuple[str, np.ndarray]] = []
    extrap_traces: list[tuple[str, np.ndarray, np.ndarray]] = []

    color_fT   = FT_FMAX_COLORS["fT"]
    color_fmax = FT_FMAX_COLORS["fmax"]

    def _label(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.1f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.1f} GHz (ext)"
        return f"{name}=n/a"

    # Trace order matters for the 2x2 legend: Plotly fills rows left→right
    # by trace insertion order.  We want the layout:
    #   [ |h21|² Meas. ]  [ |h21|² Model ]
    #   [ Mason U Meas.]  [ Mason U Model ]
    # so emit (h21 meas, h21 model, U meas, U model) in that order, with
    # extrap curves marked `showlegend=False`.

    # |h21|² measured (blue solid + markers)
    f_ext, g_ext, fT_m_ext = _do_extrap(h21_m)
    if f_ext is not None:
        extrap_used = True; f_high_track = max(f_high_track, fT_m_ext)
    fig.add_trace(go.Scattergl(
        x=f_ghz, y=h21_m, mode="lines+markers",
        name=f"|h21|² Meas. ({_label('fT', fT_m_in, fT_m_ext)})",
        line=dict(color=color_fT, width=1.4),
        marker=dict(symbol=FT_FMAX_SYMBOLS["h21"], size=6, color=color_fT)))
    sim_traces.append(("|h21|² Meas. (dB)", h21_m))
    if f_ext is not None:
        extrap_traces.append(("|h21|² Meas. (dB)", f_ext, g_ext))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines", showlegend=False,
            line=dict(color=color_fT, width=1.4, dash="dot")))

    # |h21|² modeled (blue dashed)
    f_ext, g_ext, fT_s_ext = _do_extrap(h21_s)
    if f_ext is not None:
        extrap_used = True; f_high_track = max(f_high_track, fT_s_ext)
    fig.add_trace(go.Scattergl(
        x=f_ghz, y=h21_s, mode="lines",
        name=f"|h21|² Model ({_label('fT', fT_s_in, fT_s_ext)})",
        line=dict(color=color_fT, width=2.0, dash="dash")))
    sim_traces.append(("|h21|² Model (dB)", h21_s))
    if f_ext is not None:
        extrap_traces.append(("|h21|² Model (dB)", f_ext, g_ext))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines", showlegend=False,
            line=dict(color=color_fT, width=2.0, dash="dot")))

    # Mason U measured (red solid + markers)
    f_ext, g_ext, fmax_m_ext = _do_extrap(U_m)
    if f_ext is not None:
        extrap_used = True; f_high_track = max(f_high_track, fmax_m_ext)
    fig.add_trace(go.Scattergl(
        x=f_ghz, y=U_m, mode="lines+markers",
        name=f"Mason U Meas. ({_label('fmax', fmax_m_in, fmax_m_ext)})",
        line=dict(color=color_fmax, width=1.4),
        marker=dict(symbol=FT_FMAX_SYMBOLS["U"], size=6, color=color_fmax)))
    sim_traces.append(("Mason U Meas. (dB)", U_m))
    if f_ext is not None:
        extrap_traces.append(("Mason U Meas. (dB)", f_ext, g_ext))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines", showlegend=False,
            line=dict(color=color_fmax, width=1.4, dash="dot")))

    # Mason U modeled (red dashed)
    f_ext, g_ext, fmax_s_ext = _do_extrap(U_s)
    if f_ext is not None:
        extrap_used = True; f_high_track = max(f_high_track, fmax_s_ext)
    fig.add_trace(go.Scattergl(
        x=f_ghz, y=U_s, mode="lines",
        name=f"Mason U Model ({_label('fmax', fmax_s_in, fmax_s_ext)})",
        line=dict(color=color_fmax, width=2.0, dash="dash")))
    sim_traces.append(("Mason U Model (dB)", U_s))
    if f_ext is not None:
        extrap_traces.append(("Mason U Model (dB)", f_ext, g_ext))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines", showlegend=False,
            line=dict(color=color_fmax, width=2.0, dash="dot")))

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1])
    fig.update_layout(
        title=dict(text=f"fT / fmax — {model_name}", font=dict(size=12)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=height,
        # Legend pinned bottom-left INSIDE the plot area.  Vertical stack
        # (not the 2x2 horizontal grid) so the long "Meas. (fT=… GHz)" entries
        # don't overlap inside the narrow in-plot box.
        legend=dict(orientation="v", x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=11)),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=50))
    plotly_with_dl(fig, key=key, filename=key,
                   excel_bytes=bode_excel_bytes(f_ghz, sim_traces, extrap_traces))

    # ── Extrapolation controls UNDERNEATH the chart (radio left, single-pole
    #    window slider right).  Rendered after the figure so they sit below
    #    it; the figure above reads these values from session_state on rerun.
    if any_needs and len(f_ghz) >= 2:
        ec1, ec2 = st.columns([1, 2])
        if f"{key}_extrap_method" not in st.session_state:
            st.session_state[f"{key}_extrap_method"] = "−20 dB/dec"
        with ec1:
            segmented_radio(
                tr("Extrap. method", "外插方法"),
                ["−20 dB/dec", "Single-pole"],
                key=f"{key}_extrap_method",
                format_func=_extrap_method_label,
                help=tr("−20 dB/dec anchors a slope-locked line at the last data "
                        "point (textbook fT/fmax projection).  Single-pole fits a "
                        "log-linear least-squares line over the chosen window "
                        "(default = final 5 GHz); slope is set by the data.",
                        "−20 dB/dec 於最後一個資料點錨定固定斜率的直線"
                        "（教科書式 fT/fmax 投影）。單極擬合則在所選視窗"
                        "（預設為最後 5 GHz）內做對數-線性最小平方擬合；"
                        "斜率由資料決定。"))
        if (st.session_state[f"{key}_extrap_method"] == "Single-pole"
                and len(f_ghz) >= 4):
            f_lo, f_hi = float(f_ghz[0]), float(f_ghz[-1])
            sp_default = (max(f_lo, f_hi - 5.0), f_hi)
            ec2.slider(
                tr("Single-pole fit window (GHz)", "單極擬合視窗 (GHz)"),
                min_value=f_lo, max_value=f_hi,
                value=st.session_state.get(f"{key}_sp_window", sp_default),
                step=max((f_hi - f_lo) / 400.0, 1e-3),
                key=f"{key}_sp_window")


# ════════════════════════════════════════════════════════════════════════════════
# Helper: single-dataset fT / fmax Bode block (forward-simulation only)
# ════════════════════════════════════════════════════════════════════════════════

def _build_forward_bode(S, freq_hz, title: str, *,
                        extrap_method: str = "−20 dB/dec", sp_window=None):
    """Build the fT/fmax Bode figure for a *single* simulated dataset.

    Returns ``(fig, any_needs, bode_xl)``.  ``any_needs`` is True when at least
    one trace still has gain > 0 dB at the top of the band (extrapolation is
    needed to project a 0-dB crossing).  Mirrors the built-in models' Bode card
    (FT_FMAX_COLORS: |h21|² → fT blue, Mason U → fmax red; extrap dotted).
    """
    f_ghz = freq_hz * 1e-9
    h21_db, U_db = compute_h21_U(S)
    fT, fmax = find_ft_fmax(f_ghz, h21_db, U_db)

    def _needs(in_val, gain):
        if in_val is not None:
            return False
        if not np.any(np.isfinite(gain)):
            return False
        with np.errstate(invalid="ignore"):
            return bool(np.nanmax(gain) > 0)
    any_needs = _needs(fT, h21_db) or _needs(fmax, U_db)

    def _extrap(y):
        if (extrap_method == "Single-pole" and sp_window is not None
                and len(f_ghz) >= 4):
            il = int(np.searchsorted(f_ghz, sp_window[0], side="left"))
            ih = int(np.searchsorted(f_ghz, sp_window[1], side="right")) - 1
            il = max(0, min(il, len(f_ghz) - 2))
            ih = max(il + 1, min(ih, len(f_ghz) - 1))
            r = single_pole_extrap(f_ghz, y, il, ih)
            return r[0], r[1], r[2]
        return extrap_20dbdec(f_ghz, y)

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used = False
    sim_traces: list[tuple[str, np.ndarray]] = []
    extrap_traces: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _meas_lbl(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.2f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.2f} GHz (extrap)"
        return f"{name}=n/a"

    def _add_trace(y, base_name, color, kind, in_val, symbol):
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = _extrap(y)
        ext_val = f0 if f_ext is not None else None
        legend_name = f"{base_name}  [{_meas_lbl(kind, in_val, ext_val)}]"
        fig.add_trace(go.Scatter(x=f_ghz, y=y, mode="lines+markers",
                                 name=legend_name,
                                 line=dict(color=color, width=2),
                                 marker=dict(symbol=symbol, size=6, color=color)))
        sim_traces.append((f"{base_name} (dB)", np.asarray(y)))
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            extrap_traces.append((f"{base_name} (dB)", f_ext, g_ext))
            fig.add_trace(go.Scatter(x=f_ext, y=g_ext, mode="lines",
                                     name=f"{legend_name} extrap",
                                     line=dict(color=color, width=2, dash="dot"),
                                     showlegend=False))

    _add_trace(h21_db, "|h21|²", FT_FMAX_COLORS["fT"], "fT", fT,
               FT_FMAX_SYMBOLS["h21"])
    _add_trace(U_db, "Mason U", FT_FMAX_COLORS["fmax"], "fmax", fmax,
               FT_FMAX_SYMBOLS["U"])

    bode_xl = bode_excel_bytes(f_ghz, sim_traces, extrap_traces)

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = (float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1]))
    fig.update_layout(
        title=dict(text=f"fT / fmax — {title}", font=dict(size=12)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        legend=dict(orientation="v", x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=13)),
        hovermode="x unified", margin=dict(l=55, r=20, t=40, b=50))
    return fig, any_needs, bode_xl


def render_forward_bode_block(S, freq_hz, title: str, key: str):
    """Render a single-dataset fT/fmax Bode plot with the standard download /
    copy buttons, then — when any trace needs extrapolation — a segmented
    method selector (−20 dB/dec ⇄ Single-pole) and single-pole window slider
    UNDERNEATH the chart.  The figure reads the current selection from
    session_state, so a rerun on widget change feeds it back here.  Shared by
    the custom-model forward simulator so it matches the built-in models.
    """
    f_ghz = freq_hz * 1e-9
    method = st.session_state.get(f"{key}_extrap_method", "−20 dB/dec")
    sp_window = None
    if method == "Single-pole" and len(f_ghz) >= 4:
        f_lo, f_hi = float(f_ghz[0]), float(f_ghz[-1])
        sp_window = st.session_state.get(f"{key}_sp_window",
                                         (max(f_lo, f_hi - 5.0), f_hi))
    fig, any_needs, bode_xl = _build_forward_bode(
        S, freq_hz, title, extrap_method=method, sp_window=sp_window)
    plotly_with_dl(fig, key=key, filename=key, excel_bytes=bode_xl)

    if any_needs and len(f_ghz) >= 2:
        ec1, ec2 = st.columns([1, 2])
        if f"{key}_extrap_method" not in st.session_state:
            st.session_state[f"{key}_extrap_method"] = "−20 dB/dec"
        with ec1:
            segmented_radio(
                tr("Extrap. method", "外插方法"),
                ["−20 dB/dec", "Single-pole"],
                key=f"{key}_extrap_method",
                format_func=_extrap_method_label,
                help=tr("−20 dB/dec anchors a slope-locked line at the last data "
                        "point.  Single-pole fits a log-linear line over the "
                        "chosen window (default = final 5 GHz).",
                        "−20 dB/dec 於最後一個資料點錨定固定斜率的直線。"
                        "單極擬合則在所選視窗（預設為最後 5 GHz）內做"
                        "對數-線性擬合。"))
        if (st.session_state[f"{key}_extrap_method"] == "Single-pole"
                and len(f_ghz) >= 4):
            f_lo, f_hi = float(f_ghz[0]), float(f_ghz[-1])
            sp_default = (max(f_lo, f_hi - 5.0), f_hi)
            ec2.slider(
                tr("Single-pole fit window (GHz)", "單極擬合視窗 (GHz)"),
                min_value=f_lo, max_value=f_hi,
                value=st.session_state.get(f"{key}_sp_window", sp_default),
                step=max((f_hi - f_lo) / 400.0, 1e-3),
                key=f"{key}_sp_window")


# ════════════════════════════════════════════════════════════════════════════════
# Helper: τ_total and calculated fmax expander
# ════════════════════════════════════════════════════════════════════════════════

def _extrap_f0(f_ghz, gain_db, extrap_method, sp_window):
    """0-dB crossing (GHz) via the selected extrapolation method."""
    if (extrap_method == "Single-pole" and sp_window is not None
            and len(f_ghz) >= 4):
        il = int(np.searchsorted(f_ghz, sp_window[0], side="left"))
        ih = int(np.searchsorted(f_ghz, sp_window[1], side="right")) - 1
        il = max(0, min(il, len(f_ghz) - 2))
        ih = max(il + 1, min(ih, len(f_ghz) - 1))
        return single_pole_extrap(f_ghz, gain_db, il, ih)[2]
    return extrap_20dbdec(f_ghz, gain_db)[2]


def _eff_ft_fmax_ghz(f_ghz, h21_db, U_db,
                     extrap_method="−20 dB/dec", sp_window=None):
    """Effective fT / fmax in GHz: in-band 0-dB crossing if present, else the
    extrapolated 0-dB frequency using the *selected* method (so this tracks the
    Bode plot's −20 dB/dec ⇄ single-pole radio)."""
    fT, fmax = find_ft_fmax(f_ghz, h21_db, U_db)
    if fT is None:
        fT = _extrap_f0(f_ghz, h21_db, extrap_method, sp_window)
    if fmax is None:
        fmax = _extrap_f0(f_ghz, U_db, extrap_method, sp_window)
    return fT, fmax


def _read_extrap_selection(extrap_key, f_ghz):
    """Read the extrap-method radio / single-pole window the Bode plot wrote to
    session_state under ``extrap_key`` so the τ/fmax numbers use the same fT.

    The Bode card is rendered *before* this expander, so its session_state keys
    already exist on the same run.  Defaults mirror the Bode card: −20 dB/dec,
    single-pole window = final 5 GHz.
    """
    if not extrap_key:
        return "−20 dB/dec", None
    method = st.session_state.get(f"{extrap_key}_extrap_method", "−20 dB/dec")
    sp_window = None
    if method == "Single-pole" and len(f_ghz) >= 4:
        f_lo, f_hi = float(f_ghz[0]), float(f_ghz[-1])
        sp_window = st.session_state.get(f"{extrap_key}_sp_window",
                                         (max(f_lo, f_hi - 5.0), f_hi))
    return method, sp_window


def _tau_total_ps(fT_ghz):
    """τ_total = 1/(2π fT) in ps, or None when fT is unavailable."""
    if fT_ghz is None or fT_ghz <= 0:
        return None
    return 1.0 / (2.0 * np.pi * fT_ghz * 1e9) * 1e12


def _calc_fmax_ghz(fT_ghz, CBC, Rbb):
    """fmax = sqrt(fT / (8π·C_BC·R_bb)) in GHz, or None when inputs invalid."""
    if fT_ghz is None or fT_ghz <= 0 or CBC <= 0 or Rbb <= 0:
        return None
    fT_hz = fT_ghz * 1e9
    return float(np.sqrt(fT_hz / (8.0 * np.pi * CBC * Rbb))) * 1e-9


def render_tau_fmax_expander(*, key, freq, S_meas, CBC, Rbb, S_model=None,
                             tau_sum=None, tau_sum_label="τB + τC",
                             tau_sum_tex=r"\tau_B+\tau_C", extrap_key=None):
    """Expander "Calculated Tau_total and fmax".

    Left column — τ_total = 1/(2π fT), plus the transit-time decomposition

        τ_total = τ_B + τ_C + (nkT/qI_c)·C_je + (R_c + R_e + nkT/qI_c)·C_bc

      where the model's ``tau_sum`` (= τ_B+τ_C for T models, τ for π) is shown
      and the remaining (charge-storage) term is computed as τ_total − tau_sum.

    Right column — calculated fmax = sqrt(fT / (8π·C_BC·R_bb)) compared with the
      real (0-dB crossing) fmax.  C_BC / R_bb default to the extracted values
      (Cbcx+Cbc / Rbi+Rb) but a radio lets the user supply custom values.

    ``S_model`` given (SSM extraction) → measured **and** modeled values (4 fmax
    values).  Omitted (RF simulator) → single dataset.  ``extrap_key`` points at
    the Bode card's session_state so fT/fmax follow the selected extrap method.
    """
    f_ghz = np.asarray(freq) * 1e-9
    extrap_method, sp_window = _read_extrap_selection(extrap_key, f_ghz)
    h21_m, U_m = compute_h21_U(S_meas)
    fT_m, fmax_m = _eff_ft_fmax_ghz(f_ghz, h21_m, U_m, extrap_method, sp_window)
    has_model = S_model is not None
    if has_model:
        h21_s, U_s = compute_h21_U(S_model)
        fT_s, fmax_s = _eff_ft_fmax_ghz(f_ghz, h21_s, U_s,
                                        extrap_method, sp_window)

    tau_m_ps = _tau_total_ps(fT_m)
    tau_s_ps = _tau_total_ps(fT_s) if has_model else None
    tau_sum_ps = tau_sum * 1e12 if tau_sum is not None else None

    def _rem(tau_ps):
        if tau_ps is None or tau_sum_ps is None:
            return None
        return tau_ps - tau_sum_ps

    import pandas as pd
    def _num(v, nd=4):
        return f"{v:.{nd}f}" if (v is not None and np.isfinite(v)) else "n/a"

    with st.container(
            key="hbt_exp_view_tau_" + re.sub(r"[^0-9A-Za-z_-]", "-", key)
    ), st.expander(tr("🔣 Calculated τ_total and fmax", "🔣 計算 τ_total 與 fmax"),
                   expanded=False):
        col_tau, col_fmax = st.columns(2)

        # ── τ_total ───────────────────────────────────────────────────────
        with col_tau:
            st.markdown(tr("**Total transit time, τ_total**",
                           "**總傳輸時間, τ_total**"))
            st.latex(r"\tau_{total}=\frac{1}{2\pi f_T}")
            st.latex(r"\tau_{total}=\tau_B+\tau_C+\frac{nkT}{qI_c}C_{je}"
                     r"+\left(R_c+R_e+\frac{nkT}{qI_c}\right)C_{bc}")
            # τ_B+τ_C (transit-time sum) under the equation, as latex.
            if tau_sum_ps is not None:
                st.latex(tau_sum_tex + r"=" + f"{tau_sum_ps:.4f}" + r"\,\text{ps}")

            # Table — rows = Measured / Modeled (single "Value" row when there
            # is no model).  Columns: fT (first), τ_total, then
            # τ_total − (τ_B+τ_C).
            _rows = [tr("Measured", "量測"), tr("Modeled", "模型")] if has_model \
                else [tr("Value", "數值")]
            _taus = [tau_m_ps, tau_s_ps] if has_model else [tau_m_ps]
            _fts  = [fT_m, fT_s] if has_model else [fT_m]
            _tau_tbl = {
                "fT (GHz)":     [_num(v, 2) for v in _fts],
                "τ_total (ps)": [_num(v) for v in _taus],
            }
            if tau_sum_ps is not None:
                _tau_tbl[f"τ_total − ({tau_sum_label}) (ps)"] = [
                    _num(_rem(v)) for v in _taus]
            st.table(pd.DataFrame(_tau_tbl, index=_rows))

            if tau_sum_ps is not None:
                st.caption(tr("τ_total − (τ_B+τ_C) = emitter charging time "
                              "(nkT/qI_c · C_je) + collector charging time "
                              "((R_c+R_e+nkT/qI_c) · C_bc).",
                              "τ_total − (τ_B+τ_C) = 射極充電時間 "
                              "(nkT/qI_c · C_je) + 集極充電時間 "
                              "((R_c+R_e+nkT/qI_c) · C_bc)。"))

        # ── calculated fmax ───────────────────────────────────────────────
        with col_fmax:
            st.markdown(tr("**Maximum oscillation frequency, fmax**",
                           "**最大振盪頻率, fmax**"))
            st.latex(r"f_{max}=\sqrt{\frac{f_T}{8\pi C_{BC} R_{bb}}}")

            src = segmented_radio(
                tr("C_BC / R_bb source", "C_BC / R_bb 來源"),
                ["Extracted", "Custom"],
                key=f"{key}_cbcrbb_src",
                format_func=lambda v: tr(v, {"Extracted": "萃取值",
                                              "Custom": "自訂"}.get(v, v)))
            if src == "Custom":
                cc1, cc2 = st.columns(2)
                sk_cbc, sk_rbb = f"{key}_cbc_fF", f"{key}_rbb"
                if sk_cbc not in st.session_state:
                    st.session_state[sk_cbc] = float(CBC * 1e15)
                if sk_rbb not in st.session_state:
                    st.session_state[sk_rbb] = float(Rbb)
                cc1.number_input("C_BC (fF)", min_value=0.0, step=0.1,
                                 format="%.4f", key=sk_cbc)
                cc2.number_input("R_bb (Ω)", min_value=0.0, step=0.1,
                                 format="%.4f", key=sk_rbb)
                CBC_use = float(st.session_state[sk_cbc]) * 1e-15
                Rbb_use = float(st.session_state[sk_rbb])
            else:
                CBC_use, Rbb_use = CBC, Rbb
                st.caption(f"C_BC = Cbcx + Cbc = {CBC * 1e15:.4f} fF , "
                           f"R_bb = Rbi + Rb = {Rbb:.4f} Ω")

            # Table — rows = Measured / Modeled (single "Value" row when there
            # is no model).  Columns: fT (first), fmax from the S-parameter
            # 0-dB crossing (simulation), then fmax from the formula above
            # (calculation).
            _frows = [tr("Measured", "量測"), tr("Modeled", "模型")] if has_model \
                else [tr("Value", "數值")]
            _fsim  = [fmax_m, fmax_s] if has_model else [fmax_m]
            _fts2  = [fT_m, fT_s] if has_model else [fT_m]
            _fcalc = [_calc_fmax_ghz(ft, CBC_use, Rbb_use) for ft in _fts2]
            st.table(pd.DataFrame({
                "fT (GHz)":                                    [_num(v, 2) for v in _fts2],
                tr("fmax simulation (GHz)", "fmax 模擬 (GHz)"):  [_num(v, 2) for v in _fsim],
                tr("fmax calculation (GHz)", "fmax 計算 (GHz)"): [_num(v, 2) for v in _fcalc],
            }, index=_frows))


# ════════════════════════════════════════════════════════════════════════════════
# Matplotlib Smith chart (publication-style, no in-figure legend)
# ════════════════════════════════════════════════════════════════════════════════

_MPL_SMITH_COLORS = {"S11": "#1f77b4", "S12": "#d62728",
                     "S21": "#2ca02c", "S22": "#ff7f0e"}

_MPL_LINE_STYLES   = {"solid": "-", "dashed": "--", "dotted": ":"}
_MPL_MARKER_STYLES = {"circle": "o", "square": "s", "star": "*", "triangle": "^", "x": "x"}
# Marker styles that are line-based (unfilled) — they need a visible edge
# width to be drawn at all in matplotlib.
_MPL_UNFILLED_MARKERS = {"x"}


# Smith-chart grid: how many constant-R circles to draw.  A constant-R=r
# circle has radius 1/(r+1) in the Γ-plane and is tangent to (+1, 0).  To
# distribute N circles with *evenly-spaced radii* over (0, 1], we use
# rad_i = i/N  ⇒  r_i = N/i − 1   for i = 1..N.
# At N=6 this reproduces the classic [5, 2, 1, 0.5, 0.2, 0] set.
_SMITH_GRID_MIN     = 2
_SMITH_GRID_MAX     = 12
_SMITH_GRID_DEFAULT = 4

# Text auto-format: regex substitution applied to every text annotation
# drawn on the Smith chart.  Default: find S11 / S12 / S21 / S22 anywhere in
# the input and wrap each token as upright mathtext (\mathrm) with the
# digits as a true subscript — surrounding characters pass through.
# Example: "S12/5" → "$\mathrm{S}_{12}$/5" → upright S₁₂ + "/5" regular.
_DEFAULT_AUTOFMT_PATTERN     = r"S(11|12|21|22)"
_DEFAULT_AUTOFMT_REPLACEMENT = r"$\mathrm{S}_{\1}$"


def _apply_text_autoformat(text: str, pattern: str, replacement: str) -> str:
    r"""Apply a regex substitution where the replacement is a *template*.

    Only ``\1`` … ``\9`` are interpreted as capture-group backreferences.
    All other backslash sequences (e.g. ``\mathrm``, ``\alpha``) pass
    through verbatim, so the replacement can carry matplotlib mathtext
    commands without tripping Python 3.12+'s "unknown escape" error in
    ``re.sub``.

    Returns the original text unchanged if the pattern is empty/invalid.
    """
    if not pattern:
        return text
    try:
        compiled = re.compile(pattern)
    except re.error:
        return text

    def _sub(match: re.Match) -> str:
        out = replacement
        # Substitute \1 .. \9 from highest to lowest so e.g. \12 doesn't
        # collide with \1 followed by literal "2".
        for i in range(min(9, len(match.groups())), 0, -1):
            out = out.replace(f"\\{i}", match.group(i) or "")
        return out

    return compiled.sub(_sub, text)


def _smith_grid_values(n: int) -> tuple[list[float], list[float]]:
    """Return (r_values, x_values) for N evenly-spaced Smith-chart grid lines.

    R-values include r=0 (outer unit circle).  X-values use the same set
    with r=0 stripped, mirrored over the real axis at draw time.
    """
    n = max(_SMITH_GRID_MIN, min(_SMITH_GRID_MAX, int(n)))
    r_vals = [n / i - 1.0 for i in range(1, n + 1)]
    x_vals = [v for v in r_vals if v > 0.0]
    return r_vals, x_vals


def _draw_mpl_smith_background(ax, line_lw: float, grid_lw: float,
                               density: int = _SMITH_GRID_DEFAULT):
    """Draw the constant-R / constant-X grid for a unit Smith chart on ``ax``.

    Grid (constant-R / constant-X arcs) is drawn first so the black outer-unit
    circle and real-axis line render on top of any grid intersections.

    ``density`` is the desired number of R-circles; positions are recomputed
    for each value so the visible circles are always evenly spaced.
    """
    r_values, x_values = _smith_grid_values(int(density))

    t = np.linspace(0.0, 2.0*np.pi, 500)
    for r in r_values:
        cx  = r/(r+1.0); rad = 1.0/(r+1.0)
        xc  = cx + rad*np.cos(t); yc = rad*np.sin(t)
        out = (xc**2 + yc**2) > 1.0
        xc[out] = np.nan; yc[out] = np.nan
        ax.plot(xc, yc, color="gray", linewidth=grid_lw, alpha=0.7, zorder=1)
    for x in x_values:
        for sign in (1, -1):
            xv  = sign*x; rad = 1.0/abs(xv)
            xc  = 1.0 + rad*np.cos(t); yc = (1.0/xv) + rad*np.sin(t)
            out = (xc**2 + yc**2) > 1.0
            xc[out] = np.nan; yc[out] = np.nan
            ax.plot(xc, yc, color="gray", linewidth=grid_lw, alpha=0.7, zorder=1)
    ax.plot(np.cos(t), np.sin(t), color="black", linewidth=line_lw, zorder=3)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="black",
            linewidth=line_lw*0.8, zorder=3)


_SPARAM_DEFAULT_POS = {
    "S11": ( 0.60, -0.45),
    "S12": ( 0.25,  0.10),
    "S21": (-0.40,  0.45),
    "S22": ( 0.25, -0.20),
}


def _mult_label_text(sp: str, mult) -> str:
    """On-chart label text implied by an S-param's multiplier.

      mult == 1  → ``"S21"``            (no scaling annotation)
      mult >  1  → ``"S21x3"``          (scaled up)
      mult <  1  → ``"S21/5"``          (scaled down → reciprocal, e.g. 0.2→/5)

    ``%g`` trims trailing zeros so 3.0 → "3" and 0.2 → "/5".  Falls back to the
    bare S-param name for non-positive / unparseable multipliers.
    """
    try:
        m = float(mult)
    except (TypeError, ValueError):
        return sp
    if m <= 0 or abs(m - 1.0) < 1e-9:
        return sp
    if m > 1.0:
        return f"{sp}x{m:g}"
    return f"{sp}/{(1.0 / m):g}"


def _sync_text_to_mult(skey: str, sp: str) -> None:
    """on_change callback: rewrite an S-param's Text field from its multiplier.

    Runs at the start of the rerun (before the Text widget is instantiated),
    so writing its session_state key here is safe and shows up in the field.
    """
    st.session_state[f"{skey}_text_{sp}"] = _mult_label_text(
        sp, st.session_state.get(f"{skey}_mult_{sp}"))


def _ctrl_section(text: str) -> None:
    """Consistent uppercase section header for the Smith-chart control panel.

    Every control group (appearance, trace styling, per-S-param, annotations)
    opens with one of these so the panel reads as a set of labelled cards
    instead of loose, ungrouped widgets.
    """
    st.markdown(
        "<div style='margin:0 0 8px 0;font-size:0.72rem;font-weight:700;"
        "color:#6b7280;letter-spacing:.06em;text-transform:uppercase'>"
        + text + "</div>",
        unsafe_allow_html=True)


def render_matplotlib_smith(S_mea=None, S_sim=None, fname: str = "",
                            topo_key: str = "", *,
                            sets=None, default_multiplier=1.0,
                            phase: str = "both",
                            freq_hz=None, add_pool=None,
                            return_png: bool = False):
    """
    Publication-style Smith chart drawn with matplotlib.

    ``phase`` controls which half runs — used by the SSM tab's split
    "Topology + Smith Chart" / "Smith Chart Controls" two-column
    layout (callers invoke this function twice — once with
    ``phase="controls"`` in the right column, once with ``phase="chart"``
    in the left column):

      • ``"both"`` (default, backwards-compat): widgets THEN chart.
      • ``"controls"``: widgets only — no figure rendered.
      • ``"chart"``: chart only — reads previously-written session
        state and skips widget creation.

    ``freq_hz`` (optional ndarray of frequencies in Hz) seeds a default
    "{lo:g}~{hi:g} GHz" annotation at position (0.0, -1.1) on the first
    render — the user can then edit / move / delete it through the
    "Extra text annotations" controls.

    Two ways to call:
      Backward-compatible (used by SSM extraction):
        render_matplotlib_smith(S_mea, S_sim, fname, topo_key)
            → 2 sets: Measured (× markers) + Modeled (solid line).

    Color modes (per-trace controls):
      • "Different colors for different traces" (default, legacy) — each
        S-param has its own color, shared across all sets.
      • "Different colors for measured and modeled" — each set has one color
        shared across its four S-params (defaults: Measured = #0201f0 blue,
        Modeled = #b50000 red).  Per-S-param color still drives the text
        annotation colors in this mode.

    A per-set "Decimate every Nth point" control is shown only for the
    Measured set.

      Extensible:
        render_matplotlib_smith(fname=..., topo_key=..., sets=[
            {"S": ndarray(N,2,2), "label": "Sim",
             "kind": "line"|"marker", "style": "solid"/"dashed"/"dotted"
                                            or "circle"/"square"/"star"/"triangle"},
            ...
        ], default_multiplier=1.0)

    The four S-param rows have a multiplier, text, x/y position, and color
    that always apply to the corresponding trace in every set (and to the
    text annotation drawn at the chosen x/y).  ``default_multiplier`` may be
    a scalar applied to all four traces, or a dict ``{"S11":…,"S12":…}``.
    """
    import matplotlib.pyplot as plt

    # ── Build sets list when called the legacy way ───────────────────────────
    if sets is None:
        _sets = []
        if S_mea is not None:
            _sets.append({"S": S_mea, "label": "Measured",
                          "kind": "marker", "style": "x"})
        if S_sim is not None:
            _sets.append({"S": S_sim, "label": "Modeled",
                          "kind": "line", "style": "solid"})
        sets = _sets
    sets = [s for s in sets if s.get("S") is not None]
    if not sets:
        st.info(tr("No S-parameter data available to plot.",
                   "沒有可繪製的 S 參數資料。"))
        return

    # ── Additional uploaded-file traces (the "➕" under the styling table) ────
    # ``add_pool`` is a list of {"label", "S"} for *other* uploaded files.  The
    # user picks which to overlay via the styling table's "Add a file trace"
    # button; the picks persist in session_state and are appended to ``sets`` in
    # BOTH phases so the styling rows and the drawn traces stay in lock-step.
    skey       = f"smith_mpl_{topo_key}_{fname}"
    _pool        = [p for p in (add_pool or []) if p.get("S") is not None]
    _pool_labels = [p["label"] for p in _pool]
    _pool_by_lbl = {p["label"]: p for p in _pool}
    _extra_sel_key = f"{skey}_extra_files"
    if _pool:
        _sel = [l for l in st.session_state.get(_extra_sel_key, [])
                if l in _pool_labels]
        st.session_state[_extra_sel_key] = _sel
        for _j, _lbl in enumerate(_sel):
            sets.append({"S": _pool_by_lbl[_lbl]["S"], "label": _lbl,
                         "kind": "line", "style": "solid",
                         "_extra": True, "_extra_idx": _j})
    _has_extra = any(s.get("_extra") for s in sets)

    if isinstance(default_multiplier, dict):
        default_mults = {sp: float(default_multiplier.get(sp, 1.0))
                         for sp in ("S11", "S12", "S21", "S22")}
    else:
        default_mults = {sp: float(default_multiplier)
                         for sp in ("S11", "S12", "S21", "S22")}

    extra_key  = f"{skey}_n_extra_texts"
    sparams    = ("S11", "S12", "S21", "S22")

    # ── First-time defaults for the 4 fixed S-param rows ─────────────────────
    # NOTE: text color defaults to the *darker* variant of the trace color so
    # on-chart labels stay legible against the (often light) trace strokes.
    def _hex_darken_init(h, by=45):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{max(0,r-by):02x}{max(0,g-by):02x}{max(0,b-by):02x}"

    # "#000000" is treated as a sentinel for "uninitialized / reset by
    # Streamlit on widget unmount" — when the user switches color modes,
    # some color_picker keys can come back as black on remount.  We force
    # them back to the proper palette default so the controls never
    # surprise the user with black.  (If a user genuinely wants black,
    # they can re-pick it after every mode change — rare in practice.)
    def _resolve_color(state_key, default_hex):
        cur = st.session_state.get(state_key)
        if cur is None or str(cur).lower() == "#000000":
            st.session_state[state_key] = default_hex
        return st.session_state[state_key]

    # Auto-place each S-param label where its (primary-set) trace actually is:
    # the centroid of the trace points, pushed slightly outward from the chart
    # centre so the text sits beside the curve rather than on top of it.  Falls
    # back to the curated fixed position when no finite trace data is available.
    _SP_RC = {"S11": (0, 0), "S12": (0, 1), "S21": (1, 0), "S22": (1, 1)}

    def _auto_label_pos(sp):
        try:
            r, c = _SP_RC[sp]
            z = np.asarray(sets[0]["S"])[:, r, c] * default_mults[sp]
            z = z[np.isfinite(z)]
            if z.size:
                ctr = complex(float(np.mean(z.real)), float(np.mean(z.imag)))
                rad = abs(ctr)
                if rad > 1e-3:
                    ctr = ctr + 0.13 * (ctr / rad)      # nudge outward
                return (float(np.clip(ctr.real, -1.08, 1.08)),
                        float(np.clip(ctr.imag, -1.08, 1.08)))
        except Exception:                                # noqa: BLE001
            pass
        return _SPARAM_DEFAULT_POS[sp]

    for sp in sparams:
        x0, y0 = _auto_label_pos(sp)
        if f"{skey}_text_{sp}" not in st.session_state:
            # Seed the label from the default multiplier so a non-unity
            # default already reads "Sxx×N" / "Sxx/N" on the first render
            # (matches the auto-update on later multiplier edits).
            st.session_state[f"{skey}_text_{sp}"]  = _mult_label_text(
                sp, default_mults[sp])
        if f"{skey}_x_{sp}" not in st.session_state:
            st.session_state[f"{skey}_x_{sp}"]     = float(x0)
        if f"{skey}_y_{sp}" not in st.session_state:
            st.session_state[f"{skey}_y_{sp}"]     = float(y0)
        if f"{skey}_mult_{sp}" not in st.session_state:
            st.session_state[f"{skey}_mult_{sp}"]  = float(default_mults[sp])
        _resolve_color(f"{skey}_color_{sp}", _MPL_SMITH_COLORS[sp])
        _resolve_color(f"{skey}_text_color_{sp}",
                       _hex_darken_init(_MPL_SMITH_COLORS[sp]))

    # ── Commit hand-dragged label positions from the "Drag labels" component ──
    # The drag component's value is already in session_state at the TOP of the
    # run its drag triggers, so we read it HERE — before the x/y number_input
    # widgets exist (writes allowed) and before the labels are rebuilt — so the
    # dragged position lands in the SAME run.  (Reading it later, after the
    # component call, committed one run late and the label visibly bounced back
    # to its pre-drag spot.)  ``_drag_order`` (stored by the chart phase) maps
    # the component's label index to its state key, matching the labels that
    # were on screen when the drag happened.  A nonce gate applies each drag
    # once, so it never clobbers a later manual x/y edit, and positions persist
    # when the toggle is switched off.  The freq caption is not a draggable
    # label, so it is never moved.
    _dc = st.session_state.get(f"{skey}_drag_chart")
    _dorder = st.session_state.get(f"{skey}_drag_order")
    if (isinstance(_dc, dict) and "index" in _dc and _dorder
            and _dc.get("nonce") != st.session_state.get(f"{skey}_drag_applied_nonce")):
        try:
            _di = int(_dc["index"])
            if 0 <= _di < len(_dorder):
                _dkind, _dk = _dorder[_di]
                _dpx, _dpy = float(_dc["x"]), float(_dc["y"])
                if _dkind == "sp":
                    st.session_state[f"{skey}_x_{_dk}"] = _dpx
                    st.session_state[f"{skey}_y_{_dk}"] = _dpy
                elif _dkind == "extra":
                    st.session_state[f"{skey}_ex_{_dk}"] = _dpx
                    st.session_state[f"{skey}_ey_{_dk}"] = _dpy
            st.session_state[f"{skey}_drag_applied_nonce"] = _dc.get("nonce")
        except (TypeError, ValueError, KeyError):
            pass

    # ── Follow the companion Plotly Smith multiplier ─────────────────────────
    # ``default_multiplier`` doubles as a live sync source: the RF / SSM tools
    # pass the same per-trace multiplier the user edits next to their Plotly
    # Smith chart.  It seeds these widgets on the first render (loop above);
    # here we also mirror any *change* to it into the per-S-param multiplier (and
    # text-label) state so editing the Plotly multiplier auto-updates this
    # matplotlib chart.  Runs before the widgets are instantiated, so the writes
    # take effect this rerun.  Tracked *per S-param*, so changing one Plotly
    # multiplier only updates that trace — manual edits to the other three
    # matplotlib multipliers persist.  A constant default never re-fires.
    for sp in sparams:
        _synced_sp = f"{skey}_synced_mult_{sp}"
        _new = default_mults[sp]
        if _synced_sp not in st.session_state:         # first render → seed only
            st.session_state[_synced_sp] = _new
        elif st.session_state[_synced_sp] != _new:     # Plotly value changed
            st.session_state[f"{skey}_mult_{sp}"] = _new
            st.session_state[f"{skey}_text_{sp}"] = _mult_label_text(sp, _new)
            st.session_state[_synced_sp] = _new

    if extra_key not in st.session_state:
        st.session_state[extra_key] = 0

    # ── Seed a default freq-range annotation — PHASE-INDEPENDENT ─────────────
    # Runs in every phase (controls / chart / both) so the "{lo}~{hi} GHz" text
    # is filled in by default no matter which render path touches this chart
    # first — including the chart-only `return_png` path used by the topology
    # Smith overlay.
    #
    # A dedicated "freq slot" (its own annotation index) holds this label, and a
    # *signature* of the current frequency span gates re-seeding.  Earlier this
    # used a one-shot boolean sentinel, which left the field BLANK whenever a new
    # device arrived on a chart whose key had already been seeded (e.g. after a
    # cross-page handover): the span changed but the text never refreshed.  Keying
    # on the span fixes that — a new device re-seeds, an unchanged span (so the
    # user's manual edits) is left untouched.
    _seed_sig_key = f"{skey}_freq_seed_sig"
    if freq_hz is not None:
        try:
            _f = np.asarray(freq_hz, dtype=float)
            _f = _f[np.isfinite(_f)]
            if _f.size >= 2:
                _lo = float(_f.min()) * 1e-9
                _hi = float(_f.max()) * 1e-9
                _sig = f"{_lo:g}~{_hi:g}"
                _slot = st.session_state.get(f"{skey}_freq_slot")
                _cur  = st.session_state.get(f"{skey}_etext_{_slot}") if _slot is not None else None
                # (Re)seed when the span changed, or the dedicated slot is
                # missing / blank (recovers a field left empty by an earlier run).
                if st.session_state.get(_seed_sig_key) != _sig or not _cur:
                    if _slot is None:
                        _slot = int(st.session_state.get(extra_key, 0))
                        st.session_state[extra_key] = _slot + 1
                        st.session_state[f"{skey}_freq_slot"] = _slot
                    st.session_state[f"{skey}_etext_{_slot}"]  = f"{_lo:g} ~ {_hi:g} GHz"
                    st.session_state.setdefault(f"{skey}_ex_{_slot}",     0.0)
                    st.session_state.setdefault(f"{skey}_ey_{_slot}",    -1.1)
                    st.session_state.setdefault(f"{skey}_ecolor_{_slot}", "#000000")
                    st.session_state[_seed_sig_key] = _sig
        except (TypeError, ValueError):
            pass

    _run_controls = phase in ("controls", "both")
    _run_chart    = phase in ("chart",    "both")

    # Draggable Plotly preview toggle — read from session_state up front so
    # both phases see the same value: the "controls" call below owns the
    # widget (and re-reads it fresh after instantiating it); the "chart"
    # call never creates the widget, so this session_state read is its only
    # way to know whether the preview is on.
    drag_on = bool(st.session_state.get(f"{skey}_drag_preview", False))

    # ── Chart appearance panel: line/grid thickness, grid density, text size,
    #    and the global coloring mode — grouped in one bordered card so these
    #    canvas-wide settings read as a unit instead of loose top-of-page
    #    number inputs. ────────────────────────────────────────────────────
    _PER_SET_DEFAULT_COLORS = {"Measured": "#0201f0", "Modeled": "#b50000"}
    has_measured = any(s.get("label") == "Measured" for s in sets)

    _COLOR_MODE_PER_TRACE = "trace"
    _COLOR_MODE_PER_SET   = "bicolor"
    _COLOR_MODE_CUSTOM    = "custom"

    color_mode = _COLOR_MODE_PER_TRACE
    if _run_controls:
        with st.container(border=True):
            _ctrl_section(tr("Chart appearance", "圖表外觀"))
            c_smith, c_grid, c_density, c_textsize = st.columns(4)
            smith_lw = c_smith.number_input(tr("Line thickness", "線寬"),
                                            min_value=0.1, max_value=5.0, value=3.0,
                                            step=0.1, format="%.2f",
                                            key=f"{skey}_smith_lw")
            grid_lw  = c_grid.number_input(tr("Grid thickness", "格線線寬"),
                                           min_value=0.1, max_value=5.0, value=1.0,
                                           step=0.1, format="%.2f",
                                           key=f"{skey}_grid_lw")
            grid_density = c_density.number_input(
                tr("Grid circles", "格線圓數"),
                min_value=_SMITH_GRID_MIN, max_value=_SMITH_GRID_MAX,
                value=_SMITH_GRID_DEFAULT, step=1,
                help=tr("Number of constant-R circles to draw.  Positions are "
                        "recomputed (evenly-spaced radii) for each value.",
                        "要繪製的等電阻圓數量。每次改值都會重新計算圓的位置"
                        "（半徑等間距分布）。"),
                key=f"{skey}_grid_count")
            text_size = c_textsize.number_input(
                tr("Text size", "文字大小"), min_value=4.0, max_value=48.0, value=18.0,
                step=1.0, format="%.1f",
                help=tr("Font size for all on-chart text annotations "
                        "(S-param labels + free text).",
                        "圖上所有文字標註的字級（S 參數標籤 + 自由文字）。"),
                key=f"{skey}_text_size")

            # Global coloring mode lives in the same panel — it decides which
            # colour pickers appear in the trace-styling / per-S-param tables
            # below.  Option VALUES stay the canonical "trace"/"bicolor"/
            # "custom" codes (compared below and persisted in session_state);
            # only the chip label localizes via format_func.
            if has_measured:
                _color_mode_labels = {
                    _COLOR_MODE_PER_TRACE: tr("trace", "依 S 參數"),
                    _COLOR_MODE_PER_SET:   tr("bicolor", "雙色"),
                    _COLOR_MODE_CUSTOM:    tr("custom", "自訂"),
                }
                color_mode = segmented_radio(
                    tr("Coloring mode", "配色模式"),
                    [_COLOR_MODE_PER_TRACE, _COLOR_MODE_PER_SET,
                     _COLOR_MODE_CUSTOM],
                    key=f"{skey}_color_mode",
                    format_func=lambda m: _color_mode_labels.get(m, m),
                    help=tr("**trace** — each S-param has its own color, shared "
                            "across all sets.  \n"
                            "**bicolor** — each set (Measured / Modeled) has one "
                            "color, shared across its four S-params.  \n"
                            "**custom** — pick a color independently for every "
                            "(set, S-param) combination; defaults give measured "
                            "the legacy palette and modeled a darker version.",
                            "**依 S 參數** — 每個 S 參數各有顏色，所有資料組共用。  \n"
                            "**雙色** — 每組（量測 / 模型）各一色，四個 S 參數共用。  \n"
                            "**自訂** — 每個（資料組, S 參數）組合各自挑色；預設量測用"
                            "原色盤、模型用較深版本。"))
    else:
        # Chart-only phase — read previously-set widget values direct from
        # session_state and skip widget creation entirely.
        smith_lw     = float(st.session_state.get(f"{skey}_smith_lw", 3.0))
        grid_lw      = float(st.session_state.get(f"{skey}_grid_lw",  1.0))
        grid_density = int(st.session_state.get(f"{skey}_grid_count",
                                                  _SMITH_GRID_DEFAULT))
        text_size    = float(st.session_state.get(f"{skey}_text_size", 18.0))
        color_mode   = str(st.session_state.get(f"{skey}_color_mode",
                                                  _COLOR_MODE_PER_TRACE))
    is_per_set_color = (color_mode == _COLOR_MODE_PER_SET)
    is_custom_color  = (color_mode == _COLOR_MODE_CUSTOM)

    # ── Custom-mode 8-picker grid (4 measured + 4 modeled) ───────────────
    def _hex_darken(h, by=45):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = max(0, r - by); g = max(0, g - by); b = max(0, b - by)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── Pre-seed custom-mode color keys regardless of phase ──────────────
    # Why: when the chart-only phase runs first (e.g. on the first SSM render
    # before the user opens the Controls expander), it would otherwise miss
    # these defaults and the per-(set, S-param) lookup at draw time would
    # fall back to the per-trace color.  Switching INTO custom mode would
    # then look like a "color reset" — pre-seeding makes the defaults stick
    # the moment the user picks Custom, without overwriting earlier edits.
    # Also: re-apply defaults whenever a custom-mode key has been
    # reset to "#000000" by widget remount (see _resolve_color above).
    if is_custom_color:
        for set_name, default_fn in [("meas",  lambda sp: _MPL_SMITH_COLORS[sp]),
                                      ("model", lambda sp: _hex_darken(_MPL_SMITH_COLORS[sp]))]:
            for sp in sparams:
                ck = f"{skey}_color_{set_name}_{sp}"
                _resolve_color(ck, default_fn(sp))

    if is_custom_color and _run_controls:
        with st.container(border=True):
            _ctrl_section(tr("Custom trace colors", "自訂曲線顏色"))
            st.caption(tr("Defaults: measured = legacy palette · modeled = "
                          "darker version. Your edits persist when switching "
                          "coloring modes.",
                          "預設：量測 = 原色盤 · 模型 = 較深版本。"
                          "切換配色模式時你的設定會保留。"))
            # Column header row (blank label cell + the four S-param names) so
            # each picker's S-param is unambiguous.
            _cust_head = st.columns([0.9, 1, 1, 1, 1])
            for _ci, _sp in enumerate(sparams):
                _cust_head[_ci + 1].markdown(
                    f"<span style='font-size:0.8em;color:#666;"
                    f"font-weight:600'>{_sp}</span>", unsafe_allow_html=True)
            for set_name, default_fn in [("meas",  lambda sp: _MPL_SMITH_COLORS[sp]),
                                          ("model", lambda sp: _hex_darken(_MPL_SMITH_COLORS[sp]))]:
                row_lbl = (tr("Measured", "量測") if set_name == "meas"
                           else tr("Modeled", "模型"))
                cust_cols = st.columns([0.9, 1, 1, 1, 1])
                cust_cols[0].markdown(f"**{row_lbl}**")
                for ci, sp in enumerate(sparams):
                    ck = f"{skey}_color_{set_name}_{sp}"
                    if ck not in st.session_state:
                        st.session_state[ck] = default_fn(sp)
                    cust_cols[ci + 1].color_picker(f"{row_lbl} {sp}",
                                                    key=ck,
                                                    label_visibility="collapsed")

    # ── Per-set table: trace / kind / style / size / [color] / decimate ──
    # Skip the whole widget table in chart-only phase — values are already
    # in session_state from the controls phase.
    # Show a Color column in bicolor mode OR whenever extra files are overlaid
    # (each added file gets its own color so the traces stay distinguishable).
    _show_color_col = is_per_set_color or _has_extra
    if _run_controls:
        if _show_color_col:
            _col_weights = [0.9, 1.0, 1.1, 0.8, 0.8, 0.8]
            _set_headers = ["Trace", "Kind", "Style", "Size",
                             "Color", "Decimate"]
        else:
            _col_weights = [0.9, 1.0, 1.1, 0.8, 0.8]
            _set_headers = ["Trace", "Kind", "Style", "Size", "Decimate"]

        # Header cells localize on render; the list itself keeps the canonical
        # English names because ``lbl == "Decimate"`` gates a column below.
        _set_header_zh = {"Trace": "曲線", "Kind": "類型", "Style": "樣式",
                          "Size": "大小", "Color": "顏色", "Decimate": "抽樣間隔"}
        _ctrl_section(tr("Measured / Modeled trace styling", "量測 / 模型曲線樣式"))
        with st.container(border=True):
            header_cols = st.columns(_col_weights)
            for i, lbl in enumerate(_set_headers):
                # Hide the Decimate header when there's no decimatable trace
                # (the cell is empty for non-measured / non-extra sets anyway).
                if lbl == "Decimate" and not (has_measured or _has_extra):
                    continue
                header_cols[i].markdown(
                    f"<span style='font-size:0.85em;color:#444;"
                    f"font-weight:600'>{tr(lbl, _set_header_zh[lbl])}</span>",
                    unsafe_allow_html=True)

            for si, s in enumerate(sets):
                kind_sk     = f"{skey}_set{si}_kind"
                style_sk    = f"{skey}_set{si}_style"
                size_sk     = f"{skey}_set{si}_size"
                setcolor_sk = f"{skey}_set{si}_color"
                dec_sk      = f"{skey}_set{si}_decimate"
                if kind_sk not in st.session_state:
                    st.session_state[kind_sk] = ("Line" if s.get("kind") == "line"
                                                 else "Markers")
                if style_sk not in st.session_state:
                    st.session_state[style_sk] = s.get(
                        "style", "solid" if s.get("kind") == "line" else "x")
                # Avoid black as a fallback — black is jarring on a Smith chart
                # and reads as "uninitialized" to users.  For unknown labels,
                # cycle through the trace palette by set index so each set
                # still gets a distinct sensible color.  Use _resolve_color so a
                # "#000000" left over from a widget remount is also treated as
                # uninitialized and replaced with the default.
                _palette_fallback = list(_MPL_SMITH_COLORS.values())[
                    si % len(_MPL_SMITH_COLORS)]
                _resolve_color(setcolor_sk,
                               _PER_SET_DEFAULT_COLORS.get(s.get("label"),
                                                            _palette_fallback))
                if dec_sk not in st.session_state:
                    st.session_state[dec_sk] = 1

                is_measured = (s.get("label") == "Measured")
                is_extra    = bool(s.get("_extra"))
                row_cols = st.columns(_col_weights)
                if is_extra:
                    # Column 0 becomes a file picker + remove button so the user
                    # can swap which uploaded file this overlay shows, or drop it.
                    j = s["_extra_idx"]
                    _used = set(st.session_state[_extra_sel_key]) - {s["label"]}
                    _opts = [l for l in _pool_labels if l not in _used]
                    pk = f"{skey}_extra_pick_{j}"
                    if pk not in st.session_state or st.session_state[pk] not in _opts:
                        st.session_state[pk] = (s["label"] if s["label"] in _opts
                                                else _opts[0])
                    pcol, xcol = row_cols[0].columns([4, 1])
                    pick = pcol.selectbox(f"File {si+1}", _opts, key=pk,
                                          label_visibility="collapsed")
                    if xcol.button("🗑", key=f"{skey}_extra_rm_{j}",
                                   help=tr("Remove this file from the chart",
                                           "從圖上移除此檔案")):
                        _cur = list(st.session_state[_extra_sel_key])
                        _cur.pop(j)
                        st.session_state[_extra_sel_key] = _cur
                        st.rerun()
                    if pick != s["label"]:
                        _cur = list(st.session_state[_extra_sel_key])
                        _cur[j] = pick
                        st.session_state[_extra_sel_key] = _cur
                        st.rerun()
                else:
                    row_cols[0].markdown(
                        f"**{s.get('label', f'Set {si+1}')}**")
                # Values stay "Markers"/"Line" — compared just below and
                # persisted in session_state; only the chip text localizes.
                kind = row_cols[1].selectbox(
                    f"Kind {si+1}", ["Markers", "Line"], key=kind_sk,
                    format_func=lambda k: tr(k, {"Markers": "標記",
                                                 "Line": "線條"}[k]),
                    label_visibility="collapsed")
                if kind == "Line":
                    opts = list(_MPL_LINE_STYLES.keys())
                    if st.session_state[style_sk] not in opts:
                        st.session_state[style_sk] = "solid"
                    size_default, size_max = 4.0, 10.0
                else:
                    opts = list(_MPL_MARKER_STYLES.keys())
                    if st.session_state[style_sk] not in opts:
                        st.session_state[style_sk] = "x"
                    size_default, size_max = 8.0, 30.0
                if size_sk not in st.session_state:
                    st.session_state[size_sk] = float(size_default)
                row_cols[2].selectbox(f"Style {si+1}", opts, key=style_sk,
                                       label_visibility="collapsed")
                row_cols[3].number_input(f"Size {si+1}",
                                          min_value=0.1, max_value=size_max,
                                          step=0.1, format="%.2f", key=size_sk,
                                          label_visibility="collapsed")
                _dec_ok = is_measured or is_extra
                if _show_color_col:
                    # One colour per set: bicolor mode, or per-file colouring
                    # whenever extra files are overlaid (every row — including
                    # the base file — gets a single colour picker).
                    row_cols[4].color_picker(f"Color {si+1}", key=setcolor_sk,
                                              label_visibility="collapsed")
                    if _dec_ok:
                        row_cols[5].number_input(f"Decimate {si+1}",
                                                  min_value=1, max_value=1000, step=1,
                                                  key=dec_sk,
                                                  label_visibility="collapsed")
                else:
                    if _dec_ok:
                        row_cols[4].number_input(f"Decimate {si+1}",
                                                  min_value=1, max_value=1000, step=1,
                                                  key=dec_sk,
                                                  label_visibility="collapsed")

            # ── "➕" — overlay another uploaded file's trace ────────────────
            if _pool:
                _avail = [l for l in _pool_labels
                          if l not in set(st.session_state[_extra_sel_key])]
                if _avail:
                    if st.button(tr("➕ Add a file trace", "➕ 新增檔案曲線"),
                                 key=f"{skey}_extra_add",
                                 help=tr("Overlay another uploaded S-parameter "
                                         "file on this Smith chart",
                                         "在此 Smith 圖上疊加另一個已上傳的 "
                                         "S 參數檔")):
                        _cur = list(st.session_state[_extra_sel_key])
                        _cur.append(_avail[0])
                        st.session_state[_extra_sel_key] = _cur
                        st.rerun()
                else:
                    st.caption(tr("All uploaded files are already on the chart.",
                                  "所有已上傳的檔案都已顯示在圖上。"))

    # ── Per-S-param table — S-param | Multiplier | Text | x | y | [Trace] | Text
    # Text color picker is ALWAYS shown (independent of color mode); the
    # Trace color picker appears only in per-trace coloring mode (in the
    # other modes the trace color comes from the per-set / 8-picker grid).
    if _run_controls:
        # Visual separation between this table and the Measured/Modeled
        # trace-styling table above it.
        st.markdown("<div style='height:10px'></div>",
                    unsafe_allow_html=True)
        _ctrl_section(tr("Per-S-parameter trace + label settings",
                         "各 S 參數曲線與標籤設定"))

        # Column widths — first column is a narrow label cell, "Text" input
        # is narrower than before, the rest balance out.  The per-trace "Trace"
        # colour column is dropped whenever the trace colour is decided per-set
        # (bicolor / custom) OR per-file (extra files overlaid) — in those modes
        # the user picks one colour per file, not per S-param.
        _sp_hide_trace = is_per_set_color or is_custom_color or _has_extra
        if _sp_hide_trace:
            sp_weights = [0.5, 0.7, 1.2, 0.7, 0.7, 0.7]
            sp_headers = [tr("S-param", "S 參數"), tr("Multiplier", "倍率"),
                          tr("Text", "文字"), tr("x pos", "x 位置"),
                          tr("y pos", "y 位置"), tr("Text", "文字")]
        else:
            sp_weights = [0.5, 0.7, 1.2, 0.7, 0.7, 0.7, 0.7]
            sp_headers = [tr("S-param", "S 參數"), tr("Multiplier", "倍率"),
                          tr("Text", "文字"), tr("x pos", "x 位置"),
                          tr("y pos", "y 位置"), tr("Trace", "曲線"),
                          tr("Text", "文字")]

        with st.container(border=True):
            head = st.columns(sp_weights)
            for i, lbl in enumerate(sp_headers):
                head[i].markdown(
                    f"<span style='font-size:0.85em;color:#444;"
                    f"font-weight:600'>{lbl}</span>",
                    unsafe_allow_html=True)

            for sp in sparams:
                row = st.columns(sp_weights)
                # S-param label cell — uppercase + subscript via mathtext for
                # consistency with the on-chart label rendering.
                row[0].markdown(
                    f"<div style='padding-top:6px;font-weight:600;"
                    f"font-size:0.95em'>{sp[0]}<sub>{sp[1:]}</sub></div>",
                    unsafe_allow_html=True)
                row[1].number_input(f"Multiplier {sp}",
                                    step=0.1, format="%.3f",
                                    key=f"{skey}_mult_{sp}",
                                    on_change=_sync_text_to_mult,
                                    args=(skey, sp),
                                    label_visibility="collapsed",
                                    help=tr("Scales this S-param trace.  The Text "
                                            "label auto-updates: >1 → “Sxx×N”, "
                                            "<1 → “Sxx/N”.",
                                            "縮放此 S 參數曲線。文字標籤會自動更新："
                                            ">1 → 「Sxx×N」，<1 → 「Sxx/N」。"))
                row[2].text_input(f"{sp} text", key=f"{skey}_text_{sp}",
                                  label_visibility="collapsed")
                row[3].number_input(f"{sp} x position",
                                    step=0.05, format="%.3f",
                                    key=f"{skey}_x_{sp}",
                                    label_visibility="collapsed")
                row[4].number_input(f"{sp} y position",
                                    step=0.05, format="%.3f",
                                    key=f"{skey}_y_{sp}",
                                    label_visibility="collapsed")
                if _sp_hide_trace:
                    # No trace column in these modes → text-color picker
                    # sits in column index 5.
                    row[5].color_picker(f"{sp} text color",
                                        key=f"{skey}_text_color_{sp}",
                                        label_visibility="collapsed",
                                        help=tr("On-chart label color "
                                                "(default = darker variant of "
                                                "the trace color).",
                                                "圖上標籤顏色（預設為曲線顏色的"
                                                "較深版本）。"))
                else:
                    row[5].color_picker(f"{sp} color",
                                        key=f"{skey}_color_{sp}",
                                        label_visibility="collapsed",
                                        help=tr("Trace color for this "
                                                "S-parameter.",
                                                "此 S 參數的曲線顏色。"))
                    row[6].color_picker(f"{sp} text color",
                                        key=f"{skey}_text_color_{sp}",
                                        label_visibility="collapsed",
                                        help=tr("On-chart label color "
                                                "(default = darker variant of "
                                                "the trace color).",
                                                "圖上標籤顏色（預設為曲線顏色的"
                                                "較深版本）。"))

        # One-click auto-placement: drop every S-param label just outside its
        # own trace (centroid direction, beyond the curve's outer extent) so the
        # text sits beside the curve instead of on top of it.  Must run via
        # on_click — the x/y number_inputs above are already instantiated this
        # run, so their session_state can only be mutated before the next run.
        def _auto_place_labels():
            for _sp in sparams:
                try:
                    _r, _c = _SP_RC[_sp]
                    _mult = float(st.session_state.get(
                        f"{skey}_mult_{_sp}", default_mults[_sp]))
                    _z = np.asarray(sets[0]["S"])[:, _r, _c] * _mult
                    _z = _z[np.isfinite(_z)]
                    if not _z.size:
                        continue
                    _ctr = complex(float(np.mean(_z.real)),
                                   float(np.mean(_z.imag)))
                    _rad = abs(_ctr)
                    _d = (_ctr / _rad) if _rad > 1e-6 else (1 + 0j)
                    # Outermost extent of the trace along the centroid direction,
                    # then nudge a margin further out so the label clears it.
                    _tip = float(np.max(_z.real * _d.real + _z.imag * _d.imag))
                    _pos = (_tip + 0.12) * _d
                    st.session_state[f"{skey}_x_{_sp}"] = float(
                        np.clip(_pos.real, -1.1, 1.1))
                    st.session_state[f"{skey}_y_{_sp}"] = float(
                        np.clip(_pos.imag, -1.1, 1.1))
                except Exception:                        # noqa: BLE001
                    continue
            # The frequency-range caption stays pinned at the bottom (0.0, -1.1)
            # and is NOT rearranged with the S-param labels — reset it to that
            # default spot so one click restores the standard layout.
            if freq_hz is not None:
                try:
                    _f = np.asarray(freq_hz, dtype=float)
                    _f = _f[np.isfinite(_f)]
                    if _f.size >= 2:
                        _lo = float(_f.min()) * 1e-9
                        _hi = float(_f.max()) * 1e-9
                        _slot = st.session_state.get(f"{skey}_freq_slot")
                        if _slot is None:
                            _slot = int(st.session_state.get(extra_key, 0))
                            st.session_state[extra_key] = _slot + 1
                            st.session_state[f"{skey}_freq_slot"] = _slot
                        st.session_state[f"{skey}_etext_{_slot}"] = f"{_lo:g} ~ {_hi:g} GHz"
                        st.session_state[f"{skey}_ex_{_slot}"] = 0.0
                        st.session_state[f"{skey}_ey_{_slot}"] = -1.1
                        st.session_state.setdefault(f"{skey}_ecolor_{_slot}", "#000000")
                        st.session_state[f"{skey}_freq_seed_sig"] = f"{_lo:g}~{_hi:g}"
                except (TypeError, ValueError):
                    pass

        # Label-placement actions grouped in one row — one-click auto-arrange
        # on the left, the interactive-drag switch on the right — instead of
        # two full-width controls stacked loosely under the table.
        act_place, act_drag = st.columns(2)
        act_place.button(
            tr("🎯 Auto-place labels", "🎯 自動排列標籤"),
            key=f"{skey}_autoplace_btn",
            on_click=_auto_place_labels,
            width="stretch",
            help=tr("Move each S-parameter label next to its trace — just outside "
                    "the curve so the text does not overlap it.",
                    "將每個 S 參數標籤移到對應曲線旁 — 落在曲線外側，"
                    "文字不會蓋住曲線。"))
        drag_on = act_drag.toggle(
            tr("🖱️ Drag labels (interactive)", "🖱️ 拖曳標籤（互動）"),
            key=f"{skey}_drag_preview",
            help=tr("Drag the S-parameter labels on an interactive Smith chart, "
                    "then click “Render matplotlib” to draw the publication "
                    "figure at the dragged positions. The frequency caption "
                    "stays pinned at the bottom.",
                    "在互動式 Smith 圖上拖曳 S 參數標籤，再按「產生 matplotlib 圖」"
                    "以拖曳後的位置繪製出版用圖。頻率標註固定於底部。"))

        # ── Text annotations panel ────────────────────────────────────────
        # Free text labels (the default freq-range caption is seeded phase-
        # independently near the top of this function).
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _ctrl_section(tr("Text annotations", "文字標註"))
        n_extra = int(st.session_state.get(extra_key, 0))
        with st.container(border=True):
            if n_extra > 0:
                # One header row instead of a repeated visible label on every
                # input — the rows below use collapsed labels and line up under
                # these headers.
                _ah = st.columns([2, 1, 1, 1])
                for _ai, _al in enumerate([tr("Text", "文字"), "x", "y",
                                           tr("Color", "顏色")]):
                    _ah[_ai].markdown(
                        f"<span style='font-size:0.8em;color:#666;"
                        f"font-weight:600'>{_al}</span>",
                        unsafe_allow_html=True)
            else:
                st.caption(tr("No text annotations yet — add one below.",
                              "尚無文字標註 — 於下方新增。"))
            for i in range(n_extra):
                st.session_state.setdefault(f"{skey}_etext_{i}",  "")
                st.session_state.setdefault(f"{skey}_ex_{i}",     0.0)
                st.session_state.setdefault(f"{skey}_ey_{i}",     0.0)
                st.session_state.setdefault(f"{skey}_ecolor_{i}", "#000000")
                ec1, ec2, ec3, ec4 = st.columns([2, 1, 1, 1])
                ec1.text_input(tr(f"Text {i+1}", f"文字 {i+1}"),
                               key=f"{skey}_etext_{i}",
                               label_visibility="collapsed")
                ec2.number_input(f"x {i+1}", step=0.05, format="%.3f",
                                 key=f"{skey}_ex_{i}",
                                 label_visibility="collapsed")
                ec3.number_input(f"y {i+1}", step=0.05, format="%.3f",
                                 key=f"{skey}_ey_{i}",
                                 label_visibility="collapsed")
                ec4.color_picker(tr(f"Color {i+1}", f"顏色 {i+1}"),
                                 key=f"{skey}_ecolor_{i}",
                                 label_visibility="collapsed")

            def _add_text_slot():
                st.session_state[extra_key] = int(st.session_state[extra_key]) + 1
            st.button(tr("➕ Add text", "➕ 新增文字"), key=f"{skey}_add_btn",
                      on_click=_add_text_slot, width="stretch")
    else:
        # Chart-only phase: just read how many free text slots exist
        n_extra = int(st.session_state.get(extra_key, 0))

    # ── Resolve per-S-param config (linked color/mult/text/position) ─────────
    sparam_cfg = {}
    for sp in sparams:
        sparam_cfg[sp] = {
            "mult":       float(st.session_state[f"{skey}_mult_{sp}"]),
            "color":      str(st.session_state[f"{skey}_color_{sp}"]),
            "text":       str(st.session_state[f"{skey}_text_{sp}"] or ""),
            "x":          float(st.session_state[f"{skey}_x_{sp}"]),
            "y":          float(st.session_state[f"{skey}_y_{sp}"]),
            "text_color": str(st.session_state[f"{skey}_text_color_{sp}"]),
        }

    # Skip the rest in controls-only phase — the chart half of the split
    # layout will render the figure + legend from the same session state.
    if not _run_chart:
        return

    # ── Drag-labels mode: interactive custom component + "Render matplotlib" ──
    # When the "Drag labels" toggle is on, render a draggable-annotation Smith
    # chart (a custom no-build component that streams dragged positions back to
    # Python) and hold off drawing the matplotlib figure until the user clicks
    # "Render matplotlib".  Dragged positions are stashed as a "pending" update
    # and committed into the real x/y state at the TOP of the next run (see the
    # commit block after the seeding loop), so they survive the toggle being
    # switched off.  The frequency caption is drawn as a FIXED text trace (not a
    # draggable annotation) so it stays pinned at the bottom.  `return_png` (the
    # silent topology-overlay PNG path) skips all of this.
    if drag_on and not return_png:
        _freq_slot = st.session_state.get(f"{skey}_freq_slot")
        try:
            import json as _json
            from .helpers import extended_smith_grid
            from .components import draggable_smith

            dfig = go.Figure()
            # Smith-chart grid (outer unit circle + constant-R circles +
            # constant-X arcs).  extended_smith_grid returns Scattergl (WebGL)
            # traces, which render blank inside the sandboxed component iframe —
            # rebuild them as plain SVG go.Scatter so the circles actually show.
            for _bg in extended_smith_grid(1.0):
                dfig.add_trace(go.Scatter(
                    x=_bg.x, y=_bg.y, mode="lines",
                    line=dict(color=_bg.line.color, width=_bg.line.width),
                    hoverinfo="skip", showlegend=False))

            # Traces — mirror the matplotlib draw loop's colour resolution.
            for si, s in enumerate(sets):
                S = s["S"]
                kind = st.session_state.get(
                    f"{skey}_set{si}_kind",
                    "Line" if s.get("kind") == "line" else "Markers")
                set_color = str(st.session_state.get(
                    f"{skey}_set{si}_color",
                    _PER_SET_DEFAULT_COLORS.get(s.get("label"), "#000000")))
                is_measured = (s.get("label") == "Measured")
                decimate = int(st.session_state.get(f"{skey}_set{si}_decimate", 1))
                p_mode = "lines" if kind == "Line" else "markers"
                for sp, (r, c) in [("S11", (0, 0)), ("S12", (0, 1)),
                                   ("S21", (1, 0)), ("S22", (1, 1))]:
                    cfg = sparam_cfg[sp]
                    sv = S[:, r, c] * cfg["mult"]
                    if (is_measured or s.get("_extra")) and decimate > 1:
                        sv = sv[::decimate]
                    if _has_extra:
                        color = set_color
                    elif is_custom_color:
                        _set_name = "meas" if is_measured else "model"
                        color = str(st.session_state.get(
                            f"{skey}_color_{_set_name}_{sp}", cfg["color"]))
                    elif is_per_set_color:
                        color = set_color
                    else:
                        color = cfg["color"]
                    p_kwargs = dict(x=list(sv.real), y=list(sv.imag), mode=p_mode,
                                    showlegend=False, hoverinfo="skip")
                    if p_mode == "lines":
                        p_kwargs["line"] = dict(color=color, width=2)
                    else:
                        p_kwargs["marker"] = dict(color=color, size=6)
                    dfig.add_trace(go.Scatter(**p_kwargs))

            # Draggable labels, in a KNOWN order (`_anno_order`) so the index the
            # component streams back maps to the right state key.  These are NOT
            # Plotly annotations — they become independent HTML overlay <div>s in
            # the component, so dragging one can never move another.  The freq
            # caption is EXCLUDED (fixed trace below), so it is never dragged.
            _anno_order = []
            _labels = []
            for sp in sparams:
                cfg = sparam_cfg[sp]
                if cfg["text"]:
                    _labels.append({"x": cfg["x"], "y": cfg["y"],
                                    "text": cfg["text"], "color": cfg["text_color"],
                                    "size": text_size})
                    _anno_order.append(("sp", sp))
            for i in range(n_extra):
                if i == _freq_slot:
                    continue
                _txt = str(st.session_state.get(f"{skey}_etext_{i}", "") or "")
                if not _txt:
                    continue
                _labels.append({
                    "x": float(st.session_state.get(f"{skey}_ex_{i}", 0.0)),
                    "y": float(st.session_state.get(f"{skey}_ey_{i}", 0.0)),
                    "text": _txt,
                    "color": str(st.session_state.get(f"{skey}_ecolor_{i}", "#000000")),
                    "size": text_size})
                _anno_order.append(("extra", i))

            # Frequency caption → FIXED text trace, pinned at its bottom spot.
            if _freq_slot is not None:
                _ftxt = str(st.session_state.get(f"{skey}_etext_{_freq_slot}", "") or "")
                if _ftxt:
                    dfig.add_trace(go.Scatter(
                        x=[float(st.session_state.get(f"{skey}_ex_{_freq_slot}", 0.0))],
                        y=[float(st.session_state.get(f"{skey}_ey_{_freq_slot}", -1.1))],
                        mode="text", text=[_ftxt],
                        textfont=dict(color=str(st.session_state.get(
                            f"{skey}_ecolor_{_freq_slot}", "#000000")),
                            size=text_size),
                        hoverinfo="skip", showlegend=False))

            dfig.update_layout(
                xaxis=dict(range=[-1.15, 1.15], visible=False, showgrid=False,
                           zeroline=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-1.15, 1.15], visible=False, showgrid=False,
                           zeroline=False),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10), showlegend=False)

            st.caption(tr(
                "Drag the S-parameter labels, then click **Render matplotlib** "
                "below to draw the figure at those positions. The frequency "
                "caption is pinned at the bottom.",
                "拖曳 S 參數標籤，再按下方的**產生 matplotlib 圖**以該位置繪圖。"
                "頻率標註固定於底部。"))

            # Render the draggable overlay.  Its returned value (the single
            # dragged label {"index","x","y","nonce"}) is committed at the TOP of
            # the NEXT run, read straight from session_state — so we only need to
            # remember the label order here.  ``_anno_order`` is the exact order
            # of the labels on screen now, so next run's commit maps the returned
            # index to the right state key even if the label set later changes.
            draggable_smith(figure=_json.loads(dfig.to_json()),
                            labels=_labels, height=560,
                            key=f"{skey}_drag_chart")
            st.session_state[f"{skey}_drag_order"] = _anno_order
        except Exception:                                # noqa: BLE001
            st.caption(tr(
                "Interactive drag view unavailable — use the x/y number inputs "
                "or 🎯 Auto-place labels instead.",
                "互動拖曳檢視無法使用 — 請改用 x/y 數值輸入或 🎯 自動排列標籤。"))

        # Only draw the matplotlib figure once the user asks — at the positions
        # committed so far (already written into the real x/y state).
        if not st.button(tr("🖼️ Render matplotlib", "🖼️ 產生 matplotlib 圖"),
                         key=f"{skey}_render_mpl_btn", type="primary"):
            return

    # ── Build the matplotlib figure ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    _draw_mpl_smith_background(ax, smith_lw, grid_lw, density=int(grid_density))

    for si, s in enumerate(sets):
        S = s["S"]
        kind  = st.session_state.get(f"{skey}_set{si}_kind",
                                      "Line" if s.get("kind") == "line"
                                      else "Markers")
        style = st.session_state.get(f"{skey}_set{si}_style",
                                      s.get("style", "solid"))
        size_val = float(st.session_state.get(
            f"{skey}_set{si}_size", 2.0 if kind == "Line" else 6.0))
        set_color = str(st.session_state.get(
            f"{skey}_set{si}_color",
            _PER_SET_DEFAULT_COLORS.get(s.get("label"), "#000000")))
        is_measured = (s.get("label") == "Measured")
        decimate = int(st.session_state.get(f"{skey}_set{si}_decimate", 1))
        for sp, (r, c) in [("S11", (0, 0)), ("S12", (0, 1)),
                           ("S21", (1, 0)), ("S22", (1, 1))]:
            cfg = sparam_cfg[sp]
            sv  = S[:, r, c] * cfg["mult"]
            if (is_measured or s.get("_extra")) and decimate > 1:
                sv = sv[::decimate]
            if _has_extra:
                # Overlay mode: one colour per file (base + every added file),
                # taken from that set's styling row — no per-S-param colouring.
                color = set_color
            elif is_custom_color:
                # Pick the per-(set, S-param) color stashed by the 8-picker grid.
                _set_name = "meas" if is_measured else "model"
                color = str(st.session_state.get(
                    f"{skey}_color_{_set_name}_{sp}", cfg["color"]))
            elif is_per_set_color:
                color = set_color
            else:
                color = cfg["color"]
            if kind == "Line":
                ls = _MPL_LINE_STYLES.get(style, "-")
                ax.plot(sv.real, sv.imag, linestyle=ls, color=color,
                        linewidth=size_val)
            else:
                mk = _MPL_MARKER_STYLES.get(style, "o")
                if style in _MPL_UNFILLED_MARKERS:
                    # Unfilled markers (e.g. "x") need a visible edge to render.
                    mew = max(1.0, size_val * 0.18)
                    ax.plot(sv.real, sv.imag, marker=mk, color=color,
                            markersize=size_val, markeredgewidth=mew,
                            markeredgecolor=color,
                            linestyle="None", alpha=0.85)
                else:
                    ax.plot(sv.real, sv.imag, marker=mk, color=color,
                            markersize=size_val, markeredgewidth=0,
                            linestyle="None", alpha=0.85)

    # Text auto-format pipeline — always on, fixed S-param subscript rule.
    def _fmt_text(t: str) -> str:
        return _apply_text_autoformat(
            t, _DEFAULT_AUTOFMT_PATTERN, _DEFAULT_AUTOFMT_REPLACEMENT)

    # S-param text annotations (color linked to trace).  Regular weight —
    # bold would only apply to non-mathtext parts and mismatch with the
    # auto-formatted mathtext spans, giving e.g. bold "/5" next to regular
    # "S₁₂".  Keep everything visually uniform instead.
    # Positions come straight from the committed x/y state — hand-dragged
    # positions are committed there at the top of the run, so they render the
    # same whether the "Drag labels" toggle is on or off.
    _ts = float(text_size)
    for sp in sparams:
        cfg = sparam_cfg[sp]
        if cfg["text"]:
            ax.text(cfg["x"], cfg["y"], _fmt_text(cfg["text"]),
                    ha="center", va="center", fontsize=_ts,
                    color=cfg["text_color"], zorder=5)

    # Free text annotations
    for i in range(n_extra):
        txt = str(st.session_state.get(f"{skey}_etext_{i}", "") or "")
        if not txt:
            continue
        ax.text(float(st.session_state.get(f"{skey}_ex_{i}", 0.0)),
                float(st.session_state.get(f"{skey}_ey_{i}", 0.0)),
                _fmt_text(txt), ha="center", va="center", fontsize=_ts,
                color=str(st.session_state.get(f"{skey}_ecolor_{i}", "#000000")),
                zorder=5)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # When asked, hand the *customized* figure back as PNG bytes (used by the
    # topology illustration's bottom-right Smith overlay) instead of drawing it
    # into Streamlit.  Returns before any st.* call so no UI element is created.
    if return_png:
        import io as _io
        _buf = _io.BytesIO()
        fig.savefig(_buf, format="png", bbox_inches="tight",
                    facecolor="white", dpi=150)
        plt.close(fig)
        return _buf.getvalue()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_parts = []
    for si, s in enumerate(sets):
        kind  = st.session_state.get(f"{skey}_set{si}_kind",
                                     "Line" if s.get("kind") == "line"
                                     else "Markers")
        style = st.session_state.get(f"{skey}_set{si}_style",
                                     s.get("style", "solid"))
        label = s.get("label", f"Set {si+1}")
        if is_per_set_color or _has_extra or s.get("_extra"):
            set_col = str(st.session_state.get(f"{skey}_set{si}_color", "#000"))
            legend_parts.append(
                f"<span style='color:{set_col}'>**{label}**</span> "
                f"({kind.lower()} · {style})")
        else:
            legend_parts.append(f"**{label}** ({kind.lower()} · {style})")
    if is_custom_color:
        # Two color-chip rows (measured / modeled) — one swatch per S-param.
        rows = []
        for set_name, set_label in [("meas", "Measured"), ("model", "Modeled")]:
            chip_strs = []
            for sp in sparams:
                cval = st.session_state.get(
                    f"{skey}_color_{set_name}_{sp}",
                    _MPL_SMITH_COLORS[sp])
                chip_strs.append(
                    f"<span style='color:{cval}'>**{sp}**</span>")
            rows.append(f"**{set_label}** — " + " · ".join(chip_strs))
        st.markdown(
            "**Legend** — " + " ; ".join(legend_parts) + "  \n" +
            "  \n".join(rows),
            unsafe_allow_html=True,
        )
    elif is_per_set_color or _has_extra:
        # One colour per set/file → the per-set chips already convey the colour;
        # no per-S-param "Trace colors" row.
        st.markdown("**Legend** — " + " ; ".join(legend_parts),
                    unsafe_allow_html=True)
    else:
        color_chips = " · ".join(
            f"<span style='color:{sparam_cfg[sp]['color']}'>**{sp}**</span>"
            for sp in sparams)
        st.markdown(
            "**Legend** — " + " ; ".join(legend_parts) +
            "  |  Trace colors: " + color_chips,
            unsafe_allow_html=True,
        )


def render_ft_fmax_overlay(S_raw, sim_results: dict[str, np.ndarray], freq, fname):
    """
    Bode plot: |h21|² and Mason U for measured + all simulated models.

    Standardised colour / dash convention (FT_FMAX_COLORS):
      |h21|² (fT)   → blue       Mason U (fmax) → red
      measured     : solid + markers (circle for h21, square for U)
      modeled      : dashed (one model per metric — when multiple models
                     are passed they share the colour and are
                     distinguished by name in the legend)
      extrap       : dotted (in the metric's colour, hidden from legend)

    sim_results: {model_SHORT: S_sim_array}  (None values are skipped)
    """
    f_ghz = freq * 1e-9
    h21_mea, U_mea = compute_h21_U(S_raw)

    _c_fT, _c_fmax = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]
    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used  = False
    # Collected for the standardised fT/fmax xlsx export (simulated +
    # extrapolated columns).  Labels match across the two lists.
    sim_traces:    list[tuple[str, np.ndarray]] = []
    extrap_traces: list[tuple[str, np.ndarray, np.ndarray]] = []

    # ── Measured traces (markers + solid line, full colour intensity) ──
    fig.add_trace(go.Scattergl(
        x=f_ghz, y=h21_mea, mode="lines+markers",
        name="|h21|² Meas.",
        line=dict(color=_c_fT, width=1.4),
        marker=dict(symbol=FT_FMAX_SYMBOLS["h21"], size=6, color=_c_fT)))
    sim_traces.append(("|h21|² Meas. (dB)", h21_mea))
    f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, h21_mea)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, f0)
        extrap_traces.append(("|h21|² Meas. (dB)", f_ext, g_ext))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines",
            name=f"|h21|² Meas. extrap (fT≈{f0:.1f} GHz)",
            line=dict(color=_c_fT, width=1.6, dash="dot"), showlegend=False))

    fig.add_trace(go.Scattergl(
        x=f_ghz, y=U_mea, mode="lines+markers",
        name="Mason U Meas.",
        line=dict(color=_c_fmax, width=1.4),
        marker=dict(symbol=FT_FMAX_SYMBOLS["U"], size=6, color=_c_fmax)))
    sim_traces.append(("Mason U Meas. (dB)", U_mea))
    f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, U_mea)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, f0)
        extrap_traces.append(("Mason U Meas. (dB)", f_ext, g_ext))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines",
            name=f"Mason U Meas. extrap (fmax≈{f0:.1f} GHz)",
            line=dict(color=_c_fmax, width=1.6, dash="dot"), showlegend=False))

    # ── Modeled traces — colour-by-metric, models share colour and are
    # distinguished by name (one row per model in the legend).
    for short, S_sim in sim_results.items():
        if S_sim is None:
            continue
        h21_s, U_s = compute_h21_U(S_sim)

        fig.add_trace(go.Scattergl(
            x=f_ghz, y=h21_s, mode="lines",
            name=f"|h21|² {short}",
            line=dict(color=_c_fT, width=2.0, dash="dash")))
        sim_traces.append((f"|h21|² {short} (dB)", h21_s))
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, h21_s)
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            extrap_traces.append((f"|h21|² {short} (dB)", f_ext, g_ext))
            fig.add_trace(go.Scattergl(
                x=f_ext, y=g_ext, mode="lines",
                name=f"|h21|² {short} extrap (fT≈{f0:.1f} GHz)",
                line=dict(color=_c_fT, width=2.0, dash="dot"),
                showlegend=False))

        fig.add_trace(go.Scattergl(
            x=f_ghz, y=U_s, mode="lines",
            name=f"Mason U {short}",
            line=dict(color=_c_fmax, width=2.0, dash="dash")))
        sim_traces.append((f"Mason U {short} (dB)", U_s))
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, U_s)
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            extrap_traces.append((f"Mason U {short} (dB)", f_ext, g_ext))
            fig.add_trace(go.Scattergl(
                x=f_ext, y=g_ext, mode="lines",
                name=f"Mason U {short} extrap (fmax≈{f0:.1f} GHz)",
                line=dict(color=_c_fmax, width=2.0, dash="dot"),
                showlegend=False))

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1])
    fig.update_layout(
        title="Gain vs Frequency — Measured vs Modeled",
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=450,
        legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=18)),
        hovermode="x unified", margin=dict(l=55, r=20, t=50, b=50))
    plotly_with_dl(fig, key=f"ftfmax_{fname}", filename=f"ftfmax_{fname}",
                   excel_bytes=bode_excel_bytes(f_ghz, sim_traces, extrap_traces))
    cap = ("Measured: ○ = |h21|², □ = Mason U.   Modeled: dashed lines.   "
           "Y-axis fixed 0–50 dB.")
    if extrap_used:
        cap += "   Dotted = 20 dB/dec extrapolation past the measured band."
    st.caption(cap)


# ════════════════════════════════════════════════════════════════════════════════
# Helper: Re(Z12) vs 1/IE
# ════════════════════════════════════════════════════════════════════════════════
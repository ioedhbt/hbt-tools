"""
batch_deembedding.py — Batch SSM-style de-embedding tab.

Extracts pad capacitances from the modeled Open dummy and lead inductances
from the modeled Short dummy (Gao 2015 §4.2 — same calculations as
``ssm_deembedding.step_open`` / ``step_short``), lets the user override any
element, then applies the de-embedding to every loaded DUT file.

Exposed entry point:
    render_batch_deembedding_tab(...)
"""
from __future__ import annotations

import io, re, zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from tools.SSM.ssm_deembedding import step_open, step_short, peel_parasitics
from tools.SSM.ssm_s2p          import write_s2p


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _cap_vs_f_fig(arrays_open, freq):
    f_GHz = freq * 1e-9
    fig = go.Figure()
    if arrays_open is not None:
        for key, color in [("Cpbe", "#1f77b4"),
                            ("Cpce", "#ff7f0e"),
                            ("Cpbc", "#2ca02c")]:
            arr = np.asarray(arrays_open[key]) * 1e15  # → fF
            fig.add_trace(go.Scatter(
                x=f_GHz, y=arr, mode="lines", name=key,
                line=dict(color=color, width=2),
                hovertemplate=f"{key}=%{{y:.4f}} fF<br>f=%{{x:.3f}} GHz<extra></extra>"))
    else:
        fig.add_annotation(text="No Open file uploaded — capacitances default to 0",
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(color="#888"))
    fig.update_layout(
        title=dict(text="Pad Capacitance vs Frequency (Open dummy)", font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)", showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Capacitance (fF)", showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#ccc", borderwidth=1),
        margin=dict(l=55, r=20, t=45, b=50))
    return fig


def _ind_vs_f_fig(arrays_short, freq):
    f_GHz = freq * 1e-9
    fig = go.Figure()
    if arrays_short is not None:
        for key, color in [("Lb", "#1f77b4"),
                            ("Lc", "#ff7f0e"),
                            ("Le", "#2ca02c")]:
            arr = np.asarray(arrays_short[key]) * 1e12  # → pH
            fig.add_trace(go.Scatter(
                x=f_GHz, y=arr, mode="lines", name=key,
                line=dict(color=color, width=2),
                hovertemplate=f"{key}=%{{y:.4f}} pH<br>f=%{{x:.3f}} GHz<extra></extra>"))
    else:
        fig.add_annotation(text="No Short file uploaded — inductances default to 0",
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(color="#888"))
    fig.update_layout(
        title=dict(text="Lead Inductance vs Frequency (Short dummy)", font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)", showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Inductance (pH)", showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#ccc", borderwidth=1),
        margin=dict(l=55, r=20, t=45, b=50))
    return fig


# ── Override-input helper ─────────────────────────────────────────────────────

def _ovr_input(col, label, default_si, scale, fmt, state_key):
    """number_input that initialises from `default_si * scale` once, then
    persists user edits in session state."""
    if state_key not in st.session_state:
        st.session_state[state_key] = float(default_si) * scale
    val = col.number_input(label, format=fmt, key=state_key)
    return float(val) / scale


# ── Main entry point ──────────────────────────────────────────────────────────

def render_batch_deembedding_tab(*, all_data, open_data, short_data,
                                   ui, helpers):
    """
    Parameters
    ----------
    all_data    : dict[name -> {S_raw, freq, z0, ...}]
    open_data   : (freq, S, z0) tuple or None
    short_data  : (freq, S, z0) tuple or None
    ui          : dict with keys
                    freq_min, freq_max, db_min, db_max, n_pts,
                    sh21, su, smag,
                    smith_f_min, smith_f_max, smith_max_r,
                    toggles, scales
    helpers     : dict with keys
                    y_to_s, compute_metrics, extract_limit,
                    make_bode, make_smith, card, PALETTE
    """
    st.markdown("### 🧰 Batch De-embedding")
    st.caption(
        "SSM-style modeled de-embedding (Gao 2015 §4.2): pad capacitances are "
        "extracted from the Open dummy and series-lead inductances from the "
        "Short dummy. Override any element below to retune; if no Open/Short "
        "is uploaded the corresponding parameters default to 0.")

    if not all_data:
        st.info("Upload DUT files to begin batch de-embedding.")
        return

    y_to_s         = helpers["y_to_s"]
    compute_metrics = helpers["compute_metrics"]
    extract_limit  = helpers["extract_limit"]
    make_bode      = helpers["make_bode"]
    make_smith     = helpers["make_smith"]
    card           = helpers["card"]
    PALETTE        = helpers["PALETTE"]

    # ── 1. Defaults from Open / Short dummies ─────────────────────────────────
    params_open  = {"Cpbe": 0.0, "Cpce": 0.0, "Cpbc": 0.0}
    arrays_open  = None
    params_short = {"Lb": 0.0, "Lc": 0.0, "Le": 0.0,
                     "Rpb": 0.0, "Rpc": 0.0, "Rpe": 0.0}
    arrays_short = None

    if open_data is not None:
        try:
            params_open, arrays_open = step_open(open_data)
        except Exception as e:
            st.error(f"Open processing failed: {e}")

    if short_data is not None:
        try:
            f_short = short_data[0]
            params_short, arrays_short = step_short(
                short_data, f_short,
                params_open["Cpbe"], params_open["Cpce"], params_open["Cpbc"],
                open_data=open_data,
                measured_open=(open_data is not None))
        except Exception as e:
            st.error(f"Short processing failed: {e}")

    # ── 2. Side-by-side C-vs-f and L-vs-f plots ───────────────────────────────
    f_open  = open_data[0]  if open_data  is not None else next(iter(all_data.values()))["freq"]
    f_short = short_data[0] if short_data is not None else next(iter(all_data.values()))["freq"]
    cc, cl = st.columns(2)
    cc.plotly_chart(_cap_vs_f_fig(arrays_open,  f_open),  use_container_width=True)
    cl.plotly_chart(_ind_vs_f_fig(arrays_short, f_short), use_container_width=True)

    # ── 3. Override section ───────────────────────────────────────────────────
    head_l, head_r = st.columns([4, 1])
    head_l.markdown("#### Overrides")
    if head_r.button("↺ Reset to defaults", key="bd_reset",
                      help="Restore values computed from Open/Short."):
        for k in ("bd_Cpbe","bd_Cpce","bd_Cpbc",
                  "bd_Lb","bd_Lc","bd_Le",
                  "bd_Rb","bd_Re","bd_Rc"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption("Defaults are populated from the modeled Open/Short calculation.")

    c1, c2, c3 = st.columns(3)
    Cpbe = _ovr_input(c1, "Cpbe (fF)", params_open["Cpbe"], 1e15, "%.4f", "bd_Cpbe")
    Cpce = _ovr_input(c2, "Cpce (fF)", params_open["Cpce"], 1e15, "%.4f", "bd_Cpce")
    Cpbc = _ovr_input(c3, "Cpbc (fF)", params_open["Cpbc"], 1e15, "%.4f", "bd_Cpbc")

    c4, c5, c6 = st.columns(3)
    Lb = _ovr_input(c4, "Lb (pH)", params_short["Lb"], 1e12, "%.4f", "bd_Lb")
    Lc = _ovr_input(c5, "Lc (pH)", params_short["Lc"], 1e12, "%.4f", "bd_Lc")
    Le = _ovr_input(c6, "Le (pH)", params_short["Le"], 1e12, "%.4f", "bd_Le")

    c7, c8, c9 = st.columns(3)
    Rb = _ovr_input(c7, "Rb (Ω)", 0.0, 1.0, "%.4f", "bd_Rb")
    Re = _ovr_input(c8, "Re (Ω)", 0.0, 1.0, "%.4f", "bd_Re")
    Rc = _ovr_input(c9, "Rc (Ω)", 0.0, 1.0, "%.4f", "bd_Rc")

    p_eff = dict(
        Cpbe=Cpbe, Cpce=Cpce, Cpbc=Cpbc,
        Lb=Lb, Lc=Lc, Le=Le,
        Rpb=Rb, Rpc=Rc, Rpe=Re,
    )

    # ── 4. Apply de-embedding to every DUT file ───────────────────────────────
    bd_results = {}
    bd_errors  = {}
    for name, d in all_data.items():
        try:
            Y_de  = peel_parasitics(d["S_raw"], d["freq"], d["z0"], p_eff)
            S_de  = y_to_s(Y_de, d["z0"])
            df_de = compute_metrics(Y_de, d["freq"])
            f_arr = df_de["Freq (GHz)"].values
            fT_cr,  fT_pl,  fT_m  = extract_limit(
                f_arr, df_de["|h21|² (dB)"].values,
                df_de["fT Plateau (GHz)"].values,
                ui["n_pts"], ui["freq_min"], ui["freq_max"])
            fmU_cr, fmU_pl, fmU_m = extract_limit(
                f_arr, df_de["Mason U (dB)"].values,
                df_de["fmax U Plateau (GHz)"].values,
                ui["n_pts"], ui["freq_min"], ui["freq_max"])
            fmM_cr, fmM_pl, fmM_m = extract_limit(
                f_arr, df_de["MAG/MSG (dB)"].values,
                df_de["fmax MAG Plateau (GHz)"].values,
                ui["n_pts"], ui["freq_min"], ui["freq_max"])
            bd_results[name] = dict(
                S_de=S_de, df_de=df_de, freq=d["freq"], z0=d["z0"],
                fT_cr=fT_cr,   fT_pl=fT_pl,   fT_m=fT_m,
                fmU_cr=fmU_cr, fmU_pl=fmU_pl, fmU_m=fmU_m,
                fmM_cr=fmM_cr, fmM_pl=fmM_pl, fmM_m=fmM_m,
            )
        except Exception as e:
            bd_errors[name] = str(e)

    for fname, err in bd_errors.items():
        st.error(f"**{fname}**: {err}")

    if not bd_results:
        return

    # ── 5. Per-file tabs ──────────────────────────────────────────────────────
    st.markdown("#### Per-file Results")
    file_names = list(bd_results.keys())
    stabs = st.tabs([Path(n).stem for n in file_names])
    xr = (ui["freq_min"], ui["freq_max"])
    yr = (ui["db_min"],   ui["db_max"])

    def _fc(v_cr, v_pl, method):
        if method in ("No Gain", "No Data"):
            return method
        if method == "0dB Cross":
            return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
        if method == "Extrap & Plat.":
            return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
        return "N/A"

    for stab, name in zip(stabs, file_names):
        r = bd_results[name]
        c = PALETTE[file_names.index(name) % len(PALETTE)]
        with stab:
            k1, k2, k3, k4 = st.columns(4)
            card(k1, "De-embedding", "Batch (Modeled O/S)", "mode", "#888")
            card(k2, "fT (GHz)",
                 _fc(r["fT_cr"],  r["fT_pl"],  r["fT_m"]),  r["fT_m"])
            card(k3, "fmax U",
                 _fc(r["fmU_cr"], r["fmU_pl"], r["fmU_m"]), r["fmU_m"], "#d62728")
            card(k4, "fmax MAG",
                 _fc(r["fmM_cr"], r["fmM_pl"], r["fmM_m"]), r["fmM_m"], "#2ca02c")

            cb, cs = st.columns(2)
            with cb:
                st.plotly_chart(
                    make_bode(r["df_de"], Path(name).stem,
                              xr, yr, ui["sh21"], ui["su"], ui["smag"], c),
                    use_container_width=True)
            with cs:
                st.plotly_chart(
                    make_smith(r["S_de"], r["df_de"]["Freq (GHz)"].values,
                                ui["smith_f_min"], ui["smith_f_max"],
                                ui["toggles"], ui["scales"],
                                Path(name).stem, max_r=ui["smith_max_r"]),
                    use_container_width=True)

    # ── 6. Download all de-embedded files ─────────────────────────────────────
    st.markdown("---")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, r in bd_results.items():
            stem = re.sub(r"\.(s2p|csv)$", "", name, flags=re.IGNORECASE)
            data = write_s2p(
                r["freq"], r["S_de"],
                title=f"De-embedded {stem}",
                params={
                    "Cpbe_fF": Cpbe * 1e15, "Cpce_fF": Cpce * 1e15, "Cpbc_fF": Cpbc * 1e15,
                    "Lb_pH":   Lb   * 1e12, "Lc_pH":   Lc   * 1e12, "Le_pH":   Le   * 1e12,
                    "Rb_Ohm":  Rb,          "Rc_Ohm":  Rc,          "Re_Ohm":  Re,
                })
            zf.writestr(f"{stem}_deemb.s2p", data)
    date = datetime.now().strftime("%Y-%m-%d")
    st.download_button(
        "📥 Download de-embedded measurement files",
        data=zbuf.getvalue(),
        file_name=f"Batch_Deembedded_{date}.zip",
        mime="application/zip",
        use_container_width=True)

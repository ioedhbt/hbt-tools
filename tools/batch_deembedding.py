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

from tools.common import i18n
from tools.SSM.helpers import (step_open, step_short, peel_parasitics, write_s2p,
                                deembed_open_short, s_to_y, plotly_with_dl)


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
        fig.add_annotation(text=i18n.tr("No Open file uploaded — capacitances default to 0",
                                        "未上傳 Open 檔案 — 電容預設為 0"),
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(color="#888"))
    fig.update_layout(
        title=dict(text=i18n.tr("Pad Capacitance vs Frequency (Open dummy)",
                                "Pad 電容 vs 頻率（Open dummy）"), font=dict(size=13)),
        xaxis=dict(title=i18n.tr("Frequency (GHz)", "頻率 (GHz)"), showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title=i18n.tr("Capacitance (fF)", "電容 (fF)"), range=[0, 30],
                   showgrid=True, gridcolor="#ebebeb"),
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
        fig.add_annotation(text=i18n.tr("No Short file uploaded — inductances default to 0",
                                        "未上傳 Short 檔案 — 電感預設為 0"),
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(color="#888"))
    fig.update_layout(
        title=dict(text=i18n.tr("Lead Inductance vs Frequency (Short dummy)",
                                "引線電感 vs 頻率（Short dummy）"), font=dict(size=13)),
        xaxis=dict(title=i18n.tr("Frequency (GHz)", "頻率 (GHz)"), showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title=i18n.tr("Inductance (pH)", "電感 (pH)"), range=[0, 100],
                   showgrid=True, gridcolor="#ebebeb"),
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
    st.markdown(f"### 🧰 {i18n.tr('Batch De-embedding', '批次去嵌入')}")
    st.caption(i18n.tr(
        "SSM-style modeled de-embedding (Gao 2015 §4.2): pad capacitances are "
        "extracted from the Open dummy and series-lead inductances from the "
        "Short dummy. Override any element below to retune; if no Open/Short "
        "is uploaded the corresponding parameters default to 0.",
        "SSM 風格模型去嵌入（Gao 2015 §4.2）：pad 電容取自 Open dummy，"
        "串聯引線電感取自 Short dummy。可於下方覆寫任一元件重新調校；"
        "若未上傳 Open/Short，對應參數預設為 0。"))

    if not all_data:
        st.info(i18n.tr("Upload DUT files to begin batch de-embedding.",
                        "請上傳 DUT 檔案以開始批次去嵌入。"))
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
            st.error(f"{i18n.tr('Open processing failed', 'Open 處理失敗')}: {e}")

    if short_data is not None:
        try:
            f_short = short_data[0]
            params_short, arrays_short = step_short(
                short_data, f_short,
                params_open["Cpbe"], params_open["Cpce"], params_open["Cpbc"],
                open_data=open_data,
                measured_open=(open_data is not None))
        except Exception as e:
            st.error(f"{i18n.tr('Short processing failed', 'Short 處理失敗')}: {e}")

    # ── 2. Side-by-side C-vs-f and L-vs-f plots ───────────────────────────────
    f_open  = open_data[0]  if open_data  is not None else next(iter(all_data.values()))["freq"]
    f_short = short_data[0] if short_data is not None else next(iter(all_data.values()))["freq"]
    cc, cl = st.columns(2)
    plotly_with_dl(_cap_vs_f_fig(arrays_open,  f_open),
                   key="bd_cap_vs_f", filename="batch_open_cap_vs_f", container=cc)
    plotly_with_dl(_ind_vs_f_fig(arrays_short, f_short),
                   key="bd_ind_vs_f", filename="batch_short_ind_vs_f", container=cl)

    # ── 3. Override section ───────────────────────────────────────────────────
    head_l, head_r = st.columns([4, 1])
    head_l.markdown(f"#### {i18n.tr('Overrides', '覆寫')}")
    if head_r.button(i18n.tr("↺ Reset to defaults", "↺ 重設為預設值"), key="bd_reset",
                      help=i18n.tr("Restore values computed from Open/Short.",
                                   "還原為 Open/Short 計算所得的數值。")):
        for k in ("bd_Cpbe","bd_Cpce","bd_Cpbc",
                  "bd_Lb","bd_Lc","bd_Le",
                  "bd_Rb","bd_Re","bd_Rc"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(i18n.tr("Defaults are populated from the modeled Open/Short calculation.",
                       "預設值取自模型 Open/Short 計算結果。"))

    # Re-seed when the calibration itself changes.  _ovr_input only seeds a
    # key that does not exist yet, so uploading a *different* Open/Short dummy
    # mid-session left the override fields showing the first dummy's values —
    # and the de-embedding silently applied the old parasitics to the new
    # calibration data, with nothing on screen to say so.
    _cal_fp = tuple(round(float(d[k]), 18) for d, k in (
        (params_open, "Cpbe"), (params_open, "Cpce"), (params_open, "Cpbc"),
        (params_short, "Lb"), (params_short, "Lc"), (params_short, "Le")))
    if st.session_state.get("bd_cal_fp") != _cal_fp:
        if "bd_cal_fp" in st.session_state:
            st.info(i18n.tr(
                "Open/Short calibration changed — override values reseeded.",
                "Open/Short 校準已變更 — 覆寫值已重新載入。"))
        for k in ("bd_Cpbe", "bd_Cpce", "bd_Cpbc", "bd_Lb", "bd_Lc", "bd_Le"):
            st.session_state.pop(k, None)
        st.session_state["bd_cal_fp"] = _cal_fp

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

    # ── 4. Open / Short source selector ───────────────────────────────────────
    st.markdown(f"#### {i18n.tr('De-embedding source', '去嵌入來源')}")
    st.caption(i18n.tr(
        "**Modeled** uses the override values above to build analytical pad/lead "
        "matrices and subtract them. **Measured** subtracts the raw Open/Short "
        "S-parameters directly (Gao §4.2 open-short). Measured falls back to "
        "Modeled automatically when the corresponding dummy file is missing.",
        "**模型計算** 使用上方覆寫值建立解析 pad/引線矩陣並扣除。**量測值** "
        "直接扣除原始 Open/Short S 參數（Gao §4.2 open-short）。當對應的 "
        "dummy 檔案缺失時，量測值會自動退回模型計算。"))
    # Values stay English ("Modeled"/"Measured") — open_mode/short_mode are
    # compared against these literals below and embedded in mode_lbl further
    # down; only the on-screen label localizes via format_func.
    _src_opts = ["Modeled", "Measured"]
    _src_labels_zh = {"Modeled": "模型計算", "Measured": "量測值"}
    src_o, src_s = st.columns(2)
    open_mode  = src_o.radio(i18n.tr("Open source", "Open 來源"), _src_opts,
                              horizontal=True, key="bd_open_src",
                              disabled=(open_data is None),
                              format_func=lambda v: i18n.tr(v, _src_labels_zh[v]),
                              help=(i18n.tr("Upload an Open dummy to enable Measured.",
                                            "請上傳 Open dummy 以啟用量測值。")
                                    if open_data is None else None))
    short_mode = src_s.radio(i18n.tr("Short source", "Short 來源"), _src_opts,
                              horizontal=True, key="bd_short_src",
                              disabled=(short_data is None),
                              format_func=lambda v: i18n.tr(v, _src_labels_zh[v]),
                              help=(i18n.tr("Upload a Short dummy to enable Measured.",
                                            "請上傳 Short dummy 以啟用量測值。")
                                    if short_data is None else None))
    use_meas_open  = (open_mode  == "Measured") and (open_data  is not None)
    use_meas_short = (short_mode == "Measured") and (short_data is not None)

    # Pre-compute Y_open / Y_short on the DUT frequency grid where needed.
    Y_open_meas = Y_short_meas = None
    if use_meas_open:
        f_o, S_o, z0_o = open_data
        Y_open_meas = s_to_y(S_o, z0_o)
    if use_meas_short:
        f_s, S_s, z0_s = short_data
        Y_short_meas = s_to_y(S_s, z0_s)

    # ── 5. Apply de-embedding to every DUT file ───────────────────────────────
    bd_results = {}
    bd_errors  = {}
    for name, d in all_data.items():
        try:
            if use_meas_open and use_meas_short:
                # Both sides measured: pure Open-Short subtraction (Gao §4.2)
                if (len(d["freq"]) != len(open_data[0])
                        or not np.allclose(d["freq"], open_data[0], rtol=1e-5)
                        or len(d["freq"]) != len(short_data[0])
                        or not np.allclose(d["freq"], short_data[0], rtol=1e-5)):
                    raise ValueError("DUT frequency grid does not match Open/Short.")
                Y_dut = s_to_y(d["S_raw"], d["z0"])
                Y_de  = deembed_open_short(Y_dut, Y_open_meas, Y_short_meas)
            elif use_meas_open and not use_meas_short:
                # Open from measurement, Short from model
                Y_dut = s_to_y(d["S_raw"], d["z0"])
                Y_after_open = Y_dut - Y_open_meas
                # Re-use peel_parasitics' Short half by passing zeroed pad caps
                p_short_only = {**p_eff,
                                "Cpbe": 0.0, "Cpce": 0.0, "Cpbc": 0.0}
                # Build a virtual "post-Open" S so peel_parasitics applies only Short
                # (peel does Y_dut - Y_pad first; with Y_pad=0 this is a no-op)
                from tools.SSM.helpers import y_to_s_batch as _y2s
                S_after_open = _y2s(Y_after_open, d["z0"])
                Y_de = peel_parasitics(S_after_open, d["freq"], d["z0"], p_short_only)
            elif use_meas_short and not use_meas_open:
                # Open from model, Short from measurement (rare, but supported)
                p_open_only = {**p_eff,
                               "Lb": 0.0, "Lc": 0.0, "Le": 0.0,
                               "Rpb": 0.0, "Rpc": 0.0, "Rpe": 0.0}
                Y_after_open = peel_parasitics(d["S_raw"], d["freq"], d["z0"], p_open_only)
                # Subtract measured Y_short (already Open-corrected by Gao convention
                # ⇒ users supplying Short S2P should ensure it includes pads)
                from tools.SSM.helpers import z_to_y, y_to_z
                Y_de = z_to_y(y_to_z(Y_after_open) - y_to_z(Y_short_meas - (Y_open_meas if Y_open_meas is not None else 0)))
            else:
                # Both modeled (original behaviour)
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
    st.markdown(f"#### {i18n.tr('Per-file Results', '各檔案結果')}")
    file_names = list(bd_results.keys())
    stabs = st.tabs([Path(n).stem for n in file_names])
    xr = (ui["freq_min"], ui["freq_max"])
    yr = (ui["db_min"],   ui["db_max"])

    # extract_limit()'s method tags — comparisons below stay against the raw
    # English tag; only the displayed text localizes.
    _METHOD_ZH = {"No Gain": "無增益", "No Data": "無資料",
                  "0dB Cross": "0dB 交越", "Extrap & Plat.": "外插與平台"}
    def _method_disp(m):
        return i18n.tr(m, _METHOD_ZH.get(m, m))

    def _fc(v_cr, v_pl, method):
        if method in ("No Gain", "No Data"):
            return _method_disp(method)
        if method == "0dB Cross":
            return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
        if method == "Extrap & Plat.":
            return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
        return "N/A"

    def _mode_disp(mode, data_present):
        val = mode if data_present else "Modeled"
        return i18n.tr(val, _src_labels_zh[val])

    for stab, name in zip(stabs, file_names):
        r = bd_results[name]
        c = PALETTE[file_names.index(name) % len(PALETTE)]
        with stab:
            k1, k2, k3, k4 = st.columns(4)
            mode_lbl = (f"Open: {_mode_disp(open_mode, open_data is not None)} · "
                        f"Short: {_mode_disp(short_mode, short_data is not None)}")
            card(k1, i18n.tr("De-embedding", "去嵌入"), mode_lbl, i18n.tr("mode", "模式"), "#888")
            card(k2, "fT (GHz)",
                 _fc(r["fT_cr"],  r["fT_pl"],  r["fT_m"]),  _method_disp(r["fT_m"]))
            card(k3, "fmax U",
                 _fc(r["fmU_cr"], r["fmU_pl"], r["fmU_m"]), _method_disp(r["fmU_m"]), "#d62728")
            card(k4, "fmax MAG",
                 _fc(r["fmM_cr"], r["fmM_pl"], r["fmM_m"]), _method_disp(r["fmM_m"]), "#2ca02c")

            cb, cs = st.columns(2)
            with cb:
                st.plotly_chart(
                    make_bode(r["df_de"], Path(name).stem,
                              xr, yr, ui["sh21"], ui["su"], ui["smag"], c),
                    width="stretch")
            with cs:
                st.plotly_chart(
                    make_smith(r["S_de"], r["df_de"]["Freq (GHz)"].values,
                                ui["smith_f_min"], ui["smith_f_max"],
                                ui["toggles"], ui["scales"],
                                Path(name).stem, max_r=ui["smith_max_r"]),
                    width="stretch")

    # ── 6. Download all de-embedded files ─────────────────────────────────────
    st.markdown("---")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, r in bd_results.items():
            stem = re.sub(r"\.(s2p|csv)$", "", name, flags=re.IGNORECASE)
            data = write_s2p(
                r["freq"], r["S_de"],
                title=f"De-embedded {stem}",
                # Key/unit convention must match
                # models/__init__.py::AbstractSSMModel.get_s2p_header_params —
                # bare key, unit as trailing text on the value — because that
                # is what agent_api's header parser (_HDR_KV_RE +
                # _PAD_HEADER_KEYS) reads back.  These were previously written
                # as "Cpbe_fF = 12.34": the regex swallowed "Cpbe_fF" as the
                # key, matched nothing in _PAD_HEADER_KEYS, and every file this
                # tab exported reported removed_params == {} while claiming
                # deembedded == True.
                params={
                    "Cpbe": f"{Cpbe * 1e15:.4f} fF",
                    "Cpce": f"{Cpce * 1e15:.4f} fF",
                    "Cpbc": f"{Cpbc * 1e15:.4f} fF",
                    "Lb":   f"{Lb * 1e12:.4f} pH",
                    "Lc":   f"{Lc * 1e12:.4f} pH",
                    "Le":   f"{Le * 1e12:.4f} pH",
                    "Rb":   f"{Rb:.4f} Ω",
                    "Rc":   f"{Rc:.4f} Ω",
                    "Re":   f"{Re:.4f} Ω",
                })
            zf.writestr(f"{stem}_deemb.s2p", data)
    date = datetime.now().strftime("%Y-%m-%d")
    st.download_button(
        i18n.tr("📥 Download de-embedded measurement files", "📥 下載去嵌入後的量測檔案"),
        data=zbuf.getvalue(),
        file_name=f"Batch_Deembedded_{date}.zip",
        mime="application/zip",
        width="stretch")

    # ── 7. Hand the de-embedded device(s) to the SSM pages ────────────────────
    from tools.common import handoff
    with st.container(border=True):
        st.markdown(f"**🔁 {i18n.tr('Send de-embedded device(s) to an SSM page', '將去嵌入元件傳送至 SSM 頁面')}**")
        names = list(bd_results.keys())
        # Guard the persisted selection: a stale value (file set changed) would
        # make st.selectbox raise.
        if st.session_state.get("bd_handoff_sel") not in names:
            st.session_state["bd_handoff_sel"] = names[0]
        hs1, hs2, hs3 = st.columns([1.6, 1, 1])
        sel = hs1.selectbox(i18n.tr("Primary device", "主要元件"), names,
                            format_func=lambda s: Path(s).stem,
                            key="bd_handoff_sel",
                            help=i18n.tr(
                                "Extraction also receives every other "
                                "de-embedded file here (the Z-parameter, "
                                "Cold-HBT and τ_total methods need them); "
                                "Simulation & Fitting receives just this one.",
                                "萃取頁面會同時接收此處其餘去嵌入檔案（Z 參數、"
                                "Cold-HBT 與 τ_total 方法需要用到）；"
                                "模擬與擬合頁面則只接收此選取檔案。"))
        r_sel = bd_results[sel]
        # Extraction: send all de-embedded files (primary + the rest as extras).
        if hs2.button(i18n.tr("→ SSM Extraction", "→ 小訊號模型萃取"), key="bd_handoff_ext",
                      width="stretch", type="primary"):
            extras = {n: {"S": r["S_de"], "freq": r["freq"], "z0": r["z0"]}
                      for n, r in bd_results.items() if n != sel}
            handoff.send(handoff.TARGET_EXTRACTION,
                         S=r_sel["S_de"], freq=r_sel["freq"], z0=r_sel["z0"],
                         label=Path(sel).stem, stage="deembedded",
                         extras=extras)
            st.switch_page(handoff.PAGE_EXTRACTION)
        # Fitting: only the selected device.
        if hs3.button(i18n.tr("→ Simulation & Fitting", "→ 模擬與擬合"), key="bd_handoff_sim",
                      width="stretch"):
            handoff.send(handoff.TARGET_SIMFIT,
                         S=r_sel["S_de"], freq=r_sel["freq"], z0=r_sel["z0"],
                         label=Path(sel).stem, stage="deembedded")
            st.switch_page(handoff.PAGE_SIMFIT)

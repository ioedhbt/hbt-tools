"""
hbt_rf_tool.py
============================
Main Streamlit application for HBT RF extraction.

SSM Extraction functionality lives entirely in ssm_extraction.py.
The only SSM-related line in this file is:

    from cheng_extraction import render_ssm_tab

which is then called once inside the "SSM Extraction" sub-tab.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
__version__ = "5.0"

import io, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from tools.SSM.main_ssm_extraction import render_ssm_tab   # ← SSM module (Cheng 2022)
from tools.SSM.ssm_plots      import render_matplotlib_smith
from tools.batch_deembedding  import render_batch_deembedding_tab
from tools.SSM.helpers        import (
    parse_s2p, parse_csv,
    s_to_y, y_to_s_batch as y_to_s,
    strict_freq_check,
    deembed_open_short, deembed_thru_half,
    compute_metrics, extract_limit,
    PALETTE, darken, bode_layout, make_smith, make_bode, make_plateau,
    metric_card, build_excel, load_cal,
)

# ─────────────────────────────────────────────────────────────────────────────
if "rf_uploader_key" not in st.session_state:
    st.session_state["rf_uploader_key"] = 0

st.title(f"📡 IOED HBT RF Extraction Tool (v{__version__})")

with st.expander(f"What's new in v{__version__}", expanded=False):
    st.markdown(
        "- Refactored SSM into an OOP class hierarchy (`SSMModelTemplate` parent "
        "in `base_ui.py` with concrete `ChengT`, `ChengPi`, `XuModel` subclasses)\n"
        "- Added Xu's forward simulation\n"
        "- Added more Smith chart plot options\n\n"
        "Full version history: [`CHANGELOG.md`](CHANGELOG.md)"
    )

# ═════════════════════════════════════════════════════════════════════════════
#  CORE RF UTILITIES — moved to tools/SSM/helpers/ (rf_math, s2p_io,
#  deembed_math, metrics). See helpers/INDEX.md for the catalog.
# ═════════════════════════════════════════════════════════════════════════════

def process_dut(content, filename, s1_o, s1_s, s2_o, s2_s, s3_t, n_pts, f_min, f_max):
    """Parse, de-embed, compute metrics, extract fT/fmax for one DUT file."""
    if filename.lower().endswith(".csv"):
        freq, S_raw, z0 = parse_csv(content)
    else:
        freq, S_raw, z0 = parse_s2p(content)
    Y_raw = s_to_y(S_raw,z0)
    df_raw = compute_metrics(Y_raw,freq)
    Y_fin,stages,d1_o,d1_s = Y_raw,[],None,None
    if s1_o and s1_s:
        f1o,S1o,z1o=s1_o; f1s,S1s,z1s=s1_s
        strict_freq_check(freq,f1o,"Probe Open")
        d1_o,d1_s=s_to_y(S1o,z1o),s_to_y(S1s,z1s)
        Y_fin=deembed_open_short(Y_fin,d1_o,d1_s); stages.append("Probe")
    Y2o=Y2s=None
    if s2_o and s2_s:
        f2o,S2o,z2o=s2_o; f2s,S2s,z2s=s2_s
        strict_freq_check(freq,f2o,"Dev Open")
        Y2o,Y2s=s_to_y(S2o,z2o),s_to_y(S2s,z2s)
        if d1_o is not None:
            Y2o=deembed_open_short(Y2o,d1_o,d1_s)
            Y2s=deembed_open_short(Y2s,d1_o,d1_s)
        Y_fin=deembed_open_short(Y_fin,Y2o,Y2s); stages.append("Dev(O/S)")
    if s3_t:
        f3t,S3t,z3t=s3_t; strict_freq_check(freq,f3t,"Dev Thru")
        Y3t=s_to_y(S3t,z3t)
        if d1_o is not None: Y3t=deembed_open_short(Y3t,d1_o,d1_s)
        if "Dev(O/S)" in stages:
            Y3t_r=s_to_y(S3t,z3t)
            if d1_o is not None: Y3t_r=deembed_open_short(Y3t_r,d1_o,d1_s)
            Y3t=deembed_open_short(Y3t_r,Y2o,Y2s)
        Y_fin=deembed_thru_half(Y_fin,Y3t); stages.append("Dev(Thru)")
    note=" + ".join(stages) if stages else "None"
    df_fin=compute_metrics(Y_fin,freq) if stages else None
    S_fin=y_to_s(Y_fin,z0) if stages else S_raw
    df_e=df_fin if df_fin is not None else df_raw
    f_arr=df_e["Freq (GHz)"].values
    fT_cr,fT_pl,ft_m    = extract_limit(f_arr,df_e["|h21|² (dB)"].values,  df_e["fT Plateau (GHz)"].values,  n_pts,f_min,f_max)
    fmU_cr,fmU_pl,fmU_m = extract_limit(f_arr,df_e["Mason U (dB)"].values,  df_e["fmax U Plateau (GHz)"].values,n_pts,f_min,f_max)
    fmM_cr,fmM_pl,fmM_m = extract_limit(f_arr,df_e["MAG/MSG (dB)"].values,  df_e["fmax MAG Plateau (GHz)"].values,n_pts,f_min,f_max)
    stem=re.sub(r"\.(s2p|csv)$","",filename,flags=re.IGNORECASE)
    m=re.search(r"[Vv][Cc][Ee][_\-]?([\d]+(?:p\d+)?)\s*[Vv]",stem)
    vce=float(m.group(1).replace("p",".")) if m else None
    m=re.search(r"[Ii][Bb][_\-]?([\d]+(?:p\d+)?)\s*([pnuUmM]?)[Aa]?",stem)
    ib=float(m.group(1).replace("p","."))*{"p":1e-12,"n":1e-9,"u":1e-6,"U":1e-6,"m":1e-3,"M":1e-3,"":1.0}.get(m.group(2) if m else "",1.0) if m else None
    return df_raw,df_fin,S_fin,S_raw,freq,z0,{
        "Label":stem,"Vce (V)":vce,"Ib (A)":ib,"De-embedding":note,
        "fT Cross/Extrap (GHz)":fT_cr,"fT Plateau (GHz)":fT_pl,"fT Method":ft_m,
        "fmax U Cross/Extrap (GHz)":fmU_cr,"fmax U Plateau (GHz)":fmU_pl,"fmax U Method":fmU_m,
        "fmax MAG Cross/Extrap (GHz)":fmM_cr,"fmax MAG Plateau (GHz)":fmM_pl,"fmax MAG Method":fmM_m,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING UTILITIES — moved to tools/SSM/helpers/plotly_plots.py
#  (PALETTE, darken, bode_layout, make_smith, make_bode, make_plateau)
#  and tools/SSM/helpers/chart_export.py (build_excel, metric_card, load_cal).
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("#### 🔧 3-Step De-embedding")
    sw1=st.toggle("① Probe (Open-Short)",value=False)
    f1o=st.file_uploader("Probe Open",  type=["s2p"],key="p1o") if sw1 else None
    f1s=st.file_uploader("Probe Short", type=["s2p"],key="p1s") if sw1 else None
    st.divider()
    sw2=st.toggle("② Device Dummy (Open-Short)",value=False,help="Required for SSM extraction")
    f2o=st.file_uploader("Dev Open",  type=["s2p"],key="d2o") if sw2 else None
    f2s=st.file_uploader("Dev Short", type=["s2p"],key="d2s") if sw2 else None
    if sw2: st.caption("✅ SSM Extraction tab enabled.")
    st.divider()
    sw3=st.toggle("③ Device Thru (Half-Z)",value=False)
    f3t=st.file_uploader("Dev Thru",  type=["s2p"],key="d3t") if sw3 else None
    st.divider()
    st.markdown("#### 📊 Chart Control")
    freq_min=st.number_input("Freq Min (GHz)",value=0.01,min_value=0.01,format="%.4f")
    freq_max=st.number_input("Freq Max (GHz)",value=50.0,min_value=1.0)
    db_min=st.number_input("Bode Y Min (dB)",value=0.0)
    db_max=st.number_input("Bode Y Max (dB)",value=50.0)
    n_pts=st.slider("Interpolation pts",2,20,2)
    show_raw=st.checkbox("Overlay Raw",value=True,disabled=not(sw1 or sw2 or sw3))
    st.markdown("##### Trace Selection")
    sh21=st.checkbox("|h21|² → fT",value=True,key="sh21")
    su=st.checkbox("Mason U → fmax(U)",value=True,key="su")
    smag=st.checkbox("MAG/MSG → fmax",value=True,key="smag")
    st.divider()
    st.markdown("#### 🍩 Smith Chart")
    smith_f_min=st.number_input("Smith Freq Min (GHz)",value=freq_min,min_value=0.01,format="%.4f")
    smith_f_max=st.number_input("Smith Freq Max (GHz)",value=freq_max,min_value=1.0)
    smith_max_r=st.slider("|Γ| Max Radius",1.0,5.0,1.0,0.5)
    st.markdown("##### Display & Scale")
    ca,cb=st.columns(2); show_s11=ca.checkbox("S11",value=True,key="ss11"); scale_s11=cb.number_input("S11 ×",value=1.0,step=0.1,key="sc11")
    ca,cb=st.columns(2); show_s22=ca.checkbox("S22",value=True,key="ss22"); scale_s22=cb.number_input("S22 ×",value=1.0,step=0.1,key="sc22")
    ca,cb=st.columns(2); show_s21=ca.checkbox("S21",value=True,key="ss21"); scale_s21=cb.number_input("S21 ×",value=1.0,step=0.1,key="sc21")
    ca,cb=st.columns(2); show_s12=ca.checkbox("S12",value=True,key="ss12"); scale_s12=cb.number_input("S12 ×",value=1.0,step=0.1,key="sc12")

# ═════════════════════════════════════════════════════════════════════════════
#  FILE UPLOADER & PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
col_up1,col_up2=st.columns([4,1])
with col_up1:
    dut_files=st.file_uploader("Upload DUT .s2p / .csv files",type=["s2p","csv"],
                               accept_multiple_files=True,
                               key=st.session_state["rf_uploader_key"])
with col_up2:
    st.write(""); st.write("")
    if st.button("🗑️ Clear uploads",use_container_width=True):
        st.session_state["rf_uploader_key"]+=1
        st.session_state.pop("rf_ms_files",None)
        st.session_state.pop("rf_prev_uploaded",None)
        st.rerun()

s1o=load_cal(f1o) if sw1 else None
s1s=load_cal(f1s) if sw1 else None
s2o=load_cal(f2o) if sw2 else None   # passed to render_ssm_tab
s2s=load_cal(f2s) if sw2 else None   # passed to render_ssm_tab
s3t=load_cal(f3t) if sw3 else None

all_data,errors={},{}
if dut_files:
    for f in dut_files:
        try:
            df_raw,df_fin,S_fin,S_raw,freq,z0_dut,res=process_dut(
                f.getvalue().decode("utf-8",errors="ignore"),
                f.name,s1o,s1s,s2o,s2s,s3t,n_pts,freq_min,freq_max)
            all_data[f.name]={
                "df_raw":df_raw,"df_fin":df_fin,
                "S_fin":S_fin,"S_raw":S_raw,
                "freq":freq,"z0":z0_dut,**res}
        except Exception as e:
            errors[f.name]=str(e)

for fname,err in errors.items():
    st.error(f"**{fname}**: {err}")

if all_data:
    file_options=list(all_data.keys())
    cur=set(file_options); prev=st.session_state.get("rf_prev_uploaded",set())
    new_=cur-prev
    sel=[f for f in st.session_state.get("rf_ms_files",[]) if f in cur]
    for nf in new_:
        if nf not in sel: sel.append(nf)
    st.session_state["rf_ms_files"]=sel
    st.session_state["rf_prev_uploaded"]=cur

    c1,c2,_=st.columns([1.5,1.5,7])
    if c1.button("✅ Select All"):  st.session_state["rf_ms_files"]=file_options
    if c2.button("❌ Clear"):       st.session_state["rf_ms_files"]=[]
    selected_files=st.multiselect("📂 Files to analyse:",options=file_options,
                                  key="rf_ms_files",format_func=lambda x:Path(x).stem)
else:
    selected_files=[]
    st.info("Upload DUT .s2p files above to begin.")

xr,yr=(freq_min,freq_max),(db_min,db_max)

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_ov,tab_ind,tab_sum,tab_bd=st.tabs(["📊 Overlay","📁 Individual","📋 Summary","🧰 Batch De-embed"])

with tab_ov:
    st.markdown("### 📊 Bode Plot Overlay")
    f_bode=go.Figure()
    if all_data and selected_files:
        for i,n in enumerate(selected_files):
            d,c,lbl=all_data[n],PALETTE[i%len(PALETTE)],Path(n).stem
            df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
            hov="Freq:%{x:.4f}GHz<br>%{y:.4f}dB<extra></extra>"
            if show_raw and d["df_fin"] is not None and sh21:
                f_bode.add_trace(go.Scattergl(x=d["df_raw"]["Freq (GHz)"],y=d["df_raw"]["|h21|² (dB)"],
                                            name=f"|h21|² raw–{lbl}",line=dict(color=c,width=1.2,dash="dot"),
                                            opacity=0.35,hovertemplate=hov))
            if sh21: f_bode.add_trace(go.Scattergl(x=df_p["Freq (GHz)"],y=df_p["|h21|² (dB)"],name=f"|h21|²–{lbl}",line=dict(color=c,width=2.5),hovertemplate=hov))
            if su:   f_bode.add_trace(go.Scattergl(x=df_p["Freq (GHz)"],y=df_p["Mason U (dB)"],name=f"U–{lbl}",line=dict(color=darken(c),width=2.5,dash="dash"),hovertemplate=hov))
            if smag: f_bode.add_trace(go.Scattergl(x=df_p["Freq (GHz)"],y=df_p["MAG/MSG (dB)"],name=f"MAG–{lbl}",line=dict(color=c,width=2,dash="dot"),opacity=0.7,hovertemplate=hov))
    f_bode.add_hline(y=0,line_dash="dash",line_color="black")
    f_bode.update_layout(**bode_layout("Overlay — Bode Plot","Gain (dB)",yr,xr)); f_bode.update_layout(height=550)
    st.plotly_chart(f_bode,use_container_width=True)

    st.markdown("### 📊 Plateau Plot Overlay")
    f_plat=go.Figure(); all_v=[]
    if all_data and selected_files:
        for i,n in enumerate(selected_files):
            d,c,lbl=all_data[n],PALETTE[i%len(PALETTE)],Path(n).stem
            df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
            hov="Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
            if sh21:
                f_plat.add_trace(go.Scattergl(x=df_p["Freq (GHz)"],y=df_p["fT Plateau (GHz)"],name=f"fT–{lbl}",line=dict(color=c,width=2.5),hovertemplate=hov))
                all_v+=df_p["fT Plateau (GHz)"].dropna().tolist()
            if su:
                f_plat.add_trace(go.Scattergl(x=df_p["Freq (GHz)"],y=df_p["fmax U Plateau (GHz)"],name=f"fmax(U)–{lbl}",line=dict(color=darken(c),width=2.5,dash="dash"),hovertemplate=hov))
                all_v+=df_p["fmax U Plateau (GHz)"].dropna().tolist()
    arr=np.array([v for v in all_v if np.isfinite(v) and v>0])
    ym=float(np.quantile(arr,0.97))*1.3 if len(arr) else 100
    f_plat.update_layout(**bode_layout("Overlay — Plateau","GBP (GHz)",[0,ym],xr)); f_plat.update_layout(height=550)
    st.plotly_chart(f_plat,use_container_width=True)

with tab_ind:
    if not all_data or not selected_files:
        st.info("Upload and select files to view individual analysis.")
    else:
        # Single-active-file selector — replaces st.tabs(...) over selected_files.
        # Streamlit executes the body of every st.tabs branch on every rerun (only
        # the visibility is toggled), so with N files a single slider drag would
        # re-run all N pipelines. A selectbox conditionally renders only the
        # selected file's body, so cost no longer scales with N.
        n=st.selectbox(
            "📁 Active file",
            options=selected_files,
            format_func=lambda fn: Path(fn).stem,
            key="active_file_n",
            help="Only the selected file is rendered. Use the dropdown or arrow keys to switch.",
        )
        file_names=list(all_data.keys())
        c=PALETTE[file_names.index(n)%len(PALETTE)]
        d=all_data[n]
        df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]

        def _fc(v_cr,v_pl,method):
            if method in ["No Gain","No Data"]: return method
            if method=="0dB Cross":      return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
            if method=="Extrap & Plat.": return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
            return "N/A"

        c1,c2,c3,c4,c5=st.columns(5)
        metric_card(c1,"De-embedding",d["De-embedding"],"mode","#888")
        metric_card(c2,"fT (GHz)",_fc(d["fT Cross/Extrap (GHz)"],d["fT Plateau (GHz)"],d["fT Method"]),d["fT Method"])
        metric_card(c3,"fmax U",_fc(d["fmax U Cross/Extrap (GHz)"],d["fmax U Plateau (GHz)"],d["fmax U Method"]),d["fmax U Method"],"#d62728")
        metric_card(c4,"fmax MAG",_fc(d["fmax MAG Cross/Extrap (GHz)"],d["fmax MAG Plateau (GHz)"],d["fmax MAG Method"]),d["fmax MAG Method"],"#2ca02c")
        if d["Vce (V)"] is not None:  metric_card(c5,"Vce",f"{d['Vce (V)']} V","bias","#9467bd")
        elif d["Ib (A)"] is not None: metric_card(c5,"Ib",f"{d['Ib (A)']*1e6:.1f} µA","bias","#9467bd")

        toggles={"S11":show_s11,"S22":show_s22,"S21":show_s21,"S12":show_s12}
        scales ={"S11":scale_s11,"S22":scale_s22,"S21":scale_s21,"S12":scale_s12}

        ta,tb,tc,td=st.tabs(["Bode Plot","Plateau Plot","Smith Chart","🔬 SSM Extraction"])
        with ta:
            f_arr = df_p["Freq (GHz)"].values
            n_freq = len(f_arr)

            # ── Extrapolation controls ───────────────────────────────────────
            bc1, bc2 = st.columns([1, 1])
            show_20db = bc1.checkbox(
                "Show −20 dB/dec extrapolation",
                value=True, key=f"bode_show20_{n}",
                help="Anchors a line of slope −20 dB/dec at the last "
                     "measured point (textbook fT/fmax extraction).")
            show_sp = bc2.checkbox(
                "Show single-pole fit",
                value=False, key=f"bode_showsp_{n}",
                help="Log-linear (single-pole) least-squares fit on a "
                     "user-chosen window — slope is determined by the data.")

            sp_window_idx = None
            if show_sp and n_freq >= 4:
                f_lo, f_hi = float(f_arr[0]), float(f_arr[-1])
                _spkey = f"bode_spwin_{n}"
                _dflt  = (max(f_lo, f_hi * 0.5), f_hi)
                sp_win = st.slider(
                    "Single-pole fit window (GHz)",
                    min_value=f_lo, max_value=f_hi,
                    value=st.session_state.get(_spkey, _dflt),
                    step=max((f_hi - f_lo) / 400.0, 1e-3),
                    key=_spkey)
                _il = int(np.searchsorted(f_arr, sp_win[0], side="left"))
                _ih = int(np.searchsorted(f_arr, sp_win[1], side="right")) - 1
                _il = max(0, min(_il, n_freq - 2))
                _ih = max(_il + 1, min(_ih, n_freq - 1))
                sp_window_idx = (_il, _ih)

            # f_max_target so extrap curves extend past the largest fT/fmax
            _ft_now  = d.get("fT Cross/Extrap (GHz)")
            _fmU_now = d.get("fmax U Cross/Extrap (GHz)")
            _maxes = [v for v in (_ft_now, _fmU_now, f_arr[-1] if n_freq else 0)
                      if v is not None and np.isfinite(v)]
            f_max_target = max(_maxes) * 1.2 if _maxes else None

            fig_bode, extrap_df = make_bode(
                df_p, Path(n).stem, xr, yr, sh21, su, smag, c,
                show_20db=show_20db, show_sp=show_sp,
                sp_window_idx=sp_window_idx,
                extrap_f_max=f_max_target,
                return_extrap_df=True)
            st.plotly_chart(fig_bode, use_container_width=True)

            # ── Excel download for extrapolated/fitted data ──────────────────
            if extrap_df is not None and not extrap_df.empty:
                _buf = io.BytesIO()
                extrap_df.to_excel(_buf, index=False, engine="openpyxl")
                st.download_button(
                    "📥 Download extrapolated data (Excel)",
                    data=_buf.getvalue(),
                    file_name=f"{Path(n).stem}_bode_extrap.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"bode_dl_{n}")
        with tb: st.plotly_chart(make_plateau(df_p,d,Path(n).stem,xr,sh21,su,smag,c),use_container_width=True)
        with tc:
            smith_sub_plotly, smith_sub_mpl = st.tabs(
                ["Plotly", "Matplotlib"])
            with smith_sub_plotly:
                st.plotly_chart(make_smith(
                    d["S_fin"], df_p["Freq (GHz)"].values,
                    smith_f_min, smith_f_max, toggles, scales,
                    Path(n).stem, max_r=smith_max_r),
                    use_container_width=True)
            with smith_sub_mpl:
                render_matplotlib_smith(
                    fname=n, topo_key="meas",
                    sets=[{"S": d["S_fin"], "label": Path(n).stem,
                           "kind": "line", "style": "solid"}],
                    default_multiplier=1.0,
                )
        with td:
            # Gate key unique per file
            run_key = f"ssm_run_{n}"

            if not st.session_state.get(run_key, False):
                st.markdown(" ")
                col_ctr, _, _ = st.columns([1, 2, 2])
                if col_ctr.button(
                    "▶ Run SSM Extraction",
                    key=f"ssm_btn_{n}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state[run_key] = True
                    st.rerun()
                st.caption(
                    "SSM extraction is skipped until activated to keep the app fast. "
                    "Click above to run it for this file."
                )
            else:
                # Optional: allow the user to reset / clear the results
                if st.button(
                    "✕ Clear SSM results",
                    key=f"ssm_clear_{n}",
                    help="Frees cached computation for this file.",
                ):
                    st.session_state[run_key] = False
                    # Also clear any downstream caches for this file
                    for k in list(st.session_state.keys()):
                        if k.endswith(f"_{n}") and k != run_key:
                            del st.session_state[k]
                    st.rerun()

                render_ssm_tab(
                    n, d["S_raw"], d["freq"], d["z0"],
                    s2o, s2s, all_data=all_data,
                )

        with st.expander("📋 Data Table"):
            if d["df_fin"] is not None:
                ta2,tb2=st.tabs(["De-embedded","Raw"])
                with ta2: st.dataframe(df_p.round(4),use_container_width=True,hide_index=True)
                with tb2: st.dataframe(d["df_raw"].round(4),use_container_width=True,hide_index=True)
            else: st.dataframe(df_p.round(4),use_container_width=True,hide_index=True)

with tab_sum:
    if not all_data:
        st.info("Upload files to generate summary.")
    else:
        rows=[{"File":k,"De-embedding":d["De-embedding"],"Vce (V)":d["Vce (V)"],
               "Ib (µA)":round(d["Ib (A)"]*1e6,1) if d["Ib (A)"] else None,
               "fT Cross":d["fT Cross/Extrap (GHz)"],"fT Plat":d["fT Plateau (GHz)"],
               "fmax U Cross":d["fmax U Cross/Extrap (GHz)"],"fmax U Plat":d["fmax U Plateau (GHz)"]}
              for k,d in all_data.items()]
        sum_df=pd.DataFrame(rows)
        fmt={c:"{:.4f}" for c in sum_df.columns if "Cross" in c or "Plat" in c}
        fmt["Vce (V)"]="{:.3f}"; fmt["Ib (µA)"]="{:.1f}"
        st.dataframe(sum_df.style.format(fmt,na_rep="—"),use_container_width=True,hide_index=True)
        date=datetime.now().strftime("%Y-%m-%d")
        d1,d2=st.columns(2)
        with d1:
            st.download_button("📥 Excel",data=build_excel(sum_df,all_data),
                               file_name=f"RF_Extraction_{date}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with d2:
            zbuf=io.BytesIO()
            with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Summary.csv",sum_df.to_csv(index=False).encode())
                for k,d in all_data.items():
                    dp=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    zf.writestr(f"{Path(k).stem}.csv",dp.to_csv(index=False).encode())
            st.download_button("📦 ZIP (CSV)",data=zbuf.getvalue(),
                               file_name=f"RF_Extraction_{date}.zip",
                               mime="application/zip",use_container_width=True)

with tab_bd:
    render_batch_deembedding_tab(
        all_data=all_data,
        open_data=s2o,
        short_data=s2s,
        ui=dict(
            freq_min=freq_min, freq_max=freq_max,
            db_min=db_min, db_max=db_max, n_pts=n_pts,
            sh21=sh21, su=su, smag=smag,
            smith_f_min=smith_f_min, smith_f_max=smith_f_max, smith_max_r=smith_max_r,
            toggles={"S11":show_s11,"S22":show_s22,"S21":show_s21,"S12":show_s12},
            scales ={"S11":scale_s11,"S22":scale_s22,"S21":scale_s21,"S12":scale_s12},
        ),
        helpers=dict(
            y_to_s=y_to_s, compute_metrics=compute_metrics, extract_limit=extract_limit,
            make_bode=make_bode, make_smith=make_smith, card=metric_card, PALETTE=PALETTE,
        ),
    )
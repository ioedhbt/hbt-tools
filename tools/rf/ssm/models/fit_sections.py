"""
models/fit_sections.py — Standalone fit tools embedded inside the
Interactive Parameter Extraction expander: the Cbex sweep tool and the
multi-file 1/(2πfT) vs 1/IC (τ_total) reference fit.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ..helpers import plotly_with_dl
from tools.common.i18n import tr


def _render_cbex_sweep_tool(*, cbex_arr, freq, f_ghz, f_min_v, f_max_v,
                            cbex_scale, cbex_unit, cbex_param_key,
                            model_short, fname, g_idx, rng_tag,
                            cbex_sweep_fn, param_groups):
    """Render the Min / Step / Max inputs + Calculate button for the Cbex
    sweep.  On button click: generate candidate Cbex values, call
    `cbex_sweep_fn(cbex_SI_array, mask)` to get per-candidate std(Cbcx_arr),
    pick the argmin, and write the best value into the Cbex number_input's
    session-state key so the widget below renders the new value on the next
    natural rerun.

    The std window (mask) comes from the Cbcx group's slider session state
    if the user has already moved it, otherwise the full frequency range.
    """
    # Defaults: min = min|Cbex_arr| (fF), max = max|Cbex_arr| (fF), step = 10 fF
    fin = cbex_arr[np.isfinite(cbex_arr)]
    if len(fin) == 0:
        st.caption(tr("Cbex sweep unavailable — no finite Cbex samples.",
                      "Cbex 掃描無法使用 — 沒有有限的 Cbex 樣本。"))
        return
    abs_disp = np.abs(fin) * cbex_scale
    default_min = float(np.min(abs_disp))
    default_max = float(np.max(abs_disp))
    if default_max <= default_min:
        default_max = default_min + 1.0
    default_step = 0.1

    # Per-group state keys — tied to rng_tag so defaults refresh on slider move
    sweep_key_base = f"cbex_sweep_{model_short}_{fname}_{rng_tag}"
    k_min  = f"{sweep_key_base}_min"
    k_step = f"{sweep_key_base}_step"
    k_max  = f"{sweep_key_base}_max"
    k_res  = f"cbex_sweep_result_{model_short}_{fname}"  # persists across reruns

    st.markdown("---")
    st.markdown(f"**🔍 {tr('Cbex sweep — minimise std(Cbcx)', 'Cbex 掃描 — 最小化 std(Cbcx)')}**")

    c_min, c_step, c_max = st.columns(3)
    sweep_min = c_min.number_input(
        f"{tr('Min', '最小值')} ({cbex_unit})",
        value=float(st.session_state.get(k_min, default_min)),
        min_value=0.0, format="%.4f", key=k_min)
    sweep_step = c_step.number_input(
        f"{tr('Step', '步進')} ({cbex_unit})",
        value=float(st.session_state.get(k_step, default_step)),
        min_value=1e-6, format="%.4f", key=k_step)
    sweep_max = c_max.number_input(
        f"{tr('Max', '最大值')} ({cbex_unit})",
        value=float(st.session_state.get(k_max, default_max)),
        min_value=0.0, format="%.4f", key=k_max)
    run_sweep = st.button(
        tr("Calculate", "計算"),
        key=f"{sweep_key_base}_btn",
        width="stretch")

    if run_sweep:
        if sweep_max < sweep_min or sweep_step <= 0:
            st.error(tr("Sweep range is invalid: need max ≥ min and step > 0.",
                        "掃描範圍無效：需 max ≥ min 且 step > 0。"))
        else:
            # Build candidate array in display units, then convert to SI
            n_pts = int(np.floor((sweep_max - sweep_min) / sweep_step)) + 1
            n_pts = max(2, min(n_pts, 10000))  # sanity cap
            cand_disp = sweep_min + np.arange(n_pts) * sweep_step
            cand_disp = cand_disp[cand_disp <= sweep_max + 1e-12]
            cand_SI = cand_disp / float(cbex_scale)

            # Mask: use the Cbcx group's slider range if the user has set it,
            # otherwise the full freq range.  Find the Cbcx group by searching
            # param_groups for one whose params reference "Cbcx_arr" (was
            # `g_idx + 1` but Ccex now sits between Cbex and Cbcx for ChengT).
            cbcx_g_idx = next(
                (i for i, g in enumerate(param_groups)
                 if any(spec[0] == "Cbcx_arr" for spec in g.get("params", []))),
                g_idx + 1,
            )
            cbcx_sl_key = f"pfp_sl_{model_short}_{cbcx_g_idx}_{fname}"
            cbcx_range = st.session_state.get(cbcx_sl_key, (f_min_v, f_max_v))
            try:
                f_lo_cbcx, f_hi_cbcx = float(cbcx_range[0]), float(cbcx_range[1])
            except Exception:
                f_lo_cbcx, f_hi_cbcx = f_min_v, f_max_v
            mask = (f_ghz >= f_lo_cbcx) & (f_ghz <= f_hi_cbcx)
            if not mask.any():
                mask = np.ones_like(f_ghz, dtype=bool)

            try:
                stds = cbex_sweep_fn(cand_SI, mask)
            except Exception as exc:
                st.error(tr(f"Sweep failed: {exc!r}", f"掃描失敗：{exc!r}"))
                return
            stds = np.asarray(stds, dtype=float)
            if not np.any(np.isfinite(stds)):
                st.error(tr("All candidates produced non-finite std(Cbcx) — "
                            "try a different range.",
                            "所有候選值皆產生非有限的 std(Cbcx) — 請嘗試其他範圍。"))
                return
            best_k = int(np.nanargmin(stds))
            best_cbex_SI = float(cand_SI[best_k])
            best_cbex_disp = best_cbex_SI * float(cbex_scale)
            best_std = float(stds[best_k])

            # Stage the result in a "pending" key.  We CAN'T write directly
            # to the number_input's session-state key here because that
            # widget has already been instantiated earlier in this same run
            # (it lives in cols[0] of the same row).  On the next rerun the
            # plot loop will see this pending value and copy it into the
            # widget key BEFORE the widget is created — see the
            # `pfp_pending_*` lookup just above the number_input render.
            pending_key = (f"pfp_pending_{model_short}_{cbex_param_key}_"
                           f"{fname}")
            st.session_state[pending_key] = float(best_cbex_disp)

            # Persist the message so it survives the rerun
            st.session_state[k_res] = {
                "best_disp": best_cbex_disp,
                "best_std":  best_std,
                "n_pts":     len(cand_SI),
                "f_lo":      f_lo_cbcx,
                "f_hi":      f_hi_cbcx,
                "rng_tag":   rng_tag,
            }
            st.rerun()

    # Show the last sweep result (if any) — scoped to this rng_tag so it
    # clears when the Cbex slider is moved.
    last = st.session_state.get(k_res)
    if last and last.get("rng_tag") == rng_tag:
        st.success(tr(
            f"Best Cbex = **{last['best_disp']:.4f} {cbex_unit}**  "
            f"(std(Cbcx) = {last['best_std']:.3e}, "
            f"{last['n_pts']} candidates, "
            f"Cbcx window {last['f_lo']:.2f}–{last['f_hi']:.2f} GHz)",
            f"最佳 Cbex = **{last['best_disp']:.4f} {cbex_unit}**  "
            f"（std(Cbcx) = {last['best_std']:.3e}，"
            f"{last['n_pts']} 個候選值，"
            f"Cbcx 視窗 {last['f_lo']:.2f}–{last['f_hi']:.2f} GHz）"))


def _render_tau_total_fit_section(*, all_data, fname, model_short,
                                  params, para_eff):
    """
    Multi-file 1/(2πfT) vs 1/IC linear fit (T-model reference only).

    For each bias file in ``all_data``, computes τ_total = 1/(2π f_T) from
    the de-embedded |h21|² 0-dB crossing.  Plots τ_total (ps) versus 1/IC
    (1/mA), linear-fits, and reports:
      - Cje (from slope):  slope = (η kT/q) · CJE  →  CJE = slope / (η · Vt).
        (Cbc contribution to slope is neglected per Cheng et al., paper Eq. 1.)
      - τB + τC (from intercept): intercept − (RC + REE) · CBC.
      - Per-file τCC = (rE + REE + RC) · CBC and τE = rE · CJE,
        with rE = η kT / (q IC).

    References:
      - Equation (1) of K. Y. D. Cheng et al., "Hot electron injection
        effect on the microwave performance of type-I/II AlInP/GaAsSb/InP
        DHBTs" — supplies the total-delay formula.  Paper notation:
        REE/RC/rE → code: Rpe/Rpc/(ηkT/qIC).
      - H. G. Liu, N. Tao, S. P. Watkins, C. R. Bolognesi, "Extraction
        of the Average Collector Velocity in High-Speed Type-II
        InP–GaAsSb–InP DHBTs," IEEE EDL 25(12), 2004 — supplies the
        v_c = W_C/(2 τ_C) split with default v_c = 4×10⁷ cm/s (peak
        across a 2000 Å InP collector).

    Hidden when ``all_data`` has < 2 files (the fit needs ≥ 2 bias points).
    The extracted Cje is for reference only — it does NOT feed back into
    the model's own extracted parameters.
    """
    import pandas as pd
    from pathlib import Path
    from ..helpers import peel_parasitics, compute_metrics, extract_limit

    if not all_data or len(all_data) < 2:
        return False

    with st.expander(tr("📐 Cje / τB+τC / τCC / τE from 1/(2πfT) vs 1/IC fit  "
                        "(T-model reference)",
                        "📐 由 1/(2πfT) 對 1/IC 擬合萃取 Cje / τB+τC / τCC / τE"
                        "（T 模型參考）"),
                     expanded=False):
        st.caption(
            tr("Reference extraction (extracted values do NOT feed back into "
               "the model fit). Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12), 2004 'Extraction of the average collector velocity in high-speed Type-II InP-GaAsSb-InP_DHBTs.pdf'",
               "僅供參考的萃取（萃取值不會回饋至模型擬合）。Liu, Tao, Watkins, "
               "Bolognesi, IEEE EDL 25(12), 2004 'Extraction of the average "
               "collector velocity in high-speed Type-II InP-GaAsSb-InP_DHBTs.pdf'"))
        st.latex(
            r"\frac{1}{2\pi f_T}=\tau_B+\tau_C+\frac{\eta k T}{q I_C}\,C_{JE}"
            r"+\left(R_C+R_{EE}+\frac{\eta k T}{q I_C}\right)C_{BC}")
        st.caption(
            tr("Notation: REE → emitter access resistance (Rpe here); "
               "RC → collector access (Rpc here); rE = ηkT/(qIC) → intrinsic "
               "base-emitter resistance.",
               "符號說明：REE → 射極存取電阻（此處為 Rpe）；"
               "RC → 集極存取電阻（此處為 Rpc）；"
               "rE = ηkT/(qIC) → 本徵射基電阻。"))

        # ── fT per file (Open+Short de-embedded, access R RETAINED) ─────────
        # The Cheng formula's RC/REE refer to the access resistance that the
        # device sees at the fT-measurement plane.  If we used the same
        # `para_eff` that the model extraction uses, ``peel_parasitics`` would
        # also strip Rpb/Rpc/Rpe (when sourced from Z-param / open-collector
        # / Cold-HBT) — that puts the fT plane past the access R and the
        # (RC+REE)·CBC intercept correction over-subtracts.
        #
        # Smart behavior: zero out Rpb/Rpc/Rpe before peeling.  When the user
        # has NOT entered any pad caps / lead L in the previous section
        # (because the files are already pre-de-embedded), every C and L in
        # ``para_eff`` is zero — and peel_parasitics becomes an algebraic
        # no-op (Y_pad=0, Z_ser=0 → returns Y_dut unchanged).  Otherwise it
        # peels only the caps and leads, exactly as requested.
        _para_pad_lead_only = dict(para_eff)
        _para_pad_lead_only["Rpb"] = 0.0
        _para_pad_lead_only["Rpc"] = 0.0
        _para_pad_lead_only["Rpe"] = 0.0

        recs = []
        for fn, d in all_data.items():
            try:
                Y     = peel_parasitics(d["S_raw"], d["freq"], d["z0"],
                                        _para_pad_lead_only)
                dfm   = compute_metrics(Y, d["freq"])
                f_ghz = dfm["Freq (GHz)"].to_numpy()
                fT_v, _, _ = extract_limit(
                    f_ghz, dfm["|h21|² (dB)"].to_numpy(),
                    dfm["fT Plateau (GHz)"].to_numpy(),
                    n_pts=2,
                    f_min=float(f_ghz[0]), f_max=float(f_ghz[-1]))
                fT_GHz   = float(fT_v) if np.isfinite(fT_v) else np.nan
                tau_tot  = (1.0 / (2.0 * np.pi * fT_GHz * 1e9)
                            if (np.isfinite(fT_GHz) and fT_GHz > 0) else np.nan)
                recs.append({"fn": fn, "stem": Path(fn).stem,
                             "fT_GHz": fT_GHz, "tau_s": tau_tot})
            except Exception as ex:
                recs.append({"fn": fn, "stem": Path(fn).stem,
                             "fT_GHz": np.nan, "tau_s": np.nan,
                             "err": str(ex)})

        # Sort by fT descending (matches Z-param method's "sort by extracted-
        # quantity desc" convention — higher fT files appear first).
        recs.sort(key=lambda r: (r["fT_GHz"] if np.isfinite(r["fT_GHz"])
                                  else -np.inf),
                  reverse=True)

        # st.markdown(
        #     "**Files (fT measured after Open+Short pad/lead de-embedding "
        #     "— access R RETAINED so RC/REE in the formula remain meaningful; "
        #     "no-op when the file is already pre-de-embedded):**")
        hcols = st.columns([0.3, 2.0, 1.2, 1.4, 1.4])
        for h, t in zip(hcols, ["", tr("File", "檔案"), "fT (GHz)",
                                  "τ_total (ps)", "IC (mA)"]):
            h.markdown(f"<small><b>{t}</b></small>", unsafe_allow_html=True)

        # Per-file checkbox + IC input.  Seed IC from rz12_Ie_{fn} (Z-param's
        # IE input) since IE ≈ IC in normal HBT operation; the user can refine.
        points = []   # list of (1/IC[1/mA], τ_total[ps], stem, IC_mA)
        for r in recs:
            c0, c1, c2, c3, c4 = st.columns([0.3, 2.0, 1.2, 1.4, 1.4])
            use_key = f"taut_use_{r['fn']}__{fname}__{model_short}"
            if use_key not in st.session_state:
                st.session_state[use_key] = True
            use = c0.checkbox(f"{tr('Use', '使用')} {r['stem']}",
                              key=use_key + "_w",
                              value=st.session_state[use_key],
                              label_visibility="collapsed")
            st.session_state[use_key] = use

            c1.markdown(f"<small>{r['stem']}</small>", unsafe_allow_html=True)
            c2.markdown(
                (f"<small>{r['fT_GHz']:.3f}</small>"
                 if np.isfinite(r["fT_GHz"]) else "<small>—</small>"),
                unsafe_allow_html=True)
            c3.markdown(
                (f"<small>{r['tau_s']*1e12:.4f}</small>"
                 if np.isfinite(r["tau_s"]) else "<small>—</small>"),
                unsafe_allow_html=True)

            ic_key = f"taut_Ic_{r['fn']}__{fname}__{model_short}"
            if ic_key not in st.session_state:
                # Default to Z-param Ie (≈ Ic in normal mode), else 0.
                st.session_state[ic_key] = float(
                    st.session_state.get(f"rz12_Ie_{r['fn']}", 0.0))
            ic_mA = c4.number_input(
                f"IC {tr('for', '對象')} {r['stem']}",
                min_value=0.0, step=0.1, format="%.3f",
                value=float(st.session_state[ic_key]),
                key=ic_key + "_w",
                label_visibility="collapsed")
            st.session_state[ic_key] = ic_mA

            if use and ic_mA > 0 and np.isfinite(r["tau_s"]):
                points.append((1.0 / ic_mA,
                                r["tau_s"] * 1e12,
                                r["stem"], ic_mA))

        if len(points) < 2:
            st.info(tr("Enter IC for at least two enabled files to fit.",
                       "請至少為兩個已啟用的檔案輸入 IC 才能進行擬合。"))
            return

        x = np.array([p[0] for p in points])     # 1/IC (1/mA)
        y = np.array([p[1] for p in points])     # τ_total (ps)
        lbls = [p[2] for p in points]
        try:
            slope, intercept = np.polyfit(x, y, 1)    # slope: ps·mA, int: ps
        except Exception as ex:
            st.error(tr(f"Linear fit failed: {ex}", f"線性擬合失敗：{ex}"))
            return

        # ── Plot ──────────────────────────────────────────────────────────────
        x_fit = np.linspace(0.0, float(x.max() * 1.08), 200)
        y_fit = slope * x_fit + intercept
        fig = go.Figure()
        # Trace names become Excel sheet names in the xlsx download, so they
        # must avoid characters Excel forbids in sheet titles: / \ ? * [ ]
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers+text", text=lbls,
            textposition="top center", name="τ_total",
            marker=dict(size=11, color="#1f77b4",
                        line=dict(color="#0d4a7a", width=1.5))))
        fig.add_trace(go.Scattergl(
            x=x_fit, y=y_fit, mode="lines",
            name=f"{tr('Fit', '擬合')}  slope={slope:.4g} ps·mA   int={intercept:.4g} ps",
            line=dict(color="#d62728", width=2, dash="dash")))
        fig.add_trace(go.Scattergl(
            x=[0.0], y=[intercept], mode="markers",
            name=f"{tr('Intercept', '截距')} = {intercept:.4f} ps",
            marker=dict(size=14, symbol="star", color="#d62728")))
        fig.update_layout(
            title=f"1/(2π f_T) vs 1/I_C  —  {model_short} {tr('model (reference)', '模型（參考）')}",
            xaxis=dict(title="1/I_C (1/mA)", rangemode="tozero",
                       showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(title="τ_total = 1/(2π f_T) (ps)",
                       showgrid=True, gridcolor="#ebebeb"),
            plot_bgcolor="white", paper_bgcolor="white", height=380,
            legend=dict(x=0.45, y=0.05,
                        xanchor="left", yanchor="bottom",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#ccc", borderwidth=1,
                        font=dict(size=10)),
            margin=dict(l=55, r=20, t=50, b=50))
        plotly_with_dl(fig,
                       key=f"taut_fit_{model_short}_{fname}",
                       filename=f"taut_fit_{model_short}_{fname}")

        # ── Inputs (defaults from current file's extraction) ─────────────────
        _inputs_hdr = tr("Inputs (defaults from this file's extraction):",
                         "輸入值（預設取自此檔案的萃取結果）：")
        st.markdown(f"**{_inputs_hdr}**")
        Re_def  = float(para_eff.get("Rpe", 0.0))
        Rc_def  = float(para_eff.get("Rpc", 0.0))
        Cbc_def = (float(params.get("Cbc",  0.0))
                   + float(params.get("Cbcx", 0.0)))   # total = intrinsic + extrinsic

        ci1, ci2, ci3, ci4, ci5 = st.columns(5)
        Re_val = ci1.number_input(
            "Re — REE (Ω)", min_value=0.0, value=Re_def, format="%.4f",
            key=f"taut_Re_{model_short}_{fname}",
            help=tr("Emitter access resistance.  Default = Rpe used in extraction.",
                    "射極存取電阻。預設值 = 萃取所用的 Rpe。"))
        Rc_val = ci2.number_input(
            "Rc (Ω)", min_value=0.0, value=Rc_def, format="%.4f",
            key=f"taut_Rc_{model_short}_{fname}",
            help=tr("Collector access resistance.  Default = Rpc used in extraction.",
                    "集極存取電阻。預設值 = 萃取所用的 Rpc。"))
        Cbc_val_fF = ci3.number_input(
            f"Cbc {tr('total', '總計')} (fF)", min_value=0.0,
            value=Cbc_def * 1e15, format="%.4f",
            key=f"taut_Cbc_{model_short}_{fname}",
            help=tr("Total base-collector cap.  Default = Cbc + Cbcx (intrinsic + extrinsic).",
                    "總基極-集極電容。預設值 = Cbc + Cbcx（本徵 + 外徵）。"))
        eta_val = ci4.number_input(
            f"η（{tr('ideality', '理想因子')}）", min_value=0.5, max_value=3.0,
            value=1.0, step=0.05, format="%.3f",
            key=f"taut_eta_{model_short}_{fname}",
            help=tr("Ideality factor for r_E = η kT/(q IC).  Set this from a "
                    "Gummel-plot fit of your device (typical InP HBT: 1.0–1.2).",
                    "r_E = η kT/(q IC) 的理想因子。請由元件的 Gummel 圖擬合設定"
                    "（典型 InP HBT：1.0–1.2）。"))
        T_K = ci5.number_input(
            "T (K)", min_value=1.0, value=300.0, step=5.0, format="%.1f",
            key=f"taut_T_{model_short}_{fname}",
            help=tr("Temperature for kT/q.", "kT/q 計算所用的溫度。"))

        Cbc_val = Cbc_val_fF * 1e-15
        Vt = 1.380649e-23 * T_K / 1.602176634e-19         # kT/q  (V)

        # ── Derived (slope → Cje; intercept → τB+τC) ─────────────────────────
        # Units conversion: slope is in ps·mA = (s·1e-12)·(A·1e-3) = s·A · 1e-15.
        # In SI, slope_SI = (η · Vt) · Cje  with  [V · F] = [s · A].
        # ⇒ Cje[F] = slope_SI / (η · Vt) = slope[ps·mA] · 1e-15 / (η · Vt).
        # ⇒ Cje[fF] = slope[ps·mA] / (η · Vt[V]).
        Cje_fF = (slope / (eta_val * Vt)) if (eta_val * Vt) > 0 else 0.0
        Cje_F  = Cje_fF * 1e-15

        tau_BC_ps = intercept - (Re_val + Rc_val) * Cbc_val * 1e12

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("Slope", "斜率"), f"{slope:.4g} ps·mA",
                  help=tr("d(τ_total)/d(1/I_C) — drives Cje.",
                          "d(τ_total)/d(1/I_C) — 決定 Cje。"))
        m2.metric(tr("Intercept", "截距"), f"{intercept:.4f} ps",
                  help=tr("τ_total extrapolated to 1/I_C → 0.",
                          "τ_total 外插至 1/I_C → 0 的值。"))
        m3.metric(f"Cje  ({tr('ref.', '參考')})", f"{Cje_fF:.4f} fF",
                  help=tr("Cje = slope / (η · kT/q).  Reference only.",
                          "Cje = slope / (η · kT/q)。僅供參考。"))
        m4.metric(f"τB + τC  ({tr('ref.', '參考')})", f"{tau_BC_ps:.4f} ps",
                  help=tr("τB+τC = intercept − (Rc + Re) · Cbc.",
                          "τB+τC = intercept − (Rc + Re) · Cbc。"))

        # ── Split τB / τC using assumed collector velocity v_c ───────────────
        # Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12), 2004 — "Extraction
        # of the Average Collector Velocity in High-Speed Type-II
        # InP–GaAsSb–InP DHBTs" — found v_c peaks at 4×10⁷ cm/s across a
        # 2000 Å InP collector at V_CB ≈ 0.4 V.  Using the same definition
        # v_c = W_C / (2 τ_C):
        #     τ_C = W_C / (2 v_c)
        #     τ_B = (τ_B + τ_C)_intercept − τ_C
        # Default v_c is the Liu peak for InP collectors; user can edit for
        # other collector materials / thicknesses / biases.  Publishes to
        # session state so the τB / τC number_inputs farther down offer a
        # "v_c = …" quickset button.
        st.markdown(tr(
            "**Split τB / τC using assumed average collector velocity "
            "(Liu et al. 2004 — default for InP collector):**",
            "**依假設之平均集極速度拆分 τB / τC**"
            "**（Liu et al. 2004 — InP 集極預設值）：**"))
        cv1, cv2, cv3, cv4 = st.columns(4)
        Wc_nm = cv1.number_input(
            "W_C (nm)", min_value=1.0, value=120.0, step=10.0, format="%.2f",
            key=f"taut_Wc_{model_short}_{fname}",
            help=tr("Collector depletion width.", "集極空乏區寬度。"))
        v_c_cms = cv2.number_input(
            "v_c (cm/s)", min_value=1.0e5, value=4.0e7,
            step=1.0e6, format="%.3e",
            key=f"taut_vc_{model_short}_{fname}",
            help=tr("Average collector velocity.  Default 4×10⁷ cm/s — peak "
                    "value extracted for 2000 Å InP collectors in Liu, Tao, "
                    "Watkins, Bolognesi, IEEE EDL 25(12), 2004.  Adjust for "
                    "other collector materials / thicknesses / biases.",
                    "平均集極速度。預設 4×10⁷ cm/s — 取自 Liu, Tao, Watkins, "
                    "Bolognesi, IEEE EDL 25(12), 2004 對 2000 Å InP 集極萃取"
                    "之峰值。可依不同集極材料 / 厚度 / 偏壓調整。"))
        v_c_ms     = v_c_cms * 1e-2                      # cm/s → m/s
        Wc_m       = Wc_nm * 1e-9
        tauC_vc_s  = Wc_m / (2.0 * v_c_ms)               # seconds
        tauC_vc_ps = tauC_vc_s * 1e12
        tauB_vc_ps = tau_BC_ps - tauC_vc_ps              # ps
        tauB_vc_s  = tauB_vc_ps * 1e-12

        cv3.metric(f"τC  ({tr('from v_c', '由 v_c 計算')})", f"{tauC_vc_ps:.4f} ps",
                   help=tr("τ_C = W_C / (2 v_c).", "τ_C = W_C / (2 v_c)。"))
        cv4.metric(f"τB  ({tr('from v_c', '由 v_c 計算')})", f"{tauB_vc_ps:.4f} ps",
                   help=tr("τ_B = (τ_B+τ_C) − τ_C.", "τ_B = (τ_B+τ_C) − τ_C。"))

        # Publish v_c-derived values (SI seconds) so τB / τC number_inputs
        # can read them via the "v_c = …" quickset button.  Stored only when
        # finite; deleted otherwise so the button auto-hides when the fit
        # degrades.
        for pub_key, val in (
            (f"taut_pub_tauB_{model_short}_{fname}", tauB_vc_s),
            (f"taut_pub_tauC_{model_short}_{fname}", tauC_vc_s),
        ):
            if np.isfinite(val) and abs(val) > 0:
                st.session_state[pub_key] = float(val)
            else:
                st.session_state.pop(pub_key, None)

        # ── Per-file τCC and τE ─────────────────────────────────────────────
        rows = []
        for _, tau_total_ps, stem, ic_mA in points:
            ic_A   = ic_mA * 1e-3
            rE_i   = (eta_val * Vt) / ic_A if ic_A > 0 else np.nan
            tau_cc = (rE_i + Re_val + Rc_val) * Cbc_val * 1e12  # ps
            tau_E  = rE_i * Cje_F * 1e12                         # ps
            rows.append({
                "File":              stem,
                "IC (mA)":           f"{ic_mA:.4f}",
                "rE = ηkT/(qIC) (Ω)": f"{rE_i:.4f}",
                "τCC = (rE+Re+Rc)·Cbc (ps)": f"{tau_cc:.4f}",
                "τE = rE·Cje (ps)":  f"{tau_E:.4f}",
                "τ_total measured (ps)": f"{tau_total_ps:.4f}",
            })
        st.markdown(f"**{tr('Per-file derived delays (using inputs above):', '各檔案推導延遲（依上述輸入值）：')}**")
        st.dataframe(pd.DataFrame(rows),
                     width="stretch", hide_index=True)
    return True

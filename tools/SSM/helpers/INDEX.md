# SSM — function index

Single source of truth for every function across `tools/SSM/`. Skim this
file to find which submodule owns a function before grep'ing the codebase.

All paths are relative to `tools/SSM/`. Line numbers point at the `def`
(or `class`) line and may drift slightly as files evolve — use them as
jump-start hints, not as fixed addresses.

Helpers (`helpers/*.py`) re-export from `tools.SSM.helpers`, so call
sites can simply do:

```python
from tools.SSM.helpers import parse_s2p, make_smith, compute_metrics
```

The submodule paths below are for jump-to-definition only.


---

## Top-level orchestration

### [`main_ssm_extraction.py`](../main_ssm_extraction.py) — UI entry point

| Function | Line | Purpose |
|---|---|---|
| `_agg(arr, n0, n1, method, trim_pct=20)` | 48 | Median / trimmed-mean reducer used by Step 1 UI controls. |
| `_extract_ui(fname, key, freq, default_frac_lo, default_frac_hi)` | 61 | Frequency-range + method selector. Returns `(n0, n1, method, trim_pct)`. |
| `render_ssm_tab(fname, S_raw, freq, z0, open_data, short_data, all_data=None)` | 82 | Public entry — renders the complete SSM extraction tab for one DUT file (Steps 1–3, all model panels, downloads, summary). |
| `_render_s2p_downloads(fname, freq, z0, para_eff, sim_results)` | 595 | Modeled-DUT `.s2p` Touchstone download UI per registered model. |
| `_render_summary_table(fname, para_eff, cold_res, extract_results, registry)` | 635 | Multi-layer parameter summary table + CSV download. Pulls live (post fine-tune) values when available. |

### [`ssm_override.py`](../ssm_override.py) — pre-extraction series-R picker

| Function | Line | Purpose |
|---|---|---|
| `render_unified_pre_override(fname, para_step1, cold_res, rz12_Re)` | 20 | Unified pad parameter UI. For Rb/Rc/Re it selects the highest source (Short Step 1b / Cold-HBT / Z-param method / Open-collector / Custom). Returns the effective `para_eff` dict (SI). |

### [`ssm_access_resistance.py`](../ssm_access_resistance.py) — series/access-R extractors

| Function | Line | Purpose |
|---|---|---|
| `render_rz12_section(all_data, para_eff, fname)` | 41 | Z-parameter method UI: Re(Z₁₂) vs 1/IE → Re. (Gao Ch. 5.5.1) |
| `render_open_collector_section(all_data, para_eff, fname)` | 191 | Open-collector method UI: Re(Z₁₁−Z₁₂)/Re(Z₂₂−Z₁₂)/Re(Z₁₂) vs 1/IB → Rb/Rc/Re. Writes `ocm_R*_{fname}` session keys. |
| `_render_cold_hbt(fname, open_data, para_step1, do_measured, freq, re_zparam=None, open_arr=None, short_arr=None)` | 316 | Cold-HBT extraction UI (Gao §5.5.2). Strips pad caps + lead L + Re from Z_cor before A/B/C/D. Returns a `cold_res` dict or `None`. |
| `_cold_input(container, label, key, init_disp, fmt, step, arr_si=None, scale=1.0, unit="")` | 350 | Cold-HBT inner helper: `number_input` paired with quickset buttons for one parasitic. |

### [`ssm_plots.py`](../ssm_plots.py) — Step 1 / Step 2 diagnostic Streamlit blocks

| Function | Line | Purpose |
|---|---|---|
| `render_open_plots(open_data, para_caps, open_arr, fname="")` | 32 | Five expanders for Open dummy: extra-element controls, capacitance plot, conductance plot, Im(Y)/ω vs 1/ω², measured vs modeled Smith. Returns `{cap: (mode, extra_SI)}`. |
| `_get_mode_extra(cap)` | 74 | Inner helper — return `(mode, extra_SI)` pulled from session state for one cap. |
| `render_short_plots(short_arr, para_short, fname="", freq=None)` | 196 | Lead-inductance plot for Short dummy (Lb/Lc/Le vs frequency, fixed 0–150 pH). |
| `_rlc_params_summary(p, fname)` | 254 | Build header param dict for a de-embedded `.s2p` file. |
| `_compare_bode_smith(*, S_a, S_b, freq, fname, key_suffix, label_a, label_b, color_a, color_b, smith_meas_label, smith_sim_label, gain_title)` | 275 | Side-by-side Bode (h21²+U) and Smith chart comparing two S-param datasets. Returns `(fT, fmax)` of the right-hand trace. |
| `render_os_deemb_preview(S_raw, freq, z0, para_step1, fname)` | 371 | Step 2 — Raw vs Open/Short de-embedded preview + download. Returns `S_step1`. |
| `render_intrinsic_preview(S_raw, freq, z0, para_step1, para_eff, fname, S_step1=None)` | 442 | Step 3 footer — OS de-embedded vs Intrinsic preview + download. |
| `render_ft_fmax_card(S_mea, S_sim, freq, *, model_name, key, height=560)` | 505 | Compact per-model fT/fmax mini-plot used beside each Smith chart (legend reports both meas + model fT/fmax with extrap annotation). |
| `_apply_text_autoformat(text, pattern, replacement)` | 660 | Regex substitution with `\1..\9` capture-group templating (used for Smith-label mathtext). |
| `_smith_grid_values(n)` | 689 | Return `(r_values, x_values)` for N evenly-spaced Smith-chart grid lines. |
| `_draw_mpl_smith_background(ax, line_lw, grid_lw, density)` | 701 | Draw constant-R / constant-X grid arcs for a unit Smith chart on a matplotlib axis. |
| `render_matplotlib_smith(S_mea=None, S_sim=None, fname="", topo_key="", *, sets=None, default_multiplier=1.0)` | 740 | Publication-style matplotlib Smith chart. Backward-compatible (mea/sim) and extensible (`sets=[…]`) call forms. Per-trace decimation, color modes, free-text overlays. |
| `render_ft_fmax_overlay(S_raw, sim_results, freq, fname)` | 1092 | Bode plot showing |h21|² and Mason U for the measured DUT plus every simulated model, with auto 20 dB/dec extrapolation past the band. |

---

## Helpers (`helpers/*.py`)

### [`helpers/rf_math.py`](rf_math.py) — Pure RF math (no Streamlit, no plotting)

| Function | Line | Purpose |
|---|---|---|
| `s_to_y(S, z0=50.0)` | 14 | S → Y conversion, vectorised across all frequency points. |
| `_inv2(M)` | 25 | Per-row 2×2 analytic inverse (no per-matrix Python loop). |
| `y_to_z`, `z_to_y` | 39, 40 | Aliases for `_inv2`. |
| `y_to_s_single(Y, z0=50.0)` | 43 | Y → S for a single 2×2 matrix. |
| `y_to_s_batch(Y, z0=50.0)` | 49 | Y → S over a batch (delegates to `y_to_s_vec`). |
| `y_to_s_vec(Y, z0=50.0, xp=None)` | 54 | GPU-aware vectorised Y → S using hand-inlined 2×2 algebra (numpy or cupy). |
| `inv2x2(M, xp=None)` | 97 | Analytic 2×2 batched inverse — bypasses cuSOLVER. |
| `mm2x2(A, B, xp=None)` | 124 | Analytic 2×2 batched matmul — bypasses cuBLAS. |
| `safe_median(arr, n=None)` | 149 | Median of finite values, returns `0.0` if empty. |
| `strict_freq_check(f_dut, f_dummy, label)` | 156 | Raise `ValueError` if frequency grids differ. |
| `open_elem_Y(C, mode, extra, w)` | 164 | Pad capacitor admittance — None / Parallel L / Series L / Series R. |
| `short_lead_Z(R, L, Cpar, w)` | 184 | Series-lead impedance with optional parallel cap. |
| `_smith_grid_xy(max_r=1.0)` | 200 | Cached coordinate arrays for the Smith-chart background. |
| `extended_smith_grid(max_r=1.0)` | 241 | Plotly Smith chart background traces (built fresh each call from cached coords). |
| `params_hash(p)` | 258 | MD5 hash of a parameter dict, for caching. |

### [`helpers/s2p_io.py`](s2p_io.py) — Touchstone / CSV I/O and dummy simulators

| Function | Line | Purpose |
|---|---|---|
| `build_Y_pad(p, w)` | 34 | 2×2 pad admittance matrix at angular freq `w`. |
| `build_Z_ser(p, w)` | 49 | 2×2 series-lead impedance matrix at angular freq `w`. |
| `write_s2p(freq_hz, S, title="", params=None)` | 65 | Serialize to Touchstone `.s2p` (DB format) bytes. Header `!` lines list `params`. |
| `parse_s2p(content)` | 93 | Parse a Touchstone `.s2p` (accepts str or bytes) → `(freq, S, z0)`. |
| `parse_s2p_bytes(raw)` | 138 | Bytes-only alias of `parse_s2p` (backwards-compat). |
| `parse_csv(content, z0=50.0)` | 143 | Parse VNA CSV export (RI columns) → `(freq, S, z0)`. |
| `interpolate_s2f(f_src, S_src, f_tgt)` | 159 | Interpolate S to a new frequency grid. |
| `load_cal(fobj)` | 172 | Streamlit-aware S2P loader (handles `UploadedFile`, sidebar error on parse fail). |
| `simulate_open(p, freq, z0=50.0)` | 192 | Forward-simulate Open dummy from pad params (cached). |
| `simulate_short(p, freq, z0=50.0)` | 208 | Forward-simulate Short dummy from pad+lead params (cached). |

### [`helpers/deembed_math.py`](deembed_math.py) — Open/Short extraction & de-embedding

| Function | Line | Purpose |
|---|---|---|
| `_agg_arr(arr, n0, n1, method="Median", trim_pct=20)` | 35 | Median / trimmed-mean aggregator over an index window. |
| `_open_elem_Y_vec(C, mode, extra, omega, xp)` | 50 | Vectorised pad-cap admittance over an `(N,)` omega array. |
| `_short_lead_Z_vec(R, L, Cpar, omega, xp)` | 63 | Vectorised series-lead impedance over an `(N,)` omega array. |
| `build_Y_pad_vec(p, omega, xp)` | 71 | `(N,2,2)` pad admittance — fully vectorised. |
| `build_Z_ser_vec(p, omega, xp)` | 85 | `(N,2,2)` series-lead impedance — fully vectorised. |
| `_open_elem_Y_batch(C, mode, extra, omega, xp)` | 103 | `_open_elem_Y_vec` variant where C may be `(B,1)`. |
| `_short_lead_Z_batch(R, L, Cpar, omega, xp)` | 116 | `_short_lead_Z_vec` variant where R/L may be `(B,1)`. |
| `_b1(p, key, default, xp)` | 124 | Fetch `p[key]` (or default), reshape `(B,)` → `(B,1)`; scalars stay scalar. |
| `build_Y_pad_batch(p, omega, B, N, xp)` | 133 | Pad admittance as 4 broadcast planes for sweep tuning. |
| `build_Z_ser_batch(p, omega, B, N, xp)` | 151 | Series-lead impedance as 4 broadcast planes for sweep tuning. |
| `step_open(open_data, n0, n1, method, trim_pct)` | 165 | Extract Cpbe/Cpce/Cpbc + diagnostic conductance arrays from Open dummy. |
| `step_short(short_data, freq, Cpbe, Cpce, Cpbc, ...)` | 209 | Extract Lb/Lc/Le, Rpb/Rpc/Rpe from Short dummy (Open subtracted first). |
| `peel_parasitics(S_raw, freq, z0, p)` | 284 | Remove pad+lead parasitics → Y_intrinsic (cached). |
| `deembed_open_short(Y_dut, Y_open, Y_short)` | 307 | Standard open-short de-embedding (Gao §4.2). |
| `deembed_thru_half(Y_dut, Y_thru_deemb)` | 312 | THRU/2 half-impedance subtraction. |

### [`helpers/metrics.py`](metrics.py) — Gain figures of merit

| Function | Line | Purpose |
|---|---|---|
| `compute_h21_U(S)` | 20 | Compute |h21|² (dB) and Mason U (dB) from S-parameters (z0 = 50 Ω). |
| `find_ft_fmax(f_ghz, h21_db, U_db)` | 37 | Linear interpolation of 0 dB crossings → `(fT, fmax)`; either may be `None`. |
| `extrap_20dbdec(f_ghz, gain_db, n_pts=60, f_max_target=None)` | 52 | Slope-locked −20 dB/dec extrapolation → `(f_ext, g_ext, f_zero)`. |
| `single_pole_extrap(f_ghz, gain_db, idx_lo, idx_hi, n_pts=60, f_max_target=None)` | 87 | Log-linear (single-pole) fit over a user index window → `(f_ext, g_ext, f_zero, slope, intercept)`. Slope may differ from −20 — that disagreement is itself diagnostic. |
| `compute_metrics(Y, freq_hz)` | 154 | Build DataFrame: Freq, |h21|², Mason U, MAG/MSG, K, fT/fmax plateaus. |
| `extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max)` | 185 | Genuine-crossing fT/fmax extractor with extrapolation fallback (≥10 consecutive points above 0 dB). |

### [`helpers/plotly_plots.py`](plotly_plots.py) — Plotly Smith / Bode / Plateau builders

| Function | Line | Purpose |
|---|---|---|
| `PALETTE` | 27 | 10-color hex palette for trace cycling. |
| `darken(c)` | 33 | Darken a `#rrggbb` color by 45 units per channel. |
| `bode_layout(title, ytitle, yr, xr)` | 43 | Plotly layout dict for Bode/plateau plots (log-x). |
| `make_smith(S, f_array, f_min, f_max, toggles, scales, title, max_r=1.0)` | 59 | Plotly Smith chart with up to 4 selectable S-param traces. |
| `make_bode(df, title, xr, yr, sh21, su, smag, color, *, show_20db=True, show_sp=False, sp_window_idx=None, extrap_f_max=None, return_extrap_df=False)` | 112 | Plotly Bode plot with optional −20 dB/dec and single-pole-fit extrapolations. May return `(fig, extrap_df)`. |
| `_add_20db(y_vals, color_, kind, key)` | 149 | Inner helper inside `make_bode` — append a slope-locked extrap trace. |
| `_add_sp(y_vals, color_, kind, key)` | 163 | Inner helper inside `make_bode` — append a single-pole-fit extrap trace. |
| `_build_bode_extrap_df(df, extrap_curves, sh21, su)` | 219 | Combine measured + extrapolated values into one stitched DataFrame for download. |
| `make_plateau(df, res, title, xr, sh21, su, smag, color)` | 269 | Plotly GBP plateau plot (fT, fmax(U), fmax(MAG)). |

### [`helpers/widgets.py`](widgets.py) — Streamlit input widgets

| Function | Line | Purpose |
|---|---|---|
| `apply_pending(target_key)` | 27 | Promote a pending quickset value into the widget's session-state key. **Call BEFORE the paired `number_input`.** |
| `_candidates(arr_disp, default_disp, cold_disp=None)` | 36 | Build `[(label, value), …]` list from optional sources. |
| `quickset_buttons(*, container, key_prefix, target_key, arr_disp=None, default_disp=None, cold_disp=None, unit="", fmt="%.4g", layout="side")` | 53 | Row of one-click buttons (mean / median / low f / high f / default / cold) that overwrite a paired `number_input` via the pending session-state + rerun pattern. |

### [`helpers/chart_export.py`](chart_export.py) — Excel export & Streamlit UI helpers

| Function | Line | Purpose |
|---|---|---|
| `EXCEL_MIME` | 19 | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`. |
| `_axis_text(axis_obj)` | 26 | Safely fetch a Plotly axis title text. |
| `_is_smith(fig)` | 33 | True when the figure is a Smith chart (Re(Γ) / Im(Γ) axes). |
| `_smith_col_name(trace_name)` | 38 | Map a Smith-trace name → compact Excel column prefix (e.g. `S11_meas`). |
| `_freq_sheet_name(x_lbl, x_arr)` | 61 | Build an Excel sheet name from the frequency range (e.g. `100m_to_67g`). |
| `_collect_traces(fig)` | 108 | Return `[(name, x, y), …]` for exportable traces, filtering grid lines and constant references. |
| `fig_to_excel_bytes(fig)` | 141 | Extract Plotly traces → `.xlsx` bytes (smart Smith vs normal layout). |
| `plotly_with_dl(fig, key, filename="", width="stretch", container=None, **kwargs)` | 225 | Render Plotly chart + compact xlsx download button below it. |
| `build_excel(summary_df, all_data)` | 267 | Multi-sheet workbook: Summary + per-DUT DataFrames. |
| `metric_card(col, title, val, sub, color="#4A90D9")` | 291 | Styled HTML metric tile (used in IOED tab_ind, batch tab). |

---

## Models (`models/*.py`)

### [`models/__init__.py`](../models/__init__.py) — Registry + abstract base

| Class / function | Line | Purpose |
|---|---|---|
| `class AbstractSSMModel(ABC)` | 18 | Public API every SSM model must implement (`extract`, `simulate`, `render_*`). Class attrs `NAME`, `SHORT`, `TOPOLOGY_CHAR`. |
| `AbstractSSMModel.extract(Y_ex1, freq, n_low, **kwargs)` | 37 | Extract intrinsic params from the de-embedded admittance. Returns `(params, arrays)`. |
| `AbstractSSMModel.simulate(params, freq, z0=50.0)` | 59 | Forward-simulate S-parameters from a fully populated params dict. |
| `AbstractSSMModel.render_step_formulas(cls)` | 72 | Render extraction-step LaTeX in Streamlit. |
| `AbstractSSMModel.render_results_table(cls, params)` | 81 | Render extracted scalar parameters as a Streamlit dataframe. |
| `AbstractSSMModel.render_formula_trace(cls)` | 86 | Render a collapsible expander with the full extraction + simulation chain. |
| `AbstractSSMModel.render_override_and_smith(cls, fname, S_raw, freq, z0, para_eff, extract_result, **kwargs)` | 94 | Render fine-tune override UI + Smith chart. Returns `S_sim` or `None`. |
| `AbstractSSMModel.get_s2p_header_params(cls, params, para_eff)` | 118 | Default human-readable param dict for `.s2p` header (override to customise). |
| `REGISTRY` | 140 | `{SHORT: ModelClass}` — drives the model UI loop in `main_ssm_extraction.py`. |
| `DEFAULT_SELECTION` | 149 | List of model SHORTs shown on first run. |

### [`models/base_ui.py`](../models/base_ui.py) — Shared Streamlit UI helpers for all models

| Function | Line | Purpose |
|---|---|---|
| `PAD_SPECS` | 63 | Tuple list `(key, label, SI_scale, unit, fmt, step)` for the nine pad parameters. |
| `ssm_residual(S_mea, S_mod)` | 79 | RMS relative S-parameter residual across all four ports (%). |
| `_port_residuals(S_mea, S_mod)` | 93 | Per-port residual dict (`Total`, S11, S12, S21, S22) in %. |
| `_port_residuals_batch(S_mea, S_mod_batch, xp)` | 110 | Batched fused per-port residuals — minimises GPU kernel launches. |
| `render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith", show_title=True, meas_label="Meas.", sim_label="Model")` | 140 | Per-model Plotly Smith with measured markers + dashed model traces. |
| `render_smith_with_ftfmax(S_raw, S_sim, freq, model_name, model_short, fname, scales=None)` | 177 | 2-column: Smith chart (left) + fT/fmax mini-card (right) + residual line. |
| `smith_scale_controls(fname, topo_key)` | 209 | Four `number_input` fields for per-trace Smith display scaling (S11/S12/S21/S22 ×). |
| `sync_pad_from_preov(fname, topo_key, para_eff)` | 227 | Copy current pre-extraction pad values into per-topology session state when the upstream MD5 hash changes. |
| `_render_cbex_sweep_tool(*, cbex_arr, freq, f_ghz, f_min_v, f_max_v, cbex_scale, cbex_unit, cbex_param_key, model_short, fname, g_idx, rng_tag, cbex_sweep_fn, param_groups)` | 240 | Min/Step/Max + Calculate UI for the Cbex sweep that minimises std(Cbcx) over the Cbcx group's frequency window. Stages the result into a pending key + rerun. |
| `render_interactive_param_groups(params, arrays, freq, fname, model_short, param_groups, cold_res=None, cold_param_map=None, reextract_fn=None, cbex_sweep_fn=None)` | 378 | The big interactive expander: per-group section heading, dependency info, "same range as previous" button, frequency-range slider, per-frequency line plots with median dashed line, per-param `number_input`, optional Cbex-sweep tool. Returns a copy of `params` with all overrides applied (SI). |
| `_make_sweep_values(min_val, max_val, step)` | 918 | Generate sweep values (always includes `max_val` as the last point). |
| `render_tuning_expander(model_cls, all_p, S_raw, freq, z0, tuning_specs, fname, topo_key)` | 933 | Tuning expander below the Smith chart: per-param min/step/max + total-combination count + residual table over the full sweep. Optionally fp32 main loop with fp64 top-K rerank. |

`class SSMModelTemplate` (line 3110):

Mixin-style parent class for SSM model classes. Provides default implementations of `simulate_vec`, `simulate_batch`, `_cached_simulate_vec`, `render_results_table`, and `render_override_and_smith`. Concrete model classes inherit from both `SSMModelTemplate` and `AbstractSSMModel`, then supply per-model behaviour via class attributes (`_INT_SPECS`, `_EXT_SPECS`, `_Y_INT_VEC_FN`, `_Y_INT_BATCH_FN`, `_SIM_WRAP_VEC_FN`, `_SIM_WRAP_BATCH_FN`, optional `_TUNING_PAD_SPECS`) and four classmethod hooks (`_do_override_ui`, `_render_topology`, `_results_rows`, optional `_render_results_trace`). See the docstring in `base_ui.py` for the full contract.

### [`models/_shared.py`](../models/_shared.py) — Helpers lifted from cheng.py / xu.py

| Function | Line | Purpose |
|---|---|---|
| `_b1(p, key, default, xp, dtype=None)` | 16 | Reshape `(B,)` → `(B,1)`; scalars stay scalar. Optional dtype coercion for fp32 sweep path. |
| `_detect_B(p, xp)` | 33 | Determine batch size B from any `(B,)`-shaped value in `p`. |
| `_stack22(a00, a01, a10, a11, xp)` | 48 | Stack four planes into a `(..., 2, 2)` tensor in 3 launches. |
| `_try_download_inter()` | 73 | One-time fetch of Inter font into `models/fonts/` (used for topology illustration overlay). |
| `has_inter()` | 96 | True if Inter is installed locally or successfully cached. |
| `_load_font(size)` | 105 | Load a TrueType font with Inter → Arial → fallback chain. |

### [`models/cheng.py`](../models/cheng.py) — Cheng (2022) T and π topologies

Module-level helpers:

| Function | Line | Purpose |
|---|---|---|
| `_sweep_cbex_stds_cheng(Y_ex1, freq, cbex_SI_array, mask)` | 35 | For each candidate Cbex, rebuild Y_ex2 and return std(Cbcx_arr[mask]). Shared by ChengT and ChengPi (both use the same Cbcx formula post-peel). |
| `_step2_T(Y_ex1, freq, n_low)` | 74 | Cheng [Eq. 13, 22] — Cbex_T (T variant) and Cbcx. Returns Y_ex2 in arrays. |
| `_step2_pi(Y_ex1, freq, n_low)` | 110 | Cheng [Eqs. 26–28] — Cbex_π (π variant) and Cbcx (same Eq. 22). |
| `_step3_T(Y_ex2, freq, Cbcx, n_low)` | 149 | Cheng [Eqs. 16, 29–31] — T-topology intrinsic Rbi/Rbe/Cbe/Rbc/Cbc/α₀/τB/τC. |
| `_step3_pi(Y_ex2, freq, Cbcx, n_low)` | 209 | Zhang et al. — π-topology intrinsic Rbi/Rbe/Cbe/Cbc/Gm0/τ. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | 259 | Add extrinsic caps + pad/lead parasitics around an intrinsic-Y matrix (per-freq loop, scalar). |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | 280 | Vectorised forward sim — no per-freq loop. Works on numpy or cupy. |
| `_Y_int_T_vec(p, omega, xp)` | 319 | Vectorised T-topology intrinsic-Y matrix → (N,2,2). |
| `_Y_int_Pi_vec(p, omega, xp)` | 334 | Vectorised π-topology intrinsic-Y matrix → (N,2,2). |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | 355 | Batched forward sim over (param_combo × freq). Hand-inlined 2×2 algebra, optional fp32 path via `cache["_cdtype"]`. |
| `_Y_int_T_batch(p, omega, B, N, xp, cache=None)` | 469 | Batched T-topology intrinsic-Y, returns 4 broadcastable planes. |
| `_Y_int_Pi_batch(p, omega, B, N, xp, cache=None)` | 534 | Batched π-topology intrinsic-Y, returns 4 broadcastable planes. |
| `_fmt_param(key, val_si)` | 715 | Format a parameter SI value for display on the topology illustration (uses `_PARAM_DISPLAY` ladder). |
| `_render_topology_illustration(all_p, topology, fname)` | 733 | Overlay live parameter values on the schematic template PNG; render via `st.image`. `topology` is `"T"` or `"pi"`. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS)` | 823 | Render the fine-tune override expander (Pad / Extrinsic / Intrinsic) for one Cheng topology. Returns `all_p` dict in SI. |

Shared helpers `_b1`, `_detect_B`, `_stack22`, `_try_download_inter`, `has_inter`, `_load_font` are imported from [`models/_shared.py`](../models/_shared.py).

`class ChengT(SSMModelTemplate, AbstractSSMModel)` (line 875):

Inherited from `SSMModelTemplate` (in `base_ui.py`):
`simulate_vec`, `simulate_batch`, `render_results_table`, `render_override_and_smith`.

| Method | Line | Purpose |
|---|---|---|
| `extract(Y_ex1, freq, n_low, **kwargs)` | 961 | Step 2 (Cbex_T, Cbcx) → Step 3 (intrinsic T params). |
| `sweep_cbex(Y_ex1, freq, cbex_SI_array, mask)` | 974 | Wrapper around `_sweep_cbex_stds_cheng` for the interactive Cbex stability search. |
| `simulate(params, freq, z0=50.0)` | 984 | Scalar forward sim using the analytic T intrinsic-Y. |
| `reextract(Y_ex1, freq, n_low, overrides, changed_group_idx, live_arrays)` | 1005 | Cascade re-extract when an upstream interactive group is overridden (Cbex → Cbcx → Step 3 → τB → τC). |
| `_results_rows(params)` | 1081 | Template hook — list of `(symbol, value, unit)` rows for the results table. |
| `_render_results_trace()` | 1098 | Template hook — formula-trace expander (📐 LaTeX dependency chain). |
| `_do_override_ui(fname, calc_vals)` | 1123 | Template hook — call `_override_ui` with Cheng-T specs. |
| `_render_topology(all_p, fname)` | 1128 | Template hook — render T-topology schematic illustration. |

`class ChengPi(SSMModelTemplate, AbstractSSMModel)` (line 1136): π variant. Same template hooks; no formula-trace expander.

### [`models/xu.py`](../models/xu.py) — Xu's T (2014)

Module-level helpers:

| Function | Line | Purpose |
|---|---|---|
| `_step2_T(Y_ex1, freq)` | 36 | Xu — Cbcx direct from Y_ex1 (no Cbex peel). Rbcx defaults to 285 kΩ (user-tunable, not extracted). |
| `_step3_T(Y_ex1, freq, Cbcx, Rbcx, n_low)` | 59 | Xu — peel Ybcx (=1/Rbcx + jωCbcx) then run the Cheng-style Step 3 intrinsic extraction. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | 127 | Scalar forward sim with Ybcx parallel network. |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | 143 | Vectorised forward sim. |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | 182 | Batched forward sim for tuning sweeps. |
| `_Y_int_T_vec(p, omega, xp)` | 292 | Vectorised T-topology intrinsic-Y (shared formula with Cheng-T). |
| `_Y_int_T_batch(p, omega, B, N, xp, cache=None)` | 307 | Batched intrinsic-Y planes. |
| `_aka(cheng_lbl, xu_lbl)` | 381 | Render combined "Cheng / Xu" symbol label for the override UI. |
| `_fmt_param(key, val_si)` | 486 | Format a parameter SI value for display on the topology illustration. |
| `_render_topology_illustration(all_p, fname)` | 505 | Overlay live values on the Xu schematic template PNG. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS)` | 592 | Fine-tune override expander for the Xu T topology. |

Shared helpers `_b1`, `_detect_B`, `_stack22`, `_try_download_inter`, `has_inter`, `_load_font` are imported from [`models/_shared.py`](../models/_shared.py).

`class XuModel(SSMModelTemplate, AbstractSSMModel)` (line 641, `SHORT="XuT"`):

Inherited from `SSMModelTemplate` (in `base_ui.py`):
`simulate_vec`, `simulate_batch`, `render_results_table`, `render_override_and_smith`.
Overrides class attribute `_TUNING_PAD_SPECS = _XU_PAD_SPECS` so the tuning expander uses Xu's relabelled pad-cap names.

| Method | Line | Purpose |
|---|---|---|
| `extract(Y_ex1, freq, n_low, **kwargs)` | 715 | Step 2 (Cbcx; Rbcx defaulted) → Step 3 (intrinsic T params). |
| `simulate(params, freq, z0=50.0)` | 729 | Scalar forward sim with parallel Rbcx ∥ Cbcx. |
| `reextract(Y_ex1, freq, n_low, overrides, changed_group_idx, live_arrays)` | 750 | Cascade re-extract; Rbcx read from overrides or 285 kΩ default. |
| `_results_rows(params)` | 818 | Template hook — list of `(symbol, value, unit)` rows for the results table. |
| `_render_results_trace()` | 835 | Template hook — formula-trace expander (📐 LaTeX dependency chain). |
| `_do_override_ui(fname, calc_vals)` | 859 | Template hook — call `_override_ui` with Xu specs. |
| `_render_topology(all_p, fname)` | 864 | Template hook — render Xu schematic illustration. |

### [`models/degachi.py`](../models/degachi.py) — Degachi & Ghannouchi (2008) augmented π *(currently disabled in the registry — uncomment in `models/__init__.py` to re-enable)*

Module-level helpers (Eqs. follow the 2008 IEEE TED paper):

| Function | Line | Purpose |
|---|---|---|
| `_h_Tbi(Z1, Z3, omega, omega2, n_fit)` | 59 | [Eq. 8, 12] Fbi = ω/Im(Z₁/Z₃) = A₀+ω²B₀ → Tbi = √(B₀/A₀). |
| `_h_Tbe(Z1, omega, omega2, Tbi, n_fit)` | 76 | [Eq. 19–20] F1 fit → Tbe = √(B/A). |
| `_h_ratios(Z1, Z3, omega, Tbi)` | 93 | [Eq. 13–14] Rbi/Rbc and Rbi·Cbc per-freq arrays. |
| `_h_R_RT(Z1, omega, Tbi, Tbe)` | 102 | [Eq. 23–24] R, R·T arrays from F2. |
| `_h_solve_pf(ror_arr, rbc_arr, R_arr, RT_arr, Tbi, Tbe)` | 111 | [Eq. 25] per-frequency 2×2 solve for Rbe, Rbi. |
| `_h_solve_sc(ror, rbc, R, RT, Tbi, Tbe, Z1, Z3, n_low)` | 129 | [Eq. 25] scalar solve with heuristic fallback. |
| `_h_derived_pf(Rbe_a, Rbi_a, ror_a, rbc_a, Tbi, Tbe)` | 144 | Per-freq Rbc, Cbc, Cbe, Cbi arrays from Rbe/Rbi arrays. |
| `_h_derived_sc(Rbi, Rbe, ror, rbc, Tbi, Tbe)` | 154 | Scalar Rbc, Cbc, Cbe, Cbi from Rbe/Rbi scalars. |
| `_h_Rcx(Z4, Rbc, Cbc, omega)` | 163 | [Eq. 26] 1/Rcx = Re(1/Z4 − 1/Z2), Z2 = Rbc/(1+jωRbcCbc). |
| `_h_Gm0_tau(Y_ex1, freq)` | 174 | Gm0 = Re(Y22)|f→0, τ = −(1/2π)·d∠Y21/df. |
| `_h_Ccx(Y_ex1, omega, n_hi)` | 184 | Ccx ≈ −Im(Y12)/ω at high frequency. |
| `_extract(Y_ex1, freq, n_low, n_fit=None, n_fit_f1=None)` | 195 | Run all helper steps end-to-end → `(params, arrays)`. |
| `_simulate(p, freq, z0=50.0)` | 250 | 5-layer inside-out scalar forward sim. |
| `_simulate_vec(p, freq, z0=50.0, xp=None)` | 293 | Vectorised forward sim. |
| `_b1(p, key, default, xp)` / `_detect_B(p, xp)` | 355 / 364 | Batched-tuning utilities. |
| `_simulate_batch(p, freq, z0=50.0, xp=None, cache=None)` | 378 | Batched forward sim for tuning sweeps. |

`class Degachi(AbstractSSMModel)` (line 550, `SHORT="D"`): cascade re-extraction across 9 PARAM_GROUPS (Step 1 / Fbi fit / Tbi / Part A / F1 fit + Tbe / Part C / Part D / Part E / Part F). Methods at lines 692, 698, 702, 709, 718, 830, 851, 856.

---

## Migration map (history — where helpers used to live)

| Helper | Was in | Now in |
|---|---|---|
| `s_to_y`, `y_to_s_*`, `_inv2`, `inv2x2`, `mm2x2`, smith grid, etc. | `ssm_core.py` | `helpers/rf_math.py` |
| `parse_s2p_bytes`, `write_s2p`, `simulate_open/short`, `build_Y_pad/Z_ser` | `ssm_s2p.py` | `helpers/s2p_io.py` |
| `parse_s2p` (str), `parse_csv`, `_load_cal` | `IOED_HBT_RF_extract.py` | `helpers/s2p_io.py` |
| `step_open`, `step_short`, `peel_parasitics`, `build_*_vec/_batch` | `ssm_deembedding.py` | `helpers/deembed_math.py` |
| `deembed_open_short`, `deembed_thru_half` | `IOED_HBT_RF_extract.py` | `helpers/deembed_math.py` |
| `_compute_h21_U`, `_find_ft_fmax`, `extrap_20dbdec` | `ssm_plots.py` | `helpers/metrics.py` (renamed without `_`) |
| `compute_metrics`, `extract_limit` | `IOED_HBT_RF_extract.py` | `helpers/metrics.py` |
| `make_smith`, `make_bode`, `make_plateau`, `_layout`, `_darken`, `PALETTE` | `IOED_HBT_RF_extract.py` | `helpers/plotly_plots.py` |
| `fig_to_excel_bytes`, `plotly_with_dl`, `_EXCEL_MIME` | `ssm_chart_utils.py` | `helpers/chart_export.py` |
| `build_excel`, `_card` | `IOED_HBT_RF_extract.py` | `helpers/chart_export.py` (renamed `_card` → `metric_card`) |

> **Status:** migration complete. `ssm_core.py`, `ssm_s2p.py`, and
> `ssm_chart_utils.py` have been deleted. `ssm_deembedding.py` and the
> math half of `ssm_plots.py` now live in `helpers/`. Every consumer
> imports through `tools.SSM.helpers`.

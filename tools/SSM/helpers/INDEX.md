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
| `_extract_ui(fname, key, freq, default_frac_lo, default_frac_hi)` | 56 | Frequency-range + method selector. Returns `(n0, n1, method, trim_pct)`. |
| `render_ssm_tab(fname, S_raw, freq, z0, open_data, short_data, all_data=None)` | 77 | Public entry — renders the complete SSM extraction tab for one DUT file (Steps 1–3, all model panels, summary, fit cache). |
| `_render_summary_table(fname, para_eff, cold_res, extract_results, registry)` | 677 | Multi-layer parameter summary table + CSV download. Pulls live (post fine-tune) values when available. |
| `_render_fit_cache_panel(fname)` | 793 | Fit-cache UI: list cached `(file, model)` entries for this DUT (with delete buttons), plus whole-cache export/import (sync local ↔ Streamlit Cloud). |

**Removed since the previous index:**
- `_agg` — was a dead duplicate of `helpers/deembed_math.py::_agg_arr`; never called within this module.
- `_render_s2p_downloads` — its functionality lives under each model's measured-vs-modeled Smith chart now (📥 modeled S2P button beside ⬇ xlsx, wired in `models/base_ui.py::render_smith_with_ftfmax`).

### [`ssm_override.py`](../ssm_override.py) — pre-extraction series-R picker

| Function | Line | Purpose |
|---|---|---|
| `render_unified_pre_override(fname, para_step1, cold_res, rz12_Re)` | 20 | Unified pad parameter UI. For Rb/Rc/Re it selects the highest source (Short Step 1b / Cold-HBT / Z-param method / Open-collector / Custom). Returns the effective `para_eff` dict (SI). |

### [`ssm_access_resistance.py`](../ssm_access_resistance.py) — series/access-R extractors

| Function | Line | Purpose |
|---|---|---|
| `render_rz12_section(all_data, para_eff, fname)` | 41 | Z-parameter method UI: Re(Z₁₂) vs 1/IE → Re. (Gao Ch. 5.5.1) Metrics row shows Re (intercept) and Rbe (current file). The Z₁₂-extraction-frequency input is a `selectbox` of the 10 lowest measured frequencies (default = lowest). η is intentionally NOT extracted here — slope-based ideality from Re(Z₁₂) is a derived quantity that often disagrees with the diode I–V; set η in the τ-total fit from a Gummel-plot measurement instead. |
| `render_open_collector_section(all_data, para_eff, fname)` | 194 | Open-collector method UI: Re(Z₁₁−Z₁₂)/Re(Z₂₂−Z₁₂)/Re(Z₁₂) vs 1/IB → Rb/Rc/Re. Writes `ocm_R*_{fname}` session keys. |
| `_render_cold_hbt(fname, open_data, para_step1, do_measured, freq, re_zparam=None, open_arr=None, short_arr=None)` | 321 | Cold-HBT extraction UI (Gao §5.5.2). Strips pad caps + lead L + Re from Z_cor before A/B/C/D. Returns a `cold_res` dict or `None`. **Each extracted variable (Cex, Cbc, Rbi, Cbe, Rb, Rc) is rendered inside its own bordered `st.container`** (inside `_cold_plot`) so individual parameters are visually distinct. |

### [`ssm_plots.py`](../ssm_plots.py) — Step 1 / Step 2 diagnostic Streamlit blocks

| Function | Line | Purpose |
|---|---|---|
| `render_open_plots(open_data, para_caps, open_arr, fname="")` | 33 | Five expanders for Open dummy: extra-element controls, capacitance plot, conductance plot, Im(Y)/ω vs 1/ω², measured vs modeled Smith. Returns `{cap: (mode, extra_SI)}`. |
| `render_short_plots(short_arr, para_short, fname="", freq=None)` | 199 | Lead-inductance plot for Short dummy (Lb/Lc/Le vs frequency, fixed 0–150 pH). |
| `_rlc_params_summary(p, fname)` | 257 | Build header param dict for a de-embedded `.s2p` file. |
| `_compare_bode_smith(*, S_a, S_b, freq, fname, key_suffix, label_a, label_b, color_a, color_b, smith_meas_label, smith_sim_label, gain_title, extra_download_fn=None)` | 278 | Side-by-side Bode (h21²+U) and Smith chart comparing two S-param datasets. **`color_a`/`color_b` kwargs are kept in the signature but ignored** — colours now come from `FT_FMAX_COLORS` (fT=blue, fmax=red); A vs B is distinguished by **dash** (A=solid, B=dashed). Returns `(fT, fmax)` of the right-hand trace. xlsx download uses the standardised `bode_excel_bytes` (simulated + extrapolated columns). **`extra_download_fn(fT_b, fmax_b)`** (optional) returns a `(label, data, file_name, mime)` tuple forwarded to the Smith chart's `plotly_with_dl` so a download (e.g. OS-deembedded / intrinsic S2P) sits **beside that chart's ⬇ xlsx button** — the OS-deembed and intrinsic preview sections use this for their S2P downloads. |
| `render_os_deemb_preview(S_raw, freq, z0, para_step1, fname)` | 374 | Step 2 — Raw vs Open/Short de-embedded preview + download. Returns `S_step1`. |
| `render_intrinsic_preview(S_raw, freq, z0, para_step1, para_eff, fname, S_step1=None)` | 446 | Step 3 footer — OS de-embedded vs Intrinsic preview + download. |
| `render_ft_fmax_card(S_mea, S_sim, freq, *, model_name, key, height=560, compact=False)` | 523 | **Per-model fT/fmax mini-card with standardised colour scheme + 2x2 legend + extrap method radio.** Colour: fT (|h21|²) = blue, fmax (Mason U) = red (`FT_FMAX_COLORS`). Dash: measured = solid+markers, modeled = dashed, extrap = dotted. Legend: **vertical stack pinned bottom-left inside the plot** (small font, semi-transparent bg — avoids the overlap a 2×2 grid caused in the narrow in-plot box). When any trace needs extrap, a Streamlit radio **underneath the chart** picks −20 dB/dec or single-pole (slider to the radio's right, default window = final 5 GHz). Values read from session_state before the figure is built so the widgets can sit below it. xlsx download uses the standardised `bode_excel_bytes` (simulated + extrapolated columns). |
| `_extrap_f0(f_ghz, gain_db, extrap_method, sp_window)` | ~734 | 0-dB crossing (GHz) via the selected method (single-pole window or −20 dB/dec). |
| `_eff_ft_fmax_ghz(f_ghz, h21_db, U_db, extrap_method="−20 dB/dec", sp_window=None)` | ~747 | Effective fT/fmax (GHz): in-band 0-dB crossing if present, else the *selected* extrap method (tracks the Bode radio). |
| `_read_extrap_selection(extrap_key, f_ghz)` | ~763 | Read the Bode card's extrap-method radio + single-pole window from session_state so τ/fmax use the same fT. |
| `_tau_total_ps(fT_ghz)` / `_calc_fmax_ghz(fT_ghz, CBC, Rbb)` | ~782 / ~790 | τ_total = 1/(2π fT) in ps; fmax = √(fT/(8π·CBC·Rbb)) in GHz (both None on invalid input). |
| `render_tau_fmax_expander(*, key, freq, S_meas, CBC, Rbb, S_model=None, tau_sum=None, tau_sum_label, tau_sum_tex, extrap_key=None)` | ~802 | Expander "Calculated τ_total and fmax". Left col: τ_total eqns, then `tau_sum` (τ_B+τ_C / τ) as latex, then a **table** (rows Measured/Modeled — single "Value" row when no model; columns fT, τ_total, τ_total − tau_sum). Right col: fmax eqn + **Extracted ⇄ Custom** radio for C_BC / R_bb (number inputs), then a **table** (rows Measured/Modeled; columns fT, fmax simulation = S-param 0-dB crossing, fmax calculation = √ formula). Cells via local `_num` formatter (n/a on None/non-finite). `S_model` given → measured **and** modeled (4 fmax values, SSM extraction); omitted → single dataset (RF simulator). `extrap_key` makes fT/fmax follow the Bode plot's extrap-method radio. Called by `models/base_ui.py::render_override_and_smith` (T/π/XuT only) and `RF_simulator.py` (all SSM models except Kun-Yang). |
| `_apply_text_autoformat(text, pattern, replacement)` | 671 | Regex substitution with `\1..\9` capture-group templating (used for Smith-label mathtext). |
| `_smith_grid_values(n)` | 700 | Return `(r_values, x_values)` for N evenly-spaced Smith-chart grid lines. |
| `_draw_mpl_smith_background(ax, line_lw, grid_lw, density)` | 712 | Draw constant-R / constant-X grid arcs for a unit Smith chart on a matplotlib axis. |
| `_mult_label_text(sp, mult)` | 1048 | On-chart label implied by an S-param multiplier: `1`→`"S21"`, `>1`→`"S21x3"`, `<1`→`"S21/5"` (reciprocal). `%g`-trimmed; bare name for non-positive/unparseable. |
| `_sync_text_to_mult(skey, sp)` | 1069 | `on_change` callback that rewrites an S-param's Text field from its multiplier via `_mult_label_text`. |
| `render_matplotlib_smith(S_mea=None, S_sim=None, fname="", topo_key="", *, sets=None, default_multiplier=1.0, phase="both", freq_hz=None)` | 1079 | Publication-style matplotlib Smith chart. Backward-compatible (mea/sim) and extensible (`sets=[…]`) call forms. `phase="controls"` renders widgets only; `phase="chart"` renders the figure only; `phase="both"` is the legacy combined form. SSM, RF simulator **and RF parameter extraction** call it twice (right column = controls, left column = chart) inside a side-by-side block, passing `freq_hz` for the auto frequency-range annotation. Editing a Multiplier auto-updates that S-param's Text label (`_sync_text_to_mult`). |
| `render_ft_fmax_overlay(S_raw, sim_results, freq, fname)` | 1433 | Multi-model Bode plot using `FT_FMAX_COLORS` — all fT traces blue, all fmax traces red. Models are distinguished by line name in the legend; modeled traces are dashed, extrap traces dotted. xlsx download uses the standardised `bode_excel_bytes` (simulated + extrapolated columns). |

---

## Helpers (`helpers/*.py`)

### [`helpers/_array_utils.py`](_array_utils.py) — Shared array helpers (NEW)

Canonical home for the small shape/broadcast utilities formerly duplicated across `helpers/deembed_math.py`, `models/_shared.py`, and `models/degachi.py`. Placed under `helpers/` (not `models/`) so the helpers package — which can't reach back into `models/` without a load-order cycle — can use them directly. `models/_shared.py` re-exports the names so `from ._shared import _b1, _detect_B, _stack22` keeps working everywhere it used to.

| Function | Line | Purpose |
|---|---|---|
| `_b1(p, key, default, xp, dtype=None)` | 29 | Fetch `p[key]` (or default), reshape `(B,)` → `(B,1)`; scalars stay scalar. Optional `dtype=` coerces for the fp32 sweep path. |
| `_detect_B(p, xp)` | 46 | Determine batch size B from any `(B,)`-shaped value in `p`. |
| `_stack22(a00, a01, a10, a11, xp)` | 66 | Stack four planes into a `(..., 2, 2)` tensor in 3 launches. |

### [`helpers/rf_math.py`](rf_math.py) — Pure RF math (no Streamlit, no plotting)

| Function | Line | Purpose |
|---|---|---|
| `s_to_y(S, z0=50.0)` | 14 | S → Y conversion, vectorised across all frequency points. |
| `_inv2(M)` | 36 | Per-row 2×2 analytic inverse (no per-matrix Python loop). |
| `y_to_z`, `z_to_y` | 50, 51 | Aliases for `_inv2`. |
| `y_to_s_single(Y, z0=50.0)` | 54 | Y → S for a single 2×2 matrix. |
| `y_to_s_batch(Y, z0=50.0)` | 60 | Y → S over a batch (delegates to `y_to_s_vec`). |
| `y_to_s_vec(Y, z0=50.0, xp=None)` | 65 | GPU-aware vectorised Y → S using hand-inlined 2×2 algebra (numpy or cupy). |
| `inv2x2(M, xp=None)` | 108 | Analytic 2×2 batched inverse — bypasses cuSOLVER. |
| `mm2x2(A, B, xp=None)` | 135 | Analytic 2×2 batched matmul — bypasses cuBLAS. |
| `safe_median(arr, n=None)` | 160 | Median of finite values, returns `0.0` if empty. |
| `strict_freq_check(f_dut, f_dummy, label)` | 167 | Raise `ValueError` if frequency grids differ. |
| `open_elem_Y(C, mode, extra, w)` | 175 | Pad capacitor admittance — None / Parallel L / Series L / Series R. |
| `short_lead_Z(R, L, Cpar, w)` | 195 | Series-lead impedance with optional parallel cap. |
| `_smith_grid_xy(max_r=1.0)` | 211 | Cached coordinate arrays for the Smith-chart background. |
| `extended_smith_grid(max_r=1.0)` | 252 | Plotly Smith chart background traces (built fresh each call from cached coords). |
| `params_hash(p)` | 269 | MD5 hash of a parameter dict, for caching. |

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
| `load_cal(fobj)` | 172 | Streamlit-aware S2P loader. |
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
| `_b1` | (re-export) | Imported from `helpers/_array_utils.py` (was a stale local copy that lacked the `dtype=` arg). |
| `build_Y_pad_batch(p, omega, B, N, xp)` | 127 | Pad admittance as 4 broadcast planes for sweep tuning. |
| `build_Z_ser_batch(p, omega, B, N, xp)` | 145 | Series-lead impedance as 4 broadcast planes for sweep tuning. |
| `step_open(open_data, n0, n1, method, trim_pct)` | 159 | Extract Cpbe/Cpce/Cpbc + diagnostic conductance arrays from Open dummy. |
| `step_short(short_data, freq, Cpbe, Cpce, Cpbc, ...)` | 203 | Extract Lb/Lc/Le, Rpb/Rpc/Rpe from Short dummy (Open subtracted first). |
| `peel_parasitics(S_raw, freq, z0, p)` | 278 | Remove pad+lead parasitics → Y_intrinsic (cached). |
| `deembed_open_short(Y_dut, Y_open, Y_short)` | 301 | Standard open-short de-embedding (Gao §4.2). |
| `deembed_thru_half(Y_dut, Y_thru_deemb)` | 306 | THRU/2 half-impedance subtraction. |

### [`helpers/metrics.py`](metrics.py) — Gain figures of merit

| Function | Line | Purpose |
|---|---|---|
| `compute_h21_U(S)` | 20 | Compute |h21|² (dB) and Mason U (dB) from S-parameters (z0 = 50 Ω). |
| `find_ft_fmax(f_ghz, h21_db, U_db)` | 37 | Vectorised search for the first 0 dB crossing (boolean masks + `np.flatnonzero`) → `(fT, fmax)`. The earlier Python `for i in range(len(arr))` loop ate ~30 ms × N_DUTs in the SSM tab; the vectorised version is ~50× faster. |
| `extrap_20dbdec(f_ghz, gain_db, n_pts=60, f_max_target=None)` | 62 | Slope-locked −20 dB/dec extrapolation → `(f_ext, g_ext, f_zero)`. |
| `single_pole_extrap(f_ghz, gain_db, idx_lo, idx_hi, n_pts=60, f_max_target=None)` | 97 | Log-linear (single-pole) fit over a user index window → `(f_ext, g_ext, f_zero, slope, intercept)`. Slope may differ from −20 — that disagreement is itself diagnostic. |
| `compute_metrics(Y, freq_hz)` | 164 | Build DataFrame: Freq, |h21|², Mason U, MAG/MSG, K, fT/fmax plateaus. |
| `extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max)` | 195 | Genuine-crossing fT/fmax extractor with extrapolation fallback (≥10 consecutive points above 0 dB). Ported to Rust as `_extract_limit_rust` inside `parse_and_compute_batch` — see `helpers/rust_kernels.py`. |

### [`helpers/plotly_plots.py`](plotly_plots.py) — Plotly Smith / Bode / Plateau builders

| Function / Constant | Line | Purpose |
|---|---|---|
| `PALETTE` | 27 | 10-color hex palette for trace cycling. |
| `FT_FMAX_SYMBOLS` | 39 | Standardised marker symbols for fT / fmax traces: `{"h21": "circle", "U": "square", "MAG": "diamond"}`. |
| `FT_FMAX_COLORS` | 59 | **Standard colour map** — `{"fT": "#1f77b4", "fmax": "#d62728"}` (blue / red). Applied everywhere a Bode plot draws fT/fmax measurements so users can identify a metric by colour alone. Modeled traces use the same colour with `dash="dash"`; extrap uses `dash="dot"`. |
| `thinned_indices(n, max_markers=25)` | 65 | Log-spaced sample of indices into a length-`n` array. Lets overlay plots cut WebGL marker primitives ~40× while keeping the line trace at full resolution. |
| `add_overlay_trace_with_markers(fig, x, y, *, name, color, symbol, ...)` | 81 | Add a Bode trace as (full-resolution line, hidden from legend) + (thinned marker overlay, owns the legend entry). |
| `darken(c)` | 118 | Darken a `#rrggbb` color by 45 units per channel. |
| `bode_layout(title, ytitle, yr, xr)` | 128 | Plotly layout dict for Bode/plateau plots (log-x). Legend font is 18 pt. |
| `make_smith(S, f_array, f_min, f_max, toggles, scales, title, max_r=1.0)` | 154 | Plotly Smith chart with up to 4 selectable S-param traces. |
| `make_bode(df, title, xr, yr, sh21, su, smag, color, *, show_20db=True, show_sp=False, sp_window_idx=None, extrap_f_max=None, return_extrap_df=False, return_excel_bytes=False)` | 207 | Plotly Bode plot with optional −20 dB/dec and single-pole-fit extrapolations. **Uses `FT_FMAX_COLORS`** — `color` arg kept for back-compat but ignored. `return_excel_bytes=True` → `(fig, xlsx)` via `bode_excel_bytes` (standardised sim + extrap export). |
| `_build_bode_extrap_df(df, extrap_curves, sh21, su)` | 345 | Combine measured + extrapolated values into one stitched DataFrame (legacy `return_extrap_df` path; superseded by `bode_excel_bytes`). |
| `make_plateau(df, res, title, xr, sh21, su, smag, color)` | 370 | Plotly GBP plateau plot (fT, fmax(U), fmax(MAG)). Uses `FT_FMAX_COLORS` + `FT_FMAX_SYMBOLS`. |
| `make_smith_bode_slider_fig(...)` | 415 | Smith chart + fT/fmax bode side-by-side with multi-slider semantics. |
| `_decimate_freq(S_batch, freq, max_points=200)` | 621 | Stride-decimate the frequency axis to `≤ max_points`. |
| `_compact_json_1d(arr, digits=5)` | 635 | Format a 1-D float array as compact JSON (~3× smaller). |
| `make_smith_bode_joint_slider_html(...)` | 647 | HTML-slider variant — embeds full joint sweep as JS array, injects `<input type="range">` per axis. |

### [`helpers/widgets.py`](widgets.py) — Streamlit input widgets

| Function | Line | Purpose |
|---|---|---|
| `info_icon_html(text, label="ⓘ")` | 28 | Hover-tooltip span snippet. |
| `apply_pending(target_key)` | 43 | Promote a pending quickset value into the widget's session-state key. Call BEFORE the paired `number_input`. |
| `_candidates(arr_disp, default_disp, cold_disp=None)` | 52 | Build `[(label, value), …]` list from optional sources. |
| `quickset_buttons(...)` | 69 | Row of one-click buttons (mean / median / low f / high f / default / cold). |

### [`helpers/fit_cache.py`](fit_cache.py) — Persistent per-(DUT, model) fitted-value cache

| Function | Line | Purpose |
|---|---|---|
| `_detect_cache_disabled()` | 79 | Honor `HBT_DISABLE_FIT_CACHE=1` env var. |
| `is_cache_disabled()` | 97 | Public predicate — cached during the session. |
| `_resolve_cache_dir()` | 107 | Pick the cache root: `$HBT_FIT_CACHE_DIR` ?: `~/.hbt-tools/`. |
| `_fits_dir()` | 124 | The `fits/` subdir under the cache root. |
| `cache_path_str()` | 130 | Return the cache **root** directory string. |
| `_basename_no_ext(fname)` | 135 | Strip directory + trailing `.s2p`. |
| `_norm_key(fname)` | 144 | Canonicalise to the extension-stripped basename. |
| `_dut_dir(fname, create=False)` | 150 | Resolve the per-DUT folder. |
| `_model_file(fname, model_short)` | 157 | Resolve the per-model JSON file path. |
| `_read_model_file(path)` | 164 | Tolerant JSON load → `{saved_at, params}` or `None`. |
| `_atomic_write_json(path, payload)` | 177 | Temp-file + rename so a crashed write never leaves a corrupt cache. |
| `_migrate_legacy_if_present()` | 190 | One-shot migrator: split a v1 monolithic `fit_cache.json` into per-(DUT, model) files. |
| `load_cache()` | 244 | Walk `fits/` and aggregate into `{basename: {model_short: {saved_at, params}}}`. |
| `get_fit(fname, model_short)` | 282 | Return cached SI-unit param dict, or `None`. |
| `get_fit_timestamp(fname, model_short)` | 294 | Return ISO timestamp string, or `None`. |
| `list_fits(fname)` | 302 | Return `{model_short: timestamp}` for the given s2p basename. |
| `_all_numeric_zero(params)` | 324 | True if every numeric value in `params` is exactly `0.0`. |
| `save_fit(fname, model_short, params_si)` | 340 | Persist a fine-tuned param dict. Refuses to save when every numeric value is `0.0`. |
| `delete_fit(fname, model_short=None)` | 377 | Drop one model's file (or the whole DUT folder). |
| `export_cache_bytes()` | 413 | Serialize all per-model files into one unified JSON. |
| `import_cache_bytes(raw, merge=True)` | 419 | Parse a unified JSON blob and write back as per-model files. |
| `differs_from(p_si, ref_si, keys=None, ...)` | 472 | True if any numeric value in `p_si` deviates from `ref_si`. |

### [`helpers/rust_kernels.py`](rust_kernels.py) — Optional Rust SIMD/Rayon acceleration

| Symbol | Line | Purpose |
|---|---|---|
| `HAS_RUST` (bool) | (module) | `True` iff the `hbt_rust_kernels` Rust extension is built and loaded. |
| `_arch_tag()` | 49 | Map platform → bin subfolder name (e.g. `win_amd64`). |
| `rust_diagnostic()` | 149 | Snapshot of the loading state (HAS_RUST, binary path, import error if any). |
| `_np_inv2x2_batch(Y)` | 184 | NumPy reference for the (B,2,2) analytic inverse. |
| `_np_mm2x2_batch(A, B)` | 199 | NumPy reference for the (B,2,2)×(B,2,2) matmul. |
| `_np_y_to_s_batch(Y, z0=50.0)` | 212 | NumPy reference for batched Y → S. |
| `_np_y_to_s_4d(Y, z0=50.0)` | 221 | NumPy reference for 4-D (B,N,2,2) Y → S. |
| `_np_port_residuals_batch(S_mea, S_mod_batch)` | 225 | NumPy reference for the (B,5) `[Total, S11, S12, S21, S22]` residual table. |
| `_c128(arr)` | 248 | Coerce to complex128 + C-contiguous. |
| `inv2x2_batch(Y)` | 255 | Public wrapper — Rust if available, NumPy fallback. |
| `mm2x2_batch(A, B)` | 263 | Public wrapper — Rust if available, NumPy fallback. |
| `y_to_s_batch(Y, z0=50.0)` | 270 | Public wrapper — Rust if available, NumPy fallback. |
| `y_to_s_4d(Y, z0=50.0)` | 277 | Public wrapper — Rust if available, NumPy fallback. Primary entry from the visual-tuning sweep loop. |
| `port_residuals_batch(S_mea, S_mod_batch)` | 286 | Public wrapper — Rust if available, NumPy fallback. |
| `_phase2_dispatch_enabled()` | 323 | Env-gate for the Phase 2 sim_*_batch kernels. |
| `_phase2_parity_check_enabled()` | 348 | Env-gate for the optional parity check on every Phase 2 call. |
| `_normalize_params_for_rust(params)` | 361 | Broadcast + flatten multi-dim sweep tensors to 1-D float64. |
| `_phase2_dispatch(rust_fn_name, params, freq, z0, np_fallback)` | 420 | Common dispatch helper for all topology wrappers. |
| `sim_cheng_t_batch(params, freq, z0=50.0, *, np_fallback)` | 478 | Cheng T-topology end-to-end batched simulation. |
| `sim_cheng_pi_batch(params, freq, z0=50.0, *, np_fallback)` | 507 | Cheng π-topology end-to-end batched simulation. |
| `sim_xu_t_batch(params, freq, z0=50.0, *, np_fallback)` | 516 | Xu T-topology end-to-end batched simulation (parallel `Rbcx ∥ Cbcx`). |
| `sim_kunyang_batch(params, freq, z0=50.0, *, np_fallback)` | 526 | Kun-Yang HEMT pi-topology end-to-end batched simulation. |
| `SIM_FOR_TOPOLOGY` | 543 | `{topology_short → wrapper}` lookup consumed by `SSMModelTemplate.simulate_batch`. |
| `_EXTRACT_METHOD_LABELS` | (module) | u8 method code → string label ("No Data" / "No Gain" / "0dB Cross" / "Extrap & Plat."). |
| `_np_parse_and_compute_batch(files_bytes, n_pts=2, f_min=0.01, f_max=50.0)` | 579 | NumPy reference for the bulk-upload accelerator (parse_s2p + s_to_y + compute_metrics + extract_limit). Byte-identical fallback. |
| `_expand_soa_to_per_file(soa)` | 633 | Convert the SoA dict returned by the Rust kernel into per-file dict list with **zero-copy NumPy views**. ~300 µs vs ~62 ms PyArray-alloc overhead in the earlier list-of-dicts shape. |
| `parse_and_compute_batch(files_bytes, n_pts=2, f_min=0.01, f_max=50.0)` | 690 | **Public bulk-upload accelerator.** Parses every `.s2p`, computes metric columns, runs `extract_limit ×3` per file — all in one Rayon-parallel section. Used by the IOED bulk-upload loop. |

Re-exported from `helpers/__init__.py` with a `rust_` prefix (e.g. `rust_inv2x2_batch`, `rust_parse_and_compute_batch`) and `RUST_KERNELS_AVAILABLE` for `HAS_RUST`.

Build instructions live in [`rust_kernels/README.md`](../rust_kernels/README.md). The Linux x86_64 binary is built by `.github/workflows/build-rust-linux.yml` and committed back to `bin/linux_x86_64/`. The CI smoke test exercises both `y_to_s_4d` and `parse_and_compute_batch` (verifies the SoA dict has all 20 expected keys).

---

**fit_cache.py storage layout (v2)**:

```
~/.hbt-tools/                              # or $HBT_FIT_CACHE_DIR
├── fit_cache.legacy.json                  # one-shot rename of the v1 file (if any)
└── fits/
    └── <basename_no_ext>/                 # one folder per DUT
        ├── <basename_no_ext>_T.json
        ├── <basename_no_ext>_pi.json
        └── <basename_no_ext>_XuT.json
```

Each per-model file: `{"saved_at": "<ISO>", "params": {<SI-unit dict>}}`.

### [`helpers/chart_export.py`](chart_export.py) — Excel export & Streamlit UI helpers

| Function | Line | Purpose |
|---|---|---|
| `EXCEL_MIME` | 19 | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`. |
| `_axis_text(axis_obj)` | 26 | Safely fetch a Plotly axis title text. |
| `_is_smith(fig)` | 33 | True when the figure is a Smith chart (Re(Γ) / Im(Γ) axes). |
| `_smith_col_name(trace_name)` | 38 | Map a Smith-trace name → compact Excel column prefix (e.g. `S11_meas`). |
| `_freq_sheet_name(x_lbl, x_arr)` | 61 | Build an Excel sheet name from the frequency range. |
| `_collect_traces(fig)` | 108 | Return `[(name, x, y), …]` for exportable traces. |
| `fig_to_excel_bytes(fig)` | 141 | Extract Plotly traces → `.xlsx` bytes (smart Smith vs normal layout). |
| `bode_excel_bytes(freq_ghz, sim_traces, extrap_traces=None)` | 221 | Canonical fT/fmax Bode export → single-sheet `.xlsx` bytes: simulated block (`Freq (GHz)` + each gain trace) plus, when extrapolation is present, a side-by-side `… (extrap)` block on a unified extrap freq axis. Used by every Bode plot (RF simulator, SSM, RF extraction) via `plotly_with_dl(excel_bytes=…)` / `make_bode(return_excel_bytes=True)`. |
| `plotly_with_dl(fig, key, filename="", width="stretch", container=None, extra_download=None, excel_bytes=None, **kwargs)` | 299 | Render Plotly chart + compact xlsx download button below it. **`extra_download=(label, data_bytes, file_name, mime)`** adds a second button alongside xlsx (SSM measured-vs-modeled Smith → "📥 modeled S2P"). **`excel_bytes`** supplies pre-built xlsx (e.g. `bode_excel_bytes`) instead of auto-extracting from the figure. |
| `build_excel(summary_df, all_data)` | 380 | Multi-sheet workbook: Summary + per-DUT DataFrames. |
| `metric_card(col, title, val, sub, color="#4A90D9")` | 404 | Styled HTML metric tile (used in IOED tab_ind, batch tab). |

---

## Models (`models/*.py`)

### [`models/__init__.py`](../models/__init__.py) — Registry + abstract base

| Class / function | Line | Purpose |
|---|---|---|
| `class AbstractSSMModel(ABC)` | 18 | Public API every SSM model must implement. Class attrs `NAME`, `SHORT`, `TOPOLOGY_CHAR`. |
| `AbstractSSMModel.extract(Y_ex1, freq, n_low, **kwargs)` | 37 | Extract intrinsic params from the de-embedded admittance. Returns `(params, arrays)`. |
| `AbstractSSMModel.simulate(params, freq, z0=50.0)` | 59 | Forward-simulate S-parameters from a fully populated params dict. |
| `AbstractSSMModel.render_step_formulas(cls)` | 72 | Render extraction-step LaTeX in Streamlit. |
| `AbstractSSMModel.render_results_table(cls, params)` | 81 | Render extracted scalar parameters as a Streamlit dataframe. |
| `AbstractSSMModel.render_formula_trace(cls)` | 86 | Render a collapsible expander with the full extraction + simulation chain. |
| `AbstractSSMModel.render_override_and_smith(cls, fname, S_raw, freq, z0, para_eff, extract_result, **kwargs)` | 94 | Render fine-tune override UI + Smith chart. Returns `S_sim` or `None`. |
| `AbstractSSMModel.get_s2p_header_params(cls, params, para_eff)` | 118 | Default human-readable param dict for `.s2p` header. |
| `REGISTRY` | 140 | `{SHORT: ModelClass}` — drives the model UI loop in `main_ssm_extraction.py`. |
| `DEFAULT_SELECTION` | 149 | List of model SHORTs shown on first run. |

### [`models/_shared.py`](../models/_shared.py) — Font helpers (`_b1`/`_detect_B`/`_stack22` re-exports)

After the array-utility consolidation, this module is mostly font code. The three array helpers (`_b1`, `_detect_B`, `_stack22`) are now re-exported from `helpers/_array_utils.py` so existing call sites (`from ._shared import _b1, ...`) keep working unchanged.

| Function | Line | Purpose |
|---|---|---|
| `_b1, _detect_B, _stack22` | (re-export) | See `helpers/_array_utils.py`. |
| `_try_download_inter()` | 35 | One-time fetch of Inter font into `models/fonts/`. |
| `has_inter()` | 58 | True if Inter is installed locally or successfully cached. |
| `_load_font(size)` | 68 | Load a TrueType font with Inter → Arial → fallback chain. |

### [`models/base_ui.py`](../models/base_ui.py) — Shared Streamlit UI helpers for all models

| Function | Line | Purpose |
|---|---|---|
| `PAD_SPECS` | 63 | Tuple list `(key, label, SI_scale, unit, fmt, step)` for the nine pad parameters. |
| `ssm_residual(S_mea, S_mod)` | 79 | RMS relative S-parameter residual across all four ports (%). |
| `_port_residuals(S_mea, S_mod)` | 93 | Per-port residual dict (`Total`, S11, S12, S21, S22) in %. |
| `_port_residuals_batch(S_mea, S_mod_batch, xp)` | 110 | Batched fused per-port residuals. |
| `render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith", show_title=True, meas_label="Meas.", sim_label="Model", *, compact=False, height=None, extra_download=None)` | 140 | Per-model Plotly Smith with measured markers + dashed model traces. **`extra_download`** is forwarded to `plotly_with_dl` so the download row can host a second button (📥 modeled S2P beside ⬇ xlsx). |
| `render_smith_with_ftfmax(S_raw, S_sim, freq, model_name, model_short, fname, scales=None, *, s2p_bytes=None, s2p_filename=None)` | 210 | 2-column: Smith chart (left) + fT/fmax mini-card (right). When `s2p_bytes` is supplied, the Smith chart's download row gains a `📥 modeled S2P` button beside the xlsx — this replaced the standalone "Download Modeled DUT S2P" section at the bottom of the SSM tab. |
| `smith_scale_controls(fname, topo_key)` | 252 | Four `number_input` fields for per-trace Smith display scaling. |
| `sync_pad_from_preov(fname, topo_key, para_eff)` | 275 | Copy pre-extraction pad values into per-topology session state when the upstream MD5 hash changes. |
| `_render_cbex_sweep_tool(...)` | 288 | Cbex sweep UI for the Cbcx stability search. |
| `_render_tau_total_fit_section(*, all_data, fname, model_short, params, para_eff)` | 426 | Multi-file 1/(2πfT) vs 1/IC reference fit (T-models only — ChengT, XuT). Computes fT per bias file from Open+Short pad/lead de-embedded \|h21\|² 0-dB crossing — but **access R is RETAINED** (Rpb/Rpc/Rpe forced to 0 in the dict passed to `peel_parasitics`) so RC/REE in the Cheng formula remain meaningful. Auto-no-op for pre-de-embedded files (when caps/Ls are zero, `peel_parasitics` returns Y_dut unchanged). Plots τ_total (ps) vs 1/IC (1/mA); linear-fits → reference Cje (=slope/(η·Vt)) and τB+τC (=intercept−(Rc+Re)·Cbc). Per-file τCC=(rE+Re+Rc)·Cbc and τE=rE·Cje. Also splits τB / τC via assumed average collector velocity v_c (Liu et al., IEEE EDL 25(12), 2004) — `W_C` (nm, default 120) and `v_c` (cm/s, default 4×10⁷ for 2000 Å InP collectors) are inputs → τ_C = W_C/(2 v_c), τ_B = (τ_B+τ_C) − τ_C. Publishes the v_c-derived τB / τC into session state (`taut_pub_tauB/τC_{short}_{fname}`) so the downstream τB / τC `number_input`s render a "v_c = …" quickset button. Hidden when `< 2` s2p files loaded. Reference-only — does NOT feed back into the model fit. **Returns `True` when it rendered, `False` when skipped (single file)** so the caller can suppress the trailing `---` separator and avoid a double rule with one file. References: Cheng et al. (equation) + Liu et al. (v_c default for InP collector). |
| `render_interactive_param_groups(...)` | (after _render_tau_total_fit_section) | The big interactive expander: each normal parameter group (Cbex, Cbcx, intrinsic, τB, τC, …) is wrapped in one large card keyed `pfp_groupbox_{model_short}_{g_idx}_{fname}` (`box = st.container(border=False, key=…)`; its title + slider + plots render into `box` / `box.columns(...)`, with the existing per-parameter bordered sub-containers nested inside). The thick outline is drawn by a scoped `<style>` block (injected once at the top of the expander) targeting `div[class*="st-key-pfp_groupbox_"]` directly — Streamlit puts the `key` class on the inner `stVerticalBlock` (not the border wrapper), so the CSS styles that element and `border=False` avoids a double frame. Keep the key prefix and that CSS selector in sync. Special fit groups (`z_plots_group` / `fbi_fit_group` / `f1_fit_group`) keep the legacy heading-outside layout. Per-group: section heading, dependency info, "same range as previous" button, frequency-range slider, per-frequency line plots, per-param `number_input`, optional Cbex-sweep tool, optional `tau_total_fit_group` (multi-file reference fit; its trailing `---` is suppressed when the fit is hidden for a single file). Accepts `all_data` and `para_eff` kwargs to enable the tau-total fit section. Per-param inputs render extra quickset buttons sourced from elsewhere in the UI: **"v_c = …"** for τB / τC (from the tau_total_fit_group's v_c split), and **"Z-param = …"** for Rbe (from `rz12_Rbe_{fname}` when the Z-parameter method has been run and the current DUT is in the fit). |
| `_FRAGMENT` (module-level) | ~984 | `st.fragment` (≥ 1.37) / `st.experimental_fragment` (1.33–1.36) / identity fallback. |
| `_make_sweep_values(min_val, max_val, step)` | 991 | Generate sweep values. |
| `_render_slider_preview(model_cls, all_p, S_raw, freq, z0, tuning_specs, fname, topo_key)` | 1006 | Dispatcher inside the Visual Tuning expander. |
| `_slider_default_range(current_disp)` | 1047 | Sane `(min, max, step)` for a slider. |
| `_multi_metric_top_n(arr, per_metric=10)` | 1058 | Take an `(N, n_cols)` residual table and return the union of top-`per_metric` rows by each metric. |
| `_render_live_slider_preview(...)` *@fragment* | 1087 | Streamlit-rerun-per-drag preview with pre-baked static cache + `simulate_batch(B=1)`. |
| `_quantize_S_batch_int16(S_batch)` | 1327 | Per-element int16 quantization. 4× memory savings vs complex128. |
| `_dequantize_S_frame(quant, joint)` | 1353 | Inverse of `_quantize_S_batch_int16`. |
| `_chunked_simulate_batch_to_host(...)` | 1363 | Run `simulate_batch` in slabs so OOM doesn't bite on million-frame sweeps. |
| `_render_plotly_server_cached_view(...)` *@fragment* | 1405 | Server-cached rendering. |
| `_render_plotly_slider_preview(...)` *@fragment* | 1540 | Pre-computed Plotly slider — joint cartesian sweep. |
| `render_visual_tuning_expander(...)` | 1875 | 🎚️ Visual Tuning expander — wraps `_render_slider_preview`. |
| `render_tuning_expander(...)` | 1944 | 🔧 Auto Tuning for Minimum Residuals — grid sweep + per-metric top-10 ranking + residual table. Toolbar above the table (under "Compute backend" line) has `Use default values` + `Select all` + `De-select all` buttons. |
| `class SSMModelTemplate` | 4318 | Mixin parent for SSM model classes. |
| `SSMModelTemplate.prebake_static_keys(cls, swept_keys)` | 4347 | Truth-table query: given which params are sweeping, return the list of pre-bakeable sub-network names. |
| `SSMModelTemplate.build_static_cache(cls, all_p, freq, *, xp=None, swept_keys=())` | 4360 | Build the `cache` dict matching `_sim_wrap_batch`'s lookup keys. |
| `SSMModelTemplate._build_intrinsic_static_cache(...)` | 4396 | Default no-op — concrete classes override. |
| `SSMModelTemplate.simulate_vec(...)` | 4405 | Default vectorised forward sim. |
| `SSMModelTemplate.simulate_batch(...)` | 4412 | Default batched forward sim. |
| `SSMModelTemplate._cached_simulate_vec(...)` | 4478 | Cached scalar→vector simulation with `params_hash` keying. |
| `SSMModelTemplate.render_results_table(cls, params)` | 4513 | Default — concrete classes override. |
| `SSMModelTemplate.render_formula_trace / _render_results_trace / has_formula_trace` | 4526 / 4536 / 4542 | Template hooks for the formula-trace expander. |
| `SSMModelTemplate.render_override_and_smith(cls, fname, S_raw, freq, z0, para_eff, extract_result)` | 4553 | Default — extracts param dict + simulates + writes a modeled S2P bytes blob, then renders **two top-level expanders** below the measured-vs-modeled Smith chart: "🖼️ Topology illustration" (the schematic with live param values) and **"🍩 Smith Chart (Matplotlib)"** which contains the controls (right column) AND the chart (left column) side-by-side — matches the RF simulator layout. The measured-vs-modeled Smith chart now exposes a `📥 modeled S2P` download next to its xlsx button (s2p bytes built from `S_sim + freq + para_eff` using the same template the deleted `_render_s2p_downloads` used). |

**Pre-bake truth table API** — `SSMModelTemplate` exposes `STATIC_SUBNETWORKS` (declarative dependency map, set per concrete model) and uses it to skip re-computing static sub-networks across slider ticks. Per-model `_build_intrinsic_static_cache` overrides in `cheng.py:1149` (ChengT), `cheng.py:1400` (ChengPi), `xu.py:880` (XuT), and `kunyang.py:686` (KunYangHEMT).

### [`models/cheng.py`](../models/cheng.py) — Cheng (2022) T and π topologies

Module-level helpers:

| Function | Line | Purpose |
|---|---|---|
| `_sweep_cbex_stds_cheng(Y_ex1, freq, cbex_SI_array, mask)` | 35 | For each candidate Cbex, rebuild Y_ex2 and return std(Cbcx_arr[mask]). Shared by ChengT and ChengPi. |
| `_step2_T(Y_ex1, freq, n_low)` | 74 | Cheng [Eq. 13, 22] — Cbex_T (T variant) and Cbcx. |
| `_step2_pi(Y_ex1, freq, n_low)` | 110 | Cheng [Eqs. 26–28] — Cbex_π and Cbcx. |
| `_step3_T(Y_ex2, freq, Cbcx, n_low)` | 149 | Cheng [Eqs. 16, 29–31] — T-topology intrinsic Rbi/Rbe/Cbe/Rbc/Cbc/α₀/τB/τC. |
| `_step3_pi(Y_ex2, freq, Cbcx, n_low)` | 209 | Zhang et al. — π-topology intrinsic Rbi/Rbe/Cbe/Cbc/Gm0/τ. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | 259 | Add extrinsic caps + pad/lead parasitics around an intrinsic-Y matrix (per-freq loop, scalar). |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | 280 | Vectorised forward sim. Works on numpy or cupy. |
| `_Y_int_T_vec(p, omega, xp)` | 319 | Vectorised T-topology intrinsic-Y matrix → (N,2,2). |
| `_Y_int_Pi_vec(p, omega, xp)` | 334 | Vectorised π-topology intrinsic-Y matrix → (N,2,2). |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | 355 | Batched forward sim over (param_combo × freq). |
| `_Y_int_T_batch(p, omega, B, N, xp, cache=None)` | 469 | Batched T-topology intrinsic-Y, returns 4 broadcastable planes. |
| `_Y_int_Pi_batch(p, omega, B, N, xp, cache=None)` | 534 | Batched π-topology intrinsic-Y, returns 4 broadcastable planes. |
| `_fmt_param(key, val_si)` | 715 | Format a parameter SI value for display on the topology illustration. |
| `_render_topology_illustration(all_p, topology, fname)` | 733 | Overlay live parameter values on the schematic template PNG. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS)` | 823 | Render the fine-tune override expander (Pad / Extrinsic / Intrinsic) for one Cheng topology. |

Shared helpers `_b1`, `_detect_B`, `_stack22`, `_try_download_inter`, `has_inter`, `_load_font` are imported from [`models/_shared.py`](../models/_shared.py).

`class ChengT(SSMModelTemplate, AbstractSSMModel)` (line 875): methods at lines 978 (extract), 991 (sweep_cbex), 1001 (simulate), 1022 (reextract), 1098 (_results_rows), 1115 (_render_results_trace), 1140 (_do_override_ui), 1145 (_render_topology), 1149 (_build_intrinsic_static_cache).

`class ChengPi(SSMModelTemplate, AbstractSSMModel)` (line 1195): methods at lines 1275 (extract), 1284 (sweep_cbex), 1290 (simulate), 1322 (reextract), 1373 (_results_rows), 1391 (_do_override_ui), 1396 (_render_topology), 1400 (_build_intrinsic_static_cache).

### [`models/xu.py`](../models/xu.py) — Xu's T (2014)

Module-level helpers:

| Function | Line | Purpose |
|---|---|---|
| `_step2_T(Y_ex1, freq)` | 36 | Xu — Cbcx direct from Y_ex1 (no Cbex peel). Rbcx defaults to 285 kΩ. |
| `_step3_T(Y_ex1, freq, Cbcx, Rbcx, n_low)` | 59 | Xu — peel Ybcx then run the Cheng-style Step 3 intrinsic extraction. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | 127 | Scalar forward sim with Ybcx parallel network. |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | 143 | Vectorised forward sim. |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | 182 | Batched forward sim for tuning sweeps. |
| `_Y_int_T_vec(p, omega, xp)` | 292 | Vectorised T-topology intrinsic-Y (shared formula with Cheng-T). |
| `_Y_int_T_batch(p, omega, B, N, xp, cache=None)` | 307 | Batched intrinsic-Y planes. |
| `_aka(cheng_lbl, xu_lbl)` | 381 | Render combined "Cheng / Xu" symbol label. |
| `_fmt_param(key, val_si)` | 486 | Format a parameter SI value for display on the topology illustration. |
| `_render_topology_illustration(all_p, fname)` | 505 | Overlay live values on the Xu schematic template PNG. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS)` | 592 | Fine-tune override expander for the Xu T topology. |

`class XuModel(SSMModelTemplate, AbstractSSMModel)` (line 641, `SHORT="XuT"`): methods at lines 727 (extract), 741 (simulate), 762 (reextract), 830 (_results_rows), 847 (_render_results_trace), 871 (_do_override_ui), 876 (_render_topology), 880 (_build_intrinsic_static_cache).

### [`models/kunyang.py`](../models/kunyang.py) — Kun-Yang HEMT (pi-model, forward sim only)

Forward-simulation-only model — no extraction is performed. Built inside → out:

1. Intrinsic 3-component pi: series Cgs/Ri (gate-source shunt), series Cgd/Rgd (gate-drain), parallel Rds∥Cds + transconductance gm = Gm0·exp(−jωτ) (drain-source).
2. Z_ser wrap with gate/drain/source lead L+R.
3. Kun-Yang custom substrate pad in parallel: Cgsp series Rsub1, Cdsp series Rsub2, Cgdp (port-1 to port-2).
4. Standard open-dummy pad (Cpbe, Cpce, Cpbc) on top → S.

Module-level helpers:

| Function | Line | Purpose |
|---|---|---|
| `_render_topology_illustration(all_p, fname)` | 51 | Render Kun-Yang topology illustration (placeholder — no schematic shipped). |
| `_Y_int_KY_vec(p, omega, xp)` | 64 | Vectorised intrinsic pi-model Y matrix. |
| `_Y_int_KY_batch(p, omega, B, N, xp, cache=None)` | 98 | Batched intrinsic Y → 4 (B, N) planes; cache-aware. |
| `_Y_kypad_vec(p, omega, xp)` | 155 | Vectorised Kun-Yang substrate pad Y. |
| `_Y_kypad_batch(p, omega, B, N, xp, cache=None)` | 187 | Batched Kun-Yang substrate pad as 4 (B, N) planes. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | 219 | Per-frequency scalar forward sim. |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | 261 | Vectorised forward sim. |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | 295 | Batched (param_combo × freq) forward sim. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs)` | 432 | Fine-tune override UI. |

`class KunYangHEMT(SSMModelTemplate, AbstractSSMModel)` (line 517, `SHORT="KY"`): methods at 566 (extract), 575 (simulate), 590 (render_step_formulas), 624 (_results_rows), 648 (_render_results_trace), 676 (_do_override_ui), 682 (_render_topology), 686 (_build_intrinsic_static_cache), 737 (get_s2p_header_params).

### [`models/degachi.py`](../models/degachi.py) — Degachi & Ghannouchi (2008) augmented π *(currently disabled in the registry — uncomment in `models/__init__.py` to re-enable)*

| Function | Line | Purpose |
|---|---|---|
| `_h_Tbi / _h_Tbe / _h_ratios / _h_R_RT / _h_solve_pf / _h_solve_sc / _h_derived_pf / _h_derived_sc / _h_Rcx / _h_Gm0_tau / _h_Ccx` | 59 / 76 / 93 / 102 / 111 / 129 / 144 / 154 / 163 / 174 / 184 | Per-equation helpers following the 2008 IEEE TED paper. |
| `_extract(Y_ex1, freq, n_low, n_fit=None, n_fit_f1=None)` | 195 | Run all helper steps end-to-end → `(params, arrays)`. |
| `_simulate(p, freq, z0=50.0)` | 250 | 5-layer inside-out scalar forward sim. |
| `_simulate_vec(p, freq, z0=50.0, xp=None)` | 293 | Vectorised forward sim. |
| `_b1, _detect_B` | (re-export) | Imported from `models/_shared.py` (which re-exports from `helpers/_array_utils.py`) — formerly duplicated here. |
| `_simulate_batch(p, freq, z0=50.0, xp=None, cache=None)` | 378 | Batched forward sim for tuning sweeps. |

`class Degachi(AbstractSSMModel)` (line 550, `SHORT="D"`): cascade re-extraction across 9 PARAM_GROUPS.

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
| `_b1`, `_detect_B`, `_stack22` (was duplicated in 3 places) | `models/_shared.py`, `helpers/deembed_math.py`, `models/degachi.py` | `helpers/_array_utils.py` (canonical) — `_shared.py` re-exports for back-compat |
| `_render_s2p_downloads` (Download Modeled DUT S2P section) | `main_ssm_extraction.py` | **Deleted** — replaced by `📥 modeled S2P` button beside `⬇ xlsx` under each model's measured-vs-modeled Smith chart (`models/base_ui.py::render_smith_with_ftfmax`). |

> **Status:** migration complete. `ssm_core.py`, `ssm_s2p.py`, and
> `ssm_chart_utils.py` have been deleted. `ssm_deembedding.py` and the
> math half of `ssm_plots.py` now live in `helpers/`. Every consumer
> imports through `tools.SSM.helpers`.

# SSM — function index

Single source of truth for every function across `tools/SSM/`. Skim this
file to find which submodule owns a function before grep'ing the codebase.

All paths are relative to `tools/SSM/`. The index deliberately carries
**no line numbers** (they drift as files evolve) — grep for the function
name to jump to its definition.

Helpers (`helpers/*.py`) re-export from `tools.SSM.helpers`, so call
sites can simply do:

```python
from tools.SSM.helpers import parse_s2p, make_smith, compute_metrics
```

The submodule paths below are for jump-to-definition only.


---

## Top-level orchestration

### [`main_ssm_extraction.py`](../main_ssm_extraction.py) — UI entry point

| Function | Purpose |
|---|---|
| `BUILTIN_SHORTS` | `("T","pi")` — models shown in the built-in **analytic-extraction** flow (Cheng T/π only). Xu & Kun-Yang were moved out to the Custom-model section's forward-sim views. Drives the model-checkbox loop in `render_ssm_tab`. |
| `render_builtin_forward_sim(short, S_raw, freq, z0, fname, show_header=True, show_cache_banner=True)` | **Forward-simulation-only** view for a built-in model (`"XuT"` / `"KY"`), surfaced under the Custom-model radio. Seeds params from a one-shot guess on the device (pads=0, **no Open/Short de-embed**) then hands off to `ModelClass.render_override_and_smith` (override → simulate → Smith/residual/fT-fmax → grid-sweep tuning), forwarding `show_cache_banner`. `show_header=False` suppresses the "### 🧩 {NAME} — forward simulation" markdown; `RF_simulator.py`'s fit-mode call passes `False` for both since it already shows its own "### 🎯 Fit — {NAME}" header + compact file/cache/model pill row immediately above (avoids the model name rendering twice and the full cache banner duplicating the cache pill). |
| `_extract_ui(fname, key, freq, default_frac_lo, default_frac_hi)` | Frequency-range + method selector. Returns `(n0, n1, method, trim_pct)`. |
| `render_ssm_tab(fname, S_raw, freq, z0, open_data, short_data, all_data=None)` | Public entry — renders the complete SSM extraction tab for one DUT file (Steps 1–3, **Cheng T/π only**, summary, fit cache). |
| `_render_summary_table(fname, para_eff, cold_res, extract_results, registry)` | Multi-layer parameter summary table + CSV download. Pulls live (post fine-tune) values when available. |
| `_render_fit_cache_panel(fname)` | Fit-cache UI: list cached `(file, model)` entries for this DUT (with delete buttons), plus whole-cache export/import (sync local ↔ Streamlit Cloud). |

**Removed since the previous index:**
- `_agg` — was a dead duplicate of `helpers/deembed_math.py::_agg_arr`; never called within this module.
- `_render_s2p_downloads` — its functionality lives under each model's measured-vs-modeled Smith chart now (📥 modeled S2P button beside ⬇ xlsx, wired in `models/base_ui.py::render_smith_with_ftfmax`).

### [`handoff.py`](../handoff.py) — cross-page workflow bus (NEW)

Lets the three RF pages pass the **active DUT** between each other via
`st.session_state` (survives `st.switch_page`): 📡 RF At a Glance → 🔬 SSM
Extraction → 🛠️ SSM Simulation & Fitting, with no re-uploading.

| Symbol | Purpose |
|---|---|
| `PAGE_AT_A_GLANCE` / `PAGE_EXTRACTION` / `PAGE_SIMFIT` | `st.switch_page` target paths. |
| `TARGET_EXTRACTION` / `TARGET_SIMFIT` | Handoff target ids. |
| `send(target, *, S, freq, z0, label, stage="raw", params=None, model_short=None, extras=None)` | Stash a payload (`stage` ∈ raw/deembedded/intrinsic; `params`+`model_short` seed the Sim/Fit override fields; `extras={label:{S,freq,z0}}` carries the *other* batch-de-embedded bias files so Extraction's Z-param / Cold-HBT / τ_total methods have them). |
| `peek(target)` / `take(target)` | Read / one-shot consume the pending payload. |

### [`ssm_override.py`](../ssm_override.py) — pre-extraction series-R picker

| Function | Purpose |
|---|---|
| `render_unified_pre_override(fname, para_step1, cold_res, rz12_Re)` | Unified pad parameter UI. For Rb/Rc/Re it selects the highest source (Short Step 1b / Cold-HBT / Z-param method / Open-collector / Custom). Returns the effective `para_eff` dict (SI). |

### [`ssm_access_resistance.py`](../ssm_access_resistance.py) — series/access-R extractors

| Function | Purpose |
|---|---|
| `render_rz12_section(all_data, para_eff, fname)` | Z-parameter method UI: Re(Z₁₂) vs 1/IE → Re. (Gao Ch. 5.5.1) Metrics row shows Re (intercept) and Rbe (current file). The Z₁₂-extraction-frequency input is a `selectbox` of the 10 lowest measured frequencies (default = lowest). η is intentionally NOT extracted here — slope-based ideality from Re(Z₁₂) is a derived quantity that often disagrees with the diode I–V; set η in the τ-total fit from a Gummel-plot measurement instead. |
| `render_open_collector_section(all_data, para_eff, fname)` | Open-collector method UI: Re(Z₁₁−Z₁₂)/Re(Z₂₂−Z₁₂)/Re(Z₁₂) vs 1/IB → Rb/Rc/Re. Writes `ocm_R*_{fname}` session keys. |
| `_render_cold_hbt(fname, open_data, para_step1, do_measured, freq, re_zparam=None, open_arr=None, short_arr=None, all_data=None)` | Cold-HBT extraction UI (Gao §5.5.2). **`all_data`** lets the user pick the cold device from already-loaded bias files (segmented source toggle) instead of uploading — defaults to a loaded file with "cold" in its name; falls back to the uploader when no other files are loaded. Strips pad caps + lead L + Re from Z_cor before A/B/C/D. Returns a `cold_res` dict or `None`. **Each extracted variable (Cex, Cbc, Rbi, Cbe, Rb, Rc) is rendered inside its own bordered `st.container`** (inside `_cold_plot`) so individual parameters are visually distinct. |

### [`ssm_plots.py`](../ssm_plots.py) — Step 1 / Step 2 diagnostic Streamlit blocks

| Function | Purpose |
|---|---|
| `render_open_plots(open_data, para_caps, open_arr, fname="")` | Five expanders for Open dummy: extra-element controls, capacitance plot, conductance plot, Im(Y)/ω vs 1/ω², measured vs modeled Smith. Returns `{cap: (mode, extra_SI)}`. |
| `render_short_plots(short_arr, para_short, fname="", freq=None)` | Lead-inductance plot for Short dummy (Lb/Lc/Le vs frequency, fixed 0–150 pH). |
| `_rlc_params_summary(p, fname)` | Build header param dict for a de-embedded `.s2p` file. |
| `_compare_bode_smith(*, S_a, S_b, freq, fname, key_suffix, label_a, label_b, color_a, color_b, smith_meas_label, smith_sim_label, gain_title, extra_download_fn=None)` | Side-by-side Bode (h21²+U) and Smith chart comparing two S-param datasets. **`color_a`/`color_b` kwargs are kept in the signature but ignored** — colours now come from `FT_FMAX_COLORS` (fT=blue, fmax=red); A vs B is distinguished by **dash** (A=solid, B=dashed). Returns `(fT, fmax)` of the right-hand trace. xlsx download uses the standardised `bode_excel_bytes` (simulated + extrapolated columns). **`extra_download_fn(fT_b, fmax_b)`** (optional) returns a `(label, data, file_name, mime)` tuple forwarded to the Smith chart's `plotly_with_dl` so a download (e.g. OS-deembedded / intrinsic S2P) sits **beside that chart's ⬇ xlsx button** — the OS-deembed and intrinsic preview sections use this for their S2P downloads. |
| `render_os_deemb_preview(S_raw, freq, z0, para_step1, fname)` | Step 2 — Raw vs Open/Short de-embedded preview + download. Returns `S_step1`. |
| `render_intrinsic_preview(S_raw, freq, z0, para_step1, para_eff, fname, S_step1=None)` | Step 3 footer — OS de-embedded vs Intrinsic preview + download. |
| `render_ft_fmax_card(S_mea, S_sim, freq, *, model_name, key, height=560, compact=False)` | **Per-model fT/fmax mini-card with standardised colour scheme + 2x2 legend + extrap method radio.** Colour: fT (|h21|²) = blue, fmax (Mason U) = red (`FT_FMAX_COLORS`). Dash: measured = solid+markers, modeled = dashed, extrap = dotted. Legend: **vertical stack pinned bottom-left inside the plot** (small font, semi-transparent bg — avoids the overlap a 2×2 grid caused in the narrow in-plot box). When any trace needs extrap, a Streamlit radio **underneath the chart** picks −20 dB/dec or single-pole (slider to the radio's right, default window = final 5 GHz). Values read from session_state before the figure is built so the widgets can sit below it. xlsx download uses the standardised `bode_excel_bytes` (simulated + extrapolated columns). |
| `_build_forward_bode(S, freq_hz, title, *, extrap_method="−20 dB/dec", sp_window=None)` | Build a **single-dataset** fT/fmax Bode figure (FT_FMAX_COLORS, extrap dotted). Returns `(fig, any_needs, bode_xl)`. Forward-sim sibling of `render_ft_fmax_card` (which is measured-vs-modeled). |
| `render_forward_bode_block(S, freq_hz, title, key)` | Render the single-dataset Bode plot + download/copy, then (when extrapolation is needed) a `segmented_radio` method selector + single-pole window slider underneath. Used by the custom-model forward simulator (`custom_model/ui_use.py`) so it matches the built-in models. |
| `_extrap_f0(f_ghz, gain_db, extrap_method, sp_window)` | 0-dB crossing (GHz) via the selected method (single-pole window or −20 dB/dec). |
| `_eff_ft_fmax_ghz(f_ghz, h21_db, U_db, extrap_method="−20 dB/dec", sp_window=None)` | Effective fT/fmax (GHz): in-band 0-dB crossing if present, else the *selected* extrap method (tracks the Bode radio). |
| `_read_extrap_selection(extrap_key, f_ghz)` | Read the Bode card's extrap-method radio + single-pole window from session_state so τ/fmax use the same fT. |
| `_tau_total_ps(fT_ghz)` / `_calc_fmax_ghz(fT_ghz, CBC, Rbb)` | τ_total = 1/(2π fT) in ps; fmax = √(fT/(8π·CBC·Rbb)) in GHz (both None on invalid input). |
| `render_tau_fmax_expander(*, key, freq, S_meas, CBC, Rbb, S_model=None, tau_sum=None, tau_sum_label, tau_sum_tex, extrap_key=None)` | Expander "Calculated τ_total and fmax". Left col: τ_total eqns, then `tau_sum` (τ_B+τ_C / τ) as latex, then a **table** (rows Measured/Modeled — single "Value" row when no model; columns fT, τ_total, τ_total − tau_sum). Right col: fmax eqn + **Extracted ⇄ Custom** radio for C_BC / R_bb (number inputs), then a **table** (rows Measured/Modeled; columns fT, fmax simulation = S-param 0-dB crossing, fmax calculation = √ formula). Cells via local `_num` formatter (n/a on None/non-finite). `S_model` given → measured **and** modeled (4 fmax values, SSM extraction); omitted → single dataset (RF simulator). `extrap_key` makes fT/fmax follow the Bode plot's extrap-method radio. Called by `models/base_ui.py::render_override_and_smith` (T/π/XuT only) and `RF_simulator.py` (all SSM models except Kun-Yang). The expander is wrapped in `st.container(key="hbt_exp_view_tau_" + sanitized(key))` — gray left-border, part of the fit page's read-only view cluster (`tools/ui_theme.py` action-color system). |
| `_apply_text_autoformat(text, pattern, replacement)` | Regex substitution with `\1..\9` capture-group templating (used for Smith-label mathtext). |
| `_smith_grid_values(n)` | Return `(r_values, x_values)` for N evenly-spaced Smith-chart grid lines. |
| `_draw_mpl_smith_background(ax, line_lw, grid_lw, density)` | Draw constant-R / constant-X grid arcs for a unit Smith chart on a matplotlib axis. |
| `_mult_label_text(sp, mult)` | On-chart label implied by an S-param multiplier: `1`→`"S21"`, `>1`→`"S21x3"`, `<1`→`"S21/5"` (reciprocal). `%g`-trimmed; bare name for non-positive/unparseable. |
| `_sync_text_to_mult(skey, sp)` | `on_change` callback that rewrites an S-param's Text field from its multiplier via `_mult_label_text`. |
| `render_matplotlib_smith(S_mea=None, S_sim=None, fname="", topo_key="", *, sets=None, default_multiplier=1.0, phase="both", freq_hz=None, add_pool=None)` | Publication-style matplotlib Smith chart. Backward-compatible (mea/sim) and extensible (`sets=[…]`) call forms. `phase="controls"` renders widgets only; `phase="chart"` renders the figure only; `phase="both"` is the legacy combined form. SSM, RF simulator **and RF parameter extraction** call it twice (right column = controls, left column = chart) inside a side-by-side block, passing `freq_hz` for the auto frequency-range annotation. The freq-range label is (re)seeded by a **frequency-span signature** into a dedicated annotation slot, so a new device (e.g. after a cross-page handover) refreshes the label instead of leaving it blank, while an unchanged span preserves manual edits. Editing a Multiplier auto-updates that S-param's Text label (`_sync_text_to_mult`). **`add_pool`** = list of `{"label","S"}` for *other* uploaded files; the styling table then shows a **"➕ Add a file trace"** button + per-row file picker / 🗑 remove / own colour, appending the chosen files to `sets` (selection persisted in `…_extra_files`) so multiple measured files overlay on one chart (used by RF parameter extraction). When any extra file is overlaid the chart switches to **one colour per file** (base + each added file get a single colour; the per-S-param "Trace" colour column is hidden). S-param labels (S11…S22) **auto-position** at each primary-set trace's centroid (nudged outward via `_auto_label_pos`) instead of fixed defaults — x/y stay adjustable and the text-colour picker is always available. |
| `render_ft_fmax_overlay(S_raw, sim_results, freq, fname)` | Multi-model Bode plot using `FT_FMAX_COLORS` — all fT traces blue, all fmax traces red. Models are distinguished by line name in the legend; modeled traces are dashed, extrap traces dotted. xlsx download uses the standardised `bode_excel_bytes` (simulated + extrapolated columns). |

### [`agent_api.py`](../agent_api.py) — headless AI-agent fitting API (NEW)

No Streamlit UI required — see [`INDEX.md`](../../../INDEX.md#ai-agent-fitting-api-toolsssmagent_apypy)
(repo root) for the full write-up. Bootstraps `sys.path` to the repo root at
module import time (two parents up from this file) so `python
tools/SSM/agent_api.py ...` works as a CLI from any cwd; imports Streamlit
transitively (through `tools.SSM.models`) but never calls `st.*` itself.

| Function | Purpose |
|---|---|
| `load_data(path)` | `.s2p`/`.csv` → `{freq, S, z0, header_lines, meta}`. Parses the file's leading `!` header comments (`_read_header_lines`, `_parse_header_values`, `_interpret_header`) to detect a prior de-embedding step (pad caps + lead L already removed) — `meta` carries `deembedded` / `removed_params` / `status`. |
| `list_models()` | Builtin SHORT → `NAME` from `models.REGISTRY`, plus a custom-model note. |
| `param_specs(model)` | Every parameter a builtin or custom model accepts, as `(key, label, si_scale, unit)` — via `_model_context`. |
| `_model_context(model)` | Dispatch `model` (builtin SHORT vs custom) → `(is_builtin, cls, cm, plan, spec_list)`; resolves `_TUNING_PAD_SPECS` overrides (Xu/Kun-Yang) and, for custom models, canonicalises each component to the shared `base_ui` param table (with a `Rbc`/`Rbcx` kΩ-scale override, `_CANON_SCALE_OVERRIDE`) so scale/unit match the builtin convention. |
| `_default_params_for(model, low_perf=False)` | Physics-informed SI-unit starting dict: 0 for pad caps/lead L/access R, `informed_default_range` mid-range (optionally the low-performance table) for everything else. |
| `_resolve_custom_model(model)` | Coerce a path / JSON string / dict / `CustomModel` → `CustomModel`, with a clear `ValueError` on failure. |
| `_HAS_CUDA` / `_cp` / `_cuda_device_name()` | CUDA (cupy) detection — same try-import convention as `models/base_ui.py`; never raises when cupy/a GPU is absent. |
| `HAS_RUST` | Re-exported from `helpers.rust_kernels.HAS_RUST`. |
| `_resolve_backend(backend, *, is_builtin, cm=None, plan=None)` | `"auto"/"cuda"/"rust"/"numpy"` → `(resolved, note)` — priority **CUDA > Rust > NumPy**; explicit `"cuda"`/`"rust"` fall back gracefully (with a note) when unavailable. |
| `_custom_rust_is_trustworthy(cm, plan)` / `_CUSTOM_RUST_TRUST` | One-time, cached-per-topology parity probe (small fixed freq grid) comparing `rust_kernels.sim_custom_batch` against the NumPy reference. The kernel matches NumPy (≤ ~1e-11) at physically meaningful values; it only diverges on **degenerate value sets** (an access leg with R and L both exactly 0 stamps chained ±1e12 near-shorts whose ill-conditioned LU solve makes Rust and LAPACK legitimately disagree — neither is authoritative there), so the probe floors zero values to small benign per-kind values and only fails on a true topology-level mismatch. `backend="auto"` silently skips Rust for a custom model when the probe fails; `backend="rust"` explicitly still uses it, with a warning note in the result. |
| `backend_status()` | `{cuda_available, cuda_device, rust (rust_kernels.rust_diagnostic()), auto_resolves_to, priority}` — backs the `backend` CLI subcommand. |
| `_dispatch_simulate(is_builtin, cls, plan, params, freq, z0, resolved)` | Low-level simulate given an ALREADY-resolved backend. Builtins: `cuda`/`rust` → `cls.simulate_batch(xp=cupy/np)` (Rust auto-dispatch lives inside `simulate_batch`); `numpy` → calls `cls._SIM_WRAP_BATCH_FN` directly, bypassing Rust. Custom: `cuda`/`numpy` → `custom_core.simulate_custom_model_batch(xp=cupy/np)`; `rust` → `helpers.rust_kernels.sim_custom_batch`. |
| `simulate(model, params, freq, z0=50.0, backend="auto")` | Forward-simulate `S[N,2,2]` via `_dispatch_simulate` after `_resolve_backend`; builtins route through `simulate_batch` (`B=1`), custom models compile a `SimPlan` once and call `simulate_custom_model_batch`/`sim_custom_batch`. |
| `residuals(S_meas, S_model)` | Thin wrapper around `models.base_ui._port_residuals` (the exact UI metric). |
| `_residual_vector(S_meas, S_mod)` | Stacked real/imag per-port residual vector, normalised like `_port_residuals`, for the `least_squares` fit method. |
| `_resolve_bounds(key, label, current_si, scale, user_bounds, low_perf=False)` | Per-parameter `(lo, hi)` SI bounds: explicit override, else `informed_default_range` clamped into `tune_hard_limits`, else a generous fallback box. |
| `fit(data, model, initial=None, fit_keys=None, fixed=None, bounds=None, method="auto", maxiter=400, backend="auto")` | Fit `model` to `data` (from `load_data()`). Auto-freezes the six header-removed parasitics (Cpbe/Cpce/Cpbc/Lb/Lc/Le) at 0 when `data["meta"]["deembedded"]` and neither `fit_keys` nor `fixed` was given. `method="auto"`/`"nelder-mead"` runs bounded `scipy.optimize.minimize` directly on the Total residual %% (matches the UI number exactly); `"least_squares"` runs bounded `scipy.optimize.least_squares` (`trf`) on `_residual_vector` — faster, same metric family. `backend` is resolved ONCE up front (`_resolve_backend`) and reused for every objective evaluation. Detects a low-performance device (`base_ui._detect_low_perf_device`) to widen the starting guess/bounds. Returns `{params, residuals, success, n_evals, message, backend, backend_note?}`. |
| `build_custom_model(base="Cheng T", modifications=None, name=None)` | `custom_model.core.builtin_custom_model(label)` (label resolved via `_resolve_base_label`/`_BASE_ALIASES`) + optional edits (`_apply_modification`). |
| `add_series_element` / `add_parallel_element` | Append an R/L/C `Element` to a named `CustomModel` Network (`intrinsic_base/be/bc/ce`, `port1`, `port2`, `emitter`) — new series group vs. parallel-in-existing-group. |
| `add_shunt_branch(model, section, place, *groups)` | Append a new `ShuntBranch` to `model.extrinsic` or `model.parasitic` at `place` (`p1-p2`/`p1-gnd`/`p2-gnd`). |
| `add_parallel_to_shunt(model, section, place, kind, name, ...)` | Add an element in parallel within an existing `ShuntBranch`'s group (e.g. Rbcx ∥ an existing Cbcx). |
| `save_custom_model(model, path)` | Serialise a `CustomModel` to an explicit JSON path (unlike `custom_model.core.save_model`, which always writes into the repo's `custom_models/` library folder). |
| `_cmd_inspect` / `_cmd_models` / `_cmd_backend` / `_cmd_fit` / `main(argv=None)` | CLI subcommands (`inspect FILE`, `models`, `backend`, `fit FILE --model … [--initial/--fit-keys/--fixed/--bounds/--method/--backend/--maxiter/--out]`) — JSON in/out. |

---

## Helpers (`helpers/*.py`)

### [`helpers/_array_utils.py`](_array_utils.py) — Shared array helpers (NEW)

Canonical home for the small shape/broadcast utilities formerly duplicated across `helpers/deembed_math.py`, `models/_shared.py`, and `models/degachi.py`. Placed under `helpers/` (not `models/`) so the helpers package — which can't reach back into `models/` without a load-order cycle — can use them directly. `models/_shared.py` re-exports the names so `from ._shared import _b1, _detect_B, _stack22` keeps working everywhere it used to.

| Function | Purpose |
|---|---|
| `_b1(p, key, default, xp, dtype=None)` | Fetch `p[key]` (or default), reshape `(B,)` → `(B,1)`; scalars stay scalar. Optional `dtype=` coerces for the fp32 sweep path. |
| `_detect_B(p, xp)` | Determine batch size B from any `(B,)`-shaped value in `p`. |
| `_stack22(a00, a01, a10, a11, xp)` | Stack four planes into a `(..., 2, 2)` tensor in 3 launches. |

### [`helpers/rf_math.py`](rf_math.py) — Pure RF math (no Streamlit, no plotting)

| Function | Purpose |
|---|---|
| `s_to_y(S, z0=50.0)` | S → Y conversion, vectorised across all frequency points. |
| `_inv2(M)` | Per-row 2×2 analytic inverse (no per-matrix Python loop). |
| `y_to_z`, `z_to_y` | Aliases for `_inv2`. |
| `y_to_s_single(Y, z0=50.0)` | Y → S for a single 2×2 matrix. |
| `y_to_s_batch(Y, z0=50.0)` | Y → S over a batch (delegates to `y_to_s_vec`). |
| `y_to_s_vec(Y, z0=50.0, xp=None)` | GPU-aware vectorised Y → S using hand-inlined 2×2 algebra (numpy or cupy). |
| `inv2x2(M, xp=None)` | Analytic 2×2 batched inverse — bypasses cuSOLVER. |
| `mm2x2(A, B, xp=None)` | Analytic 2×2 batched matmul — bypasses cuBLAS. |
| `safe_median(arr, n=None)` | Median of finite values, returns `0.0` if empty. |
| `strict_freq_check(f_dut, f_dummy, label)` | Raise `ValueError` if frequency grids differ. |
| `open_elem_Y(C, mode, extra, w)` | Pad capacitor admittance — None / Parallel L / Series L / Series R. |
| `short_lead_Z(R, L, Cpar, w)` | Series-lead impedance with optional parallel cap. |
| `_smith_grid_xy(max_r=1.0)` | Cached coordinate arrays for the Smith-chart background. |
| `extended_smith_grid(max_r=1.0)` | Plotly Smith chart background traces (built fresh each call from cached coords). |
| `params_hash(p)` | MD5 hash of a parameter dict, for caching. |

### [`helpers/s2p_io.py`](s2p_io.py) — Touchstone / CSV I/O and dummy simulators

| Function | Purpose |
|---|---|
| `build_Y_pad(p, w)` | 2×2 pad admittance matrix at angular freq `w`. |
| `build_Z_ser(p, w)` | 2×2 series-lead impedance matrix at angular freq `w`. |
| `write_s2p(freq_hz, S, title="", params=None)` | Serialize to Touchstone `.s2p` (DB format) bytes. Header `!` lines list `params`. |
| `parse_s2p(content)` | Parse a Touchstone `.s2p` (accepts str or bytes) → `(freq, S, z0)`. Strips trailing `!` comments on any line (option + data rows) and skips stray non-numeric tokens — same tolerance as the Rust `parse_and_compute_batch` parser, so both paths accept the same files. Raises `ValueError` when no 9-column data rows are found (was: silently returned empty arrays). |
| `_maybe_float(tok)` | `float(tok)` or `None` — tolerant token parser used by `parse_s2p`'s data rows. |
| `parse_s2p_bytes(raw)` | Bytes-only alias of `parse_s2p` (backwards-compat). |
| `parse_csv(content, z0=50.0)` | Parse VNA CSV export (RI columns) → `(freq, S, z0)`. |
| `interpolate_s2f(f_src, S_src, f_tgt)` | Interpolate S to a new frequency grid. |
| `load_cal(fobj)` | Streamlit-aware S2P loader. |
| `simulate_open(p, freq, z0=50.0)` | Forward-simulate Open dummy from pad params (cached). |
| `simulate_short(p, freq, z0=50.0)` | Forward-simulate Short dummy from pad+lead params (cached). |

### [`helpers/deembed_math.py`](deembed_math.py) — Open/Short extraction & de-embedding

| Function | Purpose |
|---|---|
| `_agg_arr(arr, n0, n1, method="Median", trim_pct=20)` | Median / trimmed-mean aggregator over an index window. |
| `_open_elem_Y_vec(C, mode, extra, omega, xp)` | Vectorised pad-cap admittance over an `(N,)` omega array. |
| `_short_lead_Z_vec(R, L, Cpar, omega, xp)` | Vectorised series-lead impedance over an `(N,)` omega array. |
| `build_Y_pad_vec(p, omega, xp)` | `(N,2,2)` pad admittance — fully vectorised. |
| `build_Z_ser_vec(p, omega, xp)` | `(N,2,2)` series-lead impedance — fully vectorised. |
| `_open_elem_Y_batch(C, mode, extra, omega, xp)` | `_open_elem_Y_vec` variant where C may be `(B,1)`. |
| `_short_lead_Z_batch(R, L, Cpar, omega, xp)` | `_short_lead_Z_vec` variant where R/L may be `(B,1)`. |
| `_b1` | Imported from `helpers/_array_utils.py` (was a stale local copy that lacked the `dtype=` arg). |
| `build_Y_pad_batch(p, omega, B, N, xp)` | Pad admittance as 4 broadcast planes for sweep tuning. |
| `build_Z_ser_batch(p, omega, B, N, xp)` | Series-lead impedance as 4 broadcast planes for sweep tuning. |
| `step_open(open_data, n0, n1, method, trim_pct)` | Extract Cpbe/Cpce/Cpbc + diagnostic conductance arrays from Open dummy. |
| `step_short(short_data, freq, Cpbe, Cpce, Cpbc, ...)` | Extract Lb/Lc/Le, Rpb/Rpc/Rpe from Short dummy (Open subtracted first). |
| `peel_parasitics(S_raw, freq, z0, p)` | Remove pad+lead parasitics → Y_intrinsic (cached). |
| `deembed_open_short(Y_dut, Y_open, Y_short)` | Standard open-short de-embedding (Gao §4.2). |
| `deembed_thru_half(Y_dut, Y_thru_deemb)` | THRU/2 half-impedance subtraction. |

### [`helpers/metrics.py`](metrics.py) — Gain figures of merit

| Function | Purpose |
|---|---|
| `compute_h21_U(S)` | Compute |h21|² (dB) and Mason U (dB) from S-parameters (z0 = 50 Ω). |
| `find_ft_fmax(f_ghz, h21_db, U_db)` | Vectorised search for the first 0 dB crossing (boolean masks + `np.flatnonzero`) → `(fT, fmax)`. The earlier Python `for i in range(len(arr))` loop ate ~30 ms × N_DUTs in the SSM tab; the vectorised version is ~50× faster. |
| `extrap_20dbdec(f_ghz, gain_db, n_pts=60, f_max_target=None)` | Slope-locked −20 dB/dec extrapolation → `(f_ext, g_ext, f_zero)`. |
| `single_pole_extrap(f_ghz, gain_db, idx_lo, idx_hi, n_pts=60, f_max_target=None)` | Log-linear (single-pole) fit over a user index window → `(f_ext, g_ext, f_zero, slope, intercept)`. Slope may differ from −20 — that disagreement is itself diagnostic. |
| `compute_metrics(Y, freq_hz)` | Build DataFrame: Freq, |h21|², Mason U, MAG/MSG, K, fT/fmax plateaus. |
| `extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max)` | Genuine-crossing fT/fmax extractor with extrapolation fallback (≥10 consecutive points above 0 dB). Ported to Rust as `_extract_limit_rust` inside `parse_and_compute_batch` — see `helpers/rust_kernels.py`. |

### [`helpers/plotly_plots.py`](plotly_plots.py) — Plotly Smith / Bode / Plateau builders

| Function / Constant | Purpose |
|---|---|
| `PALETTE` | 10-color hex palette for trace cycling. |
| `FT_FMAX_SYMBOLS` | Standardised marker symbols for fT / fmax traces: `{"h21": "circle", "U": "square", "MAG": "diamond"}`. |
| `FT_FMAX_COLORS` | **Standard colour map** — `{"fT": "#1f77b4", "fmax": "#d62728"}` (blue / red). Applied everywhere a Bode plot draws fT/fmax measurements so users can identify a metric by colour alone. Modeled traces use the same colour with `dash="dash"`; extrap uses `dash="dot"`. |
| `thinned_indices(n, max_markers=25)` | Log-spaced sample of indices into a length-`n` array. Lets overlay plots cut WebGL marker primitives ~40× while keeping the line trace at full resolution. |
| `add_overlay_trace_with_markers(fig, x, y, *, name, color, symbol, ...)` | Add a Bode trace as (full-resolution line, hidden from legend) + (thinned marker overlay, owns the legend entry). |
| `darken(c)` | Darken a `#rrggbb` color by 45 units per channel. |
| `bode_layout(title, ytitle, yr, xr)` | Plotly layout dict for Bode/plateau plots (log-x). Legend font is 18 pt. |
| `make_smith(S, f_array, f_min, f_max, toggles, scales, title, max_r=1.0)` | Plotly Smith chart with up to 4 selectable S-param traces. |
| `make_bode(df, title, xr, yr, sh21, su, smag, color, *, show_20db=True, show_sp=False, sp_window_idx=None, extrap_f_max=None, return_extrap_df=False, return_excel_bytes=False)` | Plotly Bode plot with optional −20 dB/dec and single-pole-fit extrapolations. **Uses `FT_FMAX_COLORS`** — `color` arg kept for back-compat but ignored. `return_excel_bytes=True` → `(fig, xlsx)` via `bode_excel_bytes` (standardised sim + extrap export). |
| `_build_bode_extrap_df(df, extrap_curves, sh21, su)` | Combine measured + extrapolated values into one stitched DataFrame (legacy `return_extrap_df` path; superseded by `bode_excel_bytes`). |
| `make_plateau(df, res, title, xr, sh21, su, smag, color)` | Plotly GBP plateau plot (fT, fmax(U), fmax(MAG)). Uses `FT_FMAX_COLORS` + `FT_FMAX_SYMBOLS`. |
| `make_smith_bode_slider_fig(...)` | Smith chart + fT/fmax bode side-by-side with multi-slider semantics. |
| `_decimate_freq(S_batch, freq, max_points=200)` | Stride-decimate the frequency axis to `≤ max_points`. |
| `_compact_json_1d(arr, digits=5)` | Format a 1-D float array as compact JSON (~3× smaller). **Non-finite values are emitted as `null`** — a bare `nan`/`inf` token is invalid JS and previously threw a SyntaxError that froze the whole pre-computed slider. |
| `make_smith_bode_joint_slider_html(...)` | HTML-slider variant — embeds full joint sweep as JS array, injects `<input type="range">` per axis. JS now **mutates trace arrays + `Plotly.redraw`** (scattergl-safe; `restyle` silently no-ops on WebGL) and resolves the graph div via `getGD()` (`#hbtSlPlot` → `.js-plotly-plot` fallback) so the sliders never fail to wire. Smith S-params are labelled **inline on the traces**; only the Bode legend remains (small, bottom-left). |

### [`helpers/widgets.py`](widgets.py) — Streamlit input widgets

| Function | Purpose |
|---|---|
| `info_icon_html(text, label="ⓘ")` | Hover-tooltip span snippet. |
| `segmented_radio(label, options, *, index=0, key=None, horizontal=True, format_func=None, help=None, label_visibility="visible", disabled=False)` | **Drop-in `st.radio` replacement** rendering as a segmented button group (`st.segmented_control`) — the chip selector the custom-model builder uses. Returns the selected option **value**; guarantees a non-None selection (deselect falls back to the remembered/default value). Falls back to `st.radio` when `segmented_control` is unavailable (< Streamlit 1.40). Now used everywhere a radio used to be (RF simulator, SSM models' Editor-mode, extrap-method selectors, ebeam calculator, Gummel analyzer, custom-model nav). |
| `apply_pending(target_key)` | Promote a pending quickset value into the widget's session-state key. Call BEFORE the paired `number_input`. |
| `_candidates(arr_disp, default_disp, cold_disp=None)` | Build `[(label, value), …]` list from optional sources. |
| `quickset_buttons(...)` | Row of one-click buttons (mean / median / low f / high f / default / cold). |

### [`helpers/mem_budget.py`](mem_budget.py) — Container/cgroup-aware RAM budget probes

Pure Python, no Streamlit import (safe to import from anywhere, including
`tools/ui_theme.py`). `psutil.virtual_memory()` reports the HOST's memory,
not a container's cgroup limit, so Streamlit Cloud's Auto Tuning chunk
sizing (`models/base_ui.py`) and the sidebar RAM badge
(`tools/ui_theme.py::render_ram_badge`) both go through this module
instead of calling psutil directly.

| Function | Purpose |
|---|---|
| `_read_int(path)` | Read + parse a cgroup accounting file as `int`; `None` on any failure (missing file, "max" sentinel, garbage). Never raises. |
| `_cgroup_limit_bytes()` | cgroup v2 `memory.max` → cgroup v1 `memory.limit_in_bytes` (huge-sentinel ≥ `2**60` treated as unlimited) → `None`. Cached at module level — the limit never changes for the container's lifetime. |
| `_cgroup_used_bytes()` | cgroup v2 `memory.current` → cgroup v1 `memory.usage_in_bytes` → `None`. **Not** cached — usage changes constantly. |
| `ram_limit_bytes()` | Public RAM ceiling: cgroup limit → `psutil.virtual_memory().total` → `None`. |
| `ram_used_bytes()` | Public RAM in use: cgroup usage → `psutil.virtual_memory().used` → `None`. |
| `ram_available_bytes()` | Best-effort allocatable bytes right now: `min()` of (cgroup limit − usage) and psutil's `available`, whichever is known/more pessimistic; falls back to a conservative 1 GiB constant when nothing is knowable. Never raises — this is what `base_ui.py`'s Auto Tuning chunk sizing calls. |
| `ram_usage()` | `(used_bytes, limit_bytes)` for UI display (prefers the cgroup pair, falls back to psutil `(used, total)`); `None` when neither source works — `tools/ui_theme.py::render_ram_badge` renders nothing in that case. |

### [`helpers/fit_cache.py`](fit_cache.py) — Persistent per-(DUT, model) fitted-value cache

| Function | Purpose |
|---|---|
| `_detect_cache_disabled()` | Honor `HBT_DISABLE_FIT_CACHE=1` env var. |
| `is_cache_disabled()` | Public predicate — cached during the session. |
| `_resolve_cache_dir()` | Pick the cache root: `$HBT_FIT_CACHE_DIR` ?: `~/.hbt-tools/`. |
| `_fits_dir()` | The `fits/` subdir under the cache root. |
| `cache_path_str()` | Return the cache **root** directory string. |
| `_basename_no_ext(fname)` | Strip directory + trailing `.s2p`. |
| `_norm_key(fname)` | Canonicalise to the extension-stripped basename. |
| `_dut_dir(fname, create=False)` | Resolve the per-DUT folder. |
| `_model_file(fname, model_short)` | Resolve the per-model JSON file path. |
| `_read_model_file(path)` | Tolerant JSON load → `{saved_at, params}` or `None`. |
| `_atomic_write_json(path, payload)` | Temp-file + rename so a crashed write never leaves a corrupt cache. |
| `_migrate_legacy_if_present()` | One-shot migrator: split a v1 monolithic `fit_cache.json` into per-(DUT, model) files. |
| `load_cache()` | Walk `fits/` and aggregate into `{basename: {model_short: {saved_at, params}}}`. |
| `get_fit(fname, model_short)` | Return cached SI-unit param dict, or `None`. |
| `get_fit_timestamp(fname, model_short)` | Return ISO timestamp string, or `None`. |
| `list_fits(fname)` | Return `{model_short: timestamp}` for the given s2p basename. |
| `_all_numeric_zero(params)` | True if every numeric value in `params` is exactly `0.0`. |
| `save_fit(fname, model_short, params_si)` | Persist a fine-tuned param dict. Refuses to save when every numeric value is `0.0`. |
| `delete_fit(fname, model_short=None)` | Drop one model's file (or the whole DUT folder). |
| `export_cache_bytes()` | Serialize all per-model files into one unified JSON. |
| `import_cache_bytes(raw, merge=True)` | Parse a unified JSON blob and write back as per-model files. |
| `differs_from(p_si, ref_si, keys=None, ...)` | True if any numeric value in `p_si` deviates from `ref_si`. |

### [`helpers/rust_kernels.py`](rust_kernels.py) — Optional Rust SIMD/Rayon acceleration

| Symbol | Purpose |
|---|---|
| `HAS_RUST` (bool) | `True` iff the `hbt_rust_kernels` Rust extension is built and loaded. |
| `_arch_tag()` | Map platform → bin subfolder name (e.g. `win_amd64`). |
| `rust_diagnostic()` | Snapshot of the loading state (HAS_RUST, binary path, import error if any). |
| `_np_inv2x2_batch(Y)` | NumPy reference for the (B,2,2) analytic inverse. |
| `_np_mm2x2_batch(A, B)` | NumPy reference for the (B,2,2)×(B,2,2) matmul. |
| `_np_y_to_s_batch(Y, z0=50.0)` | NumPy reference for batched Y → S. |
| `_np_y_to_s_4d(Y, z0=50.0)` | NumPy reference for 4-D (B,N,2,2) Y → S. |
| `_np_port_residuals_batch(S_mea, S_mod_batch)` | NumPy reference for the (B,5) `[Total, S11, S12, S21, S22]` residual table. |
| `_c128(arr)` | Coerce to complex128 + C-contiguous. |
| `inv2x2_batch(Y)` | Public wrapper — Rust if available, NumPy fallback. |
| `mm2x2_batch(A, B)` | Public wrapper — Rust if available, NumPy fallback. |
| `y_to_s_batch(Y, z0=50.0)` | Public wrapper — Rust if available, NumPy fallback. |
| `y_to_s_4d(Y, z0=50.0)` | Public wrapper — Rust if available, NumPy fallback. Primary entry from the visual-tuning sweep loop. |
| `port_residuals_batch(S_mea, S_mod_batch)` | Public wrapper — Rust if available, NumPy fallback. |
| `_phase2_dispatch_enabled()` | Env-gate for the Phase 2 sim_*_batch kernels. |
| `_phase2_parity_check_enabled()` | Env-gate for the optional parity check on every Phase 2 call. |
| `_normalize_params_for_rust(params)` | Broadcast + flatten multi-dim sweep tensors to 1-D float64. |
| `_phase2_dispatch(rust_fn_name, params, freq, z0, np_fallback)` | Common dispatch helper for all topology wrappers. |
| `sim_cheng_t_batch(params, freq, z0=50.0, *, np_fallback)` | Cheng T-topology end-to-end batched simulation. |
| `sim_cheng_pi_batch(params, freq, z0=50.0, *, np_fallback)` | Cheng π-topology end-to-end batched simulation. |
| `sim_xu_t_batch(params, freq, z0=50.0, *, np_fallback)` | Xu T-topology end-to-end batched simulation (parallel `Rbcx ∥ Cbcx`). |
| `sim_kunyang_batch(params, freq, z0=50.0, *, np_fallback)` | Kun-Yang HEMT pi-topology end-to-end batched simulation. |
| `sim_custom_batch(plan, params, freq, z0=50.0, *, np_fallback)` + `_encode_custom_plan(plan)` | **Generic, data-driven** custom-model end-to-end batched simulation. `_encode_custom_plan` serialises a `SimPlan` to a plain-int dict; the Rust `sim_custom_batch` kernel stamps the `(n×n)` nodal Y, Kron-reduces via a dense complex LU solve, and converts Y→S per `(B,N)`. Silent NumPy-vectorised fallback. |
| `SIM_FOR_TOPOLOGY` | `{topology_short → wrapper}` lookup consumed by `SSMModelTemplate.simulate_batch`. |
| `_EXTRACT_METHOD_LABELS` | u8 method code → string label ("No Data" / "No Gain" / "0dB Cross" / "Extrap & Plat."). |
| `_np_parse_and_compute_batch(files_bytes, n_pts=2, f_min=0.01, f_max=50.0)` | NumPy reference for the bulk-upload accelerator (parse_s2p + s_to_y + compute_metrics + extract_limit). Byte-identical fallback. |
| `_expand_soa_to_per_file(soa)` | Convert the SoA dict returned by the Rust kernel into per-file dict list with **zero-copy NumPy views**. ~300 µs vs ~62 ms PyArray-alloc overhead in the earlier list-of-dicts shape. |
| `parse_and_compute_batch(files_bytes, n_pts=2, f_min=0.01, f_max=50.0)` | **Public bulk-upload accelerator.** Parses every `.s2p`, computes metric columns, runs `extract_limit ×3` per file — all in one Rayon-parallel section. Used by the IOED bulk-upload loop. |

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

| Function | Purpose |
|---|---|
| `EXCEL_MIME` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`. |
| `_axis_text(axis_obj)` | Safely fetch a Plotly axis title text. |
| `_is_smith(fig)` | True when the figure is a Smith chart (Re(Γ) / Im(Γ) axes). |
| `_smith_col_name(trace_name)` | Map a Smith-trace name → compact Excel column prefix (e.g. `S11_meas`). |
| `_freq_sheet_name(x_lbl, x_arr)` | Build an Excel sheet name from the frequency range. |
| `_collect_traces(fig)` | Return `[(name, x, y), …]` for exportable traces. |
| `_fig_frames(fig)` | Extract Plotly traces → `[(sheet_name, DataFrame), …]` (smart Smith vs wide vs per-trace layout). **Single source of truth** shared by `fig_to_excel_bytes` and `fig_to_tsv` so xlsx and clipboard never drift. Returns None when no exportable data. |
| `fig_to_excel_bytes(fig)` | Write `_fig_frames(fig)` → `.xlsx` bytes. |
| `bode_excel_bytes(freq_ghz, sim_traces, extrap_traces=None)` | Canonical fT/fmax Bode export → single-sheet `.xlsx` bytes: simulated block (`Freq (GHz)` + each gain trace) plus, when extrapolation is present, a side-by-side `… (extrap)` block on a unified extrap freq axis. Used by every Bode plot (RF simulator, SSM, RF extraction) via `plotly_with_dl(excel_bytes=…)` / `make_bode(return_excel_bytes=True)`. |
| `frames_to_tsv(frames)` | Join `[(name, DataFrame), …]` → tab-separated clipboard text (single → one table; multi → stacked with `# <name>` headers). Returns None when empty. |
| `fig_to_tsv(fig)` | Plotly figure → TSV for clipboard paste, built directly from `_fig_frames` (no workbook write/read — cheap to recompute each rerun). Matches `fig_to_excel_bytes` content exactly. **Preferred** feed for `copy_button` whenever a figure is in hand. |
| `xlsx_bytes_to_tsv(xl_bytes)` | Read `.xlsx` bytes back → TSV. Used only when just the finished workbook is available (manually-built sheets, or pre-built Bode bytes whose extrap columns are hidden traces). Prefer `fig_to_tsv` otherwise — it skips this openpyxl read-back. |
| `copy_button(text, key, *, container=None, label="📋 copy", height=46)` | Clipboard "copy" button via `st.iframe` + inline JS. Copies with `navigator.clipboard.writeText` first (no focus → **page doesn't scroll to top**), falling back to a hidden-`<textarea>` + `execCommand('copy')` with `focus({preventScroll:true})`. Flashes an **opaque "✓ Copied" toast over the button for ~1 s**. Styled to match the xlsx button. Used beside every plot's xlsx download (in `plotly_with_dl`, plus RF Smith, IOED Bode, HP4155A export). |
| `plotly_with_dl(fig, key, filename="", width="stretch", container=None, extra_download=None, excel_bytes=None, **kwargs)` | Render Plotly chart + an xlsx download button **and a 📋 copy button** below it. Copy text comes from `fig_to_tsv(fig)` (cheap) — or `xlsx_bytes_to_tsv(excel_bytes)` for the Bode `excel_bytes` path so hidden extrap columns are kept. **`extra_download=(label, data_bytes, file_name, mime)`** adds a button alongside xlsx (SSM measured-vs-modeled Smith → "📥 modeled S2P"); the copy button sits after it. **`excel_bytes`** supplies pre-built xlsx (e.g. `bode_excel_bytes`) instead of auto-extracting from the figure. |
| `build_excel(summary_df, all_data)` | Multi-sheet workbook: Summary + per-DUT DataFrames. |
| `metric_card(col, title, val, sub, color="#4A90D9")` | Styled HTML metric tile (used in IOED tab_ind, batch tab). |

---

## Models (`models/*.py`)

### [`models/__init__.py`](../models/__init__.py) — Registry + abstract base

| Class / function | Purpose |
|---|---|
| `class AbstractSSMModel(ABC)` | Public API every SSM model must implement. Class attrs `NAME`, `SHORT`, `TOPOLOGY_CHAR`. |
| `AbstractSSMModel.extract(Y_ex1, freq, n_low, **kwargs)` | Extract intrinsic params from the de-embedded admittance. Returns `(params, arrays)`. |
| `AbstractSSMModel.simulate(params, freq, z0=50.0)` | Forward-simulate S-parameters from a fully populated params dict. |
| `AbstractSSMModel.render_step_formulas(cls)` | Render extraction-step LaTeX in Streamlit. |
| `AbstractSSMModel.render_results_table(cls, params)` | Render extracted scalar parameters as a Streamlit dataframe. |
| `AbstractSSMModel.render_formula_trace(cls)` | Render a collapsible expander with the full extraction + simulation chain. |
| `AbstractSSMModel.render_override_and_smith(cls, fname, S_raw, freq, z0, para_eff, extract_result, **kwargs)` | Render fine-tune override UI + Smith chart. Returns `S_sim` or `None`. |
| `AbstractSSMModel.get_s2p_header_params(cls, params, para_eff)` | Default human-readable param dict for `.s2p` header. |
| `REGISTRY` | `{SHORT: ModelClass}` — drives the model UI loop in `main_ssm_extraction.py`. |
| `DEFAULT_SELECTION` | List of model SHORTs shown on first run. |

### [`models/_shared.py`](../models/_shared.py) — Font helpers (`_b1`/`_detect_B`/`_stack22` re-exports)

After the array-utility consolidation, this module is mostly font code. The three array helpers (`_b1`, `_detect_B`, `_stack22`) are now re-exported from `helpers/_array_utils.py` so existing call sites (`from ._shared import _b1, ...`) keep working unchanged.

| Function | Purpose |
|---|---|
| `_b1, _detect_B, _stack22` | See `helpers/_array_utils.py`. |
| `_try_download_inter()` | One-time fetch of Inter font into `models/fonts/`. |
| `has_inter()` | True if Inter is installed locally or successfully cached. |
| `_load_font(size)` | Load a TrueType font with Inter → Arial → fallback chain. |

### [`models/base_ui.py`](../models/base_ui.py) — Shared Streamlit UI helpers for all models

| Function | Purpose |
|---|---|
| `PAD_SPECS` | Tuple list `(key, label, SI_scale, unit, fmt, step)` for the nine pad parameters. |
| `TUNE_HARD_LIMITS` / `TUNE_DEFAULT_RANGES` / `TUNE_LOW_PERF_RANGES` (module-level dicts) | Physics-informed sweep tables in **display units**, keyed by canonical param key. `TUNE_HARD_LIMITS`: `key → (lo, hi)` hard physical bounds (default `(0.0, None)`; `alpha0` boxed to `[0.95, 0.99]`). `TUNE_DEFAULT_RANGES`: default `(lo, hi)` sweep box per key — HBT entries anchored on this-work / Xu 2014 / Cheng 2022 InP extractions, plus a **Kun-Yang HEMT** group (`Cgs/Cgd/Cds`, `Ri/Rgd/Rds`, `R_delay/C_delay`, substrate-pad `Cgsp/Cdsp/Cgdp` + `Rsub1/Rsub2`; KY pads reuse the canonical `Rpb/Rpc/Rpe`+`Lb/Lc/Le` keys relabelled Rg/Rd/Rs + Lg/Ld/Ls). The shared `tau` entry spans (0.05, 5.0) ps to cover both the π-HBT and the KY τ. `TUNE_LOW_PERF_RANGES`: wider `(lo, hi)` for slow / lossy devices (higher Cbc/Cbcx/Rbi, slower τ). `_PARASITIC_KEYS` = pad caps + lead Ls (deliberately-zeroed → stay at 0). |
| `_canonical_tune_key(key, label="")` | Map a spec key OR custom-model component name to the canonical table key: exact key → Greek/typeset label alias (α₀→alpha0, τB→tauB, …) → standalone-token match of key+label (`re.findall(r"[A-Za-z]+[0-9]*")`, longest table key first) → secondary token aliases (`_TUNE_TOKEN_ALIASES`: custom access names Rb/Rc/Re → Rpb/Rpc/Rpe, checked only after every table key missed so "Rbe … aka rE" still wins as Rbe). Returns None on no match. |
| `tune_hard_limits(key, label="")` | Hard `(lo, hi)` physical limits (display units, `hi` may be None) for a param; canonicalises via `_canonical_tune_key`; defaults `(0.0, None)`. |
| `informed_default_range(key, label, current_disp, *, low_perf=False, spec_step=None)` | Physics-informed `(min, step, max)` for one param (display units). Canonical match → `TUNE_LOW_PERF_RANGES`/`TUNE_DEFAULT_RANGES` (widened to include the current value); zeroed parasitics stay `(0,0,0)`; no match → decade box around current (or `(0,0,0)` if current 0). Clamped into the hard limits (limits win). Step = `span/20` @ 2 sig-figs, floored at `spec_step`. |
| `_detect_low_perf_device(S_raw, freq)` | Heuristic low-perf flag: from `compute_h21_U`/`find_ft_fmax` (fallback: `extrap_20dbdec` 0-dB crossing), True iff `fmax < fT` or `fT < 40 GHz`. Any failure → False. Cached once per (topo, file) in `render_tuning_expander`. |
| `_prog_axis_values(lo, hi, n_pts, floor, include=None)` / `_prog_signature(row_vals, floors)` / `_prog_next_bounds(lo, hi, best, hard_lo, hard_hi, floor)` (module-level, pure) | 🪜 Progressive auto-fit geometry helpers (unit-testable). `_prog_axis_values`: one sweep axis over `[lo,hi]` (log spacing when `hi/lo ≥ 50`, else linear; spacing-floor enforced; optional inserted value). `_prog_signature`: quantised memo key (`v / (floor/2)`) — dedups values closer than floor/2, distinguishes a floor apart. `_prog_next_bounds`: per-cycle shrink (`best ± 0.35·span`) / edge-expand (`1.6·span`), clipped to hard limits, span ≥ 2×floor; when the expand is fully clipped away (best pinned against a hard limit) it falls through to the shrink rule so the axis still refines. `_PROG_*` module constants set the budgets / chunk sizes (`_PROG_CHUNK_CPU` is the CPU chunk-size ceiling — `_run_progressive` derives the actual per-cycle chunk from `helpers/mem_budget.py::ram_available_bytes()` instead of using this constant directly, so a run at high freq-point counts can't out-allocate the Streamlit Cloud container) / shrink-expand factors plus the escape knobs (`_PROG_PORT_GOAL` 5 %, `_PROG_STALL_CYCLES`, `_PROG_STALL_REL`, `_PROG_ESCAPE_PTS`); `_PROG_GROUPS` are the canonical refinement pairs — HBT (Zbe/Zbc/extrinsic/transport/gm/access/base) plus Kun-Yang HEMT (Ygs/Ygd/Yds/ky-delay/ky-pad-gs/ky-pad-ds/ky-pad-gd) and generic leads/pads groups. |
| `ssm_residual(S_mea, S_mod)` | RMS relative S-parameter residual across all four ports (%). |
| `_port_residuals(S_mea, S_mod)` | Per-port residual dict (`Total`, S11, S12, S21, S22) in %. |
| `_port_residuals_batch(S_mea, S_mod_batch, xp)` | Batched fused per-port residuals. |
| `render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith", show_title=True, meas_label="Meas.", sim_label="Model", *, compact=False, height=None, extra_download=None, inline_labels=False)` | Per-model Plotly Smith with measured markers + dashed model traces. **`extra_download`** is forwarded to `plotly_with_dl` so the download row can host a second button (📥 modeled S2P beside ⬇ xlsx). **`inline_labels=True`** drops the legend box and labels S11/S12/S21/S22 in colour right on their (model) trace centroids (matplotlib-style) with one tiny `● Meas / - Model` corner hint — used by the Live-tweak preview to save space. |
| `render_smith_with_ftfmax(S_raw, S_sim, freq, model_name, model_short, fname, scales=None, *, s2p_bytes=None, s2p_filename=None)` | Sticky `st.metric` row (Total + S11/S12/S21/S22 residuals, keyed `st.container(key="hbt_resrow_{model_short}")`, delta vs. the previous simulation via `hbt_res_prev_{model_short}_{fname}` session state) above a 2-column layout: Smith chart (left) + fT/fmax mini-card (right). `ui_theme.py` pins the strip with `position: sticky` applied to its auto-generated `stLayoutWrapper` ancestor (via `:has()` — targeting the keyed div itself doesn't work, see `ui_theme.py` comment). When `s2p_bytes` is supplied, the Smith chart's download row gains a `📥 modeled S2P` button beside the xlsx — this replaced the standalone "Download Modeled DUT S2P" section at the bottom of the SSM tab. |
| `smith_scale_controls(fname, topo_key)` | Four `number_input` fields for per-trace Smith display scaling. |
| `sync_pad_from_preov(fname, topo_key, para_eff)` | Copy pre-extraction pad values into per-topology session state when the upstream MD5 hash changes. Also reseeds when any `sim_{topo_key}_{key}_{fname}` pad widget key is missing (e.g. Streamlit GC'd it after a page switch, while the plain sync-hash key survived and would otherwise block the reseed and leave the pad inputs at 0). |
| `_render_cbex_sweep_tool(...)` | Cbex sweep UI for the Cbcx stability search. |
| `_render_tau_total_fit_section(*, all_data, fname, model_short, params, para_eff)` | Multi-file 1/(2πfT) vs 1/IC reference fit (T-models only — ChengT, XuT). Computes fT per bias file from Open+Short pad/lead de-embedded \|h21\|² 0-dB crossing — but **access R is RETAINED** (Rpb/Rpc/Rpe forced to 0 in the dict passed to `peel_parasitics`) so RC/REE in the Cheng formula remain meaningful. Auto-no-op for pre-de-embedded files (when caps/Ls are zero, `peel_parasitics` returns Y_dut unchanged). Plots τ_total (ps) vs 1/IC (1/mA); linear-fits → reference Cje (=slope/(η·Vt)) and τB+τC (=intercept−(Rc+Re)·Cbc). Per-file τCC=(rE+Re+Rc)·Cbc and τE=rE·Cje. Also splits τB / τC via assumed average collector velocity v_c (Liu et al., IEEE EDL 25(12), 2004) — `W_C` (nm, default 120) and `v_c` (cm/s, default 4×10⁷ for 2000 Å InP collectors) are inputs → τ_C = W_C/(2 v_c), τ_B = (τ_B+τ_C) − τ_C. Publishes the v_c-derived τB / τC into session state (`taut_pub_tauB/τC_{short}_{fname}`) so the downstream τB / τC `number_input`s render a "v_c = …" quickset button. Hidden when `< 2` s2p files loaded. Reference-only — does NOT feed back into the model fit. **Returns `True` when it rendered, `False` when skipped (single file)** so the caller can suppress the trailing `---` separator and avoid a double rule with one file. References: Cheng et al. (equation) + Liu et al. (v_c default for InP collector). |
| `_FINETUNE_DIAGRAM_GROUPS` (module-level) | Ordered `(label, [param_keys])` grouping for the diagram-mode fine-tune editor: Pad parasitics / Lead inductance / Access resistance / Extrinsic C / Delay / Intrinsic. Any spec key not named here lands in a trailing **"Other"** group so nothing is hidden. |
| `render_finetune_diagram(*, fname, topo_key, all_specs, calc_vals, render_illustration)` | Diagram-mode alternative inside each model's **"✏️ Fine-tune" override expander** (`_override_ui` in cheng/xu/kunyang). Left column: `render_illustration(preview_all_p_SI, highlight_key)` draws the topology schematic with the **last-edited** component ringed in red. Right column: every override value, grouped via `_FINETUNE_DIAGRAM_GROUPS`; each `number_input` writes the SAME `sim_{topo_key}_{key}_{fname}` key the List view uses (so both modes stay in lock-step and the caller's downstream `all_p` assembly is unchanged), and its `on_change` records the active component in `sim_dia_active_{topo_key}_{fname}` to drive the highlight. `all_specs` items are `(key, label, scale, unit, fmt, step)`; the schematic preview is rebuilt from session state each run. Models wire it in `_override_ui` behind an "Editor mode" `List`/`Diagram` radio (`sim_mode_{topo_key}_{fname}`); the model illustration's `highlight_key` param (cheng `_render_topology_illustration(all_p, topology, fname, highlight_key=…)`, xu/ky `(all_p, fname, highlight_key=…)`) draws the ring. |
| `render_interactive_param_groups(...)` | The big interactive expander: each normal parameter group (Cbex, Cbcx, intrinsic, τB, τC, …) is wrapped in one large card keyed `pfp_groupbox_{model_short}_{g_idx}_{fname}` (`box = st.container(border=False, key=…)`; its title + slider + plots render into `box` / `box.columns(...)`, with the existing per-parameter bordered sub-containers nested inside). The thick outline is drawn by a scoped `<style>` block (injected once at the top of the expander) targeting `div[class*="st-key-pfp_groupbox_"]` directly — Streamlit puts the `key` class on the inner `stVerticalBlock` (not the border wrapper), so the CSS styles that element and `border=False` avoids a double frame. Keep the key prefix and that CSS selector in sync. Special fit groups (`z_plots_group` / `fbi_fit_group` / `f1_fit_group`) keep the legacy heading-outside layout. Per-group: section heading, dependency info, "same range as previous" button, frequency-range slider, per-frequency line plots, per-param `number_input`, optional Cbex-sweep tool, optional `tau_total_fit_group` (multi-file reference fit; its trailing `---` is suppressed when the fit is hidden for a single file). Accepts `all_data` and `para_eff` kwargs to enable the tau-total fit section. Per-param inputs render extra quickset buttons sourced from elsewhere in the UI: **"v_c = …"** for τB / τC (from the tau_total_fit_group's v_c split), and **"Z-param = …"** for Rbe (from `rz12_Rbe_{fname}` when the Z-parameter method has been run and the current DUT is in the fit). |
| `_FRAGMENT` (module-level) | `st.fragment` (≥ 1.37) / `st.experimental_fragment` (1.33–1.36) / identity fallback. |
| `_make_sweep_values(min_val, max_val, step)` | Generate sweep values. |
| `_render_slider_preview(model_cls, all_p, S_raw, freq, z0, tuning_specs, fname, topo_key)` | Dispatcher inside the Visual Tuning expander. |
| `_slider_default_range(current_disp, key="", label="")` | Sane `(min, max, step)` for a slider, clamped into `tune_hard_limits(key, label)` (R/L/C floor at 0, alpha0 in `[0.95,0.99]`); zero current → `(0.0, 1.0, 0.01)`. |
| `_multi_metric_top_n(arr, per_metric=10)` | Take an `(N, n_cols)` residual table and return the union of top-`per_metric` rows by each metric. |
| `_render_live_slider_preview(...)` *@fragment* | Streamlit-rerun-per-drag preview with pre-baked static cache + `simulate_batch(B=1)`. |
| `_chunked_simulate_batch_to_host(...)` | Run `simulate_batch` in slabs so OOM doesn't bite on million-frame sweeps. On CPU (`xp is np`) the caller-supplied `chunk_size` is additionally capped against `helpers/mem_budget.py::ram_available_bytes()` (same per-row working-set model as the sweep budget below) so a large default chunk can't out-allocate the Streamlit Cloud container; GPU path unchanged. |
| `_render_plotly_slider_preview(...)` *@fragment* | Pre-computed Plotly slider — joint cartesian sweep, always embedded client-side (the 📡 server-cached mode + int16 quantization were removed — they duplicated Live tweak's UX). Freq points default to full fidelity (≤ 1001); default frames-per-axis shrinks with selection count (11 / 7 / 5 for ≤2 / 3 / ≥4 params) to keep the embedded payload under Streamlit's 200 MB message limit. |
| `render_visual_tuning_expander(...)` | 🎚️ Visual Tuning expander — wraps `_render_slider_preview`. Expander wrapped in `st.container(key="hbt_exp_tune_vis_" + topo_key)` — blue left-border tuning-cluster (`tools/ui_theme.py` action-color system). Inside, the "✅ Use these values" slider-commit button (`_render_slider_preview`) is separately wrapped in `st.container(key=f"hbt_amber_slcommit_{topo_key}")` — amber apply-into-form styling. |
| `render_tuning_expander(..., *, default_fit_keys=None)` | 🔧 Auto Tuning for Minimum Residuals — grid sweep + per-metric top-10 ranking + residual table. Expander wrapped in `st.container(key="hbt_exp_tune_auto_" + topo_key)` — blue left-border tuning-cluster. The "🏆 Use best values" button is separately wrapped in `st.container(key=f"hbt_amber_best_{topo_key}")` — amber apply-into-form styling. The CPU inner-block budget (`_run_one_sweep`'s `avail`/`budget`) comes from `helpers/mem_budget.py::ram_available_bytes()` (cgroup-aware — reads the Streamlit Cloud container limit, not just the host's psutil total) instead of a raw `psutil.virtual_memory().available` call; the free-RAM caption still reports GiB free. Sweep rows now seed from `informed_default_range` (physics-informed ranges, low-perf-aware via cached `_detect_low_perf_device`) and the Min/Step/Max widgets are hard-limited via `tune_hard_limits` (pre-clamped so stale out-of-range values can't raise); "↩️ Use default values" + "🏆 Use best values" + the Nelder-Mead box all re-seed from the same resolver. **Layout (top → bottom):** compute-backend badge → **🪜 Full Auto Tune** card (first; `Parameters to fit` multiselect defaulting to non-parasitic canonical keys / `default_fit_keys`, plus `Evaluate with CPU` / `⚡ Evaluate with CUDA` buttons — no target-residual input, it always runs to the step floor or ⏹ Stop) → **🔬 Semi-Auto Tune** toggle-collapsed section (toolbar `Use default values`/`Select all`/`De-select all`, per-param Sweep/Min/Step/Max rows, total-calc count, and the 🧮 Brute force / 🎯 Optimized / 🥇 Prioritized cards) → **📈 Parameter sensitivity** toggle-collapsed section → results block (`Evaluated in …`, best-residual line, 🏆 Use best values, top-K table, Excel download). The two inner sections are `st.toggle`-gated bordered containers, NOT expanders — Streamlit forbids nesting `st.expander`. `param_rows` is built from the keyed session state on every run (single `_row_entry` source of truth), so all drivers work even while Semi-Auto is collapsed. |
| `_run_progressive(*, use_cuda, fit_keys, target_pct)` (nested in `render_tuning_expander`) | 🪜 **Full Auto Tune** driver (progressive coarse → fine; dispatch passes `target_pct=0.0`). Global coarse scan over physics-informed ranges → pair-wise (Zbe/Zbc/…) refinement cycles with shrinking boxes (`_prog_next_bounds`) and log/linear axis grids (`_prog_axis_values`); every candidate memoised via `_prog_signature` so no combo is re-evaluated. **No-zero floor:** each *selected* param's effective lower limit is `max(hard_lo, step-floor)` (> 0), so the fit never proposes 0 — deselect a param in the multiselect to pin it at its current value (including 0). On CPU, `_chunk_max` (the per-`simulate_batch`-call chunk size) is derived from `helpers/mem_budget.py::ram_available_bytes()` rather than the fixed `_PROG_CHUNK_CPU` constant (`per_row_bytes = 16 * 4 * len(freq) * 12`, floored at 256 rows) — at high freq-point counts the fixed constant could allocate several GB in one call and get SIGKILLed by the Streamlit Cloud cgroup OOM-killer before any `except MemoryError` path ran; GPU keeps the fixed `_PROG_CHUNK_GPU`. Chunks survivors through `simulate_batch` (CUDA-OOM chunk-halving guard), scores with `_port_residuals_batch`, keeps a top-100 host table (`_multi_metric_top_n`), throttles UI to ~2 Hz with ⏹ Stop cancellation (progress text shows the worst per-port residual). **Escape phase:** stall detection (`_PROG_STALL_CYCLES` consecutive cycles with relative Total drop < `_PROG_STALL_REL`, or a fully-memoised cycle) while any port is above `_PROG_PORT_GOAL` (5 %) triggers one `_escape_pass`: every box-pinned param gets a wide log 1-D re-scan up to `max(informed hi, 100× current, 100× floor)` plus a 2-D scan with its first `_PROG_GROUPS` partner, and its bounds re-widened to `[pos_floor, wide_hi]`; a fruitless escape falls back to one global random re-scan (`budget // 4` samples over the re-widened boxes) before the normal termination paths may end the run — the "all boxes at floor" stop is gated on `worst_port ≤ goal or escape_spent`. Finish message: `st.success` when all ports ≤ goal, else `st.info` naming the worst port. Stops on target residual / step floor / zero-new-eval / ⏹ Stop; persists results + `tune_elapsed_*` and frees the GPU in `finally` (mirrors `_run_nelder_mead`'s except/finally). |
| `class SSMModelTemplate` | Mixin parent for SSM model classes. |
| `SSMModelTemplate.prebake_static_keys(cls, swept_keys)` | Truth-table query: given which params are sweeping, return the list of pre-bakeable sub-network names. |
| `SSMModelTemplate.build_static_cache(cls, all_p, freq, *, xp=None, swept_keys=())` | Build the `cache` dict matching `_sim_wrap_batch`'s lookup keys. |
| `SSMModelTemplate._build_intrinsic_static_cache(...)` | Default no-op — concrete classes override. |
| `SSMModelTemplate.simulate_vec(...)` | Default vectorised forward sim. |
| `SSMModelTemplate.simulate_batch(...)` | Default batched forward sim. |
| `SSMModelTemplate._cached_simulate_vec(...)` | Cached scalar→vector simulation with `params_hash` keying. |
| `SSMModelTemplate.render_results_table(cls, params)` | Default — concrete classes override. |
| `SSMModelTemplate.render_formula_trace / _render_results_trace / has_formula_trace` | Template hooks for the formula-trace expander. |
| `SSMModelTemplate.render_override_and_smith(cls, fname, S_raw, freq, z0, para_eff, extract_result, *, show_tuning=True, prefer_calc_vals=False, show_cache_banner=True)` | Default — extracts param dict + simulates + writes a modeled S2P bytes blob, then renders **two top-level expanders** below the measured-vs-modeled Smith chart: "🖼️ Topology illustration" (the schematic with live param values) and **"🍩 Smith Chart (Matplotlib)"** which contains the controls (right column) AND the chart (left column) side-by-side — matches the RF simulator layout. The measured-vs-modeled Smith chart exposes a `📥 modeled S2P` download next to its xlsx button. **`show_tuning=False`** (passed by the Extraction page — extraction is peeling-only) suppresses the Visual + Auto Tuning expanders; the **Simulation & Fitting** page calls it with `show_tuning=True` (default) for fit-mode (measured-file overlay + tuning). **`prefer_calc_vals=True`** (set by the Sim & Fitting page only right after a fresh extraction handoff) clears the `sim_{SHORT}_*_{fname}` widget keys + both sync hashes for this (model, fname) before the cache-restore step, so calc_vals wins over both stale (possibly GC'd) widget state and any cached fit; the cache-restore branch is skipped for this render (guarded by `not prefer_calc_vals`) but the banner + "📌 Use cache" button inside Fine-tune still work. **`show_cache_banner=False`** (set by `RF_simulator.py`'s fit-mode call) suppresses the "📌 Cached fit for..." text banner entirely — the fit page shows a compact `.hbt-chip-cache` pill in its header row instead; the "📌 Use cache" button inside Fine-tune keeps working regardless. A new **cache-apply request** block (`cache_apply_request_{SHORT}_{fname}` session key, popped+consumed before any sim widgets instantiate) lets the "📌 Use cache" button (added to `_override_ui`'s Fine-tune expander in cheng/xu/kunyang) reload the cached snapshot via `st.session_state[...] = True; st.rerun()` instead of a button embedded in the banner itself. `cache_ctx={"has_cache", "ts", "req_key"}` is passed into `cls._do_override_ui(fname, calc_vals, cache_ctx=...)` so the per-model Fine-tune expander can render that button. The cache-restore branch also re-applies on a **missing widget key** (`_widgets_missing`, not just `not applied_key`) — Streamlit GCs `sim_*` widget state for widgets that skip a script run (page switch), so re-entering a page must not let inputs recreate at 0. The banner slot lives in an unconditionally-rendered `st.container()` so the element tree above the "✏️ Fine-tune" expander never shifts (a shifting tree was resetting the expander's client-side open state). The auto-save gate additionally refuses to persist a state where fewer than half of `int_ext_keys` are nonzero (`_n_nonzero < max(2, len(int_ext_keys)//2)`) — guards against a GC-wiped near-zero widget set (or a "0️⃣ Reset all to 0" click) poisoning the cache. |

**Pre-bake truth table API** — `SSMModelTemplate` exposes `STATIC_SUBNETWORKS` (declarative dependency map, set per concrete model) and uses it to skip re-computing static sub-networks across slider ticks. Per-model `_build_intrinsic_static_cache` overrides live in `cheng.py` (ChengT + ChengPi), `xu.py` (XuT), and `kunyang.py` (KunYangHEMT).

### [`models/cheng.py`](../models/cheng.py) — Cheng (2022) T and π topologies

Module-level helpers:

| Function | Purpose |
|---|---|
| `_sweep_cbex_stds_cheng(Y_ex1, freq, cbex_SI_array, mask)` | For each candidate Cbex, rebuild Y_ex2 and return std(Cbcx_arr[mask]). Shared by ChengT and ChengPi. |
| `_step2_T(Y_ex1, freq, n_low)` | Cheng [Eq. 13, 22] — Cbex_T (T variant) and Cbcx. |
| `_step2_pi(Y_ex1, freq, n_low)` | Cheng [Eqs. 26–28] — Cbex_π and Cbcx. |
| `_step3_T(Y_ex2, freq, Cbcx, n_low)` | Cheng [Eqs. 16, 29–31] — T-topology intrinsic Rbi/Rbe/Cbe/Rbc/Cbc/α₀/τB/τC. |
| `_step3_pi(Y_ex2, freq, Cbcx, n_low)` | Zhang et al. — π-topology intrinsic Rbi/Rbe/Cbe/Cbc/Gm0/τ. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | Add extrinsic caps + pad/lead parasitics around an intrinsic-Y matrix (per-freq loop, scalar). |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | Vectorised forward sim. Works on numpy or cupy. |
| `_Y_int_T_vec(p, omega, xp)` | Vectorised T-topology intrinsic-Y matrix → (N,2,2). |
| `_Y_int_Pi_vec(p, omega, xp)` | Vectorised π-topology intrinsic-Y matrix → (N,2,2). |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | Batched forward sim over (param_combo × freq). |
| `_Y_int_T_batch(p, omega, B, N, xp, cache=None)` | Batched T-topology intrinsic-Y, returns 4 broadcastable planes. |
| `_Y_int_Pi_batch(p, omega, B, N, xp, cache=None)` | Batched π-topology intrinsic-Y, returns 4 broadcastable planes. |
| `_fmt_param(key, val_si)` | Format a parameter SI value for display on the topology illustration. |
| `_render_topology_illustration(all_p, topology, fname)` | Overlay live parameter values on the schematic template PNG. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS, cache_ctx=None)` | Render the fine-tune override expander (Pad / Extrinsic / Intrinsic) for one Cheng topology. Reseeds from `calc_vals` when the sync hash mismatches **or** any `sim_{tK}_{key}_{fname}` widget key is missing (Streamlit GC'd it after a page switch) — the missing-key check catches what the hash-only check couldn't: the hash's own session key survives GC while the widget keys it's supposed to guard don't. The "✏️ Fine-tune ..." expander is wrapped in `st.container(key=f"hbt_exp_edit_{tK}")` — amber left-border edit-cluster (`tools/ui_theme.py` action-color system). First element inside the expander is a 3-column button row: reset-to-interactive-values (`rst_sim_{tK}_{fname}`), "📌 Use cache" (only when `cache_ctx["has_cache"]` — wrapped in `st.container(key=f"hbt_amber_usecache_{tK}")` for the amber apply-action style; sets `st.session_state[cache_ctx["req_key"]] = True` and reruns so `render_override_and_smith`'s cache-apply-request block loads the cached snapshot before widgets instantiate), and "0️⃣ Reset all to 0" (`zero_sim_{tK}_{fname}` — zeroes every `sim_{tK}_{key}_{fname}` widget key across `all_specs`; does not touch the on-disk cache, and the auto-save "mostly-zero" guard in `render_override_and_smith` refuses to persist the resulting near-zero state). |

Shared helpers `_b1`, `_detect_B`, `_stack22`, `_try_download_inter`, `has_inter`, `_load_font` are imported from [`models/_shared.py`](../models/_shared.py).

`class ChengT(SSMModelTemplate, AbstractSSMModel)`: methods `extract`, `sweep_cbex`, `simulate`, `reextract`, `_results_rows`, `_render_results_trace`, `_do_override_ui`, `_render_topology`, `_build_intrinsic_static_cache`.

`class ChengPi(SSMModelTemplate, AbstractSSMModel)`: methods `extract`, `sweep_cbex`, `simulate`, `reextract`, `_results_rows`, `_do_override_ui`, `_render_topology`, `_build_intrinsic_static_cache`.

### [`models/xu.py`](../models/xu.py) — Xu's T (2014)

Module-level helpers:

| Function | Purpose |
|---|---|
| `_step2_T(Y_ex1, freq)` | Xu — Cbcx direct from Y_ex1 (no Cbex peel). Rbcx defaults to 285 kΩ. |
| `_step3_T(Y_ex1, freq, Cbcx, Rbcx, n_low)` | Xu — peel Ybcx then run the Cheng-style Step 3 intrinsic extraction. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | Scalar forward sim with Ybcx parallel network. |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | Vectorised forward sim. |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | Batched forward sim for tuning sweeps. |
| `_Y_int_T_vec(p, omega, xp)` | Vectorised T-topology intrinsic-Y (shared formula with Cheng-T). |
| `_Y_int_T_batch(p, omega, B, N, xp, cache=None)` | Batched intrinsic-Y planes. |
| `_aka(cheng_lbl, xu_lbl)` | Render combined "Cheng / Xu" symbol label. |
| `_fmt_param(key, val_si)` | Format a parameter SI value for display on the topology illustration. |
| `_render_topology_illustration(all_p, fname)` | Overlay live values on the Xu schematic template PNG. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS, cache_ctx=None)` | Fine-tune override expander for the Xu T topology. Same missing-widget-key reseed as `cheng.py::_override_ui`. Same `st.container(key=f"hbt_exp_edit_{tK}")` amber edit-cluster wrap. Same 3-column reset / "📌 Use cache" / "0️⃣ Reset all to 0" button row as `cheng.py::_override_ui`. |

`class XuModel(SSMModelTemplate, AbstractSSMModel)` (`SHORT="XuT"`): methods `extract`, `simulate`, `reextract`, `_results_rows`, `_render_results_trace`, `_do_override_ui`, `_render_topology`, `_build_intrinsic_static_cache`.

### [`models/kunyang.py`](../models/kunyang.py) — Kun-Yang HEMT (pi-model, forward sim only)

Forward-simulation-only model — no extraction is performed. Built inside → out:

1. Intrinsic 3-component pi: series Cgs/Ri (gate-source shunt), series Cgd/Rgd (gate-drain), parallel Rds∥Cds + transconductance gm = Gm0·exp(−jωτ) (drain-source).
2. Z_ser wrap with gate/drain/source lead L+R.
3. Kun-Yang custom substrate pad in parallel: Cgsp series Rsub1, Cdsp series Rsub2, Cgdp (port-1 to port-2).
4. Standard open-dummy pad (Cpbe, Cpce, Cpbc) on top → S.

Module-level helpers:

| Function | Purpose |
|---|---|
| `_render_topology_illustration(all_p, fname)` | Render Kun-Yang topology illustration (placeholder — no schematic shipped). |
| `_Y_int_KY_vec(p, omega, xp)` | Vectorised intrinsic pi-model Y matrix. |
| `_Y_int_KY_batch(p, omega, B, N, xp, cache=None)` | Batched intrinsic Y → 4 (B, N) planes; cache-aware. |
| `_Y_kypad_vec(p, omega, xp)` | Vectorised Kun-Yang substrate pad Y. |
| `_Y_kypad_batch(p, omega, B, N, xp, cache=None)` | Batched Kun-Yang substrate pad as 4 (B, N) planes. |
| `_sim_wrap(Y_int_fn, p, freq, z0)` | Per-frequency scalar forward sim. |
| `_sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp)` | Vectorised forward sim. |
| `_sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None)` | Batched (param_combo × freq) forward sim. |
| `_override_ui(fname, tK, calc_vals, int_specs, label, ext_specs, cache_ctx=None)` | Fine-tune override UI. Same missing-widget-key reseed as `cheng.py::_override_ui`. Same `st.container(key=f"hbt_exp_edit_{tK}")` amber edit-cluster wrap. Same 3-column reset / "📌 Use cache" / "0️⃣ Reset all to 0" button row as `cheng.py::_override_ui` (reset button label differs: "to default values"). |

`class KunYangHEMT(SSMModelTemplate, AbstractSSMModel)` (`SHORT="KY"`): methods `extract`, `simulate`, `render_step_formulas`, `_results_rows`, `_render_results_trace`, `_do_override_ui`, `_render_topology`, `_build_intrinsic_static_cache`, `get_s2p_header_params`.

### [`models/degachi.py`](../models/degachi.py) — Degachi & Ghannouchi (2008) augmented π *(currently disabled in the registry — uncomment in `models/__init__.py` to re-enable)*

| Function | Purpose |
|---|---|
| `_h_Tbi / _h_Tbe / _h_ratios / _h_R_RT / _h_solve_pf / _h_solve_sc / _h_derived_pf / _h_derived_sc / _h_Rcx / _h_Gm0_tau / _h_Ccx` | Per-equation helpers following the 2008 IEEE TED paper. |
| `_extract(Y_ex1, freq, n_low, n_fit=None, n_fit_f1=None)` | Run all helper steps end-to-end → `(params, arrays)`. |
| `_simulate(p, freq, z0=50.0)` | 5-layer inside-out scalar forward sim. |
| `_simulate_vec(p, freq, z0=50.0, xp=None)` | Vectorised forward sim. |
| `_b1, _detect_B` | Imported from `models/_shared.py` (which re-exports from `helpers/_array_utils.py`) — formerly duplicated here. |
| `_simulate_batch(p, freq, z0=50.0, xp=None, cache=None)` | Batched forward sim for tuning sweeps. |

`class Degachi(AbstractSSMModel)` (`SHORT="D"`): cascade re-extraction across 9 PARAM_GROUPS.

### [`models/svg_topology.py`](../models/svg_topology.py) — "non-zero only" live SVG schematics

Toggle-ON alternative to each model's static PNG template: builds the matching
built-in custom-model preset, prunes every R/L/C whose current value is zero /
absent / non-finite, and renders the survivors through
`custom_model/schematic.py`. Also hosts the two hand-drawn pad-dummy
schematics (no preset exists for them — they are a probe-pad network, not a
device topology).

| Function | Purpose |
|---|---|
| `_present(v)` / `_prune_network(net, all_p, values)` / `_prune_branches(branches, all_p, values)` | Non-zero test; drop dead Elements/branches and record survivors' values by `Element.id`. |
| `build_pruned_model(short, all_p)` | Streamlit-free: preset for model `short` stripped to its currently non-zero components → `(model, values)` for `render_schematic`. |
| `render_svg_topology(short, all_p, fname)` | Streamlit wrapper — pruned SVG + iframe + PNG download/copy buttons. Called from `base_ui.render_override_and_smith`'s "🖼️ Topology illustration" expander when the model sets `_SVG_TOPOLOGY` and the user flips the "Simplified schematic" toggle. |
| `_pad_svg(caps, inds=None)` / `pad_open_svg(caps)` / `pad_short_svg(caps, inds)` | Hand-drawn Open (3-cap π: Cpbe/Cpce/Cpbc) and Short (same caps + Lb/Lc/Le center-node T) pad-dummy SVGs, with each component's live value printed next to it (SI-unit inputs; `_fmt` prints blank for zero/absent). |
| `render_pad_topology(kind, params, fname, container=None)` | Streamlit wrapper for the two pad SVGs. **Single call site:** the RF simulator's "Open and Short Pad" model (`tools/RF_simulator.py`, "🖼️ Open/short topology" expander). Deliberately *not* rendered under the SSM models or the custom model — their own topology illustration already draws the pad parasitics in place. |

---

## Custom model builder (`custom_model/*.py`)

**Bilingual UI:** every user-facing string in `ui_build.py` / `ui_use.py` / `ui_fit.py` / `__init__.py` is wrapped in `tr(en, zh)` from [`custom_model/_i18n.py`](../custom_model/_i18n.py) — an inline EN/中文 helper that defers the active language to the portal's `tools.i18n.is_zh()` (the 🌐 Language radio). Add new builder strings as `tr("English", "中文")`, not bare literals. Grouped value-input section titles (English, from `CustomModel.grouped_value_specs()`) are display-translated via `ui_fit._GROUP_TITLE_ZH`.


User-built ("custom") small-signal models. Two modes (Simulate-and-Fit was
formerly two separate Load/Fit modes — now **combined**): a visual **Build**
workflow builds a topology inside→outward; a **Simulate & fit** workflow
(`ui_use.py::render_use_ui(measured=None)`) loads a topology, takes a value per
component, and forward-simulates — and when `render_custom_section` is handed a
**measured device** it overlays it on the *same* Smith/Bode (markers vs dashed),
shows the residual, and adds the grid-sweep **Auto Tuning** expander; with no
device it's a plain forward simulator (user-set frequency).  In
`tools/RF_simulator.py` it's a **"🧩 Custom model" entry in the Model radio** →
`render_custom_section(measured)` (`custom_model/__init__.py`), a **Simulate &
fit / Build model radio**.  **Build** = the wizard, ending in **Download .json**
/ **Send to Simulate/Fit** (no save-to-library); it also takes a **"Modify an
existing model"** upload that loads the whole saved setup (device, π/T,
junctions, all sections + names) into the editor (named `…_modified`).
`CustomModel.from_dict` migrates schema-v1 files (`_migrate_v1`) so old saved
models load faithfully.  The combined view reuses `ui_fit`'s `_load_model`
(shared `cmf_model` source), `_value_inputs` (`sim_custom_{cid}_{fname}` keys),
`_clear_tuning_state`, and `_make_adapter` so model + tuning state are shared
across the with/without-device cases.

The combined **Simulate & fit** view (`ui_use.py::render_use_ui`) reuses the
**same result UI as the built-in models** whenever a device is present:
`render_smith_with_ftfmax` (sticky Total/per-trace residual `st.metric` row above
the Smith + fT/fmax card, download/copy below), the topology illustration (PNG download **+
copy-image** via `schematic.copy_image_button`), the two-column matplotlib Smith
chart (measured + modeled), and **both** shared tuning expanders —
`render_visual_tuning_expander` (🎯 Live tweak / ⚡ Wide sweep) and
`render_tuning_expander` grid-sweep (Brute/Optimized/Prioritized, per-param
Min/Step/Max, "Use best values").  With no device it shows a model-only Smith
(`_smith_fig`) + forward Bode and Visual Tuning only.  `_make_adapter` wraps the
`CustomModel` in an AbstractSSMModel-shaped class (`simulate` / `simulate_vec` /
plan-backed `simulate_batch`) so the tuning works unchanged; the value inputs use
`sim_custom_{cid}_{fname}` keys (floored at 0) so applied sweeps flow back.
`_clear_tuning_state` drops the previous model's stale tuning/value state on
every model change (token-tracked) so added components can't KeyError the
sensitivity table.

Device-aware: `CustomModel.device` ∈ {Bipolar (B/C/E), Unipolar (G/D/S)} sets
port long-names (`port_label`) and the default controlled-source label
(`gm·Vbe`/`Ids` for π, `α·Ie`/`α·Is` for T).

The intrinsic core is **fully editable**: four junction `Network`s —
`intrinsic_base` (Rbi, BB→BI), `intrinsic_be` (BI→EI), `intrinsic_bc` (BI→CI),
`intrinsic_ce` (CI→EI) — each defaulting to Cheng's and freely extended with
series/parallel R-L-C (or trashed). The solver computes each junction's
equivalent admittance/impedance from its Network, so a Cheng-default junction
reproduces the old analytic to machine epsilon (verified ~4e-16 over 200 random
sets, π and T) while user edits are honoured. The generic nodal solver still
reproduces `ChengPi`/`ChengT`/`XuModel`. Mapping: Rbi = `intrinsic_base`;
extrinsic caps = shunts referenced to the intrinsic emitter (Cheng Cbex p1-gnd
+ Cbcx p1-p2; Xu Cbcx∥Rbcx p1-p2); leads = access R+L; pads = parasitic caps.

Intrinsic cores (`core.py::_intrinsic_Y(itype, Ybe, Ybc, Yce, src, omega)`):
**π** = hybrid-π `[[Ybe+Ybc,−Ybc],[gm−Ybc,Ybc+Yce]]`, gm = Gm0·e^(−jωτ);
**T** = current-source HBT/HEMT T-model (Zbe=1/Ybe, Zbc=1/Ybc +
α₀·e^(−jωτ_C)/(1+jωτ_B)). Both stamped as a 3-terminal common-emitter 2-port
between BI/CI/EI; the controlled source is scalar (params gm/τ or α₀/τ_B/τ_C).

### [`custom_model/core.py`](../custom_model/core.py) — data model + solver

| Function / class | Purpose |
|---|---|
| `Element` / `Network` / `ShuntBranch` | Leaf R/L/C; series-of-parallel branch; placed shunt/bridge cap branch. |
| `CustomModel` | Whole topology: `device`, `intrinsic_type`, four editable intrinsic junction Networks (`intrinsic_base/be/bc/ce`), `source_name`, per-section branches, **`ie_after_cbex`** (T-core flag: sense the α·Ie emitter current *after* the extrinsic Cbex tap — Ie includes the Cbex displacement current; no effect when Cbex absent or core is π). Helpers: `source_keys`, `source_disp`, `terminals`, `port_label`, `intrinsic_junctions`, `ensure_intrinsic`. `to_dict`/`from_dict` (schema v2)/`all_value_specs`. |
| `_default_source_name(itype, device)` | Default controlled-source label per topology+device. |
| `models_dir` / `save_model` / `load_model` / `list_saved_models` | JSON persistence under repo-root `custom_models/`. |
| `BUILTIN_PRESETS` / `builtin_custom_model(label)` | Built-in topology presets so the builder's "modify" flow can start from a registered model. `BUILTIN_PRESETS` = `{human label → factory}` (Cheng T `_cheng_t`, Cheng π `_cheng_pi`, Xu T `_xu_t`, Kun-Yang `_kunyang`); `builtin_custom_model` returns a fresh `CustomModel` for a label. Structures mirror each model's `_sim_wrap_vec`/forward-sim netlist exactly (intrinsic junctions, extrinsic/parasitic caps, access legs, KY source-delay branch + substrate pads) so the loaded copy reproduces the same topology. Helpers `_net(*groups)` / `_shunt(place, *groups)` / `_hbt_pads()` / `_HBT_ACCESS` build the Networks. |
| `SimPlan` (dataclass) | Value-free compiled topology: `n` nodes, `branches` `(ia,ib,series,groups)`, `twoport` `(ia,ib,iref,be,bc,ce)`, `itype`, `source_keys`, `value_keys`, **`alpha_cbex`** (`None` or `(i_bb,i_ci,i_ei,[groups,…])` for the "Ie after Cbex" extra α-controlled source). Consumed by the vectorised evaluator **and** the Rust `sim_custom_batch` (plans with `alpha_cbex` set bypass Rust → NumPy/CuPy path). |
| `compile_plan(model)` | Resolve topology to a `SimPlan` **once** (structural node merges via union-find — a structurally-empty series branch is a wire). Raises on degenerate (port-to-GND / merged ports). |
| `simulate_custom_model_batch(plan, freq, values, z0, xp=np, max_batch_elems)` | Vectorised forward sim over a parameter batch → `S[B,N,2,2]`. Scatter-stamps a `(B,N,n,n)` Y, batched `xp.linalg.solve` Kron reduction, `y_to_s_vec`. `xp=cupy` ⇒ runs on GPU; chunks the batch axis to bound memory. Raises `ValueError` when two value arrays disagree on batch size (was: silently mis-sliced). |
| `simulate_custom_model(model, freq, values, z0)` | Thin scalar wrapper: `compile_plan` → `simulate_custom_model_batch` with scalar values → `S[N,2,2]` (forward-sim / "Use" mode, back-compat). |
| `_elem_adm_b` / `_group_adm_b` / `_branch_series_b` / `_branch_shunt_b` / `_junction_adm_b` | Batched `xp`-aware admittance kernels. Series: Σ group impedances (zero group = short; all-zero ⇒ near-short `_SHORT_Y`). Shunt/junction: any absent group ⇒ open. A present inductor at a `jw = 0` (DC) sweep point reads as a short (`_SHORT_Y`), not a spurious 1 S — mirrored in the Rust `_elem_adm`. |
| `_intrinsic_Y_b(itype, Ybe, Ybc, Yce, src, jw, xp)` | Batched common-emitter 2-port from junction admittances + scalar source params (π / T α-source). |
| `_simulate_plan_core(plan, jw, vals, z0, xp, B)` / `_stamp(...)` | One-chunk evaluator: build `(B,N,n,n)` Y, embed indefinite 2-port, Kron-reduce, Y→S. When `plan.alpha_cbex` is set, also stamps the extra α·Y_cbex·(V_bb−V_ei) collector source ("Ie after Cbex"). |

### [`custom_model/schematic.py`](../custom_model/schematic.py) — live SVG

| Function | Purpose |
|---|---|
| `build_schematic(model)` | Render the **structure once** (symbols + names + junction dots), recording value slots in `_SVG.vanchors`. Cache this; overlay numbers cheaply with `s.render(s.value_layer(values, model))` so a parameter change does *not* re-render the whole drawing. |
| `render_schematic(model, values=None)` | One-shot convenience: `build_schematic` + value overlay. |
| `_draw(s, model)` | Draws the full two-port: device-aware P1/P2 port long-names, signal path, the data-driven intrinsic sub-circuit, shunts/bridges, and the emitter/source leg (`model.emitter` extras → access R/L → GND). Dots only at ≥3-wire junctions; component values go in a deferred layer. For **any T-core** it draws an **"Ie" current-sense arrow** (`_SVG.current_arrow`, overlaid on a dedicated clear stub): before-Cbex → on the b–e junction's lengthened bottom lead (the intrinsic is grown by `_IE_STUB`); after-Cbex **or no Cbex** → on a clear `_IE_STUB` stub in the emitter leg above Re (placement keyed off `model.ie_after_cbex`, which only matters when an extrinsic Cbex P1↔GND is present). `_last_group_bottom(net, y_top, y_bot)` locates the clear lead below a vertical junction's glyphs. |
| `_draw_intrinsic(s, …, model, values)` | Data-driven intrinsic from the four junction Networks: base spreading (horizontal series), `intrinsic_be` (vertical), `intrinsic_bc` (horizontal; T source ← here), `intrinsic_ce` (vertical; π source ↓ here). |
| `_draw_junction_v` / `_draw_junction_h` | Stack a junction Network's series groups vertically/horizontally, appending the controlled source to the last group as a parallel branch. |
| `intrinsic_thumbnail(model, selected)` | Abstracted intrinsic view: each editable part (base / B–E / B–C / C–E / source) as a labelled box around the base node, `selected` highlighted — pairs with the build-UI chip selector. |
| `section_thumbnail(model, section, selected)` | Cumulative abstracted view for an outer section (extrinsic / delay / access / parasitic): inner circuit collapsed into one labelled **core block** + that section's components, `selected` highlighted, absent parts dashed. The **extrinsic** view of a T-core also draws the **"Ie" arrow** on the emitter leg (high = before-Cbex, low = after-Cbex / no Cbex) so the build page's before/after radio gives live feedback. |
| `svg_png_buttons(svg, filename, …)` | Renders **Download PNG** + **Copy image** buttons in one iframe that rasterise the SVG to PNG **in the browser** (HTML `<canvas>`, single cached raster) — no server-side `rsvg-convert`/`cairosvg` and no native packages, so PNG export works identically on Streamlit Cloud and any local machine. Powers the schematic PNG export in the build/use pages. |
| `svg_to_png(svg, zoom=2)` | *(legacy server-side path, no longer wired into the UI)* Rasterise an SVG string to PNG bytes via `rsvg-convert` (15 s timeout) → `cairosvg` fallback → `None`. |
| `copy_image_button(png, …)` | A clipboard "copy image" button (mirrors `copy_button` styling) that writes an already-rasterised PNG via the `ClipboardItem` API. Superseded in the schematic UI by `svg_png_buttons`. |

`CustomModel.emitter` (a `Network`) holds source/emitter-leg "delay" extras drawn between the intrinsic emitter and the emitter access R/L — e.g. the Kun-Yang R_delay∥C_delay above the Rs node.

### [`custom_model/__init__.py`](../custom_model/__init__.py)

| Function | Purpose |
|---|---|
| `render_custom_section()` | Load / Build radio (RF Forward Simulator). On "Send to Load/Fit" from build it consumes `cm_nav_to_loadfit` (before the radio) to switch to Load. |

### [`custom_model/ui_build.py`](../custom_model/ui_build.py) — "Build" UI

| Function | Purpose |
|---|---|
| `render_build_ui()` | Progressive wizard: §1 device + editable intrinsic core, §2 extrinsic caps, §3 delay/port extras, §4 access R/L, §5 parasitic caps. §1 uses `intrinsic_thumbnail` + an `st.segmented_control` chip selector to edit **one part at a time** (`cmb_isel`); outer sections show `section_thumbnail`. Top: **🔄 Start over** + **📂 Modify an existing model** — either **start from a built-in model** (`BUILTIN_PRESETS` selectbox + Load → Cheng/Xu/Kun-Yang) **or upload a saved `.json`**; both load fully with all sections revealed and named `…_modified`. Live schematic has PNG / 📋-copy / SVG. Footer: **Download .json** + **Send to Load/Fit** (loads `cmu_model`+`cmf_model`, sets nav + pending-download flags, reruns). |
| `_install_for_modify(loaded, *, keep=())` | Shared installer for both modify paths: wipe `cmb_*` widget state (except `keep`), install `loaded` as the working model with stage=4 (all sections revealed), seed access names, drive the device/topology radios explicitly. |
| `_relabel_for_device(model, old, new)` | On a device switch, auto-rename default-named components (Cbc→Cgd, Rb→Rg, …); `_DEV_NAMES` holds per-device defaults. Name inputs refresh because their keys carry `_ver()` and the handler calls `_bump_namever()`. |
| `_auto_download_json` / `fire_pending_download` | Trigger a browser download via a hidden data-URI anchor; `fire_pending_download` (called atop the Load + Fit views) downloads a model "sent" from build after the navigation rerun. |
| `_network_editor` / `_shunt_list_editor` | Reusable series-of-parallel and shunt-branch editors (🗑️ trash; name keys carry `_ver()`; "➕ Cap" pre-adds a C). |

### [`custom_model/ui_use.py`](../custom_model/ui_use.py) — combined Simulate & Fit UI

| Function | Purpose |
|---|---|
| `render_use_ui(measured=None)` | **Unified** workbench (merges the old Load/simulate + Fit modes). Loads via `ui_fit._load_model` (shared `cmf_model`) → `ui_fit._value_inputs` (keys `sim_custom_{cid}_{fname}`) → forward-simulate. **With a device** (`measured` present): freq locked to the device grid; `render_smith_with_ftfmax` overlay + residual + fT/fmax, matplotlib Smith (measured+modeled), and Visual + Auto Tuning (ref = `S_meas`). **Without**: user-set freq, model-only `_smith_fig` + forward Bode, Visual Tuning only (ref = current sim). `fname` = device label when fitting else `"forward"`; clears stale tuning state on model change via `ui_fit._clear_tuning_state`. |
| `_smith_fig(S, freq, title, scales=None)` | Plotly Smith for the model-only (no-device) view; `scales` applies the per-trace × multiplier. |

### [`custom_model/ui_fit.py`](../custom_model/ui_fit.py) — Fit helpers + model adapter

| Function | Purpose |
|---|---|
| `_load_model(fname)` / `_value_inputs(model, fname)` / `_val_key` | `.json` uploader → `cmf_model`; value-input grid keyed `sim_custom_{cid}_{fname}` (floored at 0; `_FIT_SPEC` start defaults). Consumed by the combined `render_use_ui`. |
| `install_fit_model(model)` | Set `cmf_model` + bump `cmf_model_token` (used by the uploader and build's "Send to Simulate/Fit"). |
| `_make_adapter(model)` | Wrap a `CustomModel` as an AbstractSSMModel-shaped class so `render_tuning_expander` runs unchanged. Compiles a `SimPlan` once (`cls._plan`); implements `simulate`, `simulate_vec` (→`(N,2,2)`), and `simulate_batch` (→`inner+(N,2,2)`). CPU (`xp` numpy) routes through the Rust `sim_custom_batch` kernel (silent NumPy fallback); CUDA (`xp` cupy) runs the vectorised evaluator on the GPU. |
| `_clear_tuning_state(fname)` | Drop the shared tuning expander's cached results + per-param sweep widgets + stale `sim_*` value inputs for this DUT (exact key boundaries). |

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

# hbt-tools — repository function & functionality index

Single source of truth for every Python file **outside** `tools/SSM/`.
Skim this file to find which module owns a function or feature before
grep'ing the codebase.

The index deliberately carries **no line numbers** (they drift as files
evolve) — grep for the function name to jump to its definition.

> **The SSM tree has its own, more detailed index:**
> [`tools/SSM/helpers/INDEX.md`](tools/SSM/helpers/INDEX.md) — covers
> `tools/SSM/` (orchestration, `helpers/`, `models/`, `custom_model/`).
> Read that one for anything SSM-related.
>
> **For the big picture** (boot flow, page anatomy, data flow, layers,
> "task → file" lookup) read [`ARCHITECTURE.md`](ARCHITECTURE.md) first.

**Maintenance rule:** whenever a function is added, removed, renamed, or
its behaviour meaningfully changes, update the matching row here (or in
the SSM index) in the same change.

---

## Repository map

| Area | What lives there |
|---|---|
| repo root | Portal entry point (`IOED_Tool_Web.py`) + bootstrap launchers (`LAUNCH_Tool.py`, `launch_ebl_calculator.py`) |
| `rust_things/` | Rust-kernel build + status scripts |
| `tools/` | One `.py` per portal page (DC, RF, EBL tools) + shared portal infrastructure (`i18n`, `home`, `diagrams`) |
| `tools/SSM/` | HBT small-signal-model extraction engine + UI (own index, see above) |
| `tools/SSM/rust_kernels/` | Rust acceleration crate, prebuilt binaries, benchmark |
| `.streamlit/`, `.hbttools/` | Streamlit config / auto-created local venv (not source) |

---

## AI-agent fitting API (`tools/SSM/agent_api.py`)

**Start here if you are an AI agent or script that needs to fit measured
S-parameter data to an SSM model programmatically.** This module is a
headless (no Streamlit UI) entry point into the same extraction/fitting
engine the app uses — load a `.s2p`/`.csv`, pick a built-in model (or build a
custom one), and fit it, all as plain function calls returning dicts / numpy
arrays in SI units. Importing it transitively imports Streamlit (works fine
without a `ScriptRunContext`), but the module itself never calls `st.*`.

**Header-driven de-embedding awareness:** `.s2p` files this app writes carry
leading `!` comment lines stating what was done to the data — e.g.
[`s2p/deemb_preext_vce3.5_ib280u.s2p`](s2p/deemb_preext_vce3.5_ib280u.s2p)'s
header lists `Cpbe/Cpce/Cpbc` (pad caps) and `Lb/Lc/Le` (lead inductances)
with `Rb=Rc=Re=0.0`, meaning the pad caps + lead inductances have **already
been removed** from that S-parameter data (only the access resistances were
left in). `load_data()` parses that header into `meta` (`deembedded`,
`removed_params`, `status`), and `fit()` reads it to **automatically freeze
Cpbe/Cpce/Cpbc/Lb/Lc/Le at 0** during the fit — the key convenience this
gives an agent: point it at a de-embedded file and it fits the right
parameter subset without being told which ones to skip. A file with no such
header is treated as raw/measured data — every parasitic should be fitted
(or de-embedded first through the Streamlit app).

**Compute backend — CUDA > Rust > NumPy:** `simulate()`/`fit()` take
`backend="auto"` (default), which prefers a CUDA GPU (cupy), then the
project's Rust kernels (`helpers/rust_kernels.py`), then falls back to
NumPy; pass `"cuda"`/`"rust"`/`"numpy"` to force one (gracefully falling
back with a note when unavailable). `fit()`'s result dict includes
`backend` (and `backend_note` on fallback). `python tools/SSM/agent_api.py
backend` reports what's available and what `auto` resolves to on this
machine.

| Function | Purpose |
|---|---|
| `load_data(path)` | Parse a `.s2p`/`.csv` → `{freq, S, z0, header_lines, meta}`; `meta` carries the de-embedding interpretation above. |
| `list_models()` | Built-in model SHORT → full name (from `tools.SSM.models.REGISTRY`), plus a note on custom-model support. |
| `param_specs(model)` | Every parameter `model` accepts, as `(key, label, si_scale, unit)` — builtin SHORT or a custom model. |
| `simulate(model, params, freq, z0=50.0, backend="auto")` | Forward-simulate `S[N,2,2]` from an SI-unit `params` dict (missing keys fall back to physics-informed defaults). |
| `residuals(S_meas, S_model)` | `{Total, S11, S12, S21, S22}` residual in % — the exact metric the Streamlit UI shows. |
| `fit(data, model, initial=None, fit_keys=None, fixed=None, bounds=None, method="auto", maxiter=400, backend="auto")` | Fit `model` to `data` (from `load_data()`) → `{params, residuals, success, n_evals, message, backend}`. Auto-freezes de-embedded parasitics per the header (see above) unless `fit_keys`/`fixed` are given explicitly. |
| `backend_status()` | Snapshot of CUDA/Rust availability + what `"auto"` resolves to — backs the `backend` CLI subcommand. |
| `build_custom_model(base="Cheng T", modifications=None, name=None)` | Start a `CustomModel` from a built-in topology (`custom_model.core.builtin_custom_model`) and apply edits. |
| `add_series_element` / `add_parallel_element` / `add_shunt_branch` / `add_parallel_to_shunt` | Thin wrappers to add R/L/C components to a `CustomModel`'s junction/section Networks or shunt branches — e.g. a Cce cap collector↔emitter, or Rbcx parallel to Cbcx. |
| `save_custom_model(model, path)` | Serialise a `CustomModel` to an explicit JSON path. `fit()`/`simulate()` also accept the `CustomModel` object (or dict) directly — no save/reload round-trip required. |

Python usage:

```python
from tools.SSM.agent_api import load_data, fit

data = load_data("s2p/deemb_preext_vce3.5_ib280u.s2p")
result = fit(data, model="T", maxiter=3000)   # Cheng T-topology
print(result["residuals"]["Total"], result["params"])
```

CLI usage (run from the repo root, or pass absolute paths — the CLI
bootstraps `sys.path` to the repo root itself either way):

```bash
python tools/SSM/agent_api.py inspect s2p/deemb_preext_vce3.5_ib280u.s2p
python tools/SSM/agent_api.py backend
python tools/SSM/agent_api.py fit s2p/deemb_preext_vce3.5_ib280u.s2p --model T --maxiter 3000 --out result.json
```

---

## Entry points & launchers

### `IOED_Tool_Web.py` — Streamlit portal entry point

Declares the single `st.set_page_config` for the whole app, gates access
behind a password (skipped when `HBT_LOCAL_LAUNCH=1`, set by
`LAUNCH_Tool.py`), renders the top-right 🌐 language toggle, then builds
the grouped sidebar navigation from the `tools/i18n.py` registry
(`st.navigation` + `st.Page`). Lives at the repo root, so all
`st.Page` / `st.page_link` / `st.switch_page` paths carry a `tools/`
prefix.

| Function | Purpose |
|---|---|
| `_inject_button_css()` | Global CSS: light-gray fill on all secondary buttons (st.button / download / form submit / popover / uploader Browse) so they read as buttons. Primary buttons and segmented chips excluded. Kept in sync with the inline copies in `tools/ebeam_calculator.py` (standalone) and the iframe copy-button in `helpers/chart_export.py`. |
| `check_password()` | Password gate; correct password from `st.secrets["APP_PASSWORD"]` (fallback `"IOED"` for local testing). |
| `_page(tool_key, *, default=False)` | Build an `st.Page` for a tool key, pulling path/title/icon from the i18n registry so sidebar and in-page titles can't drift. |

### `LAUNCH_Tool.py` — bootstrap launcher for the whole portal

Creates (or reuses) a local venv at `.hbttools/`, auto-updates the
checkout (git pull, or GitHub zip overlay for non-git installs), installs
missing packages (incl. the right CuPy wheel for the detected CUDA), then
launches Streamlit with `HBT_LOCAL_LAUNCH=1`.

| Function | Purpose |
|---|---|
| `_run_silent(cmd)` | Run a command, return stdout+stderr as a string, `''` on failure. |
| `detect_cuda_major()` | CUDA major version (12, 13, …) or `None`. |
| `cupy_pip_name(cuda_major)` | Correct CuPy wheel name for the CUDA major. |
| `_autoupdate_disabled()` | Honour the auto-update kill switch (env var). |
| `_git(args, timeout=30)` | Run `git -C ROOT <args>` → `(returncode, output)`. |
| `_update_git_checkout()` | Fast-forward the git checkout to the canonical branch. |
| `_http_get(url, timeout, accept)` | Minimal stdlib HTTP GET. |
| `_remote_head_sha()` | Latest canonical-branch commit SHA via GitHub REST. |
| `_stored_sha()` | SHA recorded by the last zip-overlay update. |
| `_overlay_tree(src, dst)` | Copy files from `src` over `dst` (add/replace only, never delete). |
| `_purge_project_pycache()` | Drop project `__pycache__` so overlaid `.py` files aren't shadowed. |
| `_update_zip_install()` | Zip-download update path for non-git installs. |
| `auto_update()` | Best-effort update to the latest canonical commit (git or zip). |
| `run(cmd, **kwargs)` | Run a command, streaming output live. |
| `pip_works()` / `recreate_venv()` / `create_venv()` | Venv health check / rebuild / create. |
| `pip_install(packages)` | Install pip packages into the venv. |
| `is_importable(name)` / `check_missing(packages)` | Import probe inside the venv / list of not-yet-importable packages. |
| `_pause_if_interactive(msg)` | Block on Enter only when stdin is a TTY (double-click launches). |
| `main()` | Orchestrate: update → venv → deps → `streamlit run IOED_Tool_Web.py`. |

### `launch_ebl_calculator.py` — standalone launcher for the EBL calculator

Same bootstrap pattern as `LAUNCH_Tool.py` but minimal: venv at
`.ebl_venv/`, only the packages `tools/ebeam_calculator.py` needs, then
`streamlit run` on just that page (no portal, no password).

| Function | Purpose |
|---|---|
| `_find_app_file()` | Locate `ebeam_calculator.py` beside the launcher, else under `tools/`. |
| `_autoupdate_disabled()` / `auto_update()` | Same best-effort update as the main launcher. |
| `run` / `pip_works` / `_create_venv(fresh=False)` / `pip_install` / `is_importable` / `check_missing` / `_pause_if_interactive` / `main` | Same roles as their `LAUNCH_Tool.py` counterparts. |

### `rust_things/build_rust_kernels.py` — one-shot Rust → binary build

Builds the SSM Rust extension with maturin (in a dedicated
`.hbttools_build` venv) and copies the compiled `.pyd`/`.so`/`.dylib`
into `tools/SSM/rust_kernels/bin/<arch>/` so end users never need a Rust
toolchain. Run once per OS.

| Function | Purpose |
|---|---|
| `_have(cmd)` | True iff `cmd --version` works. |
| `_arch_tag()` | Platform → bin subfolder tag (must match `helpers/rust_kernels.py::_arch_tag`). |
| `_run(cmd)` | Run a command, streaming, raise on non-zero. |
| `_ensure_build_venv()` / `_ensure_maturin(py)` | Create build venv / install maturin into it. |
| `_clean_wheel_dir()` / `_maturin_build(py)` / `_extract_binary(wheel, dest)` | Clean → build wheel → pull the compiled binary out of it. |
| `main()` | End-to-end build + install into `bin/<arch>/`. |

### `rust_things/check_rust_status.py` — Rust acceleration status CLI

`python rust_things/check_rust_status.py` — reports whether the Rust extension
loads, which binary is used, and the import error if not
(`_arch_tag()`, `main()`).

---

## Portal infrastructure (`tools/`)

### `tools/home.py` — landing page

Bilingual orientation screen: intro + one clickable `st.page_link` card
per tool, grouped by measurement domain. No functions — straight-line
Streamlit script driven by the i18n registry.

### `tools/i18n.py` — bilingual (EN / 中文) registry

Single source of truth for tool names + icons + descriptions + feature
bullets and portal-level UI strings. The sidebar nav and each page's
`st.title` both read from here. Key data: `LANGS`, `TOOLS`
(`{tool_key: {path, icon, group, name…}}`), `GROUP_ORDER`.

| Function | Purpose |
|---|---|
| `get_lang()` / `is_zh()` | Active UI language (from the 🌐 toggle), default English / predicate. |
| `_pick(d)` | Select the `en`/`zh` field of a bilingual dict. |
| `t(key)` | Localized portal-level UI string. |
| `group_label(group_key)` | Bilingual sidebar group header. |
| `tool_name` / `tool_icon` / `title` / `tool_desc` / `tool_features` | Localized tool metadata accessors. |

### `tools/diagrams.py` — "how it works" pipeline drawer

| Function | Purpose |
|---|---|
| `pipeline_png(stages, accent)` | Cached matplotlib left-to-right arrow pipeline PNG used by each tool's collapsed "ℹ️ How it works" block. |

### `tools/ui_theme.py` — single home for all app-wide CSS

| Function | Purpose |
|---|---|
| `inject_css()` | Emit every app-wide `<style>` rule in one `st.markdown` call (button fills, chips, sticky residual strip, action-color system, language-toggle pinning, …). On Streamlit Cloud (`_IS_STREAMLIT_CLOUD`, same `/mount/src` detection as `tools/SSM/helpers/fit_cache.py`) an extra override block is appended that pushes `div.st-key-lang_toggle` down to `top: 3.25rem` — Cloud's own header + toolbar sit at the same top-right strip the toggle uses locally (`top: 0.5rem`) at a much higher z-index, so the toggle is fought below the header instead of trying to float above it. |
| `render_ram_badge()` | Sidebar RAM-usage bar — thin rounded track + colored fill (green `<60%`, amber `60–85%`, red `≥85%`) + a `RAM used / limit GB (pct%)` label, sourced from `tools/SSM/helpers/mem_budget.py::ram_usage()`. Renders nothing when that returns `None`. Wrapped in `st.fragment(run_every="10s")` so it self-refreshes without a full-page rerun (falls back to plain rendering if the installed Streamlit lacks `run_every`). Called from `IOED_Tool_Web.py` inside `with st.sidebar:`, right after `st.sidebar.divider()`. |

---

## DC / curve-tracer tools (`tools/`)

### `tools/csv_process.py` — B1500A / CITI / HP4155A CSV batch converter

Five sub-pages via a sidebar radio: **B1500A Smart Batch Tool**
(auto-detect measurement type, batch download), **TLM Resistance Avg**,
**E5270B CITI File Tool**, **HP4155A Data Processing Tool**, and
**B1500A Column Selection & Batch** (preset template converter:
Family / Transfer / Gummel).

| Function | Purpose |
|---|---|
| `detect_header_row(text)` | Find the header row by `DataName` in the first column. |
| `process_b1500a_file(file, template_cols, header_idx, preset)` | Map one raw B1500A CSV onto the preset template columns. |
| `process_file(file, selected_cols, template_header_idx)` | Column-selection batch: read a CSV with the detected/forced header and keep selected columns. |
| `sanitize_sheet_name(name)` | Excel-safe sheet name. |
| `detect_preset_from_filename(filename)` / `preset_to_type` / `get_preset_columns` / `format_output_filename` | Preset helpers (guess preset from filename, map to measurement type / expected columns / output name). |
| `convert_units(val)` | Current → `(scaled value, engineering unit)` (A/mA/µA/nA). |
| `_lines` / `parse_var_headers` / `parse_data_header` / `extract_var_list_blocks` / `extract_data_blocks` | Raw text-block splitting of B1500A exports. |
| `parse_gummel_from_text` / `parse_diode_by_header` / `parse_family_by_header` / `parse_gummel_vb_ib_ic` | Measurement-specific parsers → tidy DataFrames. |
| `parse_citi_file(content)` | CITI file → DataFrames for Excel export. |
| `parse_special_csv(content)` | Tolerant parser for headerless numeric CSV variants. |
| `parse_smu_table(df, smu_map)` | HP4155A SMU-table → named V/I columns. |
| `group_family(vc, ic, ib)` | Split a concatenated family sweep into per-Ib branches. |
| `format_ib_label(ib)` | Engineering-notation Ib legend label. |

### `tools/B1500A_Plot.py` — B1500A curve viewer + TLM analysis

Two sidebar pages: **B1500A Viewer** (auto-detects Diode / Gummel /
Family from the file, plots I–V with parameter extraction) and **TLM
Analysis** (sheet/contact resistance + transfer length from resistance
vs pad spacing).

| Function | Purpose |
|---|---|
| `find_col_like(df, keywords)` / `find_cols_starting(df, prefix)` | Fuzzy column lookup in the loaded CSV. |
| `ideality_factor(v, i, vmin, vmax)` | Diode/Gummel ideality η over a voltage window. |
| `parse_ib_current(label)` | Ib drive current (A) from a column label. |
| `fmt_current(a)` | Human-readable current with engineering units. |
| `zero_crossing_x(x, y)` | Interpolated x where y rises through zero. |
| `linear_fit_resistance(vce, ic, vmin, vmax)` | `(R_out, V_early, slope)` from a linear Ic–Vce fit. |
| `knee_voltage(vce, ic, frac)` | Vce where Ic first reaches `frac` of saturation. |
| `diode_knee_voltage(v, i)` | Turn-on voltage from linear extrapolation of the steep forward branch. |
| `saturation_current(v, i, vmin, vmax)` | Reverse-saturation current Is from the semilog-fit y-intercept. |

### `tools/HP4155A_plot.py` — HP/Agilent 4155A data viewer

Three-step page: ① persistent SMU assignment, ② data file upload,
③ plot settings (axes limits, grids, current/power scaling).

| Function | Purpose |
|---|---|
| `read_table(raw)` | Read the raw 4155A export into a DataFrame. |
| `parse_smu_table(df, smu_map)` | SMU-table → named V/I columns. |
| `group_by_ib(vc, y, ib)` | Split traces per Ib step. |
| `format_ib_label(ib)` / `scale_current(y)` / `scale_power(y)` | Legend + engineering-unit scaling helpers. |
| `apply_axes(fig, xlim, ylim, grid, minor_grid, show_right_ticks)` | Apply the shared axis/grid settings to a figure. |

### `tools/IOED_Gummel_Analyzer.py` — Gummel plot analyzer

Upload simulated/measured Gummel CSVs, overlay against the bundled UIUC
reference device, extract ideality factors in a live-adjustable current
window. Step 1 = upload + noise injection, Step 2 = Overlay / Single /
Summary tabs.

| Function | Purpose |
|---|---|
| `load_uiuc_ref()` | Cached parse of the embedded UIUC reference Gummel CSV. |
| `normalize_cols(cols)` / `pick_column_by_keywords(df, include, exclude)` | Tolerant column matching for arbitrary Gummel CSVs. |
| `load_and_standardize(content, add_noise, ic_noise, ib_noise)` | CSV → standardized `Vbase/Ic_abs/Ib_abs` DataFrame, with optional noise injection. |
| `calc_ideality(v, i, i_min, i_max, vt)` | Ideality η from a semilog fit inside a current window. |
| `extract_metrics(df, n_min, n_max, Vt)` | Per-file summary: η(Ic), η(Ib), max currents, peak β. |
| `update_axes(fig, y_title, log_y)` | Shared plot-axis styling. |

---

## RF tools (`tools/`)

### `tools/IOED_HBT_RF_extract.py` — RF S-parameter extraction (main RF page)

Four tabs: **📊 Overlay** (multi-file Bode/plateau/Smith overlay),
**📁 Individual** (per-file Bode / Plateau / Smith + de-embedding
preview, matplotlib Smith export), **📋 Summary** (fT/fmax table +
Excel), **🧰 Batch De-embed** (delegates to
`tools/batch_deembedding.py`). Bulk uploads go through the Rust
`parse_and_compute_batch` accelerator when available. Cross-page DUT
handoff via `tools/SSM/handoff.py`.

| Function | Purpose |
|---|---|
| `_df_from_rust_entry(prepared)` | Metrics DataFrame matching `compute_metrics()`'s schema from a Rust batch-kernel entry. |
| `process_dut(content, filename, s1_o, s1_s, s2_o, s2_s, s3_t, n_pts, f_min, f_max, *, prepared=None)` | Parse, de-embed, compute metrics, extract fT/fmax for one DUT file. |
| `_cal_sig(cal)` / `_dut_cache_key(...)` | Cache-key builders for calibration files / processed DUTs. |
| `_selected_dut_keys(names)` | Ordered cache keys of the currently-selected files. |
| `_cached_fig(key, build_fn)` / `_evict_stale_figs()` | Session-level figure cache + eviction of figures whose DUTs are gone. |

### `tools/SSM_extraction.py` — HBT SSM extraction page (thin wrapper)

Standalone portal page for SSM extraction: file upload + Open/Short
dummy selection, then delegates everything to
`tools/SSM/main_ssm_extraction.render_ssm_tab` (see the SSM index).
Also hosts the "🧩 Custom model" section entry.

### `tools/RF_simulator.py` — forward RF S-parameter simulator

Pick an SSM model (Cheng T / Cheng π / Xu T / Kun-Yang HEMT / Open &
Short pad / custom model), key in all parameters from scratch, inspect
Smith chart + fT/fmax Bode, download S2P/xlsx. Optional fit-target
overlay of a measured device handed over from the extraction pages.

| Function | Purpose |
|---|---|
| `_resolve_fit_target()` | Optional measured-device target (from cross-page handoff) for fit mode. |
| `_render_spec_inputs(specs, prefix, label, cols_per_row=4)` | `number_input` grid for a spec list. |
| `_collect_specs(specs, prefix)` | Read inputs back from session state → SI-unit dict. |
| `_build_smith(S, freq_hz, mults, title)` | Smith figure with per-trace multipliers. |
| `_smith_chart_with_dl(fig, key, filename, s2p_data, s2p_filename)` | Smith chart + side-by-side xlsx / s2p download buttons. |
| `_smith_multiplier_inputs(prefix, label)` | Four per-trace multiplier inputs → dict. |
| `_render_slider_preview(...)` | Mode-toggled slider preview block (Live tweak vs pre-computed Plotly). |
| `_slider_default_range(current_disp)` | Sane `(min, max, step)` for a slider. |
| `_render_rfsim_live_slider_preview(...)` | Streamlit-rerun-per-drag live preview. |
| `_render_rfsim_plotly_slider_preview(...)` | Pre-computed joint cartesian Plotly slider. |
| `_build_bode(S, freq_hz, title, *, extrap_method, sp_window)` | fT/fmax Bode figure. |
| `_render_bode_block(S, freq_hz, title, key)` | Bode plot + extrapolation-method selector underneath. |

### `tools/batch_deembedding.py` — batch SSM-style de-embedding tab

Extracts pad caps from the modeled Open dummy and lead inductances from
the modeled Short dummy (Gao 2015 §4.2), lets the user override any
element, then de-embeds every loaded DUT and offers a zip download.
Rendered inside the RF extraction page's 🧰 tab.

| Function | Purpose |
|---|---|
| `_cap_vs_f_fig(arrays_open, freq)` / `_ind_vs_f_fig(arrays_short, freq)` | Diagnostic capacitance / inductance vs frequency figures. |
| `_ovr_input(col, label, default_si, scale, fmt, state_key)` | Override `number_input` that seeds once from the extracted value. |
| `render_batch_deembedding_tab(*, all_data, open_data, short_data, ui, helpers)` | Public entry — renders the whole tab. |

### `tools/SSM/` — small-signal-model engine

See [`tools/SSM/helpers/INDEX.md`](tools/SSM/helpers/INDEX.md) for the
full per-function catalogue (orchestration, helpers, models, custom
model builder).

---

## E-beam lithography (`tools/ebeam_calculator.py`)

Self-contained page (imports no repo modules, so
`launch_ebl_calculator.py` can run it standalone). Sections: **1** chip
position in the e-beam holder, **2** left-computer origin setup,
**3** GDS mask viewer (streaming GDSII parser with a RAM budget so
Streamlit Cloud's ~1 GB limit isn't blown), **4** workflow mode selector
(dose-time test / first exposure / second alignment), each with a
per-mode exposure Time Calculator.

UI & small helpers:

| Function | Purpose |
|---|---|
| `segmented_radio(...)` | Local copy of the segmented-button radio replacement (kept inline so the file stays dependency-free). |
| `_chip_corner_guide_png()` | Static sketch explaining chip-corner labelling. |
| `_hex_to_rgba(hex, alpha)` / `_hex_rgb(hex)` | Colour conversions. |
| `_round_up_even(value)` | Smallest even integer ≥ value (floor 2). |
| `_show_outside_pattern_notice(msg)` | Centered red pill warning. |
| `_format_hms(seconds)` | Duration → `HH:MM:SS.sss`. |

Streaming GDSII parser + layer model:

| Symbol | Purpose |
|---|---|
| `class _PolyLayer` | All polygons of one (layer, datatype) as flat float64 arrays; `bbox`, `poly_areas` (vectorized shoelace), `first_vertices`. |
| `class _InstancedLayer` | A layer kept in instanced form (base polygons × offsets per group); `bbox`, `total_area_units2`, `instance_count`, `base_poly_count`. |
| `_apply_ref_transform(cx, cy, rotation, magnification, x_reflection)` | Apply an SREF/AREF transform to coordinates. |
| `_gds_real8(b)` | Decode a GDSII 8-byte excess-64 real. |
| `_element_run(a, mv, p0, blk, xy_off, xy_len)` | Bulk-decode a run of element records. |
| `_consolidate_cell(...)` | Merge one structure's parse accumulators into a raw-cell dict. |
| `_parse_gds(buf)` | Single-pass streaming parse of a GDSII byte buffer. |
| `_flatten_instanced(cell_name, cells, cache, budget)` | Resolve references → `{(layer, datatype): [group, …]}` under a RAM budget. |
| `_expand_groups_to_flat(groups)` | Tile group bases by their offsets into flat arrays. |
| `_load_gds(digest, _upload)` / `_load_gds_layers(upload)` / `_spec_to_layer(spec)` | Cached parse → per-cell layer specs → layer objects. |

Mask placement, plotting & coverage:

| Function | Purpose |
|---|---|
| `_layer_trace(name, polys, color, scale)` | Filled outline trace of every polygon. |
| `_layer_bbox_mm` / `_placed_bbox_mm` / `_bbox_rect_trace` | Layer bounding boxes (raw / placed) + dashed bbox rectangle. |
| `_rasterize_coverage` / `_coverage_grid` / `_coverage_heatmap_trace` | Low-res coverage raster for dense instanced layers. |
| `_unit_pattern_traces(layer, color, scale)` | Base (unit) pattern per group. |
| `_decimated_centers(layer, ...)` | Evenly-spread sample of instance centers (cap 5000). |
| `_nan_xy_from_flat(...)` | Flat polygons → NaN-separated closed x/y lists. |
| `_instances_in_window(layer, ...)` | Expand only instances inside a zoom window. |
| `_mask_overlay_traces(layer, ...)` / `_dense_layer_note(layer)` | Placed-mask traces (density-adaptive) + one-line rendering note. |

Per-cell area binning & exposure Time Calculator:

| Function | Purpose |
|---|---|
| `_bin_points(out, px, py, weights, ...)` | Accumulate weights into the flat exposure-cell grid. |
| `_fast_cell_areas_binned` / `_instanced_cell_areas_binned` / `_cell_areas_binned` | Vectorized per-cell mm² pattern area (flat / instanced / dispatch). |
| `_polygon_clip_per_cell_mm(polys_mm, cells)` | Exact per-cell polygon clipping (accurate small-N path). |
| `_render_time_calculator(prefix, polys_mm, cells, chip_size_mm, dotmap, ...)` | The per-mode Time Calculator section (dose, ramp, per-cell times, total HH:MM:SS). |

---

## Dev / profiling scripts

| File | Purpose |
|---|---|
| `tools/_profile_bulk_upload.py` | Profile the bulk-s2p-upload hot path (`profile_file`, `simulate_streamlit_loop`) — no Streamlit, per-stage `perf_counter` timings. |
| `tools/_profile_rust_batch.py` | Benchmark `rust_parse_and_compute_batch` vs the per-file Python loop (`py_loop` vs `rust_batch`). |
| `tools/_profile_ssm_extraction.py` | Profile the SSM extraction math pipeline end-to-end (`profile_extraction`, `_synth_dummies`). |
| `tools/SSM/rust_kernels/benchmark.py` | Rust-vs-NumPy parity test + microbenchmark for every kernel; exits 0 when Rust isn't built (NumPy fallback is supported). |

# Changelog

Per-tool version history for the IOED HBT Tools portal. Each tool's main
entry file declares its current version via a `__version__` constant near
the top, and the running app reads that constant into the page title.

When adding a feature, bump the relevant tool's `__version__` and add an
entry under that tool below.

---

## RF S-Parameter Extraction — [`tools/IOED_HBT_RF_extract.py`](tools/IOED_HBT_RF_extract.py)

### v1.1
- 🔁 **Workflow handoff.** The Individual tab gained a *"Send this device to an
  SSM page"* panel: choose **Raw** or **De-embedded** S-parameters, then jump
  straight to **SSM Extraction** or **Simulation & Fitting** with the device
  pre-loaded (no re-upload). Defaults to the de-embedded trace when present.
- 🧰 **Batch De-embed → handoff.** The batch tab can send the selected
  de-embedded device onward; **Extraction also receives every other
  de-embedded file** (the Z-parameter / Cold-HBT / τ_total methods need them),
  while Simulation & Fitting receives just the selected one.
- 🧹 Removed the now-redundant "🔬 SSM Extraction" pointer sub-tab (Individual
  tab is now Bode / Plateau / Smith).
- ✨ All `st.radio` controls switched to the segmented-button selector.

### v1.0
- 🪧 **SSM extraction split out into its own page**
  ([`tools/SSM_extraction.py`](tools/SSM_extraction.py), versioned
  independently from v6.3 — see below). The Individual tab's
  "🔬 SSM Extraction" sub-tab is now a pointer with a **Go there!**
  button that switches to the new page. This tool retains the RF
  metrics workflow (overlay / individual Bode·Plateau·Smith, summary,
  bulk upload, 3-step de-embedding + batch de-embed). The version
  history below (v4–v6) predates the split and is preserved under the
  **HBT SSM Extraction** heading, since that line of work became the
  SSM page.

---

## HBT SSM Extraction — [`tools/SSM_extraction.py`](tools/SSM_extraction.py)

### v7.1
- ✂️ **Extraction-only page.** Now does just the analytic peeling extraction
  (Cheng T/π). Forward simulation, custom models, Xu / Kun-Yang and **all
  tuning** (visual + auto) moved to **SSM Simulation & Fitting**; the model
  picker is gone.
- 🔁 **Handoff in/out.** Receives a (de-embedded) device — plus the other bias
  files as extras — from *RF At a Glance* / batch de-embed, no re-upload. After
  extracting, **"→ Send Cheng T/π to Simulation & Fitting"** forwards the exact
  same S-parameters (unchanged, with the correct `raw`/`deembedded` stage) plus
  the extracted values for final tuning.
- 🧊 **Cold-HBT file picker.** The Cold-HBT method can now select the cold
  device from the already-loaded bias files (defaults to one with "cold" in its
  name) instead of only uploading a separate S2P.
- 🐛 **Interactive section no longer collapses on edit.** The cache-first
  (skip-interactive) path is now latched at *session entry* — a cache written by
  the fine-tune auto-save mid-session no longer flips the page into cache mode
  and hides the interactive extraction.
- 🗂️ **Complete Parameter Summary** moved into a collapsed expander.

### v7.0
- 🧩 **Custom model** added to the per-DUT **Model** selection (alongside the
  built-in Cheng T/π, Xu, Kun-Yang extraction), with two sub-modes:
  - **Build / modify a model** — a visual builder
    ([`tools/SSM/custom_model/`](tools/SSM/custom_model/)): pick a device
    (Bipolar HBT B/C/E or Unipolar HEMT G/D/S) and a π/T intrinsic core, edit
    the B/C/E junction Networks + controlled source one part at a time against
    a live abstracted illustration, then add delay/port extras, extrinsic caps,
    access R/L and parasitic pad caps. Device switch auto-renames defaults
    (Cbc→Cgd, Rb→Rg, …). Upload a saved `.json` to **modify** it (loads the full
    setup, names it `…_modified`); export the schematic as PNG/SVG/📋-copy;
    **Download .json** or **Send to Load/Fit** (auto-loads + navigates +
    auto-downloads).
  - **Load & fit to this device** — overlay measured-vs-simulated on the
    **same** `render_smith_with_ftfmax` UI (Total/per-trace `ssm_residual` %
    above the Smith + fT/fmax card, downloads/📋-copy below) + the topology
    illustration + 2-column matplotlib Smith, and the **same grid-sweep Auto
    Tuning** (`render_tuning_expander`: Brute force / Optimized / Prioritized,
    per-parameter Min/Step/Max, "Use best values") via a generic
    `simulate_batch` adapter over the netlist solver.
- 🛠 Generic **netlist→Y→S solver** ([`custom_model/core.py`](tools/SSM/custom_model/core.py))
  reproduces Cheng-π/T and Xu to machine precision (~4e-16) when the equivalent
  topology is built; schema-v1 model files migrate automatically on load.
- 🩹 Robustness: component value inputs floored at 0 (negatives clamp to 0),
  stale tuning/value state cleared on every model change, and the schematic
  spacing widened so component values never overlap.

### v6.3
- 🪧 **Promoted to its own portal page** (split out of the RF
  S-Parameter Extraction tool's Individual tab). Upload DUT
  `.s2p`/`.csv` bias files — plus an optional device-dummy Open/Short
  pair for pad de-embedding — directly on this page
- 🧮 **"Calculated Tau_total and fmax" expander** under the
  measured-vs-modeled fT/fmax card for HBT T/π models (ChengT /
  ChengPi / XuT — not Kun-Yang). Left column: τ_total = 1/(2π f_T)
  with the full transit-time decomposition
  τ_total = τ_B + τ_C + (nkT/qI_c)·C_je + (R_c + R_e + nkT/qI_c)·C_bc;
  the model's τ_B+τ_C (T) or τ (π) is shown and the charge-storage
  term is reported as τ_total − (τ_B+τ_C), i.e. the emitter + collector
  charging times. Right column: calculated
  **f_max = √(f_T / (8π·C_BC·R_bb))** (C_BC = Cbcx+Cbc, R_bb = Rbi+Rb)
  compared against the real 0-dB-crossing f_max — for **both** measured
  and modeled (4 f_max values) — with a radio to switch C_BC / R_bb
  between the extracted values and custom inputs
- 📈 **Bode extrapolation control** rendered beneath the fT/fmax card:
  when a trace needs extrapolation, a radio picks **−20 dB/dec** or
  **single-pole** (least-squares fit over a window slider to the radio's
  right, defaulting to the final 5 GHz). The τ_total / f_max numbers in
  the expander follow whichever method is selected. The legend moved to
  the bottom-left **inside** the plot (vertical stack) instead of below it
- ⏱️ **Auto Tuning**: live ETA now rolls into minutes past 60 s and
  hours past 60 min; a total **"Evaluated in xx s (xx h: xx m: xx s)"**
  run time is shown above the best-residual line and persists across
  reruns. The six sweep buttons were relabeled to **Brute force /
  Optimized / Prioritized with CPU / CUDA** (the Rust/NumPy backend
  moved from the button face into the hover tooltip)
- 🐞 **Fine-tune cache fix**: fine-tune values no longer have to be
  entered twice. Two coupled causes were fixed — (1) the base_ui cache
  auto-restore now runs exactly once per Run-SSM cycle (the applied flag
  is set even when no cache existed); (2) the cache-first flow now
  **snapshots the cached params once per session** instead of re-reading
  the on-disk cache every render, so the per-keystroke auto-save can't
  feed back into `calc_vals`, shift the override sync hash, and clobber
  the next keystroke with the just-saved value

### v6.2
- ⏱️ **Transit-time extraction (Liu et al. method)** for T-topology
  models (ChengT / XuT): when ≥ 2 s2p files are loaded, a new
  "1/(2π f_T) vs 1/I_C" reference fit renders inside the Step-3
  intrinsic expander, just before the τ_B group. For each bias
  file we compute τ_total = 1/(2π f_T) from the Open+Short pad/lead
  de-embedded |h21|² 0-dB crossing — **access R retained**
  (Rpb/Rpc/Rpe forced to zero during the peel) so the formula's
  (R_C + R_EE)·C_BC intercept correction stays self-consistent; the
  peel is an algebraic no-op when no caps/Ls were set in Section 1,
  so pre-de-embedded files are handled transparently
- 📈 Linear fit → reference **C_JE = slope / (η · k T/q)** and
  **τ_B + τ_C = intercept − (R_C + R_E)·C_BC**. Re / Rc / Cbc / η /
  T are user-editable with defaults from the current file's
  extraction (Rpe, Rpc, Cbc + Cbcx, 1.0, 300 K). Per-file table of
  r_E = ηkT/(qI_C), τ_CC = (r_E + R_E + R_C)·C_BC, and
  τ_E = r_E · C_JE
- 🚀 **v_c split** (Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12),
  2004): inputs for the collector depletion width **W_C (nm,
  default 120)** and average collector velocity **v_c (cm/s,
  default 4×10⁷ for 2000 Å InP collectors)** → τ_C = W_C/(2 v_c),
  τ_B = (τ_B + τ_C) − τ_C
- 🔘 **One-click quickset buttons** on the τ_B / τ_C `number_input`s
  (label "v_c = …") populated from the v_c split, and on the **Rbe**
  input (label "Z-param = …") populated from the Z-parameter method
  when the current DUT participates in the fit
- 🩹 **Z-parameter method UI cleanup**: the Z₁₂ extraction frequency
  is now a `selectbox` of the 10 lowest measured frequencies
  (default = lowest) instead of a free-form `number_input`.
  Removed the slope-based ideality factor (η) display and
  persistence — derived η from Re(Z₁₂) vs 1/I_E disagrees with the
  Gummel-plot diode ideality when the simple T-model formula
  doesn't apply, so showing it was misleading. Set η in the
  τ-total fit from a Gummel-plot fit instead (typical InP HBT
  1.0–1.2)
- 🐞 **Cold-HBT → intrinsic-model parameter mapping fix**: Gao
  §5.5.2 "C_ex" is the extrinsic base–collector cap (= our **Cbcx**
  in intrinsic-model nomenclature). Previous versions routed
  C_ex_cold into the **Cbex** slot, so the green "Cold: …"
  reference line surfaced on the wrong parameter card. Corrected
  in `main_ssm_extraction._cold_map`
- 🎨 **Per-variable bordered containers** in the Cold-HBT extraction
  panel (`_cold_plot` wraps each of C_ex / C_bc / R_bi / C_be /
  R_b / R_c in its own card) and in the intrinsic-model per-param
  loop (`render_interactive_param_groups` wraps each parameter's
  plot + input + quickset row in its own bordered card)
- 🩹 Excel-export safety: trace name `"1/(2π f_T)"` in the τ-total
  plot was rejected by openpyxl (Excel forbids `/` in sheet
  titles); renamed to `"τ_total"`
- 📚 `helpers/INDEX.md` updated to document the new
  `_render_tau_total_fit_section` helper, the new `all_data` /
  `para_eff` kwargs on `render_interactive_param_groups`, and the
  Z-parameter / Cold-HBT UI changes

### v6.1
- ➕ New SSM model: **Kun-Yang HEMT** (pi-topology, forward simulation
  only). Inside → out: intrinsic pi (Cgs/Ri + Cgd/Rgd + Cds∥Rds + gm) →
  source-side Z_delay (R_delay ∥ C_delay in series with Rs+jωLs) →
  series-lead Z_ser (gate/drain/source) → Kun-Yang substrate pad
  (Cgsp/Rsub1, Cdsp/Rsub2, Cgdp). No standard open-dummy pad layer for
  this model — the substrate network IS the pad
- 🦀 Rust kernel `sim_kunyang_batch` (Rayon-parallel over the batch
  axis); CUDA via CuPy and NumPy fallback all parity-tested
- Static-cache pre-bake of Z_delay / Y_gs / Y_gd / Y_ds / gm /
  KY_int_planes / KY_pad_planes for fast Visual + Auto Tuning sweeps
- Also exposed in `tools/RF_simulator.py` with the KY-specific section
  ordering (substrate pad first, then access R & leads)

### v6.0
- 🦀 Rust acceleration for SSM hot paths: end-to-end batched simulators
  (Cheng T, Cheng π, Xu T) with Rayon parallelism — ~25× faster Auto
  Tuning sweeps on CPU; binary auto-built by the launcher when a Rust
  toolchain is available, NumPy fallback otherwise
- Backend status chips on Visual Tuning ("Compute backends available:
  ⚡ CUDA · 🦀 Rust") and Auto Tuning (sticky-True so transient
  re-import hiccups don't flip the badge); built-in diagnostic
  expander when Rust is missing
- Robust Rust loader: stashes the live extension on `builtins` so
  Streamlit's hot-reload module-clear doesn't trip PyO3's
  single-init guard
- 🎚️ Visual Tuning with live sliders (new section): drag-to-see
  preview with two modes — 🐢 Live Streamlit (rerun per drag,
  multi-param) and ⚡ Plotly slider (pre-computed frames, instant
  client-side scrub); side-by-side compact Smith + Bode preview,
  fixed-height scrollable variable cards (2-column grid), sticky
  right column for plots
- Fragment-scoped slider reruns (~10× faster per drag) via
  `st.fragment` — only the preview pane re-executes, not the whole
  script
- 💾 Fit caching: extracted parameters are saved per DUT/model and
  auto-restored when re-opening a file; "Use saved" button on the
  fine-tune section; revert-when-editing for in-flight values;
  zero-value write guard
- 🎨 Smith chart controls: per-S-param text-color picker, table-style
  scale-multiplier layout, color-mode persistence across reruns
- Multi-file S-parameter overlay uses Scattergl (WebGL) for fast
  loading with many files
- Streamlit deprecation migrations: `use_container_width=True` →
  `width="stretch"` (54 sites), `st.components.v1.html` → `st.iframe`,
  empty widget labels → descriptive labels + `label_visibility="collapsed"`
- Removed "Model low-freq pts" slider (hardcoded n_low=10); hid
  "Minimize deviation" Auto Tuning section behind a feature flag
- `s_to_y` divisions wrapped in `np.errstate(divide="ignore",
  invalid="ignore")` to silence RuntimeWarning floods near singular
  S-matrices

### v5.0
- Refactored SSM into an OOP class hierarchy (`SSMModelTemplate` parent in `base_ui.py` with concrete `ChengT`, `ChengPi`, `XuModel` subclasses)
- Added Xu's forward simulation
- Added more Smith chart plot options

### v4.7
- Switched scatter to scattergl for faster graph load
- Optimized calculation and loading

### v4.6
- Refactored SSM module into modular structure (`main_ssm_extraction` + `ssm_access_resistance` + `helpers/widgets`)
- Added quickset buttons (mean / median / low-f / high-f) to interactive parameter inputs
- Added Modeled/Measured Open-Short source selector for batch de-embedding
- Frequency-axis x-axis for short-dummy lead-inductance plots
- Linux/macOS launcher instructions

### v4.5
- Code refactoring and minor improvements

### v4.4
- Added optimized tuning strategies
- Added Smith chart with matplotlib
- Removed Ccex from Cheng's T
- Improved user usability

### v4.3
- Added other-OS support for launcher
- Added Ccex term for Cheng's T
- Fixed topology illustration for π
- Fixed some plotting

### v4.2
- Topology illustration
- Graph-data download
- Code refactoring

### v4.1
- Cosmetic improvements

### v4.0
- Extraction tuning with CPU and GPU optimizations

---

## Sim Gummel Plot Analyzer — [`tools/IOED_Gummel_Analyzer.py`](tools/IOED_Gummel_Analyzer.py)

### v2.0
- ✨ Log/Linear y-scale switched from a radio to the segmented-button selector.

### v1.9
- Implemented dynamic data slicing — auto-scale Y-axis now strictly binds to the selected X-axis range, eliminating out-of-bound numerical artifacts (e.g. Beta exploding to 5000+)

---

## SSM Simulation & Fitting — [`tools/RF_simulator.py`](tools/RF_simulator.py)

*(Renamed from "RF Forward Simulator" / "SSM Forward Simulation".)*

### v1.2
- 🏷️ **Renamed → "SSM Simulation & Fitting"** (forward-simulate *or* fit).
- 🎯 **Fit mode.** Optional measured-file upload (or a handoff from the other RF
  pages) switches the page to a measured-vs-modeled view: residual readout +
  **visual & auto tuning**, seeded from handed-over extracted values or
  auto-guessed from the device. The frequency axis follows the measured grid
  (start/stop/points hidden); a **📥 .s2p** download of the exact fitted device
  sits beside **✕ Clear**.
- 🧩 **Custom-model parity.** With a file present the custom section opens in
  *Fit to measurement* mode automatically (device shown, no freq inputs) and now
  carries a Smith multiplier — matching the built-in models.
- 🐛 **Pre-computed Plotly slider fixed.** It was frozen because (a) non-finite
  Bode/extrap points serialised to invalid JS (`nan`/`inf`) and threw a
  SyntaxError, and (b) `Plotly.restyle` silently no-ops on WebGL traces. Now
  emits `null`, mutates trace data + `Plotly.redraw`, and resolves the graph div
  robustly; Smith S-params are labelled inline with a small legend.
- 🐛 **Smith multiplier persists across models** (was reset on every model
  switch by a per-model key).
- 🔀 Reordered options (Custom before Open/Short); preview modes renamed to
  **🎯 Live tweak** / **⚡ Wide sweep** with when-to-use tooltips; live-tweak
  Smith uses inline labels + the Smith multipliers and a narrower slider column.
- ✨ Custom forward-sim gained fT/fmax extrapolation, the τ_total/fmax expander,
  topology illustration, matplotlib Smith and a tuning preview.

### v1.1
- 🧮 **"Calculated Tau_total and fmax" expander** below the Smith /
  fT-fmax row for every SSM model except Kun-Yang: τ_total = 1/(2π f_T)
  with the τ_B+τ_C (or τ) split and charge-storage residual, plus
  calculated f_max = √(f_T / (8π·C_BC·R_bb)) vs the real f_max, with an
  Extracted/Custom C_BC / R_bb toggle (shared helper with the SSM tab)
- 📈 **Bode extrapolation control** beneath the fT/fmax plot: −20 dB/dec
  or single-pole (window slider, default = final 5 GHz); the legend now
  sits bottom-left inside the plot

### v1.0
- Initial version tracking

---

## EBL Calculator — [`tools/ebeam_calculator.py`](tools/ebeam_calculator.py)

### v1.4
- ✨ All mode/preset/unit radios switched to the segmented-button selector.

### v1.3
- Fixed out-of-memory crash on Streamlit Cloud when uploading large (heavily
  arrayed) GDS files. The parser flattens array/AREF references by tiling
  coordinates with numpy instead of materializing millions of polygon
  objects, and stores each layer as flat float64 arrays (`_PolyLayer`,
  ~16 B/vertex, no per-polygon Python objects). The gdstk library is freed
  with `gc.collect()` after parsing, the parse cache is bounded to one entry
  (returning picklable numpy primitives wrapped into layer objects outside
  the cache), and a polygon budget surfaces a clear error instead of
  OOM-killing the app.
- **Automatic repetition detection / instanced rendering.** Repeated
  geometry is kept in instanced form (`_InstancedLayer`: a small base
  pattern + an offset lattice) rather than expanded, so a unit cell tiled
  ~1 M× stays a ~30-polygon base + offset table instead of millions of
  polygons. Detection groups references by target cell + transform and
  treats their origins as the lattice, so it catches both true AREF arrays
  **and** the common case of a CAD tool emitting an array as hundreds of
  thousands of individual single-placement SREFs (nested references combine
  by multiplying offset lattices, never polygon arrays). A real 4.64 M-polygon
  mask (969 k SREFs) parses in ~2 s (gdstk's own flatten alone takes ~15 s)
  and renders instantly; count/area/bbox match gdstk exactly. The viewer and
  workflow modes draw the **unit pattern + array footprint** instead of every
  polygon, and the Time Calculator computes area as `unit_area × tile_count`
  (exact, and independent of the on-screen rendering). Instanced layers are
  shown as a **low-res coverage view** (pattern coloured, empty white) — a
  white-background `go.Image` raster in the GDS viewer and a transparent
  `go.Heatmap` overlay on the workflow grid plots and the Time Calculator
  result plot — instead of tens of thousands of vector points, so the UI
  stays fast. The viewer also has a **box-select region inspector**: drag a
  box and that region redraws below with every polygon at full detail
  (KLayout-style), capped at 15 k polygons. Layers classify automatically:
  small → expanded
  `_PolyLayer`; dense + repetitive → `_InstancedLayer`; dense +
  non-repetitive → flat with a bounding-box / vectorized-area fallback.
  Plot/area/bbox helpers
  (`_layer_bbox_mm`, `_cell_areas_binned`, `_mask_overlay_traces`) dispatch
  on the layer type.

### v1.1
- Chip Size selector switched to μm presets; resolution is now displayed live
- Added Time Calculator to each workflow mode — computes filled resolution boxes, exposure + stage-movement time, total as HH:MM:SS; empty grids are hidden in the result plot
- Dose Time Testing: dose ramp inputs (initial + incremental)
- Custom alignment-mark inputs: mm / μm unit toggle, canonical storage in mm at 1 nm precision; Second Alignment registration display now reads the actually-selected Mark 1 / Mark 2


### v1.0
- Initial version tracking (includes custom second alignment mark)

---

## B1500A Plot & TLM — [`tools/B1500A_Plot.py`](tools/B1500A_Plot.py)

### v1.1
- **Diode:** η ideality window now drawn as dashed markers on the plot;
  extracted saturation current Is, knee/turn-on voltage, rectification ratio,
  plus forward- and leakage-current readouts at user-set voltages (default
  ±1 V, clamped to the sweep).
- **Gummel:** dashed η-window markers on the plot; added β(max) value, the Vb
  and Ic at which it occurs; switched ideality symbol to η.
- **Family:** hover shows per-curve DC gain β (parsed from the `Ib=…` column
  label) and per-curve offset voltage (Ic zero-crossing); added average offset
  voltage, saturation/output resistance with an adjustable Vc fit window
  (default = upper 40 % of the sweep, shaded on the plot), Early voltage |VA|,
  and knee voltage.
- Extraction controls and results grouped into bordered containers under each
  plot.

### v1.0
- Initial version tracking

---

## HP4155A Quick Plot — [`tools/HP4155A_plot.py`](tools/HP4155A_plot.py)

### v1.0
- Initial version tracking

---

## Measurement Data Multi-Process — [`tools/csv_process.py`](tools/csv_process.py)

### v1.0
- Initial version tracking

---

## Launcher & Portal — [`LAUNCH_Tool.py`](LAUNCH_Tool.py) / [`IOED_Tool_Web.py`](IOED_Tool_Web.py)

### 2026-06-24
- 🧭 **RF group reorganised into three clear pages:** *RF At a Glance*
  (de-embed + FOM + Smith), *SSM Extraction* (peeling only) and *SSM
  Simulation & Fitting* (all topologies + fit), wired together by a
  session-state **handoff bus** ([`tools/SSM/handoff.py`](tools/SSM/handoff.py))
  so a device flows between them without re-uploading.
- 🏠 **Home cards** now list each tool's key functionalities (bilingual).
- 🌐 **Language selector** switched from a dropdown to the segmented-button
  selector; new shared `segmented_radio` helper replaces every `st.radio`
  across the portal.
- 🦊 Added (then disabled, kept for reference) a Firefox-scoped CSS fix for the
  macOS-Firefox faint-input-border rendering.

### 2026-05-31
- ⬇️ **Auto-update on local launch** from the **canonical repo**
  (`ioedhbt/hbt-tools@IOED-Tools`, hardcoded — not the user's `origin`),
  run before venv setup so app/code/requirement updates land the same
  launch. Two paths: a **git checkout** does a safe
  `git fetch <canonical> + merge --ff-only` (never clobbers uncommitted
  tracked edits); a **zip install with no git** uses a pure-Python
  GitHub API + zipball download via `urllib` (no git binary, no extra
  pip deps) overlaid onto the folder, with the last-applied commit
  tracked in `.hbttools_update_sha`. Best-effort and non-destructive:
  protects the venv / `.git` / user data, never deletes files, and is
  silently skipped when offline. Opt out with `HBT_NO_AUTOUPDATE=1`.
  (Launcher self-changes apply on the next run.)
- 🔬 **HBT SSM Extraction registered as its own portal page** (sidebar →
  高頻量測 / RF), split out of the RF S-Parameter Extraction tool.

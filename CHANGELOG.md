# Changelog

Per-tool version history for the IOED HBT Tools portal. Each tool's main
entry file declares its current version via a `__version__` constant near
the top, and the running app reads that constant into the page title.

When adding a feature, bump the relevant tool's `__version__` and add an
entry under that tool below.

---

## RF S-Parameter Extraction — [`tools/IOED_HBT_RF_extract.py`](tools/IOED_HBT_RF_extract.py)

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

### v1.9
- Implemented dynamic data slicing — auto-scale Y-axis now strictly binds to the selected X-axis range, eliminating out-of-bound numerical artifacts (e.g. Beta exploding to 5000+)

---

## RF Forward Simulator — [`tools/RF_simulator.py`](tools/RF_simulator.py)

### v1.0
- Initial version tracking

---

## EBL Calculator — [`tools/ebeam_calculator.py`](tools/ebeam_calculator.py)

### v1.1
- Chip Size selector switched to μm presets; resolution is now displayed live
- Added Time Calculator to each workflow mode — computes filled resolution boxes, exposure + stage-movement time, total as HH:MM:SS; empty grids are hidden in the result plot
- Dose Time Testing: dose ramp inputs (initial + incremental)
- Custom alignment-mark inputs: mm / μm unit toggle, canonical storage in mm at 1 nm precision; Second Alignment registration display now reads the actually-selected Mark 1 / Mark 2


### v1.0
- Initial version tracking (includes custom second alignment mark)

---

## B1500A Plot & TLM — [`tools/B1500A_Plot.py`](tools/B1500A_Plot.py)

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

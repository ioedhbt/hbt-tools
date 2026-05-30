# Changelog

Per-tool version history for the IOED HBT Tools portal. Each tool's main
entry file declares its current version via a `__version__` constant near
the top, and the running app reads that constant into the page title.

When adding a feature, bump the relevant tool's `__version__` and add an
entry under that tool below.

---

## RF S-Parameter Extraction — [`tools/IOED_HBT_RF_extract.py`](tools/IOED_HBT_RF_extract.py)

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

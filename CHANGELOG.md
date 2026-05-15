# Changelog

Per-tool version history for the IOED HBT Tools portal. Each tool's main
entry file declares its current version via a `__version__` constant near
the top, and the running app reads that constant into the page title.

When adding a feature, bump the relevant tool's `__version__` and add an
entry under that tool below.

---

## RF S-Parameter Extraction — [`tools/IOED_HBT_RF_extract.py`](tools/IOED_HBT_RF_extract.py)

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

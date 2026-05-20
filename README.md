# IOED HBT Tools

A Streamlit-based web application for HBT (Heterojunction Bipolar Transistor) DC analysis, RF characterization, small-signal model extraction, parameter tuning, and EBL process planning.

The app is a multi-page Streamlit portal ([`IOED_Tool_Web.py`](IOED_Tool_Web.py)) that routes to individual tools grouped by measurement domain.

## Features

### RF (高頻量測)

**RF S-Parameter Extraction** ([`tools/IOED_HBT_RF_extract.py`](tools/IOED_HBT_RF_extract.py))
- Upload multi-bias S-parameter files (`.s2p`) or VNA CSV exports
- Automatic Open / Short de-embedding (Gao §4.2); optional Thru-half for symmetric calibration
- Gain & stability metrics: |h₂₁|², Mason's U, K-factor
- Auto-extrapolated fT / fmax with 20 dB/dec fit; download fitted data and extrapolation traces
- Smith and Bode chart generators (Plotly + a publication-style matplotlib Smith)
- Multi-DUT batch de-embedding tab — same Open / Short calibration applied to many DUTs with per-element overrides
- Small-signal model (SSM) extraction with three topologies built on a shared template class:
  - **Cheng T-topology** — *Microelectronics Journal* 121 (2022)
  - **Cheng π-topology** — Step 2 from Cheng, Step 3 after Zhang et al. (2015)
  - **Xu's T (2014)** — parallel Rbcx ∥ Cbcx variant
- Three access-resistance methods (Z-parameter, Cold-HBT, Open-Collector) plus a unified pre-extraction override that picks the highest-source value by default
- Per-parameter interactive fine-tuning with mean / median / low-f / high-f quickset buttons
- Global parameter tuning sweep (min/step/max per parameter) with full residual table, GPU-accelerated via CuPy

See [`tools/SSM/helpers/INDEX.md`](tools/SSM/helpers/INDEX.md) for a function-level catalog of every SSM submodule — read it before grep'ing the SSM tree.

**RF Forward Simulator** ([`tools/RF_simulator.py`](tools/RF_simulator.py))
- Simulate a small-signal model with customizable parameters
- Topology illustration and Smith chart generator

### Process (製程)

**EBL Calculator** ([`tools/ebeam_calculator.py`](tools/ebeam_calculator.py))
- JEOL ELS-7000 chip-corner position calculator with origin offset
- Per-mode workflows: dose-time test, first exposure, second alignment (incl. custom alignment marks)
- Optional GDS export via `gdstk`

### DC (直流量測)

- **B1500A Plot & TLM** ([`tools/B1500A_Plot.py`](tools/B1500A_Plot.py)) — Keysight B1500A output/transfer curve viewer with Transfer Length Method extraction and ideality-factor fit
- **HP4155A Quick Plot** ([`tools/HP4155A_plot.py`](tools/HP4155A_plot.py)) — HP/Agilent 4155A curve tracer data viewer

### TCAD

**Gummel Plot Analyzer** ([`tools/IOED_Gummel_Analyzer.py`](tools/IOED_Gummel_Analyzer.py))
- Upload and overlay Gummel plot data
- Interactive bias-point annotation

### Data Processing (資料處理)

**Measurement Data Multi-Process** ([`tools/csv_process.py`](tools/csv_process.py))
- B1500A CSV → preset template (Ic-Vc Family / Transfer / Gummel) converter
- Batch processing with ZIP export

---

## GPU-Accelerated Tuning

The parameter tuning engine supports CUDA (NVIDIA GPU) for massive-parallel combo sweeps. Key properties:

- **Adaptive block sizing** — automatically sizes the per-iteration batch to fit available VRAM, expands when headroom allows, shrinks on OOM
- **Calibrated memory model** — measures actual per-combo GPU footprint on the first iteration and uses it for subsequent sizing decisions
- **Low overhead** — no per-iteration pool flushing; pool-cache-aware free-memory estimation to avoid Copy engine thrashing on Windows WDDM
- **Mixed precision** — Cheng kernels run the main sweep in fp32 for ~5–10× speedup on consumer GPUs (fp64 is gimped 1/64 vs fp32 on e.g. RTX 3050); top-K candidates are reranked at fp64 to keep published residuals fully precise
- Tested on CUDA 12 and CUDA 13

---

## Repository Layout

- [`LAUNCH_Tool.py`](LAUNCH_Tool.py) — bootstrap launcher: creates a `.hbttools/` venv next to itself, installs all dependencies (incl. CuPy if CUDA is detected), then starts the Streamlit app
- [`IOED_Tool_Web.py`](IOED_Tool_Web.py) — Streamlit portal that routes to each tool (password-protected)
- [`tools/`](tools/) — individual tool modules
- [`tools/SSM/`](tools/SSM/) — small-signal model extraction subpackage
  - [`tools/SSM/helpers/INDEX.md`](tools/SSM/helpers/INDEX.md) — function-level catalog of every helper, model, and submodule
  - [`tools/SSM/models/`](tools/SSM/models/) — model classes (ChengT, ChengPi, XuModel) sharing a common `SSMModelTemplate` parent in [`base_ui.py`](tools/SSM/models/base_ui.py)

---

## Quick Start

Requires **Python 3.9+**, and optionally CUDA 12 or CUDA 13.

In Windows, double click ```LAUNCH_Tool.py``` to start the UI.

On Linux or macOS, open a terminal in the project directory and run:

```bash
python3 LAUNCH_Tool.py
```

If `python3` is not found, install Python 3.9+ from your package manager (e.g. `sudo apt install python3 python3-venv` on Debian/Ubuntu, or `brew install python` on macOS). The launcher will create a local virtual environment at `.hbttools/` and install all required packages on first run.

---

## Where are extracted SSM fits cached?

Fine-tuned SSM parameters are persisted **per DUT, per model** in JSON files so you can re-open the same `.s2p` later and pick up where you left off.

Resolution order (first writable path wins):

1. `$HBT_FIT_CACHE_DIR` — env-var override (set this to point the cache anywhere you like)
2. `~/.hbt-tools/` — the default on every OS
3. `<repo>/.fit_cache/` — fallback if the home dir isn't writable

Layout under the cache root:

```
fits/
  <dut_basename>/
    <dut_basename>_<model_short>.json     # e.g. mydut_T.json, mydut_pi.json, mydut_XuT.json
```

On Windows the default is `C:\Users\<you>\.hbt-tools\fits\…`, on macOS / Linux it's `~/.hbt-tools/fits/…`. Each `.json` holds a `saved_at` timestamp plus the SI-unit params dict for that one (file, model) pair — corrupt or all-zero saves for one model can no longer overwrite a sibling.

**Streamlit Community Cloud:** caching is **automatically disabled** there because the Cloud VM is ephemeral (any disk write evaporates on the next cold start) and a stale cache would otherwise fight your fine-tune edits. The cache UI is hidden on Cloud. To force one mode or the other set `HBT_DISABLE_FIT_CACHE=1` (off) or `HBT_FIT_CACHE_FORCE_ON=1` (on).

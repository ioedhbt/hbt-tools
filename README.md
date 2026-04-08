# IOED HBT Tools

A Streamlit-based web application for HBT (Heterojunction Bipolar Transistor) DC analysis, RF characterization, small-signal model extraction, and parameter tuning.

## Features

### HBT RF Extraction (`IOED_HBT_RF_extract.py`)
- Upload multi-bias S-parameter files (`.s2p`)
- Automatic de-embedding (open/short structures)
- Gain and stability metrics: |h₂₁|², Mason's U, K-factor
- Small-signal model (SSM) extraction with multiple published models:
  | Model | Reference |
  |---|---|
  | **Cheng T / π** | Cheng et al., *Microelectronics Journal* 121, 2022 |
  | **Degachi π** | Degachi & Ghannouchi, *IEEE TED* 55(4), 2008 |
  | **Xu CBC/Cπ** | Dvorak & Bolognesi, *IEEE MTT-S* 2003 |
- Global parameter tuning with GPU acceleration (CUDA via CuPy)

### Gummel Analyzer (`IOED_Gummel_Analyzer.py`)
- Upload and overlay Gummel plot data
- Interactive bias-point annotation

### DC Characterization
- **B1500A Plot** — Keysight B1500A output/transfer curve viewer
- **HP4155A Plot** — HP/Agilent 4155A curve tracer data viewer

---

## GPU-Accelerated Tuning

The parameter tuning engine supports CUDA (NVIDIA GPU) for massive-parallel combo sweeps. Key properties:

- **Adaptive block sizing** — automatically sizes the per-iteration batch to fit available VRAM, expands when headroom allows, shrinks on OOM
- **Calibrated memory model** — measures actual per-combo GPU footprint on the first iteration and uses it for subsequent sizing decisions
- **Low overhead** — no per-iteration pool flushing; pool-cache-aware free-memory estimation to avoid Copy engine thrashing on Windows WDDM
- Tested on CUDA 12 and CUDA 13

---

## Quick Start

Requires **Python 3.9+** (no other pre-installation needed).

In Windows, double click ```LAUNCH_Tool.py``` to start the UI.

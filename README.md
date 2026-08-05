# IOED HBT Tools

A Streamlit web app for HBT (Heterojunction Bipolar Transistor) lab
work: DC curve analysis, RF characterization, small-signal model (SSM)
extraction & fitting, and e-beam lithography process planning.

---

## Quick Start

Requires **Python 3.9+**. Everything else (virtual environment,
packages) is installed automatically by the launcher on first run.

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/)
   (check **"Add python.exe to PATH"** during install).
2. **Double-click `LAUNCH_Tool.py`** in the project folder.
3. A terminal opens, sets up `.hbttools\` (first run takes a few
   minutes), then the app opens in your browser.

### macOS

1. Install Python: `brew install python`
   (or from [python.org](https://www.python.org/downloads/)).
2. Open Terminal in the project folder and run:

   ```bash
   python3 LAUNCH_Tool.py
   ```

### Linux

1. Install Python + venv, e.g. Debian/Ubuntu:

   ```bash
   sudo apt install python3 python3-venv
   ```

2. In the project folder:

   ```bash
   python3 LAUNCH_Tool.py
   ```

### Optional: NVIDIA GPU acceleration (all OSes)

The SSM parameter-tuning sweeps run massively parallel on an NVIDIA GPU
if CUDA is present. Install **CUDA 12 or 13** from the official
NVIDIA page: **<https://developer.nvidia.com/cuda-downloads>** — then
(re)run the launcher; it detects CUDA and installs the matching CuPy
wheel automatically. No GPU? Everything still works on CPU.

### EBL calculator only

The e-beam calculator can run standalone (no portal, no password):
double-click `launch_ebl_calculator.py` (Windows) or run
`python3 launch_ebl_calculator.py` (macOS/Linux).

---

## What it can do

| Tool | Page | In short |
|---|---|---|
| **RF At a Glance** | [`tools/rf/at_a_glance.py`](tools/rf/at_a_glance.py) | Upload multi-bias `.s2p` / VNA CSV files; Open/Short (and Thru-half) de-embedding; \|h₂₁\|², Mason U, K; auto-extrapolated fT/fmax; Smith & Bode charts; batch de-embedding; Excel/S2P export everywhere |
| **HBT SSM Extraction** | [`tools/rf/extraction.py`](tools/rf/extraction.py) | Peeling extraction with Cheng T / Cheng π / Xu T topologies; Z-param / Cold-HBT / open-collector access resistances; per-parameter fine-tuning; GPU/Rust-accelerated auto-tuning sweeps; per-DUT fit cache |
| **RF Forward Simulator** | [`tools/rf/simulator.py`](tools/rf/simulator.py) | Forward-simulate any SSM (or a custom user-built model) from scratch; live sliders; Smith + fT/fmax Bode; fit against a measured device handed over from the other RF pages |
| **EBL Calculator** | [`tools/process/ebeam/calculator.py`](tools/process/ebeam/calculator.py) | JEOL ELS-7000 chip-position & origin calculator; GDS mask viewer (streaming parser); dose-time test / first exposure / second alignment workflows with exposure-time estimates |
| **B1500A Plot & TLM** | [`tools/dc/b1500a_plot.py`](tools/dc/b1500a_plot.py) | Diode / Gummel / Family curve viewer with ideality, knee voltage, R_out, Early voltage; TLM analysis |
| **HP4155A Quick Plot** | [`tools/dc/hp4155a_plot.py`](tools/dc/hp4155a_plot.py) | HP/Agilent 4155A curve-tracer data viewer |
| **Gummel Plot Analyzer** | [`tools/tcad/gummel_analyzer.py`](tools/tcad/gummel_analyzer.py) | Overlay Gummel data vs a reference device; live ideality-factor extraction |
| **Measurement Data Multi-Process** | [`tools/data/csv_process.py`](tools/data/csv_process.py) | B1500A / CITI / HP4155A CSV batch converters with ZIP export |

The UI is bilingual (English / 中文) — toggle at the top right.

---

## AI-agent / scripting API

Need to fit `.s2p`/`.csv` data to an SSM model (built-in or custom) from a
script or AI agent, without the Streamlit UI? Use
[`tools/rf/ssm/agent_api.py`](tools/rf/ssm/agent_api.py) — see
[`docs/INDEX.md`](docs/INDEX.md#ai-agent-fitting-api-toolsssmagent_apipy) for the full
function table and usage examples. It reads the same de-embedding-status
`!` header the app writes, so pointing it at an already de-embedded file
automatically freezes the removed pad/lead parasitics during the fit.
Fitting uses CUDA (cupy) > the project's Rust kernels > NumPy automatically
(`backend="auto"`, override with `backend=`/`--backend`).

```bash
python tools/rf/ssm/agent_api.py fit s2p/deemb_preext_vce3.5_ib280u.s2p --model T --out result.json
```

---

## Documentation

| Doc | What's inside |
|---|---|
| [`MAP.md`](MAP.md) | **Start here** — one line per module, the repo's invariants, and a "task → file" table |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **How things work** — boot flow, page anatomy, data flow, SSM engine layers, acceleration, "task → file" lookup table |
| [`docs/INDEX.md`](docs/INDEX.md) | Function-level catalogue of every Python file outside `tools/rf/ssm/` |
| [`docs/SSM_INDEX.md`](docs/SSM_INDEX.md) | Function-level catalogue of the SSM extraction tree |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Version history |

---

## Notes

- **Fit cache** — fine-tuned SSM parameters persist per (DUT, model) in
  `~/.hbt-tools/fits/` (override with `HBT_FIT_CACHE_DIR`; auto-disabled
  on Streamlit Cloud). Details in
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#5-acceleration--persistence).
- **Rust acceleration** — prebuilt kernel binaries ship with the repo.
  `python dev/check_rust_status.py` reports whether they load;
  `python dev/build_rust_kernels.py` rebuilds them (needs a Rust toolchain,
  once per OS).
- **Password** — deployed instances are gated by
  `st.secrets["APP_PASSWORD"]`; local launches via `LAUNCH_Tool.py` skip
  the gate.

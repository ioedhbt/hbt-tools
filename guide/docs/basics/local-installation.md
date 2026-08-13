# Local installation

For most uses the [streamlit online web app](https://hbt-tools.streamlit.app/)
is enough. Install locally when you need more RAM for large mask files, GPU
acceleration, or no upload cap.

## 1. Install Python

Install **Python 3.9+** from [python.org](https://www.python.org/downloads/).
On Windows, tick **"Add python.exe to PATH"** during install.

## 2. Download the code

1. Go to the [HBT tools](https://github.com/ioedhbt/hbt-tools) Github repo.
2. Click the green **"<> Code"** button, then **Download ZIP**.
3. Extract the zip to a folder.

## 3. Launch

Open a terminal in the project folder and run:

```bash
python3 LAUNCH_Tool.py
```

On Windows, double-clicking `LAUNCH_Tool.py` works too. The first run takes
a few minutes to set up, then opens the app in your browser. No password gate
locally.

!!! tip
    The e-beam calculator can run standalone (no portal): double-click
    `launch_ebl_calculator.py` on Windows, or run
    `python3 launch_ebl_calculator.py` on macOS/Linux.

## 4. Optional: NVIDIA GPU acceleration

The SSM tuning sweeps run much faster on an NVIDIA GPU. Install **CUDA 12 or
13** from the official NVIDIA page:
<https://developer.nvidia.com/cuda-downloads>. Then run the launcher again —
it detects CUDA and installs the matching GPU packages automatically. No GPU?
Everything still works on CPU.

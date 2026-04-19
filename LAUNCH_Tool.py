"""
LAUNCH_Tool.py — Bootstrap launcher for IOED Tool.

Creates (or reuses) a local Python virtual environment at .hbttools/ next to
this file, installs all required packages into it, then launches the
Streamlit app using the venv interpreter.

Run with any system Python 3.9+:
    python LAUNCH_Tool.py

CuPy (CUDA GPU acceleration) is installed automatically when a compatible
NVIDIA GPU and CUDA 12 or 13 toolkit are detected.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.resolve()
VENV_DIR = ROOT / ".hbttools"
APP_FILE = ROOT / "IOED_Tool_Web.py"

# ── Core packages always installed ───────────────────────────────────────────
REQUIRED = [
    ("streamlit",  "streamlit"),
    ("numpy",      "numpy"),
    ("pandas",     "pandas"),
    ("plotly",     "plotly"),
    ("matplotlib", "matplotlib"),
    ("openpyxl",   "openpyxl"),
    ("psutil",     "psutil"),
    ("PIL",        "pillow"),
]

# ── Resolve venv interpreter path (Windows vs. Unix) ─────────────────────────
_win = sys.platform == "win32"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if _win else "bin/python")
VENV_STREAMLIT = VENV_DIR / ("Scripts/streamlit.exe" if _win else "bin/streamlit")


# ── CUDA detection ────────────────────────────────────────────────────────────

def _run_silent(cmd):
    """Run a command, return stdout+stderr as a string, or '' on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def detect_cuda_major():
    """Return the CUDA major version (12, 13, …) or None if not found.

    Strategy:
      1. `nvcc --version`   — gives the installed toolkit version.
      2. `nvidia-smi`       — gives the driver's max supported CUDA version.
    Either source is sufficient; nvcc is preferred as it reflects the actual
    runtime used to build CuPy wheels.
    """
    # macOS has no NVIDIA CUDA support — skip detection entirely.
    if sys.platform == "darwin":
        return None

    # nvcc --version output: "Cuda compilation tools, release 12.4, V12.4.99"
    out = _run_silent(["nvcc", "--version"])
    m = re.search(r"release\s+(\d+)\.(\d+)", out)
    if m:
        return int(m.group(1))

    # nvidia-smi output header: "CUDA Version: 12.4"
    out = _run_silent(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if m:
        return int(m.group(1))

    return None


def cupy_pip_name(cuda_major):
    """Return the correct CuPy wheel name for the given CUDA major version."""
    mapping = {12: "cupy-cuda12x", 13: "cupy-cuda13x"}
    return mapping.get(cuda_major)


# ── Venv helpers ──────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a command, streaming output live."""
    subprocess.check_call(cmd, **kwargs)


def pip_works():
    """Return True if pip is functional inside the venv."""
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "--version"],
        capture_output=True,
    )
    return result.returncode == 0


def recreate_venv():
    """Delete and recreate the venv from scratch."""
    import shutil, stat, os
    if VENV_DIR.exists():
        print(f"Removing broken virtual environment at {VENV_DIR} ...")

        def _force_remove(func, path, _exc):
            # Clear read-only flag then retry — common on Windows/OneDrive,
            # harmless on POSIX where files usually aren't read-only.
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass  # best-effort; rmtree will surface any real failure

        # `onexc` is Python 3.12+; `onerror` is the older equivalent.
        if sys.version_info >= (3, 12):
            shutil.rmtree(str(VENV_DIR), onexc=_force_remove)
        else:
            shutil.rmtree(str(VENV_DIR), onerror=_force_remove)
    print(f"Creating fresh virtual environment at {VENV_DIR} ...")
    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        if sys.platform.startswith("linux"):
            print("\nERROR: Failed to create virtual environment.")
            print("On Debian/Ubuntu you may need to install the venv package:")
            print(f"    sudo apt install python3-venv  python{sys.version_info.major}.{sys.version_info.minor}-venv")
        raise
    print("Virtual environment created.")


def create_venv():
    print(f"Creating virtual environment at {VENV_DIR} ...")
    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        # On Debian/Ubuntu the venv module ships in a separate apt package.
        if sys.platform.startswith("linux"):
            print("\nERROR: Failed to create virtual environment.")
            print("On Debian/Ubuntu you may need to install the venv package:")
            print(f"    sudo apt install python3-venv  python{sys.version_info.major}.{sys.version_info.minor}-venv")
        raise
    print("Virtual environment created.")


def pip_install(packages):
    """Install a list of pip package names into the venv."""
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade"] + packages)


def is_importable(import_name):
    """Return True if import_name can be imported inside the venv."""
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", f"import {import_name}"],
        capture_output=True,
    )
    return result.returncode == 0


def check_missing(packages):
    """Return list of pip names from `packages` not yet importable in the venv."""
    return [pip for imp, pip in packages if not is_importable(imp)]


# ── Main ──────────────────────────────────────────────────────────────────────

def _pause_if_interactive(msg="Press Enter to exit..."):
    """Block on user input only when stdin is a TTY.

    On macOS/Linux double-click launches there is no controlling terminal,
    so reading stdin would raise EOFError and obscure the real error.
    """
    try:
        if sys.stdin and sys.stdin.isatty():
            input(msg)
    except (EOFError, OSError):
        pass


def main():
    if not APP_FILE.exists():
        print(f"ERROR: Cannot find {APP_FILE.name} in {ROOT}")
        print("Make sure LAUNCH_Tool.py and IOED_Tool_Web.py are in the same folder.")
        _pause_if_interactive()
        sys.exit(1)

    # ── Detect CUDA and build the full package list ───────────────────────────
    cuda_major = detect_cuda_major()
    cupy_pkg   = cupy_pip_name(cuda_major) if cuda_major else None

    all_packages = list(REQUIRED)
    if cupy_pkg:
        print(f"CUDA {cuda_major} detected — will install {cupy_pkg} for GPU acceleration.")
        all_packages.append(("cupy", cupy_pkg))
    elif cuda_major:
        print(f"CUDA {cuda_major} detected but no matching CuPy wheel "
              f"(supported: 12, 13). GPU acceleration will not be available.")
    else:
        print("No CUDA toolkit found. GPU acceleration will not be available.")

    # ── Create or repair venv ─────────────────────────────────────────────────
    # A broken venv (missing pip, missing CLI entry points) is treated the
    # same as no venv: delete and recreate so the user never gets stuck.
    if not VENV_PYTHON.exists() or not pip_works():
        recreate_venv()
        run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        pip_install([pip for _, pip in all_packages])
    else:
        # Venv is healthy — only install what's missing
        missing = check_missing(all_packages)
        if missing:
            print(f"Installing missing packages: {', '.join(missing)}")
            pip_install(missing)
        else:
            print("All required packages are present.")

    # ── Launch the app ────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  Launching IOED Integrated Component Analysis")
    print("  and Extraction Platform ...")
    print("=" * 50)
    print()

    try:
        run([str(VENV_STREAMLIT), "run", str(APP_FILE)])
    except KeyboardInterrupt:
        print("\nServer shut down cleanly.")


if __name__ == "__main__":
    main()

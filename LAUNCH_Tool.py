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

import os
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
    ("scipy",      "scipy"),
    ("gdstk",      "gdstk"),
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

    # ── Rust acceleration status (informational only — never a blocker) ────────
    # Check whether the compiled hbt_rust_kernels extension is committed for
    # this OS+arch.  No pip install or rust toolchain is queried; we just
    # look for the binary on disk in tools/SSM/rust_kernels/bin/<arch>/.
    import platform as _platform
    _arch_machine = (_platform.machine() or "unknown").lower()
    if _win:
        _arch_dir = f"win_{_arch_machine}"
    elif sys.platform == "darwin":
        _arch_dir = f"macosx_{_arch_machine}"
    else:
        _arch_dir = f"linux_{_arch_machine}"
    _rust_bin_dir = (ROOT / "tools" / "SSM" / "rust_kernels" / "bin"
                     / _arch_dir)
    def _has_active_binary(_dir):
        # Same filter as check_rust_status.py: ignore .old-<ts>.<ext>
        # files that the build script leaves behind when a previously-
        # loaded .pyd couldn't be deleted in place.
        if not _dir.is_dir():
            return False
        for pat in ("hbt_rust_kernels*.pyd",
                    "hbt_rust_kernels*.so",
                    "hbt_rust_kernels*.dylib"):
            for f in _dir.glob(pat):
                if ".old-" not in f.name:
                    return True
        return False

    _rust_binary = _has_active_binary(_rust_bin_dir)
    # Auto-build path — only triggers on the local launcher.  Streamlit
    # Cloud doesn't run this script, so the user keeps full control
    # over Linux builds (build locally on a WSL/Docker box and commit
    # the .so).
    if not _rust_binary:
        # Probe for cargo (Rust compiler) without crashing on missing.
        _cargo_ok = False
        try:
            _r = subprocess.run(
                ["cargo", "--version"],
                capture_output=True, text=True, timeout=10)
            _cargo_ok = (_r.returncode == 0)
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            _cargo_ok = False

        if _cargo_ok:
            print(f"Rust toolchain detected; building hbt_rust_kernels "
                  f"for {_arch_dir} (one-time, ~30 s)…")
            try:
                run([sys.executable,
                     str(ROOT / "build_rust_kernels.py")])
                _rust_binary = _has_active_binary(_rust_bin_dir)
            except subprocess.CalledProcessError as e:
                print(f"  Build failed ({e}).  Continuing with "
                      "NumPy fallback — the tool still works, "
                      "just without the ~25× sweep speedup.")
                _rust_binary = False

    if _rust_binary:
        print(f"Rust acceleration: ENABLED ({_arch_dir} binary present).")
        print("  HBT_USE_RUST_SIM_BATCH auto-set for this session "
              "(unset HBT_USE_RUST_SIM_BATCH=0 to force NumPy).")
    else:
        print(f"Rust acceleration: not available for {_arch_dir} — "
              "NumPy fallback active.")
        print("  (To enable: install rustup from https://rustup.rs/ "
              "then re-run this launcher; the binary is auto-built.)")

    # ── Launch the app ────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  Launching IOED Integrated Component Analysis")
    print("  and Extraction Platform ...")
    print("=" * 50)
    print()

    # Hand a local-launch flag to the Streamlit subprocess so
    # IOED_Tool_Web.py skips its password gate (the gate is preserved
    # for Streamlit Cloud / public deployments, which don't go through
    # this launcher).
    _child_env = dict(os.environ)
    _child_env["HBT_LOCAL_LAUNCH"] = "1"
    try:
        subprocess.check_call(
            [str(VENV_STREAMLIT), "run", str(APP_FILE)],
            env=_child_env)
    except KeyboardInterrupt:
        print("\nServer shut down cleanly.")


if __name__ == "__main__":
    main()

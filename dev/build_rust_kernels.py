#!/usr/bin/env python3
"""
build_rust_kernels.py — one-shot Rust → binary build for the SSM kernels.

Run this **once per OS** to produce a self-contained extension binary that
ships with the repo.  End users / Streamlit Cloud / anyone who just clones
the repo gets the prebuilt binary on their platform automatically — no pip
install, no maturin, no Rust toolchain on their side.

    python dev/build_rust_kernels.py

What it does
------------
1. Verifies that a Rust toolchain (``cargo``) is installed.  If not, prints
   install instructions and exits — we deliberately don't shell out to
   rustup to keep this script auditable.
2. Creates (or reuses) a separate build-only venv at ``.hbttools_build/``
   so the LAUNCH_Tool runtime venv (``.hbttools/``) stays untouched and
   ``requirements.txt`` never grows a maturin dependency.
3. Installs ``maturin`` into the build venv.
4. Runs ``maturin build --release`` against ``tools/rf/ssm/rust_kernels/``,
   producing an ABI3 wheel.  ABI3-py39 means the same .pyd / .so works on
   every Python ≥ 3.9 — no per-Python-version rebuilds.
5. Extracts the compiled extension from the wheel and drops it into
   ``tools/rf/ssm/rust_kernels/bin/<platform_arch>/hbt_rust_kernels.<ext>``.
   That's exactly where ``helpers/rust_kernels.py`` looks at import time.

Re-running is idempotent: the script overwrites the previous binary in
place.  For a clean rebuild, delete the ``.hbttools_build`` venv and the
``tools/rf/ssm/rust_kernels/target`` directory.

Cross-platform shipping
-----------------------
Each binary is OS- and arch-specific.  To support every platform, run this
script once on each target OS and commit the resulting
``tools/rf/ssm/rust_kernels/bin/<tag>/...`` file to the repo.  The loader
picks the right one based on the running platform.  Typical tags:

    win_amd64       (Windows 64-bit)
    linux_x86_64    (Linux x86-64, incl. Streamlit Cloud)
    macosx_arm64    (Apple Silicon)
    macosx_x86_64   (Intel Mac)
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# This file lives in dev/ — the repo root is one level up.
ROOT       = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Single source of truth for the crate location — three scripts used to build
# this path by hand and drifted apart when the tree moved.
from tools.common.paths import RUST_CRATE_DIR              # noqa: E402
CRATE_DIR  = RUST_CRATE_DIR
BIN_BASE   = CRATE_DIR / "bin"
BUILD_VENV = ROOT / ".hbttools_build"
TARGET_DIR = CRATE_DIR / "target"
WHEEL_OUT  = TARGET_DIR / "wheels"

_WIN = sys.platform == "win32"
_BUILD_VENV_PY = BUILD_VENV / ("Scripts/python.exe" if _WIN else "bin/python")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _have(cmd: str) -> bool:
    """True iff `cmd` can be invoked with `--version`."""
    try:
        subprocess.run([cmd, "--version"],
                       capture_output=True, check=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return False


def _arch_tag() -> str:
    """Match the tag scheme in helpers/rust_kernels.py::_arch_tag."""
    machine = (platform.machine() or "unknown").lower()
    if sys.platform == "win32":
        return f"win_{machine}"
    if sys.platform == "darwin":
        return f"macosx_{machine}"
    return f"linux_{machine}"


def _run(cmd: list[str], **kw):
    """Run a command, streaming output, raising on non-zero."""
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, **kw)


def _ensure_build_venv() -> Path:
    """Create the .hbttools_build venv if missing.  Return its python exe."""
    if _BUILD_VENV_PY.exists():
        r = subprocess.run([str(_BUILD_VENV_PY), "-m", "pip", "--version"],
                           capture_output=True)
        if r.returncode == 0:
            return _BUILD_VENV_PY
        # Half-created venv (interrupted setup, or ensurepip failed once):
        # the interpreter is there but pip never landed.
        print("Build venv is missing pip; bootstrapping…")
        _run([str(_BUILD_VENV_PY), "-m", "ensurepip", "--upgrade"])
        return _BUILD_VENV_PY
    print(f"Creating build-only venv at {BUILD_VENV} (one-time setup)…")
    _run([sys.executable, "-m", "venv", str(BUILD_VENV)])
    _run([str(_BUILD_VENV_PY), "-m", "pip", "install", "--upgrade", "pip"])
    return _BUILD_VENV_PY


def _ensure_maturin(py: Path) -> None:
    """Install maturin into the build venv if it isn't there."""
    r = subprocess.run([str(py), "-c", "import maturin"],
                       capture_output=True)
    if r.returncode == 0:
        return
    print("Installing maturin into the build venv…")
    _run([str(py), "-m", "pip", "install", "--upgrade", "maturin>=1.5,<2.0"])


def _clean_wheel_dir():
    if WHEEL_OUT.exists():
        for f in WHEEL_OUT.glob("*.whl"):
            try:
                f.unlink()
            except OSError:
                pass


def _maturin_build(py: Path) -> Path:
    """Invoke maturin and return the resulting .whl path."""
    _clean_wheel_dir()
    _run(
        [str(py), "-m", "maturin", "build",
         "--release",
         "--out", str(WHEEL_OUT)],
        cwd=str(CRATE_DIR),
    )
    wheels = sorted(WHEEL_OUT.glob("*.whl"))
    if not wheels:
        raise SystemExit(
            "ERROR: maturin produced no wheel.  "
            "Inspect the output above for the underlying cargo error.")
    return wheels[-1]


def _extract_binary(wheel: Path, dest_dir: Path) -> Path:
    """Pull the compiled .pyd / .so / .dylib out of the wheel.

    The wheel layout is ``hbt_rust_kernels-<ver>-<tags>.whl`` containing
    ``hbt_rust_kernels/hbt_rust_kernels.<ext>`` or
    ``hbt_rust_kernels.<ext>`` at the root.  We pick whichever binary is
    inside and drop it flat in ``dest_dir``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate_exts = (".pyd", ".so", ".dylib")

    with zipfile.ZipFile(wheel, "r") as z:
        members = z.namelist()
        # Prefer abi3-tagged binaries; fall back to any matching extension.
        binaries = [
            m for m in members
            if Path(m).name.startswith("hbt_rust_kernels")
            and Path(m).name.endswith(candidate_exts)
        ]
        if not binaries:
            raise SystemExit(
                "ERROR: wheel contains no hbt_rust_kernels.{pyd,so,dylib}.  "
                "Did Cargo.toml lose the crate-type=cdylib setting?")

        # Heuristic: shortest name first — favours plain
        # `hbt_rust_kernels.abi3.so` over deeper nested paths.
        binaries.sort(key=lambda m: (len(m), m))
        chosen = binaries[0]
        out_name = Path(chosen).name
        dst_path = dest_dir / out_name

        # Wipe stale binaries first so an old abi3-py38 doesn't shadow new.
        for existing in dest_dir.iterdir():
            if (existing.is_file()
                    and existing.name.startswith("hbt_rust_kernels")
                    and existing.name != out_name):
                try:
                    existing.unlink()
                except OSError:
                    pass  # locked by a running process — harmless

        # Windows refuses to OVERWRITE a loaded .pyd, but it does allow
        # RENAMING one out of the way.  Move the in-use file to a side
        # name, then write the new binary into its place.  Streamlit /
        # any running Python keeps the old (now-renamed) handle; the
        # next import sees the fresh file.  On the next clean run any
        # leftover .old-* files are unlinked.
        if dst_path.exists():
            import time as _time
            for stale in dest_dir.glob("hbt_rust_kernels*.old-*"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            try:
                dst_path.unlink()
            except PermissionError:
                side = dst_path.with_suffix(
                    f".old-{int(_time.time())}{dst_path.suffix}")
                try:
                    dst_path.rename(side)
                    print(f"  (note: renamed in-use binary to "
                          f"{side.name} — restart Streamlit to load "
                          "the fresh build.)")
                except OSError as e:
                    raise SystemExit(
                        f"ERROR: cannot replace {dst_path} "
                        f"({type(e).__name__}: {e}).  Close any running "
                        "Streamlit / Python processes that have the "
                        "extension loaded and re-run.")

        with z.open(chosen) as src, dst_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return dst_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not CRATE_DIR.is_dir():
        print(f"ERROR: crate directory missing at {CRATE_DIR}")
        return 1

    if not _have("cargo"):
        print("=" * 64)
        print("  Rust toolchain not found.")
        print()
        print("  Install rustup (the official Rust installer) from:")
        print("      https://rustup.rs/")
        print()
        print("  After installation, open a NEW terminal and re-run:")
        print("      python dev/build_rust_kernels.py")
        print("=" * 64)
        return 1

    py = _ensure_build_venv()
    _ensure_maturin(py)

    print()
    print(f"Building hbt_rust_kernels (release, ABI3-py39) for {_arch_tag()}…")
    wheel = _maturin_build(py)
    print(f"  Built wheel: {wheel.name}")

    bin_dir = BIN_BASE / _arch_tag()
    out = _extract_binary(wheel, bin_dir)

    print()
    print("=" * 64)
    print(f"  Installed binary: {out}")
    print()
    print("  The Python wrapper at tools/rf/ssm/helpers/rust_kernels.py")
    print("  will pick it up automatically on next import.  Verify with:")
    print()
    print("      python tools/rf/ssm/rust_kernels/benchmark.py")
    print()
    print("  To ship to another OS (e.g. Streamlit Cloud Linux):")
    print("    1. Run this script on that OS.")
    print("    2. Commit the new file under")
    print("         tools/rf/ssm/rust_kernels/bin/<platform_arch>/")
    print("       to the repo.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

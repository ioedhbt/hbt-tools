"""
launch_ebl_calculator.py — Standalone bootstrap launcher for the
E-Beam Lithography Calculator.

Creates (or reuses) a local Python virtual environment at .ebl_venv/ next
to this file, installs the packages the EBL calculator needs into it, then
launches the Streamlit app using the venv interpreter.

Run with any system Python 3.9+:
    python launch_ebl_calculator.py

The launcher finds the app at ``ebeam_calculator.py`` beside this file, or
at ``tools/ebeam_calculator.py`` inside the repo — so it works whether you
copy the two files out on their own or run it from the full checkout.

Auto-update from GitHub is scaffolded below but DISABLED (no canonical
repository is wired up yet — see AUTOUPDATE_ENABLED / REPO_* constants).
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV_DIR = ROOT / ".ebl_venv"


def _find_app_file():
    """Locate ebeam_calculator.py: beside this launcher first, then under
    tools/ in the repo layout."""
    for candidate in (ROOT / "ebeam_calculator.py",
                       ROOT / "tools" / "ebeam_calculator.py"):
        if candidate.exists():
            return candidate
    return ROOT / "ebeam_calculator.py"   # default for the error message


APP_FILE = _find_app_file()

# ── Packages the EBL calculator needs (import name, pip name) ─────────────────
# The app imports only these third-party modules (plus stdlib math/os/tempfile).
REQUIRED = [
    ("streamlit",  "streamlit"),
    ("numpy",      "numpy"),
    ("plotly",     "plotly"),
    ("matplotlib", "matplotlib"),
    ("gdstk",      "gdstk"),
]

# ── Resolve venv interpreter paths (Windows vs. Unix) ─────────────────────────
_win = sys.platform == "win32"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if _win else "bin/python")
VENV_STREAMLIT = VENV_DIR / ("Scripts/streamlit.exe" if _win else "bin/streamlit")


# ── Auto-update from a canonical GitHub repo (DISABLED for now) ───────────────
# When enabled, updates pull from the OFFICIAL repository below regardless of
# how the user obtained the files. Two code paths mirror LAUNCH_Tool.py:
#   • git checkout present  → safe `git fetch <canonical> + merge --ff-only`;
#   • no .git/ (zip install) → pure-Python GitHub API + zipball overlay.
# To enable later: set AUTOUPDATE_ENABLED = True and fill in REPO_OWNER /
# REPO_NAME / REPO_BRANCH with the canonical repository.
AUTOUPDATE_ENABLED = False
REPO_OWNER = ""        # e.g. "ioedhbt"
REPO_NAME = ""         # e.g. "hbt-tools"
REPO_BRANCH = ""       # e.g. "IOED-Tools"


def _autoupdate_disabled():
    if not AUTOUPDATE_ENABLED:
        return True
    if not (REPO_OWNER and REPO_NAME and REPO_BRANCH):
        return True
    return os.environ.get("HBT_NO_AUTOUPDATE", "").strip().lower() in {
        "1", "true", "yes", "on"}


def auto_update():
    """Update the install to the latest canonical commit (best-effort).

    Currently a no-op: auto-update is disabled until a canonical GitHub
    repository is configured (see the constants above). The hook is kept
    here so launch order and messaging don't change when it's turned on.
    """
    if _autoupdate_disabled():
        print("Auto-update disabled (no canonical repo configured yet).")
        return
    # Placeholder for the real updater — wire up the git/zip logic from
    # LAUNCH_Tool.py here once REPO_* point at the official repository.
    print(f"Auto-update — source: {REPO_OWNER}/{REPO_NAME}@{REPO_BRANCH}")


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


def _create_venv(fresh=False):
    """Create the venv (deleting an existing one first when ``fresh``)."""
    if fresh and VENV_DIR.exists():
        import shutil
        import stat
        print(f"Removing broken virtual environment at {VENV_DIR} ...")

        def _force_remove(func, path, _exc):
            # Clear read-only flag then retry — common on Windows/OneDrive.
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass  # best-effort; rmtree will surface any real failure

        if sys.version_info >= (3, 12):
            shutil.rmtree(str(VENV_DIR), onexc=_force_remove)
        else:
            shutil.rmtree(str(VENV_DIR), onerror=_force_remove)

    print(f"Creating virtual environment at {VENV_DIR} ...")
    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        # On Debian/Ubuntu the venv module ships in a separate apt package.
        if sys.platform.startswith("linux"):
            print("\nERROR: Failed to create virtual environment.")
            print("On Debian/Ubuntu you may need to install the venv package:")
            print(f"    sudo apt install python3-venv  "
                  f"python{sys.version_info.major}.{sys.version_info.minor}-venv")
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
    """Block on user input only when stdin is a TTY (so double-click
    launches without a terminal don't raise EOFError)."""
    try:
        if sys.stdin and sys.stdin.isatty():
            input(msg)
    except (EOFError, OSError):
        pass


def main():
    if not APP_FILE.exists():
        print(f"ERROR: Cannot find ebeam_calculator.py near {ROOT}")
        print("Place launch_ebl_calculator.py beside ebeam_calculator.py, "
              "or run it from the repo root (tools/ebeam_calculator.py).")
        _pause_if_interactive()
        sys.exit(1)

    # ── Pull the latest version (best-effort; currently a disabled no-op) ─────
    try:
        auto_update()
    except Exception as _upd_exc:        # never let an update hiccup block launch
        print(f"Auto-update skipped ({_upd_exc}).")

    # ── Create or repair venv ─────────────────────────────────────────────────
    # A broken venv (missing pip / CLI entry points) is treated the same as no
    # venv: recreate it so the user never gets stuck.
    if not VENV_PYTHON.exists() or not pip_works():
        _create_venv(fresh=VENV_DIR.exists())
        run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        pip_install([pip for _, pip in REQUIRED])
    else:
        missing = check_missing(REQUIRED)
        if missing:
            print(f"Installing missing packages: {', '.join(missing)}")
            pip_install(missing)
        else:
            print("All required packages are present.")

    # ── Launch the app ────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  Launching E-Beam Lithography Calculator ...")
    print("=" * 50)
    print()

    try:
        subprocess.check_call([str(VENV_STREAMLIT), "run", str(APP_FILE)])
    except KeyboardInterrupt:
        print("\nServer shut down cleanly.")


if __name__ == "__main__":
    main()

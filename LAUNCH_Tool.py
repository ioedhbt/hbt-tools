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


# ── Auto-update from the canonical GitHub repo ────────────────────────────────
# Updates always come from the project's OFFICIAL repository below, regardless
# of how the user obtained the files (zip download, or a git clone of a fork).
# Two code paths:
#   • git checkout present  → safe `git fetch <canonical> + merge --ff-only`
#     (never clobbers a contributor's uncommitted work; needs git, which a
#      cloner already has);
#   • no .git/ (zip install) → pure-Python GitHub API + zipball download via
#     urllib (NO git binary required), overlaid onto the install folder.

REPO_OWNER  = "ioedhbt"
REPO_NAME   = "hbt-tools"
REPO_BRANCH = "IOED-Tools"
CANONICAL_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
_GH_API       = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
_HTTP_UA      = f"{REPO_NAME}-launcher"
_SHA_MARKER   = ROOT / ".hbttools_update_sha"   # last-applied commit (zip path)

# Top-level names the zip overlay must NEVER overwrite/recurse into.
_PROTECT_TOP  = {".hbttools", ".hbttools_build", ".git", _SHA_MARKER.name}


def _autoupdate_disabled():
    return os.environ.get("HBT_NO_AUTOUPDATE", "").strip().lower() in {
        "1", "true", "yes", "on"}


# ---- git checkout path (safe, ff-only, forces the canonical remote) ----------

def _git(args, timeout=30):
    """Run `git -C ROOT <args>`; return (returncode, combined_output) or
    (None, "") when git isn't installed / times out."""
    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, ""


def _update_git_checkout():
    rc, _ = _git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        print("  Git checkout but git isn't available — skipping. "
              "(install git, or set HBT_NO_AUTOUPDATE=1 to silence.)")
        return
    # Never update over uncommitted *tracked* edits (a developer at work).
    rc, dirty = _git(["status", "--porcelain", "--untracked-files=no"])
    if rc == 0 and dirty.strip():
        print("  Local changes detected — skipping auto-update. "
              "(commit/stash, or set HBT_NO_AUTOUPDATE=1 to silence.)")
        return
    print("  Checking for updates…")
    # Fetch the canonical branch explicitly (NOT `origin`), so updates come
    # from the official repo even if the user cloned a fork.
    rc, _ = _git(["fetch", CANONICAL_URL, REPO_BRANCH, "--quiet"], timeout=30)
    if rc != 0:
        print("  Offline or GitHub unreachable — using the local version.")
        return
    rc, counts = _git(["rev-list", "--count", "--left-right",
                       "FETCH_HEAD...HEAD"])
    try:
        behind, ahead = (int(x) for x in counts.split())
    except (ValueError, AttributeError):
        return
    if behind == 0:
        print("  Already up to date.")
        return
    if ahead > 0:
        print(f"  Local branch is ahead by {ahead} commit(s) — skipping "
              "to avoid a merge.")
        return
    print(f"  {behind} new commit(s) available — updating…")
    rc, out = _git(["merge", "--ff-only", "FETCH_HEAD", "--quiet"], timeout=60)
    if rc == 0:
        print("  ✓ Updated to the latest version. "
              "(launcher changes apply next run)")
    else:
        reason = (out.strip().splitlines() or ["unknown error"])[0]
        print(f"  Could not fast-forward ({reason}); keeping current version.")


# ---- zip-install path (pure Python, no git binary needed) --------------------

def _http_get(url, timeout, accept):
    import urllib.request
    req = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": _HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _remote_head_sha():
    """Latest commit SHA of the canonical branch (GitHub REST API)."""
    import json
    raw = _http_get(f"{_GH_API}/commits/{REPO_BRANCH}", 20,
                    "application/vnd.github+json")
    return json.loads(raw.decode("utf-8")).get("sha")


def _stored_sha():
    try:
        return (_SHA_MARKER.read_text(encoding="utf-8").strip() or None)
    except OSError:
        return None


def _overlay_tree(src, dst):
    """Copy every file from `src` over `dst` (add/replace only — never delete),
    skipping protected top-level entries (venv, .git, the SHA marker)."""
    import shutil, stat as _stat
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if rel.parts and rel.parts[0] in _PROTECT_TOP:
            continue
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.exists():
                try:
                    os.chmod(target, _stat.S_IWRITE)   # clear read-only (Win)
                except OSError:
                    pass
            shutil.copy2(item, target)
        except OSError as e:
            # One locked/in-use file shouldn't abort the whole update.
            print(f"    (skipped {rel}: {e})")


def _purge_project_pycache():
    """Drop project __pycache__ so freshly-overlaid .py files aren't shadowed
    by stale bytecode.  The venv's caches are left untouched."""
    import shutil
    targets = [ROOT / "__pycache__"]
    tools = ROOT / "tools"
    if tools.is_dir():
        targets += list(tools.rglob("__pycache__"))
    for pc in targets:
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)


def _update_zip_install():
    import io, zipfile, tempfile
    print("  Checking for updates…")
    try:
        remote = _remote_head_sha()                 # also the connectivity test
    except Exception:
        print("  Offline or GitHub unreachable — using the local version.")
        return
    if not remote:
        return
    if _stored_sha() == remote:
        print("  Already up to date.")
        return
    print(f"  New version available ({remote[:7]}) — downloading…")
    try:
        blob = _http_get(f"{_GH_API}/zipball/{REPO_BRANCH}", 180,
                         "application/zip")
    except Exception as e:
        print(f"  Download failed ({e}); keeping current version.")
        return
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(tdp)
            tops = [p for p in tdp.iterdir() if p.is_dir()]
            if len(tops) != 1:                      # GitHub wraps in one folder
                raise RuntimeError("unexpected archive layout")
            _overlay_tree(tops[0], ROOT)
    except Exception as e:
        print(f"  Update failed ({e}); keeping current version.")
        return
    try:
        _SHA_MARKER.write_text(remote, encoding="utf-8")
    except OSError:
        pass
    _purge_project_pycache()
    print("  ✓ Updated to the latest version. (launcher changes apply next run)")


def auto_update():
    """Update the install to the latest canonical commit (best-effort).

    Picks the git path for git checkouts (safe, never clobbers local edits)
    and the pure-Python zip path otherwise (works with no git installed).
    Any failure is non-fatal — the launcher continues with the current code.
    """
    if _autoupdate_disabled():
        print("Auto-update disabled (HBT_NO_AUTOUPDATE set).")
        return
    print(f"Auto-update — source: {REPO_OWNER}/{REPO_NAME}@{REPO_BRANCH}")
    if (ROOT / ".git").exists():
        _update_git_checkout()
    else:
        _update_zip_install()


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

    # ── Pull the latest version from GitHub (best-effort) ─────────────────────
    # Runs before venv setup so any updated requirements / code are picked up
    # this same launch.  Works with or without git installed; silently skipped
    # when offline, opted out, or when a git checkout has local edits.
    try:
        auto_update()
    except Exception as _upd_exc:        # never let an update hiccup block launch
        print(f"Auto-update skipped ({_upd_exc}).")

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
    # look for the binary on disk in tools/rf/ssm/rust_kernels/bin/<arch>/.
    import platform as _platform
    _arch_machine = (_platform.machine() or "unknown").lower()
    if _win:
        _arch_dir = f"win_{_arch_machine}"
    elif sys.platform == "darwin":
        _arch_dir = f"macosx_{_arch_machine}"
    else:
        _arch_dir = f"linux_{_arch_machine}"
    # NOTE: duplicated literal, deliberately.  This runs before the venv
    # exists, so it cannot import tools.common.paths.  If the crate moves,
    # this and dev/{build_rust_kernels,check_rust_status}.py must move with
    # it — dev/smoke_test.py asserts all three agree.
    _rust_bin_dir = (ROOT / "tools" / "rf" / "ssm" / "rust_kernels" / "bin"
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
                     str(ROOT / "dev" / "build_rust_kernels.py")])
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
            env=_child_env, cwd=str(ROOT))   # cwd: pages load assets relative
                                             # to the repo root, and streamlit
                                             # does not chdir to the script dir
    except KeyboardInterrupt:
        print("\nServer shut down cleanly.")


if __name__ == "__main__":
    # Without this, a failure during first-time setup (pip_install /
    # recreate_venv raising CalledProcessError on a network drop, a full
    # disk, or a bad wheel) printed a traceback and exited — and on a
    # Windows double-click the console closed before it could be read,
    # leaving a half-built .hbttools/ and no visible diagnostic.
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except SystemExit:
        raise
    except BaseException as exc:                                  # noqa: BLE001
        import traceback
        print()
        print("=" * 60)
        print(f"  Setup failed: {type(exc).__name__}: {exc}")
        print("=" * 60)
        traceback.print_exc()
        print()
        print("The environment may be incomplete. Re-running this launcher "
              "repairs it (missing packages are reinstalled).")
        print("If it keeps failing, delete the .hbttools folder and try again.")
        _pause_if_interactive()
        sys.exit(1)

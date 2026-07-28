"""
fit_cache.py — Persistent per-s2p, per-model fitted SSM parameters.

Storage layout (v2, one JSON file per (DUT, model)):

    <cache_root>/
      fits/
        <basename_no_ext>/
          <basename_no_ext>_<model_short>.json    # one file per model
          <basename_no_ext>_<model_short>.json
        <basename_no_ext>/
          ...
      fit_cache.legacy.json   # one-shot rename of the v1 monolithic file

Each per-model file holds a single entry::

    {
      "saved_at": "2026-05-17T12:00:17",
      "params":   {"Cpbe": 1.2e-15, "Rbe": 60.0, ...}   # SI units
    }

Why per-model files instead of one monolithic JSON?

  • Isolation — a corrupt / all-zero save for one model can no longer
    overwrite a sibling model's good cache (the v1 monolith made this
    failure mode possible during a concurrent rerender).
  • Easier manual surgery — users can delete a single bad fit by
    removing one file, without hand-editing JSON.
  • Smaller writes — saving one model touches one small file, not the
    whole cross-DUT dictionary.

The public API surface is unchanged from v1.  Callers still see the
same nested `{fname: {model_short: {saved_at, params}}}` shape from
``load_cache`` / ``export_cache_bytes``; the on-disk split is hidden.

Resolution order for the cache root directory:
  1. ``$HBT_FIT_CACHE_DIR`` if set (expanded).
  2. ``~/.hbt-tools/`` if the home dir is writable.
  3. ``<repo>/.fit_cache/`` fallback (Streamlit Community Cloud).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


_LEGACY_FILENAME = "fit_cache.json"
_LEGACY_RENAMED  = "fit_cache.legacy.json"
_FITS_SUBDIR     = "fits"
_FALLBACK_DIR    = Path(__file__).resolve().parents[3] / ".fit_cache"


# ── Cloud / ephemeral-host detection ─────────────────────────────────────────
#
# Streamlit Community Cloud spins up an ephemeral VM that's recycled after
# any idle period — anything we write to disk evaporates on the next cold
# start, so the "persistent" cache is effectively write-only there.
# Worse, a stale cache entry that survives within a single VM lifetime
# fights the fine-tune section: every rerender of a fine-tune number
# input would have its value re-overwritten by the cached value, so the
# user can't actually edit anything.
#
# Detection rule (in priority order):
#   1. ``HBT_DISABLE_FIT_CACHE=1`` (or true/yes/on)  → explicit OFF
#   2. ``HBT_FIT_CACHE_FORCE_ON=1``                  → explicit ON
#      (escape hatch for testing on Cloud-like hosts)
#   3. ``/mount/src`` exists on disk                 → Streamlit Cloud → OFF
#   4. Otherwise                                     → ON (local default)
#
# When `_CACHE_DISABLED` is True, every public function in this module
# returns the empty / negative result without touching the filesystem
# (load_cache → {}, get_fit → None, save_fit → False, etc.).  The
# callers' UI branches then collapse naturally — no banner, no "Use
# saved" button, no auto-save that would clobber the user's edits.
def _detect_cache_disabled() -> bool:
    env = os.environ.get("HBT_DISABLE_FIT_CACHE", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    force_on = os.environ.get("HBT_FIT_CACHE_FORCE_ON", "").strip().lower()
    if force_on in {"1", "true", "yes", "on"}:
        return False
    # Streamlit Community Cloud-specific path — present on Cloud, absent
    # on every other host we care about.
    try:
        return Path("/mount/src").exists()
    except OSError:
        return False


_CACHE_DISABLED: bool = _detect_cache_disabled()


def is_cache_disabled() -> bool:
    """Public predicate — True when the persistent fit cache is OFF.

    Useful for callers that want to hide cache-only UI elements (e.g.
    the "Use saved" button, the "Re-extract" toggle) instead of just
    letting them no-op silently.
    """
    return _CACHE_DISABLED


def _resolve_cache_dir() -> Path:
    env = os.environ.get("HBT_FIT_CACHE_DIR")
    if env:
        p = Path(env).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass
    home = Path.home() / ".hbt-tools"
    try:
        home.mkdir(parents=True, exist_ok=True)
        return home
    except OSError:
        return _FALLBACK_DIR


def _fits_dir() -> Path:
    d = _resolve_cache_dir() / _FITS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path_str() -> str:
    """Return the cache root directory (the per-DUT folders live in ``fits/``)."""
    return str(_resolve_cache_dir())


def _basename_no_ext(fname: str) -> str:
    """File-system safe DUT identifier — strip directory + ``.s2p`` extension."""
    name = Path(str(fname)).name
    # Strip a trailing .s2p (case-insensitive) so the folder doesn't carry
    # ".s2p" literally — keeps the folder names tidy.
    name = re.sub(r"\.s2p$", "", name, flags=re.IGNORECASE)
    return name


def _norm_key(fname: str) -> str:
    """Legacy normalisation used by the v1 monolith — still returns the
    full basename WITH extension so old in-memory snapshots load cleanly."""
    return Path(str(fname)).name


def _dut_dir(fname: str, create: bool = False) -> Path:
    p = _fits_dir() / _basename_no_ext(fname)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _model_file(fname: str, model_short: str) -> Path:
    base = _basename_no_ext(fname)
    # Per the layout: <dut_dir>/<basename>_<model_short>.json
    safe_short = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_short))
    return _dut_dir(fname) / f"{base}_{safe_short}.json"


def _read_model_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "params" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write `payload` to `path` atomically.

    The temp file gets a unique name.  A fixed `path + ".tmp"` was shared by
    every writer of the same (DUT, model), so two browser tabs saving at once
    could interleave their json.dump calls into one file before either
    replace() ran — corrupting the entry the "atomic" write existed to
    protect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ── One-shot legacy migration ────────────────────────────────────────────────

_MIGRATED_FLAG = False


def _migrate_legacy_if_present() -> None:
    """If the v1 monolithic ``fit_cache.json`` exists, split it into the new
    per-(DUT, model) layout and rename the old file so we never migrate twice.
    Idempotent and silent — safe to call from every entry point."""
    global _MIGRATED_FLAG
    if _MIGRATED_FLAG:
        return
    _MIGRATED_FLAG = True

    root = _resolve_cache_dir()
    legacy = root / _LEGACY_FILENAME
    if not legacy.exists():
        return
    try:
        with legacy.open("r", encoding="utf-8") as f:
            old = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(old, dict):
        return

    for fname, models in old.items():
        if not isinstance(models, dict):
            continue
        for short, entry in models.items():
            if not isinstance(entry, dict) or "params" not in entry:
                continue
            # _SANITY_NONZERO_GATE intentionally NOT applied here — preserve
            # whatever the user had on disk, even if it's all zeros.  They
            # can manually delete the bad file under fits/<dut>/.
            target = _model_file(fname, short)
            if target.exists():
                # Don't clobber an already-migrated file.
                continue
            try:
                _atomic_write_json(target, {
                    "saved_at": entry.get("saved_at",
                                          datetime.now().isoformat(timespec="seconds")),
                    "params":   entry.get("params") or {},
                })
            except OSError:
                continue

    try:
        legacy.rename(root / _LEGACY_RENAMED)
    except OSError:
        # Best-effort — if rename fails the next migration call will redo the
        # work (harmless because we skip already-existing target files).
        pass


# ── Public API ───────────────────────────────────────────────────────────────


def load_cache() -> dict:
    """Aggregate every per-model file under ``fits/`` into the nested dict
    shape ``{basename_no_ext: {model_short: {saved_at, params}}}``.  Keys
    are the canonical (extension-stripped) form; ``get_fit`` / ``save_fit``
    / ``list_fits`` still accept full ``foo.s2p`` filenames and strip the
    extension internally."""
    if _CACHE_DISABLED:
        return {}
    _migrate_legacy_if_present()
    out: dict[str, dict] = {}
    try:
        fits = _fits_dir()
    except OSError:
        return out
    if not fits.exists():
        return out
    for dut_dir in fits.iterdir():
        if not dut_dir.is_dir():
            continue
        base = dut_dir.name
        per_dut: dict[str, dict] = {}
        prefix = f"{base}_"
        for jf in dut_dir.glob("*.json"):
            stem = jf.stem
            if not stem.startswith(prefix):
                continue
            short = stem[len(prefix):]
            entry = _read_model_file(jf)
            if entry is not None:
                per_dut[short] = {
                    "saved_at": entry.get("saved_at"),
                    "params":   entry.get("params") or {},
                }
        if per_dut:
            out[base] = per_dut
    return out


def get_fit(fname: str, model_short: str) -> Optional[dict]:
    """Return cached params (SI units) for (fname, model_short), or None."""
    if _CACHE_DISABLED:
        return None
    _migrate_legacy_if_present()
    entry = _read_model_file(_model_file(fname, model_short))
    if not entry:
        return None
    p = entry.get("params")
    return dict(p) if isinstance(p, dict) else None


def get_fit_timestamp(fname: str, model_short: str) -> Optional[str]:
    if _CACHE_DISABLED:
        return None
    _migrate_legacy_if_present()
    entry = _read_model_file(_model_file(fname, model_short))
    return entry.get("saved_at") if entry else None


def list_fits(fname: str) -> dict:
    """Return ``{model_short: timestamp}`` for the given s2p basename."""
    if _CACHE_DISABLED:
        return {}
    _migrate_legacy_if_present()
    out: dict[str, str] = {}
    d = _dut_dir(fname)
    if not d.exists():
        return out
    base = _basename_no_ext(fname)
    prefix = f"{base}_"
    for jf in d.glob("*.json"):
        stem = jf.stem
        if not stem.startswith(prefix):
            continue
        short = stem[len(prefix):]
        entry = _read_model_file(jf)
        if entry and entry.get("saved_at"):
            out[short] = entry["saved_at"]
    return out


def _all_numeric_zero(params: dict) -> bool:
    """True if every numeric value in ``params`` is exactly 0.0 (or there are
    no numeric values at all).  Used as a sanity gate in ``save_fit`` —
    a freshly-rendered model with all-zero session_state is not worth caching
    and would shadow any later non-zero extraction on the next reload."""
    saw_numeric = False
    for v in params.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            saw_numeric = True
            if float(v) != 0.0:
                return False
    return saw_numeric  # all-numeric-zero → True; no numerics → False (don't gate)


def save_fit(fname: str, model_short: str, params_si: dict) -> bool:
    """Persist a fine-tuned param dict for one (DUT, model).

    Returns True on disk write success.  Returns False (no write) if the
    incoming params look like a fresh / uninitialised render (every
    numeric value is 0.0) — this guards against the self-perpetuating
    all-zero cache that the v1 layout was vulnerable to."""
    if _CACHE_DISABLED:
        return False
    _migrate_legacy_if_present()

    payload: dict = {}
    for k, v in params_si.items():
        if isinstance(v, bool):
            payload[k] = v
        elif isinstance(v, (int, float)):
            payload[k] = float(v)
        elif isinstance(v, str):
            payload[k] = v

    if _all_numeric_zero(payload):
        # Refuse to save the all-zero state — a freshly opened file before
        # extraction has populated session_state can otherwise overwrite a
        # good cache entry on the next rerender.
        return False

    target = _model_file(fname, model_short)
    try:
        _atomic_write_json(target, {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "params":   payload,
        })
        return True
    except OSError:
        return False


def delete_fit(fname: str, model_short: Optional[str] = None) -> bool:
    """Drop the cache for one model (when ``model_short`` is given), or for
    every model under this DUT (when ``None``)."""
    if _CACHE_DISABLED:
        return False
    _migrate_legacy_if_present()
    if model_short is None:
        d = _dut_dir(fname)
        if not d.exists():
            return False
        try:
            for jf in d.glob("*.json"):
                jf.unlink(missing_ok=True)
            # Remove the folder if it's now empty (best-effort).
            try:
                d.rmdir()
            except OSError:
                pass
            return True
        except OSError:
            return False
    p = _model_file(fname, model_short)
    if not p.exists():
        return False
    try:
        p.unlink()
        # If the DUT folder is now empty, drop it too.
        try:
            p.parent.rmdir()
        except OSError:
            pass
        return True
    except OSError:
        return False


def export_cache_bytes() -> bytes:
    """Dump the entire cache as one unified JSON blob (the v1 schema) for
    portability — useful for syncing local ↔ Streamlit Cloud."""
    return json.dumps(load_cache(), indent=2, sort_keys=True).encode("utf-8")


def import_cache_bytes(raw: bytes, merge: bool = True) -> tuple[int, int]:
    """Import a v1-shaped JSON blob (single object keyed by basename).  When
    ``merge`` is False, every existing per-model file is wiped before the
    import (one-shot replace).  Returns ``(n_files, n_fits)``."""
    if _CACHE_DISABLED:
        # No filesystem to write to (or a write-only ephemeral one) —
        # treat import as a successful no-op so the UI doesn't error.
        return (0, 0)
    _migrate_legacy_if_present()
    try:
        incoming = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Cache import failed — not valid UTF-8 JSON ({e}).") from e
    if not isinstance(incoming, dict):
        raise ValueError("Cache import failed — top-level must be a JSON object.")

    if not merge:
        # Clean wipe of fits/ so the import is the only source of truth.
        fits = _fits_dir()
        for dut_dir in fits.iterdir():
            if dut_dir.is_dir():
                for jf in dut_dir.glob("*.json"):
                    try:
                        jf.unlink()
                    except OSError:
                        pass
                try:
                    dut_dir.rmdir()
                except OSError:
                    pass

    n_fits = 0
    touched: set[str] = set()
    for fname, models in incoming.items():
        if not isinstance(models, dict):
            continue
        for short, entry in models.items():
            if not isinstance(entry, dict) or "params" not in entry:
                continue
            target = _model_file(fname, short)
            try:
                _atomic_write_json(target, {
                    "saved_at": entry.get("saved_at",
                                          datetime.now().isoformat(timespec="seconds")),
                    "params":   entry.get("params") or {},
                })
                touched.add(_basename_no_ext(fname))
                n_fits += 1
            except OSError:
                continue
    return len(touched), n_fits


def differs_from(params_si: dict, reference_si: dict,
                 keys: Optional[list] = None, rtol: float = 1e-9,
                 atol: float = 1e-30) -> bool:
    """True if any numeric value in ``params_si`` deviates from ``reference_si``
    (over the given ``keys``, or the intersection of numeric keys when None).
    Used to gate auto-save so we don't persist unedited extraction defaults.
    """
    if keys is None:
        keys = [k for k in params_si
                if isinstance(params_si.get(k), (int, float))
                and isinstance(reference_si.get(k), (int, float))]
    for k in keys:
        a = params_si.get(k)
        b = reference_si.get(k)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if abs(float(a) - float(b)) > atol + rtol * max(abs(float(a)), abs(float(b))):
            return True
    return False

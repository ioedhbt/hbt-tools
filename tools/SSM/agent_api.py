"""
agent_api.py — headless SSM fitting API for AI agents and scripts.

No Streamlit UI is required to use this module: everything a script needs to
load measured S-parameter data, inspect/build an SSM model (built-in or
custom), forward-simulate it, and fit it to data lives here as plain
functions returning dicts / numpy arrays (SI units throughout).  Importing
Streamlit transitively (through `tools.SSM.models`) is fine — it works
without a ScriptRunContext — but this module itself never calls `st.*`.

Public API
----------
load_data(path)                          -> dict (freq, S, z0, header_lines, meta)
list_models()                            -> dict of builtin SHORT -> description
param_specs(model)                       -> list of (key, label, si_scale, unit)
simulate(model, params, freq, z0=50.0, backend="auto") -> S[N,2,2] complex
residuals(S_meas, S_model)               -> dict Total/S11/S12/S21/S22 (%)
fit(data, model, ..., backend="auto")    -> dict (params, residuals, success, n_evals, message, backend)
build_custom_model(base, modifications, name=None) -> CustomModel
add_series_element / add_parallel_element / add_shunt_branch /
add_parallel_to_shunt(model, ...)        -> the same CustomModel (mutated)
save_custom_model(model, path)           -> Path

CLI
---
    python tools/SSM/agent_api.py inspect FILE
    python tools/SSM/agent_api.py models
    python tools/SSM/agent_api.py backend
    python tools/SSM/agent_api.py fit FILE --model T [--initial p.json]
        [--fit-keys a,b,c] [--fixed f.json] [--bounds b.json]
        [--method auto] [--backend auto] [--maxiter 400] [--out result.json]

Compute backend — CUDA > Rust > NumPy
--------------------------------------
`simulate()`/`fit()` take a `backend=` kwarg: `"auto"` (default) picks CUDA
(cupy) if a GPU is available, else the project's Rust kernels
(`helpers/rust_kernels.py`, `HAS_RUST`) if built for this platform, else pure
NumPy; `"cuda"`/`"rust"`/`"numpy"` force a specific backend (falling back —
with a note — when the requested one isn't available). Builtin models route
through `SSMModelTemplate.simulate_batch`, which already dispatches to Rust
(CPU) or accepts `xp=cupy` (GPU); custom models route through
`helpers.rust_kernels.sim_custom_batch` / `custom_model.core.
simulate_custom_model_batch(..., xp=cupy)`, exactly like
`custom_model/ui_fit.py::_make_adapter`. The builtin per-topology Rust
kernels (T/pi/XuT/KY) verify bit-identical to NumPy. The generic
custom-model Rust kernel (`sim_custom_batch`) also matches NumPy
(≤ ~1e-11 on |S|) at physically meaningful values; it only diverges when a
value set is *degenerate* — e.g. an access leg with both R and L exactly 0
stamps chained ±1e12 near-short admittances whose nodal LU solve is so
ill-conditioned that Rust's solver and LAPACK legitimately disagree (neither
answer is authoritative there). Custom-model `backend="auto"` therefore runs
a cheap one-time parity probe per distinct topology at well-conditioned
probe values and silently prefers NumPy on a true topology-level
disagreement (`backend="rust"` explicitly still uses it, with a note).

Header-driven de-embedding awareness
-------------------------------------
Files this app writes (`.s2p`) carry leading `!` comment lines describing
what was done to the data (see `helpers/s2p_io.py::write_s2p`).  A
"Pre-ext override" / "deemb" header means the pad capacitances (Cpbe/Cpce/
Cpbc) and lead inductances (Lb/Lc/Le) have ALREADY been removed from this
S-parameter data — `load_data()` parses that header into `meta`, and `fit()`
freezes those six parameters at 0 automatically unless the caller overrides
`fit_keys`/`fixed` explicitly.  Access resistances (Rb/Rc/Re -> Rpb/Rpc/Rpe)
are NOT removed by that step and remain fittable.  A file with no such
header is raw/measured data — every parasitic should be fitted (or
de-embedded first through the Streamlit app).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Bootstrap sys.path so `python tools/SSM/agent_api.py ...` works from any
# cwd — this file lives at <repo_root>/tools/SSM/agent_api.py, so two
# parents up from its own directory is the repo root.
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import re

import numpy as np
from scipy.optimize import least_squares, minimize

from tools.SSM.helpers import parse_s2p, parse_csv
from tools.SSM.helpers import rust_kernels as _rust_kernels
from tools.SSM.models import REGISTRY
from tools.SSM.models.base_ui import (
    PAD_SPECS, _PARASITIC_KEYS, tune_hard_limits, informed_default_range,
    _canonical_tune_key, _port_residuals, _detect_low_perf_device,
)
import tools.SSM.custom_model.core as custom_core

# ── CUDA detection (same try-import convention as models/base_ui.py) ─────────
try:
    import cupy as _cp
    _HAS_CUDA = True
    _CUDA_VER = f"{_cp.cuda.runtime.runtimeGetVersion():d}"
except Exception:
    _cp = None
    _HAS_CUDA = False
    _CUDA_VER = ""

HAS_RUST = _rust_kernels.HAS_RUST


def _cuda_device_name() -> str:
    """Cheap best-effort CUDA device name for the `backend` CLI report."""
    if not _HAS_CUDA:
        return ""
    try:
        return _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        try:
            return str(_cp.cuda.Device(0))
        except Exception:
            return ""


# ════════════════════════════════════════════════════════════════════════════
# 1. Data loading — .s2p / .csv, with header interpretation
# ════════════════════════════════════════════════════════════════════════════

_HDR_KV_RE = re.compile(
    r"^!\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(.*)$")

# Header keys that describe the pad-cap / lead-L / access-R values used by a
# de-embedding step (see helpers/s2p_io.py::write_s2p and
# models/__init__.py::AbstractSSMModel.get_s2p_header_params — this is the
# exact key set that tool writes to a de-embedded .s2p's "!" header).
_PAD_HEADER_KEYS = ("Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le", "Rb", "Rc", "Re")

_DEEMBED_TOKENS = ("deemb", "pre-ext", "preext", "de-embed")


def _unit_scale(unit: str) -> float:
    """Best-effort unit-string -> SI scale factor for header values."""
    u = unit.strip()
    if not u:
        return 1.0
    if u.endswith("Ω") or u.lower().endswith("ohm"):
        prefix = (u[:-1] if u.endswith("Ω") else u[:-3]).lower()
        return {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}.get(prefix, 1.0)
    table = {
        "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9,
        "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
        "f": 1.0, "ff": 1e-15, "pf": 1e-12, "nf": 1e-9, "uf": 1e-6, "mf": 1e-3,
        "h": 1.0, "ph": 1e-12, "nh": 1e-9, "uh": 1e-6, "mh": 1e-3,
    }
    return table.get(u.lower(), 1.0)


def _read_header_lines(text: str) -> list:
    """Leading '!' comment lines at the top of a Touchstone file (before the
    '#' option line / first data row).  Blank lines are skipped; the first
    non-comment, non-blank line ends the header block."""
    lines = []
    for raw_line in text.splitlines():
        s = raw_line.strip()
        if not s:
            continue
        if s.startswith("!"):
            lines.append(s)
            continue
        break
    return lines


def _parse_header_values(header_lines: list) -> dict:
    """Parse '!  KEY = NUMBER UNIT' header lines -> {KEY: SI value}."""
    out = {}
    for line in header_lines:
        m = _HDR_KV_RE.match(line)
        if not m:
            continue
        key, num, unit = m.group(1), m.group(2), m.group(3).strip()
        try:
            out[key] = float(num) * _unit_scale(unit)
        except ValueError:
            continue
    return out


def _interpret_header(header_lines: list, filename: str) -> dict:
    """Build the `meta` dict load_data() returns from a file's leading '!'
    header comments (+ filename).  See module docstring for the semantics."""
    header_text = " ".join(header_lines).lower()
    name_l = filename.lower()
    deembedded = ("deemb" in name_l
                  or any(tok in header_text for tok in _DEEMBED_TOKENS))

    values = _parse_header_values(header_lines)
    removed_params = {k: values[k] for k in _PAD_HEADER_KEYS if k in values}

    if deembedded:
        status = (
            "De-embedded / pre-extraction-override file: the header shows pad "
            "capacitances (Cpbe/Cpce/Cpbc) and lead inductances (Lb/Lc/Le) "
            "already removed from this S-parameter data — fit() freezes those "
            "six at 0 by default. Access resistances (header Rb/Rc/Re -> model "
            "Rpb/Rpc/Rpe) were NOT removed and remain fittable.")
    elif header_lines:
        status = ("Header comments present but do not indicate de-embedding — "
                   "treating as raw/measured data; fit all parasitics.")
    else:
        status = ("No '!' header found — raw/measured data; all parasitics "
                  "(pad caps, lead inductances, access resistances) should be "
                  "fitted, or de-embedded first with the Streamlit app.")

    meta = {"deembedded": bool(deembedded), "removed_params": removed_params,
            "status": status}
    if header_lines:
        meta["header_title"] = header_lines[0].lstrip("!").strip()
    return meta


def load_data(path) -> dict:
    """Load a `.s2p` or `.csv` S-parameter file.

    Returns dict: freq (Hz, (N,) real), S ((N,2,2) complex), z0 (float),
    header_lines (list of the leading '!' comment lines, [] for csv), meta
    (see `_interpret_header` / module docstring — notably `meta['deembedded']`
    and `meta['removed_params']`, used by `fit()` to auto-freeze parasitics
    that a previous de-embedding step already removed).
    """
    p = _Path(path)
    raw = p.read_bytes()
    ext = p.suffix.lower()
    text = raw.decode("utf-8", errors="ignore")

    if ext == ".s2p":
        freq, S, z0 = parse_s2p(raw)
        header_lines = _read_header_lines(text)
    elif ext == ".csv":
        freq, S, z0 = parse_csv(text)
        header_lines = []
    else:
        raise ValueError(f"load_data: unsupported extension {ext!r} "
                         f"(expected .s2p or .csv)")

    meta = _interpret_header(header_lines, p.name)
    return {
        "freq": np.asarray(freq, dtype=float),
        "S": np.asarray(S, dtype=complex),
        "z0": float(z0),
        "header_lines": header_lines,
        "meta": meta,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Model catalogue — builtins (REGISTRY) + custom models
# ════════════════════════════════════════════════════════════════════════════

# Display (SI-scale, unit) per custom-model component "kind" — mirrors the
# fF/pH/Ω/ps/mS convention used throughout models/*.py (_PARAM_DISPLAY, etc.)
_KIND_SCALE_UNIT = {
    "R": (1.0, "Ω"), "L": (1e12, "pH"), "C": (1e15, "fF"),
    "gm": (1e3, "mS"), "tau": (1e12, "ps"), "alpha": (1.0, ""),
}

# The two canonical resistances the builtin models display in kΩ rather than
# Ω (cheng.py/xu.py _INT_SPECS: Rbc, Rbcx) — override the generic kind-based
# scale above so a custom-model component named e.g. "Rbcx" gets the same
# TUNE_DEFAULT_RANGES-compatible scale as the builtin spec tables use.
_CANON_SCALE_OVERRIDE = {"Rbc": (1e-3, "kΩ"), "Rbcx": (1e-3, "kΩ")}

# Keys that get a zero SI default (physically-optional parasitics) rather
# than a TUNE_DEFAULT_RANGES mid-range guess.
_ZERO_DEFAULT_CANON = frozenset(
    {"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le", "Rpb", "Rpc", "Rpe"})


def _resolve_custom_model(model) -> "custom_core.CustomModel":
    """Coerce `model` (path / JSON string / dict / CustomModel) -> CustomModel."""
    if isinstance(model, custom_core.CustomModel):
        return model
    if isinstance(model, dict):
        return custom_core.CustomModel.from_dict(model)
    try:
        return custom_core.load_model(model)
    except Exception as e:
        raise ValueError(
            f"Could not resolve {model!r} as a builtin model short "
            f"({list(REGISTRY)}) or a custom model (path to .json / JSON "
            f"string / dict / CustomModel instance): {e}") from e


def _model_context(model):
    """Dispatch `model` -> (is_builtin, cls, cm, plan, spec_list).

    spec_list is [(key, label, si_scale, unit), ...] describing every
    parameter the model accepts, in SI units (value = display / si_scale is
    wrong; display = SI * si_scale, matching PAD_SPECS's own convention).
    """
    if isinstance(model, str) and model in REGISTRY:
        cls = REGISTRY[model]
        pad_specs = (cls._TUNING_PAD_SPECS
                     if getattr(cls, "_TUNING_PAD_SPECS", None) is not None
                     else PAD_SPECS)
        raw_specs = (list(pad_specs) + list(getattr(cls, "_EXT_SPECS", []))
                    + list(getattr(cls, "_INT_SPECS", [])))
        spec_list = [(k, lbl, sc, u) for k, lbl, sc, u, *_ in raw_specs]
        return True, cls, None, None, spec_list

    cm = _resolve_custom_model(model)
    plan = custom_core.compile_plan(cm)
    spec_list = []
    for key, kind, label in cm.all_value_specs():
        scale, unit = _KIND_SCALE_UNIT.get(kind, (1.0, ""))
        canon = _canonical_tune_key(key, label)
        if canon in _CANON_SCALE_OVERRIDE:
            scale, unit = _CANON_SCALE_OVERRIDE[canon]
        spec_list.append((key, label, scale, unit))
    return False, None, cm, plan, spec_list


def list_models() -> dict:
    """Built-in model SHORT -> full name (from `tools.SSM.models.REGISTRY`),
    plus a `_note` explaining custom-model support."""
    out = {short: cls.NAME for short, cls in REGISTRY.items()}
    out["_note"] = (
        "Custom models are also supported everywhere a `model` argument is "
        "accepted: pass a path to a saved custom_model .json, a JSON string, "
        "a plain dict (CustomModel.to_dict() shape), or a CustomModel "
        "instance directly (see build_custom_model()).")
    return out


def param_specs(model) -> list:
    """Every parameter `model` accepts, as (key, label, si_scale, unit).

    `model` is a builtin SHORT ("T"/"pi"/"XuT"/"KY") or anything
    `simulate()`/`fit()` accept for a custom model.
    """
    _is_builtin, _cls, _cm, _plan, spec_list = _model_context(model)
    return spec_list


def _default_params_for(model, low_perf: bool = False) -> dict:
    """Sensible SI-unit starting params for `model`: 0 for pad caps / lead
    inductances / access resistances, TUNE_DEFAULT_RANGES (or, when
    `low_perf`, TUNE_LOW_PERF_RANGES) mid-range for everything else
    (intrinsic + extrinsic)."""
    is_builtin, cls, _cm, _plan, spec_list = _model_context(model)
    zero_keys = set()
    if is_builtin:
        pad_specs = (cls._TUNING_PAD_SPECS
                     if getattr(cls, "_TUNING_PAD_SPECS", None) is not None
                     else PAD_SPECS)
        zero_keys = {k for k, *_ in pad_specs}

    out = {}
    for key, label, scale, _unit in spec_list:
        canon = _canonical_tune_key(key, label)
        if key in zero_keys or canon in _ZERO_DEFAULT_CANON:
            out[key] = 0.0
            continue
        lo, _step, hi = informed_default_range(key, label, 0.0, low_perf=low_perf)
        mid = (lo + hi) / 2.0
        out[key] = mid / scale if scale else mid
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Compute-backend resolution — CUDA > Rust > NumPy
# ════════════════════════════════════════════════════════════════════════════

_BACKENDS = ("auto", "cuda", "rust", "numpy")

# Custom-model Rust parity probe.  `sim_custom_batch` matches NumPy at
# physically meaningful values; it only diverges on degenerate value sets
# (an access leg with R and L both exactly 0 stamps chained ±1e12 near-short
# admittances — the LU solve is then ill-conditioned enough that Rust and
# LAPACK legitimately disagree, and neither is authoritative).  The probe
# therefore floors zero-valued probe inputs to small benign values per kind
# so it tests kernel algebra, not that degenerate corner.  Cache the verdict
# per distinct topology (structural hash of CustomModel.to_dict()) so
# `backend="auto"` only pays the one-time probe cost once per topology, not
# once per fit-iteration.
_CUSTOM_RUST_TRUST: dict = {}
_CUSTOM_RUST_PROBE_FREQ = np.array(
    [1e8, 5e8, 1e9, 2e9, 3e9, 4e9, 5e9, 8e9], dtype=float)


def _custom_model_hash(cm) -> str:
    import hashlib
    blob = json.dumps(cm.to_dict(), sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _custom_rust_is_trustworthy(cm, plan) -> bool:
    """One-time (cached) probe: does the Rust `sim_custom_batch` kernel agree
    with the NumPy reference for this topology?  Probes at well-conditioned
    values — zero defaults are floored to small benign values per component
    kind, because an all-zero access leg stamps chained ±1e12 near-shorts
    whose ill-conditioned LU solve makes the two backends legitimately
    disagree (see module docstring).  Returns False (skip Rust) on a true
    topology-level disagreement, or when Rust isn't available."""
    if not HAS_RUST:
        return False
    key = _custom_model_hash(cm)
    if key in _CUSTOM_RUST_TRUST:
        return _CUSTOM_RUST_TRUST[key]

    defaults = _default_params_for(cm)
    kind_by_id = {vid: kind for vid, kind, _lbl in cm.all_value_specs()}
    zero_floor = {"R": 1.0, "L": 1e-12, "C": 1e-15,
                  "tau": 1e-12, "gm": 1e-3, "alpha": 0.9}
    probe_vals = {}
    for k in plan.value_keys:
        v = defaults.get(k, 0.0)
        if not v:
            v = zero_floor.get(kind_by_id.get(k, ""), 1e-3)
        probe_vals[k] = v
    try:
        S_np = custom_core.simulate_custom_model_batch(
            plan, _CUSTOM_RUST_PROBE_FREQ, probe_vals, z0=50.0, xp=np)

        def _np_fb(p, f, zz):
            return custom_core.simulate_custom_model_batch(plan, f, p, zz, xp=np)

        S_rust = _rust_kernels.sim_custom_batch(
            plan, probe_vals, _CUSTOM_RUST_PROBE_FREQ, 50.0, np_fallback=_np_fb)
        trust = bool(np.allclose(S_rust, S_np, rtol=1e-9, atol=1e-12))
    except Exception:
        trust = False
    _CUSTOM_RUST_TRUST[key] = trust
    return trust


def _resolve_backend(backend: str, *, is_builtin: bool, cm=None, plan=None):
    """`backend` ("auto"/"cuda"/"rust"/"numpy") -> (resolved, note).

    `resolved` is always one of "cuda"/"rust"/"numpy" — the backend actually
    used.  `note` is None unless a fallback happened (explicit request for an
    unavailable/untrustworthy backend), in which case it explains why.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")

    rust_ok = HAS_RUST if is_builtin else (
        HAS_RUST and (cm is None or _custom_rust_is_trustworthy(cm, plan)))
    rust_untrusted = (not is_builtin) and HAS_RUST and not rust_ok

    if backend == "cuda":
        if _HAS_CUDA:
            return "cuda", None
        if rust_ok:
            return "rust", "cuda requested but unavailable on this machine; using rust"
        return "numpy", "cuda requested but unavailable on this machine; using numpy"

    if backend == "rust":
        if HAS_RUST:
            note = ("rust requested for a custom model whose Rust kernel "
                    "disagreed with NumPy in the parity probe — results may "
                    "be inaccurate; consider backend='numpy'"
                    if rust_untrusted else None)
            return "rust", note
        return "numpy", "rust requested but not built/loaded on this machine; using numpy"

    if backend == "numpy":
        return "numpy", None

    # auto
    if _HAS_CUDA:
        return "cuda", None
    if rust_ok:
        return "rust", None
    return "numpy", None


def backend_status() -> dict:
    """Snapshot of compute-backend availability for the `backend` CLI command."""
    resolved, _note = _resolve_backend("auto", is_builtin=True)
    diag = _rust_kernels.rust_diagnostic()
    return {
        "cuda_available": _HAS_CUDA,
        "cuda_device": _cuda_device_name() if _HAS_CUDA else None,
        "rust": diag,
        "auto_resolves_to": resolved,
        "priority": "cuda > rust > numpy",
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Forward simulation
# ════════════════════════════════════════════════════════════════════════════

def _dispatch_simulate(is_builtin, cls, plan, full_params, freq, z0, resolved):
    """Low-level simulate given an already-RESOLVED backend
    ("cuda"/"rust"/"numpy"). Returns (N,2,2) on the host (numpy)."""
    if is_builtin:
        if resolved == "cuda":
            S = cls.simulate_batch(full_params, freq, z0=z0, xp=_cp)
            return np.asarray(_cp.asnumpy(S[0]))
        if resolved == "rust":
            # simulate_batch auto-dispatches to Rust on the CPU (xp=np) path
            # when HAS_RUST + phase2 dispatch are enabled (the normal case).
            S = cls.simulate_batch(full_params, freq, z0=z0, xp=np)
            return np.asarray(S[0])
        # numpy — bypass simulate_batch's Rust auto-dispatch explicitly.
        S = cls._SIM_WRAP_BATCH_FN(cls._Y_INT_BATCH_FN, full_params, freq, z0, np, None)
        return np.asarray(S[0])

    values = {k: full_params.get(k, 0.0) for k in plan.value_keys}
    if resolved == "cuda":
        S = custom_core.simulate_custom_model_batch(plan, freq, values, z0=z0, xp=_cp)
        return np.asarray(_cp.asnumpy(S[0]))
    if resolved == "rust":
        def _np_fb(p, f, zz):
            return custom_core.simulate_custom_model_batch(plan, f, p, zz, xp=np)
        S = _rust_kernels.sim_custom_batch(plan, values, freq, z0, np_fallback=_np_fb)
        return np.asarray(S[0])
    # numpy
    S = custom_core.simulate_custom_model_batch(plan, freq, values, z0=z0, xp=np)
    return np.asarray(S[0])


def simulate(model, params: dict, freq, z0: float = 50.0,
            backend: str = "auto") -> np.ndarray:
    """Forward-simulate S[N,2,2] from `model` + `params` (SI units) at `freq`
    (Hz).  `model` is a builtin SHORT ("T"/"pi"/"XuT"/"KY") or a custom model
    (path to a saved .json / JSON string / dict / CustomModel instance).
    Missing keys in `params` fall back to `_default_params_for(model)`.

    `backend`: "auto" (default, CUDA > Rust > NumPy — see module docstring),
    "cuda", "rust", or "numpy". Falls back gracefully when the requested
    backend is unavailable/untrustworthy for this model.
    """
    freq = np.asarray(freq, dtype=float)
    is_builtin, cls, cm, plan, _spec_list = _model_context(model)
    full = _default_params_for(model)
    full.update({k: float(v) for k, v in (params or {}).items()})

    resolved, _note = _resolve_backend(backend, is_builtin=is_builtin, cm=cm, plan=plan)
    return _dispatch_simulate(is_builtin, cls, plan, full, freq, z0, resolved)


def residuals(S_meas, S_model) -> dict:
    """Per-port + Total residual (%) — the exact metric the Streamlit UI
    shows (thin wrapper around `models.base_ui._port_residuals`)."""
    out = _port_residuals(np.asarray(S_meas, dtype=complex),
                          np.asarray(S_model, dtype=complex))
    return {k: float(v) for k, v in out.items()}


def _residual_vector(S_meas, S_mod) -> np.ndarray:
    """Stacked real/imag residual vector, normalised per-port exactly like
    `_port_residuals` (so `sum(vec**2)` for one port's slice equals
    `(port_residual_pct/100)**2`).  Used by the `least_squares` fit method."""
    parts = []
    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sm, sk = S_meas[:, r, c], S_mod[:, r, c]
        denom = float(np.sqrt(np.sum(np.abs(sm) ** 2)))
        parts.append((sm - sk) / denom if denom > 0 else np.zeros_like(sm))
    vec = np.concatenate(parts)
    return np.concatenate([vec.real, vec.imag])


# ════════════════════════════════════════════════════════════════════════════
# 5. Fitting
# ════════════════════════════════════════════════════════════════════════════

def _resolve_bounds(key, label, current_si, scale, user_bounds, low_perf=False):
    """(lo, hi) in SI units for one fit variable: explicit `user_bounds`
    entry, else `informed_default_range` widened to include the current
    value and clamped into `tune_hard_limits`, else a generous fallback box."""
    if key in user_bounds:
        lo, hi = user_bounds[key]
        return float(lo), float(hi)

    cur_disp = current_si * scale if scale else current_si
    lo_disp, _step, hi_disp = informed_default_range(key, label, cur_disp,
                                                      low_perf=low_perf)
    if hi_disp <= lo_disp:
        hard_lo, hard_hi = tune_hard_limits(key, label)
        lo_disp = hard_lo if hard_lo is not None else 0.0
        hi_disp = (hard_hi if hard_hi is not None
                  else max(lo_disp + 1.0, abs(cur_disp) * 10.0 + 1.0))
    lo = lo_disp / scale if scale else lo_disp
    hi = hi_disp / scale if scale else hi_disp
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 1e-3
    return float(lo), float(hi)


def fit(data: dict, model, initial: dict = None, fit_keys: list = None,
        fixed: dict = None, bounds: dict = None, method: str = "auto",
        maxiter: int = 400, backend: str = "auto") -> dict:
    """Fit `model` to measured S-parameter `data` (as returned by
    `load_data()`, or any dict with `freq`/`S`/optionally `z0`/`meta`).

    Parameters
    ----------
    model     : builtin SHORT, or a custom model (path/.json string/dict/
                CustomModel instance) — same as `simulate()`.
    initial   : optional dict of SI-unit starting values overriding the
                physics-informed defaults (`_default_params_for`).
    fit_keys  : optional list of parameter keys to optimise. Default: every
                parameter `model` declares, EXCEPT that when
                `data['meta']['deembedded']` is True (and neither `fit_keys`
                nor `fixed` was supplied), the six header-removed parasitics
                (Cpbe/Cpce/Cpbc/Lb/Lc/Le) are automatically frozen at 0.
    fixed     : optional dict of {key: SI value} held constant (not fit).
                Supplying `fit_keys` or `fixed` disables the automatic
                de-embed freeze above — you're in full control.
    bounds    : optional dict of {key: (lo, hi)} in SI units, overriding the
                `tune_hard_limits`/`informed_default_range`-derived default
                box for that key.
    method    : "auto" (Nelder-Mead, bounded, deterministic — minimises the
                exact Total residual %% shown in the app), "nelder-mead", or
                "least_squares" (scipy trf on the normalised per-port
                residual vector — faster for many free parameters).
    maxiter   : iteration / function-eval budget passed to the optimizer.
    backend   : "auto" (default, CUDA > Rust > NumPy), "cuda", "rust", or
                "numpy" — see module docstring. Resolved ONCE up front and
                reused for every objective evaluation (the plan/backend
                choice is not re-decided per iteration).

    Returns
    -------
    dict: params (full SI dict), residuals (Total/S11/S12/S21/S22 %%),
    success (bool), n_evals (int), message (str), backend (the resolved
    backend actually used), and backend_note (str, only present when the
    requested backend fell back to something else).
    """
    freq = np.asarray(data["freq"], dtype=float)
    S_meas = np.asarray(data["S"], dtype=complex)
    z0 = float(data.get("z0", 50.0))
    meta = data.get("meta", {}) or {}

    is_builtin, cls, cm, plan, spec_list = _model_context(model)
    spec_by_key = {k: (lbl, sc) for k, lbl, sc, _u in spec_list}
    all_keys = [k for k, *_ in spec_list]

    resolved_backend, backend_note = _resolve_backend(
        backend, is_builtin=is_builtin, cm=cm, plan=plan)

    def _simulate_for_fit(p):
        return _dispatch_simulate(is_builtin, cls, plan, p, freq, z0, resolved_backend)

    # Slow/lossy devices (fmax < fT, or fT below ~40 GHz) sit at much higher
    # Cbc/Rbi and slower tau — widen the starting guess + bounds accordingly
    # (same heuristic the Streamlit tuning UI uses). Best-effort: never lets
    # a metrics hiccup abort the fit.
    try:
        low_perf = bool(_detect_low_perf_device(S_meas, freq))
    except Exception:
        low_perf = False

    params = _default_params_for(model, low_perf=low_perf)
    if initial:
        params.update({k: float(v) for k, v in initial.items()})

    fixed = dict(fixed) if fixed else {}
    user_specified = (fit_keys is not None) or bool(fixed)
    fit_keys = list(all_keys) if fit_keys is None else list(fit_keys)

    if not user_specified and meta.get("deembedded"):
        for k in all_keys:
            label, _sc = spec_by_key.get(k, ("", 1.0))
            if _canonical_tune_key(k, label) in _PARASITIC_KEYS:
                fixed[k] = 0.0

    fit_keys = [k for k in fit_keys if k not in fixed]
    for k, v in fixed.items():
        params[k] = float(v)

    if method not in ("auto", "nelder-mead", "least_squares"):
        raise ValueError(f"fit: unknown method {method!r} — use 'auto', "
                         f"'nelder-mead' or 'least_squares'")

    if not fit_keys:
        S_mod = _simulate_for_fit(params)
        result = {
            "params": {k: float(v) for k, v in params.items()},
            "residuals": residuals(S_meas, S_mod),
            "success": True,
            "n_evals": 0,
            "message": "nothing to fit — fit_keys is empty after freezing/fixed",
            "backend": resolved_backend,
        }
        if backend_note:
            result["backend_note"] = backend_note
        return result

    user_bounds = dict(bounds) if bounds else {}
    lo_arr, hi_arr, x0 = [], [], []
    for k in fit_keys:
        label, scale = spec_by_key.get(k, ("", 1.0))
        lo, hi = _resolve_bounds(k, label, params.get(k, 0.0), scale,
                                 user_bounds, low_perf=low_perf)
        cur = min(max(float(params.get(k, 0.0)), lo), hi)
        lo_arr.append(lo); hi_arr.append(hi); x0.append(cur)
    x0 = np.array(x0, dtype=float)

    def _apply(x):
        p = dict(params)
        for k, v in zip(fit_keys, x):
            p[k] = float(v)
        return p

    def _obj_total(x):
        try:
            S_mod = _simulate_for_fit(_apply(x))
        except Exception:
            return 1e6
        if not np.all(np.isfinite(S_mod)):
            return 1e6
        return _port_residuals(S_meas, S_mod)["Total"]

    def _obj_vec(x):
        try:
            S_mod = _simulate_for_fit(_apply(x))
        except Exception:
            return np.full(16 * len(freq), 1e2)
        if not np.all(np.isfinite(S_mod)):
            return np.full(16 * len(freq), 1e2)
        return _residual_vector(S_meas, S_mod)

    # Fit variables span ~1e-16 (τ, F) to ~1e9 (Ω) in SI units — optimizing
    # in raw SI space breaks both optimizers (trust region / simplex geometry
    # dominated by the large-magnitude variables), so scale per variable.
    lo_np, hi_np = np.array(lo_arr), np.array(hi_arr)
    span = np.where(hi_np - lo_np > 0, hi_np - lo_np, 1.0)

    use_method = "nelder-mead" if method == "auto" else method
    if use_method == "nelder-mead":
        # Normalized space: z ∈ [0,1] per variable.  (Named zstart, NOT z0 —
        # that would shadow the reference impedance captured by
        # _simulate_for_fit.)
        zstart = (x0 - lo_np) / span

        def _obj_total_z(z):
            return _obj_total(lo_np + np.clip(z, 0.0, 1.0) * span)

        res = minimize(_obj_total_z, zstart, method="Nelder-Mead",
                       bounds=[(0.0, 1.0)] * len(zstart),
                       options={"maxiter": int(maxiter),
                                "maxfev": int(maxiter) * 4,
                                "xatol": 1e-9, "fatol": 1e-6,
                                "adaptive": True})
        x_best = lo_np + np.clip(res.x, 0.0, 1.0) * span
    else:
        res = least_squares(_obj_vec, x0, bounds=(lo_np, hi_np),
                            max_nfev=int(maxiter), method="trf",
                            x_scale=np.maximum(np.abs(x0), span * 1e-3))
        x_best = res.x

    # The optimizers report their last iterate; never return something worse
    # than the (clamped) starting point the caller supplied.
    if _obj_total(x_best) > _obj_total(x0):
        x_best = x0

    params = _apply(x_best)
    S_mod = _simulate_for_fit(params)

    result = {
        "params": {k: float(v) for k, v in params.items()},
        "residuals": residuals(S_meas, S_mod),
        "success": bool(res.success),
        "n_evals": int(getattr(res, "nfev", 0)),
        "message": str(res.message),
        "backend": resolved_backend,
    }
    if backend_note:
        result["backend_note"] = backend_note
    return result


# ════════════════════════════════════════════════════════════════════════════
# 6. Custom-model creation helpers
# ════════════════════════════════════════════════════════════════════════════

_BASE_ALIASES = {
    "cheng t": "Cheng — T (current-source T HBT)",
    "t": "Cheng — T (current-source T HBT)",
    "cheng pi": "Cheng — π (hybrid-π HBT)",
    "cheng π": "Cheng — π (hybrid-π HBT)",
    "pi": "Cheng — π (hybrid-π HBT)",
    "xu t": "Xu — T (Rbcx∥Cbcx HBT)",
    "xut": "Xu — T (Rbcx∥Cbcx HBT)",
    "kun-yang": "Kun-Yang — HEMT (π)",
    "kunyang": "Kun-Yang — HEMT (π)",
    "ky": "Kun-Yang — HEMT (π)",
}


def _resolve_base_label(base: str) -> str:
    if base in custom_core.BUILTIN_PRESETS:
        return base
    key = _BASE_ALIASES.get(str(base).strip().lower())
    if key is None:
        raise ValueError(
            f"build_custom_model: unknown base {base!r}; use one of "
            f"{list(custom_core.BUILTIN_PRESETS)} or an alias like "
            f'"Cheng T" / "Cheng pi" / "Xu T" / "Kun-Yang"')
    return key


def build_custom_model(base: str = "Cheng T", modifications: list = None,
                       name: str = None) -> "custom_core.CustomModel":
    """Start a `CustomModel` from a built-in topology and apply optional
    edits — a thin wrapper over `custom_model.core.builtin_custom_model`.

    `base` : "Cheng T" | "Cheng pi" (or "Cheng π") | "Xu T" | "Kun-Yang"
             — or any full `custom_model.core.BUILTIN_PRESETS` label.
    `modifications` : optional list of operation dicts, each one of:
        {"op": "add_series",   "network": <name>, "kind": "R"|"L"|"C", "name": <label>}
        {"op": "add_parallel", "network": <name>, "kind": "R"|"L"|"C", "name": <label>}
        {"op": "add_shunt",    "section": "extrinsic"|"parasitic",
         "place": "p1-p2"|"p1-gnd"|"p2-gnd", "components": [[("C","Cx")], ...]}
        {"op": "parallel_to_shunt", "section": ..., "place": ...,
         "kind": "R"|"L"|"C", "name": <label>}
    `network` names are the editable `CustomModel` Networks: "intrinsic_base",
    "intrinsic_be", "intrinsic_bc", "intrinsic_ce", "port1", "port2", "emitter".

    Equivalent, more explicit form — call the standalone helpers directly on
    the returned model (each mutates it in place and returns it):

        m = build_custom_model("Cheng T")
        add_parallel_element(m, "intrinsic_ce", "C", "Cce")      # Cce, C<->E
        add_parallel_to_shunt(m, "extrinsic", "p1-p2", "R", "Rbcx")  # Rbcx || Cbcx
        save_custom_model(m, "my_model.json")

    `fit()` / `simulate()` accept the returned `CustomModel` object directly
    — no need to save/reload it first.
    """
    label = _resolve_base_label(base)
    model = custom_core.builtin_custom_model(label)
    if name:
        model.name = name
    for spec in (modifications or []):
        _apply_modification(model, spec)
    return model


def add_series_element(model, network_name: str, kind: str, name: str,
                       group_index: int = None):
    """Append a new series group containing one R/L/C element to a named
    editable Network on `model` (e.g. `network_name="port1"` adds a new
    series step between the intrinsic base and the base access leg).
    `network_name` is one of "intrinsic_base"/"intrinsic_be"/"intrinsic_bc"/
    "intrinsic_ce"/"port1"/"port2"/"emitter". Mutates and returns `model`."""
    net = getattr(model, network_name)
    el = custom_core.Element(kind=kind, name=name)
    if group_index is None or group_index >= len(net.groups):
        net.groups.append([el])
    else:
        net.groups[group_index].append(el)
    return model


def add_parallel_element(model, network_name: str, kind: str, name: str,
                         group_index: int = -1):
    """Add an R/L/C element IN PARALLEL with an existing group of a named
    Network (default: the last group; creates one if the Network is empty).
    E.g. `add_parallel_element(model, "intrinsic_ce", "C", "Cce")` adds a Cce
    shunt cap across the intrinsic collector-emitter junction. Mutates and
    returns `model`."""
    net = getattr(model, network_name)
    if not net.groups:
        net.groups.append([])
        group_index = -1
    net.groups[group_index].append(custom_core.Element(kind=kind, name=name))
    return model


def add_shunt_branch(model, section: str, place: str, *groups):
    """Append a brand-new `ShuntBranch` to `model.extrinsic` (taps the inner
    base/collector nodes, Cheng-style Cbex/Cbcx) or `model.parasitic` (true
    pad-to-ground caps).  `place` is "p1-p2" | "p1-gnd" | "p2-gnd".  Each of
    `*groups` is a parallel-group spec — a list of (kind, name) tuples, e.g.
    `add_shunt_branch(m, "parasitic", "p2-gnd", [("C", "Cextra")])`.
    Mutates and returns `model`."""
    if section not in ("extrinsic", "parasitic"):
        raise ValueError('add_shunt_branch: section must be '
                         '"extrinsic" or "parasitic"')
    branch = custom_core._shunt(place, *groups)
    getattr(model, section).append(branch)
    return model


def add_parallel_to_shunt(model, section: str, place: str, kind: str, name: str,
                          branch_index: int = 0, group_index: int = 0):
    """Add an R/L/C element in parallel within an EXISTING `ShuntBranch`'s
    group — e.g. `add_parallel_to_shunt(model, "extrinsic", "p1-p2", "R",
    "Rbcx")` adds Rbcx in parallel with the Cbcx cap already sitting in a
    Cheng-T/π model's p1-p2 extrinsic branch. Mutates and returns `model`."""
    branches = [b for b in getattr(model, section) if b.place == place]
    if not branches:
        raise ValueError(f"add_parallel_to_shunt: no {section!r} branch at "
                         f"place={place!r} — use add_shunt_branch() first")
    branch = branches[branch_index]
    if not branch.network.groups:
        branch.network.groups.append([])
    branch.network.groups[group_index].append(
        custom_core.Element(kind=kind, name=name))
    return model


def _apply_modification(model, spec: dict):
    op = spec.get("op")
    if op == "add_series":
        add_series_element(model, spec["network"], spec["kind"], spec["name"],
                           group_index=spec.get("group_index"))
    elif op == "add_parallel":
        add_parallel_element(model, spec["network"], spec["kind"], spec["name"],
                             group_index=spec.get("group_index", -1))
    elif op == "add_shunt":
        add_shunt_branch(model, spec.get("section", "extrinsic"), spec["place"],
                         *spec["components"])
    elif op == "parallel_to_shunt":
        add_parallel_to_shunt(model, spec.get("section", "extrinsic"),
                              spec["place"], spec["kind"], spec["name"],
                              branch_index=spec.get("branch_index", 0),
                              group_index=spec.get("group_index", 0))
    else:
        raise ValueError(f"build_custom_model: unknown modification op {op!r}")
    return model


def save_custom_model(model, path) -> _Path:
    """Serialise `model` (a CustomModel) to an explicit JSON `path` (any
    directory — unlike `custom_model.core.save_model`, which always writes
    into the repo's `custom_models/<name>.json` library folder)."""
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(custom_core.model_to_json(model), encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ════════════════════════════════════════════════════════════════════════════

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, _Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def _cmd_inspect(args):
    data = load_data(args.file)
    out = {
        "file": str(args.file),
        "n_points": int(len(data["freq"])),
        "freq_min_hz": float(data["freq"].min()),
        "freq_max_hz": float(data["freq"].max()),
        "z0": data["z0"],
        "header_lines": data["header_lines"],
        "meta": data["meta"],
    }
    print(json.dumps(out, indent=2, default=_json_default))


def _cmd_models(args):
    out = {"models": list_models(), "param_specs": {}}
    for short in REGISTRY:
        out["param_specs"][short] = [
            {"key": k, "label": lbl, "si_scale": sc, "unit": u}
            for k, lbl, sc, u in param_specs(short)
        ]
    print(json.dumps(out, indent=2, default=_json_default))


def _cmd_backend(args):
    print(json.dumps(backend_status(), indent=2, default=_json_default))


def _cmd_fit(args):
    data = load_data(args.file)
    initial = json.loads(_Path(args.initial).read_text()) if args.initial else None
    fixed = json.loads(_Path(args.fixed).read_text()) if args.fixed else None
    bounds = None
    if args.bounds:
        raw_b = json.loads(_Path(args.bounds).read_text())
        bounds = {k: tuple(v) for k, v in raw_b.items()}
    fit_keys = args.fit_keys.split(",") if args.fit_keys else None

    result = fit(data, args.model, initial=initial, fit_keys=fit_keys,
                fixed=fixed, bounds=bounds, method=args.method,
                maxiter=args.maxiter, backend=args.backend)
    print(f"[backend: {result['backend']}"
          f"{' — ' + result['backend_note'] if result.get('backend_note') else ''}]",
          file=_sys.stderr)
    text = json.dumps(result, indent=2, default=_json_default)
    if args.out:
        _Path(args.out).write_text(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="agent_api.py",
        description="Headless SSM fitting API for AI agents / scripts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect", help="Parse a .s2p/.csv and print header interpretation "
                        "+ freq range as JSON.")
    p_inspect.add_argument("file")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_models = sub.add_parser(
        "models", help="List built-in models + parameter specs as JSON.")
    p_models.set_defaults(func=_cmd_models)

    p_backend = sub.add_parser(
        "backend", help="Report compute-backend availability (CUDA / Rust / "
                        "NumPy) and what 'auto' resolves to, as JSON.")
    p_backend.set_defaults(func=_cmd_backend)

    p_fit = sub.add_parser("fit", help="Fit a model to a data file.")
    p_fit.add_argument("file")
    p_fit.add_argument("--model", required=True,
                       help="Builtin short (T/pi/XuT/KY) or path to a "
                            "custom-model .json.")
    p_fit.add_argument("--initial", default=None,
                       help="Path to a JSON file of initial SI-unit params.")
    p_fit.add_argument("--fit-keys", default=None,
                       help="Comma-separated parameter keys to fit "
                            "(default: all non-frozen).")
    p_fit.add_argument("--fixed", default=None,
                       help="Path to a JSON file of {key: SI value} held "
                            "constant.")
    p_fit.add_argument("--bounds", default=None,
                       help="Path to a JSON file of {key: [lo, hi]} in SI "
                            "units.")
    p_fit.add_argument("--method", default="auto",
                       choices=["auto", "nelder-mead", "least_squares"])
    p_fit.add_argument("--backend", default="auto",
                       choices=list(_BACKENDS),
                       help="Compute backend: auto (CUDA > Rust > NumPy), "
                            "cuda, rust, or numpy.")
    p_fit.add_argument("--maxiter", type=int, default=400)
    p_fit.add_argument("--out", default=None,
                       help="Also write the result JSON to this path.")
    p_fit.set_defaults(func=_cmd_fit)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

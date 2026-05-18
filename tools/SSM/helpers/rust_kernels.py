"""
helpers/rust_kernels.py — Optional Rust acceleration for hot kernels.

If the `hbt_rust_kernels` extension module is built and installed
(``pip install -e tools/SSM/rust_kernels``), the wrappers in this file
delegate to it.  Otherwise they fall back to the NumPy implementations
already in :mod:`tools.SSM.helpers.rf_math` and friends — so this
module is **purely additive**: removing or never building the Rust
crate leaves the Python tool fully functional.

Public API (all input arrays must be contiguous; non-contiguous inputs
are ``ascontiguousarray``-ed automatically, at one extra allocation):

  • :func:`inv2x2_batch(Y)`               – (B,2,2) → (B,2,2)
  • :func:`mm2x2_batch(A, B)`             – (B,2,2)×(B,2,2) → (B,2,2)
  • :func:`y_to_s_batch(Y, z0=50.0)`      – (B,2,2) → (B,2,2)
  • :func:`y_to_s_4d(Y, z0=50.0)`         – (B,N,2,2) → (B,N,2,2)
  • :func:`port_residuals_batch(Sm, Sb)`  – (N,2,2),(B,N,2,2) → (B,5)
  • :data:`HAS_RUST`                      – ``True`` iff the Rust ext is loaded

A flag controlled by the ``HBT_DISABLE_RUST`` env var forces the
NumPy path even when the extension is present — handy for benchmark
parity tests (see :mod:`tools.SSM.rust_kernels.benchmark`).
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
import numpy as np

# ── Locate a committed-in-repo binary, before falling back to pip install ───
#
# The build script (`build_rust_kernels.py` at the repo root) compiles the
# crate once per host and drops the resulting `.pyd` / `.so` into
# `tools/SSM/rust_kernels/bin/<platform_arch>/`.  Adding that directory
# to `sys.path` lets `import hbt_rust_kernels` succeed without any pip
# install or maturin step on end-user machines — they just check out the
# repo, get the prebuilt binary for their OS, and the wrapper picks it up.
#
# Resolution order:
#   1. Local committed binary in tools/SSM/rust_kernels/bin/<arch>/
#   2. Any pip-installed `hbt_rust_kernels` package in the active venv
#      (e.g. when a developer ran `pip install -e .` for live iteration).
#   3. NumPy fallback (silent — the tool runs identically to before).


def _arch_tag() -> str:
    """Return the directory tag we use for the prebuilt-binary subfolder.

    Mirrors the conventions Python uses for wheel tags (with a trimmed
    name space) so a humans-reading-the-dir-tree can map them to wheels
    they've seen.  Example values: ``win_amd64``, ``linux_x86_64``,
    ``macosx_arm64``.
    """
    machine = platform.machine().lower() or "unknown"
    if sys.platform == "win32":
        return f"win_{machine}"
    if sys.platform == "darwin":
        return f"macosx_{machine}"
    return f"linux_{machine}"


_BIN_DIR = (Path(__file__).resolve().parent.parent
            / "rust_kernels" / "bin" / _arch_tag())
if _BIN_DIR.is_dir() and str(_BIN_DIR) not in sys.path:
    # Prepend so the committed binary wins over any stale pip install.
    sys.path.insert(0, str(_BIN_DIR))

# ── Try to import the Rust extension; never raise on failure ────────────────

# `_RUST_IMPORT_ERROR` captures the exception message when the import fails
# so the UI (or `check_rust_status.py`) can show *why* HAS_RUST is False
# instead of silently falling back to NumPy.  It stays None on success.
_RUST_IMPORT_ERROR: str | None = None

# Sticky cross-reload cache.
#
# Why this exists
# ---------------
# PyO3 ≤ 0.22 enforces a single-init guard at the C level: any second
# attempt to load `hbt_rust_kernels` in the same process raises
#
#     ImportError: PyO3 modules compiled for CPython 3.8 or older
#                  may only be initialized once per interpreter process
#
# even when the wheel is tagged `cp39-abi3` and the `abi3-py39` feature
# is enabled (a known quirk — `Py_LIMITED_API` doesn't bypass this
# guard in pyo3 0.22).
#
# Streamlit's hot-reload runtime clears all local modules from
# `sys.modules` between script reruns, so on the *second* run the
# wrapper's `import hbt_rust_kernels` line trips that guard and we
# silently fall back to NumPy — even though the previously-loaded
# extension is still alive and fully functional somewhere in the
# interpreter.
#
# The fix is to stash the module on `builtins`, which Streamlit never
# clears.  On subsequent reloads we grab the live reference from there
# instead of attempting a fresh init.
import builtins as _builtins

_BUILTINS_KEY = "_hbt_rust_kernels_cache"
_cached_rk = getattr(_builtins, _BUILTINS_KEY, None)

if _cached_rk is not None:
    _rk = _cached_rk
    HAS_RUST: bool = True
    _RUST_IMPORT_ERROR = (
        "reusing cached extension from builtins (avoids PyO3 single-init "
        "guard on Streamlit hot reload)"
    )
else:
    try:
        import hbt_rust_kernels as _rk        # type: ignore[import-not-found]
        HAS_RUST = True
        setattr(_builtins, _BUILTINS_KEY, _rk)
    except ImportError as _exc:                # pragma: no cover
        # Last-ditch: maybe a parallel codepath imported it and left it
        # in sys.modules.  This catches the case where the first reload
        # happens *before* the builtins cache was seeded.
        if "hbt_rust_kernels" in sys.modules:
            _rk = sys.modules["hbt_rust_kernels"]
            HAS_RUST = True
            setattr(_builtins, _BUILTINS_KEY, _rk)
            _RUST_IMPORT_ERROR = (
                f"recovered after re-init failure ({type(_exc).__name__}: "
                f"{_exc}) — using the module already cached in sys.modules"
            )
        else:
            _rk = None
            HAS_RUST = False
            _RUST_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    except Exception as _exc:                  # pragma: no cover
        _rk = None
        HAS_RUST = False
        _RUST_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# Env override — force NumPy path even when Rust is available.
_FORCE_NUMPY = os.environ.get("HBT_DISABLE_RUST", "").strip().lower() in {
    "1", "true", "yes", "on"
}
if _FORCE_NUMPY:
    HAS_RUST = False
    _RUST_IMPORT_ERROR = "HBT_DISABLE_RUST env var is set"


def rust_diagnostic() -> dict:
    """Return a snapshot of the Rust-loading state for UI diagnostics.

    Keys:
      • ``has_rust``        – bool, the public HAS_RUST flag
      • ``bin_dir``         – str, expected directory for the prebuilt binary
      • ``bin_dir_exists``  – bool, the bin folder is present on disk
      • ``binary_files``    – list[str], filenames of active binaries found
      • ``import_error``    – str | None, captured exception when import fails
      • ``force_numpy``     – bool, HBT_DISABLE_RUST override active
      • ``arch_tag``        – str, the platform tag used for the lookup
    """
    arch = _arch_tag()
    bdir = Path(__file__).resolve().parent.parent / "rust_kernels" / "bin" / arch
    bins: list[str] = []
    if bdir.is_dir():
        for pat in ("hbt_rust_kernels*.pyd",
                    "hbt_rust_kernels*.so",
                    "hbt_rust_kernels*.dylib"):
            for f in bdir.glob(pat):
                if ".old-" not in f.name:
                    bins.append(f.name)
    return {
        "has_rust":       HAS_RUST,
        "bin_dir":        str(bdir),
        "bin_dir_exists": bdir.is_dir(),
        "binary_files":   bins,
        "import_error":   _RUST_IMPORT_ERROR,
        "force_numpy":    _FORCE_NUMPY,
        "arch_tag":       arch,
    }


# ── NumPy reference implementations (used as the fallback) ──────────────────

def _np_inv2x2_batch(Y: np.ndarray) -> np.ndarray:
    """Reference NumPy inverse of a (B, 2, 2) batch — same as
    :func:`helpers.rf_math.inv2x2`, here for completeness."""
    Y = np.ascontiguousarray(Y)
    a, b = Y[..., 0, 0], Y[..., 0, 1]
    c, d = Y[..., 1, 0], Y[..., 1, 1]
    inv_det = 1.0 / (a * d - b * c)
    out = np.empty_like(Y)
    out[..., 0, 0] =  d * inv_det
    out[..., 0, 1] = -b * inv_det
    out[..., 1, 0] = -c * inv_det
    out[..., 1, 1] =  a * inv_det
    return out


def _np_mm2x2_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a00, a01 = A[..., 0, 0], A[..., 0, 1]
    a10, a11 = A[..., 1, 0], A[..., 1, 1]
    b00, b01 = B[..., 0, 0], B[..., 0, 1]
    b10, b11 = B[..., 1, 0], B[..., 1, 1]
    out = np.empty_like(A)
    out[..., 0, 0] = a00 * b00 + a01 * b10
    out[..., 0, 1] = a00 * b01 + a01 * b11
    out[..., 1, 0] = a10 * b00 + a11 * b10
    out[..., 1, 1] = a10 * b01 + a11 * b11
    return out


def _np_y_to_s_batch(Y: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Vectorised Y → S, identical math to ``helpers.rf_math.y_to_s_vec``."""
    Y = np.ascontiguousarray(Y)
    I = np.eye(2, dtype=Y.dtype)
    M = I + Y * z0
    N = I - Y * z0
    return _np_mm2x2_batch(N, _np_inv2x2_batch(M))


def _np_y_to_s_4d(Y: np.ndarray, z0: float = 50.0) -> np.ndarray:
    return _np_y_to_s_batch(Y, z0)  # NumPy broadcasts (B, N, 2, 2) naturally


def _np_port_residuals_batch(S_mea: np.ndarray,
                              S_mod_batch: np.ndarray) -> np.ndarray:
    """Reference for the Rust port_residuals_batch — produces a (B, 5)
    float array: ``[Total, S11, S12, S21, S22]`` in percent."""
    # |S_mea|² sums per port, shared across the batch.
    denoms = np.einsum("ijk,ijk->jk", S_mea, S_mea.conj()).real      # (2,2)
    diffs  = S_mod_batch - S_mea[np.newaxis, ...]                    # (B,N,2,2)
    num    = np.einsum("bijk,bijk->bjk", diffs, diffs.conj()).real   # (B,2,2)
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.sqrt(num / denoms) * 100.0                         # (B,2,2)
    port[~np.isfinite(port)] = 0.0
    B = port.shape[0]
    out = np.empty((B, 5), dtype=np.float64)
    out[:, 1] = port[:, 0, 0]   # S11
    out[:, 2] = port[:, 0, 1]   # S12
    out[:, 3] = port[:, 1, 0]   # S21
    out[:, 4] = port[:, 1, 1]   # S22
    out[:, 0] = out[:, 1:].mean(axis=1)
    return out


# ── Public wrappers ─────────────────────────────────────────────────────────

def _c128(arr: np.ndarray) -> np.ndarray:
    """Ensure complex128 + C-contiguous (Rust expects exactly this)."""
    if arr.dtype != np.complex128:
        arr = arr.astype(np.complex128)
    return np.ascontiguousarray(arr)


def inv2x2_batch(Y: np.ndarray) -> np.ndarray:
    """Per-row 2×2 complex inverse over a `(B, 2, 2)` batch.  Falls back
    to the NumPy implementation when the Rust extension isn't loaded."""
    if HAS_RUST:
        return _rk.inv2x2_batch(_c128(Y))
    return _np_inv2x2_batch(Y)


def mm2x2_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row 2×2 complex matmul over a `(B, 2, 2)` batch."""
    if HAS_RUST:
        return _rk.mm2x2_batch(_c128(A), _c128(B))
    return _np_mm2x2_batch(A, B)


def y_to_s_batch(Y: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Per-row Y → S over a `(B, 2, 2)` batch."""
    if HAS_RUST:
        return _rk.y_to_s_batch(_c128(Y), float(z0))
    return _np_y_to_s_batch(Y, z0)


def y_to_s_4d(Y: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Per-element Y → S over a `(B, N, 2, 2)` tensor.  The outer (B)
    axis is parallelised by Rayon in the Rust path — this is the
    primary kernel called from the visual-tuning sweep loop."""
    if HAS_RUST:
        return _rk.y_to_s_4d(_c128(Y), float(z0))
    return _np_y_to_s_4d(Y, z0)


def port_residuals_batch(S_mea: np.ndarray,
                         S_mod_batch: np.ndarray) -> np.ndarray:
    """Per-port residuals for a batched simulation.

    Parameters
    ----------
    S_mea : ndarray complex128, shape ``(N, 2, 2)``
        Measured S-parameters.
    S_mod_batch : ndarray complex128, shape ``(B, N, 2, 2)``
        Stack of model S-parameters, one per param combination.

    Returns
    -------
    ndarray float64, shape ``(B, 5)``
        Per-row ``[Total, S11, S12, S21, S22]`` residuals in **percent**.
    """
    if HAS_RUST:
        return _rk.port_residuals_batch(_c128(S_mea), _c128(S_mod_batch))
    return _np_port_residuals_batch(S_mea, S_mod_batch)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — per-topology end-to-end batched simulators (scaffolding)
# ════════════════════════════════════════════════════════════════════════════
#
# These wrappers reserve the public API for the upcoming
# `_sim_wrap_batch` end-to-end ports.  Until the Rust implementation
# lands for a given topology, the Rust stub raises `NotImplementedError`
# and the wrapper falls through to the NumPy reference passed in via
# the `np_fallback` kwarg — keeping the production code path
# **identical** to today's behaviour.
#
# Opt-in dispatch lives in `SSMModelTemplate.simulate_batch`
# (`tools/SSM/models/base_ui.py`), gated on the env var
# `HBT_USE_RUST_SIM_BATCH=1`.  When unset, these wrappers are not
# called at all.

def _phase2_dispatch_enabled() -> bool:
    """Whether the Phase 2 Rust dispatch fires for batched simulations.

    Resolution rules:

      • ``HBT_USE_RUST_SIM_BATCH=0`` (or false/no/off) → forced OFF.
        Useful for A/B parity tests and bug isolation.
      • ``HBT_USE_RUST_SIM_BATCH=1`` (or true/yes/on)  → forced ON.
      • UNSET (the normal case) → auto-enabled iff a Rust binary is
        present (``HAS_RUST`` is True).  The reasoning: if the user
        built and committed a Rust binary, they want to use it; we
        shouldn't make them set an env var on top.

    Streamlit Cloud / `streamlit run` invocations get the auto-enable
    behaviour without any orchestration: the wrapper sees the
    committed binary, imports it, and the dispatch fires.
    """
    val = os.environ.get("HBT_USE_RUST_SIM_BATCH", "").strip().lower()
    if val in {"0", "false", "no", "off"}:
        return False
    if val in {"1", "true", "yes", "on"}:
        return True
    return HAS_RUST


def _phase2_parity_check_enabled() -> bool:
    """True iff `HBT_RUST_PARITY_CHECK=1`.

    When True, the Phase 2 wrappers run BOTH the Rust kernel and the
    NumPy fallback for every call and `assert np.allclose(...)`.  Adds
    noticeable overhead — strictly a development / regression-testing
    aid, not a production path.
    """
    return os.environ.get("HBT_RUST_PARITY_CHECK", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _normalize_params_for_rust(params: dict) -> dict:
    """Coerce a param dict into the shape the Rust kernels expect:
    every numeric value is either a Python float or a **1-D float64
    NumPy array of length B** (the common flat batch size).

    Why this is needed
    ------------------
    The Auto Tuning sweep loop (``base_ui.py::_run_one_sweep``) packs
    each swept parameter as an N-D NumPy tensor with shape
    ``(1, …, L_i, …, 1, 1)`` — broadcast-ready against the rest of the
    grid and the frequency axis.  The NumPy ``_sim_wrap_batch`` handles
    that natively via array broadcasting; the Rust kernels, by design,
    take a flat (B,) batch axis and iterate it linearly.

    This helper bridges the two conventions:
      • Detects all multi-dim numeric arrays in ``params``.
      • Casts them to float64 and broadcasts them all to the common
        mesh shape via ``np.broadcast_arrays`` (zero-copy strided
        views — no per-element materialisation cost until the next
        step).
      • Flattens each broadcast result to 1-D contiguous float64.
        That materialisation is ~B × 8 bytes per swept param — at
        B=729 typical, ≈ 6 KB per param per call.  Negligible.
      • Scalars / strings / non-numeric values pass through unchanged.

    If no array values are present, the dict is returned as-is.
    """
    arr_keys = [k for k, v in params.items()
                if isinstance(v, np.ndarray) and v.size > 0]
    if not arr_keys:
        return params

    # Cast to float64 first so broadcasting + flattening produce the
    # contract dtype.  Cheng/Xu pre-bake passes float32 in fp32-sweep
    # mode on CUDA; CPU paths use float64.  Cast to fp64 is a no-op
    # for already-fp64 arrays.
    arrs = [np.asarray(params[k], dtype=np.float64) for k in arr_keys]

    # `broadcast_arrays` returns views; the actual allocation happens
    # in the ascontiguousarray(...).reshape(-1) line below.
    try:
        bcasts = np.broadcast_arrays(*arrs)
    except ValueError:
        # Incompatible shapes — let Rust try anyway (it'll raise a
        # clear error) or, more likely, the caller has a real bug.
        return params

    out = dict(params)
    for k, b in zip(arr_keys, bcasts):
        out[k] = np.ascontiguousarray(b, dtype=np.float64).reshape(-1)
    return out


# Module-level set of (rust_fn_name, error_type) pairs we've already
# warned about — keeps a stuck Rust path from spamming the log on every
# iteration of an Auto Tuning sweep.
_WARNED_ABOUT: set[tuple[str, str]] = set()


def _phase2_dispatch(rust_fn_name: str, params, freq, z0, np_fallback):
    """Common dispatch helper used by all three topology wrappers.

    Resolution order:
      1. If Rust isn't loaded OR the user hasn't opted into Phase 2,
         go straight to NumPy.
      2. Normalise the param dict (broadcast + flatten N-D tensors so
         the Rust kernel's 1-D contract is honoured).
      3. Try the Rust function by name.  On `NotImplementedError`
         (the scaffolding stub or exotic pad modes), fall back to
         NumPy silently.
      4. On any other exception from the Rust path, log ONCE per
         (function, error-type) combination and fall back.
      5. If `HBT_RUST_PARITY_CHECK=1` is also set, run NumPy too and
         assert agreement.
    """
    if not (HAS_RUST and _phase2_dispatch_enabled()):
        return np_fallback(params, freq, z0)

    rust_fn = getattr(_rk, rust_fn_name, None)
    if rust_fn is None:
        return np_fallback(params, freq, z0)

    try:
        freq_c = np.ascontiguousarray(freq, dtype=np.float64)
        norm_params = _normalize_params_for_rust(params)
        rust_out = rust_fn(norm_params, freq_c, float(z0))
    except NotImplementedError:
        # Scaffolding stub or exotic pad mode — silent fallback.
        return np_fallback(params, freq, z0)
    except Exception as e:                                  # pragma: no cover
        import warnings
        key = (rust_fn_name, type(e).__name__)
        if key not in _WARNED_ABOUT:
            _WARNED_ABOUT.add(key)
            warnings.warn(
                f"Rust {rust_fn_name} raised {type(e).__name__}: {e}.  "
                f"Falling back to NumPy for this and any subsequent "
                f"{type(e).__name__} from the same kernel "
                f"(further warnings suppressed).",
                RuntimeWarning,
                stacklevel=3,
            )
        return np_fallback(params, freq, z0)

    if _phase2_parity_check_enabled():
        np_out = np_fallback(params, freq, z0)
        if not np.allclose(rust_out, np_out, rtol=1e-9, atol=1e-12):
            max_err = float(np.max(np.abs(rust_out - np_out)))
            raise AssertionError(
                f"HBT_RUST_PARITY_CHECK: {rust_fn_name} disagrees with "
                f"NumPy reference (max |abs diff| = {max_err:.3e}).  "
                f"Set HBT_RUST_PARITY_CHECK=0 to silence."
            )

    return rust_out


def sim_cheng_t_batch(params, freq, z0=50.0, *, np_fallback):
    """Cheng (2022) T-topology — end-to-end batched simulation.

    Phase 2 entry point.  Currently routes to ``np_fallback`` for every
    call because the Rust implementation is a scaffolding stub that
    raises ``NotImplementedError``.  When the Rust kernel is filled
    in, set the env var ``HBT_USE_RUST_SIM_BATCH=1`` to start using
    it; the wrapper then prefers the Rust path with NumPy as the
    safety net.

    Parameters
    ----------
    params : dict
        SI-unit parameter dict.  Values can be scalars (broadcast
        across the batch) or 1-D arrays of shape ``(B,)`` (per-frame).
    freq : ndarray float64, shape ``(N,)``
    z0 : float
    np_fallback : callable(params, freq, z0) -> ndarray (B, N, 2, 2)
        The canonical NumPy implementation (typically
        ``_sim_wrap_batch(_Y_int_T_batch, params, freq, z0, np, None)``).
        Passed in by the caller to avoid a helpers→models import cycle.

    Returns
    -------
    ndarray complex128, shape ``(B, N, 2, 2)``
    """
    return _phase2_dispatch("sim_cheng_t_batch", params, freq, z0, np_fallback)


def sim_cheng_pi_batch(params, freq, z0=50.0, *, np_fallback):
    """Cheng (2022) π-topology — end-to-end batched simulation.

    See :func:`sim_cheng_t_batch` for the dispatch semantics and the
    ``np_fallback`` contract.
    """
    return _phase2_dispatch("sim_cheng_pi_batch", params, freq, z0, np_fallback)


def sim_xu_t_batch(params, freq, z0=50.0, *, np_fallback):
    """Xu (2014) T-topology with parallel ``Rbcx ‖ Cbcx`` — end-to-end
    batched simulation.

    See :func:`sim_cheng_t_batch` for the dispatch semantics and the
    ``np_fallback`` contract.
    """
    return _phase2_dispatch("sim_xu_t_batch", params, freq, z0, np_fallback)


# Lookup table consumed by `SSMModelTemplate.simulate_batch` to pick
# the right Rust wrapper for a model's `SHORT` identifier.  Adding a
# new topology only requires (a) a Rust `#[pyfunction]`, (b) a wrapper
# above, and (c) one entry here.
SIM_FOR_TOPOLOGY = {
    "T":   sim_cheng_t_batch,
    "pi":  sim_cheng_pi_batch,
    "XuT": sim_xu_t_batch,
}


__all__ = [
    "HAS_RUST",
    "inv2x2_batch",
    "mm2x2_batch",
    "y_to_s_batch",
    "y_to_s_4d",
    "port_residuals_batch",
    # Phase 2 (scaffolding)
    "sim_cheng_t_batch",
    "sim_cheng_pi_batch",
    "sim_xu_t_batch",
    "SIM_FOR_TOPOLOGY",
]

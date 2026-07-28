//! HBT Rust kernels — SIMD/Rayon-accelerated CPU implementations of the
//! hottest math kernels used during SSM model extraction.
//!
//! The Python side (`tools/SSM/helpers/rust_kernels.py`) imports this
//! module if it's built and installed, and silently falls back to the
//! NumPy implementations otherwise — so this crate is **optional**: the
//! tool runs fine without it.
//!
//! Exposed Python entry points (all expect contiguous `complex128`
//! NumPy arrays; outputs are freshly-allocated `complex128` arrays):
//!
//!   • `inv2x2_batch(Y) -> Y_inv`           — `(B, 2, 2)`  →  `(B, 2, 2)`
//!   • `mm2x2_batch(A, B) -> AB`            — `(B, 2, 2)`, `(B, 2, 2)` → `(B, 2, 2)`
//!   • `y_to_s_batch(Y, z0) -> S`           — `(B, 2, 2)`  →  `(B, 2, 2)`
//!   • `y_to_s_4d(Y, z0) -> S`              — `(B, N, 2, 2)` → `(B, N, 2, 2)`
//!   • `port_residuals_batch(S_mea, S_mod)` — `(N, 2, 2)`, `(B, N, 2, 2)`
//!                                            → `(B, 5)` `[Total, S11, S12, S21, S22]` in %
//!
//! All multi-batch kernels parallelise across the outer (B) axis via
//! Rayon, so a typical 1 000-frame tuning sweep keeps every physical
//! core busy without any per-call thread-pool spin-up overhead.

// ndarray 0.16 with `features = ["rayon"]` exposes `par_for_each` as an
// inherent method on `Zip`, so no prelude import is needed.
use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4,
            PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
// rayon::slice::ParallelSliceMut provides `par_chunks_mut` used by
// the Phase 2 end-to-end simulators below.
use rayon::slice::ParallelSliceMut;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

// mimalloc as the global allocator — see Cargo.toml note.  The default
// Windows allocator (HeapAlloc) serialises across threads on small/medium
// allocations, throttling Rayon parse paths that grow many small Vecs
// per task.  mimalloc's per-thread arenas remove the contention.
//
// Windows ONLY.  On macOS/Linux this cdylib exports mimalloc's `mi_*`
// symbols into a process that already contains a second, statically
// linked mimalloc inside pyarrow's libarrow.  dyld cross-binds the two
// copies, so a buffer arrow allocates via its mimalloc gets collected
// via ours — SIGSEGV (EXC_BAD_ACCESS) during DataFrame→Arrow conversion,
// which Streamlit does on every render.  Windows resolves DLL imports
// per-module, so the collision does not arise there and the perf win is
// kept where the HeapAlloc contention it addresses actually exists.
#[cfg(target_os = "windows")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type C = Complex64;

// ── Inlineable 2×2 primitives ────────────────────────────────────────────────

/// Analytic inverse of a 2×2 complex matrix (4-element row-major).
/// 6 mul + 2 sub + 1 div, total 9 FLOPs (counting `Complex64` ops as one).
#[inline(always)]
fn inv2x2_flat(m: &[C; 4]) -> [C; 4] {
    let det = m[0] * m[3] - m[1] * m[2];
    let inv = C::new(1.0, 0.0) / det;
    [m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv]
}

/// 2×2 × 2×2 complex matmul — 8 mul + 4 add, 12 FLOPs.
#[inline(always)]
fn mm2x2_flat(a: &[C; 4], b: &[C; 4]) -> [C; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Y → S analytic formula for a single 2×2:  S = (E − Y·Z0) · (E + Y·Z0)⁻¹
/// expanded into 25 muls + 12 adds — same as the Python `y_to_s_vec` path
/// but with no per-element NumPy overhead.
#[inline(always)]
fn y_to_s_one(y: &[C; 4], z0: C) -> [C; 4] {
    // M = E + Y·z0
    let m00 = C::new(1.0, 0.0) + y[0] * z0;
    let m01 = y[1] * z0;
    let m10 = y[2] * z0;
    let m11 = C::new(1.0, 0.0) + y[3] * z0;
    // N = E − Y·z0
    let n00 = C::new(1.0, 0.0) - y[0] * z0;
    let n01 = -y[1] * z0;
    let n10 = -y[2] * z0;
    let n11 = C::new(1.0, 0.0) - y[3] * z0;
    // S = N · M⁻¹
    let m_inv = inv2x2_flat(&[m00, m01, m10, m11]);
    mm2x2_flat(&[n00, n01, n10, n11], &m_inv)
}

// ── Python entry points ──────────────────────────────────────────────────────

#[pyfunction]
fn inv2x2_batch<'py>(
    py: Python<'py>,
    y: PyReadonlyArray3<'py, C>,
) -> PyResult<Bound<'py, PyArray3<C>>> {
    let arr = y.as_array();
    let b   = arr.shape()[0];
    let mut out = Array3::<C>::zeros((b, 2, 2));
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(arr.axis_iter(Axis(0)))
        .par_for_each(|mut o, m| {
            let mf = [m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]];
            let r  = inv2x2_flat(&mf);
            o[(0, 0)] = r[0]; o[(0, 1)] = r[1];
            o[(1, 0)] = r[2]; o[(1, 1)] = r[3];
        });
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn mm2x2_batch<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, C>,
    b: PyReadonlyArray3<'py, C>,
) -> PyResult<Bound<'py, PyArray3<C>>> {
    let aa = a.as_array();
    let bb = b.as_array();
    let bn = aa.shape()[0];
    let mut out = Array3::<C>::zeros((bn, 2, 2));
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(aa.axis_iter(Axis(0)))
        .and(bb.axis_iter(Axis(0)))
        .par_for_each(|mut o, m1, m2| {
            let af = [m1[(0, 0)], m1[(0, 1)], m1[(1, 0)], m1[(1, 1)]];
            let bf = [m2[(0, 0)], m2[(0, 1)], m2[(1, 0)], m2[(1, 1)]];
            let r  = mm2x2_flat(&af, &bf);
            o[(0, 0)] = r[0]; o[(0, 1)] = r[1];
            o[(1, 0)] = r[2]; o[(1, 1)] = r[3];
        });
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn y_to_s_batch<'py>(
    py: Python<'py>,
    y: PyReadonlyArray3<'py, C>,
    z0: f64,
) -> PyResult<Bound<'py, PyArray3<C>>> {
    let arr = y.as_array();
    let b   = arr.shape()[0];
    let z0c = C::new(z0, 0.0);
    let mut out = Array3::<C>::zeros((b, 2, 2));
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(arr.axis_iter(Axis(0)))
        .par_for_each(|mut o, m| {
            let yf = [m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]];
            let s  = y_to_s_one(&yf, z0c);
            o[(0, 0)] = s[0]; o[(0, 1)] = s[1];
            o[(1, 0)] = s[2]; o[(1, 1)] = s[3];
        });
    Ok(out.into_pyarray_bound(py))
}

/// 4-D batched Y → S, where the **outer** axis is the param batch (B)
/// and the **inner** axis is frequency (N).  We parallelise across B
/// and run a tight scalar loop over N inside each task — this is the
/// hottest shape in the tuning-sweep path.
#[pyfunction]
fn y_to_s_4d<'py>(
    py: Python<'py>,
    y: PyReadonlyArray4<'py, C>,
    z0: f64,
) -> PyResult<Bound<'py, PyArray4<C>>> {
    let arr = y.as_array();
    let shp = arr.shape();
    let (b, n) = (shp[0], shp[1]);
    let z0c = C::new(z0, 0.0);
    let mut out = Array4::<C>::zeros((b, n, 2, 2));
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(arr.axis_iter(Axis(0)))
        .par_for_each(|mut o_slab, m_slab| {
            for i in 0..n {
                let yf = [
                    m_slab[(i, 0, 0)], m_slab[(i, 0, 1)],
                    m_slab[(i, 1, 0)], m_slab[(i, 1, 1)],
                ];
                let s = y_to_s_one(&yf, z0c);
                o_slab[(i, 0, 0)] = s[0];
                o_slab[(i, 0, 1)] = s[1];
                o_slab[(i, 1, 0)] = s[2];
                o_slab[(i, 1, 1)] = s[3];
            }
        });
    Ok(out.into_pyarray_bound(py))
}

/// Per-port residuals across a parameter sweep.
///
/// ``S_mea`` is a single (N, 2, 2) measured trace; ``S_mod_batch`` is a
/// (B, N, 2, 2) stack of simulated traces (one per param combination).
/// Returns a (B, 5) f64 matrix where each row is
/// ``[Total, S11, S12, S21, S22]`` in **percent**.
///
/// Hot path: the auto-tuning expander calls this once per batch sweep,
/// often with B ≥ 10 000.  The Python reference (``_port_residuals_batch``)
/// does it with a couple of NumPy reductions; here we fuse them in a
/// single Rayon-parallel pass, which on the typical 8-core laptop is
/// 4–6× faster.
#[pyfunction]
fn port_residuals_batch<'py>(
    py: Python<'py>,
    s_mea: PyReadonlyArray3<'py, C>,
    s_mod_batch: PyReadonlyArray4<'py, C>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mea  = s_mea.as_array();
    let modb = s_mod_batch.as_array();
    let b    = modb.shape()[0];
    let n    = mea.shape()[0];

    // Per-port |S_mea|² sums — shared across the batch, compute once.
    let mut denoms = [0.0_f64; 4];
    for r in 0..2 {
        for c in 0..2 {
            let mut d = 0.0_f64;
            for i in 0..n {
                d += mea[(i, r, c)].norm_sqr();
            }
            denoms[r * 2 + c] = d;
        }
    }

    let mut out = Array2::<f64>::zeros((b, 5));
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(modb.axis_iter(Axis(0)))
        .par_for_each(|mut row, slab| {
            let mut port_res = [0.0_f64; 4];
            for r in 0..2 {
                for c in 0..2 {
                    let denom = denoms[r * 2 + c];
                    if denom > 0.0 {
                        let mut num = 0.0_f64;
                        for i in 0..n {
                            let diff = mea[(i, r, c)] - slab[(i, r, c)];
                            num += diff.norm_sqr();
                        }
                        port_res[r * 2 + c] = (num / denom).sqrt() * 100.0;
                    }
                }
            }
            let total = (port_res[0] + port_res[1]
                       + port_res[2] + port_res[3]) / 4.0;
            row[0] = total;
            row[1] = port_res[0]; // S11
            row[2] = port_res[1]; // S12
            row[3] = port_res[2]; // S21
            row[4] = port_res[3]; // S22
        });

    Ok(out.into_pyarray_bound(py))
}

// ── Phase 2 scaffolding — per-topology end-to-end batched simulators ─────────
//
// These three `#[pyfunction]`s reserve the names + signatures for the
// upcoming `_sim_wrap_batch` end-to-end ports.  They currently raise
// `NotImplementedError` so the Python wrapper layer can detect "not
// yet implemented" and fall through to the canonical NumPy
// implementation in `tools/SSM/models/cheng.py` / `xu.py`.
//
// When a topology is implemented:
//   1. Replace the `Err(...)` body below with the real composition.
//   2. The Python wrapper at `helpers/rust_kernels.py::sim_<topology>_batch`
//      starts taking the Rust branch automatically when the env var
//      `HBT_USE_RUST_SIM_BATCH=1` is set.
//   3. Bump the binary by re-running `python rust_things/build_rust_kernels.py`.
//
// The stubs deliberately accept the FULL signature now so adding the
// implementation later is purely additive — no Python-side wrapper
// changes needed when one topology gets implemented while the others
// remain stubs.

fn _phase2_not_implemented(name: &str) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "{name}: Phase 2 Rust implementation has not landed yet. \
         The Python wrapper will fall back to the NumPy `_sim_wrap_batch` \
         in tools/SSM/models/.  To track progress see \
         tools/SSM/rust_kernels/PHASE2_PLAN.md."
    ))
}

// ── Phase 2 helpers ──────────────────────────────────────────────────────────

/// A parameter value that's either a scalar (broadcast across the batch)
/// or a per-frame array of length B.  Mirrors the role of
/// `_b1(p, key, default, xp)` in the Python implementation, but instead
/// of reshape-and-broadcast we keep the two cases as an enum and pick
/// the value at index `b` cheaply in the inner loop.
enum BcVal {
    Scalar(f64),
    PerBatch(Vec<f64>),
}

impl BcVal {
    #[inline(always)]
    fn at(&self, b: usize) -> f64 {
        match self {
            BcVal::Scalar(v)   => *v,
            BcVal::PerBatch(v) => v[b],
        }
    }

    fn batch_len(&self) -> Option<usize> {
        match self {
            BcVal::PerBatch(v) => Some(v.len()),
            BcVal::Scalar(_)   => None,
        }
    }
}

/// Pull one parameter out of the Python dict.  Accepts a Python float,
/// numpy scalar, or 1-D float64 array.  Falls back to `default` when
/// the key is absent.
fn _extract_bcval(
    params: &Bound<'_, PyDict>,
    key: &str,
    default: f64,
) -> PyResult<BcVal> {
    match params.get_item(key)? {
        None       => Ok(BcVal::Scalar(default)),
        Some(item) => {
            // Scalar path — covers Python int, float, numpy 0-d.
            if let Ok(v) = item.extract::<f64>() {
                return Ok(BcVal::Scalar(v));
            }
            // Array path — strict on float64 dtype.  Callers should
            // convert int arrays to float64 on the Python side; doing
            // it here would mask precision bugs.
            if let Ok(arr) = item.extract::<PyReadonlyArray1<'_, f64>>() {
                let slice = arr.as_slice()?;
                if slice.len() == 1 {
                    Ok(BcVal::Scalar(slice[0]))
                } else {
                    Ok(BcVal::PerBatch(slice.to_vec()))
                }
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "sim_*_batch param '{}' must be a Python float, \
                     numpy scalar, or 1-D float64 array.",
                    key
                )))
            }
        }
    }
}

/// True when the params dict requests an exotic pad-cap mode
/// (Parallel L / Series L / Series R) or a non-zero `Cpar_L*` — the
/// Rust kernel currently only handles the trivial `jωC` pad and zero
/// parasitic L-cap.  Returning true causes the wrapper to raise
/// `NotImplementedError` and the Python side falls back to NumPy.
fn _params_use_extra_modes(params: &Bound<'_, PyDict>) -> PyResult<bool> {
    for key in &["Cpbe_mode", "Cpce_mode", "Cpbc_mode"] {
        if let Some(item) = params.get_item(key)? {
            if let Ok(s) = item.extract::<String>() {
                if s != "None" {
                    return Ok(true);
                }
            }
        }
    }
    for key in &["Cpar_Lb", "Cpar_Lc", "Cpar_Le"] {
        if let Some(item) = params.get_item(key)? {
            if let Ok(v) = item.extract::<f64>() {
                if v != 0.0 {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

#[pyfunction]
fn sim_cheng_t_batch<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyDict>,
    freq:   PyReadonlyArray1<'py, f64>,
    z0:     f64,
) -> PyResult<Bound<'py, PyArray4<C>>> {
    // ── Pre-flight: if the params dict requests a pad mode we don't
    //    handle yet, raise NotImplementedError so the Python wrapper
    //    falls back to NumPy.  This keeps the Rust path scoped to the
    //    common case (~95% of real use); exotic Open-cap modes stay
    //    on the canonical implementation.
    if _params_use_extra_modes(params)? {
        return Err(PyNotImplementedError::new_err(
            "sim_cheng_t_batch: pad extra-mode (Parallel L / Series L / \
             Series R / non-zero Cpar_Lb/Lc/Le) is not yet supported by \
             the Rust kernel.  Falling back to NumPy `_sim_wrap_batch`."
        ));
    }

    // ── Extract all parameters.  The defaults match the Python
    //    `_b1(p, key, default, xp)` calls in cheng.py + deembed_math.py.
    let p_cpbe   = _extract_bcval(params, "Cpbe",   0.0)?;
    let p_cpce   = _extract_bcval(params, "Cpce",   0.0)?;
    let p_cpbc   = _extract_bcval(params, "Cpbc",   0.0)?;
    let p_lb     = _extract_bcval(params, "Lb",     0.0)?;
    let p_lc     = _extract_bcval(params, "Lc",     0.0)?;
    let p_le     = _extract_bcval(params, "Le",     0.0)?;
    let p_rpb    = _extract_bcval(params, "Rpb",    0.0)?;
    let p_rpc    = _extract_bcval(params, "Rpc",    0.0)?;
    let p_rpe    = _extract_bcval(params, "Rpe",    0.0)?;
    let p_cbex   = _extract_bcval(params, "Cbex",   0.0)?;
    let p_cbcx   = _extract_bcval(params, "Cbcx",   0.0)?;
    let p_rbi    = _extract_bcval(params, "Rbi",    0.0)?;
    let p_rbe    = _extract_bcval(params, "Rbe",    1.0)?;
    let p_cbe    = _extract_bcval(params, "Cbe",    0.0)?;
    let p_rbc    = _extract_bcval(params, "Rbc",    1.0)?;
    let p_cbc    = _extract_bcval(params, "Cbc",    0.0)?;
    let p_alpha0 = _extract_bcval(params, "alpha0", 0.0)?;
    let p_taub   = _extract_bcval(params, "tauB",   0.0)?;
    let p_tauc   = _extract_bcval(params, "tauC",   0.0)?;

    // Detect batch size — the longest PerBatch length, else 1.
    let all = [
        &p_cpbe, &p_cpce, &p_cpbc, &p_lb, &p_lc, &p_le,
        &p_rpb,  &p_rpc,  &p_rpe,  &p_cbex, &p_cbcx,
        &p_rbi,  &p_rbe,  &p_cbe,  &p_rbc,  &p_cbc,
        &p_alpha0, &p_taub, &p_tauc,
    ];
    let b: usize = all.iter()
        .filter_map(|v| v.batch_len())
        .max()
        .unwrap_or(1);

    // Validate every PerBatch length agrees with the detected B —
    // catches the "I passed two arrays of different length" footgun.
    for (i, v) in all.iter().enumerate() {
        if let BcVal::PerBatch(arr) = v {
            if arr.len() != b {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("sim_cheng_t_batch: parameter at index {} has \
                             length {}, expected B={}.", i, arr.len(), b)
                ));
            }
        }
    }

    let freq_view = freq.as_array();
    let n = freq_view.len();
    let freq_slice = freq_view.as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "sim_cheng_t_batch: freq array must be C-contiguous."
        ))?
        .to_vec();

    // Allocate the output (B, N, 2, 2) contiguous in C order.  We
    // write into it via the flat slice — each (b) slab is `n * 4`
    // complex64 values, contiguous, and Rayon can parallelise across
    // the B axis trivially.
    let mut out = Array4::<C>::zeros((b, n, 2, 2));
    let slab_len = n * 4;
    let flat = out.as_slice_mut().expect("Array4 is always C-contig");

    let one = C::new(1.0, 0.0);
    let z0c = C::new(z0, 0.0);

    // Release the GIL for the parallel compute — we own all the data
    // already (extracted into BcVal Vecs above), so no Python touches
    // happen inside the worker threads.
    py.allow_threads(|| {
        flat.par_chunks_mut(slab_len)
            .enumerate()
            .for_each(|(bi, chunk)| {
                // Pre-load this batch's scalar values once.
                let cpbe   = p_cpbe.at(bi);
                let cpce   = p_cpce.at(bi);
                let cpbc   = p_cpbc.at(bi);
                let lb     = p_lb.at(bi);
                let lc     = p_lc.at(bi);
                let le     = p_le.at(bi);
                let rpb    = p_rpb.at(bi);
                let rpc    = p_rpc.at(bi);
                let rpe    = p_rpe.at(bi);
                let cbex   = p_cbex.at(bi);
                let cbcx   = p_cbcx.at(bi);
                let rbi    = p_rbi.at(bi);
                let rbe    = p_rbe.at(bi);
                let cbe    = p_cbe.at(bi);
                let rbc    = p_rbc.at(bi);
                let cbc    = p_cbc.at(bi);
                let alpha0 = p_alpha0.at(bi);
                let taub   = p_taub.at(bi);
                let tauc   = p_tauc.at(bi);

                // ── Per-batch composition shortcuts ─────────────────────
                // Decided ONCE per batch.  The inner-loop branches on
                // these booleans, which the predictor nails after the
                // first iteration since they're loop-invariant.
                //
                //  ser_active   — any of {Rpb, Rpc, Rpe, Lb, Lc, Le} ≠ 0.
                //                 When false, Y_tot = Y_ex and we skip
                //                 TWO 2×2 inversions per (b, n).
                //  lead_l_active — any of {Lb, Lc, Le} ≠ 0.  When false
                //                 *and* ser_active is true (the typical
                //                 "R-only Z_ser" case), the jωL term
                //                 inside Z_ser is constant-zero so we
                //                 skip 3 complex muls per (b, n).
                //  pad_active   — any of {Cpbe, Cpce, Cpbc} ≠ 0.  When
                //                 false, Y_a = Y_tot (no addition).
                //  cpbe/cpce/cpbc_active — per-cap; lets us skip a
                //                 single jωC multiply when only one or
                //                 two of the three caps are present.
                //
                // None of these change correctness — they're identities
                // that the compiler can't deduce because the params are
                // loaded values, not constants.
                let lead_l_active = lb != 0.0 || lc != 0.0 || le != 0.0;
                let ser_active    = rpb != 0.0 || rpc != 0.0 || rpe != 0.0
                                  || lead_l_active;
                let cpbe_active   = cpbe != 0.0;
                let cpce_active   = cpce != 0.0;
                let cpbc_active   = cpbc != 0.0;
                let pad_active    = cpbe_active || cpce_active || cpbc_active;

                for ni in 0..n {
                    let omega = 2.0 * std::f64::consts::PI * freq_slice[ni];
                    let jw    = C::new(0.0, omega);

                    // ── Intrinsic T (mirrors _Y_int_T_batch) ──
                    let zbe = C::new(rbe, 0.0) / (one + jw * (rbe * cbe));
                    let zbc = C::new(rbc, 0.0) / (one + jw * (rbc * cbc));
                    let alpha = C::new(alpha0, 0.0)
                                * (-jw * tauc).exp()
                                / (one + jw * taub);

                    let z00 = C::new(rbi, 0.0) + zbe;
                    let z01 = zbe;
                    let z10 = zbe - alpha * zbc;
                    let z11 = (one - alpha) * zbc + zbe;

                    let inv_det_i = one / (z00 * z11 - z01 * z10);
                    let yi00 =  z11 * inv_det_i;
                    let yi01 = -z01 * inv_det_i;
                    let yi10 = -z10 * inv_det_i;
                    let yi11 =  z00 * inv_det_i;

                    // ── Extrinsic caps (Cbex / Cbcx) added in Y space ──
                    let ybex = jw * cbex;
                    let ybcx = jw * cbcx;
                    let ye00 = yi00 + ybcx + ybex;
                    let ye01 = yi01 - ybcx;
                    let ye10 = yi10 - ybcx;
                    let ye11 = yi11 + ybcx;

                    // ── Z_ser branch: invert Y_ex → +Z_ser → invert back
                    //    to Y_tot.  Skipped entirely when ser_active is
                    //    false (Y_tot == Y_ex; the two inversions cancel).
                    //    When the resistive parts are non-zero but the
                    //    inductive parts (Lb/Lc/Le) are all zero — the
                    //    typical real-device case — we skip the three
                    //    `jω·L` multiplies per (b, n).
                    let (yt00, yt01, yt10, yt11) = if ser_active {
                        // Z_ex = inv(Y_ex)
                        let inv_det_e = one / (ye00 * ye11 - ye01 * ye10);
                        let ze00 =  ye11 * inv_det_e;
                        let ze01 = -ye01 * inv_det_e;
                        let ze10 = -ye10 * inv_det_e;
                        let ze11 =  ye00 * inv_det_e;

                        let (zb, zc, ze) = if lead_l_active {
                            (C::new(rpb, 0.0) + jw * lb,
                             C::new(rpc, 0.0) + jw * lc,
                             C::new(rpe, 0.0) + jw * le)
                        } else {
                            // R-only Z_ser — real-valued, freq-independent.
                            (C::new(rpb, 0.0),
                             C::new(rpc, 0.0),
                             C::new(rpe, 0.0))
                        };

                        // Z_tot = Z_ex + Z_ser
                        let zt00 = ze00 + zb + ze;
                        let zt01 = ze01 + ze;
                        let zt10 = ze10 + ze;
                        let zt11 = ze11 + zc + ze;

                        // Y_tot = inv(Z_tot)
                        let inv_det_t = one / (zt00 * zt11 - zt01 * zt10);
                        ( zt11 * inv_det_t,
                         -zt01 * inv_det_t,
                         -zt10 * inv_det_t,
                          zt00 * inv_det_t)
                    } else {
                        // No Z_ser → Y_tot = Y_ex directly.  Saves 2 mat-
                        // rix inversions per (b, n).
                        (ye00, ye01, ye10, ye11)
                    };

                    // ── Y_pad branch: skipped entirely when pad_active
                    //    is false.  When only one or two of {Cpbe, Cpce,
                    //    Cpbc} are non-zero, we skip the zero caps'
                    //    `jω·C` multiplies individually.
                    let (ya00, ya01, ya10, ya11) = if pad_active {
                        let zeroc = C::new(0.0, 0.0);
                        let ypbe = if cpbe_active { jw * cpbe } else { zeroc };
                        let ypce = if cpce_active { jw * cpce } else { zeroc };
                        let ypbc = if cpbc_active { jw * cpbc } else { zeroc };
                        ( yt00 + ypbe + ypbc,
                          yt01 - ypbc,
                          yt10 - ypbc,
                          yt11 + ypce + ypbc)
                    } else {
                        (yt00, yt01, yt10, yt11)
                    };

                    // ── Y → S, inlined (matches _sim_wrap_batch exactly) ──
                    let yn00 = ya00 * z0c;
                    let yn01 = ya01 * z0c;
                    let yn10 = ya10 * z0c;
                    let yn11 = ya11 * z0c;

                    let m00 = one + yn00;
                    let m11 = one + yn11;
                    let inv_det_m = one / (m00 * m11 - yn01 * yn10);
                    let mi00 =  m11  * inv_det_m;
                    let mi01 = -yn01 * inv_det_m;
                    let mi10 = -yn10 * inv_det_m;
                    let mi11 =  m00  * inv_det_m;

                    let nn00 = one - yn00;
                    let nn11 = one - yn11;
                    let s00 = nn00  * mi00 + (-yn01) * mi10;
                    let s01 = nn00  * mi01 + (-yn01) * mi11;
                    let s10 = (-yn10) * mi00 + nn11  * mi10;
                    let s11 = (-yn10) * mi01 + nn11  * mi11;

                    let off = ni * 4;
                    chunk[off]     = s00;
                    chunk[off + 1] = s01;
                    chunk[off + 2] = s10;
                    chunk[off + 3] = s11;
                }
            });
    });

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn sim_cheng_pi_batch<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyDict>,
    freq:   PyReadonlyArray1<'py, f64>,
    z0:     f64,
) -> PyResult<Bound<'py, PyArray4<C>>> {
    // Same pad-extra-mode pre-flight as Cheng-T.
    if _params_use_extra_modes(params)? {
        return Err(PyNotImplementedError::new_err(
            "sim_cheng_pi_batch: pad extra-mode (Parallel L / Series L / \
             Series R / non-zero Cpar_Lb/Lc/Le) is not yet supported by \
             the Rust kernel.  Falling back to NumPy `_sim_wrap_batch`."
        ));
    }

    // ── Extract params (defaults mirror `cheng.py::_Y_int_Pi_batch`) ──
    // Cheng-π uses a Y-mesh intrinsic so Rbc default is 1e9 (≈ open)
    // rather than 1.0 — division by Rbc in the Ybc formula.
    let p_cpbe   = _extract_bcval(params, "Cpbe",   0.0)?;
    let p_cpce   = _extract_bcval(params, "Cpce",   0.0)?;
    let p_cpbc   = _extract_bcval(params, "Cpbc",   0.0)?;
    let p_lb     = _extract_bcval(params, "Lb",     0.0)?;
    let p_lc     = _extract_bcval(params, "Lc",     0.0)?;
    let p_le     = _extract_bcval(params, "Le",     0.0)?;
    let p_rpb    = _extract_bcval(params, "Rpb",    0.0)?;
    let p_rpc    = _extract_bcval(params, "Rpc",    0.0)?;
    let p_rpe    = _extract_bcval(params, "Rpe",    0.0)?;
    let p_cbex   = _extract_bcval(params, "Cbex",   0.0)?;
    let p_cbcx   = _extract_bcval(params, "Cbcx",   0.0)?;
    let p_rbi    = _extract_bcval(params, "Rbi",    0.0)?;
    let p_rbe    = _extract_bcval(params, "Rbe",    1.0)?;
    let p_cbe    = _extract_bcval(params, "Cbe",    0.0)?;
    let p_rbc    = _extract_bcval(params, "Rbc",    1.0e9)?;
    let p_cbc    = _extract_bcval(params, "Cbc",    0.0)?;
    let p_gm0    = _extract_bcval(params, "Gm0",    0.0)?;
    let p_tau    = _extract_bcval(params, "tau",    0.0)?;

    let all = [
        &p_cpbe, &p_cpce, &p_cpbc, &p_lb, &p_lc, &p_le,
        &p_rpb,  &p_rpc,  &p_rpe,  &p_cbex, &p_cbcx,
        &p_rbi,  &p_rbe,  &p_cbe,  &p_rbc,  &p_cbc, &p_gm0, &p_tau,
    ];
    let b: usize = all.iter().filter_map(|v| v.batch_len()).max().unwrap_or(1);
    for (i, v) in all.iter().enumerate() {
        if let BcVal::PerBatch(arr) = v {
            if arr.len() != b {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("sim_cheng_pi_batch: param at idx {} has len {}, expected B={}.",
                            i, arr.len(), b)
                ));
            }
        }
    }

    let freq_view = freq.as_array();
    let n = freq_view.len();
    let freq_slice = freq_view.as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "sim_cheng_pi_batch: freq array must be C-contiguous."
        ))?
        .to_vec();

    let mut out = Array4::<C>::zeros((b, n, 2, 2));
    let slab_len = n * 4;
    let flat = out.as_slice_mut().expect("C-contig");
    let one = C::new(1.0, 0.0);
    let z0c = C::new(z0, 0.0);

    py.allow_threads(|| {
        flat.par_chunks_mut(slab_len)
            .enumerate()
            .for_each(|(bi, chunk)| {
                let cpbe = p_cpbe.at(bi); let cpce = p_cpce.at(bi); let cpbc = p_cpbc.at(bi);
                let lb = p_lb.at(bi); let lc = p_lc.at(bi); let le = p_le.at(bi);
                let rpb = p_rpb.at(bi); let rpc = p_rpc.at(bi); let rpe = p_rpe.at(bi);
                let cbex = p_cbex.at(bi); let cbcx = p_cbcx.at(bi);
                let rbi = p_rbi.at(bi); let rbe = p_rbe.at(bi); let cbe = p_cbe.at(bi);
                let rbc = p_rbc.at(bi); let cbc = p_cbc.at(bi);
                let gm0 = p_gm0.at(bi); let tau = p_tau.at(bi);

                let lead_l_active = lb != 0.0 || lc != 0.0 || le != 0.0;
                let ser_active    = rpb != 0.0 || rpc != 0.0 || rpe != 0.0
                                  || lead_l_active;
                let cpbe_active   = cpbe != 0.0;
                let cpce_active   = cpce != 0.0;
                let cpbc_active   = cpbc != 0.0;
                let pad_active    = cpbe_active || cpce_active || cpbc_active;
                let inv_rbe = if rbe != 0.0 { 1.0 / rbe } else { 0.0 };
                let inv_rbc = if rbc != 0.0 { 1.0 / rbc } else { 0.0 };

                for ni in 0..n {
                    let omega = 2.0 * std::f64::consts::PI * freq_slice[ni];
                    let jw    = C::new(0.0, omega);

                    // ── Intrinsic π (mirrors _Y_int_Pi_batch) ──
                    let ybe = C::new(inv_rbe, 0.0) + jw * cbe;
                    let ybc = C::new(inv_rbc, 0.0) + jw * cbc;
                    let gm  = C::new(gm0, 0.0) * (-jw * tau).exp();

                    // Y_core
                    let yc00 = ybe + ybc;
                    let yc01 = -ybc;
                    let yc10 = gm - ybc;
                    let yc11 = ybc;
                    // Z_core = inv(Y_core); then Z_core[0,0] += Rbi
                    let inv_det_c = one / (yc00 * yc11 - yc01 * yc10);
                    let zc00 =  yc11 * inv_det_c + C::new(rbi, 0.0);
                    let zc01 = -yc01 * inv_det_c;
                    let zc10 = -yc10 * inv_det_c;
                    let zc11 =  yc00 * inv_det_c;
                    // Y_int = inv(Z_core)
                    let inv_det_i = one / (zc00 * zc11 - zc01 * zc10);
                    let yi00 =  zc11 * inv_det_i;
                    let yi01 = -zc01 * inv_det_i;
                    let yi10 = -zc10 * inv_det_i;
                    let yi11 =  zc00 * inv_det_i;

                    // Extrinsic (same as Cheng-T)
                    let ybex = jw * cbex;
                    let ybcx = jw * cbcx;
                    let ye00 = yi00 + ybcx + ybex;
                    let ye01 = yi01 - ybcx;
                    let ye10 = yi10 - ybcx;
                    let ye11 = yi11 + ybcx;

                    // Z_ser branch (per-component zero-skip — see Cheng-T comments)
                    let (yt00, yt01, yt10, yt11) = if ser_active {
                        let inv_det_e = one / (ye00 * ye11 - ye01 * ye10);
                        let ze00 =  ye11 * inv_det_e;
                        let ze01 = -ye01 * inv_det_e;
                        let ze10 = -ye10 * inv_det_e;
                        let ze11 =  ye00 * inv_det_e;

                        let (zb, zc, ze) = if lead_l_active {
                            (C::new(rpb, 0.0) + jw * lb,
                             C::new(rpc, 0.0) + jw * lc,
                             C::new(rpe, 0.0) + jw * le)
                        } else {
                            (C::new(rpb, 0.0),
                             C::new(rpc, 0.0),
                             C::new(rpe, 0.0))
                        };
                        let zt00 = ze00 + zb + ze;
                        let zt01 = ze01 + ze;
                        let zt10 = ze10 + ze;
                        let zt11 = ze11 + zc + ze;

                        let inv_det_t = one / (zt00 * zt11 - zt01 * zt10);
                        ( zt11 * inv_det_t, -zt01 * inv_det_t,
                         -zt10 * inv_det_t,  zt00 * inv_det_t)
                    } else {
                        (ye00, ye01, ye10, ye11)
                    };

                    let (ya00, ya01, ya10, ya11) = if pad_active {
                        let zeroc = C::new(0.0, 0.0);
                        let ypbe = if cpbe_active { jw * cpbe } else { zeroc };
                        let ypce = if cpce_active { jw * cpce } else { zeroc };
                        let ypbc = if cpbc_active { jw * cpbc } else { zeroc };
                        ( yt00 + ypbe + ypbc, yt01 - ypbc,
                          yt10 - ypbc,        yt11 + ypce + ypbc)
                    } else {
                        (yt00, yt01, yt10, yt11)
                    };

                    // Y → S
                    let yn00 = ya00 * z0c;
                    let yn01 = ya01 * z0c;
                    let yn10 = ya10 * z0c;
                    let yn11 = ya11 * z0c;
                    let m00 = one + yn00;
                    let m11 = one + yn11;
                    let inv_det_m = one / (m00 * m11 - yn01 * yn10);
                    let mi00 =  m11  * inv_det_m;
                    let mi01 = -yn01 * inv_det_m;
                    let mi10 = -yn10 * inv_det_m;
                    let mi11 =  m00  * inv_det_m;
                    let nn00 = one - yn00;
                    let nn11 = one - yn11;
                    let s00 = nn00 * mi00 + (-yn01) * mi10;
                    let s01 = nn00 * mi01 + (-yn01) * mi11;
                    let s10 = (-yn10) * mi00 + nn11 * mi10;
                    let s11 = (-yn10) * mi01 + nn11 * mi11;

                    let off = ni * 4;
                    chunk[off]     = s00;
                    chunk[off + 1] = s01;
                    chunk[off + 2] = s10;
                    chunk[off + 3] = s11;
                }
            });
    });

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn sim_xu_t_batch<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyDict>,
    freq:   PyReadonlyArray1<'py, f64>,
    z0:     f64,
) -> PyResult<Bound<'py, PyArray4<C>>> {
    if _params_use_extra_modes(params)? {
        return Err(PyNotImplementedError::new_err(
            "sim_xu_t_batch: pad extra-mode not supported by Rust kernel; \
             falling back to NumPy."
        ));
    }

    // Defaults match `xu.py::_sim_wrap_batch` and `_Y_int_T_batch`.
    // Note `Rbcx` defaults to 285 kΩ (Xu's _RBCX_DEFAULT) — the model
    // uses Rbcx for the parallel Rbcx ‖ Cbcx extrinsic branch.
    let p_cpbe   = _extract_bcval(params, "Cpbe",   0.0)?;
    let p_cpce   = _extract_bcval(params, "Cpce",   0.0)?;
    let p_cpbc   = _extract_bcval(params, "Cpbc",   0.0)?;
    let p_lb     = _extract_bcval(params, "Lb",     0.0)?;
    let p_lc     = _extract_bcval(params, "Lc",     0.0)?;
    let p_le     = _extract_bcval(params, "Le",     0.0)?;
    let p_rpb    = _extract_bcval(params, "Rpb",    0.0)?;
    let p_rpc    = _extract_bcval(params, "Rpc",    0.0)?;
    let p_rpe    = _extract_bcval(params, "Rpe",    0.0)?;
    let p_cbcx   = _extract_bcval(params, "Cbcx",   0.0)?;
    let p_rbcx   = _extract_bcval(params, "Rbcx",   285_000.0)?;
    let p_rbi    = _extract_bcval(params, "Rbi",    0.0)?;
    let p_rbe    = _extract_bcval(params, "Rbe",    1.0)?;
    let p_cbe    = _extract_bcval(params, "Cbe",    0.0)?;
    let p_rbc    = _extract_bcval(params, "Rbc",    1.0)?;
    let p_cbc    = _extract_bcval(params, "Cbc",    0.0)?;
    let p_alpha0 = _extract_bcval(params, "alpha0", 0.0)?;
    let p_taub   = _extract_bcval(params, "tauB",   0.0)?;
    let p_tauc   = _extract_bcval(params, "tauC",   0.0)?;

    let all = [
        &p_cpbe, &p_cpce, &p_cpbc, &p_lb, &p_lc, &p_le,
        &p_rpb,  &p_rpc,  &p_rpe,  &p_cbcx, &p_rbcx,
        &p_rbi,  &p_rbe,  &p_cbe,  &p_rbc,  &p_cbc,
        &p_alpha0, &p_taub, &p_tauc,
    ];
    let b: usize = all.iter().filter_map(|v| v.batch_len()).max().unwrap_or(1);
    for (i, v) in all.iter().enumerate() {
        if let BcVal::PerBatch(arr) = v {
            if arr.len() != b {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("sim_xu_t_batch: param at idx {} has len {}, expected B={}.",
                            i, arr.len(), b)
                ));
            }
        }
    }

    let freq_view = freq.as_array();
    let n = freq_view.len();
    let freq_slice = freq_view.as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "sim_xu_t_batch: freq array must be C-contiguous."
        ))?
        .to_vec();

    let mut out = Array4::<C>::zeros((b, n, 2, 2));
    let slab_len = n * 4;
    let flat = out.as_slice_mut().expect("C-contig");
    let one = C::new(1.0, 0.0);
    let z0c = C::new(z0, 0.0);

    py.allow_threads(|| {
        flat.par_chunks_mut(slab_len)
            .enumerate()
            .for_each(|(bi, chunk)| {
                let cpbe = p_cpbe.at(bi); let cpce = p_cpce.at(bi); let cpbc = p_cpbc.at(bi);
                let lb = p_lb.at(bi); let lc = p_lc.at(bi); let le = p_le.at(bi);
                let rpb = p_rpb.at(bi); let rpc = p_rpc.at(bi); let rpe = p_rpe.at(bi);
                let cbcx = p_cbcx.at(bi); let rbcx = p_rbcx.at(bi);
                let rbi = p_rbi.at(bi); let rbe = p_rbe.at(bi); let cbe = p_cbe.at(bi);
                let rbc = p_rbc.at(bi); let cbc = p_cbc.at(bi);
                let alpha0 = p_alpha0.at(bi); let taub = p_taub.at(bi); let tauc = p_tauc.at(bi);

                let lead_l_active = lb != 0.0 || lc != 0.0 || le != 0.0;
                let ser_active    = rpb != 0.0 || rpc != 0.0 || rpe != 0.0
                                  || lead_l_active;
                let cpbe_active   = cpbe != 0.0;
                let cpce_active   = cpce != 0.0;
                let cpbc_active   = cpbc != 0.0;
                let pad_active    = cpbe_active || cpce_active || cpbc_active;
                let inv_rbcx = if rbcx != 0.0 { 1.0 / rbcx } else { 0.0 };

                for ni in 0..n {
                    let omega = 2.0 * std::f64::consts::PI * freq_slice[ni];
                    let jw    = C::new(0.0, omega);

                    // Intrinsic T (same as Cheng-T)
                    let zbe = C::new(rbe, 0.0) / (one + jw * (rbe * cbe));
                    let zbc = C::new(rbc, 0.0) / (one + jw * (rbc * cbc));
                    let alpha = C::new(alpha0, 0.0)
                                * (-jw * tauc).exp()
                                / (one + jw * taub);

                    let z00 = C::new(rbi, 0.0) + zbe;
                    let z01 = zbe;
                    let z10 = zbe - alpha * zbc;
                    let z11 = (one - alpha) * zbc + zbe;

                    let inv_det_i = one / (z00 * z11 - z01 * z10);
                    let yi00 =  z11 * inv_det_i;
                    let yi01 = -z01 * inv_det_i;
                    let yi10 = -z10 * inv_det_i;
                    let yi11 =  z00 * inv_det_i;

                    // Xu extrinsic: Ybcx = 1/Rbcx + jωCbcx, applied with
                    // [[+1,-1],[-1,+1]] pattern (no separate Ybex term).
                    let ybcx = C::new(inv_rbcx, 0.0) + jw * cbcx;
                    let ye00 = yi00 + ybcx;
                    let ye01 = yi01 - ybcx;
                    let ye10 = yi10 - ybcx;
                    let ye11 = yi11 + ybcx;

                    // Z_ser branch (per-component zero-skip — see Cheng-T comments)
                    let (yt00, yt01, yt10, yt11) = if ser_active {
                        let inv_det_e = one / (ye00 * ye11 - ye01 * ye10);
                        let ze00 =  ye11 * inv_det_e;
                        let ze01 = -ye01 * inv_det_e;
                        let ze10 = -ye10 * inv_det_e;
                        let ze11 =  ye00 * inv_det_e;

                        let (zb, zc, ze) = if lead_l_active {
                            (C::new(rpb, 0.0) + jw * lb,
                             C::new(rpc, 0.0) + jw * lc,
                             C::new(rpe, 0.0) + jw * le)
                        } else {
                            (C::new(rpb, 0.0),
                             C::new(rpc, 0.0),
                             C::new(rpe, 0.0))
                        };
                        let zt00 = ze00 + zb + ze;
                        let zt01 = ze01 + ze;
                        let zt10 = ze10 + ze;
                        let zt11 = ze11 + zc + ze;

                        let inv_det_t = one / (zt00 * zt11 - zt01 * zt10);
                        ( zt11 * inv_det_t, -zt01 * inv_det_t,
                         -zt10 * inv_det_t,  zt00 * inv_det_t)
                    } else {
                        (ye00, ye01, ye10, ye11)
                    };

                    let (ya00, ya01, ya10, ya11) = if pad_active {
                        let zeroc = C::new(0.0, 0.0);
                        let ypbe = if cpbe_active { jw * cpbe } else { zeroc };
                        let ypce = if cpce_active { jw * cpce } else { zeroc };
                        let ypbc = if cpbc_active { jw * cpbc } else { zeroc };
                        ( yt00 + ypbe + ypbc, yt01 - ypbc,
                          yt10 - ypbc,        yt11 + ypce + ypbc)
                    } else {
                        (yt00, yt01, yt10, yt11)
                    };

                    // Y → S
                    let yn00 = ya00 * z0c;
                    let yn01 = ya01 * z0c;
                    let yn10 = ya10 * z0c;
                    let yn11 = ya11 * z0c;
                    let m00 = one + yn00;
                    let m11 = one + yn11;
                    let inv_det_m = one / (m00 * m11 - yn01 * yn10);
                    let mi00 =  m11  * inv_det_m;
                    let mi01 = -yn01 * inv_det_m;
                    let mi10 = -yn10 * inv_det_m;
                    let mi11 =  m00  * inv_det_m;
                    let nn00 = one - yn00;
                    let nn11 = one - yn11;
                    let s00 = nn00 * mi00 + (-yn01) * mi10;
                    let s01 = nn00 * mi01 + (-yn01) * mi11;
                    let s10 = (-yn10) * mi00 + nn11 * mi10;
                    let s11 = (-yn10) * mi01 + nn11 * mi11;

                    let off = ni * 4;
                    chunk[off]     = s00;
                    chunk[off + 1] = s01;
                    chunk[off + 2] = s10;
                    chunk[off + 3] = s11;
                }
            });
    });

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn sim_kunyang_batch<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyDict>,
    freq:   PyReadonlyArray1<'py, f64>,
    z0:     f64,
) -> PyResult<Bound<'py, PyArray4<C>>> {
    // Kun-Yang model has NO standard Cpbe/Cpce/Cpbc pad layer (the KY
    // substrate network replaces it).  Cpar_L* parasitic-cap modes are
    // also ignored — they were a HBT-only de-embedding feature.  No
    // _params_use_extra_modes() check here.

    // Series-lead + access-resistance (re-uses HBT key names — see
    // models/kunyang.py).
    let p_lb     = _extract_bcval(params, "Lb",     0.0)?;
    let p_lc     = _extract_bcval(params, "Lc",     0.0)?;
    let p_le     = _extract_bcval(params, "Le",     0.0)?;
    let p_rpb    = _extract_bcval(params, "Rpb",    0.0)?;
    let p_rpc    = _extract_bcval(params, "Rpc",    0.0)?;
    let p_rpe    = _extract_bcval(params, "Rpe",    0.0)?;
    // Source-side delay network (R_delay ‖ C_delay in series with Rs+jωLs)
    let p_rdelay = _extract_bcval(params, "R_delay", 0.0)?;
    let p_cdelay = _extract_bcval(params, "C_delay", 0.0)?;
    // Intrinsic pi-model
    let p_cgs    = _extract_bcval(params, "Cgs",    0.0)?;
    let p_ri     = _extract_bcval(params, "Ri",     0.0)?;
    let p_cgd    = _extract_bcval(params, "Cgd",    0.0)?;
    let p_rgd    = _extract_bcval(params, "Rgd",    0.0)?;
    let p_cds    = _extract_bcval(params, "Cds",    0.0)?;
    let p_rds    = _extract_bcval(params, "Rds",    1.0)?;
    let p_gm0    = _extract_bcval(params, "Gm0",    0.0)?;
    let p_tau    = _extract_bcval(params, "tau",    0.0)?;
    // Kun-Yang custom pad / substrate
    let p_cgsp   = _extract_bcval(params, "Cgsp",   0.0)?;
    let p_rsub1  = _extract_bcval(params, "Rsub1",  0.0)?;
    let p_cdsp   = _extract_bcval(params, "Cdsp",   0.0)?;
    let p_rsub2  = _extract_bcval(params, "Rsub2",  0.0)?;
    let p_cgdp   = _extract_bcval(params, "Cgdp",   0.0)?;

    let all = [
        &p_lb, &p_lc, &p_le, &p_rpb, &p_rpc, &p_rpe,
        &p_rdelay, &p_cdelay,
        &p_cgs, &p_ri, &p_cgd, &p_rgd, &p_cds, &p_rds, &p_gm0, &p_tau,
        &p_cgsp, &p_rsub1, &p_cdsp, &p_rsub2, &p_cgdp,
    ];
    let b: usize = all.iter().filter_map(|v| v.batch_len()).max().unwrap_or(1);
    for (i, v) in all.iter().enumerate() {
        if let BcVal::PerBatch(arr) = v {
            if arr.len() != b {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("sim_kunyang_batch: param at idx {} has len {}, expected B={}.",
                            i, arr.len(), b)
                ));
            }
        }
    }

    let freq_view = freq.as_array();
    let n = freq_view.len();
    let freq_slice = freq_view.as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "sim_kunyang_batch: freq array must be C-contiguous."
        ))?
        .to_vec();

    let mut out = Array4::<C>::zeros((b, n, 2, 2));
    let slab_len = n * 4;
    let flat = out.as_slice_mut().expect("C-contig");
    let one = C::new(1.0, 0.0);
    let zeroc = C::new(0.0, 0.0);
    let z0c = C::new(z0, 0.0);

    py.allow_threads(|| {
        flat.par_chunks_mut(slab_len)
            .enumerate()
            .for_each(|(bi, chunk)| {
                // Per-batch scalar loads.
                let lb = p_lb.at(bi); let lc = p_lc.at(bi); let le = p_le.at(bi);
                let rpb = p_rpb.at(bi); let rpc = p_rpc.at(bi); let rpe = p_rpe.at(bi);
                let r_d = p_rdelay.at(bi); let c_d = p_cdelay.at(bi);
                let cgs = p_cgs.at(bi); let ri  = p_ri.at(bi);
                let cgd = p_cgd.at(bi); let rgd = p_rgd.at(bi);
                let cds = p_cds.at(bi); let rds = p_rds.at(bi);
                let gm0 = p_gm0.at(bi); let tau = p_tau.at(bi);
                let cgsp = p_cgsp.at(bi); let rsub1 = p_rsub1.at(bi);
                let cdsp = p_cdsp.at(bi); let rsub2 = p_rsub2.at(bi);
                let cgdp = p_cgdp.at(bi);

                // Loop-invariant compositional shortcuts.
                let lead_l_active  = lb != 0.0 || lc != 0.0 || le != 0.0;
                let ser_active     = rpb != 0.0 || rpc != 0.0 || rpe != 0.0
                                   || lead_l_active;
                let delay_active   = r_d != 0.0;   // r_d == 0 → Z_delay = 0
                let ky_pad_active  = cgsp != 0.0 || cdsp != 0.0 || cgdp != 0.0;
                // Y_ds = 1/Rds + jω·Cds; guard against Rds == 0 (NumPy would
                // emit inf; we treat it as the open-circuit term being skipped).
                let inv_rds = if rds != 0.0 { 1.0 / rds } else { 0.0 };

                for ni in 0..n {
                    let omega = 2.0 * std::f64::consts::PI * freq_slice[ni];
                    let jw    = C::new(0.0, omega);

                    // ── Intrinsic pi-model ────────────────────────────────
                    //   Y_gs = jωCgs / (1 + jω·Ri·Cgs)
                    //   Y_gd = jωCgd / (1 + jω·Rgd·Cgd)
                    //   Y_ds = 1/Rds + jωCds
                    //   gm   = Gm0 · exp(-jωτ)
                    let jw_cgs = jw * cgs;
                    let y_gs   = jw_cgs / (one + jw_cgs * ri);
                    let jw_cgd = jw * cgd;
                    let y_gd   = jw_cgd / (one + jw_cgd * rgd);
                    let y_ds   = C::new(inv_rds, 0.0) + jw * cds;
                    let gm     = C::new(gm0, 0.0) * (-jw * tau).exp();

                    // Y_in pi-matrix:
                    //   [ Y_gs+Y_gd      -Y_gd     ]
                    //   [ gm  - Y_gd     Y_ds+Y_gd ]
                    let yi00 = y_gs + y_gd;
                    let yi01 = -y_gd;
                    let yi10 = gm - y_gd;
                    let yi11 = y_ds + y_gd;

                    // ── Z_DUT = inv(Y_in) + Z_ser (+ Z_delay on source) ──
                    //  Z_delay = R_delay / (1 + jω·R_delay·C_delay), added to
                    //  every element of Z_ser since the indefinite-T matrix
                    //  carries Z_source in all four entries.
                    let z_delay = if delay_active {
                        C::new(r_d, 0.0) / (one + jw * (r_d * c_d))
                    } else {
                        zeroc
                    };

                    // Build Z_ser planes (R + jωL on diag, common source on
                    // off-diag) — analytic shortcut when L's are all zero.
                    let (zb, zc, ze) = if lead_l_active {
                        (C::new(rpb, 0.0) + jw * lb,
                         C::new(rpc, 0.0) + jw * lc,
                         C::new(rpe, 0.0) + jw * le)
                    } else {
                        (C::new(rpb, 0.0),
                         C::new(rpc, 0.0),
                         C::new(rpe, 0.0))
                    };
                    // Add Z_delay onto the common source path (all 4 elems).
                    let ze_eff = ze + z_delay;

                    let (yd00, yd01, yd10, yd11) = if ser_active || delay_active {
                        // Z_in = inv(Y_in)
                        let inv_det_i = one / (yi00 * yi11 - yi01 * yi10);
                        let zi00 =  yi11 * inv_det_i;
                        let zi01 = -yi01 * inv_det_i;
                        let zi10 = -yi10 * inv_det_i;
                        let zi11 =  yi00 * inv_det_i;

                        let zt00 = zi00 + zb + ze_eff;
                        let zt01 = zi01 + ze_eff;
                        let zt10 = zi10 + ze_eff;
                        let zt11 = zi11 + zc + ze_eff;

                        // Y_DUT = inv(Z_tot)
                        let inv_det_t = one / (zt00 * zt11 - zt01 * zt10);
                        ( zt11 * inv_det_t, -zt01 * inv_det_t,
                         -zt10 * inv_det_t,  zt00 * inv_det_t)
                    } else {
                        // Z_ser == 0 and Z_delay == 0 → Y_DUT == Y_in.
                        (yi00, yi01, yi10, yi11)
                    };

                    // ── Kun-Yang custom pad (parallel) ────────────────────
                    //   Y_gsp = jωCgsp / (1 + jω·Rsub1·Cgsp)
                    //   Y_dsp = jωCdsp / (1 + jω·Rsub2·Cdsp)
                    //   Y_gdp = jωCgdp
                    let (ya00, ya01, ya10, ya11) = if ky_pad_active {
                        let jw_cgsp = jw * cgsp;
                        let jw_cdsp = jw * cdsp;
                        let y_gsp = jw_cgsp / (one + jw_cgsp * rsub1);
                        let y_dsp = jw_cdsp / (one + jw_cdsp * rsub2);
                        let y_gdp = jw * cgdp;
                        ( yd00 + y_gsp + y_gdp,
                          yd01 - y_gdp,
                          yd10 - y_gdp,
                          yd11 + y_dsp + y_gdp)
                    } else {
                        (yd00, yd01, yd10, yd11)
                    };

                    // ── Y → S (inlined, matches y_to_s_vec exactly) ───────
                    let yn00 = ya00 * z0c;
                    let yn01 = ya01 * z0c;
                    let yn10 = ya10 * z0c;
                    let yn11 = ya11 * z0c;
                    let m00 = one + yn00;
                    let m11 = one + yn11;
                    let inv_det_m = one / (m00 * m11 - yn01 * yn10);
                    let mi00 =  m11  * inv_det_m;
                    let mi01 = -yn01 * inv_det_m;
                    let mi10 = -yn10 * inv_det_m;
                    let mi11 =  m00  * inv_det_m;
                    let nn00 = one - yn00;
                    let nn11 = one - yn11;
                    let s00 = nn00 * mi00 + (-yn01) * mi10;
                    let s01 = nn00 * mi01 + (-yn01) * mi11;
                    let s10 = (-yn10) * mi00 + nn11 * mi10;
                    let s11 = (-yn10) * mi01 + nn11 * mi11;

                    let off = ni * 4;
                    chunk[off]     = s00;
                    chunk[off + 1] = s01;
                    chunk[off + 2] = s10;
                    chunk[off + 3] = s11;
                }
            });
    });

    Ok(out.into_pyarray_bound(py))
}

// ── Phase 1.5 — bulk-upload parse + metrics, parallel across files ──────────
//
// Replaces the Python `parse_s2p → s_to_y → compute_metrics` chain that the
// IOED bulk-upload loop runs per-file.  Profiling on N=1001-pt sweeps:
//   parse_s2p          5.3 ms  (65 %)
//   compute_metrics    0.9 ms  (11 %)
//   s_to_y             0.2 ms  ( 3 %)
//   ───────────────────────────────────
//   total per file     ~6.5 ms  →  30 files = ~200 ms serial in Python
//
// In Rust with Rayon across files: ~200 / cores wall-clock, GIL released
// during the parallel block.
//
// Output schema (one PyDict per file, in input order):
//   freq          : (N,)         f64
//   S             : (N, 2, 2)    complex128   (Touchstone S11/S12/S21/S22)
//   z0            : float
//   h21_db        : (N,)         f64
//   u_db          : (N,)         f64
//   mag_db        : (N,)         f64
//   k             : (N,)         f64
//   ft_plat       : (N,)         f64
//   fmax_u_plat   : (N,)         f64
//   fmax_mag_plat : (N,)         f64
//
// Or, on per-file parse failure:
//   error : str    (only this key set; downstream Python treats as failure)
//
// CSV files / extracted Touchstone variants the Python parser handles but
// this Rust port doesn't yet (notably .csv) should NOT be passed here —
// the wrapper dispatches them to the Python fallback.

// ── extract_limit Rust impl ─────────────────────────────────────────────────
//
// Mirrors `helpers/metrics.extract_limit` exactly.  Returns a fT/fmax value
// from a (gain, plateau) trace pair using a "genuine 0-dB crossing" search
// (gain stays above 0 for >=10 consecutive points) with a log-linear
// extrapolation fallback.  Called 3× per file (h21 → fT, U → fmax_U,
// MAG/MSG → fmax_MAG) inside the parallel parse_and_compute_batch loop —
// this saves ~1.6 ms/file of Python work that used to run downstream.
//
// method codes are exported to Python as small ints to avoid allocating
// PyString per file; the Python wrapper maps these back to the same string
// labels the Python `extract_limit` returned ("No Data", "No Gain",
// "0dB Cross", "Extrap & Plat.").

const METHOD_NO_DATA:     u8 = 0;
const METHOD_NO_GAIN:     u8 = 1;
const METHOD_ZERO_CROSS:  u8 = 2;
const METHOD_EXTRAP_PLAT: u8 = 3;

/// Linear least-squares fit (degree 1) returning `(slope, intercept)`.
/// Caller must ensure x.len() == y.len() >= 2.
#[inline]
fn _polyfit_1(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n  = x.len() as f64;
    let sx : f64 = x.iter().sum();
    let sy : f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let denom = n * sxx - sx * sx;
    let m = (n * sxy - sx * sy) / denom;
    let c = (sy - m * sx) / n;
    (m, c)
}

/// Polynomial fit of degree `deg ∈ {1, 2}` of (x, y) pairs, evaluated at
/// `x = 0`.  Returns the constant-term of the fitted polynomial (which is
/// what `np.polyval(np.polyfit(g, f, deg), 0.0)` computes when the caller
/// is searching for the 0-dB crossing of f(g) ≈ poly(g)).
///
/// Degree 1 uses closed-form least-squares; degree 2 solves the 3×3 normal
/// equations via Cramer's rule.  Both match `np.polyfit` to within fp64
/// rounding for well-conditioned inputs.
fn _polyfit_eval0(x: &[f64], y: &[f64], deg: usize) -> f64 {
    if x.len() < 2 { return f64::NAN; }
    if deg <= 1 || x.len() < 3 {
        // y(0) = intercept of linear fit.
        let (_m, c) = _polyfit_1(x, y);
        return c;
    }
    // Degree-2 least squares: minimise Σ (yi − (a xi² + b xi + c))².
    // Normal equations:  [s4 s3 s2; s3 s2 s1; s2 s1 n] · [a b c]ᵀ = [t2 t1 t0]ᵀ
    let n  = x.len() as f64;
    let s1: f64 = x.iter().sum();
    let s2: f64 = x.iter().map(|v| v * v).sum();
    let s3: f64 = x.iter().map(|v| v * v * v).sum();
    let s4: f64 = x.iter().map(|v| v.powi(4)).sum();
    let t0: f64 = y.iter().sum();
    let t1: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let t2: f64 = x.iter().zip(y).map(|(a, b)| a * a * b).sum();
    // We only need `c` (the constant term).  Solve via Cramer.
    let det =
        s4 * (s2 * n - s1 * s1)
      - s3 * (s3 * n - s1 * s2)
      + s2 * (s3 * s1 - s2 * s2);
    if det == 0.0 || !det.is_finite() {
        // Fall back to linear if the quadratic system is singular.
        let (_m, c) = _polyfit_1(x, y);
        return c;
    }
    let c =
        s4 * (s2 * t0 - s1 * t1)
      - s3 * (s3 * t0 - s1 * t2)
      + t2 * (s3 * s1 - s2 * s2);
    c / det
}

#[derive(Default, Clone, Copy)]
struct ExtractResult {
    cross:  f64,
    plat:   f64,
    method: u8,
}

/// Port of `helpers/metrics.extract_limit`.  See that docstring for the
/// genuine-crossing logic; we mirror it line-for-line so the per-file
/// (fT, fmax_U, fmax_MAG) values match what the Python downstream would
/// have produced.
fn _extract_limit_rust(
    freq_ghz: &[f64],
    gain_db:  &[f64],
    plateau:  &[f64],
    n_pts:    usize,
    f_min:    f64,
    f_max:    f64,
) -> ExtractResult {
    let nn = freq_ghz.len();
    // Mask = in-window & finite gain.  We materialise the valid indices
    // rather than three separate Vecs upfront, then build f_v/g_v/p_v
    // once we know N — saves a few allocations on the parse-failure path.
    let mut valid: Vec<usize> = Vec::with_capacity(nn);
    for i in 0..nn {
        if freq_ghz[i] >= f_min && freq_ghz[i] <= f_max && gain_db[i].is_finite() {
            valid.push(i);
        }
    }
    if valid.is_empty() {
        return ExtractResult { cross: f64::NAN, plat: f64::NAN, method: METHOD_NO_DATA };
    }
    let n = valid.len();
    let f_v: Vec<f64> = valid.iter().map(|&i| freq_ghz[i]).collect();
    let g_v: Vec<f64> = valid.iter().map(|&i| gain_db[i]).collect();
    let p_v: Vec<f64> = valid.iter().map(|&i| plateau[i]).collect();

    let max_g = g_v.iter().copied()
        .fold(f64::NEG_INFINITY,
              |a, b| if b.is_nan() { a } else { a.max(b) });
    if max_g <= 0.0 {
        return ExtractResult { cross: f64::NAN, plat: f64::NAN, method: METHOD_NO_GAIN };
    }

    // Genuine-crossing search (mirrors the Python `for idx in crossings[::-1]`).
    let above: Vec<bool> = g_v.iter().map(|&v| v >= 0.0).collect();
    let mut crossings: Vec<usize> = Vec::new();
    for i in 0..n.saturating_sub(1) {
        if above[i] && !above[i + 1] {
            crossings.push(i);
        }
    }
    let mut genuine_idx: Option<usize> = None;
    let n80 = (0.80 * n as f64) as usize;
    'outer: for &idx in crossings.iter().rev() {
        // Count consecutive above-zero points ending at idx.
        let mut cnt = 0usize;
        let mut j = idx as isize;
        while j >= 0 {
            if above[j as usize] { cnt += 1; } else { break; }
            j -= 1;
        }
        if cnt < 10 { continue; }
        // Skip if the entire above-zero run starts in the top 20% of the
        // window AND the run is short — that's almost certainly a tail
        // noise excursion above 0 dB, not a real fT crossing.
        let run_start = idx.saturating_sub(cnt - 1);
        if run_start > n80 && cnt < 20 { continue; }
        genuine_idx = Some(idx);
        break 'outer;
    }

    if let Some(idx) = genuine_idx {
        // Polyfit window around the crossing — matches Python's
        //   s,e = max(0,idx-n_pts//2+1), min(N,idx+n_pts//2+1+(n_pts%2))
        // arithmetic exactly (with saturating math so we don't underflow
        // when idx < n_pts/2).
        let half = n_pts / 2;
        let mut s = idx.saturating_sub(half).saturating_add(1).min(n);
        let mut e = (idx + half + 1 + (n_pts % 2)).min(n);
        if e.saturating_sub(s) < 2 {
            s = idx.min(n);
            e = (idx + 2).min(n);
        }
        let xs = &g_v[s..e];
        let ys = &f_v[s..e];
        let deg = (e - s - 1).min(2);
        let mut v_cross = _polyfit_eval0(xs, ys, deg);

        // Sanity-check: if the polyfit overshot the window or went
        // negative, fall back to two-point linear interpolation across
        // the actual crossing pair.
        let bad = !v_cross.is_finite() || v_cross <= 0.0
                  || v_cross < f_v[s] || v_cross > f_v[e - 1];
        if bad && idx + 1 < n {
            let dy = g_v[idx + 1] - g_v[idx];
            if dy != 0.0 {
                v_cross = f_v[idx] + (0.0 - g_v[idx]) * (f_v[idx + 1] - f_v[idx]) / dy;
            }
        }
        return ExtractResult { cross: v_cross, plat: f64::NAN, method: METHOD_ZERO_CROSS };
    }

    // Extrap & plat. fallback — log-linear fit of the last n_use points.
    let median = {
        let mut sorted: Vec<f64> = g_v.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() { 0.0 } else {
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // NumPy median: average of the two middle elements for even N.
            let m = sorted.len();
            if m % 2 == 1 { sorted[m / 2] } else { 0.5 * (sorted[m / 2 - 1] + sorted[m / 2]) }
        }
    };
    if median > 0.0 {
        let mut v_plat = f64::NEG_INFINITY;
        let mut any_finite = false;
        for &v in &p_v {
            if v.is_finite() { v_plat = v_plat.max(v); any_finite = true; }
        }
        let v_plat = if any_finite { v_plat } else { f64::NAN };

        let n_use = n_pts.min(f_v.len());
        let mut v_extrap = f64::NAN;
        if n_use >= 2 {
            let tail = f_v.len() - n_use;
            let logf: Vec<f64> = f_v[tail..].iter().map(|v| v.log10()).collect();
            let gtail: Vec<f64> = g_v[tail..].to_vec();
            let (m, c) = _polyfit_1(&logf, &gtail);
            if m < 0.0 && m.is_finite() && c.is_finite() {
                v_extrap = 10f64.powf(-c / m);
            }
        }
        return ExtractResult { cross: v_extrap, plat: v_plat, method: METHOD_EXTRAP_PLAT };
    }
    ExtractResult { cross: f64::NAN, plat: f64::NAN, method: METHOD_NO_GAIN }
}

struct DUTRustResult {
    freq:          Vec<f64>,
    s_flat:        Vec<Complex64>,   // length n*4 row-major (N, 2, 2)
    z0:            f64,
    h21_db:        Vec<f64>,
    u_db:          Vec<f64>,
    mag_db:        Vec<f64>,
    k:             Vec<f64>,
    ft_plat:       Vec<f64>,
    fmax_u_plat:   Vec<f64>,
    fmax_mag_plat: Vec<f64>,
    // Per-file extract_limit results (filled after compute_metrics).
    ext_ft:        ExtractResult,
    ext_fmax_u:    ExtractResult,
    ext_fmax_mag:  ExtractResult,
    error:         Option<String>,
}

impl DUTRustResult {
    fn empty_err(msg: impl Into<String>) -> Self {
        Self {
            freq: Vec::new(), s_flat: Vec::new(), z0: 50.0,
            h21_db: Vec::new(), u_db: Vec::new(), mag_db: Vec::new(),
            k: Vec::new(), ft_plat: Vec::new(),
            fmax_u_plat: Vec::new(), fmax_mag_plat: Vec::new(),
            ext_ft: ExtractResult::default(),
            ext_fmax_u: ExtractResult::default(),
            ext_fmax_mag: ExtractResult::default(),
            error: Some(msg.into()),
        }
    }
}

#[inline(always)]
fn _parse_one_s(a: f64, b: f64, fmt: u8) -> Complex64 {
    // fmt: 0 = MA (mag/ang°), 1 = DB (dB/ang°), 2 = RI (real/imag)
    match fmt {
        1 => Complex64::from_polar(10.0_f64.powf(a / 20.0), b.to_radians()),
        2 => Complex64::new(a, b),
        _ => Complex64::from_polar(a, b.to_radians()),  // MA is the default
    }
}

fn _parse_s2p_rust(content: &[u8]) -> Result<(Vec<f64>, Vec<Complex64>, f64), String> {
    // Decode lossily — matches the Python `decode("utf-8", errors="ignore")`.
    let text = std::str::from_utf8(content)
        .map_or_else(|_| String::from_utf8_lossy(content).into_owned(),
                     |s| s.to_string());

    let mut freq_unit_scale: f64 = 1.0;  // hz default
    let mut fmt: u8 = 0;                  // MA default
    let mut z0: f64 = 50.0;
    // Pre-size for a typical 1000-pt sweep (9 floats per row).
    let mut data_vals: Vec<f64> = Vec::with_capacity(1024 * 9);

    for raw_line in text.lines() {
        // Touchstone allows trailing "!" comments on any line — cut them off
        // BEFORE tokenising so a comment containing numbers (e.g. "! 25 mA")
        // can't shift the 9-column data alignment and scramble the S-params.
        let line = raw_line.split('!').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            let lower: String = line[1..].to_ascii_lowercase();
            let parts: Vec<&str> = lower.split_whitespace().collect();
            for (i, p) in parts.iter().enumerate() {
                match *p {
                    "hz"  => freq_unit_scale = 1.0,
                    "khz" => freq_unit_scale = 1e3,
                    "mhz" => freq_unit_scale = 1e6,
                    "ghz" => freq_unit_scale = 1e9,
                    "ma"  => fmt = 0,
                    "db"  => fmt = 1,
                    "ri"  => fmt = 2,
                    "r"   => if let Some(nxt) = parts.get(i + 1) {
                        if let Ok(v) = nxt.parse() { z0 = v; }
                    },
                    _ => {}
                }
            }
            continue;
        }
        // Data row — split on whitespace and parse floats.  Ignore any
        // token that doesn't parse (matches Python's tolerant behaviour
        // for stray inline comments).
        for tok in line.split_whitespace() {
            if let Ok(v) = tok.parse::<f64>() {
                data_vals.push(v);
            }
        }
    }

    let n = data_vals.len() / 9;
    if n == 0 {
        return Err("parse_s2p: no 9-column data rows found".to_string());
    }

    let mut freq = Vec::with_capacity(n);
    let mut s_flat = vec![Complex64::new(0.0, 0.0); n * 4];
    for i in 0..n {
        let row = &data_vals[i * 9..(i + 1) * 9];
        freq.push(row[0] * freq_unit_scale);
        // Touchstone column order: S11, S21, S12, S22 at (1,2)(3,4)(5,6)(7,8).
        // Output layout row-major (N, 2, 2) → flat indices [s11, s12, s21, s22].
        let base = i * 4;
        s_flat[base + 0] = _parse_one_s(row[1], row[2], fmt);   // S11
        s_flat[base + 2] = _parse_one_s(row[3], row[4], fmt);   // S21
        s_flat[base + 1] = _parse_one_s(row[5], row[6], fmt);   // S12
        s_flat[base + 3] = _parse_one_s(row[7], row[8], fmt);   // S22
    }
    Ok((freq, s_flat, z0))
}

fn _compute_metrics_rust(r: &mut DUTRustResult) {
    let n = r.freq.len();
    let z0c = Complex64::new(r.z0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let eps = Complex64::new(1e-30, 0.0);

    r.h21_db        = Vec::with_capacity(n);
    r.u_db          = Vec::with_capacity(n);
    r.mag_db        = Vec::with_capacity(n);
    r.k             = Vec::with_capacity(n);
    r.ft_plat       = Vec::with_capacity(n);
    r.fmax_u_plat   = Vec::with_capacity(n);
    r.fmax_mag_plat = Vec::with_capacity(n);

    for i in 0..n {
        let base = i * 4;
        let s11 = r.s_flat[base + 0];
        let s12 = r.s_flat[base + 1];
        let s21 = r.s_flat[base + 2];
        let s22 = r.s_flat[base + 3];

        // S → Y, analytic (matches helpers.rf_math.s_to_y exactly).
        let d   = (one + s11) * (one + s22) - s12 * s21;
        let dz0 = d * z0c;
        let y11 = ((one - s11) * (one + s22) + s12 * s21) / dz0;
        let y12 = Complex64::new(-2.0, 0.0) * s12 / dz0;
        let y21 = Complex64::new(-2.0, 0.0) * s21 / dz0;
        let y22 = ((one + s11) * (one - s22) + s12 * s21) / dz0;

        // |h21|² → dB
        let h21    = -y21 / (y11 + eps);
        let h21_n2 = h21.norm_sqr();
        let h21_db = 10.0 * (h21_n2 + 1e-30).log10();

        // Mason U (NaN when denominator non-positive, matching Python where()).
        let num_u  = (y21 - y12).norm_sqr();
        let den_u  = 4.0 * (y11.re * y22.re - y12.re * y21.re);
        let u_val  = if den_u > 0.0 { num_u / den_u } else { f64::NAN };
        let u_db   = 10.0 * (u_val.abs() + 1e-30).log10();

        // K factor
        let num_k = 2.0 * y11.re * y22.re - (y12 * y21).re;
        let k_val = num_k / ((y12 * y21).norm() + 1e-60);

        // MAG / MSG
        let msg = y21.norm() / (y12.norm() + 1e-30);
        let mag_msg = if k_val > 1.0 {
            msg * (k_val - ((k_val * k_val - 1.0).max(0.0)).sqrt())
        } else {
            msg
        };
        let mag_db = 10.0 * (mag_msg.abs() + 1e-30).log10();

        // Plateau values (f in GHz so the curves match the DataFrame outputs).
        let f_ghz = r.freq[i] * 1e-9;
        let ft_plat       = f_ghz * h21_n2.sqrt();
        let fmax_u_plat   = f_ghz * u_val.abs().sqrt();
        let fmax_mag_plat = f_ghz * mag_msg.abs().sqrt();

        r.h21_db.push(h21_db);
        r.u_db.push(u_db);
        r.mag_db.push(mag_db);
        r.k.push(k_val);
        r.ft_plat.push(ft_plat);
        r.fmax_u_plat.push(fmax_u_plat);
        r.fmax_mag_plat.push(fmax_mag_plat);
    }
}

#[pyfunction]
#[pyo3(signature = (files, n_pts=2, f_min=0.01, f_max=50.0))]
fn parse_and_compute_batch<'py>(
    py: Python<'py>,
    files: Vec<Vec<u8>>,
    n_pts: usize,
    f_min: f64,
    f_max: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // Return shape: a single PyDict with stacked arrays + per-file offsets.
    //
    // The earlier version returned ``Vec<PyDict>`` — one dict-of-arrays per
    // file.  At 30 files × 9 PyArray allocations per file = 270 PyArray
    // creations, the PyO3 boundary work dominated the kernel time and the
    // Rayon speedup didn't materialise (1.2× vs 8× expected).
    //
    // SoA layout:
    //   n_per_file    int64  (N_files,)      — rows per file, 0 on parse failure
    //   z0_per_file   float64 (N_files,)
    //   errors        list[Optional[str]]    — one per file, None on success
    //   freq          float64 (total_N,)     — concatenated across all files
    //   S             complex128 (total_N, 2, 2)
    //   h21_db, u_db, mag_db, k,
    //   ft_plat, fmax_u_plat, fmax_mag_plat   — float64 (total_N,)
    //
    // Python wrapper slices these into per-file views (zero-copy) and
    // hands the same list-of-dicts shape downstream consumers expect.
    //
    // PyArray allocations: 30 × 9 = 270  →  9 total.

    let n_files = files.len();

    // ── Step 1: parallel parse + compute + extract, GIL released ──────
    // extract_limit ×3 (h21 → fT, U → fmax_U, MAG/MSG → fmax_MAG) runs
    // inside the same closure as parse + s_to_y + compute_metrics so
    // every per-file CPU op lives on the same thread — better cache
    // locality and one shared parallelism budget.  Was previously called
    // per-file in Python downstream, which added ~1.6 ms/file of work
    // and blocked the GIL.
    let results: Vec<DUTRustResult> = py.allow_threads(|| {
        files.par_iter().map(|bytes| {
            match _parse_s2p_rust(bytes) {
                Err(e) => DUTRustResult::empty_err(e),
                Ok((freq, s_flat, z0)) => {
                    let mut r = DUTRustResult {
                        freq, s_flat, z0,
                        h21_db: Vec::new(), u_db: Vec::new(),
                        mag_db: Vec::new(), k: Vec::new(),
                        ft_plat: Vec::new(), fmax_u_plat: Vec::new(),
                        fmax_mag_plat: Vec::new(),
                        ext_ft: ExtractResult::default(),
                        ext_fmax_u: ExtractResult::default(),
                        ext_fmax_mag: ExtractResult::default(),
                        error: None,
                    };
                    _compute_metrics_rust(&mut r);
                    // extract_limit consumes the freq array in GHz; the
                    // r.freq we hold is in Hz (matches Python's parse_s2p
                    // return), so convert to a temp Vec once.
                    let freq_ghz: Vec<f64> = r.freq.iter().map(|f| f * 1e-9).collect();
                    r.ext_ft = _extract_limit_rust(
                        &freq_ghz, &r.h21_db, &r.ft_plat, n_pts, f_min, f_max);
                    r.ext_fmax_u = _extract_limit_rust(
                        &freq_ghz, &r.u_db,   &r.fmax_u_plat, n_pts, f_min, f_max);
                    r.ext_fmax_mag = _extract_limit_rust(
                        &freq_ghz, &r.mag_db, &r.fmax_mag_plat, n_pts, f_min, f_max);
                    r
                }
            }
        }).collect()
    });

    // ── Step 2: compute per-file row counts and total length ──────────
    let mut n_per_file:   Vec<i64> = Vec::with_capacity(n_files);
    let mut z0_per_file:  Vec<f64> = Vec::with_capacity(n_files);
    // Per-file extract_limit results (one entry per file, in input order).
    let mut ft_cr_per_file:    Vec<f64> = Vec::with_capacity(n_files);
    let mut ft_pl_per_file:    Vec<f64> = Vec::with_capacity(n_files);
    let mut ft_m_per_file:     Vec<u8>  = Vec::with_capacity(n_files);
    let mut fmu_cr_per_file:   Vec<f64> = Vec::with_capacity(n_files);
    let mut fmu_pl_per_file:   Vec<f64> = Vec::with_capacity(n_files);
    let mut fmu_m_per_file:    Vec<u8>  = Vec::with_capacity(n_files);
    let mut fmag_cr_per_file:  Vec<f64> = Vec::with_capacity(n_files);
    let mut fmag_pl_per_file:  Vec<f64> = Vec::with_capacity(n_files);
    let mut fmag_m_per_file:   Vec<u8>  = Vec::with_capacity(n_files);
    for r in &results {
        n_per_file.push(r.freq.len() as i64);
        z0_per_file.push(r.z0);
        ft_cr_per_file  .push(r.ext_ft.cross);
        ft_pl_per_file  .push(r.ext_ft.plat);
        ft_m_per_file   .push(r.ext_ft.method);
        fmu_cr_per_file .push(r.ext_fmax_u.cross);
        fmu_pl_per_file .push(r.ext_fmax_u.plat);
        fmu_m_per_file  .push(r.ext_fmax_u.method);
        fmag_cr_per_file.push(r.ext_fmax_mag.cross);
        fmag_pl_per_file.push(r.ext_fmax_mag.plat);
        fmag_m_per_file .push(r.ext_fmax_mag.method);
    }
    let total_n: usize = n_per_file.iter().map(|&n| n as usize).sum();

    // ── Step 3: concatenate per-file Vecs into single stacked Vecs ────
    // Drain each result so we can move out of `r.s_flat` etc.  This is
    // the only serial-after-parallel work; it's pure memcpy and bounded
    // by total_N bytes (≈ N_files × 1000 × 64B = 1.9 MB for 30 files).
    let mut freq_all  : Vec<f64>       = Vec::with_capacity(total_n);
    let mut s_all     : Vec<Complex64> = Vec::with_capacity(total_n * 4);
    let mut h21_all   : Vec<f64>       = Vec::with_capacity(total_n);
    let mut u_all     : Vec<f64>       = Vec::with_capacity(total_n);
    let mut mag_all   : Vec<f64>       = Vec::with_capacity(total_n);
    let mut k_all     : Vec<f64>       = Vec::with_capacity(total_n);
    let mut ft_all    : Vec<f64>       = Vec::with_capacity(total_n);
    let mut fmu_all   : Vec<f64>       = Vec::with_capacity(total_n);
    let mut fmag_all  : Vec<f64>       = Vec::with_capacity(total_n);
    let mut errors    : Vec<Option<String>> = Vec::with_capacity(n_files);

    for mut r in results {
        errors.push(r.error.take());
        freq_all .extend(r.freq.drain(..));
        s_all    .extend(r.s_flat.drain(..));
        h21_all  .extend(r.h21_db.drain(..));
        u_all    .extend(r.u_db.drain(..));
        mag_all  .extend(r.mag_db.drain(..));
        k_all    .extend(r.k.drain(..));
        ft_all   .extend(r.ft_plat.drain(..));
        fmu_all  .extend(r.fmax_u_plat.drain(..));
        fmag_all .extend(r.fmax_mag_plat.drain(..));
    }

    // ── Step 4: build the single output PyDict ────────────────────────
    let s_arr = Array3::<Complex64>::from_shape_vec((total_n, 2, 2), s_all)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new_bound(py);
    d.set_item("n_per_file",    Array1::from_vec(n_per_file).into_pyarray_bound(py))?;
    d.set_item("z0_per_file",   Array1::from_vec(z0_per_file).into_pyarray_bound(py))?;
    d.set_item("freq",          Array1::from_vec(freq_all).into_pyarray_bound(py))?;
    d.set_item("S",             s_arr.into_pyarray_bound(py))?;
    d.set_item("h21_db",        Array1::from_vec(h21_all).into_pyarray_bound(py))?;
    d.set_item("u_db",          Array1::from_vec(u_all).into_pyarray_bound(py))?;
    d.set_item("mag_db",        Array1::from_vec(mag_all).into_pyarray_bound(py))?;
    d.set_item("k",             Array1::from_vec(k_all).into_pyarray_bound(py))?;
    d.set_item("ft_plat",       Array1::from_vec(ft_all).into_pyarray_bound(py))?;
    d.set_item("fmax_u_plat",   Array1::from_vec(fmu_all).into_pyarray_bound(py))?;
    d.set_item("fmax_mag_plat", Array1::from_vec(fmag_all).into_pyarray_bound(py))?;
    // Per-file extract_limit results (length = N_files).
    d.set_item("ft_cross",      Array1::from_vec(ft_cr_per_file).into_pyarray_bound(py))?;
    d.set_item("ft_plateau",    Array1::from_vec(ft_pl_per_file).into_pyarray_bound(py))?;
    d.set_item("ft_method",     Array1::from_vec(ft_m_per_file).into_pyarray_bound(py))?;
    d.set_item("fmax_u_cross",   Array1::from_vec(fmu_cr_per_file).into_pyarray_bound(py))?;
    d.set_item("fmax_u_plateau", Array1::from_vec(fmu_pl_per_file).into_pyarray_bound(py))?;
    d.set_item("fmax_u_method",  Array1::from_vec(fmu_m_per_file).into_pyarray_bound(py))?;
    d.set_item("fmax_mag_cross",   Array1::from_vec(fmag_cr_per_file).into_pyarray_bound(py))?;
    d.set_item("fmax_mag_plateau", Array1::from_vec(fmag_pl_per_file).into_pyarray_bound(py))?;
    d.set_item("fmax_mag_method",  Array1::from_vec(fmag_m_per_file).into_pyarray_bound(py))?;

    let errs_list = PyList::empty_bound(py);
    for e in errors {
        match e {
            Some(s) => errs_list.append(s)?,
            None    => errs_list.append(py.None())?,
        }
    }
    d.set_item("errors", errs_list)?;

    Ok(d)
}

// ── Phase 3 — generic data-driven custom-model batched simulator ─────────────
//
// Unlike the hard-coded built-in topology kernels above, the custom model's
// topology is user-built, so it can't be inlined.  Instead the Python side
// compiles the topology once into a flat, value-free `SimPlan` (see
// `tools/SSM/custom_model/core.py::compile_plan`) and hands it to this kernel
// as a dict of plain ints.  We evaluate it for every (batch, freq) pair:
// stamp a small (n×n) complex nodal matrix, Kron-reduce the internal nodes via
// a dense complex LU solve, and convert the surviving 2-port Y → S — exactly
// mirroring `simulate_custom_model_batch`.  Parallel across the batch (B) axis.
//
// Plan dict schema (all ints/lists, no strings beyond `value_keys`):
//   n            : usize                       node count, order [P1, P2, …]
//   value_keys   : list[str]                   master key order; everything
//                                              below indexes into it
//   branches     : list[(ia, ib, series, groups)]
//                    ia/ib : i64 node index (-1 == GND/reference)
//                    series: 0|1
//                    groups: [[(kind, key_idx), …], …]   kind 0=R 1=L 2=C
//   twoport      : (ia, ib, iref, be_groups, bc_groups, ce_groups)
//   itype_pi     : 0|1                         1 == hybrid-π, 0 == T α-source
//   source_idx   : list[usize]                 [gm,tau] (π) or [α0,τB,τC] (T)

const SHORT_Y: f64 = 1e12;   // wire limit for an all-zero series branch

type CGroups = Vec<Vec<(u8, usize)>>;

fn _plan_get<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: pyo3::FromPyObject<'py>,
{
    d.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("sim_custom_batch: plan missing '{key}'")))?
        .extract()
}

#[inline(always)]
fn _elem_adm(kind: u8, val: f64, jw: C) -> C {
    match kind {
        0 => if val != 0.0 { C::new(1.0 / val, 0.0) } else { C::new(0.0, 0.0) }, // R
        1 => if val != 0.0 {
            let z = jw * val;
            // jw = 0 (DC point): a present inductor is a short, not inf/NaN —
            // matches the Python evaluator's _elem_adm_b.
            if z.norm_sqr() > 0.0 { C::new(1.0, 0.0) / z } else { C::new(SHORT_Y, 0.0) }
        } else { C::new(0.0, 0.0) }, // L
        2 => jw * val,                                                            // C
        _ => C::new(0.0, 0.0),
    }
}

#[inline(always)]
fn _group_adm(group: &[(u8, usize)], v: &[f64], jw: C) -> C {
    let mut y = C::new(0.0, 0.0);
    for &(kind, ki) in group {
        y += _elem_adm(kind, v[ki], jw);
    }
    y
}

/// Series branch: impedance sum of groups (zero group → 0 impedance / short);
/// an all-zero branch collapses to a near-short `SHORT_Y` (the fixed-node-count
/// analogue of the Python union-find merge).
#[inline(always)]
fn _branch_series(groups: &[Vec<(u8, usize)>], v: &[f64], jw: C) -> C {
    let mut z = C::new(0.0, 0.0);
    for g in groups {
        let yg = _group_adm(g, v, jw);
        if yg.norm_sqr() > 0.0 {
            z += C::new(1.0, 0.0) / yg;
        }
    }
    if z.norm_sqr() > 0.0 { C::new(1.0, 0.0) / z } else { C::new(SHORT_Y, 0.0) }
}

/// Shunt / junction branch: any absent group breaks the path → open (0);
/// otherwise 1 / Σ(1/yg).  Empty group list → 0 (matches `_junction_adm_b`).
#[inline(always)]
fn _branch_shunt(groups: &[Vec<(u8, usize)>], v: &[f64], jw: C) -> C {
    let mut inv = C::new(0.0, 0.0);
    let mut open = false;
    for g in groups {
        let yg = _group_adm(g, v, jw);
        if yg.norm_sqr() == 0.0 {
            open = true;
        } else {
            inv += C::new(1.0, 0.0) / yg;
        }
    }
    if open || inv.norm_sqr() == 0.0 { C::new(0.0, 0.0) } else { C::new(1.0, 0.0) / inv }
}

#[inline(always)]
fn _stamp(y: &mut [C], n: usize, ia: i64, ib: i64, yv: C) {
    if ia >= 0 {
        let i = ia as usize;
        y[i * n + i] += yv;
    }
    if ib >= 0 {
        let j = ib as usize;
        y[j * n + j] += yv;
    }
    if ia >= 0 && ib >= 0 {
        let i = ia as usize;
        let j = ib as usize;
        y[i * n + j] -= yv;
        y[j * n + i] -= yv;
    }
}

#[inline(always)]
fn _cell(y: &mut [C], n: usize, ir: i64, ic: i64, val: C) {
    if ir >= 0 && ic >= 0 {
        y[(ir as usize) * n + (ic as usize)] += val;
    }
}

#[inline(always)]
fn _intrinsic_y(pi: bool, ybe: C, ybc: C, yce: C, v: &[f64], src: &[usize], jw: C)
    -> (C, C, C, C)
{
    let one = C::new(1.0, 0.0);
    if pi {
        let gm = C::new(v[src[0]], 0.0) * (-jw * v[src[1]]).exp();
        return (ybe + ybc, -ybc, gm - ybc, ybc + yce);
    }
    let zbe = if ybe.norm_sqr() > 0.0 { one / ybe } else { C::new(0.0, 0.0) };
    let zbc = if ybc.norm_sqr() > 0.0 { one / ybc } else { C::new(0.0, 0.0) };
    let alpha = C::new(v[src[0]], 0.0) * (-jw * v[src[2]]).exp() / (one + jw * v[src[1]]);
    let z11 = zbe;
    let z12 = zbe;
    let z21 = zbe - alpha * zbc;
    let z22 = (one - alpha) * zbc + zbe;
    let mut det = z11 * z22 - z12 * z21;
    if det.norm_sqr() == 0.0 {
        det = C::new(1e-30, 0.0);
    }
    (z22 / det, -z12 / det, -z21 / det, z11 / det + yce)
}

/// Solve `a · X = rhs` in place (a is m×m, rhs is m×nrhs, both row-major) via
/// Gaussian elimination with partial pivoting; on return `rhs` holds X.
fn _solve_inplace(a: &mut [C], rhs: &mut [C], m: usize, nrhs: usize) {
    for col in 0..m {
        let mut piv = col;
        let mut best = a[col * m + col].norm_sqr();
        for r in (col + 1)..m {
            let val = a[r * m + col].norm_sqr();
            if val > best {
                best = val;
                piv = r;
            }
        }
        if piv != col {
            for j in 0..m {
                a.swap(col * m + j, piv * m + j);
            }
            for j in 0..nrhs {
                rhs.swap(col * nrhs + j, piv * nrhs + j);
            }
        }
        let mut diag = a[col * m + col];
        if diag.norm_sqr() == 0.0 {
            diag = C::new(1e-30, 0.0);
        }
        for r in (col + 1)..m {
            let factor = a[r * m + col] / diag;
            if factor.norm_sqr() == 0.0 {
                continue;
            }
            for j in col..m {
                let t = a[col * m + j];
                a[r * m + j] -= factor * t;
            }
            for j in 0..nrhs {
                let t = rhs[col * nrhs + j];
                rhs[r * nrhs + j] -= factor * t;
            }
        }
    }
    for col in (0..m).rev() {
        let mut diag = a[col * m + col];
        if diag.norm_sqr() == 0.0 {
            diag = C::new(1e-30, 0.0);
        }
        for j in 0..nrhs {
            let mut s = rhs[col * nrhs + j];
            for k in (col + 1)..m {
                s -= a[col * m + k] * rhs[k * nrhs + j];
            }
            rhs[col * nrhs + j] = s / diag;
        }
    }
}

/// Kron-reduce the (n×n) nodal Y to a 2×2 port matrix (keeps indices 0,1).
fn _kron_reduce(y: &[C], n: usize) -> [C; 4] {
    if n == 2 {
        return [y[0], y[1], y[2], y[3]];
    }
    let m = n - 2;
    let mut a = vec![C::new(0.0, 0.0); m * m];
    let mut rhs = vec![C::new(0.0, 0.0); m * 2];
    for i in 0..m {
        for j in 0..m {
            a[i * m + j] = y[(i + 2) * n + (j + 2)];
        }
        a[i * m + i] += C::new(1e-15, 0.0);          // regularise floating nodes
        rhs[i * 2] = y[(i + 2) * n];                 // Yia col 0
        rhs[i * 2 + 1] = y[(i + 2) * n + 1];         // Yia col 1
    }
    _solve_inplace(&mut a, &mut rhs, m, 2);          // rhs ← Yii⁻¹ · Yia
    let mut out = [y[0], y[1], y[n], y[n + 1]];      // Yaa
    for r in 0..2 {
        for c in 0..2 {
            let mut s = C::new(0.0, 0.0);
            for k in 0..m {
                s += y[r * n + (k + 2)] * rhs[k * 2 + c];   // Yai · X
            }
            out[r * 2 + c] -= s;
        }
    }
    out
}

#[pyfunction]
fn sim_custom_batch<'py>(
    py: Python<'py>,
    plan: &Bound<'py, PyDict>,
    params: &Bound<'py, PyDict>,
    freq: PyReadonlyArray1<'py, f64>,
    z0: f64,
) -> PyResult<Bound<'py, PyArray4<C>>> {
    let n: usize = _plan_get(plan, "n")?;
    let value_keys: Vec<String> = _plan_get(plan, "value_keys")?;
    let branches: Vec<(i64, i64, i64, CGroups)> = _plan_get(plan, "branches")?;
    let (tp_a, tp_b, tp_ref, be_g, bc_g, ce_g):
        (i64, i64, i64, CGroups, CGroups, CGroups) = _plan_get(plan, "twoport")?;
    let itype_pi: i64 = _plan_get(plan, "itype_pi")?;
    let source_idx: Vec<usize> = _plan_get(plan, "source_idx")?;
    let pi_model = itype_pi != 0;

    // Pre-extract every referenced value as a BcVal (scalar / per-batch).
    let vals: Vec<BcVal> = value_keys.iter()
        .map(|k| _extract_bcval(params, k, 0.0))
        .collect::<PyResult<_>>()?;
    let b: usize = vals.iter().filter_map(|v| v.batch_len()).max().unwrap_or(1);
    for v in &vals {
        if let Some(l) = v.batch_len() {
            if l != b {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sim_custom_batch: inconsistent per-batch param lengths."));
            }
        }
    }

    let freq_view = freq.as_array();
    let nf = freq_view.len();
    let freq_slice = freq_view.as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "sim_custom_batch: freq array must be C-contiguous."))?
        .to_vec();

    let mut out = Array4::<C>::zeros((b, nf, 2, 2));
    let slab_len = nf * 4;
    let flat = out.as_slice_mut().expect("Array4 is always C-contig");
    let z0c = C::new(z0, 0.0);

    py.allow_threads(|| {
        flat.par_chunks_mut(slab_len)
            .enumerate()
            .for_each(|(bi, chunk)| {
                // This batch's scalar values, indexed by value-key index.
                let v: Vec<f64> = vals.iter().map(|bc| bc.at(bi)).collect();
                let mut ymat = vec![C::new(0.0, 0.0); n * n];
                for ni in 0..nf {
                    let omega = 2.0 * std::f64::consts::PI * freq_slice[ni];
                    let jw = C::new(0.0, omega);
                    for cc in ymat.iter_mut() {
                        *cc = C::new(0.0, 0.0);
                    }

                    // ── Passive branches ──
                    for (ia, ib, series, groups) in &branches {
                        let y = if *series != 0 {
                            _branch_series(groups, &v, jw)
                        } else {
                            _branch_shunt(groups, &v, jw)
                        };
                        _stamp(&mut ymat, n, *ia, *ib, y);
                    }

                    // ── Intrinsic controlled-source 2-port ──
                    let ybe = _branch_shunt(&be_g, &v, jw);
                    let ybc = _branch_shunt(&bc_g, &v, jw);
                    let yce = _branch_shunt(&ce_g, &v, jw);
                    let (y11, y12, y21, y22) =
                        _intrinsic_y(pi_model, ybe, ybc, yce, &v, &source_idx, jw);
                    _cell(&mut ymat, n, tp_a, tp_a, y11);
                    _cell(&mut ymat, n, tp_a, tp_b, y12);
                    _cell(&mut ymat, n, tp_b, tp_a, y21);
                    _cell(&mut ymat, n, tp_b, tp_b, y22);
                    _cell(&mut ymat, n, tp_a, tp_ref, -(y11 + y12));
                    _cell(&mut ymat, n, tp_b, tp_ref, -(y21 + y22));
                    _cell(&mut ymat, n, tp_ref, tp_a, -(y11 + y21));
                    _cell(&mut ymat, n, tp_ref, tp_b, -(y12 + y22));
                    _cell(&mut ymat, n, tp_ref, tp_ref, y11 + y12 + y21 + y22);

                    // ── Kron reduce → 2×2 Y → S ──
                    let y2 = _kron_reduce(&ymat, n);
                    let s = y_to_s_one(&y2, z0c);
                    let off = ni * 4;
                    chunk[off] = s[0];
                    chunk[off + 1] = s[1];
                    chunk[off + 2] = s[2];
                    chunk[off + 3] = s[3];
                }
            });
    });

    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn hbt_rust_kernels(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1 primitives
    m.add_function(wrap_pyfunction!(inv2x2_batch,         m)?)?;
    m.add_function(wrap_pyfunction!(mm2x2_batch,          m)?)?;
    m.add_function(wrap_pyfunction!(y_to_s_batch,         m)?)?;
    m.add_function(wrap_pyfunction!(y_to_s_4d,            m)?)?;
    m.add_function(wrap_pyfunction!(port_residuals_batch, m)?)?;
    // Phase 1.5 — bulk-upload accelerator
    m.add_function(wrap_pyfunction!(parse_and_compute_batch, m)?)?;
    // Phase 2 stubs (raise NotImplementedError — Python wrapper falls back)
    m.add_function(wrap_pyfunction!(sim_cheng_t_batch,    m)?)?;
    m.add_function(wrap_pyfunction!(sim_cheng_pi_batch,   m)?)?;
    m.add_function(wrap_pyfunction!(sim_xu_t_batch,       m)?)?;
    m.add_function(wrap_pyfunction!(sim_kunyang_batch,    m)?)?;
    // Phase 3 — generic data-driven custom-model simulator
    m.add_function(wrap_pyfunction!(sim_custom_batch,     m)?)?;
    m.add("__doc__", "HBT Rust kernels — see Python wrapper at \
                      tools/SSM/helpers/rust_kernels.py")?;
    Ok(())
}

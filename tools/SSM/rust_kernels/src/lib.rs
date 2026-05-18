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
use ndarray::{Array2, Array3, Array4, Axis, Zip};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4,
            PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
// rayon::slice::ParallelSliceMut provides `par_chunks_mut` used by
// the Phase 2 end-to-end simulators below.
use rayon::slice::ParallelSliceMut;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

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
//   3. Bump the binary by re-running `python build_rust_kernels.py`.
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

#[pymodule]
fn hbt_rust_kernels(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1 primitives
    m.add_function(wrap_pyfunction!(inv2x2_batch,         m)?)?;
    m.add_function(wrap_pyfunction!(mm2x2_batch,          m)?)?;
    m.add_function(wrap_pyfunction!(y_to_s_batch,         m)?)?;
    m.add_function(wrap_pyfunction!(y_to_s_4d,            m)?)?;
    m.add_function(wrap_pyfunction!(port_residuals_batch, m)?)?;
    // Phase 2 stubs (raise NotImplementedError — Python wrapper falls back)
    m.add_function(wrap_pyfunction!(sim_cheng_t_batch,    m)?)?;
    m.add_function(wrap_pyfunction!(sim_cheng_pi_batch,   m)?)?;
    m.add_function(wrap_pyfunction!(sim_xu_t_batch,       m)?)?;
    m.add("__doc__", "HBT Rust kernels — see Python wrapper at \
                      tools/SSM/helpers/rust_kernels.py")?;
    Ok(())
}

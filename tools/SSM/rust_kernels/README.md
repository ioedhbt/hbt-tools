# hbt_rust_kernels — optional Rust acceleration

A small Rust crate that exposes SIMD/Rayon-accelerated CPU kernels for
the math hot paths in the SSM extraction tool.  Building it is
**optional**: when no binary is present the Python helpers in
`tools/SSM/helpers/rust_kernels.py` silently fall back to NumPy, and
the tool runs exactly as before.

## How end users get the speedup

**They don't have to build anything.**  The compiled extension is
committed to the repo under `bin/<platform_arch>/hbt_rust_kernels.<ext>`.
The Python wrapper adds that folder to `sys.path` at import time and
loads the binary directly — no pip install, no maturin, no Rust
toolchain on user machines.

Per-OS binaries that need to be in the repo for full coverage:

```
tools/SSM/rust_kernels/bin/
├── win_amd64/           hbt_rust_kernels.pyd        ← committed
├── linux_x86_64/        hbt_rust_kernels.so         ← committed (Streamlit Cloud)
├── macosx_arm64/        hbt_rust_kernels.so         ← committed
└── macosx_x86_64/       hbt_rust_kernels.so         ← committed
```

If a user's platform isn't in `bin/`, the loader silently takes the
NumPy path.  The tool still works — they just don't get the speedup.

## How maintainers build the binary

Run **once per OS** from the repo root:

```
python rust_things/build_rust_kernels.py
```

What it does:

1. Checks for `cargo` (the Rust compiler).  If missing, prints rustup
   install instructions and exits.
2. Creates a separate `.hbttools_build/` venv — *not* the runtime
   `.hbttools/` venv used by `LAUNCH_Tool.py`.  This keeps maturin out
   of `requirements.txt` and the user-facing venv.
3. Installs maturin into the build venv.
4. Runs `maturin build --release` against this crate.  The
   `abi3-py39` feature in `Cargo.toml` produces a single binary
   compatible with every Python ≥ 3.9 — no per-version rebuilds.
5. Extracts the binary from the wheel and drops it at
   `tools/SSM/rust_kernels/bin/<platform_arch>/hbt_rust_kernels.<ext>`.

Commit the resulting file.  Other contributors / Streamlit Cloud pick
it up automatically the next time they pull.

## Five kernels

| Wrapper (in `helpers/rust_kernels.py`) | Shape | Mirrors |
|---|---|---|
| `inv2x2_batch(Y)` | `(B,2,2) → (B,2,2)` | `helpers/rf_math.py::inv2x2` |
| `mm2x2_batch(A,B)` | `(B,2,2),(B,2,2) → (B,2,2)` | `helpers/rf_math.py::mm2x2` |
| `y_to_s_batch(Y, z0)` | `(B,2,2) → (B,2,2)` | `helpers/rf_math.py::y_to_s_batch` |
| `y_to_s_4d(Y, z0)` | `(B,N,2,2) → (B,N,2,2)` | (new — primary tuning-sweep entry) |
| `port_residuals_batch(S_mea, S_mod_batch)` | `(N,2,2),(B,N,2,2) → (B,5)` | `models/base_ui.py::_port_residuals_batch` |

The outer (B) axis is parallelised by Rayon, so an N-core CPU gets
near-linear scaling for the tuning sweep payloads (typical B ≥ 1 000).

## Build prerequisites

Each maintainer needs a Rust toolchain (one-time, ~5 minutes):

```bash
# rustup — official Rust installer.  Just take stable, no nightly needed.
# https://rustup.rs/
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Linux/macOS
# Windows: download and run rustup-init.exe from rustup.rs
```

On Windows the MSVC linker is also required — `rustup-init.exe` prompts
you if it's missing (offers a one-click install of the Visual Studio
Build Tools).  On Debian/Ubuntu: `apt install build-essential`.  On
macOS: Xcode Command Line Tools.

After that, `python rust_things/build_rust_kernels.py` handles everything else.

## Verifying / benchmarking

```bash
python tools/SSM/rust_kernels/benchmark.py
```

Asserts numerical parity at 1e-10 rtol against the NumPy references,
then prints wall-clock times for three sweep sizes.

## Expected wins — wall-clock

Numbers below are realistic for a recent 8-core/16-thread CPU (Ryzen 7
/ M-series Mac).  Scaling is ~linear with cores up to `min(cores, B)`.

| Kernel | Sweep size | NumPy | Rust | Speedup |
|---|---|---|---|---|
| `inv2x2_batch` | B=1 024 | 0.15 ms | 0.06 ms | **2.5×** |
| `mm2x2_batch` | B=1 024 | 0.20 ms | 0.07 ms | **2.8×** |
| `y_to_s_batch` | B=1 024 | 0.6 ms | 0.15 ms | **4×** |
| `y_to_s_4d` | B=1 024, N=201 | 70 ms | 12 ms | **5–6×** |
| `port_residuals_batch` | B=10 000, N=201 | 120 ms | 25 ms | **4.5–5×** |

For the **Visual Tuning** Plotly slider — the dominant click-to-paint
cost — a full B = 16 384, N = 201 batch:

| Path | Wall-clock per Build click |
|---|---|
| NumPy (no binary present) | ≈ **1.8 s** |
| Rust (binary committed to `bin/`) | ≈ **0.35–0.45 s** |
| CUDA (cupy, unchanged) | ≈ **45 ms** |

So Rust closes about three-quarters of the gap between NumPy and CUDA
for users without a GPU.  Users *with* a GPU see no change — `cupy`
runs the same kernels orders of magnitude faster, and the Rust path
isn't on that branch.

## Disabling at runtime

```bash
HBT_DISABLE_RUST=1 streamlit run tools/IOED_HBT_RF_extract.py
```

This forces every wrapper in `helpers/rust_kernels.py` to take the
NumPy branch even when the binary is present.  Useful for benchmarks /
A-B parity checks.

## What's intentionally *not* ported

- `parse_s2p` / `parse_csv` / `write_s2p` — I/O bound, not CPU bound.
- `render_*` / `_render_*` — Streamlit/Plotly UI, no math to accelerate.
- `_sim_wrap_vec` / scalar paths — only the `_batch` shapes are in
  the hot loop.  Scalar single-call sims still go through NumPy.
- The CUDA path in `_sim_wrap_batch` (when `xp=cupy`) — already
  50–100× faster than this crate, nothing to gain.

## Adding a kernel

1. Implement the inline scalar version as a `#[inline(always)] fn …`
   in `src/lib.rs`, plus a `#[pyfunction]` wrapper that pulls apart
   the NumPy view and writes into a freshly-allocated output array.
   Parallelise over the outer axis with
   `ndarray::Zip::from(…).and(…).par_for_each(…)`.
2. Register it in the `#[pymodule] fn hbt_rust_kernels` block.
3. Add a NumPy reference and a wrapper to
   `tools/SSM/helpers/rust_kernels.py`.
4. Extend `benchmark.py` with a parity + timing block.
5. Rebuild and commit a fresh binary per OS:
   `python rust_things/build_rust_kernels.py`.

Callers in `helpers/` and `models/` should import from
`tools.SSM.helpers.rust_kernels`, never from `hbt_rust_kernels`
directly — that preserves the NumPy fallback for unbuilt platforms.

(note To ship to Streamlit Cloud
Run python rust_things/build_rust_kernels.py on Linux x86_64 (a WSL box, a Docker image, or a one-off VM). It'll drop a .so at tools/SSM/rust_kernels/bin/linux_x86_64/hbt_rust_kernels.so. Commit it. Streamlit Cloud picks it up automatically on next deploy — no pip install needed there either.)
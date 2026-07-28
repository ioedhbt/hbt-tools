# `bin/` — prebuilt extension binaries

Each subdirectory holds one compiled Rust extension binary for a specific
host platform.  The Python wrapper at
[`tools/rf/ssm/helpers/rust_kernels.py`](../../helpers/rust_kernels.py)
adds the matching subdir to `sys.path` at import time and loads
`hbt_rust_kernels` directly — **no pip install, no Rust toolchain, no
maturin** needed on end-user machines.

## Directory layout

```
bin/
├── win_amd64/         hbt_rust_kernels.<ext>     # Windows 64-bit
├── linux_x86_64/      hbt_rust_kernels.<ext>     # Linux x86-64 (incl. Streamlit Cloud)
├── macosx_arm64/      hbt_rust_kernels.<ext>     # Apple Silicon
└── macosx_x86_64/     hbt_rust_kernels.<ext>     # Intel Mac
```

The exact filename inside each folder ends in:

- `.pyd` on Windows
- `.so`  on Linux
- `.so`  or `.dylib` on macOS

ABI3-py39 means the same binary works on every CPython ≥ 3.9 — you don't
need to rebuild when upgrading the runtime Python.

## How to populate

Run **once per OS** from the repo root:

```
python dev/build_rust_kernels.py
```

The script creates a separate `.hbttools_build/` venv (so the runtime
`.hbttools/` venv used by `LAUNCH_Tool.py` stays clean), pulls in
maturin, builds the crate with `--release`, and drops the resulting
binary into the correct `bin/<platform_arch>/` directory.

Commit the new file to the repo so other contributors / Streamlit
Cloud pick it up automatically.

## Why the binaries aren't in this commit

Cross-compiling reliably is hard; the build script is the source of
truth.  Each maintainer / CI runner builds for their own platform once
and commits.  Until then, the Python NumPy fallback is in effect — the
tool runs identically, just slower for very large tuning sweeps.

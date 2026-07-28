"""
paths.py — one definition of every repo-relative location.

Layer:       pure (stdlib only)
Imported by: helpers/fit_cache, custom_model/core, agent_api, page modules,
             dev/build_rust_kernels.py, dev/check_rust_status.py
Gotchas:     never re-derive any of these with `parents[N]` or by joining
             path segments by hand — see below for why.

Why this exists
---------------
The repo root used to be recomputed in six places as
``Path(__file__).parents[N]`` with N baked to each file's depth
(``parents[3]`` in ``helpers/fit_cache.py`` and ``custom_model/core.py``,
``parents[2]`` in ``agent_api.py``, …). Every one breaks the moment a file
moves a level, and the breakage is silent: the fit cache simply starts
writing somewhere else.

The Rust crate location had the same problem in a nastier form — four places
built ``ROOT / "tools" / "SSM" / "rust_kernels"`` independently. After the
tree moved, three of them still pointed at the old path, so
``build_rust_kernels.py`` aborted with "crate directory missing" and the
launcher announced "Rust acceleration: not available" while a perfectly good
binary sat on disk. String-literal search-and-replace does not catch a path
assembled from segments; a single definition does.

Import from here instead.
"""
from __future__ import annotations

from pathlib import Path

# tools/common/paths.py -> tools/common -> tools -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Bundled example S-parameter files ("📂 Load example files" on the SSM
# extraction page).
EXAMPLES_DIR = REPO_ROOT / "examples"

# The Rust acceleration crate and the committed per-platform binaries under
# it.  `helpers/rust_kernels.py` derives its own copy relative to its own
# file (correct, and independent of this module); everything else should
# import these.
#
# The one legitimate exception is LAUNCH_Tool.py: it runs before the venv
# exists and so cannot import this module. It keeps a literal copy, and
# dev/smoke_test.py asserts the two agree.
RUST_CRATE_DIR = REPO_ROOT / "tools" / "rf" / "ssm" / "rust_kernels"
RUST_BIN_BASE = RUST_CRATE_DIR / "bin"

__all__ = ["REPO_ROOT", "EXAMPLES_DIR", "RUST_CRATE_DIR", "RUST_BIN_BASE"]

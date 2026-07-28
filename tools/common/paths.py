"""
paths.py — one definition of where the repository root is.

Layer:       pure (stdlib only)
Imported by: helpers/fit_cache, custom_model/core, agent_api, page modules
Gotchas:     none — but never re-derive the root with `parents[N]` elsewhere.

Why this exists
---------------
The root used to be recomputed in six places as ``Path(__file__).parents[N]``
with N baked to each file's depth (``parents[3]`` in ``helpers/fit_cache.py``
and ``custom_model/core.py``, ``parents[2]`` in ``agent_api.py``, and so on).
Every one of those breaks the moment a file moves one level up or down, and
the breakage is silent: the fit cache simply starts writing somewhere else.

Import ``REPO_ROOT`` instead. It is correct regardless of how deeply the
importing module is nested, because it is anchored to *this* file.
"""
from __future__ import annotations

from pathlib import Path

# tools/common/paths.py -> tools/common -> tools -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

def _examples_dir():
    """Bundled example S-parameter files ("Load example files" on the SSM
    extraction page).

    Accepts the legacy ``dummy_data_practice/`` name so an install that has
    not picked up the rename still finds its examples.
    """
    for name in ("examples", "dummy_data_practice"):
        d = REPO_ROOT / name
        if d.is_dir():
            return d
    return REPO_ROOT / "examples"


EXAMPLES_DIR = _examples_dir()

__all__ = ["REPO_ROOT", "EXAMPLES_DIR"]

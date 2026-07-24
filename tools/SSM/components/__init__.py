"""tools.SSM.components — hand-written (no-build) Streamlit components.

``draggable_smith`` renders a Plotly Smith chart whose text annotations are
draggable in the browser and streams the dragged annotation positions back to
Python via the standard component ``setComponentValue`` protocol.  This is what
``render_matplotlib_smith`` uses for its "Drag labels" mode: the user drags the
S-parameter labels, and the returned positions drive the matplotlib figure when
they click "Render matplotlib".

The frontend is a single static ``draggable_smith/index.html`` (Plotly from CDN
+ ~40 lines of vanilla JS implementing the component message protocol), so there
is no npm/build step — ``declare_component(path=...)`` serves the folder as-is.
"""
from __future__ import annotations

import os

import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draggable_smith")
_draggable_smith = components.declare_component("draggable_smith", path=_DIR)


def draggable_smith(*, figure: dict, labels: list, height: int = 560, key=None):
    """Render a Smith chart with independent draggable HTML label overlays.

    ``figure`` — a JSON-safe Plotly figure dict (``json.loads(fig.to_json())``)
    holding the grid + traces (and any FIXED text such as the freq caption).
    ``labels`` — the draggable labels, one dict per label in a stable order:
    ``{"x", "y", "text", "color", "size"}``.  Each is drawn as its own overlay
    <div>, so dragging one never moves another.

    Returns the single most-recently dragged label as
    ``{"index": i, "x": ..., "y": ..., "nonce": ...}`` (``index`` is the
    position in ``labels``), or ``None`` before any drag.
    """
    return _draggable_smith(figure=figure, labels=labels, height=height,
                            key=key, default=None)

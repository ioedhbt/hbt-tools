"""tools.rf.ssm.components — hand-written (no-build) Streamlit components.

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


_smith_bode_slider = components.declare_component("smith_bode_slider",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "smith_bode_slider"))


def smith_bode_slider(*, payload: dict, height: int = 560,
                      use_label: str = "Use these values", key=None):
    """Bidirectional Smith+Bode scrub-slider (the ⚡ Smooth-sweep preview).

    ``payload`` is the dict returned by
    ``tools.rf.ssm.helpers.build_smith_bode_slider_payload`` — a JSON-safe
    Plotly figure plus a gzip+base64 blob of per-frame Smith/Bode data
    (``data_b64``), cartesian axis sizes (``dims``), the initial midpoint
    frame per axis (``mids``), per-axis display-value label strings
    (``labels``), and the trace-index maps (``smith``/``bode``/``extrap``)
    ``update()`` needs to swap frame data on scrub.  The browser inflates
    ``data_b64`` itself (native ``DecompressionStream``) and renders one
    range-slider per axis, exactly replicating the frame-swap logic of
    ``make_smith_bode_joint_slider_html`` but without re-embedding a fresh
    multi-MB HTML blob on every Streamlit rerun.

    Returns ``{"indices": [...], "nonce": ...}`` — the per-axis slider
    positions at the moment ``use_label`` was clicked — or ``None`` before
    any click.  ``indices`` line up positionally with the ``slider_specs``
    list the payload was built from; the caller maps index → value.
    """
    return _smith_bode_slider(payload=payload, height=height,
                              use_label=use_label, key=key, default=None)

"""
models/svg_topology.py — "non-zero only" live SVG schematic for the four
built-in analytic topologies (Cheng T, Cheng π, Xu T, Kun-Yang HEMT).

This is the toggle-ON alternative to each model's static PNG-template
illustration (``_render_topology`` on the model class): instead of overlaying
values on a fixed drawing, it starts from the matching built-in custom-model
preset (``custom_model.core.builtin_custom_model``) — which mirrors each
model's forward-sim topology exactly — and prunes away every R/L/C component
whose *current* value (from the model's live ``all_p`` dict) is zero, absent,
or non-finite, then renders the survivors with the custom-model builder's SVG
engine (``custom_model.schematic``).

The heavy lifting (layout, glyphs, value formatting) already lives in
``custom_model.schematic``; this module only maps each preset's element names
onto ``all_p`` keys and prunes the dead ones. ``build_pruned_model`` is kept
Streamlit-free so it can be exercised directly in a script/test.
"""
from __future__ import annotations

import math

from ..custom_model.core import CustomModel, Network, ShuntBranch, builtin_custom_model
from ..custom_model.schematic import (render_schematic, svg_pixel_height,
                                      svg_png_buttons, _SVG, _GND, _fmt)


# Model SHORT → built-in preset label (custom_model.core.BUILTIN_PRESETS key).
_PRESET_LABEL = {
    "T":   "Cheng — T (current-source T HBT)",
    "pi":  "Cheng — π (hybrid-π HBT)",
    "XuT": "Xu — T (Rbcx∥Cbcx HBT)",
    "KY":  "Kun-Yang — HEMT (π)",
}

# CustomModel.access_names key → all_p key.  Fixed across all four presets:
# the access *resistors* live under the shared PAD_SPECS "Rp*" keys (Rpb/Rpc/
# Rpe — relabelled Rg/Rd/Rs for Kun-Yang, but the storage key is unchanged),
# while the access inductors keep their bare "L*" keys.
_ACCESS_KEY = {"Rb": "Rpb", "Lb": "Lb", "Rc": "Rpc", "Lc": "Lc",
              "Re": "Rpe", "Le": "Le"}

# Controlled-source parameter key (CustomModel.source_keys()) → all_p key.
# Only the π core's transconductance differs ("gm" is stored as "Gm0"); every
# other source key (tau, alpha0, tauB, tauC) is identical to its all_p key.
_SOURCE_KEY = {"gm": "Gm0"}


def _present(v) -> bool:
    """True when ``v`` is a usable, non-zero component value."""
    if v is None:
        return False
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v != 0.0


def _prune_network(net: Network, all_p: dict, values: dict) -> None:
    """Drop every Element of ``net`` whose ``all_p`` value is absent, zero, or
    non-finite, and any parallel group left empty by the drop.  Surviving
    elements' values are recorded into ``values`` keyed by ``Element.id`` —
    the lookup key ``render_schematic`` expects."""
    kept_groups = []
    for group in net.groups:
        kept = []
        for e in group:
            v = all_p.get(e.name)
            if _present(v):
                values[e.id] = float(v)
                kept.append(e)
        if kept:
            kept_groups.append(kept)
    net.groups = kept_groups


def _prune_branches(branches: list[ShuntBranch], all_p: dict,
                    values: dict) -> list[ShuntBranch]:
    """Prune each branch's Network, then drop branches left with no elements."""
    for b in branches:
        _prune_network(b.network, all_p, values)
    return [b for b in branches if not b.network.is_empty]


def build_pruned_model(short: str, all_p: dict) -> tuple[CustomModel, dict]:
    """Build the built-in preset for model ``short`` and strip it down to only
    the components whose current value (in ``all_p``) is present and
    non-zero.  Pure/Streamlit-free — returns ``(model, values)`` ready for
    ``render_schematic``.

    ``CustomModel.ensure_intrinsic()`` (run once by ``__post_init__`` during
    ``builtin_custom_model``) only fills in a default controlled-source label
    when unset — it never rebuilds/reseeds the intrinsic Networks — so pruning
    the freshly built preset here is never undone downstream.
    """
    model = builtin_custom_model(_PRESET_LABEL[short])
    values: dict = {}

    for net in (model.intrinsic_base, model.intrinsic_be, model.intrinsic_bc,
                model.intrinsic_ce, model.port1, model.port2, model.emitter):
        _prune_network(net, all_p, values)

    model.extrinsic = _prune_branches(model.extrinsic, all_p, values)
    model.parasitic = _prune_branches(model.parasitic, all_p, values)

    for k, ap_key in _ACCESS_KEY.items():
        if not model.access_names.get(k):
            continue
        v = all_p.get(ap_key)
        if _present(v):
            values[f"access_{k}"] = float(v)
        else:
            model.access_names[k] = ""     # blank name ⇒ leg absent (drawn as a wire)

    # The controlled source itself is a structural part of the core (always
    # drawn); only its printed value can go blank — _fmt() already renders 0
    # as "" — so its keys are passed straight through, no pruning.
    for k in model.source_keys():
        v = all_p.get(_SOURCE_KEY.get(k, k))
        values[k] = float(v) if _present(v) else 0.0

    return model, values


def render_svg_topology(short: str, all_p: dict, fname: str) -> None:
    """Streamlit wrapper: build the pruned model, render its SVG, and display
    it with the same iframe + download/copy pattern as the custom-model
    builder's Use tab (``custom_model/ui_use.py``)."""
    import streamlit as st

    model, values = build_pruned_model(short, all_p)
    svg = render_schematic(model, values)
    st.iframe(svg, height=svg_pixel_height(svg) + 12)
    svg_png_buttons(svg, filename=f"{short}_{fname}_topology.png", zoom=2)


# ════════════════════════════════════════════════════════════════════════════
# Open / Short pad-dummy illustrations
# ════════════════════════════════════════════════════════════════════════════
# These two pads have no CustomModel/preset (they're not a device topology,
# just the probe-pad parasitic network measured before the DUT), so they're
# hand-drawn directly with the same low-level ``_SVG`` primitives / palette /
# value formatting (``_fmt``) that ``custom_model.schematic._draw`` uses for
# every other component — see that module for the drawing-primitive reference.

def _pad_svg(caps: dict, inds: dict | None = None) -> str:
    """Shared Open/Short pad-dummy builder.

    Open  (``inds=None``): a pure π of 3 caps — Cpbe (P1→GND), Cpce (P2→GND),
    Cpbc (P1↔P2 bridge) — P1 and P2 have *no* direct path between them, only
    the capacitive coupling (that's what "open" means).

    Short (``inds`` given): the same 3 pad caps, PLUS a center-node T of leads
    on the main line — Lb (P1→center), Lc (center→P2), Le (center→GND) — the
    finite-inductance metal short across the DUT position.
    """
    s = _SVG()
    values = dict(caps)
    if inds:
        values.update(inds)

    y_main = 90
    y_bridge = y_main - 60
    y_gnd = y_main + 100
    x1 = 50                                  # P1 node
    x2 = 300 if inds else 220                 # P2 node (wider to fit the L chain)
    xc = (x1 + x2) / 2                        # short-pad center node (Lb/Lc/Le)

    # ── ports (same dot + label convention as the main engine's P1/P2) ───────
    s.dot(x1, y_main, r=6, color=_GND)
    s.text(x1 - 13, y_main + 5, "P1", color=_GND, weight="800", size=13, anchor="end")
    s.text(x1 - 13, y_main + 20, "Port 1 (B)", color=_GND, weight="600", size=9, anchor="end")
    s.dot(x2, y_main, r=6, color=_GND)
    s.text(x2 + 13, y_main + 5, "P2", color=_GND, weight="800", size=13, anchor="start")
    s.text(x2 + 13, y_main + 20, "Port 2 (C)", color=_GND, weight="600", size=9, anchor="start")

    # ── short: Lb — center — Lc on the main line, Le dropping to GND ────────
    if inds:
        s.comp("L", x1, y_main, xc, y_main, "Lb", "Lb", lpos="above")
        s.comp("L", xc, y_main, x2, y_main, "Lc", "Lc", lpos="above")
        s.comp("L", xc, y_main, xc, y_gnd, "Le", "Le", lside="right")

    # ── Cpbc bridge (P1↔P2, arcing over the top) ─────────────────────────────
    s.wire(x1, y_main, x1, y_bridge)
    s.wire(x2, y_main, x2, y_bridge)
    s.comp("C", x1, y_bridge, x2, y_bridge, "Cpbc", "Cpbc", lpos="above")

    # ── Cpbe / Cpce (P1/P2 → GND rail) ───────────────────────────────────────
    s.comp("C", x1, y_main, x1, y_gnd, "Cpbe", "Cpbe", lside="left")
    s.comp("C", x2, y_main, x2, y_gnd, "Cpce", "Cpce", lside="right")

    # ── ground rail + terminal marker (dot + "GND", matching the P1/P2 style) ─
    s.wire(x1, y_gnd, x2, y_gnd, w=3, color=_GND)
    s.dot(x1, y_gnd, r=5, color=_GND)
    s.text(x1 - 10, y_gnd + 4, "GND", color=_GND, weight="800", size=11, anchor="end")

    s.junction_dots()
    return s.render(s.value_layer(values, None))


def pad_open_svg(caps: dict) -> str:
    """Open pad-dummy schematic: Cpbe/Cpce/Cpbc only.  ``caps`` values are in
    farads (SI) — pass whatever's absent/zero and ``_fmt`` prints blank."""
    return _pad_svg(caps, None)


def pad_short_svg(caps: dict, inds: dict) -> str:
    """Short pad-dummy schematic: the open pad's 3 caps plus the Lb/Lc/Le
    center-node T.  ``caps`` in farads, ``inds`` in henries (both SI)."""
    return _pad_svg(caps, inds)


def render_pad_topology(kind: str, params: dict, fname: str, container=None) -> None:
    """Streamlit wrapper for the Open/Short pad-dummy illustrations.

    ``kind`` is ``"open"`` or ``"short"``; ``params`` holds Cpbe/Cpce/Cpbc
    (SI farads) and, for ``"short"``, also Lb/Lc/Le (SI henries) — extra keys
    (e.g. Rpb/Cpar_Lb) are simply ignored by the two pure builders above.
    ``container`` lets the caller place this in a column (e.g. the open/short
    side-by-side layout in ``base_ui.render_override_and_smith``); defaults to
    the main script body, matching ``svg_png_buttons``'s own convention."""
    import streamlit as st

    target = container if container is not None else st
    svg = pad_open_svg(params) if kind == "open" else pad_short_svg(params, params)
    target.iframe(svg, height=svg_pixel_height(svg) + 12)
    svg_png_buttons(svg, filename=f"{kind}_pad_{fname}.png",
                    container=container, zoom=2)

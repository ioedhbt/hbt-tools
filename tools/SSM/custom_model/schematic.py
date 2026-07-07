"""
schematic.py — live SVG schematic for a :class:`CustomModel`.

Draws a left→right two-port with real component glyphs (resistor zig-zag,
capacitor plates, inductor humps), the intrinsic π/T core drawn out as an
explicit sub-circuit (junction C∥R pairs + the controlled source), and — when
a ``values`` dict is supplied — each component's value printed beside it.

Layout
------
    top arcs ........ port-1↔port-2 bridges (extrinsic + parasitic caps)
    main line ....... P1 —[access R,L]— XB —[port extras]— ‹INTRINSIC› —
                      XC —[access R,L]— P2
    emitter rail .... intrinsic returns + extrinsic p1-gnd shunts (node EI)
    ground rail ..... parasitic pad caps to GND, fed from EI via the
                      emitter access R,L lead

Pure string templating — returned as an ``<svg>`` snippet embedded with
``st.iframe``.
"""
from __future__ import annotations

from .core import CustomModel, Network


# ── palette ──────────────────────────────────────────────────────────────
_BG = "#ffffff"
_LINE = "#1f2937"
_GND = "#374151"
_TXT = "#111827"
_VAL = "#6d28d9"
_R = "#d97706"   # resistor accent
_C = "#059669"   # capacitor accent
_L = "#2563eb"   # inductor accent
_SRC = "#7c3aed"  # controlled source
_SEL = "#dc2626"       # selection highlight (red) in the editor illustrations
_SEL_FILL = "#fef2f2"  # selection fill (light red)

_FONT = "Inter,Segoe UI,Arial,sans-serif"

# geometry
_Y_MAIN = 300
_PAD = 30
_SERIES_L = 82          # length of a series element cell (room for a value line)
_GAP_V = 36             # consistent vertical gap between stacked rows


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt(kind: str, val) -> str:
    """Engineering-formatted value string for a component kind."""
    if val is None or val == 0:
        return ""
    a = abs(val)
    if kind == "R":
        if a >= 1e6:
            return f"{val/1e6:.3g}MΩ"
        if a >= 1e3:
            return f"{val/1e3:.3g}kΩ"
        return f"{val:.3g}Ω"
    if kind == "C":
        if a < 1e-12:
            return f"{val*1e15:.3g}fF"
        if a < 1e-9:
            return f"{val*1e12:.3g}pF"
        return f"{val*1e9:.3g}nF"
    if kind == "L":
        if a < 1e-9:
            return f"{val*1e12:.3g}pH"
        if a < 1e-6:
            return f"{val*1e9:.3g}nH"
        return f"{val*1e6:.3g}µH"
    if kind == "gm":
        return f"{val*1e3:.3g}mS"
    if kind == "tau":
        return f"{val*1e12:.3g}ps"
    if kind == "alpha":
        return f"{val:.3g}"
    return f"{val:.3g}"


_ACCENT = {"R": _R, "L": _L, "C": _C}


class _SVG:
    def __init__(self):
        self.p: list[str] = []
        self.minx = 1e9
        self.maxx = 0
        self.miny = 1e9
        self.maxy = 0
        self.joints: dict = {}        # (x,y) → how many wire ends meet there
        self.values = None            # value dict (None ⇒ structure only)
        self.model = None             # for the controlled-source params caption
        self.vanchors: list = []      # (kind, cid, x, y, anchor) for the value layer

    def _bump(self, *xs, y=None, ys=None):
        for x in xs:
            self.minx = min(self.minx, x)
            self.maxx = max(self.maxx, x)
        for v in ([] if y is None else [y]) + (list(ys) if ys else []):
            self.miny = min(self.miny, v)
            self.maxy = max(self.maxy, v)

    def _joint(self, x, y):
        k = (round(x), round(y))
        self.joints[k] = self.joints.get(k, 0) + 1

    def wire(self, x1, y1, x2, y2, w=2, color=_LINE):
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                      f'y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" '
                      f'stroke-linecap="round"/>')
        self._bump(x1, x2, ys=(y1, y2))
        self._joint(x1, y1)
        self._joint(x2, y2)

    def junction_dots(self):
        """Dot only points where ≥3 wire ends meet (true intersections)."""
        for (x, y), deg in self.joints.items():
            if deg >= 3:
                self.dot(x, y)

    def _value(self, x, y, anchor, kind, cid):
        """Register a value-text slot; draw it now only if values are present.

        The slot is always recorded in ``vanchors`` so the value layer can be
        re-emitted cheaply (structure rendered once, numbers overlaid)."""
        self.vanchors.append((kind, cid, x, y, anchor))
        if self.values is not None:
            txt = _value_text(self.model, self.values, kind, cid)
            for i, ln in enumerate((txt or "").split("\n")):
                if ln:
                    self.text(x, y + i * 12, ln, size=10, anchor=anchor,
                              color=_VAL, weight="600")

    def value_layer(self, values, model) -> list:
        """Return SVG <text> strings for every value slot — the overlay layer.

        A value string may carry ``\\n`` (the controlled-source parameters are
        stacked one per line) so each line is short and never runs into a
        neighbouring component's label."""
        out = []
        for kind, cid, x, y, anchor in self.vanchors:
            txt = _value_text(model, values, kind, cid)
            for i, ln in enumerate((txt or "").split("\n")):
                if ln:
                    out.append(f'<text x="{x:.1f}" y="{y + i*12:.1f}" '
                               f'font-size="10" font-weight="600" fill="{_VAL}" '
                               f'text-anchor="{anchor}" '
                               f'font-family="{_FONT}">{_esc(ln)}</text>')
        return out

    def dot(self, x, y, r=4, color=_LINE):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')
        self._bump(x, y=y)

    def text(self, x, y, s, size=12, anchor="middle", color=_TXT, weight="600"):
        if not s:
            return
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                      f'font-weight="{weight}" fill="{color}" '
                      f'text-anchor="{anchor}" font-family="{_FONT}">{_esc(s)}</text>')
        self._bump(x, y=y)

    def current_arrow(self, x, y1, y2, label, side="left", color=_SRC):
        """Overlay a current-flow arrow (``y1→y2``) *on* the conductor at ``x``
        (drawn over the wire so it reads as the current through it), with the
        label offset to ``side`` into clear space.  The caller is responsible
        for giving the arrow a long-enough clear segment of wire to sit on."""
        dy = 1 if y2 >= y1 else -1
        self.p.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" '
                      f'y2="{y2 - 6 * dy:.1f}" stroke="{color}" '
                      f'stroke-width="2.5" stroke-linecap="round"/>')
        self._poly([(x - 4, y2 - 6 * dy), (x, y2), (x + 4, y2 - 6 * dy)], color)
        ox = 11 if side == "right" else -11
        anc = "start" if side == "right" else "end"
        self.text(x + ox, (y1 + y2) / 2 + 4, label, size=11, anchor=anc,
                  color=color, weight="700")
        self._bump(x + ox, ys=(y1, y2))

    # ── component glyphs (endpoints define the wire; body centred) ──────────
    def _poly(self, pts, color=_LINE, w=2):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.p.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                      f'stroke-width="{w}" stroke-linejoin="round" '
                      f'stroke-linecap="round"/>')

    def resistor(self, x1, y1, x2, y2, color=_R):
        horiz = abs(x2 - x1) >= abs(y2 - y1)
        amp = 6
        if horiz:
            L = x2 - x1
            b = 30
            a = x1 + (L - b) / 2
            self.wire(x1, y1, a, y1, color=color)
            self.wire(a + b, y1, x2, y1, color=color)
            n = 6
            pts = [(a, y1)]
            for i in range(n):
                xx = a + b * (i + 0.5) / n
                pts.append((xx, y1 + (amp if i % 2 == 0 else -amp)))
            pts.append((a + b, y1))
            self._poly(pts, color)
        else:
            L = y2 - y1
            b = 30
            a = y1 + (L - b) / 2
            self.wire(x1, y1, x1, a, color=color)
            self.wire(x1, a + b, x1, y2, color=color)
            n = 6
            pts = [(x1, a)]
            for i in range(n):
                yy = a + b * (i + 0.5) / n
                pts.append((x1 + (amp if i % 2 == 0 else -amp), yy))
            pts.append((x1, a + b))
            self._poly(pts, color)

    def capacitor(self, x1, y1, x2, y2, color=_C):
        horiz = abs(x2 - x1) >= abs(y2 - y1)
        gap, plate = 7, 18
        if horiz:
            cx = (x1 + x2) / 2
            self.wire(x1, y1, cx - gap / 2, y1, color=color)
            self.wire(cx + gap / 2, y1, x2, y1, color=color)
            for dx in (-gap / 2, gap / 2):
                self.wire(cx + dx, y1 - plate / 2, cx + dx, y1 + plate / 2, w=3, color=color)
        else:
            cy = (y1 + y2) / 2
            self.wire(x1, y1, x1, cy - gap / 2, color=color)
            self.wire(x1, cy + gap / 2, x1, y2, color=color)
            for dy in (-gap / 2, gap / 2):
                self.wire(x1 - plate / 2, cy + dy, x1 + plate / 2, cy + dy, w=3, color=color)

    def inductor(self, x1, y1, x2, y2, color=_L):
        horiz = abs(x2 - x1) >= abs(y2 - y1)
        if horiz:
            L = x2 - x1
            b = 32
            a = x1 + (L - b) / 2
            self.wire(x1, y1, a, y1, color=color)
            self.wire(a + b, y1, x2, y1, color=color)
            r = b / 8
            d = [f"M {a:.1f} {y1:.1f}"]
            for i in range(4):
                d.append(f"a {r:.1f} {r:.1f} 0 0 1 {2*r:.1f} 0")
            self.p.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
                          f'stroke-width="2"/>')
        else:
            L = y2 - y1
            b = 32
            a = y1 + (L - b) / 2
            self.wire(x1, y1, x1, a, color=color)
            self.wire(x1, a + b, x1, y2, color=color)
            r = b / 8
            d = [f"M {x1:.1f} {a:.1f}"]
            for i in range(4):
                d.append(f"a {r:.1f} {r:.1f} 0 0 0 0 {2*r:.1f}")
            self.p.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
                          f'stroke-width="2"/>')

    def source(self, x1, y1, x2, y2, label, direction="down",
               labelpos="below", color=_SRC, params=False):
        """Dependent (controlled) current source — diamond + current arrow.

        Drawn along the (x1,y1)→(x2,y2) segment (horizontal or vertical).
        ``direction`` ∈ {up, down, left, right} sets the arrow; ``labelpos`` ∈
        {below, above, left, right} sets the label placement.
        """
        horiz = abs(x2 - x1) >= abs(y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rad = 16
        if horiz:
            self.wire(x1, y1, cx - rad, cy, color=color)
            self.wire(cx + rad, cy, x2, y2, color=color)
        else:
            self.wire(x1, y1, cx, cy - rad, color=color)
            self.wire(cx, cy + rad, x2, y2, color=color)
        self.p.append(
            f'<polygon points="{cx:.1f},{cy-rad:.1f} {cx+rad:.1f},{cy:.1f} '
            f'{cx:.1f},{cy+rad:.1f} {cx-rad:.1f},{cy:.1f}" fill="#f5f3ff" '
            f'stroke="{color}" stroke-width="2"/>')
        if direction == "down":
            self.wire(cx, cy - 6, cx, cy + 5, color=color)
            self._poly([(cx - 4, cy + 1), (cx, cy + 7), (cx + 4, cy + 1)], color)
        elif direction == "up":
            self.wire(cx, cy + 6, cx, cy - 5, color=color)
            self._poly([(cx - 4, cy - 1), (cx, cy - 7), (cx + 4, cy - 1)], color)
        elif direction == "right":
            self.wire(cx - 6, cy, cx + 5, cy, color=color)
            self._poly([(cx + 1, cy - 4), (cx + 7, cy), (cx + 1, cy + 4)], color)
        else:  # left
            self.wire(cx + 6, cy, cx - 5, cy, color=color)
            self._poly([(cx - 1, cy - 4), (cx - 7, cy), (cx - 1, cy + 4)], color)
        # Place the label so 'below'/'above' line up with neighbouring component
        # labels: for a vertical source use the block endpoints, not the diamond.
        yb, yt = max(y1, y2), min(y1, y2)
        below_y = (yb + 14) if not horiz else (cy + rad + 14)
        above_y = (yt - 18) if not horiz else (cy - rad - 8)
        lx, ly, anc = cx, below_y, "middle"
        if labelpos == "above":
            ly = above_y
        elif labelpos == "right":
            lx, ly, anc = cx + rad + 6, cy + 4, "start"
        elif labelpos == "left":
            lx, ly, anc = cx - rad - 6, cy + 4, "end"
        self.text(lx, ly, label, size=11, anchor=anc, color=color, weight="700")
        if params:                       # the gm/τ or α params go in the value layer
            self._value(lx, ly + 13, anc, "params", None)

    def comp(self, kind, x1, y1, x2, y2, name="", cid=None,
             lside="right", lpos="above"):
        col = _ACCENT.get(kind, _LINE)
        {"R": self.resistor, "L": self.inductor, "C": self.capacitor}[kind](
            x1, y1, x2, y2, color=col)
        horiz = abs(x2 - x1) >= abs(y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if horiz:
            ny = cy - 14 if lpos == "above" else cy + 16
            vy = cy - 26 if lpos == "above" else cy + 28
            self.text(cx, ny, name, size=11, weight="700")
            self._value(cx, vy, "middle", kind, cid)
        else:
            yt, yb = min(y1, y2), max(y1, y2)
            if lside == "above":
                self.text(cx, yt - 18, name, size=11, weight="700")
                self._value(cx, yt - 6, "middle", kind, cid)
            elif lside == "below":
                self.text(cx, yb + 14, name, size=11, weight="700")
                self._value(cx, yb + 26, "middle", kind, cid)
            else:
                lx, anc = (x1 - 14, "end") if lside == "left" else (x1 + 14, "start")
                self.text(lx, cy - 2, name, size=11, anchor=anc, weight="700")
                self._value(lx, cy + 12, anc, kind, cid)

    def render(self, extra: list | None = None) -> str:
        mx, my = 60, 34                   # consistent outer margins (x, y)
        w = int(self.maxx - self.minx + 2 * mx)
        h = int(self.maxy - self.miny + 2 * my)
        sx, sy = mx - self.minx, my - self.miny
        parts = self.p + (extra or [])
        body = (f'<g transform="translate({sx:.1f},{sy:.1f})">'
                + "\n".join(parts) + "</g>")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                f'viewBox="0 0 {w} {h}" style="background:{_BG};border-radius:10px">'
                f'{body}</svg>')


# ════════════════════════════════════════════════════════════════════════════
def _vget(values, cid):
    if not values:
        return None
    return values.get(cid)


def _value_text(model, values, kind, cid):
    """Formatted value string for a slot ('params' = source caption)."""
    if values is None:
        return ""
    if kind == "params":
        return _src_params(model, values) or ""
    return _fmt(kind, _vget(values, cid))


def _bridge_half(net):
    """How far a p1↔p2 bridge reaches above/below its centre line, including
    labels — used to keep a uniform gap between stacked bridges."""
    groups = _net_series(net)
    n = max((len(g) for g in groups), default=1)
    if n == 1:
        return 34.0, 16.0                     # single cap: label above, plates below
    half = 60 * (n - 1) / 2 + 34              # 'h' spacing (60) + label
    return half, half


def _net_series(net: Network):
    """Flatten a Network into ordered series groups, each a list of
    (kind, name) parallel elements (skipping empty groups)."""
    return [[(e.kind, e.name, e.id) for e in g] for g in net.groups if g]


def _has_ce(model):
    """Structure decision (value-independent): show the c–e output branch only
    when the model's intrinsic c–e Network has at least one element."""
    return not model.intrinsic_ce.is_empty


def _src_params(model, values):
    """Compact controlled-source parameter caption (shown beside the source),
    or None when no values are supplied.  Every parameter carries its notation
    (Gm0=, τ=, α₀=, τB=, τC=)."""
    if values is None:
        return None
    iv = lambda k: _vget(values, k)
    # Newline-separated → drawn stacked (short lines that don't crowd neighbours).
    if model.intrinsic_type == "Pi":
        return f"Gm0={_fmt('gm', iv('gm'))}\nτ={_fmt('tau', iv('tau'))}"
    return (f"α₀={_fmt('alpha', iv('alpha0'))}\n"
            f"τB={_fmt('tau', iv('tauB'))}\nτC={_fmt('tau', iv('tauC'))}")


# ── self-contained parallel "group blocks" ─────────────────────────────────
# A group is a compact parallel block: its elements span between two rails
# (perpendicular to the elements), and the block joins the rest of the circuit
# through a *single* lead line on each side — so groups read as discrete units
# rather than the main wires threading through the parallel arrangement.
#
# A branch is a 4-tuple: (kind, a, b, side) where kind ∈ {R,L,C} → (kind, name,
# value, label-side) and kind == "src" → ("src", label, arrow-direction,
# label-side).

def _branch_v(s: _SVG, x, y_t, y_b, br):
    kind, a, b, side = br[0], br[1], br[2], br[3]
    if kind == "src":
        s.source(x, y_t, x, y_b, a, direction=b, labelpos=side,
                 params=(br[4] if len(br) > 4 else False))
    else:
        s.comp(kind, x, y_t, x, y_b, a, b, lside=side)


def _branch_h(s: _SVG, x_l, x_r, y, br):
    kind, a, b, pos = br[0], br[1], br[2], br[3]
    if kind == "src":
        s.source(x_l, y, x_r, y, a, direction=b, labelpos=pos,
                 params=(br[4] if len(br) > 4 else False))
    else:
        s.comp(kind, x_l, y, x_r, y, a, b, lpos=pos)


def _group_v(s: _SVG, x, y_top, y_bot, branches, offsets, block_h=66):
    """Vertical parallel group: a compact block joined by single leads at ``x``
    from ``y_top`` (top node) and to ``y_bot`` (bottom node)."""
    block_h = min(block_h, max((y_bot - y_top) - 16, 20))   # keep a lead each end
    cy = (y_top + y_bot) / 2
    yt, yb = cy - block_h / 2, cy + block_h / 2
    xs = [x + o for o in offsets]
    if len(branches) > 1:                       # rails split at x so the T-joins register
        s.wire(min(xs), yt, x, yt); s.wire(x, yt, max(xs), yt)
        s.wire(min(xs), yb, x, yb); s.wire(x, yb, max(xs), yb)
    s.wire(x, y_top, x, yt)                      # single connecting leads
    s.wire(x, yb, x, y_bot)
    for xi, br in zip(xs, branches):
        _branch_v(s, xi, yt, yb, br)


def _group_h(s: _SVG, x_left, x_right, y, branches, offsets, block_w=116):
    """Horizontal parallel group: a compact block joined by single leads at
    ``y`` from ``x_left`` (left node) and to ``x_right`` (right node)."""
    block_w = min(block_w, max((x_right - x_left) - 16, 20))  # keep a lead each end
    cx = (x_left + x_right) / 2
    xl, xr = cx - block_w / 2, cx + block_w / 2
    ys = [y + o for o in offsets]
    if len(branches) > 1:                       # rails split at y so the T-joins register
        s.wire(xl, min(ys), xl, y); s.wire(xl, y, xl, max(ys))
        s.wire(xr, min(ys), xr, y); s.wire(xr, y, xr, max(ys))
    s.wire(x_left, y, xl, y)                     # single connecting leads
    s.wire(xr, y, x_right, y)
    for yi, br in zip(ys, branches):
        _branch_h(s, xl, xr, yi, br)


def _net_groups(net: Network):
    """[(kind, name, id), …] per non-empty series group of a junction Network."""
    return [[(e.kind, e.name, e.id) for e in g] for g in net.groups if g]


def _src_branch(model, direction, side):
    """The controlled-source branch tuple for the current device/topology."""
    return ("src", model.source_name, direction, side, True)


def _junction_branches(group, orient, src=None):
    """Build (branches, offsets) for one parallel group, optionally appending
    the controlled source as an extra parallel branch.  Symmetric offsets keep
    the connecting leads centred for any element count."""
    items = list(group)
    n = len(items) + (1 if src else 0)
    if orient == "v":
        sp = 48 if n <= 2 else 84
        sides = (["right"] if n == 1 else
                 ["left", "right"] if n == 2 else ["left"] * n)
    else:
        sp = 60
        sides = (["above"] if n == 1 else
                 [("above" if 2 * i < n else "below") for i in range(n)])
    offsets = [(i - (n - 1) / 2) * sp for i in range(n)]
    branches = []
    seq = [(k, nm, cid) for (k, nm, cid) in items]
    for (k, nm, cid), side in zip(seq, sides):
        branches.append((k, nm, cid, side))
    if src:
        s_side = sides[len(seq)]
        branches.append((src[0], src[1], src[2], s_side, src[4]))
    return branches, offsets


def _grp_to_branches(group, orient):
    """Convert a parallel group ``[(kind, name, id), …]`` into
    ``(branches, offsets)`` for ``_group_h`` (orient='h') or ``_group_v``
    (orient='v').  Offsets are symmetric about 0 so the single connecting lead
    always lands on the block's rail — works for *any* number of parallel
    elements (a customisable model may add 1, 2, 3, …)."""
    n = len(group)
    if orient == "v":
        # ≤2 elements: labels on the outer sides. 3+: spread wide so each label
        # sits *beside* its component (to the left), all at the same height.
        if n <= 2:
            sp = 48
            sides = ["right"] if n == 1 else ["left", "right"]
        else:
            sp = 84
            sides = ["left"] * n
    else:
        sp = 60
        sides = (["above"] if n == 1 else
                 [("above" if 2 * i < n else "below") for i in range(n)])
    offsets = [(i - (n - 1) / 2) * sp for i in range(n)]
    branches = [(k, nm, cid, side)              # 3rd item = value id
                for (k, nm, cid), side in zip(group, sides)]
    return branches, offsets


def _draw_junction_v(s, x, y_top, y_bot, net: Network, src=None):
    """Draw a junction Network vertically (top node → bottom rail).  Multiple
    series groups stack into equal slots; ``src`` is appended to the last group
    as a parallel branch.  Returns nothing (single lead lands on ``x``)."""
    groups = _net_groups(net)
    if not groups:
        groups = [[]]
    seg = (y_bot - y_top) / len(groups)
    for i, group in enumerate(groups):
        gsrc = src if i == len(groups) - 1 else None
        br, offs = _junction_branches(group, "v", gsrc)
        _group_v(s, x, y_top + seg * i, y_top + seg * (i + 1), br, offs)


def _draw_junction_h(s, x_l, x_r, y, net: Network, src=None):
    """Draw a junction Network horizontally (left node → right node)."""
    groups = _net_groups(net)
    if not groups:
        groups = [[]]
    seg = (x_r - x_l) / len(groups)
    for i, group in enumerate(groups):
        gsrc = src if i == len(groups) - 1 else None
        br, offs = _junction_branches(group, "h", gsrc)
        _group_h(s, x_l + seg * i, x_l + seg * (i + 1), y, br, offs)


def _tw(text, fs: float) -> float:
    """Rough pixel width of a label string at font size ``fs`` (Inter/Arial)."""
    return len(str(text or "")) * fs * 0.56


def _lblw(net: Network, values, extra: float = 0.0) -> float:
    """Widest reserved-box width over a junction's element labels (name + value
    when ``values`` is given), plus ``extra`` (e.g. the source label).  Drives
    the content-aware spacing so each component owns enough white space."""
    w = extra
    for g in net.groups:
        for e in g:
            w = max(w, _tw(e.name, 11))
            if values is not None:
                w = max(w, _tw(_fmt(e.kind, _vget(values, e.id)), 10))
    return w


def _left_extent(net: Network, values) -> float:
    """How far left of a vertical junction's x its *left-pointing* labels reach
    (px) — the parallel-branch offset + lead (14) + label/value width.  Used to
    reserve horizontal white space so a left label clears whatever sits to the
    left (e.g. an extrinsic Cbex/Cgsx shunt leg)."""
    ext = 0.0
    for group in _net_groups(net):
        branches, offs = _grp_to_branches(group, "v")
        for (k, _nm, cid, side), off in zip(branches, offs):
            if side != "left":
                continue
            w = _tw(_nm, 11)
            if values is not None:
                w = max(w, _tw(_fmt(k, _vget(values, cid)), 10))
            ext = max(ext, -off + 14 + w)
    return ext


def _draw_intrinsic(s, xL, y_main, y_ei, model: CustomModel, values,
                    left_shunt: bool = False):
    """Draw the editable intrinsic core (base spreading + b–e / b–c / c–e
    junctions + controlled source) from the four Networks.  The b–c span is
    sized from the junction label widths so long names/values never overlap.
    Returns ``(emit_xs, x_right)`` — the emitter-rail x's and the core's right
    edge (the caller continues the main line from there).

    ``left_shunt`` — True when an extrinsic shunt (Cbex/Cgsx) taps the inner
    base node at ``xL``; the b–e junction's left-pointing label then needs
    extra horizontal room so it clears that shunt's vertical leg.

    * **π** — source (gm·Vbe / Ids) sits in the c–e branch, arrow ↓.
    * **T** — source (α·Ie / α·Is) sits in the b–c branch, arrow ←.
    """
    is_t = model.intrinsic_type == "T"
    has_ce = _has_ce(model)

    # ── content-aware label widths (drive the reserved white space) ───────────
    be_w = _lblw(model.intrinsic_be, values)
    ce_w = _lblw(model.intrinsic_ce, values)
    bc_w = _lblw(model.intrinsic_bc, values,
                 extra=_tw(model.source_name, 11) if is_t else 0.0)

    # Base spreading network (Rbi …) as horizontal series block(s).
    x = xL
    for group in _net_series(model.intrinsic_base):
        br, offs = _grp_to_branches(group, "h")
        _group_h(s, x, x + _SERIES_L, y_main, br, offs)
        x += _SERIES_L

    # Left gap from the inner base node to the b–e junction.  A single-element
    # b–e labels on the right only; with ≥2 parallel elements the first label
    # points *left* (toward xL).  When an extrinsic shunt leg sits at xL, widen
    # this gap by the label's actual left reach so it clears the leg.
    left_pad = (max(44, int(_left_extent(model.intrinsic_be, values) + 10))
                if left_shunt else 44)
    x_B = x + left_pad
    s.wire(x, y_main, x_B, y_main)

    # ── content-aware b–c span: reserve room for the b–e (right) and c–e (left)
    # labels plus the b–c centre label (and the T source caption) ─────────────
    span = max(280, int(116 + 2 * max(be_w, ce_w) + bc_w))
    x_C = x_B + span

    # π needs room on the right for the source (its own vertical leg so its
    # params point into clear space and never crowd the Cce∥Rce labels).
    src_pad = 0 if is_t else (96 if has_ce else 70)
    xR = x_C + 44 + (0 if is_t else src_pad)

    # b–e junction (down to emitter rail)
    _draw_junction_v(s, x_B, y_main, y_ei, model.intrinsic_be)

    if is_t:
        # b–c junction (B→C) carries the α source (arrow ←, params below).
        bc_src = _src_branch(model, "left", "below")
        _draw_junction_h(s, x_B, x_C, y_main, model.intrinsic_bc, src=bc_src)
        emit_xs = [x_B]
        if has_ce:
            _draw_junction_v(s, x_C, y_main, y_ei, model.intrinsic_ce)
            emit_xs.append(x_C)
        s.wire(x_C, y_main, xR, y_main)
        return emit_xs, xR

    # ── π : passive b–c, optional passive c–e, then the source on its own leg ──
    _draw_junction_h(s, x_B, x_C, y_main, model.intrinsic_bc)
    emit_xs = [x_B]
    if has_ce:
        _draw_junction_v(s, x_C, y_main, y_ei, model.intrinsic_ce)
        emit_xs.append(x_C)
    x_S = x_C + src_pad                      # source leg, clear to the right
    s.wire(x_C, y_main, x_S, y_main)
    ce_src = _src_branch(model, "down", "right")    # params point right → clear
    _draw_junction_v(s, x_S, y_main, y_ei, Network(), src=ce_src)
    emit_xs.append(x_S)
    s.wire(x_S, y_main, xR, y_main)
    return emit_xs, xR


_LEG_SLOT = 54
# Extra clear wire reserved for the "Ie" current-sense arrow (after-Cbex: a stub
# above Re; before-Cbex: extra b–e bottom-lead room via a taller intrinsic).
_IE_STUB = 40


def _last_group_bottom(net: Network, y_top: float, y_bot: float,
                       block_max: float = 66.0) -> float:
    """Y of the bottom of a vertical junction's *last* glyph block — i.e. where
    its bottom connecting lead begins.  Mirrors the slot/block math in
    :func:`_draw_junction_v` + :func:`_group_v` so a caller can place an
    annotation on the clear lead below the glyphs."""
    groups = _net_groups(net) or [[]]
    ng = len(groups)
    seg = (y_bot - y_top) / ng
    block_h = min(block_max, max(seg - 16.0, 20.0))
    cy = y_top + (ng - 0.5) * seg
    return cy + block_h / 2.0


def _draw(s: _SVG, model: CustomModel, values=None) -> None:
    """Draw the whole structure (symbols + names) into ``s``; value slots are
    recorded in ``s.vanchors`` (drawn only if ``s.values`` is set).  ``values``
    (when given) is used only to *size* the content-aware spacing so values fit
    without overlap — the value text itself is still overlaid via the value
    layer."""
    an = model.access_names

    emitter_groups = _net_series(model.emitter)
    leg_access = ([("R", "Re")] if an.get("Re") else []) + \
                 ([("L", "Le")] if an.get("Le") else [])
    n_leg = len(emitter_groups) + len(leg_access)
    has_em = n_leg > 0

    # "Ie" current-sense arrow geometry — drawn for any T-core.  Placement:
    #   • before-Cbex  → on the b–e junction's (lengthened) bottom lead
    #   • after-Cbex / no Cbex → on a clear stub in the emitter leg above Re
    # ("before/after" only differ when an extrinsic Cbex P1↔GND is present; with
    # no Cbex the sensing point is unambiguous, so the arrow sits above Re.)
    has_cbex = any(b.place == "p1-gnd" and not b.network.is_empty
                   for b in model.extrinsic)
    is_t = model.intrinsic_type == "T"
    ie_before = is_t and has_cbex and not model.ie_after_cbex
    ie_after = is_t and not ie_before     # leg stub above Re (incl. no-Cbex)
    leg_stub = _IE_STUB if ie_after else 0

    y_ei = _Y_MAIN + 116 + (_IE_STUB if ie_before else 0)
    y_gnd = y_ei + (_LEG_SLOT * n_leg if has_em else 0) + leg_stub

    x = _PAD

    def series_sym(kind, name, cid):
        # Fill the whole cell so consecutive elements share endpoints — no gaps.
        nonlocal x
        s.comp(kind, x, _Y_MAIN, x + _SERIES_L, _Y_MAIN, name, cid)
        x += _SERIES_L

    # P1 terminal — short tag left of the node, device port name beneath it
    s.dot(x, _Y_MAIN, r=6, color=_GND)
    s.text(x - 13, _Y_MAIN + 5, "P1", color=_GND, weight="800", size=13, anchor="end")
    s.text(x - 13, _Y_MAIN + 20, model.port_label("p1"), color=_GND,
           weight="600", size=9, anchor="end")
    s._bump(x - 13 - 6 * len(model.port_label("p1")))   # label extends leftward
    p1_x = x
    s.wire(p1_x, _Y_MAIN, p1_x + 10, _Y_MAIN)
    x += 10

    # base access: Lb (lead L) nearest the pad/parasitics, Rb nearest the core
    if an.get("Lb"):
        series_sym("L", an["Lb"], "access_Lb")
    if an.get("Rb"):
        series_sym("R", an["Rb"], "access_Rb")

    s.wire(x, _Y_MAIN, x + 14, _Y_MAIN)
    x += 14
    xb_x = x

    for group in _net_series(model.port1):
        br, offs = _grp_to_branches(group, "h")
        _group_h(s, x, x + _SERIES_L, _Y_MAIN, br, offs)
        x += _SERIES_L

    s.wire(x, _Y_MAIN, x + 14, _Y_MAIN)
    x += 14
    xL = x
    # An extrinsic shunt (Cbex/Cgsx) on the p1-side taps the inner base node at
    # xL → tell the intrinsic to reserve horizontal room for the b–e label.
    left_shunt = any(b.place == "p1-gnd" and not b.network.is_empty
                     for b in model.extrinsic)
    emitter_xs, xR = _draw_intrinsic(s, xL, _Y_MAIN, y_ei, model, values,
                                     left_shunt)
    x = xR

    s.wire(x, _Y_MAIN, x + 14, _Y_MAIN)
    x += 14
    for group in _net_series(model.port2):
        br, offs = _grp_to_branches(group, "h")
        _group_h(s, x, x + _SERIES_L, _Y_MAIN, br, offs)
        x += _SERIES_L

    xc_x = x
    s.wire(x, _Y_MAIN, x + 14, _Y_MAIN)
    x += 14

    if an.get("Rc"):
        series_sym("R", an["Rc"], "access_Rc")
    if an.get("Lc"):
        series_sym("L", an["Lc"], "access_Lc")

    s.wire(x, _Y_MAIN, x + 10, _Y_MAIN)
    x += 10
    p2_x = x
    s.dot(p2_x, _Y_MAIN, r=6, color=_GND)
    s.text(p2_x + 13, _Y_MAIN + 5, "P2", color=_GND, weight="800", size=13, anchor="start")
    s.text(p2_x + 13, _Y_MAIN + 20, model.port_label("p2"), color=_GND,
           weight="600", size=9, anchor="start")
    s._bump(p2_x + 13 + 6 * len(model.port_label("p2")))   # label extends rightward

    # ── emitter (EI) rail spanning only its connected x-coordinates ─────────
    em_pts = list(emitter_xs)
    if any(b.place == "p1-gnd" and not b.network.is_empty for b in model.extrinsic):
        em_pts.append(xL)      # extrinsic taps the inner base node (inside port extras)
    leg_x = (emitter_xs[0] if model.intrinsic_type == "T"
             else (min(emitter_xs) + max(emitter_xs)) / 2)
    em_pts.append(leg_x)
    if max(em_pts) - min(em_pts) > 1:
        s.wire(min(em_pts), y_ei, max(em_pts), y_ei, w=2, color=_GND)

    gnd_pts = [leg_x]
    if any(b.place == "p1-gnd" and not b.network.is_empty for b in model.parasitic):
        gnd_pts.append(p1_x)
    if any(b.place == "p2-gnd" and not b.network.is_empty for b in model.parasitic):
        gnd_pts.append(p2_x)

    # emitter leg: EI → [emitter extras] → [Ie stub] → [Re] → [Le] → GND
    ie_arrow_seg = None                  # (y_top, y_bot) clear stub for the arrow
    if has_em:
        yy = y_ei
        for group in emitter_groups:
            br, offs = _grp_to_branches(group, "v")
            _group_v(s, leg_x, yy, yy + _LEG_SLOT, br, offs)
            yy += _LEG_SLOT
        if leg_stub:                     # clear stub above the access R/L (Re)
            s.wire(leg_x, yy, leg_x, yy + leg_stub, w=2, color=_GND)
            ie_arrow_seg = (yy, yy + leg_stub)
            yy += leg_stub
        for k, key in leg_access:
            s.comp(k, leg_x, yy, leg_x, yy + _LEG_SLOT, an[key], f"access_{key}")
            yy += _LEG_SLOT
        s.wire(min(gnd_pts), y_gnd, max(gnd_pts), y_gnd, w=3, color=_GND)
    elif leg_stub:                       # no access leg — drop a stub to GND
        s.wire(leg_x, y_ei, leg_x, y_gnd, w=2, color=_GND)
        ie_arrow_seg = (y_ei, y_gnd)
        s.wire(min(gnd_pts), y_gnd, max(gnd_pts), y_gnd, w=3, color=_GND)
    else:
        s.wire(min(gnd_pts), y_ei, max(gnd_pts), y_ei, w=3, color=_GND)

    # ── shunt branches (node → rail) ───────────────────────────────────────
    # ``side`` forces the element labels to point *away* from the core (left for
    # p1-side shunts, right for p2-side) so they never collide with the
    # intrinsic junction labels that point inward.
    def shunt(node_x, net, y_bot, side="left"):
        groups = _net_series(net)
        if not groups:
            return                       # empty cap branch → open, draw nothing
        seg = (y_bot - _Y_MAIN) / len(groups)
        for i, group in enumerate(groups):
            br, offs = _grp_to_branches(group, "v")
            br = [(b[0], b[1], b[2], side) for b in br]   # force label side
            _group_v(s, node_x, _Y_MAIN + seg * i, _Y_MAIN + seg * (i + 1), br, offs)

    for b in model.extrinsic:
        if b.place == "p1-gnd" and not b.network.is_empty:
            shunt(xL, b.network, y_ei, side="left")   # inner base node
    for b in model.parasitic:
        if b.network.is_empty:
            continue
        if b.place == "p1-gnd":
            shunt(p1_x, b.network, y_gnd, side="left")
        elif b.place == "p2-gnd":
            shunt(p2_x, b.network, y_gnd, side="right")

    # ── "Ie" current-sense arrow (T-core with an extrinsic Cbex P1↔GND) ──────
    # Overlaid on a dedicated clear stub so it never crowds a component/label:
    # after-Cbex → the emitter-leg stub above Re; before-Cbex → the b–e
    # junction's (lengthened) bottom lead, after the Cbe/Rbe glyphs.
    if ie_after and ie_arrow_seg is not None:
        yt, yb = ie_arrow_seg
        s.current_arrow(leg_x, yt + 5, yb - 5, "Ie", side="left")
    elif ie_before:
        yb_last = _last_group_bottom(model.intrinsic_be, _Y_MAIN, y_ei)
        ya_lo, ya_hi = yb_last + 5, y_ei - 5
        if ya_hi - ya_lo >= 12:
            s.current_arrow(emitter_xs[0], ya_lo, ya_hi, "Ie", side="left")

    # ── top bridges (p1-p2): stack at a *consistent* gap above the intrinsic
    # and from each other (widest outermost/highest), accounting for each
    # bridge's own height so gaps stay uniform regardless of element count.
    top = [(xL, xR, b.network) for b in model.extrinsic     # inner base/collector
           if b.place == "p1-p2" and not b.network.is_empty]
    top += [(p1_x, p2_x, b.network) for b in model.parasitic
            if b.place == "p1-p2" and not b.network.is_empty]
    top.sort(key=lambda t: t[1] - t[0], reverse=True)
    if top:
        intr_top = _Y_MAIN - (70 if model.intrinsic_type == "T" else 48)
        y_edge = intr_top - _GAP_V                 # bottom edge of the lowest bridge
        ys_for = {}
        for lvl in reversed(range(len(top))):      # narrowest (innermost) first
            above, below = _bridge_half(top[lvl][2])
            line_y = y_edge - below
            ys_for[lvl] = line_y
            y_edge = line_y - above - _GAP_V
        for lvl, (xa, xb, net) in enumerate(top):
            y = ys_for[lvl]
            groups = _net_series(net)
            s.wire(xa, _Y_MAIN, xa, y)
            s.wire(xb, _Y_MAIN, xb, y)
            if not groups:
                s.wire(xa, y, xb, y)
                continue
            seg = (xb - xa) / len(groups)
            for i, group in enumerate(groups):
                br, offs = _grp_to_branches(group, "h")
                _group_h(s, xa + seg * i, xa + seg * (i + 1), y, br, offs)


def build_schematic(model: CustomModel, values: dict | None = None) -> _SVG:
    """Render the structure once (symbols + names + junction dots), recording
    value slots.  ``values`` sizes the content-aware spacing so value text fits;
    overlay the numbers cheaply via ``s.value_layer``."""
    s = _SVG()
    s.model = model
    _draw(s, model, values)
    s.junction_dots()
    return s


def render_schematic(model: CustomModel, values: dict | None = None) -> str:
    """One-shot render (structure + optional value numbers, content-aware)."""
    s = build_schematic(model, values)
    return s.render(s.value_layer(values, model) if values else None)


def svg_to_png(svg: str, zoom: int = 2) -> bytes | None:
    """Rasterise an SVG string to PNG bytes.  Tries ``rsvg-convert`` (system
    binary), then ``cairosvg`` (pure-python).  Returns None if neither is
    available — callers should fall back to offering the SVG download."""
    import shutil
    import subprocess
    exe = shutil.which("rsvg-convert")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--zoom", str(zoom), "-b", "white"],
                input=svg.encode("utf-8"), capture_output=True, check=True,
                timeout=15)          # never block the Streamlit thread forever
            return out.stdout
        except Exception:
            pass
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg.encode("utf-8"), scale=zoom)
    except Exception:
        return None


def svg_pixel_height(svg: str, default: int = 360) -> int:
    """Native pixel height declared in the ``<svg height="…">`` tag — used to
    size the embedding iframe so the (value-dependent, now taller) schematic is
    never vertically clipped."""
    import re
    m = re.search(r'<svg[^>]*\bheight="(\d+)', svg)
    return int(m.group(1)) if m else default


def copy_image_button(png: bytes, *, container=None,
                      label: str = "📋 copy image", height: int = 46) -> None:
    """Render a button that copies a PNG image to the clipboard (as an image,
    pasteable into docs/slides), styled to match the project's ``copy_button``.
    Uses the async ``ClipboardItem`` API."""
    import base64
    import html as _html
    import streamlit as st
    target = container if container is not None else st
    if not png:
        return
    b64 = base64.b64encode(png).decode("ascii")
    safe_label = _html.escape(label)
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;background:transparent;overflow:hidden;}}
  .wrap{{position:relative;}}
  button{{
    width:100%;box-sizing:border-box;cursor:pointer;
    font-family:"Source Sans Pro","Segoe UI",sans-serif;font-size:0.875rem;
    line-height:1.6;padding:0.25rem 0.75rem;min-height:38.4px;
    border:1px solid rgba(49,51,63,0.2);border-radius:0.5rem;
    background:#fff;color:rgb(38,39,48);transition:border-color .15s,color .15s;
  }}
  button:hover{{border-color:#4A90D9;color:#4A90D9;}}
  #toast{{
    position:absolute;left:50%;top:50%;
    transform:translate(-50%,-50%) scale(0.96);
    background:#1f8a4c;color:#fff;font-weight:600;font-size:0.8rem;
    font-family:"Source Sans Pro","Segoe UI",sans-serif;
    padding:5px 12px;border-radius:6px;white-space:nowrap;
    box-shadow:0 2px 8px rgba(0,0,0,0.28);opacity:0;pointer-events:none;
    transition:opacity .15s ease,transform .15s ease;
  }}
  #toast.show{{opacity:1;transform:translate(-50%,-50%) scale(1);}}
</style></head><body>
<div class="wrap">
  <button id="cb">{safe_label}</button>
  <div id="toast">✓ Image copied</div>
</div>
<script>
  const b64 = "{b64}";
  const btn = document.getElementById("cb");
  const toast = document.getElementById("toast");
  let timer = null;
  function flash(msg) {{
    if (msg) toast.textContent = msg;
    toast.classList.add("show");
    if (timer) clearTimeout(timer);
    timer = setTimeout(function() {{ toast.classList.remove("show"); }}, 1100);
  }}
  function b64ToBlob(b) {{
    const bin = atob(b); const len = bin.length;
    const arr = new Uint8Array(len);
    for (let i=0;i<len;i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], {{type:"image/png"}});
  }}
  btn.addEventListener("click", function() {{
    try {{
      const blob = b64ToBlob(b64);
      if (navigator.clipboard && window.ClipboardItem) {{
        navigator.clipboard.write([new ClipboardItem({{"image/png": blob}})])
          .then(function(){{ flash("✓ Image copied"); }})
          .catch(function(){{ flash("⚠ Use Download"); }});
      }} else {{ flash("⚠ Use Download"); }}
    }} catch (e) {{ flash("⚠ Use Download"); }}
  }});
</script></body></html>"""
    target.iframe(doc, height=height)


def svg_png_buttons(svg: str, *, filename: str = "schematic.png",
                    container=None, zoom: int = 2,
                    dl_label: str = "🖼️ Download PNG",
                    copy_label: str = "📋 copy image",
                    height: int = 46) -> None:
    """Render *Download PNG* + *Copy image* buttons that rasterise ``svg`` to a
    PNG **entirely in the browser** (an HTML ``<canvas>``) — no server-side
    ``rsvg-convert`` / ``cairosvg`` and therefore no native packages required.
    PNG export works identically on Streamlit Cloud and any local machine.  The
    two buttons share one iframe and a single (cached) rasterisation."""
    import base64
    import html as _html
    import json
    import streamlit as st
    target = container if container is not None else st
    if not svg:
        return
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    dl = _html.escape(dl_label)
    cp = _html.escape(copy_label)
    fname = json.dumps(filename)
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;background:transparent;overflow:hidden;}}
  .row{{position:relative;display:flex;gap:8px;}}
  button{{
    flex:1;box-sizing:border-box;cursor:pointer;
    font-family:"Source Sans Pro","Segoe UI",sans-serif;font-size:0.875rem;
    line-height:1.6;padding:0.25rem 0.75rem;min-height:38.4px;
    border:1px solid rgba(49,51,63,0.2);border-radius:0.5rem;
    background:#fff;color:rgb(38,39,48);transition:border-color .15s,color .15s;
  }}
  button:hover{{border-color:#4A90D9;color:#4A90D9;}}
  #toast{{
    position:absolute;left:50%;top:50%;
    transform:translate(-50%,-50%) scale(0.96);
    background:#1f8a4c;color:#fff;font-weight:600;font-size:0.8rem;
    font-family:"Source Sans Pro","Segoe UI",sans-serif;
    padding:5px 12px;border-radius:6px;white-space:nowrap;
    box-shadow:0 2px 8px rgba(0,0,0,0.28);opacity:0;pointer-events:none;
    transition:opacity .15s ease,transform .15s ease;z-index:2;
  }}
  #toast.show{{opacity:1;transform:translate(-50%,-50%) scale(1);}}
</style></head><body>
<div class="row">
  <button id="dl">{dl}</button>
  <button id="cp">{cp}</button>
  <div id="toast">✓ Image copied</div>
</div>
<script>
  const B64 = "{b64}";
  const ZOOM = {zoom};
  const FNAME = {fname};
  const toast = document.getElementById("toast");
  let timer = null, pngBlob = null;
  function flash(msg) {{
    if (msg) toast.textContent = msg;
    toast.classList.add("show");
    if (timer) clearTimeout(timer);
    timer = setTimeout(function() {{ toast.classList.remove("show"); }}, 1100);
  }}
  function rasterize(cb) {{
    if (pngBlob) {{ cb(pngBlob); return; }}
    const img = new Image();
    img.onload = function() {{
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(w * ZOOM));
      c.height = Math.max(1, Math.round(h * ZOOM));
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, c.width, c.height);
      ctx.setTransform(ZOOM, 0, 0, ZOOM, 0, 0);
      ctx.drawImage(img, 0, 0);
      c.toBlob(function(b) {{ pngBlob = b; cb(b); }}, "image/png");
    }};
    img.onerror = function() {{ cb(null); }};
    img.src = "data:image/svg+xml;base64," + B64;
  }}
  rasterize(function() {{}});           // preload so the click is instant
  document.getElementById("dl").addEventListener("click", function() {{
    rasterize(function(b) {{
      if (!b) {{ flash("⚠ export failed"); return; }}
      const url = URL.createObjectURL(b);
      const a = document.createElement("a");
      a.href = url; a.download = FNAME;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function() {{ URL.revokeObjectURL(url); }}, 4000);
    }});
  }});
  document.getElementById("cp").addEventListener("click", function() {{
    rasterize(function(b) {{
      if (!b) {{ flash("⚠ export failed"); return; }}
      try {{
        if (navigator.clipboard && window.ClipboardItem) {{
          navigator.clipboard.write([new ClipboardItem({{"image/png": b}})])
            .then(function(){{ flash("✓ Image copied"); }})
            .catch(function(){{ flash("⚠ Use Download"); }});
        }} else {{ flash("⚠ Use Download"); }}
      }} catch (e) {{ flash("⚠ Use Download"); }}
    }});
  }});
</script></body></html>"""
    target.iframe(doc, height=height)


# ════════════════════════════════════════════════════════════════════════════
# Abstracted intrinsic-editor illustration
# ════════════════════════════════════════════════════════════════════════════
def _part_names(net: Network) -> str:
    els = [e.name for g in net.groups for e in g if e.name]
    return " ∥ ".join(els) if els else "—"


def intrinsic_thumbnail(model: CustomModel, selected: str | None = None) -> str:
    """A compact, *abstracted* view of the intrinsic core: each editable part
    (base spreading, B–E / B–C / C–E junctions, controlled source) drawn as a
    labelled box around a base node, with ``selected`` highlighted.  Pairs with
    the chip selector so the user 'clicks' a part to edit just that one."""
    t = model.terminals()
    is_t = model.intrinsic_type == "T"
    W = 600
    y_main = 62
    box_top, box_h = 96, 150    # taller B–E / C–E junction boxes
    box_bot = box_top + box_h
    y_emit = box_bot + 48
    H = y_emit + 66
    NB, NC = 162, 408           # base node, collector node
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" style="background:{_BG};border-radius:10px" '
         f'font-family="{_FONT}">']

    def line(x1, y1, x2, y2, dash=False, color=_LINE, w=2):
        d = ' stroke-dasharray="4 3"' if dash else ""
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{w}" stroke-linecap="round"{d}/>')

    def node(x, y):
        p.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{_LINE}"/>')

    def term(x, y, lbl, anchor, sub=None):
        p.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{_GND}"/>')
        dx = -11 if anchor == "end" else 11
        p.append(f'<text x="{x+dx}" y="{y+4}" font-size="13" font-weight="800" '
                 f'text-anchor="{anchor}" fill="{_GND}">{_esc(lbl)}</text>')
        if sub:
            p.append(f'<text x="{x+dx}" y="{y+18}" font-size="10" '
                     f'font-weight="600" text-anchor="{anchor}" '
                     f'fill="#6b7280">{_esc(sub)}</text>')

    def box(x, y, w, h, key, title, sub):
        sel = (selected == key)
        accent = _SEL if sel else "#9ca3af"
        fill = _SEL_FILL if sel else "#ffffff"
        cx, cy = x + w / 2, y + h / 2
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                 f'fill="{fill}" stroke="{accent}" stroke-width="{3 if sel else 1.5}"/>')
        p.append(f'<text x="{cx}" y="{cy-2}" font-size="11" font-weight="700" '
                 f'text-anchor="middle" fill="#111827">{_esc(title)}</text>')
        p.append(f'<text x="{cx}" y="{cy+13}" font-size="10" font-weight="600" '
                 f'text-anchor="middle" fill="{_VAL}">{_esc(sub)}</text>')

    # signal-path lines + nodes
    term(26, y_main, t["p1"], "end", "P1")
    line(26, y_main, 44, y_main)
    line(134, y_main, NB, y_main); node(NB, y_main)
    line(NB, y_main, 196, y_main)
    line(388, y_main, NC, y_main); node(NC, y_main)
    line(NC, y_main, 470, y_main); term(470, y_main, t["p2"], "start", "P2")
    # emitter rail
    line(NB, y_emit, NC, y_emit, color=_GND, w=3)
    term(NB - 26, y_emit, t["com"], "end")
    # vertical legs (B–E, C–E)
    line(NB, y_main, NB, box_top); line(NB, box_bot, NB, y_emit)
    line(NC, y_main, NC, box_top); line(NC, box_bot, NC, y_emit)

    # boxes
    box(44, y_main - 18, 90, 36, "base", "Base", _part_names(model.intrinsic_base))
    box(196, y_main - 18, 192, 36, "bc",
        f"{t['p1']}–{t['p2']}", _part_names(model.intrinsic_bc))
    box(NB - 46, box_top, 92, box_h, "be",
        f"{t['p1']}–{t['com']}", _part_names(model.intrinsic_be))
    ce_sub = _part_names(model.intrinsic_ce)
    box(NC - 46, box_top, 92, box_h, "ce",
        f"{t['p2']}–{t['com']}", ce_sub if ce_sub != "—" else "(add)")

    # controlled source diamond — in the b–c leg (T) or c–e leg (π)
    sx, sy = (285, box_top + 44) if is_t else (NC + 70, box_top + box_h / 2)
    sel = (selected == "src")
    accent = _SEL if sel else "#9ca3af"
    if is_t:
        line(285, y_main + 18, sx, sy - 14, dash=True, color=accent, w=1.5)
    else:
        line(NC, sy, sx, sy, dash=True, color=accent, w=1.5)
    p.append(f'<polygon points="{sx},{sy-14} {sx+14},{sy} {sx},{sy+14} '
             f'{sx-14},{sy}" fill="{_SEL_FILL if sel else "#fff"}" '
             f'stroke="{accent}" stroke-width="{3 if sel else 1.5}"/>')
    p.append(f'<text x="{sx}" y="{sy+30}" font-size="10" font-weight="700" '
             f'text-anchor="middle" fill="{_SEL if sel else _SRC}">'
             f'{_esc(model.source_name)}</text>')

    p.append("</svg>")
    return "".join(p)


_CORE_LABEL = {
    "extrinsic": ["Intrinsic"],
    "delay":     ["Intrinsic", "+ Extrinsic C"],
    "access":    ["Intrinsic + Ext.C", "+ Delay"],
    "parasitic": ["Core", "+ Access R/L"],
}


def section_thumbnail(model: CustomModel, section: str,
                      selected: str | None = None) -> str:
    """Abstracted, *cumulative* illustration for an outer build section: the
    already-built inner circuit is collapsed into one labelled **core block**
    (two ports + common rail), and only this section's components are drawn
    around it, with ``selected`` highlighted.  ``section`` ∈ {extrinsic, delay,
    access, parasitic}."""
    t = model.terminals()
    W, H = 580, 384
    BX1, BY1, CW, CH = 214, 150, 152, 92
    BX2, BY2 = BX1 + CW, BY1 + CH
    YM = BY1 + CH / 2
    CXB = BX1 + CW / 2
    P1X, P2X, GNDY = 54, 526, 340
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" style="background:{_BG};border-radius:10px" '
         f'font-family="{_FONT}">']

    def line(x1, y1, x2, y2, dash=False, color=_LINE, w=2):
        d = ' stroke-dasharray="4 3"' if dash else ""
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{w}" stroke-linecap="round"{d}/>')

    def node(x, y):
        p.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{_LINE}"/>')

    def term(x, y, lbl, anchor, sub=None):
        p.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{_GND}"/>')
        dx = -11 if anchor == "end" else 11
        p.append(f'<text x="{x+dx}" y="{y+4}" font-size="13" font-weight="800" '
                 f'text-anchor="{anchor}" fill="{_GND}">{_esc(lbl)}</text>')
        if sub:
            p.append(f'<text x="{x+dx}" y="{y+18}" font-size="10" '
                     f'font-weight="600" text-anchor="{anchor}" '
                     f'fill="#6b7280">{_esc(sub)}</text>')

    def box(cx, cy, w, h, key, title, sub, present=True):
        sel = (selected == key)
        accent = _SEL if sel else ("#9ca3af" if present else "#cbd5e1")
        fill = _SEL_FILL if sel else "#ffffff"
        x, y = cx - w / 2, cy - h / 2
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="{accent}" stroke-width="{3 if sel else 1.5}"'
                 f'{"" if present else " stroke-dasharray=\"4 3\""}/>')
        tc = "#111827" if present else "#94a3b8"
        p.append(f'<text x="{cx}" y="{cy-1}" font-size="10.5" font-weight="700" '
                 f'text-anchor="middle" fill="{tc}">{_esc(title)}</text>')
        p.append(f'<text x="{cx}" y="{cy+12}" font-size="9.5" font-weight="600" '
                 f'text-anchor="middle" fill="{_VAL if present else "#cbd5e1"}">'
                 f'{_esc(sub)}</text>')

    # ── core block + ports + common rail ────────────────────────────────────
    p.append(f'<rect x="{BX1}" y="{BY1}" width="{CW}" height="{CH}" rx="10" '
             f'fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>')
    lines = _CORE_LABEL[section]
    ystart = YM - (len(lines) - 1) * 8
    for i, ln in enumerate(lines):
        p.append(f'<text x="{CXB}" y="{ystart + i*16 + 4}" font-size="12" '
                 f'font-weight="800" text-anchor="middle" fill="#3730a3">'
                 f'{_esc(ln)}</text>')
    term(P1X, YM, t["p1"], "end", "P1")
    term(P2X, YM, t["p2"], "start", "P2")
    rail_l, rail_r = (44, 536) if section == "parasitic" else (150, 430)
    line(rail_l, GNDY, rail_r, GNDY, color=_GND, w=3)      # common rail
    term(rail_l - 4, GNDY - 16, t["com"], "start")

    # ── section-specific components ─────────────────────────────────────────
    if section == "delay":
        line(P1X, YM, BX1, YM); line(BX2, YM, P2X, YM)
        line(CXB, BY2, CXB, GNDY)
        box((P1X + BX1) / 2, YM, 92, 34, "port1",
            f"Port 1 ({t['p1']})", _part_names(model.port1),
            present=not model.port1.is_empty)
        box((BX2 + P2X) / 2, YM, 92, 34, "port2",
            f"Port 2 ({t['p2']})", _part_names(model.port2),
            present=not model.port2.is_empty)
        box(CXB, (BY2 + GNDY) / 2, 120, 32, "emitter",
            f"{t['com']} delay", _part_names(model.emitter),
            present=not model.emitter.is_empty)

    elif section == "access":
        line(P1X, YM, BX1, YM); line(BX2, YM, P2X, YM)
        line(CXB, BY2, CXB, GNDY)
        an = model.access_names
        # Lead L is outermost (toward P1/P2); access R is inner (toward core).
        box(108, YM, 48, 30, "Lb", an.get("Lb") or "Lb", "L", present=bool(an.get("Lb")))
        box(166, YM, 48, 30, "Rb", an.get("Rb") or "Rb", "R", present=bool(an.get("Rb")))
        box(414, YM, 48, 30, "Rc", an.get("Rc") or "Rc", "R", present=bool(an.get("Rc")))
        box(472, YM, 48, 30, "Lc", an.get("Lc") or "Lc", "L", present=bool(an.get("Lc")))
        box(CXB, BY2 + 30, 48, 28, "Re", an.get("Re") or "Re", "R", present=bool(an.get("Re")))
        box(CXB, BY2 + 64, 48, 28, "Le", an.get("Le") or "Le", "L", present=bool(an.get("Le")))

    else:   # extrinsic / parasitic — one placeholder box per standard place
        line(P1X, YM, BX1, YM); line(BX2, YM, P2X, YM)
        line(CXB, BY2, CXB, GNDY)
        is_par = section == "parasitic"
        branches = model.parasitic if is_par else model.extrinsic

        def pnames(place):
            els = [e.name for b in branches if b.place == place
                   for e in b.network.elements() if e.name]
            return " ∥ ".join(els) if els else "(add)"

        def ppresent(place):
            return any(not b.network.is_empty
                       for b in branches if b.place == place)

        # P1 ↔ P2 top bridge (dashed until a component is added)
        by = 92
        lx, rx = (P1X, P2X) if is_par else (BX1 - 16, BX2 + 16)
        pr = ppresent("p1-p2")
        line(lx, YM, lx, by, dash=not pr); line(rx, YM, rx, by, dash=not pr)
        line(lx, by, rx, by, dash=not pr)
        box(CXB, by, 132, 30, "p1-p2", "P1 ↔ P2", pnames("p1-p2"), present=pr)

        # P1 ↔ GND left shunt
        lx2 = P1X if is_par else BX1 - 16
        pr = ppresent("p1-gnd")
        line(lx2, YM, lx2, GNDY, dash=not pr)
        box(lx2, (YM + GNDY) / 2, 104, 30, "p1-gnd", "P1 ↔ GND",
            pnames("p1-gnd"), present=pr)

        # ── "Ie" current-sense arrow (T-core only) ──────────────────────────
        # The Cbex (P1↔GND) tap merges at the common rail, so anything on the
        # emitter leg above it is *before* Cbex.  After-Cbex therefore drops the
        # arrow on a stub *below the whole emitter* (past the merge); before-Cbex
        # / no-Cbex keep it on the emitter leg, just above the rail.  Mirrors the
        # live schematic so the before/after radio gives feedback in this preview.
        if not is_par and model.intrinsic_type == "T":
            if pr and bool(model.ie_after_cbex):          # after Cbex → below rail
                line(CXB, GNDY, CXB, GNDY + 30, color=_GND)
                ay1, ay2 = GNDY + 6, GNDY + 28
            else:                                          # before Cbex / no Cbex
                ay1, ay2 = GNDY - 32, GNDY - 8
            p.append(f'<line x1="{CXB}" y1="{ay1}" x2="{CXB}" y2="{ay2 - 6}" '
                     f'stroke="{_SRC}" stroke-width="2.5" '
                     f'stroke-linecap="round"/>')
            p.append(f'<polyline points="{CXB-4},{ay2-6} {CXB},{ay2} '
                     f'{CXB+4},{ay2-6}" fill="none" stroke="{_SRC}" '
                     f'stroke-width="2.5" stroke-linejoin="round" '
                     f'stroke-linecap="round"/>')
            p.append(f'<text x="{CXB+9}" y="{(ay1+ay2)/2+4}" font-size="11" '
                     f'font-weight="700" text-anchor="start" fill="{_SRC}">'
                     f'Ie</text>')

        # P2 ↔ GND right shunt (parasitic only)
        if is_par:
            pr = ppresent("p2-gnd")
            line(P2X, YM, P2X, GNDY, dash=not pr)
            box(P2X, (YM + GNDY) / 2, 104, 30, "p2-gnd", "P2 ↔ GND",
                pnames("p2-gnd"), present=pr)

    p.append("</svg>")
    return "".join(p)

"""Draw arrows, circles, boxes and numbered callouts on a screenshot.

Coordinates are given in *CSS pixels* (what Playwright reports), not device
pixels — the harness captures at device_scale_factor=2, so this module scales
by the ratio it measures from the image width against a declared page width.

    from annotate import Annotator

    a = Annotator("raw/step1.png", css_width=1600)
    a.circle(320, 480, r=34)
    a.arrow(from_=(520, 400), to=(360, 470))
    a.box(100, 200, 600, 260, label="1")
    a.caption("Upload the open and short files here")
    a.save("../docs/en/assets/rf/step1.png")
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ACCENT = (220, 38, 38, 255)      # red-600, reads on both light and dark UI
ACCENT_SOFT = (220, 38, 38, 60)
INK = (31, 41, 51, 255)
PAPER = (255, 255, 255, 235)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int):
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


class Annotator:
    def __init__(self, src, css_width: int = 1600):
        self.img = Image.open(src).convert("RGBA")
        self.k = self.img.width / float(css_width)
        self.overlay = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.overlay)
        self.lw = max(2, round(3 * self.k))

    # ── geometry helpers ────────────────────────────────────────────────
    def _p(self, x, y):
        return (x * self.k, y * self.k)

    def circle(self, x, y, r=30, color=ACCENT):
        cx, cy = self._p(x, y)
        rr = r * self.k
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                       outline=color, width=self.lw)
        return self

    def box(self, x, y, w, h, color=ACCENT, radius=8):
        x0, y0 = self._p(x, y)
        x1, y1 = self._p(x + w, y + h)
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=radius * self.k,
                                 outline=color, width=self.lw)
        return self

    def arrow(self, from_, to, color=ACCENT, head=16):
        import math

        x0, y0 = self._p(*from_)
        x1, y1 = self._p(*to)
        self.d.line([x0, y0, x1, y1], fill=color, width=self.lw)
        ang = math.atan2(y1 - y0, x1 - x0)
        h = head * self.k
        for sign in (+1, -1):
            a = ang + sign * math.radians(150)
            self.d.line([x1, y1, x1 + h * math.cos(a), y1 + h * math.sin(a)],
                        fill=color, width=self.lw)
        return self

    def marker(self, x, y, n, r=17, color=ACCENT):
        """A numbered disc — pairs with a numbered list in the prose."""
        cx, cy = self._p(x, y)
        rr = r * self.k
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color)
        f = _font(int(23 * self.k))
        t = str(n)
        bb = self.d.textbbox((0, 0), t, font=f)
        self.d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2 - bb[1]),
                    t, font=f, fill=(255, 255, 255, 255))
        return self

    def label(self, x, y, text, color=ACCENT, anchor="lt"):
        f = _font(int(20 * self.k))
        px, py = self._p(x, y)
        bb = self.d.textbbox((0, 0), text, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        pad = 7 * self.k
        if "r" in anchor:
            px -= w + 2 * pad
        if "b" in anchor:
            py -= h + 2 * pad
        self.d.rounded_rectangle([px, py, px + w + 2 * pad, py + h + 2 * pad],
                                 radius=5 * self.k, fill=PAPER, outline=color,
                                 width=max(1, self.lw - 1))
        self.d.text((px + pad, py + pad - bb[1]), text, font=f, fill=INK)
        return self

    def dim(self, keep=None):
        """Grey out everything except an optional (x, y, w, h) region."""
        veil = Image.new("RGBA", self.img.size, (255, 255, 255, 120))
        if keep:
            x, y, w, h = keep
            x0, y0 = self._p(x, y)
            x1, y1 = self._p(x + w, y + h)
            ImageDraw.Draw(veil).rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))
        self.img = Image.alpha_composite(self.img, veil)
        return self

    def crop(self, x, y, w, h):
        x0, y0 = self._p(x, y)
        x1, y1 = self._p(x + w, y + h)
        box = tuple(int(v) for v in (x0, y0, x1, y1))
        self.img = self.img.crop(box)
        self.overlay = self.overlay.crop(box)
        self.d = ImageDraw.Draw(self.overlay)
        return self

    def save(self, dest, max_width=1400):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = Image.alpha_composite(self.img, self.overlay)
        if max_width and out.width > max_width:
            h = round(out.height * max_width / out.width)
            out = out.resize((max_width, h), Image.LANCZOS)
        out.convert("RGB").save(dest, "PNG", optimize=True)
        return dest

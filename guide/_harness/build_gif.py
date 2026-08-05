"""Assemble frame_NN.png captures into an animated GIF.

Plain PIL, no Playwright, so it can run long after the capture session.
Frames ping-pong (forward then back) so the loop reads as a drag rather than
a jump-cut at the wrap.

    python3 guide/_harness/build_gif.py <shots-dir> <out.gif> [--width 900]
                                        [--ms 420] [--no-pingpong]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shots")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--ms", type=int, default=420)
    ap.add_argument("--no-pingpong", action="store_true")
    a = ap.parse_args()

    frames = sorted(Path(a.shots).glob("frame_*.png"))
    if not frames:
        print("no frames in", a.shots)
        return 1

    imgs = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        if a.width and im.width != a.width:
            im = im.resize((a.width, round(im.height * a.width / im.width)),
                           Image.LANCZOS)
        imgs.append(im)

    seq = imgs if a.no_pingpong else imgs + imgs[-2:0:-1]
    # Quantize against one shared palette; per-frame palettes make the
    # background shimmer between frames.
    pal = seq[0].quantize(colors=192, method=Image.MEDIANCUT)
    seq = [im.quantize(palette=pal, dither=Image.NONE) for im in seq]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seq[0].save(out, save_all=True, append_images=seq[1:],
                duration=a.ms, loop=0, optimize=True, disposal=2)
    kb = out.stat().st_size / 1024
    print(f"{out}  {len(seq)} frames  {seq[0].size[0]}x{seq[0].size[1]}  {kb:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

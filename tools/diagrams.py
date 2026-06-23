"""
diagrams.py — Reusable, code-drawn "how it works" illustrations.

A single horizontal-pipeline drawer used by each tool's collapsed
"ℹ️ How it works" block.  Matplotlib (Agg, no pyplot global state), cached on
its arguments so each distinct pipeline is rendered only once.  Labels are
plain text (no emoji — matplotlib can't render them) so they stay legible.
"""
from __future__ import annotations

import io

import streamlit as st


@st.cache_data(show_spinner=False)
def pipeline_png(stages: tuple, accent: str = "#1f77b4") -> bytes:
    """Render a left-to-right arrow pipeline.

    ``stages`` is a tuple of ``(title, subtitle)`` pairs (subtitle may be "").
    ``accent`` is the box/border colour.  Returned as PNG bytes.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

    n = len(stages)
    gap = 0.04
    w = (1.0 - gap * (n - 1)) / n
    y0, h = 0.18, 0.64

    fig = Figure(figsize=(max(8.0, 2.3 * n), 1.9), dpi=130)
    ax = fig.add_subplot(111)

    centers = []
    for i, (title, subtitle) in enumerate(stages):
        x_left = i * (w + gap)
        cx = x_left + w / 2
        centers.append(cx)
        ax.add_patch(FancyBboxPatch(
            (x_left, y0), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.03",
            linewidth=1.8, edgecolor=accent, facecolor=accent,
            alpha=0.10, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x_left, y0), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.03",
            linewidth=1.8, edgecolor=accent, facecolor="none", zorder=3))
        ax.text(cx, y0 + h - 0.12, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=accent, zorder=4)
        if subtitle:
            ax.text(cx, y0 + h - 0.34, subtitle, ha="center", va="top",
                    fontsize=7.5, color="0.25", zorder=4)

    for i in range(n - 1):
        ax.annotate("", xy=(centers[i + 1] - w / 2 - 0.004, y0 + h / 2),
                    xytext=(centers[i] + w / 2 + 0.004, y0 + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.6))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

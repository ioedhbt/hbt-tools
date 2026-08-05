"""tuning.md — Auto Tuning section, "before" state: sticky residual row +
Smith chart + fT/fmax (Bode) panel, all in one frame, before either tuner
has touched the model.

Uses the raw (not de-embedded) demo file so the loaded model (seeded by
seed_fit_cache.py) sits at the documented 18.06% total residual
(S11 6.44%, S12 5.98%, S21 24.27%, S22 35.54%).

Shots:
  01_before_charts — hbt_resrow metric strip + Smith overlay + fT/fmax card.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf/raw")


def hide_badge(page):
    page.evaluate(
        "() => {"
        "let s = document.getElementById('_hide_badge');"
        "if (!s) { s = document.createElement('style'); s.id = '_hide_badge';"
        "document.head.appendChild(s); }"
        "s.textContent = 'div.st-key-lang_toggle{display:none !important;}';"
        "}"
    )


def run(page, shot):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent))
    from session import settle
    from b_chartfix import revive_visible_charts

    page.set_default_timeout(8000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)
    hide_badge(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page)
    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=700)

    # Default viewport (1600x1000 css) is shorter than resrow + Smith +
    # fT/fmax combined, and a clip taller than the viewport gets truncated
    # to whatever is actually painted -- grow the viewport for this shot.
    page.set_viewport_size({"width": 1600, "height": 1500})
    settle(page, quiet_ms=300)

    resrow = page.locator('div[class*="st-key-hbt_resrow_"]').first
    # scrollIntoView(block:'start') lands the row exactly at the sticky
    # engagement boundary (top: 2.9rem), and its solid background then
    # paints over the Smith-chart title sitting in normal flow just below
    # it. Overshoot upward by 160px so the row sits well clear of that
    # boundary -- comfortably in normal flow -- while the now-tall (1500px)
    # viewport still leaves ~1300px of room below for Smith + fT/fmax.
    resrow.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    page.evaluate("() => window.scrollBy(0, -160)")
    settle(page, quiet_ms=300)
    hide_badge(page)
    revive_visible_charts(page)
    rb = resrow.bounding_box()
    print("RESROW BOX", rb)
    shot("01_before_charts", clip={"x": 370, "y": max(0, rb["y"] - 8),
                                    "width": 1500, "height": 660})
    print("DONE")

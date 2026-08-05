"""Recapture the Cold-HBT Rb/Rc panel.

The first attempt was cropped through the left edge of the formulas and let the
fixed RAM badge bleed into the frame.  This one hides the badge and clips to
the panel's own bounding box.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_chartfix import revive_visible_charts  # noqa: E402

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")

BIAS_FILES = ["deemb_preext_vce3.5_ib200u.s2p", "deemb_preext_cold.s2p"]


def hide_badge(page):
    page.evaluate(
        "() => {let s=document.getElementById('_hb');"
        "if(!s){s=document.createElement('style');s.id='_hb';"
        "document.head.appendChild(s);}"
        "s.textContent='div.st-key-lang_toggle{display:none !important;}';}"
    )


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role(
        "link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=250)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=250)
    sb = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=250)
    sb.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=250)

    page.locator('[data-testid="stMain"] input[type="file"]').first.set_input_files(
        [str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page, quiet_ms=350)
    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=450)

    page.get_by_text("🍊 Access Resistance Extraction", exact=False).first.click()
    page.get_by_text("Cold-HBT Extraction", exact=True).first.click(timeout=8000)
    settle(page, quiet_ms=350)
    page.get_by_text("📊 Interactive Parameter Extraction",
                     exact=True).first.click(timeout=8000)
    settle(page, quiet_ms=350)

    head = page.get_by_text("Step 5 — Rb, Rc", exact=False).first
    head.wait_for(timeout=8000)
    head.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=350)
    revive_visible_charts(page, wait_ms=900)
    hide_badge(page)
    box = head.bounding_box()
    print("HEAD BOX", box, flush=True)
    shot("00_rbrc", clip={"x": box["x"] - 12, "y": box["y"] - 12,
                          "width": 1560, "height": 940})

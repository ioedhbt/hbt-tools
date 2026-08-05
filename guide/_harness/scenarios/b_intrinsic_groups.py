"""ssm-intrinsic.md — Step 2 Cbex/Cbcx cards + Step 3 Cbc median control.
Split off from b_intrinsic_final.py (model selection already captured) to
stay comfortably inside the 40s scenario ceiling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_chartfix import revive_visible_charts  # noqa: E402

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")

BIAS_FILES = [
    "deemb_preext_vce3.5_ib200u.s2p",
    "deemb_preext_cold.s2p",
]

_HIDE_CSS = """() => {
    const css = `header, [data-testid="stHeader"] { visibility: hidden !important; }`;
    const s = document.createElement("style");
    s.textContent = css;
    document.head.appendChild(s);
}"""


def _clip_from(box_lo, box_hi, pad=16):
    x0 = max(0, box_lo["x"] - pad)
    y0 = max(0, box_lo["y"] - pad)
    x1 = min(1600, box_hi["x"] + box_hi["width"] + pad)
    y1 = min(1000, box_hi["y"] + box_hi["height"] + pad)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _clip_card(top_box, pad=16, width=1060, height=950):
    """Wide/tall fixed-size clip anchored at a heading's top-left — wide
    enough for a full parameter card (plot + input + all quickset rows),
    tall enough that nothing below the input gets cut off."""
    x0 = max(0, top_box["x"] - pad)
    y0 = max(0, top_box["y"] - pad)
    x1 = min(1600, x0 + width)
    y1 = min(1000, y0 + height)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=200)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=200)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=200)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=200)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files([str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page, quiet_ms=300)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=400)
    page.evaluate(_HIDE_CSS)

    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True).last
    exp.wait_for(timeout=15000)
    exp.click()
    settle(page, quiet_ms=300)

    # Step 2 — Cbex card
    h_cbex = page.get_by_text("Step 2 — Cbex", exact=False).first
    h_cbex.wait_for(timeout=10000)
    h_cbex.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=200)
    revive_visible_charts(page, wait_ms=600)
    page.evaluate(_HIDE_CSS)
    clip2 = _clip_card(h_cbex.bounding_box(), height=850)
    shot("01_cbex", clip=clip2)

    # Step 2 — Cbcx card
    h_cbcx = page.get_by_text("Step 2 — Cbcx", exact=False).first
    h_cbcx.wait_for(timeout=10000)
    h_cbcx.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=200)
    revive_visible_charts(page, wait_ms=600)
    page.evaluate(_HIDE_CSS)
    clip3 = _clip_card(h_cbcx.bounding_box(), height=850)
    shot("02_cbcx", clip=clip3)

    # Step 3 Intrinsic — Cbc card (median default control)
    cbc_label = page.get_by_text("Cbc (low frequency range)", exact=False).first
    cbc_label.wait_for(timeout=10000)
    cbc_label.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=200)
    revive_visible_charts(page, wait_ms=600)
    page.evaluate(_HIDE_CSS)
    med_btn = cbc_label.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'median')][1]")
    med_btn.wait_for(timeout=10000)
    print("CBC label box", cbc_label.bounding_box())
    print("MEDIAN button box", med_btn.bounding_box())
    clip4 = _clip_card(cbc_label.bounding_box(), width=560, height=340)
    shot("03_intrinsic_cbc", clip=clip4)

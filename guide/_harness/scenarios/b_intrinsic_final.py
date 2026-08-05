"""ssm-intrinsic.md — final precise captures using viewport clips (not
full_page), so printed bounding_box() coordinates map 1:1 onto the saved
PNG (full_page screenshots stitch the *whole* document, which does not
line up with viewport-relative bounding boxes after a scrollIntoView).

Shots:
  00_model_selection — Model Selection checkboxes (T-topology highlighted).
  01_cbex            — Step 2 Cbex card.
  02_cbcx             — Step 2 Cbcx card.
  03_intrinsic_cbc    — Step 3 Intrinsic group, Cbc card + quickset row.
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


def _clip_from(page, box_lo, box_hi, pad=14):
    x0 = max(0, box_lo["x"] - pad)
    y0 = max(0, box_lo["y"] - pad)
    x1 = min(1600, box_hi["x"] + box_hi["width"] + pad)
    y1 = min(1000, box_hi["y"] + box_hi["height"] + pad)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=250)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=250)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=250)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=250)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files([str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page, quiet_ms=350)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=450)
    page.evaluate(_HIDE_CSS)

    # ── Model Selection ──────────────────────────────────────────────────
    heading = page.get_by_text("🔘 Model Selection", exact=False).first
    heading.wait_for(timeout=10000)
    heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=250)
    cb_pi = page.get_by_role("checkbox", name="π-topology (Cheng 2022)").first
    cb_pi.wait_for(timeout=8000)
    hb, cbb = heading.bounding_box(), cb_pi.bounding_box()
    clip = _clip_from(page, hb, cbb, pad=18)
    shot("00_model_selection", clip=clip)

    # ── Interactive Parameter Extraction (.last = T model's, not Cold-HBT's) ──
    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True).last
    exp.wait_for(timeout=10000)
    exp.click()
    settle(page, quiet_ms=350)

    # Step 2 — Cbex card
    h_cbex = page.get_by_text("Step 2 — Cbex", exact=False).first
    h_cbex.wait_for(timeout=8000)
    h_cbex.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=250)
    revive_visible_charts(page, wait_ms=700)
    page.evaluate(_HIDE_CSS)
    cbex_default = h_cbex.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'default')][1]")
    cbex_default.wait_for(timeout=8000)
    hb2 = h_cbex.bounding_box()
    db2 = cbex_default.bounding_box()
    clip2 = _clip_from(page, hb2, db2, pad=16)
    shot("01_cbex", clip=clip2)

    # Step 2 — Cbcx card
    h_cbcx = page.get_by_text("Step 2 — Cbcx", exact=False).first
    h_cbcx.wait_for(timeout=8000)
    h_cbcx.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=250)
    revive_visible_charts(page, wait_ms=700)
    page.evaluate(_HIDE_CSS)
    cbcx_default = h_cbcx.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'default')][1]")
    cbcx_default.wait_for(timeout=8000)
    hb3 = h_cbcx.bounding_box()
    db3 = cbcx_default.bounding_box()
    clip3 = _clip_from(page, hb3, db3, pad=16)
    shot("02_cbcx", clip=clip3)

    # Step 3 Intrinsic — Cbc card (median default control)
    cbc_label = page.get_by_text("Cbc (low frequency range)", exact=False).first
    cbc_label.wait_for(timeout=8000)
    cbc_label.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=250)
    revive_visible_charts(page, wait_ms=700)
    page.evaluate(_HIDE_CSS)
    med_btn = cbc_label.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'median')][1]"
    )
    med_btn.wait_for(timeout=8000)
    hb4 = cbc_label.bounding_box()
    mb4 = med_btn.bounding_box()
    clip4 = _clip_from(page, hb4, mb4, pad=16)
    shot("03_intrinsic_cbc", clip=clip4)

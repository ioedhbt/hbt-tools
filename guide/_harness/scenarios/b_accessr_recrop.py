"""ssm-access-r.md — tight recrop of the Cold-HBT source picker (the
previous accessr_cold_source.png was a loose full-page shot that let the
sidebar and RAM badge bleed in).  Also grabs the Cold-HBT Model Fit
Verification plot (Z11-Z12 / Z12 / Z22-Z12, measured vs model) so the page
can show "how to judge the fit".

Shots:
  00_cold_source_tight — Cold-HBT source segmented control + Cold file
                          selectbox, clipped tightly (no sidebar).
  01_cold_fit_verify    — Measured vs Model Z-plane comparison figure.
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
    const css = `[data-testid="stStatusWidget"], .hbt-ram-badge,
                  div[class*="ram"], header, [data-testid="stHeader"] {
                      visibility: hidden !important; }`;
    const s = document.createElement("style");
    s.textContent = css;
    document.head.appendChild(s);
}"""


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=300)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=300)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=300)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=300)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files([str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page, quiet_ms=400)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=500)

    page.get_by_text("🍊 Access Resistance Extraction", exact=False).first.click()
    page.get_by_text("Cold-HBT Extraction", exact=True).first.click(timeout=8000)
    settle(page, quiet_ms=500)

    heading = page.get_by_text("Cold-HBT source", exact=False).first
    heading.wait_for(timeout=8000)
    sel = page.get_by_text("Cold file (from loaded DUTs)", exact=False).first
    sel.wait_for(timeout=8000)
    heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    page.wait_for_timeout(300)

    page.evaluate(_HIDE_CSS)
    hb = heading.bounding_box()
    # The selectbox control itself sits below its label — walk to the
    # combobox button so the clip includes the dropdown's full height.
    combo = page.locator('[data-testid="stSidebar"] ~ * [role="combobox"]').first
    # Fallback: nearest combobox after the "Cold file" label.
    combo2 = sel.locator("xpath=following::*[@role='combobox'][1]")
    cb = (combo2.bounding_box() if combo2.count() else None) or combo.bounding_box()
    print("HEADING BOX", hb)
    print("COMBO BOX", cb)

    if hb and cb:
        pad = 14
        x0 = hb["x"] - pad
        y0 = hb["y"] - pad
        x1 = cb["x"] + cb["width"] + pad
        y1 = cb["y"] + cb["height"] + pad
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(1600, x1); y1 = min(1000, y1)
        clip = {"x": x0, "y": y0, "width": (x1 - x0), "height": (y1 - y0)}
        print("CLIP", clip)
        shot("00_cold_source_tight", clip=clip)
    else:
        shot("00_cold_source_tight", full_page=True)

    # ── Model Fit Verification (matplotlib figure inside an expander) ──────
    exp = page.get_by_text("Cold-HBT Model Fit Verification", exact=False).first
    exp.wait_for(timeout=8000)
    exp.click()
    settle(page, quiet_ms=400)
    img = exp.locator(
        "xpath=ancestor::details[1]//img"
    ).first
    img.wait_for(timeout=8000)
    img.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=400)
    page.evaluate(_HIDE_CSS)
    ib = img.bounding_box()
    print("IMG BOX", ib)
    shot("01_cold_fit_verify", locator=img)

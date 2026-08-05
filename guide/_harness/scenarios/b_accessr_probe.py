"""Probe: multi-file upload + Z-param section locators."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")
BIAS_FILES = [
    "deemb_preext_vce3.5_ib80u.s2p",
    "deemb_preext_vce3.5_ib120u.s2p",
    "deemb_preext_vce3.5_ib160u.s2p",
    "deemb_preext_vce3.5_ib200u.s2p",
    "deemb_preext_vce3.5_ib240u.s2p",
    "deemb_preext_vce3.5_ib280u.s2p",
]


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files([str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page)
    shot("00_after_run", full_page=True)

    page.get_by_text("🍊 Access Resistance Extraction", exact=False).first.click()
    settle(page)
    page.get_by_text("Z-Parameter Method — Re(Z₁₂) vs 1/IE", exact=False).first.click()
    settle(page)

    row_text = "deemb_preext_vce3.5_ib120u"
    row = page.get_by_text(row_text, exact=False).first \
        .locator("xpath=ancestor::div[contains(@data-testid,'stHorizontalBlock')][1]")
    print("ROW BOX", row.bounding_box())
    inp = row.locator('input[inputmode="numeric"], input[inputmode="decimal"], input[type="number"]')
    print("INPUT COUNT IN ROW", inp.count())
    shot("01_zparam_expanded", full_page=True)

"""Probe 3: locator-based screenshot of a nested expander block."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

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
    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page)

    page.get_by_text("📌 Open & Short Dummy De-embedding", exact=False).first.click()
    settle(page)

    box_loc = page.get_by_text("Open Dummy: Pad Capacitances", exact=False) \
        .first.locator("xpath=ancestor::div[@data-testid='stExpander'][1]")
    print("BOX", box_loc.bounding_box())
    shot("00_open_caps", locator=box_loc)

    # Scroll to short section and grab its box too
    short_loc = page.get_by_text("Short Dummy: Lead Inductances", exact=False) \
        .first.locator("xpath=ancestor::div[@data-testid='stExpander'][1]")
    short_loc.scroll_into_view_if_needed()
    settle(page, quiet_ms=300)
    print("SHORT BOX", short_loc.bounding_box())
    shot("01_short_leads", locator=short_loc)

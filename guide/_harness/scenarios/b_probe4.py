"""Probe 4: heading-to-heading clip screenshot pattern."""
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

    heading = page.get_by_text("Open Dummy: Pad Capacitances", exact=False).first
    heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hb = heading.bounding_box()
    print("HEADING BOX", hb)

    table = page.locator('[data-testid="stDataFrame"]').first
    tb = table.bounding_box()
    print("TABLE BOX", tb)

    clip = {"x": 0, "y": max(0, hb["y"] - 10), "width": 1600,
            "height": (tb["y"] + tb["height"]) - hb["y"] + 20}
    print("CLIP", clip)
    shot("00_open_caps_clip", clip=clip)

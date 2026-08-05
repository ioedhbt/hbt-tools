"""ssm-setup.md — sidebar after Open+Short dummy files are loaded (own crop)."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1500})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page)
    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page)

    shot("00_sidebar_tall")

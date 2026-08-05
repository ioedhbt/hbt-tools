"""Find the fine-tune / residual / Smith / summary controls on the page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")
BIAS = ["deemb_preext_vce3.5_ib200u.s2p", "deemb_preext_cold.s2p"]


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role(
        "link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=200)
    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=200)
    sb = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=200)
    sb.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=200)
    page.locator('[data-testid="stMain"] input[type="file"]').first.set_input_files(
        [str(DATA / "raw" / f) for f in BIAS])
    settle(page, quiet_ms=300)
    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=400)
    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True)
    print("interactive count", exp.count(), flush=True)
    if exp.count():
        exp.last.click()
        settle(page, quiet_ms=350)

    body = page.locator('[data-testid="stMain"]').inner_text()
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    print("LINES", len(lines), flush=True)
    for s in lines:
        print(repr(s[:70]), flush=True)

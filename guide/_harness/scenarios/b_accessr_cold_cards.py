"""ssm-access-r.md — Cold-HBT extracted parameter cards (Cex/Cbc/Rbi)."""
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

    page.get_by_text("🍊 Access Resistance Extraction", exact=False).first.click()
    page.get_by_text("Cold-HBT Extraction", exact=True).first.click(timeout=8000)
    settle(page, quiet_ms=350)
    page.get_by_text("📊 Interactive Parameter Extraction", exact=True).first.click(timeout=8000)
    settle(page, quiet_ms=350)

    heading = page.get_by_text("Step 2 — Cex", exact=False).first
    heading.wait_for(timeout=8000)
    heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=350)
    revive_visible_charts(page, wait_ms=900)
    shot("00_cold_cex", full_page=True)

    rb_heading = page.get_by_text("Step 5 — Rb, Rc", exact=False).first
    rb_heading.wait_for(timeout=8000)
    rb_heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=350)
    revive_visible_charts(page, wait_ms=900)
    shot("01_cold_rb_rc", full_page=True)

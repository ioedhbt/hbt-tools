"""ssm-finetune.md — the fine-tune value grid and the S display scale row."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_chartfix import revive_visible_charts  # noqa: E402

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")
BIAS_FILES = ["deemb_preext_vce3.5_ib200u.s2p", "deemb_preext_cold.s2p"]

_HIDE_CSS = """() => {
    const s = document.createElement("style");
    s.textContent = `header, [data-testid="stHeader"],
        div.st-key-lang_toggle { visibility: hidden !important; }`;
    document.head.appendChild(s);
}"""


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
    page.evaluate(_HIDE_CSS)

    head = page.get_by_text("5 — Review", exact=False).first
    head.wait_for(timeout=20000)
    head.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    page.mouse.wheel(0, -1000)
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=800)
    page.evaluate(_HIDE_CSS)
    shot("00_scale_and_smith")

    page.mouse.wheel(0, -950)
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=800)
    page.evaluate(_HIDE_CSS)
    shot("01_above_scale")

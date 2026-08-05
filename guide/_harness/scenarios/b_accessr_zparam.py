"""ssm-access-r.md — Z-parameter method: fill Ie (mA) per bias file, screenshot
the resulting table + Re(Z12) vs 1/IE fit.

Shots:
  00_zparam_table_plot — filled bias table + fit plot + Re/Rbe metrics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_chartfix import revive_visible_charts  # noqa: E402

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")

# stem -> Ie = Ic + Ib (mA), from the bias table in the task brief.
IE_MA = {
    "deemb_preext_vce3.5_ib80u":  1.68,
    "deemb_preext_vce3.5_ib120u": 3.048,
    "deemb_preext_vce3.5_ib160u": 4.578,
    "deemb_preext_vce3.5_ib200u": 6.275,
    "deemb_preext_vce3.5_ib240u": 8.099,
    "deemb_preext_vce3.5_ib280u": 8.298,
}
BIAS_FILES = [f"{stem}.s2p" for stem in IE_MA]


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
    page.get_by_text("Z-Parameter Method — Re(Z₁₂) vs 1/IE", exact=False).first.click()
    settle(page, quiet_ms=400)

    for stem, ie in IE_MA.items():
        row = page.get_by_text(stem, exact=False).first \
            .locator("xpath=ancestor::div[contains(@data-testid,'stHorizontalBlock')][1]")
        inp = row.locator('input').last
        inp.fill(str(ie))
        inp.press("Tab")

    anchor = page.get_by_text("Z-Parameter Method — Re(Z₁₂) vs 1/IE", exact=False).first
    anchor.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=600)
    shot("00_zparam_table_plot", full_page=True)

    page.mouse.wheel(0, 500)
    settle(page, quiet_ms=400)
    revive_visible_charts(page)
    shot("01_zparam_chart", full_page=True)

    page.mouse.wheel(0, 500)
    settle(page, quiet_ms=400)
    shot("02_zparam_metrics", full_page=True)

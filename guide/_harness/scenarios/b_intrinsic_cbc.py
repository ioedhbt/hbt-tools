"""ssm-intrinsic.md — Step 3 Intrinsic group, Cbc card only: circle the
"median" quickset button (the control that overrides Cbc's low-frequency
default).  Leaner than b_intrinsic_model.py (skips Cbex/Cbcx shots) to stay
inside the 40 s scenario ceiling.
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

    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True).last
    exp.wait_for(timeout=10000)
    exp.click()
    settle(page, quiet_ms=350)

    cbc_label = page.get_by_text("Cbc (low frequency range)", exact=False).first
    cbc_label.wait_for(timeout=8000)
    cbc_label.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=800)

    med_btn = cbc_label.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'median')][1]"
    )
    med_btn.wait_for(timeout=8000)
    lb = cbc_label.bounding_box()
    mb = med_btn.bounding_box()
    print("CBC label box", lb)
    print("MEDIAN button box", mb)
    shot("00_intrinsic_cbc_median", full_page=True)

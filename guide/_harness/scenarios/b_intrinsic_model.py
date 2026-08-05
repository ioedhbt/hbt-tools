"""ssm-intrinsic.md — Model Selection (T-topology checkbox) + Interactive
Parameter Extraction: Step 2 Cbex/Cbcx groups and the Step 3 Intrinsic
group's Cbc card (median-vs-low-f default control).

Shots (full_page — cropped afterward with annotate.py using the printed
bounding boxes):
  00_model_selection — 🔘 Model Selection checkboxes.
  01_cbex_cbcx        — Step 2 Cbex + Cbcx groups.
  02_intrinsic_top    — Step 3 Intrinsic group, first row (Rbi, Rbe).
  03_intrinsic_cbc    — Step 3 Intrinsic group, Cbc card + quickset row.
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

    # ── Model Selection ──────────────────────────────────────────────────
    heading = page.get_by_text("🔘 Model Selection", exact=False).first
    heading.wait_for(timeout=10000)
    heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    cb = page.get_by_role("checkbox", name="T-topology (Cheng 2022)").first
    cb.wait_for(timeout=8000)
    print("MODEL_SEL heading box", heading.bounding_box())
    print("MODEL_SEL checkbox box", cb.bounding_box())
    shot("00_model_selection", full_page=True)

    # ── Interactive Parameter Extraction (T model's — .last, since Cold-HBT
    #    has its own expander of the same name earlier on the page) ───────
    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True).last
    exp.wait_for(timeout=10000)
    exp.click()
    settle(page, quiet_ms=400)

    h_cbex = page.get_by_text("Step 2 — Cbex", exact=False).first
    h_cbex.wait_for(timeout=8000)
    h_cbex.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=900)
    print("CBEX heading box", h_cbex.bounding_box())
    h_cbcx = page.get_by_text("Step 2 — Cbcx", exact=False).first
    print("CBCX heading box", h_cbcx.bounding_box())
    shot("01_cbex_cbcx", full_page=True)

    h_int = page.get_by_text("Step 3 — Intrinsic", exact=False).first
    h_int.wait_for(timeout=8000)
    h_int.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=900)
    print("INTRINSIC heading box", h_int.bounding_box())
    shot("02_intrinsic_top", full_page=True)

    cbc_label = page.get_by_text("Cbc (low frequency range)", exact=False).first
    cbc_label.wait_for(timeout=8000)
    cbc_label.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=900)
    med_btn = cbc_label.locator(
        "xpath=ancestor::div[contains(@data-testid,'stVerticalBlockBorderWrapper')][1]"
    ).get_by_role("button", name="median", exact=False).first
    print("CBC label box", cbc_label.bounding_box())
    print("CBC median button box", med_btn.bounding_box())
    shot("03_intrinsic_cbc", full_page=True)

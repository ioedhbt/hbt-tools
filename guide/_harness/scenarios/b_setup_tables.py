"""ssm-setup.md — post-run: Cpxx / Lx tables + the pre-override resistance panel.

Shots:
  00_open_caps   — Cpbe/Cpce/Cpbc table (clipped, main column only).
  01_short_leads — Lb/Lc/Le table (clipped).
  02_preoverride — "Choose series resistance" expander, Rb/Rc/Re = Custom 0 Ω.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def clip_from_to(page, top_loc, bottom_loc, pad=14):
    top_loc.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    from session import settle
    settle(page, quiet_ms=300)
    tb = top_loc.bounding_box()
    bb = bottom_loc.bounding_box()
    y0 = max(0, tb["y"] - pad)
    y1 = bb["y"] + bb["height"] + pad
    return {"x": 390, "y": y0, "width": 1180, "height": min(y1 - y0, 1000 - y0)}


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

    heading1 = page.get_by_text("Open Dummy: Pad Capacitances", exact=False).first
    table1 = page.locator('[data-testid="stDataFrame"]').first
    clip1 = clip_from_to(page, heading1, table1)
    print("CLIP1", clip1)
    shot("00_open_caps", clip=clip1)

    heading2 = page.get_by_text("Short Dummy: Lead Inductances", exact=False).first
    table2 = page.locator('[data-testid="stDataFrame"]').nth(1)
    clip2 = clip_from_to(page, heading2, table2)
    print("CLIP2", clip2)
    shot("01_short_leads", clip=clip2)

    # ── Pre-override panel: scroll to Section 3, open it ──────────────────
    page.get_by_text("✏️ Choose series resistance", exact=False).first \
        .evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    page.get_by_text("✏️ Choose series resistance", exact=False).first.click()
    settle(page)
    exp = page.get_by_text("✏️ Choose series resistance", exact=False).first \
        .locator("xpath=ancestor::div[@data-testid='stExpander'][1]")
    exp.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    eb = exp.bounding_box()
    print("EXP BOX", eb)
    shot("02_preoverride", clip={"x": 380, "y": max(0, eb["y"] - 10),
                                  "width": 1180,
                                  "height": min(eb["height"] + 20, 1000)})

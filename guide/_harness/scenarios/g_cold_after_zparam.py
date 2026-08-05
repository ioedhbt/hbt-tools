"""Cold-HBT with Re already known from the Z-parameter fit.

The cold extraction takes Re as an input (Gao §5.5.2 removes Re from Z_cor
before A/B/C/D are computed), so the Z-parameter fit has to run first. Filling
the six-row Z-parameter table costs a Streamlit rerun per row and blows the 40 s
budget, so this scenario types the Re that fit produces (64.3966 Ω) straight
into the cold section's own Re box, which is the same value by the same route.

Prints the numbers instead of shooting them, so they can be checked before any
page quotes them.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")
RE_ZPARAM = 64.3966
FILES = ["deemb_preext_vce3.5_ib200u.s2p", "deemb_preext_cold.s2p"]


def p(*a):
    print(*a, flush=True)


def val(page, label):
    return page.get_by_text(label, exact=True).first.locator(
        "xpath=following::input[1]").input_value()


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
        [str(DATA / "raw" / f) for f in FILES])
    settle(page, quiet_ms=300)
    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=400)

    page.get_by_text("🍊 Access Resistance Extraction", exact=False).first.click()
    page.get_by_text("Cold-HBT Extraction", exact=True).first.click(timeout=10000)
    settle(page, quiet_ms=350)
    p("cold opened")

    re_box = page.get_by_text("Re (Ω)", exact=True).first.locator(
        "xpath=following::input[1]")
    p("Re before:", re_box.input_value())
    re_box.fill(str(RE_ZPARAM))
    re_box.press("Tab")
    settle(page, quiet_ms=400)

    page.get_by_text("📊 Interactive Parameter Extraction",
                     exact=True).first.click(timeout=10000)
    settle(page, quiet_ms=350)

    cex_label = page.get_by_text("Cex (fF)", exact=True).first
    cex_label.wait_for(timeout=10000)
    cex_label.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    med = cex_label.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'median')][1]")
    p("median chip:", med.inner_text(), med.bounding_box())
    med.scroll_into_view_if_needed()
    med.click(force=True)
    settle(page, quiet_ms=900)
    page.wait_for_timeout(800)
    p("after click, Cex =", page.get_by_text("Cex (fF)", exact=True).first
      .locator("xpath=following::input[1]").input_value())
    cex_label.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    page.wait_for_timeout(400)
    shot("00_after_median")

    for label in ("Cex (fF)", "Rb (Ω)", "Rc (Ω)", "Re (Ω)", "Cbc (fF)",
                  "Rbi (Ω)", "Cbe (fF)"):
        loc = page.get_by_text(label, exact=True)
        n = loc.count()
        vals = []
        for i in range(n):
            try:
                vals.append(loc.nth(i).locator(
                    "xpath=following::input[1]").input_value())
            except Exception:
                vals.append("?")
        p(f"{label} x{n}: {vals}")

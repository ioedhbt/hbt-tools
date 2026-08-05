"""ssm-access-r.md — Cold-HBT run *after* the Z-parameter fit, Cex on median.

Order matters: the cold section's Re box reads 0.0000 until the Z-parameter fit
has produced one, and Gao §5.5.2 removes Re from Z_cor before A/B/C/D, so a cold
run done first returns Rb/Rc computed against Re = 0.

Filling the six-row Z-parameter table costs a rerun per row and does not fit the
40 s budget, so Re is typed straight in here. It is the same number the fit
produces (64.3966 Ω).

One quirk worth keeping: the quickset chips need two rerun cycles before the
number_input reflects the new value. Read it too early and you get the old one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_chartfix import revive_visible_charts  # noqa: E402

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")
RE_ZPARAM = 64.3966
FILES = ["deemb_preext_vce3.5_ib200u.s2p", "deemb_preext_cold.s2p"]

_HIDE = """() => {let s=document.getElementById('_hb');
  if(!s){s=document.createElement('style');s.id='_hb';
  document.head.appendChild(s);}
  s.textContent='div.st-key-lang_toggle{display:none !important;}';}"""


def p(*a):
    print(*a, flush=True)


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
    settle(page, quiet_ms=300)

    re_label = page.get_by_text("Re (Ω)", exact=True).first
    re_box = re_label.locator("xpath=following::input[1]")
    re_box.fill(str(RE_ZPARAM))
    re_box.press("Tab")
    settle(page, quiet_ms=350)

    page.get_by_text("📊 Interactive Parameter Extraction",
                     exact=True).first.click(timeout=10000)
    settle(page, quiet_ms=300)

    cex = page.get_by_text("Cex (fF)", exact=True).first
    cex.wait_for(timeout=10000)
    cex.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    med = cex.locator(
        "xpath=following::button[starts-with(normalize-space(.), 'median')][1]")
    med.scroll_into_view_if_needed()
    med.click(force=True)
    settle(page, quiet_ms=400)
    box = cex.locator("xpath=following::input[1]")
    for _ in range(20):                 # chips need a second rerun to land
        if box.input_value().startswith("57.8"):
            break
        page.wait_for_timeout(400)
    p("Cex =", box.input_value())

    cex.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    page.mouse.wheel(0, -420)
    revive_visible_charts(page, wait_ms=800)
    page.evaluate(_HIDE)
    shot("01_cex_median")

    rb = page.get_by_text("Step 5", exact=False).first
    rb.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    page.wait_for_timeout(600)
    page.evaluate(_HIDE)
    p("Rb =", page.get_by_text("Rb (Ω)", exact=True).first
      .locator("xpath=following::input[1]").input_value())
    p("Rc =", page.get_by_text("Rc (Ω)", exact=True).first
      .locator("xpath=following::input[1]").input_value())
    print("RB BOX", rb.bounding_box(), flush=True)
    shot("02_rb_rc")

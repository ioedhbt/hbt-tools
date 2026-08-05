"""Probe: multiselect dropdown structure for Visual Tuning."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/deembed_these/deembedded/4x10_left")


def run(page, shot):
    from session import settle

    page.set_default_timeout(6000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page)
    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)

    vt = page.get_by_text("Visual Tuning", exact=False).first
    vt.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    vt.click()
    settle(page, quiet_ms=500)

    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(has_text="Parameters to slide").first
    print("WRAP COUNT", page.locator('div[data-testid="stMultiSelect"]').filter(has_text="Parameters to slide").count())
    cb = wrap.get_by_role("combobox").first
    print("CB COUNT", wrap.get_by_role("combobox").count())
    cb.click()
    settle(page, quiet_ms=500)
    shot("00_dropdown_open", full_page=False)

    html = page.evaluate("""() => {
        const els = document.querySelectorAll('[role="listbox"], [role="option"], ul[data-baseweb], li');
        return Array.from(els).slice(0, 40).map(e => e.outerHTML.slice(0, 200));
    }""")
    for h in html:
        print("EL:", h)
    print("DONE")

"""Diagnostic: does opening T-model's Interactive Parameter Extraction with
[ib80u, ib280u] (no cold file) alone complete in time, and does filling the
tau_total fit's IC boxes afterward hang?  All prints use flush=True so
output survives even if the process is later force-killed by the outer
scenario timeout (unbuffered stdout writes reach the pipe immediately)."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")
BIAS_FILES = [
    "deemb_preext_vce3.5_ib80u.s2p",
    "deemb_preext_vce3.5_ib280u.s2p",
]


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=200)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=200)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=200)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=200)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files([str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page, quiet_ms=300)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=400)
    print("P00 after run", flush=True)

    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True).last
    exp.wait_for(timeout=15000)
    exp.click()
    settle(page, quiet_ms=300)
    print("P01 after expand", flush=True)

    tau_hdr = page.get_by_text("Cje / τB+τC / τCC / τE from 1/(2πfT) vs 1/IC fit",
                                exact=False).first
    tau_hdr.wait_for(timeout=12000, state="attached")
    print("P02 tau_hdr found, attached", flush=True)
    tau_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    print("P03 scrolled", flush=True)
    tau_hdr.click()
    print("P04 clicked", flush=True)
    settle(page, quiet_ms=300)
    print("P05 settled after click", flush=True)

    IE_MA = {
        "deemb_preext_vce3.5_ib80u":  1.68,
        "deemb_preext_vce3.5_ib280u": 8.298,
    }
    ic_header = page.get_by_text("IC (mA)", exact=True)
    print("P06 IC header count", ic_header.count(), flush=True)
    header_row = ic_header.first.locator(
        "xpath=ancestor::div[contains(@data-testid,'stHorizontalBlock')][1]")
    data_rows = header_row.locator(
        "xpath=following::div[contains(@data-testid,'stHorizontalBlock')]")
    n_rows = data_rows.count()
    print("P07 data row count", n_rows, flush=True)
    for i in range(min(n_rows, 8)):
        row = data_rows.nth(i)
        row_text = row.inner_text()
        print("P08 row", i, "text", repr(row_text[:60]), flush=True)
        stem_match = next((s for s in IE_MA if s in row_text), None)
        print("P09 matched stem", stem_match, flush=True)
        if stem_match is None:
            continue
        inp = row.locator('input').last
        print("P10 input count", inp.count(), flush=True)
        inp.fill(str(IE_MA[stem_match]))
        print("P11 filled", stem_match, flush=True)
        inp.press("Tab")
        print("P12 tabbed", stem_match, flush=True)
    settle(page, quiet_ms=400)
    print("P11 settled after fills", flush=True)

    shot("00_final", full_page=False)
    print("P12 shot done", flush=True)

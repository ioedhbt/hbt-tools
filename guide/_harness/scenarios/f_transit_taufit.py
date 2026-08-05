"""ssm-transit-time.md — type Ie = Ic + Ib into the tau_total fit's per-file
IC (mA) boxes, screenshot that table, then the tau_total vs 1/IC plot
+ metrics (Cje, tauB+tauC).

Built from the b_diag1.py bisection:
  - A real click on the nested tau_total expander IS required (its content
    is not actionable pre-click).
  - `get_by_text(stem)` / `get_by_text("IC (mA)")` are ambiguous (11-22
    matches on this heavy page) and picking the wrong one hangs .fill() —
    anchor on the unique "IC (mA)" table header instead, then walk forward
    through row blocks (`following::div[stHorizontalBlock]`) and match each
    row's own text to a stem — this is what actually finds the real inputs.
  - `input[aria-label*=stem]` looked promising but matches a small (13x13,
    off-screen) unrelated element — likely a hidden download-button label
    mentioning the filename — not the number_input. Avoid it.
  - Playwright's own `locator.bounding_box()` can go stale after several
    reruns; re-derive real coordinates with `getBoundingClientRect()` via
    `evaluate()` right before using them.

Shots:
  00_ic_table — tau_total fit's per-file IC (mA) column, filled with
                Ie = Ic + Ib.
  01_tau_fit  — tau_total vs 1/IC plot + slope/intercept/Cje/tauB+tauC.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_chartfix import revive_visible_charts  # noqa: E402

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")

IE_MA = {
    "deemb_preext_vce3.5_ib120u": 3.048,
    "deemb_preext_vce3.5_ib160u": 4.578,
    "deemb_preext_vce3.5_ib200u": 6.275,
    "deemb_preext_vce3.5_ib240u": 8.099,
    "deemb_preext_vce3.5_ib80u":  1.68,
    "deemb_preext_vce3.5_ib280u": 8.298,
}
BIAS_FILES = [f"{stem}.s2p" for stem in IE_MA]

_HIDE_CSS = """() => {
    const css = `header, [data-testid="stHeader"] { visibility: hidden !important; }`;
    const s = document.createElement("style");
    s.textContent = css;
    document.head.appendChild(s);
}"""


def run(page, shot):
    from session import settle

    def p(*a):
        print(*a, flush=True)

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
    page.evaluate(_HIDE_CSS)

    exp = page.get_by_text("📊 Interactive Parameter Extraction", exact=True).last
    exp.wait_for(timeout=15000)
    exp.click()
    settle(page, quiet_ms=300)

    tau_hdr = page.get_by_text("Cje / τB+τC / τCC / τE from 1/(2πfT) vs 1/IC fit",
                                exact=False).first
    tau_hdr.wait_for(timeout=12000)
    tau_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    tau_hdr.click()
    p("clicked tau_hdr")
    settle(page, quiet_ms=400)
    revive_visible_charts(page, wait_ms=700)
    page.evaluate(_HIDE_CSS)

    ic_header = page.get_by_text("IC (mA)", exact=True)
    p("IC header count", ic_header.count())
    header_row = ic_header.first.locator(
        "xpath=ancestor::div[contains(@data-testid,'stHorizontalBlock')][1]")
    data_rows = header_row.locator(
        "xpath=following::div[contains(@data-testid,'stHorizontalBlock')]")
    n_rows = min(data_rows.count(), 8)
    p("data row count", n_rows)
    filled_row = None
    filled = 0
    for i in range(n_rows):
        row = data_rows.nth(i)
        row_text = row.inner_text()
        stem_match = next((s for s in IE_MA if s in row_text), None)
        if stem_match is None:
            continue
        row.locator('input').last.fill(str(IE_MA[stem_match]))
        row.locator('input').last.press("Tab")
        p("filled", stem_match)
        filled_row = row
        filled += 1
        if filled == len(IE_MA):
            break
    settle(page, quiet_ms=400)
    p("settled after fills, filled =", filled)

    if filled_row is not None:
        rect = filled_row.evaluate(
            """(el) => { el.scrollIntoView({block:'center', behavior:'instant'});
                         const r = el.getBoundingClientRect();
                         return {x:r.x, y:r.y, width:r.width, height:r.height}; }""")
    else:
        rect = {"x": 400, "y": 300, "width": 200, "height": 40}
    p("row rect", rect)
    settle(page, quiet_ms=300)
    page.evaluate(_HIDE_CSS)
    x0 = max(0, rect["x"] - 400)
    y0 = max(0, rect["y"] - 300)
    clip_ic = {"x": x0, "y": y0,
               "width": min(1160, 1600 - x0),
               "height": min(430, 1000 - y0)}
    page.wait_for_timeout(500)
    page.screenshot(path="/tmp/_warm2.png")
    shot("00_ic_table", clip=clip_ic)
    p("shot 00 done")

    fit_hdr = page.get_by_text("Inputs (defaults from this file's extraction)",
                                exact=False).first
    fit_hdr.wait_for(timeout=10000, state="attached")
    fit_hdr.evaluate("el => el.scrollIntoView({block:'end', behavior:'instant'})")
    settle(page, quiet_ms=300)
    revive_visible_charts(page, wait_ms=700)
    page.evaluate(_HIDE_CSS)
    page.wait_for_timeout(700)
    page.screenshot(path="/tmp/_warm3.png")
    shot("01_tau_fit")
    p("shot 01 done")

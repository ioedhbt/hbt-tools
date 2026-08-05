"""Probe: purge every OTHER Plotly WebGL context on the page, then force a
fresh newPlot on the target chart, to test whether context exhaustion is
why Scattergl traces render blank inside expanders on this page."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1400})
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
    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=400)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=500)

    page.get_by_text("📌 Open & Short Dummy De-embedding", exact=False).first.click()
    settle(page, quiet_ms=500)

    chart_heading = page.get_by_text("Short — Lead Inductances vs Frequency", exact=False).first
    chart_heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=400)

    result = page.evaluate(
        """() => {
            const gds = Array.from(document.querySelectorAll('.js-plotly-plot'));
            const vh = window.innerHeight;
            let target = null, minDist = Infinity;
            for (const gd of gds) {
                const r = gd.getBoundingClientRect();
                if (r.top >= -50 && r.top < vh) {
                    const d = Math.abs(r.top);
                    if (d < minDist) { minDist = d; target = gd; }
                }
            }
            let purged = 0;
            for (const gd of gds) {
                if (gd !== target) {
                    try { window.Plotly.purge(gd); purged++; } catch(e) {}
                }
            }
            let revived = false;
            if (target) {
                try {
                    const data = target.data, layout = target.layout;
                    window.Plotly.purge(target);
                    window.Plotly.newPlot(target, data, layout, {responsive: true});
                    revived = true;
                } catch(e) { revived = String(e); }
            }
            return {total: gds.length, purged, revived, hasTarget: !!target};
        }"""
    )
    print("RESULT", result)
    page.wait_for_timeout(1200)
    shot("00_purged_revived", full_page=True)

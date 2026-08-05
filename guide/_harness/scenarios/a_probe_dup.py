"""Diagnose the apparent 2x DOM duplication of stPlotlyChart / stDataFrame."""
ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    def count(sel):
        return page.locator(sel).count()

    def vcount(sel):
        return page.locator(sel + ":visible").count()

    print("home charts:", count('[data-testid="stPlotlyChart"]'), flush=True)

    page.get_by_role("link", name="RF At a Glance").first.click()
    settle(page)
    print("at-a-glance charts total:", count('[data-testid="stPlotlyChart"]'),
          "visible:", vcount('[data-testid="stPlotlyChart"]'), flush=True)

    # Inspect the ancestry of each stPlotlyChart to see if there are two
    # separate top-level app roots.
    info = page.evaluate("""
        () => {
            const els = Array.from(document.querySelectorAll('[data-testid="stPlotlyChart"]'));
            return els.map(e => {
                let vis = !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
                // Find nearest ancestor with an id, to distinguish app roots.
                let anc = e;
                let path = [];
                while (anc && path.length < 8) {
                    if (anc.id) path.push('#'+anc.id);
                    anc = anc.parentElement;
                }
                return {visible: vis, w: e.offsetWidth, h: e.offsetHeight, ids: path};
            });
        }
    """)
    for i, item in enumerate(info):
        print(i, item, flush=True)

    shot("00", full_page=True)

    import glob
    FILES = sorted(glob.glob("/tmp/app/guide/_data/rf/raw/*.s2p"))[1:3]
    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)
    print("after upload charts total:", count('[data-testid="stPlotlyChart"]'),
          "visible:", vcount('[data-testid="stPlotlyChart"]'), flush=True)

    info2 = page.evaluate("""
        () => {
            const els = Array.from(document.querySelectorAll('[data-testid="stPlotlyChart"]'));
            return els.map(e => {
                let vis = !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
                let anc = e;
                let path = [];
                while (anc && path.length < 8) {
                    if (anc.id) path.push('#'+anc.id);
                    anc = anc.parentElement;
                }
                return {visible: vis, w: e.offsetWidth, h: e.offsetHeight, ids: path};
            });
        }
    """)
    for i, item in enumerate(info2):
        print("post-upload", i, item, flush=True)

    charts = page.locator('[data-testid="stPlotlyChart"]')
    for idx in (0, 1, 8, 9):
        try:
            charts.nth(idx).screenshot(path=f"/tmp/app/guide/_shots/a_probe_dup/chart_{idx}.png",
                                        timeout=3000)
            print(f"chart {idx} shot ok", flush=True)
        except Exception as e:
            print(f"chart {idx} shot failed:", str(e)[:150], flush=True)

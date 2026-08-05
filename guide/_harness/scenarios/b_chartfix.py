"""Shared helper: this sandbox's Chromium exhausts its WebGL context budget
on the SSM extraction page (every parameter card gets its own Plotly
Scattergl figure, and Streamlit mounts collapsed-expander content too, so a
single "Run SSM Extraction" can create 40-100+ contexts). Once the browser
hits its cap, new/older `.js-plotly-plot` canvases silently paint blank —
axes and legend (SVG) show, but no traces (WebGL layer).

Fix: purge every Plotly context NOT currently on screen, then destroy and
recreate (`Plotly.newPlot`) each one that IS on screen. That frees enough of
the budget for the visible chart(s) to actually paint.

Call `revive_visible_charts(page)` right before a `shot(...)` of anything
containing a Plotly chart, after scrolling the target into view.
"""

_REVIVE_JS = """() => {
    const gds = Array.from(document.querySelectorAll('.js-plotly-plot'));
    const vh = window.innerHeight, vw = window.innerWidth;
    const targets = [];
    for (const gd of gds) {
        const r = gd.getBoundingClientRect();
        if (r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw) {
            targets.push(gd);
        }
    }
    for (const gd of gds) {
        if (!targets.includes(gd)) {
            try { window.Plotly.purge(gd); } catch (e) {}
        }
    }
    for (const gd of targets) {
        try {
            const data = gd.data, layout = gd.layout;
            window.Plotly.purge(gd);
            window.Plotly.newPlot(gd, data, layout, {responsive: true});
        } catch (e) {}
    }
    return {total: gds.length, revived: targets.length};
}"""


def revive_visible_charts(page, wait_ms=1200):
    """Purge off-screen Plotly contexts, force-redraw on-screen ones.

    Call after scrolling so the chart(s) you want in the next screenshot are
    inside the viewport.  Returns {"total": N, "revived": M}.
    """
    result = page.evaluate(_REVIVE_JS)
    page.wait_for_timeout(wait_ms)
    return result

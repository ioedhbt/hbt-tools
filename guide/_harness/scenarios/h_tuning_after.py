"""tuning.md — Auto Tuning section, "after" state: residual row + Smith +
fT/fmax once Full Auto Tune has run and its best row has been committed to
Fine-tune via "Use best values".

Same upload as h_tuning_before.py (raw demo file, 18.06% starting point).
Scope is trimmed to Rbi/Rbc right before "Evaluate with CPU" so the
coarse-to-fine search converges inside the capture budget -- the untrimmed
default scope is what tuning_scope.png already shows (see g_tuning_auto.py).

Shots:
  02_after_charts — hbt_resrow metric strip + Smith overlay + fT/fmax card,
                     after Stop -> Use best values.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf/raw")


def hide_badge(page):
    page.evaluate(
        "() => {"
        "let s = document.getElementById('_hide_badge');"
        "if (!s) { s = document.createElement('style'); s.id = '_hide_badge';"
        "document.head.appendChild(s); }"
        "s.textContent = 'div.st-key-lang_toggle{display:none !important;}';"
        "}"
    )


def run(page, shot):
    import sys
    import time
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent))
    from session import settle
    from b_chartfix import revive_visible_charts

    t0 = time.time()
    def mark(label):
        print(f"T+{time.time() - t0:5.1f}s  {label}")

    page.set_default_timeout(8000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page, quiet_ms=500)
    hide_badge(page)
    mark("nav done")

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page, quiet_ms=500)
    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=700)
    mark("upload done")

    at = page.get_by_text("Auto Tuning for Minimum Residuals", exact=False).first
    at.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    at.click()
    settle(page, quiet_ms=400)
    hide_badge(page)
    mark("auto-tuning expander open")

    # Trim scope to Rbi/Rbc so the coarse-to-fine search converges inside
    # the capture budget (see g_tuning_auto.py -- same trim, same reasoning:
    # each selected param is a BaseWeb tag with its own delete icon, click
    # that instead of guessing at backspace-repeat-count).
    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(
        has_text="Parameters to fit").first
    keep = {"Rbi (Ω)", "Rbc (kΩ)"}
    all_tags = ["Cbex (fF)", "Cbcx (fF)", "Rbi (Ω)", "Rbe (Ω)", "Cbe (fF)",
                "Rbc (kΩ)", "Cbc (fF)", "α₀", "τB (ps)", "τC (ps)",
                "Rb (Ω)", "Rc (Ω)", "Re (Ω)"]
    for label in all_tags:
        if label in keep:
            continue
        tag = wrap.locator('span[data-baseweb="tag"]').filter(has_text=label).first
        if tag.count() == 0:
            continue
        icon = tag.locator('svg[title="Delete"]')
        icon.click()
        page.wait_for_timeout(200)
    mark("scope trimmed")

    btn = page.get_by_role("button", name="Evaluate with CPU", exact=False).first
    btn.click()
    mark("evaluate clicked")
    try:
        page.locator('[data-testid="stStatusWidget"]').wait_for(
            state="visible", timeout=3000)
    except Exception:
        pass
    settle(page, quiet_ms=500, timeout_ms=20_000)
    hide_badge(page)
    mark("post-evaluate settle done")

    best = page.get_by_text("Best so far", exact=False).first
    best.wait_for(state="visible", timeout=10_000)
    mark("Best so far visible")

    # Let it run a couple of cycles so Stop doesn't catch cycle 0, then stop
    # and keep the best row -- this is the "one-click run, then commit"
    # sequence a reader actually performs.
    page.wait_for_timeout(2500)
    stop = page.get_by_role("button", name="Stop", exact=False).first
    stop.click()
    mark("stop clicked")
    settle(page, quiet_ms=600, timeout_ms=20_000)
    hide_badge(page)
    mark("post-stop settle done")

    use_best = page.get_by_role("button", name="Use best values", exact=False).first
    use_best.wait_for(state="visible", timeout=10_000)
    use_best.click()
    mark("use best values clicked")
    settle(page, quiet_ms=600, timeout_ms=20_000)
    hide_badge(page)
    mark("post-use-best settle done")

    # Tall viewport + scrollIntoView(block:'center'): centering (rather than
    # 'start') keeps the resrow well clear of the sticky engagement boundary
    # (top: 2.9rem) in *either* direction -- unlike 'start', it doesn't
    # depend on how much scroll-up room happens to exist above the row at
    # this point in the run (Streamlit reruns from the Stop/Use-best-values
    # clicks can leave scrollTop small, which starves a 'start'+overshoot
    # trick of room to reveal content above). 1700px leaves ~850px on each
    # side of a centered resrow -- plenty for the ~650px Smith+fT/fmax block
    # below it.
    page.set_viewport_size({"width": 1600, "height": 1700})
    settle(page, quiet_ms=300)

    resrow = page.locator('div[class*="st-key-hbt_resrow_"]').first
    resrow.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    revive_visible_charts(page)
    # Move the mouse off the Smith chart so Plotly's hover modebar (camera /
    # zoom icons) doesn't float over the chart title in the capture.
    page.mouse.move(50, 50)
    page.wait_for_timeout(300)
    rb = resrow.bounding_box()
    print("RESROW BOX", rb)
    shot("02_after_charts", clip={"x": 370, "y": max(0, rb["y"] - 8),
                                    "width": 1500, "height": 660})
    mark("02_after_charts shot")
    print("DONE")

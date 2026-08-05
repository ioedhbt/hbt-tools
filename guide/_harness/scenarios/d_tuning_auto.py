"""tuning.md — Auto Tuning: default parameter scope (Cpxx/Lx unselected) and
the residual before/after "Evaluate with CPU".

Shots:
  00_scope            — Full Auto Tune card: "Parameters to fit" multiselect,
                         default scope, before running.
  01_before_residual  — sticky residual row before running (the "before").
  02_after_results    — live "Best so far" line + Stop button a couple of
                         cycles into Evaluate with CPU (scope trimmed to
                         Rbi/Rbc right before the click, purely to keep the
                         search inside the capture budget -- Full Auto Tune
                         does not block, it refines until you press Stop or
                         it hits the step floor).
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/deembed_these/deembedded/4x10_left")


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
    from session import settle
    import time
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
    mark("auto-tuning clicked")
    settle(page, quiet_ms=400)
    hide_badge(page)
    mark("auto-tuning expander settled")

    resrow = page.locator('div[class*="st-key-hbt_resrow_"]').first
    rb = resrow.bounding_box()
    print("RESROW BOX", rb)
    shot("01_before_residual", clip={"x": 370, "y": max(0, rb["y"] - 10),
                                      "width": 1550, "height": 140})
    mark("01_before_residual shot")

    scope = page.get_by_text("Full Auto Tune", exact=False).first
    scope.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    sb = scope.bounding_box()
    print("SCOPE BOX", sb)
    shot("00_scope", clip={"x": 370, "y": max(0, sb["y"] - 10),
                           "width": 1550, "height": 260})
    mark("00_scope shot")

    def tag_count():
        return page.evaluate("""() => {
            const wrap = [...document.querySelectorAll('div[data-testid="stMultiSelect"]')]
                .find(w => w.innerText.includes('Parameters to fit'));
            return wrap ? wrap.querySelectorAll('span[data-baseweb="tag"]').length : -1;
        }""")

    # Trim the scope to 2 parameters so the coarse-to-fine search converges
    # inside the capture budget. 00_scope above already shows the real
    # default (untrimmed) scope. Each selected param renders as a BaseWeb
    # tag with its own delete icon (<span data-baseweb="tag" aria-label="X,
    # close by backspace"><svg title="Delete">) -- click that icon directly
    # instead of guessing at backspace-repeat-count (backspace only edits
    # one character of the last tag's label per press, it does not reliably
    # remove a whole tag). A short wait between clicks avoids losing a
    # removal to an in-flight rerun.
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
    mark(f"scope trimmed to {tag_count()} tags")

    btn = page.get_by_role("button", name="Evaluate with CPU", exact=False).first
    btn.click()
    mark("evaluate clicked")
    # settle() can race the click: the busy indicator sometimes takes a beat
    # to appear, so a quiet DOM right after click looks "ready" before the
    # real computation has even started.  Give it a moment to go busy first.
    try:
        page.locator('[data-testid="stStatusWidget"]').wait_for(
            state="visible", timeout=3000)
    except Exception:
        pass
    mark("status widget appeared (or 3s timeout)")
    settle(page, quiet_ms=500, timeout_ms=20_000)
    hide_badge(page)
    mark("post-evaluate settle done")

    # Full Auto Tune keeps refining until you hit the step floor or press
    # Stop -- it does not block. Right after "Evaluate with CPU" the button
    # becomes "Stop" and a live "Best so far" line updates every cycle. That
    # live state (not a final "stopped" summary) is what a one-click run
    # actually looks like, so that's what we capture.
    best = page.get_by_text("Best so far", exact=False).first
    best.wait_for(state="visible", timeout=10_000)
    best.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    bb = best.bounding_box()
    print("BEST BOX", bb)
    shot("02_after_results", clip={"x": 370, "y": max(0, bb["y"] - 90),
                                    "width": 1550, "height": 260})
    mark("02_after_results shot")
    print("DONE")

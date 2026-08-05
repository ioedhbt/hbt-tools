"""tuning.md — Auto Tuning walkthrough: run Full Auto Tune with Rbc removed
from "Parameters to fit" (pinned at its extracted 58.7816 kOhm) and read back
the residual actually reached, for the one-sentence result callout.

No image asset required here (tuning_scope_no_rbc.png already shows the
removal) -- this just measures the number quoted in prose, straight from the
"Best so far" line's text content, printed to stdout/_log.txt.

Scope is trimmed to Rbi only (Rbc already excluded) so the search converges
inside the capture budget, same reasoning as g_tuning_auto.py / h_tuning_after.py.
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
    import time
    from session import settle

    t0 = time.time()
    def mark(label):
        print(f"T+{time.time() - t0:5.1f}s  {label}")

    page.set_default_timeout(8000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page, quiet_ms=500)
    hide_badge(page)

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

    # Keep only Rbi -- Rbc is excluded (pinned) per the walkthrough, and
    # trimming everything else keeps the coarse-to-fine search inside the
    # capture budget (same trick as g_tuning_auto.py / h_tuning_after.py).
    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(
        has_text="Parameters to fit").first
    keep = {"Rbi (Ω)"}
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
    mark(f"scope trimmed (Rbc pinned, kept {keep})")

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

    # Let a couple of cycles run, then stop -- but a single-parameter scope
    # can hit its step floor and finish on its own before we get to click,
    # reverting the button away from "Stop" first (it prints its own
    # "Full Auto Tune done/stopped ... best Total = X.XX%" line in that
    # case). Either outcome leaves a committed result, so tolerate a
    # missing Stop button instead of failing the run.
    page.wait_for_timeout(5000)
    try:
        stop = page.get_by_role("button", name="Stop", exact=False).first
        stop.click(timeout=3000)
        mark("stop clicked")
    except Exception as e:
        mark(f"stop button not clicked ({e.__class__.__name__}) -- "
             f"assuming the search already finished on its own")
    settle(page, quiet_ms=600, timeout_ms=20_000)
    hide_badge(page)
    mark("post-stop settle done")

    # Pull whatever residual text is on screen: either the driver's own
    # "Full Auto Tune done/stopped ... best Total = X.XX%" completion line
    # (natural convergence) or the live/committed "Best so far" /
    # "Best residual" summary block (we stopped it ourselves).
    found = False
    for pat in ("Full Auto Tune done", "Full Auto Tune stopped",
                "Best so far", "Best residual"):
        loc = page.get_by_text(pat, exact=False).first
        if loc.count() == 0:
            continue
        try:
            txt = loc.evaluate(
                "el => (el.closest('div[data-testid=\"stMarkdown\"]') "
                "|| el.closest('div[data-testid=\"stAlert\"]') || el).innerText",
                timeout=3000)
        except Exception as e:
            print(f"  ({pat!r} evaluate failed: {e.__class__.__name__})")
            continue
        print(f"MATCHED {pat!r}:")
        print(txt)
        found = True
    if not found:
        print("No residual text found by any selector.")

    rb = page.locator('div[class*="st-key-hbt_resrow_"]').first.bounding_box()
    print("RESROW BOX", rb)
    mark("read done")
    print("DONE")

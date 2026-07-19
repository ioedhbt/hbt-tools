"""
tools/ui_theme.py — single home for ALL app-wide CSS.

Selectors on Streamlit internals (data-testid, .st-key-*) are version-sensitive
— Streamlit is pinned in requirements.txt; re-verify selectors after upgrades.

Public API
----------
inject_css()
    Emit one <style> block via st.markdown.  Call once, right after
    st.set_page_config, before any other st.* call that renders visible content.
render_ram_badge()
    Compact RAM-usage bar (server-side, cgroup-aware), rendered inside the
    fixed top-right lang_toggle container, left of the language toggle.
    Call inside ``with st.container(key="lang_toggle"):``.  Renders nothing
    when usage can't be determined.
"""

from pathlib import Path

import streamlit as st

from tools.SSM.helpers import mem_budget


# Streamlit Community Cloud mounts the checkout under /mount/src — same
# detection rule as tools/SSM/helpers/fit_cache.py.  Used to move the
# language-toggle CSS below the Cloud header (see inject_css()).
try:
    _IS_STREAMLIT_CLOUD = Path("/mount/src").exists()
except OSError:
    _IS_STREAMLIT_CLOUD = False


def inject_css() -> None:
    """Inject all app-wide CSS in a single st.markdown call."""
    css = """
        <style>

        /* ── Secondary button fill ────────────────────────────────────────────
           Streamlit's default secondary buttons are white-with-border, which
           users mistake for labels.  Give them a light-gray fill.  Primary
           buttons (theme blue) and segmented-control chips are deliberately
           excluded.  The theme is locked to light mode in .streamlit/config.toml
           so fixed hex grays are safe.
           Keep in sync with: tools/ebeam_calculator.py (standalone copy)
                              tools/SSM/helpers/chart_export.py (iframe copy-button)
        */
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-secondaryFormSubmit"] {
            background-color: #E9EDF3;
        }
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
            background-color: #DDE3EB;
        }
        button[data-testid="stBaseButton-secondary"]:active,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:active {
            background-color: #D1D8E2;
        }
        button[data-testid="stBaseButton-secondary"]:disabled,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:disabled {
            background-color: #F1F3F7;
        }

        /* ── .hbt-help — small circled "?" hint icon ─────────────────────────
           Usage: <span class="hbt-help" title="Tooltip text here">?</span>
           Shared contract with other agents/modules — do not rename this class.
        */
        .hbt-help {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.05em;
            height: 1.05em;
            margin-left: 0.35em;
            border: 1px solid #94A3B8;
            border-radius: 50%;
            color: #64748B;
            font-size: 0.72em;
            font-weight: 700;
            line-height: 1;
            cursor: help;
            vertical-align: 15%;
            user-select: none;
        }
        .hbt-help:hover {
            color: #1F2933;
            border-color: #64748B;
        }

        /* ── Status chip classes ─────────────────────────────────────────────
           Usage:
             <span class="hbt-chip">Neutral / pending</span>
             <span class="hbt-chip-ok">Success / ready</span>
             <span class="hbt-chip-file">Measured-device filename pill</span>
             <span class="hbt-chip-cache">Cached-fit pill</span>
             <span class="hbt-chip-model">Model-name pill</span>
           Shared contract with other agents/modules — do not rename these classes.
        */
        .hbt-chip,
        .hbt-chip-ok,
        .hbt-chip-file,
        .hbt-chip-cache,
        .hbt-chip-model {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            color: #37474F;
            background: #F1F5F9;
        }
        .hbt-chip-ok {
            background: #E7F5EC;
            color: #1B5E20;
        }
        .hbt-chip-file {
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
        }
        .hbt-chip-cache {
            background: #FFFBEB;
            color: #B45309;
            border: 1px solid #FDE68A;
        }
        .hbt-chip-model {
            background: #F5F3FF;
            color: #6D28D9;
            border: 1px solid #DDD6FE;
        }

        /* ── Density: reduce main block top padding ───────────────────────────
           Keeps the page title from being buried behind the header bar while
           reclaiming whitespace above the first widget.  Raise toward 2.5rem
           if the page title is visually clipped under the header bar.
        */
        div.block-container {
            padding-top: 1.8rem;
        }

        /* ── Expander polish ─────────────────────────────────────────────────
           Bold summary text and a hover tint to clarify the target is clickable.
        */
        div[data-testid="stExpander"] summary:hover {
            background: #F1F5F9;
        }
        div[data-testid="stExpander"] summary p {
            font-weight: 600;
        }

        /* ── Visible keyboard focus ───────────────────────────────────────────
           Restore an explicit focus ring for accessibility; Streamlit strips
           the browser default in several places.
        */
        button:focus-visible,
        a:focus-visible,
        [role="radio"]:focus-visible,
        [role="checkbox"]:focus-visible {
            outline: 2px solid #2563EB;
            outline-offset: 2px;
        }

        /* ── Language-toggle pinning ─────────────────────────────────────────
           The segmented radio rendered inside st.container(key="lang_toggle")
           gets Streamlit's automatic class st-key-lang_toggle.  Position it
           fixed top-right so it floats above content and stops consuming
           vertical space in the main flow.
        */
        div.st-key-lang_toggle {
            position: fixed;
            top: 1.1rem;
            right: 1.2rem;
            z-index: 999;
            width: auto;
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 0.6rem;
        }

        /* The keyed container div IS the stVerticalBlock, so the row-flex
           props live on the base rule above.  Nested blocks (the RAM-badge
           fragment wraps its content in its own stVerticalBlock) stay
           column; children shrink to content so the fixed container
           shrink-wraps around badge + toggle. */
        div.st-key-lang_toggle > div {
            width: auto;
        }

        /* ── Sticky residual metric strip ────────────────────────────────────
           render_smith_with_ftfmax renders its residual metrics inside
           st.container(key="hbt_resrow_<model_short>").  Pin it while the
           tuning expanders scroll; solid background so content slides under.
           top offset clears Streamlit's fixed header (toolbarMode=minimal).

           position:sticky is applied to the auto-generated stLayoutWrapper
           ANCESTOR (via :has()), not to the keyed div itself. Streamlit
           wraps every st.container(key=...) in its own stLayoutWrapper that
           is auto-sized to exactly fit that one container — per the CSS
           Flexbox spec a flex item's containing block is its flex
           container, so sticky applied to the inner div is constrained by
           that wrapper's box, which has ~0px of extra height to move
           within and unsticks again within about a pixel of scroll.
           Promoting the sticky rule to stLayoutWrapper works because that
           wrapper is itself a flex item of the page-spanning outer
           stVerticalBlock, giving it a tall containing block to stick
           within. Verified in live Chromium (148) via Playwright:
           inner-targeted rule unstuck after ~1px of scroll past engagement;
           wrapper-targeted rule stayed pinned through max scroll.
           Requires :has() (Chromium 105+ / Safari 15.4+ / Firefox 121+).
        */
        div[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-hbt_resrow_"]) {
            position: sticky;
            top: 2.9rem;
            z-index: 50;
            background: #FFFFFF;
            padding: 0.15rem 0 0.3rem 0;
            border-bottom: 1px solid #E2E8F0;
        }
        /* Tag-agnostic attribute selectors: verified live that stMetricLabel
           renders on a <label> (not a <div>) — a div-qualified selector
           silently never matches it.  stMetricValue/stMetricDelta are <div>s
           today but left tag-agnostic too so a future Streamlit markup
           change can't quietly kill these rules again. */
        div[class*="st-key-hbt_resrow_"] [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        div[class*="st-key-hbt_resrow_"] [data-testid="stMetricLabel"] p {
            font-size: 0.72rem;
        }
        div[class*="st-key-hbt_resrow_"] [data-testid="stMetricDelta"] {
            font-size: 0.8rem;
        }

        /* ── Action-color system ─────────────────────────────────────────────
           Wrap idiom: st.container(key="hbt_amber_*" / "hbt_danger_*") around
           a button; "hbt_exp_tune_*" / "hbt_exp_edit_*" / "hbt_exp_view_*"
           around an expander.  Substring attribute selectors tolerate
           sanitized-filename key suffixes.  Amber = apply-a-result-into-the-
           form actions; red outline = destructive; expander left borders
           group the fit page's clusters: blue tuning, amber editing, gray
           read-only views.  These outrank the generic gray secondary-button
           fill above by selector specificity. */
        div[class*="st-key-hbt_amber_"] button {
            background-color: #D97706;
            border-color: #D97706;
            color: #FFFFFF;
        }
        div[class*="st-key-hbt_amber_"] button:hover {
            background-color: #B45309;
            border-color: #B45309;
            color: #FFFFFF;
        }
        div[class*="st-key-hbt_amber_"] button:active {
            background-color: #92400E;
        }
        div[class*="st-key-hbt_danger_"] button {
            background-color: #FFFFFF;
            border: 1px solid #DC2626;
            color: #B91C1C;
        }
        div[class*="st-key-hbt_danger_"] button:hover {
            background-color: #FEF2F2;
            border-color: #B91C1C;
            color: #B91C1C;
        }
        div[class*="st-key-hbt_danger_"] button:active {
            background-color: #FEE2E2;
        }
        div[class*="st-key-hbt_exp_tune_"] div[data-testid="stExpander"] {
            border-left: 3px solid #2563EB;
        }
        div[class*="st-key-hbt_exp_edit_"] div[data-testid="stExpander"] {
            border-left: 3px solid #D97706;
        }
        div[class*="st-key-hbt_exp_view_"] div[data-testid="stExpander"] {
            border-left: 3px solid #CBD5E1;
        }

        </style>
        """
    if _IS_STREAMLIT_CLOUD:
        # ── Cloud-only language-toggle offset ───────────────────────────────
        # Streamlit Community Cloud renders its own header + toolbar ("Manage
        # app", share, hamburger) at a very high z-index (~999990) and opaque
        # background, pinned across the same top-right strip the toggle uses
        # locally (top: 0.5rem) — so on Cloud the toolbar simply covers the
        # toggle. Fighting that z-index is a losing game (Cloud reserves the
        # right to raise its own chrome further), so instead of stacking
        # above it we push the toggle DOWN below the header, which is about
        # 2.875rem tall. Locally (toolbarMode=minimal, no Cloud chrome) the
        # base top: 1.1rem position above is fine and stays unchanged.
        # top: 3.25rem still collided with Cloud's fork/GitHub buttons, so
        # it is pushed a further ~1.25rem (about half the toggle's height)
        # down.
        css += """
        <style>
        div.st-key-lang_toggle {
            top: 4.5rem;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ── RAM usage badge (top-right, left of language toggle) ────────────────────
#
# Streamlit >= 1.36 (pinned in requirements.txt) supports st.fragment's
# run_every= kwarg, so the badge body re-renders on its own timer without
# forcing a full-page rerun. If a future downgrade drops that support the
# TypeError falls back to plain per-rerun rendering (still correct, just
# not self-refreshing).
try:
    _ram_fragment_decorator = st.fragment(run_every="10s")
except TypeError:
    _ram_fragment_decorator = st.fragment


@_ram_fragment_decorator
def _render_ram_badge_body() -> None:
    """Auto-refreshing RAM-usage bar, rendered inside the fixed top-right
    lang_toggle container, left of the language toggle. No-ops when
    mem_budget can't determine usage/limit (e.g. neither cgroup accounting
    nor psutil is available)."""
    usage = mem_budget.ram_usage()
    if usage is None:
        return
    used, limit = usage
    if not limit:
        return
    pct = max(0.0, min(100.0, used / limit * 100.0))
    if pct < 60:
        fill_color = "#16a34a"       # green
    elif pct < 85:
        fill_color = "#d97706"       # amber
    else:
        fill_color = "#dc2626"       # red
    st.markdown(
        f"""
        <div style="width: 10rem; margin: 0;">
          <div style="
              width: 100%;
              height: 0.55rem;
              border-radius: 999px;
              background: rgba(128,128,128,.25);
              overflow: hidden;
          ">
            <div style="
                width: {pct:.1f}%;
                height: 100%;
                background: {fill_color};
                border-radius: 999px;
            "></div>
          </div>
          <div style="font-size: 0.72rem; opacity: 0.75; margin-top: 0.2rem;">
            RAM {used/1024**3:.2f} / {limit/1024**3:.2f} GB ({pct:.0f}%)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ram_badge() -> None:
    """Render the RAM-usage bar. Call inside
    ``with st.container(key="lang_toggle"):``, before the language toggle,
    so it renders left of it in the fixed top-right strip.

    Renders nothing when ``mem_budget.ram_usage()`` returns ``None`` (no
    cgroup accounting and no psutil available).
    """
    _render_ram_badge_body()

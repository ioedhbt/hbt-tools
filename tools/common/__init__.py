"""
tools.common — infrastructure shared by every tool group.

Layer:       ui-helper / io  (no domain math lives here)
Imported by: the portal entry point and every page, RF and DC alike
Gotchas:     nothing here may import from tools.rf / tools.dc / tools.process —
             that is the whole point of the package.

Why this package exists
-----------------------
These modules used to live inside the SSM (RF) engine, so the DC pages, the
Gummel analyser and even the portal entry point reached into
``tools.rf.ssm.helpers`` for a radio widget, a clipboard button or a RAM probe.
Tracing a DC bug therefore led straight into the RF small-signal-model tree.
None of the modules here depend on anything RF.

Contents
--------
paths         — REPO_ROOT / EXAMPLES_DIR (one definition, see the module docstring)
i18n          — bilingual registry: TOOLS, GROUP_ORDER, t(), tr()
ui_theme      — all app-wide CSS + the sidebar RAM badge
diagrams      — the "ℹ️ How it works" pipeline drawer
widgets       — segmented_radio, quickset_buttons, dedupe_upload_names, …
chart_export  — Excel/TSV export, plotly_with_dl, unique_sheet_name, metric_card
mem_budget    — cgroup-aware available-RAM probe
handoff       — cross-page session_state bus (RF + DC) and the page-path registry
"""
from __future__ import annotations

from .paths import REPO_ROOT, EXAMPLES_DIR

from .widgets import (
    quickset_buttons,
    apply_pending,
    info_icon_html,
    segmented_radio,
    dedupe_upload_names,
)

from .chart_export import (
    fig_to_excel_bytes,
    bode_excel_bytes,
    plotly_with_dl,
    build_excel,
    metric_card,
    fig_to_tsv,
    frames_to_tsv,
    xlsx_bytes_to_tsv,
    copy_button,
    unique_sheet_name,
    EXCEL_MIME,
)

__all__ = [
    "REPO_ROOT", "EXAMPLES_DIR",
    "quickset_buttons", "apply_pending", "info_icon_html", "segmented_radio",
    "dedupe_upload_names",
    "fig_to_excel_bytes", "bode_excel_bytes", "plotly_with_dl", "build_excel",
    "metric_card", "fig_to_tsv", "frames_to_tsv", "xlsx_bytes_to_tsv",
    "copy_button", "unique_sheet_name", "EXCEL_MIME",
]

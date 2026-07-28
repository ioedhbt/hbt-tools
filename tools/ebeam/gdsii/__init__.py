"""
tools/ebeam/gdsii — GDSII loading pipeline for the EBL calculator.

Split out of the former monolithic ``tools/ebeam/calculator.py`` (mechanical
refactor, no behaviour change): ``limits`` (RAM-budget sizing), ``parser``
(single-pass streaming GDSII decoder + in-memory layer objects) and
``stream`` (bounded-memory scan/window path for masks too big to fully
parse). See ``tools/ebeam/AGENTS.md`` for the self-containment rule this
whole page follows.
"""

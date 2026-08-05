"""
gdsii/limits.py — RAM-budget sizing for the GDSII parser/streamer.

Split out of ``tools/process/ebeam/calculator.py`` (mechanical refactor: moved
as-is, no behaviour change, no numbers changed).

Self-contained like the rest of the EBL calculator (see
``tools/process/ebeam/AGENTS.md``): this module keeps its own inline ``_is_zh()`` /
``tr()`` copy instead of importing them back from ``calculator.py``.  That
is not optional here — Streamlit always executes the page script as
``__main__`` (see ``streamlit.runtime.scriptrunner.script_runner``), never
under the dotted name ``tools.process.ebeam.calculator``, so an import of
``tools.process.ebeam.calculator`` from a module *it* imports would re-run the
whole page a second time under that dotted name (a second
``st.set_page_config()`` call, a second render). ``calculator.py`` keeps
its own separate copy of the same two functions; the other ``tools/process/ebeam``
submodules that need ``tr()`` import it from here instead (submodules
importing each other is fine — only importing back up to ``calculator.py``
is not).
"""
from __future__ import annotations

from typing import NamedTuple

import streamlit as st


def _is_zh() -> bool:
    """Whether the shared portal UI language is 中文 (Traditional Chinese)."""
    return st.session_state.get("ui_lang") == "中文"


def tr(en: str, zh: str) -> str:
    """Return ``zh`` when the portal UI language is 中文, else ``en``.

    Inline mirror of ``tools.i18n.tr()`` / ``tools.process.ebeam.calculator.tr()`` —
    kept local (no import of ``calculator.py`` or the repo-wide i18n
    module) so this file stays fully self-contained; see the module
    docstring above for why it cannot simply import ``calculator.py``'s
    copy. Reads the same ``st.session_state["ui_lang"]`` key the portal's
    language toggle writes, so this page follows it automatically when
    embedded, and defaults to English when run standalone (no toggle
    present).
    """
    return zh if _is_zh() else en


# ─── Memory budgets (sized from the RAM this machine actually has) ──────────
# A pathological GDS must fail with a clear message instead of OOM-killing
# the app — but "pathological" depends entirely on the host. The same file
# that would kill a 3 GB Streamlit Cloud container is unremarkable on a
# 32 GB workstation, so the budgets below are computed at parse time from
# the free RAM probed right then, not hard-coded.
#
# Geometry is stored as flat float64 arrays (~16 B per vertex / per
# placement row, no per-polygon Python objects), but the *peak* during
# parsing runs several times the stored size: raw chunks, the concatenated
# copy and the upload buffer are all live at once. Measured with
# gds/_profile_gds_limits.py (peak RSS above the import baseline, second
# upload of the same size — the worst moment):
#
#   distinct geometry   9.3 M vertices (100 MB file)  → 0.93 GB
#                      18.6 M vertices (200 MB file)  → 1.84 GB
#   repeated cells      8.7 M placements (250 MB)     → 1.25 GB
#                      17.5 M placements (500 MB)     → 1.45 GB
#
# Subtracting the upload buffer (~2× the file, held twice while Streamlit
# receives it) leaves ~90 MB of peak per million vertices and per million
# placements — the coefficients used below.
_MB_PER_M_VERTICES = 90.0     # peak MB per 1 M polygon vertices
_MB_PER_M_ROWS = 90.0         # peak MB per 1 M reference placements
_UPLOAD_BUFFER_FACTOR = 2.0   # file bytes held live while parsing

# How much of the machine one mask may claim. Two regimes, because the two
# hosts fail differently:
#
#   * Inside a container (a cgroup limit exists) going over means SIGKILL —
#     no swap, no `except MemoryError`, no warning. So the budget is bound
#     strictly to what is free inside the limit right now.
#   * On a PC there is no such cliff: over-committing means paging, which
#     is slow but survivable, and a failed allocation raises a catchable
#     MemoryError. Binding to instantaneous "available" there would make
#     the app flaky — the same mask would load in the morning and be
#     refused in the afternoon because a browser grew. So a share of TOTAL
#     RAM acts as a floor under the available-memory figure.
_RAM_CLAIM_FRACTION = 0.75    # of free RAM (both regimes)
_RAM_TOTAL_SHARE = 0.35       # of total RAM — the PC floor
_RAM_TOTAL_CEILING = 0.60     # of total RAM — never claim more than this
_MIN_BUDGET_MB = 600.0
_MAX_BUDGET_MB = 24_000.0     # stops a 512 GB server setting budgets so
                              # large a bad file churns for minutes

# Fallback budget when nothing about the host can be probed — the measured
# safe value for Streamlit Community Cloud (3 GB container, ~1 GB resident
# when idle).
_FALLBACK_BUDGET_MB = 1_500.0

# Granularity at which a changed RAM budget invalidates the parse cache.
# The budget itself moves continuously (it is read from free memory), so
# keying the cache on it directly would miss on every rerun.
_BUDGET_BUCKET_MB = 256.0

# Clamps on the derived counts. The floors are what the smallest sensible
# host must still accept; the ceilings bound parse time, not memory.
_MIN_VERTICES, _MAX_VERTICES_CAP = 2_000_000, 200_000_000
_MIN_ROWS, _MAX_ROWS_CAP = 4_000_000, 400_000_000

# Above this polygon count a layer is too dense to draw individually in
# the browser (Plotly chokes well before this) or to clip per-grid with
# gdstk. The viewer and workflow modes fall back to a bounding-box outline
# + a vectorized area estimate instead. Unlike the budgets above this is a
# rendering limit, not a memory one, so it does not scale with RAM.
_POLY_LIMIT = 50_000

# cgroup accounting files — the only way to see a container's real limit
# (psutil reports the *host's* memory, which on Streamlit Cloud is far
# more than the container may use, so sizing off it alone gets the process
# SIGKILLed before any `except MemoryError` can run). Inline copies of the
# probes in tools/common/mem_budget.py; this file imports no repo
# modules so it can run standalone (see the module docstring).
_CGROUP_FILES = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),        # v2
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes",                        # v1
     "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
_CGROUP_UNLIMITED = 1 << 60   # v1 sentinel for "no limit", not a real value


def _read_int_file(path: str):
    """Parse a cgroup accounting file; None on any failure or "max"."""
    try:
        with open(path) as fh:
            text = fh.read().strip()
    except Exception:
        return None
    if text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value >= _CGROUP_UNLIMITED else value


def _cgroup_free_mb():
    """MB left inside this process's cgroup memory limit, or None when
    there is no limit (i.e. not in a constrained container)."""
    for limit_path, used_path in _CGROUP_FILES:
        limit = _read_int_file(limit_path)
        used = _read_int_file(used_path)
        if limit is not None and used is not None:
            return max(0, limit - used) / 1e6
    return None


def _free_ram_mb():
    """``(free_mb, total_mb, source)`` for this host.

    ``free_mb`` is what can still be allocated; ``total_mb`` is the
    ceiling this process lives under (the cgroup limit in a container,
    otherwise physical RAM). ``source`` is ``"container"`` when a cgroup
    limit was found — the caller must not over-commit in that case.
    Any field may be ``None`` when it can't be probed.
    """
    cgroup_free = _cgroup_free_mb()
    try:
        import psutil
        vm = psutil.virtual_memory()
        avail_mb, total_mb = vm.available / 1e6, vm.total / 1e6
    except Exception:
        avail_mb = total_mb = None

    if cgroup_free is not None:
        # A container's own accounting beats the host-wide numbers psutil
        # reports (which describe the machine the container runs on).
        return cgroup_free, None, "container"
    if avail_mb is None:
        return None, None, "unknown"
    return avail_mb, total_mb, "system"


def _mask_budget_mb():
    """``(budget_mb, note, strict)`` — peak RAM one mask may use, now.

    ``strict`` marks the container regime, where the budget is a hard
    ceiling (exceeding it is a SIGKILL) and callers must not apply
    comfort floors on top of it.
    """
    free_mb, total_mb, source = _free_ram_mb()
    if free_mb is None:
        return _FALLBACK_BUDGET_MB, tr(
            "could not read this machine's memory — using the safe default",
            "無法讀取本機記憶體資訊 — 使用安全預設值"), True

    budget = free_mb * _RAM_CLAIM_FRACTION
    if source == "container":
        # No floor here: if only 250 MB is free, 250 MB is the truth, and
        # rounding it up to a friendlier number is how you get OOM-killed.
        return budget, tr(
            f"{free_mb / 1024:.1f} GB free in this container",
            f"容器內可用 {free_mb / 1024:.1f} GB"), True

    # Not in a container: paging is the penalty for over-committing, not a
    # kill, so keep a stable floor tied to installed RAM rather than
    # tracking every fluctuation in what's free.
    if total_mb:
        budget = min(max(budget, total_mb * _RAM_TOTAL_SHARE),
                     total_mb * _RAM_TOTAL_CEILING)
    budget = min(_MAX_BUDGET_MB, max(_MIN_BUDGET_MB, budget))
    return budget, tr(
        f"{free_mb / 1024:.1f} GB free of {total_mb / 1024:.0f} GB",
        f"可用 {free_mb / 1024:.1f} GB／共 {total_mb / 1024:.0f} GB"), False


class _Limits(NamedTuple):
    """The three parse budgets, derived per file from the RAM budget."""
    src_verts: int      # polygon vertices stored while parsing
    rows: int           # reference placements after flattening
    verts: int          # vertices when expanding a layer flat
    budget_mb: float    # the RAM budget they came from
    note: str           # human-readable "where that number came from"


def _limits_for(file_mb: float) -> _Limits:
    """Budgets for parsing a ``file_mb`` upload on this host, now.

    The upload buffer is charged first (Streamlit holds the bytes for as
    long as the widget lives, and briefly twice while receiving them);
    what remains is what the geometry may spend. A file big enough to eat
    the whole budget on its own yields the floor budgets, so it will fail
    on a guard with a message rather than by allocating until the kernel
    steps in.
    """
    budget_mb, note, strict = _mask_budget_mb()
    geom_mb = budget_mb - _UPLOAD_BUFFER_FACTOR * file_mb
    verts = min(_MAX_VERTICES_CAP, geom_mb / _MB_PER_M_VERTICES * 1e6)
    rows = min(_MAX_ROWS_CAP, geom_mb / _MB_PER_M_ROWS * 1e6)
    if not strict:
        # On a PC, don't let a momentarily-busy machine shrink the budgets
        # below what any ordinary mask needs — worst case it pages.
        verts = max(_MIN_VERTICES, verts)
        rows = max(_MIN_ROWS, rows)
    return _Limits(int(max(0, verts)), int(max(0, rows)), int(max(0, verts)),
                   budget_mb, note)


# Static fallbacks: the measured-safe values for a 3 GB container, used
# when a caller has no file size to size against (tests, direct calls).
_DEFAULT_LIMITS = _Limits(10_000_000, 20_000_000, 10_000_000,
                          _FALLBACK_BUDGET_MB, "default")


def _rows_budget_msg(limit: int) -> str:
    """User-facing message for "this file places cells too many times".
    Shared by the parser (SREF runs, AREF expansion) and the flattener so
    the wording stays identical wherever the budget trips."""
    return tr(
        f"GDS places cell references more than {limit:,} "
        "times — more than this machine's free memory allows. Expose a "
        "smaller layer, then re-upload.",
        f"GDS 檔案的元件參照放置次數超過 {limit:,} 次 — "
        "超出本機可用記憶體的負荷。請匯出較小的圖層後重新上傳。"
    )


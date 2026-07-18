"""
mem_budget.py — Container/cgroup-aware RAM budget probes.

Streamlit Community Cloud runs each app inside a cgroup-limited container
(today ~2.7 GB), but ``psutil.virtual_memory()`` reports the HOST's memory,
not the cgroup limit — code that sizes chunks off psutil alone can allocate
far more than the container actually has and get SIGKILLed by the cgroup
OOM-killer before any ``except MemoryError`` recovery path ever runs. These
probes read the cgroup v2 / v1 accounting files directly (falling back to
psutil, then a conservative constant) so callers can size work off what's
*actually* available right now.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


_CGROUP_V2_MAX     = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V1_LIMIT   = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CGROUP_V1_USAGE   = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")

# cgroup v1 hosts report a huge sentinel (e.g. 2**63-1, rounded to a page)
# instead of a real limit_in_bytes when the container is unconstrained.
_UNLIMITED_SENTINEL = 1 << 60

# Only matters on hosts where neither cgroup nor psutil could be read —
# conservative so a constrained-but-unprobeable host doesn't over-allocate.
_FALLBACK_AVAILABLE_BYTES = 1 * 1024**3


def _read_int(path: Path) -> Optional[int]:
    """Read + parse a cgroup accounting file. ``None`` on any failure
    (missing file, wrong cgroup version, "max" sentinel, permission error,
    garbage contents) — never raises."""
    try:
        text = path.read_text().strip()
    except Exception:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


# The cgroup limit is fixed for the container's lifetime, so it's cached at
# module level after the first successful (or unsuccessful) probe. Usage
# changes constantly and must NEVER be cached.
_cgroup_limit_cached = False
_cgroup_limit_value: Optional[int] = None


def _cgroup_limit_bytes() -> Optional[int]:
    global _cgroup_limit_cached, _cgroup_limit_value
    if _cgroup_limit_cached:
        return _cgroup_limit_value
    _cgroup_limit_cached = True
    limit = _read_int(_CGROUP_V2_MAX)
    if limit is None:
        limit = _read_int(_CGROUP_V1_LIMIT)
        if limit is not None and limit >= _UNLIMITED_SENTINEL:
            limit = None
    _cgroup_limit_value = limit
    return limit


def _cgroup_used_bytes() -> Optional[int]:
    used = _read_int(_CGROUP_V2_CURRENT)
    if used is None:
        used = _read_int(_CGROUP_V1_USAGE)
    return used


def ram_limit_bytes() -> int | None:
    """Best-known RAM ceiling for this process, in bytes.

    cgroup v2 ``memory.max`` → cgroup v1 ``memory.limit_in_bytes`` (treating
    the "unlimited" sentinel as no limit) → ``psutil.virtual_memory().total``
    → ``None`` if nothing is knowable.
    """
    limit = _cgroup_limit_bytes()
    if limit is not None:
        return limit
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def ram_used_bytes() -> int | None:
    """Bytes currently in use, preferring the cgroup accounting (never
    cached — this changes constantly) and falling back to psutil."""
    used = _cgroup_used_bytes()
    if used is not None:
        return used
    try:
        import psutil
        return int(psutil.virtual_memory().used)
    except Exception:
        return None


def ram_available_bytes() -> int:
    """Best-effort bytes this process can still allocate right now.

    Takes the minimum of (cgroup limit − cgroup used, when both are known)
    and (``psutil.virtual_memory().available``, when psutil is importable) —
    whichever source is more pessimistic wins. Falls back to a conservative
    1 GiB when nothing is knowable (only matters on constrained hosts we
    can't otherwise probe). Never raises.
    """
    candidates = []

    limit = _cgroup_limit_bytes()
    used = _cgroup_used_bytes()
    if limit is not None and used is not None:
        candidates.append(max(0, limit - used))

    try:
        import psutil
        candidates.append(int(psutil.virtual_memory().available))
    except Exception:
        pass

    if candidates:
        return min(candidates)
    return _FALLBACK_AVAILABLE_BYTES


def ram_usage() -> tuple[int, int] | None:
    """``(used_bytes, limit_bytes)`` for UI display, or ``None`` when neither
    the cgroup accounting nor psutil is available. Prefers the cgroup pair
    (accurate inside a Streamlit Cloud container); falls back to psutil's
    host-wide ``(used, total)``.
    """
    limit = _cgroup_limit_bytes()
    used = _cgroup_used_bytes()
    if limit is not None and used is not None:
        return used, limit
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.used), int(vm.total)
    except Exception:
        return None

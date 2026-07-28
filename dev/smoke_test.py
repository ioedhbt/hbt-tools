#!/usr/bin/env python3
"""
smoke_test.py — the repo's regression net.  No test framework required.

Run it after every structural change.  It answers three questions:

  1. Does everything still import?          (catches broken/relative imports)
  2. Do all page paths still resolve?       (catches sidebar/switch_page breakage)
  3. Did any number change?                 (catches silent numerical regressions)

Usage
-----
    python dev/smoke_test.py --baseline    # record current numbers as truth
    python dev/smoke_test.py               # compare against the recorded truth

The baseline lives in ``dev/_smoke_baseline.json`` and is committed, so a
diff on it is a visible, reviewable claim that a number was *meant* to move.

This file deliberately resolves each import against **both** the pre- and
post-restructure module paths (see ``_imp``), so the same test — and the
same baseline — is valid on either side of the move.  Once the restructure
has landed the legacy candidates can be dropped.

Exit code is 0 on pass, 1 on any failure.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_PATH = Path(__file__).resolve().parent / "_smoke_baseline.json"

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

TOL = 1e-9

# Page modules execute Streamlit code top-to-bottom on import, and the EBL
# calculator is the deliberately self-contained standalone app
# (ARCHITECTURE.md §1).  Both are covered by the path check instead.
_SKIP_IMPORT_SUFFIXES = (
    "ebeam.calculator", "ebeam_calculator",
    "dc.b1500a_plot", "dc.hp4155a_plot",
    "tcad.gummel_analyzer",
    "rf.at_a_glance", "rf.simulator", "rf.extraction",
    "data.csv_process", "portal.home",
)

_failures: list[str] = []
_checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if not ok:
        _failures.append(f"{label}{(' — ' + detail) if detail else ''}")


def _imp(*candidates: str):
    """Import the first importable module from ``candidates``.

    Lets one test file span the restructure: pass the new path first, the
    legacy path second.
    """
    last = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last = exc
    raise ImportError(f"none of {candidates} importable: {last}")


# ─────────────────────────────────────────────────────────────────────────────
#  1. Import everything
# ─────────────────────────────────────────────────────────────────────────────

def test_imports() -> None:
    import tools
    for info in pkgutil.walk_packages(tools.__path__, prefix="tools."):
        name = info.name
        if "__pycache__" in name:
            continue
        if any(name.endswith(s) for s in _SKIP_IMPORT_SUFFIXES):
            continue
        try:
            importlib.import_module(name)
            check(f"import {name}", True)
        except Exception as exc:                                  # noqa: BLE001
            check(f"import {name}", False, f"{type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Every page path / switch_page target resolves on disk
# ─────────────────────────────────────────────────────────────────────────────

def test_page_paths() -> None:
    i18n = _imp("tools.common.i18n", "tools.i18n")

    for key, meta in i18n.TOOLS.items():
        check(f"TOOLS[{key}].path", (ROOT / meta["path"]).is_file(),
              f"missing: {meta['path']}")
        g = meta["group"]
        check(f"TOOLS[{key}].group", g in i18n.GROUPS and g in i18n.GROUP_ORDER, g)

    seen_pages = 0
    for modname in ("tools.common.handoff", "tools.rf.ssm.handoff", "tools.dc_handoff"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for const in dir(mod):
            if not const.startswith("PAGE_"):
                continue
            rel = getattr(mod, const)
            seen_pages += 1
            check(f"{modname}.{const}", (ROOT / rel).is_file(), f"missing: {rel}")
    check("handoff PAGE_* constants found", seen_pages >= 4, f"found {seen_pages}")

    entry = (ROOT / "IOED_Tool_Web.py").read_text(encoding="utf-8")
    for m in re.finditer(r'st\.Page\(\s*"([^"]+)"', entry):
        check(f"IOED_Tool_Web st.Page({m.group(1)})",
              (ROOT / m.group(1)).is_file(), f"missing: {m.group(1)}")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Path resolution that a folder move silently breaks
# ─────────────────────────────────────────────────────────────────────────────

def collect_paths() -> dict:
    rk = _imp("tools.rf.ssm.helpers.rust_kernels", "tools.rf.ssm.helpers.rust_kernels")
    fc = _imp("tools.rf.ssm.helpers.fit_cache", "tools.rf.ssm.helpers.fit_cache")
    return {
        "rust_bin_under_repo": str(rk._BIN_DIR).startswith(str(ROOT)),
        "rust_bin_tail": "/".join(Path(rk._BIN_DIR).parts[-3:]),
        "fit_cache_tail": "/".join(Path(fc.cache_path_str()).parts[-2:]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4. Numerical goldens — the part that makes a refactor safe
# ─────────────────────────────────────────────────────────────────────────────

def _sig(arr) -> float:
    """Order-sensitive scalar signature of an array (NaN-tolerant)."""
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        a = np.concatenate([a.real.ravel(), a.imag.ravel()])
    a = a.ravel().astype(float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    w = np.arange(1, a.size + 1, dtype=float)
    return float(np.sum(a * w) / a.size)


FREQ = np.linspace(1e9, 50e9, 21)
_EXAMPLE_CANDIDATES = (
    "examples/ADSsim_measured_ChengT_1p5V_Ib100u.s2p",
    "dummy_data_practice/ADSsim_measured_ChengT_1p5V_Ib100u.s2p",
)

# Fixed pad/lead values for the de-embedding golden — deliberately not the
# defaults, so a sign/ordering change in the peel math actually shows up.
_PAD = {"Cpbe": 12e-15, "Cpce": 9e-15, "Cpbc": 4e-15,
        "Lb": 40e-12, "Lc": 35e-12, "Le": 6e-12,
        "Rpb": 1.5, "Rpc": 2.0, "Rpe": 0.8}


def _example_path() -> Path:
    for rel in _EXAMPLE_CANDIDATES:
        p = ROOT / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"none of {_EXAMPLE_CANDIDATES} exists")


def H_load(api, path):
    return api.load_data(str(path))


def collect_numbers() -> dict:
    api = _imp("tools.rf.ssm.agent_api", "tools.rf.ssm.agent_api")
    H = _imp("tools.rf.ssm.helpers", "tools.rf.ssm.helpers")
    models = _imp("tools.rf.ssm.models", "tools.rf.ssm.models")

    out: dict = {}

    # -- forward simulation of every registered model -------------------------
    for short in sorted(models.REGISTRY):
        out[f"simulate.{short}"] = _sig(api.simulate(short, {}, FREQ,
                                                     backend="numpy"))

    # -- parsing --------------------------------------------------------------
    d = api.load_data(str(_example_path()))
    out["parse.freq"] = _sig(d["freq"])
    out["parse.S"] = _sig(d["S"])
    out["parse.z0"] = float(d["z0"])
    out["parse.deembedded"] = bool(d["meta"]["deembedded"])
    out["parse.deembed_source"] = d["meta"].get("deembed_source")
    out["parse.removed_params"] = sorted(d["meta"]["removed_params"])

    # De-embed detection + per-key freeze policy (agent_api).  Filenames are
    # sniffed, so the negated forms below are the ones that used to be
    # misclassified as de-embedded and have their parasitics pinned to 0.
    _hdr = {"Cpbe": "12.3400 fF", "Cpce": "9.0000 fF", "Cpbc": "4.0000 fF",
            "Lb": "40.0000 pH", "Lc": "35.0000 pH", "Le": "6.0000 pH",
            "Rb": "0.0000 Ω", "Rc": "0.0000 Ω", "Re": "0.0000 Ω"}
    _seed = {"Cpbe": 11e-15, "Cpce": 8e-15, "Cpbc": 3e-15,
             "Lb": 30e-12, "Lc": 25e-12, "Le": 5e-12,
             "Rpb": 3.0, "Rpc": 4.0, "Rpe": 1.0}
    import tempfile
    _fsim = np.linspace(1e9, 50e9, 15)
    _Ssim = np.full((15, 2, 2), 0.1 + 0.05j)
    for fname, title, hdr in (
            ("wafer_deemb.s2p", "De-embedded wafer", _hdr),
            ("raw_not_deembedded.s2p", "Measured raw data", None),
            ("plain_measured.s2p", "Measured raw data", None),
            ("legacy_deemb.s2p", "Measured raw data", None)):
        p = Path(tempfile.mkdtemp()) / fname
        p.write_bytes(H.write_s2p(_fsim, _Ssim, title=title, params=hdr))
        dd = H_load(api, p)
        rr = api.fit(dd, model="T", initial=dict(_seed), maxiter=1)
        out[f"deembed.{fname}"] = {
            "deembedded": bool(dd["meta"]["deembedded"]),
            "source": dd["meta"].get("deembed_source"),
            "pinned": sorted(k for k, v in _seed.items()
                             if float(rr["params"].get(k, v)) == 0.0),
        }

    freq, S, z0 = d["freq"], d["S"], d["z0"]

    # -- conversions ----------------------------------------------------------
    Y = H.s_to_y(S, z0)
    out["s_to_y"] = _sig(Y)
    out["y_to_z"] = _sig(H.y_to_z(Y))
    out["y_to_s_roundtrip"] = _sig(H.y_to_s_batch(Y, z0) - S)

    # -- de-embedding math ----------------------------------------------------
    S_open = H.simulate_open(_PAD, freq, z0)
    S_short = H.simulate_short(_PAD, freq, z0)
    out["simulate_open"] = _sig(S_open)
    out["simulate_short"] = _sig(S_short)

    po, _ = H.step_open((freq, S_open, z0))
    out["step_open"] = {k: round(float(v), 21) for k, v in sorted(po.items())}
    ps, _ = H.step_short((freq, S_short, z0), freq,
                         po["Cpbe"], po["Cpce"], po["Cpbc"])
    out["step_short"] = {k: round(float(v), 18) for k, v in sorted(ps.items())}

    Y_ex1 = H.peel_parasitics(S, freq, z0, {**po, **ps})
    out["peel_parasitics"] = _sig(Y_ex1)

    # -- metrics --------------------------------------------------------------
    h21_db, U_db = H.compute_h21_U(S)
    out["h21_db"] = _sig(h21_db)
    out["U_db"] = _sig(U_db)
    ft, fmax = H.find_ft_fmax(freq * 1e-9, h21_db, U_db)
    out["find_ft"] = None if ft is None else round(float(ft), 9)
    out["find_fmax"] = None if fmax is None else round(float(fmax), 9)

    df = H.compute_metrics(Y, freq)
    out["compute_metrics.cols"] = sorted(map(str, df.columns))
    out["compute_metrics.sig"] = _sig(df.select_dtypes("number").to_numpy())

    # -- analytic extraction (the physics that must not drift) ----------------
    for short in ("T", "pi"):
        try:
            params, _arr = models.REGISTRY[short].extract(Y_ex1, freq, 10)
            out[f"extract.{short}"] = {
                k: round(float(v), 18) for k, v in sorted(params.items())
                if np.isscalar(v) and np.isfinite(v)}
        except Exception as exc:                                  # noqa: BLE001
            out[f"extract.{short}"] = f"ERROR {type(exc).__name__}: {exc}"

    # -- interactive re-extraction -------------------------------------------
    # reextract() is what the "📊 Interactive Parameter Extraction" fields
    # call on every edit.  It re-derives Step 2/3 under user overrides and is
    # a completely separate code path from extract() — it is where τC used to
    # silently revert to the uncorrected [Eq. 31] form, and where a
    # deliberate Cbex = 0 used to be discarded as falsy.  Pinned here because
    # nothing else in this file touches it.
    for short in ("T", "pi"):
        cls = models.REGISTRY[short]
        base_p, base_a = cls.extract(Y_ex1, freq, 10)

        # (a) no overrides — must reproduce extract() exactly
        p0, _ = cls.reextract(Y_ex1, freq, 10, {}, 0, base_a)
        out[f"reextract.{short}.noop_matches_extract"] = all(
            abs(float(p0[k]) - float(base_p[k])) <= 1e-12 * max(1.0, abs(float(base_p[k])))
            for k in base_p if np.isscalar(base_p[k]) and np.isfinite(base_p[k]))

        # (b) alpha0 override — exercises the tauC branch
        if "alpha0" in base_p:
            p1, _ = cls.reextract(Y_ex1, freq, 10,
                                  {"alpha0": float(base_p["alpha0"]) * 0.98}, 4,
                                  base_a)
            out[f"reextract.{short}.alpha0_ov.tauC"] = round(float(p1["tauC"]), 18)
            out[f"reextract.{short}.alpha0_ov.tauB"] = round(float(p1["tauB"]), 18)

        # (c) Cbex = 0 must survive as 0, not be replaced by the recomputed value
        p2, _ = cls.reextract(Y_ex1, freq, 10, {"Cbex": 0.0}, 0, base_a)
        out[f"reextract.{short}.cbex_zero_honoured"] = float(p2["Cbex"]) == 0.0

    # -- residual metric ------------------------------------------------------
    S_mod = api.simulate("T", {}, freq, backend="numpy")
    out["residuals"] = {k: round(float(v), 9)
                        for k, v in api.residuals(S, S_mod).items()}

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────

def compare(actual, expected, prefix: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        for k in sorted(set(actual) | set(expected)):
            if k not in expected:
                check(f"{prefix}.{k}", False, "new key (re-baseline if intended)")
            elif k not in actual:
                check(f"{prefix}.{k}", False, "key disappeared")
            else:
                compare(actual[k], expected[k], f"{prefix}.{k}")
        return
    if isinstance(actual, float) and isinstance(expected, float):
        ok = abs(actual - expected) <= TOL * max(1.0, abs(expected))
        check(prefix, ok, f"{expected!r} -> {actual!r}")
        return
    check(prefix, actual == expected, f"{expected!r} -> {actual!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="record current numbers as the new truth")
    args = ap.parse_args()

    test_imports()
    test_page_paths()

    numbers = collect_numbers()
    paths = collect_paths()

    if args.baseline:
        BASELINE_PATH.write_text(
            json.dumps({"numbers": numbers, "paths": paths},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE_PATH.relative_to(ROOT)}")
    else:
        if not BASELINE_PATH.exists():
            print("no baseline — run with --baseline first", file=sys.stderr)
            return 1
        base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        compare(numbers, base["numbers"], "num")
        compare(paths, base["paths"], "path")

    if _failures:
        print(f"\nFAIL — {len(_failures)} of {_checks} checks:\n")
        for f in _failures:
            print(f"  ✗ {f}")
        return 1
    print(f"PASS — {_checks} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())

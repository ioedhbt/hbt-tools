"""
Memory / time limits sweep for the EBL page's GDS mask viewer.

Answers the deployment question directly: **how big a .gds can Streamlit
Community Cloud take before the container is OOM-killed?**  For each
synthetic mask from ``gds/_gen_test_gds.py`` it runs the app's real
pipeline (``_load_gds_layers`` → per-cell area binning → coverage raster
→ zoom-window expansion) in a *fresh subprocess* and records peak RSS
(``ru_maxrss``, the kernel's own high-water mark) and per-stage wall
time.

Each case is scored against a RAM budget rather than raw RSS, because
the numbers that matter on the host are relative:

    projected host RSS  =  idle app RSS  +  (worker peak − worker baseline)

``--idle-mb`` is what the deployed app already occupies doing nothing
(~1 GB on Community Cloud) and ``--limit-mb`` is the container ceiling
(3 GB), so the usable budget for one upload is the difference.  A case
"passes" when the projection stays under the ceiling **or** when the
parser refuses the file with a friendly ``ValueError`` from one of its
budget guards — being told "too big" is a pass; being OOM-killed is not.

Usage (from repo root) — the second form is the one that set the current
``maxUploadSize`` (1 GB idle + 1.5 GB usable on a 2.5 GB working ceiling):

    .hbttools/bin/python gds/_profile_gds_limits.py
    .hbttools/bin/python gds/_profile_gds_limits.py \\
        --sizes 100,200,250,300,400 --idle-mb 1000 --limit-mb 2500 --reupload

Add ``--json out.json`` to keep the raw numbers, ``--keep`` to keep the
generated masks (handy for opening one in the app by hand).

Two things this sweep learned the hard way, worth keeping in mind when
reading its output:

* ``--reupload`` matters. The first parse of a mask is the cheap case;
  replacing a loaded mask costs several hundred MB more.
* Peak RSS is sticky. Freeing the previous parse does not necessarily
  hand the pages back to the OS (it does not on macOS), so a *drop* in
  peak between two variants is strong evidence while a flat result is
  weak evidence — the allocator can hide a real saving.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
_RSS_SCALE = 1.0 if sys.platform == "darwin" else 1024.0

# What the workflow modes ask of a mask once it is loaded: an 18 × 18 grid
# of 600 µm exposure cells (the page's defaults).
_GRID_N = 18
_CHIP_MM = 0.6


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_SCALE / 1e6


def _cap_address_space(mb: float) -> str:
    """Hard-cap the worker's address space so an over-budget file dies
    with a ``MemoryError`` here instead of an OOM kill in production.

    Only meaningful on Linux — the deployment target.  Darwin ignores
    ``RLIMIT_AS`` (``setrlimit`` raises), so the sweep falls back to
    comparing measured peaks against the budget.  Returns a status
    string for the report.
    """
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (int(mb * 1e6), hard))
        return f"capped at {mb:,.0f} MB"
    except (ValueError, OSError) as e:
        return f"not enforced on {sys.platform} ({e})"


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:                     # psutil is optional here
        return float("nan")


# ─── Worker: one file, one fresh process ────────────────────────────────────

def run_worker(path: str, rlimit_mb: float | None = None,
               reupload: bool = False) -> dict:
    """Load ``path`` exactly the way the Streamlit page does and report
    timings, geometry counts and RSS.  Runs in its own process so the
    peak it reports belongs to this file alone."""
    import io

    out: dict = {"path": path, "bytes": os.path.getsize(path)}
    if rlimit_mb:
        out["rlimit"] = _cap_address_space(rlimit_mb)
    t_imports = time.perf_counter()
    import tools.ebeam.calculator as m       # noqa: E402  (timed on purpose)
    out["t_import_s"] = time.perf_counter() - t_imports
    out["rss_baseline_mb"] = _rss_mb()
    out["peak_baseline_mb"] = _peak_mb()

    # Streamlit's file_uploader keeps the whole upload in RAM for as long
    # as the widget exists, so hold the buffer live across the parse —
    # dropping it here would understate the real peak by the file size.
    t0 = time.perf_counter()
    upload = io.BytesIO(Path(path).read_bytes())
    out["t_read_s"] = time.perf_counter() - t0

    try:
        t0 = time.perf_counter()
        unit_m, cells = m._load_gds_layers(upload)
        out["t_parse_s"] = time.perf_counter() - t0
    except ValueError as e:                 # a budget guard fired — by design
        out["guard"] = str(e)[:160]
        out["rss_peak_mb"] = _peak_mb()
        return out
    except MemoryError as e:
        out["error"] = f"MemoryError: {e}"
        out["rss_peak_mb"] = _peak_mb()
        return out

    out["rss_after_parse_mb"] = _rss_mb()
    scale_mm = unit_m * 1000.0

    # Pick the heaviest layer — the one the user would expose.
    best = None
    layers = []
    for cname, by_layer in cells.items():
        for key, layer in by_layer.items():
            info = {
                "cell": cname, "layer": key[0], "datatype": key[1],
                "kind": type(layer).__name__, "polys": len(layer),
            }
            if isinstance(layer, m._InstancedLayer):
                info["instances"] = layer.instance_count()
                info["base_polys"] = layer.base_poly_count()
            layers.append(info)
            if best is None or len(layer) > len(best[1]):
                best = (key, layer)
    out["layers"] = layers

    key, layer = best
    out["biggest"] = {"layer": key[0], "datatype": key[1],
                      "kind": type(layer).__name__, "polys": len(layer)}
    bb = m._layer_bbox_mm(layer, scale_mm)
    out["bbox_mm"] = list(bb) if bb else None

    # Stage: per-cell exposure area (what the Time Calculator runs on
    # every rerun) over an 18 × 18 grid anchored at the mask corner.
    gx0, gy0 = (bb[0], bb[1]) if bb else (0.0, 0.0)
    t0 = time.perf_counter()
    areas = m._cell_areas_binned(layer, scale_mm, 0.0, 0.0, gx0, gy0,
                                 _CHIP_MM, _GRID_N, _GRID_N)
    out["t_area_s"] = time.perf_counter() - t0
    out["area_total_mm2"] = float(sum(areas))

    # Stage: the overview raster / bbox traces the viewer draws.
    t0 = time.perf_counter()
    if isinstance(layer, m._InstancedLayer):
        m._rasterize_coverage(layer, scale_mm, 0.0, 0.0, m._PALETTE[0])
        m._decimated_centers(layer, scale_mm, 0.0, 0.0)
    else:
        m._layer_bbox_mm(layer, scale_mm)
    out["t_render_s"] = time.perf_counter() - t0

    # Stage: box-select zoom — expands only the instances in a window.
    t0 = time.perf_counter()
    if isinstance(layer, m._InstancedLayer) and bb:
        w = min(0.2, (bb[2] - bb[0]) or 0.2)
        res = m._instances_in_window(layer, scale_mm, 0.0, 0.0,
                                     bb[0], bb[0] + w, bb[1], bb[1] + w)
        # ("over", n) signals "too dense to expand"; anything else is
        # either None (empty window) or the (cx, cy, starts) arrays.
        over = isinstance(res, tuple) and isinstance(res[0], str)
        out["zoom"] = "over" if over else ("empty" if res is None else "ok")
    out["t_zoom_s"] = time.perf_counter() - t0

    out["rss_peak_mb"] = _peak_mb()
    out["rss_end_mb"] = _rss_mb()

    if reupload:
        # Uploading a second mask is the real high-water mark, because
        # `st.cache_resource(max_entries=1)` evicts an entry only after
        # its replacement has been computed — both parsed masks are live
        # unless the page drops the old one first.  This mirrors what
        # `_load_gds_layers` now does on a digest change; drop the
        # `clear()` here to measure the un-fixed behaviour.
        #
        # Dropping our own references first is not cheating: a Streamlit
        # rerun's locals die with the rerun, so by the time the next
        # upload is parsed the cache is the only thing still holding the
        # old mask.  Keeping `cells`/`layer` alive here would measure a
        # process the app never has.
        del cells, best, layer, areas
        t0 = time.perf_counter()
        m._load_gds.clear()
        gc.collect()
        m._load_gds(str(out["bytes"]) + ":second", upload)
        out["t_reparse_s"] = time.perf_counter() - t0
        out["rss_peak_reupload_mb"] = _peak_mb()
    return out


# ─── Parent: generate, run, tabulate ────────────────────────────────────────

def app_constant(name: str) -> int:
    """Read one of the page's fixed constants straight from its source.

    Importing ``ebeam_calculator`` here would drag Streamlit into the
    parent process for one integer; a regex avoids that. Only valid for
    genuinely constant values — the memory budgets are *not* constants
    any more (see ``app_limits``).
    """
    src = (ROOT / "tools" / "ebeam_calculator.py").read_text(encoding="utf-8")
    hit = re.search(rf"^{name}\s*=\s*([\d_]+)", src, re.M)
    if not hit:
        raise SystemExit(f"could not find {name} in ebeam_calculator.py")
    return int(hit.group(1).replace("_", ""))


def app_limits(file_mb: float) -> dict:
    """The page's parse budgets for a ``file_mb`` upload on this host.

    Since the budgets are derived from free RAM at parse time rather than
    hard-coded, the only honest way to size a "just under the guard" case
    is to ask the app what it would allow *right now* — done in a
    throwaway subprocess so the parent stays free of Streamlit and of the
    memory it would hold.
    """
    code = ("import json,sys;sys.path.insert(0,%r);"
            "import tools.ebeam.calculator as m;"
            "print('__LIMITS__'+json.dumps(m._limits_for(%f)._asdict()))"
            % (str(ROOT), file_mb))
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=str(ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith("__LIMITS__"):
            return json.loads(line[len("__LIMITS__"):])
    raise SystemExit(f"could not read the app's limits:\n{proc.stderr[-400:]}")


def _cases(sizes: list[float], full: bool) -> list[dict]:
    """The sweep plan: realistic repeated-cell masks at several sizes,
    plus the pathological shapes that probe the parser's guards."""
    plan = [{"label": f"sref {s:g} MB", "mode": "sref", "target_mb": s}
            for s in sizes]
    if not full:
        return plan
    # Size the "just under the guard" cases from what the app would
    # allow on THIS host right now — the budgets track free RAM, so a
    # hard-coded edge would test the wrong thing on a different machine
    # (or on the same machine an hour later).  ~9.3 M vertices per
    # 100 MB of flat geometry, so probe the limits at that scale.
    probe = app_limits(100.0)
    poly_limit = app_constant("_POLY_LIMIT")
    # The budget shrinks as the file grows (the upload buffer is charged
    # against it), so the vertex ceiling and the file size that reaches it
    # are mutually dependent.  One fixed-point step lands close enough for
    # the ≤ / > guard pair to straddle the real edge.
    flat_mb_at_guard = probe["src_verts"] / 9.3e6 * 100.0
    probe = app_limits(flat_mb_at_guard)
    flat_mb_at_guard = probe["src_verts"] / 9.3e6 * 100.0
    src_verts = probe["src_verts"]
    exp_verts = probe["verts"]
    print(f"    app budget here: {probe['budget_mb']:,.0f} MB "
          f"({probe['note']}) → {src_verts:,} vertices, "
          f"{probe['rows']:,} placements")

    plan += [
        # No repetition at all: every shape written out, so the parser
        # must hold real vertices instead of an offset table.  One case
        # just under the vertex guard, one just over it.
        {"label": "flat 100 MB", "mode": "flat", "target_mb": 100.0},
        {"label": f"flat {flat_mb_at_guard * 0.9:.0f} MB (≤ guard)",
         "mode": "flat", "target_mb": flat_mb_at_guard * 0.9},
        {"label": f"flat {flat_mb_at_guard * 1.2:.0f} MB (> guard)",
         "mode": "flat", "target_mb": flat_mb_at_guard * 1.2},
        # A big cell stepped a few times: a small file whose layer cannot
        # stay instanced (its base has more than _POLY_LIMIT polygons)
        # and so must be expanded flat — the _MAX_VERTICES path.  Pick
        # the placement count that puts the base at ~2 × _POLY_LIMIT:
        # 8 polygons / 78 vertices per unit pattern.
        {"label": "bigcell ≤ guard", "mode": "bigcell",
         "places": max(2, int(exp_verts * 0.9 * (8 / 78)
                              / (2 * poly_limit))),
         "vert_budget": int(exp_verts * 0.9)},
        {"label": "bigcell > guard", "mode": "bigcell",
         "places": max(2, int(exp_verts * 1.3 * (8 / 78)
                              / (2 * poly_limit))),
         "vert_budget": int(exp_verts * 1.3)},
        # A few KB on disk, hundreds of millions of placements: the AREF
        # bomb.  Must be refused without ever building the lattice.
        {"label": "aref 20 M places", "mode": "aref", "places": 20_000_000},
        {"label": "aref 100 M places", "mode": "aref", "places": 100_000_000},
        {"label": "aref 1 G places", "mode": "aref", "places": 1_000_000_000},
        # Runs broken by a rotation that changes every element: the slow
        # per-element Python path.
        {"label": "sref_norun 50 MB", "mode": "sref_norun",
         "target_mb": 50.0},
        {"label": "sref_norun 250 MB", "mode": "sref_norun",
         "target_mb": 250.0},
        # Flat geometry AND references, each just under its own guard —
        # the most expensive thing a file of a given size can be.
        {"label": "mixed 250 MB", "mode": "mixed", "target_mb": 250.0,
         "vert_budget": int(src_verts * 0.9)},
        {"label": "mixed 300 MB", "mode": "mixed", "target_mb": 300.0,
         "vert_budget": int(src_verts * 0.9)},
        {"label": "mixed 400 MB", "mode": "mixed", "target_mb": 400.0,
         "vert_budget": int(src_verts * 0.9)},
    ]
    return plan


def _generate(case: dict, outdir: Path) -> dict:
    from gds._gen_test_gds import generate_gds
    name = (case["label"].replace(" ", "_").replace(".", "p")
            .replace("≤", "le").replace(">", "gt").replace("(", "")
            .replace(")", "") + ".gds")
    kw = {}
    if case.get("vert_budget"):
        kw["vert_budget"] = case["vert_budget"]
    t0 = time.perf_counter()
    st = generate_gds(outdir / name, mode=case["mode"],
                      target_mb=case.get("target_mb"),
                      places=case.get("places"), **kw)
    st["t_generate_s"] = time.perf_counter() - t0
    return st


def _run_case(case: dict, outdir: Path, keep: bool,
              rlimit_mb: float | None = None,
              reupload: bool = False) -> dict:
    gen = _generate(case, outdir)
    path = gen["path"]
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", path]
    if rlimit_mb:
        cmd += ["--rlimit-mb", str(rlimit_mb)]
    if reupload:
        cmd.append("--reupload")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    res: dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            res = json.loads(line[len("__RESULT__"):])
    if not res:
        # No result line → the process died (OOM kill, segfault, …).
        res = {"error": f"worker exit {proc.returncode}: "
                        f"{proc.stderr.strip().splitlines()[-1:] or ''}"}
    if not keep:
        os.unlink(path)
    return {"case": case["label"], "gen": gen, "res": res}


def _verdict(row: dict, budget_mb: float) -> tuple[str, float]:
    """(status, peak RSS above baseline in MB) for one finished case."""
    res = row["res"]
    if res.get("guard"):
        return "REFUSED (guard)", float("nan")
    if res.get("error"):
        return "CRASH", float("nan")
    peak = max(res["rss_peak_mb"], res.get("rss_peak_reupload_mb", 0.0))
    delta = peak - res["peak_baseline_mb"]
    return ("OK" if delta <= budget_mb else "OVER BUDGET"), delta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--outdir", default=None,
                    help="where to write the generated .gds files")
    ap.add_argument("--sizes", default="50,100,200,250,300,400",
                    help="comma-separated sref file sizes in MB")
    ap.add_argument("--idle-mb", type=float, default=1000.0,
                    help="RSS the deployed app already uses when idle")
    ap.add_argument("--limit-mb", type=float, default=3000.0,
                    help="container memory ceiling")
    ap.add_argument("--quick", action="store_true",
                    help="sizes only — skip the flat/aref/no-run cases")
    ap.add_argument("--keep", action="store_true",
                    help="keep the generated .gds files")
    ap.add_argument("--rlimit-mb", type=float, default=None,
                    help="hard-cap each worker's address space (Linux only) "
                         "— defaults to the computed budget when --enforce")
    ap.add_argument("--enforce", action="store_true",
                    help="run every case under the RAM budget as a hard cap")
    ap.add_argument("--reupload", action="store_true",
                    help="also parse each mask a second time under a new "
                         "cache key — the two-masks-live peak")
    ap.add_argument("--json", default=None, help="write raw results here")
    args = ap.parse_args()

    if args.worker:
        print("__RESULT__" + json.dumps(
            run_worker(args.worker, args.rlimit_mb, args.reupload)))
        return

    outdir = Path(args.outdir or (ROOT / "_gds_stress"))
    outdir.mkdir(parents=True, exist_ok=True)
    budget = args.limit_mb - args.idle_mb
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    plan = _cases(sizes, full=not args.quick)

    # The worker carries the same ~100 MB import baseline the idle app
    # already has, so cap it at idle + budget when asked to enforce.
    rlimit = args.rlimit_mb or (args.limit_mb if args.enforce else None)

    print(f"\nGDS limits sweep — budget {budget:,.0f} MB "
          f"(ceiling {args.limit_mb:,.0f} MB − idle {args.idle_mb:,.0f} MB)")
    print(f"files in {outdir}"
          + (f", address space capped at {rlimit:,.0f} MB" if rlimit else "")
          + "\n")

    rows = []
    for case in plan:
        print(f"  · {case['label']} …", end="", flush=True)
        try:
            row = _run_case(case, outdir, args.keep, rlimit, args.reupload)
        except ValueError as e:             # unbuildable request
            print(f" skipped ({e})")
            continue
        rows.append(row)
        status, delta = _verdict(row, budget)
        print(f" {status}  (+{delta:,.0f} MB)" if delta == delta
              else f" {status}")

    hdr = (f"{'case':<20}{'file MB':>9}{'places':>13}{'polygons':>15}"
           f"{'parse s':>9}{'area s':>8}{'peak MB':>9}{'Δ MB':>8}"
           f"{'host MB':>9}  verdict")
    print("\n" + hdr)
    print("─" * len(hdr))
    for row in rows:
        gen, res = row["gen"], row["res"]
        status, delta = _verdict(row, budget)
        host = args.idle_mb + delta if delta == delta else float("nan")
        peak = max(res.get("rss_peak_mb", float("nan")),
                   res.get("rss_peak_reupload_mb", 0.0))
        print(f"{row['case']:<20}{gen['mb']:>9.1f}{gen['places']:>13,}"
              f"{gen['polys']:>15,}"
              f"{res.get('t_parse_s', float('nan')):>9.2f}"
              f"{res.get('t_area_s', float('nan')):>8.2f}"
              f"{peak:>9.0f}{delta:>8.0f}{host:>9.0f}  {status}")

    print("\nnotes")
    for row in rows:
        res = row["res"]
        if res.get("guard"):
            print(f"  {row['case']}: guard → {res['guard']}")
        elif res.get("error"):
            print(f"  {row['case']}: {res['error']}")
        elif res.get("biggest"):
            b = res["biggest"]
            extra = ""
            if res.get("rss_peak_reupload_mb"):
                extra = (f", 2nd upload peak "
                         f"{res['rss_peak_reupload_mb']:,.0f} MB")
            print(f"  {row['case']}: L{b['layer']}/D{b['datatype']} "
                  f"{b['kind']} {b['polys']:,} polys, "
                  f"pattern area {res.get('area_total_mm2', 0):.3f} mm², "
                  f"render {res.get('t_render_s', 0):.2f} s, "
                  f"zoom {res.get('t_zoom_s', 0):.2f} s{extra}")

    ok = [r for r in rows
          if _verdict(r, budget)[0] == "OK" and r["gen"]["mode"] == "sref"]
    if ok:
        biggest = max(ok, key=lambda r: r["gen"]["mb"])
        print(f"\nlargest repeated-cell file that fits: "
              f"{biggest['gen']['mb']:.0f} MB "
              f"({biggest['gen']['places']:,} placements, "
              f"{biggest['gen']['polys']:,} polygons, "
              f"+{_verdict(biggest, budget)[1]:,.0f} MB over baseline)")
        print("set .streamlit/config.toml [server] maxUploadSize with margin "
              "below that number.")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nraw results → {args.json}")
    if not args.keep:
        try:
            outdir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()

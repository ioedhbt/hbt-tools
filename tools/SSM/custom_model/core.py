"""
core.py — data model, JSON persistence, and the netlist→Y→S solver for
user-built custom small-signal models.

Topology data model
────────────────────
Everything the user draws reduces to a handful of **2-terminal branches**
between named nodes, plus one **intrinsic π/T stamp**.

* :class:`Element` — a single R, L or C leaf.  Carries an ``id`` (stable key
  used to look up the user-entered value at simulate time) and a ``name``
  (display label, e.g. ``"Cbe"``).
* :class:`Network` — a *series chain of parallel groups* (``groups``): the
  outer list is combined in series, each inner group (list of Elements) in
  parallel.  This two-level form is exactly what the builder exposes ("add a
  series step" appends a group; "add a parallel component" appends an element
  to a group) and covers essentially every practical pad/parasitic branch.
  One Network == one 2-terminal branch between two nodes.
* :class:`CustomModel` — the whole topology: intrinsic type + the per-section
  branch collections + editable intrinsic element names.

Node graph (built inside→outside)
─────────────────────────────────
    GND  : reference (emitter rail)
    BI,CI: intrinsic base / collector
    EI   : intrinsic emitter        (EI→GND through emitter access R,L)
    EJ   : T-model emitter knee      (EJ→EI through intrinsic ``re``; π: EJ≡EI)
    XB,XC: extrinsic base / collector (after the port-1/port-2 series extras)
    P1,P2: external ports            (XB/XC→P1/P2 through base/collector access)

Branches whose Network is empty merge their two end nodes (a plain wire).

Solver
──────
Stamp every passive branch and the intrinsic VCCS (gm·e^{-jωτ}) into a nodal
admittance matrix, Kron-reduce the internal nodes away, and convert the
resulting 2×2 Y to S.  Fully vectorised over frequency.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..helpers import y_to_s_vec


SCHEMA_VERSION = 2
KINDS = ("R", "L", "C")

# Devices: bipolar HBT (terminals B/C/E) or unipolar HEMT (terminals G/D/S).
DEVICES = ("Bipolar", "Unipolar")
# Per-device terminal letters and long port names (port1, port2, common rail).
_TERMINALS = {
    "Bipolar":  {"p1": "B", "p2": "C", "com": "E",
                 "p1_long": "base", "p2_long": "collector", "com_long": "emitter"},
    "Unipolar": {"p1": "G", "p2": "D", "com": "S",
                 "p1_long": "gate", "p2_long": "drain", "com_long": "source"},
}

# The intrinsic controlled source is the only non-R/L/C part of the core.  Its
# scalar parameters are keyed and formatted here; the junction admittances are
# now ordinary editable Networks (see CustomModel.intrinsic_*).
#   π : voltage-controlled source  gm·V_be·e^(−jωτ)              → keys gm, tau
#   T : current-controlled source  α = α₀·e^(−jωτ_C)/(1+jωτ_B)   → keys α₀,τ_B,τ_C
_SOURCE_KEYS = {"Pi": ["gm", "tau"], "T": ["alpha0", "tauB", "tauC"]}
_SOURCE_KIND = {"gm": "gm", "tau": "tau",
                "alpha0": "alpha", "tauB": "tau", "tauC": "tau"}
# Display notation for each source parameter (shown in value inputs + caption).
_SOURCE_DISP = {"gm": "Gm0", "tau": "τ", "alpha0": "α₀", "tauB": "τB", "tauC": "τC"}
_RLC_KINDS = ("R", "L", "C")


def _default_source_name(itype: str, device: str) -> str:
    """Default label for the intrinsic controlled source per topology+device."""
    if itype == "Pi":
        return "gm·Vbe" if device == "Bipolar" else "Ids"
    return "α·Ie" if device == "Bipolar" else "α·Is"


def _migrate_v1(d: dict) -> dict:
    """Rebuild v2 junction Networks from a schema-v1 dict (flat
    ``intrinsic_names`` + Rbi-in-port1).  Topology is preserved exactly: the
    junctions take their v1 names, the base spreading stays empty (so the old
    Rbi kept in ``port1`` lands at the same node), and the source label defaults
    by topology+device.  Values are entered fresh in Use mode, so only the
    structure + names need migrating."""
    nm = d.get("intrinsic_names", {})

    def jn(ck: str, rk: str) -> dict:
        els = []
        if nm.get(ck):
            els.append({"kind": "C", "name": nm[ck]})
        if nm.get(rk):
            els.append({"kind": "R", "name": nm[rk]})
        return {"groups": [els] if els else []}

    d["intrinsic_be"] = jn("Cbe", "Rbe")
    d["intrinsic_bc"] = jn("Cbc", "Rbc")
    d["intrinsic_ce"] = jn("Cce", "Rce")
    d["intrinsic_base"] = {"groups": []}
    d.setdefault("source_name",
                 _default_source_name(d.get("intrinsic_type", "Pi"),
                                      d.get("device", "Bipolar")))
    return d


# ════════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════════
def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Element:
    """A single R / L / C leaf component."""
    kind: str                              # "R" | "L" | "C"
    name: str                              # display label, e.g. "Cbe"
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        return cls(kind=d["kind"], name=d["name"], id=d.get("id") or _new_id())


@dataclass
class Network:
    """A 2-terminal branch: series chain of parallel groups."""
    groups: list[list[Element]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return all(len(g) == 0 for g in self.groups)

    def elements(self) -> list[Element]:
        return [e for g in self.groups for e in g]

    def to_dict(self) -> dict:
        return {"groups": [[e.to_dict() for e in g] for g in self.groups]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Network":
        if not d:
            return cls()
        return cls(groups=[[Element.from_dict(e) for e in g]
                           for g in d.get("groups", [])])


@dataclass
class ShuntBranch:
    """A capacitive shunt/series branch placed between two named terminals.

    ``place`` is one of ``"p1-p2"``, ``"p1-gnd"``, ``"p2-gnd"`` (parasitic) or
    ``"p1-p2"`` / ``"p1-gnd"`` (extrinsic).  ``network`` is the (series +
    parallel) composite the user built for that branch.
    """
    place: str
    network: Network = field(default_factory=Network)
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict:
        return {"id": self.id, "place": self.place,
                "network": self.network.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "ShuntBranch":
        return cls(place=d["place"], network=Network.from_dict(d.get("network")),
                   id=d.get("id") or _new_id())


def _be(): return Network([[Element("C", "Cbe"), Element("R", "Rbe")]])
def _bc(): return Network([[Element("C", "Cbc"), Element("R", "Rbc")]])
def _base(): return Network([[Element("R", "Rbi")]])


@dataclass
class CustomModel:
    name: str = "MyModel"
    intrinsic_type: str = "Pi"                       # "Pi" | "T"
    device: str = "Bipolar"                          # "Bipolar" | "Unipolar"

    # ── Intrinsic core — four editable Networks + one controlled source ──────
    # Each junction is a 2-terminal branch (series chain of parallel groups),
    # defaulting to Cheng's topology.  The controlled source is scalar (its
    # parameters live in the value dict under the source keys gm/tau or
    # α₀/τ_B/τ_C; its label is ``source_name``).
    intrinsic_base: Network = field(default_factory=_base)   # spreading Rbi: BB→BI
    intrinsic_be: Network = field(default_factory=_be)       # b–e junction: BI→EI
    intrinsic_bc: Network = field(default_factory=_bc)       # b–c junction: BI→CI
    intrinsic_ce: Network = field(default_factory=Network)   # c–e output: CI→EI
    source_name: str = ""                                    # set per device+type

    # Section 1 — port extras (series chain of parallel groups), inside→out.
    port1: Network = field(default_factory=Network)  # BI → XB
    port2: Network = field(default_factory=Network)  # CI → XC

    # Section 2 — extrinsic capacitances (p1-p2, p1-gnd).
    extrinsic: list[ShuntBranch] = field(default_factory=list)

    # Section 3 — access R + lead L (series).  Editable names; blank/0 ⇒ absent.
    # Start empty so a fresh model shows only the intrinsic core (no Rx/Lx).
    access_names: dict = field(default_factory=lambda: {
        "Rb": "", "Lb": "", "Rc": "", "Lc": "", "Re": "", "Le": "",
    })

    # Section 4 — parasitic pad capacitances (p1-p2, p1-gnd, p2-gnd).
    parasitic: list[ShuntBranch] = field(default_factory=list)

    # Emitter/source-leg extras (series chain of parallel groups) inserted
    # between the intrinsic emitter and the emitter access R/L — e.g. the
    # Kun-Yang source-side R_delay∥C_delay network above the Rs node.
    emitter: Network = field(default_factory=Network)

    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        self.ensure_intrinsic()

    # ── intrinsic helpers ──────────────────────────────────────────────────
    def source_keys(self) -> list[str]:
        return list(_SOURCE_KEYS[self.intrinsic_type])

    def source_disp(self, key: str) -> str:
        return _SOURCE_DISP.get(key, key)

    def terminals(self) -> dict:
        return _TERMINALS.get(self.device, _TERMINALS["Bipolar"])

    def port_label(self, which: str) -> str:
        """e.g. ``port_label('p1')`` → 'Port 1 (base)' / 'Port 1 (gate)'."""
        t = self.terminals()
        num = {"p1": "1", "p2": "2"}[which]
        return f"Port {num} ({t[f'{which}_long']})"

    def intrinsic_junctions(self):
        """(node_a, node_b, network, role) for each editable junction."""
        return [
            ("BB", "BI", self.intrinsic_base, "base"),   # spreading Rbi
            ("BI", "EI", self.intrinsic_be, "be"),       # b–e junction
            ("BI", "CI", self.intrinsic_bc, "bc"),       # b–c junction
            ("CI", "EI", self.intrinsic_ce, "ce"),       # c–e output
        ]

    def ensure_intrinsic(self) -> None:
        """Fill in the device-dependent default source label if unset."""
        if not self.source_name:
            self.source_name = _default_source_name(self.intrinsic_type,
                                                    self.device)

    # Back-compat alias (older callers used ensure_intrinsic_names()).
    def ensure_intrinsic_names(self) -> None:
        self.ensure_intrinsic()

    # ── (de)serialisation ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        self.ensure_intrinsic()
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "intrinsic_type": self.intrinsic_type,
            "device": self.device,
            "intrinsic_base": self.intrinsic_base.to_dict(),
            "intrinsic_be": self.intrinsic_be.to_dict(),
            "intrinsic_bc": self.intrinsic_bc.to_dict(),
            "intrinsic_ce": self.intrinsic_ce.to_dict(),
            "source_name": self.source_name,
            "port1": self.port1.to_dict(),
            "port2": self.port2.to_dict(),
            "extrinsic": [b.to_dict() for b in self.extrinsic],
            "access_names": dict(self.access_names),
            "parasitic": [b.to_dict() for b in self.parasitic],
            "emitter": self.emitter.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CustomModel":
        # ── v1→v2 migration ────────────────────────────────────────────────
        # Schema-v1 stored the intrinsic as flat ``intrinsic_names`` keys
        # (Cbe/Rbe/…) with Rbi carried in port1, instead of the v2 junction
        # Networks.  Rebuild the junctions so old saved models load faithfully
        # rather than silently falling back to a default Cheng core.
        if "intrinsic_be" not in d and "intrinsic_names" in d:
            d = _migrate_v1(dict(d))

        def net(key, fallback):
            return Network.from_dict(d[key]) if key in d else fallback()
        m = cls(
            name=d.get("name", "MyModel"),
            intrinsic_type=d.get("intrinsic_type", "Pi"),
            device=d.get("device", "Bipolar"),
            intrinsic_base=net("intrinsic_base", _base),
            intrinsic_be=net("intrinsic_be", _be),
            intrinsic_bc=net("intrinsic_bc", _bc),
            intrinsic_ce=net("intrinsic_ce", Network),
            source_name=d.get("source_name", ""),
            port1=Network.from_dict(d.get("port1")),
            port2=Network.from_dict(d.get("port2")),
            extrinsic=[ShuntBranch.from_dict(b) for b in d.get("extrinsic", [])],
            access_names=dict(d.get("access_names", {})),
            parasitic=[ShuntBranch.from_dict(b) for b in d.get("parasitic", [])],
            emitter=Network.from_dict(d.get("emitter")),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )
        m.ensure_intrinsic()
        return m

    def all_value_specs(self) -> list[tuple[str, str, str]]:
        """Every value the user must supply, as (id, kind, label) tuples.

        ``id`` is the lookup key for the value dict passed to the solver.
        Source parameters use their canonical key as id (e.g. ``"gm"``);
        junction R/L/C use their element id.
        """
        specs: list[tuple[str, str, str]] = []
        # Intrinsic junction R/L/C
        for _a, _b, net, role in self.intrinsic_junctions():
            for e in net.elements():
                specs.append((e.id, e.kind, f"Intrinsic · {e.name}"))
        # Controlled-source scalar params
        for k in self.source_keys():
            specs.append((k, _SOURCE_KIND[k], self.source_disp(k)))
        # Port + emitter extras
        for net, lbl in ((self.port1, "Port-1"), (self.port2, "Port-2"),
                         (self.emitter, "Emitter")):
            for e in net.elements():
                specs.append((e.id, e.kind, f"{lbl} · {e.name}"))
        # Extrinsic
        for b in self.extrinsic:
            for e in b.network.elements():
                specs.append((e.id, e.kind, f"Extrinsic {b.place} · {e.name}"))
        # Access
        for k, nm in self.access_names.items():
            if not nm:
                continue
            kind = "C" if k.startswith("C") else ("L" if k.startswith("L") else "R")
            specs.append((f"access_{k}", kind, nm))
        # Parasitic
        for b in self.parasitic:
            for e in b.network.elements():
                specs.append((e.id, e.kind, f"Parasitic {b.place} · {e.name}"))
        return specs

    def grouped_value_specs(self) -> list[tuple[str, list[tuple[str, str, str]]]]:
        """Value specs split into sections **outside→inside** with *clean* names
        (just the component name — no ``p1-p2`` / ``Extrinsic ·`` tags), for the
        value-input UIs.  Returns ``[(section_title, [(id, kind, name), …]), …]``
        with empty sections omitted.  Order: parasitic pad caps → lead L →
        access R → extrinsic caps → port/delay extras → intrinsic core."""
        def shunt_items(branches):
            return [(e.id, e.kind, e.name)
                    for b in branches for e in b.network.elements()]

        groups: list[tuple[str, list]] = [
            ("Parasitic pad capacitances", shunt_items(self.parasitic)),
            ("Lead inductance",
             [(f"access_{k}", "L", nm) for k, nm in self.access_names.items()
              if nm and k.startswith("L")]),
            ("Access resistance",
             [(f"access_{k}", "R", nm) for k, nm in self.access_names.items()
              if nm and k.startswith("R")]),
            ("Extrinsic capacitances", shunt_items(self.extrinsic)),
            ("Port / delay extras",
             [(e.id, e.kind, e.name)
              for net in (self.port1, self.port2, self.emitter)
              for e in net.elements()]),
            ("Intrinsic core",
             [(e.id, e.kind, e.name)
              for _a, _b, net, _r in self.intrinsic_junctions()
              for e in net.elements()]
             + [(k, _SOURCE_KIND[k], self.source_disp(k))
                for k in self.source_keys()]),
        ]
        return [(title, items) for title, items in groups if items]


# ════════════════════════════════════════════════════════════════════════════
# Persistence — custom_models/ folder (+ caller-side download/upload)
# ════════════════════════════════════════════════════════════════════════════
def _repo_root() -> Path:
    # tools/SSM/custom_model/core.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    d = _repo_root() / "custom_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename(name: str) -> str:
    keep = "-_. ()"
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return (cleaned or "model").replace(" ", "_")


def save_model(model: CustomModel) -> Path:
    """Serialise ``model`` to ``custom_models/<name>.json`` and return the path."""
    path = models_dir() / f"{_safe_filename(model.name)}.json"
    path.write_text(model_to_json(model), encoding="utf-8")
    return path


def model_to_json(model: CustomModel) -> str:
    return json.dumps(model.to_dict(), indent=2, ensure_ascii=False)


def load_model(source) -> CustomModel:
    """Load from a path, a JSON string, or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        data = json.loads(bytes(source).decode("utf-8"))
    elif isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                      and source.endswith(".json")):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, str):
        data = json.loads(source)
    else:
        data = dict(source)
    return CustomModel.from_dict(data)


def list_saved_models() -> list[Path]:
    return sorted(models_dir().glob("*.json"))


# ════════════════════════════════════════════════════════════════════════════
# Solver — netlist → nodal Y → 2-port Y → S
# ════════════════════════════════════════════════════════════════════════════
# Near-short admittance used to represent an all-zero (wire) *series* branch
# without changing the node count — the fixed-topology analogue of the scalar
# path's union-find merge (see _branch_series_b / compile_plan).
_SHORT_Y = 1e12


def _safe(x, xp):
    """Replace exact-zero entries with 1 so a reciprocal stays finite; callers
    mask the genuine result back to 0 with a parallel ``where``."""
    return xp.where(xp.abs(x) > 0, x, 1.0)


def _elem_adm_b(kind: str, v, jw, xp):
    """Vectorised admittance of one R/L/C leaf.

    ``v`` is a value array of shape ``(B, 1)`` (or scalar-broadcastable) and
    ``jw = 1j·ω`` has shape ``(1, N)``.  A zero value ⇒ absent (open) → 0
    admittance, matching the scalar convention of the former ``_y_element``.
    """
    if kind == "R":
        return xp.where(v != 0, 1.0 / _safe(v, xp), 0.0)            # (B, 1)
    if kind == "L":
        return xp.where(v != 0, 1.0 / _safe(jw * v, xp), 0.0)       # (B, N)
    if kind == "C":
        return jw * v                                              # (B, N); 0 if v=0
    return xp.zeros_like(jw)


def _group_adm_b(group, vals: dict, jw, xp):
    """Parallel combination (admittance sum) of one group's elements."""
    y = 0.0
    for kind, key in group:
        y = y + _elem_adm_b(kind, vals[key], jw, xp)
    return y


def _branch_series_b(groups, vals: dict, jw, xp):
    """Admittance of a structurally-present *series* branch.

    Sum the group impedances (a zero-value group contributes 0 impedance — a
    short — exactly as the scalar ``_y_network`` skipped absent series groups).
    An all-zero branch collapses to a wire, represented here by the large
    finite admittance ``_SHORT_Y`` (numerically a short to ~1e-12 relative).
    """
    Z = 0.0
    for g in groups:
        yg = _group_adm_b(g, vals, jw, xp)
        Z = Z + xp.where(xp.abs(yg) > 0, 1.0 / _safe(yg, xp), 0.0)
    return xp.where(xp.abs(Z) > 0, 1.0 / _safe(Z, xp), _SHORT_Y)


def _branch_shunt_b(groups, vals: dict, jw, xp):
    """Admittance of a structurally-present *shunt* branch.  Any absent group
    breaks the path → the whole branch is open (0), matching the scalar
    ``_y_network(series=False)`` early-return-None semantics."""
    inv = 0.0
    open_mask = None
    for g in groups:
        yg = _group_adm_b(g, vals, jw, xp)
        gz = xp.abs(yg) == 0
        open_mask = gz if open_mask is None else (open_mask | gz)
        inv = inv + xp.where(xp.abs(yg) > 0, 1.0 / _safe(yg, xp), 0.0)
    y = xp.where(xp.abs(inv) > 0, 1.0 / _safe(inv, xp), 0.0)
    return xp.where(open_mask, 0.0, y) if open_mask is not None else y


def _junction_adm_b(groups, vals: dict, jw, xp):
    """Equivalent admittance of an intrinsic junction Network (shunt
    semantics; empty ⇒ 0).  Vectorised form of the scalar ``_net_admittance``."""
    if not groups:
        return xp.zeros_like(jw)
    return _branch_shunt_b(groups, vals, jw, xp)


def _intrinsic_Y_b(itype: str, Ybe, Ybc, Yce, src: dict, jw, xp) -> tuple:
    """Batched common-emitter 2-port intrinsic admittance (y11, y12, y21, y22).

    Vectorised form of the former scalar ``_intrinsic_Y``.  Referenced to the
    intrinsic emitter node; junction admittances are precomputed from the
    editable junction Networks and ``src`` holds the controlled-source scalars.

    * **Pi** — hybrid-π:  y = [[Ybe+Ybc, −Ybc], [gm−Ybc, Ybc+Yce]],
      gm = gm₀·e^(−jωτ).
    * **T**  — current-source T (Cheng-T / Xu):  Y = inv(Z) where
      Z = [[Zbe, Zbe], [Zbe − αZbc, (1−α)Zbc + Zbe]],
      Zbe = 1/Ybe, Zbc = 1/Ybc, α = α₀·e^(−jωτ_C)/(1+jωτ_B); Yce added in
      parallel at the output.
    """
    if itype == "Pi":
        gm = src["gm"] * xp.exp(-jw * src["tau"])
        return Ybe + Ybc, -Ybc, gm - Ybc, Ybc + Yce
    # ── T (α current-source) ──
    Zbe = xp.where(xp.abs(Ybe) > 0, 1.0 / _safe(Ybe, xp), 0.0)
    Zbc = xp.where(xp.abs(Ybc) > 0, 1.0 / _safe(Ybc, xp), 0.0)
    alpha = src["alpha0"] * xp.exp(-jw * src["tauC"]) / (1.0 + jw * src["tauB"])
    z11, z12 = Zbe, Zbe
    z21, z22 = Zbe - alpha * Zbc, (1.0 - alpha) * Zbc + Zbe
    det = z11 * z22 - z12 * z21
    det = xp.where(xp.abs(det) > 0, det, 1e-30)
    return z22 / det, -z12 / det, -z21 / det, z11 / det + Yce


class _UnionFind:
    def __init__(self):
        self._p: dict[str, str] = {}

    def find(self, a: str) -> str:
        self._p.setdefault(a, a)
        root = a
        while self._p[root] != root:
            root = self._p[root]
        while self._p[a] != root:
            self._p[a], a = root, self._p[a]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep GND / port names as roots when possible for readability.
        priority = {"GND": 3, "P1": 2, "P2": 2}
        if priority.get(rb, 0) > priority.get(ra, 0):
            ra, rb = rb, ra
        self._p[rb] = ra


@dataclass
class SimPlan:
    """Value-free, flattened topology of a :class:`CustomModel`, compiled once
    (:func:`compile_plan`) so a tuning sweep can evaluate thousands of parameter
    sets without rebuilding the node graph each time.  Consumed by the
    vectorised evaluator (:func:`simulate_custom_model_batch`) and, when built,
    the Rust kernel.

    * ``n`` — node count; ordering is ``[P1, P2, internal…]`` so a Kron
      reduction that keeps the first two indices yields the 2-port.
    * ``branches`` — ``(ia, ib, series, groups)`` per passive branch, where
      ``ia``/``ib`` are node indices (``-1`` == GND/reference) and ``groups`` is
      ``[[(kind, value_key), …], …]`` (series chain of parallel groups).
    * ``twoport`` — ``(ia, ib, iref, be_groups, bc_groups, ce_groups)`` for the
      intrinsic controlled-source 2-port.
    * ``itype`` / ``source_keys`` — controlled-source flavour + scalar keys.
    * ``value_keys`` — every value key referenced (scalar wrapper + Rust).
    """
    n: int
    branches: list
    twoport: tuple
    itype: str
    source_keys: list
    value_keys: list


def _net_groups(net: "Network") -> list:
    """Structural (value-free) group spec for a Network: ``[[(kind, id), …], …]``
    with empty steps dropped."""
    return [[(e.kind, e.id) for e in g] for g in net.groups if g]


def _access_groups(model: "CustomModel", rkey: str, lkey: str) -> list:
    """Series groups for an access leg — only the components the user actually
    named (blank name ⇒ absent), so a fully-undefined leg compiles to a wire."""
    groups = []
    if model.access_names.get(rkey):
        groups.append([("R", f"access_{rkey}")])
    if model.access_names.get(lkey):
        groups.append([("L", f"access_{lkey}")])
    return groups


def compile_plan(model: CustomModel) -> SimPlan:
    """Resolve a CustomModel's topology to a flat, value-free :class:`SimPlan`.

    Node merges are decided **structurally** (a series branch with no defined
    components is a wire) rather than from runtime values, so the node count is
    fixed for the whole sweep.  Mirrors the build order of the former
    ``simulate_custom_model`` exactly (intrinsic → port extras → extrinsic →
    access → pads).
    """
    uf = _UnionFind()
    series_specs: list = []   # (a, b, groups)
    shunt_specs: list = []    # (a, b, groups)

    def add_series(a, b, groups):
        if groups:
            series_specs.append((a, b, groups))
        else:
            uf.union(a, b)            # absent series branch → wire (merge nodes)

    def add_shunt(a, b, groups):
        if groups:                    # absent shunt branch → open → drop
            shunt_specs.append((a, b, groups))

    # ── Intrinsic core: 2-port between BI, CI (ref EI) + base spreading ─────
    be_g = _net_groups(model.intrinsic_be)
    bc_g = _net_groups(model.intrinsic_bc)
    ce_g = _net_groups(model.intrinsic_ce)
    # Base spreading network (Rbi) in series ahead of the intrinsic base node.
    add_series("BB", "BI", _net_groups(model.intrinsic_base))

    # ── Section 1: port extras (XB→BB base, XC→CI collector) ───────────────
    add_series("XB", "BB", _net_groups(model.port1))
    add_series("XC", "CI", _net_groups(model.port2))

    # ── Section 2: extrinsic caps ──────────────────────────────────────────
    # Extrinsic caps tap the *inner* base/collector nodes (BB/CI); a "p1-gnd"
    # shunt returns to the intrinsic emitter node EI (above the emitter lead),
    # not true ground (parasitic pad caps in Section 4 reference GND instead).
    for b in model.extrinsic:
        a, c = ("BB", "CI") if b.place == "p1-p2" else ("BB", "EI")
        add_shunt(a, c, _net_groups(b.network))

    # ── Section 3: access R + lead L (XB→P1, XC→P2, EI→GND) ─────────────────
    add_series("XB", "P1", _access_groups(model, "Rb", "Lb"))
    add_series("XC", "P2", _access_groups(model, "Rc", "Lc"))
    # Emitter leg: EI →(emitter extras, e.g. R_delay∥C_delay)→ EM →(Re,Le)→ GND
    add_series("EI", "EM", _net_groups(model.emitter))
    add_series("EM", "GND", _access_groups(model, "Re", "Le"))

    # ── Section 4: parasitic pad caps ──────────────────────────────────────
    place_nodes = {"p1-p2": ("P1", "P2"), "p1-gnd": ("P1", "GND"),
                   "p2-gnd": ("P2", "GND")}
    for b in model.parasitic:
        a, c = place_nodes[b.place]
        add_shunt(a, c, _net_groups(b.network))

    # ── Resolve canonical node roots; GND excluded as the reference ────────
    R = uf.find
    node_set = set()
    for a, b, _ in series_specs:
        node_set.add(R(a)); node_set.add(R(b))
    for a, b, _ in shunt_specs:
        node_set.add(R(a)); node_set.add(R(b))
    for nd in ("BI", "CI", "EI"):
        node_set.add(R(nd))
    gnd = R("GND")
    node_set.discard(gnd)

    p1, p2 = R("P1"), R("P2")
    if p1 == gnd or p2 == gnd or p1 == p2:
        raise ValueError("Degenerate topology: a port collapsed onto ground or "
                         "the two ports merged. Add the access/parasitic "
                         "branches that separate the ports before simulating.")

    # Order: ports first, then internal nodes (so reduction keeps [P1,P2]).
    internal = sorted(node_set - {p1, p2})
    order = [p1, p2] + internal
    idx = {nd: i for i, nd in enumerate(order)}
    n = len(order)

    def ni(name):
        r = R(name)
        return -1 if r == gnd else idx[r]

    branches = [(ni(a), ni(b), True, groups) for a, b, groups in series_specs]
    branches += [(ni(a), ni(b), False, groups) for a, b, groups in shunt_specs]
    twoport = (ni("BI"), ni("CI"), ni("EI"), be_g, bc_g, ce_g)

    # Collect every value key referenced (scalar wrapper defaults + Rust).
    value_keys: list = []
    seen: set = set()

    def _collect(groups):
        for g in groups:
            for _kind, key in g:
                if key not in seen:
                    seen.add(key); value_keys.append(key)

    for _ia, _ib, _s, groups in branches:
        _collect(groups)
    _collect(be_g); _collect(bc_g); _collect(ce_g)
    for k in model.source_keys():
        if k not in seen:
            seen.add(k); value_keys.append(k)

    return SimPlan(n=n, branches=branches, twoport=twoport,
                   itype=model.intrinsic_type, source_keys=model.source_keys(),
                   value_keys=value_keys)


# ── Batched evaluator ────────────────────────────────────────────────────────
def _stamp(Yb, ia: int, ib: int, y) -> None:
    """Stamp a 2-terminal admittance ``y`` (shape (B,1) or (B,N)) into the
    nodal matrix ``Yb`` (B,N,n,n).  ``-1`` indices are the GND reference."""
    if ia >= 0:
        Yb[:, :, ia, ia] += y
    if ib >= 0:
        Yb[:, :, ib, ib] += y
    if ia >= 0 and ib >= 0:
        Yb[:, :, ia, ib] -= y
        Yb[:, :, ib, ia] -= y


def _simulate_plan_core(plan: SimPlan, jw, vals: dict, z0: float, xp, B: int):
    """Evaluate one (already-chunked) batch: build (B,N,n,n) Y, Kron-reduce the
    internals, convert to S[B,N,2,2].  ``vals`` maps key → (B,1) (or (1,1))
    array; ``jw`` is (1,N)."""
    N = jw.shape[1]
    n = plan.n
    Yb = xp.zeros((B, N, n, n), dtype=complex)

    for ia, ib, series, groups in plan.branches:
        y = (_branch_series_b(groups, vals, jw, xp) if series
             else _branch_shunt_b(groups, vals, jw, xp))
        _stamp(Yb, ia, ib, y)

    ia, ib, iref, be_g, bc_g, ce_g = plan.twoport
    Ybe = _junction_adm_b(be_g, vals, jw, xp)
    Ybc = _junction_adm_b(bc_g, vals, jw, xp)
    Yce = _junction_adm_b(ce_g, vals, jw, xp)
    src = {k: vals[k] for k in plan.source_keys}
    y11, y12, y21, y22 = _intrinsic_Y_b(plan.itype, Ybe, Ybc, Yce, src, jw, xp)

    def cell(ir, ic, val):
        if ir >= 0 and ic >= 0:
            Yb[:, :, ir, ic] += val

    # Embed the 3-terminal common-reference 2-port via its indefinite matrix.
    cell(ia, ia, y11); cell(ia, ib, y12)
    cell(ib, ia, y21); cell(ib, ib, y22)
    cell(ia, iref, -(y11 + y12)); cell(ib, iref, -(y21 + y22))
    cell(iref, ia, -(y11 + y21)); cell(iref, ib, -(y12 + y22))
    cell(iref, iref, y11 + y12 + y21 + y22)

    # Kron reduction of internal nodes (indices 2..n-1).
    if n == 2:
        Y2 = Yb
    else:
        Yaa = Yb[..., :2, :2]
        Yai = Yb[..., :2, 2:]
        Yia = Yb[..., 2:, :2]
        Yii = Yb[..., 2:, 2:]
        # Regularise to avoid singular internal blocks (floating nodes).
        eye = xp.eye(n - 2, dtype=complex)
        Yii = Yii + 1e-15 * eye
        Y2 = Yaa - Yai @ xp.linalg.solve(Yii, Yia)

    return y_to_s_vec(Y2, z0, xp)


def simulate_custom_model_batch(plan: SimPlan, freq: np.ndarray, values: dict,
                                z0: float = 50.0, xp=np,
                                max_batch_elems: int = 16_000_000) -> np.ndarray:
    """Vectorised forward simulation over a parameter batch.

    ``values`` maps each value key to a scalar or a 1-D array (length B) of
    parameter values.  Returns ``S[B, N, 2, 2]`` (pass scalars for a single
    simulation and index ``[0]``).

    Pass ``xp=cupy`` to run the whole evaluation on the GPU.  Large sweeps are
    chunked along the batch axis so peak memory stays under ``max_batch_elems``
    complex entries of the (B,N,n,n) tensor.
    """
    freq = np.asarray(freq, dtype=float)
    N = freq.size
    jw = xp.asarray(1j * 2.0 * np.pi * freq).reshape(1, N)

    raw: dict = {}
    B = 1
    for key in plan.value_keys:
        arr = xp.asarray(values.get(key, 0.0), dtype=complex).reshape(-1)
        raw[key] = arr
        if arr.size > 1:
            B = max(B, int(arr.size))

    def view(key, lo, hi):
        arr = raw[key]
        return arr.reshape(1, 1) if arr.size == 1 else arr[lo:hi].reshape(-1, 1)

    per = max(1, N * plan.n * plan.n)
    chunk = max(1, min(B, max_batch_elems // per))
    outs = []
    for lo in range(0, B, chunk):
        hi = min(B, lo + chunk)
        vals = {key: view(key, lo, hi) for key in plan.value_keys}
        outs.append(_simulate_plan_core(plan, jw, vals, z0, xp, hi - lo))
    return outs[0] if len(outs) == 1 else xp.concatenate(outs, axis=0)


def simulate_custom_model(model: CustomModel, freq: np.ndarray,
                          values: dict, z0: float = 50.0) -> np.ndarray:
    """Forward-simulate S[N,2,2] from a :class:`CustomModel` + value dict.

    Thin scalar wrapper over :func:`compile_plan` + :func:`simulate_custom_model_batch`
    (kept for the forward-sim / "Use" UI and back-compat).  ``values`` maps
    element id (or intrinsic/access key) → value in SI units.
    """
    plan = compile_plan(model)
    vals = {k: float(values.get(k, 0.0) or 0.0) for k in plan.value_keys}
    return simulate_custom_model_batch(plan, freq, vals, z0, xp=np)[0]

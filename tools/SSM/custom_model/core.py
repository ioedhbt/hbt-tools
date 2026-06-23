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
def _y_element(kind: str, val: float, omega: np.ndarray) -> np.ndarray:
    """Admittance array for one R/L/C leaf.  Value 0/None ⇒ absent (open)."""
    z = np.zeros_like(omega, dtype=complex)
    if not val:
        return z
    if kind == "R":
        return z + (1.0 / val)
    if kind == "L":
        return 1.0 / (1j * omega * val)
    if kind == "C":
        return 1j * omega * val * np.ones_like(omega)
    return z


def _y_group(group: list[Element], values: dict, omega: np.ndarray) -> np.ndarray:
    """Parallel combination of the elements in one group."""
    y = np.zeros_like(omega, dtype=complex)
    for e in group:
        y = y + _y_element(e.kind, float(values.get(e.id, 0.0) or 0.0), omega)
    return y


def _y_network(net: Network, values: dict, omega: np.ndarray,
               series: bool) -> np.ndarray | None:
    """Admittance of a series-of-parallel-groups branch.

    Empty groups (no elements yet) are ignored.  A group whose elements are
    all absent (value 0) evaluates to 0 admittance — what that means depends
    on the branch role:

    * ``series=True``  (signal-path branch: port extras, access) — an absent
      group is a **short** (skipped); a fully-absent branch returns ``None``
      so the caller merges its end nodes with a plain wire.
    * ``series=False`` (shunt/bridge branch: extrinsic, parasitic caps) — an
      absent group breaks the path → the branch is **open** → returns ``None``
      so the caller drops it entirely.
    """
    groups = [g for g in net.groups if g]            # ignore empty steps
    if not groups:
        return None
    z = np.zeros_like(omega, dtype=complex)
    used = 0
    for g in groups:
        yg = _y_group(g, values, omega)
        if not np.any(np.abs(yg) > 0):               # group fully absent
            if series:
                continue                              # short — skip
            return None                               # shunt — open → drop
        with np.errstate(divide="ignore", invalid="ignore"):
            z = z + 1.0 / yg
        used += 1
    if used == 0:                                     # everything shorted
        return None                                   # → wire (series) / open
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(z) > 0, 1.0 / z, 0.0)


def _rc_parallel(c_val: float, r_val: float, omega: np.ndarray) -> np.ndarray:
    """C ∥ R admittance for intrinsic junctions."""
    y = np.zeros_like(omega, dtype=complex)
    if c_val:
        y = y + 1j * omega * c_val
    if r_val:
        y = y + 1.0 / r_val
    return y


def _net_admittance(net: "Network", values: dict, omega: np.ndarray) -> np.ndarray:
    """Equivalent admittance array of a 2-terminal junction Network.

    The junction is a series chain of parallel groups (default = a single
    Cbe∥Rbe-style group).  Returns 0 where the branch is open/absent.  For a
    plain C∥R group this is identical to ``_rc_parallel`` — so a Cheng-default
    junction reproduces the previous analytic admittance exactly, while a
    user-edited junction (extra parallels / series steps) is honoured.
    """
    y = _y_network(net, values, omega, series=False)
    return np.zeros_like(omega, dtype=complex) if y is None else y


def _intrinsic_Y(itype: str, Ybe, Ybc, Yce, src: dict,
                 omega: np.ndarray) -> tuple:
    """Common-emitter 2-port intrinsic admittance (y11, y12, y21, y22).

    Referenced to the intrinsic emitter node.  Junction admittances ``Ybe``,
    ``Ybc`` (b–e, b–c) and the optional output ``Yce`` are precomputed from the
    editable junction Networks; ``src`` holds the controlled-source scalar
    parameters.  Base spreading resistance (Rbi) is stamped separately as a
    series branch BB→BI.

    * **Pi** — hybrid-π:  y = [[Ybe+Ybc, −Ybc], [gm−Ybc, Ybc+Yce]],
      gm = gm₀·e^(−jωτ).
    * **T**  — current-source T (Cheng-T / Xu):  Y = inv(Z) where
      Z = [[Zbe, Zbe], [Zbe − αZbc, (1−α)Zbc + Zbe]],
      Zbe = 1/Ybe, Zbc = 1/Ybc, α = α₀·e^(−jωτ_C)/(1+jωτ_B); Yce added in
      parallel at the output.
    """
    jw = 1j * omega
    if itype == "Pi":
        gm = src["gm"] * np.exp(-jw * src["tau"])
        return Ybe + Ybc, -Ybc, gm - Ybc, Ybc + Yce
    # ── T (α current-source) ──
    z = np.zeros_like(omega, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        Zbe = np.where(np.abs(Ybe) > 0, 1.0 / Ybe, z)
        Zbc = np.where(np.abs(Ybc) > 0, 1.0 / Ybc, z)
    alpha = src["alpha0"] * np.exp(-jw * src["tauC"]) / (1.0 + jw * src["tauB"])
    z11, z12 = Zbe, Zbe
    z21, z22 = Zbe - alpha * Zbc, (1.0 - alpha) * Zbc + Zbe
    det = z11 * z22 - z12 * z21
    det = np.where(np.abs(det) > 0, det, 1e-30)
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


def simulate_custom_model(model: CustomModel, freq: np.ndarray,
                          values: dict, z0: float = 50.0) -> np.ndarray:
    """Forward-simulate S[N,2,2] from a :class:`CustomModel` + value dict.

    ``values`` maps element id (or intrinsic/access key) → value in SI units.
    """
    freq = np.asarray(freq, dtype=float)
    omega = 2.0 * np.pi * freq
    F = freq.size

    uf = _UnionFind()
    branches: list[tuple[str, str, np.ndarray]] = []   # (nodeA, nodeB, Y)
    twoports: list[tuple] = []                          # (a, b, ref, y11..y22)

    def add_network(a: str, b: str, net: Network, series: bool) -> None:
        y = _y_network(net, values, omega, series)
        if y is None:
            if series:
                uf.union(a, b)      # absent series branch → wire (merge nodes)
            # absent shunt branch → open → drop (no stamp)
        else:
            branches.append((a, b, y))

    # ── Intrinsic core: stamp the common-emitter 2-port between BI, CI, EI ──
    Ybe = _net_admittance(model.intrinsic_be, values, omega)
    Ybc = _net_admittance(model.intrinsic_bc, values, omega)
    Yce = _net_admittance(model.intrinsic_ce, values, omega)
    src = {k: float(values.get(k, 0.0) or 0.0) for k in model.source_keys()}
    y11, y12, y21, y22 = _intrinsic_Y(model.intrinsic_type, Ybe, Ybc, Yce, src,
                                      omega)
    twoports.append(("BI", "CI", "EI", y11, y12, y21, y22))
    # Base spreading network (Rbi) in series ahead of the intrinsic base node.
    add_network("BB", "BI", model.intrinsic_base, series=True)

    # ── Section 1: port extras (XB→BB base, XC→CI collector) ───────────────
    # Port / delay extras sit *outside* the extrinsic caps (build order goes
    # inside→out: intrinsic → extrinsic → port/delay → access → pads), so they
    # are placed between the access node (XB/XC) and the inner extrinsic node
    # (BB/CI).  When empty they merge XB≡BB / XC≡CI (e.g. Cheng/Xu).
    add_network("XB", "BB", model.port1, series=True)
    add_network("XC", "CI", model.port2, series=True)

    # ── Section 2: extrinsic caps ──────────────────────────────────────────
    # Extrinsic caps tap the *inner* base/collector nodes (BB/CI) — i.e. inside
    # the port/delay extras and the access leads — so a "p1-gnd" extrinsic shunt
    # returns to the intrinsic emitter node EI (above the emitter lead), not to
    # true ground.  This matches the conventional textbook placement (e.g.
    # Cheng's Cbex at the common intrinsic-emitter node).  Parasitic pad caps,
    # which sit *outside* the leads, reference true ground instead (Section 4).
    for b in model.extrinsic:
        a, c = ("BB", "CI") if b.place == "p1-p2" else ("BB", "EI")
        add_network(a, c, b.network, series=False)

    # ── Section 3: access R + lead L (XB→P1, XC→P2, EI→GND) ─────────────────
    def access_branch(a: str, b: str, rkey: str, lkey: str) -> None:
        net = Network(groups=[
            [Element(kind="R", name=model.access_names.get(rkey, rkey), id=f"access_{rkey}")],
            [Element(kind="L", name=model.access_names.get(lkey, lkey), id=f"access_{lkey}")],
        ])
        add_network(a, b, net, series=True)

    access_branch("XB", "P1", "Rb", "Lb")
    access_branch("XC", "P2", "Rc", "Lc")
    # Emitter leg: EI →(emitter extras, e.g. R_delay∥C_delay)→ EM →(Re,Le)→ GND
    add_network("EI", "EM", model.emitter, series=True)
    access_branch("EM", "GND", "Re", "Le")

    # ── Section 4: parasitic pad caps ──────────────────────────────────────
    place_nodes = {"p1-p2": ("P1", "P2"), "p1-gnd": ("P1", "GND"),
                   "p2-gnd": ("P2", "GND")}
    for b in model.parasitic:
        a, c = place_nodes[b.place]
        add_network(a, c, b.network, series=False)

    return _assemble_and_reduce(branches, twoports, uf, F, z0)


def _assemble_and_reduce(branches, twoports, uf, F: int, z0: float) -> np.ndarray:
    """Stamp the nodal matrix, Kron-reduce internals, convert Y→S."""
    def R(n: str) -> str:
        return uf.find(n)

    # Node set (canonical), GND excluded as reference.
    node_set = set()
    for a, b, _ in branches:
        node_set.add(R(a)); node_set.add(R(b))
    for a, b, ref, *_ in twoports:
        for n in (a, b, ref):
            node_set.add(R(n))
    node_set.discard(R("GND"))

    p1, p2 = R("P1"), R("P2")
    if p1 == R("GND") or p2 == R("GND") or p1 == p2:
        raise ValueError("Degenerate topology: a port collapsed onto ground or "
                         "the two ports merged. Add the access/parasitic "
                         "branches that separate the ports before simulating.")

    # Order: ports first, then internal nodes (so reduction keeps [P1,P2]).
    internal = sorted(node_set - {p1, p2})
    order = [p1, p2] + internal
    idx = {n: i for i, n in enumerate(order)}
    n = len(order)

    Yb = np.zeros((F, n, n), dtype=complex)
    gnd = R("GND")

    def stamp(a: str, b: str, y: np.ndarray) -> None:
        a, b = R(a), R(b)
        ia = idx.get(a) if a != gnd else None
        ib = idx.get(b) if b != gnd else None
        if ia is not None:
            Yb[:, ia, ia] += y
        if ib is not None:
            Yb[:, ib, ib] += y
        if ia is not None and ib is not None:
            Yb[:, ia, ib] -= y
            Yb[:, ib, ia] -= y

    for a, b, y in branches:
        stamp(a, b, y)

    def cell(r_node: str, c_node: str, val: np.ndarray) -> None:
        ir = idx.get(R(r_node)) if R(r_node) != gnd else None
        ic = idx.get(R(c_node)) if R(c_node) != gnd else None
        if ir is not None and ic is not None:
            Yb[:, ir, ic] += val

    for a, b, ref, y11, y12, y21, y22 in twoports:
        # Embed a 3-terminal common-reference 2-port (ports a, b; reference
        # ref) via its indefinite admittance matrix.
        cell(a, a, y11); cell(a, b, y12)
        cell(b, a, y21); cell(b, b, y22)
        cell(a, ref, -(y11 + y12)); cell(b, ref, -(y21 + y22))
        cell(ref, a, -(y11 + y21)); cell(ref, b, -(y12 + y22))
        cell(ref, ref, y11 + y12 + y21 + y22)

    # Kron reduction of internal nodes (indices 2..n-1).
    if n == 2:
        Y2 = Yb
    else:
        Yaa = Yb[:, :2, :2]
        Yai = Yb[:, :2, 2:]
        Yia = Yb[:, 2:, :2]
        Yii = Yb[:, 2:, 2:]
        # Regularise to avoid singular internal blocks (floating nodes).
        eye = np.eye(n - 2, dtype=complex)[None, :, :]
        Yii = Yii + 1e-15 * eye
        Y2 = Yaa - Yai @ np.linalg.solve(Yii, Yia)

    return y_to_s_vec(Y2, z0, np)

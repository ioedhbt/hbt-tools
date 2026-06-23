"""
ui_build.py — progressive "Make" workflow for the custom SSM builder.

Sections reveal one at a time, inside→outward.  The next section's controls
only appear after the user confirms the current one:

    0  Device + intrinsic core — pick Bipolar/Unipolar, π/T, edit the B/C/E
       junction Networks (default = Cheng's) and name the controlled source
    1  Delay / port extras    — series-of-parallel R/L/C at each port + the
       common (emitter/source) port
    2  Extrinsic caps         — shunt P1↔P2 or P1↔GND (series + parallel)
    3  Access R + lead L      — standard series Rb/Lb, Rc/Lc, Re/Le
    4  Parasitic caps         — pad caps P1↔P2, P1↔GND, P2↔GND  →  Save

A live SVG schematic redraws above the controls on every interaction.  A fresh
model shows only the intrinsic core (the access Rx/Lx stay hidden until the
user reaches and names them).
"""
from __future__ import annotations

import streamlit as st

from .core import (CustomModel, Network, Element, ShuntBranch,
                   model_to_json, load_model, _default_source_name)
from .schematic import (render_schematic, intrinsic_thumbnail,
                        section_thumbnail, svg_to_png, copy_image_button,
                        svg_pixel_height)


_MODEL_KEY = "cmb_model"
_STAGE_KEY = "cmb_stage"

_TRASH = "🗑️"          # explicit emoji-variation selector so it always renders


def _ver() -> int:
    """Monotonic suffix mixed into name-input keys so they re-initialise from
    the model after a programmatic rename (device switch / delete)."""
    return int(st.session_state.get("cmb_namever", 0))


def _bump_namever() -> None:
    st.session_state["cmb_namever"] = _ver() + 1


def _auto_download_json(filename: str, data_str: str) -> None:
    """Trigger a browser download of ``data_str`` as ``filename`` via a hidden
    data-URI anchor that auto-clicks (the ⬇ button stays as a fallback if the
    browser blocks the programmatic download)."""
    import base64
    import html as _html
    b64 = base64.b64encode(data_str.encode("utf-8")).decode("ascii")
    safe = _html.escape(filename, quote=True)
    doc = (f'<!DOCTYPE html><html><body style="margin:0">'
           f'<a id="dl" download="{safe}" '
           f'href="data:application/json;base64,{b64}"></a>'
           f'<script>document.getElementById("dl").click();</script>'
           f'</body></html>')
    st.iframe(doc, height=1)


def fire_pending_download() -> None:
    """Download a model's .json if one was queued by the build view's "Send to
    Load / Fit" button — called at the top of the Load and Fit views, i.e. on
    whichever destination the navigation rerun landed on."""
    pend = st.session_state.pop("cm_pending_download", None)
    if pend:
        _auto_download_json(pend[0], pend[1])


def _cur_part(state_key: str, keys: list, default: str) -> str:
    """Current chip selection (read before the illustration is drawn)."""
    s = st.session_state.get(state_key, default)
    return s if s in keys else default


def _part_chips(part_opts: dict, keys: list, state_key: str, default: str) -> str:
    """Render the chip selector; persist the selection by key and return it."""
    labels = [part_opts[k] for k in keys]
    sel = _cur_part(state_key, keys, default)
    picked = st.segmented_control(
        "Edit which part", labels, default=part_opts[sel],
        key=f"{state_key}_chip_{_ver()}", label_visibility="collapsed")
    if picked in labels:
        ns = keys[labels.index(picked)]
        if ns != sel:
            st.session_state[state_key] = ns
            st.rerun()
        sel = ns
    return sel


def _place_net(branches: list, place: str) -> Network:
    """The shunt-branch Network for ``place`` (P1↔P2 / P1↔GND / P2↔GND) — created
    empty if none exists yet, so the chip editor always has something to edit.
    Empty branches read as 'absent' (dashed in the illustration, dropped by the
    solver and the schematic)."""
    for b in branches:
        if b.place == place:
            return b.network
    nb = ShuntBranch(place=place, network=Network(groups=[[]]))
    branches.append(nb)
    return nb.network


def _model() -> CustomModel:
    if _MODEL_KEY not in st.session_state:
        st.session_state[_MODEL_KEY] = CustomModel()
    return st.session_state[_MODEL_KEY]


def _stage() -> int:
    return int(st.session_state.get(_STAGE_KEY, 0))


def _set_stage(v: int) -> None:
    st.session_state[_STAGE_KEY] = max(_stage(), v)  # never regress on reveal


def _default_name(model: CustomModel, kind: str) -> str:
    n = sum(1 for net in (model.intrinsic_base, model.intrinsic_be,
                          model.intrinsic_bc, model.intrinsic_ce,
                          model.port1, model.port2, model.emitter)
            for e in net.elements() if e.kind == kind)
    n += sum(1 for b in (model.extrinsic + model.parasitic)
             for e in b.network.elements() if e.kind == kind)
    return f"{kind}{n + 1}"


# Terminal letters per device (P1, P2, common), used to seed location-aware
# default names for the extrinsic caps and delay branches.
_DEV_LETTERS = {"Bipolar": ("B", "C", "E"), "Unipolar": ("G", "D", "S")}


def _place_pair(device: str, place: str) -> str:
    """Two-letter terminal pair for a shunt placement, e.g. ``bc`` / ``gd``."""
    p1, p2, com = _DEV_LETTERS.get(device, ("B", "C", "E"))
    return {"p1-p2": f"{p1}{p2}", "p1-gnd": f"{p1}{com}",
            "p2-gnd": f"{p2}{com}"}.get(place, f"{p1}{p2}").lower()


def _extrinsic_default(model: CustomModel, place: str, kind: str) -> str:
    """Default name for an extrinsic-cap component: ``Cbcx`` / ``Cgdx`` across
    the ports (P1↔P2), ``Cbex`` / ``Cgsx`` to ground (P1↔GND).  Non-cap parts
    fall back to the generic counter."""
    if kind != "C":
        return _default_name(model, kind)
    return f"C{_place_pair(model.device, place)}x"


def _parasitic_default(model: CustomModel, place: str, kind: str) -> str:
    """Default name for a parasitic pad-cap component: ``Cpbc`` / ``Cpgd``
    (P1↔P2), ``Cpbe`` / ``Cpgs`` (P1↔GND), ``Cpce`` / ``Cpds`` (P2↔GND).
    Non-cap parts fall back to the generic counter."""
    if kind != "C":
        return _default_name(model, kind)
    return f"Cp{_place_pair(model.device, place)}"


def _shunt_name_map(old: str, new: str) -> dict:
    """{old default cap name → new default} for extrinsic + parasitic-pad shunt
    caps across every placement — used to auto-rename on a device switch."""
    m = {}
    op1, op2, ocom = _DEV_LETTERS[old]
    np1, np2, ncom = _DEV_LETTERS[new]
    for (oa, ob), (na, nb) in zip(
            [(op1, op2), (op1, ocom), (op2, ocom)],
            [(np1, np2), (np1, ncom), (np2, ncom)]):
        o, n = f"{oa}{ob}".lower(), f"{na}{nb}".lower()
        m[f"C{o}x"] = f"C{n}x"     # extrinsic
        m[f"Cp{o}"] = f"Cp{n}"     # parasitic pad
    return m


def _delay_default(model: CustomModel, kind: str, place: str) -> str:
    """Default name for a delay/port-extra component: ``R_delay_B`` /
    ``C_delay_B`` (or ``…_G`` / ``…_D`` / ``…_S`` per device & port)."""
    p1, p2, com = _DEV_LETTERS.get(model.device, ("B", "C", "E"))
    letter = {"port1": p1, "port2": p2, "emitter": com}.get(place, p1)
    return f"{kind}_delay_{letter}"


# ── device-dependent default component names ────────────────────────────────
# Positional defaults per junction role + the access R/L names.  Switching the
# device auto-renames any element/field still carrying the *other* device's
# default (custom names are left untouched).
_DEV_NAMES = {
    "Bipolar": {
        "base": ["Rbi"], "be": ["Cbe", "Rbe"], "bc": ["Cbc", "Rbc"],
        "ce": ["Cce", "Rce"],
        "access": {"Rb": "Rb", "Lb": "Lb", "Rc": "Rc", "Lc": "Lc",
                   "Re": "Re", "Le": "Le"},
    },
    "Unipolar": {
        "base": ["Rgi"], "be": ["Cgs", "Ri"], "bc": ["Cgd", "Rgd"],
        "ce": ["Cds", "Rds"],
        "access": {"Rb": "Rg", "Lb": "Lg", "Rc": "Rd", "Lc": "Ld",
                   "Re": "Rs", "Le": "Ls"},
    },
}
# (model attribute, widget-key prefix) per junction role.
_JUNCTIONS = [("intrinsic_base", "cmb_ibase", "base"),
              ("intrinsic_be", "cmb_ibe", "be"),
              ("intrinsic_bc", "cmb_ibc", "bc"),
              ("intrinsic_ce", "cmb_ice", "ce")]


def _intrinsic_name_map(old: str, new: str) -> dict:
    """{old default name → new default name} across every junction role."""
    m = {}
    for role in ("base", "be", "bc", "ce"):
        for a, b in zip(_DEV_NAMES[old][role], _DEV_NAMES[new][role]):
            m[a] = b
    return m


def _access_default(device: str, key: str) -> str:
    return _DEV_NAMES[device]["access"][key]


def _relabel_for_device(model: CustomModel, old: str, new: str) -> None:
    """Auto-rename default-named components when the device changes.  The
    name-input widgets refresh because their keys carry ``_ver()`` and the
    caller bumps the version — so the displayed text follows the model."""
    nmap = _intrinsic_name_map(old, new)
    for attr, _prefix, _role in _JUNCTIONS:
        for el in getattr(model, attr).elements():
            if el.name in nmap:
                el.name = nmap[el.name]
    # Extrinsic + parasitic-pad shunt caps (Cbcx→Cgdx, Cpbe→Cpgs, …).
    smap = _shunt_name_map(old, new)
    for b in (model.extrinsic + model.parasitic):
        for el in b.network.elements():
            if el.name in smap:
                el.name = smap[el.name]
    # Access names: only remap the ones already seeded (non-empty) that still
    # carry the old default — leave blanks blank (they seed when reached).
    for k in list(model.access_names):
        cur = model.access_names.get(k, "")
        if cur and cur == _access_default(old, k):
            model.access_names[k] = _access_default(new, k)


# ════════════════════════════════════════════════════════════════════════════
# Reusable network (series-of-parallel) editor
# ════════════════════════════════════════════════════════════════════════════
def _network_editor(net: Network, prefix: str, model: CustomModel,
                    allow_series: bool = True, kinds=("R", "L", "C"),
                    name_fn=None) -> None:
    """Edit a Network in place. ``prefix`` namespaces the widget keys.
    ``name_fn(kind) -> str`` overrides the default name given to a freshly
    added component (used for location-aware extrinsic / delay defaults)."""
    def _new_name(k: str) -> str:
        return name_fn(k) if name_fn else _default_name(model, k)
    if not net.groups:
        st.caption("_No components yet._")
    for gi, group in enumerate(net.groups):
        with st.container(border=True):
            st.markdown(f"**Series step {gi + 1}** &nbsp; "
                        "_(components below are in parallel)_",
                        unsafe_allow_html=True)
            for el in list(group):
                c1, c2, c3 = st.columns([1, 3, 1])
                c1.markdown(f"`{el.kind}`")
                el.name = c2.text_input("name", value=el.name,
                                        key=f"{prefix}_g{gi}_e{el.id}_nm_{_ver()}",
                                        label_visibility="collapsed")
                if c3.button(_TRASH, key=f"{prefix}_g{gi}_e{el.id}_del",
                             help="Remove this component"):
                    group.remove(el)
                    st.rerun()
            bc = st.columns(len(kinds))
            for j, k in enumerate(kinds):
                if bc[j].button(f"➕ {k} (parallel)", key=f"{prefix}_g{gi}_add{k}",
                                width="stretch"):
                    group.append(Element(kind=k, name=_new_name(k)))
                    st.rerun()
    if allow_series:
        if st.button("➕ Add series step", key=f"{prefix}_addseries",
                     width="stretch"):
            net.groups.append([])
            st.rerun()
    elif not net.groups:
        if st.button("➕ Add components", key=f"{prefix}_addseries",
                     width="stretch"):
            net.groups.append([])
            st.rerun()


def _shunt_list_editor(branches: list, allowed_places: dict, prefix: str,
                       model: CustomModel) -> None:
    """Editor for a list of ShuntBranch (extrinsic / parasitic)."""
    add_cols = st.columns(len(allowed_places))
    for col, (place, label) in zip(add_cols, allowed_places.items()):
        if col.button(f"➕ {label}", key=f"{prefix}_add_{place}",
                      width="stretch"):
            # Pre-populate with a capacitor (these branches are caps) so the
            # user doesn't have to click "C" after adding the branch.
            branches.append(ShuntBranch(
                place=place,
                network=Network(groups=[[Element("C", _default_name(model, "C"))]])))
            st.rerun()
    for b in list(branches):
        with st.container(border=True):
            head = st.columns([4, 1])
            head[0].markdown(f"**{allowed_places.get(b.place, b.place)}**")
            if head[1].button(f"{_TRASH} branch", key=f"{prefix}_b{b.id}_del"):
                branches.remove(b)
                st.rerun()
            _network_editor(b.network, f"{prefix}_b{b.id}", model,
                            allow_series=True)


# ════════════════════════════════════════════════════════════════════════════
# Page
# ════════════════════════════════════════════════════════════════════════════
def render_build_ui() -> None:
    model = _model()
    stage = _stage()
    t = model.terminals()

    top = st.columns([4, 1])
    top[0].subheader("🧩 Build a custom small-signal model")
    if top[1].button("🔄 Start over", help="Discard this draft"):
        for k in [kk for kk in st.session_state if kk.startswith("cmb_")]:
            st.session_state.pop(k, None)
        st.session_state.pop(_MODEL_KEY, None)
        st.session_state.pop(_STAGE_KEY, None)
        st.rerun()

    # ── Modify an existing model — upload its .json to edit it in place ──────
    with st.expander("📂 Modify an existing model (upload a .json)",
                     expanded=False):
        st.caption("Loads the whole saved topology — device, π/T, intrinsic "
                   "junctions, port/delay extras, extrinsic & parasitic caps, "
                   "access R/L and every name — into the editor with all "
                   "sections revealed.  Edit anything, then download the result "
                   "at the bottom.")
        mod = st.file_uploader("Custom model .json", type=["json"],
                               key="cmb_modify_up", label_visibility="collapsed")
        if mod is not None:
            data = mod.getvalue()
            sig = (mod.name, len(data), hash(data))
            if st.session_state.get("cmb_modify_sig") != sig:
                try:
                    loaded = load_model(data)
                except Exception as exc:                   # noqa: BLE001
                    st.error(f"Couldn't read that model: {exc}")
                else:
                    # Name the working copy "<name>_modified" so the edited
                    # model downloads as a distinct file from the original.
                    if not loaded.name.endswith("_modified"):
                        loaded.name = f"{loaded.name}_modified"
                    # Wipe builder widget state, then install the model with
                    # every section revealed so any part can be edited.  Widget
                    # keys re-initialise from the loaded model on rerun.
                    for k in [kk for kk in st.session_state
                              if kk.startswith("cmb_") and kk != "cmb_modify_up"]:
                        st.session_state.pop(k, None)
                    st.session_state[_MODEL_KEY] = loaded
                    st.session_state[_STAGE_KEY] = 4
                    st.session_state["cmb_access_seeded"] = True
                    # Explicitly drive the Device + Intrinsic-topology radios to
                    # the uploaded model's values (their option strings *are* the
                    # session-state values) so the selection is guaranteed
                    # correct regardless of widget re-init timing.
                    st.session_state["cmb_dev"] = loaded.device
                    st.session_state["cmb_itype"] = loaded.intrinsic_type
                    st.session_state["cmb_modify_sig"] = sig
                    st.rerun()

    # Re-read in case a modify-upload just replaced the working model.
    model = _model()
    t = model.terminals()

    # ── live schematic ──────────────────────────────────────────────────────
    st.markdown("##### Live schematic")
    svg = render_schematic(model)
    st.iframe(svg, height=svg_pixel_height(svg) + 12)
    png = svg_to_png(svg, zoom=2)
    bc = st.columns(3)
    if png is not None:
        bc[0].download_button("🖼️ Download PNG", data=png,
                              file_name=f"{model.name or 'model'}.png",
                              mime="image/png", key="cmb_dl_png", width="stretch")
        copy_image_button(png, container=bc[1], label="📋 copy image")
    else:
        bc[0].caption("PNG export needs `rsvg-convert`/`cairosvg` — SVG below.")
    bc[2].download_button("⬇ Download SVG", data=svg,
                          file_name=f"{model.name or 'model'}.svg",
                          mime="image/svg+xml", key="cmb_dl_svg", width="stretch")

    st.divider()

    # ── Section 0: device + intrinsic core ──────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 1 · Device & intrinsic core")
        c1, c2, c3 = st.columns([2, 2, 2])
        model.name = c1.text_input("Model name", value=model.name, key="cmb_name")
        new_dev = c2.radio(
            "Device", ["Bipolar", "Unipolar"],
            format_func=lambda d: ("Bipolar (HBT) · B/C/E" if d == "Bipolar"
                                   else "Unipolar (HEMT) · G/D/S"),
            index=0 if model.device == "Bipolar" else 1, key="cmb_dev")
        new_type = c3.radio(
            "Intrinsic topology", ["Pi", "T"],
            format_func=lambda x: "Intrinsic π" if x == "Pi" else "Intrinsic T",
            horizontal=True,
            index=0 if model.intrinsic_type == "Pi" else 1, key="cmb_itype")
        # On a device change, auto-rename every default-named component
        # (Cbc→Cgd, Rb→Rg, …) and refresh the source label.
        if new_dev != model.device or new_type != model.intrinsic_type:
            was_default = (model.source_name ==
                           _default_source_name(model.intrinsic_type, model.device))
            old_dev = model.device
            model.device, model.intrinsic_type = new_dev, new_type
            if old_dev != new_dev:
                _relabel_for_device(model, old_dev, new_dev)
            if was_default or not model.source_name:
                model.source_name = _default_source_name(new_type, new_dev)
            _bump_namever()        # force every name input to follow the model
            st.rerun()

        # ── Intuitive one-part-at-a-time editor (illustration | controls) ───
        part_keys = ["base", "be", "bc", "ce", "src"]
        part_opts = {
            "base": f"⬚ Base ({t['p1']})",
            "be":   f"⬚ {t['p1']}–{t['com']}",
            "bc":   f"⬚ {t['p1']}–{t['p2']}",
            "ce":   f"⬚ {t['p2']}–{t['com']}",
            "src":  "◇ Source",
        }
        ill, ctrl = st.columns([5, 6])
        sel = _cur_part("cmb_isel", part_keys, "be")
        with ill:
            st.iframe(intrinsic_thumbnail(model, sel), height=380)
        ctrl_box = ctrl.container()
        with ctrl_box:
            sel = _part_chips(part_opts, part_keys, "cmb_isel", "be")
            if sel == "base":
                st.markdown(f"**Base spreading** — series network into node "
                            f"**{t['p1']}** (e.g. {t['p1']}-spreading R).")
                _network_editor(model.intrinsic_base, "cmb_ibase", model)
            elif sel == "be":
                st.markdown(f"**{t['p1']}–{t['com']} junction** "
                            f"_({t['p1_long']}–{t['com_long']})_",
                            unsafe_allow_html=True)
                _network_editor(model.intrinsic_be, "cmb_ibe", model)
            elif sel == "bc":
                st.markdown(f"**{t['p1']}–{t['p2']} junction** "
                            f"_({t['p1_long']}–{t['p2_long']})_",
                            unsafe_allow_html=True)
                _network_editor(model.intrinsic_bc, "cmb_ibc", model)
            elif sel == "ce":
                st.markdown(f"**{t['p2']}–{t['com']} output** "
                            f"_({t['p2_long']}–{t['com_long']}, optional)_",
                            unsafe_allow_html=True)
                _network_editor(model.intrinsic_ce, "cmb_ice", model)
            else:  # src
                model.source_name = st.text_input(
                    "Controlled-source label", value=model.source_name,
                    key=f"cmb_srcnm_{_ver()}")

        if stage == 0:
            if st.button("✅ Confirm intrinsic core", type="primary"):
                _set_stage(1)
                st.rerun()

    if stage < 1:
        st.info("Confirm the intrinsic core to add the delay / port branches.")
        return

    # ── Section 2: extrinsic caps ───────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 2 · Extrinsic capacitances")
        e_keys = ["p1-p2", "p1-gnd"]
        e_opts = {"p1-p2": "⬚ P1 ↔ P2", "p1-gnd": "⬚ P1 ↔ GND"}
        ill, ctrl = st.columns([5, 6])
        esel = _cur_part("cmb_extsel", e_keys, "p1-p2")
        with ill:
            st.iframe(section_thumbnail(model, "extrinsic", esel), height=404)
        with ctrl.container():
            esel = _part_chips(e_opts, e_keys, "cmb_extsel", "p1-p2")
            st.markdown(f"**{e_opts[esel][2:]}** capacitance")
            _network_editor(_place_net(model.extrinsic, esel),
                            f"cmb_ext_{esel}", model,
                            name_fn=lambda k: _extrinsic_default(model, esel, k))
        if stage == 1:
            if st.button("✅ Done — add delay / port extras", type="primary"):
                _set_stage(2)
                st.rerun()

    if stage < 2:
        return

    # ── Section 3: delay / port extras ──────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 3 · Delay / port extras")
        d_keys = ["port1", "port2", "emitter"]
        d_opts = {"port1": f"⬚ Port 1 ({t['p1']})",
                  "port2": f"⬚ Port 2 ({t['p2']})",
                  "emitter": f"⬚ {t['com']} delay"}
        ill, ctrl = st.columns([5, 6])
        dsel = _cur_part("cmb_dsel", d_keys, "port1")
        with ill:
            st.iframe(section_thumbnail(model, "delay", dsel), height=404)
        with ctrl.container():
            dsel = _part_chips(d_opts, d_keys, "cmb_dsel", "port1")
            if dsel == "port1":
                st.markdown(f"**Port 1 ({t['p1_long']} side)** — series extras.")
                _network_editor(model.port1, "cmb_p1", model,
                                name_fn=lambda k: _delay_default(model, k, "port1"))
            elif dsel == "port2":
                st.markdown(f"**Port 2 ({t['p2_long']} side)** — series extras.")
                _network_editor(model.port2, "cmb_p2", model,
                                name_fn=lambda k: _delay_default(model, k, "port2"))
            else:
                st.markdown(f"**Common ({t['com_long']}) delay branch** &nbsp; "
                            "_(between the intrinsic source and access Re/Le — "
                            "e.g. the Kun-Yang R_delay∥C_delay above Rs)_",
                            unsafe_allow_html=True)
                _network_editor(model.emitter, "cmb_emit", model,
                                name_fn=lambda k: _delay_default(model, k, "emitter"))
        if stage == 2:
            if st.button("✅ Done — add access R / lead L", type="primary"):
                _set_stage(3)
                st.rerun()

    if stage < 3:
        return

    # ── Section 4: access R + lead L ────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 4 · Access resistance + lead inductance")
        # Seed device-appropriate default names the first time this section is
        # reached (kept blank earlier so the schematic shows intrinsic-only).
        if not st.session_state.get("cmb_access_seeded"):
            for k in model.access_names:
                if not model.access_names.get(k):
                    model.access_names[k] = _access_default(model.device, k)
            st.session_state["cmb_access_seeded"] = True

        def _access_cell(col, key, unit):
            """One access element: name input + delete, or a re-add button."""
            if model.access_names.get(key):
                ic, dc = col.columns([5, 1])
                model.access_names[key] = ic.text_input(
                    f"{key} ({unit})", value=model.access_names.get(key, ""),
                    key=f"cmb_acc_{key}_{_ver()}")
                if dc.button(_TRASH, key=f"cmb_acc_del_{key}",
                             help=f"Delete {key}"):
                    model.access_names[key] = ""
                    _bump_namever()
                    st.rerun()
            else:
                if col.button(f"➕ add {_access_default(model.device, key)}",
                              key=f"cmb_acc_add_{key}", width="stretch"):
                    model.access_names[key] = _access_default(model.device, key)
                    _bump_namever()
                    st.rerun()

        ill, ctrl = st.columns([5, 6])
        with ill:
            st.iframe(section_thumbnail(model, "access"), height=404)
        with ctrl.container():
            rows = [(t["p1_long"].capitalize(), "Rb", "Lb"),
                    (t["p2_long"].capitalize(), "Rc", "Lc"),
                    (t["com_long"].capitalize(), "Re", "Le")]
            for lbl, rk, lk in rows:
                c = st.columns([1, 2, 2])
                c[0].markdown(f"**{lbl}**")
                _access_cell(c[1], rk, "series R")
                _access_cell(c[2], lk, "series L")
        if stage == 3:
            if st.button("✅ Done — add parasitic caps", type="primary"):
                _set_stage(4)
                st.rerun()

    if stage < 4:
        return

    # ── Section 5: parasitic caps ───────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 5 · Parasitic pad capacitances")
        pa_keys = ["p1-p2", "p1-gnd", "p2-gnd"]
        pa_opts = {"p1-p2": "⬚ P1 ↔ P2", "p1-gnd": "⬚ P1 ↔ GND",
                   "p2-gnd": "⬚ P2 ↔ GND"}
        ill, ctrl = st.columns([5, 6])
        psel = _cur_part("cmb_parsel", pa_keys, "p1-p2")
        with ill:
            st.iframe(section_thumbnail(model, "parasitic", psel), height=404)
        with ctrl.container():
            psel = _part_chips(pa_opts, pa_keys, "cmb_parsel", "p1-p2")
            st.markdown(f"**{pa_opts[psel][2:]}** pad capacitance")
            _network_editor(_place_net(model.parasitic, psel),
                            f"cmb_par_{psel}", model,
                            name_fn=lambda k: _parasitic_default(model, psel, k))

    # ── Save (download the .json, or send straight to the Load/Fit views) ────
    st.divider()
    st.markdown("#### 💾 Save / use model")
    st.caption("Download the topology as a `.json`, or send it straight to the "
               "**Load** (simulate) and **Fit-to-device** views — no re-upload "
               "needed; just switch to them.")
    sd = st.columns(2)
    sd[0].download_button(
        "⬇ Download .json", data=model_to_json(model),
        file_name=f"{model.name or 'model'}.json", mime="application/json",
        type="primary", width="stretch")
    if sd[1].button("📤 Send to Load / Fit views", width="stretch",
                    help="Load this model into the Load (simulate) and "
                         "Fit-to-device views, switch to them, and download "
                         "its .json"):
        from .ui_fit import install_fit_model
        js = model_to_json(model)
        st.session_state["cmu_model"] = load_model(js)
        st.session_state.pop("cmu_base", None)
        install_fit_model(load_model(js))
        # Defer the download to the destination view (this rerun navigates away)
        # and signal the host page to switch its radio to Load/Fit.
        st.session_state["cm_pending_download"] = (
            f"{model.name or 'model'}.json", js)
        st.session_state["cm_nav_to_loadfit"] = True
        st.rerun()

"""PartForge - two photos of a broken part -> measured, editable, printable CAD.

Run: .venv\\Scripts\\streamlit.exe run app.py
Flow: upload photos -> 3D viewer -> click a dimension -> edit live -> STL/STEP.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import streamlit as st

from partforge import cad, meta_api
from partforge.card import render_card
from partforge.fusion import fuse
from partforge.measure import annotate, measure_photo
from partforge.spec import Bend, Hole, PartSpec, Slot, rectangle_spec
from partforge.synthetic import make_synthetic_bracket_side, \
    make_synthetic_photo, make_synthetic_side
from partforge.viewer import part_viewer
from partforge.voice_edit import apply_edit, parse_edit_text

st.set_page_config(page_title="PartForge", layout="wide", page_icon="🔩")

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("spec_json", None)
ss.setdefault("selected", None)
ss.setdefault("views_png", {})
ss.setdefault("photo_names", {})
ss.setdefault("pending", {})   # role -> raw bytes staged for processing
ss.setdefault("log", [])


def log(msg: str) -> None:
    ss.log = (ss.log + [msg])[-6:]


def get_spec() -> PartSpec | None:
    return PartSpec(**ss.spec_json) if ss.spec_json else None


def clear_inputs(prefixes=("hole_", "slot_", "ans_", "dim_", "view_")):
    for k in list(ss.keys()):
        if k.startswith(prefixes):
            del ss[k]


def set_spec(spec: PartSpec) -> None:
    ss.spec_json = json.loads(spec.model_dump_json())
    clear_inputs()


def save(spec: PartSpec) -> None:
    ss.spec_json = json.loads(spec.model_dump_json())


@st.cache_data(show_spinner=False)
def build_exports(spec_json: str, tol: float):
    spec = PartSpec(**json.loads(spec_json))
    return (cad.export_bytes(spec, "stl", tol),
            cad.export_bytes(spec, "step", tol),
            cad.top_view_png(spec))


def process_photo(img_bgr: np.ndarray, name: str, thresh: int,
                  role: str = "top") -> None:
    """role: top | side | front. Top replaces the master; sides fuse into it."""
    spec, rect, ppm, info = measure_photo(img_bgr, thresh=thresh)
    if spec is None:
        st.error(f"Could not measure '{name}' ({role}): "
                 f"{info.get('error', 'unknown')} "
                 f"(markers found: {info.get('card_markers')}). "
                 "Need all 4 card markers visible.")
        return
    annotated = annotate(rect, spec, ppm)
    ok, buf = cv2.imencode(".png", annotated)
    ss.views_png[role] = buf.tobytes() if ok else None
    ss.photo_names[role] = name
    if role == "top":
        set_spec(spec)
        ss.selected = None
        log(f"Top: {spec.summary()}")
    else:
        master = get_spec()
        if master is None:
            set_spec(spec)  # side-only edge case: at least show something
            log(f"{role}: no top yet — showing it as master for now")
            return
        reports = fuse(master, {role: spec})
        clear_inputs(("ans_",))  # questions changed; keep dimension edits
        save(master)
        log(f"{role}: {reports.get(role, 'fused')}")


def cb_hole(hid: str, field: str):
    spec = get_spec()
    val = ss[f"hole_{hid}_{field}"]
    for h in spec.holes:
        if h.id == hid:
            setattr(h, field, float(val))
            h.confidence = 1.0
    save(spec)


def cb_slot(sid: str, field: str):
    spec = get_spec()
    val = ss[f"slot_{sid}_{field}"]
    for s in spec.slots:
        if s.id == sid:
            setattr(s, field, float(val))
            s.confidence = 1.0
    save(spec)


def cb_thickness():
    spec = get_spec()
    spec.thickness_mm = float(ss["dim_thickness"])
    save(spec)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔩 PartForge")
    st.caption("Photos in → measured, printable CAD out. "
               "Geometry is measured, never hallucinated.")
    cfg = meta_api.configured()
    st.caption("Meta APIs: " + ", ".join(
        f"{k}={'on' if v else 'off'}" for k, v in cfg.items()))

    top_file = st.file_uploader("Top view — part lying flat on the card",
                                type=["jpg", "jpeg", "png"], key="up_top")
    side_file = st.file_uploader("Side view — standing on its long edge",
                                 type=["jpg", "jpeg", "png"], key="up_side")
    front_file = st.file_uploader("Front view — standing on short edge (or other leg)",
                                  type=["jpg", "jpeg", "png"], key="up_front")
    thresh = st.slider("Segmentation threshold (fallback)", 120, 250, 210,
                       help="Lower if the part is missed, higher if card text is picked up.")
    tol = st.slider("Print hole tolerance (mm)", 0.0, 0.4, 0.2, 0.05,
                    help="Extra clearance so printed holes fit real bolts.")

    c1, c2 = st.columns(2)
    if c1.button("▶ Demo part", use_container_width=True):
        process_photo(make_synthetic_photo(), "synthetic demo top", thresh, "top")
        process_photo(make_synthetic_side("W"), "synthetic demo side", thresh, "side")
        process_photo(make_synthetic_side("H"), "synthetic demo front", thresh, "front")
        st.rerun()
    if c1.button("▶ Demo bracket", use_container_width=True):
        process_photo(make_synthetic_photo(), "synthetic plate top", thresh, "top")
        process_photo(make_synthetic_bracket_side(), "synthetic L side", thresh, "side")
        st.rerun()
    card_png = cv2.imencode(".png", render_card())[1].tobytes()
    c2.download_button("⬇ Print card", card_png, "partforge_card.png",
                       use_container_width=True,
                       help="Print at exactly 210x148 mm (100% scale).")

    with st.expander("No photo? Start from a blank plate"):
        bw = st.number_input("Width mm", 5.0, 500.0, 80.0)
        bh = st.number_input("Height mm", 5.0, 500.0, 40.0)
        bt = st.number_input("Thickness mm", 0.5, 50.0, 3.0)
        if st.button("Create blank plate"):
            set_spec(rectangle_spec(bw, bh, bt))
            ss.views_png = {}
            st.rerun()

    for line in ss.log:
        st.caption("• " + line)


def _stage(upload, role: str) -> None:
    """Stage a photo; processing happens only on the Measure button."""
    if upload is None:
        return
    data = upload.getvalue()
    if ss.pending.get(role) != data:
        ss.pending[role] = data
        ss.photo_names[role] = upload.name


_stage(top_file, "top")
_stage(side_file, "side")
_stage(front_file, "front")

# Measure button lives in its own sidebar block AFTER staging, so it
# appears on the same run the photo is attached.
with st.sidebar:
    st.markdown("---")
    st.caption("**Step 1** — attach photos above (they stage here, nothing runs yet).")
    staged = [r for r in ("top", "side", "front") if ss.pending.get(r)]
    if staged:
        st.caption("Staged: " + ", ".join(
            f"{r} ({len(ss.pending[r]) // 1024} KB)" for r in staged))
        if st.button("🔨 Measure → build CAD", type="primary", use_container_width=True,
                     help="Top becomes the master spec; side/front fuse into it."):
            with st.spinner("Measuring… (rectify → segment → measure → fuse)"):
                for role in staged:
                    img = cv2.imdecode(
                        np.frombuffer(ss.pending[role], np.uint8), cv2.IMREAD_COLOR)
                    process_photo(img, ss.photo_names.get(role, f"{role}.jpg"),
                                  thresh, role)
                    del ss.pending[role]
            st.rerun()
        if st.button("Clear staged photos"):
            ss.pending = {}
            st.rerun()
    else:
        st.caption("No photos staged yet — attach top/side/front above.")

spec = get_spec()

# ---------------------------------------------------------------- main
if spec is None:
    st.header("Photograph the part. Get a printable part back.")
    st.markdown(
        "1. Print the calibration card (sidebar), place the part on it.\n"
        "2. **Top**: part lying flat → outline + holes + slots.\n"
        "3. **Side / front**: part standing on edge → thickness vote; "
        "an L-profile proposes a bracket bend.\n"
        "4. Click a marker in the 3D viewer, fix dimensions, export **STL** or **STEP**.\n\n"
        "Scope: flat plates and single-bend brackets with holes/slots. "
        "Thickness comes from side views or your caliper — never from the top photo.\n\n"
        "No part handy? Hit **▶ Demo part** or **▶ Demo bracket** in the sidebar.")
    st.stop()

st.subheader(f"Measured part — {spec.summary()}")
spec_json = json.dumps(ss.spec_json, sort_keys=True)
stl_bytes, step_bytes, diagram_png = build_exports(spec_json, tol)

left, right = st.columns([1.55, 1.0])

with left:
    clicked = None
    try:
        clicked = part_viewer(stl_bytes, spec, ss.selected)
    except Exception as e:  # viewer is an enhancement; panel below always works
        st.warning(f"3D viewer unavailable ({e.__class__.__name__}). "
                   "Use the dimension diagram + panel on the right.")
    if clicked and clicked != ss.selected:
        ss.selected = clicked
        st.rerun()
    st.image(diagram_png, caption="Dimension diagram — numbers match the panel on the right")
    for role in ("top", "side", "front"):
        if ss.views_png.get(role):
            with st.expander(f"{role.capitalize()} view (rectified, annotated) — "
                             f"{ss.photo_names.get(role, '')}"):
                st.image(ss.views_png[role])
                rep = (get_spec().view_reports or {}).get(role)
                if rep:
                    st.caption(rep)

with right:
    # ----- asks
    questions = spec.questions()
    if questions:
        st.markdown("### ❓ PartForge asks")
        for q in questions:
            st.warning(q.text)
            if q.kind == "choice" and q.choices:
                cols = st.columns(len(q.choices) + 1)
                for i, ch in enumerate(q.choices):
                    if cols[i].button(f"{ch:.1f} mm", key=f"ans_{q.id}_{ch}"):
                        spec.apply_answer(q, float(ch))
                        save(spec)
                        st.rerun()
                val = cols[-1].number_input("other", 0.5, 100.0,
                                            float(q.suggested or 3.0),
                                            key=f"ans_{q.id}_custom", label_visibility="collapsed")
                if st.button("Use other", key=f"ans_{q.id}_use"):
                    spec.apply_answer(q, float(val)); save(spec); st.rerun()
            else:
                val = st.number_input("Value (mm)", 0.1, 200.0,
                                      float(q.suggested or 3.0), key=f"ans_{q.id}")
                if st.button("Apply", key=f"ans_{q.id}_apply"):
                    spec.apply_answer(q, float(val)); save(spec); st.rerun()

    # ----- dimensions
    st.markdown("### 📐 Dimensions (live)")
    st.caption("Click a marker in the 3D viewer, or pick a dimension here. "
               "The model rebuilds on every change.")
    d1, d2, d3 = st.columns(3)
    d1.metric("Width", f"{spec.width_mm:.1f} mm")
    d2.metric("Height", f"{spec.height_mm:.1f} mm")
    tv = spec.thickness_votes
    tlabel = (f"{spec.thickness_mm:.1f} mm"
              + (f" (votes {', '.join(f'{v:.1f}' for v in tv)})" if tv else "")
              if spec.thickness_mm is not None else "—")
    d3.metric("Thickness", tlabel)

    nw = d1.number_input("Set width", 1.0, 1000.0, float(round(spec.width_mm, 1)), key="resize_w")
    nh = d2.number_input("Set height", 1.0, 1000.0, float(round(spec.height_mm, 1)), key="resize_h")
    if st.button("Rescale outline to width/height"):
        spec.rescale_outline(float(nw), float(nh)); save(spec); st.rerun()

    st.number_input("Thickness (mm) — caliper value", 0.4, 50.0,
                    float(spec.thickness_mm or 3.0), 0.1,
                    key="dim_thickness", on_change=cb_thickness)

    for h in spec.holes:
        sel = ss.selected == h.id
        if st.button(f"{'🔴' if sel else '⚪'} {h.id} — Ø{h.d:.1f} mm at ({h.x:.1f}, {h.y:.1f})",
                     key=f"pick_{h.id}", use_container_width=True):
            ss.selected = h.id; st.rerun()
        if sel:
            a, b, c = st.columns(3)
            a.number_input("x", 0.0, 1000.0, float(h.x), 0.5, key=f"hole_{h.id}_x",
                           on_change=cb_hole, args=(h.id, "x"))
            b.number_input("y", 0.0, 1000.0, float(h.y), 0.5, key=f"hole_{h.id}_y",
                           on_change=cb_hole, args=(h.id, "y"))
            c.number_input("Ø mm", 0.5, 100.0, float(h.d), 0.1, key=f"hole_{h.id}_d",
                           on_change=cb_hole, args=(h.id, "d"))
            if st.button(f"Delete {h.id}", key=f"del_{h.id}"):
                spec.holes = [x for x in spec.holes if x.id != h.id]
                ss.selected = None; save(spec); st.rerun()

    for s in spec.slots:
        sel = ss.selected == s.id
        if st.button(f"{'🔴' if sel else '🟠'} {s.id} — {s.length:.1f}x{s.width:.1f} mm at ({s.cx:.1f}, {s.cy:.1f})",
                     key=f"pick_{s.id}", use_container_width=True):
            ss.selected = s.id; st.rerun()
        if sel:
            a, b, c, e = st.columns(4)
            a.number_input("cx", 0.0, 1000.0, float(s.cx), 0.5, key=f"slot_{s.id}_cx",
                           on_change=cb_slot, args=(s.id, "cx"))
            b.number_input("cy", 0.0, 1000.0, float(s.cy), 0.5, key=f"slot_{s.id}_cy",
                           on_change=cb_slot, args=(s.id, "cy"))
            c.number_input("len", 1.0, 500.0, float(s.length), 0.5, key=f"slot_{s.id}_length",
                           on_change=cb_slot, args=(s.id, "length"))
            e.number_input("wid", 0.5, 100.0, float(s.width), 0.1, key=f"slot_{s.id}_width",
                           on_change=cb_slot, args=(s.id, "width"))

    # ----- bracket
    st.markdown("#### Bracket")
    is_bracket = st.checkbox("This part has a bend", value=(spec.type == "bracket"),
                             key="view_is_bracket")
    if is_bracket and spec.type != "bracket":
        spec.type = "bracket"
        if spec.bend is None:
            spec.bend = Bend(position_mm=round(spec.width_mm / 2, 1))
        save(spec); st.rerun()
    if not is_bracket and spec.type != "plate":
        spec.type = "plate"
        save(spec); st.rerun()
    if spec.type == "bracket":
        b = spec.bend or Bend(position_mm=round(spec.width_mm / 2, 1))
        spec.bend = b
        sel_b = ss.selected == "bend"
        if st.button(f"{'🔴' if sel_b else '🟣'} Bend — {b.angle_deg:.0f}° at {b.position_mm:.1f} mm",
                     key="pick_bend", use_container_width=True):
            ss.selected = "bend"; st.rerun()
        if sel_b or True:
            p1, p2, p3 = st.columns(3)
            pos = p1.number_input("position mm", 0.5, 2000.0, float(b.position_mm), 0.5,
                                  key="dim_bend_pos")
            ang = p2.number_input("angle °", 15.0, 175.0, float(b.angle_deg), 1.0,
                                  key="dim_bend_ang")
            ax = p3.selectbox("bend line ∥", ["y", "x"],
                              index=0 if b.axis == "y" else 1, key="dim_bend_ax")
            if st.button("Apply bend", key="dim_bend_apply"):
                b.position_mm, b.angle_deg, b.axis = float(pos), float(ang), ax
                try:
                    cad.build_solid(spec)
                    b.confidence = 1.0
                    save(spec); st.rerun()
                except ValueError as e:
                    st.error(str(e))
        note = getattr(cad.build_solid(spec), "_partforge_note", "")
        if note and "1 on folded" not in note and "0 on folded" not in note:
            st.caption(note)

    ac1, ac2 = st.columns(2)
    if ac1.button("＋ Add hole"):
        nid = f"H{len(spec.holes) + 1}"
        while any(h.id == nid for h in spec.holes):
            nid += "x"
        spec.holes.append(Hole(id=nid, x=spec.width_mm / 2, y=spec.height_mm / 2, d=5.0))
        ss.selected = nid; save(spec); st.rerun()
    if ac2.button("＋ Add slot"):
        nid = f"S{len(spec.slots) + 1}"
        spec.slots.append(Slot(id=nid, cx=spec.width_mm / 2, cy=spec.height_mm / 2,
                               length=18.0, width=5.0))
        ss.selected = nid; save(spec); st.rerun()

    # ----- voice / typed correction
    st.markdown("### 🎙️ Correct it hands-free")
    edit_text = st.text_input("Say or type: 'hole 2 is 5, not 4' / 'thickness 4'",
                              key="edit_text")
    audio = st.audio_input("Record a correction")
    if st.button("Apply correction") and edit_text:
        edit = parse_edit_text(edit_text, spec)
        if edit:
            log(apply_edit(spec, edit)); save(spec); st.rerun()
        else:
            st.error("I could not parse that. Try 'hole 2 is 5' or 'thickness 4'.")
    if audio is not None:
        text = meta_api.transcribe(audio.getvalue())
        if text:
            st.info(f"Heard: {text}")
            edit = parse_edit_text(text, spec)
            if edit:
                log(apply_edit(spec, edit)); save(spec); st.rerun()
        else:
            st.caption("Voice transcription is off (Muse Voice API not configured) — "
                       "typed correction above works now.")

    # ----- export
    st.markdown("### ⬇ Export")
    e1, e2 = st.columns(2)
    e1.download_button("STL — print it", stl_bytes, "partforge_part.stl",
                       use_container_width=True)
    e2.download_button("STEP — edit it", step_bytes, "partforge_part.step",
                       use_container_width=True)
    with st.expander("Part spec (JSON)"):
        st.code(json.dumps(ss.spec_json, indent=1), language="json")

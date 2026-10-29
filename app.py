"""PartForge - two photos of a broken part -> measured, editable, printable CAD.

Run: .venv\\Scripts\\streamlit.exe run app.py
Flow: upload photos -> 3D viewer -> click a dimension -> edit live -> STL/STEP.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import streamlit as st

from partforge import cad, meta_api, ui
from partforge.fusion import fuse
from partforge.measure import annotate, measure_photo
from partforge.spec import Bend, Hole, PartSpec, Slot, rectangle_spec
from partforge.synthetic import make_synthetic_bracket_side, \
    make_synthetic_photo, make_synthetic_side
from partforge.viewer import part_viewer
from partforge.voice_edit import apply_edit, parse_edit_text

st.set_page_config(page_title="PartForge", layout="wide", page_icon="🔩",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("spec_json", None)
ss.setdefault("selected", None)
ss.setdefault("views_png", {})
ss.setdefault("views_mask", {})
ss.setdefault("photo_names", {})
ss.setdefault("pending", {})   # role -> raw bytes staged for processing
ss.setdefault("originals", {})  # role -> original jpg bytes (tap re-measure)
ss.setdefault("tap_mode", {})   # role -> True when tap-to-select is active
ss.setdefault("staged_side", {})  # side/front specs waiting for a top
ss.setdefault("errors", {})    # role -> last measurement error (persistent)
ss.setdefault("pf_hist", [])   # undo stack of spec_json
ss.setdefault("log", [])
ss.setdefault("pf_tol", 0.2)


def log(msg: str) -> None:
    ss.log = (ss.log + [msg])[-6:]


def get_spec() -> PartSpec | None:
    j = ss.get("spec_json")
    return PartSpec(**j) if j else None


def clear_inputs(prefixes=("hole_", "slot_", "ans_", "dim_", "view_")):
    for k in list(ss.keys()):
        if k.startswith(prefixes):
            del ss[k]


def set_spec(spec: PartSpec) -> None:
    ss.spec_json = json.loads(spec.model_dump_json())
    clear_inputs()


def save(spec: PartSpec) -> None:
    ss.pf_hist = (ss.pf_hist + [ss.spec_json])[-20:] if ss.spec_json else ss.pf_hist
    ss.spec_json = json.loads(spec.model_dump_json())


@st.cache_data(show_spinner=False)
def build_exports(spec_json: str, tol: float):
    spec = PartSpec(**json.loads(spec_json))
    return (cad.export_bytes(spec, "stl", tol),
            cad.export_bytes(spec, "step", tol),
            cad.top_view_png(spec))


def process_photo(img_bgr: np.ndarray, name: str, thresh: int,
                  role: str = "top") -> None:
    """role: top | front. Top gives profile+holes+slots; front gives height.

    One SAM call per photo (cached by sha1 so edits never re-call).
    No taps, no thresholds — SAM text prompts only.
    Returns True on success, False on failure (error is stored in
    ss.errors[role] so it survives reruns instead of flashing away).
    """
    import hashlib
    from partforge.pipeline import measure_view
    ok, obuf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        ss.errors[role] = "could not decode photo"
        return False
    key = hashlib.sha1(obuf.tobytes()).hexdigest()
    cached = ss.sam_cache.get(key)
    if cached is None:
        try:
            spec, rect, ppm, info = measure_view(img_bgr, role)
        except RuntimeError as e:
            spec, rect, ppm, info = None, None, 0.0, {"error": str(e)}
        cached = {"spec": spec.model_dump() if spec else None,
                  "rect": rect.tolist() if rect is not None else None,
                  "ppm": ppm, "info": info}
        ss.sam_cache[key] = cached
    else:
        info = cached["info"]
        ppm = cached["ppm"]
        rect = np.array(cached["rect"], dtype=np.uint8) \
            if cached["rect"] is not None else None
        spec = PartSpec(**cached["spec"]) if cached["spec"] else None
    if spec is None:
        err = info.get("error", "unknown")
        # keep the forensic layers even on failure (mask/rect explain why)
        if info.get("rect") is not None:
            ok, buf = cv2.imencode(".png", info["rect"])
            ss.views_png[role] = buf.tobytes() if ok else None
        if info.get("mask") is not None:
            okm, mbuf = cv2.imencode(".png", info["mask"])
            ss.views_mask[role] = mbuf.tobytes() if okm else None
        ss.photo_names[role] = name
        ss.errors[role] = err
        # keep originals even on failure: tap-to-select needs them
        ok_o, obuf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok_o:
            ss.originals[role] = obuf.tobytes()
        if role == "top":
            # a failed top leaves nothing to build on: clear stale CAD
            ss.pop("spec_json", None)
            ss.views_png.pop("top", None)
            ss.views_mask.pop("top", None)
            ss.selected = None
        # a failed side/front keeps the top master; it just doesn't fuse
        log(f"{role}: FAILED — {err[:100]}")
        return False
    ss.errors.pop(role, None)
    ss.originals[role] = obuf.tobytes()
    annotated = annotate(rect, spec, ppm)
    ok, buf = cv2.imencode(".png", annotated)
    ss.views_png[role] = buf.tobytes() if ok else None
    mask = info.get("mask")
    if mask is not None:
        okm, mbuf = cv2.imencode(".png", np.array(mask, dtype=np.uint8))
        ss.views_mask[role] = mbuf.tobytes() if okm else None
    ss.photo_names[role] = name
    log(f"{role}: {info.get('note', 'measured')} "
        f"[SAM3:{info.get('backend', '?')}]")
    if role == "top":
        set_spec(spec)
        ss.selected = None
        log(f"Top: {spec.summary()}")
        # fuse any side/front views that arrived before the top
        pending = dict(ss.get("staged_side", {}))
        if pending:
            reports = fuse(spec, pending)
            save(spec)
            ss.staged_side = {}
            for r, rep in reports.items():
                if r != "top":
                    log(f"{r}: {rep}")
    else:
        master = get_spec()
        if master is None:
            # side before any top: stage it, don't pretend it's the master.
            # it fuses automatically once a top succeeds (see below).
            ss.staged_side[role] = spec
            log(f"{role}: staged — run a top view to fuse it in")
            return True
        reports = fuse(master, {role: spec})
        clear_inputs(("ans_",))  # questions changed; keep dimension edits
        save(master)
        log(f"{role}: {reports.get(role, 'fused')}")
    return True


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


# ---------------------------------------------------------------- menu + steps FIRST
# Photoshop-style: title bar is the first thing on the page, menus touching.
spec = get_spec()
doc = ss.photo_names.get("top") or "untitled part"
if spec is not None and not ss.views_png:
    doc += " (manual)"
step_idx = 0 if spec is None else 1
ui.menubar(doc, step_idx, spec is not None)

# ---------------------------------------------------------------- Capture panel
# Collapsible Photoshop-panel style. Collapses itself after a successful
# Measure so the menus stay near the top of the viewport.
cap_open = ss.get("cap_open", True)
cap_label = "➖ ① Capture — photos in" if cap_open else "➕ ① Capture — photos in"
if st.button(cap_label, use_container_width=True, key="cap_toggle"):
    ss.cap_open = not cap_open
    st.rerun()

if cap_open:
    st.markdown("<div class='pf-cap'>", unsafe_allow_html=True)
    cu1, cu2, cu3 = st.columns(3)
    with cu1:
        up_top = st.file_uploader("Top — lying flat (required)",
                                  type=["jpg", "jpeg", "png"], key="up_top")
    with cu2:
        up_side = st.file_uploader("Side — on long edge (thickness)",
                                   type=["jpg", "jpeg", "png"], key="up_side")
    with cu3:
        up_front = st.file_uploader("Front — on short edge (thickness)",
                                    type=["jpg", "jpeg", "png"], key="up_front")
    for role, up in (("top", up_top), ("side", up_side), ("front", up_front)):
        if up is not None and ss.pending.get(role) != up.getvalue():
            ss.pending[role] = up.getvalue()
            ss.photo_names[role] = up.name
    staged = [r for r in ("top", "side", "front") if ss.pending.get(r)]
    co1, co2, co3 = st.columns([1.4, 1.1, 1.5])
    with co1:
        auto_t = st.checkbox("Auto threshold (Otsu)", value=True,
                             key="auto_thresh",
                             help="Per-photo threshold from the card background. "
                                  "Uncheck to set manually.")
        if auto_t:
            thresh = 0
        else:
            thresh = st.slider("Segmentation threshold", 60, 250, 150,
                               help="Lower if the part is missed, higher if card text is picked up.")
    with co2:
        tol = st.slider("Print hole tolerance (mm)", 0.0, 0.4, 0.2, 0.05,
                        help="Extra clearance so printed holes fit real bolts.")
        ss.pf_tol = tol
    with co3:
        st.write("")
        st.write("")
        if staged:
            if st.button("🔨 Measure → build CAD", type="primary",
                         use_container_width=True):
                with st.spinner("Measuring… (rectify → segment → measure → fuse)"):
                    if "top" not in staged:
                        st.error("Top view is required — attach the flat-lay photo.")
                    else:
                        for role in staged:
                            img = cv2.imdecode(
                                np.frombuffer(ss.pending[role], np.uint8),
                                cv2.IMREAD_COLOR)
                            process_photo(img, ss.photo_names.get(role, f"{role}.jpg"),
                                          thresh, role)
                            del ss.pending[role]
                        ss.cap_open = False  # collapse capture: menus on top
                        st.rerun()
            if st.button("Clear staged photos", use_container_width=True):
                ss.pending = {}
                st.rerun()
        else:
            st.caption("Attach photos above, or try a synthetic demo:")
            dd1, dd2, dd3 = st.columns(3)
            if dd1.button("▶ Plate", use_container_width=True, key="demo_plate"):
                process_photo(make_synthetic_photo(), "synthetic top", thresh, "top")
                process_photo(make_synthetic_side("W"), "synthetic side", thresh, "side")
                process_photo(make_synthetic_side("H"), "synthetic front", thresh, "front")
                ss.cap_open = False
                st.rerun()
            if dd2.button("▶ Bracket", use_container_width=True, key="demo_bracket"):
                process_photo(make_synthetic_photo(), "synthetic top", thresh, "top")
                process_photo(make_synthetic_bracket_side(), "synthetic L", thresh, "side")
                ss.cap_open = False
                st.rerun()
            if dd3.button("▢ Blank plate", use_container_width=True, key="demo_blank"):
                set_spec(rectangle_spec(80, 40, 3.0))
                ss.views_png = {}
                ss.cap_open = False
                log("Blank 80x40x3 plate created — edit freely")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
else:
    thresh = 210
    tol = ss.get("pf_tol", 0.2)

# ---------------------------------------------------------------- persistent error panel
# process_photo stores failures in ss.errors; this panel survives reruns.
# Rectified/mask layers render INLINE here — the Review tab doesn't exist
# when top failed, so View-menu gating can't apply to forensics.
# ss.resegment (set of roles) opens the same tap picker for views that
# measured but produced a wrong shape.
tap_roles = ([r for r in ss.errors] +
             [r for r in getattr(ss, "resegment", set()) if r not in ss.errors])
def tap_resegment_panel(role: str, thresh: float, key_suffix: str = "") -> None:
    """Rectified photo + mask + tap picker + re-measure for one view."""
    ks = f"{role}{key_suffix}"
    if ss.views_mask.get(role) is not None:
        m1, m2 = st.columns(2)
        with m1:
            st.image(ss.views_png.get(role), caption=f"{ks}: rectified photo")
        with m2:
            st.image(ss.views_mask.get(role),
                     caption=f"{ks}: What I segmented — white = treated as part")
        if ss.originals.get(role) and ss.views_png.get(role):
            if st.button("👆 Tap to select the part", key=f"tapmode_{ks}",
                         help="Auto-segmentation merged shadows/glare. "
                              "Tap the metal face; PartForge cuts from your taps."):
                ss.tap_mode[role] = not ss.tap_mode.get(role, False)
                st.rerun()
            if ss.tap_mode.get(role):
                from partforge.tap_picker import tap_picker
                picked = tap_picker(
                    ss.views_png[role],
                    ss.get(f"taps_{ks}", []),
                    ss.get(f"eraser_{ks}", []),
                    key=f"tap_{ks}")
                if picked:
                    ss[f"taps_{ks}"] = picked.get("taps", [])
                    ss[f"eraser_{ks}"] = picked.get("eraser", [])
                if st.button("🔁 Re-measure from taps",
                             key=f"retap_{ks}", type="primary",
                             disabled=not ss.get(f"taps_{ks}")):
                    from partforge.card import detect_card, CARD_H_MM, CARD_W_MM
                    orig = cv2.imdecode(
                        np.frombuffer(ss.originals[role], np.uint8),
                        cv2.IMREAD_COLOR)
                    H, _ = detect_card(orig)
                    if H is None:
                        st.error("Card lost in stored photo — retake the view.")
                    else:
                        import numpy as _np
                        # rectified = scale @ H @ orig; invert for taps
                        rp = cv2.imdecode(
                            np.frombuffer(ss.views_png[role], np.uint8),
                            cv2.IMREAD_COLOR)
                        rh, rw = rp.shape[:2]
                        S = _np.diag([rw / CARD_W_MM, rh / CARD_H_MM, 1.0])
                        inv = _np.linalg.inv(S @ H)

                        def back(pts):
                            out = []
                            for x, y in pts:
                                p = inv @ _np.array([x, y, 1.0])
                                out.append((float(p[0] / p[2]),
                                            float(p[1] / p[2])))
                            return out
                        process_photo(
                            orig, ss.photo_names.get(role, f"{ks}.jpg"),
                            thresh, role,
                            taps=back(ss[f"taps_{ks}"]),
                            eraser=back(ss.get(f"eraser_{ks}", [])))
                        ss.tap_mode[role] = False
                        (getattr(ss, "resegment", set()) or set()).discard(role)
                        st.rerun()
    else:
        st.caption("No segmentation to show — the card itself "
                   "wasn't found (not all 4 markers detected).")


if ss.get("errors"):
    with st.container():
        st.error("**Could not measure " +
                 ", ".join(f"{r} ({ss.photo_names.get(r, '')})"
                            for r in ss.errors) + "**")
        for role, err in ss.errors.items():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{role.capitalize()}:** {err}")
            with c2:
                if st.button("Retake view", key=f"retake_{role}"):
                    ss.pending.pop(role, None)
                    ss.views_png.pop(role, None)
                    ss.views_mask.pop(role, None)
                    ss.errors.pop(role, None)
                    ss.cap_open = True
                    st.rerun()
            tap_resegment_panel(role, thresh, key_suffix="_err")
        st.markdown("---")

if getattr(ss, "resegment", set()):
    with st.container():
        st.warning("**Re-segmenting from your taps** — click the metal face, "
                   "then Re-measure. (Auto threshold merged shadows/glare.)")
        for role in sorted(getattr(ss, "resegment", set())):
            st.markdown(f"**{role.capitalize()}**")
            tap_resegment_panel(role, thresh, key_suffix="_err")
        st.markdown("---")

# ---------------------------------------------------------------- main
spec = get_spec()
if spec is None and not ss.get("errors"):
    st.markdown(
        """<div class="pf-hero">
        <h1>🔩 PartForge</h1>
        <p>Two photos of a broken part in, a printable CAD model out.
        Geometry is <b>measured</b> from the photos — never hallucinated.</p>
        <p>① Print the calibration card (<span class="pf-kbd">File</span> →
        <span class="pf-kbd">Print calibration card</span>) →
        ② attach <b>top / side / front</b> photos in the ① Capture panel →
        ③ <span class="pf-kbd">🔨 Measure → build CAD</span></p>
        </div>""", unsafe_allow_html=True)
    st.stop()

if spec is None:
    # total failure (top refused): nothing to review, error panel above says why
    st.info("No measurable part yet. Fix the views above (Retake view), "
            "then press Measure again.")
    st.stop()

st.subheader(f"Measured part — {spec.summary()}")
spec_json = json.dumps(ss.spec_json, sort_keys=True)
stl_bytes, step_bytes, diagram_png = build_exports(spec_json, tol)
ss.dl_stl, ss.dl_step = stl_bytes, step_bytes  # Export menu downloads need these

if ss.get("pf_go_export"):
    ss.pf_go_export = False
tab_review, tab_export = st.tabs(["🔍 ② Review — dimensions & questions",
                                  "⬇ ③ Export — STL / STEP"])

with tab_review:
    left, right = st.columns([1.55, 1.0])

    with left:
        clicked = None
        try:
            clicked = part_viewer(stl_bytes, spec, ss.selected,
                                  key=f"viewer_{ss.get('viewer_reset', 0)}")
        except Exception as e:  # viewer is an enhancement; panel below always works
            st.warning(f"3D viewer unavailable ({e.__class__.__name__}). "
                       "Use the dimension diagram + panel on the right.")
        if clicked and clicked != ss.selected:
            ss.selected = clicked
            st.rerun()
        if ss.get("show_diagram", True):
            st.image(diagram_png,
                     caption="Dimension diagram — numbers match the panel on the right")
        if ss.get("show_rect", False):
            tabs = st.tabs([r.capitalize() for r in ("top", "side", "front")
                            if ss.views_png.get(r)])
            i = 0
            for role in ("top", "side", "front"):
                if not ss.views_png.get(role):
                    continue
                with tabs[i]:
                    i += 1
                    sel = st.radio("Layer", ("Measured photo", "What I segmented"),
                                   key=f"layer_{role}", horizontal=True,
                                   help="Measured photo = annotated rectified view. "
                                        "What I segmented = the mask PartForge treated as the part. "
                                        "If the mask is wrong, the CAD will be wrong.")
                    if sel == "Measured photo":
                        st.image(ss.views_png[role])
                    else:
                        st.image(ss.views_mask.get(role),
                                 caption="White = treated as part. Grey specks / shadows included = bad mask.")
                    rep = (get_spec().view_reports or {}).get(role)
                    if rep:
                        st.caption(rep)
                    if st.button("👆 Wrong shape? tap to re-select",
                                 key=f"retap_review_{role}",
                                 help="Re-segment this view from your taps instead of "
                                      "the auto threshold."):
                        ss.resegment = set(getattr(ss, "resegment", set())) | {role}
                        ss.tap_mode[role] = True
                        st.rerun()
                    if role in getattr(ss, "resegment", set()):
                        tap_resegment_panel(role, tol, key_suffix="_rev")

    with right:
        # ----- asks
        with st.container():
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
                            if q.id == "fragmented_mask":
                                save(spec); st.rerun()  # dismissed, no value applied
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

with tab_export:
    ss.pf_go_export = True
    st.markdown("### ⬇ Export")
    st.caption("STL goes to the slicer/printer. STEP opens in Fusion/Onshape for editing. "
               "Holes are exported with the print tolerance added.")
    e1, e2 = st.columns(2)
    e1.download_button("🖨 STL — print it", stl_bytes, "partforge_part.stl",
                       use_container_width=True)
    e2.download_button("📐 STEP — edit it", step_bytes, "partforge_part.step",
                       use_container_width=True)
    e3, e4 = st.columns(2)
    with e3:
        st.metric("STL size", f"{len(stl_bytes) // 1024} KB")
    with e4:
        st.metric("STEP size", f"{len(step_bytes) // 1024} KB")
    if ss.get("show_json", False):
        with st.expander("Part spec (JSON)", expanded=True):
            st.code(json.dumps(ss.spec_json, indent=1), language="json")

ui.statusbar(spec, meta_api.configured(), ss.log)

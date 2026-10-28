"""PartForge Pro UI theme + shell components (menu bar, stepper, status bar).

Everything visual that makes the app read as a desktop CAD program instead
of a Streamlit demo lives here, so app.py stays about workflow, not CSS.

Layout (Photoshop-style):
  [≡] PartForge | File Edit View Window Help || <doc title> || Export ▾
  stepper: 1 Capture - 2 Review - 3 Export
  main area (sidebar hidden by default; Capture lives in the menu)
"""
from __future__ import annotations

import streamlit as st

ACCENT = "#f59e0b"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #0e1117; }

/* ---- kill Streamlit chrome: header, Deploy, footer, sidebar nav ---- */
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
footer { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }
section[data-testid="stSidebar"] { background:#11151c; }
.block-container { padding-top: 0.6rem !important; padding-bottom: 48px; max-width: 100% !important; }

/* ---- app title bar: ONE strip, menus touching ---- */
.pf-titlebar { display:flex; align-items:stretch; gap:0; background:#1e1e1e;
  border:1px solid #333; border-radius:8px 8px 0 0; padding:0 6px 0 0;
  margin-bottom:0; border-bottom:none; }
.pf-brand { font-weight:700; font-size:14px; color:#f5f5f4; margin-right:0;
  display:flex; align-items:center; gap:7px; white-space:nowrap;
  padding:6px 10px 6px 10px; border-right:1px solid #333; }
.pf-brand .dot { width:11px; height:11px; border-radius:3px;
  background:linear-gradient(135deg,#f59e0b,#ef4444); display:inline-block; }
.pf-doctitle { flex:1; text-align:center; font-size:12.5px; color:#9aa4b2;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:0 12px;
  align-self:center; }
/* popover menu buttons: flush against each other like Photoshop menus */
div[data-testid="stPopover"] { margin:0 !important; padding:0 !important; }
div[data-testid="stPopover"] > button { background:transparent !important;
  border:none !important; border-radius:0 !important; margin:0 !important;
  color:#d6dbe2 !important; font-size:13px !important;
  padding:8px 12px !important; }
/* kill the gaps between the menu buttons (title bar only) */
.pf-menus div[data-testid="column"] { padding:0 1px 0 0 !important; min-width:0 !important; }
.pf-menus div[data-testid="stHorizontalBlock"] { gap:0 !important; }
div[data-testid="stPopover"] > button:hover { background:#3a3a3a !important;
  color:#fff !important; }
div[data-testid="stPopover"] button[kind="primary"] { background:#f59e0b !important;
  color:#111 !important; font-weight:700 !important; border-radius:6px !important; }
.pf-export-wrap { display:flex; align-items:center; }
.pf-export-wrap button { font-size:13px !important; }

/* ---- stepper (Photoshop has none; ours is the workflow guide) ---- */
.pf-steps { display:flex; gap:0; margin:0 0 12px 0; background:#1e1e1e;
  border:1px solid #333; border-top:1px solid #2a2a2a; border-radius:0 0 8px 8px;
  padding:0 12px; }
.pf-step { flex:1; text-align:center; font-size:12px; font-weight:600;
  padding:7px 4px; color:#6b7280; border-bottom:3px solid transparent; }
.pf-step.done { color:#4ade80; border-color:#4ade80; }
.pf-step.now { color:#fff; border-color:#f59e0b; }

/* ---- hero (empty state) ---- */
.pf-hero { background:linear-gradient(180deg,#1a2230,#12161d);
  border:1px solid #2d333b; border-radius:12px; padding:28px; text-align:center; }
.pf-hero h1 { font-size:30px; margin:0 0 8px 0; color:#f5f5f4; }
.pf-hero p { color:#9aa4b2; font-size:14px; max-width:640px; margin:6px auto; }
.pf-kbd { background:#2d333b; border-radius:5px; padding:1px 7px; font-size:12px;
  color:#e6e9ee; border:1px solid #444c56; }

/* ---- capture panel (in-menu) ---- */
.pf-cap { background:#161b22; border:1px solid #2d333b; border-radius:10px;
  padding:12px; margin-bottom:12px; }
.pf-cap h4 { margin:0 0 4px 0; font-size:12px; letter-spacing:.04em;
  text-transform:uppercase; color:#9aa4b2; }

/* ---- status bar (Photoshop-style bottom bar) ---- */
.pf-status { position:fixed; left:0; right:0; bottom:0; z-index:999;
  display:flex; gap:16px; align-items:center; background:#1e1e1e;
  border-top:1px solid #333; padding:5px 16px; font-size:12px; color:#9aa4b2; }
.pf-status .ok { color:#4ade80; }
.pf-status .mono { font-family:ui-monospace, monospace; }

/* primary buttons */
.stButton > button[kind="primary"] { background:#f59e0b; border-color:#f59e0b;
  color:#111; font-weight:700; }
.stButton > button[kind="primary"]:hover { background:#fbbf24; border-color:#fbbf24; }
</style>
"""

STEPS = ["Capture", "Review", "Export"]


def menubar(doc_title: str, step_idx: int, has_spec: bool):
    """Photoshop-style title bar: brand | menus | doc title | Export.

    Menus (only what we need):
      File   - New from photos, Print calibration card, Clear project
      Edit   - Undo, Add hole, Add slot
      View   - Fit 3D view, Dimension diagram, Rectified photos
      Window - Correction box, Part spec (JSON)
      Help   - 3-step workflow
    Export popover on the right holds STL/STEP downloads.
    """
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown("<div class='pf-menus'>", unsafe_allow_html=True)
    bar = st.columns([5.2, 3.0, 1.1])
    with bar[0]:
        # one row, zero gaps: brand | File Edit View Window Help, all touching
        row = st.columns([1.35, 0.62, 0.62, 0.62, 0.72, 0.62], gap="small")
        with row[0]:
            st.markdown(
                "<div class='pf-brand'><span class='dot'></span>PartForge</div>",
                unsafe_allow_html=True)
        with row[1]:
            with st.popover("File"):
                if st.button("📷 New from photos",
                             use_container_width=True, key="m_new"):
                    st.session_state["pf_step"] = 0
                    st.rerun()
                st.download_button("⬇ Print calibration card",
                                   _card_bytes(), "partforge_card.png",
                                   use_container_width=True, key="m_card",
                                   help="Print at exactly 210x148 mm (100% scale).")
                if st.button("🗑 Clear project", use_container_width=True,
                             key="m_clear"):
                    for k in ("spec_json", "selected", "views_png", "views_mask",
                              "photo_names", "pending", "staged_side", "errors",
                              "dl_stl", "dl_step",
                              "show_rect", "show_diagram"):
                        st.session_state.pop(k, None)
                    st.rerun()
        with row[2]:
            with st.popover("Edit"):
                if st.button("↩ Undo", use_container_width=True,
                             key="m_undo", disabled=not has_spec):
                    hist = st.session_state.get("pf_hist", [])
                    if hist:
                        st.session_state["spec_json"] = hist.pop()
                        st.rerun()
                if st.button("＋ Add hole", use_container_width=True,
                             key="m_hole", disabled=not has_spec):
                    _add_feature("hole")
                if st.button("＋ Add slot", use_container_width=True,
                             key="m_slot", disabled=not has_spec):
                    _add_feature("slot")
        with row[3]:
            with st.popover("View"):
                if st.button("◉ Fit 3D view", use_container_width=True,
                             key="m_fit", disabled=not has_spec):
                    st.session_state["viewer_reset"] = \
                        st.session_state.get("viewer_reset", 0) + 1
                    st.rerun()
                st.toggle("Dimension diagram", value=True, key="show_diagram")
                if st.button("🖼 Rectified photos", use_container_width=True,
                             key="m_rect", disabled=not has_spec):
                    st.session_state["show_rect"] = \
                        not st.session_state.get("show_rect", False)
                    st.rerun()
        with row[4]:
            with st.popover("Window"):
                st.toggle("Correction box", value=True, key="show_correct")
                st.toggle("Part spec (JSON)", value=False, key="show_json")
        with row[5]:
            with st.popover("Help"):
                st.markdown(
                    "**PartForge in 3 steps**\n\n"
                    "1. **Capture** — print the calibration card\n"
                    "   (File → Print calibration card), photograph the part\n"
                    "   on it: top, then standing on edge for side/front.\n"
                    "2. **Review** — click a marker in the 3D view, answer\n"
                    "   what PartForge asks (thickness, uncertain holes).\n"
                    "3. **Export** — STL to print, STEP to edit.\n\n"
                    "The card's 4 markers turn photo pixels into millimetres. "
                    "Thickness is measured from side views or your caliper — "
                    "never from the top photo.")
    with bar[1]:
        st.markdown(f"<div class='pf-doctitle'>{doc_title}</div>",
                    unsafe_allow_html=True)
    with bar[2]:
        with st.popover("Export ▾"):
            st.download_button("🖨 STL — print it",
                               st.session_state.get("dl_stl", b""),
                               "partforge_part.stl",
                               disabled=not has_spec, use_container_width=True,
                               type="primary", key="m_stl")
            st.download_button("📐 STEP — edit it",
                               st.session_state.get("dl_step", b""),
                               "partforge_part.step",
                               disabled=not has_spec, use_container_width=True,
                               key="m_step")
    steps_html = "<div class='pf-steps'>" + "".join(
        f"<div class='pf-step {'now' if i == step_idx else ('done' if i < step_idx else '')}'>"
        f"{'✓ ' if i < step_idx else ''}{i + 1}. {name}</div>"
        for i, name in enumerate(STEPS)) + "</div>"
    st.markdown(steps_html + "</div>", unsafe_allow_html=True)


def _add_feature(kind: str):
    from partforge.spec import Hole, Slot
    from app import get_spec, save
    spec = get_spec()
    if kind == "hole":
        nid = f"H{len(spec.holes) + 1}"
        while any(h.id == nid for h in spec.holes):
            nid += "x"
        spec.holes.append(Hole(id=nid, x=spec.width_mm / 2,
                               y=spec.height_mm / 2, d=5.0))
        st.session_state["selected"] = nid
    else:
        nid = f"S{len(spec.slots) + 1}"
        spec.slots.append(Slot(id=nid, cx=spec.width_mm / 2,
                               cy=spec.height_mm / 2, length=18.0, width=5.0))
        st.session_state["selected"] = nid
    save(spec)
    st.rerun()


def _card_bytes() -> bytes:
    import cv2
    from partforge.card import render_card
    return cv2.imencode(".png", render_card())[1].tobytes()


def statusbar(spec, meta: dict, log: list[str]):
    """Fixed bottom bar: part, units, tolerance, Meta API state, last log."""
    if spec is not None:
        part = (f"● {spec.type} {spec.width_mm:.1f}×{spec.height_mm:.1f}×"
                f"{spec.thickness_mm if spec.thickness_mm is not None else '—'} mm · "
                f"{len(spec.holes)}H/{len(spec.slots)}S"
                + (f" · bend {spec.bend.angle_deg:.0f}°" if spec.bend else ""))
    else:
        part = "○ no part loaded"
    apis = " ".join(
        f"<span class='{'ok' if v else ''}'>{k}</span>" for k, v in meta.items())
    last = (log or ["ready"])[-1]
    st.markdown(
        f"<div class='pf-status'><span class='mono'>{part}</span>"
        f"<span>mm</span><span>tolerance +{st.session_state.get('pf_tol', 0.2):.2f}</span>"
        f"<span style='margin-left:auto'>{apis}</span>"
        f"<span>{last[:80]}</span></div>",
        unsafe_allow_html=True)

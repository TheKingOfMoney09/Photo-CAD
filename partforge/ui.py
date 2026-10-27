"""PartForge Pro UI theme + shell components (menu bar, stepper, status bar).

Everything visual that makes the app read as a desktop CAD program instead
of a Streamlit demo lives here, so app.py stays about workflow, not CSS.
"""
from __future__ import annotations

import streamlit as st

ACCENT = "#f59e0b"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #0e1117; }

/* ---- menu bar ---- */
.pf-menubar { display:flex; align-items:center; gap:2px; background:#161b22;
  border:1px solid #2d333b; border-radius:8px; padding:4px 8px; margin-bottom:6px; }
.pf-brand { font-weight:700; font-size:15px; color:#f5f5f4; margin-right:10px;
  display:flex; align-items:center; gap:6px; white-space:nowrap; }
.pf-brand .dot { width:10px; height:10px; border-radius:3px;
  background:linear-gradient(135deg,#f59e0b,#ef4444); display:inline-block; }
.pf-step-chip { margin-left:auto; font-size:12px; color:#9aa4b2; white-space:nowrap; }
div[data-testid="stPopover"] > button { background:transparent !important;
  border:none !important; color:#d6dbe2 !important; font-size:13.5px !important;
  padding:6px 10px !important; }
div[data-testid="stPopover"] > button:hover { background:#2d333b !important;
  border-radius:6px; color:#fff !important; }

/* ---- stepper ---- */
.pf-steps { display:flex; gap:0; margin:2px 0 12px 0; }
.pf-step { flex:1; text-align:center; font-size:12.5px; font-weight:600;
  padding:8px 4px; color:#6b7280; border-bottom:3px solid #2d333b; }
.pf-step.done { color:#4ade80; border-color:#4ade80; }
.pf-step.now { color:#fff; border-color:#f59e0b; }

/* ---- cards ---- */
.pf-card { background:#161b22; border:1px solid #2d333b; border-radius:10px;
  padding:14px 16px; margin-bottom:12px; }
.pf-card h4 { margin:0 0 6px 0; font-size:13px; letter-spacing:.04em;
  text-transform:uppercase; color:#9aa4b2; }
.pf-hero { background:linear-gradient(180deg,#1a2230,#12161d);
  border:1px solid #2d333b; border-radius:12px; padding:28px; text-align:center; }
.pf-hero h1 { font-size:30px; margin:0 0 8px 0; color:#f5f5f4; }
.pf-hero p { color:#9aa4b2; font-size:14px; max-width:640px; margin:6px auto; }
.pf-kbd { background:#2d333b; border-radius:5px; padding:1px 7px; font-size:12px;
  color:#e6e9ee; border:1px solid #444c56; }

/* ---- status bar ---- */
.pf-status { position:fixed; left:0; right:0; bottom:0; z-index:999;
  display:flex; gap:16px; align-items:center; background:#161b22;
  border-top:1px solid #2d333b; padding:5px 16px; font-size:12px; color:#9aa4b2; }
.pf-status .ok { color:#4ade80; } .pf-status .warn { color:#f59e0b; }
.pf-status .mono { font-family:ui-monospace, monospace; }
.block-container { padding-bottom:48px; }

/* primary buttons */
.stButton > button[kind="primary"] { background:#f59e0b; border-color:#f59e0b;
  color:#111; font-weight:700; }
.stButton > button[kind="primary"]:hover { background:#fbbf24; border-color:#fbbf24; }
section[data-testid="stSidebar"] { background:#11151c; }
</style>
"""

STEPS = ["Capture", "Review", "Export"]


def menubar(step_idx: int, has_spec: bool):
    """Desktop-style menu bar with popover menus + workflow stepper."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    with st.container():
        m1, m2, m3, m4, m5, chip = st.columns([0.62, 0.62, 0.62, 0.72, 0.9, 3.0])
        with m1:
            with st.popover("File"):
                if st.button("📷 New from photos…",
                             use_container_width=True, key="m_new"):
                    st.session_state["pf_step"] = 0
                    st.rerun()
                st.download_button("⬇ Print calibration card",
                                   _card_bytes(), "partforge_card.png",
                                   use_container_width=True, key="m_card",
                                   help="Print at exactly 210x148 mm (100% scale).")
                c1, c2 = st.columns(2)
                stl = c1.download_button(
                    "STL", st.session_state.get("dl_stl", b""),
                    "partforge_part.stl", disabled=not has_spec,
                    use_container_width=True, key="m_stl")
                stp = c2.download_button(
                    "STEP", st.session_state.get("dl_step", b""),
                    "partforge_part.step", disabled=not has_spec,
                    use_container_width=True, key="m_step")
                if st.button("🗑 Clear project", use_container_width=True, key="m_clear"):
                    for k in ("spec_json", "selected", "views_png", "photo_names",
                              "pending", "dl_stl", "dl_step", "pf_step"):
                        st.session_state.pop(k, None)
                    st.rerun()
        with m2:
            with st.popover("Edit"):
                if st.button("↩ Undo last edit", use_container_width=True,
                             key="m_undo", disabled=not has_spec):
                    hist = st.session_state.get("pf_hist", [])
                    if hist:
                        st.session_state["spec_json"] = hist.pop()
                        st.rerun()
                if st.button("＋ Add hole", use_container_width=True,
                             key="m_hole", disabled=not has_spec):
                    from partforge.spec import Hole
                    from app import get_spec, save
                    spec = get_spec()
                    nid = f"H{len(spec.holes) + 1}"
                    while any(h.id == nid for h in spec.holes):
                        nid += "x"
                    spec.holes.append(Hole(id=nid, x=spec.width_mm / 2,
                                           y=spec.height_mm / 2, d=5.0))
                    save(spec)
                    st.rerun()
        with m3:
            with st.popover("View"):
                if st.button("◉ Fit 3D view", use_container_width=True, key="m_fit"):
                    st.session_state["viewer_reset"] = \
                        st.session_state.get("viewer_reset", 0) + 1
                    st.rerun()
                if st.button("🖼 Show rectified photos", use_container_width=True,
                             key="m_rect"):
                    st.session_state["show_rect"] = \
                        not st.session_state.get("show_rect", False)
                    st.rerun()
        with m4:
            with st.popover("Window"):
                st.caption("Panel layout")
                st.toggle("Dimension diagram", value=True, key="show_diagram")
                st.toggle("Correction box", value=True, key="show_correct")
                st.toggle("Part spec (JSON)", value=False, key="show_json")
        with m5:
            with st.popover("Help"):
                st.markdown(
                    "**PartForge in 3 steps**\n\n"
                    "1. **Capture** — print the calibration card, photograph "
                    "the part on it (top / side / front).\n"
                    "2. **Review** — click a marker in the 3D view, fix what "
                    "PartForge asks.\n"
                    "3. **Export** — download STL to print or STEP to edit.\n\n"
                    "The card's 4 markers turn photo pixels into millimetres. "
                    "Thickness is measured from side views or your caliper — "
                    "never from the top photo.")
        with chip:
            st.markdown(
                f"<div class='pf-step-chip'>PartForge · "
                f"step {step_idx + 1}/3 — {STEPS[step_idx]}</div>",
                unsafe_allow_html=True)
    steps_html = "<div class='pf-steps'>" + "".join(
        f"<div class='pf-step {'now' if i == step_idx else ('done' if i < step_idx else '')}'>"
        f"{'✓ ' if i < step_idx else ''}{i + 1}. {name}</div>"
        for i, name in enumerate(STEPS)) + "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)


def _card_bytes() -> bytes:
    import cv2
    from partforge.card import render_card
    return cv2.imencode(".png", render_card())[1].tobytes()


def statusbar(spec, meta: dict, log: list[str]):
    """Fixed bottom status bar: part, units, tolerance, Meta API state, last log."""
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

"""SAM 3.1 -> shapes -> PartSpec, one view at a time. No taps, no thresholds.

measure_view(img, role) -> (spec|None, rect|None, ppm, info).
  role "top":   part mask -> outline; hole/slot masks -> features.
  role "front": part mask -> fitted rectangle -> long side must match
                top W/H (checked by fusion), short side = thickness vote.
  role "side":  alias of front (kept for old sessions).

One request per prompt ("part", then "hole", then "slot"), never combined.
Per-photo result cached by the caller (sha1) so edits never re-call SAM.
Debug images go to debug/<role>_sam_<prompt>.png and debug/<role>_fitted.png.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .card import detect_card, rectify
from .sam_engine import segment
from .shapes import fit_holes, fit_slots, polygon_to_mask
from .spec import PartSpec

PART_PROMPTS = ["metal bracket plate", "flat metal part on white card",
                "rectangular bar", "plate", "part"]
HOLE_PROMPT = "hole"
SLOT_PROMPT = "slot"

DEBUG_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "debug")


def _debug(name: str, img: np.ndarray) -> None:
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, name), img)
    except Exception:
        pass


def _mask_to_poly(mask: np.ndarray) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.float64) for c in cnts
            if cv2.contourArea(c) >= 50]


def _warp_polys(polys: list[np.ndarray], H: np.ndarray,
                ppm: float) -> list[np.ndarray]:
    out = []
    for p in polys:
        pts = np.hstack([p.reshape(-1, 2),
                         np.ones((len(p.reshape(-1, 2)), 1))])
        w = (np.diag([ppm, ppm, 1.0]) @ H @ pts.T).T
        w = w[:, :2] / w[:, 2:3]
        out.append(w)
    return out


def _backend() -> str:
    if os.environ.get("SAM_MOCK") == "1":
        return "mock"
    return "live" if os.environ.get("MODEL_API_KEY", "").strip() else "missing"


def annotate(rect_bgr: np.ndarray, spec, px_per_mm: float) -> np.ndarray:
    """Draw fitted outline/holes/slots back onto the rectified photo."""
    out = rect_bgr.copy()
    ox = (spec.offset_mm[0] * px_per_mm, spec.offset_mm[1] * px_per_mm)
    pts = np.array([[(p[0] * px_per_mm + ox[0]), (p[1] * px_per_mm + ox[1])]
                    for p in spec.outline_mm], dtype=np.int32)
    cv2.polylines(out, [pts], True, (0, 200, 0), 3)
    for h in spec.holes:
        c = (int(h.x * px_per_mm + ox[0]), int(h.y * px_per_mm + ox[1]))
        cv2.circle(out, c, max(3, int(h.d / 2 * px_per_mm)), (0, 0, 255), 3)
        cv2.putText(out, f"{h.id} d{h.d:.1f}", (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for s in spec.slots:
        c = (int(s.cx * px_per_mm + ox[0]), int(s.cy * px_per_mm + ox[1]))
        cv2.circle(out, c, max(3, int(s.width / 2 * px_per_mm)), (255, 128, 0), 3)
        cv2.putText(out, f"{s.id} {s.length:.0f}x{s.width:.1f}", (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2)
    return out


def measure_view(img_bgr: np.ndarray, role: str):
    """Run SAM 3.1 on one photo, fit shapes, return a PartSpec (or None)."""
    H, ids = detect_card(img_bgr)
    if H is None:
        return None, None, 0.0, {
            "error": "Calibration card not found (need all 4 markers).",
            "card_markers": ids, "backend": _backend()}
    rect, ppm = rectify(img_bgr, H)
    backend = _backend()
    try:
        parts = []
        for pp in PART_PROMPTS:
            parts.extend(segment(img_bgr, pp))
            if parts:
                break
        holes = segment(img_bgr, HOLE_PROMPT) if role == "top" else []
        slots = segment(img_bgr, SLOT_PROMPT) if role == "top" else []
    except RuntimeError as e:
        return None, rect, ppm, {"error": str(e), "card_markers": ids,
                                 "rect": rect, "backend": backend}
    h, w = img_bgr.shape[:2]
    for prompt, recs in (("part", parts), ("hole", holes), ("slot", slots)):
        dbg = img_bgr.copy()
        for (_box, m) in recs:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, cnts, -1, (0, 255, 0), 3)
        _debug(f"{role}_sam_{prompt}.png", dbg)
    if not parts:
        return None, rect, ppm, {
            "error": f"SAM found no 'part' in the {role} view.",
            "card_markers": ids, "rect": rect, "backend": backend,
            "mask": np.zeros(rect.shape[:2], np.uint8)}
    # biggest part mask wins (one part per photo)
    _box, part_mask = max(parts, key=lambda r: int(r[1].sum()))
    from .shapes import fit_outline
    rparts = _warp_polys(_mask_to_poly(part_mask), H, ppm)
    if not rparts:
        return None, rect, ppm, {
            "error": f"SAM part mask in the {role} view has no contour.",
            "card_markers": ids, "rect": rect, "backend": backend}
    rpart = max(rparts, key=lambda p: cv2.contourArea(
        p.reshape(-1, 1, 2).astype(np.float32)))
    full = polygon_to_mask(
        [p.reshape(-1, 1, 2).astype(np.int32) for p in rparts],
        rect.shape[:2])
    frac = float((full > 0).mean())
    if frac > 0.55:
        return None, rect, ppm, {
            "error": (f"SAM mask covers {frac:.0%} of the card — wrong "
                      f"instance. Retake with only the part centred."),
            "card_markers": ids, "rect": rect, "mask": full,
            "backend": backend}
    rholes = []
    for (_box, m) in holes:
        rholes.extend(_warp_polys(_mask_to_poly(m), H, ppm))
    rslots = []
    for (_box, m) in slots:
        rslots.extend(_warp_polys(_mask_to_poly(m), H, ppm))
    outline = fit_outline(rpart, ppm)
    minx = min(p[0] for p in outline)
    miny = min(p[1] for p in outline)
    outline = [[p[0] - minx, p[1] - miny] for p in outline]
    hole_objs = fit_holes(rholes, ppm)
    for hh in hole_objs:
        hh.x = round(hh.x - minx, 2)
        hh.y = round(hh.y - miny, 2)
    slot_objs = fit_slots(rslots, ppm)
    for sl in slot_objs:
        sl.cx = round(sl.cx - minx, 2)
        sl.cy = round(sl.cy - miny, 2)
    if role == "top":
        spec = PartSpec(outline_mm=outline, holes=hole_objs, slots=slot_objs,
                        source=f"measured:sam31:{backend}",
                        offset_mm=[round(minx, 2), round(miny, 2)])
        note = (f"Top: {spec.width_mm:.1f}x{spec.height_mm:.1f} mm, "
                f"{len(hole_objs)} holes, {len(slot_objs)} slots")
    else:
        # front/side: rectangle silhouette; fusion matches + votes thickness
        spec = PartSpec(outline_mm=outline, holes=[], slots=[],
                        source=f"measured:sam31:{backend}",
                        offset_mm=[round(minx, 2), round(miny, 2)])
        note = (f"{role}: silhouette {spec.width_mm:.1f}x{spec.height_mm:.1f}")
    _debug(f"{role}_fitted.png",
           _draw_fitted(rect, outline, minx, miny, ppm))
    return spec, rect, ppm, {"card_markers": ids, "mask": full,
                             "backend": backend, "note": note}


def _draw_fitted(rect: np.ndarray, outline: list, minx: float, miny: float,
                 ppm: float) -> np.ndarray:
    out = rect.copy()
    pts = np.array([[(p[0] + minx) * ppm, (p[1] + miny) * ppm]
                    for p in outline], dtype=np.int32)
    cv2.polylines(out, [pts], True, (0, 200, 0), 3)
    return out

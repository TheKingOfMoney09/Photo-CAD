"""SAM -> shapes -> PartSpec, one view at a time. No taps, no thresholds.

measure_view(img, role) -> (spec|None, rect|None, ppm, info).
  role "top":   part polygon -> outline; hole/slot polygons -> features.
  role "front": part polygon -> fitted rectangle -> long side must match
                top W/H (checked by fusion), short side = thickness vote.
  role "side":  alias of front (kept for old sessions).

One SAM call per photo (client does all prompts at once). Debug images go
to debug/<role>_sam_<prompt>.png and debug/<role>_fitted.png.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .card import detect_card, rectify
from .sam3 import PROMPTS, get_client
from .shapes import fit_holes, fit_outline, fit_slots, polygon_to_mask
from .spec import PartSpec

DEBUG_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "debug")


def _debug(name: str, img: np.ndarray) -> None:
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, name), img)
    except Exception:
        pass


def _warp_polys(polys: list[np.ndarray], H: np.ndarray,
                ppm: float, shape: tuple[int, int]) -> list[np.ndarray]:
    out = []
    for p in polys:
        pts = np.hstack([p.reshape(-1, 2),
                         np.ones((len(p.reshape(-1, 2)), 1))])
        w = (np.diag([ppm, ppm, 1.0]) @ H @ pts.T).T
        w = w[:, :2] / w[:, 2:3]
        out.append(w)
    return out


def measure_view(img_bgr: np.ndarray, role: str):
    """Run SAM on one photo, fit shapes, return a PartSpec (or None)."""
    H, ids = detect_card(img_bgr)
    if H is None:
        return None, None, 0.0, {
            "error": "Calibration card not found (need all 4 markers).",
            "card_markers": ids, "backend": get_client().name}
    rect, ppm = rectify(img_bgr, H)
    client = get_client()
    try:
        polys = client.segment(img_bgr, PROMPTS)
    except RuntimeError as e:
        return None, rect, ppm, {"error": str(e), "card_markers": ids,
                                 "rect": rect, "backend": client.name}
    h, w = img_bgr.shape[:2]
    for prompt, pp in polys.items():
        dbg = rect.copy() if False else img_bgr.copy()
        for poly in pp:
            cv2.polylines(dbg, [poly.reshape(-1, 1, 2).astype(np.int32)],
                          True, (0, 255, 0), 3)
        _debug(f"{role}_sam_{prompt}.png", dbg)
    parts = polys.get("part", [])
    if not parts:
        return None, rect, ppm, {
            "error": f"SAM found no 'part' in the {role} view.",
            "card_markers": ids, "rect": rect, "backend": client.name,
            "mask": np.zeros(rect.shape[:2], np.uint8)}
    # biggest part polygon wins (one part per photo)
    part = max(parts, key=lambda p: cv2.contourArea(
        p.reshape(-1, 1, 2).astype(np.float32)))
    rparts = _warp_polys([part], H, ppm, rect.shape[:2])
    # mask in rect space for the UI layer + guards
    full = polygon_to_mask(
        [p.reshape(-1, 1, 2).astype(np.int32)
         for p in _warp_polys(parts, H, ppm, rect.shape[:2])],
        rect.shape[:2])
    frac = float((full > 0).mean())
    if frac > 0.55:
        return None, rect, ppm, {
            "error": (f"SAM mask covers {frac:.0%} of the card — wrong "
                      f"instance. Retake with only the part centred."),
            "card_markers": ids, "rect": rect, "mask": full,
            "backend": client.name}
    rholes = _warp_polys(polys.get("hole", []), H, ppm, rect.shape[:2])
    rslots = _warp_polys(polys.get("slot", []), H, ppm, rect.shape[:2])
    outline = fit_outline(rparts[0], ppm)
    minx = min(p[0] for p in outline)
    miny = min(p[1] for p in outline)
    outline = [[p[0] - minx, p[1] - miny] for p in outline]
    holes = fit_holes(rholes, ppm)
    for hh in holes:
        hh.x = round(hh.x - minx, 2)
        hh.y = round(hh.y - miny, 2)
    slots = fit_slots(rslots, ppm)
    for sl in slots:
        sl.cx = round(sl.cx - minx, 2)
        sl.cy = round(sl.cy - miny, 2)
    if role == "top":
        spec = PartSpec(outline_mm=outline, holes=holes, slots=slots,
                        source=f"measured:sam3:{client.name}",
                        offset_mm=[round(minx, 2), round(miny, 2)])
        note = (f"Top: {spec.width_mm:.1f}x{spec.height_mm:.1f} mm, "
                f"{len(holes)} holes, {len(slots)} slots")
    else:
        # front/side: rectangle silhouette; fusion matches + votes thickness
        spec = PartSpec(outline_mm=outline, holes=[], slots=[],
                        source=f"measured:sam3:{client.name}",
                        offset_mm=[round(minx, 2), round(miny, 2)])
        note = (f"{role}: silhouette {spec.width_mm:.1f}x{spec.height_mm:.1f}")
    _debug(f"{role}_fitted.png",
           _draw_fitted(rect, outline, minx, miny, ppm))
    return spec, rect, ppm, {"card_markers": ids, "mask": full,
                             "backend": client.name, "note": note}


def _draw_fitted(rect: np.ndarray, outline: list, minx: float, miny: float,
                 ppm: float) -> np.ndarray:
    out = rect.copy()
    pts = np.array([[(p[0] + minx) * ppm, (p[1] + miny) * ppm]
                    for p in outline], dtype=np.int32)
    cv2.polylines(out, [pts], True, (0, 200, 0), 3)
    return out

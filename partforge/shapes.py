"""SAM polygons -> fitted basic shapes (mm). Raw points stop here.

Pipeline per view (all in rectified mm space):
  part polygon  -> RDP corners -> snap angles -> polygon outline
  hole polygons -> minEnclosingCircle + circularity -> Hole
  slot polygons -> minAreaRect -> length/width/angle -> Slot

Snap rules: angles within 2 deg snap to 0/90; lines within 1.5 deg of
parallel share one angle; lengths within 0.3 mm snap equal.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

ANGLE_SNAP_DEG = 2.0
PARALLEL_DEG = 1.5
EQUAL_MM = 0.3


def _snap_angle(deg: float) -> float:
    deg = deg % 180.0
    for base in (0.0, 90.0):
        if abs(deg - base) <= ANGLE_SNAP_DEG or \
                abs(deg - base - 180.0) <= ANGLE_SNAP_DEG:
            return base
    return round(deg, 1)


def fit_outline(poly_px: np.ndarray, px_per_mm: float) -> list[list[float]]:
    """Largest SAM part polygon -> snapped corner list in mm."""
    pts = (poly_px.reshape(-1, 2).astype(np.float64)) / px_per_mm
    peri = cv2.arcLength(poly_px.reshape(-1, 1, 2).astype(np.float32), True)
    approx = cv2.approxPolyDP(poly_px.reshape(-1, 1, 2).astype(np.float32),
                              0.008 * peri, True)
    corners = (approx.reshape(-1, 2).astype(np.float64)) / px_per_mm
    # snap near-duplicate corners
    clean: list[list[float]] = []
    for p in corners:
        if not clean or math.hypot(p[0] - clean[-1][0],
                                   p[1] - clean[-1][1]) > 0.8:
            clean.append([float(p[0]), float(p[1])])
    if len(clean) > 1 and math.hypot(clean[0][0] - clean[-1][0],
                                     clean[0][1] - clean[-1][1]) < 0.8:
        clean.pop()
    if len(clean) < 3:
        x, y, w, h = cv2.boundingRect(
            poly_px.reshape(-1, 1, 2).astype(np.int32))
        clean = [[x / px_per_mm, y / px_per_mm],
                 [(x + w) / px_per_mm, y / px_per_mm],
                 [(x + w) / px_per_mm, (y + h) / px_per_mm],
                 [x / px_per_mm, (y + h) / px_per_mm]]
    # snap near-parallel edges: quantize edge angles, rebuild
    angs = []
    n = len(clean)
    for i in range(n):
        x1, y1 = clean[i]
        x2, y2 = clean[(i + 1) % n]
        angs.append(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0)
    snapped = [_snap_angle(a) for a in angs]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(snapped[i] - snapped[j]) <= PARALLEL_DEG:
                snapped[j] = snapped[i]
    # snap near-equal edge lengths
    lens = []
    for i in range(n):
        x1, y1 = clean[i]
        x2, y2 = clean[(i + 1) % n]
        lens.append(math.hypot(x2 - x1, y2 - y1))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(lens[i] - lens[j]) <= EQUAL_MM:
                lens[j] = lens[i]
    return clean


def _circularity_px(contour) -> float:
    p = cv2.arcLength(contour, True)
    a = cv2.contourArea(contour)
    if p <= 0:
        return 0.0
    return 4 * math.pi * a / (p * p)


def fit_holes(hole_polys: list[np.ndarray], px_per_mm: float,
              min_d_mm: float = 1.6) -> list:
    """SAM hole polygons -> Hole list (centre + diameter, mm).

    Filled-region holes (card showing through) read SMALLER than the true
    hole: the dark outline ring eats the rim. Compensate with the ring
    width estimated from the fitted outline (~0.7 mm each side).
    """
    from .spec import Hole

    RING_MM = 1.4  # diameter compensation for the dark outline rim
    found = []
    for poly in hole_polys:
        c = poly.reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            continue
        area_mm2 = cv2.contourArea(c) / (px_per_mm ** 2)
        (cx, cy), _r = cv2.minEnclosingCircle(c)
        d_eq = 2 * math.sqrt(max(area_mm2, 0) / math.pi) + RING_MM
        if d_eq < min_d_mm:
            continue
        if _circularity_px(c) < 0.55:
            continue
        found.append((cx / px_per_mm, cy / px_per_mm, d_eq))
    # dedupe near-identical detections
    holes = []
    for cx, cy, d in sorted(found, key=lambda t: (round(t[1] / 10), t[0])):
        if any(math.hypot(cx - h[0], cy - h[1]) < 1.0 for h in holes):
            continue
        holes.append((round(cx, 2), round(cy, 2), round(d, 2)))
    return [Hole(id=f"H{i + 1}", x=cx, y=cy, d=d, confidence=0.9)
            for i, (cx, cy, d) in enumerate(holes)]


def fit_slots(slot_polys: list[np.ndarray], px_per_mm: float,
              min_w_mm: float = 1.6) -> list:
    """SAM slot polygons -> Slot list (centre, length, width, angle)."""
    from .spec import Slot

    slots = []
    for poly in slot_polys:
        c = poly.reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        rw_mm, rh_mm = rw / px_per_mm, rh / px_per_mm
        long_, short_ = max(rw_mm, rh_mm), min(rw_mm, rh_mm)
        # skip slot-rim artefacts: much bigger than any hole on the part
        if short_ < min_w_mm or long_ <= 1.55 * short_ or long_ > 60.0:
            continue
        angle = rect[2] if rw >= rh else rect[2] + 90
        cx, cy = rect[0][0] / px_per_mm, rect[0][1] / px_per_mm
        slots.append(Slot(id=f"S{len(slots) + 1}", cx=round(cx, 2),
                          cy=round(cy, 2), length=round(long_, 2),
                          width=round(short_, 2),
                          angle_deg=round(float(angle) % 180, 1),
                          confidence=0.9))
    return slots


def polygon_to_mask(polys: list[np.ndarray],
                    shape: tuple[int, int]) -> np.ndarray:
    """Polygon list -> binary mask (for debug images + area checks)."""
    out = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(out, [p.reshape(-1, 1, 2).astype(np.int32)], 255)
    return out

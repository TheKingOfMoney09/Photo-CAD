"""Deterministic measurement from a calibrated top photo.

No learned model touches a number here: outline, hole centres/diameters and
slots come from contour geometry in millimetre space. A segmentation mask
(SAM 3.1, or the OpenCV fallback below) only decides *which* pixels are part.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .card import detect_card, rectify
from .spec import Hole, PartSpec, Slot


def segment_fallback(rect_bgr: np.ndarray, thresh: int = 210) -> np.ndarray:
    """Fallback when SAM is unavailable: part = non-card-white region.

    Works when the part lies on the white calibration card. The card's own
    graphics (border, ArUco markers, scale bar) sit at known mm positions
    in rectified space and are masked out first. Returns uint8 mask.
    """
    from .card import CARD_H_MM, CARD_W_MM, MARKER_CENTERS, MARKER_MM

    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    dark_or_colored = ((gray < thresh) | (sat > 60)).astype(np.uint8) * 255
    h_px, w_px = dark_or_colored.shape
    ppm_x, ppm_y = w_px / CARD_W_MM, h_px / CARD_H_MM

    def blank(x_mm, y_mm, w_mm, h_mm):
        x0, y0 = int(x_mm * ppm_x), int(y_mm * ppm_y)
        x1, y1 = int((x_mm + w_mm) * ppm_x), int((y_mm + h_mm) * ppm_y)
        dark_or_colored[max(0, y0):y1, max(0, x0):x1] = 0

    blank(0, 0, CARD_W_MM, 7); blank(0, CARD_H_MM - 7, CARD_W_MM, 7)
    blank(0, 0, 7, CARD_H_MM); blank(CARD_W_MM - 7, 0, 7, CARD_H_MM)
    for cx, cy in MARKER_CENTERS.values():
        s = MARKER_MM + 8
        blank(cx - s / 2, cy - s / 2, s, s)
    blank(65, CARD_H_MM - 24, 75, 17)  # scale bar + label
    blank(48, 4, 115, 17)  # title text strip
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(dark_or_colored, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if hierarchy is None:
        return out
    hierarchy = hierarchy[0]
    h, w = mask.shape
    for i, c in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue  # roots only; holes are re-cut below
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)
        touches = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
        if area > 0.002 * h * w and not touches:
            cv2.drawContours(out, [c], -1, 255, -1)
            child = hierarchy[i][2]
            while child != -1:  # keep holes/slots as holes in the mask
                cv2.drawContours(out, [contours[child]], -1, 0, -1)
                child = hierarchy[child][0]
    return out


def _circularity(contour) -> float:
    p = cv2.arcLength(contour, True)
    a = cv2.contourArea(contour)
    if p <= 0:
        return 0.0
    return 4 * math.pi * a / (p * p)


def measure_mask(mask: np.ndarray, px_per_mm: float,
                 min_hole_mm: float = 1.6) -> PartSpec | None:
    """Measure a binary part mask (in mm-space image) into a PartSpec."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    # outer contour = largest root contour
    roots = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not roots:
        return None
    outer_i = max(roots, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_i]
    if cv2.contourArea(outer) < 50:
        return None

    approx = cv2.approxPolyDP(outer, 0.6 * px_per_mm, True)
    pts = approx.reshape(-1, 2).astype(np.float64) / px_per_mm
    # drop near-duplicate / collinear points
    clean: list[list[float]] = []
    for p in pts:
        if not clean or np.hypot(*(p - np.array(clean[-1]))) > 0.8:
            clean.append([float(p[0]), float(p[1])])
    if len(clean) > 1 and np.hypot(*(np.array(clean[0]) - np.array(clean[-1]))) < 0.8:
        clean.pop()
    if len(clean) < 3:  # fall back to bounding rectangle
        x, y, w, h = cv2.boundingRect(outer)
        clean = [[x / px_per_mm, y / px_per_mm],
                 [(x + w) / px_per_mm, y / px_per_mm],
                 [(x + w) / px_per_mm, (y + h) / px_per_mm],
                 [x / px_per_mm, (y + h) / px_per_mm]]

    minx = min(p[0] for p in clean)
    miny = min(p[1] for p in clean)
    outline = [[p[0] - minx, p[1] - miny] for p in clean]

    holes: list[Hole] = []
    slots: list[Slot] = []
    # children of the outer contour = holes / slots
    child = hierarchy[outer_i][2]
    seen: list[tuple[float, float]] = []
    while child != -1:
        c = contours[child]
        area_mm2 = cv2.contourArea(c) / (px_per_mm ** 2)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cx_mm, cy_mm = cx / px_per_mm - minx, cy / px_per_mm - miny
        circ = _circularity(c)
        d_eq = 2 * math.sqrt(max(area_mm2, 0) / math.pi)
        if d_eq >= min_hole_mm and not any(
                math.hypot(cx_mm - sx, cy_mm - sy) < 1.0 for sx, sy in seen):
            seen.append((cx_mm, cy_mm))
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            rw_mm, rh_mm = rw / px_per_mm, rh / px_per_mm
            long_, short_ = max(rw_mm, rh_mm), min(rw_mm, rh_mm)
            if circ >= 0.72:
                conf = float(min(1.0, 0.5 + circ / 2))
                holes.append(Hole(id=f"H{len(holes)+1}", x=round(cx_mm, 2),
                                  y=round(cy_mm, 2), d=round(d_eq, 2),
                                  confidence=round(conf, 2)))
            elif short_ >= min_hole_mm and long_ > 1.55 * short_:
                angle = rect[2] if rw >= rh else rect[2] + 90
                slots.append(Slot(id=f"S{len(slots)+1}", cx=round(cx_mm, 2),
                                  cy=round(cy_mm, 2), length=round(long_, 2),
                                  width=round(short_, 2),
                                  angle_deg=round(float(angle) % 180, 1),
                                  confidence=round(float(min(1, 0.4 + circ)), 2)))
        child = hierarchy[child][0]

    holes.sort(key=lambda h: (round(h.y / 10), h.x))
    for i, h in enumerate(holes):
        h.id = f"H{i+1}"
    return PartSpec(outline_mm=outline, holes=holes, slots=slots,
                    source="measured", offset_mm=[round(minx, 2), round(miny, 2)])


def measure_photo(img_bgr: np.ndarray, mask: np.ndarray | None = None,
                  thresh: int = 210):
    """Full path: photo -> (spec, rectified_img, px_per_mm, info)."""
    H, ids = detect_card(img_bgr)
    if H is None:
        return None, None, 0.0, {"card_markers": ids,
                                 "error": "Calibration card not found (need all 4 markers)."}
    rect, ppm = rectify(img_bgr, H)
    if mask is None:
        mask = segment_fallback(rect, thresh=thresh)
    else:
        mask = cv2.warpPerspective(
            mask, np.diag([ppm, ppm, 1.0]) @ H,
            (rect.shape[1], rect.shape[0]))
    spec = measure_mask(mask, ppm)
    return spec, rect, ppm, {"card_markers": ids}


def annotate(rect_bgr: np.ndarray, spec: PartSpec, px_per_mm: float) -> np.ndarray:
    """Draw measured outline/holes/slots back onto the rectified photo."""
    out = rect_bgr.copy()
    # outline offset: spec coords are bbox-relative; bbox min in rect px:
    # recompute from outline is identity here because measure used same min.
    # For display we need the offset; stored by caller via spec offset attrs.
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

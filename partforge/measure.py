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

    thresh <= 0 means Otsu auto-threshold on the background peak: lighting
    varies between phones/flash/daylight, so a fixed 210 breaks on dim
    shots (card goes grey -> whole card segments as "part").
    """
    from .card import CARD_H_MM, CARD_W_MM, MARKER_CENTERS, MARKER_MM

    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    if thresh and thresh > 0:
        t = thresh
    else:
        # Otsu on the card area only (centre 60% x 30% is usually background)
        h, w = gray.shape
        probe = gray[int(h * 0.35):int(h * 0.65), int(w * 0.2):int(w * 0.8)]
        t, _ = cv2.threshold(probe, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(max(60.0, min(225.0, t - 12.0)))  # margin: card stays bg
    dark_or_colored = ((gray < t) | (sat > 60)).astype(np.uint8) * 255
    h_px, w_px = dark_or_colored.shape
    ppm_x, ppm_y = w_px / CARD_W_MM, h_px / CARD_H_MM
    _blank_card_graphics(dark_or_colored, ppm_x, ppm_y)

    # --- specular repair: blown highlights inside the part -----------------
    # Shiny metal blows out to near-white, same as the card. Card-white is
    # reachable from the image border; blowout is fully enclosed by dark
    # part pixels. Flood-fill the background from the border on the BLANKED
    # mask (blanked graphics read as background, so the card interior stays
    # connected to the border through them), then merge large/irregular
    # enclosed-dark regions... no: merge enclosed-BRIGHT regions in the gray
    # image back into the part mask. Real holes survive via circularity.
    bright = (gray >= t).astype(np.uint8) * 255
    # carve the same blanked zones as background so card stays connected
    bg = bright.copy()
    bg[dark_or_colored == 0] = 0
    ff = bg.copy()
    ffmask = np.zeros((h_px + 2, w_px + 2), np.uint8)
    try:
        cv2.floodFill(ff, ffmask, (0, 0), 128)
    except cv2.error:
        pass
    enclosed = ((bg == 255) & (ff != 128)).astype(np.uint8)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(enclosed)
    ppm = (ppm_x + ppm_y) / 2
    for i in range(1, nlab):
        area_px = stats[i, cv2.CC_STAT_AREA]
        comp = (lab == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        circ = _circularity(max(cnts, key=cv2.contourArea))
        # merge blowout: irregular shape, or implausibly large for a screw hole
        if circ < 0.45 or area_px > math.pi * ((15.0 / 2) * ppm) ** 2:
            dark_or_colored[lab == i] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(dark_or_colored, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if hierarchy is None:
        return out, t, 0
    hierarchy = hierarchy[0]
    h, w = mask.shape
    kept = 0
    for i, c in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue  # roots only; holes are re-cut below
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)
        touches = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
        if area > 0.002 * h * w and not touches:
            cv2.drawContours(out, [c], -1, 255, -1)
            kept += 1
            child = hierarchy[i][2]
            while child != -1:  # keep holes/slots as holes in the mask
                cv2.drawContours(out, [contours[child]], -1, 0, -1)
                child = hierarchy[child][0]
    return out, t, kept


def _blank_card_graphics(img: np.ndarray, ppm_x: float, ppm_y: float) -> None:
    """Zero out the card's own printed graphics (in place)."""
    from .card import CARD_H_MM, CARD_W_MM, MARKER_CENTERS, MARKER_MM

    def blank(x_mm, y_mm, w_mm, h_mm):
        x0, y0 = int(x_mm * ppm_x), int(y_mm * ppm_y)
        x1, y1 = int((x_mm + w_mm) * ppm_x), int((y_mm + h_mm) * ppm_y)
        img[max(0, y0):y1, max(0, x0):x1] = 0

    blank(0, 0, CARD_W_MM, 7); blank(0, CARD_H_MM - 7, CARD_W_MM, 7)
    blank(0, 0, 7, CARD_H_MM); blank(CARD_W_MM - 7, 0, 7, CARD_H_MM)
    for cx, cy in MARKER_CENTERS.values():
        s = MARKER_MM + 8
        blank(cx - s / 2, cy - s / 2, s, s)
    blank(65, CARD_H_MM - 24, 75, 17)  # scale bar + label
    blank(48, 4, 115, 17)  # title text strip


def segment_tap(rect_bgr: np.ndarray, taps_px: list[tuple[int, int]],
                eraser_px: list[tuple[int, int]] | None = None) -> np.ndarray | None:
    """GrabCut segmentation seeded by user taps on the part (tap-to-select).

    taps = sure-foreground discs, eraser taps = sure-background discs,
    image border ring = sure background. Returns a binary mask in
    rectified space, or None on failure (caller keeps the old mask).
    """
    from .card import CARD_H_MM, CARD_W_MM

    if not taps_px:
        return None
    h_px, w_px = rect_bgr.shape[:2]
    ppm_x, ppm_y = w_px / CARD_W_MM, h_px / CARD_H_MM
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    _blank_card_graphics(gray, ppm_x, ppm_y)  # seeds can't leak into graphics
    fg = np.zeros((h_px, w_px), np.uint8)
    for tx, ty in taps_px:
        cv2.circle(fg, (int(tx), int(ty)), 40, 255, -1)
    seed = np.full((h_px, w_px), cv2.GC_PR_BGD, np.uint8)
    seed[fg > 0] = cv2.GC_FGD
    seed[6:-6, 6:-6] = np.where(seed[6:-6, 6:-6] == cv2.GC_FGD,
                                cv2.GC_FGD, seed[6:-6, 6:-6])
    # border ring = sure background (but never override a tap)
    ring = np.zeros((h_px, w_px), np.uint8)
    ring[:6, :] = ring[-6:, :] = ring[:, :6] = ring[:, -6:] = 255
    seed[(ring > 0) & (fg == 0)] = cv2.GC_BGD
    seed[gray == 0] = np.where(seed[gray == 0] == cv2.GC_FGD,
                               cv2.GC_FGD, cv2.GC_BGD)
    for ex, ey in (eraser_px or []):
        cv2.circle(seed, (int(ex), int(ey)), 30, cv2.GC_BGD, -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rect_bgr, seed, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = (((seed == cv2.GC_FGD) | (seed == cv2.GC_PR_FGD)).astype(np.uint8)
           * 255)
    # re-cut through-holes: they show card-white through the metal, i.e.
    # small ROUND bright enclosures inside the GrabCut area. (Large or
    # irregular bright = glare blowout, stays part.) Threshold mode finds
    # holes as mask children; GrabCut's solid mask needs this explicit pass.
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    t_hole, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_in = ((gray >= t_hole) & (out > 0)).astype(np.uint8) * 255
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(bright_in)
    ppm = (ppm_x + ppm_y) / 2
    for i in range(1, nlab):
        area_px = stats[i, cv2.CC_STAT_AREA]
        d_eq = 2 * (area_px / 3.14159) ** 0.5 / ppm
        if not 1.6 <= d_eq <= 15.0:
            continue
        comp = (lab == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts and _circularity(max(cnts, key=cv2.contourArea)) >= 0.6:
            out[lab == i] = 0
    _blank_card_graphics(out, ppm_x, ppm_y)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out


def _segment_from_taps(img_bgr, H, ppm, taps, eraser, sam_first=True):
    """Tap -> mask in RECTIFIED space. SAM 3.1 (API) first, GrabCut fallback.

    Returns (mask_or_None, method_name). SAM runs on the ORIGINAL photo
    (its native input); the mask is warped into rect mm-space with the
    same homography as the image.
    """
    import numpy as _np

    method = "grabcut"
    if sam_first:
        try:
            from .sam_provider import segment_points
            got = segment_points(img_bgr, taps, eraser)
        except Exception:
            got = None
        if got is not None:
            sam_mask, _box, conf = got
            warped = cv2.warpPerspective(
                sam_mask, _np.diag([ppm, ppm, 1.0]) @ H,
                (int(210 * ppm), int(148 * ppm)))
            _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
            _blank_card_graphics(warped,
                                 warped.shape[1] / 210.0,
                                 warped.shape[0] / 148.0)
            return warped, f"sam:{conf:.2f}"
    scale = _np.diag([ppm, ppm, 1.0]) @ H
    taps_rect = []
    for tx, ty in taps:
        p = scale @ _np.array([tx, ty, 1.0])
        taps_rect.append((float(p[0] / p[2]), float(p[1] / p[2])))
    er_rect = []
    for ex, ey in (eraser or []):
        p = scale @ _np.array([ex, ey, 1.0])
        er_rect.append((float(p[0] / p[2]), float(p[1] / p[2])))
    import numpy as _np
    rect = cv2.warpPerspective(img_bgr, scale,
                               (int(210 * ppm), int(148 * ppm)))
    return segment_tap(rect, taps_rect, er_rect), method


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
                  thresh: int = 0, taps: list[tuple[int, int]] | None = None,
                  eraser: list[tuple[int, int]] | None = None,
                  sam_first: bool = True):
    """Full path: photo -> (spec, rectified_img, px_per_mm, info).

    thresh <= 0 (default) = Otsu auto-threshold per photo.
    taps/eraser are in ORIGINAL photo pixels.
    Tap mode: SAM 3.1 (API) first when sam_first, GrabCut offline fallback.
    """
    H, ids = detect_card(img_bgr)
    if H is None:
        return None, None, 0.0, {"card_markers": ids,
                                 "error": "Calibration card not found (need all 4 markers)."}
    rect, ppm = rectify(img_bgr, H)
    fragments = 1
    seg_method = "threshold"
    if taps:
        import numpy as _np
        mask, seg_method = _segment_from_taps(img_bgr, H, ppm, taps, eraser,
                                              sam_first)
        thresh_used = thresh
        if mask is None or int((mask > 0).sum()) < 50:
            return None, None, 0.0, {
                "card_markers": ids,
                "error": ("Tap segmentation found nothing — tap on the metal "
                          "face, not on holes or shadows."),
                "mask": mask if mask is not None else _np.zeros(rect.shape[:2],
                                                                 dtype=_np.uint8),
                "rect": rect, "thresh_used": thresh_used,
                "method": seg_method}
    elif mask is None:
        mask, thresh_used, fragments = segment_fallback(rect, thresh=thresh)
    else:
        thresh_used = thresh
        mask = cv2.warpPerspective(
            mask, np.diag([ppm, ppm, 1.0]) @ H,
            (rect.shape[1], rect.shape[0]))
    spec = measure_mask(mask, ppm)
    if spec is None:
        return None, None, 0.0, {
            "card_markers": ids,
            "error": ("Card found, but no measurable part outline survived "
                      "(mask empty or fragmented — check 'What I segmented')."),
            "mask": mask, "rect": rect, "thresh_used": thresh_used}
    if spec is not None:
        spec.outline_fragments = fragments
        spec.source = ("measured:" + seg_method) if taps else spec.source if hasattr(spec, "source") else "measured"
    if spec is None:
        return None, None, 0.0, {
            "card_markers": ids,
            "error": ("Card found, but no measurable part outline survived "
                      "(mask empty or fragmented — check 'What I segmented')."),
            "mask": mask, "rect": rect, "thresh_used": thresh_used}
    frac = float((mask > 0).mean())
    # Guard: if the "part" covers most of the card interior, we segmented
    # the CARD (dim light / glare), not the part. Refuse, don't CAD garbage.
    if frac > 0.55:
        return None, None, 0.0, {
            "card_markers": ids,
            "error": (f"Segmented {frac:.0%} of the card — that's the card, "
                      f"not the part (dim/flash light?). Retake with diffuse "
                      f"daylight, or check the 'What I segmented' layer."),
            "mask": mask, "rect": rect, "thresh_used": thresh_used}
    return spec, rect, ppm, {"card_markers": ids,
                             "mask": mask,
                             "rect": rect,
                             "thresh_used": thresh_used}


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

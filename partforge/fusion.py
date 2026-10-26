"""Multi-view fusion.

Top is the master: outline, holes, slots. Side/front are secondary views of
the SAME part in a different resting pose on the card:

- plate standing on its LONG edge (side) or SHORT edge (front): the long
  silhouette dimension must match top W or H; the short one is a
  **thickness vote** (measured, not guessed).
- bracket resting on one leg (side): an L-profile kink proposes
  type=bracket + bend position/angle, always as a confirmable question.

Guards are honest: non-rectangular side profiles and dimension mismatches
produce warnings, never numbers.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .spec import Bend, PartSpec

RECT_FILL = 0.85   # contour/bbox fill required to accept a thickness vote
AGREE_MM = 0.6     # votes within this spread agree
MATCH_TOL = 0.06   # long-dim match to top outline (relative)
KINK_MIN_MM = 4.0  # convexity-defect depth that counts as a bend
NONCONVEX = 0.93   # contour/hull area below this = not a plain plate


def rectangularity(outline_mm: list[list[float]]) -> float:
    cnt = (np.array(outline_mm) * 10).astype(np.int32).reshape(-1, 1, 2)
    _, _, w, h = cv2.boundingRect(cnt)
    if w * h <= 0:
        return 0.0
    return float(cv2.contourArea(cnt) / (w * h))


def _match(long_dim: float, top_w: float, top_h: float):
    """Match a side silhouette's long dim to top W or H. Returns 'W'/'H'/None."""
    ew = abs(long_dim - top_w) / max(top_w, 1e-6)
    eh = abs(long_dim - top_h) / max(top_h, 1e-6)
    if ew <= MATCH_TOL and ew <= eh:
        return "W"
    if eh <= MATCH_TOL and eh <= ew:
        return "H"
    return None


def detect_bend(outline_mm: list[list[float]]) -> tuple[Bend | None, dict]:
    """Find an L-kink in a side silhouette. Returns (bend, info)."""
    info: dict = {}
    if len(outline_mm) < 5:
        return None, info
    cnt = (np.array(outline_mm) * 10).astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area <= 0 or area / hull_area > NONCONVEX:
        return None, info
    try:
        defects = cv2.convexityDefects(
            cnt, cv2.convexHull(cnt, returnPoints=False))
    except cv2.error:
        return None, info
    if defects is None:
        return None, info
    deep = [row.reshape(-1) for row in defects
            if row.reshape(-1)[3] / 256.0 > KINK_MIN_MM]
    if len(deep) != 1:
        return None, info  # 0 = convex, 2+ = too complex for one bend
    far = cnt[deep[0][2]][0].astype(float) / 10.0  # mm
    approx = cv2.approxPolyDP(cnt, 15, True).reshape(-1, 2).astype(float) / 10.0
    dists = [math.hypot(p[0] - far[0], p[1] - far[1]) for p in approx]
    i = int(np.argmin(dists))
    v = approx[i]
    d1 = v - approx[i - 1]
    d2 = approx[(i + 1) % len(approx)] - v
    n1, n2 = math.hypot(*d1), math.hypot(*d2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None, info
    turn = math.degrees(math.acos(
        max(-1.0, min(1.0, float(d1 @ d2) / (n1 * n2)))))
    angle = 180.0 - turn
    if not 30.0 <= angle <= 170.0:
        return None, info
    x, y, w, h = cv2.boundingRect(cnt)
    x, y, w, h = x / 10.0, y / 10.0, w / 10.0, h / 10.0
    along_x = w >= h
    # bend pos measured from the L's outer corner along the matched axis
    pos = (v[0] - x) if along_x else (y + h - v[1])
    info = {"pos": round(float(pos), 1), "long": round(max(w, h), 1),
            "along": "x" if along_x else "y"}
    return (Bend(position_mm=round(float(pos), 1),
                 angle_deg=round(float(angle), 1), confidence=0.6), info)


def fuse(top: PartSpec, views: dict[str, PartSpec]) -> dict[str, str]:
    """Fuse side/front view specs into the top master (mutated)."""
    reports = {
        "top": (f"Top: {top.width_mm:.1f}x{top.height_mm:.1f} mm, "
                f"{len(top.holes)} holes, {len(top.slots)} slots")}
    votes: list[float] = []
    for role in ("side", "front"):
        v = views.get(role)
        if v is None or not v.outline_mm:
            continue
        bend, info = detect_bend(v.outline_mm)
        if bend is not None:
            if top.source == "measured" and top.type == "plate":
                match = _match(info["long"], top.width_mm, top.height_mm)
                # position varies along the matched top dim; bend line ∥ other axis
                axis = "y" if match == "W" else "x"
                dim = top.width_mm if match == "W" else top.height_mm
                scale = dim / info["long"] if match and info["long"] > 0 else 1.0
                top.type = "bracket"
                top.bend = Bend(position_mm=round(info["pos"] * scale, 1),
                                angle_deg=bend.angle_deg, axis=axis,
                                confidence=0.6)
                reports[role] = (
                    f"{role}: L-kink ~{bend.angle_deg:.0f} deg at "
                    f"{info['pos']:.1f} mm on the "
                    f"{'width' if axis == 'y' else 'height'} side "
                    f"-> bracket? Confirm the angle below.")
            else:
                reports[role] = (
                    f"{role}: L-kink found, but master is a manual "
                    f"{top.type} - left alone.")
            continue
        xs = [p[0] for p in v.outline_mm]
        ys = [p[1] for p in v.outline_mm]
        long, short = max(max(xs) - min(xs), max(ys) - min(ys)), \
            min(max(xs) - min(xs), max(ys) - min(ys))
        fill = rectangularity(v.outline_mm)
        m = _match(long, top.width_mm, top.height_mm)
        if fill < RECT_FILL:
            reports[role] = (
                f"{role}: profile is not a clean rectangle "
                f"(fill {fill:.2f}) - no thickness vote, check the pose.")
        elif m is None:
            reports[role] = (
                f"{role}: long side {long:.1f} mm matches neither top W "
                f"{top.width_mm:.1f} nor H {top.height_mm:.1f} - check the pose.")
        else:
            votes.append(round(short, 2))
            reports[role] = (
                f"{role}: long side {long:.1f} ~= top "
                f"{'W' if m == 'W' else 'H'} ok, thickness vote "
                f"{short:.1f} mm.")
    if votes:
        seen = [round(v, 2) for v in (top.thickness_votes or [])]
        for v in votes:
            if not any(abs(v - s) < 0.05 for s in seen):
                seen.append(v)
        top.thickness_votes = seen
        top.thickness_mm = round(sum(seen) / len(seen), 2)
        if len(seen) >= 2 and max(seen) - min(seen) <= AGREE_MM:
            top.thickness_confidence = 1.0
        elif len(seen) >= 2:
            top.thickness_confidence = 0.4
        else:
            top.thickness_confidence = max(top.thickness_confidence, 0.7)
    top.view_reports = reports
    return reports

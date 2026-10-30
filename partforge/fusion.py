"""Multi-view fusion.

Top is the master: outline, holes, slots. Front/side are secondary views
of the SAME part standing on edge: the long silhouette dimension must
match top W or H; the short one is a **thickness vote** (measured, not
guessed). Guards are honest: non-rectangular profiles and dimension
mismatches produce warnings, never numbers.

Bend detection lives in partforge/bends.py (mask-based, automatic).
"""
from __future__ import annotations

import cv2
import numpy as np

from .spec import PartSpec

RECT_FILL = 0.82   # contour/bbox fill required to accept a thickness vote
AGREE_MM = 0.6     # votes within this spread agree
MATCH_TOL = 0.06   # long-dim match to top outline (relative)


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


def fuse(top: PartSpec, views: dict[str, PartSpec]) -> dict[str, str]:
    """Fuse side/front view specs into the top master (mutated)."""
    reports = {
        "top": (f"Top: {top.width_mm:.1f}x{top.height_mm:.1f} mm, "
                f"{len(top.holes)} holes, {len(top.slots)} slots")}
    votes: list[float] = []
    roles: dict[str, float] = dict(top.thickness_vote_roles or {})
    for role in ("front", "side"):
        v = views.get(role)
        if v is None or not v.outline_mm:
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
                f"(fill {fill:.2f} < {RECT_FILL}) - no thickness vote, "
                f"check the pose.")
        elif m is None:
            reports[role] = (
                f"{role}: long side {long:.1f} mm matches neither top W "
                f"{top.width_mm:.1f} nor H {top.height_mm:.1f} - check the pose.")
        else:
            votes.append(round(short, 2))
            roles[role] = round(short, 2)
            reports[role] = (
                f"{role}: long side {long:.1f} ~= top "
                f"{'W' if m == 'W' else 'H'} ok, thickness vote "
                f"{short:.1f} mm.")
    if roles:
        seen = sorted(roles.values())
        top.thickness_vote_roles = roles
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

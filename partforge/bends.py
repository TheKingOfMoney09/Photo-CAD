"""Automatic bend detection from SAM part masks. No user input.

A flat plate photographed standing on edge is a thin, nearly rectangular
silhouette (bbox fill ~= 1). A bent part (L/Z/U flange out of plane) shows
two arms at an angle: the mask leaves a large empty corner inside its own
bounding box. All geometry is computed in mm via the ArUco scale.

Features (per side/front view mask, rectified mm space):
  fill        mask area / bbox area (flat bar ~= 0.95+, L ~= 0.15-0.6)
  solidity    mask area / convex-hull area (flat ~= 1.0, L ~= 0.3)
  empty_frac  largest empty bbox corner region / bbox area (L ~= 0.5+)
  runs        distinct thickness runs along the main axis (L splits runs)

Decision: has_bend if >= 2 of (fill < 0.75, solidity < 0.75,
empty_frac > 0.30) agree on a side/front view. Top view is a weak
secondary signal only (edge-thickness change along one line).
No side/front photo -> top-only fallback with capped confidence (0.5).

Output: {"has_bend", "bend_confidence", "bend_line_mm", "source_views",
         "features"} — JSON-serialisable. bend_line_mm = bend position
along the part in mm (inner corner of the L); None when no bend.
No CAD is built here (task: detection only).
"""
from __future__ import annotations

import cv2
import numpy as np

FILL_BENT = 0.75     # below -> bent vote (flat bars read 0.95+)
SOLID_BENT = 0.75    # below -> bent vote (flat reads ~1.0)
EMPTY_BENT = 0.30    # above -> bent vote (flat reads ~0.0)
TOP_ONLY_CAP = 0.5   # confidence cap when no side/front photo exists


def _largest_empty_corner(mask: np.ndarray):
    """Largest empty bbox corner region: connected components of the
    inverted mask inside the bbox, fraction of bbox area + its bbox."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw <= 0 or bh <= 0:
        return 0.0, None
    inv = np.zeros_like(mask)
    inv[y0:y1 + 1, x0:x1 + 1] = 255 - mask[y0:y1 + 1, x0:x1 + 1]
    nlab, _lab, stats, _cent = cv2.connectedComponentsWithStats(inv)
    if nlab <= 1:
        return 0.0, None
    best = max(range(1, nlab), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    area = float(stats[best, cv2.CC_STAT_AREA])
    bx = (float(stats[best, cv2.CC_STAT_LEFT]),
          float(stats[best, cv2.CC_STAT_TOP]),
          float(stats[best, cv2.CC_STAT_WIDTH]),
          float(stats[best, cv2.CC_STAT_HEIGHT]))
    return area / (bw * bh), bx


def _thickness_runs(mask: np.ndarray, axis: int) -> int:
    """Distinct thickness runs along the main axis (L splits runs)."""
    runs = (mask > 0).sum(axis=axis)
    runs = runs[runs > 0]
    if len(runs) == 0:
        return 0
    med = float(np.median(runs))
    # a run counts as distinct when it differs from the median by 2x
    lo = runs < med * 0.5
    hi = runs > med * 2.0
    return int(lo.sum() + hi.sum())


def _jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    if isinstance(v, (tuple, list)):
        return [float(x) for x in v]
    return v


def mask_features(mask: np.ndarray, ppm: float) -> dict:
    """Features of one rectified binary part mask. Pure function of pixels."""
    m = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {"fill": 0.0, "solidity": 0.0, "empty_frac": 0.0,
                "runs": 0, "bw_mm": 0.0, "bh_mm": 0.0}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    fill = float(area) / (bw * bh)
    solidity = float(area) / hull if hull > 0 else 0.0
    empty_frac, empty_box = _largest_empty_corner(m)
    axis = 0 if bw >= bh else 1
    runs = _thickness_runs(m, axis)
    return {"fill": round(fill, 3), "solidity": round(solidity, 3),
            "empty_frac": round(empty_frac, 3), "runs": runs,
            "bw_mm": round(float(bw) / ppm, 1),
            "bh_mm": round(float(bh) / ppm, 1),
            "empty_box_px": _jsonable(empty_box),
            "axis": "x" if axis == 0 else "y"}


def _bend_line_mm(mask: np.ndarray, feats: dict, ppm: float):
    """Bend line = inner corner of the L: the empty region's corner
    nearest the mask centroid, projected on the main axis, in mm."""
    eb = feats.get("empty_box_px")
    if not eb:
        return None
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    ex, ey, ew, eh = [int(v) for v in eb]
    # inner corner = empty-box corner closest to the mask centroid
    corners = [(ex, ey), (ex + ew, ey), (ex, ey + eh), (ex + ew, ey + eh)]
    ix, iy = min(corners, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    # project on the main axis, measured from the bbox min, in mm
    x0, y0 = int(xs.min()), int(ys.min())
    pos_px = (ix - x0) if feats.get("axis") == "x" else (iy - y0)
    return round(float(pos_px) / ppm, 1)


def detect_bend_view(mask: np.ndarray, ppm: float) -> dict:
    """One view's verdict from its SAM part mask (rectified, binary)."""
    feats = mask_features(mask, ppm)
    votes = [feats["fill"] < FILL_BENT, feats["solidity"] < SOLID_BENT,
             feats["empty_frac"] > EMPTY_BENT]
    has = sum(votes) >= 2
    agree = sum(votes) / 3.0
    conf = round(0.5 + 0.5 * agree if has else 0.5 * (1.0 - agree) + 0.2, 2)
    conf = min(0.97, max(0.05, conf))
    line = _bend_line_mm(mask, feats, ppm) if has else None
    return {"has_bend": has, "bend_confidence": conf,
            "bend_line_mm": line, "features": feats}


def detect_bends(masks: dict[str, tuple[np.ndarray, float]]) -> dict:
    """Fuse per-view verdicts. masks: view -> (rectified mask, ppm).

    Side/front views decide (majority); top is a weak secondary signal.
    No side/front -> top-only fallback, confidence capped at TOP_ONLY_CAP.
    """
    per_view = {}
    for view, (mask, ppm) in masks.items():
        if mask is None:
            continue
        per_view[view] = detect_bend_view(mask, ppm)
    voters = [v for k, v in per_view.items() if k in ("side", "front")]
    if voters:
        trues = sum(1 for v in voters if v["has_bend"])
        has = trues * 2 >= len(voters)
        conf = round(sum(v["bend_confidence"] for v in voters
                         if v["has_bend"] == has) / max(trues if has
                                                       else len(voters)
                                                       - trues, 1), 2)
        src = [k for k, v in per_view.items()
               if k in ("side", "front") and v["has_bend"] == has]
        line_v = next((v for v in voters if v["has_bend"] == has
                       and v["bend_line_mm"] is not None), None)
        line = line_v["bend_line_mm"] if line_v else None
    elif "top" in per_view:
        v = per_view["top"]
        has, conf, src = v["has_bend"], min(v["bend_confidence"],
                                            TOP_ONLY_CAP), ["top"]
        line = v["bend_line_mm"]
    else:
        return {"has_bend": False, "bend_confidence": 0.0,
                "bend_line_mm": None, "source_views": [],
                "features": {k: v["features"] for k, v in per_view.items()}}
    return {"has_bend": has, "bend_confidence": conf,
            "bend_line_mm": line, "source_views": src,
            "features": {k: v["features"] for k, v in per_view.items()}}

"""Printable calibration card.

A4-ish card with 4 ArUco markers whose centres sit at known mm positions.
Photograph the part on/near the card; detection gives a homography from
photo pixels to real millimetres. A 50 mm scale bar is printed as a
human-checkable backup.
"""
from __future__ import annotations

import cv2
import numpy as np

CARD_W_MM = 210.0
CARD_H_MM = 148.0
MARKER_MM = 28.0
# marker id -> centre position in card mm coords (origin top-left, x right, y down)
MARKER_CENTERS = {
    0: (24.0, 24.0),
    1: (CARD_W_MM - 24.0, 24.0),
    2: (CARD_W_MM - 24.0, CARD_H_MM - 24.0),
    3: (24.0, CARD_H_MM - 24.0),
}
PX_PER_MM_CARD = 8  # render resolution of the card image

_DICT = cv2.aruco.DICT_4X4_50


def render_card(px_per_mm: int = PX_PER_MM_CARD) -> np.ndarray:
    w, h = int(CARD_W_MM * px_per_mm), int(CARD_H_MM * px_per_mm)
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), (0, 0, 0), 3)
    dic = cv2.aruco.getPredefinedDictionary(_DICT)
    for mid, (cx, cy) in MARKER_CENTERS.items():
        m = cv2.aruco.generateImageMarker(dic, mid, int(MARKER_MM * px_per_mm))
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        x0, y0 = int((cx - MARKER_MM / 2) * px_per_mm), int((cy - MARKER_MM / 2) * px_per_mm)
        img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    # 50 mm scale bar
    bx, by = int(70 * px_per_mm), int((CARD_H_MM - 18) * px_per_mm)
    cv2.rectangle(img, (bx, by), (bx + int(50 * px_per_mm), by + int(4 * px_per_mm)), (0, 0, 0), -1)
    cv2.putText(img, "50 mm", (bx, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(img, "PartForge card - print 100%",
                (int(62 * px_per_mm), int(11 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
    cv2.putText(img, "part goes in the middle",
                (int(68 * px_per_mm), int(18 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 60, 60), 2)
    return img


def detect_card(img_bgr: np.ndarray):
    """Return (homography px->card-mm, marker_ids) or (None, [])."""
    dic = cv2.aruco.getPredefinedDictionary(_DICT)
    try:
        detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(img_bgr)
    except Exception:
        corners, ids = cv2.aruco.detectMarkers(img_bgr, dic)
    if ids is None:
        return None, []
    found = {}
    for c, mid in zip(corners, ids.flatten()):
        if int(mid) in MARKER_CENTERS:
            found[int(mid)] = c[0].mean(axis=0)  # centre px
    if len(found) < 4:
        return None, sorted(found)
    src = np.array([found[i] for i in sorted(found)], dtype=np.float64)
    dst = np.array([MARKER_CENTERS[i] for i in sorted(found)], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H, sorted(found)


def rectify(img_bgr: np.ndarray, H: np.ndarray, px_per_mm: float = 8.0):
    """Warp photo into top-down card-mm space. Returns (image, mm->px scale)."""
    out_w, out_h = int(CARD_W_MM * px_per_mm), int(CARD_H_MM * px_per_mm)
    scale = np.diag([px_per_mm, px_per_mm, 1.0])
    warped = cv2.warpPerspective(img_bgr, scale @ H, (out_w, out_h))
    return warped, px_per_mm

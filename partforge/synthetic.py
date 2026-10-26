"""Synthetic bench photo with known ground truth (demo + regression test)."""
from __future__ import annotations

import cv2
import numpy as np

from .card import PX_PER_MM_CARD, render_card

GT = {"w": 80.0, "h": 40.0,
      "holes": [(10, 10, 5.0), (70, 10, 5.0), (10, 30, 4.0), (70, 30, 4.0)],
      "slot": (40, 20, 18, 5)}
PLATE_X, PLATE_Y = 60.0, 50.0


def make_synthetic_photo() -> np.ndarray:
    ppm = PX_PER_MM_CARD
    scene = render_card()
    x0, y0 = int(PLATE_X * ppm), int(PLATE_Y * ppm)
    x1, y1 = int((PLATE_X + GT["w"]) * ppm), int((PLATE_Y + GT["h"]) * ppm)
    cv2.rectangle(scene, (x0, y0), (x1, y1), (115, 115, 120), -1)
    cv2.rectangle(scene, (x0, y0), (x1, y1), (40, 40, 40), 3)
    for hx, hy, d in GT["holes"]:
        c = (int((PLATE_X + hx) * ppm), int((PLATE_Y + hy) * ppm))
        cv2.circle(scene, c, int(d / 2 * ppm), (245, 245, 245), -1)
        cv2.circle(scene, c, int(d / 2 * ppm), (40, 40, 40), 2)
    sx, sy, sl, sw = GT["slot"]
    cx, cy = int((PLATE_X + sx) * ppm), int((PLATE_Y + sy) * ppm)
    cv2.rectangle(scene, (cx - int((sl - sw) / 2 * ppm), cy - int(sw / 2 * ppm)),
                  (cx + int((sl - sw) / 2 * ppm), cy + int(sw / 2 * ppm)),
                  (245, 245, 245), -1)
    for dx in (-(sl - sw) / 2, (sl - sw) / 2):
        cv2.circle(scene, (cx + int(dx * ppm), cy), int(sw / 2 * ppm),
                   (245, 245, 245), -1)
    photo = np.full((1300, 1700, 3), (70, 80, 90), np.uint8)
    src = np.float32([[0, 0], [scene.shape[1], 0],
                      [scene.shape[1], scene.shape[0]], [0, scene.shape[0]]])
    dst = np.float32([[180, 120], [1560, 60], [1640, 1180], [90, 1240]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(scene, M, (photo.shape[1], photo.shape[0]))
    m = cv2.warpPerspective(np.full(scene.shape[:2], 255, np.uint8), M,
                            (photo.shape[1], photo.shape[0]))
    photo[m > 0] = warped[m > 0]
    return photo

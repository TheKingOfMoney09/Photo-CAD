"""Headless SAM tap test on the user's real bracket photos."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from dotenv import load_dotenv

load_dotenv()

from partforge.card import detect_card, rectify
from partforge.measure import measure_mask, measure_photo
from partforge.sam_provider import segment_points

TOP = r"C:\Users\david\Downloads\top (2).jpg"
SIDE = r"C:\Users\david\Downloads\side (2).jpg"

for path, taps in [
    (TOP, [(1050, 1250)]),    # guess: bracket face centre-ish
    (SIDE, [(1000, 1300)]),
]:
    img = cv2.imread(path)
    print("=" * 60)
    print(path, img.shape)
    H, ids = detect_card(img)
    print("markers:", ids)
    if H is None:
        continue
    got = segment_points(img, taps, None)
    if got is None:
        print("SAM: None (rejected or failed)")
        continue
    mask, box, conf = got
    print(f"SAM: conf {conf:.3f} box {tuple(round(float(v), 1) for v in box)} "
          f"frac {float((mask > 0).mean()):.4f}")
    rect, ppm = rectify(img, H)
    import numpy as _np
    warped = cv2.warpPerspective(mask, _np.diag([ppm, ppm, 1.0]) @ H,
                                 (rect.shape[1], rect.shape[0]))
    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
    spec = measure_mask(warped, ppm)
    print("measured:", spec.summary() if spec else None)
    if spec:
        print("  holes:", [(h.id, h.x, h.y, h.d) for h in spec.holes])
        print("  slots:", [(s.id, s.cx, s.cy, s.length, s.width) for s in spec.slots])

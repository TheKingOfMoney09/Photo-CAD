"""Synthetic end-to-end test.

Renders a fake bench photo: calibration card + a known 80x40 plate with
4 holes and 1 slot, warped by a perspective transform. Runs the real
measurement pipeline and prints measured vs ground-truth errors.
Run: .venv\\Scripts\\python.exe scripts\\demo_synthetic.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partforge.measure import measure_photo
from partforge.synthetic import GT, make_synthetic_photo as make_photo


def main() -> int:
    photo = make_photo()
    cv2.imwrite("tmp_synthetic_photo.jpg", photo)
    spec, rect, ppm, info = measure_photo(photo)
    print("info:", info)
    if spec is None:
        print("FAIL: no spec measured")
        return 1
    cv2.imwrite("tmp_synthetic_rectified.jpg", rect)
    print(f"outline: {spec.width_mm:.2f} x {spec.height_mm:.2f} mm "
          f"(GT {GT['w']} x {GT['h']}) -> err "
          f"{abs(spec.width_mm-GT['w']):.2f} / {abs(spec.height_mm-GT['h']):.2f} mm")
    ok = True
    for gx, gy, gd in GT["holes"]:
        best = min(spec.holes, key=lambda h: abs(h.x - gx) + abs(h.y - gy),
                   default=None)
        if best is None:
            print(f"hole GT ({gx},{gy},d{gd}): MISSING"); ok = False; continue
        err = (abs(best.x - gx), abs(best.y - gy), abs(best.d - gd))
        print(f"hole GT ({gx},{gy},d{gd}): measured ({best.x:.2f},{best.y:.2f},"
              f"d{best.d:.2f}) err x/y/d = {err[0]:.2f}/{err[1]:.2f}/{err[2]:.2f} mm")
        ok = ok and max(err) < 1.0
    print(f"slots measured: {[(s.cx, s.cy, s.length, s.width) for s in spec.slots]} "
          f"(GT {GT['slot']})")
    from partforge import cad
    spec.thickness_mm = 3.0
    stl = cad.export_bytes(spec, "stl")
    step = cad.export_bytes(spec, "step")
    print(f"export: STL {len(stl)} bytes, STEP {len(step)} bytes")
    print("questions:", [q.id for q in spec.questions()])
    print("RESULT:", "PASS" if ok else "CHECK errors above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

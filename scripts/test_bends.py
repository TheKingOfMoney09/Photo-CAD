"""Unit test: bend detector on synthetic masks (no photos, no SAM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from partforge.bends import detect_bend_view, detect_bends


def l_mask(size=200, t=24):
    m = np.zeros((size, size), np.uint8)
    m[:, :t] = 255            # vertical arm
    m[-t:, :] = 255           # horizontal foot
    return m


def rect_mask(w=160, h=24, size=200):
    m = np.zeros((size, size), np.uint8)
    m[(size - h) // 2:(size + h) // 2, (size - w) // 2:(size + w) // 2] = 255
    return m


def main() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name +
              (f" — {detail}" if detail else ""))
        ok = ok and cond

    r = detect_bend_view(l_mask(), 8.0)
    check("L mask -> bend", r["has_bend"] is True,
          f"conf {r['bend_confidence']} line {r['bend_line_mm']} "
          f"feats {r['features']}")
    check("L bend line sane",
          r["bend_line_mm"] is not None and 0 < r["bend_line_mm"] < 25,
          f"line {r['bend_line_mm']}")

    r = detect_bend_view(rect_mask(), 8.0)
    check("rect mask -> flat", r["has_bend"] is False,
          f"conf {r['bend_confidence']} feats {r['features']}")

    fused = detect_bends({"side": (l_mask(), 8.0), "top": (rect_mask(), 8.0)})
    check("fused side-L + top-rect -> bend", fused["has_bend"] is True,
          f"conf {fused['bend_confidence']} src {fused['source_views']}")
    fused = detect_bends({"top": (rect_mask(), 8.0)})
    check("top-only flat capped", fused["has_bend"] is False
          and fused["bend_confidence"] <= 0.5,
          f"conf {fused['bend_confidence']}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

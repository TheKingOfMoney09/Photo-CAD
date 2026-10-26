"""Multi-view regression: top + side + bracket-side synthetic photos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partforge import cad
from partforge.fusion import fuse
from partforge.measure import measure_photo
from partforge.synthetic import GT, make_synthetic_bracket_side, \
    make_synthetic_photo, make_synthetic_side


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    ok = True
    top, _, _, _ = measure_photo(make_synthetic_photo())
    ok &= check("top measured", top is not None and len(top.holes) == 4,
                top.summary() if top else "none")

    side, _, _, _ = measure_photo(make_synthetic_side("W"))
    ok &= check("side measured", side is not None,
                f"{side.width_mm:.1f}x{side.height_mm:.1f}" if side else "none")
    fuse(top, {"side": side})
    th = top.thickness_mm
    ok &= check("thickness vote", th is not None and abs(th - GT["t"]) <= 0.8,
                f"vote {th} vs GT {GT['t']}")
    ok &= check("side matched long edge",
                "long side" in str(top.view_reports.get("side", "")),
                str(top.view_reports.get("side", ""))[:100])

    front, _, _, _ = measure_photo(make_synthetic_side("H"))
    fuse(top, {"side": side, "front": front})
    th2 = top.thickness_mm
    agree = len(top.thickness_votes) == 2
    ok &= check("two votes agree", agree and top.thickness_confidence == 1.0,
                f"votes {top.thickness_votes} -> {th2}")

    br_side, _, _, _ = measure_photo(make_synthetic_bracket_side())
    tb = top.model_copy(deep=True)
    tb.type, tb.thickness_mm, tb.bend = "plate", 3.0, None
    fuse(tb, {"side": br_side})
    bend = tb.bend
    ok &= check("bracket proposed", tb.type == "bracket" and bend is not None,
                str(tb.view_reports.get("side", ""))[:110])
    if bend is not None:
        ok &= check("bend angle ~90", abs(bend.angle_deg - 90) <= 8,
                    f"{bend.angle_deg:.1f} deg at {bend.position_mm:.1f} mm")
        s_ = cad.build_solid(tb)
        ok &= check("bracket CAD builds", s_.volume > 0,
                    f"vol {s_.volume:.0f}")
        stl = cad.export_bytes(tb, "stl")
        ok &= check("bracket STL non-empty", len(stl) > 1000, str(len(stl)))

    stl = cad.export_bytes(top, "stl")
    step = cad.export_bytes(top, "step")
    ok &= check("STL/STEP export", len(stl) > 1000 and len(step) > 1000,
                f"{len(stl)}/{len(step)} bytes")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

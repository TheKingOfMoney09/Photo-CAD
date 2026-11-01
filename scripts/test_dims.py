"""Unit tests: dim_xy constraints (hole↔edge, hole↔hole, validation)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partforge.spec import Hole, PartSpec, rectangle_spec


def make_plate():
    s = rectangle_spec(100, 60, 3.0)
    s.holes = [Hole(id="H1", x=20.0, y=30.0, d=5.0),
               Hole(id="H2", x=60.0, y=30.0, d=5.0)]
    return s


def main() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name +
              (f" — {detail}" if detail else ""))
        ok = ok and cond

    # 1. hole 20 mm from left edge -> set 25 mm: moves 5, others stay
    s = make_plate()
    check("H1 x-edge is 20", s.dim_current("H1", "edge_left", "x") == 20.0)
    err = s.apply_dim("H1", "edge_left", "x", 25.0)
    check("edge edit applies", err == "", err)
    check("H1 moved +5", s._hole("H1").x == 25.0, f"x={s._hole('H1').x}")
    check("H1 y kept", s._hole("H1").y == 30.0)
    check("H2 untouched", (s._hole("H2").x, s._hole("H2").y) == (60.0, 30.0))
    check("constraint stored",
          len(s.dims) == 1 and s.dims[0].value_mm == 25.0)

    # 2. two holes: set X distance 30 -> second moves, first stays
    s = make_plate()
    check("H1-H2 dx is 40", s.dim_current("H1", "H2", "x") == 40.0)
    err = s.apply_dim("H1", "H2", "x", 30.0)
    check("pair edit applies", err == "", err)
    check("H2 moved to 50", s._hole("H2").x == 50.0,
          f"x={s._hole('H2').x}")
    check("H1 fixed at 20", s._hole("H1").x == 20.0)

    # 3. edit pushing a hole outside the outline is rejected, value kept
    s = make_plate()
    err = s.apply_dim("H1", "edge_left", "x", -5.0)
    check("outside rejected", err.startswith("Invalid:"),
          err or "no error!")
    check("old value kept", s._hole("H1").x == 20.0)
    err = s.apply_dim("H1", "H2", "x", 200.0)
    check("overlap/escape rejected", err.startswith("Invalid:"), err)
    check("H2 kept", s._hole("H2").x == 60.0)

    # 4. overlap with another hole rejected
    s = make_plate()
    err = s.apply_dim("H1", "H2", "x", 4.0)
    check("overlap rejected", err.startswith("Invalid:"), err)

    # 5. constraints survive a rebuild (reapply)
    s = make_plate()
    s.apply_dim("H1", "edge_left", "x", 25.0)
    s2 = PartSpec(**s.model_dump())
    flagged = s2.reapply_dims()
    check("reapply clean", flagged == [], str(flagged))
    check("reapplied value", s2._hole("H1").x == 25.0)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

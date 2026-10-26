"""Deterministic spoken/typed correction parsing.

'the hole is 5, not 4' / 'hole 2 is 5' / 'thickness 4' -> structured edit.
Works offline; Muse parsing can replace this later, same output shape.
"""
from __future__ import annotations

import re


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_edit_text(text: str, spec) -> dict | None:
    t = text.lower().strip()
    # thickness
    m = re.search(r"thickness[^0-9]*([0-9]+(?:[.,][0-9]+)?)", t)
    if m:
        return {"field": "thickness_mm", "value": _num(m.group(1))}
    # hole by id: 'hole 2 is 5' / 'h2 = 5'
    m = re.search(r"(?:hole|h)\s*([0-9]+)[^0-9]*([0-9]+(?:[.,][0-9]+)?)\s*(?:mm)?\s*$", t)
    if m and spec.holes:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(spec.holes):
            return {"field": f"hole:{spec.holes[idx].id}.d", "value": _num(m.group(2))}
    # 'the hole is 5, not 4' -> the hole whose current d matches the 'not' value,
    # else the only / first hole.
    m = re.search(r"hole[^0-9]*?(?:is\s*)?([0-9]+(?:[.,][0-9]+)?)\s*,?\s*not\s*([0-9]+(?:[.,][0-9]+)?)", t)
    if m and spec.holes:
        new, old = _num(m.group(1)), _num(m.group(2))
        for h in spec.holes:
            if abs(h.d - old) < 0.6:
                return {"field": f"hole:{h.id}.d", "value": new}
        return {"field": f"hole:{spec.holes[0].id}.d", "value": new}
    m = re.search(r"hole[^0-9]*([0-9]+(?:[.,][0-9]+)?)", t)
    if m and len(spec.holes) == 1:
        return {"field": f"hole:{spec.holes[0].id}.d", "value": _num(m.group(1))}
    # overall size: 'width 90' / 'height 45' / 'make it 90 by 45'
    m = re.search(r"(?:width|wide)[^0-9]*([0-9]+(?:[.,][0-9]+)?)", t)
    if m:
        return {"field": "width_mm", "value": _num(m.group(1))}
    m = re.search(r"(?:height|tall|long)[^0-9]*([0-9]+(?:[.,][0-9]+)?)", t)
    if m:
        return {"field": "height_mm", "value": _num(m.group(1))}
    return None


def apply_edit(spec, edit: dict) -> str:
    f, v = edit["field"], edit["value"]
    if f == "thickness_mm":
        spec.thickness_mm = v
    elif f.startswith("hole:"):
        hid = f.split(":")[1].split(".")[0]
        for h in spec.holes:
            if h.id == hid:
                h.d = v
                h.confidence = 1.0
    elif f == "width_mm":
        spec.rescale_outline(v, spec.height_mm)
    elif f == "height_mm":
        spec.rescale_outline(spec.width_mm, v)
    else:
        return f"Not applied: {f}"
    return f"Set {f} = {v}"

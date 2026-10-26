"""Parametric CAD from a PartSpec (build123d -> STEP / STL).

Spec is the source of truth; this module only turns numbers into geometry.
A print-tolerance is applied to holes/slots at export time for FDM fit.
"""
from __future__ import annotations

import io
import os
import tempfile

from build123d import (
    Circle, Polygon, Pos, Rot, SlotOverall, export_step, export_stl, extrude,
)

from .spec import PartSpec


def build_solid(spec: PartSpec, hole_tolerance_mm: float = 0.0):
    t = spec.thickness_mm or 3.0
    pts = [(p[0], p[1]) for p in spec.outline_mm]
    if len(pts) < 3:
        raise ValueError("Outline needs at least 3 points")
    profile = Polygon(*pts, align=None)
    for hole in spec.holes:
        profile -= Pos(hole.x, hole.y) * Circle(
            radius=(hole.d + hole_tolerance_mm) / 2)
    for slot in spec.slots:
        profile -= Pos(slot.cx, slot.cy) * Rot(0, 0, slot.angle_deg) * SlotOverall(
            width=slot.length + hole_tolerance_mm,
            height=slot.width + hole_tolerance_mm)
    return extrude(profile, amount=t)


def export_bytes(spec: PartSpec, fmt: str, hole_tolerance_mm: float = 0.2) -> bytes:
    solid = build_solid(spec, hole_tolerance_mm=hole_tolerance_mm)
    ext = ".stl" if fmt == "stl" else ".step"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        if fmt == "stl":
            export_stl(solid, path)
        elif fmt == "step":
            export_step(solid, path)
        else:
            raise ValueError(fmt)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def top_view_png(spec: PartSpec, selected: str | None = None) -> bytes:
    """2D dimension diagram: outline, numbered holes/slots, overall dims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    poly = list(spec.outline_mm) + [spec.outline_mm[0]]
    ax.plot([p[0] for p in poly], [p[1] for p in poly], "k-", lw=2)
    for h in spec.holes:
        sel = selected == h.id
        color = "red" if sel else "tab:red"
        ax.add_patch(plt.Circle((h.x, h.y), h.d / 2, fill=False,
                                color=color, lw=3 if sel else 1.5))
        ax.plot(h.x, h.y, "+", color=color)
        ax.annotate(f"{h.id} d={h.d:.1f}", (h.x, h.y),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=color)
    for s in spec.slots:
        sel = selected == s.id
        color = "red" if sel else "tab:orange"
        ax.add_patch(plt.Circle((s.cx, s.cy), s.width / 2, fill=False, color=color))
        ax.annotate(f"{s.id} {s.length:.0f}x{s.width:.1f}", (s.cx, s.cy),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=color)
    ax.set_xlim(-8, spec.width_mm + 8)
    ax.set_ylim(-8, spec.height_mm + 10)
    ax.set_aspect("equal")
    ax.set_xlabel(f"width {spec.width_mm:.1f} mm")
    ax.set_ylabel(f"height {spec.height_mm:.1f} mm")
    ax.set_title(f"{spec.width_mm:.1f} x {spec.height_mm:.1f} x "
                 f"{spec.thickness_mm or '?'} mm")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png")
    plt.close(fig)
    return out.getvalue()

"""Parametric CAD from a PartSpec (build123d -> STEP / STL).

Spec is the source of truth; this module only turns numbers into geometry.
A print-tolerance is applied to holes/slots at export time for FDM fit.

Bracket support is deliberately narrow: one straight bend, outline treated
as its flat-pattern rectangle. Top-view holes cut on the leg they fall on;
features on the folded leg are kept in the spec but not cut (flagged in UI).
"""
from __future__ import annotations

import io
import math
import os
import tempfile

from build123d import (
    Axis, Box, Circle, Polygon, Pos, Rot, SlotOverall,
    export_step, export_stl, extrude,
)

from .spec import PartSpec


def build_solid(spec: PartSpec, hole_tolerance_mm: float = 0.0):
    t = spec.thickness_mm or 3.0
    if spec.type == "bracket" and spec.bend is not None:
        return _build_bracket(spec, t, hole_tolerance_mm)
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


def _build_bracket(spec: PartSpec, t: float, tol: float):
    xs = [p[0] for p in spec.outline_mm]
    ys = [p[1] for p in spec.outline_mm]
    minx, miny = min(xs), min(ys)
    W, H = max(xs) - minx, max(ys) - miny
    axis, pos = spec.bend.axis, spec.bend.position_mm
    if axis == "y":
        fixed, moving, leg_h = pos, W - pos, H
        if fixed <= 0.5 or moving <= 0.5:
            raise ValueError(
                f"Bend at {pos:.1f} mm leaves a degenerate leg "
                f"({fixed:.1f}/{moving:.1f} mm) — fix the position.")
        fixed_box = Box(fixed, leg_h, t)
        moving_box = Box(moving, leg_h, t).translate((pos, 0, 0))
        bend_line = Axis((pos, 0, 0), (0, 1, 0))
        fixed_span = (0.0, pos)
    else:
        fixed, moving, leg_h = pos, H - pos, W
        if fixed <= 0.5 or moving <= 0.5:
            raise ValueError(
                f"Bend at {pos:.1f} mm leaves a degenerate leg "
                f"({fixed:.1f}/{moving:.1f} mm) — fix the position.")
        fixed_box = Box(leg_h, fixed, t)
        moving_box = Box(leg_h, moving, t).translate((0, pos, 0))
        bend_line = Axis((0, pos, 0), (1, 0, 0))
        fixed_span = (0.0, pos)
    fold = math.radians(180.0 - spec.bend.angle_deg)  # away from flat
    base = fixed_box.fuse(moving_box.rotate(bend_line, math.degrees(fold)))

    def on_fixed(x: float, y: float) -> bool:
        return (0 <= x <= fixed) if axis == "y" else (0 <= y <= fixed)

    cut, kept = 0, 0
    for hole in spec.holes:
        hx, hy = hole.x - minx, hole.y - miny
        if not on_fixed(hx, hy):
            kept += 1
            continue
        punch = extrude(
            Pos(hx, hy) * Circle(radius=(hole.d + tol) / 2),
            amount=t * 3).translate((0, 0, -t))
        base -= punch
        cut += 1
    for slot in spec.slots:
        sx_, sy_ = slot.cx - minx, slot.cy - miny
        if not on_fixed(sx_, sy_):
            kept += 1
            continue
        punch = extrude(
            Pos(sx_, sy_) * Rot(0, 0, slot.angle_deg) * SlotOverall(
                width=slot.length + tol, height=slot.width + tol),
            amount=t * 3).translate((0, 0, -t))
        base -= punch
        cut += 1
    base._partforge_note = (f"{cut} features cut on flat leg, "
                            f"{kept} on folded leg kept in spec only")
    return base.translate((-minx, -miny, 0))


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
    if spec.type == "bracket" and spec.bend is not None:
        b = spec.bend
        if b.axis == "y":
            ax.axvline(b.position_mm, color="tab:purple", ls="--", lw=1.5)
            ax.text(b.position_mm, spec.height_mm + 4,
                    f"bend {b.angle_deg:.0f}°", color="tab:purple",
                    ha="center", fontsize=9)
        else:
            ax.axhline(b.position_mm, color="tab:purple", ls="--", lw=1.5)
            ax.text(-6, b.position_mm, f"bend {b.angle_deg:.0f}°",
                    color="tab:purple", va="center", fontsize=9)
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
    title = f"{spec.width_mm:.1f} x {spec.height_mm:.1f} x {spec.thickness_mm or '?'} mm"
    if spec.type == "bracket" and spec.bend is not None:
        title += f" + bend {spec.bend.angle_deg:.0f}°"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png")
    plt.close(fig)
    return out.getvalue()

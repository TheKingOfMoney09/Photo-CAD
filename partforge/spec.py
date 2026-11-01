"""PartForge part specification.

The spec is the single source of truth. Photos are measured into a spec,
the user (or voice) edits the spec, and CAD is generated from the spec.
No module other than measurement creates dimensions, and measurement
never invents them: anything uncertain becomes a Question.
"""
from __future__ import annotations

import math
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field


class Hole(BaseModel):
    id: str
    x: float  # mm, from outline bbox min corner
    y: float  # mm
    d: float  # diameter mm
    confidence: float = 1.0  # 0..1 from measurement


class Slot(BaseModel):
    id: str
    cx: float
    cy: float
    length: float  # overall length mm (incl. end radii)
    width: float
    angle_deg: float = 0.0
    confidence: float = 1.0


class DimConstraint(BaseModel):
    """A hole↔edge or hole↔hole X/Y dimension, stored in the shape IR.

    a is always a hole id. b is a hole id or an edge id
    (edge_left/edge_right/edge_bottom/edge_top = outline bbox sides).
    axis is "x" or "y". value_mm is the target distance. flagged is set
    when the constraint could not be applied on a rebuild.
    """
    type: Literal["dim_xy"] = "dim_xy"
    a: str
    b: str
    axis: Literal["x", "y"] = "x"
    value_mm: float = 0.0
    flagged: str = ""


class Bend(BaseModel):
    """Detected bend state. Set ONLY by partforge/bends.py, never by UI."""
    position_mm: float = 0.0  # bend line position along the part, mm
    angle_deg: float = 90.0
    radius_mm: float = 2.0
    axis: str = "y"  # bend line runs parallel to this spec axis
    confidence: float = 0.0  # detector confidence (bend_confidence)


class Question(BaseModel):
    id: str
    field: str  # spec field this answers, e.g. "thickness_mm" or "hole:H2.d"
    text: str
    kind: Literal["number", "choice"] = "number"
    choices: list[float] = Field(default_factory=list)
    suggested: Optional[float] = None


# Standard clearance-ish hole sizes we snap-suggest (mm)
STANDARD_HOLES = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]


class PartSpec(BaseModel):
    type: Literal["plate", "bracket"] = "plate"
    thickness_mm: Optional[float] = None
    thickness_votes: list[float] = Field(default_factory=list)
    thickness_vote_roles: dict[str, float] = Field(default_factory=dict)
    thickness_confidence: float = 0.0  # 0..1; photo can never make this 1.0
    view_reports: dict[str, str] = Field(default_factory=dict)  # per-view fusion notes
    outline_fragments: int = 1  # root contours kept; >1 = split mask, low trust
    outline_mm: list[list[float]] = Field(default_factory=list)  # polygon, mm
    holes: list[Hole] = Field(default_factory=list)
    slots: list[Slot] = Field(default_factory=list)
    bend: Optional[Bend] = None
    dims: list[DimConstraint] = Field(default_factory=list)  # dimension constraints
    source: str = "measured"
    offset_mm: list[float] = Field(default_factory=lambda: [0.0, 0.0])  # bbox min in photo mm coords

    # -- derived ------------------------------------------------------------
    @property
    def width_mm(self) -> float:
        if not self.outline_mm:
            return 0.0
        xs = [p[0] for p in self.outline_mm]
        return max(xs) - min(xs)

    @property
    def height_mm(self) -> float:
        if not self.outline_mm:
            return 0.0
        ys = [p[1] for p in self.outline_mm]
        return max(ys) - min(ys)

    def rescale_outline(self, new_w: float, new_h: float) -> None:
        """Scale outline + feature positions to a corrected overall size."""
        w, h = self.width_mm, self.height_mm
        if w <= 0 or h <= 0:
            return
        sx, sy = new_w / w, new_h / h
        minx = min(p[0] for p in self.outline_mm)
        miny = min(p[1] for p in self.outline_mm)
        self.outline_mm = [
            [(p[0] - minx) * sx, (p[1] - miny) * sy] for p in self.outline_mm
        ]
        for hole in self.holes:
            hole.x, hole.y = (hole.x - minx) * sx, (hole.y - miny) * sy
        for slot in self.slots:
            slot.cx, slot.cy = (slot.cx - minx) * sx, (slot.cy - miny) * sy

    # -- the "asks" -----------------------------------------------------------
    def questions(self) -> list[Question]:
        qs: list[Question] = []
        if getattr(self, "outline_fragments", 1) > 1:
            qs.append(Question(
                id="fragmented_mask", field="fragment_note",
                text=(f"Segmentation split the part into "
                       f"{self.outline_fragments} pieces (glare/shadows?). "
                       f"Outline {self.width_mm:.1f}x{self.height_mm:.1f} may be "
                       f"wrong — retake with diffuse light, or fix it by hand."),
                kind="number", suggested=self.width_mm))
        if self.thickness_mm is None:
            qs.append(Question(
                id="thickness", field="thickness_mm",
                text="Thickness: no side or front view yet. "
                     "Photograph the part standing on edge, or caliper it?",
                kind="number", suggested=3.0))
        elif self.thickness_votes and self.thickness_confidence < 1.0:
            votes = ", ".join(f"{v:.1f}" for v in self.thickness_votes)
            qs.append(Question(
                id="thickness_confirm", field="thickness_mm",
                text=f"Thickness votes disagree ({votes} mm). Caliper wins — "
                     f"sides on my current average {self.thickness_mm:.1f} mm?",
                kind="number", suggested=self.thickness_mm))
        if self.type == "bracket" and self.bend is not None and \
                self.bend.confidence < 1.0:
            qs.append(Question(
                id="bend_angle_confirm", field="bend.angle_deg",
                text=f"Detector found a {self.bend.angle_deg:.0f}° bend at "
                     f"{self.bend.position_mm:.1f} mm "
                     f"(confidence {self.bend.confidence:.2f}). "
                     f"Confirm the angle?",
                kind="number", suggested=self.bend.angle_deg))
        for hole in self.holes:
            near = [s for s in STANDARD_HOLES if abs(s - hole.d) <= 0.45]
            ambiguous = len(near) >= 1 and abs(near[0] - hole.d) > 0.15
            if hole.confidence < 0.8 or ambiguous:
                choices = near if near else STANDARD_HOLES
                qs.append(Question(
                    id=f"{hole.id}_d", field=f"hole:{hole.id}.d",
                    text=f"Hole {hole.id}: I measured {hole.d:.1f} mm but I am "
                         f"not sure. Is it a standard size?",
                    kind="choice", choices=choices, suggested=hole.d))
            edge = min(hole.x, hole.y,
                       self.width_mm - hole.x, self.height_mm - hole.y)
            if edge < hole.d / 2 + 1.5:
                qs.append(Question(
                    id=f"{hole.id}_edge", field=f"hole:{hole.id}.d",
                    text=f"Hole {hole.id} is very close to an edge "
                         f"({edge:.1f} mm). Keep diameter {hole.d:.1f} mm?",
                    kind="choice", choices=[hole.d], suggested=hole.d))
        for slot in self.slots:
            if slot.confidence < 0.8:
                qs.append(Question(
                    id=f"{slot.id}_w", field=f"slot:{slot.id}.width",
                    text=f"Slot {slot.id}: width measured {slot.width:.1f} mm, "
                         f"low confidence. Confirm width?",
                    kind="number", suggested=slot.width))
        if self.type == "bracket" and self.bend is None:
            # detector found no bend geometry yet (e.g. side view missing):
            # ask once, do not invent a position.
            qs.append(Question(
                id="bend_angle", field="bend.angle_deg",
                text="Bend detected but the side view is missing. "
                     "What is the bend angle?",
                kind="number", suggested=90.0))
        return qs

    def apply_answer(self, question: Question, value: float) -> None:
        f = question.field
        if f == "fragment_note":
            return  # informational only; user fixes the photo or edits dims
        if f == "thickness_mm":
            self.thickness_mm = value
            self.thickness_confidence = 1.0  # user (caliper) outranks photo votes
        elif f.startswith("hole:"):
            hid = f.split(":")[1].split(".")[0]
            for hole in self.holes:
                if hole.id == hid:
                    hole.d = value
                    hole.confidence = 1.0
        elif f.startswith("slot:"):
            sid = f.split(":")[1].split(".")[0]
            for slot in self.slots:
                if slot.id == sid:
                    slot.width = value
                    slot.confidence = 1.0
        elif f == "bend.angle_deg":
            # user confirms/corrects the ANGLE only; position stays detected.
            if self.bend is None:
                self.bend = Bend(angle_deg=value)
            else:
                self.bend.angle_deg = value
                self.bend.confidence = 1.0

    def summary(self) -> str:
        t = f"{self.thickness_mm:.1f}" if self.thickness_mm is not None else "—"
        parts = [f"{self.type} {self.width_mm:.1f}x{self.height_mm:.1f}x{t} mm",
                 f"{len(self.holes)} holes", f"{len(self.slots)} slots"]
        if self.type == "bracket" and self.bend is not None:
            parts.append(f"bend {self.bend.angle_deg:.0f}° at {self.bend.position_mm:.1f}")
        if self.dims:
            parts.append(f"{len(self.dims)} dims")
        return ", ".join(parts)

    # -- dimension constraints (hole↔edge, hole↔hole, X/Y) --------------------
    EDGE_IDS: ClassVar[tuple] = ("edge_left", "edge_right",
                                 "edge_bottom", "edge_top")

    def _hole(self, hid: str) -> Optional[Hole]:
        for h in self.holes:
            if h.id == hid:
                return h
        return None

    def _bbox(self):
        xs = [p[0] for p in self.outline_mm]
        ys = [p[1] for p in self.outline_mm]
        return min(xs), min(ys), max(xs), max(ys)

    def _edge_pos(self, edge: str, axis: str) -> Optional[float]:
        """Outline-bbox side position along axis. For non-axis outline
        segments this is the bbox projection — callers label it as such."""
        if not self.outline_mm or edge not in self.EDGE_IDS:
            return None
        minx, miny, maxx, maxy = self._bbox()
        if axis == "x":
            return {"edge_left": minx, "edge_right": maxx}.get(edge)
        return {"edge_bottom": miny, "edge_top": maxy}.get(edge)

    def dim_current(self, a: str, b: str, axis: str) -> Optional[float]:
        """Current X/Y distance between references (hole centre to hole
        centre, or hole centre to bbox-projected edge)."""
        ha = self._hole(a)
        if ha is None:
            return None
        pa = ha.x if axis == "x" else ha.y
        hb = self._hole(b)
        if hb is not None:
            pb = hb.x if axis == "x" else hb.y
            return round(abs(pa - pb), 2)
        pe = self._edge_pos(b, axis)
        if pe is None:
            return None
        return round(abs(pa - pe), 2)

    def _hole_inside(self, x: float, y: float, d: float,
                     ignore_id: str = "") -> str:
        """Return "" if a hole at (x,y,d) is valid, else a reason."""
        if not self.outline_mm:
            return "no outline"
        minx, miny, maxx, maxy = self._bbox()
        if not (minx + d / 2 <= x <= maxx - d / 2 and
                miny + d / 2 <= y <= maxy - d / 2):
            return "hole would leave the outline"
        for h in self.holes:
            if h.id == ignore_id:
                continue
            if math.hypot(x - h.x, y - h.y) < (d + h.d) / 2 + 0.2:
                return f"hole would overlap {h.id}"
        for s in self.slots:
            if math.hypot(x - s.cx, y - s.cy) < d / 2 + min(s.length, s.width) / 2:
                return f"hole would overlap {s.id}"
        return ""

    def apply_dim(self, a: str, b: str, axis: str, value_mm: float) -> str:
        """Set dimension a↔b along axis to value_mm. Moves the SECOND hole
        (b) for hole↔hole, or the hole (a) for hole↔edge; edges never move.
        Returns "" on success or "Invalid: <reason>" (old value kept)."""
        ha = self._hole(a)
        if ha is None:
            return f"Invalid: hole {a} not found"
        hb = self._hole(b)
        if hb is not None:
            # hole↔hole: move b, keep a fixed
            cur = (hb.x if axis == "x" else hb.y) - \
                (ha.x if axis == "x" else ha.y)
            delta = value_mm - abs(cur)
            sgn = 1.0 if cur >= 0 else -1.0
            nx = hb.x + (delta * sgn if axis == "x" else 0.0)
            ny = hb.y + (delta * sgn if axis == "y" else 0.0)
            bad = self._hole_inside(nx, ny, hb.d, ignore_id=hb.id)
            if bad:
                return f"Invalid: {bad}"
            if axis == "x":
                hb.x = round(nx, 2)
            else:
                hb.y = round(ny, 2)
            hb.confidence = 1.0
        else:
            pe = self._edge_pos(b, axis)
            if pe is None:
                return f"Invalid: reference {b} not found"
            # hole↔edge: move a so |a - edge| == value, same side as now
            pa = ha.x if axis == "x" else ha.y
            sgn = 1.0 if pa >= pe else -1.0
            npos = pe + sgn * value_mm
            nx = npos if axis == "x" else ha.x
            ny = npos if axis == "y" else ha.y
            bad = self._hole_inside(nx, ny, ha.d, ignore_id=ha.id)
            if bad:
                return f"Invalid: {bad}"
            if axis == "x":
                ha.x = round(nx, 2)
            else:
                ha.y = round(ny, 2)
            ha.confidence = 1.0
        # store (or update) the constraint
        for dc in self.dims:
            if dc.a == a and dc.b == b and dc.axis == axis:
                dc.value_mm = round(value_mm, 2)
                dc.flagged = ""
                break
        else:
            self.dims.append(DimConstraint(a=a, b=b, axis=axis,
                                           value_mm=round(value_mm, 2)))
        return ""

    def reapply_dims(self) -> list[str]:
        """Reapply all stored constraints (rebuild / re-run survival).
        Returns ids of flagged constraints that could not be applied."""
        flagged = []
        for dc in self.dims:
            err = self.apply_dim(dc.a, dc.b, dc.axis, dc.value_mm)
            if err:
                dc.flagged = err.replace("Invalid: ", "")
                flagged.append(f"{dc.a}↔{dc.b}.{dc.axis}")
            else:
                dc.flagged = ""
        return flagged


def rectangle_spec(w: float, h: float, thickness: Optional[float] = 3.0) -> PartSpec:
    return PartSpec(
        thickness_mm=thickness,
        outline_mm=[[0, 0], [w, 0], [w, h], [0, h]],
        source="manual")

"""PartForge part specification.

The spec is the single source of truth. Photos are measured into a spec,
the user (or voice) edits the spec, and CAD is generated from the spec.
No module other than measurement creates dimensions, and measurement
never invents them: anything uncertain becomes a Question.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

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


class Bend(BaseModel):
    position_mm: float  # distance from one end along the flat pattern
    angle_deg: float = 90.0
    radius_mm: float = 2.0


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
    outline_mm: list[list[float]] = Field(default_factory=list)  # polygon, mm
    holes: list[Hole] = Field(default_factory=list)
    slots: list[Slot] = Field(default_factory=list)
    bend: Optional[Bend] = None
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
        if self.thickness_mm is None:
            qs.append(Question(
                id="thickness", field="thickness_mm",
                text="Thickness: I cannot measure this from the top photo. "
                     "What does the caliper say?",
                kind="number", suggested=3.0))
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
            qs.append(Question(
                id="bend_angle", field="bend.angle_deg",
                text="This looks like a bent bracket. What is the bend angle?",
                kind="number", suggested=90.0))
        return qs

    def apply_answer(self, question: Question, value: float) -> None:
        f = question.field
        if f == "thickness_mm":
            self.thickness_mm = value
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
            if self.bend is None:
                self.bend = Bend(position_mm=self.width_mm / 2, angle_deg=value)
            else:
                self.bend.angle_deg = value

    def summary(self) -> str:
        return (f"{self.type} {self.width_mm:.1f}x{self.height_mm:.1f} mm, "
                f"thickness {self.thickness_mm}, {len(self.holes)} holes, "
                f"{len(self.slots)} slots")


def rectangle_spec(w: float, h: float, thickness: Optional[float] = 3.0) -> PartSpec:
    return PartSpec(
        thickness_mm=thickness,
        outline_mm=[[0, 0], [w, 0], [w, h], [0, h]],
        source="manual")

"""Bidirectional 3D viewer component: renders the STL, click a marker
to select that hole/slot for editing. Falls back gracefully (returns
None / raises) if the component cannot load - the side panel still works.
"""
from __future__ import annotations

import base64
import os

import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(__file__), "viewer_component")
_component = components.declare_component("partforge_viewer", path=_DIR)


def part_viewer(stl_bytes: bytes, spec, selected: str | None,
                second: str | None = None, dims: list | None = None,
                key: str = "viewer"):
    markers = [{"id": h.id, "x": h.x, "y": h.y} for h in spec.holes]
    markers += [{"id": s.id, "x": s.cx, "y": s.cy} for s in spec.slots]
    return _component(
        stl_b64=base64.b64encode(stl_bytes).decode(),
        markers=markers,
        selected=selected or "",
        second=second or "",
        outline=[[p[0], p[1]] for p in spec.outline_mm],
        dims=dims or [],
        thickness=spec.thickness_mm or 3.0,
        key=key, default=None)

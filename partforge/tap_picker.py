"""Tap-to-select photo picker: click the part face -> pixel coords.

Same component pattern as viewer.py. Coords are in rectified-image pixels;
app.py maps them back to original photo pixels before calling measure.
"""
from __future__ import annotations

import base64
import os

import streamlit.components.v1 as components

_component = components.declare_component(
    "partforge_tap_picker",
    path=os.path.join(os.path.dirname(__file__), "tap_picker_component"))


def tap_picker(rect_png: bytes, taps: list, eraser: list,
               key: str = "tap_picker"):
    return _component(png_b64=base64.b64encode(rect_png).decode(),
                      taps=taps, eraser=eraser, key=key, default=None)

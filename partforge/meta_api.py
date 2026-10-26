"""Meta Model API wrappers (SAM 3.1 / Muse Spark / Muse Voice Transcribe).

Design rule: every function returns None when the API is not configured,
so the app always has a working fallback path (OpenCV segmentation,
rule-based questions, typed correction). The LLM never writes a measured
dimension; Muse only classifies, phrases questions, and parses edits.

Configuration (env or .env): META_API_KEY, plus per-model endpoint/model
settings once confirmed from the Meta developer console.
"""
from __future__ import annotations

import os

import numpy as np


def configured() -> dict:
    return {
        "api_key": bool(os.environ.get("META_API_KEY")),
        "sam": bool(os.environ.get("SAM_API_URL")),
        "muse": bool(os.environ.get("MUSE_API_URL")),
        "voice": bool(os.environ.get("VOICE_API_URL")),
    }


def segment_part(image_bgr: np.ndarray, click_xy: tuple[int, int] | None = None):
    """SAM 3.1: select the clicked instance on the bench -> binary mask.

    Returns None until SAM_API_URL is configured; caller falls back to
    partforge.measure.segment_fallback.
    """
    if not configured()["sam"]:
        return None
    raise NotImplementedError("SAM endpoint configured but client not wired yet")


def interpret_spec(spec, annotated_png: bytes):
    """Muse Spark: plate vs bracket + which questions are worth asking.

    Returns dict like {"type": "plate", "note": "..."} or None.
    Must never modify measured numbers in `spec`.
    """
    if not configured()["muse"]:
        return None
    raise NotImplementedError("Muse endpoint configured but client not wired yet")


def transcribe(audio_bytes: bytes) -> str | None:
    """Muse Voice Transcribe: spoken correction -> text, or None."""
    if not configured()["voice"]:
        return None
    raise NotImplementedError("Voice endpoint configured but client not wired yet")


def parse_voice_edit(text: str, spec) -> dict | None:
    """Turn 'the hole is 5, not 4' into a structured edit, or None.

    Deterministic regex fallback lives in partforge.voice_edit; Muse
    parsing replaces it once wired.
    """
    from .voice_edit import parse_edit_text
    return parse_edit_text(text, spec)

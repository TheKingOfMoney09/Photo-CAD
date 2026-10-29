"""SAM 3.1 segmentation provider (Meta Model API, Responses API).

segment_points(image, taps, eraser) -> (mask, box, confidence) or None.

Wire contract: taps/eraser are ORIGINAL photo pixels. The point prompt uses
the photo's own w/h, exactly as the API requires. The returned mask raster
is decoded and resized to the input image size.

Design: every failure returns None so callers fall back (GrabCut, then
classical threshold). The response is never trusted blindly: masks that
touch the image border, cover >80% of the frame, or come back empty/low
confidence are rejected with a reason in the log.
"""
from __future__ import annotations

import asyncio
import base64
import os

import cv2
import numpy as np

MODEL_ID = "sam-3.1"
BASE_URL = "https://api.meta.ai/v1"
MIN_CONF = 0.30
MIN_AREA_FRAC = 0.0005
MAX_AREA_FRAC = 0.80


def _client():
    from openai import OpenAI

    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY not set")
    return OpenAI(base_url=BASE_URL, api_key=key)


def _build_prompt(taps, eraser, w, h) -> str:
    from meta_sam_parser import (
        FramePrompt,
        PromptObject,
        PromptPoint,
        build_object_prompt,
    )

    pts = [PromptPoint(int(x), int(y)) for x, y in taps]
    for x, y in eraser or []:
        pts.append(PromptPoint(int(x), int(y), label="negative"))
    return build_object_prompt(
        size=(w, h),
        objects=[PromptObject(id=1, prompts=[FramePrompt(frame=0, points=pts)])],
    )


async def _segment_async(image_bgr, taps, eraser):
    from meta_sam_parser import (
        SegmentationBoxRecord,
        SegmentationMaskRecord,
        decode_mask_to_raster,
        image_segmentation_format,
        parse_responses_stream,
    )

    h, w = image_bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = _build_prompt(taps, eraser, w, h)
    client = _client()

    stream = client.responses.create(
        model=MODEL_ID,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image",
             "image_url": f"data:image/jpeg;base64,{b64}"}]}],
        stream=True,
        metadata={"include_confidence": "true"},
    )

    async def gen():
        for event in stream:
            yield event

    parsed = parse_responses_stream(gen(), format=image_segmentation_format())
    async for _ in parsed:
        pass
    res = parsed.final_result
    result = res() if callable(res) else res
    if asyncio.iscoroutine(result):
        result = await result

    masks = [r for r in result.records
             if isinstance(r, SegmentationMaskRecord)]
    boxes = [r for r in result.records
             if isinstance(r, SegmentationBoxRecord)]
    if not masks:
        return None
    m = masks[0]
    conf = m.confidence if m.confidence is not None else 0.0
    if conf < MIN_CONF:
        return None
    arr = np.frombuffer(bytes(decode_mask_to_raster(m.mask)),
                        dtype=np.uint8)
    arr = arr.reshape(m.mask.height, m.mask.width)
    full = np.zeros((h, w), dtype=np.uint8)
    if boxes:
        b = boxes[0]
        x1, y1 = max(0, int(b.left)), max(0, int(b.top))
        x2, y2 = min(w, int(b.right) + 1), min(h, int(b.bottom) + 1)
        if x2 > x1 and y2 > y1:
            full[y1:y2, x1:x2] = cv2.resize(
                arr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    else:
        full = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    frac = float((full > 0).mean())
    if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
        return None
    # border contact: instance masks shouldn't span the whole frame
    edge = np.zeros_like(full)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = 1
    if float(((full > 0) & (edge > 0)).sum()) / max(full.sum(), 1) > 0.25:
        return None
    box = None
    if boxes:
        b = boxes[0]
        box = (b.left, b.top, b.right, b.bottom)
    return (full * 255).astype(np.uint8), box, float(conf)


def segment_points(image_bgr: np.ndarray,
                   taps: list[tuple[int, int]],
                   eraser: list[tuple[int, int]] | None = None):
    """SAM 3.1 point-prompt segmentation. Returns (mask, box, conf) or None."""
    if not taps:
        return None
    try:
        return asyncio.run(_segment_async(image_bgr, taps, eraser))
    except Exception:
        return None

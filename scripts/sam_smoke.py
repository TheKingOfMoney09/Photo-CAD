"""Live SAM 3.1 smoke test: point prompt on the synthetic plate photo."""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from meta_sam_parser import (
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)

from partforge.synthetic import make_synthetic_photo

photo = make_synthetic_photo()
h, w = photo.shape[:2]
ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
b64 = base64.b64encode(buf.tobytes()).decode()

# tap the plate centre (photo pixels)
cx, cy = int(w * 0.52), int(h * 0.55)
prompt = f"<0f>1<|point;x={cx};y={cy};w={w};h={h}|>"

client = OpenAI(base_url="https://api.meta.ai/v1",
                api_key=os.environ["MODEL_API_KEY"])
stream = client.responses.create(
    model="sam-3.1",
    input=[{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{b64}"}]}],
    stream=True,
    metadata={"include_confidence": "true"},
)

import asyncio


async def _drain(events):
    parsed = parse_responses_stream(
        events, format=image_segmentation_format())
    async for _ in parsed:
        pass
    res = parsed.final_result
    out = res() if callable(res) else res
    if asyncio.iscoroutine(out):
        out = await out
    return out


async def _sdk_stream():
    async def gen():
        for event in stream:
            yield event
    return await _drain(gen())


parsed = asyncio.run(_sdk_stream())
masks = [r for r in parsed.records
         if isinstance(r, SegmentationMaskRecord)]
boxes = [r for r in parsed.records
         if isinstance(r, SegmentationBoxRecord)]
print("records:", len(masks), "masks,", len(boxes), "boxes")
for b in boxes:
    print("box:", b.object_id, round(b.left), round(b.top), round(b.right),
          round(b.bottom), "conf:",
          round(b.confidence, 4) if b.confidence is not None else None)
for m in masks:
    raster = decode_mask_to_raster(m.mask)
    import numpy as _np
    arr = _np.frombuffer(bytes(raster), dtype=_np.uint8)
    print("mask raster:", m.mask.height, "x", m.mask.width,
          "ones:", int(arr.sum()), "of", len(arr))
    cv2.imwrite("tmp_sam_mask.png",
                (arr.reshape(m.mask.height, m.mask.width) * 255)
                .astype("uint8"))
    print("wrote tmp_sam_mask.png")

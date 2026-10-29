"""Meta SAM 3.1 engine — text-prompted segmentation (Meta Model API).

One function for the rest of the app:

    segment(image, prompt) -> list[(box, mask)]

- image: BGR photo (any size). prompt: one noun phrase ("part"/"hole"/"slot").
- One POST /v1/responses per call, non-streaming, mask_encoding one_bit.
- Returns [(box_xyxy, mask_bool_fullframe)] in INPUT-IMAGE pixels.
  Empty list = zero matches (valid, not an error).
- Transport/API/parse problems raise RuntimeError("SAM unavailable: ...").
  Callers show that string in the UI. NO fallback to edge detection.
- Mock runs only behind SAM_MOCK=1 (offline dev). Status bar must show it.

Mask wire format (docs/sams/reading-segmentation): the mask token carries
a box-cropped raster: data=H,W,enc+payload where H,W are the BOX size, and
the raster must be placed at the record's box. decode_mask_to_raster gives
row-major 0/1 of length H*W; paste at (x1,y1)-(x2,y2).

Box coords: x2/y2 INCLUSIVE (docs). Paste region is [x1:x2+1, y1:y2+1].
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"


def _client():
    from openai import OpenAI

    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SAM unavailable: MODEL_API_KEY not set")
    return OpenAI(base_url=BASE_URL, api_key=key)


def _jpg_b64(img_bgr: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("SAM unavailable: could not JPEG-encode photo")
    return base64.b64encode(buf.tobytes()).decode()


def _parse(text: str):
    """Wire text -> list of (box_xyxy_inclusive, raster01 HW, H, W)."""
    import re

    recs = []
    for m in re.finditer(
            r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>"
            r"<\|mask;x=0;y=0;data=(\d+),(\d+),([!~])(.*?)\|>", text):
        x1, y1, x2, y2, _w, _h, mh, mw, enc, payload = m.groups()
        recs.append({"box": (int(x1), int(y1), int(x2), int(y2)),
                     "mh": int(mh), "mw": int(mw), "enc": enc,
                     "payload": payload})
    return recs


def _decode_one_bit(payload: str, h: int, w: int) -> np.ndarray:
    """one_bit payload (marker already stripped) -> bool raster HxW.

    Decoded with meta_sam_parser's own codec (custom base85 + range
    coder) — NOT stdlib base85. Raises RuntimeError on failure.
    """
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    try:
        raw = bytes(decode_mask_to_raster(
            SegmentationMask(encoding="one_bit", payload="!" + payload,
                             width=w, height=h)))
    except Exception as e:
        raise RuntimeError(f"SAM unavailable: mask decode failed: {e}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
    return arr.astype(bool)


def _decode_lossless(payload: str, h: int, w: int) -> np.ndarray:
    from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster

    rec = SegmentationMaskRecord(mask=payload, height=h, width=w)
    try:
        raw = bytes(decode_mask_to_raster(rec))
    except TypeError:
        try:
            from meta_sam_parser import SegmentationMask

            raw = bytes(decode_mask_to_raster(
                SegmentationMask(data=payload, height=h, width=w)))
        except Exception as e:
            raise RuntimeError(f"SAM unavailable: mask decode failed: {e}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
    return arr.astype(bool)


def segment(image: np.ndarray, prompt: str) -> list:
    """segment(image, prompt) -> [(box_xyxy, mask_bool_fullframe)]."""
    from openai import BadRequestError

    if os.environ.get("SAM_MOCK") == "1":
        return _mock_segment(image, prompt)
    client = _client()
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image",
                 "image_url": f"data:image/jpeg;base64,{_jpg_b64(image)}"}]}],
            metadata={"mask_encoding": "one_bit"},
        )
    except BadRequestError as e:
        raise RuntimeError(f"SAM unavailable: bad request: {e}")
    except Exception as e:
        raise RuntimeError(f"SAM unavailable: {type(e).__name__}: {e}")
    text = resp.output_text or ""
    if not text.strip():
        return []
    out = []
    ih, iw = image.shape[:2]
    for rec in _parse(text):
        x1, y1, x2, y2 = rec["box"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(iw - 1, x2), min(ih - 1, y2)
        if x2c < x1c or y2c < y1c:
            continue
        bw, bh = x2c - x1c + 1, y2c - y1c + 1
        try:
            if rec["enc"] == "!":
                raster = _decode_one_bit(rec["payload"], rec["mh"], rec["mw"])
            else:
                raster = _decode_lossless(rec["payload"], rec["mh"], rec["mw"])
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"SAM unavailable: mask decode failed: {e}")
        r = cv2.resize(raster.astype(np.uint8), (bw, bh),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((ih, iw), dtype=bool)
        full[y1c:y2c + 1, x1c:x2c + 1] = r[
            :(y2c + 1 - y1c), :(x2c + 1 - x1c)]
        out.append(((x1, y1, x2, y2), full))
    return out


def _mock_segment(image: np.ndarray, prompt: str) -> list:
    """Offline dev stand-in behind SAM_MOCK=1. Same I/O, NOT trustworthy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(t)
    dark = ((gray < max(60.0, t - 12.0))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    ih, iw = gray.shape
    if prompt == "part":
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best, best_a = None, 0.0
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + cw >= iw - 2 or y + ch >= ih - 2:
                continue
            a = cv2.contourArea(c)
            if a > best_a:
                best, best_a = c, a
        if best is None:
            return []
        x, y, w, h = cv2.boundingRect(best)
        full = np.zeros((ih, iw), dtype=bool)
        cv2.drawContours(full.view(np.uint8), [best], -1, 1, -1)
        return [((x, y, x + w - 1, y + h - 1), full)]
    # hole/slot: enclosed bright blobs inside the dark part region
    bright = ((gray >= max(60.0, t - 12.0))).astype(np.uint8) * 255
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    out = []
    for i in range(1, nlab):
        m = (lab == i)
        if not m.any() or m[0, :].any() or m[-1, :].any() \
                or m[:, 0].any() or m[:, -1].any():
            continue  # touches border = background, not a hole
        area = float(stats[i, cv2.CC_STAT_AREA])
        d_eq = 2 * (area / 3.14159) ** 0.5
        ys, xs = np.nonzero(m)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        ww, hh = x2 - x1 + 1, y2 - y1 + 1
        elong = max(ww, hh) / max(min(ww, hh), 1)
        if prompt == "hole" and d_eq >= 8 and elong < 1.6:
            out.append(((x1, y1, x2, y2), m))
        elif prompt == "slot" and elong >= 1.6 and min(ww, hh) >= 8:
            out.append(((x1, y1, x2, y2), m))
    return out

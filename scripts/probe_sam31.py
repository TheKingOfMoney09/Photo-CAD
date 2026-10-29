"""Milestone 1 probe: one real SAM 3.1 call on a sample photo.

Run: .venv\\Scripts\\python.exe scripts\\probe_sam31.py
Needs MODEL_API_KEY in .env. Exits nonzero with the exact error on failure.
"""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from partforge.synthetic import make_synthetic_photo

img = make_synthetic_photo()
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
b64 = base64.b64encode(buf.tobytes()).decode()

key = os.environ.get("MODEL_API_KEY", "")
if not key:
    print("MISSING: MODEL_API_KEY not set")
    raise SystemExit(2)

client = OpenAI(base_url="https://api.meta.ai/v1", api_key=key)
resp = client.responses.create(
    model="sam-3.1",
    input=[{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "metal bracket plate"},
        {"type": "input_image",
         "image_url": f"data:image/jpeg;base64,{b64}"}]}],
    metadata={"mask_encoding": "one_bit"},
)
text = resp.output_text
print("output_text len:", len(text))
print("head:", text[:300])
open("debug_probe_raw.txt", "w", encoding="utf-8").write(text)
print("full response saved to debug_probe_raw.txt")

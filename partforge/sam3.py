"""SAM 3 vision engine — text-prompted segmentation behind one interface.

Real backend: Roboflow Serverless Hosted API, /sam3/concept_segment,
polygon output. Key comes from the API_KEY env variable.
Mock backend: classical OpenCV with the SAME interface (offline dev only).

segment(img, prompts) -> {prompt: [polygon, ...]}, polygons as Nx2 float
arrays in INPUT-IMAGE pixels. Any prompt may return [] (no instances —
normal result, not an error). All transport/parse failures raise
RuntimeError with the cause; callers decide fallback policy (task: no
fallback to old edge detection — a failed SAM call is a failed view).

get_client() -> RoboflowSam3 when API_KEY is set, else MockSam3.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

ENDPOINT = "https://serverless.roboflow.com/sam3/concept_segment"
PROMPTS = ["part", "hole", "slot"]


def _jpg_b64(img_bgr: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not JPEG-encode image")
    return base64.b64encode(buf.tobytes()).decode()


class RoboflowSam3:
    """SAM 3 via Roboflow Serverless. One HTTP call per photo."""

    def __init__(self, api_key: str, timeout_s: float = 180.0,
                 prob_thresh: float = 0.5):
        import requests

        self._requests = requests
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.prob_thresh = prob_thresh

    @property
    def name(self) -> str:
        return "live"

    def segment(self, img_bgr: np.ndarray,
                prompts: list[str]) -> dict[str, list[np.ndarray]]:
        payload = {
            "image": {"type": "base64", "value": _jpg_b64(img_bgr)},
            "prompts": [{"type": "text", "text": p} for p in prompts],
            "output_prob_thresh": self.prob_thresh,
            "format": "polygon",
        }
        try:
            r = self._requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"SAM3 API call failed: {e}") from e
        try:
            results = data["prompt_results"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"SAM3 bad envelope: {str(data)[:200]}") from e
        out: dict[str, list[np.ndarray]] = {}
        for prompt, res in zip(prompts, results):
            polys: list[np.ndarray] = []
            for pred in res.get("predictions", []):
                for poly in pred.get("masks", []):
                    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                    if len(arr) >= 3:
                        polys.append(arr)
            out[prompt] = polys
        return out


class MockSam3:
    """Offline stand-in: classical CV behind the SAM3 interface.

    'part' -> largest interior blob; 'hole'/'slot' -> enclosed blobs
    classified by circularity/elongation. Same I/O as RoboflowSam3.
    Dev-only: results are NOT trustworthy measurements.
    """

    @property
    def name(self) -> str:
        return "mock"

    def segment(self, img_bgr: np.ndarray,
                prompts: list[str]) -> dict[str, list[np.ndarray]]:
        h, w = img_bgr.shape[:2]
        out: dict[str, list[np.ndarray]] = {p: [] for p in prompts}
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, t = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = ((gray < max(60.0, t - 12.0))).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0.0
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2:
                continue
            a = cv2.contourArea(c)
            if a > best_area:
                best, best_area = c, a
        if best is None or best_area < 50:
            return out
        if "part" in prompts:
            out["part"] = [best.reshape(-1, 2).astype(np.float64)]
        if "hole" in prompts or "slot" in prompts:
            inv = cv2.bitwise_not(
                cv2.compare(gray, int(max(60.0, t - 12.0)), cv2.CMP_LT))
            part_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(part_mask, [best], -1, 255, -1)
            inside = cv2.bitwise_and(inv, inv, mask=part_mask)
            nlab, lab, _, _ = cv2.connectedComponentsWithStats(inside)
            for i in range(1, nlab):
                m = (lab == i).astype(np.uint8) * 255
                ccs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                if not ccs:
                    continue
                c = max(ccs, key=cv2.contourArea)
                per = cv2.arcLength(c, True)
                if per <= 0:
                    continue
                circ = 4 * np.pi * cv2.contourArea(c) / (per * per)
                rw, rh = cv2.minAreaRect(c)[1]
                long_, short_ = max(rw, rh), min(rw, rh)
                poly = c.reshape(-1, 2).astype(np.float64)
                if circ >= 0.6 and short_ >= 3.0:
                    if "hole" in prompts:
                        out["hole"].append(poly)
                elif short_ >= 3.0 and long_ > 1.55 * short_:
                    if "slot" in prompts:
                        out["slot"].append(poly)
        return out


def get_client():
    """RoboflowSam3 when API_KEY is set, else MockSam3. Never raises."""
    key = os.environ.get("API_KEY", "").strip()
    if key:
        try:
            return RoboflowSam3(key)
        except Exception:
            pass
    return MockSam3()

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

    Works in RECTIFIED card space would be ideal, but the mock only sees
    the raw photo — so it restricts itself to the card quad: detect the
    ArUco markers, warp to card space, run the old threshold path there,
    warp polygons back to photo pixels. Same I/O as RoboflowSam3.
    Dev-only: results are NOT trustworthy measurements.
    """

    @property
    def name(self) -> str:
        return "mock"

    def segment(self, img_bgr: np.ndarray,
                prompts: list[str]) -> dict[str, list[np.ndarray]]:
        from .card import CARD_H_MM, CARD_W_MM, detect_card
        from .measure import _blank_card_graphics, segment_fallback

        h, w = img_bgr.shape[:2]
        out: dict[str, list[np.ndarray]] = {p: [] for p in prompts}
        H, _ids = detect_card(img_bgr)
        if H is None:
            return out
        ppm = 8.0
        rect = cv2.warpPerspective(
            img_bgr, np.diag([ppm, ppm, 1.0]) @ H,
            (int(CARD_W_MM * ppm), int(CARD_H_MM * ppm)))
        mask, _t, _kept = segment_fallback(rect, thresh=0)
        # part polygon: largest root contour in rect space -> photo pixels
        cnts, hier = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or hier is None:
            return out
        hier = hier[0]
        roots = [i for i, hh in enumerate(hier) if hh[3] == -1]
        if not roots:
            return out
        outer = max(roots, key=lambda i: cv2.contourArea(cnts[i]))
        if cv2.contourArea(cnts[outer]) < 50:
            return out
        inv = np.linalg.inv(np.diag([ppm, ppm, 1.0]) @ H)

        def back(pts):
            p = np.hstack([pts.reshape(-1, 2),
                           np.ones((len(pts.reshape(-1, 2)), 1))])
            q = (inv @ p.T).T
            return (q[:, :2] / q[:, 2:3]).reshape(-1, 2)

        if "part" in prompts:
            out["part"] = [back(cnts[outer].reshape(-1, 2))]
        if "hole" in prompts or "slot" in prompts:
            child = hier[outer][2]
            while child != -1:
                c = cnts[child]
                per = cv2.arcLength(c, True)
                area = cv2.contourArea(c)
                if per > 0 and area / (ppm * ppm) > 2.0:
                    circ = 4 * np.pi * area / (per * per)
                    rw, rh = cv2.minAreaRect(c)[1]
                    long_, short_ = max(rw, rh) / ppm, min(rw, rh) / ppm
                    poly = back(c.reshape(-1, 2))
                    if circ >= 0.6 and short_ >= 1.6:
                        if "hole" in prompts:
                            out["hole"].append(poly)
                    elif short_ >= 1.6 and long_ > 1.55 * short_:
                        if "slot" in prompts:
                            out["slot"].append(poly)
                child = hier[child][0]
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

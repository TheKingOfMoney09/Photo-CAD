# PartForge — engine decisions (SAM3 via Roboflow API, branch `sam3-api`)

## D1 — No local weights, no GPU server: mock with the same interface
Checked 2026-10-30: no torch in venv, no `sam*` weights on disk, no SAM env
vars, nothing SAM-like listening (port 8000 answers "Not Found" on
/v1/models). Task says: build against a mock, report what's missing.
**Missing: `API_KEY` env var (Roboflow key).** Without it every SAM call
401s (verified with a live probe: `{"status":401,"message":"Unauthorized
api_key"}`). `get_client()` returns `RoboflowSam3` when `API_KEY` is set,
else `MockSam3`. One env var flips backends, zero code changes.

## D2 — Roboflow wire contract (from current docs, verified endpoint shape)
`POST https://serverless.roboflow.com/sam3/concept_segment`
headers `Authorization: Bearer $API_KEY`, body:
`{"image": {"type": "base64", "value": "<b64 jpg>"},
   "prompts": [{"type": "text", "text": "metal bracket plate"}, ...],
   "output_prob_thresh": 0.5, "format": "polygon"}`
Response: `{"prompt_results": [{"echo": {...}, "predictions":
[{"confidence": f, "format": "polygon",
   "masks": [[[x,y], ...], ...]}]}]}`. Polygon points are input-image
pixels. Any prompt may return zero predictions (normal, not an error).
Doc gaps: RLE payload shape is under-documented — polygon is primary,
RLE decode left unimplemented until a real response needs it.

## D3 — One SAM call per photo, all prompts at once
`segment(photo, ["part", "hole", "slot"])` sends the image once with all
three text prompts and returns `{prompt: [polygon, ...]}` in image pixels.
Per-photo result cached in `ss.sam_cache[sha1(jpg)]` so measurement edits
rebuild CAD only and never re-call the API.

## D4 — Mock = old engine repackaged, labeled, never trusted
`MockSam3.segment()` = Otsu threshold + contour children, same I/O as the
real client (dict of prompt → polygon lists). For offline dev only.
Status bar shows `SAM3: live` vs `SAM3: mock`.

## D5 — Polygons → fitted shapes, never raw points, downstream
`shapes.py`: RDP → corners; Hough → lines (snap 2° → 0/90, parallel 1.5°,
equal length 0.3 mm); minEnclosingCircle + circularity → holes;
minAreaRect → slots. `PartSpec` only sees fitted values.

## D6 — Front view keeps its role (thickness vote), new engine
Front "part" polygon → fitted rectangle → long side matches top W/H,
short side = thickness vote. `fusion.py` untouched (works on PartSpecs).

## D7 — Viewer clicks select detected shapes; edits rebuild CAD only
Markers already carry shape ids (H1, S1, bend). SAM runs once per photo
at Measure time (cached). All edits mutate the spec + `build_solid`.

## D8 — Debug images on disk for every view
`debug/<role>_sam_<prompt>.png` (polygons over photo) +
`debug/<role>_fitted.png` (fitted shapes over rectified photo).
Gitignored, never uploaded.

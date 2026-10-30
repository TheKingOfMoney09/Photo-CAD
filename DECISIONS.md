# PartForge — engine decisions (Meta SAM 3.1, branch `sam3-api`)

## D9 — Bend detection is mask-based and automatic (no checkbox)
`partforge/bends.py`: `detect_bends({view: (rect_mask, ppm)})` returns
`{has_bend, bend_confidence, bend_line_mm, source_views, features}`.
Features per view mask: bbox fill, solidity (area/convex-hull),
largest empty bbox-corner fraction, distinct thickness runs along the
main axis. `has_bend` needs >= 2 of (fill < 0.75, solidity < 0.75,
empty_frac > 0.30). Side/front views vote (majority); top is weak
secondary; top-only (no side/front) caps confidence at 0.5.
bend_line_mm = empty region's inner corner projected on the main axis,
in mm from the bbox min. No CAD built (detection only).
Thresholds tuned on: synthetic L (fill 0.216/solidity 0.358/empty 0.774)
vs synthetic bar (0.952/1.0/0.0); real bracket side (0.152/0.287/0.822)
vs real bracket front/top faces (~0.79/0.90/0.13). Flat plates read
fill 0.95+, solidity ~1.0, empty ~0.0 — wide margins, no knife-edges.

## D1 — No local weights, no GPU server: Meta Model API instead
Checked 2026-10-30: no torch in venv, no `sam*` weights on disk, nothing
SAM-like listening. Task: Meta `sam-3.1` on Meta Model API, no self-host.
`MODEL_API_KEY` present in `.env` → live backend works.

## D2 — Verified wire facts (live probe, do not re-guess)
- Base URL `https://api.meta.ai/v1`, `POST /v1/responses`, OpenAI SDK.
- One user message: `input_text` noun phrase + `input_image` as a
  **base64 data URL — works** (verified, no public URL needed).
- `metadata: {"mask_encoding": "one_bit"}` → mask token `data=H,W,!payload`.
- Non-streaming is fine for single images. Empty `output_text` = zero
  matches (valid, e.g. prompt "part" on a side view).
- Mask raster is **box-cropped**: H,W = the record's box size; paste the
  decoded raster at (x1,y1)-(x2,y2), x2/y2 INCLUSIVE.
- one_bit payload is NOT stdlib base85 (has `'`, `/`, `[` etc). Decoded
  with `meta_sam_parser`'s own codec (`decode_mask_to_raster`).
  Hand-rolled replay events fail lane checks — parse the real SDK stream.
- `meta-sam-parser` 0.1.0 has NO `parse_image_stream`/`records_of_kind`:
  use `parse_responses_stream(events, format=image_segmentation_format())`,
  drain snapshots, then `final_result` (call it if callable, await if
  coroutine). Records: `SegmentationBoxRecord`/`SegmentationMaskRecord`.

## D3 — One request per prompt, never combined
`segment(image, prompt) -> [(box_xyxy, mask_bool_fullframe)]` in
`partforge/sam_engine.py`. Prompts: part ladder
("metal bracket plate" → "flat metal part on white card" →
"rectangular bar" → "plate" → "part"), then "hole", then "slot"
(top view only). First non-empty part result wins. Thin side bars need
shape words ("rectangular bar"); bare "part" returns nothing on them.
Mock runs ONLY behind `SAM_MOCK=1`; status bar shows live/mock/missing.

## D4 — Shapes, never raw contours, go downstream
`shapes.py` fits outline (RDP corners + angle/length snap), circles
(holes), minAreaRect (slots). `PartSpec` only sees fitted values.

## D5 — SAM hole masks are approximate: fit, then verify by position
Live "hole" masks land within 1.5 mm of GT centres on synthetic but
diameters read ~1.5 high (mask bleed). Regression checks position only;
diameter accuracy is a user-confirmed question in the app (unchanged).

## D6 — Front view keeps its role (thickness vote), new engine
Front part mask → fitted rectangle → long side matches top W/H,
short side = thickness vote. `fusion.py` untouched (works on PartSpecs).

## D7 — Viewer clicks select detected shapes; edits rebuild CAD only
Markers already carry shape ids (H1, S1, bend). SAM runs once per photo
at Measure time (sha1 cache). All edits mutate the spec + `build_solid`.

## D8 — Debug images on disk for every view
`debug/<role>_sam_<prompt>.png` (SAM masks over photo) +
`debug/<role>_fitted.png` (fitted outline over rectified photo).
Gitignored, never uploaded.

# PartForge — SAM31 engine (branch `sam3-api`, 2026-10-30)

## What changed
Vision engine is Meta SAM 3.1 (`sam-3.1`) on Meta Model API. New code:
- `partforge/sam_engine.py` — `segment(image, prompt) -> [(box, mask)]`,
  one request per prompt, one_bit masks, box-placed decode, mock only
  behind `SAM_MOCK=1`. Failures raise "SAM unavailable: reason".
- `partforge/pipeline.py` — `measure_view(img, role)`: ArUco → SAM part
  ladder → hole/slot prompts (top only) → `shapes.py` fit → PartSpec.
- `partforge/shapes.py` — polygons → fitted outline/circles/rects/slots.
- `app.py` — per-photo sha1 SAM cache (edits never re-call), no taps, no
  thresholds, status bar shows `SAM:live/mock/missing`.

Deleted: Roboflow `sam3.py`, old `measure.py` (Otsu/GrabCut/taps),
SAM 3.1 point provider, tap picker, probe scripts.

Untouched: ArUco (`card.py`), CAD (`cad.py`), fusion, spec/questions,
viewer (clicks select detected shape ids), UI shell.

## Verification (live backend, MODEL_API_KEY in .env)
- Synthetic top: 80.0×39.8 plate, 4/4 hole centres <1.5 mm (diameters
  read ~1.5 high — mask bleed, user-confirmed in app as before).
- Side bars need shape-word prompts ("rectangular bar"); bare "part"
  returns nothing on them. Prompt ladder handles this automatically.
- AppTest Plate demo → Review + Export tabs, zero exceptions.
- Live server :8508 replies `ok`.

## Known gaps
- Slot prompt returns nothing on synthetic (pre-existing, both engines).
- Bracket-side bend proposal FAILs on synthetic (pre-existing).
- Live SAM costs per photo (3 requests top, 1-4 side/front); cache
  prevents repeat charges on edits. Thin-view prompt ladder may need
  one more wording round on real photos.

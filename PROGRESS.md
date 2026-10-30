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

## Bend test set (2026-10-30, detector `partforge/bends.py`)

| photo | has_bend | conf | bend_line_mm | features (fill/solid/empty) |
|---|---|---|---|---|
| side (2).jpg (L profile) | true | 0.97 | 3.2 | 0.152/0.287/0.822 |
| front (2).jpg (face) | false | 0.70 | — | 0.793/0.903/0.124 |
| top (2).jpg (face) | false | 0.70 | — | 0.794/0.898/0.130 |
| fused (all three) | true | 0.97 | 3.2 | src: [side] |
| synthetic flat plate (control) | false | 0.50 | — | 0.986/0.992/0.002, top-only cap |
| synthetic L mask (unit) | true | 0.97 | 3.0 | 0.216/0.358/0.774 |
| synthetic bar mask (unit) | false | 0.70 | — | 0.952/1.0/0.0 |

Flat control: no real flat-bracket photo on hand — control is the
synthetic flat plate top view (fill 0.986). A real flat-plate photo on
the card is still wanted as a second control.
`scripts/test_bends.py`: 5/5 PASS. `scripts/demo_synthetic.py` unchanged
(8/10, same 2 pre-existing FAILs: slot prompt, bracket-side bend — the
old outline-kink path in fusion, separate from this detector).

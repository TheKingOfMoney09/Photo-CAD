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

## Hole↔edge / hole↔hole dimensions — bracket `top (2).jpg` (2026-11-01)
Measured plate 54.4×48.9 mm, 7 holes (H4 d=9.5, rest d≈5.4).

Hole→edge (bbox-projected; outline is near-rectangular so ≈ true edge):
| hole | X←left | X→right | Y←bottom | Y→top |
|---|---|---|---|---|
| H1 (14.3, 9.3) | 14.3 | 40.1 | 9.3 | 39.6 |
| H2 (49.1, 9.4) | 49.1 | 5.3 | 9.4 | 39.5 |
| H3 (29.0, 13.1) | 29.0 | 25.4 | 13.1 | 35.9 |
| H4 (47.4, 22.1) | 47.4 | 7.0 | 22.1 | 26.8 |
| H5 (14.2, 34.5) | 14.2 | 40.2 | 34.5 | 14.4 |
| H6 (29.0, 31.0) | 28.9 | 25.4 | 31.0 | 17.9 |
| H7 (48.9, 34.6) | 48.9 | 5.5 | 34.6 | 14.3 |

Hole↔hole (dX / dY): H1-H2 34.8/0.1 · H1-H5 0.2/25.2 · H5-H7 34.8/0.1 ·
H2-H7 0.2/25.2 · H3-H6 0.1/17.9 (rows/cols read square: dY≈0.1, dX≈34.8).

# PartForge — SAM3 engine swap (branch `sam3-api`, 2026-10-30)

## What changed
Replaced the whole vision engine (Otsu threshold + GrabCut + manual taps +
SAM 3.1 point API) with SAM 3 text prompts via Roboflow
`/sam3/concept_segment` (polygon output). New modules only:
- `partforge/sam3.py` — one interface, two backends: `RoboflowSam3`
  (live, needs `API_KEY`) / `MockSam3` (offline classical CV, same I/O).
- `partforge/shapes.py` — polygons → fitted lines/arcs/circles/rects/slots
  in mm, with snap rules. Raw contour points never go downstream.
- `partforge/pipeline.py` — `measure_view(img, role)`: ArUco → SAM call →
  fit → PartSpec. One SAM call per photo, cached by sha1 (edits never
  re-call). Debug images in `debug/` (gitignored).
- `app.py` — Capture without thresholds/taps; Review/Export unchanged;
  viewer clicks select detected shape ids; status bar shows SAM3 backend.

Deleted: `tap_picker*`, `sam_provider.py` (SAM 3.1), `sam_bracket/smoke`
scripts, all threshold/tap UI and code paths in `app.py`/`measure.py`.

Untouched: ArUco (`card.py`), CAD (`cad.py`), fusion (`fusion.py`),
spec/questions, viewer component, voice edit, UI shell.

## Verification (mock backend — no API_KEY in this env)
- `scripts/demo_synthetic.py`: 8/10 PASS — identical to the pre-swap
  baseline (top 79.2×39.2, 4 holes, side/front votes 3.5→3.5 vs GT 3.0).
  The 2 FAILs are pre-existing (slot + bracket-side, broken before swap).
- Streamlit AppTest: Plate demo → Review + Export tabs, STL/STEP
  downloads, zero exceptions.
- Live server on :8506 replies `ok`.

## What's missing (needs David)
1. **`API_KEY` env var (Roboflow key)** — without it everything runs on
   `MockSam3` (classical CV, dev-only). Live probe confirmed the endpoint
   shape: missing key → `401 {"Unauthorized api_key"}`.
2. **Confirm the wire contract** (`DECISIONS.md` D2): polygon point layout
   is input-image pixels per docs, but unverified against a real response.
   First live call should dump `prompt_results` and check parsing.
3. **Slot + bracket-side FAILs** predate the swap; the live backend may fix
   the slot (real "slot" concept vs mock contour rules), but the bracket
   side fixture needs review regardless.
4. `debug/` images are written locally only — check them after the first
   live run.

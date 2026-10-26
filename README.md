# PartForge

**Two photos of a broken part → measured, editable, printable CAD.**

Built for the [Meta Global AI Developer Hackathon 2026](https://dev.meta.ai/events/global-hackathon)
(Oct 26 – Nov 4, 2026).

| | |
|---|---|
| Track | Indie developer (individual / student) |
| Theme | Push the Frontier — manufacturing & physical systems |
| Status | Day 1 working slice — photo → measured plate → live-edited CAD → STL/STEP |

## The idea

Reverse-engineering a replacement part usually starts with calipers and a lot of
guesswork. Photo-CAD aims to take one or more photos of a physical part and return a
parametric CAD model you can actually edit — dimensions, features, and sketch
geometry — rather than a dead triangle mesh.

Pipeline (measured, never hallucinated):

1. **Capture** — top (+ optional side) photo of the part on the printed calibration card.
2. **Segment** — click-select the broken instance with SAM 3.1 (OpenCV card fallback runs now).
3. **Measure** — deterministic OpenCV: outline, hole centres/diameters, slots, in millimetres.
4. **Ask** — PartForge asks only the ambiguous dimensions (thickness, uncertain holes), via
   Muse Spark reasoning; typed/voice correction loop ("hole 2 is 5, not 4").
5. **Build** — spec → parametric build123d model, regenerated live on every edit.
6. **Verify** — CAD overlay on the rectified photo + caliper error table (coming).
7. **Print** — export STL (print) or STEP (edit in Fusion/Onshape).

Scope: flat plates + single-bend brackets with holes/slots. Thickness is confirmed
with calipers, never guessed. No gears, threads, or freeform shapes.

## Tech

- **Meta Model APIs** — SAM 3.1 (instance selection), Muse Spark (interpretation,
  the one ambiguous question), Muse Voice Transcribe (hands-dirty correction loop).
  Thin wrappers in `partforge/meta_api.py`; app runs fully offline without keys.
- **Geometry** — OpenCV + ArUco calibration, build123d CAD, Streamlit + Three.js viewer
- **Output** — `PartSpec` JSON (single source of truth), STL, STEP

## Getting started

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

Then: hit **▶ Demo part** in the sidebar, or print the card
(sidebar → ⬇ Print card, at exactly 210×148 mm), lay the part on it, and upload a photo.

## Repo layout

```
.
├── app.py                  # Streamlit UI: upload -> 3D viewer -> edit -> export
├── partforge/
│   ├── spec.py             # PartSpec: outline/holes/slots/bend + questions
│   ├── card.py             # printable calibration card + ArUco rectification
│   ├── measure.py          # deterministic mm measurement from a photo
│   ├── cad.py              # spec -> build123d solid -> STL/STEP, dimension diagram
│   ├── viewer.py           # clickable Three.js viewer (hole markers select dims)
│   ├── meta_api.py         # Meta API wrappers (SAM/Muse/Voice, fallback-safe)
│   └── voice_edit.py       # 'hole 2 is 5, not 4' -> structured edit
├── scripts/
│   └── demo_synthetic.py   # regression test on a known-truth fake photo
└── requirements.txt
```

## Hackathon

- Applications closed Oct 25, 2026 · Kickoff Oct 26 · Submissions due Nov 4, 9:00 PM ET
- Judged on creativity, technical ambition, and whether it works
- Official Rules are published before the hackathon begins and take precedence over
  anything written here

## License

MIT — see [LICENSE](LICENSE).

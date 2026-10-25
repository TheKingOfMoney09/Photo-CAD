# Photo-CAD

**Turn photos of real-world parts into editable CAD models.**

Built for the [Meta Global AI Developer Hackathon 2026](https://dev.meta.ai/events/global-hackathon)
(Oct 26 – Nov 4, 2026).

| | |
|---|---|
| Track | Indie developer (individual / student) |
| Theme | Push the Frontier — manufacturing & physical systems |
| Status | Scaffold — implementation starts at kickoff, Oct 26 2026 |

## The idea

Reverse-engineering a replacement part usually starts with calipers and a lot of
guesswork. Photo-CAD aims to take one or more photos of a physical part and return a
parametric CAD model you can actually edit — dimensions, features, and sketch
geometry — rather than a dead triangle mesh.

Planned pipeline:

1. **Capture** — one or more photos of the part next to a known reference for scale.
2. **Segment** — isolate the part and its features with SAM 3.1.
3. **Understand** — extract dimensions, primitives, and topology with Meta's vision
   and reasoning models (Muse Image / Muse Spark).
4. **Build** — emit parametric CAD (STEP / sketch primitives) instead of a mesh.
5. **Verify** — render the result back and compare it to the original photo.

## Tech

- **Meta Model API** — SAM 3.1 (segmentation), Muse Image / Muse Spark (vision + reasoning)
- **Muse Code** — agentic coding during the build
- Output targets: STEP, STL, and parametric sketch data

> Exact stack is finalized during the hackathon. This section will be updated as the
> implementation lands.

## Getting started

Nothing to run yet. Setup instructions will be added with the first working slice.

## Repo layout

```
.
├── README.md
├── LICENSE
└── .gitignore
```

## Hackathon

- Applications closed Oct 25, 2026 · Kickoff Oct 26 · Submissions due Nov 4, 9:00 PM ET
- Judged on creativity, technical ambition, and whether it works
- Official Rules are published before the hackathon begins and take precedence over
  anything written here

## License

MIT — see [LICENSE](LICENSE).

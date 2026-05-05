# GR Tracer

A physically accurate general relativistic ray tracer for Schwarzschild and Kerr black holes.

---

## Physics Background

### Gravitational Lensing

General relativity predicts that mass curves spacetime, and light follows the straightest possible paths (geodesics) through that curved geometry. Near a black hole, this bending becomes extreme: a distant star directly behind a black hole appears as a bright ring (the **Einstein ring**), and multiple images of the same source can form around it. At the **photon sphere** (r = 3M for a Schwarzschild black hole), photons can orbit in unstable circular paths — light from behind the black hole can loop around and reach the observer.

### Ray Tracing in GR

Classical ray tracers propagate straight rays through flat Euclidean space. In GR, we instead integrate **geodesic equations** in curved spacetime:

```
d²xμ/dλ² + Γμαβ (dxα/dλ)(dxβ/dλ) = 0
```

where Γμαβ are Christoffel symbols encoding the curvature of spacetime induced by the black hole's mass (and spin, for Kerr). We trace rays *backwards* from the camera — each pixel corresponds to a null geodesic integrated through the metric until it hits an object (accretion disk, background sky) or falls past the event horizon.

---

## Features

- [ ] **Schwarzschild metric** — non-rotating black hole in Boyer-Lindquist coordinates
- [ ] **Kerr metric** — rotating black hole with frame dragging
- [ ] **Accretion disk** — geometrically thin, optically thick disk with Novikov-Thorne temperature profile
- [ ] **Gravitational lensing** — full multi-order image formation, Einstein rings
- [ ] **Doppler & redshift effects** — relativistic beaming and color shift from disk orbital velocity
- [ ] **Background sky texture** — map arbitrary HDR/equirectangular images to the celestial sphere
- [ ] **Adaptive integration** — RK45 with step-size control for accuracy near the horizon
- [ ] **Multi-resolution rendering** — fast preview mode + high-quality final render

---

## Roadmap

| Phase | Goal |
|-------|------|
| **1 — Flat baseline** | Pin-hole camera, straight ray casting, background sky texture |
| **2 — Schwarzschild** | Geodesic integrator, photon sphere, event horizon absorption, shadow |
| **3 — Visual effects** | Accretion disk geometry, temperature coloring, gravitational redshift, Doppler boosting |
| **4 — Optimization** | Vectorized batch integration, adaptive step control, optional GPU path |
| **5 — Kerr** | Kerr metric, Carter constant, ergosphere, ISCO shifts, frame dragging |

---

## Installation

> **Prerequisites:** Python 3.10+

```bash
git clone https://github.com/AminaZa/black-hole-raytracer.git
cd black-hole-raytracer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Usage

> Usage instructions will be added as the project develops.

```python
# Placeholder — full API documented in docs/
from gr_tracer import render

render.render_schwarzschild(mass=1.0, output="gallery/schwarzschild.png")
```

---

## License

MIT © 2026

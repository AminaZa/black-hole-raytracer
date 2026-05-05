# GR Black Hole Ray Tracer — Claude Code Guide

## Project Description

A physically accurate general relativistic ray tracer for Schwarzschild and Kerr black holes. Photon paths are computed by numerically integrating geodesic equations in curved spacetime, producing images of black holes with accretion disks, gravitational lensing, and Doppler/redshift effects.

## Tech Stack

- **Python 3.10+**
- **NumPy** — numerical integration, linear algebra, array operations
- **Matplotlib** — image display, diagnostic plots
- **Pillow** — final image output (PNG/JPEG)

## Project Structure

```
GR-tracer/
├── src/
│   ├── metrics/     # Spacetime metric tensors and Christoffel symbols
│   ├── geodesic/    # Geodesic integrators (RK4, RK45, etc.)
│   ├── camera/      # Camera model, ray generation
│   ├── scene/       # Scene description: black hole, accretion disk, background
│   ├── render/      # Render loop, tone mapping, image assembly
│   └── utils/       # Math helpers, coordinate transforms, constants
├── tests/           # Unit and integration tests (mirrors src/ layout)
├── examples/        # Runnable example scripts
├── gallery/         # Output images (git-ignored except .gitkeep)
├── docs/            # Supplementary documentation and derivations
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Coding Conventions

- **Type hints** on all function signatures; use `numpy.typing.NDArray` for arrays.
- **Docstrings** on every public function and class (NumPy docstring style).
- **snake_case** for all identifiers; `PascalCase` for classes.
- **Module size** — keep every `.py` file under 300 lines. Split by responsibility if a module grows larger.
- **No global mutable state.** Pass configuration explicitly via dataclasses or typed dicts.
- **SI units** throughout unless a function explicitly converts; document unit assumptions in docstrings.
- **Geometric units** (G = c = 1) inside the physics core; conversion utilities live in `src/utils/units.py`.
- Tests go in `tests/` mirroring the `src/` layout, named `test_<module>.py`.

## Physics Summary

We trace photon paths backwards from the camera through curved spacetime to a source (background sky texture or accretion disk). The spacetime geometry is encoded in a **metric tensor** gμν. Photons travel along **null geodesics** — curves satisfying:

```
d²xμ/dλ² + Γμαβ (dxα/dλ)(dxβ/dλ) = 0
```

where Γμαβ are the Christoffel symbols derived from the metric. We numerically integrate this ODE system in Boyer-Lindquist coordinates.

**Key conserved quantities** (Schwarzschild):
- Energy: E = (1 − 2M/r) ṫ
- Angular momentum: L = r² φ̇

**Key conserved quantities** (Kerr):
- Energy E, axial angular momentum Lz, Carter constant Q

The **photon sphere** at r = 3M (Schwarzschild) marks the innermost unstable circular orbit for photons. Rays that spiral inward past the **event horizon** (r = 2M) are absorbed and rendered black.

Accretion disk emission is modeled as a geometrically thin disk in the equatorial plane with a Novikov-Thorne temperature profile; Doppler boosting and gravitational redshift shift the observed color.

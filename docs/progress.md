# Progress Log

### 2026-05-06 — Phase 2.5: parallel + closed-form-Christoffel optimisation

Cut single-thread renderer overhead and fanned the work out across CPU cores.
The Schwarzschild integration is now ~25× faster end-to-end at 80×60 (1067
rays/s vs 42 rays/s baseline) and should land the 400×300 reference render in
2–3 minutes instead of ~48 minutes.

Changes:
- `src/metrics/schwarzschild.py` — added `acceleration(x, p)` returning
  `-Γ^μ_{αβ} p^α p^β` directly from the 11 non-zero connection coefficients,
  skipping the (4, 4, 4) tensor allocation and the einsum dispatch on every
  RK4 substep. Verified against `christoffel_symbols + einsum` to 4e-15.
- `src/geodesic/integrator.py` — the integrator picks up
  `metric.acceleration` automatically when present (still falls back to
  einsum/Christoffel for metrics without it, keeping the protocol generic).
  Step sizing is now three-tier: `r < 3 rs` uses `near_field_factor`,
  `r > 10 rs` uses a new `far_field_factor` (default 2.0), middle band uses
  `base_step`. Per-step constants are hoisted out of the loop and the accel
  closure is built once per call.
- `src/render/curved_renderer.py` — rewritten to vectorise per-pixel initial
  conditions (basis projection + null 4-momentum) and dispatch chunks to a
  `multiprocessing.Pool` (spawn context, so it works on Windows). Per-chunk
  progress percentage is printed as results land via `imap_unordered`. New
  `n_workers` (default = `os.cpu_count()`) and `chunks_per_worker` knobs;
  `n_workers=1` skips multiprocessing entirely for debugging.
- `examples/schwarzschild_render.py` — passes `far_field_factor=2.0`, prints
  total wall-clock render time at the end.

All 14 existing tests still pass without modification. Energy and L_z
conservation tests use explicit `base_step`/`near_field_factor` so they're
unaffected by the new defaults; the metric-agnostic stub test still
exercises the einsum fallback (the stub has no `acceleration`).

### 2026-05-06 — Phase 2 complete: Schwarzschild geodesic ray tracer

Phase 2 (curved spacetime, Schwarzschild) is working end-to-end. The first
ray-traced black hole image is at `gallery/schwarzschild_test.png`: it shows
the event horizon shadow, the lensed back-side of the accretion disk arching
over the BH ("Interstellar" effect), the photon ring at the shadow edge, and
the underside of the disk wrapping below.

Key files:
- `src/metrics/schwarzschild.py` — `SchwarzschildMetric` (M=1, rs=2 by default)
  with `metric_tensor()` and `christoffel_symbols()` in (t, r, θ, φ).
- `src/geodesic/integrator.py` — `GeodesicIntegrator` (RK4) and `GeodesicResult`
  dataclass. Metric-agnostic via `MetricProtocol`. Adaptive step (×0.1 inside
  r < 5 rs). Termination on horizon, escape (r ≥ r_max), in-annulus equatorial
  crossing (disk hit), or max_steps.
- `src/render/curved_renderer.py` — `CurvedRenderer` with Cartesian↔spherical
  conversion helpers and a static-observer null-momentum builder.
- `examples/schwarzschild_render.py` — camera at (0, 5, −30), M=1, disk
  r ∈ [3 rs, 12 rs] = [6, 24], 60° FOV, 400×300.
- `tests/metrics/test_schwarzschild.py`, `tests/geodesic/test_integrator.py` —
  14 tests covering metric components, Christoffel values & symmetry,
  horizon/escape/disk termination, energy & axial-angular-momentum
  conservation along a geodesic, and metric-agnostic integration with a
  zero-connection stub. All pass.

Benchmark: 400×300 = 120 000 rays in **2853.78 s (~47.6 min) at 42 rays/s**
on a single Python thread. The hot loop is per-ray RK4 with `np.einsum` over
the (4,4,4) Christoffel tensor; far-field rays escape quickly while near-BH
rays loop many steps around the photon sphere, so the rate is uneven through
the frame.

**Next session — Phase 3:** vectorise the integrator across rays so all pixels
advance in lockstep. Expected speed-up of 50–200× from amortising Python /
NumPy call overhead. After that, add gravitational redshift / Doppler shifting
for the disk emission and a starfield background.

### 2026-05-06 — Phase 1 complete

Phase 1 (flat spacetime ray tracer) is complete. A pinhole camera model,
accretion disk geometry, and closest-hit renderer are all working and produce
a correct perspective image (`gallery/flat_test.png`).

Key files:
- `src/camera/camera.py` — `Camera` class, fully vectorised ray generation
- `src/scene/accretion_disk.py` — `AccretionDisk` class, y=0 plane intersection, checkerboard colour
- `src/render/renderer.py` — `Renderer` class, closest-hit compositing, PNG export
- `examples/flat_render.py` — end-to-end example: camera at (0, 5, −15), disk r∈[3, 12], 800×600

**Next session — Phase 2:** implement the Schwarzschild metric and geodesic
equations. Entry points will be `src/metrics/` (metric tensor, Christoffel
symbols) and `src/geodesic/` (RK45 null geodesic integrator).

### 2026-05-06 — Docstring cleanup; Progress Log convention added

Trimmed all docstrings in `src/` and `examples/` to be concise and natural.
Removed verbose prose, redundant parameter explanations, and the numbered
Algorithm block in `renderer.py`. Protocol method stubs now have single-line
docstrings. No logic was changed.

Added the "Working Convention" section to this file establishing the rule:
append a Progress Log entry after every completed task.

### 2026-05-06 — Expanded CLAUDE.md with project standards

Added four new sections: Code Quality Standards, Documentation Standards,
Performance, and Architecture Principles. Key decisions recorded:
- Geodesic integrator must be metric-agnostic (Schwarzschild ↔ Kerr = one-line swap).
- Scene objects share a common `intersect()` / `color()` interface.
- Renderer is decoupled from spacetime geometry entirely.
- Render times must be benchmarked and logged.
- Academic paper references required where applicable.

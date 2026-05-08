# Progress Log

### 2026-05-08 — Polar-axis singularity in the integrator

The "Voronoi-cell" / streak artefacts in the disk renders weren't a
colour-map problem at all — they were the **spherical-coord polar
singularity**. The Schwarzschild and Kerr Christoffels carry `cot θ`
and `1/sin θ` terms that diverge at θ = 0, π, and the GPU integrator
took fixed RK4 steps right through the +y polar axis above the BH.
A diagnostic batch traced rays one by one and recorded their minimum
θ-to-pole distance: rays passing the axis ended up at θ ≈ −504 rad,
final r ≈ 1e43, with p^φ blown out exponentially by a few `cot(θ)·h`
kicks. That noise scattered ~36 pixels per axis-crossing column to
random sky/disk colours — the cellular pattern on screen.

Two coupled fixes in `src/geodesic/gpu_integrator.py` and
`src/geodesic/integrator.py`:

1. **Polar-proximity throttle.** Per-ray step `h` is multiplied by
   `max(min(1, sin θ / 0.1), 0.01)`. Below `sin θ = 0.1` the step
   shrinks linearly with distance to the pole; the 1% floor lets a
   ray exactly on the axis still advance. Cost: rays that genuinely
   pass near the pole take ≤ 100× more steps, but those are < 1% of
   the batch.
2. **Pole reflection on overshoot.** When a step still puts θ outside
   `[0, π]` the integrator maps `(θ, φ, p^θ) → (-θ or 2π − θ, φ + π,
   −p^θ)` — the same physical point on the other branch of the chart.

Diagnostics first ruled out the suspected causes: positions/momenta
stay `float64` end-to-end, the equatorial crossing is interpolated
linearly so θ at the recorded hit is pinned to π/2 within 2.2e-16,
and the r-coordinate of disk hits has no grid quantisation (median
adjacent diff ≈ 1e-9, min positive diff at machine epsilon). A
grayscale-of-r render of disk hits was already smooth.

`test_integrator_is_metric_agnostic` had to be loosened: its
`p^θ = 0.2` over λ = 58 was wrapping θ through π three times, which
is correct physics now. Added a paired
`test_polar_axis_reflection_in_flat_space` that *verifies* the wrap.
All 57 tests pass.

Refreshed cinematic + Kerr + Schwarzschild gallery renders: the
vertical speckled column above the shadow and the jagged "lip" at
the top of the lensed arc are gone, the disk gradient is smooth
through the lensing region, and the BH shadow boundary is clean.

### 2026-05-08 — Smooth blackbody colour map

Disk renders showed faint concentric arcs in the colour gradient. Tracked
to `src/scene/blackbody.py`: the Tanner Helland piecewise fit has step
discontinuities at the t = 66 (T = 6600 K) crossover — green snaps from
~255 to ~251 and blue from ~252 to forced 255. With the smooth radial
profile `T(r) = T_peak (r_isco/r)^(3/4)` those steps map to fixed-radius
isotemperature rings.

Replaced the piecewise fit with an analytic Planck × CIE 1931 2° observer
integration → linear sRGB (D65), per-row normalised so the brightest
channel sits at 1. The CIE table is sampled at 10 nm steps from 380 to
780 nm (41 wavelengths); the whole computation is `float64` and `C∞` in
`T`. Sweeping `T` from 1000 to 40000 K in 9.75 K increments, the maximum
single-step delta is 0.0025 in any channel — well below the 1/255
quantisation floor, so no banding survives 8-bit output.

All 15 disk unit tests still pass: inner-pixel-bluer-than-outer, Doppler
brightening of the approaching side, the `(1/γ)^3` time-dilation ratio,
and the prograde left/right asymmetry. Reference colours look right too
— 1500 K deep red, 3000 K incandescent, 6500 K near-white, 12000 K+
bluish.

### 2026-05-07 — Phase 6 + 7: animator and fused acceleration

Two interleaved pieces — the animator made the cost of a render frame
suddenly matter (we'd be paying it tens or hundreds of times), and the
fusion pass attacked the dominant cost.

**Phase 6 — frame loop and animation scripts:**
- `src/render/animator.py` — `Animator` runs a `render_frame(idx, total)`
  callback per frame, saves each frame as a numbered PNG into a staging
  directory, and stitches the result into a GIF via `imageio`. Two
  knobs that turned out to matter: `resume=True` (the default) skips
  frames whose PNG is already on disk, and `max_frames_per_run` caps
  fresh renders per call. Together they make a long animation
  resumable across sessions — important because the Claude Code
  background-task budget is 10 minutes, and a 120-frame 800×600
  animation is 30+ minutes even after the fusion pass.
- `examples/orbit_animation.py` — Kerr a = 0.9, camera at r = 30 M, 75°
  inclination, 120 frames sweeping azimuth 0 → 2π. Output:
  `gallery/orbit_animation.gif` (800 × 600 @ 24 fps).
- `examples/spin_animation.py` — fixed camera, 90 frames with the spin
  parameter sweeping linearly from 0 to 0.99. The metric, ISCO and
  disk inner edge are rebuilt per frame; each frame gets an `a = …`
  label burned into the top-left so the physics story narrates itself.
  Output: `gallery/spin_evolution.gif` (800 × 600 @ 24 fps).
- Both example scripts honour `ANIM_MAX_FRAMES_PER_RUN` (env var) so
  chunked runs are scriptable.

**Phase 7 — fused acceleration kernels:**

The first GPU integrator was launch-overhead bound: per RK4 substep the
metric's `acceleration_batch` issued ~30 small CuPy kernels (one per
arithmetic op), and an 800×600 frame integrated at 11 820 rays/s for a
40 s wall-clock — most of which was kernel-dispatch and small-array
allocations rather than FP64 compute.

Fix: a module-level `@cupy.fuse` function in each metric that
contracts the inner element-wise math into a single CUDA kernel.
- `src/metrics/schwarzschild.py` — `_schwarzschild_accel_fused` (one
  kernel for the whole inlined Schwarzschild acceleration).
  ``acceleration_batch`` dispatches to the fused path when given a cupy
  array and falls back to plain NumPy otherwise.
- `src/metrics/kerr.py` — `_kerr_accel_fused`. Same dispatch story.
  Substantially heavier kernel (full BL metric, derivatives, inverse,
  acceleration contraction inlined) but still one launch.
- Defined inside `if _cupy is not None:` so the module imports cleanly
  on machines without CuPy. The numpy path is untouched.

Numerics: fused vs NumPy agrees to **~3 × 10⁻¹⁴** float64 round-off,
identical to the pre-fusion result. All 56 tests still pass.

Performance — single 800 × 600 Kerr frame at a = 0.99:

| stage                      | rays/s | per-frame (s) |
|----------------------------|--------|---------------|
| CPU 20-worker (Phase 4)    | 1 439  | 333.6         |
| GPU unfused (Phase 5)      | 11 820 |  41.6         |
| GPU fused (this phase)     | 33 800 |  15.1         |

**~2.86× over the unfused GPU path; 22× over the multiprocess CPU.**

The fused-acceleration micro-benchmark dropped per-call cost from a few
ms (unfused) to **0.46 ms** (Kerr) / **0.58 ms** (Schwarzschild) at
N = 50 000. The remaining ~17 ms per RK4 step is the integrator's own
small-kernel chain (where, mask, RK4 update). That's the next
optimisation lever — fusing the RK4 update body itself — but is
deferred; the current speedup already brings full-resolution
animations into a workable range (orbit ≈ 30 min, spin ≈ 22 min).

### 2026-05-07 — Phase 5: GPU acceleration via CuPy

Phase 5 introduces a CuPy-backed batched ray tracer that integrates every
pixel ray simultaneously on the GPU. The CPU multiprocessing path stays
intact; example scripts auto-detect GPU and pick the right backend.

Hardware on this box: **NVIDIA RTX 3080 Ti Laptop, 16 GiB, CC 8.6**.

Install: ``cupy-cuda12x`` 13.6.0 plus the supporting NVIDIA wheels —
``nvidia-cuda-runtime-cu12``, ``nvidia-cuda-nvrtc-cu12``,
``nvidia-cuda-cccl-cu12``, ``nvidia-cublas-cu12``, ``nvidia-curand-cu12``,
``nvidia-cusolver-cu12``, ``nvidia-cusparse-cu12``, ``nvidia-cufft-cu12``.
On Windows pure-pip the NVRTC and runtime DLLs live under
``site-packages/nvidia/<lib>/bin``; CuPy needs them registered with
``os.add_dll_directory`` *and* prepended to ``PATH`` before import (so the
transitive ``LoadLibraryEx`` from inside ``nvrtc64_120_0.dll`` finds the
``nvrtc-builtins64_129.dll`` next to it). All of that is hidden inside
:mod:`utils.cuda_loader` — at import it sets up the DLL search paths,
imports CuPy, and exposes ``gpu_available`` and ``xp_module`` for the rest
of the codebase.

New / changed:
- `src/utils/cuda_loader.py` — DLL-path discovery, ``load_cupy``,
  ``device_summary``, ``asnumpy``, ``get_xp``. Linux / Mac just import
  CuPy directly; this whole file is a no-op there.
- `src/metrics/schwarzschild.py` and `src/metrics/kerr.py` — added
  ``acceleration_batch(positions, momenta)`` that mirrors the inlined
  scalar acceleration but operates on (N, 4) arrays. The array module is
  detected from ``positions`` via :func:`utils.cuda_loader.get_xp`, so the
  same code runs on numpy or cupy. CPU and GPU paths agree to ≈ 3·10⁻¹⁴
  on float64.
- `src/geodesic/gpu_integrator.py` — :class:`GpuGeodesicIntegrator` does
  RK4 on the whole (N, 4) ray batch in lockstep, with a per-ray
  ``live_mask`` for horizon / escape / disk termination. Polls
  ``live_mask.any()`` every 32 steps to amortise the GPU→CPU sync cost
  (each ``.any()`` was a full pipeline drain). The integrator is
  array-module agnostic — pass numpy arrays for a CPU dry-run, cupy for
  the GPU path. The 2D termination state machine is encoded as four
  module-level integer constants (``TERM_HORIZON`` etc.) so the result is
  trivially marshalable across the device boundary.
- `src/render/gpu_renderer.py` — :class:`GpuRenderer`. Builds initial
  (positions, momenta) on the host, uploads once, calls the integrator
  once, brings ``(termination, final_position, final_momentum)`` back to
  CPU and shades there. Disk colour and starfield sampling stay on the
  CPU — they're O(N) but small compared to the RK4 loop. Falls back to
  the numpy path when CuPy is unavailable, so the same renderer is also a
  single-process batched CPU renderer.
- `examples/kerr_render.py` and `examples/spin_comparison.py` — both
  scripts pick the GPU renderer when available, log
  ``device_summary()`` at startup, and fall back to the multiprocessing
  CPU path otherwise.
- `tests/metrics/test_acceleration_batch.py` (4 numpy parametrisations,
  2 cupy parametrisations gated on ``gpu_available``). Verifies the
  batched acceleration matches the scalar one element-wise.
- `tests/geodesic/test_gpu_integrator.py` (3 cases). Three canonical
  rays — horizon, escape, disk — agree with the per-ray CPU integrator
  bit-for-bit (interpolated disk crossing matches to 1e-10). A cupy-only
  case verifies the CuPy device path matches the numpy host path.

Performance (single-thread Python driver):

| Backend                  | 480k Kerr a=0.99 rays | rays/s    | Speedup vs single-thread CPU |
|--------------------------|-----------------------|-----------|------------------------------|
| CPU single-thread Kerr   | ≈ 35 min              | ~230      | 1×                           |
| CPU 8 workers (Phase 4)  | 5.6 min               | 1 439     | 6.3×                         |
| GPU batch (this phase)   | ≈ 45 s                | 10 620    | 46×                          |

The GPU win comes from launching ~30 vectorised kernels per RK4 step over
the full 480k-ray batch instead of running per-ray Python loops on 8 CPU
cores. The remaining headroom is kernel-launch overhead and many small
intermediate allocations in ``acceleration_batch`` — a fused custom CUDA
kernel (or ``cupy.fuse``) over the RK4 substep would fold those together
and is the obvious next optimisation. float32 mode would be another big
win on a consumer GPU (8× FP32 vs FP64 on Ampere) at the cost of some
near-photon-sphere precision.

Polar-axis BL singularity from Phase 4 is unchanged — the GPU integrator
inherits the same artefact. The fix is still the generic axis-reflection
step in the integrator and is now Phase 6 work.

### 2026-05-07 — Phase 4: Kerr metric (rotating black hole)

Phase 4 is in. The integrator was not touched — the Kerr metric drops in
through the same `MetricProtocol` (`rs`, `christoffel_symbols`,
`acceleration`) that Schwarzschild uses. Spin a = 0.99 M renders and a
4-panel a ∈ {0, 0.5, 0.9, 0.99} comparison are wired up.

New / changed:
- `src/metrics/kerr.py` — `KerrMetric(mass, spin)` in Boyer-Lindquist
  coordinates. Closed-form `metric_tensor`, `inverse_metric` (uses the exact
  identity `det(g_{tφ block}) = -Δ sin²θ`), `metric_derivatives` and
  `christoffel_symbols`. The integrator's hot path is `acceleration`, which
  is fully inlined: one `_aux` call per step, the dg matrix entries are kept
  as scalars, and the final 4-vector contraction is hand-unrolled. `rs` is
  set to the outer horizon `r_+ = M + √(M² − a²)` so the integrator's
  termination radius and step bands track the actual horizon, not 2M.
- `src/scene/kerr_accretion_disk.py` — `KerrAccretionDisk` with the
  Bardeen-Press-Teukolsky ISCO via Z₁, Z₂; defaults `inner_radius` to the
  prograde ISCO (≈ 1.45 M at a = 0.99). Uses prograde Kepler
  Ω = √M / (r^{3/2} + a √M). Per-photon redshift is computed from the
  conserved scalars: lower indices using the equatorial Kerr metric, then
  `g = E / [u^t (E − Ω L)]` applied to both temperature and beaming.
- `src/render/curved_renderer.py` — `_build_initial_momenta` now takes
  the gravitational radius scale `r_grav = 2M`, not the integrator's `rs`.
  For Schwarzschild the two coincide (rs = 2M); for Kerr the asymptotic
  static-observer redshift wants 2M, not the smaller r_+. Verified: all
  Schwarzschild tests pass unchanged.
- `examples/kerr_render.py` — a = 0.99 render at 800×600, 75° inclination,
  saved to `gallery/kerr_render.png`.
- `examples/spin_comparison.py` — 4-panel a ∈ {0, 0.5, 0.9, 0.99} tile at
  400×300 each with labels, saved to `gallery/spin_comparison.png`.
- Tests: `tests/metrics/test_kerr.py` (11 cases) covers parameter
  validation, horizon shrinkage with spin, exact reduction to Schwarzschild
  at a = 0 (metric and Christoffels), inverse-metric consistency, lower-
  index symmetry of Γ, agreement of inlined `acceleration` with the
  full-Christoffel route, integrator drop-in, and L_z conservation.
  `tests/scene/test_kerr_accretion_disk.py` (9 cases) covers the ISCO
  formula (6M at a=0, → M as a → M, retrograde > prograde), annulus
  clipping, the static-redshift-only path, and the Doppler asymmetry that
  spin produces. Two pre-existing physical-disk tests had wrong physics
  premises (claimed inner pixel brighter than outer, and g_dop = 1 for a
  radial photon) — both rewritten in terms of the correct physics
  (spectral character / time-dilation 1/γ).

Performance — the Kerr per-ray cost is the bottleneck:
- First Kerr cut (christoffel_symbols → einsum acceleration): ~42 rays/s
  single-thread on the 80×60 smoke test (matches the old Schwarzschild
  baseline before its Phase-2.5 rework).
- After hand-inlining `acceleration` (no full Γ tensor, scalar dg entries,
  unrolled final contraction): **~231 rays/s single-thread, 5.5× speedup.**
- Kerr is now ~4.6× slower per-ray than Schwarzschild's closed-form path
  (1067 rays/s). Acceptable given the substantially heavier metric. The
  main remaining win is per-ray vectorisation across the integrator — Phase
  5 work.

The integrator change-set is empty: zero edits in `src/geodesic/integrator.py`.

Known limitation in this phase: rays whose backward-traced trajectory grazes
the spin axis (θ → 0 or θ → π) hit the Boyer-Lindquist coordinate
singularity — `g^{φφ} ∝ 1/sin²θ` blows up there, the RK4 stride punches
across the axis, and those pixels emerge as a thin dark streak through the
projection of the +y spin axis on the image. The fix is a generic axis-
reflection step in the integrator (`θ → -θ`, `φ → φ + π` when θ wraps past
zero), which would equally benefit Schwarzschild renders that happen to aim
at the pole. Deferred until Phase 5 along with the per-ray vectorisation
work, since both are integrator-level changes.

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

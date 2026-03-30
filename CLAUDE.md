# AVAKG Project Notes

## Fast FlowPy (numba) Validation

Faithful port of avaframe `com4FlowPy` (flowClass.py + flowCore.py) to a single `@njit` numba function. Validated on demo dataset (KG mining site, 5x5 km DEM at 5m, 8767 release cells):

| Metric | Fast (numba) | AvaFrame | Match |
|--------|-------------|----------|-------|
| zDelta max | 93.52 m | 93.5 m | exact |
| cellCounts max | 3205 | 3205 | exact |
| travelLength max | 853.0 m | 853 m | exact |
| travelAngle range | 25.0-49.2 deg | 25.0-49.2 deg | exact |
| **Time** | **13 s** | **~20 min** | **~100x faster** |

### What's ported
- Persistence-based routing (parent direction weighting with 0.707 diagonal)
- `tan(beta/2)^exp` terrain routing
- Cumulative z_delta energy line (`z_delta + z_gamma - z_alpha`)
- Flux distribution with threshold redistribution
- `no_flow` backflow prevention
- `calcDistMin` (shortest path through parents)
- Flow-path and straight-line travel angles
- Cell count (first-visit only, matching avaframe)

### What's NOT ported
- Forest interaction (friction/detrainment)
- Infrastructure back-tracking
- Variable alpha/exp/uMax per cell
- Preview mode (skip already-hit release cells)
- Tiling (not needed — numba handles full domain)

### Architecture
- `flowpy_fast.py`: single `@njit(cache=True)` function `_process_release_cell()`
- Cell list flattened to pre-allocated numpy arrays (max 200k cells per release)
- Parent tracking: up to 16 parents per cell with (idx, dr, dc, zdelta, mindist)
- Outer loop in Python with progress logging
- First call includes numba JIT compilation (~2s overhead)

## com1DFA Profiling (21k particles, 50m mesh, 1188 timesteps)

| Function | Time | % | Description |
|----------|------|---|-------------|
| `computeForceSPHC` | 34.2s | 36% | SPH neighbor search + pressure gradient |
| `updateFieldsC` | 36.8s | 39% | Particle-to-grid interpolation |
| `computeForceC` | 8.1s | 9% | Gravity + friction per particle |
| `updatePositionC` | 7.1s | 7% | Velocity + position update |
| `getNeighborsC` | 3.8s | 4% | Grid cell assignment |
| Python overhead | 5.1s | 5% | Dict access, config reads |

**SPH + Fields = 75% of runtime.** Both are particle loops suitable for `numba.prange` parallelization.

### Numba port strategy (com1DFA)
1. Port `computeForceSPHC` to numba with `prange` — biggest win (36%)
2. Port `updateFieldsC` to numba with `prange` — second biggest (39%)
3. Port `computeForceC` (samosAT friction only) — 9%
4. Port `updatePositionC` — 7%
5. Bundle as drop-in replacement module

Expected speedup: 3-6x on Mac (8-10 cores), 2-3x on server (4 cores)

## TODO

- [x] Port `computeForceSPHC` to numba prange (36% → parallel, done)
- [ ] ~~Port `updateFieldsC` to numba prange~~ (chunk-reduce too slow, needs full rewrite)
- [x] Port `computeForceC` for samosAT friction to numba (9% → parallel, done)
- [ ] ~~Port `updatePositionC` to numba~~ (7%, marginal gain)
- [ ] **Full com1DFA rewrite in numba** — separate project, see architecture below

### com1DFA Full Rewrite Architecture (planned)
Goal: 5-10x over current Cython for large simulations (100k+ particles)

Design:
- Pre-allocated struct-of-arrays (no Python dicts in hot loop)
- All config parameters baked into flat struct at init time
- Single `@njit` time loop function — Python only for I/O
- Grid update via cell-sorted particle scatter (avoids race conditions)
- `prange` on all particle loops (force, SPH, position, fields)
- Persistent buffers: zero-and-reuse instead of allocate-per-step

Modules:
1. `init.py` — DEM loading, particle initialization from release polygons
2. `kernels.py` — njit force, SPH, position, fields kernels (all prange)
3. `timeloop.py` — njit time integration loop
4. `io.py` — peak field output, GeoTIFF writing
5. `run.py` — CLI entry point, config loading

Friction: samosAT only (covers 95% of use cases)
SPH: option 1 (SamosAT style, dz=0)
Interpolation: bilinear only
- [ ] Port forest interaction (friction/detrainment) to numba FlowPy
- [ ] Port infrastructure back-tracking to numba FlowPy
- [ ] Port variable alpha/exp/uMax per cell (spatially varying parameters from rasters)
- [ ] Port preview mode (skip release cells already hit by prior paths)
- [ ] Validate numba FlowPy on a larger domain with multiple release areas
- [ ] Add com4FlowPy output support to dashboard download ZIP
- [ ] Refine dashboard for FlowPy: better legends with actual value ranges
- [ ] ERA5 snow climatology end-to-end test on server
- [ ] Copernicus DEM auto-download test on server
- [ ] KML upload end-to-end test
- [ ] Multi-return-period (com1DFA) end-to-end test

## Deployment

- **Live**: https://apps.mountainfutures.ch/avaframe/
- **Server**: myserver (joel@192.168.1.120)
- **Service**: `systemctl --user status avaframe-webui`
- **Process**: gunicorn, 2 workers, 2 threads, port 5050
- **Caddy**: `handle_path /avaframe*` reverse proxy
- **Linger**: enabled (`loginctl enable-linger joel`)
- **Data**: `/home/joel/avaframe_projects/`
- **App**: `/home/joel/avaframe-webui/`
- **Repo**: `/home/joel/src/AVAKG/`

### Deploy updates
```bash
ssh joel@myserver
cd ~/src/AVAKG && git pull
cp -r * ~/avaframe-webui/
cp -r demo ~/avaframe-webui/  # if demo changed
# No restart needed for template/pipeline changes
# Restart for app.py changes:
lsof -ti :5050 | xargs kill; systemctl --user restart avaframe-webui
```

## Demo

- Cropped DEM from KG mining site (5x5 km, 3.8 MB)
- Single release polygon (`rel_demo.shp`, thickness=0.5m)
- Demo endpoint pre-creates step 1-4 checkpoints (skips to simulation)
- Default model: com4FlowPy (fast numba version)
- Runs in ~15 seconds including dashboard generation

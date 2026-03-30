"""Fast energy-line avalanche propagation model.

A simplified, performance-optimized reimplementation of the Flow-Py
algorithm (Huber et al. 2022) using numba JIT compilation.

Algorithm per release cell:
    1. Start at release cell (z_release)
    2. Propagate flux to downhill neighbors
    3. At each cell: zDelta = z_release - z_current - tan(alpha) * distance
    4. Spread flux weighted by slope angle^exp / sum(slope_angle^exp)
    5. Stop when zDelta <= 0 or flux < threshold

Outputs:
    - zDelta: energy line height at each cell (m)
    - cellCounts: number of flow paths affecting each cell
    - travelLength: maximum 3D travel distance (m)
    - fpTravelAngle: travel angle from release to each cell (degrees)

References:
    Huber et al. (2022): Flow-Py — routing and spreading.
    D'Amboise et al. (2022): Forest interaction.
"""

import numpy as np
from pathlib import Path

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: define njit as passthrough decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ---------------------------------------------------------------------------
# Core computation (numba-accelerated)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _propagate_single(dem, release_row, release_col, alpha_rad, exp,
                      flux_threshold, max_z, cellsize,
                      out_zdelta, out_counts, out_travlen, out_travang):
    """Propagate flow from a single release cell using BFS with flux spreading.

    Updates output arrays in-place (accumulated across all release cells).
    """
    nrows, ncols = dem.shape
    z0 = dem[release_row, release_col]
    if np.isnan(z0) or z0 <= -9000:
        return

    tan_alpha = np.tan(alpha_rad)

    # BFS queue: (row, col, flux, distance_2d, distance_3d)
    # Use fixed-size arrays instead of Python lists for numba
    max_queue = nrows * ncols
    q_row = np.empty(max_queue, dtype=np.int32)
    q_col = np.empty(max_queue, dtype=np.int32)
    q_flux = np.empty(max_queue, dtype=np.float64)
    q_dist = np.empty(max_queue, dtype=np.float64)

    # Visited tracking (avoid revisiting with lower flux)
    visited_flux = np.zeros((nrows, ncols), dtype=np.float64)

    # Initialize with release cell
    head = 0
    tail = 1
    q_row[0] = release_row
    q_col[0] = release_col
    q_flux[0] = 1.0
    q_dist[0] = 0.0
    visited_flux[release_row, release_col] = 1.0

    # 8-connectivity neighbors: (drow, dcol, distance_factor)
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
    diag = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])

    while head < tail:
        r = q_row[head]
        c = q_col[head]
        flux = q_flux[head]
        dist = q_dist[head]
        head += 1

        z_here = dem[r, c]
        if np.isnan(z_here) or z_here <= -9000:
            continue

        # Energy line height
        zdelta = z0 - z_here - tan_alpha * dist
        if zdelta < 0:
            continue
        if zdelta > max_z:
            zdelta = max_z

        # Record results (take max zdelta, accumulate counts)
        if zdelta > out_zdelta[r, c]:
            out_zdelta[r, c] = zdelta
        out_counts[r, c] += 1

        # Travel length and angle
        dz = z0 - z_here
        dist_3d = np.sqrt(dist * dist + dz * dz)
        if dist_3d > out_travlen[r, c]:
            out_travlen[r, c] = dist_3d
        if dist > 0:
            ang = np.degrees(np.arctan(dz / dist))
            if ang > out_travang[r, c]:
                out_travang[r, c] = ang

        # Spread to downhill neighbors
        # Compute slope angles to neighbors
        slopes = np.zeros(8, dtype=np.float64)
        total = 0.0
        for i in range(8):
            nr = r + dr[i]
            nc = c + dc[i]
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                continue
            z_nb = dem[nr, nc]
            if np.isnan(z_nb) or z_nb <= -9000:
                continue
            dz_nb = z_here - z_nb
            if dz_nb <= 0:
                continue  # only downhill
            d_nb = cellsize * diag[i]
            slope_tan = dz_nb / d_nb
            # Use slope^exp for weighting
            w = slope_tan ** exp
            slopes[i] = w
            total += w

        if total <= 0:
            continue

        # Distribute flux to downhill neighbors
        for i in range(8):
            if slopes[i] <= 0:
                continue
            nr = r + dr[i]
            nc = c + dc[i]
            d_nb = cellsize * diag[i]
            new_dist = dist + d_nb
            frac = slopes[i] / total
            new_flux = flux * frac

            if new_flux < flux_threshold:
                continue

            # Check energy line
            z_nb = dem[nr, nc]
            new_zdelta = z0 - z_nb - tan_alpha * new_dist
            if new_zdelta < 0:
                continue

            # Only visit if we bring more flux than before
            if new_flux > visited_flux[nr, nc]:
                visited_flux[nr, nc] = new_flux
                if tail < max_queue:
                    q_row[tail] = nr
                    q_col[tail] = nc
                    q_flux[tail] = new_flux
                    q_dist[tail] = new_dist
                    tail += 1


def run_flowpy_fast(dem, release_mask, cellsize, alpha=25.0, exp=8.0,
                    flux_threshold=3e-4, max_z=8848.0, log_fn=print):
    """Run fast energy-line propagation for all release cells.

    Parameters
    ----------
    dem : 2D numpy array
        DEM elevations. NaN or <= -9000 treated as nodata.
    release_mask : 2D numpy array (bool or uint8)
        True/nonzero where release cells are.
    cellsize : float
        DEM cell size in meters.
    alpha : float
        Angle of reach in degrees (default 25).
    exp : float
        Spreading exponent (default 8).
    flux_threshold : float
        Minimum flux for lateral spreading (default 3e-4).
    max_z : float
        Maximum energy line height in meters (default 8848).
    log_fn : callable
        Logging function.

    Returns
    -------
    dict with keys: zdelta, cellcounts, travellength, travelangle (all 2D arrays)
    """
    nrows, ncols = dem.shape
    alpha_rad = np.radians(alpha)

    # Output arrays
    out_zdelta = np.zeros((nrows, ncols), dtype=np.float64)
    out_counts = np.zeros((nrows, ncols), dtype=np.float64)
    out_travlen = np.zeros((nrows, ncols), dtype=np.float64)
    out_travang = np.zeros((nrows, ncols), dtype=np.float64)

    # Get release cell coordinates
    rel_rows, rel_cols = np.where(release_mask > 0)
    n_release = len(rel_rows)
    log_fn(f"[flowpy-fast] {n_release} release cells, DEM {ncols}x{nrows} @ {cellsize}m")

    if n_release == 0:
        log_fn("[flowpy-fast] WARNING: No release cells!")
        return {
            "zdelta": out_zdelta,
            "cellcounts": out_counts,
            "travellength": out_travlen,
            "travelangle": out_travang,
        }

    # Process each release cell
    log_interval = max(1, n_release // 20)
    for i in range(n_release):
        if i % log_interval == 0:
            pct = 100 * i / n_release
            log_fn(f"[flowpy-fast] {pct:.0f}% ({i}/{n_release})")

        _propagate_single(
            dem, rel_rows[i], rel_cols[i],
            alpha_rad, exp, flux_threshold, max_z, cellsize,
            out_zdelta, out_counts, out_travlen, out_travang,
        )

    log_fn(f"[flowpy-fast] 100% ({n_release}/{n_release})")

    # Stats
    affected = out_counts > 0
    log_fn(f"[flowpy-fast] Affected cells: {np.sum(affected)}")
    if np.any(out_zdelta > 0):
        log_fn(f"[flowpy-fast] zDelta max: {np.max(out_zdelta):.1f}m")
    if np.any(out_travlen > 0):
        log_fn(f"[flowpy-fast] Travel length max: {np.max(out_travlen):.0f}m")

    return {
        "zdelta": out_zdelta,
        "cellcounts": out_counts,
        "travellength": out_travlen,
        "travelangle": out_travang,
    }


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def run_fast_flowpy_pipeline(cfg, project_dir, log_fn=print):
    """Run fast FlowPy from pipeline config. Returns same format as simulation.py."""
    import rasterio
    from rasterio.features import rasterize
    import fiona
    from shapely.geometry import shape

    project_dir = Path(project_dir)
    sim = cfg.get("simulation", {})

    # Load DEM
    dem_files = list((project_dir / "Inputs").glob("*.tif"))
    dem_files = [f for f in dem_files if f.name != "release.tif" and "REL" not in str(f)]
    if not dem_files:
        raise FileNotFoundError("No DEM found")

    log_fn("[flowpy-fast] Loading DEM...")
    with rasterio.open(dem_files[0]) as src:
        dem = src.read(1).astype(np.float64)
        cellsize = abs(src.transform.a)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    # Build release mask from shapefiles
    rel_dir = project_dir / "Inputs" / "REL"
    geometries = []
    for shp in sorted(rel_dir.glob("*.shp")):
        with fiona.open(shp) as f:
            for feat in f:
                geom = shape(feat["geometry"])
                if geom.is_valid and not geom.is_empty:
                    geometries.append((geom, 1))

    if not geometries:
        raise ValueError("No release geometries found")

    log_fn(f"[flowpy-fast] Rasterizing {len(geometries)} release polygons...")
    release_mask = rasterize(
        geometries,
        out_shape=dem.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Run
    results = run_flowpy_fast(
        dem, release_mask, cellsize,
        alpha=float(sim.get("flowpy_alpha", 25)),
        exp=float(sim.get("flowpy_exp", 8)),
        flux_threshold=float(sim.get("flowpy_flux_threshold", 3e-4)),
        max_z=float(sim.get("flowpy_max_z", 8848)),
        log_fn=log_fn,
    )

    # Write output rasters
    out_dir = project_dir / "Outputs" / "com4FlowPy" / "peakFiles" / "res_fast"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in results.items():
        out_path = out_dir / f"fast_{name}.tif"
        p = profile.copy()
        p.update(dtype="float64", count=1, nodata=0)
        with rasterio.open(out_path, "w", **p) as dst:
            dst.write(data, 1)
        log_fn(f"[flowpy-fast] Wrote {out_path.name}")

    n_files = len(list(out_dir.glob("*.tif")))
    return {
        "runs": [{
            "label": "flowpy",
            "thickness_m": 0,
            "output_dir": str(out_dir),
            "n_sims": 1,
            "model": "com4FlowPy",
        }],
        "model": "com4FlowPy",
    }

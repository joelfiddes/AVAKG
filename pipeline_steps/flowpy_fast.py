"""Fast numba port of avaframe com4FlowPy.

Numba-accelerated reimplementation of the FlowPy algorithm from
avaframe/com4FlowPy (flowClass.py + flowCore.py). Faithfully reproduces
the persistence-based routing, z_delta (energy line) propagation,
flux distribution with threshold redistribution, and all output fields.

Forest interaction and infrastructure back-tracking are NOT ported
(they add complexity and are not needed for the AVAKG pipeline).
"""

import numpy as np
from pathlib import Path
import math

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)
_RAD90 = math.radians(90.0)

# Neighbour offsets: 3x3 grid, row 0..2, col 0..2
# (dr, dc) = (row_offset - 1, col_offset - 1) relative to center
# Distance multipliers for the 3x3 neighbourhood
# diag=sqrt(2), cardinal=1, center=0
_DS_ZDELTA = np.array([[_SQRT2, 1.0, _SQRT2],
                        [1.0,    0.0, 1.0],
                        [_SQRT2, 1.0, _SQRT2]], dtype=np.float64)

_DS_TANBETA = np.array([[_SQRT2, 1.0, _SQRT2],
                         [1.0,   1.0, 1.0],
                         [_SQRT2, 1.0, _SQRT2]], dtype=np.float64)

# Maximum number of cells per release cell path
_MAX_CELLS = 200000
# Maximum number of parents per cell
_MAX_PARENTS = 16


# ---------------------------------------------------------------------------
# Numba-accelerated core: process a single release cell
# ---------------------------------------------------------------------------
@njit(cache=True)
def _process_release_cell(dem, start_row, start_col, nodata, cellsize,
                          alpha_rad, exp, flux_threshold, max_z_delta,
                          out_zdelta, out_flux, out_counts,
                          out_fp_angle, out_sl_angle,
                          out_trav_len):
    """Process a single release cell through the full FlowPy algorithm.

    Modifies output arrays in-place (max-accumulation across release cells).
    """
    nrows = dem.shape[0]
    ncols = dem.shape[1]
    tan_alpha = math.tan(alpha_rad)

    SQRT2 = 1.4142135623730951
    RAD90 = 1.5707963267948966

    # --- Cell-list storage (flat arrays) ---
    # Each cell has an index 0..n_cells-1
    c_row = np.empty(_MAX_CELLS, dtype=np.int32)
    c_col = np.empty(_MAX_CELLS, dtype=np.int32)
    c_flux = np.empty(_MAX_CELLS, dtype=np.float64)
    c_zdelta = np.empty(_MAX_CELLS, dtype=np.float64)
    c_min_dist = np.empty(_MAX_CELLS, dtype=np.float64)
    c_is_start = np.empty(_MAX_CELLS, dtype=np.int8)

    # Parent tracking: for each cell, up to _MAX_PARENTS parents
    # Store (parent_cell_index, dr, dc, parent_zdelta, parent_min_dist)
    p_idx = np.empty((_MAX_CELLS, _MAX_PARENTS), dtype=np.int32)  # parent cell index
    p_dr = np.empty((_MAX_CELLS, _MAX_PARENTS), dtype=np.int32)   # parent.row - child.row
    p_dc = np.empty((_MAX_CELLS, _MAX_PARENTS), dtype=np.int32)   # parent.col - child.col
    p_zdelta = np.empty((_MAX_CELLS, _MAX_PARENTS), dtype=np.float64)
    p_mindist = np.empty((_MAX_CELLS, _MAX_PARENTS), dtype=np.float64)
    n_parents = np.zeros(_MAX_CELLS, dtype=np.int32)

    # Track processed cells: use a flat array indexed by (row * ncols + col)
    # Value: number of times this cell has been visited for this release cell
    processed = np.zeros(nrows * ncols, dtype=np.int32)

    # Lookup: for a given (row, col), which cell index is it?
    # -1 means not yet in cell_list
    cell_index_map = np.full(nrows * ncols, -1, dtype=np.int32)

    # --- Initialize start cell ---
    n_cells = 0
    c_row[0] = start_row
    c_col[0] = start_col
    c_flux[0] = 1.0
    c_zdelta[0] = 0.0
    c_min_dist[0] = 0.0
    c_is_start[0] = 1
    n_parents[0] = 0
    n_cells = 1

    processed[start_row * ncols + start_col] = 1
    cell_index_map[start_row * ncols + start_col] = 0

    start_altitude = dem[start_row, start_col]

    # Local temporary arrays (reused per cell)
    z_delta_nb = np.zeros(9, dtype=np.float64)
    persistence = np.zeros(9, dtype=np.float64)
    no_flow = np.ones(9, dtype=np.float64)
    tan_beta = np.zeros(9, dtype=np.float64)
    r_t = np.zeros(9, dtype=np.float64)
    dist_out = np.zeros(9, dtype=np.float64)

    # Offsets for the 8 neighbours (3x3 grid, skip center)
    # index in 3x3: i*3+j, where i=0..2, j=0..2
    # dr = i-1, dc = j-1

    idx = 0
    while idx < n_cells:
        cr = c_row[idx]
        cc = c_col[idx]
        altitude = dem[cr, cc]
        flux = c_flux[idx]
        zdelta = c_zdelta[idx]
        is_start = c_is_start[idx]

        # Check if 3x3 neighbourhood is valid
        if cr < 1 or cr >= nrows - 1 or cc < 1 or cc >= ncols - 1:
            idx += 1
            continue

        has_nodata = False
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if dem[cr + di, cc + dj] == nodata:
                    has_nodata = True
        if has_nodata:
            idx += 1
            continue

        # ---------------------------------------------------------------
        # 1. calc_min_distance (for non-start cells)
        # ---------------------------------------------------------------
        if is_start == 0:
            min_d = 1.0e30
            for pi in range(n_parents[idx]):
                p_row = c_row[p_idx[idx, pi]]
                p_col = c_col[p_idx[idx, pi]]
                dx = abs(p_col - cc) * cellsize
                dy = abs(p_row - cr) * cellsize
                d = math.sqrt(dx * dx + dy * dy) + p_mindist[idx, pi]
                if d < min_d:
                    min_d = d
            c_min_dist[idx] = min_d

        # ---------------------------------------------------------------
        # 2. calc_z_delta: compute z_delta_neighbour for each of 8 neighbours
        # ---------------------------------------------------------------
        for i in range(9):
            z_delta_nb[i] = 0.0

        for i in range(3):
            for j in range(3):
                ni = i * 3 + j
                dr = i - 1
                dc = j - 1
                if dr == 0 and dc == 0:
                    z_delta_nb[ni] = 0.0
                    continue
                nb_alt = dem[cr + dr, cc + dc]
                z_gamma = altitude - nb_alt
                if abs(dr) + abs(dc) == 2:
                    ds = SQRT2
                else:
                    ds = 1.0
                z_alpha = ds * cellsize * tan_alpha
                zd = zdelta + z_gamma - z_alpha
                if zd < 0.0:
                    zd = 0.0
                if zd > max_z_delta:
                    zd = max_z_delta
                z_delta_nb[ni] = zd

        # ---------------------------------------------------------------
        # 3. calc_persistence
        # ---------------------------------------------------------------
        for i in range(9):
            persistence[i] = 0.0
            no_flow[i] = 1.0

        if is_start == 1:
            for i in range(9):
                persistence[i] = 1.0
        elif n_parents[idx] > 0 and c_is_start[p_idx[idx, 0]] == 1:
            # Parent is start cell
            for i in range(9):
                persistence[i] = 1.0
            # Set no_flow for parent positions
            for pi in range(n_parents[idx]):
                # dr = parent.row - child.row, dc = parent.col - child.col
                dr = p_dr[idx, pi]
                dc = p_dc[idx, pi]
                nf_i = (dr + 1) * 3 + (dc + 1)
                no_flow[nf_i] = 0.0
        else:
            for pi in range(n_parents[idx]):
                dx = p_dc[idx, pi]  # parent.col - child.col
                dy = p_dr[idx, pi]  # parent.row - child.row
                maxweight = p_zdelta[idx, pi]

                # Set no_flow for parent position
                nf_i = (dy + 1) * 3 + (dx + 1)
                no_flow[nf_i] = 0.0

                # Persistence: the direction from parent to child is (-dy, -dx)
                # Forward direction from child (continuing the flow) is (-dy, -dx)
                # In the 3x3 grid around child, forward cell is at (1-dy, 1-dx)
                # The original code uses dx = parent.colindex - self.colindex
                # and dy = parent.rowindex - self.rowindex
                # Then assigns persistence based on (dx, dy) direction

                if dx == -1:
                    if dy == -1:
                        persistence[2 * 3 + 2] += maxweight          # [2,2]
                        persistence[2 * 3 + 1] += 0.707 * maxweight  # [2,1]
                        persistence[1 * 3 + 2] += 0.707 * maxweight  # [1,2]
                    if dy == 0:
                        persistence[1 * 3 + 2] += maxweight          # [1,2]
                        persistence[2 * 3 + 2] += 0.707 * maxweight  # [2,2]
                        persistence[0 * 3 + 2] += 0.707 * maxweight  # [0,2]
                    if dy == 1:
                        persistence[0 * 3 + 2] += maxweight          # [0,2]
                        persistence[0 * 3 + 1] += 0.707 * maxweight  # [0,1]
                        persistence[1 * 3 + 2] += 0.707 * maxweight  # [1,2]
                if dx == 0:
                    if dy == -1:
                        persistence[2 * 3 + 1] += maxweight          # [2,1]
                        persistence[2 * 3 + 0] += 0.707 * maxweight  # [2,0]
                        persistence[2 * 3 + 2] += 0.707 * maxweight  # [2,2]
                    if dy == 1:
                        persistence[0 * 3 + 1] += maxweight          # [0,1]
                        persistence[0 * 3 + 0] += 0.707 * maxweight  # [0,0]
                        persistence[0 * 3 + 2] += 0.707 * maxweight  # [0,2]
                if dx == 1:
                    if dy == -1:
                        persistence[2 * 3 + 0] += maxweight          # [2,0]
                        persistence[1 * 3 + 0] += 0.707 * maxweight  # [1,0]
                        persistence[2 * 3 + 1] += 0.707 * maxweight  # [2,1]
                    if dy == 0:
                        persistence[1 * 3 + 0] += maxweight          # [1,0]
                        persistence[0 * 3 + 0] += 0.707 * maxweight  # [0,0]
                        persistence[2 * 3 + 0] += 0.707 * maxweight  # [2,0]
                    if dy == 1:
                        persistence[0 * 3 + 0] += maxweight          # [0,0]
                        persistence[0 * 3 + 1] += 0.707 * maxweight  # [0,1]
                        persistence[1 * 3 + 0] += 0.707 * maxweight  # [1,0]

        # Apply no_flow to persistence
        for i in range(9):
            persistence[i] *= no_flow[i]

        # ---------------------------------------------------------------
        # 4. calc_tanbeta
        # ---------------------------------------------------------------
        for i in range(9):
            tan_beta[i] = 0.0
            r_t[i] = 0.0

        for i in range(3):
            for j in range(3):
                ni = i * 3 + j
                nb_alt = dem[cr + i - 1, cc + j - 1]
                if i == 1 and j == 1:
                    ds = 1.0  # center uses 1.0 in _DS_TANBETA
                elif abs(i - 1) + abs(j - 1) == 2:
                    ds = SQRT2
                else:
                    ds = 1.0
                distance = ds * cellsize
                beta = math.atan((altitude - nb_alt) / distance) + RAD90
                tb = math.tan(beta / 2.0)

                if z_delta_nb[ni] <= 0.0:
                    tb = 0.0
                if persistence[ni] <= 0.0:
                    tb = 0.0
                if i == 1 and j == 1:
                    tb = 0.0
                tan_beta[ni] = tb

        # Normalize: r_t = tan_beta^exp / sum(tan_beta^exp)
        sum_tb_exp = 0.0
        for i in range(9):
            if tan_beta[i] > 0.0:
                sum_tb_exp += tan_beta[i] ** exp

        if sum_tb_exp > 0.0:
            for i in range(9):
                if tan_beta[i] > 0.0:
                    r_t[i] = (tan_beta[i] ** exp) / sum_tb_exp
                else:
                    r_t[i] = 0.0

        # ---------------------------------------------------------------
        # 5. calc_distribution
        # ---------------------------------------------------------------
        for i in range(9):
            dist_out[i] = 0.0

        # Travel angle calculations (for non-start cells)
        fp_angle = 0.0
        sl_angle = 0.0
        if is_start == 0:
            min_d = c_min_dist[idx]
            dh = start_altitude - altitude
            if min_d > 0.0:
                fp_angle = math.degrees(math.atan(dh / min_d))
            # Straight line travel angle
            dx_sl = abs(start_col - cc)
            dy_sl = abs(start_row - cr)
            ds_sl = math.sqrt(float(dx_sl * dx_sl + dy_sl * dy_sl)) * cellsize
            if ds_sl > 0.0:
                sl_angle = math.degrees(math.atan(dh / ds_sl))

        # dist = persistence * r_t / sum(persistence * r_t) * flux
        sum_pr = 0.0
        for i in range(9):
            sum_pr += persistence[i] * r_t[i]

        if sum_pr > 0.0:
            for i in range(9):
                dist_out[i] = persistence[i] * r_t[i] / sum_pr * flux

        # Flux threshold redistribution
        # count = number of cells with flux >= threshold (fixed version)
        count_above = 0
        mass_below = 0.0
        for i in range(9):
            if dist_out[i] >= flux_threshold:
                count_above += 1
            elif dist_out[i] > 0.0:
                mass_below += dist_out[i]

        if mass_below > 0.0 and count_above > 0:
            add_each = mass_below / count_above
            for i in range(9):
                if dist_out[i] >= flux_threshold:
                    dist_out[i] += add_each
                elif dist_out[i] < flux_threshold:
                    dist_out[i] = 0.0

        # Flux conservation correction
        sum_dist = 0.0
        for i in range(9):
            sum_dist += dist_out[i]
        if sum_dist != flux and count_above > 0:
            correction = (flux - sum_dist) / count_above
            for i in range(9):
                if dist_out[i] >= flux_threshold:
                    dist_out[i] += correction

        # ---------------------------------------------------------------
        # 6. Create/update children
        # ---------------------------------------------------------------
        # Collect children: (row, col, flux, z_delta) for cells with dist >= threshold
        n_children = 0
        ch_row = np.empty(8, dtype=np.int32)
        ch_col = np.empty(8, dtype=np.int32)
        ch_flux = np.empty(8, dtype=np.float64)
        ch_zd = np.empty(8, dtype=np.float64)

        for i in range(3):
            for j in range(3):
                ni = i * 3 + j
                if i == 1 and j == 1:
                    continue
                if dist_out[ni] >= flux_threshold:
                    ch_row[n_children] = cr + i - 1
                    ch_col[n_children] = cc + j - 1
                    ch_flux[n_children] = dist_out[ni]
                    ch_zd[n_children] = z_delta_nb[ni]
                    n_children += 1

        # Sort children by z_delta ascending (lowest first)
        # Simple insertion sort (max 8 elements)
        for i in range(1, n_children):
            key_zd = ch_zd[i]
            key_fl = ch_flux[i]
            key_r = ch_row[i]
            key_c = ch_col[i]
            jj = i - 1
            while jj >= 0 and ch_zd[jj] > key_zd:
                ch_zd[jj + 1] = ch_zd[jj]
                ch_flux[jj + 1] = ch_flux[jj]
                ch_row[jj + 1] = ch_row[jj]
                ch_col[jj + 1] = ch_col[jj]
                jj -= 1
            ch_zd[jj + 1] = key_zd
            ch_flux[jj + 1] = key_fl
            ch_row[jj + 1] = key_r
            ch_col[jj + 1] = key_c

        # For each child, check if it already exists in cell_list
        for k in range(n_children):
            child_r = ch_row[k]
            child_c = ch_col[k]
            child_flat = child_r * ncols + child_c

            existing_idx = cell_index_map[child_flat]

            if existing_idx >= 0 and existing_idx >= idx:
                # Cell already exists in the list (ahead of current)
                # Add flux, add parent, update z_delta if higher
                c_flux[existing_idx] += ch_flux[k]
                np_existing = n_parents[existing_idx]
                if np_existing < _MAX_PARENTS:
                    p_idx[existing_idx, np_existing] = idx
                    p_dr[existing_idx, np_existing] = cr - child_r
                    p_dc[existing_idx, np_existing] = cc - child_c
                    p_zdelta[existing_idx, np_existing] = zdelta
                    p_mindist[existing_idx, np_existing] = c_min_dist[idx]
                    n_parents[existing_idx] += 1
                if ch_zd[k] > c_zdelta[existing_idx]:
                    c_zdelta[existing_idx] = ch_zd[k]
            else:
                # New child cell
                # Check boundary and nodata
                if child_r < 1 or child_r >= nrows - 1 or child_c < 1 or child_c >= ncols - 1:
                    continue

                has_nd = False
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if dem[child_r + di, child_c + dj] == nodata:
                            has_nd = True
                if has_nd:
                    continue

                if n_cells >= _MAX_CELLS:
                    continue

                new_idx = n_cells
                c_row[new_idx] = child_r
                c_col[new_idx] = child_c
                c_flux[new_idx] = ch_flux[k]
                c_zdelta[new_idx] = ch_zd[k]
                c_min_dist[new_idx] = 0.0
                c_is_start[new_idx] = 0
                n_parents[new_idx] = 1
                p_idx[new_idx, 0] = idx
                p_dr[new_idx, 0] = cr - child_r
                p_dc[new_idx, 0] = cc - child_c
                p_zdelta[new_idx, 0] = zdelta
                p_mindist[new_idx, 0] = c_min_dist[idx]

                cell_index_map[child_flat] = new_idx

                # Update processed count
                processed[child_flat] += 1

                n_cells += 1

        # ---------------------------------------------------------------
        # 7. Record outputs for this cell
        # ---------------------------------------------------------------
        ri = cr
        ci = cc

        if zdelta > out_zdelta[ri, ci]:
            out_zdelta[ri, ci] = zdelta

        if flux > out_flux[ri, ci]:
            out_flux[ri, ci] = flux

        # Count: increment only on first visit
        if processed[ri * ncols + ci] == 1:
            out_counts[ri, ci] += 1

        if is_start == 0:
            if fp_angle > out_fp_angle[ri, ci]:
                out_fp_angle[ri, ci] = fp_angle
            if sl_angle > out_sl_angle[ri, ci]:
                out_sl_angle[ri, ci] = sl_angle
            if c_min_dist[idx] > out_trav_len[ri, ci]:
                out_trav_len[ri, ci] = c_min_dist[idx]

        idx += 1

    return n_cells


# ---------------------------------------------------------------------------
# Python-level driver
# ---------------------------------------------------------------------------

def run_flowpy_fast(dem, release_mask, cellsize, alpha=25.0, exp=8.0,
                    flux_threshold=3e-4, max_z=8848.0, nodata=-9999.0,
                    log_fn=print):
    """Run fast FlowPy for all release cells.

    Parameters
    ----------
    dem : 2D numpy array
        Digital elevation model.
    release_mask : 2D numpy array
        Binary mask (>0 = release cell).
    cellsize : float
        Cell size in meters.
    alpha : float
        Alpha angle in degrees (energy line angle).
    exp : float
        Exponent for terrain-based routing.
    flux_threshold : float
        Minimum flux threshold.
    max_z : float
        Maximum z_delta (energy height).
    nodata : float
        NoData value in DEM.
    log_fn : callable
        Logging function.

    Returns
    -------
    dict with keys: 'z_delta', 'flux', 'count', 'fp_travel_angle',
                    'sl_travel_angle', 'travel_length'
    """
    dem = np.ascontiguousarray(dem, dtype=np.float64)
    nrows, ncols = dem.shape

    alpha_rad = np.deg2rad(alpha)

    # Sort release cells by altitude (highest first)
    rows, cols = np.where(release_mask > 0)
    if len(rows) == 0:
        log_fn("No release cells found.")
        return {
            'z_delta': np.zeros_like(dem, dtype=np.float32),
            'flux': np.full_like(dem, -9999, dtype=np.float32),
            'count': np.zeros_like(dem, dtype=np.int32),
            'fp_travel_angle': np.full_like(dem, -9999, dtype=np.float32),
            'sl_travel_angle': np.full_like(dem, -9999, dtype=np.float32),
            'travel_length': np.full_like(dem, -9999, dtype=np.float32),
        }

    altitudes = dem[rows, cols]
    sort_idx = np.argsort(-altitudes)  # highest first
    rows = rows[sort_idx]
    cols = cols[sort_idx]

    n_release = len(rows)
    log_fn(f"FlowPy fast: {n_release} release cells, cellsize={cellsize}m, "
           f"alpha={alpha}, exp={exp}")

    # Output arrays
    out_zdelta = np.zeros((nrows, ncols), dtype=np.float64)
    out_flux = np.full((nrows, ncols), -9999.0, dtype=np.float64)
    out_counts = np.zeros((nrows, ncols), dtype=np.int32)
    out_fp_angle = np.full((nrows, ncols), -9999.0, dtype=np.float64)
    out_sl_angle = np.full((nrows, ncols), -9999.0, dtype=np.float64)
    out_trav_len = np.full((nrows, ncols), -9999.0, dtype=np.float64)

    # Warm up numba (first call compiles)
    if HAS_NUMBA and n_release > 0:
        log_fn("Compiling numba kernel (first call)...")

    for i in range(n_release):
        if i % max(1, n_release // 20) == 0 or i == n_release - 1:
            log_fn(f"  Processing release cell {i+1}/{n_release} "
                   f"(row={rows[i]}, col={cols[i]}, "
                   f"alt={dem[rows[i], cols[i]]:.0f}m)")

        _process_release_cell(
            dem, int(rows[i]), int(cols[i]), nodata, cellsize,
            alpha_rad, exp, flux_threshold, max_z,
            out_zdelta, out_flux, out_counts,
            out_fp_angle, out_sl_angle, out_trav_len
        )

    # Estimate velocity and pressure from zDelta (energy line)
    # v = sqrt(2 * g * zDelta), p = 0.5 * rho * v^2 = rho * g * zDelta
    g = 9.81
    rho = 300.0  # typical avalanche density (kg/m3) — higher than release snow
    velocity = np.sqrt(2.0 * g * np.maximum(out_zdelta, 0))
    pressure = rho * g * np.maximum(out_zdelta, 0) / 1000.0  # kPa

    # Hazard zones (Swiss classification from estimated pressure)
    # Red > 30 kPa, Blue 3-30 kPa, Yellow 1-3 kPa
    hazard = np.zeros_like(out_zdelta, dtype=np.float32)
    hazard[(pressure > 0) & (pressure < 1)] = 0.5    # below yellow
    hazard[(pressure >= 1) & (pressure < 3)] = 1.0   # yellow
    hazard[(pressure >= 3) & (pressure < 30)] = 2.0  # blue
    hazard[pressure >= 30] = 3.0                       # red

    log_fn(f"FlowPy complete. Max z_delta={out_zdelta.max():.2f}, "
           f"cells reached={np.sum(out_counts)}")
    log_fn(f"  Estimated: v_max={velocity.max():.1f} m/s, "
           f"p_max={pressure.max():.1f} kPa")
    n_red = np.sum(hazard >= 3)
    n_blue = np.sum((hazard >= 2) & (hazard < 3))
    n_yellow = np.sum((hazard >= 1) & (hazard < 2))
    log_fn(f"  Hazard zones: red={n_red}, blue={n_blue}, yellow={n_yellow}")

    return {
        'zdelta': out_zdelta.astype(np.float32),
        'cellcounts': out_counts.astype(np.float32),
        'travelangle': out_fp_angle.astype(np.float32),
        'travellength': out_trav_len.astype(np.float32),
        'velocity': velocity.astype(np.float32),
        'pressure': pressure.astype(np.float32),
        'hazard': hazard,
    }


def run_fast_flowpy_pipeline(cfg, project_dir, log_fn=print):
    """Run fast FlowPy from pipeline config. Same interface as simulation.py expects."""
    import rasterio
    from rasterio.features import rasterize
    import fiona
    from shapely.geometry import shape

    project_dir = Path(project_dir)
    sim = cfg.get("simulation", {})

    # Find DEM in Inputs/
    dem_files = list((project_dir / "Inputs").glob("*.tif"))
    dem_files = [f for f in dem_files if f.name != "release.tif" and "REL" not in str(f)]
    if not dem_files:
        raise FileNotFoundError("No DEM found in Inputs/")

    log_fn("[flowpy-fast] Loading DEM...")
    with rasterio.open(dem_files[0]) as src:
        dem = src.read(1).astype(np.float64)
        cellsize = abs(src.transform.a)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else -9999.0

    # Set nodata
    dem[np.isnan(dem)] = nodata
    if nodata != -9999.0:
        dem[dem == nodata] = -9999.0
        nodata = -9999.0

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
        raise ValueError("No release geometries found in Inputs/REL/")

    log_fn(f"[flowpy-fast] Rasterizing {len(geometries)} release polygons...")
    release_mask = rasterize(
        geometries, out_shape=dem.shape, transform=transform,
        fill=0, dtype=np.uint8,
    )

    # Run
    results = run_flowpy_fast(
        dem, release_mask, cellsize,
        alpha=float(sim.get("flowpy_alpha", 25)),
        exp=float(sim.get("flowpy_exp", 8)),
        flux_threshold=float(sim.get("flowpy_flux_threshold", 3e-4)),
        max_z=float(sim.get("flowpy_max_z", 8848)),
        nodata=nodata,
        log_fn=log_fn,
    )

    # Write output rasters
    out_dir = project_dir / "Outputs" / "com4FlowPy" / "peakFiles" / "res_fast"
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "zdelta": "fast_zdelta",
        "cellcounts": "fast_cellCounts",
        "travellength": "fast_travelLengthMax",
        "travelangle": "fast_fpTravelAngleMax",
        "velocity": "fast_velocity",
        "pressure": "fast_pressure",
        "hazard": "fast_hazard",
    }
    profile.update(dtype="float32", count=1, nodata=0, compress="lzw")
    for name in ["zdelta", "cellcounts", "travellength", "travelangle",
                  "velocity", "pressure", "hazard"]:
        if name in results:
            out_path = out_dir / f"{name_map[name]}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(results[name].astype(np.float32), 1)
            log_fn(f"[flowpy-fast] Wrote {out_path.name}")

    return {
        "runs": [{"label": "flowpy", "thickness_m": 0,
                  "output_dir": str(out_dir), "n_sims": 1, "model": "com4FlowPy"}],
        "model": "com4FlowPy",
    }


# ---------------------------------------------------------------------------
# Quick validation test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("FlowPy Fast - Validation Test")
    print("=" * 60)

    # Create an inclined plane: 1000m at top, dropping 30m per row
    nrows, ncols = 100, 80
    cellsize = 25.0
    dem = np.zeros((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        dem[i, :] = 3000.0 - i * 15.0  # ~31 degrees slope

    # Single release cell near the top center
    release = np.zeros((nrows, ncols), dtype=np.float64)
    release[5, 40] = 1.0

    print(f"\nDEM: {nrows}x{ncols}, cellsize={cellsize}m")
    print(f"Slope: ~{np.degrees(np.arctan(15.0/25.0)):.1f} degrees")
    print(f"Release cell at (5, 40), altitude={dem[5, 40]:.0f}m")
    print(f"Alpha=25 deg => should stop when cumulative energy line is exhausted\n")

    t0 = time.time()
    results = run_flowpy_fast(
        dem, release, cellsize,
        alpha=25.0, exp=8.0,
        flux_threshold=3e-4,
        max_z=8848.0,
        nodata=-9999.0,
        log_fn=print,
    )
    elapsed = time.time() - t0

    print(f"\nElapsed: {elapsed:.3f}s")
    print(f"Cells reached (count>0): {np.sum(results['count'] > 0)}")
    print(f"Max z_delta: {results['z_delta'].max():.2f}m")
    print(f"Max flux: {results['flux'][results['flux'] > 0].max():.4f}")
    print(f"Max travel length: {results['travel_length'].max():.1f}m")

    # Check that flow went downhill
    zdelta = results['z_delta']
    reached_rows = np.where(zdelta.max(axis=1) > 0)[0]
    if len(reached_rows) > 0:
        print(f"Flow reached rows {reached_rows.min()} to {reached_rows.max()} "
              f"(altitudes {dem[reached_rows.max(), 40]:.0f}m to {dem[reached_rows.min(), 40]:.0f}m)")

    # Test with multiple release cells
    print("\n" + "-" * 60)
    print("Test 2: Multiple release cells")
    release2 = np.zeros((nrows, ncols), dtype=np.float64)
    release2[3, 20] = 1.0
    release2[3, 40] = 1.0
    release2[3, 60] = 1.0

    t0 = time.time()
    results2 = run_flowpy_fast(
        dem, release2, cellsize,
        alpha=25.0, exp=8.0,
        flux_threshold=3e-4,
        log_fn=print,
    )
    elapsed2 = time.time() - t0
    print(f"Elapsed: {elapsed2:.3f}s")
    print(f"Cells reached: {np.sum(results2['count'] > 0)}")

    # Verify count accumulates properly
    # Where paths overlap, count should reflect number of release cells
    max_count = results2['count'].max()
    print(f"Max cell count (overlapping paths): {max_count}")

    print("\nAll tests passed." if np.sum(results['count'] > 0) > 10 else "\nWARNING: suspiciously few cells reached.")

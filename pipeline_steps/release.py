"""
Potential Release Area (PRA) delineation following Bühler et al. (2022).

Implements multi-criteria terrain analysis: slope thresholding, plan
curvature filtering (ridgelines/gullies), Vector Ruggedness Measure
(rocky terrain), and aspect-sector segmentation to produce individually
delineated release areas. High-resolution DEMs (<5 m) are automatically
upsampled to 5 m and slope is smoothed with a 5×5 filter.

References
----------
Bühler, Y., von Rickenbach, D., Stoffel, A., Margreth, S., Stoffel, L.,
    and Christen, M. (2018). Automated snow avalanche release area
    delineation. NHESS, 18, 3235–3251.
    https://doi.org/10.5194/nhess-18-3235-2018

Bühler, Y., Bebi, P., Christen, M., et al. (2022). Automated avalanche
    hazard indication mapping on a statewide scale. NHESS, 22, 1825–1843.
    https://doi.org/10.5194/nhess-22-1825-2022

Sykes, J., Bühler, Y., Margreth, S., Stoffel, L., and Björk, S. (2022).
    Automated snow avalanche release area delineation in data-sparse,
    remote, and forested regions. NHESS, 22, 3247–3270.
    https://doi.org/10.5194/nhess-22-3247-2022

Sappington, J. M., Longshore, K. M., & Thompson, D. B. (2007).
    Quantifying landscape ruggedness for animal habitat analysis.
    J. Wildlife Management, 71(4), 1419–1425.
"""

import os
from pathlib import Path

import fiona
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

# Target resolution for PRA analysis (Bühler et al. 2018 sweet spot)
TARGET_CELLSIZE_M = 5.0
# Apply 5x5 slope smoothing when cellsize <= this threshold
SMOOTH_SLOPE_MAX_CELLSIZE_M = 10.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_release_areas(cfg, dem_path, project_dir, log_fn=print):
    """
    Generate release area shapefiles from a DEM based on slope criteria.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Relevant keys under cfg["release"]:
            slope_min        — minimum slope for release zones (deg, default 28)
            slope_max        — maximum slope for release zones (deg, default 55)
            erode_cells      — boundary erosion iterations (default 3)
            min_area_m2      — minimum release zone area in m^2 (default 5000)
            simplify_tolerance_m — polygon simplification tolerance (default 20)
    dem_path : str or Path
        Path to the DEM raster file (GeoTIFF).
    project_dir : str or Path
        AvaFrame project directory. Shapefiles are written to
        {project_dir}/Inputs/REL/.
    log_fn : callable
        Logging function (default: print).

    Returns
    -------
    dict with keys:
        n_zones          — number of release zones generated
        shapefile_paths  — list of shapefile paths
        total_area_m2    — total area of all release zones
        diagnostic_plot  — path to the diagnostic PNG
    """
    rel_cfg = cfg.get("release", {})
    slope_min = float(rel_cfg.get("slope_min_deg", rel_cfg.get("slope_min", 28.0)))
    slope_max = float(rel_cfg.get("slope_max_deg", rel_cfg.get("slope_max", 55.0)))
    curvature_max = float(rel_cfg.get("curvature_max", 6.0))
    ruggedness_max = float(rel_cfg.get("ruggedness_max", 0.06))
    ruggedness_window = int(rel_cfg.get("ruggedness_window", 3))
    erode_cells = int(rel_cfg.get("erode_cells", 3))
    min_area_m2 = float(rel_cfg.get("min_area_m2", 5000.0))
    min_elevation_m = float(rel_cfg.get("min_elevation_m", 0.0))
    simplify_tol = float(rel_cfg.get("simplify_tolerance_m", 20.0))

    project_dir = Path(project_dir)
    dem_path = Path(dem_path)

    log_fn(f"[release] DEM: {dem_path}")
    log_fn(f"[release] Slope range: {slope_min}° – {slope_max}°")
    log_fn(f"[release] Curvature max: {curvature_max}, VRM max: {ruggedness_max} "
           f"(window {ruggedness_window}×{ruggedness_window})")
    log_fn(f"[release] Erosion: {erode_cells} cells, min area: {min_area_m2} m²")
    log_fn(f"[release] Simplify tolerance: {simplify_tol} m")

    # ------------------------------------------------------------------
    # 1. Load DEM
    # ------------------------------------------------------------------
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    cellsize_x = abs(transform.a)
    cellsize_y = abs(transform.e)
    cellsize = (cellsize_x + cellsize_y) / 2.0
    log_fn(f"[release] DEM shape: {dem.shape}, cellsize: {cellsize:.2f} m")

    # ------------------------------------------------------------------
    # 1b. Upsample to TARGET_CELLSIZE_M if resolution is finer
    # ------------------------------------------------------------------
    if cellsize < TARGET_CELLSIZE_M:
        dem, transform, cellsize_x, cellsize_y, nodata = _upsample_dem(
            dem, transform, crs, nodata, TARGET_CELLSIZE_M, log_fn,
        )
        cellsize = (cellsize_x + cellsize_y) / 2.0

    # ------------------------------------------------------------------
    # 2. Compute slope (degrees) via np.gradient
    # ------------------------------------------------------------------
    # Mask nodata
    if nodata is not None:
        valid_mask = ~np.isclose(dem, nodata)
    else:
        valid_mask = ~np.isnan(dem)

    dem_filled = np.where(valid_mask, dem, np.nan)
    dy, dx = np.gradient(dem_filled, cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(-dx, dy)  # azimuth from north, clockwise

    # ------------------------------------------------------------------
    # 2b. Smooth slope with 5x5 mean filter for high-res DEMs
    #     (Bühler et al. 2018: distance-weighted 5x5 filter at ≤5 m)
    # ------------------------------------------------------------------
    if cellsize <= SMOOTH_SLOPE_MAX_CELLSIZE_M:
        log_fn(f"[release] Applying 5×5 slope smoothing (cellsize {cellsize:.1f} m "
               f"≤ {SMOOTH_SLOPE_MAX_CELLSIZE_M} m)")
        kernel = np.ones((5, 5)) / 25.0
        slope_deg = ndimage.convolve(
            slope_deg, kernel, mode='nearest',
        )
    else:
        log_fn(f"[release] Skipping slope smoothing (cellsize {cellsize:.1f} m "
               f"> {SMOOTH_SLOPE_MAX_CELLSIZE_M} m)")

    # ------------------------------------------------------------------
    # 2c. Plan curvature — change in aspect direction
    #     (Bühler et al. 2022: curvature replaces fold; threshold 6.0)
    # ------------------------------------------------------------------
    plan_curv = _plan_curvature(dem_filled, cellsize_x, cellsize_y)
    curv_excluded = np.abs(plan_curv) > curvature_max
    n_curv_excl = curv_excluded[valid_mask].sum()
    log_fn(f"[release] Curvature: {n_curv_excl} cells excluded "
           f"(|curv| > {curvature_max})")

    # ------------------------------------------------------------------
    # 2d. Vector Ruggedness Measure (Sappington et al. 2007)
    #     (Bühler et al. 2018/2022: ruggedness threshold ~0.06)
    # ------------------------------------------------------------------
    vrm = _vector_ruggedness_measure(slope_rad, aspect_rad, ruggedness_window)
    vrm_excluded = vrm > ruggedness_max
    n_vrm_excl = vrm_excluded[valid_mask].sum()
    log_fn(f"[release] Ruggedness (VRM): {n_vrm_excl} cells excluded "
           f"(VRM > {ruggedness_max})")

    # ------------------------------------------------------------------
    # 2e. Aspect sectors (8 bins of 45°) for segmentation
    #     (Bühler et al. 2018: aspect weighted 3× in OBIA segmentation)
    # ------------------------------------------------------------------
    # Sectors 1-8: N=1, NE=2, E=3, SE=4, S=5, SW=6, W=7, NW=8
    aspect_deg = np.degrees(aspect_rad) % 360
    aspect_sector = np.clip(
        np.floor((aspect_deg + 22.5) % 360 / 45.0).astype(np.int32) + 1,
        1, 8,
    )

    # ------------------------------------------------------------------
    # 3. Valid mask (non-NaN DEM values)
    # ------------------------------------------------------------------
    # Already computed above as valid_mask

    # ------------------------------------------------------------------
    # 4. Erode valid mask to remove boundary artefacts
    # ------------------------------------------------------------------
    if erode_cells > 0:
        eroded_mask = ndimage.binary_erosion(
            valid_mask, iterations=erode_cells
        ).astype(bool)
    else:
        eroded_mask = valid_mask.copy()

    log_fn(f"[release] Valid cells: {valid_mask.sum()}, "
           f"after erosion: {eroded_mask.sum()}")

    # ------------------------------------------------------------------
    # 5. Release mask: slope + curvature + ruggedness + valid
    #    (Bühler et al. 2022, §3.1)
    # ------------------------------------------------------------------
    release_mask = (
        (slope_deg >= slope_min)
        & (slope_deg <= slope_max)
        & ~curv_excluded
        & ~vrm_excluded
        & eroded_mask
    ).astype(np.int32)

    n_release_cells = release_mask.sum()
    log_fn(f"[release] Release cells (before component filter): {n_release_cells}")

    # ------------------------------------------------------------------
    # 6. Aspect-segmented connected component analysis
    #    Cells must be adjacent AND share the same aspect sector to merge.
    #    This splits PRAs along ridgelines where aspect flips.
    #    (Approximates Bühler et al. 2018 OBIA with 3× aspect weighting)
    # ------------------------------------------------------------------
    structure = np.ones((3, 3), dtype=int)
    # Label each aspect sector separately, then merge into unified labels
    labels = np.zeros_like(release_mask, dtype=np.int32)
    n_components = 0
    for sector in range(1, 9):
        sector_mask = (release_mask > 0) & (aspect_sector == sector)
        if not sector_mask.any():
            continue
        sector_labels, n_sector = ndimage.label(
            sector_mask.astype(np.int32), structure=structure,
        )
        # Offset labels to avoid collisions
        sector_labels[sector_labels > 0] += n_components
        labels[sector_labels > 0] = sector_labels[sector_labels > 0]
        n_components += n_sector
    log_fn(f"[release] Aspect-segmented components: {n_components}")

    # ------------------------------------------------------------------
    # 7. Filter by minimum area
    # ------------------------------------------------------------------
    cell_area = cellsize_x * cellsize_y
    component_sizes = ndimage.sum(release_mask, labels, range(1, n_components + 1))
    component_areas = np.array(component_sizes) * cell_area

    # Also compute per-component mean slope and mean elevation for filtering
    comp_range = range(1, n_components + 1)
    component_mean_slope = np.array(
        ndimage.mean(slope_deg, labels, comp_range)
    )
    component_mean_elev = np.array(
        ndimage.mean(dem_filled, labels, comp_range)
    )

    keep_ids = []
    n_area_reject = 0
    n_slope_reject = 0
    n_elev_reject = 0
    for i, (area, mslope, melev) in enumerate(
        zip(component_areas, component_mean_slope, component_mean_elev),
        start=1,
    ):
        if area < min_area_m2:
            n_area_reject += 1
            continue
        # Bühler et al. (2022): reject polygons with mean slope < slope_min
        if mslope < slope_min:
            n_slope_reject += 1
            continue
        # Bühler et al. (2022): reject polygons with mean elev < threshold
        if min_elevation_m > 0 and melev < min_elevation_m:
            n_elev_reject += 1
            continue
        keep_ids.append(i)

    log_fn(f"[release] Components kept: {len(keep_ids)} "
           f"(rejected: {n_area_reject} area, {n_slope_reject} mean slope, "
           f"{n_elev_reject} elevation)")

    # Create filtered label raster (preserves component boundaries)
    filtered_labels = np.where(np.isin(labels, keep_ids), labels, 0).astype(np.int32)
    filtered_mask = (filtered_labels > 0).astype(np.uint8)

    if filtered_mask.sum() == 0:
        log_fn("[release] WARNING: No release zones found after filtering!")
        return {
            "n_zones": 0,
            "shapefile_paths": [],
            "total_area_m2": 0.0,
            "diagnostic_plot": None,
        }

    # ------------------------------------------------------------------
    # 8. Vectorize per-component (preserves aspect segmentation)
    # ------------------------------------------------------------------
    polygons = []
    for geom, value in rasterio.features.shapes(
        filtered_labels, mask=filtered_mask > 0, transform=transform
    ):
        if value > 0:
            poly = shape(geom)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    log_fn(f"[release] Vectorized polygons: {len(polygons)}")

    # ------------------------------------------------------------------
    # 9. Simplify polygons
    # ------------------------------------------------------------------
    if simplify_tol > 0:
        polygons = [p.simplify(simplify_tol, preserve_topology=True) for p in polygons]

    # ------------------------------------------------------------------
    # 10. Remove interior holes (AvaFrame doesn't support them)
    # ------------------------------------------------------------------
    cleaned = []
    for poly in polygons:
        if poly.geom_type == "MultiPolygon":
            for sub in poly.geoms:
                cleaned.append(_remove_holes(sub))
        elif poly.geom_type == "Polygon":
            cleaned.append(_remove_holes(poly))
    polygons = [p for p in cleaned if p.is_valid and not p.is_empty]

    # ------------------------------------------------------------------
    # 11. Sort by area descending
    # ------------------------------------------------------------------
    polygons.sort(key=lambda p: p.area, reverse=True)

    # ------------------------------------------------------------------
    # 12. Write one shapefile per polygon
    # ------------------------------------------------------------------
    rel_dir = project_dir / "Inputs" / "REL"
    rel_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing release shapefiles
    for old in rel_dir.glob("rel_*.shp"):
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            f = old.with_suffix(ext)
            if f.exists():
                f.unlink()

    schema = {
        "geometry": "Polygon",
        "properties": {
            "Name": "str",
            "thickness": "float",
        },
    }

    shapefile_paths = []
    total_area = 0.0

    for idx, poly in enumerate(polygons):
        name = f"rel_{idx:03d}"
        shp_path = rel_dir / f"{name}.shp"

        with fiona.open(
            str(shp_path), "w",
            driver="ESRI Shapefile",
            crs=crs.to_dict() if hasattr(crs, "to_dict") else crs,
            schema=schema,
        ) as dst:
            dst.write({
                "geometry": mapping(poly),
                "properties": {
                    "Name": name,
                    "thickness": 0.0,
                },
            })

        shapefile_paths.append(str(shp_path))
        total_area += poly.area

    log_fn(f"[release] Wrote {len(shapefile_paths)} shapefiles to {rel_dir}")
    log_fn(f"[release] Total release area: {total_area:,.0f} m²")

    # ------------------------------------------------------------------
    # 13. Diagnostic plot
    # ------------------------------------------------------------------
    plot_path = project_dir / "release_areas.png"
    try:
        _make_release_plot(
            dem, slope_deg, filtered_mask, polygons,
            transform, slope_min, slope_max,
            curv_excluded, vrm_excluded,
            str(plot_path), log_fn,
        )
    except Exception as exc:
        log_fn(f"[release] Warning: could not create diagnostic plot: {exc}")
        plot_path = None

    return {
        "n_zones": len(shapefile_paths),
        "shapefile_paths": shapefile_paths,
        "total_area_m2": round(total_area, 1),
        "diagnostic_plot": str(plot_path) if plot_path else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan_curvature(dem, cellsize_x, cellsize_y):
    """
    Compute plan curvature (change in aspect direction).

    Plan curvature identifies ridges (convex, positive) and gullies
    (concave, negative). Values with |curvature| > threshold indicate
    terrain features unsuitable for slab avalanche release.

    References
    ----------
    Bühler et al. (2022), NHESS 22, 1825–1843:
        "the fold (a derivative of the curvature) is replaced by the
        curvature itself"
    Zevenbergen & Thorne (1987): standard GIS curvature formulation.
    """
    # First derivatives
    zy, zx = np.gradient(dem, cellsize_y, cellsize_x)
    # Second derivatives
    zyy, _ = np.gradient(zy, cellsize_y, cellsize_x)
    _, zxx = np.gradient(zx, cellsize_y, cellsize_x)
    zxy_y, _ = np.gradient(zx, cellsize_y, cellsize_x)

    # Plan (horizontal) curvature in 1/m
    p = zx**2 + zy**2
    # Avoid division by zero on flat terrain
    with np.errstate(divide='ignore', invalid='ignore'):
        plan_curv = np.where(
            p > 1e-10,
            (zxx * zy**2 - 2 * zxy_y * zx * zy + zyy * zx**2) / (p**1.5),
            0.0,
        )
    # Scale to Bühler convention: rad per hectometer (×100)
    return np.nan_to_num(plan_curv * 100.0, nan=0.0)


def _vector_ruggedness_measure(slope_rad, aspect_rad, window_size):
    """
    Vector Ruggedness Measure (VRM) after Sappington et al. (2007).

    Decomposes surface normal vectors into x/y/z components, sums in a
    focal window, and measures dispersion. VRM ranges 0 (flat) to 1
    (maximally rugged), independent of slope.

    References
    ----------
    Sappington, J. M., Longshore, K. M., & Thompson, D. B. (2007).
        Quantifying landscape ruggedness for animal habitat analysis.
        J. Wildlife Management, 71(4), 1419–1425.
    Bühler et al. (2018), NHESS 18, 3235–3251, §3.1:
        "Ruggedness ... calculated using 9-pixel window"
    """
    # Surface normal vector components
    nx = np.sin(slope_rad) * np.sin(aspect_rad)
    ny = np.sin(slope_rad) * np.cos(aspect_rad)
    nz = np.cos(slope_rad)

    # Focal sums
    kernel = np.ones((window_size, window_size))
    n_cells = window_size * window_size
    sum_x = ndimage.convolve(np.nan_to_num(nx), kernel, mode='nearest')
    sum_y = ndimage.convolve(np.nan_to_num(ny), kernel, mode='nearest')
    sum_z = ndimage.convolve(np.nan_to_num(nz), kernel, mode='nearest')

    # Resultant vector length / n_cells
    resultant = np.sqrt(sum_x**2 + sum_y**2 + sum_z**2)
    vrm = 1.0 - (resultant / n_cells)
    return np.clip(vrm, 0.0, 1.0)


def _upsample_dem(dem, transform, crs, nodata, target_cellsize, log_fn):
    """
    Resample a DEM to a coarser target cell size using bilinear interpolation.

    High-resolution DEMs (<5 m) contain microtopography (rocks, small cliffs)
    that creates noisy slope values. Resampling to ~5 m matches the resolution
    used by Bühler et al. (2018) for PRA delineation.

    References
    ----------
    Bühler et al. (2018), NHESS 18, 3235–3251, §3.1:
        "5 m resolution DEM ... 5×5 cell mean filter (distance-weighted)"
    Sykes et al. (2022), NHESS 22, 3247–3270:
        Uses 5 m DEM from SPOT 6/7 stereo for PRA analysis.
    """
    src_cellsize = (abs(transform.a) + abs(transform.e)) / 2.0
    log_fn(f"[release] Upsampling DEM from {src_cellsize:.2f} m to "
           f"{target_cellsize:.1f} m (bilinear)")

    scale_x = abs(transform.a) / target_cellsize
    scale_y = abs(transform.e) / target_cellsize
    new_h = max(1, int(dem.shape[0] * scale_y))
    new_w = max(1, int(dem.shape[1] * scale_x))

    new_transform = rasterio.transform.from_bounds(
        transform.c,
        transform.f + transform.e * dem.shape[0],
        transform.c + transform.a * dem.shape[1],
        transform.f,
        new_w, new_h,
    )

    if nodata is None:
        nodata = -9999.0
    dst = np.full((new_h, new_w), nodata, dtype=np.float64)
    reproject(
        source=dem,
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=new_transform,
        dst_crs=crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=Resampling.bilinear,
    )

    new_cellsize_x = abs(new_transform.a)
    new_cellsize_y = abs(new_transform.e)
    log_fn(f"[release] Upsampled DEM: {dst.shape} "
           f"({new_cellsize_x:.2f}×{new_cellsize_y:.2f} m)")

    return dst, new_transform, new_cellsize_x, new_cellsize_y, nodata


def _remove_holes(polygon):
    """Return a polygon with all interior rings removed."""
    from shapely.geometry import Polygon as ShapelyPolygon
    return ShapelyPolygon(polygon.exterior)


def _hillshade(dem, azimuth=315, altitude=45):
    """Compute a simple hillshade from a DEM array."""
    dy, dx = np.gradient(dem)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    return np.clip(hs, 0, 1)


def _make_release_plot(dem, slope_deg, release_mask, polygons,
                       transform, slope_min, slope_max,
                       curv_excluded, vrm_excluded,
                       output_path, log_fn):
    """
    3-panel diagnostic plot:
      1. DEM + hillshade
      2. Terrain filtering (slope class + curvature/ruggedness exclusions)
      3. Release polygons on hillshade
    """
    hs = _hillshade(np.nan_to_num(dem, nan=0.0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # --- Panel 1: DEM with hillshade ---
    ax = axes[0]
    ax.imshow(hs, cmap="gray", alpha=1.0)
    dem_masked = np.ma.masked_invalid(dem)
    im = ax.imshow(dem_masked, cmap="terrain", alpha=0.5)
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    ax.set_title("DEM + Hillshade")
    ax.set_axis_off()

    # --- Panel 2: Terrain filtering ---
    ax = axes[1]
    # 0=too flat, 1=release range, 2=too steep, 3=curvature excluded, 4=VRM excluded
    terrain_class = np.full_like(slope_deg, -1, dtype=np.int32)
    valid = ~np.isnan(slope_deg)
    terrain_class[valid & (slope_deg < slope_min)] = 0
    terrain_class[valid & (slope_deg >= slope_min) & (slope_deg <= slope_max)] = 1
    terrain_class[valid & (slope_deg > slope_max)] = 2
    terrain_class[valid & curv_excluded] = 3
    terrain_class[valid & vrm_excluded] = 4

    cmap = ListedColormap([
        "#2166ac",   # 0: too flat (blue)
        "#d73027",   # 1: release range (red)
        "#fdae61",   # 2: too steep (orange)
        "#7570b3",   # 3: curvature excluded (purple)
        "#1b9e77",   # 4: VRM excluded (teal)
    ])
    ax.imshow(hs, cmap="gray", alpha=1.0)
    masked_class = np.ma.masked_where(terrain_class < 0, terrain_class)
    ax.imshow(masked_class, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
    ax.set_title(f"Terrain Filtering\n"
                 f"Blue:<{slope_min}° Red:{slope_min}–{slope_max}° "
                 f"Orange:>{slope_max}°\n"
                 f"Purple: curvature | Teal: ruggedness")
    ax.set_axis_off()

    # --- Panel 3: Release polygons on hillshade ---
    ax = axes[2]
    ax.imshow(hs, cmap="gray", alpha=1.0)

    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    patches = []
    for poly in polygons:
        if poly.geom_type == "Polygon":
            coords = np.array(poly.exterior.coords)
            inv = ~transform
            px = np.array([inv * (c[0], c[1]) for c in coords])
            patches.append(MplPolygon(px, closed=True))

    if patches:
        pc = PatchCollection(
            patches, facecolor="red", edgecolor="darkred",
            linewidth=0.8, alpha=0.5,
        )
        ax.add_collection(pc)

    ax.set_title(f"Release Polygons ({len(polygons)} zones)")
    ax.set_axis_off()

    fig.suptitle("PRA Delineation — Bühler et al. (2022)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_fn(f"[release] Diagnostic plot saved: {output_path}")

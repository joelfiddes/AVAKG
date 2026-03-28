"""
Release area generation from slope analysis.

Refactored from the Jyrgalan approach: load DEM, compute slope, classify
potential release zones, vectorise connected components, write one shapefile
per polygon to {project_dir}/Inputs/REL/.
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
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


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
    slope_min = float(rel_cfg.get("slope_min", 28.0))
    slope_max = float(rel_cfg.get("slope_max", 55.0))
    erode_cells = int(rel_cfg.get("erode_cells", 3))
    min_area_m2 = float(rel_cfg.get("min_area_m2", 5000.0))
    simplify_tol = float(rel_cfg.get("simplify_tolerance_m", 20.0))

    project_dir = Path(project_dir)
    dem_path = Path(dem_path)

    log_fn(f"[release] DEM: {dem_path}")
    log_fn(f"[release] Slope range: {slope_min}° – {slope_max}°")
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
    # 5. Release mask: slope in [slope_min, slope_max] AND eroded valid
    # ------------------------------------------------------------------
    release_mask = (
        (slope_deg >= slope_min)
        & (slope_deg <= slope_max)
        & eroded_mask
    ).astype(np.int32)

    n_release_cells = release_mask.sum()
    log_fn(f"[release] Release cells (before component filter): {n_release_cells}")

    # ------------------------------------------------------------------
    # 6. Connected component analysis (8-connectivity)
    # ------------------------------------------------------------------
    structure = np.ones((3, 3), dtype=int)
    labels, n_components = ndimage.label(release_mask, structure=structure)
    log_fn(f"[release] Connected components found: {n_components}")

    # ------------------------------------------------------------------
    # 7. Filter by minimum area
    # ------------------------------------------------------------------
    cell_area = cellsize_x * cellsize_y
    component_sizes = ndimage.sum(release_mask, labels, range(1, n_components + 1))
    component_areas = np.array(component_sizes) * cell_area

    keep_ids = []
    for i, area in enumerate(component_areas, start=1):
        if area >= min_area_m2:
            keep_ids.append(i)

    log_fn(f"[release] Components with area >= {min_area_m2} m²: {len(keep_ids)}")

    # Create filtered mask
    filtered_mask = np.isin(labels, keep_ids).astype(np.uint8)

    if filtered_mask.sum() == 0:
        log_fn("[release] WARNING: No release zones found after filtering!")
        return {
            "n_zones": 0,
            "shapefile_paths": [],
            "total_area_m2": 0.0,
            "diagnostic_plot": None,
        }

    # ------------------------------------------------------------------
    # 8. Vectorize filtered mask
    # ------------------------------------------------------------------
    polygons = []
    for geom, value in rasterio.features.shapes(
        filtered_mask, mask=filtered_mask > 0, transform=transform
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
                       transform, slope_min, slope_max, output_path, log_fn):
    """
    3-panel diagnostic plot:
      1. DEM + hillshade
      2. Slope classification (below / release / above)
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

    # --- Panel 2: Slope classification ---
    ax = axes[1]
    # 0 = below slope_min, 1 = release range, 2 = above slope_max
    slope_class = np.zeros_like(slope_deg, dtype=np.int32)
    slope_class[(slope_deg >= slope_min) & (slope_deg <= slope_max)] = 1
    slope_class[slope_deg > slope_max] = 2
    slope_class[np.isnan(slope_deg)] = -1

    cmap = ListedColormap(["#2166ac", "#d73027", "#fdae61"])
    ax.imshow(hs, cmap="gray", alpha=1.0)
    masked_class = np.ma.masked_where(slope_class < 0, slope_class)
    ax.imshow(masked_class, cmap=cmap, alpha=0.5, vmin=0, vmax=2)
    ax.set_title(f"Slope Classification\n"
                 f"Blue: <{slope_min}° | Red: {slope_min}–{slope_max}° | "
                 f"Orange: >{slope_max}°")
    ax.set_axis_off()

    # --- Panel 3: Release polygons on hillshade ---
    ax = axes[2]
    ax.imshow(hs, cmap="gray", alpha=1.0)

    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    patches = []
    for poly in polygons:
        if poly.geom_type == "Polygon":
            # Convert geo coords to pixel coords
            coords = np.array(poly.exterior.coords)
            # Inverse transform: pixel = ~transform * (x, y)
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

    fig.suptitle("Release Area Generation — Diagnostic", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_fn(f"[release] Diagnostic plot saved: {output_path}")

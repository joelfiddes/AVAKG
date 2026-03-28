"""DEM acquisition and preprocessing for the unified AvaFrame pipeline."""
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import (
    calculate_default_transform,
    reproject,
)


def _fix_nodata(array: np.ndarray, nodata_value=None) -> np.ndarray:
    """Replace nodata sentinels (values <= 0 or < -1e8) with NaN."""
    out = array.astype(np.float32)
    mask = np.isnan(out) | (out < -1e8) | (out <= 0)
    if nodata_value is not None:
        mask |= np.isclose(out, nodata_value)
    out[mask] = np.nan
    return out


def _resample_and_save(
    src_path: str,
    dst_path: str,
    target_epsg: int,
    target_res_m: float,
    log_fn=print,
) -> dict:
    """Reproject and resample a DEM to the target CRS and resolution.

    Returns metadata dict with dem_path, crs, shape, bounds_wgs84.
    """
    dst_crs = f"EPSG:{target_epsg}"

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_res_m, target_res_m),
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            nodata=np.nan,
            driver="GTiff",
            compress="deflate",
        )

        log_fn(f"  Reprojecting to {dst_crs}, {width}x{height} @ {target_res_m} m")

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                src_data = src.read(band_idx)
                dst_data = np.empty((height, width), dtype=np.float32)

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )

                dst_data = _fix_nodata(dst_data, nodata_value=src.nodata)
                dst.write(dst_data, band_idx)

    # Read back bounds in WGS84 for metadata
    bounds_wgs84 = _get_bounds_wgs84(dst_path)

    return {
        "dem_path": str(dst_path),
        "crs": dst_crs,
        "shape": (height, width),
        "bounds_wgs84": bounds_wgs84,
    }


def _get_bounds_wgs84(raster_path: str) -> tuple:
    """Return the bounding box of a raster in WGS84 (west, south, east, north)."""
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return tuple(bounds)


def _acquire_local(
    cfg: dict, domain: dict, project_dir: str, log_fn=print
) -> dict:
    """Handle local DEM mode: copy or resample as needed."""
    dem_cfg = cfg["dem"]
    src_path = Path(dem_cfg["path"])
    target_epsg = domain["target_epsg"]
    target_res = dem_cfg.get("target_res_m", 5)
    out_dir = Path(project_dir) / "Inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / "dem.tif"

    log_fn(f"Local DEM: {src_path}")

    with rasterio.open(src_path) as src:
        src_epsg = src.crs.to_epsg() if src.crs else None
        src_res = abs(src.transform.a)
        needs_reproject = src_epsg != target_epsg
        needs_resample = abs(src_res - target_res) > 0.01

    if not needs_reproject and not needs_resample:
        log_fn("  DEM already at target CRS and resolution, copying...")
        shutil.copy2(str(src_path), str(dst_path))

        # Still fix nodata in the copy
        with rasterio.open(dst_path, "r+") as ds:
            for band_idx in range(1, ds.count + 1):
                data = ds.read(band_idx)
                data = _fix_nodata(data, nodata_value=ds.nodata)
                ds.write(data, band_idx)
            ds.nodata = np.nan

        bounds_wgs84 = _get_bounds_wgs84(str(dst_path))
        with rasterio.open(dst_path) as ds:
            shape = (ds.height, ds.width)

        return {
            "dem_path": str(dst_path),
            "crs": f"EPSG:{target_epsg}",
            "shape": shape,
            "bounds_wgs84": bounds_wgs84,
        }
    else:
        log_fn(f"  Resampling from EPSG:{src_epsg} res={src_res:.1f} m")
        return _resample_and_save(
            str(src_path), str(dst_path), target_epsg, target_res, log_fn
        )


def _acquire_copernicus(
    cfg: dict, domain: dict, project_dir: str, log_fn=print
) -> dict:
    """Download Copernicus GLO-30 DEM and reproject to target CRS/resolution."""
    from dem_stitcher import stitch_dem

    dem_cfg = cfg["dem"]
    target_epsg = domain["target_epsg"]
    target_res = dem_cfg.get("target_res_m", 5)
    bbox = domain["bbox_wgs84"]

    out_dir = Path(project_dir) / "Inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "dem_raw_wgs84.tif"
    dst_path = out_dir / "dem.tif"

    log_fn(f"Downloading Copernicus GLO-30 for bbox {bbox}")
    X, profile = stitch_dem(
        list(bbox),
        dem_name="glo_30",
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )

    # Save raw WGS84 DEM
    profile.update(driver="GTiff", compress="deflate")
    with rasterio.open(str(raw_path), "w", **profile) as ds:
        ds.write(X, 1)
    log_fn(f"  Raw DEM saved: {raw_path} ({X.shape[1]}x{X.shape[0]})")

    # Reproject and resample to target
    result = _resample_and_save(
        str(raw_path), str(dst_path), target_epsg, target_res, log_fn
    )

    # Clean up raw file
    raw_path.unlink(missing_ok=True)

    return result


def acquire_dem(
    cfg: dict, domain: dict, project_dir: str, log_fn=print
) -> dict:
    """Acquire and preprocess DEM for the pipeline.

    Parameters
    ----------
    cfg : dict
        Full merged pipeline configuration.
    domain : dict
        Resolved domain from resolve_domain(), containing bbox_wgs84 and
        target_epsg.
    project_dir : str
        Path to the project directory.
    log_fn : callable
        Logging function (default: print). Allows the webui to capture output.

    Returns
    -------
    dict
        dem_path : str — path to the processed DEM GeoTIFF
        crs : str — CRS string (e.g. "EPSG:32632")
        shape : tuple — (height, width) of the raster
        bounds_wgs84 : tuple — (west, south, east, north) in WGS84
    """
    source = cfg.get("dem", {}).get("source", "copernicus")
    log_fn(f"DEM acquisition: source={source}")

    if source == "local":
        return _acquire_local(cfg, domain, project_dir, log_fn)
    elif source == "copernicus":
        return _acquire_copernicus(cfg, domain, project_dir, log_fn)
    else:
        raise ValueError(f"Unknown DEM source: {source}")

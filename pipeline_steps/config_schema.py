"""Config schema for unified AvaFrame pipeline."""
import copy
import math
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "project": {
        "name": "avalanche_project",
        "dir": "",
    },
    "domain": {
        "bbox": None,           # [west, south, east, north] in WGS84
        "shapefile": None,      # path to study area shapefile
        "center_lat": None,
        "center_lon": None,
        "buffer_m": 6000,
        "target_epsg": None,    # auto-detect if None
    },
    "dem": {
        "source": "copernicus", # or "local"
        "path": None,           # for local
        "target_res_m": 5,
    },
    "snow": {
        "source": "fixed",      # "era5", "manual", or "fixed"
        "thickness_m": 1.5,     # for fixed
        "fracture_depths": None, # for manual: {30: 0.68, 100: 0.95}
        "latitude": None,       # for era5
        "longitude": None,
        "start_year": 1950,
        "grid_elevation_m": None,
        "target_elevation_m": None,
        "mean_slope_deg": 35.0,
        "wind_loading_m": 0.0,
        "h72_gradient_cm_per_100m": 5.0,
        "return_periods": [30, 100, 300],
    },
    "release": {
        "slope_min_deg": 28,
        "slope_max_deg": 60,
        "min_area_m2": 2500,
        "simplify_tolerance_m": 5,
        "erode_cells": 3,
    },
    "simulation": {
        "friction_model": "samosATAuto",
        "mesh_cell_size_m": 5,
        "t_end_s": 600,
        "res_type": "ppr|pft|pfv",
        "snow_density": 200,
    },
    "dashboard": {
        "generate": True,
        "title": "",
        "downsample": 2,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base.

    Keys in override take precedence.  Nested dicts are merged rather than
    replaced outright.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(yaml_path: str) -> dict:
    """Read a YAML config file and deep-merge it with DEFAULT_CONFIG.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r") as fh:
        user_cfg = yaml.safe_load(fh) or {}

    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def validate_config(cfg: dict) -> list[str]:
    """Validate a merged configuration dictionary.

    Returns a list of human-readable error strings.  An empty list means the
    config is valid.
    """
    errors: list[str] = []

    # --- project ---
    if not cfg.get("project", {}).get("name"):
        errors.append("project.name is required")

    # --- domain: exactly one mode must be specified ---
    dom = cfg.get("domain", {})
    has_bbox = dom.get("bbox") is not None
    has_shp = dom.get("shapefile") is not None
    has_center = (
        dom.get("center_lat") is not None and dom.get("center_lon") is not None
    )
    has_from_dem = dom.get("from_dem", False)

    modes = sum([has_bbox, has_shp, has_center, has_from_dem])
    if modes == 0:
        errors.append(
            "domain: specify one of bbox, shapefile, center_lat+center_lon, or from_dem"
        )
    elif modes > 1:
        errors.append(
            "domain: only one of bbox, shapefile, center_lat+center_lon, or from_dem "
            "should be set"
        )
    if has_from_dem and cfg.get("dem", {}).get("source") != "local":
        errors.append("domain.from_dem requires dem.source = 'local'")

    if has_bbox:
        bbox = dom["bbox"]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            errors.append("domain.bbox must be [west, south, east, north]")
        else:
            w, s, e, n = bbox
            if not (-180 <= w <= 180 and -180 <= e <= 180):
                errors.append("domain.bbox longitudes must be in [-180, 180]")
            if not (-90 <= s <= 90 and -90 <= n <= 90):
                errors.append("domain.bbox latitudes must be in [-90, 90]")
            if w >= e:
                errors.append("domain.bbox: west must be < east")
            if s >= n:
                errors.append("domain.bbox: south must be < north")

    if has_shp:
        shp_path = Path(dom["shapefile"])
        if not shp_path.exists():
            errors.append(f"domain.shapefile not found: {dom['shapefile']}")

    if has_center:
        lat = dom["center_lat"]
        lon = dom["center_lon"]
        if not (-90 <= lat <= 90):
            errors.append("domain.center_lat must be in [-90, 90]")
        if not (-180 <= lon <= 180):
            errors.append("domain.center_lon must be in [-180, 180]")
        buf = dom.get("buffer_m", 6000)
        if buf <= 0:
            errors.append("domain.buffer_m must be > 0")

    # --- dem ---
    dem = cfg.get("dem", {})
    source = dem.get("source", "copernicus")
    if source not in ("copernicus", "local"):
        errors.append(f"dem.source must be 'copernicus' or 'local', got '{source}'")
    if source == "local" and not dem.get("path"):
        errors.append("dem.path is required when dem.source is 'local'")
    # Don't check dem.path existence here — file may be uploaded or
    # created by demo endpoint between config validation and pipeline run
    if dem.get("target_res_m", 5) <= 0:
        errors.append("dem.target_res_m must be > 0")

    # --- snow ---
    snow = cfg.get("snow", {})
    snow_source = snow.get("source", "fixed")
    if snow_source not in ("era5", "manual", "fixed"):
        errors.append(
            f"snow.source must be 'era5', 'manual', or 'fixed', got '{snow_source}'"
        )
    if snow_source == "fixed":
        th = snow.get("thickness_m")
        if th is not None and th <= 0:
            errors.append("snow.thickness_m must be > 0")
    if snow_source == "manual":
        fd = snow.get("fracture_depths")
        if not isinstance(fd, dict) or len(fd) == 0:
            errors.append(
                "snow.fracture_depths must be a non-empty dict when source is 'manual'"
            )
    if snow_source == "era5":
        if snow.get("latitude") is None or snow.get("longitude") is None:
            errors.append(
                "snow.latitude and snow.longitude are required when source is 'era5'"
            )

    # --- release ---
    rel = cfg.get("release", {})
    if rel.get("slope_min_deg", 28) >= rel.get("slope_max_deg", 60):
        errors.append("release.slope_min_deg must be < release.slope_max_deg")

    # --- simulation ---
    sim = cfg.get("simulation", {})
    if sim.get("mesh_cell_size_m", 5) <= 0:
        errors.append("simulation.mesh_cell_size_m must be > 0")

    return errors


def auto_detect_utm(lon: float, lat: float) -> int:
    """Return the EPSG code for the UTM zone covering the given WGS84 point.

    Parameters
    ----------
    lon : float
        Longitude in degrees (WGS84).
    lat : float
        Latitude in degrees (WGS84).

    Returns
    -------
    int
        EPSG code (e.g. 32632 for UTM zone 32N).
    """
    zone_number = int((lon + 180) / 6) + 1
    # Special zones for Norway / Svalbard
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
    elif 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    if lat >= 0:
        return 32600 + zone_number  # Northern hemisphere
    else:
        return 32700 + zone_number  # Southern hemisphere


def resolve_domain(cfg: dict) -> dict:
    """Resolve the domain section of the config into a canonical form.

    Returns a dict with:
        bbox_wgs84 : tuple (west, south, east, north)
        target_epsg : int

    Supports three modes:
        1. bbox — used directly
        2. shapefile — read with geopandas, extract total_bounds
        3. center+buffer — project to UTM, add buffer, project back
    """
    dom = cfg.get("domain", {})

    bbox_wgs84 = None

    # --- Mode 1: explicit bbox ---
    if dom.get("bbox") is not None:
        bbox_wgs84 = tuple(dom["bbox"])

    # --- Mode 2: shapefile ---
    elif dom.get("shapefile") is not None:
        import geopandas as gpd

        gdf = gpd.read_file(dom["shapefile"])
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        w, s, e, n = gdf.total_bounds
        bbox_wgs84 = (w, s, e, n)

    # --- Mode 3: center + buffer ---
    elif dom.get("center_lat") is not None and dom.get("center_lon") is not None:
        from pyproj import Transformer

        lat = dom["center_lat"]
        lon = dom["center_lon"]
        buf = dom.get("buffer_m", 6000)

        utm_epsg = auto_detect_utm(lon, lat)

        to_utm = Transformer.from_crs(4326, utm_epsg, always_xy=True)
        to_wgs = Transformer.from_crs(utm_epsg, 4326, always_xy=True)

        cx, cy = to_utm.transform(lon, lat)
        w_utm = cx - buf
        e_utm = cx + buf
        s_utm = cy - buf
        n_utm = cy + buf

        w, s = to_wgs.transform(w_utm, s_utm)
        e, n = to_wgs.transform(e_utm, n_utm)
        bbox_wgs84 = (w, s, e, n)

    # --- Mode 4: from DEM extent ---
    elif dom.get("from_dem", False):
        import rasterio
        from rasterio.warp import transform_bounds

        dem_path = cfg.get("dem", {}).get("path")
        if not dem_path:
            raise ValueError("domain.from_dem requires dem.path to be set")
        with rasterio.open(dem_path) as src:
            if src.crs.to_epsg() == 4326:
                bbox_wgs84 = (src.bounds.left, src.bounds.bottom,
                              src.bounds.right, src.bounds.top)
            else:
                wb = transform_bounds(src.crs, "EPSG:4326",
                                      src.bounds.left, src.bounds.bottom,
                                      src.bounds.right, src.bounds.top)
                bbox_wgs84 = wb

    else:
        raise ValueError(
            "Cannot resolve domain: provide bbox, shapefile, center_lat+center_lon, or from_dem"
        )

    # Auto-detect UTM from centroid
    w, s, e, n = bbox_wgs84
    centroid_lon = (w + e) / 2.0
    centroid_lat = (s + n) / 2.0

    target_epsg = dom.get("target_epsg")
    if target_epsg is None:
        target_epsg = auto_detect_utm(centroid_lon, centroid_lat)

    return {
        "bbox_wgs84": bbox_wgs84,
        "target_epsg": target_epsg,
    }


def get_config_template() -> str:
    """Return a YAML template string with comments for all config sections."""
    return """\
# ============================================================
# AvaFrame Unified Pipeline Configuration
# ============================================================

project:
  name: my_avalanche_project    # Project name (used for directory naming)
  dir: ""                        # Project directory (auto-created if empty)

# Domain definition — choose ONE of three modes:
#   1. bbox:       explicit bounding box [west, south, east, north] in WGS84
#   2. shapefile:  path to a study-area shapefile
#   3. center+buffer: center_lat, center_lon, buffer_m
domain:
  # Mode 1: bounding box
  # bbox: [10.5, 46.8, 10.7, 47.0]

  # Mode 2: shapefile
  # shapefile: /path/to/study_area.shp

  # Mode 3: center + buffer
  center_lat: 46.9
  center_lon: 10.6
  buffer_m: 6000

  target_epsg: null              # Auto-detected from centroid if null

# DEM acquisition
dem:
  source: copernicus             # "copernicus" (Copernicus GLO-30) or "local"
  path: null                     # Path to local DEM file (required if source=local)
  target_res_m: 5                # Target resolution in metres

# Snow depth / fracture depth
snow:
  source: fixed                  # "era5", "manual", or "fixed"

  # -- fixed mode --
  thickness_m: 1.5               # Uniform fracture depth [m]

  # -- manual mode --
  # fracture_depths:             # Return-period -> depth [m]
  #   30: 0.68
  #   100: 0.95
  #   300: 1.25

  # -- era5 mode --
  # latitude: 46.9
  # longitude: 10.6
  # start_year: 1950
  # grid_elevation_m: 2500
  # target_elevation_m: 2800
  # mean_slope_deg: 35.0
  # wind_loading_m: 0.0
  # h72_gradient_cm_per_100m: 5.0

  return_periods: [30, 100, 300]

# Release area generation
release:
  slope_min_deg: 28              # Minimum slope for release areas
  slope_max_deg: 60              # Maximum slope for release areas
  min_area_m2: 2500              # Minimum release area size
  simplify_tolerance_m: 5        # Polygon simplification tolerance
  erode_cells: 3                 # Raster erosion (cells) before polygonising

# Simulation parameters
simulation:
  friction_model: samosATAuto    # AvaFrame friction model
  mesh_cell_size_m: 5            # Computational mesh size
  t_end_s: 600                   # Simulation end time [s]
  res_type: "ppr|pft|pfv"       # Result types (pressure, flow thickness, velocity)
  snow_density: 200              # Release snow density [kg/m3]

# Dashboard generation
dashboard:
  generate: true                 # Generate interactive HTML dashboard
  title: ""                      # Dashboard title (defaults to project name)
  downsample: 2                  # Raster downsample factor for visualisation
"""

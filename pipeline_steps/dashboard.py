"""Generate a standalone Leaflet.js HTML dashboard from AvaFrame results.

Reads peak-file rasters (pressure, flow thickness, velocity), computes
cell-wise max envelopes, renders RGBA overlay PNGs, and embeds them as
base64 data URIs in an interactive map with layer controls.
"""

import base64
import glob
import io
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _load_peak_envelope(peak_dir: str, pattern: str) -> tuple:
    """Load all TIFFs matching *pattern*, return cell-wise max + metadata.

    Parameters
    ----------
    peak_dir : str
        Directory containing peak-file TIFFs.
    pattern : str
        Glob pattern (e.g. ``*_ppr.tif``).

    Returns
    -------
    tuple
        ``(max_array, transform, crs, bounds)`` where *max_array* is 2-D
        numpy array of cell-wise maxima, *bounds* is (left, bottom, right, top).
        Returns ``(None, None, None, None)`` if no files found.
    """
    files = sorted(glob.glob(str(Path(peak_dir) / pattern)))
    if not files:
        return None, None, None, None

    # Read first file for shape / metadata
    with rasterio.open(files[0]) as src:
        envelope = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    # Stack remaining files and take max
    for fpath in files[1:]:
        with rasterio.open(fpath) as src:
            data = src.read(1).astype(np.float64)
            envelope = np.maximum(envelope, data)

    # Replace nodata / negative with 0
    envelope = np.where(np.isfinite(envelope), envelope, 0.0)
    envelope = np.clip(envelope, 0, None)

    return envelope, transform, crs, bounds


def _render_hazard_png(ppr_data: np.ndarray, downsample: int = 2) -> Image.Image:
    """Pressure array to RGBA with Swiss hazard-zone colours.

    Red:    > 30 kPa
    Blue:   3 - 30 kPa
    Yellow: 1 - 3 kPa
    Transparent: < 1 kPa or zero

    Parameters
    ----------
    ppr_data : np.ndarray
        Peak pressure in Pa.
    downsample : int
        Downsample factor (take every Nth pixel).

    Returns
    -------
    PIL.Image.Image
        RGBA image.
    """
    # Convert Pa -> kPa
    kpa = ppr_data[::downsample, ::downsample] / 1000.0
    h, w = kpa.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Red zone: > 30 kPa
    mask_red = kpa > 30
    rgba[mask_red] = [220, 40, 40, 200]

    # Blue zone: 3 - 30 kPa
    mask_blue = (kpa > 3) & (kpa <= 30)
    rgba[mask_blue] = [50, 100, 200, 180]

    # Yellow zone: 1 - 3 kPa
    mask_yellow = (kpa > 1) & (kpa <= 3)
    rgba[mask_yellow] = [240, 200, 40, 160]

    return Image.fromarray(rgba, "RGBA")


def _render_colormap_png(data: np.ndarray, cmap_name: str,
                         vmin: float, vmax: float,
                         downsample: int = 2) -> Image.Image:
    """Data array to RGBA using a matplotlib colourmap.

    Parameters
    ----------
    data : np.ndarray
        2-D data array.
    cmap_name : str
        Matplotlib colourmap name.
    vmin, vmax : float
        Colourmap range.
    downsample : int
        Downsample factor.

    Returns
    -------
    PIL.Image.Image
        RGBA image (transparent where data <= 0).
    """
    d = data[::downsample, ::downsample]
    normed = np.clip((d - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(normed) * 255).astype(np.uint8)
    # Make zero / near-zero transparent
    colored[d <= 0, 3] = 0
    return Image.fromarray(colored, "RGBA")


def _render_hillshade_png(dem: np.ndarray, cellsize: float,
                          downsample: int = 2) -> Image.Image:
    """DEM array to grayscale hillshade RGBA image.

    Uses gradient-based shading: ``shade = (-dx + dy) / sqrt(dx^2 + dy^2 + 1)``

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array.
    cellsize : float
        Cell size in map units (metres).
    downsample : int
        Downsample factor.

    Returns
    -------
    PIL.Image.Image
        RGBA hillshade image.
    """
    d = dem[::downsample, ::downsample]
    cs = cellsize * downsample
    dy, dx = np.gradient(d, cs)
    shade = (-dx + dy) / np.sqrt(dx ** 2 + dy ** 2 + 1)

    # Normalise to 0-255
    shade = ((shade + 1) / 2 * 255).astype(np.uint8)
    h, w = shade.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = shade
    rgba[:, :, 1] = shade
    rgba[:, :, 2] = shade
    rgba[:, :, 3] = 180  # semi-transparent
    return Image.fromarray(rgba, "RGBA")


def _image_to_data_uri(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _zones_to_geojson(rel_dir: str, target_crs: int = 4326) -> str:
    """Read all shapefiles from REL directory, merge, classify, return GeoJSON.

    Classification by area:
    - Large:  > 50 000 m2
    - Medium: > 10 000 m2
    - Small:  <= 10 000 m2

    Geometries are simplified (15 m tolerance) and reprojected to *target_crs*.

    Parameters
    ----------
    rel_dir : str
        Path to the REL directory containing shapefiles.
    target_crs : int
        Target CRS EPSG code.

    Returns
    -------
    str
        GeoJSON string.
    """
    shp_files = list(Path(rel_dir).glob("*.shp"))
    if not shp_files:
        return json.dumps({"type": "FeatureCollection", "features": []})

    gdfs = []
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        gdfs.append(gdf)

    merged = gpd.pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    merged = merged.copy()

    # Compute area in the native CRS (assumed projected)
    if merged.crs is not None and not merged.crs.is_geographic:
        merged["area_m2"] = merged.geometry.area
    else:
        merged["area_m2"] = 0.0

    # Classify
    def _classify(area):
        if area > 50_000:
            return "Large"
        elif area > 10_000:
            return "Medium"
        else:
            return "Small"

    merged["size_class"] = merged["area_m2"].apply(_classify)

    # Simplify in native CRS before reprojecting
    merged["geometry"] = merged.geometry.simplify(15, preserve_topology=True)

    # Reproject
    if merged.crs is not None and merged.crs.to_epsg() != target_crs:
        merged = merged.to_crs(epsg=target_crs)

    # Keep only relevant columns
    keep_cols = ["geometry", "size_class", "area_m2"]
    for col in list(merged.columns):
        if col not in keep_cols:
            merged = merged.drop(columns=[col], errors="ignore")

    return merged.to_json()


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1729; color: #e0e0e0; }}
#map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: 1; }}

/* Sidebar */
#sidebar {{
    position: absolute; top: 0; left: 0; bottom: 0;
    width: 320px; z-index: 1000;
    background: linear-gradient(180deg, #0f1729 0%, #1a2340 100%);
    border-right: 2px solid #c0392b;
    overflow-y: auto; overflow-x: hidden;
    transition: transform 0.3s ease;
    display: flex; flex-direction: column;
}}
#sidebar.collapsed {{ transform: translateX(-320px); }}

#sidebar-toggle {{
    position: absolute; top: 12px; z-index: 1001;
    left: 320px;
    width: 32px; height: 48px;
    background: #1a2340; border: 2px solid #c0392b;
    border-left: none; border-radius: 0 6px 6px 0;
    color: #e0e0e0; font-size: 18px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: left 0.3s ease;
}}
#sidebar.collapsed + #sidebar-toggle {{ left: 0px; }}

.sidebar-header {{
    padding: 16px; text-align: center;
    border-bottom: 1px solid rgba(192,57,43,0.4);
}}
.sidebar-header img {{ height: 40px; margin-bottom: 8px; }}
.sidebar-header h1 {{ font-size: 16px; color: #fff; margin: 4px 0; }}
.sidebar-header p {{ font-size: 12px; color: #888; }}

.sidebar-section {{
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.sidebar-section h3 {{
    font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
    color: #c0392b; margin-bottom: 8px;
}}

/* Radio / checkbox styling */
.layer-option {{ margin: 4px 0; display: flex; align-items: center; }}
.layer-option input {{ margin-right: 8px; accent-color: #c0392b; }}
.layer-option label {{ font-size: 13px; cursor: pointer; }}

/* Opacity slider */
.slider-row {{ display: flex; align-items: center; gap: 8px; margin: 8px 0; }}
.slider-row input[type=range] {{ flex: 1; accent-color: #c0392b; }}
.slider-row span {{ font-size: 12px; min-width: 32px; text-align: right; }}

/* Legend */
.legend-block {{ margin: 6px 0; }}
.legend-row {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; font-size: 12px; }}
.legend-swatch {{
    width: 18px; height: 14px; border-radius: 2px; flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.2);
}}

/* Return period selector */
.rp-selector {{ margin: 8px 0; }}
.rp-btn {{
    padding: 4px 10px; margin: 2px 4px 2px 0; border: 1px solid #c0392b;
    background: transparent; color: #e0e0e0; border-radius: 4px;
    cursor: pointer; font-size: 12px;
}}
.rp-btn.active {{ background: #c0392b; color: #fff; }}

/* Mobile */
@media (max-width: 768px) {{
    #sidebar {{ width: 280px; }}
    #sidebar-toggle {{ left: 280px; }}
    #sidebar.collapsed + #sidebar-toggle {{ left: 0; }}
}}

/* Attribution fix for dark background */
.leaflet-control-attribution {{ background: rgba(15,23,41,0.8) !important; color: #888 !important; }}
.leaflet-control-attribution a {{ color: #aaa !important; }}
</style>
</head>
<body>
<div id="map"></div>

<div id="sidebar">
    <div class="sidebar-header">
        <a href="https://mountainfutures.ch" target="_blank" rel="noopener">
            <img src="/static/mf_logo.svg" alt="Mountain Futures"
                 onerror="this.style.display='none'">
        </a>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>

    <div class="sidebar-section">
        <h3>Basemap</h3>
        <div class="layer-option">
            <input type="radio" name="basemap" id="bm-topo" value="topo" checked>
            <label for="bm-topo">OpenTopoMap</label>
        </div>
        <div class="layer-option">
            <input type="radio" name="basemap" id="bm-sat" value="satellite">
            <label for="bm-sat">Esri Satellite</label>
        </div>
        <div class="layer-option">
            <input type="radio" name="basemap" id="bm-osm" value="osm">
            <label for="bm-osm">OpenStreetMap</label>
        </div>
    </div>

    {rp_selector_html}

    {overlays_sidebar_html}

    {kml_layers_sidebar}

    <div class="sidebar-section">
        <h3>Opacity</h3>
        <div class="slider-row">
            <label style="font-size:12px;">Overlays</label>
            <input type="range" id="opacity-slider" min="0" max="100" value="80">
            <span id="opacity-val">80%</span>
        </div>
    </div>

    {legend_html}
</div>

<button id="sidebar-toggle" onclick="toggleSidebar()">&#9776;</button>

<script>
// --- Data ---
var overlayBounds = {overlay_bounds_js};
var releaseGeoJSON = {release_geojson_js};
var overlayData = {overlay_data_js};
var isMultiRun = {is_multi_run_js};
var runLabels = {run_labels_js};
var activeRun = runLabels[0] || "fixed";

// --- Map setup ---
var map = L.map('map', {{ zoomControl: true }}).setView([{center_lat}, {center_lon}], 13);

var basemaps = {{
    topo: L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 17,
        attribution: '&copy; OpenTopoMap contributors'
    }}),
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        maxZoom: 19,
        attribution: '&copy; Esri'
    }}),
    osm: L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
    }})
}};
basemaps.topo.addTo(map);

L.control.scale({{ imperial: false }}).addTo(map);

// --- Basemap radio ---
document.querySelectorAll('input[name="basemap"]').forEach(function(radio) {{
    radio.addEventListener('change', function() {{
        Object.values(basemaps).forEach(function(l) {{ map.removeLayer(l); }});
        basemaps[radio.value].addTo(map);
    }});
}});

// --- Overlay layers ---
var overlayLayers = {{}};

function createOverlays(runKey) {{
    var d = overlayData[runKey];
    if (!d) return;
    var b = overlayBounds;
    var layers = {{}};
    if (d.hillshade) layers.hillshade = L.imageOverlay(d.hillshade, b);
    // com1DFA overlays
    if (d.hazard) layers.hazard = L.imageOverlay(d.hazard, b);
    if (d.pressure) layers.pressure = L.imageOverlay(d.pressure, b);
    if (d.thickness) layers.thickness = L.imageOverlay(d.thickness, b);
    if (d.velocity) layers.velocity = L.imageOverlay(d.velocity, b);
    // com4FlowPy overlays
    if (d.zdelta) layers.zdelta = L.imageOverlay(d.zdelta, b);
    if (d.cellcounts) layers.cellcounts = L.imageOverlay(d.cellcounts, b);
    if (d.travelangle) layers.travelangle = L.imageOverlay(d.travelangle, b);
    if (d.travellength) layers.travellength = L.imageOverlay(d.travellength, b);
    return layers;
}}

// Release zones
var relColors = {{ Large: '#e74c3c', Medium: '#e67e22', Small: '#f1c40f' }};
var releaseLayer = L.geoJSON(releaseGeoJSON, {{
    style: function(f) {{
        var c = relColors[f.properties.size_class] || '#f1c40f';
        return {{ color: c, weight: 2, fillColor: c, fillOpacity: 0.25 }};
    }},
    onEachFeature: function(f, layer) {{
        var a = f.properties.area_m2;
        var txt = f.properties.size_class + ' (' + (a > 1000 ? (a/1000).toFixed(1) + ' km' : a.toFixed(0) + ' m') + '\\u00B2)';
        layer.bindPopup(txt);
    }}
}});

function showOverlays() {{
    // Remove existing
    Object.values(overlayLayers).forEach(function(l) {{ map.removeLayer(l); }});
    if (map.hasLayer(releaseLayer)) map.removeLayer(releaseLayer);

    var layers = createOverlays(activeRun);
    if (!layers) return;
    overlayLayers = layers;

    var opacity = parseInt(document.getElementById('opacity-slider').value) / 100;

    // Add checked layers
    var order = ['hillshade', 'hazard', 'pressure', 'thickness', 'velocity',
                 'zdelta', 'cellcounts', 'travelangle', 'travellength'];
    order.forEach(function(key) {{
        var cb = document.getElementById('cb-' + key);
        if (cb && cb.checked && layers[key]) {{
            layers[key].setOpacity(opacity);
            layers[key].addTo(map);
        }}
    }});

    if (document.getElementById('cb-release').checked) {{
        releaseLayer.addTo(map);
    }}
}}

// --- Layer toggle ---
['hillshade','hazard','pressure','thickness','velocity',
 'zdelta','cellcounts','travelangle','travellength','release'].forEach(function(key) {{
    var cb = document.getElementById('cb-' + key);
    if (cb) cb.addEventListener('change', function() {{
        showOverlays();
        // Toggle legends
        var legendId = 'legend-' + key;
        var el = document.getElementById(legendId);
        if (el) el.style.display = cb.checked ? 'block' : 'none';
    }});
}});

// --- Opacity slider ---
document.getElementById('opacity-slider').addEventListener('input', function() {{
    document.getElementById('opacity-val').textContent = this.value + '%';
    var op = parseInt(this.value) / 100;
    Object.values(overlayLayers).forEach(function(l) {{ if (map.hasLayer(l)) l.setOpacity(op); }});
}});

// --- Return period selector ---
if (isMultiRun) {{
    document.querySelectorAll('.rp-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            document.querySelectorAll('.rp-btn').forEach(function(b) {{ b.classList.remove('active'); }});
            btn.classList.add('active');
            activeRun = btn.dataset.run;
            showOverlays();
        }});
    }});
}}

// --- Sidebar toggle ---
function toggleSidebar() {{
    document.getElementById('sidebar').classList.toggle('collapsed');
}}

// --- KML custom layers ---
var kmlLayersData = {kml_layers_js};
var kmlLeafletLayers = [];
kmlLayersData.forEach(function(layerDef, idx) {{
    var geojson = JSON.parse(layerDef.geojson);
    var style = layerDef.style || {{}};
    var lyr = L.geoJSON(geojson, {{
        style: function() {{ return style; }},
        pointToLayer: function(feature, latlng) {{
            return L.circleMarker(latlng, {{
                radius: style.radius || 5,
                color: style.color || '#e6194b',
                weight: 2,
                fillColor: style.fillColor || style.color || '#e6194b',
                fillOpacity: style.fillOpacity || 0.15
            }});
        }},
        onEachFeature: function(f, layer) {{
            var parts = [];
            if (f.properties.name) parts.push('<b>' + f.properties.name + '</b>');
            if (f.properties.description) parts.push(f.properties.description);
            if (parts.length) layer.bindPopup(parts.join('<br>'));
        }}
    }});
    lyr.addTo(map);
    kmlLeafletLayers.push(lyr);

    var cb = document.querySelector('[data-kml-idx="' + idx + '"]');
    if (cb) {{
        cb.addEventListener('change', function() {{
            if (this.checked) {{ lyr.addTo(map); }}
            else {{ map.removeLayer(lyr); }}
        }});
    }}
}});

// --- Fit map to overlay bounds ---
map.fitBounds(overlayBounds);
showOverlays();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# KML parsing
# ---------------------------------------------------------------------------

_KML_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9",
]

_KML_NS = "{http://www.opengis.net/kml/2.2}"


def _parse_kml_to_geojson_layers(kml_dir: str, log_fn=print) -> list:
    """Parse all KML files in a directory into GeoJSON layer dicts.

    Each KML file becomes one or more layers. Features are grouped by
    the folder/category they belong to in the KML structure.

    Returns list of dicts: [{"name": str, "geojson": str, "color": str, "style": dict}, ...]
    """
    kml_path = Path(kml_dir)
    if not kml_path.is_dir():
        return []

    kml_files = list(kml_path.glob("*.kml")) + list(kml_path.glob("*.KML"))
    if not kml_files:
        return []

    layers = []
    color_idx = 0

    for kml_file in sorted(kml_files):
        log_fn(f"[dashboard] Parsing KML: {kml_file.name}")
        try:
            tree = ET.parse(str(kml_file))
        except ET.ParseError as e:
            log_fn(f"[dashboard]   WARNING: Could not parse {kml_file.name}: {e}")
            continue

        root = tree.getroot()

        # Collect placemarks grouped by parent folder name
        groups = {}  # folder_name -> list of GeoJSON features

        def _extract_coords(coord_text):
            """Parse KML coordinate string into list of [lon, lat, alt]."""
            coords = []
            for chunk in coord_text.strip().split():
                parts = chunk.split(",")
                if len(parts) >= 2:
                    lon = round(float(parts[0]), 5)
                    lat = round(float(parts[1]), 5)
                    coords.append([lon, lat])
            return coords

        def _placemark_to_feature(pm):
            """Convert a KML Placemark element to a GeoJSON feature dict or None."""
            name_el = pm.find(f"{_KML_NS}name")
            desc_el = pm.find(f"{_KML_NS}description")
            props = {}
            if name_el is not None and name_el.text:
                props["name"] = name_el.text
            if desc_el is not None and desc_el.text:
                props["description"] = desc_el.text[:200]

            # Point
            point = pm.find(f".//{_KML_NS}Point/{_KML_NS}coordinates")
            if point is not None and point.text:
                coords = _extract_coords(point.text)
                if coords:
                    return {"type": "Feature", "properties": props,
                            "geometry": {"type": "Point", "coordinates": coords[0]}}

            # LineString
            line = pm.find(f".//{_KML_NS}LineString/{_KML_NS}coordinates")
            if line is not None and line.text:
                coords = _extract_coords(line.text)
                if len(coords) >= 2:
                    return {"type": "Feature", "properties": props,
                            "geometry": {"type": "LineString", "coordinates": coords}}

            # Polygon
            poly = pm.find(f".//{_KML_NS}Polygon//{_KML_NS}outerBoundaryIs/{_KML_NS}LinearRing/{_KML_NS}coordinates")
            if poly is not None and poly.text:
                coords = _extract_coords(poly.text)
                if len(coords) >= 3:
                    return {"type": "Feature", "properties": props,
                            "geometry": {"type": "Polygon", "coordinates": [coords]}}

            return None

        def _walk_element(el, folder_name=None):
            """Recursively walk KML elements collecting placemarks by folder."""
            tag = el.tag.replace(_KML_NS, "")
            if tag == "Folder":
                name_el = el.find(f"{_KML_NS}name")
                folder_name = name_el.text if (name_el is not None and name_el.text) else folder_name

            if tag == "Placemark":
                feat = _placemark_to_feature(el)
                if feat is not None:
                    group_key = folder_name or kml_file.stem
                    groups.setdefault(group_key, []).append(feat)
                return

            for child in el:
                _walk_element(child, folder_name)

        _walk_element(root)

        # Convert groups to layers
        for group_name, features in groups.items():
            if len(features) > 10000:
                log_fn(f"[dashboard]   Skipping '{group_name}' ({len(features)} features, too dense)")
                continue

            color = _KML_PALETTE[color_idx % len(_KML_PALETTE)]
            color_idx += 1

            geojson = json.dumps({
                "type": "FeatureCollection",
                "features": features,
            })

            layers.append({
                "name": group_name,
                "geojson": geojson,
                "color": color,
                "style": {
                    "color": color,
                    "weight": 2,
                    "fillColor": color,
                    "fillOpacity": 0.15,
                    "radius": 5,
                },
            })

    log_fn(f"[dashboard] Parsed {len(layers)} KML layer(s)")
    return layers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_dashboard(cfg: dict, sim_results: dict, project_dir: str,
                       log_fn=print) -> str:
    """Generate a standalone HTML dashboard from simulation results.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration.
    sim_results : dict
        Output from ``run_simulations()``.
    project_dir : str
        Path to the AvaFrame project directory.
    log_fn : callable
        Logging callback.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    project_dir = Path(project_dir).resolve()
    downsample = cfg.get("dashboard", {}).get("downsample", 2)
    snow_source = cfg.get("snow", {}).get("source", "fixed")
    is_multi = snow_source in ("era5", "manual")
    runs = sim_results.get("runs", [])

    # ------------------------------------------------------------------
    # 1. Load DEM and compute hillshade
    # ------------------------------------------------------------------
    log_fn("[dashboard] Loading DEM ...")
    dem_files = list((project_dir / "Inputs").glob("*.tif"))
    if not dem_files:
        raise FileNotFoundError(f"No DEM .tif found in {project_dir / 'Inputs'}")

    dem_path = dem_files[0]
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float64)
        dem_transform = src.transform
        dem_crs = src.crs
        dem_bounds = src.bounds
        cellsize = abs(src.transform.a)

    # Reproject bounds to WGS84
    wgs_bounds = transform_bounds(dem_crs, "EPSG:4326",
                                  dem_bounds.left, dem_bounds.bottom,
                                  dem_bounds.right, dem_bounds.top)
    # wgs_bounds = (west, south, east, north)
    south, west, north, east = wgs_bounds[1], wgs_bounds[0], wgs_bounds[3], wgs_bounds[2]
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2
    overlay_bounds = [[south, west], [north, east]]

    log_fn("[dashboard] Computing hillshade ...")
    hillshade_img = _render_hillshade_png(dem_data, cellsize, downsample)
    hillshade_uri = _image_to_data_uri(hillshade_img)

    # ------------------------------------------------------------------
    # 2. Load release zones
    # ------------------------------------------------------------------
    log_fn("[dashboard] Loading release zones ...")
    rel_dir = project_dir / "Inputs" / "REL"
    release_geojson = _zones_to_geojson(str(rel_dir))

    # ------------------------------------------------------------------
    # 3. Process each run
    # ------------------------------------------------------------------
    overlay_data = {}
    run_labels = []

    is_flowpy = sim_results.get("model") == "com4FlowPy"

    for run in runs:
        label = run["label"]
        out_dir = Path(run["output_dir"])
        run_labels.append(label)

        log_fn(f"[dashboard] Processing run '{label}' (model: {'FlowPy' if is_flowpy else 'com1DFA'}) ...")

        run_overlays = {"hillshade": hillshade_uri}

        if is_flowpy:
            # --- com4FlowPy results ---
            # FlowPy outputs are directly in the result dir (not peakFiles/)
            # Try both the res_* dir and its parent
            search_dirs = [out_dir] + list(out_dir.glob("res_*"))
            tif_dir = out_dir
            for sd in search_dirs:
                if list(sd.glob("*_zdelta.tif")) or list(sd.glob("*_zDelta.tif")):
                    tif_dir = sd
                    break

            # zDelta (energy line height) -> runout extent
            zdelta, _, _, _ = _load_peak_envelope(str(tif_dir), "*_zdelta.tif")
            if zdelta is None:
                zdelta, _, _, _ = _load_peak_envelope(str(tif_dir), "*_zDelta.tif")
            if zdelta is not None:
                zdelta[zdelta <= 0] = np.nan
                vmax = float(np.nanpercentile(zdelta[zdelta > 0], 98)) if np.any(zdelta > 0) else 100
                img = _render_colormap_png(zdelta, "YlOrRd", 0, vmax, downsample)
                run_overlays["zdelta"] = _image_to_data_uri(img)
                log_fn(f"[dashboard]   zDelta: max={np.nanmax(zdelta):.1f}m, vmax={vmax:.1f}m")

            # cellCounts -> path density
            cells, _, _, _ = _load_peak_envelope(str(tif_dir), "*_cellCounts.tif")
            if cells is not None:
                cells[cells <= 0] = np.nan
                vmax = float(np.nanpercentile(cells[cells > 0], 98)) if np.any(cells > 0) else 10
                img = _render_colormap_png(cells, "hot_r", 0, vmax, downsample)
                run_overlays["cellcounts"] = _image_to_data_uri(img)
                log_fn(f"[dashboard]   cellCounts: max={np.nanmax(cells):.0f}")

            # travelAngleMax -> travel angle
            ta, _, _, _ = _load_peak_envelope(str(tif_dir), "*_fpTravelAngleMax.tif")
            if ta is not None:
                ta[ta <= 0] = np.nan
                img = _render_colormap_png(ta, "RdYlGn", 15, 45, downsample)
                run_overlays["travelangle"] = _image_to_data_uri(img)
                log_fn(f"[dashboard]   travelAngle: range={np.nanmin(ta):.1f}-{np.nanmax(ta):.1f} deg")

            # travelLengthMax
            tl, _, _, _ = _load_peak_envelope(str(tif_dir), "*_travelLengthMax.tif")
            if tl is not None:
                tl[tl <= 0] = np.nan
                vmax = float(np.nanpercentile(tl[tl > 0], 98)) if np.any(tl > 0) else 1000
                img = _render_colormap_png(tl, "viridis", 0, vmax, downsample)
                run_overlays["travellength"] = _image_to_data_uri(img)
                log_fn(f"[dashboard]   travelLength: max={np.nanmax(tl):.0f}m")

            # Estimated velocity
            vel, _, _, _ = _load_peak_envelope(str(tif_dir), "*_velocity.tif")
            if vel is not None:
                vel[vel <= 0] = np.nan
                vmax = float(np.nanpercentile(vel[vel > 0], 98)) if np.any(vel > 0) else 60
                img = _render_colormap_png(vel, "plasma", 0, vmax, downsample)
                run_overlays["velocity"] = _image_to_data_uri(img)
                log_fn(f"[dashboard]   velocity: max={np.nanmax(vel):.1f} m/s")

            # Estimated pressure
            pres, _, _, _ = _load_peak_envelope(str(tif_dir), "*_pressure.tif")
            if pres is not None:
                pres[pres <= 0] = np.nan
                img = _render_colormap_png(pres, "YlOrRd", 0, 100, downsample)
                run_overlays["pressure"] = _image_to_data_uri(img)
                log_fn(f"[dashboard]   pressure: max={np.nanmax(pres):.1f} kPa")

            # Hazard zones from estimated pressure (kPa -> Pa for _render_hazard_png)
            if pres is not None:
                pres_pa = np.nan_to_num(pres, nan=0) * 1000.0  # kPa -> Pa
                hazard_img = _render_hazard_png(pres_pa, downsample)
                run_overlays["hazard"] = _image_to_data_uri(hazard_img)
                log_fn(f"[dashboard]   hazard zones rendered")

        else:
            # --- com1DFA results ---
            peak_dir = out_dir / "peakFiles"

            # Peak pressure (ppr) -> hazard zones
            ppr_data, _, _, _ = _load_peak_envelope(str(peak_dir), "*_ppr.tif")
            if ppr_data is not None:
                hazard_img = _render_hazard_png(ppr_data, downsample)
                run_overlays["hazard"] = _image_to_data_uri(hazard_img)

                pressure_img = _render_colormap_png(
                    ppr_data / 1000.0, "YlOrRd", 0, 100, downsample)
                run_overlays["pressure"] = _image_to_data_uri(pressure_img)

            # Peak flow thickness (pft)
            pft_data, _, _, _ = _load_peak_envelope(str(peak_dir), "*_pft.tif")
            if pft_data is not None:
                thickness_img = _render_colormap_png(pft_data, "Blues", 0, 5, downsample)
                run_overlays["thickness"] = _image_to_data_uri(thickness_img)

            # Peak velocity (pfv)
            pfv_data, _, _, _ = _load_peak_envelope(str(peak_dir), "*_pfv.tif")
            if pfv_data is not None:
                velocity_img = _render_colormap_png(pfv_data, "plasma", 0, 60, downsample)
                run_overlays["velocity"] = _image_to_data_uri(velocity_img)

        overlay_data[label] = run_overlays

    # ------------------------------------------------------------------
    # 4. Build return-period selector HTML (multi-run only)
    # ------------------------------------------------------------------
    if is_multi and len(run_labels) > 1:
        btns = []
        for i, lbl in enumerate(run_labels):
            active = ' class="rp-btn active"' if i == 0 else ' class="rp-btn"'
            btns.append(f'<button{active} data-run="{lbl}">{lbl}</button>')
        rp_html = (
            '<div class="sidebar-section">'
            '<h3>Return Period</h3>'
            '<div class="rp-selector">' + "".join(btns) + '</div>'
            '</div>'
        )
    else:
        rp_html = ""

    # ------------------------------------------------------------------
    # 4b. Load KML overlays
    # ------------------------------------------------------------------
    kml_dir = project_dir / "Inputs" / "KML"
    kml_layers = _parse_kml_to_geojson_layers(str(kml_dir), log_fn=log_fn)

    # Build KML sidebar HTML and JS data
    if kml_layers:
        kml_sidebar_parts = [
            '<div class="sidebar-section">',
            '<h3>Custom Layers</h3>',
        ]
        for i, layer in enumerate(kml_layers):
            lid = f"cb-kml-{i}"
            kml_sidebar_parts.append(
                f'<div class="layer-option">'
                f'<input type="checkbox" id="{lid}" checked data-kml-idx="{i}">'
                f'<label for="{lid}">'
                f'<span class="legend-swatch" style="background:{layer["color"]};display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
                f'{layer["name"]}</label>'
                f'</div>'
            )
        kml_sidebar_parts.append('</div>')
        kml_layers_sidebar = "\n    ".join(kml_sidebar_parts)
        kml_layers_js_data = json.dumps(kml_layers)
    else:
        kml_layers_sidebar = ""
        kml_layers_js_data = "[]"

    # ------------------------------------------------------------------
    # 5. Title / subtitle
    # ------------------------------------------------------------------
    dash_cfg = cfg.get("dashboard", {})
    title = dash_cfg.get("title") or cfg.get("project", {}).get("name", "AvaFrame Results")
    if is_flowpy:
        subtitle = "com4FlowPy energy-line propagation"
    else:
        subtitle = "com1DFA dense-flow simulation"
        if is_multi:
            subtitle += f" | {len(run_labels)} return period(s)"

    # ------------------------------------------------------------------
    # 5b. Build model-specific sidebar and legend HTML
    # ------------------------------------------------------------------
    if is_flowpy:
        overlays_sidebar_html = """
    <div class="sidebar-section">
        <h3>Overlays</h3>
        <div class="layer-option"><input type="checkbox" id="cb-hillshade" checked><label for="cb-hillshade">Hillshade</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-hazard" checked><label for="cb-hazard">Hazard Zones (estimated)</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-pressure"><label for="cb-pressure">Est. Pressure (kPa)</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-velocity"><label for="cb-velocity">Est. Velocity (m/s)</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-zdelta"><label for="cb-zdelta">Energy Line Height</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-cellcounts"><label for="cb-cellcounts">Path Density</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-travelangle"><label for="cb-travelangle">Travel Angle</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-travellength"><label for="cb-travellength">Travel Length</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-release" checked><label for="cb-release">Release zones</label></div>
    </div>"""
        legend_html = """
    <div class="sidebar-section">
        <h3>Legend</h3>
        <div class="legend-block" id="legend-hazard">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Estimated Hazard Zones</div>
            <div class="legend-row"><span class="legend-swatch" style="background:rgba(220,40,40,0.8)"></span> Red zone (&gt; 30 kPa)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:rgba(50,100,200,0.7)"></span> Blue zone (3 &ndash; 30 kPa)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:rgba(240,200,40,0.6)"></span> Yellow zone (1 &ndash; 3 kPa)</div>
            <div style="font-size:10px; color:#888; margin-top:4px; font-style:italic;">Estimated from energy line: p = &rho;gz&Delta;</div>
        </div>
        <div class="legend-block" id="legend-pressure" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Est. Pressure (kPa)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#ffffcc,#fd8d3c,#bd0026); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>50</span><span>100</span></div>
        </div>
        <div class="legend-block" id="legend-velocity" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Est. Velocity (m/s)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#0d0887,#9c179e,#ed7953,#f0f921); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>30</span><span>60</span></div>
        </div>
        <div class="legend-block" id="legend-zdelta" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Energy Line Height (m)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#ffffcc,#fd8d3c,#bd0026); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>low</span><span>high</span></div>
        </div>
        <div class="legend-block" id="legend-cellcounts" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Path Density</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#fff5f0,#fc4e2a,#67000d); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>few</span><span>many</span></div>
        </div>
        <div class="legend-block" id="legend-travelangle" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Travel Angle (deg)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#1a9850,#ffffbf,#d73027); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>15</span><span>30</span><span>45</span></div>
        </div>
        <div class="legend-block" id="legend-travellength" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Travel Length (m)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#440154,#31688e,#35b779,#fde725); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>mid</span><span>max</span></div>
        </div>
        <div class="legend-block">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Release Zones</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#e74c3c"></span> Large (&gt; 50k m&sup2;)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#e67e22"></span> Medium (&gt; 10k m&sup2;)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#f1c40f"></span> Small (&le; 10k m&sup2;)</div>
        </div>
    </div>"""
    else:
        overlays_sidebar_html = """
    <div class="sidebar-section">
        <h3>Overlays</h3>
        <div class="layer-option"><input type="checkbox" id="cb-hillshade" checked><label for="cb-hillshade">Hillshade</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-hazard" checked><label for="cb-hazard">Hazard zones</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-pressure"><label for="cb-pressure">Peak pressure</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-thickness"><label for="cb-thickness">Peak flow thickness</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-velocity"><label for="cb-velocity">Peak velocity</label></div>
        <div class="layer-option"><input type="checkbox" id="cb-release" checked><label for="cb-release">Release zones</label></div>
    </div>"""
        legend_html = """
    <div class="sidebar-section">
        <h3>Legend</h3>
        <div class="legend-block" id="legend-hazard">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Hazard Zones</div>
            <div class="legend-row"><span class="legend-swatch" style="background:rgba(220,40,40,0.8)"></span> Red zone (&gt; 30 kPa)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:rgba(50,100,200,0.7)"></span> Blue zone (3 &ndash; 30 kPa)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:rgba(240,200,40,0.6)"></span> Yellow zone (1 &ndash; 3 kPa)</div>
        </div>
        <div class="legend-block" id="legend-pressure" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Peak Pressure (kPa)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#ffffcc,#fd8d3c,#bd0026); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>50</span><span>100</span></div>
        </div>
        <div class="legend-block" id="legend-thickness" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Peak Flow Thickness (m)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#f7fbff,#6baed6,#08306b); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>2.5</span><span>5</span></div>
        </div>
        <div class="legend-block" id="legend-velocity" style="display:none;">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Peak Velocity (m/s)</div>
            <div style="height:14px; border-radius:2px; background:linear-gradient(90deg,#0d0887,#9c179e,#ed7953,#f0f921); border:1px solid rgba(255,255,255,0.2);"></div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;"><span>0</span><span>30</span><span>60</span></div>
        </div>
        <div class="legend-block">
            <div style="font-size:12px; font-weight:600; margin-bottom:4px;">Release Zones</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#e74c3c"></span> Large (&gt; 50k m&sup2;)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#e67e22"></span> Medium (&gt; 10k m&sup2;)</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#f1c40f"></span> Small (&le; 10k m&sup2;)</div>
        </div>
    </div>"""

    # ------------------------------------------------------------------
    # 6. Render HTML
    # ------------------------------------------------------------------
    log_fn("[dashboard] Rendering HTML ...")

    html = _HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        center_lat=center_lat,
        center_lon=center_lon,
        overlay_bounds_js=json.dumps(overlay_bounds),
        release_geojson_js=release_geojson,
        overlay_data_js=json.dumps(overlay_data),
        is_multi_run_js="true" if is_multi else "false",
        run_labels_js=json.dumps(run_labels),
        rp_selector_html=rp_html,
        overlays_sidebar_html=overlays_sidebar_html,
        legend_html=legend_html,
        kml_layers_sidebar=kml_layers_sidebar,
        kml_layers_js=kml_layers_js_data,
    )

    out_path = project_dir / "Outputs" / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    log_fn(f"[dashboard] Dashboard saved to {out_path}")
    return str(out_path)

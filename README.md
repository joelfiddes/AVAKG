# AVAKG — AvaFrame Web Interface

Web-based interface for running [AvaFrame](https://docs.avaframe.org) avalanche simulations. Provides a unified pipeline from DEM acquisition through to interactive hazard dashboards.

**Live**: https://apps.mountainfutures.ch/avaframe/

## Features

- **Domain**: Define study area by center+buffer, bounding box, shapefile, or DEM extent
- **DEM**: Copernicus GLO-30 auto-download or upload your own
- **Snow climatology**: ERA5-based fracture depth estimation (Swiss procedure, Salm et al. 1990), manual d0 values, or fixed thickness
- **Release areas**: Automatic slope-based PRA delineation (28-60 deg, SLF/LSHIM methodology)
- **Simulation**: AvaFrame com1DFA dense flow avalanche (SPH particle method)
- **Dashboard**: Interactive Leaflet.js hazard map with Swiss zone classification (red/blue/yellow)
- **KML overlays**: Upload infrastructure plans, paths, etc. to overlay on results
- **Demo**: Built-in demo with a pre-cropped DEM from KG mining site (Kyrgyzstan)

## Quick Start (local)

```bash
conda activate avaframe     # needs avaframe 1.13+
pip install flask pyyaml
python app.py
# Open http://localhost:5050
```

Click **Load Demo** in the Pipeline tab for a quick test run.

## Deploy on server

Deployed on myserver behind Caddy reverse proxy:

```bash
# On server:
git clone https://github.com/joelfiddes/AVAKG.git ~/src/AVAKG
cd ~/src/AVAKG

# Create venv and install deps
python3 -m venv ~/avaframe-webui/venv
source ~/avaframe-webui/venv/bin/activate
pip install flask pyyaml avaframe rasterio geopandas fiona shapely scipy pillow matplotlib pandas requests pyproj dem-stitcher

# Copy app files
cp -r * ~/avaframe-webui/

# Create systemd user service
# See deploy.sh for details

# Caddy config (inside apps.mountainfutures.ch block):
#   handle /avaframe { redir /avaframe/ permanent }
#   handle_path /avaframe* { reverse_proxy localhost:5050 }
```

## CLI Pipeline

```bash
python pipeline.py --template > config.yaml    # generate config template
python pipeline.py config.yaml                  # run full pipeline (6 steps)
python pipeline.py config.yaml --from-step 5    # resume from step 5
python pipeline.py config.yaml --only-step 2    # run only step 2
```

### Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | Domain | Resolve study area (bbox/shapefile/center+buffer/DEM extent) |
| 2 | DEM | Acquire DEM (Copernicus GLO-30 or local file) |
| 3 | Snow | Compute release thickness (ERA5 H72 EVA / manual / fixed) |
| 4 | Release | Generate release areas from slope analysis |
| 5 | Simulation | Run com1DFA dense flow simulation(s) |
| 6 | Dashboard | Generate interactive Leaflet.js hazard map |

### Configuration Options

**Snow sources:**
- `fixed` — Single thickness value (e.g. 1.5m)
- `manual` — Per-return-period d0 values (e.g. 30yr: 0.68, 100yr: 0.95, 300yr: 1.22)
- `era5` — Automatic: fetches ERA5 snowfall (1950-present), computes H72, fits Gumbel distribution, applies Swiss corrections (elevation, slope, wind)

**DEM sources:**
- `copernicus` — Auto-downloads Copernicus GLO-30 (~30m) via dem-stitcher
- `local` — Your own DEM file (any resolution, auto-resampled)

**Friction models:** samosATAuto (recommended), samosAT, samosATSmall, samosATMedium, Voellmy, Coulomb

## Architecture

```
app.py                  # Flask web server
pipeline.py             # CLI entry point with checkpoint system
pipeline_steps/
  config_schema.py      # YAML config loading, validation, domain resolution
  dem.py                # DEM acquisition (local/Copernicus)
  snow.py               # Snow climatology (ERA5/manual/fixed)
  release.py            # Slope-based release area delineation
  simulation.py         # com1DFA runner
  dashboard.py          # Leaflet HTML dashboard generator
templates/index.html    # Web UI (dark theme, form-based config)
demo/                   # Built-in demo dataset (KG mining site)
```

## References

- AvaFrame: https://docs.avaframe.org
- Salm, B., Burkard, A., & Gubler, H. (1990). Berechnung von Fliesslawinen. SLF Mitteilung 47.
- Buehler, Y., et al. (2018). Automated snow avalanche release area delineation. NHESS 18, 3235-3251.

# AVAKG — AvaFrame Web Interface

Web-based interface for running AvaFrame avalanche simulations. Provides a unified pipeline from DEM acquisition through to interactive hazard dashboards.

## Features

- **DEM**: Copernicus GLO-30 auto-download or upload your own
- **Snow climatology**: ERA5-based fracture depth estimation (Swiss procedure), manual values, or fixed thickness
- **Release areas**: Automatic slope-based delineation (28-60 deg)
- **Simulation**: AvaFrame com1DFA dense flow avalanche (SPH)
- **Dashboard**: Interactive Leaflet.js hazard map with Swiss zone classification

## Quick Start (local)

```bash
conda activate avaframe
pip install flask pyyaml
python app.py
# Open http://localhost:5050
```

## Deploy on server

```bash
bash deploy.sh
# Serves at https://apps.mountainfutures.ch/avaframe/
```

## CLI Pipeline

```bash
python pipeline.py --template > config.yaml
# Edit config.yaml
python pipeline.py config.yaml
```

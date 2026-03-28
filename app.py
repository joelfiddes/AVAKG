"""
AvaFrame Web UI - A user-friendly interface for running AvaFrame simulations.

Usage:
    conda activate avaframe
    python app.py

Then open http://localhost:5050 in your browser.
"""

import configparser
import glob
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        render_template,
        request,
        send_file,
    )
except ImportError:
    print("Flask not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import (
        Flask,
        Response,
        jsonify,
        render_template,
        request,
        send_file,
    )

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

# Data directory for projects (configurable via env var)
DATA_DIR = os.environ.get("AVAFRAME_DATA_DIR", os.path.expanduser("~/avaframe_projects"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AVAFRAME_PKG = None

def _find_avaframe():
    global AVAFRAME_PKG
    if AVAFRAME_PKG is None:
        import avaframe
        AVAFRAME_PKG = pathlib.Path(avaframe.__path__[0])
    return AVAFRAME_PKG

# ---------------------------------------------------------------------------
# INI parsing with comments as docs
# ---------------------------------------------------------------------------

def parse_ini_with_docs(filepath):
    """Parse an .ini file and extract parameters with their preceding comments as documentation."""
    sections = {}
    current_section = None
    comment_buffer = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.strip()

            # Skip top-level header comments (### lines)
            if stripped.startswith("###"):
                continue

            # Section header
            m = re.match(r"^\[(.+)\]$", stripped)
            if m:
                current_section = m.group(1)
                sections[current_section] = {"_doc": "\n".join(comment_buffer), "params": []}
                comment_buffer = []
                continue

            # Comment line
            if stripped.startswith("#") or stripped.startswith("##"):
                comment_buffer.append(stripped.lstrip("#").strip())
                continue

            # Empty line — keep accumulating comments only if we're in a section
            if not stripped:
                if current_section is None:
                    comment_buffer = []
                continue

            # Parameter line
            if "=" in stripped and current_section:
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip()
                doc = "\n".join(comment_buffer)
                comment_buffer = []
                sections[current_section]["params"].append({
                    "key": key,
                    "value": value,
                    "doc": doc,
                })
            else:
                comment_buffer = []

    return sections


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULES = {
    "avaframe": {
        "name": "General Settings",
        "ini": "avaframeCfg.ini",
        "description": "Main AvaFrame configuration: project directory, CPU settings, plotting flags.",
    },
    "com1DFA": {
        "name": "com1DFA - Dense Flow Avalanche",
        "ini": "com1DFA/com1DFACfg.ini",
        "description": "Particle-based SPH simulation of dense flow avalanches. The primary simulation engine with multiple friction models (samosAT, Voellmy, Coulomb, wetsnow), entrainment, dam interactions, and more.",
    },
    "com2AB": {
        "name": "com2AB - Alpha-Beta Model",
        "ini": "com2AB/com2ABCfg.ini",
        "description": "Statistical rapid-assessment model for avalanche runout estimation. Uses regression parameters to predict runout distance from avalanche path profile and split point.",
    },
    "com3Hybrid": {
        "name": "com3Hybrid - Hybrid Model",
        "ini": "com3Hybrid/com3HybridCfg.ini",
        "description": "Combines com1DFA (dense flow simulation) with com2AB (alpha-beta model) through iterative refinement of friction parameters.",
    },
    "com4FlowPy": {
        "name": "com4FlowPy - Flow-Py Model",
        "ini": "com4FlowPy/com4FlowPyCfg.ini",
        "description": "Flow path propagation model using angle-of-reach and spreading coefficients. Suitable for rapid hazard mapping over large areas. Supports forest interaction, tiling for large DEMs, and multiprocessing.",
    },
    "com5SnowSlide": {
        "name": "com5SnowSlide - Snow Slide",
        "ini": "com5SnowSlide/com5SnowSlideCfg.ini",
        "description": "Small snow slide module built on top of com1DFA with elastic cohesion forces between particles to represent slab behavior.",
    },
    "com6RockAvalanche": {
        "name": "com6RockAvalanche - Rock Avalanche",
        "ini": "com6RockAvalanche/com6RockAvalancheCfg.ini",
        "description": "Rock avalanche simulation using com1DFA engine with rock-specific parameters (high density, Voellmy friction). EXPERIMENTAL.",
    },
    "ana3AIMEC": {
        "name": "ana3AIMEC - AIMEC Analysis",
        "ini": "ana3AIMEC/ana3AIMECCfg.ini",
        "description": "Automated Indicator-based Model Evaluation and Comparison. Transforms simulation results to an (s,l) coordinate system along the avalanche path for comparison.",
    },
    "ana4Stats": {
        "name": "ana4Stats - Probabilistic Analysis",
        "ini": "ana4Stats/probAnaCfg.ini",
        "description": "Probabilistic analysis with parameter variation (Latin Hypercube, Morris method). Generates probability maps from multiple simulation runs.",
    },
    "generateTopo": {
        "name": "Generate Topography",
        "ini": "in3Utils/generateTopoCfg.ini",
        "description": "Generate synthetic test topographies (flat plane, inclined plane, parabolic slope, hockey stick, bowl, helix, pyramid).",
    },
    "plotUtils": {
        "name": "Plot Settings",
        "ini": "out3Plot/plotUtilsCfg.ini",
        "description": "Global plot appearance settings: figure size, font, colormap levels, units, and contour line levels for all result parameters.",
    },
}

# Workflows that can be run
WORKFLOWS = {
    "com1DFA": {
        "name": "Run com1DFA (Dense Flow Avalanche)",
        "description": "Run the dense flow avalanche simulation using SPH particle method.",
        "required_inputs": ["DEM", "REL"],
        "optional_inputs": ["ENT", "RES", "SECREL", "DAM"],
        "script": "run_com1dfa",
    },
    "com2AB": {
        "name": "Run com2AB (Alpha-Beta)",
        "description": "Run the alpha-beta statistical runout model. Requires an avalanche path (LINES) and split point (POINTS).",
        "required_inputs": ["DEM", "LINES", "POINTS"],
        "optional_inputs": [],
        "script": "run_com2ab",
    },
    "com4FlowPy": {
        "name": "Run com4FlowPy (Flow-Py)",
        "description": "Run the Flow-Py flow path propagation model for rapid hazard mapping.",
        "required_inputs": ["DEM", "REL"],
        "optional_inputs": ["INFRA", "FOREST"],
        "script": "run_com4flowpy",
    },
    "operational": {
        "name": "Operational Run (com1DFA + com2AB)",
        "description": "Full operational workflow: runs com1DFA simulation, then com2AB analysis and report generation.",
        "required_inputs": ["DEM", "REL"],
        "optional_inputs": ["ENT", "RES", "SECREL", "LINES", "POINTS"],
        "script": "run_operational",
    },
    "probAna": {
        "name": "Probabilistic Analysis",
        "description": "Run multiple simulations with parameter variations and generate probability maps.",
        "required_inputs": ["DEM", "REL"],
        "optional_inputs": ["ENT"],
        "script": "run_prob_ana",
    },
}

# ---------------------------------------------------------------------------
# Running simulations state
# ---------------------------------------------------------------------------
_runs = {}  # run_id -> {"status": str, "log_lines": list, "process": subprocess.Popen}

def _generate_run_script(workflow_id, ava_dir, config_overrides):
    """Generate a temporary Python script to run the requested workflow."""
    ava_dir_escaped = ava_dir.replace("\\", "\\\\").replace("'", "\\'")
    overrides_json = json.dumps(config_overrides)

    script = f'''#!/usr/bin/env python3
"""Auto-generated AvaFrame run script."""
import configparser
import json
import os
import pathlib
import sys

# Make sure we can import avaframe
avalancheDir = r"{ava_dir}"

# Apply config overrides
overrides = json.loads('{overrides_json}')

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import initializeProject as iP

# Set avalanche dir in the main config
cfgMain = cfgUtils.getGeneralConfig()
cfgMain["MAIN"]["avalancheDir"] = avalancheDir

# Start logging
log = logUtils.initiateLogger(avalancheDir, "runLog")
log.info("AvaFrame WebUI run started")
log.info("Avalanche directory: %s", avalancheDir)
'''

    if workflow_id == "com1DFA":
        script += f'''
from avaframe.com1DFA import com1DFA
from avaframe.out1Peak import outPlotAllPeak as oP
from avaframe.log2Report import generateReport as gR

# Clean module files
iP.cleanModuleFiles(avalancheDir, com1DFA)

# Get module config
modCfg, modInfo = cfgUtils.getModuleConfig(com1DFA, toPrint=False)

# Apply overrides
for section_key, params in overrides.items():
    if section_key.startswith("com1DFA:"):
        section = section_key.split(":", 1)[1]
        if section in modCfg:
            for k, v in params.items():
                modCfg[section][k] = str(v)

# Run preprocessing
simDict, inputSimFiles, outDir = com1DFA.com1DFAPreprocess(
    cfgMain, "cfgFromObject", modCfg, com1DFA
)

# Run simulation
dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, "cfgFromObject", modCfg)
log.info("com1DFA simulation complete")

# Generate peak plots
cfgFlags = cfgMain["FLAGS"]
if cfgFlags.getboolean("savePlot"):
    log.info("Generating peak plots...")
    oP.plotAllPeakFields(avalancheDir, cfgMain, modCfg, "com1DFA")

if cfgFlags.getboolean("createReport"):
    log.info("Generating report...")
    gR.writeReport(avalancheDir, reportDictList, cfgFlags, "com1DFA")

log.info("Run complete!")
'''
    elif workflow_id == "com2AB":
        script += f'''
from avaframe.com2AB import com2AB

iP.cleanModuleFiles(avalancheDir, com2AB)
modCfg, modInfo = cfgUtils.getModuleConfig(com2AB, toPrint=False)

# Apply overrides
for section_key, params in overrides.items():
    if section_key.startswith("com2AB:"):
        section = section_key.split(":", 1)[1]
        if section in modCfg:
            for k, v in params.items():
                modCfg[section][k] = str(v)

resAB = com2AB.com2ABMain(cfgMain, modCfg)
log.info("com2AB analysis complete")
log.info("Run complete!")
'''
    elif workflow_id == "com4FlowPy":
        script += f'''
from avaframe.com4FlowPy import com4FlowPy

iP.cleanModuleFiles(avalancheDir, com4FlowPy)
modCfg, modInfo = cfgUtils.getModuleConfig(com4FlowPy, toPrint=False)

# Apply overrides
for section_key, params in overrides.items():
    if section_key.startswith("com4FlowPy:"):
        section = section_key.split(":", 1)[1]
        if section in modCfg:
            for k, v in params.items():
                modCfg[section][k] = str(v)

com4FlowPy.com4FlowPyMain(cfgMain, modCfg)
log.info("com4FlowPy simulation complete")
log.info("Run complete!")
'''
    elif workflow_id == "operational":
        script += f'''
from avaframe.com1DFA import com1DFA
from avaframe.com2AB import com2AB
from avaframe.out1Peak import outPlotAllPeak as oP
from avaframe.log2Report import generateReport as gR

# Clean and run com1DFA
iP.cleanModuleFiles(avalancheDir, com1DFA)
modCfg1, _ = cfgUtils.getModuleConfig(com1DFA, toPrint=False)

# Apply com1DFA overrides
for section_key, params in overrides.items():
    if section_key.startswith("com1DFA:"):
        section = section_key.split(":", 1)[1]
        if section in modCfg1:
            for k, v in params.items():
                modCfg1[section][k] = str(v)

dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, "cfgFromObject", modCfg1)
log.info("com1DFA complete")

cfgFlags = cfgMain["FLAGS"]
if cfgFlags.getboolean("savePlot"):
    oP.plotAllPeakFields(avalancheDir, cfgMain, modCfg1, "com1DFA")
if cfgFlags.getboolean("createReport"):
    gR.writeReport(avalancheDir, reportDictList, cfgFlags, "com1DFA")

# Run com2AB if path and split point exist
import pathlib
linesDir = pathlib.Path(avalancheDir) / "Inputs" / "LINES"
pointsDir = pathlib.Path(avalancheDir) / "Inputs" / "POINTS"
if linesDir.exists() and pointsDir.exists():
    lineFiles = list(linesDir.glob("*.shp"))
    pointFiles = list(pointsDir.glob("*.shp"))
    if lineFiles and pointFiles:
        log.info("Running com2AB...")
        iP.cleanModuleFiles(avalancheDir, com2AB)
        modCfg2, _ = cfgUtils.getModuleConfig(com2AB, toPrint=False)
        resAB = com2AB.com2ABMain(cfgMain, modCfg2)
        log.info("com2AB complete")
    else:
        log.info("Skipping com2AB: no LINES/POINTS shapefiles found")
else:
    log.info("Skipping com2AB: LINES or POINTS directory missing")

log.info("Operational run complete!")
'''
    elif workflow_id == "probAna":
        script += f'''
from avaframe.com1DFA import com1DFA
from avaframe.ana4Stats import probAna

iP.cleanModuleFiles(avalancheDir, com1DFA)

modCfg, _ = cfgUtils.getModuleConfig(probAna, toPrint=False)

# Apply overrides
for section_key, params in overrides.items():
    if section_key.startswith("ana4Stats:"):
        section = section_key.split(":", 1)[1]
        if section in modCfg:
            for k, v in params.items():
                modCfg[section][k] = str(v)

probAna.probAnalysis(avalancheDir, cfgMain, modCfg, com1DFA)
log.info("Probabilistic analysis complete!")
log.info("Run complete!")
'''

    return script


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/modules")
def api_modules():
    """Return list of configurable modules."""
    return jsonify(MODULES)


@app.route("/api/workflows")
def api_workflows():
    """Return list of runnable workflows."""
    return jsonify(WORKFLOWS)


@app.route("/api/config/<module_id>")
def api_config(module_id):
    """Return parsed config for a module with documentation."""
    if module_id not in MODULES:
        return jsonify({"error": "Unknown module"}), 404

    pkg = _find_avaframe()
    ini_path = pkg / MODULES[module_id]["ini"]
    if not ini_path.exists():
        return jsonify({"error": f"Config file not found: {ini_path}"}), 404

    sections = parse_ini_with_docs(ini_path)
    return jsonify({
        "module": MODULES[module_id],
        "sections": sections,
        "ini_path": str(ini_path),
    })


@app.route("/api/project/init", methods=["POST"])
def api_project_init():
    """Initialize an AvaFrame project directory structure."""
    data = request.json
    ava_dir = data.get("avalancheDir", "")
    if not ava_dir:
        return jsonify({"error": "No avalancheDir provided"}), 400

    ava_dir = os.path.expanduser(ava_dir)
    os.makedirs(ava_dir, exist_ok=True)

    # Create standard directory structure
    subdirs = [
        "Inputs", "Inputs/REL", "Inputs/RES", "Inputs/ENT",
        "Inputs/SECREL", "Inputs/POINTS", "Inputs/LINES",
        "Inputs/POLYGONS", "Inputs/RELTH", "Inputs/RASTERS",
        "Inputs/DAM",
        "Outputs", "Work",
    ]
    for d in subdirs:
        os.makedirs(os.path.join(ava_dir, d), exist_ok=True)

    return jsonify({"status": "ok", "avalancheDir": ava_dir, "created": subdirs})


@app.route("/api/project/info")
def api_project_info():
    """Get info about an existing project directory."""
    ava_dir = request.args.get("dir", "")
    if not ava_dir:
        return jsonify({"error": "No dir provided"}), 400

    ava_dir = os.path.expanduser(ava_dir)
    if not os.path.isdir(ava_dir):
        return jsonify({"error": "Directory does not exist"}), 404

    info = {"avalancheDir": ava_dir, "inputs": {}, "outputs": {}}

    # Scan Inputs
    inputs_dir = os.path.join(ava_dir, "Inputs")
    if os.path.isdir(inputs_dir):
        # DEM
        dems = glob.glob(os.path.join(inputs_dir, "*.tif")) + glob.glob(os.path.join(inputs_dir, "*.asc"))
        info["inputs"]["DEM"] = [os.path.basename(f) for f in dems]

        for subdir in ["REL", "ENT", "RES", "SECREL", "POINTS", "LINES", "POLYGONS", "RELTH", "RASTERS", "DAM"]:
            subpath = os.path.join(inputs_dir, subdir)
            if os.path.isdir(subpath):
                files = os.listdir(subpath)
                info["inputs"][subdir] = [f for f in files if not f.startswith(".")]

    # Scan Outputs
    outputs_dir = os.path.join(ava_dir, "Outputs")
    if os.path.isdir(outputs_dir):
        for subdir in os.listdir(outputs_dir):
            subpath = os.path.join(outputs_dir, subdir)
            if os.path.isdir(subpath):
                info["outputs"][subdir] = {}
                for root, dirs, files in os.walk(subpath):
                    rel = os.path.relpath(root, subpath)
                    visible = [f for f in files if not f.startswith(".")]
                    if visible:
                        info["outputs"][subdir][rel] = visible

    return jsonify(info)


@app.route("/api/project/local_config", methods=["GET"])
def api_get_local_config():
    """Get local config overrides from the project directory."""
    ava_dir = request.args.get("dir", "")
    module_id = request.args.get("module", "")

    if not ava_dir or not module_id:
        return jsonify({"error": "Missing dir or module"}), 400

    # Look for local_*Cfg.ini in the project dir
    local_cfg_name = f"local_{MODULES.get(module_id, {}).get('ini', '').split('/')[-1]}"
    local_path = os.path.join(os.path.expanduser(ava_dir), local_cfg_name)

    if os.path.exists(local_path):
        sections = parse_ini_with_docs(local_path)
        return jsonify({"exists": True, "path": local_path, "sections": sections})
    else:
        return jsonify({"exists": False, "path": local_path})


@app.route("/api/project/save_local_config", methods=["POST"])
def api_save_local_config():
    """Save local config overrides to the project directory."""
    data = request.json
    ava_dir = data.get("dir", "")
    module_id = data.get("module", "")
    overrides = data.get("overrides", {})  # {section: {key: value}}

    if not ava_dir or not module_id:
        return jsonify({"error": "Missing dir or module"}), 400

    ini_name = MODULES.get(module_id, {}).get("ini", "").split("/")[-1]
    local_cfg_name = f"local_{ini_name}"
    local_path = os.path.join(os.path.expanduser(ava_dir), local_cfg_name)

    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve case

    for section, params in overrides.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        for key, value in params.items():
            cfg.set(section, key, str(value))

    with open(local_path, "w") as f:
        f.write(f"### Local override for {ini_name}\n")
        f.write(f"# Generated by AvaFrame WebUI on {datetime.now().isoformat()}\n\n")
        cfg.write(f)

    return jsonify({"status": "ok", "path": local_path})


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a simulation run."""
    data = request.json
    workflow_id = data.get("workflow", "")
    ava_dir = data.get("avalancheDir", "")
    config_overrides = data.get("configOverrides", {})

    if workflow_id not in WORKFLOWS:
        return jsonify({"error": "Unknown workflow"}), 400
    if not ava_dir:
        return jsonify({"error": "No avalancheDir"}), 400

    ava_dir = os.path.expanduser(ava_dir)

    run_id = str(uuid.uuid4())[:8]
    script_content = _generate_run_script(workflow_id, ava_dir, config_overrides)

    # Write temp script
    script_path = os.path.join(ava_dir, f".webui_run_{run_id}.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    _runs[run_id] = {
        "status": "running",
        "log_lines": [],
        "workflow": workflow_id,
        "ava_dir": ava_dir,
        "started": datetime.now().isoformat(),
        "script_path": script_path,
    }

    def run_in_thread():
        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=ava_dir,
            )
            _runs[run_id]["process"] = process

            for line in process.stdout:
                _runs[run_id]["log_lines"].append(line.rstrip("\n"))

            process.wait()
            _runs[run_id]["status"] = "complete" if process.returncode == 0 else "error"
            _runs[run_id]["returncode"] = process.returncode
        except Exception as e:
            _runs[run_id]["status"] = "error"
            _runs[run_id]["log_lines"].append(f"ERROR: {e}")
        finally:
            # Clean up script
            try:
                os.remove(script_path)
            except OSError:
                pass

    t = threading.Thread(target=run_in_thread, daemon=True)
    t.start()

    return jsonify({"run_id": run_id, "status": "running"})


@app.route("/api/run/<run_id>/status")
def api_run_status(run_id):
    """Get current status and logs for a run."""
    if run_id not in _runs:
        return jsonify({"error": "Unknown run"}), 404

    run = _runs[run_id]
    offset = int(request.args.get("offset", 0))
    return jsonify({
        "status": run["status"],
        "log_lines": run["log_lines"][offset:],
        "total_lines": len(run["log_lines"]),
    })


@app.route("/api/run/<run_id>/stream")
def api_run_stream(run_id):
    """SSE stream of run logs."""
    if run_id not in _runs:
        return jsonify({"error": "Unknown run"}), 404

    def generate():
        sent = 0
        while True:
            run = _runs.get(run_id)
            if not run:
                break
            lines = run["log_lines"]
            while sent < len(lines):
                yield f"data: {json.dumps({'line': lines[sent]})}\n\n"
                sent += 1
            if run["status"] != "running":
                yield f"data: {json.dumps({'status': run['status']})}\n\n"
                break
            time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/results/files")
def api_results_files():
    """List result files in an output directory."""
    ava_dir = request.args.get("dir", "")
    if not ava_dir:
        return jsonify({"error": "No dir"}), 400

    ava_dir = os.path.expanduser(ava_dir)
    outputs_dir = os.path.join(ava_dir, "Outputs")
    results = {}

    if os.path.isdir(outputs_dir):
        for root, dirs, files in os.walk(outputs_dir):
            rel = os.path.relpath(root, outputs_dir)
            visible = sorted([f for f in files if not f.startswith(".")])
            if visible:
                results[rel] = visible

    return jsonify(results)


@app.route("/api/results/file")
def api_results_file():
    """Serve a specific result file."""
    filepath = request.args.get("path", "")
    if not filepath or ".." in filepath:
        return jsonify({"error": "Invalid path"}), 400

    filepath = os.path.expanduser(filepath)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath)


# ---------------------------------------------------------------------------
# Pipeline routes
# ---------------------------------------------------------------------------

_pipeline_runs = {}

@app.route("/api/pipeline/template")
def api_pipeline_template():
    """Return YAML config template."""
    from pipeline_steps.config_schema import get_config_template
    return Response(get_config_template(), mimetype="text/plain")


@app.route("/api/pipeline/validate", methods=["POST"])
def api_pipeline_validate():
    """Validate a pipeline config."""
    import yaml
    from pipeline_steps.config_schema import load_config, validate_config
    data = request.json
    config_text = data.get("config", "")
    try:
        cfg = yaml.safe_load(config_text)
        if cfg is None:
            return jsonify({"errors": ["Empty config"]})
        # Merge with defaults
        from pipeline_steps.config_schema import DEFAULT_CONFIG
        import copy
        merged = copy.deepcopy(DEFAULT_CONFIG)
        def _merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    _merge(base[k], v)
                else:
                    base[k] = v
        _merge(merged, cfg)
        errors = validate_config(merged)
        return jsonify({"errors": errors, "config": merged})
    except yaml.YAMLError as e:
        return jsonify({"errors": [f"YAML parse error: {e}"]})


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """Start a pipeline run."""
    import yaml
    import tempfile
    data = request.json
    config_text = data.get("config", "")

    if not config_text:
        return jsonify({"error": "No config provided"}), 400

    run_id = str(uuid.uuid4())[:8]
    _pipeline_runs[run_id] = {
        "status": "running",
        "log_lines": [],
        "started": datetime.now().isoformat(),
        "step": 0,
    }

    # Write config to temp file
    config_path = os.path.join(tempfile.gettempdir(), f"avaframe_pipeline_{run_id}.yaml")
    with open(config_path, "w") as f:
        f.write(config_text)

    def run_in_thread():
        def log_fn(msg):
            _pipeline_runs[run_id]["log_lines"].append(str(msg))

        try:
            # Run pipeline as subprocess so it gets its own process
            process = subprocess.Popen(
                [sys.executable, "pipeline.py", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

            for line in process.stdout:
                _pipeline_runs[run_id]["log_lines"].append(line.rstrip("\n"))
                # Track current step
                if line.strip().startswith("[Step"):
                    try:
                        step_num = int(line.strip().split("/")[0].replace("[Step ", ""))
                        _pipeline_runs[run_id]["step"] = step_num
                    except (ValueError, IndexError):
                        pass

            process.wait()
            _pipeline_runs[run_id]["status"] = "complete" if process.returncode == 0 else "error"
            _pipeline_runs[run_id]["returncode"] = process.returncode
        except Exception as e:
            _pipeline_runs[run_id]["status"] = "error"
            _pipeline_runs[run_id]["log_lines"].append(f"ERROR: {e}")
        finally:
            try:
                os.remove(config_path)
            except OSError:
                pass

    t = threading.Thread(target=run_in_thread, daemon=True)
    t.start()

    return jsonify({"run_id": run_id, "status": "running"})


@app.route("/api/pipeline/status/<run_id>")
def api_pipeline_status(run_id):
    """Get pipeline run status and logs."""
    if run_id not in _pipeline_runs:
        return jsonify({"error": "Unknown run"}), 404

    run = _pipeline_runs[run_id]
    offset = int(request.args.get("offset", 0))
    return jsonify({
        "status": run["status"],
        "step": run.get("step", 0),
        "log_lines": run["log_lines"][offset:],
        "total_lines": len(run["log_lines"]),
    })


@app.route("/api/pipeline/demo")
def api_pipeline_demo():
    """Set up demo project with pre-baked inputs and return config.

    The demo skips steps 1-4 entirely. We copy the DEM directly into
    Inputs/ as dem.tif and the release shapefile into Inputs/REL/.
    The pipeline runs only step 5 (simulation) and step 6 (dashboard).
    """
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
    if not os.path.isdir(demo_dir):
        return jsonify({"error": "Demo data not found"}), 404

    import shutil
    import json as json_mod

    demo_project = os.path.join(DATA_DIR, "demo_kg_mining")

    # Always start fresh for demo
    if os.path.exists(demo_project):
        shutil.rmtree(demo_project)

    for d in ["Inputs/REL", "Outputs", "Work", ".pipeline"]:
        os.makedirs(os.path.join(demo_project, d), exist_ok=True)

    # Copy DEM as dem.tif (the only DEM avaframe should see)
    shutil.copy2(
        os.path.join(demo_dir, "Inputs", "demo_dem.tif"),
        os.path.join(demo_project, "Inputs", "dem.tif"),
    )

    # Copy release shapefile
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        src = os.path.join(demo_dir, "Inputs", "REL", f"rel_demo{ext}")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(demo_project, "Inputs", "REL", f"rel_demo{ext}"))

    dem_path = os.path.join(demo_project, "Inputs", "dem.tif")

    # Pre-create checkpoints for steps 1-4 so pipeline skips to step 5
    for step in range(1, 5):
        cp = {"step": step, "timestamp": "", "cfg_hash": "demo", "result": {"demo": True}}
        with open(os.path.join(demo_project, ".pipeline", f"step{step}.json"), "w") as f:
            json_mod.dump(cp, f)

    return jsonify({
        "project_dir": demo_project,
        "dem_path": dem_path,
        "config": {
            "project": {"name": "demo_kg_mining", "dir": demo_project},
            "domain": {"from_dem": True},
            "dem": {"source": "local", "path": dem_path, "target_res_m": 25},
            "snow": {"source": "fixed", "thickness_m": 2.0},
            "release": {"slope_min_deg": 28, "slope_max_deg": 60, "min_area_m2": 50000},
            "simulation": {"friction_model": "samosATAuto", "mesh_cell_size_m": 25,
                           "t_end_s": 150, "res_type": "ppr|pft|pfv", "snow_density": 200,
                           "rel_th_from_shp": True},
            "dashboard": {"generate": True, "title": "Demo - KG Mining Site Avalanche Hazard",
                          "downsample": 2},
        },
    })


@app.route("/api/pipeline/upload", methods=["POST"])
def api_pipeline_upload():
    """Upload a file (DEM, shapefile) to a project directory."""
    project_dir = request.form.get("dir", "")
    subdir = request.form.get("subdir", "Inputs")  # e.g. "Inputs" or "Inputs/REL"

    if not project_dir:
        return jsonify({"error": "No project dir"}), 400

    target_dir = os.path.join(os.path.expanduser(project_dir), subdir)
    os.makedirs(target_dir, exist_ok=True)

    uploaded_files = []
    for key in request.files:
        f = request.files[key]
        if f.filename:
            dest = os.path.join(target_dir, f.filename)
            f.save(dest)
            uploaded_files.append(dest)

    return jsonify({"uploaded": uploaded_files})


@app.route("/api/pipeline/dashboard/<path:project_name>")
def api_pipeline_dashboard(project_name):
    """Serve a generated dashboard HTML from a project directory."""
    # Look for dashboard in common locations
    base = os.path.expanduser(f"~/avaframe_projects/{project_name}")
    if not os.path.isdir(base):
        # Try absolute path
        base = project_name

    for candidate in [
        os.path.join(base, "dashboard.html"),
        os.path.join(base, "Outputs", "dashboard.html"),
    ]:
        if os.path.isfile(candidate):
            return send_file(candidate)

    # Search for any dashboard*.html
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.startswith("dashboard") and f.endswith(".html"):
                return send_file(os.path.join(root, f))

    return jsonify({"error": "Dashboard not found"}), 404


@app.route("/api/pipeline/results/download")
def api_pipeline_results_download():
    """Download result files as a zip.

    By default downloads only .tif peak files (simulation results).
    Pass ?type=all to include .tif, .png, .html, and .csv files.
    """
    import zipfile
    project_dir = request.args.get("dir", "")
    if not project_dir:
        return jsonify({"error": "No dir"}), 400

    download_type = request.args.get("type", "tif")  # "tif" (default) or "all"

    project_dir = os.path.expanduser(project_dir)
    outputs_dir = os.path.join(project_dir, "Outputs")
    if not os.path.isdir(outputs_dir):
        return jsonify({"error": "No outputs found"}), 404

    # Derive project name for the zip filename
    project_name = os.path.basename(project_dir.rstrip("/"))
    if not project_name:
        project_name = "avaframe_results"

    if download_type == "all":
        allowed_exts = (".tif", ".asc", ".png", ".html", ".csv")
    else:
        allowed_exts = (".tif",)

    # Create zip in memory
    zip_path = os.path.join(project_dir, ".pipeline", "results.zip")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(outputs_dir):
            for f in files:
                if f.endswith(allowed_exts):
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, project_dir)
                    zf.write(full, arcname)

    return send_file(zip_path, as_attachment=True,
                     download_name=f"{project_name}_results.zip")


@app.route("/api/docs/workflow/<workflow_id>")
def api_docs_workflow(workflow_id):
    """Return detailed documentation for a workflow."""
    docs = {
        "com1DFA": {
            "title": "Dense Flow Avalanche Simulation (com1DFA)",
            "overview": """com1DFA is the primary computational module in AvaFrame. It uses a Smoothed Particle Hydrodynamics (SPH) method to simulate dense flow avalanches as a collection of particles moving over a digital elevation model (DEM).

The simulation computes pressure, flow thickness, and flow velocity fields that can be used for hazard mapping and risk assessment.""",
            "inputs": """**Required:**
- **DEM** (.tif or .asc): Digital Elevation Model raster in the Inputs/ directory
- **REL** (shapefile): Release area polygon(s) in Inputs/REL/. Must contain a 'relTh' (release thickness) attribute if relThFromShp=True

**Optional:**
- **ENT** (shapefile): Entrainment area polygon(s) in Inputs/ENT/
- **RES** (shapefile): Resistance area polygon(s) in Inputs/RES/
- **SECREL** (shapefile): Secondary release area in Inputs/SECREL/
- **DAM** (shapefile): Dam line(s) in Inputs/DAM/""",
            "key_parameters": """- **meshCellSize**: Spatial resolution in meters (default: 5m). Smaller = more accurate but slower
- **frictModel**: Friction model choice. samosATAuto is recommended (auto-selects based on volume)
- **tEnd**: Maximum simulation time in seconds (default: 400s)
- **dt**: Time step in seconds (default: 0.1s)
- **rho**: Snow density in kg/m3 (default: 200)
- **resType**: Output result types (ppr=pressure, pft=flow thickness, pfv=flow velocity)""",
            "outputs": """- Peak files (.asc): Maximum values of pressure (ppr), flow thickness (pft), flow velocity (pfv) over the entire simulation
- Report: HTML report with plots and simulation metadata
- Optionally: time-step outputs, particle trajectories""",
            "tips": """1. Start with default parameters before customizing
2. For small avalanches (<25,000 m3), samosATAuto will automatically use higher friction
3. Decrease meshCellSize for more detail (but increases computation time significantly)
4. The simulation automatically stops when kinetic energy drops below 1% of peak
5. Use tSteps to save intermediate results (e.g., tSteps = 0:10 saves every 10 seconds)"""
        },
        "com2AB": {
            "title": "Alpha-Beta Runout Model (com2AB)",
            "overview": """com2AB implements the alpha-beta statistical model for avalanche runout prediction. It uses regression parameters calibrated from historical avalanche data to estimate the runout distance from the avalanche path profile.

The model requires an avalanche path line and a split point (transition between track and runout zone).""",
            "inputs": """**Required:**
- **DEM** (.tif or .asc): Digital Elevation Model
- **LINES** (shapefile): Avalanche path profile line in Inputs/LINES/ (filename must contain 'AB')
- **POINTS** (shapefile): Split point in Inputs/POINTS/""",
            "key_parameters": """- **k1, k2, k3, k4**: Regression coefficients (calibrated values, usually leave at defaults)
- **SD**: Standard deviation (default: 1.25)
- **smallAva**: Set True for small avalanche calibration
- **distance**: Resampling step in meters (default: 10)
- **dsMin**: Threshold distance for beta point (default: 30m)""",
            "outputs": """- Alpha angle, beta angle, and runout position
- Shapefile with results
- Profile plots""",
            "tips": """1. The split point should be placed where the slope transitions from track to runout
2. Use the default (large avalanche) parameters unless you know the avalanche is small
3. The avalanche path line should start from the top and end at/beyond the expected runout"""
        },
        "com4FlowPy": {
            "title": "Flow-Py Propagation Model (com4FlowPy)",
            "overview": """com4FlowPy implements the Flow-Py model, a GIS-based approach for rapid avalanche hazard mapping. It propagates flow from release areas downslope using an angle-of-reach concept with lateral spreading.

Ideal for large-area screening and preliminary hazard assessment.""",
            "inputs": """**Required:**
- **DEM** (.tif or .asc): Digital Elevation Model
- **REL** (raster .tif): Release area raster (cell values > 0 are release cells)

**Optional:**
- **INFRA** (raster): Infrastructure for back-calculation
- **FOREST** (raster): Forest cover for friction/detrainment effects""",
            "key_parameters": """- **alpha**: Angle of reach in degrees (default: 25). Lower = longer runout
- **exp**: Spreading exponent (default: 8). Higher = less lateral spreading
- **flux_threshold**: Threshold for lateral spreading (default: 3e-4)
- **max_z**: Energy line height limit in meters (default: 8848m, i.e., no practical limit)
- **tileSize/tileOverlap**: For large DEMs, controls automatic tiling""",
            "outputs": """- zDelta: Energy line height raster
- cellCounts: Number of paths affecting each cell
- travelLengthMax: Maximum travel length
- fpTravelAngleMax: Maximum travel angle from flow path""",
            "tips": """1. Alpha angle is the most important parameter - lower values produce longer runout
2. For avalanches, typical alpha is 20-30 degrees
3. For rockfall, alpha is typically 30-40 degrees
4. Enable forest interaction for more realistic results in forested terrain
5. Use tiling for DEMs larger than 15km x 15km"""
        },
        "operational": {
            "title": "Operational Run (com1DFA + com2AB)",
            "overview": """The operational workflow runs the full AvaFrame pipeline: first the dense flow avalanche simulation (com1DFA), then optionally the alpha-beta analysis (com2AB) if path and split point data are available, followed by report generation.""",
            "inputs": """**Required:**
- **DEM** (.tif or .asc): Digital Elevation Model
- **REL** (shapefile): Release area polygon(s)

**Optional (for com2AB):**
- **LINES** (shapefile): Avalanche path for alpha-beta
- **POINTS** (shapefile): Split point for alpha-beta""",
            "key_parameters": "See com1DFA and com2AB documentation for their respective parameters.",
            "outputs": """- All com1DFA outputs (peak files, plots)
- com2AB results if path/split point provided
- Combined HTML report""",
            "tips": """1. This is the recommended workflow for a complete avalanche analysis
2. Prepare all input data before running
3. com2AB is skipped if LINES/POINTS are not provided"""
        },
        "probAna": {
            "title": "Probabilistic Analysis (ana4Stats)",
            "overview": """Performs multiple simulations with systematic parameter variations to generate probability maps. Uses Latin Hypercube Sampling or Morris method to explore parameter space efficiently.""",
            "inputs": """**Required:**
- **DEM** (.tif or .asc): Digital Elevation Model
- **REL** (shapefile): Release area polygon(s)""",
            "key_parameters": """- **varParList**: Parameters to vary (e.g., relTh|musamosat)
- **variationType**: How to vary (percent, range, rangefromci)
- **variationValue**: Variation magnitude
- **nSample**: Number of samples (default: 40)
- **sampleMethod**: latin or morris""",
            "outputs": """- Probability maps for each result parameter
- Statistical summary plots
- Individual simulation results""",
            "tips": """1. Start with fewer samples (e.g., 10) for testing, then increase for production
2. Varying release thickness (relTh) is the most common use case
3. Latin Hypercube Sampling is recommended for general use
4. Morris method is better for sensitivity analysis"""
        },
    }

    if workflow_id not in docs:
        return jsonify({"error": "No docs for this workflow"}), 404

    return jsonify(docs[workflow_id])


# ---------------------------------------------------------------------------
# Instructions / Help
# ---------------------------------------------------------------------------

@app.route("/api/docs/general")
def api_docs_general():
    """Return general AvaFrame usage instructions."""
    return jsonify({
        "project_setup": {
            "title": "Project Setup",
            "content": """## Setting Up a Project

1. **Create project directory**: Click "Initialize Project" and provide a path (e.g., `/Users/joel/sim/myAvalanche`)

2. **Add input data**: Place your files in the correct subdirectories:
   - `Inputs/` - DEM raster (.tif or .asc)
   - `Inputs/REL/` - Release area shapefiles (.shp + .dbf + .shx + .prj)
   - `Inputs/ENT/` - Entrainment area shapefiles (optional)
   - `Inputs/RES/` - Resistance area shapefiles (optional)
   - `Inputs/SECREL/` - Secondary release shapefiles (optional)
   - `Inputs/LINES/` - Avalanche path lines (for com2AB, filename must contain 'AB')
   - `Inputs/POINTS/` - Split points (for com2AB)
   - `Inputs/DAM/` - Dam line shapefiles (optional)

3. **DEM requirements**:
   - Must be a georeferenced raster (GeoTIFF or ESRI ASCII)
   - Should cover the full avalanche path from release to runout
   - Resolution of 5m is a good starting point

4. **Release area requirements**:
   - Shapefile polygon(s) defining where the avalanche starts
   - Must include a `relTh` attribute (release thickness in meters) if using relThFromShp=True
   - Must overlap with the DEM extent"""
        },
        "running": {
            "title": "Running Simulations",
            "content": """## Running a Simulation

1. **Select a workflow** from the Run tab
2. **Review configuration** - modify parameters as needed
3. **Click Run** - the simulation will start in the background
4. **Monitor progress** in the log viewer
5. **View results** in the Results tab when complete

### Typical Workflows

- **Quick assessment**: Use com4FlowPy for rapid screening
- **Detailed simulation**: Use com1DFA for full particle-based simulation
- **Full analysis**: Use Operational Run for com1DFA + com2AB + reports
- **Uncertainty analysis**: Use Probabilistic Analysis for parameter variation"""
        },
        "configuration": {
            "title": "Configuration System",
            "content": """## Configuration

AvaFrame uses a hierarchical configuration system:

1. **Default config** - Built into the package (shown in the Config tab)
2. **Local override** - `local_<module>Cfg.ini` in your project directory
3. **WebUI override** - Parameters you modify in this interface

### Priority: WebUI override > Local override > Default

### Tips:
- Hover over parameter names to see documentation
- Modified parameters are highlighted
- Use "Save as Local Config" to persist changes for reuse
- Reset individual parameters by clearing the field"""
        },
        "results": {
            "title": "Understanding Results",
            "content": """## Result Files

### Peak Files (com1DFA)
- **ppr** (peak pressure): Maximum pressure in kPa at each cell
- **pft** (peak flow thickness): Maximum flow thickness in meters
- **pfv** (peak flow velocity): Maximum flow velocity in m/s
- **pta** (peak travel angle): Travel angle in degrees

### Interpreting Pressure Values
- < 1 kPa: Minor damage to lightweight structures
- 1-5 kPa: Damage to wood-frame buildings
- 5-25 kPa: Significant structural damage
- 25-50 kPa: Severe damage to reinforced structures
- > 50 kPa: Complete destruction

### Report
The generated report (in Outputs/reports/) contains:
- Simulation parameters summary
- Peak field plots with hillshade background
- Simulation statistics"""
        }
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  AvaFrame Web UI")
    print("  Open http://localhost:5050 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)

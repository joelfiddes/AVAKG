"""Run AvaFrame simulations (com1DFA or com4FlowPy).

com1DFA: SPH particle simulation — detailed, slow, needs release thickness.
com4FlowPy: Energy-line propagation — fast, large-area screening.

Supports two modes for com1DFA:
- Single-run (snow.source == "fixed"): one simulation with thickness from shapefile
- Multi-run (snow.source in ["era5", "manual"]): one simulation per return period,
  each with a specific fracture depth, plus an envelope of all runs.
"""

import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# INI template
# ---------------------------------------------------------------------------

_INI_TEMPLATE = """\
[GENERAL]
simTypeList = null
resType = {res_type}
tSteps = 0
relThFromShp = {rel_th_from_shp}
relTh = {rel_th}
secRelArea = False
dam = False
frictModel = {friction_model}
tEnd = {t_end}
meshCellSize = {mesh_cell_size}
rho = {rho}
deltaTh = {delta_th}

[INPUT]
releaseScenario =
"""


def _write_ini(project_dir: str, cfg: dict, thickness: float | None,
               from_shp: bool) -> Path:
    """Write local_com1DFACfg.ini into the project directory.

    Parameters
    ----------
    project_dir : str
        Path to the AvaFrame project directory.
    cfg : dict
        Full pipeline config dict.
    thickness : float or None
        Release thickness in metres.  Ignored when *from_shp* is True.
    from_shp : bool
        If True, AvaFrame reads thickness from the shapefile attribute table.

    Returns
    -------
    Path
        Path to the written INI file.
    """
    sim = cfg["simulation"]
    content = _INI_TEMPLATE.format(
        res_type=sim.get("res_type", "ppr|pft|pfv"),
        rel_th_from_shp=str(from_shp),
        rel_th=f"{thickness:.2f}" if thickness is not None and not from_shp else "",
        friction_model=sim.get("friction_model", "samosATAuto"),
        t_end=sim.get("t_end_s", 600),
        mesh_cell_size=sim.get("mesh_cell_size_m", 5),
        rho=sim.get("snow_density", 200),
        delta_th=sim.get("delta_th_m", 0.25),
    )

    ini_path = Path(project_dir) / "local_com1DFACfg.ini"
    ini_path.write_text(content)
    return ini_path


def _clean_com1dfa(project_dir: str) -> None:
    """Remove Outputs/com1DFA/ and Work/com1DFA/ so AvaFrame can run fresh."""
    for subdir in ("Outputs/com1DFA", "Work/com1DFA"):
        p = Path(project_dir) / subdir
        if p.exists():
            shutil.rmtree(p)


def _run_com1dfa(project_dir: str) -> tuple:
    """Execute com1DFA and return (dem, plotDict, reportDictList, simDF).

    Imports are inside the function to avoid top-level dependency on AvaFrame
    (which may not be installed in every environment).
    """
    from avaframe.in3Utils import cfgUtils, logUtils
    from avaframe.com1DFA import com1DFA

    cfgMain = cfgUtils.getGeneralConfig()
    cfgMain["MAIN"]["avalancheDir"] = str(project_dir)

    log = logUtils.initiateLogger(project_dir, "runLog")  # noqa: F841
    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain)
    return dem, plotDict, reportDictList, simDF


def _count_sims(output_dir: Path) -> int:
    """Count the number of peak-pressure result files in *output_dir*."""
    peak_dir = output_dir / "peakFiles"
    if not peak_dir.exists():
        return 0
    return len(list(peak_dir.glob("*_ppr.tif")))


# ---------------------------------------------------------------------------
# com4FlowPy support
# ---------------------------------------------------------------------------

_FLOWPY_INI = """\
[GENERAL]
alpha = {alpha}
exp = {exp}
flux_threshold = {flux_threshold}
max_z = {max_z}
infra = False
previewMode = False
forest = False
forestModule = forestFriction
forestInteraction = False
variableUmaxLim = False
variableAlpha = False
variableExponent = False
fluxDistOldVersion = False
tileSize = 15000
tileOverlap = 5000
procPerCPUCore = 1
chunkSize = 50
maxChunks = 500

[PATHS]
outputFileFormat = .tif
outputFiles = zDelta|cellCounts|travelLengthMax|fpTravelAngleMax
useCustomPaths = False
useCustomPathDEM = False
deleteTempFolder = False

[FLAGS]
plotPath = False
plotProfile = False
saveProfile = False
writeRes = True
fullOut = False
"""


def _create_release_raster(project_dir: str, log_fn=print) -> Path:
    """Create a release raster from shapefiles in Inputs/REL/.

    FlowPy needs a raster where cell value > 0 marks release cells.
    We rasterize all release shapefiles onto the DEM grid.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    import fiona
    from shapely.geometry import shape

    inputs = Path(project_dir) / "Inputs"
    dem_files = list(inputs.glob("*.tif"))
    if not dem_files:
        raise FileNotFoundError("No DEM found in Inputs/")

    with rasterio.open(dem_files[0]) as src:
        dem_shape = (src.height, src.width)
        dem_transform = src.transform
        dem_crs = src.crs
        dem_profile = src.profile.copy()

    # Collect all release geometries
    rel_dir = inputs / "REL"
    geometries = []
    for shp in sorted(rel_dir.glob("*.shp")):
        with fiona.open(shp) as f:
            for feat in f:
                geom = shape(feat["geometry"])
                if geom.is_valid and not geom.is_empty:
                    geometries.append((geom, 1))

    if not geometries:
        raise ValueError("No valid release geometries found in Inputs/REL/")

    log_fn(f"[flowpy] Rasterizing {len(geometries)} release polygons")

    release = rasterize(
        geometries,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=0,
        dtype=np.uint8,
    )

    rel_raster = rel_dir / "release.tif"
    profile = dem_profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(rel_raster, "w", **profile) as dst:
        dst.write(release, 1)

    n_cells = int(np.sum(release > 0))
    log_fn(f"[flowpy] Release raster: {n_cells} cells, saved to {rel_raster.name}")
    return rel_raster


def _run_flowpy(project_dir: str, cfg: dict, log_fn=print):
    """Run com4FlowPy using the standard avaframe runner."""
    from avaframe.runCom4FlowPy import main as flowpy_main

    sim = cfg.get("simulation", {})

    # Write local config
    content = _FLOWPY_INI.format(
        alpha=sim.get("flowpy_alpha", 25),
        exp=sim.get("flowpy_exp", 8),
        flux_threshold=sim.get("flowpy_flux_threshold", 3.0e-4),
        max_z=sim.get("flowpy_max_z", 8848),
    )
    ini_path = Path(project_dir) / "local_com4FlowPyCfg.ini"
    ini_path.write_text(content)

    log_fn("[flowpy] Running com4FlowPy...")
    flowpy_main(avalancheDir=str(project_dir))
    log_fn("[flowpy] com4FlowPy complete")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulations(cfg: dict, thicknesses: dict, project_dir: str,
                    log_fn=print) -> dict:
    """Run com1DFA simulations for the given configuration and thicknesses.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration (merged with defaults).
    thicknesses : dict
        Mapping of label -> thickness in metres.
        - Single-run mode: ``{"fixed": 1.5}``
        - Multi-run mode: ``{"T30": 0.68, "T100": 0.95, "T300": 1.25}``
    project_dir : str
        Path to the AvaFrame project directory (must already contain
        Inputs/REL/ and Inputs/<dem>.tif).
    log_fn : callable, optional
        Logging callback, defaults to ``print``.

    Returns
    -------
    dict
        ``{"runs": [{"label": str, "thickness_m": float,
        "output_dir": str, "n_sims": int}, ...]}``
    """
    project_dir = str(Path(project_dir).resolve())
    model = cfg.get("simulation", {}).get("model", "com1DFA")

    if model == "com4FlowPy":
        return _run_flowpy_pipeline(cfg, project_dir, log_fn)

    # --- com1DFA path ---

    # AvaFrame requires exactly one DEM in Inputs/ — remove extras
    inputs_dir = Path(project_dir) / "Inputs"
    dem_files = sorted(inputs_dir.glob("*.tif")) + sorted(inputs_dir.glob("*.asc"))
    if len(dem_files) > 1:
        # Keep only "dem.tif" (pipeline-generated), remove others
        keep = [f for f in dem_files if f.name == "dem.tif"]
        if not keep:
            keep = [dem_files[0]]  # fallback: keep the first one
        for f in dem_files:
            if f not in keep:
                log_fn(f"[sim] Removing extra DEM: {f.name}")
                f.unlink()

    snow_source = cfg.get("snow", {}).get("source", "fixed")
    is_single = (snow_source == "fixed")

    runs = []

    if is_single:
        # -----------------------------------------------------------------
        # Single-run mode: use thickness from config (not shapefile)
        # Release shapefiles from pipeline have placeholder thickness=0
        # -----------------------------------------------------------------
        thickness = list(thicknesses.values())[0]
        label = "fixed"

        # Use relThFromShp if explicitly set (e.g. demo with pre-made shapefiles)
        use_shp = cfg.get("simulation", {}).get("rel_th_from_shp", False)
        log_fn(f"[sim] Single-run mode, thickness = {thickness:.2f} m, fromShp={use_shp}")
        _clean_com1dfa(project_dir)
        _write_ini(project_dir, cfg, thickness=thickness, from_shp=use_shp)

        log_fn("[sim] Running com1DFA ...")
        _run_com1dfa(project_dir)

        out_dir = Path(project_dir) / "Outputs" / "com1DFA"
        n = _count_sims(out_dir)
        log_fn(f"[sim] Completed: {n} scenario(s)")

        runs.append({
            "label": label,
            "thickness_m": thickness,
            "output_dir": str(out_dir),
            "n_sims": n,
        })

    else:
        # -----------------------------------------------------------------
        # Multi-run mode: one run per return period / thickness
        # -----------------------------------------------------------------
        log_fn(f"[sim] Multi-run mode: {len(thicknesses)} thickness value(s)")

        for label, thickness in thicknesses.items():
            log_fn(f"[sim] --- {label}: thickness = {thickness:.2f} m ---")

            # Clean previous outputs (AvaFrame asserts Work/ doesn't exist)
            _clean_com1dfa(project_dir)

            # Write INI with explicit relTh, relThFromShp=False
            _write_ini(project_dir, cfg, thickness=thickness, from_shp=False)

            log_fn(f"[sim] Running com1DFA for {label} ...")
            _run_com1dfa(project_dir)

            # Move outputs to labelled directory
            src_dir = Path(project_dir) / "Outputs" / "com1DFA"
            dst_dir = Path(project_dir) / "Outputs" / f"com1DFA_{label}"

            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            if src_dir.exists():
                shutil.move(str(src_dir), str(dst_dir))

            n = _count_sims(dst_dir)
            log_fn(f"[sim] {label}: {n} scenario(s), saved to {dst_dir.name}")

            runs.append({
                "label": label,
                "thickness_m": thickness,
                "output_dir": str(dst_dir),
                "n_sims": n,
            })

        # Create an envelope directory placeholder (actual envelope computed
        # during dashboard generation from the per-run peak files)
        envelope_dir = Path(project_dir) / "Outputs" / "com1DFA_envelope"
        envelope_dir.mkdir(parents=True, exist_ok=True)
        log_fn(f"[sim] Envelope directory created: {envelope_dir.name}")

    log_fn(f"[sim] All simulations complete: {len(runs)} run(s)")
    return {"runs": runs}


def _run_flowpy_pipeline(cfg, project_dir, log_fn):
    """Run com4FlowPy (fast energy-line model)."""
    import glob
    import numpy as np
    import rasterio

    # Ensure DEM has nodata set (FlowPy requires it)
    inputs_dir = Path(project_dir) / "Inputs"
    for dem_f in sorted(inputs_dir.glob("*.tif")):
        if dem_f.name == "release.tif" or "REL" in str(dem_f):
            continue
        with rasterio.open(dem_f) as src:
            if src.nodata is None:
                log_fn(f"[flowpy] Setting nodata=-9999 on {dem_f.name}")
                data = src.read(1)
                profile = src.profile.copy()
                profile["nodata"] = -9999.0
                data[np.isnan(data)] = -9999.0
                with rasterio.open(dem_f, "w", **profile) as dst:
                    dst.write(data, 1)
        break

    # Create release raster from shapefiles
    _create_release_raster(project_dir, log_fn)

    # Clean previous FlowPy outputs
    for d in ["Outputs/com4FlowPy", "Work/com4FlowPy"]:
        p = Path(project_dir) / d
        if p.exists():
            shutil.rmtree(p)

    # AvaFrame requires exactly one DEM
    inputs_dir = Path(project_dir) / "Inputs"
    dem_files = sorted(inputs_dir.glob("*.tif"))
    # Don't count the release raster as a DEM
    dem_files = [f for f in dem_files if f.name != "release.tif" and "REL" not in str(f)]
    if len(dem_files) > 1:
        keep = [f for f in dem_files if f.name == "dem.tif"]
        if not keep:
            keep = [dem_files[0]]
        for f in dem_files:
            if f not in keep:
                log_fn(f"[flowpy] Removing extra DEM: {f.name}")
                f.unlink()

    _run_flowpy(project_dir, cfg, log_fn)

    # Find output directory
    out_base = Path(project_dir) / "Outputs" / "com4FlowPy"
    peak_dirs = sorted(out_base.glob("peakFiles/res_*"))
    if peak_dirs:
        out_dir = peak_dirs[0]
        n_files = len(list(out_dir.glob("*.tif")))
        log_fn(f"[flowpy] Results: {n_files} output rasters in {out_dir.name}")
    else:
        out_dir = out_base
        n_files = len(list(out_base.rglob("*.tif")))

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

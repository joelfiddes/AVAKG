"""Run AvaFrame com1DFA simulations.

Supports two modes:
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
        # Single-run mode: thickness comes from shapefile or config
        # -----------------------------------------------------------------
        thickness = list(thicknesses.values())[0]
        label = "fixed"

        log_fn(f"[sim] Single-run mode, thickness = {thickness:.2f} m")
        _clean_com1dfa(project_dir)
        _write_ini(project_dir, cfg, thickness=thickness, from_shp=True)

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

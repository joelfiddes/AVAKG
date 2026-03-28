#!/usr/bin/env python3
"""
AvaFrame Unified Pipeline — CLI entry point.

Usage:
    conda activate avaframe
    python pipeline.py config.yaml
    python pipeline.py config.yaml --from-step 3
    python pipeline.py config.yaml --only-step 2
    python pipeline.py --template > my_config.yaml

Steps:
    1. Resolve domain (bbox / shapefile / center+buffer)
    2. Acquire DEM (Copernicus or local)
    3. Compute release thickness (ERA5 / manual / fixed)
    4. Generate release areas (slope-based PRA)
    5. Run com1DFA simulation(s)
    6. Generate interactive dashboard
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _config_hash(cfg, keys):
    """Hash relevant config sections for checkpoint validation."""
    subset = {k: cfg.get(k) for k in keys if k in cfg}
    return hashlib.md5(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()


def _checkpoint_path(project_dir, step):
    return Path(project_dir) / ".pipeline" / f"step{step}.json"


def _save_checkpoint(project_dir, step, result, cfg_hash):
    cp_dir = Path(project_dir) / ".pipeline"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "cfg_hash": cfg_hash,
        "result": result,
    }
    with open(cp_dir / f"step{step}.json", "w") as f:
        json.dump(cp, f, indent=2, default=str)


def _load_checkpoint(project_dir, step, expected_hash=None):
    cp_path = _checkpoint_path(project_dir, step)
    if not cp_path.exists():
        return None
    with open(cp_path) as f:
        cp = json.load(f)
    if expected_hash and cp.get("cfg_hash") != expected_hash:
        return None  # config changed, invalidate
    return cp["result"]


def run_pipeline(config_path, from_step=1, only_step=None, log_fn=print):
    """Run the full pipeline from a YAML config file.

    Returns dict with results from each completed step.
    """
    from pipeline_steps.config_schema import load_config, validate_config, resolve_domain
    from pipeline_steps.dem import acquire_dem
    from pipeline_steps.snow import compute_thicknesses
    from pipeline_steps.release import generate_release_areas
    from pipeline_steps.simulation import run_simulations
    from pipeline_steps.dashboard import generate_dashboard

    # Load and validate config
    log_fn(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            log_fn(f"  CONFIG ERROR: {e}")
        raise ValueError(f"Config validation failed with {len(errors)} errors")

    project_dir = cfg["project"]["dir"]
    project_dir = os.path.expanduser(project_dir)
    cfg["project"]["dir"] = project_dir

    # Create project structure
    for d in ["Inputs", "Inputs/REL", "Outputs", "Work", ".pipeline"]:
        os.makedirs(os.path.join(project_dir, d), exist_ok=True)

    # Save config copy
    import shutil
    shutil.copy2(config_path, os.path.join(project_dir, ".pipeline", "config.yaml"))

    results = {}
    steps = [
        (1, "Resolve domain", ["domain"], _step1_domain),
        (2, "Acquire DEM", ["domain", "dem"], _step2_dem),
        (3, "Compute release thickness", ["snow"], _step3_snow),
        (4, "Generate release areas", ["release"], _step4_release),
        (5, "Run simulation(s)", ["simulation", "snow"], _step5_simulation),
        (6, "Generate dashboard", ["dashboard"], _step6_dashboard),
    ]

    for step_num, step_name, hash_keys, step_fn in steps:
        if only_step is not None and step_num != only_step:
            # Load checkpoint if available for dependency
            cached = _load_checkpoint(project_dir, step_num)
            if cached:
                results[step_num] = cached
            continue

        if step_num < from_step:
            cached = _load_checkpoint(project_dir, step_num)
            if cached:
                results[step_num] = cached
                log_fn(f"[Step {step_num}/6] {step_name} — cached")
                continue

        cfg_hash = _config_hash(cfg, hash_keys)
        cached = _load_checkpoint(project_dir, step_num, cfg_hash)
        if cached and only_step is None:
            results[step_num] = cached
            log_fn(f"[Step {step_num}/6] {step_name} — cached (config unchanged)")
            continue

        log_fn(f"\n{'='*60}")
        log_fn(f"[Step {step_num}/6] {step_name}")
        log_fn(f"{'='*60}")

        t0 = time.time()
        try:
            result = step_fn(cfg, project_dir, results, log_fn)
            results[step_num] = result
            _save_checkpoint(project_dir, step_num, result, cfg_hash)
            elapsed = time.time() - t0
            log_fn(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            log_fn(f"  ERROR in step {step_num}: {e}")
            import traceback
            log_fn(traceback.format_exc())
            raise

    log_fn(f"\n{'='*60}")
    log_fn("Pipeline complete!")
    log_fn(f"Project directory: {project_dir}")
    if 6 in results and results[6].get("dashboard_path"):
        log_fn(f"Dashboard: {results[6]['dashboard_path']}")
    log_fn(f"{'='*60}")

    return results


def _step1_domain(cfg, project_dir, prev_results, log_fn):
    from pipeline_steps.config_schema import resolve_domain
    domain = resolve_domain(cfg)
    log_fn(f"  Domain bbox (WGS84): {domain['bbox_wgs84']}")
    log_fn(f"  Target EPSG: {domain['target_epsg']}")
    return domain


def _step2_dem(cfg, project_dir, prev_results, log_fn):
    from pipeline_steps.dem import acquire_dem
    domain = prev_results.get(1, {})
    result = acquire_dem(cfg, domain, project_dir, log_fn)
    log_fn(f"  DEM: {result['dem_path']}")
    log_fn(f"  Shape: {result['shape']}")
    return result


def _step3_snow(cfg, project_dir, prev_results, log_fn):
    from pipeline_steps.snow import compute_thicknesses
    result = compute_thicknesses(cfg, project_dir, log_fn)
    log_fn(f"  Mode: {result['mode']}")
    for label, th in result["thicknesses"].items():
        log_fn(f"  {label}: d0 = {th:.2f} m")
    return result


def _step4_release(cfg, project_dir, prev_results, log_fn):
    from pipeline_steps.release import generate_release_areas
    dem_result = prev_results.get(2, {})
    dem_path = dem_result.get("dem_path", "")
    if not dem_path:
        # Try to find DEM in Inputs/
        inputs = Path(project_dir) / "Inputs"
        dems = list(inputs.glob("*.tif")) + list(inputs.glob("*.asc"))
        if dems:
            dem_path = str(dems[0])
        else:
            raise FileNotFoundError("No DEM found. Run step 2 first.")
    result = generate_release_areas(cfg, dem_path, project_dir, log_fn)
    log_fn(f"  Release zones: {result['n_zones']}")
    log_fn(f"  Total area: {result['total_area_m2']/1e6:.3f} km2")
    return result


def _step5_simulation(cfg, project_dir, prev_results, log_fn):
    from pipeline_steps.simulation import run_simulations
    snow_result = prev_results.get(3, {})
    thicknesses = snow_result.get("thicknesses", {"all": 1.5})
    result = run_simulations(cfg, thicknesses, project_dir, log_fn)
    for run in result["runs"]:
        log_fn(f"  {run['label']}: {run['n_sims']} sims, d0={run['thickness_m']:.2f}m")
    return result


def _step6_dashboard(cfg, project_dir, prev_results, log_fn):
    from pipeline_steps.dashboard import generate_dashboard
    sim_result = prev_results.get(5, {})
    result = generate_dashboard(cfg, sim_result, project_dir, log_fn)
    # generate_dashboard returns either a path string or a dict
    if isinstance(result, str):
        result = {"dashboard_path": result}
    log_fn(f"  Dashboard: {result.get('dashboard_path', 'unknown')}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AvaFrame Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Resume from step N (1-6)")
    parser.add_argument("--only-step", type=int, default=None,
                        help="Run only step N")
    parser.add_argument("--template", action="store_true",
                        help="Print a config template to stdout")

    args = parser.parse_args()

    if args.template:
        from pipeline_steps.config_schema import get_config_template
        print(get_config_template())
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    run_pipeline(args.config, args.from_step, args.only_step)


if __name__ == "__main__":
    main()

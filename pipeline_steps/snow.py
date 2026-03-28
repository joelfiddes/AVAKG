"""
Snow climatology / release thickness computation.

Three modes:
  - fixed:  user supplies a single thickness value
  - manual: user supplies fracture depths per return period
  - era5:   fetch Open-Meteo ERA5 snowfall, compute H72, fit Gumbel,
            apply Swiss corrections (elevation, slope, wind)
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r

# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def compute_thicknesses(cfg, project_dir, log_fn=print):
    """Dispatcher. Returns {"mode": str, "thicknesses": dict, "plot_path": str|None}."""
    snow_cfg = cfg.get("snow", {})
    mode = snow_cfg.get("mode", "fixed").lower().strip()

    if mode == "fixed":
        return _mode_fixed(snow_cfg, log_fn)
    elif mode == "manual":
        return _mode_manual(snow_cfg, log_fn)
    elif mode == "era5":
        return _mode_era5(snow_cfg, project_dir, log_fn)
    else:
        raise ValueError(f"Unknown snow mode: {mode!r}. Expected fixed/manual/era5.")


# ---------------------------------------------------------------------------
# Fixed mode
# ---------------------------------------------------------------------------

def _mode_fixed(snow_cfg, log_fn):
    thickness_m = float(snow_cfg.get("thickness_m", 1.0))
    log_fn(f"[snow] Fixed mode: thickness = {thickness_m:.2f} m")
    return {"mode": "fixed", "thicknesses": {"all": thickness_m}, "plot_path": None}


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------

def _mode_manual(snow_cfg, log_fn):
    raw = snow_cfg.get("fracture_depths", {})
    if not raw:
        raise ValueError("[snow] Manual mode selected but no fracture_depths provided.")
    thicknesses = {str(k): float(v) for k, v in raw.items()}
    log_fn(f"[snow] Manual mode: {thicknesses}")
    return {"mode": "manual", "thicknesses": thicknesses, "plot_path": None}


# ---------------------------------------------------------------------------
# ERA5 mode
# ---------------------------------------------------------------------------

def _mode_era5(snow_cfg, project_dir, log_fn):
    lat = float(snow_cfg["latitude"])
    lon = float(snow_cfg["longitude"])
    start_year = int(snow_cfg.get("start_year", 1950))
    end_year = int(snow_cfg.get("end_year", datetime.now().year - 1))
    grid_elev = float(snow_cfg.get("grid_elevation_m", 0.0))
    target_elev = float(snow_cfg.get("target_elevation_m", grid_elev))
    gradient = float(snow_cfg.get("h72_gradient_cm_per_100m", 5.0))
    slope_deg = float(snow_cfg.get("mean_slope_deg", 35.0))
    wind_m = float(snow_cfg.get("wind_loading_m", 0.0))
    return_periods = snow_cfg.get("return_periods", [30, 100, 300])
    if isinstance(return_periods, str):
        return_periods = [int(x.strip()) for x in return_periods.split(",")]

    project_dir = Path(project_dir)
    cache_path = project_dir / "h72_cache.csv"

    log_fn(f"[snow] ERA5 mode: lat={lat}, lon={lon}, {start_year}-{end_year}")
    log_fn(f"[snow] Grid elev={grid_elev} m, target elev={target_elev} m, "
           f"gradient={gradient} cm/100m")
    log_fn(f"[snow] Slope={slope_deg}°, wind loading={wind_m} m")
    log_fn(f"[snow] Return periods: {return_periods}")

    # 1. Fetch snowfall data
    df = _fetch_openmeteo(lat, lon, start_year, end_year, cache_path, log_fn)

    # 2. Compute H72
    h72 = _compute_h72(df)

    # 3. Extract annual maxima
    annual_max_df = _extract_annual_maxima(h72)
    log_fn(f"[snow] {len(annual_max_df)} water years of annual maxima")

    if len(annual_max_df) < 10:
        raise ValueError(
            f"[snow] Only {len(annual_max_df)} water years found — need at least 10 "
            "for a reliable Gumbel fit."
        )

    # 4. Fit Gumbel
    annual_max = annual_max_df["h72_max_m"].values
    fits = _fit_gumbel(annual_max, return_periods)
    log_fn("[snow] Gumbel fit (grid-point H72):")
    for rp in return_periods:
        val = fits["values"][rp]
        ci_lo, ci_hi = fits["ci_95"][rp]
        log_fn(f"  {rp:>4d}-yr: {val:.2f} m  (95% CI: {ci_lo:.2f} – {ci_hi:.2f})")

    # 5. Apply corrections
    corrections = _apply_corrections(
        {rp: fits["values"][rp] for rp in return_periods},
        grid_elev, target_elev, gradient, slope_deg, wind_m,
    )
    log_fn("[snow] Corrected release thicknesses:")
    thicknesses = {}
    for rp in return_periods:
        val = corrections[rp]
        log_fn(f"  {rp:>4d}-yr: {val:.2f} m")
        thicknesses[f"{rp}yr"] = round(val, 3)

    # 6. Diagnostic plot
    plot_path = project_dir / "snow_climatology.png"
    try:
        _make_snow_plot(
            annual_max_df, h72, fits, corrections,
            return_periods, str(plot_path), snow_cfg, log_fn,
        )
        log_fn(f"[snow] Diagnostic plot saved to {plot_path}")
    except Exception as exc:
        log_fn(f"[snow] Warning: could not create diagnostic plot: {exc}")
        plot_path = None

    return {
        "mode": "era5",
        "thicknesses": thicknesses,
        "plot_path": str(plot_path) if plot_path else None,
    }


# ---------------------------------------------------------------------------
# Open-Meteo ERA5 fetch
# ---------------------------------------------------------------------------

def _fetch_openmeteo(lat, lon, start_year, end_year, cache_path, log_fn=print):
    """
    Fetch daily snowfall from Open-Meteo ERA5 archive in 5-year chunks.

    Returns a DataFrame with columns ['date', 'snowfall_cm'] indexed by date.
    Results are cached to *cache_path* as CSV.
    """
    import requests

    cache_path = Path(cache_path)

    # Return cached data if available
    if cache_path.exists():
        log_fn(f"[snow] Loading cached snowfall data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df = df.set_index("date")
        return df

    url = "https://archive-api.open-meteo.com/v1/archive"
    all_frames = []

    # Build 5-year chunks
    chunk_starts = list(range(start_year, end_year + 1, 5))
    for cs in chunk_starts:
        ce = min(cs + 4, end_year)
        start_date = f"{cs}-01-01"
        end_date = f"{ce}-12-31"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "snowfall_sum",
            "timezone": "auto",
        }

        log_fn(f"[snow] Fetching ERA5 snowfall {start_date} to {end_date} ...")

        for attempt in range(1, 6):
            try:
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log_fn(f"[snow] Rate limited, retrying in {wait}s (attempt {attempt}/5)")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                if attempt == 5:
                    raise RuntimeError(
                        f"[snow] Failed to fetch ERA5 data after 5 attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt
                log_fn(f"[snow] Request error ({exc}), retrying in {wait}s "
                       f"(attempt {attempt}/5)")
                time.sleep(wait)
        else:
            raise RuntimeError("[snow] Failed to fetch ERA5 data after 5 attempts.")

        data = resp.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        snowfall = daily.get("snowfall_sum", [])

        chunk_df = pd.DataFrame({"date": pd.to_datetime(dates), "snowfall_cm": snowfall})
        all_frames.append(chunk_df)

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    df = df.set_index("date")

    # Replace NaN snowfall with 0
    df["snowfall_cm"] = df["snowfall_cm"].fillna(0.0)

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    log_fn(f"[snow] Cached snowfall data to {cache_path}")

    return df


# ---------------------------------------------------------------------------
# H72 computation
# ---------------------------------------------------------------------------

def _compute_h72(df):
    """Rolling 3-day sum of snowfall, convert cm -> m."""
    h72 = df["snowfall_cm"].rolling(window=3, min_periods=1).sum() / 100.0
    h72.name = "h72_m"
    return h72


# ---------------------------------------------------------------------------
# Annual maxima extraction
# ---------------------------------------------------------------------------

def _extract_annual_maxima(h72):
    """One max H72 per water year (Oct-Sep), filter partial years."""
    df = h72.to_frame()
    # Water year: Oct of year N -> Sep of year N+1 => water_year = N+1
    df["water_year"] = df.index.year + (df.index.month >= 10).astype(int)

    # Drop partial first / last water years
    wy_counts = df.groupby("water_year").size()
    # A full water year has ~365 days; keep only those with >= 300 days
    valid_wys = wy_counts[wy_counts >= 300].index
    df = df[df["water_year"].isin(valid_wys)]

    annual_max = df.groupby("water_year")["h72_m"].max().reset_index()
    annual_max.columns = ["water_year", "h72_max_m"]
    return annual_max


# ---------------------------------------------------------------------------
# Gumbel fit
# ---------------------------------------------------------------------------

def _fit_gumbel(annual_max, return_periods):
    """
    Fit Gumbel distribution, compute return values + bootstrap 95% CI.

    Returns dict with keys:
        "loc", "scale" — Gumbel parameters
        "values" — {rp: value_m}
        "ci_95"  — {rp: (lo, hi)}
    """
    loc, scale = gumbel_r.fit(annual_max)

    # Return-period quantiles:  p = 1 - 1/T  =>  ppf(p)
    values = {}
    for rp in return_periods:
        p = 1.0 - 1.0 / rp
        values[rp] = float(gumbel_r.ppf(p, loc=loc, scale=scale))

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    n_boot = 5000
    n = len(annual_max)
    boot_values = {rp: np.empty(n_boot) for rp in return_periods}

    for i in range(n_boot):
        sample = rng.choice(annual_max, size=n, replace=True)
        try:
            b_loc, b_scale = gumbel_r.fit(sample)
        except Exception:
            # Degenerate sample — use original fit
            b_loc, b_scale = loc, scale
        for rp in return_periods:
            p = 1.0 - 1.0 / rp
            boot_values[rp][i] = gumbel_r.ppf(p, loc=b_loc, scale=b_scale)

    ci_95 = {}
    for rp in return_periods:
        ci_95[rp] = (
            float(np.percentile(boot_values[rp], 2.5)),
            float(np.percentile(boot_values[rp], 97.5)),
        )

    return {
        "loc": float(loc),
        "scale": float(scale),
        "values": values,
        "ci_95": ci_95,
    }


# ---------------------------------------------------------------------------
# Swiss corrections (elevation + slope + wind)
# ---------------------------------------------------------------------------

def _apply_corrections(h72_values, grid_elev, target_elev, gradient, slope_deg, wind_m):
    """
    Elevation + slope + wind corrections (Swiss procedure).

    Parameters
    ----------
    h72_values : dict {rp: h72_m}
        Grid-point H72 values from the Gumbel fit.
    grid_elev : float
        Elevation of the ERA5 grid point (m a.s.l.).
    target_elev : float
        Elevation of the release area (m a.s.l.).
    gradient : float
        H72 elevation gradient (cm per 100 m).
    slope_deg : float
        Mean slope angle of the release area (degrees).
    wind_m : float
        Wind loading addition (m).

    Returns
    -------
    dict {rp: corrected_thickness_m}
    """
    dz = target_elev - grid_elev  # elevation difference in m
    elev_corr_m = (gradient / 100.0) * (dz / 100.0)  # gradient is cm/100m -> m/100m -> m

    # Slope correction: project perpendicular depth onto slope
    slope_rad = np.radians(slope_deg)
    slope_factor = np.cos(slope_rad)

    corrected = {}
    for rp, h72 in h72_values.items():
        val = h72 + elev_corr_m          # elevation correction
        val = val * slope_factor          # slope correction (perpendicular depth)
        val = val + wind_m                # wind loading
        corrected[rp] = max(val, 0.01)   # ensure positive
    return corrected


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def _make_snow_plot(annual_max_df, h72, fits, corrections, return_periods,
                    output_path, cfg, log_fn=print):
    """
    4-panel diagnostic plot:
      1. Daily H72 time series
      2. Annual maxima bar chart
      3. Gumbel fit with CI
      4. Corrected thicknesses per return period
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: H72 time series ---
    ax = axes[0, 0]
    ax.plot(h72.index, h72.values, linewidth=0.4, color="steelblue", alpha=0.8)
    ax.set_title("Daily H72 (3-day snowfall)")
    ax.set_ylabel("H72 (m)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Annual maxima bar chart ---
    ax = axes[0, 1]
    ax.bar(annual_max_df["water_year"], annual_max_df["h72_max_m"],
           color="steelblue", edgecolor="navy", linewidth=0.5)
    ax.set_title("Annual maxima H72 (water year)")
    ax.set_ylabel("H72 max (m)")
    ax.set_xlabel("Water year")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Gumbel return-period curve ---
    ax = axes[1, 0]
    rp_curve = np.logspace(np.log10(2), np.log10(500), 200)
    loc, scale = fits["loc"], fits["scale"]
    h72_curve = gumbel_r.ppf(1.0 - 1.0 / rp_curve, loc=loc, scale=scale)
    ax.plot(rp_curve, h72_curve, "b-", linewidth=1.5, label="Gumbel fit")

    # Plot CI band for specific return periods
    rps_arr = np.array(return_periods)
    vals_arr = np.array([fits["values"][rp] for rp in return_periods])
    ci_lo = np.array([fits["ci_95"][rp][0] for rp in return_periods])
    ci_hi = np.array([fits["ci_95"][rp][1] for rp in return_periods])
    ax.errorbar(rps_arr, vals_arr, yerr=[vals_arr - ci_lo, ci_hi - vals_arr],
                fmt="ro", capsize=4, label="Return values (95% CI)")

    # Empirical plotting positions (Gringorten)
    n = len(annual_max_df)
    sorted_max = np.sort(annual_max_df["h72_max_m"].values)
    ranks = np.arange(1, n + 1)
    emp_rp = (n + 0.12) / (ranks - 0.44)
    emp_rp = emp_rp[::-1]  # ascending with sorted values
    ax.plot(emp_rp, sorted_max, "k.", markersize=4, alpha=0.6, label="Observed")

    ax.set_xscale("log")
    ax.set_title("Gumbel return-period analysis")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("H72 (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 4: Corrected thicknesses ---
    ax = axes[1, 1]
    rp_labels = [f"{rp}yr" for rp in return_periods]
    grid_vals = [fits["values"][rp] for rp in return_periods]
    corr_vals = [corrections[rp] for rp in return_periods]
    x = np.arange(len(return_periods))
    width = 0.35
    ax.bar(x - width / 2, grid_vals, width, label="Grid-point H72", color="steelblue")
    ax.bar(x + width / 2, corr_vals, width, label="Corrected thickness", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(rp_labels)
    ax.set_title("Release thickness (grid vs corrected)")
    ax.set_ylabel("Thickness (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate corrections
    grid_elev = float(cfg.get("grid_elevation_m", 0))
    target_elev = float(cfg.get("target_elevation_m", grid_elev))
    slope_deg = float(cfg.get("mean_slope_deg", 35))
    wind_m = float(cfg.get("wind_loading_m", 0))
    info = (f"Elev: {grid_elev:.0f} -> {target_elev:.0f} m | "
            f"Slope: {slope_deg:.0f}° | Wind: +{wind_m:.2f} m")
    ax.set_xlabel(info, fontsize=8)

    fig.suptitle("Snow Climatology — ERA5 H72 Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_fn(f"[snow] Plot saved: {output_path}")

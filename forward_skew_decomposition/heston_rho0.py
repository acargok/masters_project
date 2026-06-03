#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heston particle method + LSV validation with rho = 0.

Re-runs the Heston leverage extraction with the spot-variance correlation forced
to zero (kappa, theta, xi, V0 reused from the production calibration), then
reprices vanillas. The particle method is rerun rather than reused because
setting rho = 0 changes the joint (S, V) distribution the Gyongy projection sees.

Run as a standalone subprocess from the repo root so only lsv_heston modules
load, avoiding the sibling-name clash with lsv_bergomi.
"""
import json
import logging
import sys
import time

import numpy as np

import decomp_config as cfg

sys.path.insert(0, str(cfg.LSV_HESTON_DIR))
import particle_method as pm      # noqa: E402
import lsv_validation as val      # noqa: E402
from scipy import interpolate     # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heston_rho0")

ARRAYS = cfg.HESTON0_DIR / "arrays"
DATA = cfg.HESTON0_DIR / "data"
PLOTS = cfg.HESTON0_DIR / "plots"


def run_particle(seed=cfg.SEED):
    """Heston leverage extraction with rho = 0; saves arrays/params/plot."""
    inputs = pm.load_inputs()
    baseline_rho = float(inputs["heston"]["rho"])
    modified = dict(inputs["heston"])
    modified["rho"] = 0.0
    inputs["heston"] = modified
    logger.info(f"Overrode rho = 0.0 (baseline {baseline_rho:+.4f}); kept "
                f"kappa={modified['kappa']:.4f} theta={modified['theta']:.6f} "
                f"xi={modified['xi']:.4f} V0={modified['V0']:.6f}")

    t0 = time.time()
    results = pm.run_particle_method(
        inputs, N=pm.N_PARTICLES, dt=pm.DT,
        bandwidth_override=pm.BANDWIDTH_OVERRIDE, seed=seed,
        variance_scheme=pm.VARIANCE_SCHEME,
    )
    logger.info(f"Particle method done in {time.time() - t0:.1f}s")

    np.save(ARRAYS / "leverage_surface.npy", results["leverage_surface"])
    np.save(ARRAYS / "leverage_spot_grid.npy", results["spot_grid"])
    np.save(ARRAYS / "leverage_time_grid.npy", results["time_grid"])
    with open(DATA / "heston_params.json", "w") as f:
        json.dump({**modified, "rho_baseline": baseline_rho,
                   "experiment": "decomposition_rho_zero"}, f, indent=2)
    with open(DATA / "particle_log.json", "w") as f:
        json.dump(results["log"], f, indent=2)

    # plot_leverage_surface writes to pm.PLOT_DIR — redirect to the experiment.
    _orig = pm.PLOT_DIR
    pm.PLOT_DIR = PLOTS
    try:
        pm.plot_leverage_surface(results["leverage_surface"], results["spot_grid"],
                                 results["time_grid"], inputs["S"])
    finally:
        pm.PLOT_DIR = _orig
    return results


def run_validation(seed=cfg.SEED):
    """Vanilla repricing under the rho = 0 leverage; writes the summary."""
    inputs = val.load_validation_inputs()
    with open(DATA / "heston_params.json") as f:
        inputs["heston"] = json.load(f)
    lev = np.load(ARRAYS / "leverage_surface.npy")
    sg = np.load(ARRAYS / "leverage_spot_grid.npy")
    tg = np.load(ARRAYS / "leverage_time_grid.npy")
    inputs["leverage_surface"] = lev
    inputs["spot_grid"] = sg
    inputs["time_grid"] = tg
    inputs["leverage_interp"] = interpolate.RegularGridInterpolator(
        (sg, tg), lev, method="linear", bounds_error=False, fill_value=None)

    df, _ = val.lsv_monte_carlo_reprice(
        inputs, n_paths=val.MC_N_PATHS, n_reprice=val.MC_N_REPRICE, seed=seed)
    df.to_csv(DATA / "lsv_repricing_errors.csv", index=False)
    summary = val.compute_summary(df)
    with open(DATA / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Validation: n={summary.get('n_valid')} "
                f"RMSE={summary.get('lsv_iv_rmse_bps', float('nan')):.1f}bp")
    return summary


def run(seed=cfg.SEED):
    for d in (ARRAYS, DATA, PLOTS):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("  HESTON particle method + validation — rho = 0")
    logger.info("=" * 70)
    run_particle(seed)
    run_validation(seed)


if __name__ == "__main__":
    run()

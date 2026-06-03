#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bergomi two-factor particle method + LSV validation with rho1 = rho2 = 0.

Re-runs the Bergomi leverage extraction with the spot-variance correlations
forced to zero. The factor-factor correlation rho12, the variance-dynamics
params (nu, theta, kappa1, kappa2), and the initial forward variance curve are
all preserved — the latter is independent of spot-variance correlation. With
rho1 = rho2 = 0 the 3x3 correlation matrix is block-diagonal and stays
positive-definite for any |rho12| < 1, so no regularisation is needed.

Run as a standalone subprocess from the repo root so only lsv_bergomi modules
load, avoiding the sibling-name clash with lsv_heston.
"""
import json
import logging
import sys
import time

import numpy as np

import decomp_config as cfg

sys.path.insert(0, str(cfg.LSV_BERGOMI_DIR))
import particle_method as pm      # noqa: E402
import lsv_validation as val      # noqa: E402
from scipy import interpolate     # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bergomi_rho0")

ARRAYS = cfg.BERGOMI0_DIR / "arrays"
DATA = cfg.BERGOMI0_DIR / "data"
PLOTS = cfg.BERGOMI0_DIR / "plots"


def run_particle(seed=cfg.SEED):
    """Bergomi leverage extraction with rho1 = rho2 = 0; saves arrays/params/plot."""
    inputs = pm.load_inputs()
    base_rho1 = float(inputs["bergomi"]["rho1"])
    base_rho2 = float(inputs["bergomi"]["rho2"])
    modified = dict(inputs["bergomi"])
    modified["rho1"] = 0.0
    modified["rho2"] = 0.0
    inputs["bergomi"] = modified
    logger.info(f"Overrode rho1 = rho2 = 0.0 (baseline rho1={base_rho1:+.4f} "
                f"rho2={base_rho2:+.4f}); kept nu={modified['nu']:.4f} "
                f"theta={modified['theta']:.4f} kappa1={modified['kappa1']:.4f} "
                f"kappa2={modified['kappa2']:.4f} rho12={modified['rho12']:+.4f}")

    t0 = time.time()
    results = pm.run_particle_method(
        inputs, N=pm.N_PARTICLES, dt=pm.DT,
        bandwidth_override=pm.BANDWIDTH_OVERRIDE, seed=seed,
    )
    logger.info(f"Particle method done in {time.time() - t0:.1f}s")

    np.save(ARRAYS / "leverage_surface.npy", results["leverage_surface"])
    np.save(ARRAYS / "leverage_spot_grid.npy", results["spot_grid"])
    np.save(ARRAYS / "leverage_time_grid.npy", results["time_grid"])
    with open(DATA / "bergomi_params.json", "w") as f:
        json.dump({**modified, "rho1_baseline": base_rho1,
                   "rho2_baseline": base_rho2,
                   "experiment": "decomposition_rho_zero"}, f, indent=2)
    with open(DATA / "particle_log.json", "w") as f:
        json.dump(results["log"], f, indent=2)

    _orig = pm.PLOT_DIR
    pm.PLOT_DIR = PLOTS
    try:
        pm.plot_leverage_surface(results["leverage_surface"], results["spot_grid"],
                                 results["time_grid"], inputs["S"])
    finally:
        pm.PLOT_DIR = _orig
    return results


def run_validation(seed=cfg.SEED):
    """Vanilla repricing under the rho1 = rho2 = 0 leverage; writes the summary."""
    inputs = val.load_validation_inputs()
    with open(DATA / "bergomi_params.json") as f:
        inputs["bergomi"] = json.load(f)
    sg = np.load(ARRAYS / "leverage_spot_grid.npy")
    tg = np.load(ARRAYS / "leverage_time_grid.npy")
    lev = np.load(ARRAYS / "leverage_surface.npy")
    inputs["spot_grid"] = sg
    inputs["time_grid"] = tg
    inputs["leverage_interp"] = interpolate.RegularGridInterpolator(
        (sg, tg), lev, method="linear", bounds_error=False, fill_value=None)

    df = val.bergomi_lsv_mc_reprice(
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
    logger.info("  BERGOMI particle method + validation — rho1 = rho2 = 0")
    logger.info("=" * 70)
    run_particle(seed)
    run_validation(seed)


if __name__ == "__main__":
    run()

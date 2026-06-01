#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage Bergomi parameter calibration:
  Stage 1 — variance-of-variance calibration.
  Stage 2 — skew correlation calibration via (rho1, chi).
"""

import logging

import numpy as np
from scipy import optimize

from bergomi_param_config import SEED, DE_MAXITER, N_WORKERS
from bergomi_models import vol_of_vol_model, skew_order1_model

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: variance-of-variance calibration
# =============================================================================

def stage1_objective(params, T_grid, target):
    nu, theta, kappa1, kappa2, rho12 = params
    # Hard constraint: kappa1 > kappa2 (short factor faster). Soft penalty.
    if kappa1 <= kappa2:
        return 1e6 + (kappa2 - kappa1) ** 2 * 1e3
    # Validity of normalisation denominator
    denom_sq = (1 - theta) ** 2 + theta ** 2 + 2.0 * rho12 * theta * (1 - theta)
    if denom_sq <= 1e-8:
        return 1e6
    try:
        model = vol_of_vol_model(T_grid, nu, theta, kappa1, kappa2, rho12)
    except (FloatingPointError, ValueError):
        return 1e6
    if not np.all(np.isfinite(model)):
        return 1e6
    return float(np.sum((model - target) ** 2))


def calibrate_stage1(T_grid, target, seed=SEED, maxiter=DE_MAXITER):
    bounds = [
        (0.5, 2),     # nu
        (0.1, 0.5),     # theta
        (3.0, 20.0),    # kappa1
        (0.05, 0.6),    # kappa2
        (-0.99, 0.8),  # rho12
    ]
    logger.info("Stage 1: calibrating (nu, theta, kappa1, kappa2, rho12) "
                "via differential evolution + Nelder-Mead polish")

    de = optimize.differential_evolution(
        stage1_objective, bounds, args=(T_grid, target),
        seed=seed, maxiter=maxiter,
        tol=1e-10, atol=1e-10,
        polish=False, workers=N_WORKERS, updating="deferred", disp=False,
    )
    logger.info(f"  DE: fun={de.fun:.6e}, nfev={de.nfev}")

    # L-BFGS-B polish respects bounds (NM does not — it walks flat valleys
    # of the over-parameterised vol-of-vol surface to nonsensical values).
    pol = optimize.minimize(
        stage1_objective, de.x, args=(T_grid, target),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    logger.info(f"  L-BFGS-B: fun={pol.fun:.6e}, nfev={pol.nfev}")

    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x = np.clip(pol.x, lo, hi)
    nu, theta, kappa1, kappa2, rho12 = x
    if kappa1 <= kappa2:
        logger.warning(f"  kappa1 ({kappa1:.3f}) <= kappa2 ({kappa2:.3f}); "
                        "this should not happen — investigate.")
    return {
        "nu": float(nu), "theta": float(theta),
        "kappa1": float(kappa1), "kappa2": float(kappa2),
        "rho12": float(rho12),
        "objective": float(pol.fun),
    }


# =============================================================================
# Stage 2: skew correlation calibration via (rho1, chi)
# =============================================================================

def derive_rho2(rho1, chi, rho12):
    """Wang eq. 4.4: ensures (rho12, rho1, rho2) is a valid correlation matrix."""
    return rho12 * rho1 + chi * np.sqrt(max(1 - rho12 ** 2, 0.0)) \
                                 * np.sqrt(max(1 - rho1 ** 2, 0.0))


def stage2_objective(params, T_grid, target, stage1):
    rho1, chi = params
    rho2 = derive_rho2(rho1, chi, stage1["rho12"])
    # Penalty if rho2 strays outside [-0.99, 0.0] (we expect negative for SPX)
    pen = 0.0
    if rho2 < -0.99:
        pen += 1e3 * (rho2 + 0.99) ** 2
    if rho2 > 0.0:
        pen += 1e3 * rho2 ** 2
    try:
        model = skew_order1_model(
            T_grid,
            stage1["nu"], stage1["theta"],
            stage1["kappa1"], stage1["kappa2"], stage1["rho12"],
            rho1, rho2,
        )
    except (FloatingPointError, ValueError):
        return 1e6
    if not np.all(np.isfinite(model)):
        return 1e6
    return float(np.sum((model - target) ** 2)) + pen


def calibrate_stage2(T_grid, target, stage1, seed=SEED, maxiter=DE_MAXITER):
    bounds = [
        (-0.99, 0.0),   # rho1
        (-1, 1),    # chi  (parametrises rho2)
    ]
    logger.info("Stage 2: calibrating (rho1, chi -> rho2) via DE + NM polish")

    de = optimize.differential_evolution(
        stage2_objective, bounds, args=(T_grid, target, stage1),
        seed=seed, maxiter=maxiter,
        tol=1e-10, atol=1e-10,
        polish=False, workers=N_WORKERS, updating="deferred", disp=False,
    )
    logger.info(f"  DE: fun={de.fun:.6e}, nfev={de.nfev}")

    pol = optimize.minimize(
        stage2_objective, de.x, args=(T_grid, target, stage1),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    logger.info(f"  L-BFGS-B: fun={pol.fun:.6e}, nfev={pol.nfev}")

    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x = np.clip(pol.x, lo, hi)
    rho1, chi = x
    rho2 = derive_rho2(rho1, chi, stage1["rho12"])
    return {"rho1": float(rho1), "chi": float(chi), "rho2": float(rho2),
            "objective": float(pol.fun)}

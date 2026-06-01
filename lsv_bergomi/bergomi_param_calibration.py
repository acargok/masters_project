#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bergomi Two-Factor Parameter Calibration (Wang 2017 §5.2.3 — separate calibration)
====================================================================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Calibrates the Bergomi two-factor parameters (nu, theta, kappa1, kappa2, rho12,
rho1, rho2) in two stages:

  Stage 1 — Variance dynamics: fit (nu, theta, kappa1, kappa2, rho12) to a
            target vol-of-vol term structure nu^B(T) = sigma0 * (tau0 / T)^alpha
            (Wang eq. 4.3). Uses the order-0 expression for nu_t^T (Wang 3.10),
            assuming a flat forward variance curve so that
                A_i(T) = (1 - exp(-kappa_i T)) / (kappa_i T).

  Stage 2 — Skew correlations: fit (rho1, rho2) to the SPX ATMF skew term
            structure measured from the SSVI surface, using the order-1
            Bergomi-Guyon skew formula (Wang eq. 4.1):
                S_T^{ord1} = nu * alpha_theta *
                  [ (1-theta) rho1 g(kappa1 T)/(kappa1 T)^2
                  +    theta  rho2 g(kappa2 T)/(kappa2 T)^2 ]
            with g(x) = x - (1 - exp(-x)). Parametrisation
                rho2 = rho12 * rho1 + chi * sqrt(1 - rho12^2) sqrt(1 - rho1^2)
            (Wang 4.4) ensures a valid 3x3 correlation matrix by construction.

Outputs:
    lsv_bergomi/data/bergomi_params.json — calibrated parameters
    lsv_bergomi/plots/bergomi_calib_volofvol.png — vol-of-vol fit plot
    lsv_bergomi/plots/bergomi_calib_skew.png      — ATMF skew fit plot

Reference:
    Wang, J. (2017). LSV Model Calibration. PhD thesis, Imperial College London.
    See sections 3.3 (Bergomi-Guyon expansion), 4.1-4.3 (skew/vol-of-vol
    benchmarks), 5.2.3 (separate calibration, "Bergomi I").
"""

# ===== IMPORTS =====
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize

from bergomi_param_config import (
    DEFAULT_SIGMA0, DEFAULT_TAU0, DEFAULT_ALPHA, SKEW_DELTA_K,
    SEED, DE_MAXITER, N_WORKERS, T_GRID_VOLOFVOL,
    ROOT, IV_DIR, DUPIRE_DIR, BERGOMI_DIR, DATA_DIR, PLOT_DIR,
)
from bergomi_models import *
from bergomi_param_stages import *

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bergomi_param_calibration")


# =============================================================================
# Plotting
# =============================================================================

def plot_volofvol(T_grid, target, model_fitted, params, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    T_plot = np.linspace(T_grid.min(), T_grid.max(), 200)
    target_plot = vol_of_vol_benchmark(T_plot)
    model_plot = vol_of_vol_model(
        T_plot, params["nu"], params["theta"],
        params["kappa1"], params["kappa2"], params["rho12"],
    )
    ax.plot(T_plot, target_plot * 100, "k-", lw=2, label="Target")
    ax.plot(T_plot, model_plot * 100, color="#1f77b4", lw=2, label="Bergomi model")
    ax.scatter(T_grid, target * 100, c="k", marker="o", zorder=3)
    ax.scatter(T_grid, model_fitted * 100, c="#1f77b4", marker="x", zorder=3)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel(r"$\nu^{B}(T)$  (%)")
    ax.set_title(
        f"Stage 1 — vol-of-vol term structure\n"
        f"nu={params['nu']:.3f}  theta={params['theta']:.3f}  "
        f"k1={params['kappa1']:.3f}  k2={params['kappa2']:.3f}  "
        f"rho12={params['rho12']:.3f}"
    )
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved -> {out_path}")


def plot_skew(T_grid, target, model_fitted, params, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    T_plot = np.linspace(T_grid.min(), T_grid.max(), 200)
    model_plot = skew_order1_model(
        T_plot, params["nu"], params["theta"],
        params["kappa1"], params["kappa2"], params["rho12"],
        params["rho1"], params["rho2"],
    )
    ax.plot(T_plot, model_plot, color="#d62728", lw=2,
            label="Bergomi two-factor model")
    ax.scatter(T_grid, target, c="k", marker="o", label="SSVI ATMF skew")
    ax.scatter(T_grid, model_fitted, c="#d62728", marker="x")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("ATMF skew  (dvol / dlogK)")
    ax.set_title(
        f"Stage 2 — ATMF skew term structure\n"
        f"rho1={params['rho1']:.3f}  rho2={params['rho2']:.3f}  "
        f"rho12={params['rho12']:.3f}"
    )
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved -> {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def run(sigma0=DEFAULT_SIGMA0, tau0=DEFAULT_TAU0, alpha=DEFAULT_ALPHA,
        save=True):
    """Run the two-stage Bergomi parameter calibration."""
    logger.info("=" * 70)
    logger.info("  BERGOMI PARAMETER CALIBRATION (Wang 2017 §5.2.3)")
    logger.info("=" * 70)
    logger.info(f"  Vol-of-vol benchmark: sigma0={sigma0}, tau0={tau0}, alpha={alpha}")

    # ---- Stage 1 target ----
    T1 = T_GRID_VOLOFVOL
    target_volvol = vol_of_vol_benchmark(T1, sigma0=sigma0, tau0=tau0, alpha=alpha)
    logger.info(f"  Stage 1 maturities: {T1.tolist()}")

    stage1 = calibrate_stage1(T1, target_volvol)
    logger.info(f"  Stage 1: nu={stage1['nu']:.4f}  theta={stage1['theta']:.4f}  "
                f"k1={stage1['kappa1']:.4f}  k2={stage1['kappa2']:.4f}  "
                f"rho12={stage1['rho12']:.4f}")
    model_volvol = vol_of_vol_model(
        T1, stage1["nu"], stage1["theta"],
        stage1["kappa1"], stage1["kappa2"], stage1["rho12"],
    )
    volvol_rmse = float(np.sqrt(np.mean((model_volvol - target_volvol) ** 2)))
    logger.info(f"  Vol-of-vol fit RMSE: {volvol_rmse * 100:.3f} %")

    # ---- Stage 2 target: SSVI-derived empirical skew ----
    iv_surface = np.load(IV_DIR / "arrays" / "iv_surface.npy")
    log_m_grid = np.load(IV_DIR / "arrays" / "log_m_grid.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    # Use SSVI maturities >= 3 months: at very short maturities the empirical
    # skew is steep and the order-1 Bergomi-Guyon expansion underestimates it
    # (Wang 2017 observes the same — short-end skew needs higher-order terms).
    T2 = ttm_grid[(ttm_grid >= 0.25) & (ttm_grid <= 2.0)]
    if len(T2) > 25:
        T2 = T2[np.linspace(0, len(T2) - 1, 25, dtype=int)]
    target_skew = empirical_atmf_skew(iv_surface, log_m_grid, ttm_grid, T2)
    logger.info(f"  Stage 2 maturities: {T2.shape[0]} points, "
                f"T in [{T2[0]:.3f}, {T2[-1]:.3f}]")
    logger.info(f"  Empirical skew range: [{target_skew.min():+.4f}, "
                f"{target_skew.max():+.4f}]  median={np.median(target_skew):+.4f}")

    stage2 = calibrate_stage2(T2, target_skew, stage1)
    # Diagnostic: chi parametrises validity of (rho1, rho2, rho12) as a
    # correlation matrix. chi == ±1 means the optimiser sat at the boundary
    # of the valid set — implies the order-1 skew formula needed an extreme
    # correlation structure to match the target. Print full precision so we
    # can distinguish "chi = 1.0000" from "chi = 0.9998".
    chi_val = stage2["chi"]
    chi_at_bound = abs(abs(chi_val) - 1.0) < 1e-6
    logger.info(f"  Stage 2: rho1={stage2['rho1']:.6f}  rho2={stage2['rho2']:.6f}  "
                f"chi={chi_val:.6f}{' [boundary]' if chi_at_bound else ''}")
    # Sanity-check the full-precision rho2 derived from the saturated chi:
    derived = derive_rho2(stage2["rho1"], chi_val, stage1["rho12"])
    logger.info(f"  Stage 2: derived rho2 = rho12*rho1 + chi*sqrt(1-rho12^2)*sqrt(1-rho1^2) "
                f"= {derived:.6f}  (delta vs stored: {derived - stage2['rho2']:+.2e})")

    model_skew = skew_order1_model(
        T2, stage1["nu"], stage1["theta"],
        stage1["kappa1"], stage1["kappa2"], stage1["rho12"],
        stage2["rho1"], stage2["rho2"],
    )
    skew_rmse = float(np.sqrt(np.mean((model_skew - target_skew) ** 2)))
    logger.info(f"  ATMF skew fit RMSE: {skew_rmse:.5f}")

    # ---- Sanity / boundary diagnostics ----
    notes = []
    if abs(stage1["nu"] - 0.5) < 1e-3 or abs(stage1["nu"] - 3.0) < 1e-3:
        notes.append("nu hit Stage 1 bound")
    if abs(stage1["theta"] - 0.1) < 1e-3 or abs(stage1["theta"] - 0.5) < 1e-3:
        notes.append("theta hit Stage 1 bound")
    if abs(stage1["kappa1"] - 3.0) < 1e-3 or abs(stage1["kappa1"] - 10.0) < 1e-3:
        notes.append("kappa1 hit Stage 1 bound")
    if abs(stage1["kappa2"] - 0.05) < 1e-3 or abs(stage1["kappa2"] - 0.6) < 1e-3:
        notes.append("kappa2 hit Stage 1 bound")
    if abs(stage2["rho1"] - (-0.99)) < 1e-3 or abs(stage2["rho1"] - 0.0) < 1e-3:
        notes.append("rho1 hit Stage 2 bound")
    if abs(abs(stage2["chi"]) - 1.0) < 1e-6:
        notes.append(f"chi saturated at {stage2['chi']:+.0f} (rho2 on validity boundary)")
    if not notes:
        notes.append("no boundary hits; calibration looks healthy")

    out = {
        "nu": stage1["nu"], "theta": stage1["theta"],
        "kappa1": stage1["kappa1"], "kappa2": stage1["kappa2"],
        "rho12": stage1["rho12"],
        "rho1": stage2["rho1"], "rho2": stage2["rho2"],
        "stage1_objective": stage1["objective"],
        "stage2_objective": stage2["objective"],
        "vol_of_vol_fit_rmse": volvol_rmse,
        "atmf_skew_fit_rmse": skew_rmse,
        "calibration_notes": "; ".join(notes),
        "benchmark": {"sigma0": sigma0, "tau0": tau0, "alpha": alpha},
    }

    logger.info("=" * 70)
    logger.info("  CALIBRATED BERGOMI PARAMETERS")
    logger.info("=" * 70)
    for k in ["nu", "theta", "kappa1", "kappa2", "rho12", "rho1", "rho2"]:
        logger.info(f"    {k:<7} = {out[k]:+.4f}")
    logger.info(f"    notes: {out['calibration_notes']}")
    logger.info("=" * 70)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / "bergomi_params.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved Bergomi parameters -> {out_path}")
        plot_volofvol(T1, target_volvol, model_volvol, out,
                       PLOT_DIR / "bergomi_calib_volofvol.png")
        plot_skew(T2, target_skew, model_skew, out,
                   PLOT_DIR / "bergomi_calib_skew.png")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0,
                        help="Vol-of-vol level at reference maturity (default 1.0)")
    parser.add_argument("--tau0", type=float, default=DEFAULT_TAU0,
                        help="Reference maturity in years (default 0.25)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Power-law exponent (default 0.4)")
    args = parser.parse_args()
    run(sigma0=args.sigma0, tau0=args.tau0, alpha=args.alpha)

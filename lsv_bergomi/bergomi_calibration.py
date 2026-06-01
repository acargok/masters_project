#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bergomi Two-Factor Forward Variance — Step 3a (Bergomi)
=========================================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Extracts the initial forward variance curve xi^T_0 from the SSVI surface
via variance swap volatility integration, and fits a linearly mean-reverting
parametric form for numerical stability.

The Bergomi two-factor model (Wang 2017, Bergomi 2015):

    d xi^T_t = (2 nu) xi^T_t alpha_theta [
        (1-theta) e^{-kappa1(T-t)} dW^1_t + theta e^{-kappa2(T-t)} dW^2_t
    ]

    dS/S = (r-q) dt + sigma(S,t) sqrt(xi^t_t) dW^S

Inputs:
    iv_surface/arrays/iv_surface.npy     — SSVI implied vol surface
    iv_surface/arrays/total_var_surface.npy — total variance surface
    iv_surface/arrays/ttm_grid.npy       — time grid
    iv_surface/arrays/log_m_grid.npy     — log-moneyness grid
    lsv_bergomi/data/bergomi_params.json — model parameters

Outputs:
    lsv_bergomi/data/fwd_var_fit.json    — fitted NSS parameters
                                            (nss_beta_{0..3}, nss_tau_{1,2}, nss_rmse)
    lsv_bergomi/arrays/fwd_var_curve.npy — xi^T_0 on ttm_grid (analytic NSS derivative)
    lsv_bergomi/arrays/vs_vol_curve.npy  — variance swap vol on ttm_grid
    lsv_bergomi/plots/fwd_var_curve.png  — diagnostic plot
"""

# ===== IMPORTS =====
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

from bergomi_calib_config import (
    VS_METHOD_DEFAULT, _NSS_T_SMALL,
    ROOT, IV_DIR, DUPIRE_DIR, BERGOMI_DIR, DATA_DIR, PLOT_DIR, ARRAY_DIR,
)
from vs_vol_extraction import *
from nss_curve import *

warnings.filterwarnings("ignore")

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bergomi_calibration")


# =============================================================================
# Plotting
# =============================================================================

def plot_fwd_var(ttm_grid, vs_vol, vs_vol_fitted, fwd_var):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(ttm_grid, vs_vol, "o", ms=4, alpha=0.7, label="SSVI-derived")
    ax.plot(ttm_grid, vs_vol_fitted, "-", lw=2, label="Parametric fit")
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel("VS Volatility")
    ax.set_title("Variance Swap Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ttm_grid, fwd_var, "-", lw=2, color="darkorange")
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel(r"$\xi^T_0$")
    ax.set_title("Initial Forward Variance Curve")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(ttm_grid, np.sqrt(fwd_var), "-", lw=2, color="green")
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel(r"$\sqrt{\xi^T_0}$")
    ax.set_title("Forward Volatility")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / "fwd_var_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved forward variance plot -> {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def plot_vs_vol_nss_comparison(ttm_grid, vs_vol, fit_legacy, fit_nss,
                                legacy_params, nss_params, out_path):
    """Side-by-side comparison of the three-parameter fit and the NSS fit."""
    legacy_rmse = float(np.sqrt(np.mean((fit_legacy - vs_vol) ** 2)))
    nss_rmse    = float(np.sqrt(np.mean((fit_nss    - vs_vol) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              sharex=True, sharey=True)
    fig.suptitle("Variance swap volatility: parametric vs NSS",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(ttm_grid, vs_vol, "o", ms=4, alpha=0.75, color="black",
            label="SSVI-derived (Carr-Madan)")
    ax.plot(ttm_grid, fit_legacy, "-", lw=1.8, color="#d62728",
            label=(rf"Legacy 3-param fit"
                   rf"  $z_1$={legacy_params['z1']:.3f}, "
                   rf"$z_2$={legacy_params['z2']:.3f}, "
                   rf"$z_3$={legacy_params['z3']:.3f}"))
    ax.set_title(f"Monotone three-parameter (Wang eq. 5.2)\n"
                 f"RMSE = {legacy_rmse * 1e4:.1f} bp")
    ax.set_xlabel("TTM (years)"); ax.set_ylabel("VS volatility")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(ttm_grid, vs_vol, "o", ms=4, alpha=0.75, color="black",
            label="SSVI-derived (Carr-Madan)")
    ax.plot(ttm_grid, fit_nss, "-", lw=1.8, color="#1f77b4",
            label=(rf"NSS fit"
                   rf"  $\beta_0$={nss_params['nss_beta_0']:.3f}, "
                   rf"$\beta_1$={nss_params['nss_beta_1']:+.3f}, "
                   rf"$\beta_2$={nss_params['nss_beta_2']:+.3f}, "
                   rf"$\beta_3$={nss_params['nss_beta_3']:+.3f}, "
                   rf"$\tau_1$={nss_params['nss_tau_1']:.3f}, "
                   rf"$\tau_2$={nss_params['nss_tau_2']:.3f}"))
    ax.set_title(f"Nelson-Siegel-Svensson (Svensson 1994)\n"
                 f"RMSE = {nss_rmse * 1e4:.1f} bp")
    ax.set_xlabel("TTM (years)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved NSS comparison plot -> {out_path}")


def plot_vs_vol_comparison(ttm_grid, vs_proxy, vs_carr_madan, out_path):
    """Side-by-side comparison plot of proxy vs Carr-Madan VS vol."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ttm_grid, vs_proxy, "o-", ms=4, alpha=0.7, color="grey",
            label="Proxy (smile-avg)")
    ax.plot(ttm_grid, vs_carr_madan, "s-", ms=4, alpha=0.85, color="#1f77b4",
            label="Carr-Madan replication")
    ax.set_xlabel("TTM (years)"); ax.set_ylabel("VS volatility")
    ax.set_title("VS volatility curves: proxy vs Carr-Madan")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    rel_diff = (vs_carr_madan - vs_proxy) / np.maximum(vs_proxy, 1e-6) * 100
    ax.plot(ttm_grid, rel_diff, "-", lw=2, color="#d62728")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("TTM (years)"); ax.set_ylabel("(CM − proxy) / proxy  (%)")
    ax.set_title("Relative difference")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved VS vol comparison plot -> {out_path}")


def run(method=VS_METHOD_DEFAULT):
    logger.info("=" * 60)
    logger.info(f"STEP 3a (Bergomi): Forward Variance Extraction  method={method}")
    logger.info("=" * 60)

    # Load SSVI surface
    iv_surface = np.load(IV_DIR / "arrays" / "iv_surface.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    log_m_grid = np.load(IV_DIR / "arrays" / "log_m_grid.npy")
    fwd_curve = np.load(IV_DIR / "arrays" / "forward_curve.npy")

    logger.info(f"IV surface: {iv_surface.shape}, TTM: [{ttm_grid[0]:.4f}, {ttm_grid[-1]:.4f}]")
    logger.info(f"Forward curve: {fwd_curve.shape}, F range "
                f"[{fwd_curve.min():.2f}, {fwd_curve.max():.2f}]")

    # Compute both methods (always — used for the comparison plot regardless
    # of which one feeds the production pipeline).
    vs_proxy = extract_vs_vol_proxy(iv_surface, log_m_grid, ttm_grid)
    vs_carr_madan = extract_vs_vol_carr_madan(iv_surface, log_m_grid,
                                                ttm_grid, fwd_curve)
    rel_diff = (vs_carr_madan - vs_proxy) / np.maximum(vs_proxy, 1e-6)
    logger.info(f"VS vol proxy      range: [{vs_proxy.min():.4f}, {vs_proxy.max():.4f}]")
    logger.info(f"VS vol Carr-Madan range: [{vs_carr_madan.min():.4f}, {vs_carr_madan.max():.4f}]")
    logger.info(f"Relative diff (CM-proxy)/proxy: mean={rel_diff.mean()*100:+.2f}%  "
                f"max={rel_diff.max()*100:+.2f}%  min={rel_diff.min()*100:+.2f}%")

    plot_vs_vol_comparison(ttm_grid, vs_proxy, vs_carr_madan,
                            PLOT_DIR / "vs_vol_comparison.png")

    if method == "carr_madan":
        vs_vol = vs_carr_madan
    elif method == "proxy":
        vs_vol = vs_proxy
    else:
        raise ValueError(f"Unknown VS method: {method!r}")

    # ---- NSS parametric fit ----------------------------------------------
    # Also fit the three-parameter form for a side-by-side comparison plot,
    # but route the pipeline through the NSS form.
    legacy_params, vs_vol_fitted_legacy = fit_vs_vol_parametric_legacy(
        ttm_grid, vs_vol)
    legacy_rmse = float(np.sqrt(np.mean(
        (vs_vol_fitted_legacy - vs_vol) ** 2)))
    logger.info(f"Legacy 3-param fit RMSE: {legacy_rmse:.6f}  ({legacy_rmse * 1e4:.1f} bp)")

    fit_params, vs_vol_fitted = fit_vs_vol_nss(ttm_grid, vs_vol)
    fit_params["vs_method"] = method
    fit_params["legacy_rmse"] = legacy_rmse

    # ---- Forward variance: analytic NSS derivative -----------------------
    fwd_var = nss_fwd_variance(
        ttm_grid,
        fit_params["nss_beta_0"], fit_params["nss_beta_1"],
        fit_params["nss_beta_2"], fit_params["nss_beta_3"],
        fit_params["nss_tau_1"],  fit_params["nss_tau_2"],
    )
    fwd_var = np.maximum(fwd_var, 1e-6)
    logger.info(f"Forward variance range: [{fwd_var.min():.6f}, {fwd_var.max():.6f}]")
    logger.info(f"Forward vol range: [{np.sqrt(fwd_var.min()):.4f}, {np.sqrt(fwd_var.max()):.4f}]")

    # ---- NSS vs legacy comparison plot -----------------------------------
    plot_vs_vol_nss_comparison(
        ttm_grid, vs_vol, vs_vol_fitted_legacy, vs_vol_fitted,
        legacy_params, fit_params,
        PLOT_DIR / "vs_vol_nss_comparison.png")

    # ---- Save -------------------------------------------------------------
    with open(DATA_DIR / "fwd_var_fit.json", "w") as f:
        json.dump(fit_params, f, indent=2)
    np.save(ARRAY_DIR / "fwd_var_curve.npy", fwd_var)
    np.save(ARRAY_DIR / "vs_vol_curve.npy", vs_vol)
    np.save(ARRAY_DIR / "vs_vol_fitted.npy", vs_vol_fitted)
    np.save(ARRAY_DIR / "vs_vol_proxy.npy", vs_proxy)
    np.save(ARRAY_DIR / "vs_vol_carr_madan.npy", vs_carr_madan)
    logger.info(f"Saved forward variance artifacts -> {ARRAY_DIR}/")

    # Plot
    plot_fwd_var(ttm_grid, vs_vol, vs_vol_fitted, fwd_var)

    return fit_params, fwd_var, vs_vol


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["carr_madan", "proxy"],
                        default=VS_METHOD_DEFAULT)
    args = parser.parse_args()
    run(method=args.method)

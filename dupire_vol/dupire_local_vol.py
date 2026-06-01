#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dupire Local Volatility Surface — Step 2 (SSVI / Forward-Based)
================================================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Computes the Dupire local volatility surface from the SSVI total variance
surface (Step 1).

Mathematical framework
----------------------
Gatheral (2004) total-variance representation. Given total implied variance
w(k, T) = σ_BS(k,T)² · T in forward log-moneyness k = log(K / F(0,T)):

    σ²_loc(k, T) = (∂w/∂T) / g(k, T)

with the Gatheral risk-neutral density factor:

    g(k, T) = [1 - k·(∂w/∂k)/(2w)]²  -  (∂w/∂k)²/4 · (1/4 + 1/w)
              + (∂²w/∂k²)/2

Properties enforced by SSVI:
  ∂w/∂T ≥ 0        ⟺  no calendar-spread arbitrage (monotone θ(T))
  g ≥ 0             ⟺  no butterfly arbitrage
  σ²_loc = ∂w/∂T/g ≥ 0  follows from both

Working directly in w and k = log(K/F) space folds carry/dividend into F(0,T)
and avoids any call-price conversion.

Monte Carlo SDE
---------------
Under the risk-neutral measure:

    dS_t = (r − q_eff(t)) S_t dt + σ_loc(log(S_t/F(0,t)), t) S_t dW_t

where q_eff(t) is the effective dividend yield interpolated from the per-TTM
q_eff_grid (put-call parity). The log-moneyness argument log(S_t/F(0,t)) is
consistent with the k-grid used to build the surface.

Inputs (from iv_surface/ Step 1 outputs)
-----------------------------------------
    arrays/total_var_surface.npy — 2D total variance surface w(k, T)  (n_k × n_T)
    arrays/iv_surface.npy        — 2D implied vol surface σ(k, T)     (n_k × n_T)
    arrays/ttm_grid.npy          — 1D TTM grid (uniform)
    arrays/log_m_grid.npy        — 1D forward log-moneyness grid k = log(K/F) (uniform)
    arrays/q_eff_grid.npy        — per-TTM effective dividend yield
    arrays/forward_curve.npy     — [[T_1, F_1], ..., [T_n, F_n]] forward curve
    data/spx_iv_data.csv         — option data (has fwd_log_m, total_var columns)
    data/implied_forwards.csv    — per-expiry implied forwards

Outputs (to dupire_vol/)
------------------------
    arrays/local_vol_surface.npy  — 2D local vol surface  (n_k × n_T)
    arrays/local_vol_mask.npy     — boolean mask of reliable grid points
    data/local_vol_surface.csv    — local vol surface as CSV
    data/repricing_errors.csv     — MC repricing error report
    data/market_params.json       — saved market parameters (S, r, q)
    data/dupire_diagnostics.json  — stability diagnostics
    plots/local_vol_surface_3d.png
    plots/iv_vs_local_vol.png
    plots/repricing_validation.png
    plots/mc_vs_vanilla_all.png
"""

# ===== IMPORTS =====
import json
import logging
import os
import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import interpolate, optimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===== MODULE IMPORTS =====
from config import *
from io_loaders import *
from dupire_math import *
from monte_carlo import *
from plotting import *

# ===========================================================================
# SECTION 8: MAIN PIPELINE
# ===========================================================================

def main():
    """
    End-to-end pipeline for the Dupire local volatility surface.

    Steps
    -----
    1. Load Step 1 outputs (total variance surface, forward curve, option data)
       and fetch current market parameters.
    2. Compute ∂w/∂T, ∂w/∂k, ∂²w/∂k² on the uniform (k, T) grid.
    3. Compute the Gatheral density factor g(k, T).
    4. Apply σ²_loc = (∂w/∂T) / g to get the Dupire local variance.
    5. Plot local vol surface and side-by-side comparison with IV surface.
    6. Validate by Monte Carlo repricing under the Dupire SDE (Checkpoint 1).
    7. Save all outputs.

    Returns
    -------
    tuple
        (local_vol, mask, result_df) for use in downstream steps (LSV).
    """
    logger.info("=" * 60)
    logger.info("  DUPIRE LOCAL VOLATILITY SURFACE — STEP 2 (SSVI / FORWARD-BASED)")
    logger.info("=" * 60)

    for d in [DIR_DATA, DIR_PLOTS, DIR_ARRAYS]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ 1/6
    logger.info("\n[1/6]  Loading Step 1 outputs and market parameters...")
    (tv_surface, iv_surface, ttm_grid, log_m_grid,
    df, S, fwd_curve, q_eff_grid) = load_iv_surface()

    step1_params = load_step1_market_params()
    r = float(step1_params["r"])

    if q_eff_grid is not None:
        q = q_eff_grid
        q_scalar = float(np.median(q_eff_grid))
        logger.info(f"  Using Step 1 per-TTM q_eff from implied forwards (median={q_scalar:.4f})")
    else:
        q_scalar = float(step1_params["q"])
        q = q_scalar
        logger.info(f"  Using Step 1 constant q = {q_scalar:.4f}")

    params = {"S": S, "r": r, "q": q_scalar, "date": str(date.today())}
    with open(os.path.join(DIR_DATA, "market_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"  Step 1 market params reused: S={S:.2f}, r={r:.4f}, q={q_scalar:.4f}")

    # ------------------------------------------------------------------ 2/6
    logger.info("\n[2/6]  Computing total variance partial derivatives "
                "(∂w/∂T, ∂w/∂k, ∂²w/∂k²)...")
    dw_dT, dw_dk, d2w_dk2 = compute_tv_derivatives(tv_surface, log_m_grid, ttm_grid)

    # ------------------------------------------------------------------ 3/6
    logger.info("\n[3/6]  Computing Gatheral density g(k,T)...")
    g = compute_gatheral_g(tv_surface, dw_dk, d2w_dk2, log_m_grid)

    # ------------------------------------------------------------------ 4/6
    logger.info("\n[4/6]  Applying Dupire formula σ²_loc = (∂w/∂T) / g...")
    local_vol, mask, diagnostics = compute_dupire_local_vol(tv_surface, dw_dT, g)
    #local_vol -= 0.01 #acar

    np.save(os.path.join(DIR_ARRAYS, "local_vol_surface.npy"), local_vol)
    np.save(os.path.join(DIR_ARRAYS, "local_vol_mask.npy"), mask)
    logger.info(f"  Saved: {DIR_ARRAYS}/local_vol_surface.npy, local_vol_mask.npy")

    with open(os.path.join(DIR_DATA, "dupire_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # ------------------------------------------------------------------ 5/6
    logger.info("\n[5/6]  Plotting surfaces...")
    plot_local_vol_surface(local_vol, ttm_grid, log_m_grid, mask, S)
    plot_iv_vs_local_vol(iv_surface, local_vol, ttm_grid, log_m_grid, S)

    # ------------------------------------------------------------------ 6/6
    logger.info("\n[6/6]  Monte Carlo repricing validation (Checkpoint 1)...")
    result_df = monte_carlo_reprice(
        df, local_vol, log_m_grid, ttm_grid, S, r, q, fwd_curve
    )
    result_df.to_csv(os.path.join(DIR_DATA, "repricing_errors.csv"), index=False)
    plot_repricing_validation(result_df)
    plot_mc_vs_vanilla(result_df)

    # Save local vol as CSV (human-readable)
    lv_df = pd.DataFrame(
        local_vol,
        index=np.round(log_m_grid, 6),
        columns=np.round(ttm_grid, 6)
    )
    lv_df.index.name = "fwd_log_moneyness_k"
    lv_df.to_csv(os.path.join(DIR_DATA, "local_vol_surface.csv"))

    # ------------------------------------------------------------------ Done
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 2 COMPLETE — Dupire local vol surface computed and validated")
    logger.info("=" * 60)
    logger.info(f"  Formula: σ²_loc = (∂w/∂T) / g(k,T)  [Gatheral total-variance form]")
    logger.info(f"  Coordinate: k = log(K/F(0,T))  [forward log-moneyness, consistent with SSVI]")
    logger.info(f"  MC SDE: dS = (r-q_eff(t))·S dt + σ_loc(log(S/F(0,t)),t)·S dW")
    logger.info(f"  {DIR_ARRAYS}/local_vol_surface.npy   local vol surface")
    logger.info(f"  {DIR_ARRAYS}/local_vol_mask.npy      reliability mask")
    logger.info(f"  {DIR_DATA}/local_vol_surface.csv     local vol as CSV")
    logger.info(f"  {DIR_DATA}/repricing_errors.csv      MC repricing report")
    logger.info(f"  {DIR_DATA}/market_params.json        saved (S, r, q)")
    logger.info(f"  {DIR_DATA}/dupire_diagnostics.json   stability diagnostics")
    logger.info(f"  {DIR_PLOTS}/local_vol_surface_3d.png")
    logger.info(f"  {DIR_PLOTS}/iv_vs_local_vol.png")
    logger.info(f"  {DIR_PLOTS}/repricing_validation.png")
    logger.info(f"  {DIR_PLOTS}/mc_vs_vanilla_all.png")

    return local_vol, mask, result_df


# ===== EXECUTE =====
if __name__ == "__main__":
    main()

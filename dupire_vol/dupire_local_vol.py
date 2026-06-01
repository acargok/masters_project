#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dupire Local Volatility Surface — Step 2 (SSVI / forward-based).

Builds the Dupire local vol surface from the SSVI total variance w(k,T) of
Step 1, then MC-validates it (LSV thesis, Imperial College London).

Gatheral (2004) total-variance form, k = log(K/F(0,T)):
    σ²_loc(k,T) = (∂w/∂T) / g(k,T)
    g = [1 − k·w_k/(2w)]² − (w_k²/4)·(1/4 + 1/w) + w_kk/2
SSVI enforces ∂w/∂T ≥ 0 (no calendar arb), g ≥ 0 (no butterfly), so
σ²_loc ≥ 0. Working in w, k folds carry/div into F(0,T), no call-price step.

MC SDE (risk-neutral): dS = (r − q_eff(t)) S dt + σ_loc(log(S/F(0,t)), t) S dW,
q_eff(t) interpolated from per-TTM q_eff_grid (put-call parity).

Inputs:  iv_surface/ arrays (total_var_surface, iv_surface, ttm_grid,
         log_m_grid, q_eff_grid, forward_curve) + data CSVs.
Outputs: dupire_vol/ arrays (local_vol_surface, local_vol_mask), data
         (local_vol_surface.csv, repricing_errors.csv, market_params.json,
         dupire_diagnostics.json), and plots.
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from config import *
from io_loaders import *
from dupire_math import *
from monte_carlo import *
from plotting import *

# SECTION 8: Main pipeline

def main():
    """
    End-to-end Dupire local vol pipeline.

    Load Step 1 outputs + market params -> FD partials -> Gatheral g ->
    σ²_loc = (∂w/∂T)/g -> plots -> MC repricing validation (Checkpoint 1) ->
    save. Returns (local_vol, mask, result_df) for downstream LSV.
    """
    logger.info("=" * 60)
    logger.info("  DUPIRE LOCAL VOLATILITY SURFACE — STEP 2 (SSVI / FORWARD-BASED)")
    logger.info("=" * 60)

    for d in [DIR_DATA, DIR_PLOTS, DIR_ARRAYS]:
        os.makedirs(d, exist_ok=True)

    # 1/6
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

    # 2/6
    logger.info("\n[2/6]  Computing total variance partial derivatives "
                "(∂w/∂T, ∂w/∂k, ∂²w/∂k²)...")
    dw_dT, dw_dk, d2w_dk2 = compute_tv_derivatives(tv_surface, log_m_grid, ttm_grid)

    # 3/6
    logger.info("\n[3/6]  Computing Gatheral density g(k,T)...")
    g = compute_gatheral_g(tv_surface, dw_dk, d2w_dk2, log_m_grid)

    # 4/6
    logger.info("\n[4/6]  Applying Dupire formula σ²_loc = (∂w/∂T) / g...")
    local_vol, mask, diagnostics = compute_dupire_local_vol(tv_surface, dw_dT, g)
    #local_vol -= 0.01 #acar

    np.save(os.path.join(DIR_ARRAYS, "local_vol_surface.npy"), local_vol)
    np.save(os.path.join(DIR_ARRAYS, "local_vol_mask.npy"), mask)
    logger.info(f"  Saved: {DIR_ARRAYS}/local_vol_surface.npy, local_vol_mask.npy")

    with open(os.path.join(DIR_DATA, "dupire_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # 5/6
    logger.info("\n[5/6]  Plotting surfaces...")
    plot_local_vol_surface(local_vol, ttm_grid, log_m_grid, mask, S)
    plot_iv_vs_local_vol(iv_surface, local_vol, ttm_grid, log_m_grid, S)

    # 6/6
    logger.info("\n[6/6]  Monte Carlo repricing validation (Checkpoint 1)...")
    result_df = monte_carlo_reprice(
        df, local_vol, log_m_grid, ttm_grid, S, r, q, fwd_curve
    )
    result_df.to_csv(os.path.join(DIR_DATA, "repricing_errors.csv"), index=False)
    plot_repricing_validation(result_df)
    plot_mc_vs_vanilla(result_df)

    # Save local vol as CSV
    lv_df = pd.DataFrame(
        local_vol,
        index=np.round(log_m_grid, 6),
        columns=np.round(ttm_grid, 6)
    )
    lv_df.index.name = "fwd_log_moneyness_k"
    lv_df.to_csv(os.path.join(DIR_DATA, "local_vol_surface.csv"))

    # Done
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


if __name__ == "__main__":
    main()

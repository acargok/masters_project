# -*- coding: utf-8 -*-
"""Loaders for Step 1 outputs (SECTION 1)."""

import json
import logging
import os

import numpy as np
import pandas as pd

from config import *

logger = logging.getLogger(__name__)


# SECTION 1: Load Step 1 outputs

def load_iv_surface() -> tuple:
    """
    Load SSVI surface outputs from Step 1 (iv_surface/).

    Returns (tv_surface w(k,T)=σ²·T (n_k,n_T), iv_surface, ttm_grid (uniform),
    log_m_grid k=log(K/F) (uniform), df option data, S spot recovered as
    strike/moneyness, fwd_curve [[T,F]] (n_T,2), q_eff_grid per-TTM div yield
    or None).
    """
    tv_surface  = np.load(os.path.join(IV_DIR_ARRAYS, "total_var_surface.npy"))
    iv_surface  = np.load(os.path.join(IV_DIR_ARRAYS, "iv_surface.npy"))
    ttm_grid    = np.load(os.path.join(IV_DIR_ARRAYS, "ttm_grid.npy"))
    log_m_grid  = np.load(os.path.join(IV_DIR_ARRAYS, "log_m_grid.npy"))
    # forward_curve.npy is F(T_i), indexed like ttm_grid
    fwd_prices  = np.load(os.path.join(IV_DIR_ARRAYS, "forward_curve.npy"))
    # Normalise to (n_T, 2) [[T, F]] for the interpolator.
    if fwd_prices.ndim == 1:
        fwd_curve = np.column_stack([ttm_grid, fwd_prices])
    else:
        fwd_curve = fwd_prices

    q_eff_path = os.path.join(IV_DIR_ARRAYS, "q_eff_grid.npy")
    if os.path.exists(q_eff_path):
        q_eff_grid = np.load(q_eff_path)
        logger.info(f"Loaded q_eff grid: [{q_eff_grid.min():.4f}, {q_eff_grid.max():.4f}]")
    else:
        q_eff_grid = None
        logger.info("No q_eff_grid.npy — using constant q from Step 1 market_params.json")

    df = pd.read_csv(os.path.join(IV_DIR_DATA, "spx_iv_data.csv"))

    # Derive forward log-moneyness if absent.
    if "fwd_log_m" not in df.columns:
        fwd_df = pd.read_csv(os.path.join(IV_DIR_DATA, "implied_forwards.csv"))
        fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))
        df["fwd_log_m"] = np.log(df["strike"] / df["expiry"].map(fwd_map))
    if "total_var" not in df.columns:
        df["total_var"] = df["iv"] ** 2 * df["ttm"]

    # S = strike / moneyness (exact for all rows)
    S = float((df["strike"] / df["moneyness"]).median())

    logger.info(f"Loaded TV surface: {tv_surface.shape}, "
                f"w range [{tv_surface.min():.6f}, {tv_surface.max():.6f}]")
    logger.info(f"Loaded IV surface: {iv_surface.shape}, "
                f"IV range [{iv_surface.min():.4f}, {iv_surface.max():.4f}]")
    logger.info(f"TTM grid: [{ttm_grid[0]:.4f}, {ttm_grid[-1]:.4f}], {len(ttm_grid)} pts")
    logger.info(f"k = log(K/F) grid: [{log_m_grid[0]:.4f}, {log_m_grid[-1]:.4f}], "
                f"{len(log_m_grid)} pts")
    logger.info(f"Forward curve: F range [{fwd_curve[:,1].min():.2f}, "
                f"{fwd_curve[:,1].max():.2f}]  ({len(fwd_curve)} points)")
    logger.info(f"Recovered spot price: S = {S:.2f}")

    return tv_surface, iv_surface, ttm_grid, log_m_grid, df, S, fwd_curve, q_eff_grid
def load_step1_market_params() -> dict:
    """Load Step 1 market parameters (market_params.json)."""
    path = os.path.join(IV_DIR_DATA, "market_params.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing Step 1 market params: {path}"
        )
    with open(path, "r") as f:
        params = json.load(f)
    return params

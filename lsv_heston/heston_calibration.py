#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heston calibration — Step 3a (Master's Thesis, Imperial College London).

Calibrates Heston params (kappa, theta, xi, rho, V0) to the Step-1 IV surface
via the characteristic function / Carr-Madan-Lewis pricing, minimising weighted
SSE of IV errors with differential_evolution. Feller 2*kappa*theta > xi^2 as
soft penalty.

Inputs:
    iv_surface/data/spx_iv_data.csv    — option data with market IVs
    dupire_vol/data/market_params.json — S, r, q
Outputs:
    data/heston_params.json — calibrated params
    plots/heston_fit.png    — model vs market IV
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize
from scipy.stats import norm

from heston_config import *
from heston_pricing import *

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heston_calibration")


# --- Calibration objective ---

def calibration_objective(params, S, r, q, K_arr, T_arr, market_prices, vegas,
                          opt_type_arr):
    """
    SSE of IV-approx errors plus soft Feller penalty.

    Uses iv_error ~ price_error/vega (first order) to skip BS inversion in the
    loop; equivalent to IV-space calibration with balanced strike weight.
    params: [kappa, theta, xi, rho, V0]. market_prices: BS call/put per
    opt_type_arr. vegas: IV-approx denominator.
    """
    kappa, theta, xi, rho, V0 = params

    try:
        model_prices = heston_call_price_vectorised(
            S, K_arr, T_arr, r, q, kappa, theta, xi, rho, V0,
            N_quad=N_QUAD_OPT, upper_limit=UPPER_LIMIT_OPT
        )
        # Calls -> puts via put-call parity where needed
        put_mask = opt_type_arr == "put"
        if put_mask.any():
            model_prices = model_prices.copy()
            model_prices[put_mask] -= (
                S * np.exp(-q * T_arr[put_mask]) - K_arr[put_mask] * np.exp(-r * T_arr[put_mask])
            )

        if np.any(~np.isfinite(model_prices)):
            return 1e10

        # iv_error ~ price_error / vega
        iv_approx_errors = (model_prices - market_prices) / np.maximum(vegas, VEGA_FLOOR)
        sse = np.sum(iv_approx_errors**2)

    except Exception:
        return 1e10

    # Soft Feller penalty (2*kappa*theta > xi^2)
    feller_gap = xi**2 - 2.0 * kappa * theta
    if feller_gap > 0:
        sse += FELLER_PENALTY * feller_gap**2

    return sse


# --- Main calibration routine ---

def load_market_data():
    """Load Step 1/2 market data; returns dict (S, r, q, df, iv_surface,
    ttm_grid, log_m_grid)."""
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    df = pd.read_csv(IV_DIR / "data" / "spx_iv_data.csv")

    iv_surface = np.load(IV_DIR / "arrays" / "iv_surface.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    log_m_grid = np.load(IV_DIR / "arrays" / "log_m_grid.npy")

    logger.info(f"Loaded market data: S={S:.2f}, r={r:.4f}, q={q:.4f}")
    logger.info(f"Option data: {len(df)} records, "
                f"TTM range [{df['ttm'].min():.4f}, {df['ttm'].max():.4f}]")

    return {
        "S": S, "r": r, "q": q, "df": df,
        "iv_surface": iv_surface, "ttm_grid": ttm_grid, "log_m_grid": log_m_grid,
    }


def prepare_calibration_data(market_data, max_options=2000):
    """
    Build calibration set: select OTM options within TTM/IV bounds,
    stratified-subsample over TTM x moneyness if too many, return BS vegas for
    the objective's IV-approx normalisation.

    Returns K_arr, T_arr, iv_market, market_prices, vegas (raw dollar vegas),
    opt_type_arr.
    """
    S, r, q = market_data["S"], market_data["r"], market_data["q"]
    df = market_data["df"].copy()

    # OTM: puts for k<0, calls for k>=0 (preserves left-wing skew)
    fwd_moneyness = df["strike"] / (S * np.exp((r - q) * df["ttm"]))
    otm_mask = (
        ((fwd_moneyness < 1.0) & (df["option_type"] == "put")) |
        ((fwd_moneyness >= 1.0) & (df["option_type"] == "call"))
    )
    df = df[otm_mask].copy()

    # 0.80 <= K/F(T) <= 1.20; drops deep-OTM call wings where the Fourier pricer
    # underflows to ~0 price/vega and BS inversion fails.
    fwd_moneyness = fwd_moneyness[otm_mask]
    df = df[(fwd_moneyness >= 0.80) & (fwd_moneyness <= 1.20)]

    # TTM 2 weeks to 2 years
    df = df[(df["ttm"] >= 0.04) & (df["ttm"] <= 2)]

    df = df[df["iv"].between(0.01, 2.0)]   # valid IV
    df = df.drop_duplicates(subset=["strike", "ttm"])

    logger.info(f"Calibration pool: {len(df)} OTM options after filtering "
                f"({(df['option_type']=='put').sum()} puts, "
                f"{(df['option_type']=='call').sum()} calls)")

    # Stratified subsample over 5x5 TTM x fwd-moneyness grid
    if len(df) > max_options:
        df = df.reset_index(drop=True)
        fwd_m_col = df["strike"] / (S * np.exp((r - q) * df["ttm"]))
        ttm_bins = pd.cut(df["ttm"],   bins=5, labels=False)
        m_bins   = pd.cut(fwd_m_col,   bins=5, labels=False)
        df["_strata"] = ttm_bins.astype(str) + "_" + m_bins.astype(str)
        n_strata   = df["_strata"].nunique()
        per_stratum = max(1, max_options // n_strata)

        sampled = (
            df.groupby("_strata", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=SEED))
        )
        # Top up to max_options from the remainder
        if len(sampled) < max_options:
            remaining = df.loc[~df.index.isin(sampled.index)]
            n_extra = min(max_options - len(sampled), len(remaining))
            if n_extra > 0:
                sampled = pd.concat(
                    [sampled, remaining.sample(n_extra, random_state=SEED)]
                )
        df = sampled.drop(columns="_strata").head(max_options).reset_index(drop=True)
        logger.info(
            f"Stratified subsampled to {len(df)} options "
            f"across {n_strata} TTM×moneyness strata"
        )

    K_arr       = df["strike"].values.astype(np.float64)
    T_arr       = df["ttm"].values.astype(np.float64)
    # `iv` = SSVI-fitted IV (raw quote = `iv_yf`); SSVI is the reference for both
    # IV and BS-price errors throughout.
    iv_market   = df["iv"].values.astype(np.float64)
    opt_type_arr = df["option_type"].values

    # SSVI BS prices from SSVI IVs (call, or put via parity)
    market_prices = np.array([
        bs_call_price(S, K_arr[i], T_arr[i], r, q, iv_market[i])
        if opt_type_arr[i] == "call" else
        bs_call_price(S, K_arr[i], T_arr[i], r, q, iv_market[i])
        - S * np.exp(-q * T_arr[i]) + K_arr[i] * np.exp(-r * T_arr[i])
        for i in range(len(K_arr))
    ])

    # Raw BS dollar vegas (objective's IV-approx denominator)
    vegas = np.array([
        bs_vega(S, K_arr[i], T_arr[i], r, q, iv_market[i])
        for i in range(len(K_arr))
    ])

    return K_arr, T_arr, iv_market, market_prices, vegas, opt_type_arr


def calibrate_heston(market_data, max_options=2000):
    """
    Calibrate Heston to market IVs via differential_evolution (global) then a
    Nelder-Mead polish. Returns params (kappa, theta, xi, rho, V0) + fit diagnostics.
    """
    S, r, q = market_data["S"], market_data["r"], market_data["q"]
    K_arr, T_arr, iv_market, market_prices, vegas, opt_type_arr = prepare_calibration_data(
        market_data, max_options
    )

    logger.info("Starting Heston calibration via differential evolution...")

    # Bounds [kappa, theta, xi, rho, V0]
    bounds = [
        (0.1, 10.0),      # kappa: mean reversion
        (0.005, 0.50),    # theta: long-run var (~7%-70% vol)
        (0.05, 2.0),      # xi: vol of vol
        (-0.99, 0.10),    # rho: spot-vol corr (slight positive allowed)
        (0.005, 0.50),    # V0: initial var
    ]

    result = optimize.differential_evolution(
        calibration_objective,
        bounds,
        args=(S, r, q, K_arr, T_arr, market_prices, vegas, opt_type_arr),
        seed=SEED,
        maxiter=MAX_ITER,
        tol=1e-10,
        atol=1e-10,
        polish=False,    # polished separately below
        workers=N_WORKERS,
        updating="deferred",
        disp=False,
    )

    logger.info(f"Differential evolution converged: fun={result.fun:.6e}, "
                f"nfev={result.nfev}")

    # Nelder-Mead polish
    result_local = optimize.minimize(
        calibration_objective,
        result.x,
        args=(S, r, q, K_arr, T_arr, market_prices, vegas, opt_type_arr),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )

    logger.info(f"Nelder-Mead polish: fun={result_local.fun:.6e}, "
                f"nfev={result_local.nfev}")

    # Clip back to bounds (Nelder-Mead is unconstrained)
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    clipped = np.clip(result_local.x, lo, hi)
    if not np.allclose(clipped, result_local.x):
        logger.warning(f"Nelder-Mead violated bounds; clipped: "
                       f"{result_local.x} → {clipped}")
    kappa, theta, xi, rho, V0 = clipped

    # Fit diagnostics (higher-accuracy pricing)
    heston_call_prices = heston_call_price_vectorised(
        S, K_arr, T_arr, r, q, kappa, theta, xi, rho, V0,
        N_quad=N_QUAD_DIAG, upper_limit=UPPER_LIMIT_DIAG
    )
    model_prices_diag = heston_call_prices.copy()
    put_mask = opt_type_arr == "put"
    if put_mask.any():
        model_prices_diag[put_mask] -= (
            S * np.exp(-q * T_arr[put_mask]) - K_arr[put_mask] * np.exp(-r * T_arr[put_mask])
        )
    # IV-space diagnostics: invert model prices to IV
    iv_model = np.array([
        bs_implied_vol(
            model_prices_diag[i], S, K_arr[i], T_arr[i], r, q,
            option_type=opt_type_arr[i]
        )
        for i in range(len(K_arr))
    ])
    valid = np.isfinite(iv_model)
    iv_errors = iv_model[valid] - iv_market[valid]
    iv_mae  = float(np.mean(np.abs(iv_errors)))
    iv_rmse = float(np.sqrt(np.mean(iv_errors**2)))
    iv_me   = float(np.mean(iv_errors))

    # Price-space diagnostics vs SSVI BS prices (market_prices = BS(...,iv_ssvi),
    # see prepare_calibration_data).
    price_valid = np.isfinite(model_prices_diag) & (market_prices > 0.01)
    price_err_pct = (100.0 * (model_prices_diag[price_valid] - market_prices[price_valid])
                     / market_prices[price_valid])
    price_mae_pct = float(np.mean(np.abs(price_err_pct)))
    price_me_pct  = float(np.mean(price_err_pct))
    price_rmse_pct = float(np.sqrt(np.mean(price_err_pct ** 2)))

    feller = 2.0 * kappa * theta - xi**2
    feller_satisfied = feller > 0

    params = {
        "kappa": float(kappa),
        "theta": float(theta),
        "xi": float(xi),
        "rho": float(rho),
        "V0": float(V0),
        "feller_value": float(feller),
        "feller_satisfied": bool(feller_satisfied),
        # IV vs SSVI
        "iv_mae": iv_mae,
        "iv_rmse": iv_rmse,
        "iv_me": iv_me,
        # Price vs SSVI (BS price)
        "price_vs_ssvi_mae_pct": price_mae_pct,
        "price_vs_ssvi_me_pct":  price_me_pct,
        "price_vs_ssvi_rmse_pct": price_rmse_pct,
        "n_price_valid": int(price_valid.sum()),
        "n_options": len(K_arr),
        "objective": float(result_local.fun),
    }

    logger.info("=" * 60)
    logger.info("Calibrated Heston Parameters:")
    logger.info(f"  kappa = {kappa:.6f}  (mean reversion)")
    logger.info(f"  theta = {theta:.6f}  (long-run var, vol = {np.sqrt(theta):.4f})")
    logger.info(f"  xi    = {xi:.6f}  (vol of vol)")
    logger.info(f"  rho   = {rho:.6f}  (spot-vol corr)")
    logger.info(f"  V0    = {V0:.6f}  (init var, vol = {np.sqrt(V0):.4f})")
    logger.info(f"  Feller: 2kθ - ξ² = {feller:.6f}  ({'SATISFIED' if feller_satisfied else 'VIOLATED'})")
    logger.info(f"  IV fit vs SSVI:  MAE = {iv_mae*100:.2f} vpts,  RMSE = {iv_rmse*100:.2f} vpts,  "
                f"ME = {iv_me*100:+.2f} vpts  ({valid.sum()}/{len(K_arr)} options)")
    logger.info(f"  Price fit vs SSVI: MAE = {price_mae_pct:.2f}%,  RMSE = {price_rmse_pct:.2f}%,  "
                f"ME = {price_me_pct:+.2f}%")
    logger.info("=" * 60)

    return params, K_arr, T_arr, iv_market, iv_model


# --- Plotting ---

def plot_heston_fit(K_arr, T_arr, iv_market, iv_model, S, r, q, params):
    """
    Plot Heston fit: model vs SSVI IV (4 panels).

    iv_market: SSVI-fitted IV (column `iv`, not raw mid `iv_yf`). iv_model:
    Heston IV, NaN where inversion failed.
    """
    moneyness = K_arr / (S * np.exp((r - q) * T_arr))
    valid = np.isfinite(iv_model)
    iv_err_pts = (iv_model[valid] - iv_market[valid]) * 100.0   # vol pts %

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Heston Calibration — Model vs SSVI IV",
                 fontsize=14, fontweight="bold")

    # (a) model IV vs SSVI IV
    ax = axes[0, 0]
    sc = ax.scatter(iv_market[valid] * 100, iv_model[valid] * 100,
                    c=T_arr[valid], cmap="viridis", alpha=0.7, s=20, edgecolors="none")
    hi = max(iv_market[valid].max(), iv_model[valid].max()) * 100 * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="Perfect fit")
    ax.set_xlim([0, hi])
    ax.set_ylim([0, hi])
    ax.set_xlabel("SSVI IV (%)")
    ax.set_ylabel("Heston IV (%)")
    ax.set_title("(a) Model vs SSVI IV")
    ax.legend(loc="upper left")
    plt.colorbar(sc, ax=ax, label="TTM (years)")

    # (b) IV error histogram
    ax = axes[0, 1]
    ax.hist(iv_err_pts, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("IV Error (model − SSVI, vol pts %)")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) IV Error Distribution (MAE={params['iv_mae']*100:.2f} vpts)")

    # (c) IV error vs fwd moneyness
    ax = axes[1, 0]
    ax.scatter(moneyness[valid], iv_err_pts, c=T_arr[valid], cmap="viridis",
               alpha=0.7, s=20, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fwd Moneyness (K/F)")
    ax.set_ylabel("IV Error (vol pts %)")
    ax.set_title("(c) IV Error vs Fwd Moneyness")

    # (d) IV error vs TTM
    ax = axes[1, 1]
    ax.scatter(T_arr[valid], iv_err_pts, c=moneyness[valid], cmap="coolwarm",
               alpha=0.7, s=20, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel("IV Error (vol pts %)")
    ax.set_title("(d) IV Error vs TTM")

    # Param annotation
    txt = (
        f"κ={params['kappa']:.4f}  θ={params['theta']:.4f}  "
        f"ξ={params['xi']:.4f}  ρ={params['rho']:.4f}  V₀={params['V0']:.4f}\n"
        f"Feller: {'✓' if params['feller_satisfied'] else '✗'}  "
        f"IV MAE={params['iv_mae']*100:.2f} vpts  IV RMSE={params['iv_rmse']*100:.2f} vpts  "
        f"Price MAE={params['price_vs_ssvi_mae_pct']:.2f}%"
    )
    fig.text(0.5, 0.01, txt, ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = PLOT_DIR / "heston_fit.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Heston fit plot → {out_path}")


# --- Entry point ---

def run(max_options=2000):
    """Run the Heston calibration pipeline; returns calibrated params."""
    logger.info("=" * 60)
    logger.info("STEP 3a: Heston Model Calibration")
    logger.info("=" * 60)

    market_data = load_market_data()

    params, K_arr, T_arr, iv_market, iv_model = calibrate_heston(
        market_data, max_options=max_options
    )

    out_path = DATA_DIR / "heston_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved Heston parameters → {out_path}")

    plot_heston_fit(K_arr, T_arr, iv_market, iv_model,
                    market_data["S"], market_data["r"], market_data["q"], params)

    return params


if __name__ == "__main__":
    run()

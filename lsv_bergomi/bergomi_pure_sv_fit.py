
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bergomi Pure-SV Fit Diagnostic — Step 3a''
============================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Bergomi has no semi-analytic vanilla pricer, so we simulate the pure
stochastic-volatility backbone (σ(S,t) ≡ 1, i.e. no LSV leverage) under the
calibrated parameters and reprice the option pool via Monte Carlo.

The 4-panel output plot shows how well the pure Bergomi backbone fits market
IVs before the LSV leverage function is applied. Large residuals here indicate
how much work is left for the leverage function in the particle method step.

Inputs:
    lsv_bergomi/data/bergomi_params.json — calibrated parameters
    lsv_bergomi/arrays/fwd_var_curve.npy — forward variance curve
    dupire_vol/data/market_params.json   — S, r, q
    iv_surface/data/spx_iv_data.csv      — option panel
    iv_surface/arrays/forward_curve.npy  — F(0, T)
    iv_surface/arrays/ttm_grid.npy
    iv_surface/arrays/log_m_grid.npy

Outputs:
    lsv_bergomi/plots/bergomi_fit.png    — 4-panel model vs market IV
    lsv_bergomi/data/bergomi_fit.json    — summary statistics
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm

from pure_sv_config import (
    N_PATHS, STEPS_PER_YEAR, MAX_OPTIONS, SEED,
    ROOT, IV_DIR, DUPIRE_DIR, BERGOMI_DIR, DATA_DIR, PLOT_DIR, ARRAY_DIR,
)
from pure_sv_bs import *
from pure_sv_dynamics import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bergomi_pure_sv_fit")


# =============================================================================
# Plot — model vs market IV
# =============================================================================

def plot_bergomi_fit(K_arr, T_arr, iv_ssvi, iv_model, S, r, q, params,
                     out_path, summary):
    moneyness = K_arr / (S * np.exp((r - q) * T_arr))
    valid = np.isfinite(iv_model)
    iv_err_pts = (iv_model[valid] - iv_ssvi[valid]) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bergomi (pure SV, no leverage) — Model vs SSVI IV",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    sc = ax.scatter(iv_ssvi[valid] * 100, iv_model[valid] * 100,
                    c=T_arr[valid], cmap="viridis", alpha=0.7, s=20,
                    edgecolors="none")
    hi = max(iv_ssvi[valid].max(), iv_model[valid].max()) * 100 * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="Perfect fit")
    ax.set_xlim([0, hi]); ax.set_ylim([0, hi])
    ax.set_xlabel("SSVI IV (%)")
    ax.set_ylabel("Bergomi IV (%)")
    ax.set_title("(a) Model vs SSVI IV")
    ax.legend(loc="upper left")
    plt.colorbar(sc, ax=ax, label="TTM (years)")

    ax = axes[0, 1]
    ax.hist(iv_err_pts, bins=40, edgecolor="black", alpha=0.7, color="#d62728")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("IV Error (model − SSVI, vol pts %)")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) IV Error Distribution (MAE={summary['iv_mae']*100:.2f} vpts)")

    ax = axes[1, 0]
    ax.scatter(moneyness[valid], iv_err_pts, c=T_arr[valid], cmap="viridis",
               alpha=0.7, s=20, edgecolors="none")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Fwd Moneyness (K/F)")
    ax.set_ylabel("IV Error (vol pts %)")
    ax.set_title("(c) IV Error vs Fwd Moneyness")

    ax = axes[1, 1]
    ax.scatter(T_arr[valid], iv_err_pts, c=moneyness[valid], cmap="coolwarm",
               alpha=0.7, s=20, edgecolors="none")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel("IV Error (vol pts %)")
    ax.set_title("(d) IV Error vs TTM")

    txt = (
        f"ν={params['nu']:.4f}  θ={params['theta']:.4f}  "
        f"κ1={params['kappa1']:.4f}  κ2={params['kappa2']:.4f}  "
        f"ρ12={params['rho12']:.4f}  ρ1={params['rho1']:.4f}  ρ2={params['rho2']:.4f}\n"
        f"IV MAE={summary['iv_mae']*100:.2f} vpts  "
        f"IV RMSE={summary['iv_rmse']*100:.2f} vpts  "
        f"IV ME={summary['iv_me']*100:+.2f} vpts  "
        f"({summary['n_valid']}/{summary['n_total']} options)"
    )
    fig.text(0.5, 0.01, txt, ha="center", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Bergomi fit plot -> {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def run(n_paths=N_PATHS, steps_per_year=STEPS_PER_YEAR,
        max_options=MAX_OPTIONS, seed=SEED):
    logger.info("=" * 60)
    logger.info("STEP 3a'' (Bergomi): Pure-SV Fit Diagnostic")
    logger.info("=" * 60)

    # Load market parameters
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    with open(DATA_DIR / "bergomi_params.json") as f:
        bergomi = json.load(f)

    fwd_var = np.load(ARRAY_DIR / "fwd_var_curve.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    fwd_var_interp = interpolate.interp1d(
        ttm_grid, fwd_var, kind="linear",
        bounds_error=False, fill_value=(fwd_var[0], fwd_var[-1]),
    )

    logger.info(f"S={S:.2f}, r={r:.4f}, q={q:.4f}")
    logger.info(f"Bergomi: nu={bergomi['nu']:.4f}, theta={bergomi['theta']:.4f}, "
                f"k1={bergomi['kappa1']:.4f}, k2={bergomi['kappa2']:.4f}")
    logger.info(f"Correlations: rho1={bergomi['rho1']:.4f}, rho2={bergomi['rho2']:.4f}, "
                f"rho12={bergomi['rho12']:.4f}")
    logger.info(f"Forward variance range: [{fwd_var.min():.6f}, {fwd_var.max():.6f}]")

    # Option pool
    df = select_option_pool(S, r, q, max_options=max_options, seed=seed)
    logger.info(f"Option pool: {len(df)} OTM options "
                f"({(df['option_type']=='put').sum()} puts, "
                f"{(df['option_type']=='call').sum()} calls)  "
                f"TTM in [{df['ttm'].min():.3f}, {df['ttm'].max():.3f}]")

    K_arr = df["strike"].values.astype(np.float64)
    T_arr = df["ttm"].values.astype(np.float64)
    # `iv` in spx_iv_data.csv is the SSVI-fitted IV (raw quote = `iv_yf`).
    # We treat it as the SSVI reference and use BS(S, K, T, r, q, iv_ssvi)
    # as the SSVI price benchmark — never the raw market mid.
    iv_ssvi = df["iv"].values.astype(np.float64)
    opt_type_arr = df["option_type"].values

    # Simulate
    dt = 1.0 / steps_per_year
    maturities_required = sorted(set(np.round(T_arr, 6)))
    logger.info(f"Pure-Bergomi MC: {n_paths:,} paths, dt={dt:.5f} "
                f"({len(maturities_required)} distinct maturities)")

    snapshots, step_of = simulate_bergomi_no_leverage(
        S, r, q, bergomi, fwd_var_interp, ttm_grid,
        maturities_required, n_paths, dt, seed,
    )

    # Reprice and invert IVs; also build the SSVI BS price benchmark.
    iv_model = np.full(len(K_arr), np.nan)
    model_prices = np.full(len(K_arr), np.nan)
    ssvi_prices = np.full(len(K_arr), np.nan)
    for i, (K, T, opt) in enumerate(zip(K_arr, T_arr, opt_type_arr)):
        T_key = float(round(T, 6))
        S_T = snapshots[step_of[T_key]]
        if opt == "call":
            payoff = np.maximum(S_T - K, 0.0)
            ssvi_prices[i] = bs_call_price(S, K, T, r, q, iv_ssvi[i])
        else:
            payoff = np.maximum(K - S_T, 0.0)
            ssvi_prices[i] = bs_put_price(S, K, T, r, q, iv_ssvi[i])
        disc = np.exp(-r * T)
        price = disc * payoff.mean()
        model_prices[i] = price
        iv_model[i] = bs_iv(price, S, K, T, r, q, opt)

    valid = np.isfinite(iv_model)
    iv_err = iv_model[valid] - iv_ssvi[valid]
    price_valid = valid & (ssvi_prices > 0.01)
    price_err_pct = (100.0 * (model_prices[price_valid] - ssvi_prices[price_valid])
                     / ssvi_prices[price_valid])
    summary = {
        "n_total": int(len(K_arr)),
        "n_valid": int(valid.sum()),
        # IV errors vs SSVI
        "iv_mae": float(np.mean(np.abs(iv_err))),
        "iv_me":  float(np.mean(iv_err)),
        "iv_rmse": float(np.sqrt(np.mean(iv_err ** 2))),
        "iv_max_abs": float(np.max(np.abs(iv_err))),
        # Price errors vs SSVI BS price (%)
        "n_price_valid": int(price_valid.sum()),
        "price_vs_ssvi_mae_pct": float(np.mean(np.abs(price_err_pct))),
        "price_vs_ssvi_me_pct":  float(np.mean(price_err_pct)),
        "price_vs_ssvi_rmse_pct": float(np.sqrt(np.mean(price_err_pct ** 2))),
        "n_paths": n_paths,
        "dt": float(dt),
        "seed": seed,
    }
    logger.info("=" * 60)
    logger.info("Bergomi pure-SV fit:")
    logger.info(f"  IV MAE  = {summary['iv_mae']*100:.2f} vpts  "
                f"({summary['iv_mae']*10000:.0f} bps)")
    logger.info(f"  IV ME   = {summary['iv_me']*100:+.2f} vpts  "
                f"({summary['iv_me']*10000:+.0f} bps)")
    logger.info(f"  IV RMSE = {summary['iv_rmse']*100:.2f} vpts  "
                f"({summary['iv_rmse']*10000:.0f} bps)")
    logger.info(f"  Price vs SSVI MAE  = {summary['price_vs_ssvi_mae_pct']:.2f}%")
    logger.info(f"  Price vs SSVI ME   = {summary['price_vs_ssvi_me_pct']:+.2f}%")
    logger.info(f"  Price vs SSVI RMSE = {summary['price_vs_ssvi_rmse_pct']:.2f}%")
    logger.info(f"  Valid: {summary['n_valid']}/{summary['n_total']}")
    logger.info("=" * 60)

    with open(DATA_DIR / "bergomi_fit.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary -> {DATA_DIR / 'bergomi_fit.json'}")

    plot_bergomi_fit(K_arr, T_arr, iv_ssvi, iv_model, S, r, q, bergomi,
                     PLOT_DIR / "bergomi_fit.png", summary)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--steps-per-year", type=int, default=STEPS_PER_YEAR)
    parser.add_argument("--max-options", type=int, default=MAX_OPTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(n_paths=args.paths, steps_per_year=args.steps_per_year,
        max_options=args.max_options, seed=args.seed)

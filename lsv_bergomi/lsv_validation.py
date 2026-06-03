#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bergomi LSV validation, Checkpoint 2 (Master's thesis, Imperial).
Validates the calibrated leverage sigma(t,S) by MC-repricing vanillas under
    dS/S = (r-q)dt + sigma(S,t) sqrt(xi^t_t) dW^S
(leverage interpolated from the particle-method surface) and comparing to SSVI
IVs and the Step-2 Dupire baseline.
In: arrays/{leverage_*,fwd_var_curve}.npy, data/bergomi_params.json,
dupire_vol/repricing_errors.csv.
Out: data/{lsv_repricing_errors.csv,validation_summary.json},
plots/lsv_repricing_validation.png."""

import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm

from validation_config import (
    MC_N_PATHS, MC_STEPS_PER_YEAR, MC_N_REPRICE, MC_SEED,
    ROOT, IV_DIR, DUPIRE_DIR, BERGOMI_DIR, DATA_DIR, PLOT_DIR, ARRAY_DIR,
)
from validation_bs import *
from validation_spot_variance import *

warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lsv_validation_bergomi")


# Data loading

def load_validation_inputs():
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    with open(DATA_DIR / "bergomi_params.json") as f:
        bergomi = json.load(f)

    # Leverage surface.
    leverage_surface = np.load(ARRAY_DIR / "leverage_surface.npy")
    spot_grid = np.load(ARRAY_DIR / "leverage_spot_grid.npy")
    time_grid = np.load(ARRAY_DIR / "leverage_time_grid.npy")
    leverage_interp = interpolate.RegularGridInterpolator(
        (spot_grid, time_grid), leverage_surface,
        method="linear", bounds_error=False, fill_value=None,
    )

    # Forward variance curve.
    fwd_var = np.load(ARRAY_DIR / "fwd_var_curve.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    fwd_var_interp = interpolate.interp1d(
        ttm_grid, fwd_var, kind="linear",
        bounds_error=False, fill_value=(fwd_var[0], fwd_var[-1]),
    )

    # Dupire repricing baseline (Step 2).
    dupire_df = pd.read_csv(DUPIRE_DIR / "data" / "repricing_errors.csv")

    logger.info(f"Loaded: S={S:.2f}, leverage surface {leverage_surface.shape}, "
                f"{len(dupire_df)} Dupire repricing rows")

    return {
        "S": S, "r": r, "q": q,
        "bergomi": bergomi,
        "leverage_interp": leverage_interp,
        "spot_grid": spot_grid,
        "time_grid": time_grid,
        "dupire_df": dupire_df,
        "fwd_var_interp": fwd_var_interp,
        "ttm_grid": ttm_grid,
        "fwd_var": fwd_var,
    }


# Monte Carlo repricing

def bergomi_lsv_mc_reprice(inputs, n_paths=MC_N_PATHS,
                           steps_per_year=MC_STEPS_PER_YEAR,
                           n_reprice=MC_N_REPRICE, seed=MC_SEED):
    S0 = inputs["S"]
    r = inputs["r"]
    q = inputs["q"]
    bergomi = inputs["bergomi"]
    leverage_interp = inputs["leverage_interp"]
    spot_grid = inputs["spot_grid"]
    time_grid = inputs["time_grid"]
    dupire_df = inputs["dupire_df"]
    fwd_var_interp = inputs["fwd_var_interp"]
    ttm_grid = inputs["ttm_grid"]

    rho1 = bergomi["rho1"]
    rho2 = bergomi["rho2"]
    rho12 = bergomi["rho12"]
    kappa1 = bergomi["kappa1"]
    kappa2 = bergomi["kappa2"]

    rng = np.random.default_rng(seed)

    # Correlation matrix and Cholesky.
    corr_input = np.array([
        [1.0,   rho1,  rho2],
        [rho1,  1.0,   rho12],
        [rho2,  rho12, 1.0],
    ])
    # Diagnostic det/eigvals of the input.
    det_input = float(np.linalg.det(corr_input))
    eigvals_input, eigvecs = np.linalg.eigh(corr_input)
    logger.info(f"Corr matrix (input): det={det_input:+.6e}  "
                f"eigvals=[{eigvals_input[0]:+.4e}, {eigvals_input[1]:+.4e}, "
                f"{eigvals_input[2]:+.4e}]")
    logger.info(f"Corr matrix (input): rho1={corr_input[0,1]:+.6f}  "
                f"rho2={corr_input[0,2]:+.6f}  rho12={corr_input[1,2]:+.6f}")
    # Eigenvalue floor at 1e-6 (matches particle_method.py).
    eigvals = np.maximum(eigvals_input, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)
    regularised = bool(np.any(eigvals != eigvals_input))
    L_chol = np.linalg.cholesky(corr)

    # Post-regularisation diagnostic.
    det_post = float(np.linalg.det(corr))
    eigvals_post = np.linalg.eigvalsh(corr)
    tag = "regularised" if regularised else "unchanged"
    logger.info(f"Corr matrix ({tag}): det={det_post:+.6e}  "
                f"eigvals=[{eigvals_post[0]:+.4e}, {eigvals_post[1]:+.4e}, "
                f"{eigvals_post[2]:+.4e}]")
    logger.info(f"Corr matrix ({tag}): rho1={corr[0,1]:+.6f}  "
                f"rho2={corr[0,2]:+.6f}  rho12={corr[1,2]:+.6f}")
    if regularised:
        delta_rho1  = corr[0,1] - corr_input[0,1]
        delta_rho2  = corr[0,2] - corr_input[0,2]
        delta_rho12 = corr[1,2] - corr_input[1,2]
        logger.info(f"Corr matrix delta: drho1={delta_rho1:+.4e}  "
                    f"drho2={delta_rho2:+.4e}  drho12={delta_rho12:+.4e}")

    # Filter options to leverage-surface bounds.
    spot_min, spot_max = spot_grid[0], spot_grid[-1]
    t_min, t_max = time_grid[0], time_grid[-1]
    in_bounds = (
        (dupire_df["strike"] >= spot_min * 0.95) &
        (dupire_df["strike"] <= spot_max * 1.05) &
        (dupire_df["ttm"] >= t_min) &
        (dupire_df["ttm"] <= t_max) &
        (dupire_df["iv_ssvi"].between(0.01, 2.0))
    )
    pool = dupire_df[in_bounds].copy()
    if n_reprice > 0 and len(pool) > n_reprice:
        sample = pool.sample(n_reprice, random_state=seed).copy()
    else:
        sample = pool.copy()

    if len(sample) == 0:
        logger.warning("No options in bounds.")
        return pd.DataFrame()

    logger.info(f"Repricing {len(sample)} options via Bergomi LSV MC ({n_paths:,} paths)")

    T_max_sim = sample["ttm"].max()
    n_steps = max(int(T_max_sim * steps_per_year), 20)
    dt = T_max_sim / n_steps
    sqrt_dt = np.sqrt(dt)
    t_schedule = np.arange(n_steps + 1) * dt

    sample["step_idx"] = sample["ttm"].apply(
        lambda T: int(np.argmin(np.abs(t_schedule - T)))
    )
    required_steps = set(sample["step_idx"].unique())

    # OU exact-sim coefficients.
    decay1 = np.exp(-kappa1 * dt)
    decay2 = np.exp(-kappa2 * dt)
    std1 = np.sqrt((1.0 - np.exp(-2.0 * kappa1 * dt)) / (2.0 * kappa1)) if kappa1 > 1e-10 else sqrt_dt
    std2 = np.sqrt((1.0 - np.exp(-2.0 * kappa2 * dt)) / (2.0 * kappa2)) if kappa2 > 1e-10 else sqrt_dt

    S_sim = np.full(n_paths, S0, dtype=np.float64)
    X1 = np.zeros(n_paths, dtype=np.float64)
    X2 = np.zeros(n_paths, dtype=np.float64)
    step_spots = {}

    for step in range(1, n_steps + 1):
        t = (step - 1) * dt

        xi_t_t = _spot_variance(X1, X2, t, bergomi, fwd_var_interp, ttm_grid)

        S_clamped = np.clip(S_sim, spot_grid[0], spot_grid[-1])
        t_clamped = np.clip(t, time_grid[0], time_grid[-1])
        pts = np.column_stack([S_clamped, np.full(n_paths, t_clamped)])
        L_vals = leverage_interp(pts)
        L_vals = np.clip(L_vals, np.sqrt(0.01), np.sqrt(5.0))

        Z_indep = rng.standard_normal((3, n_paths))
        Z_corr = L_chol @ Z_indep
        Z_S, Z_W1, Z_W2 = Z_corr[0], Z_corr[1], Z_corr[2]

        # Spot (log-Euler).
        vol = L_vals * np.sqrt(np.maximum(xi_t_t, 0.0))
        S_sim = S_sim * np.exp(
            (r - q - 0.5 * vol**2) * dt + vol * sqrt_dt * Z_S
        )

        # OU (exact).
        X1 = X1 * decay1 + std1 * Z_W1
        X2 = X2 * decay2 + std2 * Z_W2

        if step in required_steps:
            step_spots[step] = S_sim.copy()

        if step % max(1, n_steps // 5) == 0:
            logger.info(f"  Step {step}/{n_steps} | S: [{S_sim.min():.0f}, {S_sim.max():.0f}]")

    logger.info("  Simulation complete. Computing payoffs...")

    records = []
    for _, row in sample.iterrows():
        K = row["strike"]
        T = row["ttm"]
        opt_type = row["option_type"]
        # `iv_ssvi` is the SSVI-fitted IV; rebuild the SSVI BS price from it as
        # the comparison target, not the raw market mid.
        iv_ssvi = row["iv_ssvi"]
        if opt_type == "call":
            ssvi_price = bs_call_price(S0, K, T, r, q, iv_ssvi)
        else:
            ssvi_price = bs_put_price(S0, K, T, r, q, iv_ssvi)
        dup_price = row["mc_price"]
        dup_std = row["mc_std_err"]
        step_idx = row["step_idx"]

        S_T = step_spots[step_idx]
        if opt_type == "call":
            payoff = np.maximum(S_T - K, 0)
        else:
            payoff = np.maximum(K - S_T, 0)

        disc = np.exp(-r * T)
        lsv_price = disc * payoff.mean()
        lsv_std = disc * payoff.std() / np.sqrt(n_paths)

        iv_lsv = bs_iv(lsv_price, S0, K, T, r, q, opt_type)
        lsv_iv_err_bps = (iv_lsv - iv_ssvi) * 10000 if np.isfinite(iv_lsv) else np.nan

        lsv_vs_ssvi = (100.0 * (lsv_price - ssvi_price) / ssvi_price
                       if abs(ssvi_price) > 0.01 else np.nan)
        dup_vs_ssvi = (100.0 * (dup_price - ssvi_price) / ssvi_price
                       if abs(ssvi_price) > 0.01 else np.nan)

        records.append({
            "strike": K,
            "ttm": round(T, 4),
            "option_type": opt_type,
            "moneyness": round(K / (S0 * np.exp((r - q) * T)), 4),
            "log_moneyness": round(np.log(K / (S0 * np.exp((r - q) * T))), 4),
            "ssvi_price": round(ssvi_price, 4),
            "lsv_price": round(lsv_price, 4),
            "dupire_price": round(dup_price, 4),
            "lsv_std_err": round(lsv_std, 4),
            "dupire_std_err": round(dup_std, 4),
            "iv_ssvi": round(iv_ssvi, 4),
            "iv_lsv": round(iv_lsv, 6) if np.isfinite(iv_lsv) else None,
            "lsv_iv_error_bps": round(lsv_iv_err_bps, 2) if np.isfinite(lsv_iv_err_bps) else None,
            "lsv_vs_ssvi_pct": round(lsv_vs_ssvi, 2) if np.isfinite(lsv_vs_ssvi) else None,
            "dupire_vs_ssvi_pct": round(dup_vs_ssvi, 2) if np.isfinite(dup_vs_ssvi) else None,
        })

    return pd.DataFrame(records)


# Summary statistics

def compute_summary(result_df):
    valid = result_df.dropna(subset=["lsv_iv_error_bps"])
    iv_err = valid["lsv_iv_error_bps"]
    lsv_ssvi = valid.dropna(subset=["lsv_vs_ssvi_pct"])["lsv_vs_ssvi_pct"]
    dup_ssvi = valid.dropna(subset=["dupire_vs_ssvi_pct"])["dupire_vs_ssvi_pct"]

    summary = {
        "n_repriced": len(result_df),
        "n_valid": len(valid),
        "lsv_iv_mae_bps": float(iv_err.abs().mean()),
        "lsv_iv_me_bps": float(iv_err.mean()),
        "lsv_iv_rmse_bps": float(np.sqrt((iv_err**2).mean())),
        "lsv_iv_median_bps": float(iv_err.abs().median()),
        "lsv_iv_p5_bps": float(iv_err.quantile(0.05)),
        "lsv_iv_p95_bps": float(iv_err.quantile(0.95)),
        "lsv_vs_ssvi_mae_pct": float(lsv_ssvi.abs().mean()),
        "lsv_vs_ssvi_me_pct": float(lsv_ssvi.mean()),
        "lsv_vs_ssvi_rmse_pct": float(np.sqrt((lsv_ssvi**2).mean())),
        "dupire_vs_ssvi_mae_pct": float(dup_ssvi.abs().mean()),
    }
    for min_price in [10, 20, 50]:
        f = valid[valid["ssvi_price"] >= min_price]
        if len(f) > 0:
            summary[f"lsv_iv_mae_bps_ge_{min_price}"] = float(f["lsv_iv_error_bps"].abs().mean())
            summary[f"lsv_iv_me_bps_ge_{min_price}"] = float(f["lsv_iv_error_bps"].mean())

    logger.info("=" * 60)
    logger.info("Bergomi LSV Validation Summary:")
    logger.info(f"  Options repriced: {summary['n_valid']}")
    logger.info(f"  LSV IV Error — MAE: {summary['lsv_iv_mae_bps']:.1f} bp, "
                f"ME: {summary['lsv_iv_me_bps']:+.1f} bp, "
                f"RMSE: {summary['lsv_iv_rmse_bps']:.1f} bp")
    logger.info("=" * 60)

    return summary


# Plotting

def plot_validation(result_df, summary):
    valid = result_df.dropna(subset=["lsv_iv_error_bps"])
    err = valid["lsv_iv_error_bps"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bergomi LSV Validation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    sc = ax.scatter(valid["ssvi_price"], valid["lsv_price"],
                    c=valid["ttm"], cmap="viridis", alpha=0.7, s=20, edgecolors="none")
    lims = [0, max(valid["ssvi_price"].max(), valid["lsv_price"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("SSVI Price ($)")
    ax.set_ylabel("LSV MC Price ($)")
    ax.set_title("(a) LSV Price vs SSVI")

    ax = axes[0, 1]
    ax.hist(err, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("IV Error (bp)")
    ax.set_title(f"(b) IV Error Distribution (MAE={summary['lsv_iv_mae_bps']:.1f} bp)")

    ax = axes[1, 0]
    sc = ax.scatter(valid["log_moneyness"], err,
                    c=valid["ttm"], cmap="viridis", alpha=0.7, s=25, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("IV Error (bp)")
    ax.set_title("(c) IV Error vs Log-Moneyness")
    plt.colorbar(sc, ax=ax, label="TTM")

    ax = axes[1, 1]
    sc = ax.scatter(valid["ttm"], err,
                    c=valid["log_moneyness"], cmap="coolwarm", alpha=0.7, s=25, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel("IV Error (bp)")
    ax.set_title("(d) IV Error vs TTM")
    plt.colorbar(sc, ax=ax, label="log(K/F)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = PLOT_DIR / "lsv_repricing_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved validation plot -> {out_path}")


def run(n_paths=MC_N_PATHS, n_reprice=MC_N_REPRICE, seed=MC_SEED):
    logger.info("=" * 60)
    logger.info("CHECKPOINT 2 (Bergomi): LSV Validation")
    logger.info("=" * 60)

    inputs = load_validation_inputs()
    result_df = bergomi_lsv_mc_reprice(
        inputs, n_paths=n_paths, n_reprice=n_reprice, seed=seed
    )

    if len(result_df) == 0:
        logger.error("No repricing results.")
        return result_df, {}

    result_df.to_csv(DATA_DIR / "lsv_repricing_errors.csv", index=False)
    logger.info(f"Saved repricing errors -> {DATA_DIR}/lsv_repricing_errors.csv")

    summary = compute_summary(result_df)
    with open(DATA_DIR / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_validation(result_df, summary)

    return result_df, summary


if __name__ == "__main__":
    run()

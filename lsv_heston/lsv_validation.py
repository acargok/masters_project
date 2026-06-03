#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSV validation — Checkpoint 2 (Master's Thesis, Imperial College London).

Validates the calibrated leverage L(t, S) by MC-repricing vanilla European
options under the full LSV dynamics (dS=(r-q)S dt + L(t,S) sqrt(V) S dW^S,
dV=kappa(theta-V)dt + xi sqrt(V)dW^V, d<W^S,W^V>=rho dt; L interpolated from the
Step-3b surface). Errors compared to Dupire (Step 2) as a benchmark.

Inputs:
    lsv_heston/arrays/leverage_{surface,spot_grid,time_grid}.npy — L(t,S) and grids
    lsv_heston/data/heston_params.json   — Heston params
    dupire_vol/data/market_params.json   — S, r, q
    dupire_vol/data/repricing_errors.csv — Dupire repricing for comparison
    iv_surface/data/spx_iv_data.csv      — option data
Outputs:
    data/lsv_repricing_errors.csv       — MC repricing results
    data/validation_summary.json        — summary stats
    plots/lsv_repricing_validation.png  — scatter + histogram
    plots/lsv_vs_dupire_comparison.png  — side-by-side comparison
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
from scipy import interpolate
from scipy.stats import norm

from validation_config import *
from validation_bs import *

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lsv_validation")


# --- Data loading ---

def load_validation_inputs():
    """
    Load LSV-validation inputs (Dupire repricing read from Step-2 CSV, not recomputed).

    Returns dict: S, r, q, heston, leverage_interp, leverage_surface, spot_grid,
    time_grid, dupire_df, dupire_interp, log_m_grid, ttm_grid_dup, fwd_curve.
    """
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    with open(DATA_DIR / "heston_params.json") as f:
        heston = json.load(f)

    leverage_surface = np.load(ARRAY_DIR / "leverage_surface.npy")
    spot_grid = np.load(ARRAY_DIR / "leverage_spot_grid.npy")
    time_grid = np.load(ARRAY_DIR / "leverage_time_grid.npy")

    # 2D L(t,S) interp, (spot, time) ordering. time_grid starts at first positive
    # recorded step (t=0 slice excluded: cloud degenerate at t=0); early-time
    # queries clamp to time_grid[0] via bounds_error=False.
    leverage_interp = interpolate.RegularGridInterpolator(
        (spot_grid, time_grid),
        leverage_surface,
        method="linear",
        bounds_error=False,
        fill_value=None,   # clamp to nearest boundary
    )

    # Dupire local vol surface and forward curve (for Gyöngy diagnostic)
    local_vol = np.load(DUPIRE_DIR / "arrays" / "local_vol_surface.npy")
    ttm_grid_dup = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    log_m_grid_dup = np.load(IV_DIR / "arrays" / "log_m_grid.npy")
    dupire_interp = interpolate.RegularGridInterpolator(
        (log_m_grid_dup, ttm_grid_dup),
        local_vol,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    fwd_prices = np.load(IV_DIR / "arrays" / "forward_curve.npy")
    if fwd_prices.ndim == 1:
        fwd_curve = np.column_stack([ttm_grid_dup, fwd_prices])
    else:
        fwd_curve = fwd_prices

    # Dupire repricing from Step 2 (no recompute)
    dupire_df = pd.read_csv(DUPIRE_DIR / "data" / "repricing_errors.csv")

    logger.info(f"Loaded: S={S:.2f}, leverage surface {leverage_surface.shape}, "
                f"{len(dupire_df)} Dupire repricing rows from Step 2")

    return {
        "S": S, "r": r, "q": q,
        "heston": heston,
        "leverage_interp": leverage_interp,
        "leverage_surface": leverage_surface,
        "spot_grid": spot_grid,
        "time_grid": time_grid,
        "dupire_df": dupire_df,
        "dupire_interp": dupire_interp,
        "log_m_grid": log_m_grid_dup,
        "ttm_grid_dup": ttm_grid_dup,
        "fwd_curve": fwd_curve,
    }


# --- Monte Carlo simulation under LSV dynamics ---

def lsv_monte_carlo_reprice(inputs, n_paths=MC_N_PATHS,
                              steps_per_year=MC_STEPS_PER_YEAR,
                              n_reprice=MC_N_REPRICE,
                              seed=MC_SEED,
                              variance_scheme="qe"):
    """
    MC-reprice vanilla Europeans under LSV; Dupire prices from Step-2 CSV.

    n_reprice: 0 = all in-bounds. variance_scheme: "euler"/"qe".
    Returns (result_df, diag_snapshots): repricing DataFrame (strike, ttm,
    option_type, prices, IV/price errors vs SSVI) and Gyöngy-diagnostic
    snapshots keyed by step.
    """
    S0 = inputs["S"]
    r = inputs["r"]
    q = inputs["q"]
    heston = inputs["heston"]
    leverage_interp = inputs["leverage_interp"]
    spot_grid = inputs["spot_grid"]
    time_grid = inputs["time_grid"]
    dupire_df = inputs["dupire_df"]

    kappa = heston["kappa"]
    theta = heston["theta"]
    xi = heston["xi"]
    rho = heston["rho"]
    V0 = heston["V0"]

    rng = np.random.default_rng(seed)

    # Option pool = Dupire repricing CSV (each row has a Step-2 Dupire price)
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
        logger.warning("No options in bounds for LSV repricing.")
        return pd.DataFrame()

    logger.info(f"Repricing {len(sample)} options via LSV MC ({n_paths:,} paths); "
                f"Dupire prices from Step 2 CSV")

    T_max = sample["ttm"].max()
    n_steps = max(int(T_max * steps_per_year), 20)
    dt = T_max / n_steps
    sqrt_dt = np.sqrt(dt)

    t_schedule = np.arange(n_steps + 1) * dt

    # Map each option to nearest step
    sample["step_idx"] = sample["ttm"].apply(
        lambda T: int(np.argmin(np.abs(t_schedule - T)))
    )
    required_steps = set(sample["step_idx"].unique())

    logger.info(f"  Simulation: {n_steps} steps, dt={dt:.6f}, T_max={T_max:.4f}")
    logger.info(f"  {len(required_steps)} distinct maturities required")

    # Gyöngy diagnostic snapshots at 25/50/75/100% of the sim
    diag_steps = {max(1, int(f * n_steps)) for f in (0.25, 0.50, 0.75, 1.00)}
    diag_snapshots = {}  # step -> {"t", "S", "V", "L"}

    lsv_step_spots = {}   # LSV spots at required maturities

    S_lsv = np.full(n_paths, S0, dtype=np.float64)
    V_t = np.full(n_paths, V0, dtype=np.float64)

    # Lazy import avoids circular import; particle_method exposes QE helpers
    from particle_method import step_variance_qe, step_spot_qe_bk

    if variance_scheme not in ("euler", "qe"):
        raise ValueError(f"Unknown variance_scheme: {variance_scheme!r}")
    logger.info(f"  variance_scheme = {variance_scheme}")

    for step in range(1, n_steps + 1):
        t = (step - 1) * dt

        # Leverage at start of step
        S_clamped = np.clip(S_lsv, spot_grid[0], spot_grid[-1])
        t_clamped = np.clip(t, time_grid[0], time_grid[-1])
        pts = np.column_stack([S_clamped, np.full(n_paths, t_clamped)])
        L_vals = leverage_interp(pts)
        L_vals = np.clip(L_vals, np.sqrt(0.01), np.sqrt(5.0))

        V_pos = np.maximum(V_t, 0.0)

        # Snapshot (S_t, V_t^+, L(t,S_t)) before advancing state
        if step in diag_steps:
            diag_snapshots[step] = {
                "t": t,
                "S": S_lsv.copy(),
                "V": V_pos.copy(),
                "L": L_vals.copy(),
            }

        if variance_scheme == "euler":
            Z1 = rng.standard_normal(n_paths)
            Z_indep = rng.standard_normal(n_paths)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z_indep

            sqrt_V = np.sqrt(V_pos)
            vol_lsv = L_vals * sqrt_V
            S_lsv = S_lsv * np.exp(
                (r - q - 0.5 * vol_lsv**2) * dt + vol_lsv * sqrt_dt * Z1
            )

            V_t = V_t + kappa * (theta - V_pos) * dt + xi * sqrt_V * sqrt_dt * Z2
            V_t = np.maximum(V_t, 0.0)
        else:
            Z_qe = rng.standard_normal(n_paths)
            Z_perp = rng.standard_normal(n_paths)
            V_new = step_variance_qe(V_t, dt, kappa, theta, xi, Z_qe)
            log_S = np.log(np.maximum(S_lsv, 1e-12))
            log_S = step_spot_qe_bk(
                log_S, V_t, V_new, L_vals, dt,
                r, q, rho, kappa, theta, xi, Z_perp,
            )
            S_lsv = np.exp(log_S)
            V_t = V_new

        if step in required_steps:
            lsv_step_spots[step] = S_lsv.copy()

        if step % max(1, n_steps // 5) == 0:
            logger.info(f"  Step {step}/{n_steps} | "
                        f"LSV S: [{S_lsv.min():.0f}, {S_lsv.max():.0f}]")

    logger.info("  Simulation complete. Computing payoffs...")

    # Reprice each option. `iv_ssvi` is the SSVI-fitted IV (from
    # spx_iv_data.csv:iv); rebuild the SSVI BS price as comparison target (not
    # the raw market mid).
    records = []
    for _, row in sample.iterrows():
        K = row["strike"]
        T = row["ttm"]
        opt_type = row["option_type"]
        iv_ssvi = row["iv_ssvi"]
        if opt_type == "call":
            ssvi_price = bs_call_price(S0, K, T, r, q, iv_ssvi)
        else:
            ssvi_price = bs_put_price(S0, K, T, r, q, iv_ssvi)
        dup_price = row["mc_price"]
        dup_std = row["mc_std_err"]
        step_idx = row["step_idx"]

        S_T_lsv = lsv_step_spots[step_idx]

        if opt_type == "call":
            payoff_lsv = np.maximum(S_T_lsv - K, 0)
        else:
            payoff_lsv = np.maximum(K - S_T_lsv, 0)

        disc = np.exp(-r * T)
        lsv_price = disc * payoff_lsv.mean()
        lsv_std = disc * payoff_lsv.std() / np.sqrt(n_paths)

        # IV-space error vs SSVI
        iv_lsv = bs_iv(lsv_price, S0, K, T, r, q, opt_type)
        lsv_iv_err_bps = (iv_lsv - iv_ssvi) * 10000 if np.isfinite(iv_lsv) else np.nan

        # Price-space error vs SSVI BS price
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

    result_df = pd.DataFrame(records)
    return result_df, diag_snapshots


# --- Gyöngy projection diagnostic ---

def compute_gyongy_diagnostic(diag_snapshots, S0, r, q, dupire_interp,
                               log_m_grid, ttm_grid, fwd_curve,
                               n_bins=20, min_paths=100):
    """
    Check the Gyöngy projection on validation paths.

    Per time slice, bins paths by spot and compares mean(L(t,S_t)^2 V_t | bin)
    against Dupire target sigma_Dupire(t, bin_center)^2. Ratio 1 = exact; >1 =
    projected variance exceeds target -> positive IV bias (overpricing).
    Returns DataFrame, one row per (slice, bin): step, time, spot_bin_center,
    n_paths, mean_L2V, dupire_var, ratio.
    """
    records = []
    for step in sorted(diag_snapshots):
        snap = diag_snapshots[step]
        t = snap["t"]
        S_arr = snap["S"]
        V_arr = snap["V"]   # already >= 0
        L_arr = snap["L"]
        L2V = L_arr ** 2 * V_arr

        # Bin over 5th-95th pct to keep bins populated
        S_lo = np.percentile(S_arr, 5)
        S_hi = np.percentile(S_arr, 95)
        if S_lo >= S_hi:
            continue
        edges = np.linspace(S_lo, S_hi, n_bins + 1)
        bin_idx = np.clip(np.digitize(S_arr, edges) - 1, 0, n_bins - 1)

        t_c = float(np.clip(t, ttm_grid[0], ttm_grid[-1]))

        for b in range(n_bins):
            mask = bin_idx == b
            n_in_bin = int(mask.sum())
            if n_in_bin < min_paths:
                continue

            bin_center = 0.5 * (edges[b] + edges[b + 1])
            mean_L2V = float(L2V[mask].mean())

            # Dupire local variance at (bin_center, t), per-expiry forward
            F_0_t = float(np.interp(t, fwd_curve[:, 0], fwd_curve[:, 1]))
            F_0_t = max(F_0_t, 1e-6)
            log_m = np.log(bin_center / F_0_t)
            log_m_c = float(np.clip(log_m, log_m_grid[0], log_m_grid[-1]))
            sigma_dup = float(dupire_interp([[log_m_c, t_c]])[0])
            sigma_dup = max(sigma_dup, 1e-4)
            dupire_var = sigma_dup ** 2

            records.append({
                "step": int(step),
                "time": round(t, 6),
                "spot_bin_center": round(bin_center, 4),
                "n_paths": n_in_bin,
                "mean_L2V": round(mean_L2V, 8),
                "dupire_var": round(dupire_var, 8),
                "ratio": round(mean_L2V / dupire_var, 6),
            })

    return pd.DataFrame(records)


# --- Summary statistics ---

def compute_summary(result_df):
    """Validation summary stats from repricing results; returns dict."""
    valid = result_df.dropna(subset=["lsv_iv_error_bps"])

    iv_err = valid["lsv_iv_error_bps"]                                        # bp vs SSVI
    lsv_ssvi = valid.dropna(subset=["lsv_vs_ssvi_pct"])["lsv_vs_ssvi_pct"]    # price %
    dup_ssvi = valid.dropna(subset=["dupire_vs_ssvi_pct"])["dupire_vs_ssvi_pct"]

    summary = {
        "n_repriced": len(result_df),
        "n_valid": len(valid),
        # LSV IV error vs SSVI (bp)
        "lsv_iv_mae_bps": float(iv_err.abs().mean()),
        "lsv_iv_me_bps": float(iv_err.mean()),
        "lsv_iv_rmse_bps": float(np.sqrt((iv_err ** 2).mean())),
        "lsv_iv_median_bps": float(iv_err.abs().median()),
        "lsv_iv_p5_bps": float(iv_err.quantile(0.05)),
        "lsv_iv_p95_bps": float(iv_err.quantile(0.95)),
        # LSV vs SSVI (price %)
        "lsv_vs_ssvi_mae_pct": float(lsv_ssvi.abs().mean()),
        "lsv_vs_ssvi_me_pct": float(lsv_ssvi.mean()),
        "lsv_vs_ssvi_rmse_pct": float(np.sqrt((lsv_ssvi ** 2).mean())),
        # Dupire vs SSVI (price %)
        "dupire_vs_ssvi_mae_pct": float(dup_ssvi.abs().mean()),
    }

    # Filtered by min SSVI price floor
    for min_price in [10, 20, 50]:
        f = valid[valid["ssvi_price"] >= min_price]
        if len(f) > 0:
            summary[f"lsv_iv_mae_bps_ge_{min_price}"] = float(
                f["lsv_iv_error_bps"].abs().mean()
            )
            summary[f"lsv_iv_me_bps_ge_{min_price}"] = float(
                f["lsv_iv_error_bps"].mean()
            )
            summary[f"lsv_vs_ssvi_mae_ge_{min_price}"] = float(
                f.dropna(subset=["lsv_vs_ssvi_pct"])["lsv_vs_ssvi_pct"].abs().mean()
            )
            summary[f"dupire_vs_ssvi_mae_ge_{min_price}"] = float(
                f.dropna(subset=["dupire_vs_ssvi_pct"])["dupire_vs_ssvi_pct"].abs().mean()
            )
            summary[f"n_price_ge_{min_price}"] = len(f)

    logger.info("=" * 60)
    logger.info("LSV Validation Summary (Checkpoint 2):")
    logger.info(f"  Options repriced: {summary['n_valid']}")
    logger.info("")
    logger.info("  LSV IV Error vs SSVI (bp):")
    logger.info(f"    MAE:    {summary['lsv_iv_mae_bps']:.1f} bp")
    logger.info(f"    ME:     {summary['lsv_iv_me_bps']:+.1f} bp")
    logger.info(f"    RMSE:   {summary['lsv_iv_rmse_bps']:.1f} bp")
    logger.info(f"    Median: {summary['lsv_iv_median_bps']:.1f} bp")
    logger.info(f"    [P5,P95]: [{summary['lsv_iv_p5_bps']:.1f}, "
                f"{summary['lsv_iv_p95_bps']:.1f}] bp")
    logger.info("")
    logger.info("  vs SSVI (both models, price %):")
    logger.info(f"    LSV MAE:    {summary['lsv_vs_ssvi_mae_pct']:.2f}%")
    logger.info(f"    Dupire MAE: {summary['dupire_vs_ssvi_mae_pct']:.2f}%")
    for min_price in [10, 50]:
        lk = f"lsv_vs_ssvi_mae_ge_{min_price}"
        dk = f"dupire_vs_ssvi_mae_ge_{min_price}"
        if lk in summary:
            logger.info(f"    (price>=${min_price})  LSV: {summary[lk]:.2f}%  "
                        f"Dupire: {summary[dk]:.2f}%")
    logger.info("=" * 60)

    return summary


# --- Plotting ---

def plot_validation(result_df, summary):
    """
    Two validation figures: lsv_repricing_validation.png (LSV/Dupire vs SSVI,
    4-panel) and lsv_vs_dupire_comparison.png (head-to-head, 4-panel).
    """
    valid = result_df.dropna(subset=["lsv_iv_error_bps"])

    # Figure 1: both models vs SSVI
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LSV Model Validation — Checkpoint 2", fontsize=14, fontweight="bold")

    # (a) LSV price vs SSVI price
    ax = axes[0, 0]
    sc = ax.scatter(valid["ssvi_price"], valid["lsv_price"],
                    c=valid["ttm"], cmap="viridis", alpha=0.7, s=20, edgecolors="none")
    lims = [0, max(valid["ssvi_price"].max(), valid["lsv_price"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect repricing")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("SSVI Price ($)")
    ax.set_ylabel("LSV MC Price ($)")
    ax.set_title("(a) LSV Price vs SSVI Price")
    ax.legend(loc="upper left")

    # (b) error histogram, both models vs SSVI
    ax = axes[0, 1]
    lsv_ssvi = valid.dropna(subset=["lsv_vs_ssvi_pct"])
    dup_ssvi = valid.dropna(subset=["dupire_vs_ssvi_pct"])
    ax.hist(dup_ssvi["dupire_vs_ssvi_pct"], bins=30, alpha=0.5,
            color="orange", edgecolor="black", label="Dupire vs SSVI")
    ax.hist(lsv_ssvi["lsv_vs_ssvi_pct"], bins=30, alpha=0.5,
            color="steelblue", edgecolor="black", label="LSV vs SSVI")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Error vs SSVI (%)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Both Models vs SSVI")
    ax.legend(fontsize=9)

    # (c) LSV vs SSVI error by log-moneyness
    ax = axes[1, 0]
    sc = ax.scatter(lsv_ssvi["log_moneyness"], lsv_ssvi["lsv_vs_ssvi_pct"],
                    c=lsv_ssvi["ttm"], cmap="viridis", alpha=0.7, s=20, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fwd Log-Moneyness  ln(K/F)")
    ax.set_ylabel("LSV vs SSVI Error (%)")
    ax.set_title("(c) LSV vs SSVI by Fwd Log-Moneyness")
    plt.colorbar(sc, ax=ax, label="TTM (years)")

    # (d) LSV price vs Dupire price
    ax = axes[1, 1]
    sc = ax.scatter(valid["dupire_price"], valid["lsv_price"],
                    c=valid["ttm"], cmap="viridis", alpha=0.7, s=20, edgecolors="none")
    lims = [0, max(valid["dupire_price"].max(), valid["lsv_price"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Dupire MC Price ($)")
    ax.set_ylabel("LSV MC Price ($)")
    ax.set_title("(d) LSV vs Dupire Price")
    ax.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = PLOT_DIR / "lsv_repricing_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved validation plot → {out_path}")

    # Figure 2: LSV vs Dupire head-to-head
    _plot_lsv_vs_dupire(valid, summary)


def _plot_lsv_vs_dupire(valid, summary):
    """Head-to-head LSV vs Dupire: IV error in basis points (4-panel)."""
    valid = valid.dropna(subset=["lsv_iv_error_bps"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LSV IV Error (vs SSVI IV) — basis points",
                 fontsize=14, fontweight="bold")

    err = valid["lsv_iv_error_bps"]

    # (a) IV error (bp) vs log-moneyness
    ax = axes[0, 0]
    sc = ax.scatter(valid["log_moneyness"], err,
                    c=valid["ttm"], cmap="viridis", alpha=0.7, s=25, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fwd Log-Moneyness  ln(K/F)")
    ax.set_ylabel("LSV IV Error (bp)")
    mae = summary["lsv_iv_mae_bps"]
    ax.set_title(f"(a) LSV IV Error by Fwd Log-Moneyness  (MAE={mae:.1f} bp)")
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="TTM (years)")
    y_lo, y_hi = np.percentile(err, [2.5, 97.5])
    pad = (y_hi - y_lo) * 0.3
    ax.set_ylim(y_lo - pad, y_hi + pad)

    # (b) same, filtered to price >= $10
    ax = axes[0, 1]
    liq = valid[valid["ssvi_price"] >= 10]
    err_liq = liq["lsv_iv_error_bps"]
    sc = ax.scatter(liq["log_moneyness"], err_liq,
                    c=liq["ttm"], cmap="viridis", alpha=0.7, s=25, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Fwd Log-Moneyness  ln(K/F)")
    ax.set_ylabel("LSV IV Error (bp)")
    mae_liq = err_liq.abs().mean() if len(err_liq) > 0 else float("nan")
    ax.set_title(f"(b) Price >= $10  (MAE={mae_liq:.1f} bp, N={len(liq)})")
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="TTM (years)")

    # (c) Error histogram
    ax = axes[1, 0]
    ax.hist(err, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=1)
    med = summary["lsv_iv_median_bps"]
    ax.set_xlabel("LSV IV Error (bp)")
    ax.set_ylabel("Count")
    ax.set_title(f"(c) Error Distribution  (median |err|={med:.1f} bp)")

    # (d) Error vs TTM
    ax = axes[1, 1]
    sc = ax.scatter(valid["ttm"], err,
                    c=valid["log_moneyness"], cmap="coolwarm", alpha=0.7,
                    s=25, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel("LSV IV Error (bp)")
    ax.set_title("(d) IV Error vs TTM")
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="ln(K/F)")
    ax.set_ylim(y_lo - pad, y_hi + pad)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = PLOT_DIR / "lsv_vs_dupire_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved LSV vs Dupire comparison → {out_path}")


# --- Entry point ---

def run(n_paths=MC_N_PATHS, n_reprice=MC_N_REPRICE, seed=MC_SEED,
        variance_scheme="qe"):
    """Run the LSV validation pipeline; returns (result_df, summary)."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT 2: LSV Model Validation")
    logger.info("=" * 60)

    inputs = load_validation_inputs()

    result_df, diag_snapshots = lsv_monte_carlo_reprice(
        inputs, n_paths=n_paths, n_reprice=n_reprice, seed=seed,
        variance_scheme=variance_scheme,
    )

    if len(result_df) == 0:
        logger.error("No repricing results. Validation failed.")
        return result_df, {}

    out_path = DATA_DIR / "lsv_repricing_errors.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved repricing errors → {out_path}")

    summary = compute_summary(result_df)

    out_path = DATA_DIR / "validation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved validation summary → {out_path}")

    # Gyöngy projection diagnostic
    diag_df = compute_gyongy_diagnostic(
        diag_snapshots,
        S0=inputs["S"], r=inputs["r"], q=inputs["q"],
        dupire_interp=inputs["dupire_interp"],
        log_m_grid=inputs["log_m_grid"],
        ttm_grid=inputs["ttm_grid_dup"],
        fwd_curve=inputs["fwd_curve"],
    )
    diag_path = DATA_DIR / "gyongy_diagnostic.csv"
    diag_df.to_csv(diag_path, index=False)
    logger.info(f"Saved Gyöngy diagnostic → {diag_path}")
    if len(diag_df) > 0:
        well_pop = diag_df[diag_df["n_paths"] >= 100]
        if len(well_pop) > 0:
            mean_r = well_pop["ratio"].mean()
            med_r = well_pop["ratio"].median()
            logger.info(
                f"  Gyöngy ratio E[L²V]/σ²_Dupire — "
                f"mean: {mean_r:.4f}, median: {med_r:.4f}  "
                f"(1.0 = perfect calibration, >1 = positive IV bias)"
            )

    plot_validation(result_df, summary)

    return result_df, summary


if __name__ == "__main__":
    run()

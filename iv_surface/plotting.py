import logging
import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from config import *
from market_data import _RAW_CHAIN_CACHE
from ssvi import ssvi_total_variance

logger = logging.getLogger(__name__)


# Section 5 — Plotting

def plot_iv_surface(ttm_grid: np.ndarray, log_m_grid: np.ndarray,
                    iv_surface: np.ndarray, S: float) -> None:
    """3D IV surface: X=k=ln(K/F), Y=T (years), Z=σ."""
    TTM_mesh, LM_mesh = np.meshgrid(ttm_grid, log_m_grid)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        LM_mesh, TTM_mesh, iv_surface,
        cmap=cm.RdYlGn_r, edgecolor="none", alpha=0.92,
        rcount=100, ccount=100
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.05,
                 label="Implied Volatility")
    ax.set_xlabel("Fwd Log-Moneyness  ln(K/F)", fontsize=11, labelpad=12)
    ax.set_ylabel("Time to Maturity (years)", fontsize=11, labelpad=12)
    ax.set_zlabel("Implied Volatility", fontsize=11, labelpad=10)
    snap = _RAW_CHAIN_CACHE.get("snapshot") or str(date.today())
    ax.set_title(
        f"SPX Implied Volatility Surface (SSVI, OptionMetrics)\n"
        f"Spot = {S:,.0f}   |   Snapshot: {snap}",
        fontsize=13, fontweight="bold"
    )
    ax.view_init(elev=28, azim=-55)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "iv_surface_3d.png"), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {DIR_PLOTS}/iv_surface_3d.png")


def plot_iv_smiles(df: pd.DataFrame, fwd_df: pd.DataFrame) -> None:
    """IV smiles by expiry in forward log-moneyness space."""
    fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))
    expiries = sorted(df["expiry"].unique())
    colours = plt.cm.viridis(np.linspace(0, 1, len(expiries)))

    fig, ax = plt.subplots(figsize=(13, 6))
    for exp, colour in zip(expiries, colours):
        subset = df[df["expiry"] == exp].sort_values("moneyness")
        F = fwd_map.get(exp, subset["strike"].median())
        fwd_log_m = np.log(subset["strike"].values / F)
        ax.plot(fwd_log_m, subset["iv"].values, "o-",
                color=colour, markersize=3, linewidth=1.2, alpha=0.75, label=exp)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5, label="ATM (K=F)")
    ax.set_xlabel("Forward Log-Moneyness  ln(K/F)", fontsize=11)
    ax.set_ylabel("Implied Volatility", fontsize=11)
    ax.set_title("SPX IV Smiles by Expiry (forward moneyness)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "iv_smiles.png"), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {DIR_PLOTS}/iv_smiles.png")


def plot_ssvi_fit(df: pd.DataFrame, fwd_df: pd.DataFrame,
                  ssvi_params_df: pd.DataFrame) -> None:
    """Overlay SSVI fit on market total variance per slice (up to 12 expiries,
    4-col grid); per-slice θ with shared η, γ, ρ."""
    fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))

    n_plot = min(12, len(ssvi_params_df))
    indices = np.linspace(0, len(ssvi_params_df) - 1, n_plot, dtype=int)
    slices = ssvi_params_df.iloc[indices]

    ncols = 4
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (_, row) in enumerate(slices.iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        expiry = row["expiry"]
        ttm = row["ttm"]

        slice_df = df[df["expiry"] == expiry]
        F = fwd_map.get(expiry, slice_df["strike"].median())
        k_mkt = np.log(slice_df["strike"].values / F)
        w_mkt = slice_df["iv"].values**2 * ttm

        k_fine = np.linspace(k_mkt.min() - 0.05, k_mkt.max() + 0.05, 200)
        w_ssvi = ssvi_total_variance(k_fine, row["theta"], row["phi"], row["rho"])

        ax.scatter(k_mkt, w_mkt, s=15, c="steelblue", alpha=0.7, label="Market")
        ax.plot(k_fine, w_ssvi, "r-", linewidth=1.5, label="SSVI")
        ax.set_title(f"{expiry} (T={ttm:.3f}y)\nRMSE={row['rmse']:.5f}", fontsize=9)
        ax.set_xlabel("k = ln(K/F)", fontsize=8)
        ax.set_ylabel("w = σ²T", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_plot, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    eta   = ssvi_params_df["eta"].iloc[0]
    gamma = ssvi_params_df["gamma"].iloc[0]
    p0, p1, p2 = (ssvi_params_df["p0"].iloc[0],
                   ssvi_params_df["p1"].iloc[0],
                   ssvi_params_df["p2"].iloc[0])
    plt.suptitle(
        f"SSVI Per-Slice Fits (total variance) — "
        f"η={eta:.3f}  γ={gamma:.3f}  "
        f"ρ(t): p₀={p0:.3f}, p₁={p1:.3f}, p₂={p2:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "ssvi_fit.png"), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {DIR_PLOTS}/ssvi_fit.png")

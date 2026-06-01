# -*- coding: utf-8 -*-
"""Plotting and repricing validation plots (SECTIONS 6-7)."""

import logging
import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from config import *

logger = logging.getLogger(__name__)


# SECTION 6: Plotting

def plot_local_vol_surface(
        local_vol: np.ndarray,
        ttm_grid: np.ndarray,
        log_m_grid: np.ndarray,
        mask: np.ndarray,
        S: float) -> None:
    """Render the Dupire local volatility surface as a 3D surface plot."""
    TTM_mesh, LM_mesh = np.meshgrid(ttm_grid, log_m_grid)
    pct_reliable = 100.0 * mask.sum() / mask.size

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        LM_mesh, TTM_mesh, local_vol,
        cmap=cm.RdYlGn_r,
        edgecolor="none",
        alpha=0.92,
        rcount=100, ccount=100
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.05,
                 label="Local Volatility (annualised)")

    ax.set_xlabel("k = log(K/F(0,T))", fontsize=11, labelpad=12)
    ax.set_ylabel("Time to Maturity (years)", fontsize=11, labelpad=12)
    ax.set_zlabel("Local Volatility", fontsize=11, labelpad=10)
    ax.set_title(
        f"Dupire Local Volatility Surface\n"
        f"Spot = {S:,.0f}   |   {date.today()}   |   "
        f"{pct_reliable:.0f}% reliable",
        fontsize=13, fontweight="bold"
    )
    ax.view_init(elev=28, azim=-55)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "local_vol_surface_3d.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {DIR_PLOTS}/local_vol_surface_3d.png")


def plot_iv_vs_local_vol(
        iv_surface: np.ndarray,
        local_vol: np.ndarray,
        ttm_grid: np.ndarray,
        log_m_grid: np.ndarray,
        S: float) -> None:
    """
    Side-by-side 3D IV vs Dupire local vol, both in k = log(K/F(0,T)).

    Local vol (instantaneous σ_loc) is spikier than IV (lifetime average
    σ_avg = √(w/T)).
    """
    TTM_mesh, LM_mesh = np.meshgrid(ttm_grid, log_m_grid)

    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(LM_mesh, TTM_mesh, iv_surface, cmap=cm.RdYlGn_r,
                     edgecolor="none", alpha=0.90, rcount=80, ccount=80)
    ax1.set_xlabel("k = log(K/F)", fontsize=10, labelpad=10)
    ax1.set_ylabel("TTM (years)", fontsize=10, labelpad=10)
    ax1.set_zlabel("Implied Vol σ(k,T)", fontsize=10, labelpad=8)
    ax1.set_title("Implied Vol Surface (SSVI, Step 1)",
                  fontsize=12, fontweight="bold")
    ax1.view_init(elev=28, azim=-55)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(LM_mesh, TTM_mesh, local_vol, cmap=cm.RdYlGn_r,
                     edgecolor="none", alpha=0.90, rcount=80, ccount=80)
    ax2.set_xlabel("k = log(K/F)", fontsize=10, labelpad=10)
    ax2.set_ylabel("TTM (years)", fontsize=10, labelpad=10)
    ax2.set_zlabel("Local Vol σ_loc(k,T)", fontsize=10, labelpad=8)
    ax2.set_title("Dupire Local Vol Surface (Step 2)",
                  fontsize=12, fontweight="bold")
    ax2.view_init(elev=28, azim=-55)

    plt.suptitle(
        f"SPX: Implied vs Local Volatility  |  Spot = {S:,.0f}  |  {date.today()}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "iv_vs_local_vol.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {DIR_PLOTS}/iv_vs_local_vol.png")


# SECTION 7: Repricing validation plots

# 4 categories: call/put × OTM/ITM
CATEGORY_STYLE = {
    "OTM Call": {"color": "#1f77b4", "marker": "o"},
    "ITM Call": {"color": "#7fbfff", "marker": "s"},
    "OTM Put":  {"color": "#d62728", "marker": "^"},
    "ITM Put":  {"color": "#ff9896", "marker": "D"},
}


def _classify_options(df: pd.DataFrame) -> pd.Series:
    """
    Classify each option into OTM/ITM Call/Put.

    Uses k = log(K/F) if present (k>=0: call OTM, put ITM), else spot
    moneyness K/S (old CSVs).
    """
    cat = pd.Series("", index=df.index)
    is_call = df["option_type"] == "call"

    if "fwd_log_m" in df.columns:
        is_otm = ((is_call) & (df["fwd_log_m"] >= 0)) | \
                 ((~is_call) & (df["fwd_log_m"] < 0))
    else:
        is_otm = ((is_call) & (df["moneyness"] >= 1.0)) | \
                 ((~is_call) & (df["moneyness"] < 1.0))

    cat[is_call  & is_otm]  = "OTM Call"
    cat[is_call  & ~is_otm] = "ITM Call"
    cat[~is_call & is_otm]  = "OTM Put"
    cat[~is_call & ~is_otm] = "ITM Put"
    return cat


def _scatter_by_category(ax, df, x_col, y_col, alpha=0.7, s=25):
    """Scatter, one colour/marker per option category."""
    cats = _classify_options(df)
    for label, style in CATEGORY_STYLE.items():
        mask = cats == label
        if mask.sum() == 0:
            continue
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   alpha=alpha, s=s, zorder=3,
                   color=style["color"], marker=style["marker"],
                   label=f"{label} ({mask.sum()})")


def plot_repricing_validation(result_df: pd.DataFrame) -> None:
    """Three-panel diagnostic: SSVI vs MC price, % error, error histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    liquid = result_df[result_df["ssvi_price"] >= 10.0]

    # Left: SSVI vs MC price
    ax = axes[0]
    _scatter_by_category(ax, result_df, "ssvi_price", "mc_price", alpha=0.5, s=20)
    lims = [0, max(result_df["ssvi_price"].max(),
                   result_df["mc_price"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="y = x", zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("SSVI BS Price ($)", fontsize=10)
    ax.set_ylabel("MC Price (Dupire local vol)", fontsize=10)
    ax.set_title("SSVI vs MC Price\n(all options)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

    # Middle: % error vs price (liquid)
    ax = axes[1]
    if len(liquid) > 0:
        _scatter_by_category(ax, liquid, "ssvi_price", "price_error_pct",
                             alpha=0.5, s=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("SSVI BS Price ($)", fontsize=10)
    ax.set_ylabel("MC Repricing Error vs SSVI (%)", fontsize=10)
    ax.set_title(f"Repricing Error vs SSVI Price\n(liquid: ssvi_price >= $10, n={len(liquid)})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: error histogram (liquid)
    ax = axes[2]
    if len(liquid) > 0:
        cats = _classify_options(liquid)
        for label, style in CATEGORY_STYLE.items():
            vals = liquid.loc[cats == label, "price_error_pct"].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.5, color=style["color"],
                        edgecolor="white", label=f"{label} ({len(vals)})")
        mean_err = liquid["price_error_pct"].dropna().mean()
        ax.axvline(mean_err, color="red", linestyle="-", linewidth=1.2,
                   label=f"Mean = {mean_err:.2f}%")
        for opt_type, col in [("call", "#1f77b4"), ("put", "#d62728")]:
            vals = liquid[liquid["option_type"] == opt_type]["price_error_pct"].dropna()
            if len(vals) > 0:
                ax.axvline(vals.mean(), color=col, linestyle="-", linewidth=1.2,
                           label=f"{opt_type.capitalize()} mean = {vals.mean():.2f}%")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("MC Repricing Error (%)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Repricing Errors\n(liquid options)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle("Dupire Local Vol — MC vs SSVI Prices (Checkpoint 1)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "repricing_validation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {DIR_PLOTS}/repricing_validation.png")


def plot_mc_vs_vanilla(result_df: pd.DataFrame) -> None:
    """
    Six-panel MC vs market comparison for all in-bounds options.

    Row 1 price scatter (log all / linear liquid), row 2 $ error vs TTM +
    histogram, row 3 % error vs TTM + histogram. Coloured by category.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    liquid = result_df[result_df["ssvi_price"] >= 10.0]

    # Row 1: price scatter
    ax = axes[0, 0]
    _scatter_by_category(ax, result_df, "ssvi_price", "mc_price", alpha=0.3, s=10)
    lo = max(0.01, min(result_df["ssvi_price"].min(), result_df["mc_price"].min()))
    hi = max(result_df["ssvi_price"].max(), result_df["mc_price"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("SSVI BS Price ($)", fontsize=10)
    ax.set_ylabel("MC Price (Dupire local vol)", fontsize=10)
    ax.set_title(f"All Options (n={len(result_df)}) — Log Scale",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3, which="both")

    ax = axes[0, 1]
    if len(liquid) > 0:
        _scatter_by_category(ax, liquid, "ssvi_price", "mc_price", alpha=0.35, s=12)
        lims = [0, max(liquid["ssvi_price"].max(), liquid["mc_price"].max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="y = x", zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ss_res = ((liquid["mc_price"] - liquid["ssvi_price"])**2).sum()
        ss_tot = ((liquid["ssvi_price"] - liquid["ssvi_price"].mean())**2).sum()
        r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae  = liquid["price_error"].abs().mean()
        calls_l = liquid[liquid["option_type"] == "call"]
        puts_l  = liquid[liquid["option_type"] == "put"]
        ax.text(0.05, 0.92,
                f"R² = {r2:.4f}   MAE = ${mae:.2f}\n"
                f"Call: ME=${calls_l['price_error'].mean() if len(calls_l)>0 else 0:+.2f}  "
                f"MAE=${calls_l['price_error'].abs().mean() if len(calls_l)>0 else 0:.2f}\n"
                f"Put:  ME=${puts_l['price_error'].mean() if len(puts_l)>0 else 0:+.2f}  "
                f"MAE=${puts_l['price_error'].abs().mean() if len(puts_l)>0 else 0:.2f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("SSVI BS Price ($)", fontsize=10)
    ax.set_ylabel("MC Price (Dupire local vol)", fontsize=10)
    ax.set_title(f"Liquid Options (ssvi_price >= $10, n={len(liquid)}) — Linear Scale",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)

    # Row 2: absolute error ($)
    ax = axes[1, 0]
    if len(liquid) > 0:
        _scatter_by_category(ax, liquid, "ttm", "price_error", alpha=0.4, s=12)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time to Maturity (years)", fontsize=10)
    ax.set_ylabel("Absolute Error: MC − SSVI ($)", fontsize=10)
    ax.set_title(f"Absolute Error vs Maturity (liquid, n={len(liquid)})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(liquid) > 0:
        cats = _classify_options(liquid)
        abs_all = liquid["price_error"].dropna()
        bins = np.linspace(abs_all.min() - 1, abs_all.max() + 1, 40)
        for label, style in CATEGORY_STYLE.items():
            vals = liquid.loc[cats == label, "price_error"].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=bins, alpha=0.5, color=style["color"],
                        edgecolor="white", label=f"{label} ({len(vals)})")
        ax.axvline(abs_all.mean(), color="red", linestyle="-", linewidth=1.2,
                   label=f"Mean = ${abs_all.mean():.2f}")
        for opt_type, col in [("call", "#1f77b4"), ("put", "#d62728")]:
            vals = liquid[liquid["option_type"] == opt_type]["price_error"].dropna()
            if len(vals) > 0:
                ax.axvline(vals.mean(), color=col, linestyle="-", linewidth=1.2,
                           label=f"{opt_type.capitalize()} mean = ${vals.mean():.2f}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Absolute Error: MC − SSVI ($)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Absolute Error Distribution (liquid)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 3: percentage error (%)
    ax = axes[2, 0]
    liq_pct = liquid.dropna(subset=["price_error_pct"])
    if len(liq_pct) > 0:
        _scatter_by_category(ax, liq_pct, "ttm", "price_error_pct", alpha=0.4, s=12)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time to Maturity (years)", fontsize=10)
    ax.set_ylabel("Percentage Error (%)", fontsize=10)
    ax.set_title(f"Percentage Error vs Maturity (liquid, n={len(liq_pct)})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if len(liquid) > 0:
        cats = _classify_options(liquid)
        pct_all = liquid["price_error_pct"].dropna()
        bins = np.linspace(pct_all.min() - 1, pct_all.max() + 1, 40)
        for label, style in CATEGORY_STYLE.items():
            vals = liquid.loc[cats == label, "price_error_pct"].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=bins, alpha=0.5, color=style["color"],
                        edgecolor="white", label=f"{label} ({len(vals)})")
        ax.axvline(pct_all.mean(), color="red", linestyle="-", linewidth=1.2,
                   label=f"Mean = {pct_all.mean():.2f}%")
        for opt_type, col in [("call", "#1f77b4"), ("put", "#d62728")]:
            vals = liquid[liquid["option_type"] == opt_type]["price_error_pct"].dropna()
            if len(vals) > 0:
                ax.axvline(vals.mean(), color=col, linestyle="-", linewidth=1.2,
                           label=f"{opt_type.capitalize()} mean = {vals.mean():.2f}%")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Percentage Error (%)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Percentage Error Distribution (liquid)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle("Dupire MC vs SSVI Prices — All Options (Checkpoint 1)",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "mc_vs_vanilla_all.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {DIR_PLOTS}/mc_vs_vanilla_all.png")

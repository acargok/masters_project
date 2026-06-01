import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from config import *

logger = logging.getLogger(__name__)


# Section 4b — Dupire compatibility diagnostics

def validate_dupire_compatibility(
        total_var_surface: np.ndarray,
        ttm_grid: np.ndarray,
        log_m_grid: np.ndarray) -> dict:
    """
    Check whether the total-variance surface is usable for Dupire local vol.
    A surface smooth in value space can still give negative local variance if
    its k-derivatives are unstable. Evaluates, with w'=∂w/∂k, w''=∂²w/∂k²:

      ∂w/∂T       must be ≥ 0 (calendar spread)
      ∂²w/∂k²     negative curvature → butterfly arbitrage
      g(k,T)      Gatheral (2004) density: g = (1 − k·w'/(2w))²
                  − (w')²/4·(1/w + ¼) + w''/2;  g < 0 → imaginary local vol
      σ²_loc(k,T) = (∂w/∂T) / g

    ∂w/∂T is taken holding k = ln(K/F(T)) fixed (forward-curve correction
    ignored; small for smooth forwards). Returns summary stats plus g and
    local-variance arrays for plotting.
    """
    w = total_var_surface   # (n_k, n_T)

    # ∂w/∂k, ∂²w/∂k² via np.gradient on the uniform k-grid
    dk = float(log_m_grid[1] - log_m_grid[0])
    dw_dk   = np.gradient(w, dk, axis=0)   # (n_k, n_T)
    d2w_dk2 = np.gradient(dw_dk, dk, axis=0)

    # ∂w/∂T: forward difference, aligned to T index j
    dT    = np.diff(ttm_grid)                         # (n_T-1,)
    dw_dT = np.diff(w, axis=1) / dT[np.newaxis, :]   # (n_k, n_T-1)

    # Interior T points for consistent comparison
    w_int       = w[:, :-1]
    dw_dk_int   = dw_dk[:, :-1]
    d2w_dk2_int = d2w_dk2[:, :-1]
    k_grid      = log_m_grid[:, np.newaxis]           # (n_k, 1)

    # Gatheral g-function
    w_safe = np.maximum(w_int, 1e-8)
    h      = dw_dk_int
    hp     = d2w_dk2_int
    term1  = (1.0 - k_grid * h / (2.0 * w_safe)) ** 2
    term2  = h**2 / 4.0 * (1.0 / w_safe + 0.25)
    term3  = hp / 2.0
    g      = term1 - term2 + term3   # (n_k, n_T-1)

    # Dupire local variance
    local_var = np.where(np.abs(g) > 1e-10, dw_dT / g, np.nan)

    # Summary statistics
    g_neg_frac  = float(np.mean(g < 0))
    lv_finite   = local_var[~np.isnan(local_var)]
    lv_neg_frac = float(np.mean(lv_finite < 0)) if len(lv_finite) > 0 else float("nan")

    stats = {
        "dw_dT_min":              float(dw_dT.min()),
        "dw_dk_range":            (float(dw_dk.min()), float(dw_dk.max())),
        "d2w_dk2_min":            float(d2w_dk2.min()),
        "g_min":                  float(g.min()),
        "g_neg_fraction":         g_neg_frac,
        "local_var_min":          float(np.nanmin(local_var)),
        "local_var_neg_fraction": lv_neg_frac,
        # Arrays for plot_dupire_diagnostics
        "g_surface":              g,
        "local_var_surface":      local_var,
        "ttm_int_grid":           ttm_grid[:-1],
    }

    logger.info("\n" + "=" * 60)
    logger.info("  DUPIRE COMPATIBILITY CHECK")
    logger.info("=" * 60)
    ok_dT = "OK" if stats["dw_dT_min"] >= -1e-6 else "WARNING: negative"
    logger.info(f"  ∂w/∂T  min:               {stats['dw_dT_min']:+.6f}  ({ok_dT})")
    logger.info(f"  ∂²w/∂k² min:              {stats['d2w_dk2_min']:+.6f}")
    logger.info(f"  Gatheral g  min:           {stats['g_min']:+.6f}  "
                f"({'OK' if stats['g_min'] >= 0 else 'WARNING: negative density'})")
    logger.info(f"  Gatheral g  < 0 fraction:  {stats['g_neg_fraction']*100:.1f}%")
    logger.info(f"  Local var   min:           {stats['local_var_min']:+.6f}")
    logger.info(f"  Local var   < 0 fraction:  {stats['local_var_neg_fraction']*100:.1f}%")
    if stats["g_neg_fraction"] > 0.01:
        logger.warning("  *** Butterfly arbitrage present — Dupire will be unreliable ***")
    if stats["local_var_neg_fraction"] > 0.01:
        logger.warning("  *** Local variance negative in >1%% of grid — inspect short end ***")
    logger.info("=" * 60)

    return stats


def plot_dupire_diagnostics(dupire_stats: dict,
                            ttm_grid: np.ndarray,
                            log_m_grid: np.ndarray) -> None:
    """
    Two-panel heatmap: Gatheral g(k,T) (red = butterfly arb) and local
    variance σ²_loc (red = negative). These derivative-space quantities are
    not tested by in-sample repricing.
    """
    g         = dupire_stats["g_surface"]          # (n_k, n_T-1)
    local_var = dupire_stats["local_var_surface"]  # (n_k, n_T-1)
    ttm_int   = dupire_stats["ttm_int_grid"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    TTM_m, LM_m = np.meshgrid(ttm_int, log_m_grid)

    # Left: g(k, T)
    ax = axes[0]
    vmax = max(abs(float(g.min())), abs(float(g.max())), 1e-6)
    im = ax.pcolormesh(TTM_m, LM_m, g,
                       cmap="RdYlGn", vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label="g(k,T)")
    ax.set_xlabel("Time to Maturity (years)", fontsize=11)
    ax.set_ylabel("Forward Log-Moneyness k = ln(K/F)", fontsize=11)
    ax.set_title(f"Gatheral Density g(k,T)\n"
                 f"red = butterfly arb / negative RN density  "
                 f"[neg frac: {dupire_stats['g_neg_fraction']*100:.1f}%]",
                 fontsize=10)

    # Right: local variance
    ax = axes[1]
    lv_plot  = np.where(np.isnan(local_var), 0.0, local_var)
    vmax_lv  = float(np.nanpercentile(np.abs(local_var), 99))
    im2 = ax.pcolormesh(TTM_m, LM_m, lv_plot,
                        cmap="RdYlGn", vmin=-vmax_lv, vmax=vmax_lv, shading="auto")
    plt.colorbar(im2, ax=ax, label="σ²_loc(k,T)")
    ax.set_xlabel("Time to Maturity (years)", fontsize=11)
    ax.set_ylabel("Forward Log-Moneyness k = ln(K/F)", fontsize=11)
    ax.set_title(f"Dupire Local Variance σ²_loc(k,T)\n"
                 f"red = negative → imaginary local vol  "
                 f"[neg frac: {dupire_stats['local_var_neg_fraction']*100:.1f}%]",
                 fontsize=10)

    plt.suptitle("Dupire Compatibility Diagnostics (derivative-space)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "dupire_diagnostics.png"),
                dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {DIR_PLOTS}/dupire_diagnostics.png")

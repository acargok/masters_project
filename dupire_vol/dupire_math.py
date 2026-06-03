# -*- coding: utf-8 -*-
"""Dupire math: total-variance derivatives, Gatheral g, local vol (SECTIONS 3-4)."""

import logging

import numpy as np

from config import *

logger = logging.getLogger(__name__)


# SECTION 3: Total variance partial derivatives

def compute_tv_derivatives(
        w: np.ndarray,
        k_grid: np.ndarray,
        T_grid: np.ndarray) -> tuple:
    """
    Finite-difference partials of the total variance surface w(k, T).

    Uniform grids: central differences interior, one-sided at boundaries.
    w (n_k,n_T), k_grid, T_grid -> (dw_dT, dw_dk, d2w_dk2). Constraints:
    dw_dT >= 0 (calendar-arb free), d2w_dk2 >= 0 (no butterfly arb).
    """
    dk = k_grid[1] - k_grid[0]
    dT = T_grid[1] - T_grid[0]

    # ∂w/∂T (axis 1, TTM)
    dw_dT = np.zeros_like(w)
    dw_dT[:, 0]    = (w[:, 1] - w[:, 0]) / dT
    dw_dT[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2.0 * dT)
    dw_dT[:, -1]   = (w[:, -1] - w[:, -2]) / dT

    # T_min boundary: SSVI gives w(k,0)=0 (θ(0)=0). Unequal-spacing central
    # difference through {0, T_min, T_min+dT} anchored at T=0; h1=T_min,
    # h2=dT, f' = [h1²f(x+h2)+(h2²-h1²)f(x)-h2²f(x-h1)]/(h1·h2·(h1+h2)),
    # f(x-h1)=0.
    h1 = T_grid[0]
    h2 = dT
    dw_dT[:, 0] = (h1**2 * w[:, 1] + (h2**2 - h1**2) * w[:, 0]) / (h1 * h2 * (h1 + h2))

    # ∂w/∂k (axis 0, log-moneyness)
    dw_dk = np.zeros_like(w)
    dw_dk[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2.0 * dk)    # central
    dw_dk[0,    :] = (w[1,  :] - w[0,   :]) / dk             # forward
    dw_dk[-1,   :] = (w[-1, :] - w[-2,  :]) / dk             # backward

    # ∂²w/∂k² (axis 0)
    d2w_dk2 = np.zeros_like(w)
    d2w_dk2[1:-1, :] = (w[2:, :] - 2.0*w[1:-1, :] + w[:-2, :]) / dk**2
    d2w_dk2[0,    :] = (w[2,  :] - 2.0*w[1,    :] + w[0,   :]) / dk**2
    d2w_dk2[-1,   :] = (w[-1, :] - 2.0*w[-2,   :] + w[-3,  :]) / dk**2

    logger.info(f"∂w/∂T  range: [{dw_dT.min():.4e}, {dw_dT.max():.4e}]  "
                f"(neg fraction: {(dw_dT < 0).mean()*100:.1f}%)")
    logger.info(f"∂w/∂k  range: [{dw_dk.min():.4e}, {dw_dk.max():.4e}]")
    logger.info(f"∂²w/∂k² range: [{d2w_dk2.min():.4e}, {d2w_dk2.max():.4e}]")

    return dw_dT, dw_dk, d2w_dk2


# SECTION 4: Gatheral density and Dupire local vol

def compute_gatheral_g(
        w: np.ndarray,
        dw_dk: np.ndarray,
        d2w_dk2: np.ndarray,
        k_grid: np.ndarray) -> np.ndarray:
    """
    Gatheral (2004) risk-neutral density factor g(k, T).

        g = [1 − k·w_k/(2w)]² − (w_k²/4)·(1/4 + 1/w) + w_kk/2

    g >= 0 iff no butterfly arb; g = 0 at the density wings. Inputs w, dw_dk,
    d2w_dk2 (n_k,n_T) and k_grid -> g (n_k,n_T).
    """
    k = k_grid[:, np.newaxis]            # (n_k,1), broadcasts to (n_k,n_T)
    w_safe = np.maximum(w, 1e-12)

    term1 = (1.0 - k * dw_dk / (2.0 * w_safe)) ** 2
    term2 = -(dw_dk ** 2 / 4.0) * (0.25 + 1.0 / w_safe)
    term3 = d2w_dk2 / 2.0
    g = term1 + term2 + term3

    g_neg_frac = float((g < 0).mean())
    logger.info(f"Gatheral g  range: [{g.min():.4e}, {g.max():.4e}]  "
                f"(g < 0 fraction: {g_neg_frac*100:.1f}%)")
    if g_neg_frac > 0.01:
        logger.warning("  *** Butterfly arbitrage present (g < 0 in >1% of grid) — "
                       "local vol will be unreliable in those regions ***")

    return g


def compute_dupire_local_vol(
        w: np.ndarray,
        dw_dT: np.ndarray,
        g: np.ndarray) -> tuple:
    """
    Dupire local vol via Gatheral total-variance form: σ²_loc = (∂w/∂T)/g.

    Direct from SSVI total variance, no call-price conversion. Safeguards:
    g < MIN_G_VALUE flagged unreliable (1/0); ∂w/∂T floored to 0 (FD edge
    noise, SSVI is calendar-arb free); a "filled" surface clips variance to
    [floor², cap²] for MC, raw (NaN where unreliable) returned via mask.

    Inputs w, dw_dT, g (n_k,n_T). Returns (local_vol clipped/filled, mask
    bool reliable, diagnostics dict).
    """
    # SSVI is calendar-arb free; floor handles FD edge noise only.
    dw_dT_pos = np.maximum(dw_dT, 0.0)

    g_ok = g > MIN_G_VALUE
    n_neg_g = int((~g_ok).sum())

    local_var = np.full_like(w, np.nan)
    local_var[g_ok] = dw_dT_pos[g_ok] / g[g_ok]

    # Negative variance should only be boundary float noise (dw_dT floored).
    var_positive = np.isfinite(local_var) & (local_var > 0)
    n_neg_var = int((g_ok & ~var_positive).sum())

    mask = g_ok & var_positive

    # Filled surface for MC
    g_safe = np.maximum(g, MIN_G_VALUE)
    local_var_filled = dw_dT_pos / g_safe
    local_var_filled = np.clip(local_var_filled,
                               LOCAL_VOL_FLOOR ** 2, LOCAL_VOL_CAP ** 2)
    local_vol = np.sqrt(local_var_filled)

    n_total    = mask.size
    n_reliable = int(mask.sum())
    pct_reliable = 100.0 * n_reliable / n_total
    reliable_vals = local_vol[mask]

    diag = {
        "n_total":            n_total,
        "n_reliable":         n_reliable,
        "pct_reliable":       round(pct_reliable, 1),
        "n_neg_g":            n_neg_g,
        "n_neg_var":          n_neg_var,
        "local_vol_median":   round(float(np.median(reliable_vals)), 4) if n_reliable > 0 else None,
        "local_vol_mean":     round(float(np.mean(reliable_vals)),   4) if n_reliable > 0 else None,
        "local_vol_min":      round(float(np.min(reliable_vals)),    4) if n_reliable > 0 else None,
        "local_vol_max":      round(float(np.max(reliable_vals)),    4) if n_reliable > 0 else None,
    }

    logger.info(f"Dupire local vol: {pct_reliable:.1f}% reliable "
                f"({n_reliable}/{n_total} grid points)")
    logger.info(f"  g < MIN_G_VALUE: {n_neg_g} points")
    logger.info(f"  Negative variance: {n_neg_var} points")
    if n_reliable > 0:
        logger.info(f"  Reliable local vol: median={diag['local_vol_median']:.4f}, "
                    f"range=[{diag['local_vol_min']:.4f}, {diag['local_vol_max']:.4f}]")

    return local_vol, mask, diag

# -*- coding: utf-8 -*-
"""Dupire math: total-variance derivatives, Gatheral g, local vol (SECTIONS 3-4)."""

import logging

import numpy as np

from config import *

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 3: TOTAL VARIANCE PARTIAL DERIVATIVES
# ===========================================================================

def compute_tv_derivatives(
        w: np.ndarray,
        k_grid: np.ndarray,
        T_grid: np.ndarray) -> tuple:
    """
    Compute partial derivatives of the total variance surface w(k, T).

    Both grids are uniform, so all finite differences use constant spacing.
    Interior points use second-order central differences; boundaries use
    first-order one-sided differences.

    Parameters
    ----------
    w : np.ndarray, shape (n_k, n_T)
        Total variance surface.
    k_grid : np.ndarray, shape (n_k,)
        Uniform forward log-moneyness grid.
    T_grid : np.ndarray, shape (n_T,)
        Uniform TTM grid.

    Returns
    -------
    dw_dT : np.ndarray, shape (n_k, n_T)
        ∂w/∂T — should be ≥ 0 (calendar-spread free).
    dw_dk : np.ndarray, shape (n_k, n_T)
        ∂w/∂k — smile slope.
    d2w_dk2 : np.ndarray, shape (n_k, n_T)
        ∂²w/∂k² — smile convexity (≥ 0 for no butterfly arb).
    """
    dk = k_grid[1] - k_grid[0]   # uniform k spacing
    dT = T_grid[1] - T_grid[0]   # uniform T spacing

    # ── ∂w/∂T (along axis 1 — TTM) ──────────────────────────────────
    dw_dT = np.zeros_like(w)
    dw_dT[:, 0]    = (w[:, 1] - w[:, 0]) / dT
    dw_dT[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2.0 * dT)
    dw_dT[:, -1]   = (w[:, -1] - w[:, -2]) / dT

    # Boundary at T_min: SSVI guarantees w(k, 0) = 0 (θ(0) = 0). Use an
    # unequal-spacing central difference through {T=0, T_min, T_min+dT}, which
    # uses the T=0 anchor. h1 = T_min (back to T=0), h2 = dT (forward step):
    # f'(x) = [h1^2*f(x+h2) + (h2^2-h1^2)*f(x) - h2^2*f(x-h1)] / (h1*h2*(h1+h2))
    # with f(x-h1) = w(k, 0) = 0.
    h1 = T_grid[0]   # T_min
    h2 = dT
    dw_dT[:, 0] = (h1**2 * w[:, 1] + (h2**2 - h1**2) * w[:, 0]) / (h1 * h2 * (h1 + h2))

    # ── ∂w/∂k (along axis 0 — log-moneyness) ────────────────────────
    dw_dk = np.zeros_like(w)
    dw_dk[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2.0 * dk)    # central
    dw_dk[0,    :] = (w[1,  :] - w[0,   :]) / dk             # forward
    dw_dk[-1,   :] = (w[-1, :] - w[-2,  :]) / dk             # backward

    # ── ∂²w/∂k² (along axis 0) ──────────────────────────────────────
    d2w_dk2 = np.zeros_like(w)
    d2w_dk2[1:-1, :] = (w[2:, :] - 2.0*w[1:-1, :] + w[:-2, :]) / dk**2
    d2w_dk2[0,    :] = (w[2,  :] - 2.0*w[1,    :] + w[0,   :]) / dk**2
    d2w_dk2[-1,   :] = (w[-1, :] - 2.0*w[-2,   :] + w[-3,  :]) / dk**2

    logger.info(f"∂w/∂T  range: [{dw_dT.min():.4e}, {dw_dT.max():.4e}]  "
                f"(neg fraction: {(dw_dT < 0).mean()*100:.1f}%)")
    logger.info(f"∂w/∂k  range: [{dw_dk.min():.4e}, {dw_dk.max():.4e}]")
    logger.info(f"∂²w/∂k² range: [{d2w_dk2.min():.4e}, {d2w_dk2.max():.4e}]")

    return dw_dT, dw_dk, d2w_dk2


# ===========================================================================
# SECTION 4: GATHERAL DENSITY AND DUPIRE LOCAL VOL
# ===========================================================================

def compute_gatheral_g(
        w: np.ndarray,
        dw_dk: np.ndarray,
        d2w_dk2: np.ndarray,
        k_grid: np.ndarray) -> np.ndarray:
    """
    Compute the Gatheral (2004) risk-neutral density factor g(k, T).

    Definition
    ----------
    g(k, T) = [1 − k·(∂w/∂k)/(2w)]²
              − (∂w/∂k)²/4 · (1/4 + 1/w)
              + (∂²w/∂k²)/2

    Properties
    ----------
    g ≥ 0  ⟺  no butterfly arbitrage at (k, T).
    g = 0  gives the risk-neutral distribution's wings (density touches zero).

    Parameters
    ----------
    w : np.ndarray, shape (n_k, n_T)
        Total variance surface.
    dw_dk, d2w_dk2 : np.ndarray, shape (n_k, n_T)
        First and second k-derivatives of w.
    k_grid : np.ndarray, shape (n_k,)
        Forward log-moneyness grid.

    Returns
    -------
    g : np.ndarray, shape (n_k, n_T)
        Gatheral density factor.
    """
    k = k_grid[:, np.newaxis]            # (n_k, 1) → broadcasts to (n_k, n_T)
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
    Apply the Gatheral total-variance form of the Dupire formula.

    σ²_loc(k, T) = (∂w/∂T) / g(k, T)

    This is the most numerically direct path from SSVI total variance to
    local variance.  No call price conversion needed; no carry mismatch.
    All quantities are already in forward log-moneyness coordinates.

    Numerical stability safeguards
    ------------------------------
    1. Points where g < MIN_G_VALUE are flagged unreliable (would cause 1/0).
    2. Points where ∂w/∂T < 0 (calendar-spread arb due to numerical noise)
       are floored to 0 before division — they should not arise from SSVI
       but can appear at boundaries from finite-difference edge effects.
    3. A "filled" surface clips local variance to [floor², cap²] for use in
       the MC simulation.  The raw surface (with NaN at unreliable points)
       is returned separately.

    Parameters
    ----------
    w : np.ndarray, shape (n_k, n_T)
        Total variance surface (used only for reference; NaN handling).
    dw_dT : np.ndarray, shape (n_k, n_T)
        Calendar derivative ∂w/∂T.
    g : np.ndarray, shape (n_k, n_T)
        Gatheral density factor from compute_gatheral_g().

    Returns
    -------
    local_vol : np.ndarray, shape (n_k, n_T)
        Dupire local volatility surface (clipped, filled).
    mask : np.ndarray (bool), shape (n_k, n_T)
        True where the local vol is numerically reliable.
    diagnostics : dict
        Computation statistics for logging/saving.
    """
    # Floor ∂w/∂T to 0: calendar arb should not appear from SSVI, but
    # finite-difference edge effects can give tiny negative values.
    dw_dT_pos = np.maximum(dw_dT, 0.0)

    # Identify reliable region
    g_ok = g > MIN_G_VALUE
    n_neg_g = int((~g_ok).sum())

    # Compute local variance with safe division
    local_var = np.full_like(w, np.nan)
    local_var[g_ok] = dw_dT_pos[g_ok] / g[g_ok]

    # Negative local variance is not physical (dw_dT floored above, so
    # this should only occur at exact boundaries from floating-point noise)
    var_positive = np.isfinite(local_var) & (local_var > 0)
    n_neg_var = int((g_ok & ~var_positive).sum())

    # Combined reliability mask
    mask = g_ok & var_positive

    # Build filled surface for MC simulation
    g_safe = np.maximum(g, MIN_G_VALUE)
    local_var_filled = dw_dT_pos / g_safe
    local_var_filled = np.clip(local_var_filled,
                               LOCAL_VOL_FLOOR ** 2, LOCAL_VOL_CAP ** 2)
    local_vol = np.sqrt(local_var_filled)

    # Diagnostics
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

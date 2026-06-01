#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bergomi vol-of-vol/skew formulas (flat forward-variance approximation) and
benchmarks, for the two-stage parameter calibration."""

import numpy as np
from scipy import interpolate

from bergomi_param_config import (
    DEFAULT_SIGMA0, DEFAULT_TAU0, DEFAULT_ALPHA, SKEW_DELTA_K,
)


def alpha_theta(theta, rho12):
    """Wang eq. 3.10 normalisation."""
    denom = np.sqrt((1.0 - theta) ** 2 + theta ** 2
                    + 2.0 * rho12 * theta * (1.0 - theta))
    return 1.0 / max(denom, 1e-12)


def A_i(kappa, T):
    """A_i(T) under flat forward variance curve (Wang 4.1)."""
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    small = kappa * T < 1e-8
    out[small] = 1.0
    big = ~small
    out[big] = (1.0 - np.exp(-kappa * T[big])) / (kappa * T[big])
    return out


def vol_of_vol_model(T, nu, theta, kappa1, kappa2, rho12):
    """nu_t^T under Wang 3.10 with flat forward variance (= flat A weights)."""
    a_th = alpha_theta(theta, rho12)
    a1 = A_i(kappa1, T)
    a2 = A_i(kappa2, T)
    var = (1 - theta) ** 2 * a1 ** 2 \
        + theta ** 2 * a2 ** 2 \
        + 2.0 * rho12 * theta * (1 - theta) * a1 * a2
    var = np.maximum(var, 0.0)
    return nu * a_th * np.sqrt(var)


def skew_g(x):
    """g(x) = x - (1 - exp(-x))."""
    x = np.asarray(x, dtype=float)
    return x - (1.0 - np.exp(-x))


def skew_order1_model(T, nu, theta, kappa1, kappa2, rho12, rho1, rho2):
    """Bergomi-Guyon order-1 ATMF skew under flat forward variance (Wang 4.1)."""
    a_th = alpha_theta(theta, rho12)
    k1T = np.maximum(kappa1 * T, 1e-12)
    k2T = np.maximum(kappa2 * T, 1e-12)
    term1 = (1 - theta) * rho1 * skew_g(k1T) / (k1T ** 2)
    term2 =       theta  * rho2 * skew_g(k2T) / (k2T ** 2)
    return nu * a_th * (term1 + term2)


# Benchmarks

def vol_of_vol_benchmark(T, sigma0=DEFAULT_SIGMA0, tau0=DEFAULT_TAU0,
                          alpha=DEFAULT_ALPHA):
    """Power-law vol-of-vol benchmark (Wang 4.3)."""
    return sigma0 * (tau0 / np.asarray(T, dtype=float)) ** alpha


def empirical_atmf_skew(iv_surface, log_m_grid, ttm_grid, T_query,
                         delta=SKEW_DELTA_K):
    """Empirical ATMF skew at T_query from the SSVI surface:
    skew(T) = (sigma(F(1+delta),T) - sigma(F(1-delta),T)) / (2*delta),
    via bilinear interpolation in log-moneyness k=log(K/F) and T."""
    interp = interpolate.RegularGridInterpolator(
        (log_m_grid, ttm_grid), iv_surface,
        method="linear", bounds_error=False, fill_value=None,
    )
    k_pos = np.log(1.0 + delta)
    k_neg = np.log(1.0 - delta)
    T_query = np.asarray(T_query, dtype=float)
    pts_pos = np.column_stack([np.full_like(T_query, k_pos), T_query])
    pts_neg = np.column_stack([np.full_like(T_query, k_neg), T_query])
    sig_pos = interp(pts_pos)
    sig_neg = interp(pts_neg)
    return (sig_pos - sig_neg) / (2.0 * delta)

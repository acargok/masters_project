#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variance swap vol extraction from SSVI (Step 3a).

Two methods are provided:
  "carr_madan": Carr-Madan (1998) / Demeterfi-Derman-Kamal-Zou (1999)
                model-free replication. The default.
  "proxy":      uniform smile-average of total variance over the log-moneyness
                grid; a quick approximation.
"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

from bergomi_calib_config import VS_METHOD_DEFAULT


def _bs_und_call_put_forward(F, K, T, sigma):
    """
    Undiscounted Black-76 European call/put on the *forward* (no rates,
    no dividends — applied separately by the caller). Returns (C, P) where:

        C = F * N(d1) - K * N(d2)
        P = K * N(-d2) - F * N(-d1)
        d1 = (ln(F/K) + 0.5 sigma^2 T) / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)

    These are the prices to be received at maturity if exercised — the
    quantity that the Carr-Madan replication formula consumes.
    """
    if T <= 0 or sigma <= 0:
        intrinsic_C = max(F - K, 0.0)
        intrinsic_P = max(K - F, 0.0)
        return intrinsic_C, intrinsic_P
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    C = F * norm.cdf(d1) - K * norm.cdf(d2)
    P = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return C, P


def extract_vs_vol_proxy(iv_surface, log_m_grid, ttm_grid):
    """
    Smile-averaged total variance proxy (NOT the model-free VS strike).

    sigma_VS_proxy^2(T) = [integral w(k, T) dk] / [T * (k_max - k_min)]

    Dependent on grid width.
    """
    n_T = len(ttm_grid)
    vs_vol = np.zeros(n_T)
    k_range = log_m_grid[-1] - log_m_grid[0]
    for j in range(n_T):
        T = ttm_grid[j]
        if T <= 0:
            continue
        total_var_slice = iv_surface[:, j] ** 2 * T
        fair_var = np.trapz(total_var_slice, log_m_grid) / k_range
        vs_vol[j] = np.sqrt(max(fair_var, 1e-8))
    return vs_vol


def extract_vs_vol_carr_madan(iv_surface, log_m_grid, ttm_grid, fwd_curve,
                                n_quad=2001):
    """
    Carr-Madan variance swap replication (Carr & Madan 1998; Demeterfi,
    Derman, Kamal, Zou 1999). The model-free fair variance is:

        sigma_VS^2(T) = (2 / T) * [
            int_0^F  P(K, T) / K^2  dK
          + int_F^inf C(K, T) / K^2  dK
        ]

    In log-moneyness k = ln(K/F), with K = F e^k and dK = K dk, the K^2 in
    the denominator becomes K, giving:

        sigma_VS^2(T) = (2/T) * [
            int_{-inf}^0 P(F e^k, T) / (F e^k) dk
          + int_0^{inf}  C(F e^k, T) / (F e^k) dk
        ]

    P, C are *undiscounted* (forward) European prices using the SSVI IV at
    each (k, T). For each maturity we evaluate the integrand on the SSVI
    log-moneyness grid and integrate via Simpson's rule.

    Parameters
    ----------
    iv_surface : np.ndarray, shape (n_k, n_T)
    log_m_grid : np.ndarray, shape (n_k,)
        Log-moneyness in the convention k = ln(K/F).
    ttm_grid : np.ndarray, shape (n_T,)
    fwd_curve : np.ndarray, shape (n_T,) or (n_T, 2)
        Per-expiry forward F(0, T). If 2D, expects [[T, F], ...] but only
        the F column is used; we assume rows align with ttm_grid.

    Returns
    -------
    np.ndarray, shape (n_T,)
        Variance swap volatility per maturity.
    """
    if fwd_curve.ndim == 2:
        F_arr = np.asarray(fwd_curve[:, 1], dtype=float)
    else:
        F_arr = np.asarray(fwd_curve, dtype=float)
    if F_arr.shape[0] != len(ttm_grid):
        raise ValueError(
            f"forward curve length {F_arr.shape[0]} != ttm_grid length {len(ttm_grid)}"
        )

    n_T = len(ttm_grid)
    vs_vol = np.zeros(n_T)
    k_src = np.asarray(log_m_grid, dtype=float)

    # Fine integration grid: denser than the SSVI grid so Simpson's rule
    # resolves the integrand at short maturities (where C/K and P/K are
    # sharply peaked around k=0 with width ~ sigma*sqrt(T)). A 60-point
    # SSVI grid over [-0.8, 0.8] has spacing 0.027 — too coarse for T=0.05
    # where sigma*sqrt(T) ~= 0.045. We interpolate IV linearly onto a fine
    # k-grid covering the same range and integrate there.
    k_fine = np.linspace(k_src[0], k_src[-1], n_quad)
    put_mask = k_fine <= 0.0
    call_mask = k_fine >= 0.0

    for j in range(n_T):
        T = float(ttm_grid[j])
        if T <= 0:
            continue
        F = float(F_arr[j])
        # Linear interpolation of IV onto fine k grid (no extrapolation needed
        # since k_fine spans exactly the same range as k_src).
        sigma_fine = np.interp(k_fine, k_src, iv_surface[:, j])
        K_fine = F * np.exp(k_fine)

        # Vectorised Black-76 in k-space:
        # C/K = exp(-k) N(d1) - N(d2),  P/K = N(-d2) - exp(-k) N(-d1)
        # with d1 = (-k + 0.5 sigma^2 T) / (sigma sqrt(T)), d2 = d1 - sigma sqrt(T).
        sqrt_T = np.sqrt(T)
        sigT = sigma_fine * sqrt_T
        # Avoid division by zero where sigma is degenerate
        sigT_safe = np.maximum(sigT, 1e-12)
        d1 = (-k_fine + 0.5 * sigma_fine ** 2 * T) / sigT_safe
        d2 = d1 - sigT_safe
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        C_over_K = np.exp(-k_fine) * N_d1 - N_d2
        P_over_K = (1.0 - N_d2) - np.exp(-k_fine) * (1.0 - N_d1)

        integrand = np.where(put_mask, P_over_K, C_over_K)
        # Integrate the two halves separately. With odd n_quad and the
        # midpoint exactly at k=0 (when the grid is symmetric around 0),
        # Simpson's rule on each half is well-defined and doesn't double-count.
        put_integral = integrate.simpson(integrand[put_mask], k_fine[put_mask])
        call_integral = integrate.simpson(integrand[call_mask], k_fine[call_mask])
        sigma_VS_sq = (2.0 / T) * (put_integral + call_integral)
        vs_vol[j] = np.sqrt(max(sigma_VS_sq, 1e-8))

    return vs_vol


def extract_vs_vol(iv_surface, log_m_grid, ttm_grid, fwd_curve=None,
                    method=VS_METHOD_DEFAULT):
    """
    Dispatch to the chosen VS-vol extraction method.

    method="carr_madan" requires fwd_curve. method="proxy" ignores it.
    """
    if method == "carr_madan":
        if fwd_curve is None:
            raise ValueError("Carr-Madan extraction requires fwd_curve")
        return extract_vs_vol_carr_madan(iv_surface, log_m_grid, ttm_grid, fwd_curve)
    if method == "proxy":
        return extract_vs_vol_proxy(iv_surface, log_m_grid, ttm_grid)
    raise ValueError(f"Unknown VS extraction method: {method!r}")


def fit_vs_vol_parametric_legacy(ttm_grid, vs_vol):
    """
    Monotone three-parameter VS-vol fit (Wang eq. 5.2).

    sigma_VS(t) = z2 + (z1 - z2) * exp(-z3 * t)

    Provides a parity comparison against the NSS fit. The pipeline routes the
    NSS form through `fit_vs_vol_nss`, since the three-parameter form cannot
    represent non-monotone (humped) VS-vol curves observed on SPX data.
    """
    def objective(params):
        z1, z2, z3 = params
        fitted = z2 + (z1 - z2) * np.exp(-z3 * ttm_grid)
        return np.sum((fitted - vs_vol)**2)

    z1_0 = vs_vol[0]
    z2_0 = vs_vol[-1]
    z3_0 = 1.0

    result = optimize.minimize(
        objective, [z1_0, z2_0, z3_0],
        bounds=[(0.01, 2.0), (0.01, 2.0), (0.01, 50.0)],
        method="L-BFGS-B",
    )

    z1, z2, z3 = result.x
    fitted = z2 + (z1 - z2) * np.exp(-z3 * ttm_grid)

    return {"z1": float(z1), "z2": float(z2), "z3": float(z3)}, fitted

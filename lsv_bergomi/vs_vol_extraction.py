#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variance-swap vol extraction from SSVI (Step 3a).
Methods: "carr_madan" (Carr-Madan / DDKZ model-free replication, default) and
"proxy" (uniform smile-average of total variance, grid-dependent)."""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

from bergomi_calib_config import VS_METHOD_DEFAULT


def _bs_und_call_put_forward(F, K, T, sigma):
    """Undiscounted Black-76 call/put on the forward (rates/divs applied by
    caller). Returns (C, P) = (F N(d1)-K N(d2), K N(-d2)-F N(-d1)); the
    maturity-payoff prices the Carr-Madan replication consumes."""
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
    """Smile-averaged total-variance proxy (NOT the model-free VS strike):
    sigma_VS_proxy^2(T) = int w(k,T)dk / [T (k_max-k_min)]. Grid-width dependent."""
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
    """Carr-Madan / DDKZ variance-swap replication. Model-free fair variance,
    in log-moneyness k=ln(K/F) (K=F e^k, dK=K dk):
        sigma_VS^2(T) = (2/T)[int_{-inf}^0 P/(F e^k) dk + int_0^inf C/(F e^k) dk]
    with undiscounted (forward) prices from the SSVI IV, integrated by Simpson.

    iv_surface (n_k,n_T); log_m_grid k=ln(K/F) (n_k,); ttm_grid (n_T,);
    fwd_curve F(0,T), (n_T,) or (n_T,2) (F column, rows aligned to ttm_grid).
    Returns VS volatility per maturity (n_T,)."""
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

    # Fine grid: the SSVI grid is too coarse for short maturities, where C/K
    # and P/K are peaked at k=0 with width ~sigma*sqrt(T). Interpolate IV onto
    # a denser k-grid over the same range and integrate there.
    k_fine = np.linspace(k_src[0], k_src[-1], n_quad)
    put_mask = k_fine <= 0.0
    call_mask = k_fine >= 0.0

    for j in range(n_T):
        T = float(ttm_grid[j])
        if T <= 0:
            continue
        F = float(F_arr[j])
        sigma_fine = np.interp(k_fine, k_src, iv_surface[:, j])
        K_fine = F * np.exp(k_fine)

        # Vectorised Black-76 in k-space:
        # C/K = e^-k N(d1) - N(d2), P/K = N(-d2) - e^-k N(-d1),
        # d1 = (-k + 0.5 sigma^2 T)/(sigma sqrt(T)), d2 = d1 - sigma sqrt(T).
        sqrt_T = np.sqrt(T)
        sigT = sigma_fine * sqrt_T
        sigT_safe = np.maximum(sigT, 1e-12)   # guard degenerate sigma
        d1 = (-k_fine + 0.5 * sigma_fine ** 2 * T) / sigT_safe
        d2 = d1 - sigT_safe
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        C_over_K = np.exp(-k_fine) * N_d1 - N_d2
        P_over_K = (1.0 - N_d2) - np.exp(-k_fine) * (1.0 - N_d1)

        integrand = np.where(put_mask, P_over_K, C_over_K)
        # Integrate each half separately; with odd n_quad and k=0 at the
        # midpoint, Simpson is well-defined and avoids double-counting.
        put_integral = integrate.simpson(integrand[put_mask], k_fine[put_mask])
        call_integral = integrate.simpson(integrand[call_mask], k_fine[call_mask])
        sigma_VS_sq = (2.0 / T) * (put_integral + call_integral)
        vs_vol[j] = np.sqrt(max(sigma_VS_sq, 1e-8))

    return vs_vol


def extract_vs_vol(iv_surface, log_m_grid, ttm_grid, fwd_curve=None,
                    method=VS_METHOD_DEFAULT):
    """Dispatch VS-vol extraction. "carr_madan" needs fwd_curve; "proxy" ignores it."""
    if method == "carr_madan":
        if fwd_curve is None:
            raise ValueError("Carr-Madan extraction requires fwd_curve")
        return extract_vs_vol_carr_madan(iv_surface, log_m_grid, ttm_grid, fwd_curve)
    if method == "proxy":
        return extract_vs_vol_proxy(iv_surface, log_m_grid, ttm_grid)
    raise ValueError(f"Unknown VS extraction method: {method!r}")


def fit_vs_vol_parametric_legacy(ttm_grid, vs_vol):
    """Monotone 3-parameter VS-vol fit (Wang 5.2):
    sigma_VS(t) = z2 + (z1-z2)exp(-z3 t). Parity comparison against fit_vs_vol_nss;
    the pipeline uses NSS since this form cannot fit humped SPX VS-vol curves."""
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

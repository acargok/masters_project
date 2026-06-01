#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Black-Scholes helpers and Heston semi-analytic pricing (Step 3a)."""

import numpy as np
from scipy import optimize
from scipy.stats import norm


# --- Black-Scholes helpers ---

def bs_call_price(S, K, T, r, q, sigma):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, q, sigma):
    """Black-Scholes vega (sensitivity to volatility)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def bs_implied_vol(price, S, K, T, r, q, option_type="call", tol=1e-8, max_iter=100):
    """Invert BS for implied vol via Brent on [1e-4, 5.0]; NaN if it fails."""
    if T <= 0:
        return np.nan

    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if price <= intrinsic + 1e-10:
        return np.nan

    def objective(sigma):
        c = bs_call_price(S, K, T, r, q, sigma)
        if option_type == "put":
            c = c - S * np.exp(-q * T) + K * np.exp(-r * T)
        return c - price

    try:
        iv = optimize.brentq(objective, 1e-4, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# --- Heston characteristic function and semi-analytic pricing ---

def _heston_P(u, T, kappa, theta, xi, rho, V0, j):
    """
    Heston (1993) char-fn component for P_j in C = S*exp(-qT)*P1 - K*exp(-rT)*P2.

    u: real integration variable (N,). j: 1 = stock numeraire, 2 = bond.
    Returns the complex integrand (N,) before Re and dividing by u.
    """
    i = 1j

    if j == 1:
        b = kappa - rho * xi
        u_j = 0.5
    else:
        b = kappa
        u_j = -0.5

    # d_j = sqrt((rho*xi*i*u - b)^2 - xi^2*(2*u_j*i*u - u^2))
    d = np.sqrt(
        (rho * xi * i * u - b)**2
        - xi**2 * (2.0 * u_j * i * u - u**2)
    )

    # Little Heston trap (Lord & Kahl 2006): g=(beta-d)/(beta+d) with exp(-dT),
    # keeps |g*exp(-dT)| < 1, avoiding overflow/branch-cut for large T or xi.
    beta = b - rho * xi * i * u
    g = (beta - d) / (beta + d)

    exp_neg_dT = np.exp(-d * T)

    C = (kappa * theta / xi**2) * (
        (beta - d) * T
        - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
    )

    D = ((beta - d) / xi**2) * (
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    )

    return np.exp(C + D * V0)


def heston_call_price_vectorised(S, K_arr, T_arr, r, q, kappa, theta, xi, rho, V0,
                                  N_quad=200, upper_limit=100.0):
    """
    Vectorised Heston European calls via P1/P2 decomposition with GL quadrature:

        C = S*exp(-qT)*P1 - K*exp(-rT)*P2
        P_j = 1/2 + (1/pi) int_0^inf Re[exp(-iu ln(K/S)) f_j(u) / (iu)] du

    Uses the little Heston trap. K_arr/T_arr: same length. N_quad: GL nodes.
    upper_limit: Fourier integration cap. Returns call prices.
    """
    n = len(K_arr)
    prices = np.zeros(n)

    # GL nodes/weights mapped to [0, upper_limit]
    nodes, weights_gl = np.polynomial.legendre.leggauss(N_quad)
    u_vals = 0.5 * upper_limit * (nodes + 1.0)
    w_vals = 0.5 * upper_limit * weights_gl

    # 1/(iu), guarding u~0
    inv_iu = np.zeros(N_quad, dtype=complex)
    safe = u_vals > 1e-12
    inv_iu[safe] = 1.0 / (1j * u_vals[safe])

    # Batch by unique maturity to reuse char-fn
    unique_T = np.unique(T_arr)
    for T in unique_T:
        if T <= 1e-8:
            mask_T = T_arr == T
            prices[mask_T] = np.maximum(
                S * np.exp(-q * T) - K_arr[mask_T] * np.exp(-r * T), 0.0
            )
            continue

        # Char-fn for this maturity (once)
        f1 = _heston_P(u_vals, T, kappa, theta, xi, rho, V0, j=1)
        f2 = _heston_P(u_vals, T, kappa, theta, xi, rho, V0, j=2)

        # Pre-multiply by weights and 1/(iu)
        wf1 = w_vals * f1 * inv_iu
        wf2 = w_vals * f2 * inv_iu

        mask_T = np.where(T_arr == T)[0]
        for idx in mask_T:
            K = K_arr[idx]
            x = np.log(S / K) + (r - q) * T

            exp_iux = np.exp(1j * u_vals * x)

            P1 = 0.5 + (1.0 / np.pi) * np.real(np.sum(exp_iux * wf1))
            P2 = 0.5 + (1.0 / np.pi) * np.real(np.sum(exp_iux * wf2))

            P1 = np.clip(P1, 0.0, 1.0)
            P2 = np.clip(P2, 0.0, 1.0)

            call = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
            prices[idx] = max(call, 0.0)

    return prices

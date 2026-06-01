#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dupire surface query and kernel-weighted conditional expectation (Step 3b).

Extracted from particle_method.py: query_dupire, conditional_expectation_kernel,
and nw_cv_bandwidth. Imported by the particle_method facade via
`from leverage_estimation import *`.
"""

import numpy as np


# =============================================================================
# Dupire surface query
# =============================================================================

def query_dupire(dupire_interp, spot_arr, t, S0, r, q, log_m_grid, ttm_grid,
                 fwd_curve=None):
    """
    Query the Dupire local vol surface at (spot, t) points.

    Converts spot to forward log-moneyness k = log(S_t / F(0,t)) using the
    per-expiry forward F(0,t) (from put-call parity, interpolated from fwd_curve
    if given; otherwise the flat approx S0*exp((r-q)*t)).

    Inputs: dupire_interp (2D interpolator in (log_fwd_moneyness, ttm)); spot_arr
    (particle spots); t; S0; r, q (used only if fwd_curve is None); log_m_grid,
    ttm_grid (clamping bounds); fwd_curve (optional [[T, F], ...]).
    Returns the local vol at each particle position.
    """
    if fwd_curve is not None:
        F_0_t = float(np.interp(t, fwd_curve[:, 0], fwd_curve[:, 1]))
        F_0_t = max(F_0_t, 1e-6)
    else:
        F_0_t = S0 * np.exp((r - q) * t)

    log_m = np.log(np.maximum(spot_arr, 1e-6) / F_0_t)
    log_m_clamped = np.clip(log_m, log_m_grid[0], log_m_grid[-1])
    t_clamped = np.clip(t, ttm_grid[0], ttm_grid[-1])

    pts = np.column_stack([log_m_clamped, np.full(len(spot_arr), t_clamped)])
    sigma_dupire = dupire_interp(pts)

    # Safety: ensure non-negative
    sigma_dupire = np.maximum(sigma_dupire, 1e-4)

    return sigma_dupire


# =============================================================================
# Kernel-weighted conditional expectation
# =============================================================================

def conditional_expectation_kernel(S_particles, V_particles, S_query, bandwidth):
    """
    Estimate E[V | S = S_query] by Gaussian kernel smoothing over the particles
    (Nadaraya-Watson regression).

    Inputs: S_particles, V_particles (spots and variances, shape (N,)); S_query
    (query spots, shape (M,) or scalar); bandwidth h.
    Returns E[V | S = S_query], shape (M,).
    """
    S_query = np.atleast_1d(S_query)
    M = len(S_query)
    N = len(S_particles)

    # Gaussian kernel weights K_h(S_j - S_query_i), shape (M, N)
    diff = S_query[:, None] - S_particles[None, :]   # (M, N)
    kernel_vals = np.exp(-0.5 * (diff / bandwidth)**2) / bandwidth   # (M, N)

    # Weighted average
    numerator = kernel_vals @ V_particles              # (M,)
    denominator = kernel_vals.sum(axis=1)              # (M,)

    # Avoid division by zero in sparse regions
    safe_denom = np.maximum(denominator, 1e-30)
    E_V_given_S = numerator / safe_denom

    return E_V_given_S


def nw_cv_bandwidth(S_particles, V_particles, n_subsample=500, n_h=15):
    """
    Select the NW kernel-regression bandwidth by leave-one-out CV, minimising

        CV(h) = (1/n) * sum_i (V_i - m_{-i}(S_i))^2

    where m_{-i}(S_i) is the NW estimate of E[V|S=S_i] leaving out particle i,
    computed without refitting as

        m_{-i}(S_i) = (num_i - K_h(0) * V_i) / (den_i - K_h(0)),  K_h(0) = 1/h.

    Subsamples particles for speed; the bandwidth grid is centred on the
    Silverman reference for the subsample.

    Inputs: S_particles, V_particles; n_subsample (max CV subsample, default
    500); n_h (bandwidth grid size, default 15). Returns the optimal bandwidth.
    """
    N = len(S_particles)
    if N > n_subsample:
        idx = np.random.choice(N, n_subsample, replace=False)
        S = S_particles[idx]
        V = V_particles[idx]
    else:
        S = S_particles
        V = V_particles
    n = len(S)

    # Bandwidth grid: 0.3x to 3x Silverman as reference range
    h_ref = max(1.06 * np.std(S) * n**(-0.2), 1e-6)
    h_grid = np.linspace(0.3 * h_ref, 3.0 * h_ref, n_h)

    best_h = h_ref
    best_cv = np.inf

    diff = S[:, None] - S[None, :]   # (n, n)

    for h in h_grid:
        K = np.exp(-0.5 * (diff / h)**2) / h   # (n, n)

        num = K @ V             # (n,) — full NW numerator
        den = K.sum(axis=1)     # (n,) — full NW denominator

        # Remove self-contribution: K_h(0) = 1/h
        k0 = 1.0 / h
        num_loo = num - k0 * V
        den_loo = den - k0

        # Avoid near-zero denominators (isolated particles)
        mask = np.abs(den_loo) > 1e-20
        m_loo = np.where(mask, num_loo / np.where(mask, den_loo, 1.0), V)

        cv = np.mean((V - m_loo)**2)
        if cv < best_cv:
            best_cv = cv
            best_h = h

    return max(best_h, 1e-6)

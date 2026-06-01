#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dupire surface query and kernel-weighted conditional expectation (Step 3b)."""

import numpy as np


def query_dupire(dupire_interp, spot_arr, t, S0, r, q, log_m_grid, ttm_grid,
                 fwd_curve=None):
    """
    Local vol at (spot, t) from the Dupire surface; returns vol per particle.

    Converts spot to forward log-moneyness k = log(S_t/F(0,t)) using per-expiry
    F(0,t) (interpolated from fwd_curve if given, else flat S0*exp((r-q)t)).
    dupire_interp: 2D interp in (log_fwd_moneyness, ttm). r, q used only when
    fwd_curve is None. log_m_grid, ttm_grid: clamping bounds.
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

    sigma_dupire = np.maximum(sigma_dupire, 1e-4)   # non-negative

    return sigma_dupire


def conditional_expectation_kernel(S_particles, V_particles, S_query, bandwidth):
    """
    E[V | S = S_query] via Gaussian kernel (Nadaraya-Watson) smoothing.

    S_particles, V_particles: shape (N,). S_query: shape (M,) or scalar.
    Returns E[V | S = S_query], shape (M,).
    """
    S_query = np.atleast_1d(S_query)
    M = len(S_query)
    N = len(S_particles)

    # Gaussian kernel weights K_h(S_j - S_query_i), (M, N)
    diff = S_query[:, None] - S_particles[None, :]
    kernel_vals = np.exp(-0.5 * (diff / bandwidth)**2) / bandwidth

    numerator = kernel_vals @ V_particles
    denominator = kernel_vals.sum(axis=1)

    safe_denom = np.maximum(denominator, 1e-30)   # guard sparse regions
    E_V_given_S = numerator / safe_denom

    return E_V_given_S


def nw_cv_bandwidth(S_particles, V_particles, n_subsample=500, n_h=15):
    """
    NW bandwidth by leave-one-out CV, minimising CV(h) = (1/n) sum_i (V_i - m_{-i}(S_i))^2.

    LOO estimate without refitting: m_{-i}(S_i) = (num_i - K_h(0) V_i)/(den_i - K_h(0)),
    K_h(0) = 1/h. Subsamples for speed; bandwidth grid centred on the subsample's
    Silverman reference. n_subsample: max CV subsample. n_h: grid size.
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

    # Grid 0.3x to 3x Silverman reference
    h_ref = max(1.06 * np.std(S) * n**(-0.2), 1e-6)
    h_grid = np.linspace(0.3 * h_ref, 3.0 * h_ref, n_h)

    best_h = h_ref
    best_cv = np.inf

    diff = S[:, None] - S[None, :]

    for h in h_grid:
        K = np.exp(-0.5 * (diff / h)**2) / h

        num = K @ V             # full NW numerator
        den = K.sum(axis=1)

        # Remove self-contribution K_h(0) = 1/h
        k0 = 1.0 / h
        num_loo = num - k0 * V
        den_loo = den - k0

        mask = np.abs(den_loo) > 1e-20   # guard isolated particles
        m_loo = np.where(mask, num_loo / np.where(mask, den_loo, 1.0), V)

        cv = np.mean((V - m_loo)**2)
        if cv < best_cv:
            best_cv = cv
            best_h = h

    return max(best_h, 1e-6)

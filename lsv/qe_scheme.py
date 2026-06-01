#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Andersen QE scheme for the CIR variance process (Step 3b).

Extracted from particle_method.py: step_variance_qe and step_spot_qe_bk.
Imported by the particle_method facade via `from qe_scheme import *`.
"""

import numpy as np
from scipy.stats import norm

from particle_config import QE_PSI_C


# =============================================================================
# Andersen QE scheme for the CIR variance process
# =============================================================================
# Andersen (2008), "Simple and Efficient Simulation of the Heston Model", with
# the Broadie-Kaya log-spot decomposition.
#
# Exact CIR conditional moments of V_{t+dt} | V_t:
#     m  = theta + (V_t - theta) * exp(-kappa dt)
#     s2 = V_t * xi^2 * exp(-kappa dt) / kappa * (1 - exp(-kappa dt))
#        + theta * xi^2 / (2 kappa) * (1 - exp(-kappa dt))^2
#     psi = s2 / m^2
#
# psi <= psi_c: sample a moment-matched quadratic-Gaussian (Case A).
# psi >  psi_c: sample an exponential with a point mass at zero (Case B).
# psi_c = 1.5 (Andersen) keeps moment-matching error small in both branches.

def step_variance_qe(V, dt, kappa, theta, xi, Z, psi_c=QE_PSI_C):
    """
    Vectorised Andersen QE step for the CIR variance process.

    Inputs: V (current variance, shape (N,), clipped to >= 0); dt; CIR
    parameters kappa/theta/xi; Z (standard normals, shape (N,) — used directly
    in Case A and as U = Phi(Z) for inverse-CDF sampling in Case B); psi_c
    (switching threshold, default 1.5). Returns sampled V_{t+dt}, shape (N,).
    """
    if dt <= 0:
        return np.maximum(V, 0.0).copy()

    V = np.maximum(V, 0.0)
    e_kt = np.exp(-kappa * dt)
    one_minus = 1.0 - e_kt

    m = theta + (V - theta) * e_kt
    s2 = (V * xi * xi * e_kt / kappa) * one_minus \
         + (theta * xi * xi / (2.0 * kappa)) * one_minus * one_minus

    m_safe = np.maximum(m, 1e-30)
    psi = s2 / (m_safe * m_safe)

    V_new = np.empty_like(V)
    case_A = psi <= psi_c
    case_B = ~case_A

    # --- Case A: quadratic-Gaussian, V_new = a (b + Z)^2 ---
    if np.any(case_A):
        psi_A = psi[case_A]
        m_A = m[case_A]
        Z_A = Z[case_A]
        inv = 2.0 / np.maximum(psi_A, 1e-30)
        # b^2 = 2/psi - 1 + sqrt(2/psi) sqrt(2/psi - 1).
        # In Case A, psi <= psi_c <= 2 so inv >= 1, hence inv - 1 >= 0.
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        b = np.sqrt(b2)
        a = m_A / (1.0 + b2)
        V_new[case_A] = a * (b + Z_A) ** 2

    # --- Case B: exponential with point mass at zero ---
    if np.any(case_B):
        psi_B = psi[case_B]
        m_B = m[case_B]
        U_B = norm.cdf(Z[case_B])     # uniform on (0,1) from same Z
        p = (psi_B - 1.0) / (psi_B + 1.0)
        beta = (1.0 - p) / np.maximum(m_B, 1e-30)
        below_mass = U_B <= p
        V_new_B = np.zeros_like(U_B)
        if np.any(~below_mass):
            denom = np.maximum(1.0 - U_B[~below_mass], 1e-30)
            V_new_B[~below_mass] = -np.log(
                (1.0 - p[~below_mass]) / denom
            ) / beta[~below_mass]
        V_new[case_B] = V_new_B

    return np.maximum(V_new, 0.0)


def step_spot_qe_bk(log_S, V_old, V_new, L, dt, r, q, rho, kappa, theta, xi,
                    Z_perp, gamma1=0.5, gamma2=0.5):
    """
    Broadie-Kaya log-spot update consistent with Andersen QE. The spot increment
    uses the just-sampled V_new together with V_old:

        log S_{t+dt} = log S_t + (r - q) dt
                     - 0.5 * L^2 * V_bar * dt
                     + (rho * L / xi) * [V_new - V_old - kappa theta dt + kappa V_bar dt]
                     + L * sqrt((1 - rho^2) * V_bar * dt) * Z_perp

    with V_bar = gamma1*V_old + gamma2*V_new (central choice gamma1=gamma2=1/2).

    Inputs: log_S, V_old/V_new (variance at start/end of step), L (leverage at
    S_t, t), Z_perp (independent standard normal, orthogonal to V's noise), all
    shape (N,).
    """
    V_bar = gamma1 * V_old + gamma2 * V_new
    # Numerical safety: V_bar should be >= 0, but clip to avoid sqrt of tiny negatives
    V_bar_pos = np.maximum(V_bar, 0.0)
    drift_corr = V_new - V_old - kappa * theta * dt + kappa * V_bar * dt
    log_S_new = (
        log_S
        + (r - q) * dt
        - 0.5 * (L ** 2) * V_bar * dt
        + (rho * L / xi) * drift_corr
        + L * np.sqrt((1.0 - rho ** 2) * V_bar_pos * dt) * Z_perp
    )
    return log_S_new

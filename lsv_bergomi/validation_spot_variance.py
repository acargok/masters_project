#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bergomi spot variance computation (for the MC validation)."""

import numpy as np


# =============================================================================
# Bergomi spot variance computation (for MC validation)
# =============================================================================

def _compute_alpha_theta(theta_param, rho12):
    denom = np.sqrt((1 - theta_param)**2 + theta_param**2 + 2 * rho12 * theta_param * (1 - theta_param))
    return 1.0 / max(denom, 1e-10)


def _compute_chi(t, T, kappa1, kappa2, theta_param, rho12, alpha_theta):
    th = theta_param
    tau = T - t
    var_X1 = (1.0 - np.exp(-2.0 * kappa1 * t)) / (2.0 * kappa1) if kappa1 > 1e-10 else t
    var_X2 = (1.0 - np.exp(-2.0 * kappa2 * t)) / (2.0 * kappa2) if kappa2 > 1e-10 else t
    cov_X12 = rho12 * (1.0 - np.exp(-(kappa1 + kappa2) * t)) / (kappa1 + kappa2) if (kappa1 + kappa2) > 1e-10 else rho12 * t
    chi = alpha_theta**2 * (
        (1.0 - th)**2 * np.exp(-2.0 * kappa1 * tau) * var_X1
        + th**2 * np.exp(-2.0 * kappa2 * tau) * var_X2
        + 2.0 * (1.0 - th) * th * np.exp(-(kappa1 + kappa2) * tau) * cov_X12
    )
    return max(chi, 0.0)


def _spot_variance(X1, X2, t, bergomi, fwd_var_interp, ttm_grid):
    """Compute xi^t_t for arrays of particles."""
    nu = bergomi["nu"]
    theta = bergomi["theta"]
    kappa1 = bergomi["kappa1"]
    kappa2 = bergomi["kappa2"]
    rho12 = bergomi["rho12"]
    # See particle_method.py for empirical justification: omega = 2*nu.
    omega = 2.0 * nu
    alpha_th = _compute_alpha_theta(theta, rho12)

    x_t_t = alpha_th * ((1.0 - theta) * X1 + theta * X2)
    chi_t_t = _compute_chi(t, t, kappa1, kappa2, theta, rho12, alpha_th)
    f_val = np.exp(omega * x_t_t - 0.5 * omega**2 * chi_t_t)

    t_clamped = np.clip(t, ttm_grid[0], ttm_grid[-1])
    xi_t_0 = max(float(fwd_var_interp(t_clamped)), 1e-8)

    return np.maximum(xi_t_0 * f_val, 1e-8)

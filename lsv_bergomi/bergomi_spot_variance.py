#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bergomi two-factor spot variance helpers for the particle method."""

import numpy as np


# =============================================================================
# Bergomi two-factor helpers
# =============================================================================

def compute_alpha_theta(theta_param, rho12):
    """Normalisation constant alpha_theta."""
    denom = np.sqrt((1 - theta_param)**2 + theta_param**2 + 2 * rho12 * theta_param * (1 - theta_param))
    return 1.0 / max(denom, 1e-10)


def compute_chi(t, T, kappa1, kappa2, theta_param, rho12, alpha_theta):
    """
    chi(t, T) = Var[x^T_t], the integrated variance of x^T from 0 to t,
    built from the OU variances and covariance (Wang eq. 3.8):

        x^T_t = alpha_theta [ (1-theta) e^{-kappa1(T-t)} X^1_t
                               + theta e^{-kappa2(T-t)} X^2_t ]

        Var[X^i_t]        = (1 - e^{-2*kappa_i*t}) / (2*kappa_i)
        Cov[X^1_t, X^2_t] = rho12 * (1 - e^{-(kappa1+kappa2)*t}) / (kappa1+kappa2)

        chi(t, T) = alpha_theta^2 * [
            (1-theta)^2 * e^{-2*kappa1*(T-t)} * Var[X^1_t]
            + theta^2 * e^{-2*kappa2*(T-t)} * Var[X^2_t]
            + 2*(1-theta)*theta * e^{-(kappa1+kappa2)*(T-t)} * Cov[X^1_t, X^2_t]
        ]
    """
    th = theta_param
    tau = T - t  # time to maturity from current time

    var_X1 = (1.0 - np.exp(-2.0 * kappa1 * t)) / (2.0 * kappa1) if kappa1 > 1e-10 else t
    var_X2 = (1.0 - np.exp(-2.0 * kappa2 * t)) / (2.0 * kappa2) if kappa2 > 1e-10 else t
    cov_X12 = rho12 * (1.0 - np.exp(-(kappa1 + kappa2) * t)) / (kappa1 + kappa2) if (kappa1 + kappa2) > 1e-10 else rho12 * t

    chi = alpha_theta**2 * (
        (1.0 - th)**2 * np.exp(-2.0 * kappa1 * tau) * var_X1
        + th**2 * np.exp(-2.0 * kappa2 * tau) * var_X2
        + 2.0 * (1.0 - th) * th * np.exp(-(kappa1 + kappa2) * tau) * cov_X12
    )
    return max(chi, 0.0)


def compute_spot_variance(X1, X2, t, bergomi, fwd_var_interp, ttm_grid):
    """
    Compute spot variance xi^t_t for each particle.

    xi^t_t = xi^t_0 * f_t(t, x^t_t)

    where x^t_t = alpha_theta [(1-theta) X^1_t + theta X^2_t]
    (since T=t, e^{-kappa(T-t)} = 1)

    f_t(t, x) = exp(omega * x - omega^2/2 * chi(t, t))

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (N,)
        OU process states.
    t : float
        Current time.
    bergomi : dict
        Bergomi parameters.
    fwd_var_interp : callable
        Interpolator for xi^T_0.
    ttm_grid : np.ndarray
        TTM grid for clamping.

    Returns
    -------
    np.ndarray, shape (N,)
        Spot variance for each particle.
    """
    nu = bergomi["nu"]
    theta = bergomi["theta"]
    kappa1 = bergomi["kappa1"]
    kappa2 = bergomi["kappa2"]
    rho12 = bergomi["rho12"]
    # f_T(t, x) = exp(omega * x - omega^2/2 * chi). The vol-of-vol formula
    # nu_t^T = nu * alpha_theta * sqrt(...) (Wang eq. 3.10) is consistent with
    # omega = 2*nu in the simulator: Wang's nu in the vol-of-vol formula is half
    # the f_T coefficient, following the Bergomi 2008 single-factor convention.
    omega = 2.0 * nu
    alpha_th = compute_alpha_theta(theta, rho12)

    # x^t_t: at T=t, the exponential decay factors are e^0 = 1
    x_t_t = alpha_th * ((1.0 - theta) * X1 + theta * X2)

    # chi(t, t): variance of x^t at time t with T=t
    chi_t_t = compute_chi(t, t, kappa1, kappa2, theta, rho12, alpha_th)

    # f_t(t, x) = exp(omega * x - omega^2/2 * chi)
    f_val = np.exp(omega * x_t_t - 0.5 * omega**2 * chi_t_t)

    # xi^t_0: initial forward variance at maturity t
    t_clamped = np.clip(t, ttm_grid[0], ttm_grid[-1])
    xi_t_0 = float(fwd_var_interp(t_clamped))
    xi_t_0 = max(xi_t_0, 1e-8)

    # xi^t_t = xi^t_0 * f_t(t, x^t_t)
    spot_var = xi_t_0 * f_val
    spot_var = np.maximum(spot_var, 1e-8)

    return spot_var

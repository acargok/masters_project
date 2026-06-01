#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nelson-Siegel-Svensson VS-vol fit (Svensson 1994).

sigma_VS(T) = beta0
            + beta1 * f1(T, tau1)
            + beta2 * f2(T, tau1)
            + beta3 * f2(T, tau2)

with the two NSS basis functions
    f1(T, tau) = (1 - exp(-T/tau)) / (T/tau)               -> 1 as T -> 0
    f2(T, tau) = f1(T, tau) - exp(-T/tau)                  -> 0 as T -> 0

Six parameters: beta0 (long-term level), beta1 (short-rate slope),
beta2 / beta3 (signed magnitudes of the two humps at timescales tau1, tau2).

The two-hump extension over the original Nelson-Siegel (1987) three-parameter
form lets the curve resolve non-monotone term structures (e.g. the SPX VS-vol
hump near T ~ 0.6 driven by short-end event density) that the monotone Wang
5.2 form cannot represent.

Reference:
  Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest
  Rates: Sweden 1992-1994". IMF Working Paper WP/94/114.
"""

import logging

import numpy as np
from scipy import optimize

from bergomi_calib_config import _NSS_T_SMALL

logger = logging.getLogger(__name__)


def _nss_f1(T, tau):
    """(1 - exp(-T/tau)) / (T/tau);  limit = 1 as T -> 0."""
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    small = np.abs(T) < _NSS_T_SMALL
    out[small] = 1.0
    big = ~small
    u = T[big] / tau
    out[big] = (1.0 - np.exp(-u)) / u
    return out


def _nss_f2(T, tau):
    """f1(T, tau) - exp(-T/tau);  limit = 0 as T -> 0."""
    return _nss_f1(T, tau) - np.exp(-np.asarray(T, dtype=float) / tau)


def nss_vs_vol(T, beta0, beta1, beta2, beta3, tau1, tau2):
    """Nelson-Siegel-Svensson VS-vol curve evaluated at maturities `T`."""
    return (beta0
            + beta1 * _nss_f1(T, tau1)
            + beta2 * _nss_f2(T, tau1)
            + beta3 * _nss_f2(T, tau2))


def _nss_df1_dT(T, tau):
    """
    d/dT [(1 - exp(-T/tau)) / (T/tau)]
        = tau / T**2 * [(T/tau + 1) exp(-T/tau) - 1]                (T > 0)
        = -1 / (2 tau)                                              (T -> 0 limit)

    Limit verified by Taylor expansion:
        e^-u = 1 - u + u^2/2 - u^3/6 + ...
        (u+1) e^-u - 1 = -u^2/2 + u^3/3 + O(u^4)
        / u^2  ->  -1/2  as u -> 0
    """
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    small = np.abs(T) < _NSS_T_SMALL
    out[small] = -0.5 / tau
    big = ~small
    Tb = T[big]
    u = Tb / tau
    out[big] = tau * ((u + 1.0) * np.exp(-u) - 1.0) / (Tb * Tb)
    return out


def _nss_df2_dT(T, tau):
    """
    d/dT [f1(T, tau) - exp(-T/tau)]
        = df1/dT + (1/tau) exp(-T/tau)                              (T > 0)
        =  1 / (2 tau)                                              (T -> 0 limit)
    """
    T = np.asarray(T, dtype=float)
    return _nss_df1_dT(T, tau) + (1.0 / tau) * np.exp(-T / tau)


def nss_dvs_dT(T, beta0, beta1, beta2, beta3, tau1, tau2):
    """d sigma_VS / dT at maturities `T` (beta0 has no T dependence)."""
    return (beta1 * _nss_df1_dT(T, tau1)
            + beta2 * _nss_df2_dT(T, tau1)
            + beta3 * _nss_df2_dT(T, tau2))


def nss_fwd_variance(T, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Analytic initial forward variance under the NSS form:

        xi^T_0 = d/dT [ T * sigma_VS(T)^2 ]
               = sigma_VS(T)^2  +  2 T * sigma_VS(T) * dsigma_VS/dT
               = sigma_VS(T) * ( sigma_VS(T) + 2 T * dsigma_VS/dT ).

    The NSS curve is smooth in closed form, so the analytic derivative is exact
    at every grid point, including the T = 0 boundary.
    """
    T = np.asarray(T, dtype=float)
    sigma  = nss_vs_vol(T, beta0, beta1, beta2, beta3, tau1, tau2)
    dsigma = nss_dvs_dT(T, beta0, beta1, beta2, beta3, tau1, tau2)
    return sigma * (sigma + 2.0 * T * dsigma)


def fit_vs_vol_nss(ttm_grid, vs_vol, seed=42):
    """
    Fit the NSS VS-vol form by least squares.

    Parameters
    ----------
    ttm_grid : np.ndarray
    vs_vol   : np.ndarray
        Per-maturity VS volatility (e.g. Carr-Madan replication).
    seed : int
        Unused — kept for signature parity with the pipeline.

    Returns
    -------
    dict
        nss_beta_{0..3}, nss_tau_{1,2}, nss_rmse  (all floats).
    np.ndarray
        Fitted VS-vol curve evaluated on `ttm_grid`.
    """
    ttm = np.asarray(ttm_grid, dtype=float)
    target = np.asarray(vs_vol, dtype=float)

    def model(p):
        b0, b1, b2, b3, t1, t2 = p
        return nss_vs_vol(ttm, b0, b1, b2, b3, t1, t2)

    def objective(p):
        # L-BFGS-B doesn't support coupled bounds, so the tau2 > tau1 + 0.1
        # ordering constraint is enforced as a soft penalty.
        _, _, _, _, t1, t2 = p
        pen = 0.0
        if t2 <= t1 + 0.1:
            pen = 1.0e3 * (t1 + 0.1 - t2) ** 2
        return float(np.sum((model(p) - target) ** 2)) + pen

    x0 = [float(np.mean(target)), 0.0, 0.0, 0.0, 0.5, 2.0]
    bounds = [
        (0.05, 0.50),   # beta0  - asymptotic vol level
        (-0.30, 0.30),  # beta1  - short-rate slope (signed)
        (-0.30, 0.30),  # beta2  - first hump magnitude
        (-0.30, 0.30),  # beta3  - second hump magnitude
        (0.05, 2.0),    # tau1   - first hump timescale
        (0.15, 10.0),   # tau2   - second hump timescale (penalty enforces > tau1+0.1)
    ]
    result = optimize.minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-10},
    )

    b0, b1, b2, b3, t1, t2 = result.x
    fitted = model(result.x)
    rmse = float(np.sqrt(np.mean((fitted - target) ** 2)))
    if t2 <= t1 + 0.1:
        logger.warning(f"NSS fit: tau2 ordering soft-violated  tau1={t1:.4f}  tau2={t2:.4f}")

    logger.info(f"NSS fit: beta0={b0:+.4f}  beta1={b1:+.4f}  beta2={b2:+.4f}  beta3={b3:+.4f}  "
                f"tau1={t1:.4f}  tau2={t2:.4f}")
    logger.info(f"  Fit RMSE: {rmse:.6f}  ({rmse * 1e4:.1f} bp)   objective: {result.fun:.6e}")

    return {
        "nss_beta_0": float(b0), "nss_beta_1": float(b1),
        "nss_beta_2": float(b2), "nss_beta_3": float(b3),
        "nss_tau_1":  float(t1), "nss_tau_2":  float(t2),
        "nss_rmse":   rmse,
    }, fitted


def compute_forward_variance(ttm_grid, vs_vol_fitted):
    """
    Finite-difference computation of xi^T_0 from a gridded VS-vol curve.
    The pipeline uses `nss_fwd_variance` (analytic NSS derivative); this helper
    provides a parity diagnostic against the finite-difference form.
    """
    total_var = vs_vol_fitted**2 * ttm_grid
    fwd_var = np.gradient(total_var, ttm_grid)
    fwd_var = np.maximum(fwd_var, 1e-6)
    return fwd_var

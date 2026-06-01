#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bergomi spot variance, option-pool selection, and pure-Bergomi MC simulation
(no LSV leverage) for the pure-SV fit diagnostic.
"""

import numpy as np
import pandas as pd

from pure_sv_config import IV_DIR, MAX_OPTIONS, SEED


# =============================================================================
# Bergomi spot variance
# =============================================================================

def _alpha_theta(theta, rho12):
    denom = np.sqrt((1 - theta) ** 2 + theta ** 2 + 2 * rho12 * theta * (1 - theta))
    return 1.0 / max(denom, 1e-10)


def _chi_t_t(t, kappa1, kappa2, theta, rho12, alpha_th):
    """Var[x^t_t] under analytical OU moments — used for martingale correction."""
    if t <= 0:
        return 0.0
    var_X1 = (1.0 - np.exp(-2.0 * kappa1 * t)) / (2.0 * kappa1) if kappa1 > 1e-10 else t
    var_X2 = (1.0 - np.exp(-2.0 * kappa2 * t)) / (2.0 * kappa2) if kappa2 > 1e-10 else t
    cov_X12 = (rho12 * (1.0 - np.exp(-(kappa1 + kappa2) * t)) / (kappa1 + kappa2)
               if (kappa1 + kappa2) > 1e-10 else rho12 * t)
    chi = alpha_th ** 2 * (
        (1.0 - theta) ** 2 * var_X1
        + theta ** 2 * var_X2
        + 2.0 * (1.0 - theta) * theta * cov_X12
    )
    return max(chi, 0.0)


def _spot_variance(X1, X2, t, bergomi, fwd_var_interp, ttm_grid):
    nu = bergomi["nu"]
    theta = bergomi["theta"]
    kappa1 = bergomi["kappa1"]
    kappa2 = bergomi["kappa2"]
    rho12 = bergomi["rho12"]
    omega = 2.0 * nu  # see particle_method.py for empirical justification
    alpha_th = _alpha_theta(theta, rho12)
    x_t_t = alpha_th * ((1.0 - theta) * X1 + theta * X2)
    chi_t_t = _chi_t_t(t, kappa1, kappa2, theta, rho12, alpha_th)
    f_val = np.exp(omega * x_t_t - 0.5 * omega ** 2 * chi_t_t)
    t_clamped = np.clip(t, ttm_grid[0], ttm_grid[-1])
    xi_t_0 = max(float(fwd_var_interp(t_clamped)), 1e-8)
    return np.maximum(xi_t_0 * f_val, 1e-8)


# =============================================================================
# Option-pool selection
# =============================================================================

def select_option_pool(S, r, q, max_options=MAX_OPTIONS, seed=SEED):
    df = pd.read_csv(IV_DIR / "data" / "spx_iv_data.csv")
    fwd_m = df["strike"] / (S * np.exp((r - q) * df["ttm"]))
    otm = (((fwd_m < 1.0) & (df["option_type"] == "put")) |
           ((fwd_m >= 1.0) & (df["option_type"] == "call")))
    df = df[otm].copy()
    df = df[(df["ttm"] >= 0.04) & (df["ttm"] <= 2.0)]
    df = df[df["iv"].between(0.01, 2.0)].drop_duplicates(subset=["strike", "ttm"])
    if len(df) > max_options:
        df = df.reset_index(drop=True)
        ttm_bins = pd.cut(df["ttm"], bins=5, labels=False)
        m_col = df["strike"] / (S * np.exp((r - q) * df["ttm"]))
        m_bins = pd.cut(m_col, bins=5, labels=False)
        df["_strata"] = ttm_bins.astype(str) + "_" + m_bins.astype(str)
        per_stratum = max(1, max_options // df["_strata"].nunique())
        sampled = (df.groupby("_strata", group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), per_stratum),
                                                random_state=seed)))
        if len(sampled) < max_options:
            remaining = df.loc[~df.index.isin(sampled.index)]
            n_extra = min(max_options - len(sampled), len(remaining))
            if n_extra > 0:
                sampled = pd.concat([sampled, remaining.sample(n_extra, random_state=seed)])
        df = sampled.drop(columns="_strata").head(max_options).reset_index(drop=True)
    return df


# =============================================================================
# Pure-Bergomi MC simulation (no LSV leverage)
# =============================================================================

def simulate_bergomi_no_leverage(S0, r, q, bergomi, fwd_var_interp, ttm_grid,
                                   maturities_required, n_paths, dt, seed):
    """
    Simulate Bergomi spot dynamics *without* the LSV leverage function:

        dS/S = (r - q) dt + sqrt(xi^t_t) dW^S

    Snapshot S at each requested maturity. Returns dict {step_idx: S_arr}.
    """
    rho1 = bergomi["rho1"]
    rho2 = bergomi["rho2"]
    rho12 = bergomi["rho12"]
    kappa1 = bergomi["kappa1"]
    kappa2 = bergomi["kappa2"]

    rng = np.random.default_rng(seed)

    # Correlation matrix (regularise eigenvalues)
    corr = np.array([
        [1.0,   rho1,  rho2],
        [rho1,  1.0,   rho12],
        [rho2,  rho12, 1.0],
    ])
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr_pd, 1.0)
    L_chol = np.linalg.cholesky(corr_pd)

    T_max = max(maturities_required)
    n_steps = max(int(np.ceil(T_max / dt)), 20)
    dt_actual = T_max / n_steps
    sqrt_dt = np.sqrt(dt_actual)
    t_schedule = np.arange(n_steps + 1) * dt_actual

    # Map each maturity to nearest step index (forward of 0)
    step_of = {T: int(np.argmin(np.abs(t_schedule - T))) for T in maturities_required}
    required_steps = set(step_of.values())

    decay1 = np.exp(-kappa1 * dt_actual)
    decay2 = np.exp(-kappa2 * dt_actual)
    std1 = (np.sqrt((1.0 - np.exp(-2 * kappa1 * dt_actual)) / (2 * kappa1))
            if kappa1 > 1e-10 else sqrt_dt)
    std2 = (np.sqrt((1.0 - np.exp(-2 * kappa2 * dt_actual)) / (2 * kappa2))
            if kappa2 > 1e-10 else sqrt_dt)

    S = np.full(n_paths, S0, dtype=np.float64)
    X1 = np.zeros(n_paths, dtype=np.float64)
    X2 = np.zeros(n_paths, dtype=np.float64)
    snapshots = {}

    for step in range(1, n_steps + 1):
        t = (step - 1) * dt_actual
        xi_t_t = _spot_variance(X1, X2, t, bergomi, fwd_var_interp, ttm_grid)

        Z = L_chol @ rng.standard_normal((3, n_paths))
        Z_S, Z_W1, Z_W2 = Z[0], Z[1], Z[2]

        vol = np.sqrt(np.maximum(xi_t_t, 0.0))
        S = S * np.exp((r - q - 0.5 * vol ** 2) * dt_actual + vol * sqrt_dt * Z_S)
        X1 = X1 * decay1 + std1 * Z_W1
        X2 = X2 * decay2 + std2 * Z_W2

        if step in required_steps:
            snapshots[step] = S.copy()

    return snapshots, step_of

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Method for Leverage Function sigma(t, S) — Bergomi Two-Factor
========================================================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Implements the particle method to calibrate the leverage function sigma(t, S)
in the Bergomi LSV dynamics:

    dS/S = (r-q) dt + sigma(S,t) sqrt(xi^t_t) dW^S

where xi^t_t is the spot variance from the Bergomi two-factor model:

    xi^T_t = xi^T_0 * f_T(t, x^T_t)
    x^T_t = alpha_theta [ (1-theta) e^{-kappa1(T-t)} X^1_t
                           + theta e^{-kappa2(T-t)} X^2_t ]
    f_T(t, x) = exp( omega * x - omega^2/2 * chi(t, T) )

The leverage function is determined by the Gyongy projection:

    sigma^2(K, t) * E[xi^t_t | S_t = K] = sigma_Dupire^2(K, t)

Inputs:
    lsv_bergomi/data/bergomi_params.json — model parameters
    lsv_bergomi/data/fwd_var_fit.json    — VS vol fit (z1, z2, z3)
    lsv_bergomi/arrays/fwd_var_curve.npy — initial forward variance curve
    dupire_vol/arrays/local_vol_surface.npy — Dupire local vol
    dupire_vol/data/market_params.json   — S, r, q
    iv_surface/arrays/ttm_grid.npy       — time grid
    iv_surface/arrays/log_m_grid.npy     — log-moneyness grid

Outputs:
    lsv_bergomi/arrays/leverage_surface.npy     — sigma(t, S) grid
    lsv_bergomi/arrays/leverage_spot_grid.npy   — spot grid
    lsv_bergomi/arrays/leverage_time_grid.npy   — time grid
    lsv_bergomi/data/particle_log.json          — diagnostics
"""

# ===== IMPORTS =====
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import interpolate

from particle_config import (
    N_PARTICLES, DT, BANDWIDTH_OVERRIDE, L_SQUARED_CLIP, SEED,
    N_SPOT_GRID, SPOT_GRID_RANGE,
    ROOT, IV_DIR, DUPIRE_DIR, BERGOMI_DIR, DATA_DIR, PLOT_DIR, ARRAY_DIR,
)
from bergomi_spot_variance import *
from leverage_estimation import *

warnings.filterwarnings("ignore")

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("particle_method_bergomi")


# =============================================================================
# Data loading
# =============================================================================

def load_inputs():
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    with open(DATA_DIR / "bergomi_params.json") as f:
        bergomi = json.load(f)

    with open(DATA_DIR / "fwd_var_fit.json") as f:
        fwd_var_fit = json.load(f)

    fwd_var = np.load(ARRAY_DIR / "fwd_var_curve.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    log_m_grid = np.load(IV_DIR / "arrays" / "log_m_grid.npy")

    local_vol = np.load(DUPIRE_DIR / "arrays" / "local_vol_surface.npy")
    dupire_interp = interpolate.RegularGridInterpolator(
        (log_m_grid, ttm_grid), local_vol,
        method="cubic", bounds_error=False, fill_value=None,
    )

    fwd_prices = np.load(IV_DIR / "arrays" / "forward_curve.npy")
    if fwd_prices.ndim == 1:
        fwd_curve = np.column_stack([ttm_grid, fwd_prices])
    else:
        fwd_curve = fwd_prices

    # Interpolator for initial forward variance curve
    fwd_var_interp = interpolate.interp1d(
        ttm_grid, fwd_var, kind="linear",
        bounds_error=False, fill_value=(fwd_var[0], fwd_var[-1]),
    )

    logger.info(f"Loaded inputs: S={S:.2f}, r={r:.4f}, q={q:.4f}")
    logger.info(f"Bergomi: nu={bergomi['nu']:.4f}, theta={bergomi['theta']:.4f}, "
                f"kappa1={bergomi['kappa1']:.4f}, kappa2={bergomi['kappa2']:.4f}")
    logger.info(f"Correlations: rho1={bergomi['rho1']:.4f}, rho2={bergomi['rho2']:.4f}, "
                f"rho12={bergomi['rho12']:.4f}")
    logger.info(f"Fwd var range: [{fwd_var.min():.6f}, {fwd_var.max():.6f}]")

    return {
        "S": S, "r": r, "q": q,
        "bergomi": bergomi,
        "fwd_var_fit": fwd_var_fit,
        "fwd_var": fwd_var,
        "fwd_var_interp": fwd_var_interp,
        "dupire_interp": dupire_interp,
        "local_vol": local_vol,
        "ttm_grid": ttm_grid,
        "log_m_grid": log_m_grid,
        "fwd_curve": fwd_curve,
    }


# =============================================================================
# Core particle simulation — Bergomi backbone
# =============================================================================

def run_particle_method(inputs, N=N_PARTICLES, dt=DT,
                        bandwidth_override=BANDWIDTH_OVERRIDE, seed=SEED):
    S0 = inputs["S"]
    r = inputs["r"]
    q = inputs["q"]
    bergomi = inputs["bergomi"]
    dupire_interp = inputs["dupire_interp"]
    ttm_grid = inputs["ttm_grid"]
    log_m_grid = inputs["log_m_grid"]
    fwd_curve = inputs["fwd_curve"]
    fwd_var_interp = inputs["fwd_var_interp"]

    nu = bergomi["nu"]
    theta_param = bergomi["theta"]
    kappa1 = bergomi["kappa1"]
    kappa2 = bergomi["kappa2"]
    rho12 = bergomi["rho12"]
    rho1 = bergomi["rho1"]
    rho2 = bergomi["rho2"]

    rng = np.random.default_rng(seed)

    # Precompute correlation structure for (W^S, W^1, W^2)
    # Correlation matrix:
    #   [[1,    rho1,  rho2 ],
    #    [rho1, 1,     rho12],
    #    [rho2, rho12, 1    ]]
    corr_input = np.array([
        [1.0,   rho1,  rho2],
        [rho1,  1.0,   rho12],
        [rho2,  rho12, 1.0],
    ])
    # Diagnostic: log determinant + minimum eigenvalue of the *input* matrix.
    # det >> 0 means well-conditioned PD; det near 0 means near-singular;
    # det < 0 means the input correlations form an invalid matrix.
    det_input = float(np.linalg.det(corr_input))
    eigvals_input, eigvecs = np.linalg.eigh(corr_input)
    logger.info(f"Corr matrix (input): det={det_input:+.6e}  "
                f"eigvals=[{eigvals_input[0]:+.4e}, {eigvals_input[1]:+.4e}, "
                f"{eigvals_input[2]:+.4e}]")
    logger.info(f"Corr matrix (input): rho1={corr_input[0,1]:+.6f}  "
                f"rho2={corr_input[0,2]:+.6f}  rho12={corr_input[1,2]:+.6f}")

    # Always regularise via eigenvalue floor at 1e-6 — keeps particle method
    # and lsv_validation.py using the SAME correlation matrix regardless of
    # floating-point details when the input is near-singular (e.g. chi
    # saturated at +/-1, which puts one eigenvalue at 0 in exact arithmetic).
    # Without this, the two stages can take different branches on the same
    # input and the leverage no longer projects to the same dynamics it is
    # later applied to.
    eigvals = np.maximum(eigvals_input, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)
    regularised = bool(np.any(eigvals != eigvals_input))
    L_chol = np.linalg.cholesky(corr)

    # Diagnostic: post-regularisation correlation values + det.
    # If the input was already PD this just confirms nothing changed.
    det_post = float(np.linalg.det(corr))
    eigvals_post = np.linalg.eigvalsh(corr)
    tag = "regularised" if regularised else "unchanged"
    logger.info(f"Corr matrix ({tag}): det={det_post:+.6e}  "
                f"eigvals=[{eigvals_post[0]:+.4e}, {eigvals_post[1]:+.4e}, "
                f"{eigvals_post[2]:+.4e}]")
    logger.info(f"Corr matrix ({tag}): rho1={corr[0,1]:+.6f}  "
                f"rho2={corr[0,2]:+.6f}  rho12={corr[1,2]:+.6f}")
    if regularised:
        delta_rho1  = corr[0,1] - corr_input[0,1]
        delta_rho2  = corr[0,2] - corr_input[0,2]
        delta_rho12 = corr[1,2] - corr_input[1,2]
        logger.info(f"Corr matrix delta: drho1={delta_rho1:+.4e}  "
                    f"drho2={delta_rho2:+.4e}  drho12={delta_rho12:+.4e}")

    # Time grid
    T_max = ttm_grid[-1]
    n_steps = int(np.ceil(T_max / dt))
    dt_actual = T_max / n_steps
    sqrt_dt = np.sqrt(dt_actual)
    time_schedule = np.linspace(0, T_max, n_steps + 1)

    logger.info(f"Particle method (Bergomi): N={N:,}, dt={dt_actual:.6f} ({n_steps} steps), "
                f"T_max={T_max:.4f}")

    # Spot grid for recording leverage surface
    spot_grid = np.linspace(S0 * SPOT_GRID_RANGE[0], S0 * SPOT_GRID_RANGE[1], N_SPOT_GRID)

    record_every = max(1, n_steps // 200)
    min_record_step = int(np.ceil(0.10 / dt_actual))
    first_record = max(record_every, min_record_step)
    record_steps = list(range(first_record, n_steps + 1, record_every))
    if not record_steps or record_steps[-1] != n_steps:
        record_steps.append(n_steps)
    record_times = [time_schedule[k] for k in record_steps]

    leverage_records = []

    # Initialise particles
    S_particles = np.full(N, S0, dtype=np.float64)
    X1 = np.zeros(N, dtype=np.float64)
    X2 = np.zeros(N, dtype=np.float64)

    # OU exact simulation coefficients
    decay1 = np.exp(-kappa1 * dt_actual)
    decay2 = np.exp(-kappa2 * dt_actual)
    std1 = np.sqrt((1.0 - np.exp(-2.0 * kappa1 * dt_actual)) / (2.0 * kappa1)) if kappa1 > 1e-10 else sqrt_dt
    std2 = np.sqrt((1.0 - np.exp(-2.0 * kappa2 * dt_actual)) / (2.0 * kappa2)) if kappa2 > 1e-10 else sqrt_dt

    # Diagnostics
    clip_count = 0
    total_evaluations = 0
    bandwidth_history = []
    spot_var_history = []

    logger.info("Starting particle evolution...")

    for step in range(n_steps):
        t = time_schedule[step]

        # Compute spot variance xi^t_t for each particle
        xi_t_t = compute_spot_variance(
            X1, X2, t, bergomi, fwd_var_interp, ttm_grid
        )

        # Bandwidth selection
        if bandwidth_override is not None:
            h = bandwidth_override
        else:
            h = nw_cv_bandwidth(S_particles, xi_t_t)
        bandwidth_history.append(h)

        # Conditional expectation E[xi^t_t | S_t = S]
        E_xi_given_S = conditional_expectation_kernel(
            S_particles, xi_t_t, S_particles, h
        )
        E_xi_given_S = np.maximum(E_xi_given_S, 1e-8)

        # Dupire local vol at each particle
        sigma_dupire = query_dupire(
            dupire_interp, S_particles, t, S0, r, q, log_m_grid, ttm_grid, fwd_curve
        )

        # Leverage function: sigma^2(S,t) = sigma_Dupire^2 / E[xi^t_t | S_t = S]
        L_sq = sigma_dupire**2 / E_xi_given_S
        n_clipped = np.sum((L_sq < L_SQUARED_CLIP[0]) | (L_sq > L_SQUARED_CLIP[1]))
        clip_count += n_clipped
        total_evaluations += N
        L_sq = np.clip(L_sq, L_SQUARED_CLIP[0], L_SQUARED_CLIP[1])
        L_particles = np.sqrt(L_sq)

        # Record leverage surface on spot grid
        if step in record_steps:
            E_xi_grid = conditional_expectation_kernel(
                S_particles, xi_t_t, spot_grid, h
            )
            E_xi_grid = np.maximum(E_xi_grid, 1e-8)
            sigma_dupire_grid = query_dupire(
                dupire_interp, spot_grid, t, S0, r, q, log_m_grid, ttm_grid, fwd_curve
            )
            L_sq_grid = sigma_dupire_grid**2 / E_xi_grid
            L_sq_grid = np.clip(L_sq_grid, L_SQUARED_CLIP[0], L_SQUARED_CLIP[1])
            L_grid = np.sqrt(L_sq_grid)
            leverage_records.append(L_grid)
            spot_var_history.append({
                "t": float(t),
                "xi_mean": float(xi_t_t.mean()),
                "xi_std": float(xi_t_t.std()),
                "xi_median": float(np.median(xi_t_t)),
            })

        # Generate 3 correlated normals: (Z_S, Z_1, Z_2) via Cholesky
        Z_indep = rng.standard_normal((3, N))
        Z_corr = L_chol @ Z_indep  # (3, N)
        Z_S = Z_corr[0]
        Z_W1 = Z_corr[1]
        Z_W2 = Z_corr[2]

        # Evolve spot (log-Euler)
        vol = L_particles * np.sqrt(np.maximum(xi_t_t, 0.0))
        S_particles = S_particles * np.exp(
            (r - q - 0.5 * vol**2) * dt_actual + vol * sqrt_dt * Z_S
        )

        # Evolve OU processes (exact simulation)
        X1 = X1 * decay1 + std1 * Z_W1
        X2 = X2 * decay2 + std2 * Z_W2

        # Progress
        if (step + 1) % max(1, n_steps // 10) == 0:
            pct = 100 * (step + 1) / n_steps
            logger.info(
                f"  Step {step+1:>5}/{n_steps} ({pct:5.1f}%) | "
                f"S: [{S_particles.min():.0f}, {S_particles.max():.0f}] | "
                f"xi: [{xi_t_t.min():.6f}, {xi_t_t.max():.6f}] | "
                f"h={h:.1f}"
            )

    # Assemble leverage surface
    leverage_surface = np.array(leverage_records).T  # (n_S, n_T)
    leverage_time_grid = np.array(record_times[:len(leverage_records)])

    clip_pct = 100.0 * clip_count / max(total_evaluations, 1)
    h_mean = np.mean(bandwidth_history)
    h_std = np.std(bandwidth_history)

    log_data = {
        "N_particles": N,
        "dt": float(dt_actual),
        "n_steps": n_steps,
        "T_max": float(T_max),
        "bandwidth_mean": float(h_mean),
        "bandwidth_std": float(h_std),
        "bandwidth_override": bandwidth_override,
        "L_squared_clip_range": list(L_SQUARED_CLIP),
        "clip_count": int(clip_count),
        "clip_pct": float(clip_pct),
        "total_evaluations": int(total_evaluations),
        "leverage_surface_shape": list(leverage_surface.shape),
        "spot_grid_range": [float(spot_grid[0]), float(spot_grid[-1])],
        "time_grid_range": [float(leverage_time_grid[0]), float(leverage_time_grid[-1])],
        "final_S_mean": float(S_particles.mean()),
        "final_S_std": float(S_particles.std()),
        "spot_var_snapshots": spot_var_history,
    }

    logger.info("=" * 60)
    logger.info("Particle Method (Bergomi) Complete")
    logger.info(f"  Leverage surface shape: {leverage_surface.shape}")
    logger.info(f"  Bandwidth (mean +/- std): {h_mean:.2f} +/- {h_std:.2f}")
    logger.info(f"  Clipping events: {clip_count:,} / {total_evaluations:,} ({clip_pct:.2f}%)")
    logger.info(f"  Final S: mean={S_particles.mean():.2f}, std={S_particles.std():.2f}")
    logger.info("=" * 60)

    if clip_pct > 5.0:
        logger.warning(f"  High clipping rate ({clip_pct:.1f}%).")

    return {
        "leverage_surface": leverage_surface,
        "spot_grid": spot_grid,
        "time_grid": leverage_time_grid,
        "log": log_data,
        "S_particles_final": S_particles,
        "X1_final": X1,
        "X2_final": X2,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_leverage_surface(leverage_surface, spot_grid, time_grid, S0):
    T_mesh, S_mesh = np.meshgrid(time_grid, spot_grid)
    log_moneyness = np.log(S_mesh / S0)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        log_moneyness, T_mesh, leverage_surface,
        cmap=cm.viridis, alpha=0.9, linewidth=0, antialiased=True,
    )
    ax.set_xlabel("Log-Moneyness ln(S/S0)", fontsize=11)
    ax.set_ylabel("Time (years)", fontsize=11)
    ax.set_zlabel("sigma(t, S)", fontsize=11)
    ax.set_title("Leverage Function sigma(t, S) -- Bergomi LSV", fontsize=13, fontweight="bold")
    ax.view_init(elev=25, azim=-50)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    out_path = PLOT_DIR / "leverage_surface_3d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved leverage surface plot -> {out_path}")

    # Slices
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_slices = min(8, len(time_grid))
    slice_indices = np.linspace(0, len(time_grid) - 1, n_slices, dtype=int)
    colors = cm.viridis(np.linspace(0.1, 0.9, n_slices))
    log_m_axis = np.log(spot_grid / S0)
    for i, idx in enumerate(slice_indices):
        ax.plot(log_m_axis, leverage_surface[:, idx],
                color=colors[i], lw=1.5,
                label=f"t = {time_grid[idx]:.3f}y")
    ax.set_xlabel("Log-Moneyness ln(S/S0)")
    ax.set_ylabel("sigma(t, S)")
    ax.set_title("Leverage Function -- Time Slices (Bergomi)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = PLOT_DIR / "leverage_slices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved leverage slices plot -> {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def run(N=N_PARTICLES, dt=DT, bandwidth_override=BANDWIDTH_OVERRIDE, seed=SEED):
    logger.info("=" * 60)
    logger.info("STEP 3b (Bergomi): Particle Method for Leverage Function")
    logger.info("=" * 60)

    inputs = load_inputs()
    results = run_particle_method(inputs, N=N, dt=dt,
                                  bandwidth_override=bandwidth_override, seed=seed)

    np.save(ARRAY_DIR / "leverage_surface.npy", results["leverage_surface"])
    np.save(ARRAY_DIR / "leverage_spot_grid.npy", results["spot_grid"])
    np.save(ARRAY_DIR / "leverage_time_grid.npy", results["time_grid"])
    logger.info(f"Saved leverage arrays -> {ARRAY_DIR}/")

    log_path = DATA_DIR / "particle_log.json"
    with open(log_path, "w") as f:
        json.dump(results["log"], f, indent=2)
    logger.info(f"Saved particle log -> {log_path}")

    plot_leverage_surface(
        results["leverage_surface"], results["spot_grid"],
        results["time_grid"], inputs["S"],
    )

    return results


if __name__ == "__main__":
    run()

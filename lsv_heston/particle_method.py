#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle method for leverage L(t, S) — Step 3b (Master's Thesis, Imperial).

Guyon & Henry-Labordère (2012) particle calibration of L(t, S) in the LSV
dynamics dS=(r-q)S dt + L(t,S) sqrt(V) S dW^S, dV=kappa(theta-V)dt + xi sqrt(V)dW^V,
d<W^S,W^V>=rho dt. Gyöngy projection: L(t,S)^2 = sigma_Dupire(t,S)^2 / E[V_t|S_t=S],
the conditional expectation via Gaussian kernel smoothing over the particle ensemble.

Inputs:
    lsv_heston/data/heston_params.json      — calibrated Heston params
    dupire_vol/arrays/local_vol_surface.npy — Dupire local vol surface
    dupire_vol/data/market_params.json      — S, r, q
    iv_surface/arrays/ttm_grid.npy          — time grid
    iv_surface/arrays/log_m_grid.npy        — log-moneyness grid

Outputs (to lsv_heston/):
    arrays/leverage_{surface,spot_grid,time_grid}.npy — L(t,S) on (n_S, n_T) and grids
    data/particle_log.json          — summary log (N, bandwidth, clipping, ...)
    plots/leverage_surface_3d.png   — 3D surface of L(t, S)
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import interpolate
from scipy.stats import norm

from particle_config import *
from qe_scheme import *
from leverage_estimation import *

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("particle_method")


# --- Data loading ---

def load_inputs():
    """Load particle-method inputs; returns dict (S, r, q, heston, dupire_interp,
    local_vol, ttm_grid, log_m_grid, fwd_curve)."""
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    with open(DATA_DIR / "heston_params.json") as f:
        heston = json.load(f)

    local_vol = np.load(DUPIRE_DIR / "arrays" / "local_vol_surface.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    log_m_grid = np.load(IV_DIR / "arrays" / "log_m_grid.npy")

    # 2D Dupire interpolator (log-moneyness, TTM)
    dupire_interp = interpolate.RegularGridInterpolator(
        (log_m_grid, ttm_grid),
        local_vol,
        method="cubic",
        bounds_error=False,
        fill_value=None,   # clamp to nearest boundary
    )

    # Forward curve F(0,T) from put-call parity, for log(S/F(0,t)) rather than
    # flat approx log(S/S0) - (r-q)t
    fwd_prices = np.load(IV_DIR / "arrays" / "forward_curve.npy")
    if fwd_prices.ndim == 1:
        fwd_curve = np.column_stack([ttm_grid, fwd_prices])
    else:
        fwd_curve = fwd_prices   # already [[T, F]]

    logger.info(f"Loaded inputs: S={S:.2f}, r={r:.4f}, q={q:.4f}")
    logger.info(f"Heston: κ={heston['kappa']:.4f}, θ={heston['theta']:.4f}, "
                f"ξ={heston['xi']:.4f}, ρ={heston['rho']:.4f}, V₀={heston['V0']:.4f}")
    logger.info(f"Dupire surface: {local_vol.shape}, "
                f"LogM=[{log_m_grid[0]:.4f}, {log_m_grid[-1]:.4f}], "
                f"TTM=[{ttm_grid[0]:.4f}, {ttm_grid[-1]:.4f}]")
    logger.info(f"Forward curve: F range [{fwd_curve[:,1].min():.2f}, {fwd_curve[:,1].max():.2f}]")

    return {
        "S": S, "r": r, "q": q,
        "heston": heston,
        "dupire_interp": dupire_interp,
        "local_vol": local_vol,
        "ttm_grid": ttm_grid,
        "log_m_grid": log_m_grid,
        "fwd_curve": fwd_curve,
    }


# --- Core particle simulation ---

def run_particle_method(inputs, N=N_PARTICLES, dt=DT,
                         bandwidth_override=BANDWIDTH_OVERRIDE,
                         seed=SEED, variance_scheme=VARIANCE_SCHEME):
    """
    Simulate N particles under LSV with per-step Gyöngy projection
    L(t_k,S)^2 = sigma_Dupire(t_k,S)^2 / E[V_tk|S_tk=S] (kernel-smoothed).

    dt in years. bandwidth_override: if set, replaces NW CV every step.
    variance_scheme: "euler" or "qe". Returns dict with leverage_surface (n_S,n_T),
    spot_grid (n_S,), time_grid (n_T,), and a diagnostics log.
    """
    S0 = inputs["S"]
    r = inputs["r"]
    q = inputs["q"]
    heston = inputs["heston"]
    dupire_interp = inputs["dupire_interp"]
    ttm_grid = inputs["ttm_grid"]
    log_m_grid = inputs["log_m_grid"]
    fwd_curve = inputs["fwd_curve"]

    kappa = heston["kappa"]
    theta = heston["theta"]
    xi = heston["xi"]
    rho = heston["rho"]
    V0 = heston["V0"]

    rng = np.random.default_rng(seed)

    # Time grid 0..T_max (max TTM on Dupire surface)
    T_max = ttm_grid[-1]
    n_steps = int(np.ceil(T_max / dt))
    dt_actual = T_max / n_steps
    sqrt_dt = np.sqrt(dt_actual)
    time_schedule = np.linspace(0, T_max, n_steps + 1)

    if variance_scheme not in ("euler", "qe"):
        raise ValueError(f"Unknown variance_scheme: {variance_scheme!r}")
    logger.info(f"Particle method: N={N:,}, dt={dt_actual:.6f} ({n_steps} steps), "
                f"T_max={T_max:.4f}, scheme={variance_scheme}")

    # Spot grid for recording leverage surface
    spot_grid = np.linspace(S0 * SPOT_GRID_RANGE[0], S0 * SPOT_GRID_RANGE[1],
                             N_SPOT_GRID)

    # Record L(t,S) on spot grid at ~100-200 steps. Start at TTM>=0.10: earlier
    # the cloud is too concentrated near S0 for reliable kernel estimates.
    record_every = max(1, n_steps // 200)
    min_record_step = int(np.ceil(0.10 / dt_actual))
    first_record = max(record_every, min_record_step)
    record_steps = list(range(first_record, n_steps + 1, record_every))
    if not record_steps or record_steps[-1] != n_steps:
        record_steps.append(n_steps)
    record_times = [time_schedule[k] for k in record_steps]

    leverage_records = []

    S_particles = np.full(N, S0, dtype=np.float64)
    V_particles = np.full(N, V0, dtype=np.float64)

    clip_count = 0
    total_evaluations = 0
    bandwidth_history = []

    logger.info("Starting particle evolution...")

    for step in range(n_steps):
        t = time_schedule[step]

        # Bandwidth (NW leave-one-out CV)
        if bandwidth_override is not None:
            h = bandwidth_override
        else:
            h = nw_cv_bandwidth(S_particles, V_particles)
        bandwidth_history.append(h)

        # E[V|S] per particle
        E_V_given_S_particles = conditional_expectation_kernel(
            S_particles, V_particles, S_particles, h
        )
        E_V_given_S_particles = np.maximum(E_V_given_S_particles, 1e-8)  # floor

        # Dupire local vol per particle
        sigma_dupire = query_dupire(
            dupire_interp, S_particles, t, S0, r, q, log_m_grid, ttm_grid, fwd_curve
        )

        # Leverage per particle
        L_sq = sigma_dupire**2 / E_V_given_S_particles

        n_clipped = np.sum((L_sq < L_SQUARED_CLIP[0]) | (L_sq > L_SQUARED_CLIP[1]))
        clip_count += n_clipped
        total_evaluations += N
        L_sq = np.clip(L_sq, L_SQUARED_CLIP[0], L_SQUARED_CLIP[1])
        L_particles = np.sqrt(L_sq)

        # Record leverage on spot grid
        if step in record_steps:
            E_V_grid = conditional_expectation_kernel(
                S_particles, V_particles, spot_grid, h
            )
            E_V_grid = np.maximum(E_V_grid, 1e-8)

            sigma_dupire_grid = query_dupire(
                dupire_interp, spot_grid, t, S0, r, q, log_m_grid, ttm_grid, fwd_curve
            )

            L_sq_grid = sigma_dupire_grid**2 / E_V_grid
            L_sq_grid = np.clip(L_sq_grid, L_SQUARED_CLIP[0], L_SQUARED_CLIP[1])
            L_grid = np.sqrt(L_sq_grid)
            leverage_records.append(L_grid)

        if variance_scheme == "euler":
            # Full-truncation Euler, correlated (W^S, W^V)
            Z1 = rng.standard_normal(N)
            Z_indep = rng.standard_normal(N)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z_indep

            # Log-Euler for S
            vol_particles = L_particles * np.sqrt(np.maximum(V_particles, 0.0))
            S_particles = S_particles * np.exp(
                (r - q - 0.5 * vol_particles**2) * dt_actual
                + vol_particles * sqrt_dt * Z1
            )

            # Full-truncation Euler for V
            V_pos = np.maximum(V_particles, 0.0)
            V_particles = (
                V_particles
                + kappa * (theta - V_pos) * dt_actual
                + xi * np.sqrt(V_pos) * sqrt_dt * Z2
            )
            V_particles = np.maximum(V_particles, 0.0)
        else:
            # Andersen QE for V, BK-decomposed log-spot
            Z_qe = rng.standard_normal(N)        # V step (Case A) and U=Phi(Z) (Case B)
            Z_perp = rng.standard_normal(N)      # independent spot leg

            V_new = step_variance_qe(V_particles, dt_actual,
                                     kappa, theta, xi, Z_qe)
            log_S = np.log(np.maximum(S_particles, 1e-12))
            log_S = step_spot_qe_bk(
                log_S, V_particles, V_new, L_particles, dt_actual,
                r, q, rho, kappa, theta, xi, Z_perp,
            )
            S_particles = np.exp(log_S)
            V_particles = V_new

        if (step + 1) % max(1, n_steps // 10) == 0:
            pct = 100 * (step + 1) / n_steps
            logger.info(
                f"  Step {step+1:>5}/{n_steps} ({pct:5.1f}%) | "
                f"S: [{S_particles.min():.0f}, {S_particles.max():.0f}] | "
                f"V: [{V_particles.min():.6f}, {V_particles.max():.6f}] | "
                f"h={h:.1f}"
            )

    # Assemble surface, transpose to (n_S, n_T) matching (spot_grid, time_grid)
    leverage_surface = np.array(leverage_records)   # (n_T_recorded, n_S)
    leverage_time_grid = np.array(record_times[:len(leverage_records)])
    leverage_surface = leverage_surface.T

    clip_pct = 100.0 * clip_count / max(total_evaluations, 1)
    h_mean = np.mean(bandwidth_history)
    h_std = np.std(bandwidth_history)

    log_data = {
        "N_particles": N,
        "dt": float(dt_actual),
        "n_steps": n_steps,
        "T_max": float(T_max),
        "variance_scheme": variance_scheme,
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
        "final_V_mean": float(V_particles.mean()),
        "final_V_std": float(V_particles.std()),
    }

    logger.info("=" * 60)
    logger.info("Particle Method Complete")
    logger.info(f"  Leverage surface shape: {leverage_surface.shape}")
    logger.info(f"  Bandwidth (mean ± std): {h_mean:.2f} ± {h_std:.2f}")
    logger.info(f"  Clipping events: {clip_count:,} / {total_evaluations:,} "
                f"({clip_pct:.2f}%)")
    logger.info(f"  Final S: mean={S_particles.mean():.2f}, std={S_particles.std():.2f}")
    logger.info(f"  Final V: mean={V_particles.mean():.6f}, std={V_particles.std():.6f}")
    logger.info("=" * 60)

    if clip_pct > 5.0:
        logger.warning(f"  High clipping rate ({clip_pct:.1f}%). Consider adjusting "
                        "bandwidth or clip bounds.")

    return {
        "leverage_surface": leverage_surface,
        "spot_grid": spot_grid,
        "time_grid": leverage_time_grid,
        "log": log_data,
        "S_particles_final": S_particles,
        "V_particles_final": V_particles,
    }


# --- Plotting ---

def plot_leverage_surface(leverage_surface, spot_grid, time_grid, S0):
    """3D surface (and time-slice) plot of L(t, S). leverage_surface: (n_S, n_T)."""
    T_mesh, S_mesh = np.meshgrid(time_grid, spot_grid)
    log_moneyness = np.log(S_mesh / S0)   # ln(S/S₀): L is a fn of spot, not forward moneyness

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        log_moneyness, T_mesh, leverage_surface,
        cmap=cm.viridis, alpha=0.9, linewidth=0, antialiased=True,
        rstride=1, cstride=1,
    )

    ax.set_xlabel("Log-Moneyness  ln(S/S₀)", fontsize=11)
    ax.set_ylabel("Time (years)", fontsize=11)
    ax.set_zlabel("L(t, S)", fontsize=11)
    ax.set_title("Leverage Function L(t, S) — Particle Method", fontsize=13,
                 fontweight="bold")
    ax.view_init(elev=25, azim=-50)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="L(t, S)")

    plt.tight_layout()
    out_path = PLOT_DIR / "leverage_surface_3d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved leverage surface plot → {out_path}")

    # Time slices
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_slices = min(8, len(time_grid))
    slice_indices = np.linspace(0, len(time_grid) - 1, n_slices, dtype=int)
    colors = cm.viridis(np.linspace(0.1, 0.9, n_slices))

    log_m_axis = np.log(spot_grid / S0)
    for i, idx in enumerate(slice_indices):
        ax.plot(log_m_axis, leverage_surface[:, idx],
                color=colors[i], lw=1.5,
                label=f"t = {time_grid[idx]:.3f}y")

    ax.set_xlabel("Log-Moneyness  ln(S/S₀)", fontsize=11)
    ax.set_ylabel("L(t, S)", fontsize=11)
    ax.set_title("Leverage Function — Time Slices", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / "leverage_slices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved leverage slices plot → {out_path}")


# --- Entry point ---

def run(N=N_PARTICLES, dt=DT, bandwidth_override=BANDWIDTH_OVERRIDE, seed=SEED,
        variance_scheme=VARIANCE_SCHEME):
    """Run the particle pipeline (load, simulate, save, plot); returns results."""
    logger.info("=" * 60)
    logger.info("STEP 3b: Particle Method for Leverage Function L(t, S)")
    logger.info("=" * 60)

    inputs = load_inputs()

    results = run_particle_method(inputs, N=N, dt=dt,
                                   bandwidth_override=bandwidth_override,
                                   seed=seed, variance_scheme=variance_scheme)

    np.save(ARRAY_DIR / "leverage_surface.npy", results["leverage_surface"])
    np.save(ARRAY_DIR / "leverage_spot_grid.npy", results["spot_grid"])
    np.save(ARRAY_DIR / "leverage_time_grid.npy", results["time_grid"])
    logger.info(f"Saved leverage arrays → {ARRAY_DIR}/")

    log_path = DATA_DIR / "particle_log.json"
    with open(log_path, "w") as f:
        json.dump(results["log"], f, indent=2)
    logger.info(f"Saved particle log → {log_path}")

    plot_leverage_surface(
        results["leverage_surface"],
        results["spot_grid"],
        results["time_grid"],
        inputs["S"],
    )

    return results


if __name__ == "__main__":
    run()

import logging

import numpy as np

from variance_processes import HestonVariance, BergomiVariance

logger = logging.getLogger(__name__)


def simulate_lsv(S0, r, q, rho, leverage_fn, var_process, reset_dates,
                 n_paths, dt_max=1.0 / 52.0, seed=42):
    """Simulate LSV dynamics forward, recording spot at reset dates.

    Log-Euler for S, var_process.step for V. The fine time grid hits every
    reset exactly (no spot interpolation) with intermediate steps so no step
    exceeds dt_max.

    var_process must expose initial_variance(n) and step(V, dt, Z).
    reset_dates: years from t=0, sorted with first entry > 0.
    Returns dict: S_resets (n_paths, n_resets), full S_all/V_all on the fine
    grid, time_grid, and reset_indices into it.
    """
    # Fine time grid hitting all reset dates exactly
    all_times = [0.0]
    for T in reset_dates:
        t_prev = all_times[-1]
        gap = T - t_prev
        if gap <= 0:
            continue
        n_sub = max(1, int(np.ceil(gap / dt_max)))
        sub_times = np.linspace(t_prev, T, n_sub + 1)[1:]  # exclude t_prev
        all_times.extend(sub_times.tolist())
    time_grid = np.array(all_times)

    reset_indices = np.searchsorted(time_grid, reset_dates)

    n_steps = len(time_grid) - 1
    rng = np.random.default_rng(seed)

    S = np.full(n_paths, S0)
    V = var_process.initial_variance(n_paths)

    # Full paths for diagnostics (subsampled by caller)
    S_all = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    V_all = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    S_all[:, 0] = S
    V_all[:, 0] = V

    sqrt_1_minus_rho2 = np.sqrt(1.0 - rho**2)

    # QE-mode Heston needs the BK joint update path
    use_qe = (
        isinstance(var_process, HestonVariance)
        and getattr(var_process, "scheme", "euler") == "qe"
    )
    if use_qe:
        logger.info("simulate_lsv (HestonVariance): variance_scheme='qe' "
                    "(Andersen 2008 + BK spot)")
    elif isinstance(var_process, BergomiVariance):
        logger.info("simulate_lsv (BergomiVariance): exact OU + lognormal "
                    "forward variance (omega = 2*nu)")

    for i in range(n_steps):
        dt = time_grid[i + 1] - time_grid[i]
        sqrt_dt = np.sqrt(dt)
        t = time_grid[i]

        # Leverage at start of step
        L = leverage_fn(S, t)

        if use_qe:
            Z_qe = rng.standard_normal(n_paths)
            Z_perp = rng.standard_normal(n_paths)
            V_new = var_process.qe_step(V, dt, Z_qe)
            log_S = np.log(np.maximum(S, 1e-12))
            log_S = log_S + var_process.qe_log_spot_increment(
                V, V_new, L, dt, r, q, rho, Z_perp,
            )
            S = np.exp(log_S)
            V = V_new
        else:
            # Bergomi or Heston-Euler
            Z1 = rng.standard_normal(n_paths)
            Z_indep = rng.standard_normal(n_paths)
            Z2 = rho * Z1 + sqrt_1_minus_rho2 * Z_indep

            vol = L * np.sqrt(np.maximum(V, 0.0))
            S = S * np.exp((r - q - 0.5 * vol**2) * dt + vol * sqrt_dt * Z1)

            # Bergomi: uses spot noise (Z1). Heston-Euler: uses Z2.
            Z_var = Z1 if getattr(var_process, 'uses_spot_noise', False) else Z2
            V = var_process.step(V, dt, Z_var)

        S_all[:, i + 1] = S
        V_all[:, i + 1] = V

    S_resets = S_all[:, reset_indices].astype(np.float64)

    return {
        "S_resets": S_resets,
        "S_all": S_all,
        "V_all": V_all,
        "time_grid": time_grid,
        "reset_indices": reset_indices,
    }


def compute_returns(S_resets, S0):
    """Per-period returns r_i = S_{T_i}/S_{T_{i-1}} - 1 (S_{T_0} = S0).
    S_resets: (n_paths, n_resets) -> same shape."""
    prev = np.column_stack([np.full(S_resets.shape[0], S0), S_resets[:, :-1]])
    return S_resets / prev - 1.0

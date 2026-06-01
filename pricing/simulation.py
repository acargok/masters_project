import logging

import numpy as np

from variance_processes import HestonVariance, BergomiVariance

logger = logging.getLogger(__name__)


def simulate_lsv(S0, r, q, rho, leverage_fn, var_process, reset_dates,
                 n_paths, dt_max=1.0 / 52.0, seed=42):
    """
    Simulate the LSV dynamics forward and record spot at reset dates.

    Uses a log-Euler scheme for S and the variance process's step method for V.
    The time grid is built so that every reset date is hit exactly (no
    interpolation of spot at resets). Between resets, intermediate steps are
    inserted so that no step exceeds dt_max.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    r, q : float
        Risk-free rate and dividend yield.
    rho : float
        Spot-vol correlation.
    leverage_fn : callable
        L(S_arr, t) -> leverage values.
    var_process : object
        Must expose initial_variance(n) and step(V, dt, Z) methods.
    reset_dates : np.ndarray, shape (n_resets,)
        Reset times in years from t=0 (must be sorted, first entry > 0).
    n_paths : int
        Number of MC paths.
    dt_max : float
        Maximum time step size (default: weekly = 1/52).
    seed : int
        Random seed.

    Returns
    -------
    dict
        S_resets : np.ndarray, shape (n_paths, n_resets)
            Spot at each reset date.
        S_all : np.ndarray, shape (n_paths, n_time_steps + 1)
            Full spot paths on the fine time grid.
        V_all : np.ndarray, shape (n_paths, n_time_steps + 1)
            Full variance paths on the fine time grid.
        time_grid : np.ndarray
            Fine time grid including t=0 and all reset dates.
        reset_indices : np.ndarray
            Indices into time_grid where resets occur.
    """
    # Build fine time grid: ensure all reset dates are hit exactly
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

    # Find indices of reset dates in the fine grid
    reset_indices = np.searchsorted(time_grid, reset_dates)

    n_steps = len(time_grid) - 1
    rng = np.random.default_rng(seed)

    # Initialise
    S = np.full(n_paths, S0)
    V = var_process.initial_variance(n_paths)

    # Storage — full paths for diagnostics (will be subsampled by caller)
    S_all = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    V_all = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    S_all[:, 0] = S
    V_all[:, 0] = V

    sqrt_1_minus_rho2 = np.sqrt(1.0 - rho**2)

    # Detect QE-mode Heston: needs the BK joint update path
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

        # Leverage at start of step (uses S_t, t)
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
            # Bergomi or Heston-Euler branch
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

    # Extract spot at resets
    S_resets = S_all[:, reset_indices].astype(np.float64)

    return {
        "S_resets": S_resets,
        "S_all": S_all,
        "V_all": V_all,
        "time_grid": time_grid,
        "reset_indices": reset_indices,
    }


def compute_returns(S_resets, S0):
    """
    Compute per-period returns from reset spot prices.

    r_i = S_{T_i} / S_{T_{i-1}} - 1, with S_{T_0} = S0.

    Parameters
    ----------
    S_resets : np.ndarray, shape (n_paths, n_resets)
    S0 : float

    Returns
    -------
    np.ndarray, shape (n_paths, n_resets)
    """
    prev = np.column_stack([np.full(S_resets.shape[0], S0), S_resets[:, :-1]])
    return S_resets / prev - 1.0

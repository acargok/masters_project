import json
import logging

import numpy as np
from scipy import interpolate

from config import *

logger = logging.getLogger(__name__)


def load_pricing_inputs():
    """
    Load all artifacts needed for cliquet pricing.

    Returns
    -------
    dict
        S, r, q, date, heston params, leverage surface + grids, forward curve.
    """
    # Market parameters
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    # Heston parameters
    with open(LSV_DIR / "data" / "heston_params.json") as f:
        heston = json.load(f)

    # Leverage surface
    leverage = np.load(LSV_DIR / "arrays" / "leverage_surface.npy")
    spot_grid = np.load(LSV_DIR / "arrays" / "leverage_spot_grid.npy")
    time_grid = np.load(LSV_DIR / "arrays" / "leverage_time_grid.npy")

    # Forward curve and TTM grid (for discount factors)
    fwd_prices = np.load(IV_DIR / "arrays" / "forward_curve.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")

    # Optional Bergomi params + forward variance for the pure-Bergomi baseline
    # that sits alongside the pure-Heston one. Missing files are tolerated; the
    # baseline becomes NaN downstream.
    bergomi = None
    fwd_var = None
    try:
        with open(BERGOMI_DIR / "data" / "bergomi_params.json") as f:
            bergomi = json.load(f)
        fwd_var = np.load(BERGOMI_DIR / "arrays" / "fwd_var_curve.npy")
    except FileNotFoundError:
        logger.info("Bergomi params/fwd_var not available — pure-Bergomi "
                    "baseline will be skipped.")

    logger.info(f"Loaded pricing inputs: S={S:.2f}, r={r:.4f}, q={q:.4f}")
    logger.info(f"Heston: kappa={heston['kappa']:.4f}, theta={heston['theta']:.6f}, "
                f"xi={heston['xi']:.4f}, rho={heston['rho']:.4f}, V0={heston['V0']:.6f}")
    logger.info(f"Leverage surface: {leverage.shape}, "
                f"spot=[{spot_grid[0]:.0f}, {spot_grid[-1]:.0f}], "
                f"time=[{time_grid[0]:.4f}, {time_grid[-1]:.4f}]")

    return {
        "S": S, "r": r, "q": q, "date": mkt.get("date", "unknown"),
        "heston": heston,
        "leverage": leverage,
        "spot_grid": spot_grid,
        "time_grid": time_grid,
        "fwd_prices": fwd_prices,
        "ttm_grid": ttm_grid,
        # Optional Bergomi inputs for the pure-Bergomi baseline.
        "bergomi": bergomi,
        "fwd_var": fwd_var,
    }


def build_leverage_interpolator(leverage, spot_grid, time_grid):
    """
    Build a 2D interpolator for L(t, S).

    Bilinear interpolation on the (spot, time) grid, with values outside the
    grid clamped to the nearest boundary (no gradient extrapolation), which
    keeps leverage stable for spot far from S0.

    Parameters
    ----------
    leverage : np.ndarray, shape (n_S, n_T)
    spot_grid : np.ndarray, shape (n_S,)
    time_grid : np.ndarray, shape (n_T,)

    Returns
    -------
    callable
        L(S_arr, t) -> np.ndarray of leverage values.
    """
    interp = interpolate.RegularGridInterpolator(
        (spot_grid, time_grid),
        leverage,
        method="linear",
        bounds_error=False,
        fill_value=None,  # clamp to nearest boundary
    )

    def L(S_arr, t):
        t_clamped = np.clip(t, time_grid[0], time_grid[-1])
        S_clamped = np.clip(S_arr, spot_grid[0], spot_grid[-1])
        pts = np.column_stack([S_clamped, np.full(len(S_clamped), t_clamped)])
        vals = interp(pts)
        return np.maximum(vals, 1e-4)

    return L


def load_bergomi_pricing_inputs():
    """
    Load artifacts for cliquet pricing under Bergomi LSV.

    Returns dict with same structure as load_pricing_inputs() plus bergomi params
    and forward variance curve.
    """
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        mkt = json.load(f)
    S, r, q = mkt["S"], mkt["r"], mkt["q"]

    with open(BERGOMI_DIR / "data" / "bergomi_params.json") as f:
        bergomi = json.load(f)

    leverage = np.load(BERGOMI_DIR / "arrays" / "leverage_surface.npy")
    spot_grid = np.load(BERGOMI_DIR / "arrays" / "leverage_spot_grid.npy")
    time_grid = np.load(BERGOMI_DIR / "arrays" / "leverage_time_grid.npy")

    fwd_var = np.load(BERGOMI_DIR / "arrays" / "fwd_var_curve.npy")
    ttm_grid = np.load(IV_DIR / "arrays" / "ttm_grid.npy")
    fwd_prices = np.load(IV_DIR / "arrays" / "forward_curve.npy")

    # Also load Heston for baselines
    with open(LSV_DIR / "data" / "heston_params.json") as f:
        heston = json.load(f)

    logger.info(f"Loaded Bergomi pricing inputs: S={S:.2f}, r={r:.4f}, q={q:.4f}")
    logger.info(f"Bergomi: nu={bergomi['nu']}, theta={bergomi['theta']}, "
                f"kappa1={bergomi['kappa1']}, kappa2={bergomi['kappa2']}")
    logger.info(f"Leverage surface: {leverage.shape}")

    return {
        "S": S, "r": r, "q": q, "date": mkt.get("date", "unknown"),
        "bergomi": bergomi,
        "heston": heston,
        "leverage": leverage,
        "spot_grid": spot_grid,
        "time_grid": time_grid,
        "fwd_var": fwd_var,
        "ttm_grid": ttm_grid,
        "fwd_prices": fwd_prices,
    }

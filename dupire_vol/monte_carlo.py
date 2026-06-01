# -*- coding: utf-8 -*-
"""Monte Carlo repricing validation (SECTION 5)."""

import logging

import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from scipy.stats import norm

from config import *

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 5: MONTE CARLO REPRICING VALIDATION
# ===========================================================================

def build_local_vol_interpolator(
        local_vol: np.ndarray,
        log_m_grid: np.ndarray,
        ttm_grid: np.ndarray):
    """
    Build a 2D interpolator for σ_loc(k, t).

    During MC simulation, we evaluate this at (log(S_t / F(0,t)), t),
    where F(0,t) is the initial forward curve interpolated at time t.
    Both arguments are clipped to the grid bounds (nearest-value
    extrapolation for paths wandering outside the surface).

    Parameters
    ----------
    local_vol : np.ndarray, shape (n_k, n_T)
        Dupire local vol surface (clipped, finite).
    log_m_grid : np.ndarray
        k = log(K/F) grid.
    ttm_grid : np.ndarray
        TTM grid.

    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
    """
    return interpolate.RegularGridInterpolator(
        (log_m_grid, ttm_grid),
        local_vol,
        method="linear",
        bounds_error=False,
        fill_value=None   # nearest-boundary extrapolation
    )


def _bs_price_forward(
        sigma: float, k: float, F: float,
        r: float, T: float, opt_type: str) -> float:
    """
    Black-76-style forward-log-moneyness BS price.

    Mirrors `_bsm_iv_from_price`'s convention so a price computed here and
    then inverted there round-trips to the same sigma:

        K = F·exp(k)
        d1 = (−k + σ²T/2)/(σ√T),  d2 = d1 − σ√T
        Call: e^{−rT}·[F·N(d1) − K·N(d2)]
        Put : e^{−rT}·[K·N(−d2) − F·N(−d1)]

    Used to build the SSVI BS price from the SSVI IV (the `iv` column in
    spx_iv_data.csv); that price is the comparison target for Dupire MC.
    """
    if T <= 0 or sigma <= 0:
        K = F * np.exp(k)
        intrinsic_call = max(np.exp(-r * T) * (F - K), 0.0)
        intrinsic_put  = max(np.exp(-r * T) * (K - F), 0.0)
        return intrinsic_call if opt_type == "call" else intrinsic_put
    K  = F * np.exp(k)
    df = np.exp(-r * T)
    w  = sigma ** 2 * T
    sqw = max(np.sqrt(w), 1e-12)
    d1 = (-k + w / 2.0) / sqw
    d2 = d1 - sqw
    if opt_type == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _bsm_iv_from_price(
        price: float, k: float, F: float,
        r: float, T: float, opt_type: str) -> float:
    """
    Back out Black-Scholes implied vol from an option price using Brent's method.

    Uses the forward log-moneyness form:
        d1 = (−k + σ²T/2) / (σ√T),   d2 = d1 − σ√T
        C  = e^{−rT}·[F·N(d1) − K·N(d2)],   P via put-call parity.

    Parameters
    ----------
    price    : option price to invert
    k        : forward log-moneyness  log(K/F)
    F        : forward price F(0,T)
    r        : risk-free rate
    T        : time to maturity
    opt_type : "call" or "put"

    Returns
    -------
    sigma (annualised) or np.nan if inversion fails (e.g. price ≤ intrinsic).
    """
    K  = F * np.exp(k)
    df = np.exp(-r * T)
    intrinsic = (max(df * (F - K), 0.0) if opt_type == "call"
                 else max(df * (K - F), 0.0))
    if price <= intrinsic + 1e-10 * max(F, 1.0):
        return np.nan

    def residual(sigma):
        w   = sigma ** 2 * T
        sqw = max(np.sqrt(w), 1e-10)
        d1  = (-k + w / 2.0) / sqw
        d2  = d1 - sqw
        if opt_type == "call":
            return df * (F * norm.cdf(d1) - K * norm.cdf(d2)) - price
        else:
            return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)) - price

    try:
        return float(optimize.brentq(residual, 1e-6, 10.0, xtol=1e-9, maxiter=200))
    except Exception:
        return np.nan


def monte_carlo_reprice(
        df: pd.DataFrame,
        local_vol: np.ndarray,
        log_m_grid: np.ndarray,
        ttm_grid: np.ndarray,
        S: float,
        r: float,
        q,
        fwd_curve: np.ndarray,
        n_paths: int = MC_N_PATHS,
        steps_per_year: int = MC_STEPS_PER_YEAR,
        n_reprice: int = MC_N_REPRICE,
        seed: int = MC_SEED) -> pd.DataFrame:
    """
    Validate the Dupire local vol surface by Monte Carlo repricing.

    Simulates the risk-neutral SDE (exponential Euler-Maruyama):

        S_{t+dt} = S_t · exp[(r − q_eff(t) − ½σ_loc²) dt + σ_loc · √dt · Z]

    where Z ~ N(0,1), q_eff(t) is linearly interpolated from the per-TTM
    q_eff_grid (average effective yields from put-call parity), and

        σ_loc = σ_loc(log(S_t / F(0,t)), t)

    is interpolated from the local vol surface.

    Using r − q_eff(t) as the drift is a good approximation for SPX where
    q varies slowly across maturities, giving E[S_T] ≈ F(0,T).

    F(0,t) is obtained by linear interpolation of fwd_curve = [[T, F(T)]]
    from Step 1, which was built from put-call parity.

    Agreement between MC prices and SSVI BS prices confirms that the
    Dupire surface is consistent with the IV surface and the forward curve.
    The SSVI BS price is BS(σ_SSVI, …) where σ_SSVI is the per-option
    SSVI-fitted IV in `spx_iv_data.csv:iv`. The SSVI surface is the ground
    truth benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Option data with columns [strike, ttm, option_type, fwd_log_m, iv, mid].
    local_vol : np.ndarray
        Dupire local vol surface (k × T grid, filled, no NaN).
    log_m_grid : np.ndarray
        k = log(K/F) grid (same grid as local_vol axis 0).
    ttm_grid : np.ndarray
        TTM grid (same grid as local_vol axis 1).
    S : float
        Spot price at inception (t=0).
    r : float
        Risk-free rate.
    q : float or np.ndarray, shape (n_T,)
        Dividend yield — scalar or per-TTM array.
    fwd_curve : np.ndarray, shape (n_T, 2)
        Forward curve on the TTM grid: [[T_1, F_1], ..., [T_n, F_n]].
    n_paths, steps_per_year, n_reprice, seed : int
        MC parameters.

    Returns
    -------
    pd.DataFrame
        Repricing results with columns: strike, ttm, option_type, forward,
        fwd_log_m, ssvi_price, mc_price, mc_std_err, price_error,
        price_error_pct, iv_ssvi, iv_mc, iv_error_bps.
    """
    from scipy.interpolate import interp1d

    rng = np.random.default_rng(seed)

    # Build local vol interpolator
    lv_interp = build_local_vol_interpolator(local_vol, log_m_grid, ttm_grid)

    # Forward curve interpolator: F(0, t) for any simulation time t
    fwd_interp = interp1d(fwd_curve[:, 0], fwd_curve[:, 1],
                          kind="linear", fill_value="extrapolate")

    # Per-TTM q interpolator (if array provided).
    # q_eff_grid[i] is the average effective yield over [0, T_i] from put-call
    # parity. Using it as an interpolated instantaneous rate is an approximation
    # that is accurate when q varies slowly (true for SPX), giving E[S_T] ≈ F(0,T).
    if isinstance(q, np.ndarray) and q.ndim == 1:
        q_interp_fn = interp1d(ttm_grid, q, kind="linear",
                               fill_value="extrapolate")
        q_is_array = True
    else:
        q_is_array = False

    # Select options within grid bounds using forward log-moneyness
    in_bounds = (
        (df["fwd_log_m"] >= log_m_grid[0]) &
        (df["fwd_log_m"] <= log_m_grid[-1]) &
        (df["ttm"] >= ttm_grid[0]) &
        (df["ttm"] <= ttm_grid[-1])
    )
    pool = df[in_bounds]
    if n_reprice > 0:
        sample = pool.sample(min(n_reprice, len(pool)), random_state=seed).copy()
    else:
        sample = pool.copy()

    if len(sample) == 0:
        logger.warning("No options in grid bounds for MC repricing.")
        return pd.DataFrame()

    # ── Simulate paths to T_max ──────────────────────────────────────
    T_max   = sample["ttm"].max()
    n_steps = max(int(T_max * steps_per_year), 20)
    dt      = T_max / n_steps
    sqrt_dt = np.sqrt(dt)

    t_schedule = np.arange(n_steps + 1) * dt   # (n_steps + 1,)

    sample["step_idx"] = sample["ttm"].apply(
        lambda T: int(np.argmin(np.abs(t_schedule - T)))
    )
    required_steps = set(sample["step_idx"].unique())

    logger.info(f"MC simulation: {n_paths:,} paths, {n_steps} steps, "
                f"T_max={T_max:.3f} years")
    logger.info(f"Repricing {len(sample)} options at "
                f"{len(required_steps)} distinct maturities")

    step_spots = {}
    S_t = np.full(n_paths, S, dtype=np.float64)

    for step in range(1, n_steps + 1):
        t = (step - 1) * dt   # time at START of this step
        if step % 20 == 0:
            logger.info(f"Simulating... ({step/n_steps:.2%} complete)")


        # Drift: r - q_eff(t).  q_eff_grid[i] is the average effective yield
        # over [0, T_i] from put-call parity; using it as an interpolated
        # instantaneous rate is a good approximation for SPX where q varies slowly.
        q_t = float(q_interp_fn(t)) if q_is_array else float(q)

        # Forward log-moneyness for the current paths:
        # k_t = log(S_t / F(0, t)) — consistent with the surface parameterisation
        F_0_t = float(fwd_interp(t))
        F_0_t = max(F_0_t, 1e-6)   # safety floor
        log_m_t = np.log(np.maximum(S_t, 1e-6) / F_0_t)
        log_m_clipped = np.clip(log_m_t, log_m_grid[0], log_m_grid[-1])
        # Allow t < ttm_grid[0]: the interpolator extrapolates linearly,
        # giving a smooth local vol for early steps rather than freezing at T_min.
        t_clipped = np.clip(t, 0.0, ttm_grid[-1])

        pts = np.column_stack([log_m_clipped, np.full(n_paths, t_clipped)])
        sigma_loc = lv_interp(pts)

        Z = rng.standard_normal(n_paths)
        S_t = S_t * np.exp(
            (r - q_t - 0.5 * sigma_loc ** 2) * dt + sigma_loc * sqrt_dt * Z
        )
        S_t = np.maximum(S_t, 1e-6)

        if step in required_steps:
            step_spots[step] = S_t.copy()

    logger.info("  Simulation complete. Computing payoffs...")

    # ── Reprice each option ──────────────────────────────────────────
    records = []
    for _, row in sample.iterrows():
        K        = row["strike"]
        T        = row["ttm"]
        opt_type = row["option_type"]
        fwd_k    = row["fwd_log_m"]
        step_idx = row["step_idx"]

        # Forward price for this option's expiry
        F_T = float(fwd_interp(T))

        # SSVI ground truth: per-option SSVI-fitted IV (spx_iv_data.csv:iv) →
        # BS price under the same forward-log-moneyness convention used to
        # invert MC prices.
        iv_ssvi = float(row["iv"])
        ssvi_price = _bs_price_forward(iv_ssvi, fwd_k, F_T, r, T, opt_type)

        S_T = step_spots[step_idx]
        payoff = (np.maximum(S_T - K, 0) if opt_type == "call"
                  else np.maximum(K - S_T, 0))

        mc_price   = np.exp(-r * T) * payoff.mean()
        mc_std_err = np.exp(-r * T) * payoff.std() / np.sqrt(n_paths)

        price_err     = mc_price - ssvi_price
        price_err_pct = (100.0 * price_err / ssvi_price
                         if abs(ssvi_price) > 0.01 else np.nan)

        iv_mc  = _bsm_iv_from_price(mc_price, fwd_k, F_T, r, T, opt_type)
        iv_err_bps = ((iv_mc - iv_ssvi) * 10_000
                      if not np.isnan(iv_mc) else np.nan)

        records.append({
            "strike":          K,
            "ttm":             round(T, 4),
            "option_type":     opt_type,
            "forward":         round(F_T, 2),
            "fwd_log_m":       round(fwd_k, 4),
            "ssvi_price":      round(ssvi_price,    4),
            "mc_price":        round(mc_price,      4),
            "mc_std_err":      round(mc_std_err,    4),
            "price_error":     round(price_err,     4),
            "price_error_pct": round(price_err_pct, 2) if np.isfinite(price_err_pct) else None,
            "iv_ssvi":         round(iv_ssvi,       6),
            "iv_mc":           round(iv_mc,         6) if not np.isnan(iv_mc) else None,
            "iv_error_bps":    round(iv_err_bps,    2) if not np.isnan(iv_err_bps) else None,
        })

    result_df = pd.DataFrame(records)

    # ── Report ──────────────────────────────────────────────────────
    # Liquidity filter uses the SSVI BS price as the size proxy with a $10
    # dollar threshold.
    liquid = result_df[result_df["ssvi_price"] >= 10.0]

    logger.info("\n" + "=" * 60)
    logger.info("MC REPRICING VALIDATION — CHECKPOINT 1 (vs SSVI)")
    logger.info(f"  {len(result_df)} options, {n_paths:,} paths")
    logger.info(f"  Dupire SDE: dS = (r-q)S dt + σ_loc(log(S/F(0,t)), t) S dW")
    logger.info(f"  Ground truth: SSVI BS price = BS(σ_SSVI, K, T, r, q_eff)")
    logger.info("")

    abs_errs = result_df["price_error"].abs()
    logger.info(f"  ALL OPTIONS ({len(result_df)}):")
    logger.info(f"    Price MAE  : ${abs_errs.mean():.2f}")
    logger.info(f"    Price RMSE : ${np.sqrt((result_df['price_error']**2).mean()):.2f}")
    logger.info(f"    Price Max  : ${abs_errs.max():.2f}")

    iv_valid = result_df.dropna(subset=["iv_mc"])
    if len(iv_valid) > 0:
        iv_errs = iv_valid["iv_mc"] - iv_valid["iv_ssvi"]
        logger.info(f"    IV ME      : {iv_errs.mean()*10000:+.1f} bp  "
                    f"(n={len(iv_valid)}/{len(result_df)} inverted)")
        logger.info(f"    IV MAE     : {iv_errs.abs().mean()*10000:.1f} bp")
        logger.info(f"    IV RMSE    : {np.sqrt((iv_errs**2).mean())*10000:.1f} bp")

    if len(liquid) > 0:
        liq_pct = liquid["price_error_pct"].dropna()
        logger.info("")
        logger.info(f"  LIQUID OPTIONS (ssvi_price >= $10, n={len(liquid)}):")
        logger.info(f"    Price MAE  : ${liquid['price_error'].abs().mean():.2f}")
        logger.info(f"    Price RMSE : ${np.sqrt((liquid['price_error']**2).mean()):.2f}")
        if len(liq_pct) > 0:
            logger.info(f"    %Err MAE   : {liq_pct.abs().mean():.2f}%")
            logger.info(f"    %Err RMSE  : {np.sqrt((liq_pct**2).mean()):.2f}%")
        liq_iv = liquid.dropna(subset=["iv_mc"])
        if len(liq_iv) > 0:
            liq_iv_errs = liq_iv["iv_mc"] - liq_iv["iv_ssvi"]
            logger.info(f"    IV ME      : {liq_iv_errs.mean()*10000:+.1f} bp")
            logger.info(f"    IV MAE     : {liq_iv_errs.abs().mean()*10000:.1f} bp")
            logger.info(f"    IV RMSE    : {np.sqrt((liq_iv_errs**2).mean())*10000:.1f} bp")

    logger.info("=" * 60 + "\n")
    return result_df

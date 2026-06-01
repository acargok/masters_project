import logging

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


# Section 3 — Black-Scholes and implied volatility

def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, option_type: str) -> float:
    """
    BSM price for a European option on a dividend-paying underlying.
    d1 = [ln(S/K) + (r−q+σ²/2)T]/(σ√T), d2 = d1−σ√T.
    Returns intrinsic value if T ≤ 0 or sigma ≤ 0.
    """
    if T <= 0 or sigma <= 0:
        return max(0.0,
            S * np.exp(-q * T) - K * np.exp(-r * T) if option_type == "call"
            else K * np.exp(-r * T) - S * np.exp(-q * T)
        )
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BSM vega: ∂Price/∂σ.  Same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 1e-12
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol_single(market_price: float, S: float, K: float, T: float,
                       r: float, q: float, option_type: str,
                       tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Implied vol via Newton-Raphson (σ₀=0.25, clipped to [1e-6, 5.0]) with
    Brent bisection on [1e-6, 5.0] as guaranteed-convergence fallback.
    Returns np.nan if price ≤ intrinsic or both methods fail.
    """
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    intrinsic = max(0.0, disc_S - disc_K if option_type == "call" else disc_K - disc_S)
    if market_price <= intrinsic + tol:
        return np.nan

    sigma = 0.25
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, q, sigma, option_type)
        vega = bs_vega(S, K, T, r, q, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            if 0.001 < sigma < 5.0:
                return sigma
            break
        if abs(vega) < 1e-10:
            break
        sigma -= diff / vega
        sigma = np.clip(sigma, 1e-6, 5.0)

    try:
        objective = lambda s: bs_price(S, K, T, r, q, s, option_type) - market_price
        if objective(1e-6) * objective(5.0) >= 0:
            return np.nan
        return float(optimize.brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=500))
    except (ValueError, RuntimeError):
        return np.nan


def compute_implied_vols(df: pd.DataFrame, S: float, r: float, q) -> pd.DataFrame:
    """
    Compute IVs for all options.  q can be scalar or dict (expiry → q_eff).
    Drops NaN and IVs outside [1%, 300%].
    """
    logger.info(f"Computing IVs for {len(df):,} options…")
    q_is_dict = isinstance(q, dict)
    ivs = [
        implied_vol_single(
            market_price=row["mid"], S=S, K=row["strike"], T=row["ttm"],
            r=r, q=q[row["expiry"]] if q_is_dict and row["expiry"] in q else
                  (q if not q_is_dict else 0.012),
            option_type=row["option_type"]
        )
        for _, row in df.iterrows()
    ]
    df = df.copy()
    df["iv"] = ivs
    n_before = len(df)
    df = df.dropna(subset=["iv"])
    df = df[(df["iv"] >= 0.01) & (df["iv"] <= 3.0)]
    logger.info(f"IV computation: {n_before:,} → {len(df):,} valid  "
                f"(IV range [{df['iv'].min():.3f}, {df['iv'].max():.3f}])")
    return df

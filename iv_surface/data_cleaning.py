import logging

import numpy as np
import pandas as pd

from config import *

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 2 — DATA CLEANING
# =============================================================================

def filter_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove illiquid options: OI, bid-ask spread, moneyness, mid > 0.

    The 20% bid-ask spread tolerance (MAX_BID_ASK_SPREAD_PCT=0.20) is
    permissive for SPX (typical spread ≈ 2–5% of mid for liquid strikes) while
    removing stale quotes. If the market is closed (all bids = 0), the bid/ask
    filters are skipped rather than dropping all data.
    """
    n0 = len(df)
    df = df[df["openInterest"] >= MIN_OPEN_INTEREST].copy()

    df_strict = df[df["bid"] > 0].copy()
    if len(df_strict) > 0:
        df_strict["spread_pct"] = (df_strict["ask"] - df_strict["bid"]) / df_strict["mid"]
        df_spread = df_strict[df_strict["spread_pct"] <= MAX_BID_ASK_SPREAD_PCT].copy()
        df_spread = df_spread.drop(columns=["spread_pct"])
        if len(df_spread) > 0:
            df = df_spread
        else:
            df = df_strict.drop(columns=["spread_pct"])
            logger.warning("Spread filter removed all options — skipping")
    else:
        logger.warning("No options with bid > 0 — skipping bid/ask filters")

    df = df[(df["moneyness"] >= MIN_MONEYNESS) & (df["moneyness"] <= MAX_MONEYNESS)].copy()
    df = df[df["mid"] > 0].copy()
    logger.info(f"Liquidity filter: {n0:,} → {len(df):,}")
    return df


def filter_no_arbitrage(df: pd.DataFrame, S: float, r: float, q: float) -> pd.DataFrame:
    """
    Remove options violating static no-arbitrage bounds (5% tolerance).

    Calls:  0.95·max(0, S·e^{-qT} − K·e^{-rT})  ≤  C  ≤  1.05·S·e^{-qT}
    Puts:   0.95·max(0, K·e^{-rT} − S·e^{-qT})  ≤  P  ≤  1.05·K·e^{-rT}
    """
    n0 = len(df)
    T, K, price = df["ttm"], df["strike"], df["mid"]
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    calls = df["option_type"] == "call"
    puts = df["option_type"] == "put"
    mask = pd.Series(True, index=df.index)
    mask[calls & (price < 0.95 * np.maximum(0, disc_S - disc_K))] = False
    mask[puts  & (price < 0.95 * np.maximum(0, disc_K - disc_S))] = False
    mask[calls & (price > 1.05 * disc_S)] = False
    mask[puts  & (price > 1.05 * disc_K)] = False
    df = df[mask].copy()
    logger.info(f"No-arbitrage filter: {n0:,} → {len(df):,}")
    return df


def compute_implied_forwards(df: pd.DataFrame, S: float, r: float,
                             q_fallback: float) -> pd.DataFrame:
    """
    Infer per-expiry forward prices from put-call parity.

    METHODOLOGY: For each expiry, pair calls and puts at the same strike
    within a near-ATM band (|K/S − 1| < NEAR_ATM_BAND).  Then:

        F = K + exp(r·T) · (C_mid − P_mid)

    Take the median F across strikes (robust to outlier pairs).  Compute
    the effective continuously-compounded dividend yield:

        q_eff(T) = r − ln(F/S) / T

    For expiries with insufficient pairs (< 2), fall back to F = S·exp((r−q)T)
    using the constant dividend yield.

    Parameters
    ----------
    df : pd.DataFrame
        MUST contain both calls and puts (run BEFORE OTM filtering).
    S, r : float
        Spot price and risk-free rate.
    q_fallback : float
        Constant dividend yield used as fallback.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, ttm, forward, q_eff, n_pairs.
        One row per expiry (including fallback rows).
    """
    records = []
    for expiry, grp in df.groupby("expiry"):
        ttm = grp["ttm"].iloc[0]
        calls = grp[grp["option_type"] == "call"].set_index("strike")["mid"]
        puts  = grp[grp["option_type"] == "put"].set_index("strike")["mid"]
        common = calls.index.intersection(puts.index)
        F_approx = S * np.exp((r - q_fallback) * ttm)
        near_atm = [K for K in common if abs(K / F_approx - 1.0) < NEAR_ATM_BAND]

        if len(near_atm) >= 2:
            disc = np.exp(r * ttm)
            forwards = [K + disc * (calls[K] - puts[K]) for K in near_atm]
            F = float(np.median(forwards))
            q_eff = r - np.log(F / S) / ttm if F > 0 and ttm > 0 else q_fallback
            n_pairs = len(near_atm)
        else:
            # Fallback: constant q
            F = S * np.exp((r - q_fallback) * ttm)
            q_eff = q_fallback
            n_pairs = 0

        records.append({
            "expiry": expiry, "ttm": ttm,
            "forward": F, "q_eff": q_eff, "n_pairs": n_pairs,
        })

    fwd_df = pd.DataFrame(records).sort_values("ttm").reset_index(drop=True)
    n_parity = (fwd_df["n_pairs"] > 0).sum()
    logger.info(f"Implied forwards: {n_parity}/{len(fwd_df)} from put-call parity, "
                f"rest from constant q={q_fallback:.4f}")
    if n_parity > 0:
        logger.info(f"  q_eff range: [{fwd_df['q_eff'].min():.4f}, {fwd_df['q_eff'].max():.4f}]")
    return fwd_df


def filter_option_type_forward(
        df: pd.DataFrame,
        fwd_df: pd.DataFrame,
        calls_only: bool = False) -> pd.DataFrame:
    """
    Keep only the options used to build the IV surface.

    Two modes controlled by ``calls_only``:

    calls_only=False (default) — OTM filter (forward-based)
        Keep OTM calls (K ≥ F) and OTM puts (K < F).  This is the standard
        market convention for index option surfaces: the forward accounts for
        dividends and financing, giving a cleaner ATM boundary than spot.

    calls_only=True — calls only (OTM and ITM)
        Keep every call regardless of moneyness.  Useful when put liquidity
        is poor or when you want to calibrate purely from call prices.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain (must have 'expiry', 'strike', 'option_type').
    fwd_df : pd.DataFrame
        Forward table from compute_implied_forwards() (has 'expiry', 'forward').
    calls_only : bool
        If True, keep all calls (OTM + ITM) and drop all puts.
        If False (default), keep OTM options only (forward-based).
    """
    n0 = len(df)

    if calls_only:
        df = df[df["option_type"] == "call"].copy()
        logger.info(f"Option type filter (calls only): {n0:,} → {len(df):,}")
        return df

    fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))
    df = df.copy()
    df["forward"] = df["expiry"].map(fwd_map)

    calls = df["option_type"] == "call"
    puts = df["option_type"] == "put"
    mask = (calls & (df["strike"] >= df["forward"])) | \
           (puts  & (df["strike"] <  df["forward"]))

    df = df[mask].drop(columns=["forward"]).copy()
    logger.info(f"OTM filter (forward-based): {n0:,} → {len(df):,}")
    return df

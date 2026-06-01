import logging

import numpy as np
import pandas as pd

from config import *

logger = logging.getLogger(__name__)


# Section 2 — Data cleaning

def filter_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove illiquid options by OI, bid-ask spread, moneyness, and mid > 0.
    Bid/ask filters are skipped if the market is closed (all bids = 0).
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
    Infer per-expiry forwards from put-call parity.

    Per expiry, pair near-ATM calls/puts (|K/F−1| < band):
    F = K + e^{rT}(C−P), take median F; q_eff(T) = r − ln(F/S)/T.
    < 2 pairs → fallback F = S·e^{(r−q)T} with constant q_fallback.

    df must contain both calls and puts (run BEFORE OTM filtering).
    Returns one row per expiry: expiry, ttm, forward, q_eff, n_pairs.
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
    Keep the options used to build the IV surface.

    calls_only=False (default): forward-based OTM filter — OTM calls (K≥F)
    and OTM puts (K<F); forward gives a cleaner ATM boundary than spot.
    calls_only=True: keep all calls (OTM+ITM), drop puts.

    fwd_df is the forward table from compute_implied_forwards().
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

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import *

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1 — MARKET DATA LOADING (OptionMetrics CSV)
# =============================================================================

def _resolve_snapshot_date(raw: pd.DataFrame) -> str:
    """Pick the snapshot date: SNAPSHOT_DATE if set, else the latest date in
    the file. Returns the date as an ISO 'YYYY-MM-DD' string."""
    if SNAPSHOT_DATE is not None:
        return SNAPSHOT_DATE
    dates = sorted(raw["date"].unique())
    if not dates:
        raise RuntimeError(f"No rows in {RAW_CSV_PATH}")
    chosen = dates[-1]
    return str(chosen)


def fetch_risk_free_rate() -> float:
    """Return RISK_FREE_RATE.

    Sourced from the OptionMetrics zero curve / linked rate file. Treated as a
    continuously-compounded annualised rate, consistent with every BSM formula
    in this file.
    """
    logger.info(f"Risk-free rate (hardcoded): {RISK_FREE_RATE:.4f}")
    return float(RISK_FREE_RATE)


def fetch_dividend_yield() -> float:
    """Return DIVIDEND_YIELD.

    SPX trailing-12-month dividend yield from the OptionMetrics database. Used
    as a fallback in `compute_implied_forwards()` when an expiry has too few
    near-ATM call-put pairs to imply a forward; per-expiry `q_eff` is preferred
    everywhere downstream.
    """
    logger.info(f"Dividend yield (hardcoded): {DIVIDEND_YIELD:.4f}")
    return float(DIVIDEND_YIELD)


def _infer_spot_from_parity(raw_filtered: pd.DataFrame, r: float, q: float) -> float:
    """Infer SPX spot from put-call parity at the shortest available expiry.

    Uses the same near-ATM-pair median estimator as
    `compute_implied_forwards()`, then converts F → S via S = F·exp(-(r-q)T).
    Falls back to the strike of the most-traded near-ATM contract if no
    pairs are available.
    """
    df = raw_filtered.copy()
    df = df[(df["mid"] > 0.0) & (df["best_bid"] > 0.0)]
    if df.empty:
        raise RuntimeError("No bid-positive options to infer spot from.")
    short_expiry = df.loc[df["ttm"].idxmin(), "expiry"]
    grp = df[df["expiry"] == short_expiry]
    ttm = float(grp["ttm"].iloc[0])
    calls = grp[grp["option_type"] == "call"].set_index("strike")["mid"]
    puts  = grp[grp["option_type"] == "put"].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if len(common) < 2:
        # Fallback: most-traded strike on that expiry, treated as ATM proxy.
        atm_K = float(grp.sort_values("openInterest", ascending=False)["strike"].iloc[0])
        S_guess = atm_K * np.exp(-(r - q) * ttm)
        logger.warning(f"Spot inference: <2 parity pairs at T={ttm:.3f}; "
                       f"using OI-weighted ATM proxy S≈{S_guess:.2f}")
        return float(S_guess)
    disc = np.exp(r * ttm)
    forwards = np.array([float(K) + disc * (calls[K] - puts[K]) for K in common])
    F = float(np.median(forwards))
    S = F * np.exp(-(r - q) * ttm)
    logger.info(f"Spot inferred from parity at T={ttm:.3f}: F={F:.2f}, "
                f"S={S:.2f}  ({len(common)} pairs)")
    return S


def fetch_spot_price() -> float:
    """Return SPX spot for the snapshot date.

    Uses OVERRIDE_SPOT_PRICE if set, otherwise infers from put-call parity
    on the same data the surface is built from. Inference requires the raw
    chain to have already been loaded; this function side-loads it via the
    module-level cache populated by `fetch_option_chain` on first call.
    """
    if OVERRIDE_SPOT_PRICE is not None:
        logger.info(f"SPX spot (override): {OVERRIDE_SPOT_PRICE:.2f}")
        return float(OVERRIDE_SPOT_PRICE)
    if _RAW_CHAIN_CACHE["df"] is None:
        # Force a load — but with S unknown we cannot apply the moneyness
        # filter yet. fetch_option_chain handles a None spot by deferring
        # the moneyness column to the caller.
        _ = fetch_option_chain(S=None)
    return _infer_spot_from_parity(
        _RAW_CHAIN_CACHE["df"],
        r=float(RISK_FREE_RATE),
        q=float(DIVIDEND_YIELD),
    )


# Module-level cache so that fetch_spot_price() and fetch_option_chain()
# can share the same parsed CSV without re-reading it twice.
_RAW_CHAIN_CACHE = {"df": None, "snapshot": None, "S": None}


def _load_optionmetrics_chain() -> pd.DataFrame:
    """Read the raw OptionMetrics CSV, filter to SNAPSHOT_DATE, convert to
    the column schema used by the rest of the pipeline.

    Output schema:
        strike, expiry, ttm, option_type, bid, ask, mid,
        openInterest, volume, iv_yf
    plus the OptionMetrics raw fields kept for diagnostics:
        best_bid, best_offer, impl_volatility, delta, gamma, vega, theta.
    """
    if _RAW_CHAIN_CACHE["df"] is not None:
        return _RAW_CHAIN_CACHE["df"]

    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"OptionMetrics raw CSV not found: {RAW_CSV_PATH}")

    logger.info(f"Loading OptionMetrics raw CSV: {RAW_CSV_PATH}")
    raw_all = pd.read_csv(RAW_CSV_PATH)

    snapshot = _resolve_snapshot_date(raw_all)
    raw = raw_all[raw_all["date"] == snapshot].copy()
    if raw.empty:
        raise RuntimeError(
            f"No rows for SNAPSHOT_DATE={snapshot!r} in {RAW_CSV_PATH}. "
            f"Available dates: {sorted(raw_all['date'].unique())}"
        )
    logger.info(f"Snapshot date: {snapshot}  ({len(raw):,} raw rows)")

    # ── Column conversions ──
    # OptionMetrics convention: strike_price stored as int = strike_in_dollars * 1000
    raw["strike"] = raw["strike_price"].astype(float) / 1000.0
    raw["option_type"] = raw["cp_flag"].map({"C": "call", "P": "put"})
    raw["expiry"] = raw["exdate"].astype(str)
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d").date()
    raw["ttm"] = raw["expiry"].apply(
        lambda s: (datetime.strptime(s, "%Y-%m-%d").date() - snap_dt).days / 365.0
    )
    raw["bid"] = raw["best_bid"].astype(float)
    raw["ask"] = raw["best_offer"].astype(float)
    raw["mid"] = (raw["bid"] + raw["ask"]) / 2.0
    raw["openInterest"] = raw["open_interest"].fillna(0).astype(float)
    raw["volume"] = raw.get("volume", pd.Series(0, index=raw.index)).fillna(0).astype(float)
    # Column `iv_yf` holds the vendor (OptionMetrics) implied volatility,
    # consumed by downstream comparison code.
    raw["iv_yf"] = pd.to_numeric(raw["impl_volatility"], errors="coerce")

    # Drop rows that obviously can't form a valid quote
    raw = raw.dropna(subset=["strike", "option_type", "expiry", "ttm"])
    raw = raw[raw["option_type"].isin(["call", "put"])]

    # TTM filter
    pre_ttm = len(raw)
    raw = raw[(raw["ttm"] >= MIN_TTM) & (raw["ttm"] <= MAX_TTM)].copy()
    logger.info(f"TTM filter [{MIN_TTM:.3f}, {MAX_TTM:.3f}]: "
                f"{pre_ttm:,} → {len(raw):,}")

    # Deduplicate (expiry, strike, option_type): OptionMetrics may carry
    # multiple option series per expiry/strike (e.g., AM- vs PM-settled, or
    # different `secid`/`optionid` revisions). Keep the row with the largest
    # open interest — the most actively quoted contract. The pipeline assumes
    # uniqueness on this key.
    pre_dedup = len(raw)
    raw = (raw.sort_values("openInterest", ascending=False)
              .drop_duplicates(subset=["expiry", "strike", "option_type"],
                               keep="first")
              .reset_index(drop=True))
    if pre_dedup != len(raw):
        logger.info(f"Deduplicate (expiry, strike, type): "
                    f"{pre_dedup:,} → {len(raw):,}")

    if raw.empty:
        raise RuntimeError("No options remain after TTM filter.")

    _RAW_CHAIN_CACHE["df"] = raw
    _RAW_CHAIN_CACHE["snapshot"] = snapshot
    return raw


def fetch_option_chain(S) -> pd.DataFrame:
    """Return the OptionMetrics option chain.

    `S` may be None on the very first call — used by `fetch_spot_price` when
    inferring spot via parity before the moneyness filter. Once `S` is
    available, the moneyness column is populated.
    """
    raw = _load_optionmetrics_chain()
    out = raw.copy()
    if S is not None:
        out["moneyness"] = out["strike"] / float(S)
    logger.info(
        f"OptionMetrics chain: {len(out):,} rows | "
        f"{out['expiry'].nunique()} expiries  "
        f"(TTM range [{out['ttm'].min():.3f}, {out['ttm'].max():.3f}])"
    )
    return out

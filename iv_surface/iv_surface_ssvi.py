#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPX Implied Volatility Surface — Step 1 (SSVI fit). LSV model for Asian
options; Master's Thesis, Imperial College London.

Builds the SSVI IV surface from the OptionMetrics extract at
iv_surface/spx_raw_data.csv (snapshot via SNAPSHOT_DATE; r, q are constants).

Method: per-expiry forwards F(T) from put-call parity; strike coordinate
k = ln(K/F(T)) (ATM at k=0); OTM cutoff on F(T); joint SSVI fit in total
variance w=σ²T; calendar-arb enforced; Dupire diagnostic (∂w/∂T, ∂²w/∂k²,
Gatheral g, σ²_loc = (∂w/∂T)/g).

Outputs: data/{spx_iv_data, validation_results, ssvi_params, implied_forwards,
market_params}; arrays/{ttm_grid, log_m_grid, iv_surface, total_var_surface,
q_eff_grid, forward_curve, dupire_g_surface, dupire_local_var};
plots/{iv_surface_3d, iv_smiles, ssvi_fit, validation, dupire_diagnostics}.
"""

# Imports
import json
import logging
import os
import warnings
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import interpolate, optimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module facade: re-export public names for tests and iv_explorer.
from config import *
from market_data import *
from market_data import _RAW_CHAIN_CACHE
from data_cleaning import *
from black_scholes import *
from ssvi import *
from dupire_diagnostics import *
from plotting import *
from validation import *


# Section 7 — Main pipeline

def main():
    """End-to-end SPX IV surface build.
    Returns (df_iv, ttm_grid, log_m_grid, iv_surface, val_df)."""
    logger.info("=" * 60)
    logger.info("  SPX IMPLIED VOLATILITY SURFACE — STEP 1 (v2: SVI)")
    logger.info("=" * 60)

    for d in [DIR_DATA, DIR_PLOTS, DIR_ARRAYS]:
        os.makedirs(d, exist_ok=True)

    # 1. Market data
    logger.info("\n[1/10]  Fetching market data…")
    r = fetch_risk_free_rate()
    q = fetch_dividend_yield()
    S = fetch_spot_price()

    # 2. Option chain
    logger.info("\n[2/10]  Fetching option chain…")
    raw = fetch_option_chain(S)

    # 3. Liquidity + no-arb
    logger.info("\n[3/10]  Cleaning data…")
    clean = filter_liquidity(raw)
    clean = filter_no_arbitrage(clean, S, r, q)

    # 4. Implied forwards — BEFORE OTM filtering (parity needs calls and puts).
    logger.info("\n[4/10]  Computing implied forwards…")
    fwd_df = compute_implied_forwards(clean, S, r, q_fallback=q)
    q_eff_map = dict(zip(fwd_df["expiry"], fwd_df["q_eff"]))

    # 5. Option type filter
    if USE_CALLS_ONLY:
        logger.info("\n[5/10]  Filtering to calls only (OTM + ITM)…")
    else:
        logger.info("\n[5/10]  Filtering to OTM (forward-based)…")
    clean = filter_option_type_forward(clean, fwd_df, calls_only=USE_CALLS_ONLY)

    if len(clean) < 50:
        raise RuntimeError(f"Only {len(clean)} options after filtering. Relax thresholds.")

    # 6. Implied vols
    logger.info("\n[6/10]  Computing implied volatilities…")
    df_iv = compute_implied_vols(clean, S, r, q_eff_map)

    # Drop thin slices (< MIN_OPTIONS_PER_SLICE): excluded from the fit anyway,
    # and would otherwise pollute plots and validation metrics.
    slice_counts = df_iv.groupby("expiry")["strike"].transform("count")
    n_before = len(df_iv)
    df_iv = df_iv[slice_counts >= MIN_OPTIONS_PER_SLICE].copy()
    # df_iv = df_iv[df_iv["expiry"] != "2026-01-30"]
    n_removed = n_before - len(df_iv)
    n_expiries_kept = df_iv["expiry"].nunique()
    if n_removed > 0:
        logger.info(
            f"  Thin-slice filter (< {MIN_OPTIONS_PER_SLICE} options): "
            f"{n_before:,} → {len(df_iv):,} options  ({n_removed} removed)  "
            f"{n_expiries_kept} expiries retained"
        )

    # Add forward log-moneyness k = ln(K/F) and total variance w = σ²·T
    _fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))
    df_iv["fwd_log_m"] = np.log(
        df_iv["strike"] / df_iv["expiry"].map(_fwd_map)
    )
    df_iv["total_var"] = df_iv["iv"] ** 2 * df_iv["ttm"]

    df_iv.to_csv(os.path.join(DIR_DATA, "spx_iv_data.csv"), index=False)
    logger.info(f"  → {DIR_DATA}/spx_iv_data.csv")

    # 7. Build SSVI surface (joint fit, no-butterfly penalty, monotone θ(T)).
    logger.info("\n[7/10]  Building SSVI IV surface…")
    ttm_grid, log_m_grid, iv_surface, total_var_surface, ssvi_params_df = \
        build_iv_surface(df_iv, fwd_df)

    # Save arrays
    np.save(os.path.join(DIR_ARRAYS, "ttm_grid.npy"), ttm_grid)
    np.save(os.path.join(DIR_ARRAYS, "log_m_grid.npy"), log_m_grid)
    np.save(os.path.join(DIR_ARRAYS, "iv_surface.npy"), iv_surface)
    np.save(os.path.join(DIR_ARRAYS, "total_var_surface.npy"), total_var_surface)

    # Forward curve on the TTM grid (for Dupire in forward space)
    from scipy.interpolate import interp1d
    fwd_interp = interp1d(fwd_df["ttm"].values, fwd_df["forward"].values,
                          kind="linear", fill_value="extrapolate")
    forward_curve = fwd_interp(ttm_grid).astype(np.float64)
    np.save(os.path.join(DIR_ARRAYS, "forward_curve.npy"), forward_curve)

    # q_eff on the TTM grid
    q_eff_interp = interp1d(fwd_df["ttm"].values, fwd_df["q_eff"].values,
                            kind="linear", fill_value="extrapolate")
    q_eff_grid = q_eff_interp(ttm_grid).astype(np.float64)
    np.save(os.path.join(DIR_ARRAYS, "q_eff_grid.npy"), q_eff_grid)

    logger.info(f"  Forward curve: [{forward_curve.min():.2f}, {forward_curve.max():.2f}]")
    logger.info(f"  q_eff grid: [{q_eff_grid.min():.4f}, {q_eff_grid.max():.4f}]")

    # Save SSVI params
    ssvi_params_df.to_csv(os.path.join(DIR_DATA, "ssvi_params.csv"), index=False)

    # 7b. Dupire compatibility check (before downstream Dupire/LSV use):
    # ∂w/∂T ≥ 0, Gatheral g admissibility, σ²_loc = (∂w/∂T)/g > 0 — not
    # tested by in-sample repricing, the usual LV/LSV failure mode.
    logger.info("\n[7b]  Dupire compatibility check…")
    dupire_stats = validate_dupire_compatibility(total_var_surface, ttm_grid, log_m_grid)
    np.save(os.path.join(DIR_ARRAYS, "dupire_g_surface.npy"),
            dupire_stats["g_surface"].astype(np.float64))
    np.save(os.path.join(DIR_ARRAYS, "dupire_local_var.npy"),
            np.where(np.isnan(dupire_stats["local_var_surface"]), 0.0,
                     dupire_stats["local_var_surface"]).astype(np.float64))

    # Save market params for downstream (Dupire, LSV).
    # `date` = OptionMetrics snapshot date (surface vintage), not today.
    snapshot = _RAW_CHAIN_CACHE.get("snapshot") or str(date.today())
    market_params = {
        "S": float(S), "r": float(r), "q": float(q),
        "date": snapshot,
        "source": "optionmetrics",
    }
    with open(os.path.join(DIR_DATA, "market_params.json"), "w") as f:
        json.dump(market_params, f, indent=2)

    # Save per-expiry implied forwards for downstream (`iv_explorer.py`,
    # `dupire_vol/dupire_local_vol.py` fallback path).
    fwd_df.to_csv(os.path.join(DIR_DATA, "implied_forwards.csv"), index=False)
    logger.info(f"  → {DIR_DATA}/implied_forwards.csv")

    logger.info(f"  → arrays + data saved")

    # 8. Plots
    logger.info("\n[8/10]  Plotting IV surface…")
    plot_iv_surface(ttm_grid, log_m_grid, iv_surface, S)

    logger.info("\n[9/10]  Plotting smiles + SSVI fits + Dupire diagnostics…")
    plot_iv_smiles(df_iv, fwd_df)
    plot_ssvi_fit(df_iv, fwd_df, ssvi_params_df)
    plot_dupire_diagnostics(dupire_stats, ttm_grid, log_m_grid)

    # 10. Validation
    logger.info("\n[10/10]  Validating surface…")
    val_df = validate_surface(df_iv, fwd_df, ttm_grid, log_m_grid, iv_surface, S, r)
    val_df.to_csv(os.path.join(DIR_DATA, "validation_results.csv"), index=False)
    plot_validation(val_df, min_price_for_pct=10.0)

    # Done
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 1 COMPLETE (v3: SSVI + forward log-moneyness)")
    logger.info("=" * 60)
    logger.info(f"  Key decisions:")
    logger.info(f"    - Forwards: put-call parity ({(fwd_df['n_pairs']>0).sum()}/{len(fwd_df)} expiries)")
    logger.info(f"    - Moneyness: forward log-moneyness k = ln(K/F)")
    logger.info(f"    - OTM cutoff: based on forward F(T), not spot S")
    logger.info(f"    - Surface fit: SSVI joint (η, γ, p₀, p₁, p₂ shared; θ(T) per-slice; ρ(T) from parametric form)")
    logger.info(f"    - No-butterfly: G&J (2014) Thm 4.2 enforced during fit")
    logger.info(f"    - Calendar arb: θ monotone by construction + grid sweep")
    logger.info(f"    - SSVI slices: {len(ssvi_params_df)} fitted")
    logger.info(f"    - Surface: {iv_surface.shape[0]}×{iv_surface.shape[1]} grid")
    logger.info("=" * 60)

    return df_iv, ttm_grid, log_m_grid, iv_surface, val_df


if __name__ == "__main__":
    main()

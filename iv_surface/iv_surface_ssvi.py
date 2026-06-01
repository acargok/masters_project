#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPX Implied Volatility Surface — Step 1 (SSVI fit)
==================================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Builds the SSVI implied-volatility surface from an OptionMetrics historical
extract at `iv_surface/spx_raw_data.csv`. The snapshot is selected via
`SNAPSHOT_DATE` (None → latest date in the file). The risk-free rate `r` and
dividend yield `q` are module-level constants for the snapshot date.

Methodology
-----------
- Implied forwards F(T) per expiry from put-call parity (model-free; avoids
  the constant-q assumption).
- Forward log-moneyness k = ln(K/F(T)) as the strike coordinate, so ATM is at
  k=0 and the surface is the correct input for Dupire in forward space.
- OTM/ITM cutoff on the forward F(T): calls K ≥ F(T), puts K < F(T).
- SSVI fit jointly across slices in total-variance space w(k) = σ²·T, with
  power-law φ(θ) and parameter smoothing across maturity.
- Calendar-spread arbitrage enforced: w(k, T₁) ≤ w(k, T₂) for T₁ < T₂.
- Dupire compatibility diagnostic: ∂w/∂T, ∂²w/∂k², the Gatheral density
  g(k,T), and local variance σ²_loc = (∂w/∂T)/g — the derivative-space checks
  that in-sample repricing does not cover.

Pipeline
--------
  1.  Load r, q, spot, and the OptionMetrics option chain.
  2.  Apply liquidity + no-arbitrage filters.
  3.  Infer per-expiry forwards from put-call parity (BEFORE OTM filtering).
  4.  Filter to OTM options on the forward price.
  5.  Compute implied vols via BSM inversion (Newton-Raphson + Brent).
  6.  Fit SSVI jointly in total-variance / forward-log-moneyness space.
  7.  Dupire compatibility check.
  8.  Enforce calendar-spread arbitrage; plot surface, smiles, diagnostics.
  9.  Validate by repricing a sample of market options with the surface IV.

Outputs
-------
  data/spx_iv_data.csv          — cleaned option data with computed IVs
  data/validation_results.csv   — sample repricing comparison
  data/ssvi_params.csv          — per-slice SSVI θ(T), φ(T), ρ(T) + shared params
  arrays/ttm_grid.npy           — 1D TTM grid
  arrays/log_m_grid.npy         — 1D forward-log-moneyness grid (k = ln(K/F))
  arrays/iv_surface.npy         — 2D interpolated IV surface σ(k, T)
  arrays/q_eff_grid.npy         — per-TTM effective dividend yield
  arrays/forward_curve.npy      — F(T) on the TTM grid
  arrays/total_var_surface.npy  — 2D total variance surface w(k, T)
  plots/iv_surface_3d.png       — 3D surface plot
  plots/iv_smiles.png           — smile overlay per expiry slice
  plots/ssvi_fit.png            — SSVI fit quality per slice
  plots/validation.png          — in-sample repricing diagnostics
  plots/dupire_diagnostics.png  — g(k,T) and σ²_loc heatmaps
  arrays/dupire_g_surface.npy   — Gatheral density g(k,T)
  arrays/dupire_local_var.npy   — Dupire local variance σ²_loc(k,T)
"""

# ===== IMPORTS =====
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

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===== MODULE FACADE =====
# Re-export every public name so that tests and `iv_explorer` can keep
# importing from `iv_surface_ssvi`.
from config import *
from market_data import *
from market_data import _RAW_CHAIN_CACHE
from data_cleaning import *
from black_scholes import *
from ssvi import *
from dupire_diagnostics import *
from plotting import *
from validation import *


# =============================================================================
# SECTION 7 — MAIN PIPELINE
# =============================================================================

def main():
    """
    End-to-end pipeline for constructing the SPX IV surface.

    Returns
    -------
    tuple
        (df_iv, ttm_grid, log_m_grid, iv_surface, val_df)
    """
    logger.info("=" * 60)
    logger.info("  SPX IMPLIED VOLATILITY SURFACE — STEP 1 (v2: SVI)")
    logger.info("=" * 60)

    for d in [DIR_DATA, DIR_PLOTS, DIR_ARRAYS]:
        os.makedirs(d, exist_ok=True)

    # ────────────────────────────────────────────────────── 1. Market data
    logger.info("\n[1/10]  Fetching market data…")
    r = fetch_risk_free_rate()
    q = fetch_dividend_yield()
    S = fetch_spot_price()

    # ────────────────────────────────────────────────────── 2. Option chain
    logger.info("\n[2/10]  Fetching option chain…")
    raw = fetch_option_chain(S)

    # ────────────────────────────────────────────────────── 3. Liquidity + no-arb
    logger.info("\n[3/10]  Cleaning data…")
    clean = filter_liquidity(raw)
    clean = filter_no_arbitrage(clean, S, r, q)

    # ────────────────────────────────────────────────────── 4. Implied forwards
    # Compute forwards BEFORE OTM filtering: put-call parity needs both calls
    # and puts at the same strike.
    logger.info("\n[4/10]  Computing implied forwards…")
    fwd_df = compute_implied_forwards(clean, S, r, q_fallback=q)
    q_eff_map = dict(zip(fwd_df["expiry"], fwd_df["q_eff"]))

    # ────────────────────────────────────────────────────── 5. Option type filter
    if USE_CALLS_ONLY:
        logger.info("\n[5/10]  Filtering to calls only (OTM + ITM)…")
    else:
        logger.info("\n[5/10]  Filtering to OTM (forward-based)…")
    clean = filter_option_type_forward(clean, fwd_df, calls_only=USE_CALLS_ONLY)

    if len(clean) < 50:
        raise RuntimeError(f"Only {len(clean)} options after filtering. Relax thresholds.")

    # ────────────────────────────────────────────────────── 6. Implied vols
    logger.info("\n[6/10]  Computing implied volatilities…")
    df_iv = compute_implied_vols(clean, S, r, q_eff_map)

    # Drop expiries with fewer than MIN_OPTIONS_PER_SLICE options.
    # These slices are excluded from SSVI fitting anyway; keeping them in df_iv
    # would let them appear in plots and contaminate validation metrics.
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

    # ────────────────────────────────────────────────────── 7. Build SSVI surface
    # Jointly fit SSVI across all slices with power-law φ(θ); no-butterfly
    # conditions (G&J 2014 Thm 4.2) enforced during fitting via a penalty.
    # θ(T) is monotone by parameterisation (calendar-spread free).
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

    # ─────────────────────────────────────────── 7b. Dupire compatibility check
    # Run before any downstream Dupire / LSV use. Checks ∂w/∂T ≥ 0, the Gatheral
    # g-function (butterfly admissibility), and positivity of σ²_loc =
    # (∂w/∂T)/g — quantities that in-sample repricing does NOT test and the most
    # common failure mode of smooth-looking surfaces in LV/LSV pipelines.
    logger.info("\n[7b]  Dupire compatibility check…")
    dupire_stats = validate_dupire_compatibility(total_var_surface, ttm_grid, log_m_grid)
    np.save(os.path.join(DIR_ARRAYS, "dupire_g_surface.npy"),
            dupire_stats["g_surface"].astype(np.float64))
    np.save(os.path.join(DIR_ARRAYS, "dupire_local_var.npy"),
            np.where(np.isnan(dupire_stats["local_var_surface"]), 0.0,
                     dupire_stats["local_var_surface"]).astype(np.float64))

    # Save market params (for downstream: Dupire, LSV).
    # The `date` field is the OptionMetrics snapshot date, NOT today's date —
    # downstream expects this to identify the surface vintage.
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

    # ────────────────────────────────────────────────────── 8. Plots
    logger.info("\n[8/10]  Plotting IV surface…")
    plot_iv_surface(ttm_grid, log_m_grid, iv_surface, S)

    logger.info("\n[9/10]  Plotting smiles + SSVI fits + Dupire diagnostics…")
    plot_iv_smiles(df_iv, fwd_df)
    plot_ssvi_fit(df_iv, fwd_df, ssvi_params_df)
    plot_dupire_diagnostics(dupire_stats, ttm_grid, log_m_grid)

    # ────────────────────────────────────────────────────── 10. Validation
    logger.info("\n[10/10]  Validating surface…")
    val_df = validate_surface(df_iv, fwd_df, ttm_grid, log_m_grid, iv_surface, S, r)
    val_df.to_csv(os.path.join(DIR_DATA, "validation_results.csv"), index=False)
    plot_validation(val_df, min_price_for_pct=10.0)

    # ────────────────────────────────────────────────────── Done
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

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from config import *
from black_scholes import bs_price

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 6 — VALIDATION
# =============================================================================

def validate_surface(df: pd.DataFrame,
                     fwd_df: pd.DataFrame,
                     ttm_grid: np.ndarray,
                     log_m_grid: np.ndarray,
                     iv_surface: np.ndarray,
                     S: float, r: float,
                     n_sample: int = N_VALIDATION,
                     min_price_for_pct: float = 10.0) -> pd.DataFrame:
    """
    Validate the surface by repricing a random sample of market options.

    Computes:
      - IV absolute error: |iv_interpolated − iv_computed|
      - Price error: (BSM_repriced − market_mid) / market_mid × 100%

    METHODOLOGY: We interpolate the surface at the option's forward log-
    moneyness k = ln(K/F(T)) using RegularGridInterpolator on (log_m_grid,
    ttm_grid).  Then reprice via BSM with the interpolated IV and the
    option's per-expiry q_eff.
    """
    fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))
    q_eff_map = dict(zip(fwd_df["expiry"], fwd_df["q_eff"]))

    interp_fn = interpolate.RegularGridInterpolator(
        (log_m_grid, ttm_grid), iv_surface,
        method="linear", bounds_error=False, fill_value=None
    )

    # Compute forward log-moneyness for all options
    df = df.copy()
    df["forward"] = df["expiry"].map(fwd_map)
    df["fwd_log_m"] = np.log(df["strike"] / df["forward"])

    # Restrict to options within grid bounds
    in_bounds = (
        (df["fwd_log_m"] >= log_m_grid.min()) &
        (df["fwd_log_m"] <= log_m_grid.max()) &
        (df["ttm"] >= ttm_grid.min()) &
        (df["ttm"] <= ttm_grid.max())
    )
    pool = df[in_bounds]
    if len(pool) < n_sample:
        logger.warning(f"Only {len(pool)} options in bounds (requested {n_sample})")

    sample = pool.sample(min(n_sample, len(pool)), random_state=42)

    records = []
    for _, row in sample.iterrows():
        k = row["fwd_log_m"]
        T = row["ttm"]
        q_eff = q_eff_map.get(row["expiry"], 0.012)

        iv_interp = float(interp_fn([[k, T]]))
        iv_abs_err = abs(iv_interp - row["iv"])
        repriced = bs_price(S, row["strike"], T, r, q_eff, iv_interp, row["option_type"])
        error_pct = (repriced - row["mid"]) / row["mid"] * 100.0

        records.append({
            "strike":          row["strike"],
            "expiry":          row["expiry"],
            "option_type":     row["option_type"],
            "market_price":    round(row["mid"], 4),
            "iv_computed":     round(row["iv"], 4),
            "iv_interpolated": round(iv_interp, 4),
            "iv_abs_err":      round(iv_abs_err, 4),
            "repriced":        round(repriced, 4),
            "error_pct":       round(error_pct, 4),
        })

    val_df = pd.DataFrame(records)

    # Report metrics
    iv_mae  = val_df["iv_abs_err"].mean()
    iv_rmse = np.sqrt((val_df["iv_abs_err"]**2).mean())
    liquid = val_df[val_df["market_price"] >= min_price_for_pct]
    p_mae = liquid["error_pct"].abs().mean() if len(liquid) > 0 else float("nan")

    logger.info("\n" + "=" * 60)
    logger.info(f"SURFACE VALIDATION  ({len(val_df)} sampled options)")
    logger.info(f"  IV MAE:  {iv_mae:.4f}  ({iv_mae*100:.2f} vol pts)")
    logger.info(f"  IV RMSE: {iv_rmse:.4f}  ({iv_rmse*100:.2f} vol pts)")
    logger.info(f"  Price MAE (mid≥${min_price_for_pct:.0f}, n={len(liquid)}): {p_mae:.2f}%")
    logger.info("=" * 60)

    return val_df


def plot_validation(val_df: pd.DataFrame, min_price_for_pct: float = 10.0) -> None:
    """Three-panel validation diagnostic: IV scatter, price scatter, error histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    calls_mask = val_df["option_type"] == "call"
    liquid = val_df[val_df["market_price"] >= min_price_for_pct]

    # Left: IV computed vs interpolated
    ax = axes[0]
    ax.scatter(val_df.loc[calls_mask, "iv_computed"],
               val_df.loc[calls_mask, "iv_interpolated"],
               alpha=0.7, s=35, color="steelblue", label="Call", zorder=3)
    ax.scatter(val_df.loc[~calls_mask, "iv_computed"],
               val_df.loc[~calls_mask, "iv_interpolated"],
               alpha=0.7, s=35, color="coral", label="Put", marker="^", zorder=3)
    iv_lims = [
        min(val_df["iv_computed"].min(), val_df["iv_interpolated"].min()) * 0.95,
        max(val_df["iv_computed"].max(), val_df["iv_interpolated"].max()) * 1.05
    ]
    ax.plot(iv_lims, iv_lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlim(iv_lims); ax.set_ylim(iv_lims)
    ax.set_xlabel("Computed IV"); ax.set_ylabel("Interpolated IV (SVI)")
    ax.set_title("IV: Computed vs Surface"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Middle: market vs repriced (liquid)
    ax = axes[1]
    liq_calls = liquid["option_type"] == "call"
    ax.scatter(liquid.loc[liq_calls, "market_price"], liquid.loc[liq_calls, "repriced"],
               alpha=0.7, s=35, color="steelblue", label="Call", zorder=3)
    ax.scatter(liquid.loc[~liq_calls, "market_price"], liquid.loc[~liq_calls, "repriced"],
               alpha=0.7, s=35, color="coral", label="Put", marker="^", zorder=3)
    if len(liquid) > 0:
        lims = [min(liquid["market_price"].min(), liquid["repriced"].min()) * 0.95,
                max(liquid["market_price"].max(), liquid["repriced"].max()) * 1.05]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Market Mid ($)"); ax.set_ylabel("BSM Repriced ($)")
    ax.set_title(f"Market vs Repriced (mid≥${min_price_for_pct:.0f})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: IV error histogram
    ax = axes[2]
    ax.hist(val_df["iv_abs_err"] * 100, bins=20, color="mediumpurple",
            edgecolor="white", alpha=0.85)
    ax.axvline(val_df["iv_abs_err"].mean() * 100, color="red",
               label=f"Mean = {val_df['iv_abs_err'].mean()*100:.2f} vol pts")
    ax.set_xlabel("IV Absolute Error (vol pts)"); ax.set_ylabel("Count")
    ax.set_title("IV Error Distribution"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle("SPX IV Surface Validation (SVI)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "validation.png"), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {DIR_PLOTS}/validation.png")

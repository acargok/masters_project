#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results_4_3_figures.py — thesis §4.3 (LSV Calibration) figure pack
====================================================================
Stand-alone script. Reads cached pipeline artefacts from
    lsv/arrays/, lsv/data/                       (Heston-LSV)
    lsv_bergomi/arrays/, lsv_bergomi/data/       (Bergomi-LSV)
    iv_surface/arrays/                           (forward curve / ttm grid)
    dupire_vol/data/                             (market params)
and writes thesis-ready matplotlib figures to
    iv_surface/results_4.3_plots/

Style conventions: serif font / cm mathtext, repricing trios at (4.5, 4.5)
with a square box, calls = tab:blue circles s=5, puts = tab:red circles s=5,
histograms in bp with mediumpurple bars + ME line, "Implied volatility error
(bp)" as the unit label.
"""
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

warnings.filterwarnings("ignore")

# ───────────────────────── Paths ─────────────────────────
HERE        = Path(__file__).resolve().parent          # iv_surface/
ROOT        = HERE.parent
IV_DIR      = HERE
DUPIRE_DIR  = ROOT / "dupire_vol"
LSV_DIR     = ROOT / "lsv"
BERGOMI_DIR = ROOT / "lsv_bergomi"
OUT_DIR     = HERE / "results_4.3_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────── Style ─────────────────────────
mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "mathtext.fontset":  "cm",
})

# Shared style constants (matches results_4_2_figures.py)
_REPR_FIGSIZE   = (4.5, 4.5)
_REPR_SCATTER_S = 5
_CALL_COLOR     = "tab:blue"
_PUT_COLOR      = "tab:red"
_ERR_LABEL      = "Implied volatility error (bp)"
_DIAG_FIGSIZE   = (6.5, 5.0)   # Heston-LSV vs Bergomi-LSV diagonal scatters


def _square_axes(ax) -> None:
    ax.set_box_aspect(1)


def _save(fig, name: str, tight: bool = True) -> Path:
    """Save and close. tight=False honours the declared figsize exactly so
    paired figures with different right-edge content (colourbar vs none)
    come out at identical pixel dimensions.

    Note: matplotlib's `savefig.bbox = 'tight'` rcParam overrides a
    `bbox_inches=None` kwarg, so we wrap in an rc_context to force the
    canvas-sized save when `tight=False`."""
    out = OUT_DIR / name
    if tight:
        fig.savefig(out)
    else:
        with mpl.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(out)
    plt.close(fig)
    return out


# ═════════════════════ Data loading ═══════════════════════
def load_data() -> dict:
    d = {}
    # ── Heston-LSV ────────────────────────────────────────
    d["heston_L"]       = np.load(LSV_DIR / "arrays" / "leverage_surface.npy")
    d["heston_S_grid"]  = np.load(LSV_DIR / "arrays" / "leverage_spot_grid.npy")
    d["heston_T_grid"]  = np.load(LSV_DIR / "arrays" / "leverage_time_grid.npy")
    d["heston_repr"]    = pd.read_csv(LSV_DIR / "data" / "lsv_repricing_errors.csv")
    with open(LSV_DIR / "data" / "validation_summary.json") as f:
        d["heston_val"] = json.load(f)

    # ── Bergomi-LSV ───────────────────────────────────────
    d["bergomi_L"]      = np.load(BERGOMI_DIR / "arrays" / "leverage_surface.npy")
    d["bergomi_S_grid"] = np.load(BERGOMI_DIR / "arrays" / "leverage_spot_grid.npy")
    d["bergomi_T_grid"] = np.load(BERGOMI_DIR / "arrays" / "leverage_time_grid.npy")
    d["bergomi_repr"]   = pd.read_csv(BERGOMI_DIR / "data" / "lsv_repricing_errors.csv")
    with open(BERGOMI_DIR / "data" / "validation_summary.json") as f:
        d["bergomi_val"] = json.load(f)

    # ── Market params (for the forward F(T) = S exp((r-q)T) on a uniform grid) ─
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        d["market"] = json.load(f)
    return d


# ══════════════════ §4.3 — Leverage surfaces ══════════════
# Two separate 3D-isometric surfaces, one per model. Axes and mesh ordering
# match the upstream pipeline plot_leverage_surface routines:
#   x = ln(S / S0)         (spot log-moneyness)
#   y = T                  (years)
#   z = L(t, S)            (leverage)
#   T_mesh, S_mesh = np.meshgrid(time_grid, spot_grid)
# so the data is used as-is from the .npy artefacts without any resampling.
# The colour scale is locked to [0, 5] for both plots — the particle method
# clamps L at 5 — so the two panels are directly comparable. 3D styling
# (figsize, view_init, orthographic projection, labelpad, fontsize) follows
# the §4.1 dupire/IV surface convention.

_LEV_VMIN, _LEV_VMAX = 0.0, 5.0
# Spot log-moneyness range applied to BOTH panels: the EXACT Bergomi-LSV
# calibrated grid extent. The Heston-LSV grid is far wider (≈ ±0.92) and
# carries boundary-instability spikes outside the calibrated wing; clipping
# both panels to the Bergomi extent preserves every Bergomi data point,
# crops Heston to the comparable domain, and removes the spikes from view.
# Asymmetry on the right is real (Bergomi's particle support is narrower
# above forward than below) and meaningful — not a plotting artefact.
_LEV_XLIM = (-0.357, 0.263)


def _plot_leverage_surface_3d(L, spot_grid, time_grid, S0, fname,
                                show_colorbar: bool):
    """3D isometric leverage surface in the §4.1 surface-plot style.

    Data orientation matches lsv/particle_method.py:plot_leverage_surface and
    lsv_bergomi/particle_method.py:plot_leverage_surface — no resampling.
    Both panels share the same `_LEV_XLIM` x-range and the same `[0, 5]`
    colour and z scale, so a reader can compare the surfaces directly.

    Only the Bergomi-LSV panel carries a colourbar (by convention — the
    scale is identical on both), but both panels are saved at the declared
    figsize with `bbox_inches=None` so their image dimensions match
    pixel-for-pixel; the Heston panel reserves the same horizontal area
    that the Bergomi colourbar occupies, just empty."""
    T_mesh, S_mesh = np.meshgrid(time_grid, spot_grid)
    log_moneyness = np.log(S_mesh / S0)

    fig = plt.figure(figsize=(9.0, 6.0))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(log_moneyness, T_mesh, L,
                            cmap=cm.viridis, edgecolor="none",
                            alpha=0.92, rcount=80, ccount=80,
                            vmin=_LEV_VMIN, vmax=_LEV_VMAX)
    if show_colorbar:
        cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.12)
        cb.set_label(r"Leverage  $L(t, S)$", fontsize=12)
    ax.set_xlabel(r"Spot log-moneyness  $\ln(S / S_0)$", labelpad=10, fontsize=12)
    ax.set_ylabel(r"Time to maturity $T$ (years)",       labelpad=10, fontsize=12)
    ax.set_zlabel(r"Leverage  $L(t, S)$",                labelpad=10, fontsize=12)
    ax.set_xlim(*_LEV_XLIM)
    ax.set_zlim(_LEV_VMIN, _LEV_VMAX)
    ax.view_init(elev=26, azim=-58)
    ax.set_proj_type("ortho")   # orthographic — no perspective distortion
    plt.tight_layout()
    return _save(fig, fname, tight=False)


def fig_heston_leverage_surface(d: dict) -> Path:
    return _plot_leverage_surface_3d(
        d["heston_L"], d["heston_S_grid"], d["heston_T_grid"],
        S0=d["market"]["S"],
        fname="heston_leverage_surface_3d.png",
        show_colorbar=False)


def fig_bergomi_leverage_surface(d: dict) -> Path:
    return _plot_leverage_surface_3d(
        d["bergomi_L"], d["bergomi_S_grid"], d["bergomi_T_grid"],
        S0=d["market"]["S"],
        fname="bergomi_leverage_surface_3d.png",
        show_colorbar=True)


# ══════════════════ §4.3 — Fourth Checkpoint trio ═════════
# Same styling as the §4.2 Third Checkpoint trio: square box, bp units,
# call/put circles, mediumpurple histogram with the ME line.
#
# Both `lsv/data/lsv_repricing_errors.csv` and `lsv_bergomi/data/lsv_repricing_errors.csv`
# share the same schema:
#   strike, ttm, option_type, moneyness, log_moneyness,
#   ssvi_price, lsv_price, dupire_price,
#   iv_ssvi, iv_lsv, lsv_iv_error_bps = (iv_lsv − iv_ssvi) * 1e4
# so the three trio helpers below take a generic dataframe.

def _trio_error_hist(df: pd.DataFrame, fname: str) -> Path:
    err = df["lsv_iv_error_bps"].dropna()
    me  = float(err.mean())
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.hist(err, bins=30, color="mediumpurple", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(me, color="red", lw=1.1, ls=":", label=f"ME {me:+.0f} bp")
    ax.set_xlabel(_ERR_LABEL)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def _trio_error_vs_price(df: pd.DataFrame, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    for sub, color, label in [
        (df[df["option_type"] == "call"], _CALL_COLOR, "calls"),
        (df[df["option_type"] == "put"],  _PUT_COLOR,  "puts"),
    ]:
        ax.scatter(sub["ssvi_price"], sub["lsv_iv_error_bps"],
                   s=_REPR_SCATTER_S, alpha=0.6,
                   color=color, marker="o", label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("SSVI option price (USD)")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def _trio_error_vs_moneyness(df: pd.DataFrame, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    for sub, color, label in [
        (df[df["option_type"] == "call"], _CALL_COLOR, "calls"),
        (df[df["option_type"] == "put"],  _PUT_COLOR,  "puts"),
    ]:
        ax.scatter(sub["log_moneyness"], sub["lsv_iv_error_bps"],
                   s=_REPR_SCATTER_S, alpha=0.6,
                   color=color, marker="o", label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Forward log-moneyness $k$")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def fig_heston_lsv_repricing_error_hist(d: dict) -> Path:
    return _trio_error_hist(d["heston_repr"],
                             "heston_lsv_repricing_error_hist.png")


def fig_heston_lsv_repricing_error_vs_price(d: dict) -> Path:
    return _trio_error_vs_price(d["heston_repr"],
                                 "heston_lsv_repricing_error_vs_price.png")


def fig_heston_lsv_repricing_error_vs_moneyness(d: dict) -> Path:
    return _trio_error_vs_moneyness(d["heston_repr"],
                                     "heston_lsv_repricing_error_vs_moneyness.png")


def fig_bergomi_lsv_repricing_error_hist(d: dict) -> Path:
    return _trio_error_hist(d["bergomi_repr"],
                             "bergomi_lsv_repricing_error_hist.png")


def fig_bergomi_lsv_repricing_error_vs_price(d: dict) -> Path:
    return _trio_error_vs_price(d["bergomi_repr"],
                                 "bergomi_lsv_repricing_error_vs_price.png")


def fig_bergomi_lsv_repricing_error_vs_moneyness(d: dict) -> Path:
    return _trio_error_vs_moneyness(d["bergomi_repr"],
                                     "bergomi_lsv_repricing_error_vs_moneyness.png")


# ══════════════════ §4.3 — LSV inter-model agreement ══════
# Two diagonal scatters comparing Heston-LSV against Bergomi-LSV on the
# IDENTICAL option set (1567 contracts, joined on (strike, ttm, option_type)
# from the two pipelines' lsv_repricing_errors.csv outputs). Points hugging
# the y=x diagonal indicate the two LSV backbones agree on vanilla pricing;
# spread off the diagonal exposes per-backbone bias. Colouring by option type
# instead of TTM lets us see whether disagreement clusters in calls or puts.
#
# Styling follows the §4.2 model-vs-SSVI scatter (figsize, y=x line, point size,
# edgecolor convention); only the colour mapping differs.

def _lsv_diagonal_join(d: dict) -> pd.DataFrame:
    """Inner-join the two LSV repricing CSVs on (strike, ttm, option_type)."""
    return d["heston_repr"].merge(
        d["bergomi_repr"], on=["strike", "ttm", "option_type"],
        suffixes=("_h", "_b"),
    )


def _diagonal_scatter(df, x_col, y_col, x_lab, y_lab, fname,
                       transform=None, pad_frac=0.05):
    """Generic y=x scatter coloured by call/put."""
    calls = df[df["option_type"] == "call"]
    puts  = df[df["option_type"] == "put"]
    x_all = df[x_col].to_numpy(); y_all = df[y_col].to_numpy()
    if transform is not None:
        x_all = transform(x_all); y_all = transform(y_all)
        x_c, y_c = transform(calls[x_col]), transform(calls[y_col])
        x_p, y_p = transform(puts[x_col]),  transform(puts[y_col])
    else:
        x_c, y_c = calls[x_col], calls[y_col]
        x_p, y_p = puts[x_col],  puts[y_col]

    fig, ax = plt.subplots(figsize=_DIAG_FIGSIZE)
    ax.scatter(x_c, y_c, s=_REPR_SCATTER_S, alpha=0.7,
               color=_CALL_COLOR, edgecolors="none", label="calls")
    ax.scatter(x_p, y_p, s=_REPR_SCATTER_S, alpha=0.7,
               color=_PUT_COLOR,  edgecolors="none", label="puts")

    lo = float(np.nanmin([x_all.min(), y_all.min()]))
    hi = float(np.nanmax([x_all.max(), y_all.max()]))
    pad = (hi - lo) * pad_frac
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=1.0, label=r"$y = x$")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def fig_lsv_iv_diagonal(d: dict) -> Path:
    """Heston-LSV implied vol vs Bergomi-LSV implied vol, in %."""
    m = _lsv_diagonal_join(d)
    return _diagonal_scatter(
        m, "iv_lsv_h", "iv_lsv_b",
        "Heston-LSV implied volatility (%)",
        "Bergomi-LSV implied volatility (%)",
        "lsv_iv_diagonal.png",
        transform=lambda x: np.asarray(x, dtype=float) * 100.0,
    )


def fig_lsv_price_diagonal(d: dict) -> Path:
    """Heston-LSV MC price vs Bergomi-LSV MC price (USD)."""
    m = _lsv_diagonal_join(d)
    return _diagonal_scatter(
        m, "lsv_price_h", "lsv_price_b",
        "Heston-LSV MC price (USD)",
        "Bergomi-LSV MC price (USD)",
        "lsv_price_diagonal.png",
    )


# ══════════════════════════ Main ══════════════════════════
def main():
    print(f"Output dir: {OUT_DIR.resolve()}")
    print("Loading pipeline artefacts ...")
    d = load_data()

    # Quick metric digest so the user can sanity-check vs the table source.
    print(f"  Heston-LSV  : MAE={d['heston_val']['lsv_iv_mae_bps']:.1f} bp  "
          f"RMSE={d['heston_val']['lsv_iv_rmse_bps']:.1f} bp  "
          f"ME={d['heston_val']['lsv_iv_me_bps']:+.1f} bp  "
          f"(n={d['heston_val']['n_valid']})")
    print(f"  Bergomi-LSV : MAE={d['bergomi_val']['lsv_iv_mae_bps']:.1f} bp  "
          f"RMSE={d['bergomi_val']['lsv_iv_rmse_bps']:.1f} bp  "
          f"ME={d['bergomi_val']['lsv_iv_me_bps']:+.1f} bp  "
          f"(n={d['bergomi_val']['n_valid']})")

    figures = [
        ("§4.3   Heston-LSV  —  leverage surface 3D",     fig_heston_leverage_surface),
        ("§4.3   Bergomi-LSV —  leverage surface 3D",     fig_bergomi_leverage_surface),
        ("§4.3.4 Heston-LSV  —  IV err hist",             fig_heston_lsv_repricing_error_hist),
        ("§4.3.4 Heston-LSV  —  IV err vs price",         fig_heston_lsv_repricing_error_vs_price),
        ("§4.3.4 Heston-LSV  —  IV err vs k",             fig_heston_lsv_repricing_error_vs_moneyness),
        ("§4.3.4 Bergomi-LSV —  IV err hist",             fig_bergomi_lsv_repricing_error_hist),
        ("§4.3.4 Bergomi-LSV —  IV err vs price",         fig_bergomi_lsv_repricing_error_vs_price),
        ("§4.3.4 Bergomi-LSV —  IV err vs k",             fig_bergomi_lsv_repricing_error_vs_moneyness),
        ("§4.3   Heston-LSV IV vs Bergomi-LSV IV",        fig_lsv_iv_diagonal),
        ("§4.3   Heston-LSV price vs Bergomi-LSV price",  fig_lsv_price_diagonal),
    ]

    print()
    n_ok = 0; n_fail = 0
    for label, fn in figures:
        try:
            out = fn(d)
            print(f"  OK    {label:<46}  {out.name}"); n_ok += 1
        except Exception as exc:
            print(f"  FAIL  {label:<46}  {exc!r}"); n_fail += 1

    print(f"\n{n_ok} files written, {n_fail} failed.")
    print(f"All outputs under {OUT_DIR}/")


if __name__ == "__main__":
    main()

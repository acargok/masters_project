#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
appendix_cliquet_plots.py — thesis appendix figure pack for cliquet pricing
==============================================================================
Companion to results_4_4_figures.py. Generates the deeper-dive cliquet
figures beyond the core §4.4 set (price bars, payoff CDFs, paths + variance,
decomposition table).

Figures produced (under iv_surface/appendix_plots/):

  1. cliquet_mc_convergence.png       — running price estimate ±1.96 SE
                                          per cliquet, both LSV models
  2. cliquet_reset_returns.png        — per-reset return box plots,
                                          Heston-LSV row / Bergomi-LSV row
  3. cliquet_payoff_histograms.png    — per-path payoff histograms
                                          (density counterpart to §4.4 CDFs)
  4. napoleon_worst_reset.png         — distribution of which reset becomes
                                          the worst per path (Napoleon)
  5. reverse_cliquet_consumption.png  — coupon-remaining histograms
"""
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ───────────────────────── Paths ─────────────────────────
HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
PRICING_DIR = ROOT / "pricing"
OUT_DIR     = HERE / "appendix_plots"
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

# Reuse the §4.4 model-colour palette so figures match across the section.
_HESTON_C  = "#ff7f0e"
_BERGOMI_C = "#9467bd"

_CLIQUETS    = ["accumulator", "reverse_cliquet", "napoleon"]
_CLIQ_LABEL  = {
    "accumulator":     "Accumulator",
    "reverse_cliquet": "Reverse Cliquet",
    "napoleon":        "Napoleon",
}


def _save(fig, name: str, tight: bool = True) -> Path:
    """Save and close. tight=False honours the declared figsize exactly so
    paired figures come out at identical pixel dimensions (overriding the
    global `savefig.bbox = 'tight'` rcParam)."""
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
    with open(PRICING_DIR / "data" / "pricing_results.json") as f:
        d["results"] = json.load(f)

    arr = PRICING_DIR / "arrays"
    d["payoffs"], d["returns"] = {}, {}
    for c in _CLIQUETS:
        d["payoffs"][c] = {
            "heston_lsv":  np.load(arr / f"{c}_payoffs.npy"),
            "bergomi_lsv": np.load(arr / f"bergomi_{c}_payoffs.npy"),
        }
        d["returns"][c] = {
            "heston_lsv":  np.load(arr / f"{c}_returns.npy"),
            "bergomi_lsv": np.load(arr / f"bergomi_{c}_returns.npy"),
        }
    return d


# ═════════════ 1. Monte-Carlo convergence ═════════════════
def fig_mc_convergence(d: dict) -> Path:
    """Running price estimate ±1.96 SE as a function of the number of
    paths, log-scale x-axis. One panel per cliquet, both LSV models on
    each. Pulls the discount factor from `pricing_results.json` per model
    so the running mean is in present-value units."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True)
    for ax, c in zip(axes, _CLIQUETS):
        df_h = d["results"]["heston_options"][c]["discount_factor"]
        df_b = d["results"]["bergomi_options"][c]["discount_factor"]
        p_h_final = d["results"]["heston_options"][c]["price"]
        p_b_final = d["results"]["bergomi_options"][c]["price"]

        for payoffs, df_factor, color, label, final in [
            (d["payoffs"][c]["heston_lsv"],  df_h, _HESTON_C,
             "Heston-LSV",  p_h_final),
            (d["payoffs"][c]["bergomi_lsv"], df_b, _BERGOMI_C,
             "Bergomi-LSV", p_b_final),
        ]:
            x = (payoffs * df_factor).astype(np.float64)
            n_total = x.size
            # 80 log-spaced sample points between 100 and n_total.
            counts = np.unique(np.geomspace(100, n_total, 80).astype(int))
            counts = counts[counts <= n_total]
            cum_sum = np.cumsum(x)
            cum_sum_sq = np.cumsum(x ** 2)
            means = cum_sum[counts - 1] / counts
            variances = cum_sum_sq[counts - 1] / counts - means ** 2
            ses = np.sqrt(np.maximum(variances, 0) / counts)

            ax.fill_between(counts, (means - 1.96 * ses) * 100,
                                     (means + 1.96 * ses) * 100,
                             color=color, alpha=0.18, linewidth=0)
            ax.plot(counts, means * 100, color=color, lw=1.6, label=label)
            ax.axhline(final * 100, color=color, lw=0.7, ls="--", alpha=0.85)

        ax.set_xscale("log")
        ax.set_title(_CLIQ_LABEL[c], fontsize=11)
        ax.set_xlabel("Number of paths")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("Running price (% of spot)")
    axes[0].legend(loc="best", framealpha=1.0, fontsize=9).set_zorder(10)
    plt.tight_layout()
    return _save(fig, "cliquet_mc_convergence.png")


# ═════════════ 2. Per-reset return distributions ═══════════
# Per-reset returns are independent of the cliquet payoff structure (the same
# underlying MC paths feed every cliquet; only the post-processing differs), so
# a single-panel per-model figure carries the full content for any one cliquet.
_RESET_FIGSIZE = (6.5, 5.0)


def _plot_reset_returns(returns: np.ndarray, color: str,
                         fname: str, title: str) -> Path:
    """Box-and-whisker of per-reset returns (one box per monthly reset).

    Style: alpha-0.55 box fills coloured by model, coloured whiskers/caps,
    black median, outliers hidden, dotted zero line, M1..M_n ticks."""
    n_resets = returns.shape[1]
    fig, ax = plt.subplots(figsize=_RESET_FIGSIZE)
    bp = ax.boxplot(
        [returns[:, i] * 100.0 for i in range(n_resets)],
        positions=np.arange(1, n_resets + 1),
        widths=0.65, showfliers=False, patch_artist=True,
        medianprops=dict(color="black", lw=1.0),
    )
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.55, edgecolor=color)
    for whisker in bp["whiskers"]:
        whisker.set(color=color)
    for cap in bp["caps"]:
        cap.set(color=color)
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Reset period")
    ax.set_ylabel("Return (%)")
    ax.set_xticks(np.arange(1, n_resets + 1))
    ax.set_xticklabels([f"M{i}" for i in range(1, n_resets + 1)], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save(fig, fname, tight=False)


def fig_reset_returns_heston(d: dict) -> Path:
    # Returns are identical across cliquets per model — pick any.
    return _plot_reset_returns(
        d["returns"]["accumulator"]["heston_lsv"],
        color=_HESTON_C,
        fname="cliquet_reset_returns_heston.png",
        title="Heston-LSV")


def fig_reset_returns_bergomi(d: dict) -> Path:
    return _plot_reset_returns(
        d["returns"]["accumulator"]["bergomi_lsv"],
        color=_BERGOMI_C,
        fname="cliquet_reset_returns_bergomi.png",
        title="Bergomi-LSV")


# ═════════════ 3. Per-path payoff histograms ══════════════
def fig_payoff_histograms(d: dict) -> Path:
    """Histogram of per-path payoffs for each cliquet, Heston-LSV and
    Bergomi-LSV overlaid (alpha 0.55). Mean line shown for each model in
    matching colour. Annotated with the % of zero-payoff paths so the
    reader sees the digital-tail behaviour for reverse cliquet / Napoleon."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, c in zip(axes, _CLIQUETS):
        for tag, color, label in [
            ("heston_lsv",  _HESTON_C,  "Heston-LSV"),
            ("bergomi_lsv", _BERGOMI_C, "Bergomi-LSV"),
        ]:
            p = d["payoffs"][c][tag] * 100.0  # to %
            ax.hist(p, bins=60, color=color, alpha=0.55,
                    edgecolor="white", linewidth=0.3, label=label)
            ax.axvline(p.mean(), color=color, lw=1.2, ls="--", alpha=0.95)

        # Annotation: % zero-payoff per model
        z_h = (d["payoffs"][c]["heston_lsv"] == 0.0).mean() * 100.0
        z_b = (d["payoffs"][c]["bergomi_lsv"] == 0.0).mean() * 100.0
        if z_h > 0.1 or z_b > 0.1:
            ax.text(0.97, 0.95,
                    f"zero payoff:\nH  {z_h:5.1f}%\nB  {z_b:5.1f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="grey",
                              alpha=0.92))

        ax.set_title(_CLIQ_LABEL[c], fontsize=11)
        ax.set_xlabel("Payoff (% of spot)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right", framealpha=1.0,
                    fontsize=9).set_zorder(10)
    plt.tight_layout()
    return _save(fig, "cliquet_payoff_histograms.png")


# ═════════════ 4. Napoleon worst-reset analysis ═══════════
def fig_napoleon_worst_reset(d: dict) -> Path:
    """For Napoleon: which reset is the worst per path? Two panels:
       (a) Histogram (grouped bars per reset index) of the empirical
           distribution of the worst-reset index, Heston vs Bergomi.
       (b) Scatter of worst-reset return vs final payoff, with the
           coupon level marked. Shows how the worst return drives the
           payoff curtailment.
    """
    returns_h  = d["returns"]["napoleon"]["heston_lsv"]
    returns_b  = d["returns"]["napoleon"]["bergomi_lsv"]
    payoffs_h  = d["payoffs"]["napoleon"]["heston_lsv"]
    payoffs_b  = d["payoffs"]["napoleon"]["bergomi_lsv"]
    coupon = d["results"]["heston_options"]["napoleon"]["payoff_kwargs"]["coupon"]
    n_resets = returns_h.shape[1]

    worst_idx_h = np.argmin(returns_h, axis=1)
    worst_idx_b = np.argmin(returns_b, axis=1)
    worst_val_h = returns_h[np.arange(len(returns_h)), worst_idx_h] * 100.0
    worst_val_b = returns_b[np.arange(len(returns_b)), worst_idx_b] * 100.0

    counts_h = np.bincount(worst_idx_h, minlength=n_resets) / len(worst_idx_h) * 100.0
    counts_b = np.bincount(worst_idx_b, minlength=n_resets) / len(worst_idx_b) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (a) Bar chart of worst-reset index frequency
    ax = axes[0]
    x = np.arange(1, n_resets + 1)
    bar_w = 0.4
    ax.bar(x - bar_w / 2, counts_h, width=bar_w,
           color=_HESTON_C,  alpha=0.85, edgecolor="white", linewidth=0.4,
           label="Heston-LSV")
    ax.bar(x + bar_w / 2, counts_b, width=bar_w,
           color=_BERGOMI_C, alpha=0.85, edgecolor="white", linewidth=0.4,
           label="Bergomi-LSV")
    ax.set_xticks(x); ax.set_xticklabels([f"M{i}" for i in x], fontsize=8)
    ax.set_xlabel("Reset period")
    ax.set_ylabel("Share of paths (%)")
    ax.set_title("Worst-reset index distribution", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", framealpha=1.0).set_zorder(10)

    # (b) Scatter of worst-reset return vs final payoff
    ax = axes[1]
    rng = np.random.default_rng(0)
    n_show = 5000
    for tag, worst_val, payoffs, color, label in [
        ("h", worst_val_h, payoffs_h, _HESTON_C,  "Heston-LSV"),
        ("b", worst_val_b, payoffs_b, _BERGOMI_C, "Bergomi-LSV"),
    ]:
        idx = rng.choice(len(worst_val), n_show, replace=False)
        ax.scatter(worst_val[idx], payoffs[idx] * 100.0,
                   s=3, alpha=0.25, color=color, edgecolors="none",
                   label=label)
    ax.axvline(-coupon * 100, color="black", lw=0.8, ls="--",
               label=f"$-$coupon = $-${coupon * 100:.0f}%")
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.set_xlabel("Worst-reset return (%)")
    ax.set_ylabel("Final payoff (% of spot)")
    ax.set_title("Worst return $\\to$ payoff", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=1.0, fontsize=9).set_zorder(10)

    plt.tight_layout()
    return _save(fig, "napoleon_worst_reset.png")


# ═════════════ 5. Reverse cliquet coupon consumption ══════
def fig_reverse_consumption(d: dict) -> Path:
    """For the reverse cliquet: the coupon is paid at maturity minus the
    sum of negative monthly returns (floored at zero). Plotted is the
    *remaining* coupon (= coupon + Σ min(r_i, 0), bottom-floored at zero)
    as a histogram, with three regime annotations: fully consumed, partial,
    intact. Heston-LSV and Bergomi-LSV side-by-side on a shared x-axis."""
    returns_h = d["returns"]["reverse_cliquet"]["heston_lsv"]
    returns_b = d["returns"]["reverse_cliquet"]["bergomi_lsv"]
    coupon = d["results"]["heston_options"]["reverse_cliquet"]["payoff_kwargs"]["coupon"]

    remaining_h = coupon + np.minimum(returns_h, 0.0).sum(axis=1)
    remaining_b = coupon + np.minimum(returns_b, 0.0).sum(axis=1)

    lo = float(min(remaining_h.min(), remaining_b.min())) * 100.0
    hi = float(coupon) * 100.0 * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                              sharex=True, sharey=False)
    for ax, rem, color, label in [
        (axes[0], remaining_h, _HESTON_C,  "Heston-LSV"),
        (axes[1], remaining_b, _BERGOMI_C, "Bergomi-LSV"),
    ]:
        rem_pct = rem * 100.0
        ax.hist(rem_pct, bins=80, range=(lo, hi),
                color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvline(coupon * 100.0, color="grey", lw=0.8, ls=":")
        # Regime stats.
        pct_fully    = (rem <= 0).mean() * 100.0
        pct_partial  = ((rem > 0) & (rem < coupon)).mean() * 100.0
        pct_intact   = (rem >= coupon).mean() * 100.0
        ax.text(0.03, 0.95,
                f"consumed:  {pct_fully:5.1f}%\n"
                f"partial:    {pct_partial:5.1f}%\n"
                f"intact:    {pct_intact:5.1f}%",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="grey", alpha=0.92))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Coupon remaining (% of spot)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    return _save(fig, "reverse_cliquet_consumption.png")


# ══════════════════════════ Main ══════════════════════════
def main():
    print(f"Output dir: {OUT_DIR.resolve()}")
    print("Loading pricing artefacts ...")
    d = load_data()

    figures = [
        ("Monte-Carlo convergence",        fig_mc_convergence),
        ("Reset returns — Heston-LSV",     fig_reset_returns_heston),
        ("Reset returns — Bergomi-LSV",    fig_reset_returns_bergomi),
        ("Per-path payoff histograms",     fig_payoff_histograms),
        ("Napoleon worst-reset analysis",  fig_napoleon_worst_reset),
        ("Reverse cliquet consumption",    fig_reverse_consumption),
    ]
    n_ok = 0; n_fail = 0
    for label, fn in figures:
        try:
            out = fn(d)
            print(f"  OK    {label:<38}  {out.name}"); n_ok += 1
        except Exception as exc:
            print(f"  FAIL  {label:<38}  {exc!r}"); n_fail += 1

    print(f"\n{n_ok} files written, {n_fail} failed.")
    print(f"All outputs under {OUT_DIR}/")


if __name__ == "__main__":
    main()

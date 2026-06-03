#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report and plots for the forward-skew decomposition.

Builds the headline cliquet-price table with the forward-skew contribution
G = (Δ_baseline − Δ_zero) / Δ_baseline (Δ = P_Heston − P_Bergomi), plus a
vanilla-IV-residual summary and two comparison plots.
"""
import json
import logging

import numpy as np

import decomp_config as cfg

logger = logging.getLogger("decomp_report")
NAMES = cfg.PAYOFF_NAMES


def load_baseline():
    """Production cliquet pricing results, or None if not yet generated."""
    path = cfg.PRICING_DIR / "data" / "pricing_results.json"
    if not path.exists():
        logger.warning(f"Baseline pricing not found at {path} — "
                       "report baseline columns will be NaN.")
        return None
    with open(path) as f:
        return json.load(f)


def _price(d, side, name):
    if d is None:
        return float("nan")
    return d[side][name]["price"]


def _verdict(contrib):
    if not np.isfinite(contrib):
        return "baseline H−B gap too small to attribute"
    if contrib > 0.6:
        return "gap driven primarily by forward skew (spot-variance correlation)"
    if contrib > 0.2:
        return "forward skew contributes substantially but not exclusively"
    if contrib > -0.2:
        return "gap largely independent of forward skew (vol-of-vol / factor structure dominates)"
    return "removing forward skew widens the gap"


def write_report(baseline, zero_corr, h_val, b_val):
    """Write report.md with the decomposition table and interpretation."""
    L = []
    L.append("# Forward-skew decomposition — contribution to the cliquet H–B gap")
    L.append("")
    L.append("Both backbones are re-run with spot-variance correlation forced to "
             "zero (Heston rho = 0; Bergomi rho1 = rho2 = 0), the leverage function "
             "recalibrated via the particle method, and the three cliquets re-priced. "
             "All other calibrated parameters and the forward variance curve are "
             "held at production values. Production outputs are not modified.")
    L.append("")
    L.append(f"**Config:** N_particles=5000, dt=1/504 (leverage); "
             f"N_paths={cfg.N_PATHS:,}, dt_max={cfg.DT_MAX:.4f}, seed={cfg.SEED} (pricing).")
    L.append("")
    L.append("## Cliquet prices")
    L.append("")
    L.append("| Option | Heston (base) | Heston (ρ=0) | Bergomi (base) | Bergomi (ρ₁=ρ₂=0) | H−B (base) | H−B (ρ=0) | Fwd-skew G |")
    L.append("|---|---|---|---|---|---|---|---|")

    decomp = {}
    for n in NAMES:
        bh = _price(baseline, "heston_options", n)
        bb = _price(baseline, "bergomi_options", n)
        zh = zero_corr["heston_options"][n]["price"]
        zb = zero_corr["bergomi_options"][n]["price"]
        d_base = bh - bb
        d_zero = zh - zb
        contrib = (d_base - d_zero) / d_base if abs(d_base) > 1e-12 else float("nan")
        decomp[n] = {"hb_base": d_base, "hb_zero": d_zero, "G": contrib}
        L.append(f"| {n} | {bh:.6f} | {zh:.6f} | {bb:.6f} | {zb:.6f} | "
                 f"{d_base:+.6f} | {d_zero:+.6f} | {contrib * 100:+.1f}% |")
    L.append("")
    L.append("G = (baseline H−B − zero-corr H−B) / baseline H−B. Near 100%: the gap "
             "is dominated by forward skew. Near 0%: the gap is driven by other "
             "features (vol-of-vol, mean-reversion, factor structure). Negative: the "
             "gap widens without spot-variance correlation.")
    L.append("")
    L.append("## Vanilla IV residuals (LSV repricing, ρ = 0)")
    L.append("")
    L.append("| Backbone | N | MAE (bp) | ME (bp) | RMSE (bp) |")
    L.append("|---|---|---|---|---|")
    for label, v in (("Heston (ρ=0)", h_val), ("Bergomi (ρ₁=ρ₂=0)", b_val)):
        L.append(f"| {label} | {v.get('n_valid')} | "
                 f"{v.get('lsv_iv_mae_bps', float('nan')):.1f} | "
                 f"{v.get('lsv_iv_me_bps', float('nan')):+.1f} | "
                 f"{v.get('lsv_iv_rmse_bps', float('nan')):.1f} |")
    L.append("")
    L.append("Higher residuals than the calibrated baselines are expected: with the "
             "spot-variance correlation gone, the leverage function alone must match "
             "the SPX skew.")
    L.append("")
    L.append("## Interpretation")
    L.append("")
    for n in NAMES:
        d = decomp[n]
        L.append(f"- **{n}**: baseline H−B = {d['hb_base']:+.6f}, zero-corr H−B = "
                 f"{d['hb_zero']:+.6f}, G = {d['G'] * 100:+.1f}% — {_verdict(d['G'])}.")
    L.append("")

    cfg.REPORT_PATH.write_text("\n".join(L))
    logger.info(f"Wrote report → {cfg.REPORT_PATH}")
    return decomp


def generate_plots(baseline, zero_corr):
    """Bar chart of H−B differences and leverage-slice comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # (1) H-B difference: baseline vs zero-correlation.
    base_diff = [_price(baseline, "heston_options", n) - _price(baseline, "bergomi_options", n)
                 for n in NAMES]
    zero_diff = [zero_corr["heston_options"][n]["price"] - zero_corr["bergomi_options"][n]["price"]
                 for n in NAMES]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(NAMES))
    w = 0.35
    ax.bar(x - w / 2, base_diff, w, label="Baseline (calibrated ρ)",
           color="#1f77b4", edgecolor="black")
    ax.bar(x + w / 2, zero_diff, w, label="ρ = 0", color="#d62728", edgecolor="black")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("H − B cliquet price difference")
    ax.set_title("Heston − Bergomi cliquet gap: baseline vs ρ = 0",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.PLOTS_DIR / "hb_diff_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    # (2) Leverage slices at T in {0.25, 0.50, 1.00}, baseline vs ρ = 0.
    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 9))
        targets = [0.25, 0.50, 1.00]
        colours = ["#1f77b4", "#2ca02c", "#9467bd"]
        panels = [
            (axes[0], cfg.LSV_HESTON_DIR / "arrays", cfg.HESTON0_DIR / "arrays",
             "Heston LSV leverage: baseline vs ρ = 0", "ρ=0"),
            (axes[1], cfg.LSV_BERGOMI_DIR / "arrays", cfg.BERGOMI0_DIR / "arrays",
             "Bergomi LSV leverage: baseline vs ρ₁ = ρ₂ = 0", "ρ₁=ρ₂=0"),
        ]
        for ax, base_dir, zero_dir, title, zlabel in panels:
            lb = np.load(base_dir / "leverage_surface.npy")
            sb = np.load(base_dir / "leverage_spot_grid.npy")
            tb = np.load(base_dir / "leverage_time_grid.npy")
            l0 = np.load(zero_dir / "leverage_surface.npy")
            s0 = np.load(zero_dir / "leverage_spot_grid.npy")
            t0 = np.load(zero_dir / "leverage_time_grid.npy")
            for t, c in zip(targets, colours):
                jb = int(np.argmin(np.abs(tb - t)))
                j0 = int(np.argmin(np.abs(t0 - t)))
                ax.plot(sb, lb[:, jb], color=c, lw=1.4, label=f"base T={tb[jb]:.2f}y")
                ax.plot(s0, l0[:, j0], color=c, lw=1.4, ls="--",
                        label=f"{zlabel} T={t0[j0]:.2f}y")
            ax.set_xlabel("Spot")
            ax.set_ylabel("Leverage L(t, S)")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(cfg.PLOTS_DIR / "leverage_comparison.png", dpi=140, bbox_inches="tight")
        plt.close()
    except FileNotFoundError as e:
        logger.warning(f"Skipped leverage comparison plot (missing baseline arrays): {e}")
    logger.info(f"Saved plots → {cfg.PLOTS_DIR}")

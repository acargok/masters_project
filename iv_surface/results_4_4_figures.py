#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results_4_4_figures.py — thesis §4.4 (Cliquet Pricing) figure pack
====================================================================
Stand-alone script. Reads cached pipeline artefacts from
    pricing/data/pricing_results.json                  (baseline)
    pricing/arrays/*                                   (sample paths, payoffs)
    pricing/experiments/decomposition/pricing_results.json  (ρ = 0 run)
and writes thesis-ready matplotlib figures + LaTeX tables to
    iv_surface/results_4.4_plots/

Style conventions: serif font / cm mathtext, square boxes on per-cliquet
panels, distinct colour per pricing model held constant across the section.

§4.4.1  Cliquet prices across five models (table + grouped bars)
§4.4.2  Payoff CDFs (one panel per cliquet) + paths/variance two-panel
§4.4.3  Forward-skew decomposition (table + grouped bars)
"""
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ───────────────────────── Paths ─────────────────────────
HERE         = Path(__file__).resolve().parent
ROOT         = HERE.parent
PRICING_DIR  = ROOT / "pricing"
OUT_DIR      = HERE / "results_4.4_plots"
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

# Per-cliquet ordering and display labels (used by every figure / table).
_CLIQUETS = ["accumulator", "reverse_cliquet", "napoleon"]
_CLIQ_LABEL = {
    "accumulator":     "Accumulator",
    "reverse_cliquet": "Reverse Cliquet",
    "napoleon":        "Napoleon",
}

# Model colours — fixed across every §4.4 figure so a reader can trace the
# same model across the price bars, payoff CDFs, and path overlays. The two
# orange shades pair Pure Heston with its LSV variant; the two purple shades
# pair Pure Bergomi with its LSV variant (matplotlib's tab20 light/dark pairs).
_MODEL_COLOR = {
    "bs":            "#7f7f7f",   # neutral grey
    "pure_heston":   "#ffbb78",   # light orange  (tab20 pair of tab:orange)
    "heston_lsv":    "#ff7f0e",   # tab:orange
    "pure_bergomi":  "#c5b0d5",   # light purple  (tab20 pair of tab:purple)
    "bergomi_lsv":   "#9467bd",   # tab:purple
}
_MODEL_LABEL = {
    "bs":           "Black-Scholes",
    "pure_heston":  "Pure Heston",
    "heston_lsv":   "Heston-LSV",
    "pure_bergomi": "Pure Bergomi",
    "bergomi_lsv":  "Bergomi-LSV",
}
_MODEL_ORDER = ["bs", "pure_heston", "heston_lsv", "pure_bergomi", "bergomi_lsv"]


def _save(fig, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


# ═════════════════════ Data loading ═══════════════════════
def load_data() -> dict:
    d = {}
    with open(PRICING_DIR / "data" / "pricing_results.json") as f:
        d["baseline"] = json.load(f)
    with open(PRICING_DIR / "experiments" / "decomposition" / "pricing_results.json") as f:
        d["zero_corr"] = json.load(f)

    # Sample paths + variance + payoff arrays for §4.4.2.
    arr = PRICING_DIR / "arrays"
    d["paths"] = {
        "t":         np.load(arr / "sim_time_grid.npy"),
        "t_b":       np.load(arr / "bergomi_sim_time_grid.npy"),
        "S_h":       np.load(arr / "sample_paths_S.npy"),
        "S_b":       np.load(arr / "bergomi_sample_paths_S.npy"),
        "V_h":       np.load(arr / "sample_paths_V.npy"),
        "V_b":       np.load(arr / "bergomi_sample_paths_V.npy"),
        "resets":    np.load(arr / "reset_dates.npy"),
    }
    d["payoffs"] = {}
    for c in _CLIQUETS:
        d["payoffs"][c] = {
            "heston_lsv":  np.load(arr / f"{c}_payoffs.npy"),
            "bergomi_lsv": np.load(arr / f"bergomi_{c}_payoffs.npy"),
        }
    return d


# Helper: extract per-model price + MC stats per cliquet from the baseline JSON.
def _five_model_prices(res: dict, c: str) -> dict:
    """Returns a dict keyed by model id with {price, se, ci_half}.

    Pure Heston / Bergomi prices live as side-channels inside each LSV block
    of the same JSON — the pricing engine computes all five model prices in
    one pass. The BS-flat reference uses the LSV block's own implied ATM vol,
    so the two BS columns disagree by a few bp; for the headline table we
    report the Heston-block BS value and flag the Bergomi-block value in the
    caption."""
    h = res["heston_options"][c]
    b = res["bergomi_options"][c]
    return {
        "bs":           {"price": h["bs_price"],
                          "se":    h["bs_se"],
                          "ci_half": 1.96 * h["bs_se"]},
        "pure_heston":  {"price": h["heston_price"],
                          "se":    h["heston_se"],
                          "ci_half": 1.96 * h["heston_se"]},
        "pure_bergomi": {"price": h["bergomi_price"],
                          "se":    h["bergomi_se"],
                          "ci_half": 1.96 * h["bergomi_se"]},
        "heston_lsv":   {"price": h["price"],
                          "se":    h["se"],
                          "ci_half": h["ci_half"]},
        "bergomi_lsv":  {"price": b["price"],
                          "se":    b["se"],
                          "ci_half": b["ci_half"]},
    }


# ══════════════════ §4.4.1 — Price comparison ═════════════
def fig_cliquet_prices_table(d: dict) -> Path:
    """LaTeX table: 3 cliquets × 5 models with prices and 95% MC half-widths
    in percent of the spot price."""
    lines = [
        "% Auto-generated by results_4_4_figures.py — do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cliquet prices under five pricing models. Values are "
        r"present-value prices as a percentage of the spot price.}",
        r"\label{tab:cliquet_prices}",
        r"\begin{tabular}{l" + "c" * len(_CLIQUETS) + "}",
        r"\toprule",
        "Model & " + " & ".join(_CLIQ_LABEL[c] for c in _CLIQUETS) + r" \\",
        r"\midrule",
    ]
    base = d["baseline"]
    for m in _MODEL_ORDER:
        row = [_MODEL_LABEL[m]]
        for c in _CLIQUETS:
            p = _five_model_prices(base, c)[m]
            row.append(
                f"{p['price'] * 100.0:.3f}\\% $\\pm$ "
                f"{p['ci_half'] * 100.0:.3f}\\%"
            )
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = OUT_DIR / "cliquet_prices_table.tex"
    out.write_text("\n".join(lines))
    return out


def fig_cliquet_price_comparison_bars(d: dict) -> Path:
    """Grouped bar chart: per cliquet, five bars (one per model)."""
    base = d["baseline"]
    n_groups = len(_CLIQUETS)
    n_models = len(_MODEL_ORDER)
    bar_w = 0.16
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, m in enumerate(_MODEL_ORDER):
        prices = np.array([_five_model_prices(base, c)[m]["price"] for c in _CLIQUETS])
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(x + offset, prices, width=bar_w,
               color=_MODEL_COLOR[m], label=_MODEL_LABEL[m],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([_CLIQ_LABEL[c] for c in _CLIQUETS])
    ax.set_ylabel("Price (fraction of notional)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.92)
    plt.tight_layout()
    return _save(fig, "cliquet_price_comparison_bars.png")


# ══════════════════ §4.4.2 — Payoff CDFs ══════════════════
def fig_cliquet_payoff_cdfs(d: dict) -> Path:
    """One subplot per cliquet, payoff CDFs for the two LSV models overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, c in zip(axes, _CLIQUETS):
        for tag in ("heston_lsv", "bergomi_lsv"):
            p = np.asarray(d["payoffs"][c][tag], dtype=float)
            p_sorted = np.sort(p)
            n = p_sorted.size
            F = np.arange(1, n + 1) / n
            # Subsample to keep PDF size reasonable.
            step = max(1, n // 4000)
            ax.plot(p_sorted[::step], F[::step],
                    color=_MODEL_COLOR[tag], lw=1.6,
                    label=_MODEL_LABEL[tag])
        ax.set_title(_CLIQ_LABEL[c], fontsize=11)
        ax.set_xlabel("Payoff (fraction of spot)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(r"CDF  $F_H(x)$")
    axes[0].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return _save(fig, "cliquet_payoff_cdfs.png")


def fig_cliquet_paths_panel(d: dict) -> Path:
    """Two-panel figure stacking sample S-paths and variance trajectories.

    Top panel: a few hundred sample spot paths from each LSV model, lightly
    overlaid in the model colour. Reset dates marked with thin vertical lines
    so the cliquet's monthly grid is visible.

    Bottom panel: variance trajectories V(t) under each LSV model. Mean ± 1σ
    band drawn solid, with a handful of individual paths thinly overlaid so
    the reader sees both the typical level and the tail behaviour."""
    paths = d["paths"]
    t      = paths["t"]
    t_b    = paths["t_b"]
    S_h    = paths["S_h"]
    S_b    = paths["S_b"]
    V_h    = paths["V_h"]
    V_b    = paths["V_b"]
    resets = paths["resets"]

    S0 = float(d["baseline"]["S0"])

    n_show = min(10, S_h.shape[0], S_b.shape[0], V_h.shape[0], V_b.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.5), sharex=True)

    # ── Top: spot paths (10 per model, absolute price units) ──────────
    ax = axes[0]
    for ti in resets:
        ax.axvline(float(ti), color="grey", lw=0.3, alpha=0.4, zorder=1)
    for i in range(n_show):
        ax.plot(t,   S_h[i], color=_MODEL_COLOR["heston_lsv"],
                lw=0.9, alpha=0.40, zorder=2)
        ax.plot(t_b, S_b[i], color=_MODEL_COLOR["bergomi_lsv"],
                lw=0.9, alpha=0.40, zorder=2)
    # Mean line drawn opaquely over the full 200-path ensemble — carries
    # the legend label so the legend stays uncluttered.
    ax.plot(t,   np.mean(S_h, axis=0), color=_MODEL_COLOR["heston_lsv"],
            lw=1.8, zorder=4, label=_MODEL_LABEL["heston_lsv"])
    ax.plot(t_b, np.mean(S_b, axis=0), color=_MODEL_COLOR["bergomi_lsv"],
            lw=1.8, zorder=4, label=_MODEL_LABEL["bergomi_lsv"])
    ax.axhline(S0, color="black", lw=0.5, ls=":")
    ax.set_ylabel(r"Spot price $S(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # ── Bottom: variance trajectories (10 per model, no band) ─────────
    ax = axes[1]
    for i in range(n_show):
        ax.plot(t,   V_h[i], color=_MODEL_COLOR["heston_lsv"],
                lw=0.9, alpha=0.40, zorder=2)
        ax.plot(t_b, V_b[i], color=_MODEL_COLOR["bergomi_lsv"],
                lw=0.9, alpha=0.40, zorder=2)
    ax.plot(t,   np.mean(V_h, axis=0), color=_MODEL_COLOR["heston_lsv"],
            lw=1.8, zorder=4, label=_MODEL_LABEL["heston_lsv"])
    ax.plot(t_b, np.mean(V_b, axis=0), color=_MODEL_COLOR["bergomi_lsv"],
            lw=1.8, zorder=4, label=_MODEL_LABEL["bergomi_lsv"])
    ax.set_xlabel(r"Time $t$ (years)")
    ax.set_ylabel(r"Spot variance $V(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    return _save(fig, "cliquet_paths_panel.png")


# ══════════════════ §4.4.3 — Decomposition ════════════════
def _decomp_rows(d: dict):
    """Compute per-cliquet decomposition rows.

    Returns a list of (cliquet, H_baseline, H_zero, B_baseline, B_zero,
    H_contrib_pct, B_contrib_pct, HB_diff_baseline, HB_diff_zero,
    HB_contrib_pct).

    - Per-model contribution is (price_baseline - price_zero_corr) / price_baseline
      — the fraction of each model's price that the spot-variance correlation
      accounts for (with sign).
    - The H-B contribution mirrors the existing Wang-decomposition report:
      (HB_diff_baseline − HB_diff_zero) / HB_diff_baseline."""
    rows = []
    for c in _CLIQUETS:
        h0 = d["baseline"]["heston_options"][c]["price"]
        b0 = d["baseline"]["bergomi_options"][c]["price"]
        h1 = d["zero_corr"]["heston_options"][c]["price"]
        b1 = d["zero_corr"]["bergomi_options"][c]["price"]
        h_contrib = (h0 - h1) / h0 if h0 != 0.0 else float("nan")
        b_contrib = (b0 - b1) / b0 if b0 != 0.0 else float("nan")
        hb0 = h0 - b0
        hb1 = h1 - b1
        hb_contrib = (hb0 - hb1) / hb0 if hb0 != 0.0 else float("nan")
        rows.append({
            "cliquet": c,
            "H_baseline": h0, "H_zero": h1, "H_contrib": h_contrib,
            "B_baseline": b0, "B_zero": b1, "B_contrib": b_contrib,
            "HB_baseline_diff": hb0, "HB_zero_diff": hb1,
            "HB_contrib": hb_contrib,
        })
    return rows


def fig_decomposition_table(d: dict) -> Path:
    rows = _decomp_rows(d)
    lines = [
        "% Auto-generated by results_4_4_figures.py — do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Forward skew decomposition results. \(\mathcal{G}_H\) "
        r"and \(\mathcal{G}_B\) are forward skew contributions for "
        r"Heston-LSV and Bergomi-LSV respectively, and \(\mathcal{G}\) "
        r"is the forward skew contribution to the Heston-Bergomi price gap.}",
        r"\label{tab:cliquet_decomposition}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Heston-LSV} "
        r"& \multicolumn{3}{c}{Bergomi-LSV} "
        r"& \multicolumn{2}{c}{H$-$B gap} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9}",
        r"Cliquet & baseline & $\rho = 0$ & \(\mathcal{G}_H\) & "
        r"baseline & $\rho = 0$ & \(\mathcal{G}_B\) & "
        r"baseline & \(\mathcal{G}\) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_CLIQ_LABEL[r['cliquet']]} & "
            f"{r['H_baseline'] * 100:.3f}\\% & "
            f"{r['H_zero']     * 100:.3f}\\% & "
            f"{r['H_contrib']  * 100:+.3f}\\% & "
            f"{r['B_baseline'] * 100:.3f}\\% & "
            f"{r['B_zero']     * 100:.3f}\\% & "
            f"{r['B_contrib']  * 100:+.3f}\\% & "
            f"{r['HB_baseline_diff'] * 100:+.3f}\\% & "
            f"{r['HB_contrib']  * 100:+.3f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = OUT_DIR / "decomposition_table.tex"
    out.write_text("\n".join(lines))
    return out


def fig_decomposition_bars(d: dict) -> Path:
    """Grouped bar chart of per-model forward-skew contribution (×100%)."""
    rows = _decomp_rows(d)
    n_groups = len(rows)
    bar_w = 0.36
    x = np.arange(n_groups)

    h_contribs = np.array([r["H_contrib"] * 100.0 for r in rows])
    b_contribs = np.array([r["B_contrib"] * 100.0 for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - bar_w / 2, h_contribs, width=bar_w,
           color=_MODEL_COLOR["heston_lsv"], label=_MODEL_LABEL["heston_lsv"],
           edgecolor="white", linewidth=0.4)
    ax.bar(x + bar_w / 2, b_contribs, width=bar_w,
           color=_MODEL_COLOR["bergomi_lsv"], label=_MODEL_LABEL["bergomi_lsv"],
           edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)

    # Numeric labels on each bar.
    for xi, vh, vb in zip(x, h_contribs, b_contribs):
        ax.text(xi - bar_w / 2, vh + (1.5 if vh >= 0 else -3.5),
                f"{vh:+.1f}%", ha="center", fontsize=8)
        ax.text(xi + bar_w / 2, vb + (1.5 if vb >= 0 else -3.5),
                f"{vb:+.1f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_CLIQ_LABEL[r["cliquet"]] for r in rows])
    ax.set_ylabel("Forward-skew contribution (%)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return _save(fig, "decomposition_bars.png")


# ══════════════════════════ Main ══════════════════════════
def main():
    print(f"Output dir: {OUT_DIR.resolve()}")
    print("Loading pricing artefacts ...")
    d = load_data()

    # Quick digest.
    print("Baseline prices (heston_lsv / bergomi_lsv):")
    for c in _CLIQUETS:
        h = d["baseline"]["heston_options"][c]["price"]
        b = d["baseline"]["bergomi_options"][c]["price"]
        print(f"  {_CLIQ_LABEL[c]:18s}  H={h:.5f}   B={b:.5f}   H-B={h-b:+.5f}")
    print()

    figures = [
        ("§4.4.1 Cliquet prices table",            fig_cliquet_prices_table),
        ("§4.4.1 Cliquet price comparison bars",   fig_cliquet_price_comparison_bars),
        ("§4.4.2 Payoff CDFs (3 panels)",          fig_cliquet_payoff_cdfs),
        ("§4.4.2 Sample paths + variance panel",   fig_cliquet_paths_panel),
        ("§4.4.3 Decomposition table",             fig_decomposition_table),
        ("§4.4.3 Decomposition bar chart",         fig_decomposition_bars),
    ]

    n_ok = 0; n_fail = 0
    for label, fn in figures:
        try:
            out = fn(d)
            print(f"  OK    {label:<45}  {out.name}"); n_ok += 1
        except Exception as exc:
            print(f"  FAIL  {label:<45}  {exc!r}"); n_fail += 1

    print(f"\n{n_ok} files written, {n_fail} failed.")
    print(f"All outputs under {OUT_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliquet Pricing Explorer — Interactive Plotly Dashboard
========================================================
Single HTML file with tabbed plots for exploring cliquet pricing results:
convergence analysis, payoff distributions, sample path walkthroughs,
per-reset return diagnostics, and Heston vs Bergomi model comparisons.

Usage:
    python pricing_explorer.py

Output:
    plots/pricing_explorer.html — open in any browser, switch between tabs
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Directories
DIR_DATA   = "pricing/data"
DIR_ARRAYS = "pricing/arrays"
DIR_PLOTS  = "pricing/plots"

# Colour scheme
OPTION_COLORS = {
    "accumulator": "#1f77b4",
    "reverse_cliquet": "#d62728",
    "napoleon": "#2ca02c",
}
OPTION_LABELS = {
    "accumulator": "Accumulator",
    "reverse_cliquet": "Reverse Cliquet",
    "napoleon": "Napoleon",
}

MODEL_COLORS = {
    "heston_lsv": "#1f77b4",
    "bergomi_lsv": "#d62728",
    "pure_heston": "#ff7f0e",
    "bs": "#7f7f7f",
}
MODEL_LABELS = {
    "heston_lsv": "Heston LSV",
    "bergomi_lsv": "Bergomi LSV",
    "pure_heston": "Pure Heston",
    "bs": "Black-Scholes",
}

RESET_CMAP = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def load_data():
    """Load all saved pricing results and arrays."""
    with open(os.path.join(DIR_DATA, "pricing_results.json")) as f:
        results = json.load(f)

    data = {"results": results}

    # Heston arrays
    for name in ["accumulator", "reverse_cliquet", "napoleon"]:
        payoff_path = os.path.join(DIR_ARRAYS, f"{name}_payoffs.npy")
        if os.path.exists(payoff_path):
            data[f"{name}_payoffs"] = np.load(payoff_path)
            data[f"{name}_returns"] = np.load(
                os.path.join(DIR_ARRAYS, f"{name}_returns.npy"))
            data[f"{name}_S_resets"] = np.load(
                os.path.join(DIR_ARRAYS, f"{name}_S_resets.npy"))

    # Bergomi arrays
    for name in ["accumulator", "reverse_cliquet", "napoleon"]:
        payoff_path = os.path.join(DIR_ARRAYS, f"bergomi_{name}_payoffs.npy")
        if os.path.exists(payoff_path):
            data[f"bergomi_{name}_payoffs"] = np.load(payoff_path)
            data[f"bergomi_{name}_returns"] = np.load(
                os.path.join(DIR_ARRAYS, f"bergomi_{name}_returns.npy"))
            data[f"bergomi_{name}_S_resets"] = np.load(
                os.path.join(DIR_ARRAYS, f"bergomi_{name}_S_resets.npy"))

    # Heston sample paths
    sp = os.path.join(DIR_ARRAYS, "sample_paths_S.npy")
    if os.path.exists(sp):
        data["sample_S"] = np.load(sp)
        data["sample_V"] = np.load(os.path.join(DIR_ARRAYS, "sample_paths_V.npy"))
        data["time_grid"] = np.load(os.path.join(DIR_ARRAYS, "sim_time_grid.npy"))
        data["reset_dates"] = np.load(os.path.join(DIR_ARRAYS, "reset_dates.npy"))
        data["reset_indices"] = np.load(os.path.join(DIR_ARRAYS, "reset_indices.npy"))

    # Bergomi sample paths
    bsp = os.path.join(DIR_ARRAYS, "bergomi_sample_paths_S.npy")
    if os.path.exists(bsp):
        data["bergomi_sample_S"] = np.load(bsp)
        data["bergomi_sample_V"] = np.load(
            os.path.join(DIR_ARRAYS, "bergomi_sample_paths_V.npy"))
        data["bergomi_time_grid"] = np.load(
            os.path.join(DIR_ARRAYS, "bergomi_sim_time_grid.npy"))
        data["bergomi_reset_indices"] = np.load(
            os.path.join(DIR_ARRAYS, "bergomi_reset_indices.npy"))

    return data


def _has_bergomi(data):
    return "bergomi_options" in data["results"] and len(data["results"]["bergomi_options"]) > 0


def _has_heston(data):
    return "heston_options" in data["results"] and len(data["results"]["heston_options"]) > 0


# Tab 1: Summary Table

def make_summary_table(data):
    """Comparison table: Heston LSV vs Bergomi LSV vs baselines."""
    results = data["results"]
    h_opts = results.get("heston_options", {})
    b_opts = results.get("bergomi_options", {})
    all_names = sorted(set(list(h_opts.keys()) + list(b_opts.keys())))

    if not all_names:
        return go.Figure()

    headers = [
        "Option", "Heston LSV", "H SE",
        "Bergomi LSV", "B SE",
        "Pure Heston", "BS",
        "H-B Diff", "H % Zero", "B % Zero",
    ]
    rows = []
    for name in all_names:
        h = h_opts.get(name, {})
        b = b_opts.get(name, {})
        h_price = h.get("price", float("nan"))
        b_price = b.get("price", float("nan"))
        pure_h = h.get("heston_price", b.get("heston_price", float("nan")))
        bs = h.get("bs_price", b.get("bs_price", float("nan")))
        diff = h_price - b_price

        rows.append([
            OPTION_LABELS.get(name, name),
            f"{h_price:.6f}" if h else "—",
            f"{h.get('se', 0):.6f}" if h else "—",
            f"{b_price:.6f}" if b else "—",
            f"{b.get('se', 0):.6f}" if b else "—",
            f"{pure_h:.6f}",
            f"{bs:.6f}",
            f"{diff:+.6f}" if h and b else "—",
            f"{h.get('pct_zero', 0):.1f}%" if h else "—",
            f"{b.get('pct_zero', 0):.1f}%" if b else "—",
        ])

    cell_vals = [[r[i] for r in rows] for i in range(len(headers))]

    fig = go.Figure(go.Table(
        header=dict(
            values=headers,
            fill_color="#4472C4", font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[["#f0f4ff", "white"] * 2][:len(rows)],
            font=dict(size=12), align="center", height=28,
        ),
    ))

    S0 = results.get("S0", 0)
    fig.update_layout(
        title=(f"Cliquet Pricing Summary  |  S₀={S0:,.0f}  "
               f"r={results.get('r', 0):.4f}  q={results.get('q', 0):.4f}  "
               f"paths={results.get('n_paths', 0):,}"),
        height=350,
    )
    return fig


# Tab 2: Model Comparison Bar Chart

def make_price_comparison_bars(data):
    """Grouped bar chart: 4 models x 3 options."""
    results = data["results"]
    h_opts = results.get("heston_options", {})
    b_opts = results.get("bergomi_options", {})
    all_names = sorted(set(list(h_opts.keys()) + list(b_opts.keys())))

    if not all_names:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Prices by Model", "Price Differences from BS"])

    labels = [OPTION_LABELS.get(n, n) for n in all_names]

    # Prices
    for model_key, get_fn in [
        ("heston_lsv", lambda n: h_opts.get(n, {}).get("price")),
        ("bergomi_lsv", lambda n: b_opts.get(n, {}).get("price")),
        ("pure_heston", lambda n: (h_opts.get(n, {}).get("heston_price") or
                                    b_opts.get(n, {}).get("heston_price"))),
        ("bs", lambda n: (h_opts.get(n, {}).get("bs_price") or
                          b_opts.get(n, {}).get("bs_price"))),
    ]:
        vals = [get_fn(n) for n in all_names]
        if any(v is not None for v in vals):
            ses = []
            for n in all_names:
                if model_key == "heston_lsv":
                    ses.append(h_opts.get(n, {}).get("se", 0))
                elif model_key == "bergomi_lsv":
                    ses.append(b_opts.get(n, {}).get("se", 0))
                elif model_key == "pure_heston":
                    ses.append(h_opts.get(n, {}).get("heston_se",
                               b_opts.get(n, {}).get("heston_se", 0)))
                else:
                    ses.append(h_opts.get(n, {}).get("bs_se",
                               b_opts.get(n, {}).get("bs_se", 0)))

            fig.add_trace(go.Bar(
                x=labels,
                y=[v if v is not None else 0 for v in vals],
                error_y=dict(type="data",
                             array=[1.96 * s for s in ses],
                             visible=True),
                name=MODEL_LABELS[model_key],
                marker_color=MODEL_COLORS[model_key],
                hovertemplate="%{x}<br>Price: %{y:.6f}<extra></extra>",
            ), row=1, col=1)

    # Differences from BS
    for model_key, get_price, get_bs in [
        ("heston_lsv",
         lambda n: h_opts.get(n, {}).get("price"),
         lambda n: h_opts.get(n, {}).get("bs_price")),
        ("bergomi_lsv",
         lambda n: b_opts.get(n, {}).get("price"),
         lambda n: b_opts.get(n, {}).get("bs_price")),
        ("pure_heston",
         lambda n: h_opts.get(n, {}).get("heston_price", b_opts.get(n, {}).get("heston_price")),
         lambda n: h_opts.get(n, {}).get("bs_price", b_opts.get(n, {}).get("bs_price"))),
    ]:
        diffs = []
        for n in all_names:
            p = get_price(n)
            bs = get_bs(n)
            diffs.append((p - bs) if p is not None and bs is not None else 0)
        fig.add_trace(go.Bar(
            x=labels, y=diffs,
            name=MODEL_LABELS[model_key],
            marker_color=MODEL_COLORS[model_key],
            showlegend=False,
            hovertemplate="%{x}<br>Diff from BS: %{y:+.6f}<extra></extra>",
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=2)
    fig.update_xaxes(title_text="Option", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(title_text="Option", row=1, col=2)
    fig.update_yaxes(title_text="Price - BS", row=1, col=2)
    fig.update_layout(title="Model Price Comparison", barmode="group")
    return fig


# Tab 3: Payoff Distribution Comparison

def make_payoff_comparison(data):
    """Overlaid histograms: Heston vs Bergomi payoffs per option."""
    results = data["results"]
    all_names = sorted(set(
        list(results.get("heston_options", {}).keys()) +
        list(results.get("bergomi_options", {}).keys())
    ))
    n_opts = len(all_names)
    if n_opts == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=n_opts,
                        subplot_titles=[OPTION_LABELS.get(n, n) for n in all_names])

    for col, name in enumerate(all_names, 1):
        h_payoffs = data.get(f"{name}_payoffs")
        b_payoffs = data.get(f"bergomi_{name}_payoffs")

        if h_payoffs is not None:
            nonzero = h_payoffs[h_payoffs > 0]
            fig.add_trace(go.Histogram(
                x=nonzero, nbinsx=60,
                marker_color=MODEL_COLORS["heston_lsv"], opacity=0.5,
                name="Heston LSV" if col == 1 else None,
                showlegend=(col == 1),
                legendgroup="heston",
                hovertemplate="Payoff: %{x:.4f}<br>Count: %{y}<extra></extra>",
            ), row=1, col=col)

        if b_payoffs is not None:
            nonzero = b_payoffs[b_payoffs > 0]
            fig.add_trace(go.Histogram(
                x=nonzero, nbinsx=60,
                marker_color=MODEL_COLORS["bergomi_lsv"], opacity=0.5,
                name="Bergomi LSV" if col == 1 else None,
                showlegend=(col == 1),
                legendgroup="bergomi",
                hovertemplate="Payoff: %{x:.4f}<br>Count: %{y}<extra></extra>",
            ), row=1, col=col)

        fig.update_xaxes(title_text="Payoff", row=1, col=col)
        fig.update_yaxes(title_text="Count", row=1, col=col)

    fig.update_layout(
        title="Payoff Distributions: Heston LSV vs Bergomi LSV",
        barmode="overlay",
    )
    return fig


# Tab 4: Return Distribution Comparison

def make_return_comparison(data):
    """Side-by-side box plots: Heston vs Bergomi per-reset returns."""
    results = data["results"]
    all_names = sorted(set(
        list(results.get("heston_options", {}).keys()) +
        list(results.get("bergomi_options", {}).keys())
    ))
    n_opts = len(all_names)
    if n_opts == 0:
        return go.Figure()

    fig = make_subplots(rows=2, cols=n_opts,
                        subplot_titles=(
                            [f"{OPTION_LABELS.get(n, n)} — Heston" for n in all_names] +
                            [f"{OPTION_LABELS.get(n, n)} — Bergomi" for n in all_names]
                        ))

    for col, name in enumerate(all_names, 1):
        for row, prefix in [(1, ""), (2, "bergomi_")]:
            returns = data.get(f"{prefix}{name}_returns")
            if returns is None:
                continue
            n_resets = returns.shape[1]
            for i in range(n_resets):
                col_data = returns[:, i] * 100
                std_val = float(np.std(col_data))
                q1, med, q3 = np.percentile(col_data, [25, 50, 75])
                iqr = q3 - q1
                wlo = col_data[col_data >= q1 - 1.5 * iqr].min() if len(col_data[col_data >= q1 - 1.5 * iqr]) > 0 else q1
                whi = col_data[col_data <= q3 + 1.5 * iqr].max() if len(col_data[col_data <= q3 + 1.5 * iqr]) > 0 else q3
                color = MODEL_COLORS["heston_lsv"] if row == 1 else MODEL_COLORS["bergomi_lsv"]
                fig.add_trace(go.Box(
                    y=col_data, name=f"M{i+1}",
                    marker_color=color, showlegend=False,
                    boxmean=True,
                    boxpoints="outliers", marker=dict(opacity=0),
                    hoverinfo="text",
                    hovertext=(
                        f"Reset {i+1}<br>"
                        f"Median: {med:.2f}%<br>"
                        f"Q1: {q1:.2f}%  Q3: {q3:.2f}%<br>"
                        f"Std: {std_val:.2f}%"
                    ),
                ), row=row, col=col)
            fig.update_yaxes(title_text="Return (%)", row=row, col=col)

    fig.update_layout(
        title="Per-Reset Return Distributions: Heston vs Bergomi",
        height=800,
    )
    return fig


# Tab 5: Rainbow Paths Comparison

def make_rainbow_comparison(data, n_light=100):
    """Side-by-side rainbow paths: Heston vs Bergomi."""
    has_h = "sample_S" in data
    has_b = "bergomi_sample_S" in data

    cols = int(has_h) + int(has_b)
    if cols == 0:
        return go.Figure()

    titles = []
    if has_h:
        titles.append("Heston LSV Paths")
    if has_b:
        titles.append("Bergomi LSV Paths")

    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)

    def add_paths(sample_S, time_grid, reset_indices, reset_dates, col_idx, color_base):
        n_avail = sample_S.shape[0]
        n_show = min(n_light, n_avail)

        for i in range(n_show):
            fig.add_trace(go.Scatter(
                x=time_grid, y=sample_S[i, :].astype(np.float64),
                mode="lines", line=dict(color=f"rgba({color_base},0.08)", width=0.8),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col_idx)

        terminals = sample_S[:n_avail, -1].astype(np.float64)
        bold_idx = [np.argmin(terminals), np.argmax(terminals),
                    np.argmin(np.abs(terminals - np.median(terminals)))]
        bold_colors = ["#d62728", "#2ca02c", "#1f77b4"]

        for j, idx in enumerate(bold_idx):
            fig.add_trace(go.Scatter(
                x=time_grid, y=sample_S[idx, :].astype(np.float64),
                mode="lines",
                line=dict(color=bold_colors[j], width=2.5),
                name=f"S_T={sample_S[idx, -1]:.0f}",
                showlegend=(col_idx == 1),
                legendgroup=f"bold_{j}",
                hovertemplate="t=%{x:.3f}<br>S=%{y:.1f}<extra></extra>",
            ), row=1, col=col_idx)

            if reset_dates is not None:
                fig.add_trace(go.Scatter(
                    x=reset_dates, y=sample_S[idx, reset_indices].astype(np.float64),
                    mode="markers",
                    marker=dict(size=6, color=bold_colors[j], symbol="diamond"),
                    showlegend=False,
                    hovertemplate="Reset<br>t=%{x:.3f}<br>S=%{y:.1f}<extra></extra>",
                ), row=1, col=col_idx)

    ci = 1
    if has_h:
        reset_dates = data.get("reset_dates")
        add_paths(data["sample_S"], data["time_grid"],
                  data["reset_indices"], reset_dates, ci, "100,149,237")
        fig.update_xaxes(title_text="Time (years)", row=1, col=ci)
        fig.update_yaxes(title_text="Spot Price", row=1, col=ci)
        ci += 1

    if has_b:
        reset_dates = data.get("reset_dates", None)
        add_paths(data["bergomi_sample_S"], data["bergomi_time_grid"],
                  data["bergomi_reset_indices"], reset_dates, ci, "214,39,40")
        fig.update_xaxes(title_text="Time (years)", row=1, col=ci)
        fig.update_yaxes(title_text="Spot Price", row=1, col=ci)

    fig.update_layout(title="Sample Paths: Heston LSV vs Bergomi LSV")
    return fig


# Tab 6: Variance Path Comparison

def make_variance_comparison(data, n_show=30):
    """Side-by-side variance paths: Heston vs Bergomi."""
    has_h = "sample_V" in data
    has_b = "bergomi_sample_V" in data
    cols = int(has_h) + int(has_b)
    if cols == 0:
        return go.Figure()

    titles = []
    if has_h:
        titles.append("Heston Variance Paths")
    if has_b:
        titles.append("Bergomi Spot Variance Paths")

    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)

    def add_var_paths(sample_V, time_grid, col_idx, color):
        n = min(n_show, sample_V.shape[0])
        for i in range(n):
            fig.add_trace(go.Scatter(
                x=time_grid, y=sample_V[i, :].astype(np.float64),
                mode="lines", line=dict(color=color, width=0.8),
                opacity=0.3, showlegend=False, hoverinfo="skip",
            ), row=1, col=col_idx)
        # Mean
        mean_v = sample_V[:n, :].astype(np.float64).mean(axis=0)
        fig.add_trace(go.Scatter(
            x=time_grid, y=mean_v,
            mode="lines", line=dict(color="black", width=2.5),
            name="Mean" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
            hovertemplate="t=%{x:.3f}<br>V=%{y:.6f}<extra></extra>",
        ), row=1, col=col_idx)

    ci = 1
    if has_h:
        add_var_paths(data["sample_V"], data["time_grid"], ci,
                      MODEL_COLORS["heston_lsv"])
        fig.update_xaxes(title_text="Time (years)", row=1, col=ci)
        fig.update_yaxes(title_text="Variance", row=1, col=ci)
        ci += 1
    if has_b:
        add_var_paths(data["bergomi_sample_V"], data["bergomi_time_grid"], ci,
                      MODEL_COLORS["bergomi_lsv"])
        fig.update_xaxes(title_text="Time (years)", row=1, col=ci)
        fig.update_yaxes(title_text="Spot Variance ξ", row=1, col=ci)

    fig.update_layout(title="Variance Paths: Heston CIR vs Bergomi Two-Factor")
    return fig


# Tab 7: Convergence

def make_convergence(data):
    """Price convergence for both models."""
    results = data["results"]
    h_opts = results.get("heston_options", results.get("options", {}))
    b_opts = results.get("bergomi_options", {})
    all_names = sorted(set(list(h_opts.keys()) + list(b_opts.keys())))
    n_opts = len(all_names)
    if n_opts == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=n_opts,
                        subplot_titles=[OPTION_LABELS.get(n, n) for n in all_names])

    for col, name in enumerate(all_names, 1):
        for model_key, prefix, opts in [
            ("heston_lsv", "", h_opts),
            ("bergomi_lsv", "bergomi_", b_opts),
        ]:
            d = opts.get(name)
            payoffs = data.get(f"{prefix}{name}_payoffs")
            if d is None or payoffs is None:
                continue

            df = d["discount_factor"]
            n_total = len(payoffs)
            counts = np.unique(np.geomspace(100, n_total, 80).astype(int))
            counts = counts[counts <= n_total]

            cum_sum = np.cumsum(payoffs * df)
            means = cum_sum[counts - 1] / counts
            cum_sum_sq = np.cumsum((payoffs * df)**2)
            variances = cum_sum_sq[counts - 1] / counts - means**2
            ses = np.sqrt(np.maximum(variances, 0) / counts)

            color = MODEL_COLORS[model_key]
            fig.add_trace(go.Scatter(
                x=counts, y=means,
                mode="lines", name=MODEL_LABELS[model_key] if col == 1 else None,
                showlegend=(col == 1),
                legendgroup=model_key,
                line=dict(color=color, width=2),
                hovertemplate="N=%{x:,}<br>Price=%{y:.6f}<extra></extra>",
            ), row=1, col=col)

            fig.add_trace(go.Scatter(
                x=np.concatenate([counts, counts[::-1]]),
                y=np.concatenate([means + 1.96 * ses, (means - 1.96 * ses)[::-1]]),
                fill="toself",
                fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba")
                if "rgb" in color else f"rgba(100,100,100,0.12)",
                line=dict(width=0), showlegend=False,
                hoverinfo="skip", legendgroup=model_key,
            ), row=1, col=col)

        fig.update_xaxes(title_text="Number of Paths", type="log", row=1, col=col)
        fig.update_yaxes(title_text="Price Estimate", row=1, col=col)

    fig.update_layout(title="Price Convergence: Heston LSV vs Bergomi LSV")
    return fig


# Tab 8: Payoff Distributions (Heston only)

def make_payoff_distributions(data):
    """Histogram of per-path payoffs for each option (Heston LSV)."""
    results = data["results"]
    opts = results.get("heston_options", results.get("options", {}))
    names = list(opts.keys())
    if not names:
        return go.Figure()

    fig = make_subplots(rows=1, cols=len(names),
                        subplot_titles=[OPTION_LABELS.get(n, n) for n in names])

    for col, name in enumerate(names, 1):
        payoffs = data.get(f"{name}_payoffs")
        if payoffs is None:
            continue
        d = opts[name]
        color = OPTION_COLORS.get(name, "#333")
        nonzero = payoffs[payoffs > 0]

        fig.add_trace(go.Histogram(
            x=nonzero, nbinsx=60,
            marker_color=color, opacity=0.7,
            name=OPTION_LABELS.get(name, name),
            hovertemplate="Payoff: %{x:.4f}<br>Count: %{y}<extra></extra>",
        ), row=1, col=col)

        fig.add_vline(x=d["mean_payoff"], line_color="black", line_dash="dash",
                      annotation_text=f"Mean={d['mean_payoff']:.4f}",
                      annotation_font_size=10, row=1, col=col)

        pct_zero = d["pct_zero"]
        fig.add_annotation(
            text=f"{pct_zero:.1f}% zero payoff",
            xref=f"x{'' if col == 1 else col}", yref=f"y{'' if col == 1 else col} domain",
            x=0.95, y=0.95, showarrow=False, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="grey",
            xanchor="right",
        )
        fig.update_xaxes(title_text="Payoff", row=1, col=col)
        fig.update_yaxes(title_text="Count", row=1, col=col)

    fig.update_layout(title="Per-Path Payoff Distributions — Heston LSV",
                      showlegend=False)
    return fig


# Tab 9: Per-Reset Return Distributions (Heston)

def make_reset_return_distributions(data):
    """Box plots of per-reset returns (Heston LSV)."""
    results = data["results"]
    opts = results.get("heston_options", results.get("options", {}))
    names = list(opts.keys())
    if not names:
        return go.Figure()

    fig = make_subplots(rows=1, cols=len(names),
                        subplot_titles=[OPTION_LABELS.get(n, n) for n in names])

    for col, name in enumerate(names, 1):
        returns = data.get(f"{name}_returns")
        if returns is None:
            continue
        n_resets = returns.shape[1]
        for i in range(n_resets):
            col_data = returns[:, i] * 100
            std_val = float(np.std(col_data))
            q1, med, q3 = np.percentile(col_data, [25, 50, 75])
            iqr = q3 - q1
            wlo = col_data[col_data >= q1 - 1.5 * iqr].min() if len(col_data[col_data >= q1 - 1.5 * iqr]) > 0 else q1
            whi = col_data[col_data <= q3 + 1.5 * iqr].max() if len(col_data[col_data <= q3 + 1.5 * iqr]) > 0 else q3
            fig.add_trace(go.Box(
                y=col_data, name=f"M{i+1}",
                marker_color=RESET_CMAP[i % len(RESET_CMAP)],
                showlegend=False, boxmean=True,
                boxpoints="outliers", marker=dict(opacity=0),
                hoverinfo="text",
                hovertext=(
                    f"Reset {i+1}<br>Median: {med:.2f}%<br>"
                    f"Q1: {q1:.2f}%  Q3: {q3:.2f}%<br>"
                    f"Whiskers: [{wlo:.2f}%, {whi:.2f}%]<br>Std: {std_val:.2f}%"
                ),
            ), row=1, col=col)
        fig.update_xaxes(title_text="Reset Period", row=1, col=col)
        fig.update_yaxes(title_text="Return (%)", row=1, col=col)

    fig.update_layout(title="Per-Reset Return Distributions — Heston LSV")
    return fig


# Tab 10: Sample Path Walkthroughs

def make_sample_path_walkthrough(data, name, n_show=5):
    """Walkthrough of sample paths for one option."""
    results = data["results"]
    opts = results.get("heston_options", results.get("options", {}))
    if name not in opts:
        return go.Figure()

    d = opts[name]
    S0 = results["S0"]
    sample_S = data["sample_S"].astype(np.float64)
    time_grid = data["time_grid"]
    reset_indices = data["reset_indices"]
    reset_dates = data["reset_dates"]
    returns_all = data.get(f"{name}_returns")
    payoffs_all = data.get(f"{name}_payoffs")

    if returns_all is None:
        return go.Figure()

    n_resets = len(reset_dates)
    n_sample = min(n_show, sample_S.shape[0], len(payoffs_all))
    quantiles = np.linspace(0.1, 0.9, n_show)
    sorted_idx = np.argsort(payoffs_all[:sample_S.shape[0]])
    path_indices = sorted_idx[(quantiles * len(sorted_idx)).astype(int)]
    path_indices = path_indices[:n_sample]

    fig = make_subplots(
        rows=n_sample, cols=3,
        subplot_titles=[f"Path {j+1} — Spot" if c == 0
                        else f"Path {j+1} — Returns" if c == 1
                        else f"Path {j+1} — Running Agg"
                        for j in range(n_sample) for c in range(3)],
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )

    for row_idx, pi in enumerate(path_indices, 1):
        S_path = sample_S[pi, :]
        fig.add_trace(go.Scatter(
            x=time_grid, y=S_path, mode="lines",
            line=dict(color="#1f77b4", width=1.5), showlegend=False,
            hovertemplate="t=%{x:.3f}<br>S=%{y:.1f}<extra></extra>",
        ), row=row_idx, col=1)
        S_at_resets = S_path[reset_indices]
        fig.add_trace(go.Scatter(
            x=reset_dates, y=S_at_resets, mode="markers",
            marker=dict(size=8, color="red", symbol="diamond"),
            showlegend=False,
            hovertemplate="Reset %{text}<br>t=%{x:.3f}<br>S=%{y:.1f}<extra></extra>",
            text=[str(i+1) for i in range(n_resets)],
        ), row=row_idx, col=1)

        rets = returns_all[pi, :]
        if name == "accumulator":
            cap, floor = d["payoff_kwargs"]["cap"], d["payoff_kwargs"]["floor"]
            processed = np.maximum(np.minimum(rets, cap), floor)
            running = np.cumsum(processed)
        elif name == "reverse_cliquet":
            coupon = d["payoff_kwargs"]["coupon"]
            processed = np.minimum(rets, 0.0)
            running = coupon + np.cumsum(processed)
        elif name == "napoleon":
            coupon = d["payoff_kwargs"]["coupon"]
            processed = rets
            running = coupon + np.array([rets[:i+1].min() for i in range(n_resets)])

        fig.add_trace(go.Bar(
            x=list(range(1, n_resets + 1)), y=rets * 100,
            marker_color="rgba(100,100,100,0.4)",
            name="Raw", showlegend=False,
            hovertemplate="M%{x}: raw=%{y:.2f}%<extra></extra>",
        ), row=row_idx, col=2)
        fig.add_trace(go.Bar(
            x=list(range(1, n_resets + 1)), y=processed * 100,
            marker_color=OPTION_COLORS.get(name, "#333"),
            name="Processed", showlegend=False, opacity=0.7,
            hovertemplate="M%{x}: processed=%{y:.2f}%<extra></extra>",
        ), row=row_idx, col=2)

        fig.add_trace(go.Scatter(
            x=list(range(1, n_resets + 1)), y=running * 100,
            mode="lines+markers",
            line=dict(color=OPTION_COLORS.get(name, "#333"), width=2),
            marker=dict(size=6), showlegend=False,
            hovertemplate="M%{x}: running=%{y:.2f}%<extra></extra>",
        ), row=row_idx, col=3)
        fig.add_hline(y=0, line_dash="dash", line_color="grey",
                      row=row_idx, col=3)

        final_payoff = payoffs_all[pi]
        ax = "" if row_idx == 1 else str((row_idx - 1) * 3 + 3)
        fig.add_annotation(
            text=f"Payoff: {final_payoff:.4f}",
            xref=f"x{ax if ax else 3}", yref=f"y{ax if ax else 3} domain",
            x=0.5, y=1.05, showarrow=False, font=dict(size=10, color="black"),
            bgcolor="lightyellow", bordercolor="grey",
        )

    label = OPTION_LABELS.get(name, name)
    fig.update_layout(
        title=f"{label} — Sample Path Walkthroughs",
        height=250 * n_sample + 100, barmode="overlay",
    )
    return fig


# Tab 11: Napoleon Worst-Reset Analysis

def make_napoleon_worst_reset(data):
    """Which reset period has the worst return for Napoleon."""
    results = data["results"]
    h_opts = results.get("heston_options", results.get("options", {}))
    b_opts = results.get("bergomi_options", {})

    has_h = data.get("napoleon_returns") is not None
    has_b = data.get("bergomi_napoleon_returns") is not None
    cols = int(has_h) + int(has_b)
    if cols == 0:
        return go.Figure()

    titles = []
    if has_h:
        titles.extend(["Heston — Worst Reset", "Heston — Worst Return vs Payoff"])
    if has_b:
        titles.extend(["Bergomi — Worst Reset", "Bergomi — Worst Return vs Payoff"])

    fig = make_subplots(rows=1, cols=cols * 2, subplot_titles=titles)

    ci = 1
    for prefix, opts, label in [
        ("", h_opts, "Heston"),
        ("bergomi_", b_opts, "Bergomi"),
    ]:
        returns = data.get(f"{prefix}napoleon_returns")
        payoffs = data.get(f"{prefix}napoleon_payoffs")
        if returns is None or payoffs is None:
            continue

        n_resets = returns.shape[1]
        worst_idx = np.argmin(returns, axis=1)
        worst_vals = returns[np.arange(len(returns)), worst_idx]

        counts = np.bincount(worst_idx, minlength=n_resets)
        fig.add_trace(go.Bar(
            x=[f"M{i+1}" for i in range(n_resets)], y=counts,
            marker_color=RESET_CMAP[:n_resets], showlegend=False,
            hovertemplate="Reset %{x}<br>Count: %{y:,}<extra></extra>",
        ), row=1, col=ci)
        fig.update_xaxes(title_text="Reset Period", row=1, col=ci)
        fig.update_yaxes(title_text="Count", row=1, col=ci)

        n_show = min(5000, len(worst_vals))
        idx = np.random.default_rng(0).choice(len(worst_vals), n_show, replace=False)
        fig.add_trace(go.Scatter(
            x=worst_vals[idx] * 100, y=payoffs[idx],
            mode="markers",
            marker=dict(size=3, color=worst_idx[idx], colorscale="Viridis",
                        opacity=0.5),
            showlegend=False,
            hovertemplate="Worst: %{x:.2f}%<br>Payoff: %{y:.4f}<extra></extra>",
        ), row=1, col=ci + 1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=ci + 1)

        coupon = opts.get("napoleon", {}).get("payoff_kwargs", {}).get("coupon", 0.08)
        fig.add_vline(x=-coupon * 100, line_dash="dash", line_color="red",
                      annotation_text=f"{coupon*100:.0f}%",
                      annotation_font_size=10, row=1, col=ci + 1)
        fig.update_xaxes(title_text="Worst Return (%)", row=1, col=ci + 1)
        fig.update_yaxes(title_text="Payoff", row=1, col=ci + 1)
        ci += 2

    fig.update_layout(title="Napoleon — Worst-Reset Analysis: Heston vs Bergomi")
    return fig


# Tab 12: Reverse Cliquet Coupon Consumption

def make_reverse_consumption(data):
    """Coupon consumption analysis for reverse cliquet."""
    results = data["results"]
    h_opts = results.get("heston_options", results.get("options", {}))
    b_opts = results.get("bergomi_options", {})

    has_h = data.get("reverse_cliquet_returns") is not None
    has_b = data.get("bergomi_reverse_cliquet_returns") is not None
    cols = int(has_h) + int(has_b)
    if cols == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=cols,
                        subplot_titles=(
                            (["Heston — Coupon Remaining"] if has_h else []) +
                            (["Bergomi — Coupon Remaining"] if has_b else [])
                        ))

    ci = 1
    for prefix, opts, color in [
        ("", h_opts, MODEL_COLORS["heston_lsv"]),
        ("bergomi_", b_opts, MODEL_COLORS["bergomi_lsv"]),
    ]:
        returns = data.get(f"{prefix}reverse_cliquet_returns")
        if returns is None:
            continue

        coupon = opts.get("reverse_cliquet", {}).get("payoff_kwargs", {}).get("coupon", 0.15)
        neg_sum = np.minimum(returns, 0.0).sum(axis=1)
        remaining = coupon + neg_sum

        fig.add_trace(go.Histogram(
            x=remaining * 100, nbinsx=80,
            marker_color=color, opacity=0.7,
            showlegend=False,
            hovertemplate="Remaining: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ), row=1, col=ci)

        fig.add_vline(x=0, line_dash="dash", line_color="black",
                      annotation_text="Zero payoff", annotation_font_size=10,
                      row=1, col=ci)
        fig.add_vline(x=coupon * 100, line_dash="dot", line_color="grey",
                      annotation_text=f"Full ({coupon*100:.0f}%)",
                      annotation_font_size=10, row=1, col=ci)

        pct_fully = (remaining <= 0).mean() * 100
        pct_partial = ((remaining > 0) & (remaining < coupon)).mean() * 100
        pct_full = (remaining >= coupon).mean() * 100
        fig.add_annotation(
            text=(f"Consumed: {pct_fully:.1f}%<br>"
                  f"Partial: {pct_partial:.1f}%<br>"
                  f"Intact: {pct_full:.1f}%"),
            xref=f"x{'' if ci == 1 else ci}",
            yref=f"y{'' if ci == 1 else ci} domain",
            x=remaining.mean() * 100, y=0.9,
            showarrow=False, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="grey",
        )
        fig.update_xaxes(title_text="Coupon Remaining (%)", row=1, col=ci)
        fig.update_yaxes(title_text="Count", row=1, col=ci)
        ci += 1

    fig.update_layout(title="Reverse Cliquet — Coupon Consumption: Heston vs Bergomi")
    return fig


# Tab 13: Terminal Spot & Variance Distributions

def make_terminal_distributions(data):
    """Compare terminal spot and variance distributions."""
    has_h = "sample_S" in data
    has_b = "bergomi_sample_S" in data
    if not has_h and not has_b:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Terminal Spot Distribution",
                                        "Terminal Variance Distribution"])

    if has_h:
        S_term = data["sample_S"][:, -1].astype(np.float64)
        fig.add_trace(go.Histogram(
            x=S_term, nbinsx=50,
            marker_color=MODEL_COLORS["heston_lsv"], opacity=0.5,
            name="Heston", legendgroup="heston",
        ), row=1, col=1)
        V_term = data["sample_V"][:, -1].astype(np.float64)
        fig.add_trace(go.Histogram(
            x=V_term, nbinsx=50,
            marker_color=MODEL_COLORS["heston_lsv"], opacity=0.5,
            name="Heston", legendgroup="heston", showlegend=False,
        ), row=1, col=2)

    if has_b:
        S_term = data["bergomi_sample_S"][:, -1].astype(np.float64)
        fig.add_trace(go.Histogram(
            x=S_term, nbinsx=50,
            marker_color=MODEL_COLORS["bergomi_lsv"], opacity=0.5,
            name="Bergomi", legendgroup="bergomi",
        ), row=1, col=1)
        V_term = data["bergomi_sample_V"][:, -1].astype(np.float64)
        fig.add_trace(go.Histogram(
            x=V_term, nbinsx=50,
            marker_color=MODEL_COLORS["bergomi_lsv"], opacity=0.5,
            name="Bergomi", legendgroup="bergomi", showlegend=False,
        ), row=1, col=2)

    fig.update_xaxes(title_text="Spot Price", row=1, col=1)
    fig.update_xaxes(title_text="Variance", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        title="Terminal Distributions: Heston vs Bergomi",
        barmode="overlay",
    )
    return fig


# HTML Assembly

TAB_DESCRIPTIONS = {
    "Summary": (
        "Comparison of all three cliquet prices under Heston LSV and Bergomi LSV, "
        "alongside pure Heston (L=1) and Black-Scholes baselines."
    ),
    "Price Comparison": (
        "Grouped bar chart of prices across all four models. Right panel shows "
        "each model's price difference from the BS baseline."
    ),
    "Payoff Comparison": (
        "Overlaid histograms of per-path payoffs for Heston LSV vs Bergomi LSV. "
        "Differences in tail behaviour reveal how the two stochastic vol backbones "
        "affect the payoff distribution."
    ),
    "Return Comparison": (
        "Per-reset return box plots: top row Heston, bottom row Bergomi. Compare "
        "how the two models distribute monthly returns — wider boxes in Bergomi "
        "indicate higher conditional variance."
    ),
    "Rainbow Paths": (
        "Side-by-side sample spot paths under Heston LSV and Bergomi LSV. "
        "Bergomi's two-factor structure can produce different dispersion and clustering."
    ),
    "Variance Paths": (
        "Heston's CIR variance vs Bergomi's spot variance xi^t_t. The Bergomi "
        "process can exhibit faster mean-reversion and different tail behaviour "
        "driven by two OU factors."
    ),
    "Convergence": (
        "Price convergence as a function of path count, with 95% CI bands. "
        "Both models should converge smoothly; if bands are still wide, more paths are needed."
    ),
    "Heston Payoffs": (
        "Histograms of per-path payoffs under Heston LSV (excluding zero-payoff paths)."
    ),
    "Heston Returns": (
        "Box plots of per-reset-period returns under Heston LSV."
    ),
    "Accumulator Paths": (
        "Detailed walkthrough of 5 sample accumulator paths under Heston LSV."
    ),
    "Napoleon Paths": (
        "Detailed walkthrough of 5 sample napoleon paths under Heston LSV."
    ),
    "Napoleon Analysis": (
        "Worst-reset analysis for Napoleon under both models. Which reset "
        "period most frequently contains the worst return, and how the worst "
        "return maps to the final payoff."
    ),
    "Reverse Consumption": (
        "Coupon consumption for reverse cliquet under both models. Shows the "
        "distribution of remaining coupon at maturity."
    ),
    "Terminal Distributions": (
        "Terminal spot and variance distributions side-by-side. Heavier tails in "
        "one model lead to different cliquet prices due to path-dependent payoff "
        "sensitivity."
    ),
}


def build_html(figures, tab_names, descriptions):
    """Build a single HTML file with CSS tabs and separate Plotly figures."""
    fig_json_list = []
    for fig in figures:
        fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        fig_json_list.append(fig.to_json())

    tab_buttons = []
    for i, name in enumerate(tab_names):
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab-btn{active}" onclick="switchTab({i})">{name}</button>'
        )

    tab_contents = []
    for i, name in enumerate(tab_names):
        display = "block" if i == 0 else "none"
        desc = descriptions.get(name, "")
        tab_contents.append(
            f'<div class="tab-content" id="tab-{i}" style="display:{display}">'
            f'<p class="tab-desc">{desc}</p>'
            f'<div id="plot-{i}" style="width:100%;height:650px;"></div>'
            f'</div>'
        )

    fig_specs_js = ",\n".join(fig_json_list)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cliquet Pricing Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         "Helvetica Neue", Arial, sans-serif; background: #fafafa; color: #333;
         padding: 20px 30px; }
  h1 { font-size: 22px; margin-bottom: 16px; font-weight: 600; }
  .tab-bar { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px;
              border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }
  .tab-btn { padding: 8px 16px; border: 1px solid #ccc; border-radius: 6px 6px 0 0;
              background: #f0f0f0; cursor: pointer; font-size: 13px; font-weight: 500;
              transition: all 0.15s; }
  .tab-btn:hover { background: #e0e0e0; }
  .tab-btn.active { background: white; border-bottom: 2px solid white;
                     margin-bottom: -2px; color: #1a73e8; font-weight: 600; }
  .tab-desc { font-size: 14px; line-height: 1.5; color: #555; margin: 8px 0 12px 0;
               max-width: 900px; }
  .tab-content { background: white; border-radius: 0 0 8px 8px; padding: 12px; }
</style>
</head>
<body>
<h1>Cliquet Pricing Explorer — Heston LSV vs Bergomi LSV</h1>
"""
    html += '<div class="tab-bar">\n  ' + ''.join(tab_buttons) + '\n</div>\n'
    html += ''.join(tab_contents)

    html += """
<script>
var figSpecs = [""" + fig_specs_js + """];
var rendered = {};

function renderPlot(idx) {
  if (rendered[idx]) {
    Plotly.Plots.resize(document.getElementById('plot-' + idx));
    return;
  }
  var spec = figSpecs[idx];
  Plotly.newPlot('plot-' + idx, spec.data, spec.layout, {responsive: true});
  rendered[idx] = true;
}

function switchTab(idx) {
  document.querySelectorAll('.tab-content').forEach(function(el) { el.style.display = 'none'; });
  document.querySelectorAll('.tab-btn').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById('tab-' + idx).style.display = 'block';
  document.querySelectorAll('.tab-btn')[idx].classList.add('active');
  renderPlot(idx);
}

renderPlot(0);
</script>
</body>
</html>"""

    return html


def main():
    data = load_data()

    tab_names = [
        "Summary",
        "Price Comparison",
        "Payoff Comparison",
        "Return Comparison",
        "Rainbow Paths",
        "Variance Paths",
        "Convergence",
        "Heston Payoffs",
        "Heston Returns",
        "Accumulator Paths",
        "Napoleon Paths",
        "Napoleon Analysis",
        "Reverse Consumption",
        "Terminal Distributions",
    ]

    figures = [
        make_summary_table(data),
        make_price_comparison_bars(data),
        make_payoff_comparison(data),
        make_return_comparison(data),
        make_rainbow_comparison(data),
        make_variance_comparison(data),
        make_convergence(data),
        make_payoff_distributions(data),
        make_reset_return_distributions(data),
        make_sample_path_walkthrough(data, "accumulator"),
        make_sample_path_walkthrough(data, "napoleon"),
        make_napoleon_worst_reset(data),
        make_reverse_consumption(data),
        make_terminal_distributions(data),
    ]

    html = build_html(figures, tab_names, TAB_DESCRIPTIONS)
    os.makedirs(DIR_PLOTS, exist_ok=True)
    out_path = os.path.join(DIR_PLOTS, "pricing_explorer.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path} — open in your browser")


if __name__ == "__main__":
    main()

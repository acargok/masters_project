#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSV explorer: interactive Plotly dashboard.

Single tabbed HTML for the LSV outputs (leverage surface, Heston calibration,
repricing errors vs market and vs Dupire).

Usage:  python lsv_explorer.py
Output: plots/lsv_explorer.html
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Directories
DIR_DATA   = "lsv_heston/data"
DIR_PLOTS  = "lsv_heston/plots"
DIR_ARRAYS = "lsv_heston/arrays"
IV_DIR_ARRAYS = "iv_surface/arrays"
DUPIRE_DIR_ARRAYS = "dupire_vol/arrays"
DUPIRE_DIR_DATA = "dupire_vol/data"

# 4-category colour scheme
CAT_COLORS = {
    "OTM Call": "#1f77b4",
    "ITM Call": "#7fbfff",
    "OTM Put":  "#d62728",
    "ITM Put":  "#ff9896",
}
CAT_SYMBOLS = {
    "OTM Call": "circle",
    "ITM Call": "square",
    "OTM Put":  "diamond",
    "ITM Put":  "triangle-up",
}


def load_data():
    """Load all pre-computed data."""
    df = pd.read_csv(os.path.join(DIR_DATA, "lsv_repricing_errors.csv"))

    leverage = np.load(os.path.join(DIR_ARRAYS, "leverage_surface.npy"))
    spot_grid = np.load(os.path.join(DIR_ARRAYS, "leverage_spot_grid.npy"))
    time_grid = np.load(os.path.join(DIR_ARRAYS, "leverage_time_grid.npy"))

    with open(os.path.join(DIR_DATA, "heston_params.json")) as f:
        heston = json.load(f)

    with open(os.path.join(DIR_DATA, "particle_log.json")) as f:
        particle_log = json.load(f)

    with open(os.path.join(DIR_DATA, "validation_summary.json")) as f:
        val_summary = json.load(f)

    with open(os.path.join(DUPIRE_DIR_DATA, "market_params.json")) as f:
        market = json.load(f)

    # IV surface + Dupire local vol for context
    iv_surface = np.load(os.path.join(IV_DIR_ARRAYS, "iv_surface.npy"))
    log_m_grid = np.load(os.path.join(IV_DIR_ARRAYS, "log_m_grid.npy"))
    ttm_grid = np.load(os.path.join(IV_DIR_ARRAYS, "ttm_grid.npy"))
    local_vol = np.load(os.path.join(DUPIRE_DIR_ARRAYS, "local_vol_surface.npy"))

    return (df, leverage, spot_grid, time_grid, heston, particle_log,
            val_summary, market, iv_surface, log_m_grid, ttm_grid, local_vol)


def classify_options(df):
    """Classify into OTM Call, ITM Call, OTM Put, ITM Put."""
    cat = pd.Series("", index=df.index)
    is_call = df["option_type"] == "call"
    is_otm = ((is_call) & (df["moneyness"] >= 1.0)) | ((~is_call) & (df["moneyness"] < 1.0))
    cat[is_call & is_otm]  = "OTM Call"
    cat[is_call & ~is_otm] = "ITM Call"
    cat[~is_call & is_otm] = "OTM Put"
    cat[~is_call & ~is_otm] = "ITM Put"
    return cat


def _add_cat_traces(fig, df, x_col, y_col, customdata_cols=None,
                    hovertemplate=None, row=None, col=None, size=5, opacity=0.6,
                    showlegend=True):
    """Add one scatter trace per category to a figure."""
    cats = classify_options(df)
    for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
        sub = df[cats == label]
        if len(sub) == 0:
            continue
        kwargs = dict(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            marker=dict(size=size, color=CAT_COLORS[label],
                        symbol=CAT_SYMBOLS[label], opacity=opacity),
            name=f"{label} ({len(sub)})",
            legendgroup=label,
            showlegend=showlegend,
        )
        if customdata_cols is not None:
            kwargs["customdata"] = sub[customdata_cols].values
        if hovertemplate is not None:
            kwargs["hovertemplate"] = hovertemplate
        trace = go.Scatter(**kwargs)
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


# Tab 1: 3D leverage surface

def make_leverage_surface(leverage, spot_grid, time_grid, S):
    log_spot = np.log(spot_grid / S)
    fig = go.Figure(go.Surface(
        x=log_spot, y=time_grid, z=leverage.T,
        colorscale="Inferno", colorbar_title="L(t,S)",
        contours=dict(
            x=dict(show=True, color="rgba(0,0,0,0.12)", width=1),
            y=dict(show=True, color="rgba(0,0,0,0.12)", width=1),
        ),
        hovertemplate=(
            "ln(S/S₀): %{x:.4f}<br>"
            "t: %{y:.3f} yr<br>"
            "L(t,S): %{z:.4f}<extra></extra>"
        ),
    ))
    x_range = log_spot.max() - log_spot.min()
    y_range = time_grid.max() - time_grid.min()
    z_range = leverage.max() - leverage.min()
    mx = max(x_range, y_range)
    fig.update_layout(
        title="Leverage Function L(t, S)",
        scene=dict(
            xaxis_title="ln(S/S₀)",
            yaxis_title="Time (years)",
            zaxis_title="L(t, S)",
            aspectmode="manual",
            aspectratio=dict(x=x_range/mx, y=y_range/mx, z=z_range/mx*1.5),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# Tab 2: leverage slices by time

def make_leverage_slices(leverage, spot_grid, time_grid, S):
    log_spot = np.log(spot_grid / S)
    n_slices = 10
    indices = np.linspace(0, len(time_grid) - 1, n_slices, dtype=int)

    fig = go.Figure()
    for idx in indices:
        t = time_grid[idx]
        fig.add_trace(go.Scatter(
            x=log_spot, y=leverage[:, idx],
            mode="lines", name=f"t={t:.3f}y ({t*252:.0f}d)",
            hovertemplate="ln(S/S₀)=%{x:.4f}<br>L=%{y:.4f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5,
                  annotation_text="ATM")
    fig.update_layout(
        title="Leverage Function Slices L(t, S) by Maturity",
        xaxis_title="ln(S/S₀)",
        yaxis_title="L(t, S)",
    )
    return fig


# Tab 3: leverage vs local vol vs IV

def make_surfaces_comparison(leverage, spot_grid, time_grid, iv_surface,
                             local_vol, log_m_grid, ttm_grid, S):
    log_spot = np.log(spot_grid / S)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["IV Surface", "Dupire Local Vol", "Leverage L(t,S)"],
    )
    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid, z=iv_surface.T,
        colorscale="Viridis", colorbar=dict(x=0.28, len=0.8, title="IV"),
        hovertemplate="log(K/F): %{x:.4f}<br>TTM: %{y:.3f}<br>IV: %{z:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid, z=local_vol.T,
        colorscale="Plasma", colorbar=dict(x=0.63, len=0.8, title="LV"),
        hovertemplate="log(K/F): %{x:.4f}<br>TTM: %{y:.3f}<br>LV: %{z:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Surface(
        x=log_spot, y=time_grid, z=leverage.T,
        colorscale="Inferno", colorbar=dict(x=0.98, len=0.8, title="L"),
        hovertemplate="ln(S/S₀): %{x:.4f}<br>t: %{y:.3f}<br>L: %{z:.4f}<extra></extra>",
    ), row=1, col=3)
    fig.update_layout(
        title="IV Surface → Dupire Local Vol → Leverage Function",
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
    )
    return fig


# Tab 4: Heston fit summary

def make_heston_summary(heston, particle_log, val_summary, market):
    """Summary table: Heston params, particle stats, validation metrics."""
    heston_rows = [
        ["κ (mean reversion)", f"{heston['kappa']:.4f}"],
        ["θ (long-run var)", f"{heston['theta']:.6f}"],
        ["ξ (vol of vol)", f"{heston['xi']:.4f}"],
        ["ρ (correlation)", f"{heston['rho']:.4f}"],
        ["V₀ (initial var)", f"{heston['V0']:.6f}"],
        ["√V₀ (initial vol)", f"{np.sqrt(heston['V0']):.4f}"],
        ["2κθ/ξ² (Feller)", f"{heston['feller_value']:.4f} ({'✓' if heston['feller_satisfied'] else '✗'})"],
        ["IV MAE (vol pts)", f"{heston['iv_mae']*100:.2f}"],
        ["IV RMSE (vol pts)", f"{heston['iv_rmse']*100:.2f}"],
        ["N options calibrated", f"{heston['n_options']}"],
    ]

    # Particle method
    particle_rows = [
        ["N particles", f"{particle_log['N_particles']:,}"],
        ["dt", f"{particle_log['dt']:.6f} ({1/particle_log['dt']:.0f} steps/yr)"],
        ["N time steps", f"{particle_log['n_steps']}"],
        ["Bandwidth (mean±std)", f"{particle_log['bandwidth_mean']:.1f} ± {particle_log['bandwidth_std']:.1f}"],
        ["L² clip range", f"{particle_log['L_squared_clip_range']}"],
        ["Clipped evaluations", f"{particle_log['clip_count']:,} ({particle_log['clip_pct']:.2f}%)"],
    ]

    def _fmt_bps(key):
        v = val_summary.get(key)
        return f"{v:.1f} bp" if v is not None else "N/A"

    def _fmt_pct(key):
        v = val_summary.get(key)
        return f"{v:.2f}%" if v is not None else "N/A"

    # Validation summary
    val_rows = [
        ["LSV IV MAE (all)", _fmt_bps('lsv_iv_mae_bps')],
        ["LSV IV ME (all)", _fmt_bps('lsv_iv_me_bps')],
        ["LSV IV RMSE (all)", _fmt_bps('lsv_iv_rmse_bps')],
        ["LSV IV MAE (≥$10)", _fmt_bps('lsv_iv_mae_bps_ge_10')],
        ["LSV IV MAE (≥$50)", _fmt_bps('lsv_iv_mae_bps_ge_50')],
        ["LSV vs SSVI MAE (≥$10)", _fmt_pct('lsv_vs_ssvi_mae_ge_10')],
        ["Dupire vs SSVI MAE (≥$10)", _fmt_pct('dupire_vs_ssvi_mae_ge_10')],
        [f"S₀ = {market['S']:,.2f}", f"r = {market['r']:.4f}  q = {market['q']:.4f}"],
    ]

    all_rows = heston_rows + [["─── Particle Method ───", ""]] + particle_rows + \
               [["─── Validation ───", ""]] + val_rows

    fig = go.Figure(go.Table(
        header=dict(
            values=["Parameter", "Value"],
            fill_color="#4472C4", font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=[[r[0] for r in all_rows], [r[1] for r in all_rows]],
            fill_color=[["#f0f4ff", "white"] * (len(all_rows) // 2 + 1)][:len(all_rows)],
            font=dict(size=12), align=["left", "center"], height=28,
        ),
    ))
    fig.update_layout(title="Heston Calibration & Particle Method Summary", height=700)
    return fig


# Tab 5: LSV IV error (bp) vs log-moneyness

def make_lsv_vs_dupire_moneyness(df):
    """Primary comparison: LSV IV error (bp) vs log-moneyness."""
    valid = df.dropna(subset=["lsv_iv_error_bps"])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["All Options",
                                        "Liquid Options (market price ≥ $10)"])
    hover = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "IV Mkt: %{customdata[3]:.4f}  IV LSV: %{customdata[4]:.4f}<br>"
        "IV Error: %{y:+.1f} bp<extra></extra>"
    )
    cdata = ["strike", "ttm", "option_type", "iv_ssvi", "iv_lsv"]

    _add_cat_traces(fig, valid, "log_moneyness", "lsv_iv_error_bps",
                    customdata_cols=cdata, hovertemplate=hover,
                    row=1, col=1, size=4, opacity=0.5)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.4,
                  annotation_text="ATM", row=1, col=1)

    liquid = valid[valid["ssvi_price"] >= 10.0]
    if len(liquid) > 0:
        _add_cat_traces(fig, liquid, "log_moneyness", "lsv_iv_error_bps",
                        customdata_cols=cdata, hovertemplate=hover,
                        row=1, col=2, size=5, opacity=0.6, showlegend=False)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.4,
                      annotation_text="ATM", row=1, col=2)
        mae = liquid["lsv_iv_error_bps"].abs().mean()
        me = liquid["lsv_iv_error_bps"].mean()
        fig.add_annotation(
            text=f"MAE={mae:.1f} bp  ME={me:+.1f} bp  N={len(liquid)}",
            xref="x2", yref="y2 domain", x=0, y=1.0,
            showarrow=False, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="grey",
        )

    fig.update_xaxes(title_text="Fwd Log-Moneyness ln(K/F)", row=1, col=1)
    fig.update_yaxes(title_text="LSV IV Error (bp)", row=1, col=1)
    fig.update_xaxes(title_text="Fwd Log-Moneyness ln(K/F)", row=1, col=2)
    fig.update_yaxes(title_text="LSV IV Error (bp)", row=1, col=2)
    fig.update_layout(title="LSV IV Error (bp) vs Fwd Log-Moneyness")
    return fig


# Tab 6: LSV IV error (bp) vs TTM

def make_lsv_vs_dupire_ttm(df):
    """LSV IV error (bp) vs time to maturity."""
    valid = df.dropna(subset=["lsv_iv_error_bps"])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["All Options",
                                        "Liquid Options (market price ≥ $10)"])
    hover = (
        "K=%{customdata[0]:,.0f}  log(K/F)=%{customdata[1]:.4f}<br>"
        "Type: %{customdata[2]}<br>"
        "IV Mkt: %{customdata[3]:.4f}  IV LSV: %{customdata[4]:.4f}<br>"
        "IV Error: %{y:+.1f} bp<extra></extra>"
    )
    cdata = ["strike", "log_moneyness", "option_type", "iv_ssvi", "iv_lsv"]

    _add_cat_traces(fig, valid, "ttm", "lsv_iv_error_bps",
                    customdata_cols=cdata, hovertemplate=hover,
                    row=1, col=1, size=4, opacity=0.5)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

    liquid = valid[valid["ssvi_price"] >= 10.0]
    if len(liquid) > 0:
        _add_cat_traces(fig, liquid, "ttm", "lsv_iv_error_bps",
                        customdata_cols=cdata, hovertemplate=hover,
                        row=1, col=2, size=5, opacity=0.6, showlegend=False)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    fig.update_xaxes(title_text="TTM (years)", row=1, col=1)
    fig.update_yaxes(title_text="LSV IV Error (bp)", row=1, col=1)
    fig.update_xaxes(title_text="TTM (years)", row=1, col=2)
    fig.update_yaxes(title_text="LSV IV Error (bp)", row=1, col=2)
    fig.update_layout(title="LSV IV Error (bp) vs TTM")
    return fig


# Tab 7: LSV IV error (bp) distribution

def make_lsv_vs_dupire_hist(df):
    """Histogram of LSV IV error (bp) for all vs liquid."""
    valid = df.dropna(subset=["lsv_iv_error_bps"])
    liquid = valid[valid["ssvi_price"] >= 10.0]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["All Options", "Liquid (market price ≥ $10)"])

    cats = classify_options(valid)
    for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
        vals = valid.loc[cats == label, "lsv_iv_error_bps"].dropna()
        if len(vals) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=vals, name=f"{label} ({len(vals)})", legendgroup=label,
            marker_color=CAT_COLORS[label], opacity=0.6, nbinsx=30,
        ), row=1, col=1)

    if len(liquid) > 0:
        cats_l = classify_options(liquid)
        for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
            vals = liquid.loc[cats_l == label, "lsv_iv_error_bps"].dropna()
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(
                x=vals, name=label, legendgroup=label,
                marker_color=CAT_COLORS[label], opacity=0.6, nbinsx=30,
                showlegend=False,
            ), row=1, col=2)
        me = liquid["lsv_iv_error_bps"].mean()
        fig.add_vline(x=me, line_color="red", line_width=2,
                      annotation_text=f"Mean={me:+.1f} bp",
                      annotation_font_size=10, row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)

    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.update_xaxes(title_text="LSV IV Error (bp)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="LSV IV Error (bp)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(title="LSV IV Error (bp) — Distribution", barmode="overlay")
    return fig


# Tab 8: LSV & Dupire vs SSVI scatter

def make_price_scatter(df):
    """LSV and Dupire repriced prices vs market, side by side."""
    liquid = df[df["ssvi_price"] >= 10.0]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["LSV Price vs SSVI (liquid)",
                                        "Dupire Price vs SSVI (liquid)"])
    hover_lsv = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Market: $%{x:.2f}  LSV: $%{y:.2f}<br>"
        "Error: %{customdata[3]:.2f}%<extra></extra>"
    )
    hover_dup = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Market: $%{x:.2f}  Dupire: $%{y:.2f}<br>"
        "Error: %{customdata[3]:.2f}%<extra></extra>"
    )

    if len(liquid) > 0:
        _add_cat_traces(fig, liquid, "ssvi_price", "lsv_price",
                        customdata_cols=["strike", "ttm", "option_type", "lsv_vs_ssvi_pct"],
                        hovertemplate=hover_lsv, row=1, col=1, size=5, opacity=0.6)
        _add_cat_traces(fig, liquid, "ssvi_price", "dupire_price",
                        customdata_cols=["strike", "ttm", "option_type", "dupire_vs_ssvi_pct"],
                        hovertemplate=hover_dup, row=1, col=2, size=5, opacity=0.6,
                        showlegend=False)

        hi = max(liquid["ssvi_price"].max(), liquid["lsv_price"].max(),
                 liquid["dupire_price"].max()) * 1.05
        for c in [1, 2]:
            fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines",
                                     line=dict(color="black", dash="dash"),
                                     name="y=x", showlegend=False), row=1, col=c)

        # R² for each
        for c, price_col, label in [(1, "lsv_price", "LSV"), (2, "dupire_price", "Dupire")]:
            ss_res = ((liquid[price_col] - liquid["ssvi_price"])**2).sum()
            ss_tot = ((liquid["ssvi_price"] - liquid["ssvi_price"].mean())**2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            mae = (liquid[price_col] - liquid["ssvi_price"]).abs().mean()
            ax = "" if c == 1 else str(c)
            fig.add_annotation(
                text=f"{label}: R²={r2:.4f}  MAE=${mae:.2f}",
                xref=f"x{ax}", yref=f"y{ax} domain", x=0.05*hi, y=0.95,
                showarrow=False, font=dict(size=11),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="grey",
            )

    fig.update_xaxes(title_text="Market Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="LSV Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Market Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Dupire Price ($)", row=1, col=2)
    fig.update_layout(title="Model Prices vs SSVI Prices (liquid, mid ≥ $10)")
    return fig


# Tab 9: LSV vs SSVI pct error vs log-moneyness

def make_lsv_vs_ssvi_moneyness(df):
    """LSV and Dupire pct errors vs market, by log-moneyness."""
    liquid = df[df["ssvi_price"] >= 10.0].copy()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["LSV vs SSVI (%)",
                                        "Dupire vs SSVI (%)"])
    hover = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Price: $%{customdata[3]:.2f}<br>"
        "Error: %{y:.2f}%<extra></extra>"
    )

    if len(liquid) > 0:
        _add_cat_traces(fig, liquid, "log_moneyness", "lsv_vs_ssvi_pct",
                        customdata_cols=["strike", "ttm", "option_type", "ssvi_price"],
                        hovertemplate=hover, row=1, col=1, size=4, opacity=0.5)
        _add_cat_traces(fig, liquid, "log_moneyness", "dupire_vs_ssvi_pct",
                        customdata_cols=["strike", "ttm", "option_type", "ssvi_price"],
                        hovertemplate=hover, row=1, col=2, size=4, opacity=0.5,
                        showlegend=False)
        for c in [1, 2]:
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=c)
            fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.4,
                          annotation_text="ATM", row=1, col=c)

    fig.update_xaxes(title_text="Fwd Log-Moneyness ln(K/F)", row=1, col=1)
    fig.update_yaxes(title_text="(LSV − Market) / Market (%)", row=1, col=1)
    fig.update_xaxes(title_text="Fwd Log-Moneyness ln(K/F)", row=1, col=2)
    fig.update_yaxes(title_text="(Dupire − Market) / Market (%)", row=1, col=2)
    fig.update_layout(title="Model vs SSVI: Percentage Errors by Fwd Log-Moneyness (liquid, ≥ $10)")
    return fig


# Tab 10: error diagnostic (interactive)

def prepare_diagnostic_data(df):
    """Repricing data as JSON for the interactive diagnostic tab."""
    cols = ["strike", "ttm", "option_type", "moneyness", "log_moneyness",
            "ssvi_price", "lsv_price", "dupire_price",
            "iv_ssvi", "iv_lsv", "lsv_iv_error_bps",
            "lsv_vs_ssvi_pct", "dupire_vs_ssvi_pct"]
    available = [c for c in cols if c in df.columns]
    clean = df[available].dropna(subset=["lsv_iv_error_bps"]).copy()
    clean["category"] = classify_options(clean).values
    return clean.to_dict(orient="list")


# Tab 11: summary stats table

def make_summary_table(df, val_summary, market):
    """Summary stats table."""
    def stats_row(name, subset):
        if len(subset) == 0:
            return [name, 0, "—", "—", "—", "—", "—"]
        iv_err = subset["lsv_iv_error_bps"].dropna()
        lm = subset["lsv_vs_ssvi_pct"].dropna()
        dm = subset["dupire_vs_ssvi_pct"].dropna()
        return [
            name, len(subset),
            f"{iv_err.abs().mean():.1f}" if len(iv_err) > 0 else "—",
            f"{iv_err.mean():+.1f}" if len(iv_err) > 0 else "—",
            f"{lm.abs().mean():.2f}%" if len(lm) > 0 else "—",
            f"{dm.abs().mean():.2f}%" if len(dm) > 0 else "—",
            f"{(subset['lsv_price'] - subset['ssvi_price']).abs().mean():.2f}" if len(subset) > 0 else "—",
        ]

    cats = classify_options(df)
    liquid = df[df["ssvi_price"] >= 10.0]
    liq50 = df[df["ssvi_price"] >= 50.0]

    rows = [
        stats_row("All", df),
        stats_row("Liquid (≥$10)", liquid),
        stats_row("Liquid (≥$50)", liq50),
        stats_row("Calls", df[df["option_type"] == "call"]),
        stats_row("Puts", df[df["option_type"] == "put"]),
        stats_row("OTM Calls", df[cats == "OTM Call"]),
        stats_row("OTM Puts", df[cats == "OTM Put"]),
        stats_row("ITM Calls", df[cats == "ITM Call"]),
        stats_row("ITM Puts", df[cats == "ITM Put"]),
    ]

    headers = ["Subset", "N", "LSV IV MAE (bp)", "LSV IV ME (bp)",
               "lsv/Mkt MAE%", "Dup/Mkt MAE%", "LSV MAE ($)"]
    cell_vals = [[r[i] for r in rows] for i in range(len(headers))]

    fig = go.Figure(go.Table(
        header=dict(values=headers,
                    fill_color="#4472C4", font=dict(color="white", size=13),
                    align="center"),
        cells=dict(values=cell_vals,
                   fill_color=[["#f0f4ff", "white"] * (len(rows) // 2 + 1)][:len(rows)],
                   font=dict(size=12), align="center", height=28),
    ))
    fig.update_layout(
        title=(f"Repricing Summary  |  S={market['S']:,.0f}  "
               f"r={market['r']:.4f}  q={market['q']:.4f}  date={market['date']}"),
        height=450,
    )
    return fig


# HTML assembly

TAB_DESCRIPTIONS = {
    "Leverage Surface": (
        "The calibrated leverage function L(t, S) from the particle method. "
        "This is the ratio σ_Dupire(t,S) / √E[V_t | S_t = S] that bridges the "
        "Dupire local vol surface with Heston stochastic vol. Rotate and zoom "
        "to inspect the shape across spot and time."
    ),
    "Leverage Slices": (
        "Cross-sections of the leverage function at selected time points. "
        "Each line shows L(t, S) as a function of spot for a fixed maturity. "
        "The shape should be relatively stable — erratic behaviour suggests "
        "insufficient particles or bandwidth issues."
    ),
    "Surfaces Overview": (
        "Side-by-side comparison: IV surface (input from Step 1) → Dupire local "
        "vol (Step 2) → Leverage function (Step 3). The leverage function should "
        "be smoother than the local vol since it divides out the conditional "
        "expectation of stochastic variance."
    ),
    "Heston + Params": (
        "Summary of all calibration and particle method parameters. Heston "
        "parameters (κ, θ, ξ, ρ, V₀), Feller condition status, particle "
        "method settings, and headline validation metrics."
    ),
    "LSV vs Dupire (Moneyness)": (
        "PRIMARY DIAGNOSTIC. LSV implied volatility error (IV_LSV − IV_market) in "
        "basis points plotted against log-moneyness. This is the key metric — it shows "
        "whether the LSV model reproduces the market IV surface (as it should via "
        "Gyöngy's theorem). Systematic patterns indicate calibration issues."
    ),
    "LSV vs Dupire (TTM)": (
        "LSV IV error (bp) plotted against time to maturity. "
        "This reveals whether the leverage function degrades at short or long "
        "horizons — common when the particle cloud has thinned out or the "
        "bandwidth is too wide/narrow for certain maturities."
    ),
    "LSV vs Dupire (Hist)": (
        "Distribution of the LSV IV errors in basis points. A tight distribution "
        "centered at zero means the LSV model faithfully reproduces the market IV "
        "surface. Fat tails or systematic shift indicate calibration issues."
    ),
    "Price Scatter": (
        "LSV and Dupire repriced prices vs actual market mid prices (liquid "
        "options, mid ≥ $10). Points near the diagonal mean the model prices "
        "well. Compare the two panels to see whether LSV improves over Dupire."
    ),
    "vs SSVI (Moneyness)": (
        "Both models' percentage errors against market prices, plotted by "
        "log-moneyness. Compare left (LSV) and right (Dupire) to see which "
        "model fits the market better and where each struggles."
    ),
    "Error Diagnostic": (
        "Interactive diagnostic tool: filter by strike range and TTM range using "
        "sliders to isolate specific regions. Shows LSV IV error (bp) and "
        "both models vs market price errors. Use this to pinpoint which strikes and "
        "maturities drive the IV differences."
    ),
    "Summary Stats": (
        "Quantitative summary: LSV IV MAE and ME (bp) and both models' price errors "
        "vs market, broken down by option type and moneyness category. "
        "This is the key checkpoint table for thesis Chapter 4."
    ),
}


def build_html(figures, tab_names, descriptions, diagnostic_data=None, diagnostic_tab_idx=None):
    """Build the single tabbed HTML file from the Plotly figures."""
    fig_json_list = []
    fig_idx_map = {}
    for i, fig in enumerate(figures):
        if fig is None:
            continue
        fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        fig_idx_map[i] = len(fig_json_list)
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
        if i == diagnostic_tab_idx:
            tab_contents.append(
                f'<div class="tab-content" id="tab-{i}" style="display:{display}">'
                f'<p class="tab-desc">{desc}</p>'
                f'<div id="diag-controls" style="margin-bottom:10px;">'
                f'  <div style="display:flex;gap:40px;flex-wrap:wrap;align-items:center;">'
                f'    <div style="flex:1;min-width:300px;">'
                f'      <label style="font-weight:600;font-size:13px;">Strike Range</label><br>'
                f'      <span id="strike-lo-val"></span> — <span id="strike-hi-val"></span>'
                f'      <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">'
                f'        <input type="range" id="strike-lo" style="flex:1;" oninput="diagUpdate()">'
                f'        <input type="range" id="strike-hi" style="flex:1;" oninput="diagUpdate()">'
                f'      </div>'
                f'    </div>'
                f'    <div style="flex:1;min-width:300px;">'
                f'      <label style="font-weight:600;font-size:13px;">TTM Range (years)</label><br>'
                f'      <span id="ttm-lo-val"></span> — <span id="ttm-hi-val"></span>'
                f'      <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">'
                f'        <input type="range" id="ttm-lo" style="flex:1;" step="0.001" oninput="diagUpdate()">'
                f'        <input type="range" id="ttm-hi" style="flex:1;" step="0.001" oninput="diagUpdate()">'
                f'      </div>'
                f'    </div>'
                f'    <div style="flex:1;min-width:200px;">'
                f'      <label style="font-weight:600;font-size:13px;">Min Price ($)</label><br>'
                f'      <span id="price-min-val">0</span>'
                f'      <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">'
                f'        <input type="range" id="price-min" style="flex:1;" min="0" max="200" value="0" step="5" oninput="diagUpdate()">'
                f'      </div>'
                f'    </div>'
                f'  </div>'
                f'  <div id="diag-stats" style="margin-top:8px;font-size:13px;color:#555;"></div>'
                f'</div>'
                f'<div id="diag-plot" style="width:100%;height:520px;"></div>'
                f'</div>'
            )
        else:
            tab_contents.append(
                f'<div class="tab-content" id="tab-{i}" style="display:{display}">'
                f'<p class="tab-desc">{desc}</p>'
                f'<div id="plot-{i}" style="width:100%;height:650px;"></div>'
                f'</div>'
            )

    fig_specs_js = ",\n".join(fig_json_list)
    fig_idx_map_js = json.dumps(fig_idx_map)
    diag_data_js = json.dumps(diagnostic_data) if diagnostic_data else "null"
    diag_idx_js = str(diagnostic_tab_idx) if diagnostic_tab_idx is not None else "-1"

    html_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LSV Explorer</title>
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
  input[type=range] { height: 6px; cursor: pointer; }
</style>
</head>
<body>
<h1>LSV Model Explorer</h1>
"""

    html_tabs = '<div class="tab-bar">\n  ' + ''.join(tab_buttons) + '\n</div>\n'
    html_tabs += ''.join(tab_contents)

    html_script = """
<script>
var figSpecs = [""" + fig_specs_js + """];
var figIdxMap = """ + fig_idx_map_js + """;
var rendered = {};
var DIAG_TAB = """ + diag_idx_js + """;
var diagData = """ + diag_data_js + """;
var diagInitialised = false;

var catColors = {
  "OTM Call": "#1f77b4", "ITM Call": "#7fbfff",
  "OTM Put": "#d62728", "ITM Put": "#ff9896"
};
var catSymbols = {
  "OTM Call": "circle", "ITM Call": "square",
  "OTM Put": "diamond", "ITM Put": "triangle-up"
};

function renderPlot(idx) {
  if (idx === DIAG_TAB) {
    initDiagnostic();
    return;
  }
  if (rendered[idx]) {
    Plotly.Plots.resize(document.getElementById('plot-' + idx));
    return;
  }
  var specIdx = figIdxMap[String(idx)];
  if (specIdx === undefined) return;
  var spec = figSpecs[specIdx];
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

function initDiagnostic() {
  if (!diagData || diagInitialised) {
    if (diagInitialised) Plotly.Plots.resize(document.getElementById('diag-plot'));
    return;
  }
  var strikes = diagData.strike;
  var ttms = diagData.ttm;
  var sMin = Math.min.apply(null, strikes), sMax = Math.max.apply(null, strikes);
  var tMin = Math.min.apply(null, ttms), tMax = Math.max.apply(null, ttms);

  var sloEl = document.getElementById('strike-lo');
  var shiEl = document.getElementById('strike-hi');
  var tloEl = document.getElementById('ttm-lo');
  var thiEl = document.getElementById('ttm-hi');

  sloEl.min = sMin; sloEl.max = sMax; sloEl.value = sMin; sloEl.step = 5;
  shiEl.min = sMin; shiEl.max = sMax; shiEl.value = sMax; shiEl.step = 5;
  tloEl.min = tMin; tloEl.max = tMax; tloEl.value = tMin;
  thiEl.min = tMin; thiEl.max = tMax; thiEl.value = tMax;

  diagInitialised = true;
  diagUpdate();
}

function diagUpdate() {
  var sLo = parseFloat(document.getElementById('strike-lo').value);
  var sHi = parseFloat(document.getElementById('strike-hi').value);
  var tLo = parseFloat(document.getElementById('ttm-lo').value);
  var tHi = parseFloat(document.getElementById('ttm-hi').value);
  var pMin = parseFloat(document.getElementById('price-min').value);
  if (sLo > sHi) { var tmp = sLo; sLo = sHi; sHi = tmp; }
  if (tLo > tHi) { var tmp = tLo; tLo = tHi; tHi = tmp; }

  document.getElementById('strike-lo-val').textContent = sLo.toFixed(0);
  document.getElementById('strike-hi-val').textContent = sHi.toFixed(0);
  document.getElementById('ttm-lo-val').textContent = tLo.toFixed(3);
  document.getElementById('ttm-hi-val').textContent = tHi.toFixed(3);
  document.getElementById('price-min-val').textContent = pMin.toFixed(0);

  // Filter data
  var n = diagData.strike.length;
  var filtered = {};
  var keys = Object.keys(diagData);
  keys.forEach(function(k) { filtered[k] = []; });
  for (var i = 0; i < n; i++) {
    if (diagData.strike[i] >= sLo && diagData.strike[i] <= sHi &&
        diagData.ttm[i] >= tLo && diagData.ttm[i] <= tHi &&
        diagData.dupire_price[i] >= pMin) {
      keys.forEach(function(k) { filtered[k].push(diagData[k][i]); });
    }
  }

  // Build traces — LSV IV error (bp) vs log-moneyness, coloured by category
  var traces = [];
  var cats = ["OTM Call", "ITM Call", "OTM Put", "ITM Put"];
  for (var ci = 0; ci < cats.length; ci++) {
    var cat = cats[ci];
    var x = [], y = [], cd = [];
    for (var i = 0; i < filtered.strike.length; i++) {
      if (filtered.category[i] === cat) {
        x.push(filtered.log_moneyness[i]);
        y.push(filtered.lsv_iv_error_bps[i]);
        cd.push([filtered.strike[i], filtered.ttm[i], filtered.option_type[i],
                 filtered.iv_ssvi[i], filtered.iv_lsv[i],
                 filtered.lsv_vs_ssvi_pct[i], filtered.dupire_vs_ssvi_pct[i]]);
      }
    }
    if (x.length === 0) continue;
    traces.push({
      x: x, y: y, customdata: cd, mode: 'markers',
      marker: {size: 5, color: catColors[cat], symbol: catSymbols[cat], opacity: 0.7},
      name: cat + ' (' + x.length + ')',
      hovertemplate: 'K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>' +
                     'Type: %{customdata[2]}<br>' +
                     'IV Mkt: %{customdata[3]:.4f}  IV LSV: %{customdata[4]:.4f}<br>' +
                     'IV Error: %{y:+.1f} bp<br>' +
                     'lsv/Mkt: %{customdata[5]:.2f}%  Dup/Mkt: %{customdata[6]:.2f}%<extra></extra>'
    });
  }
  // Zero line
  var xRange = filtered.log_moneyness.length > 0 ?
    [Math.min.apply(null, filtered.log_moneyness), Math.max.apply(null, filtered.log_moneyness)] : [-0.3, 0.3];
  traces.push({x: xRange, y: [0, 0], mode: 'lines',
               line: {color: 'black', dash: 'dash', width: 1.2}, showlegend: false});
  // ATM line
  var yExt = filtered.lsv_iv_error_bps.length > 0 ?
    [Math.min.apply(null, filtered.lsv_iv_error_bps) * 1.5, Math.max.apply(null, filtered.lsv_iv_error_bps) * 1.5] : [-200, 200];
  traces.push({x: [0, 0], y: yExt, mode: 'lines',
               line: {color: 'grey', dash: 'dot', width: 1}, showlegend: false});

  var layout = {
    template: 'plotly_white',
    title: 'LSV IV Error (bp) vs Fwd Log-Moneyness  [' + filtered.strike.length + ' options, price >= $' + pMin.toFixed(0) + ']',
    xaxis: {title: 'Fwd Log-Moneyness ln(K/F)'}, yaxis: {title: 'LSV IV Error (bp)'},
    height: 520, margin: {l: 60, r: 40, t: 50, b: 50},
    legend: {font: {size: 11}}
  };
  Plotly.react('diag-plot', traces, layout, {responsive: true});

  // Stats
  var nf = filtered.strike.length;
  if (nf === 0) {
    document.getElementById('diag-stats').innerHTML = '<b>No options in range.</b>';
    return;
  }
  var sumIV = 0, sumAbsIV = 0, sumLM = 0, sumAbsLM = 0, sumDM = 0, sumAbsDM = 0;
  for (var i = 0; i < nf; i++) {
    sumIV += filtered.lsv_iv_error_bps[i];
    sumAbsIV += Math.abs(filtered.lsv_iv_error_bps[i]);
    sumLM += filtered.lsv_vs_ssvi_pct[i];
    sumAbsLM += Math.abs(filtered.lsv_vs_ssvi_pct[i]);
    sumDM += filtered.dupire_vs_ssvi_pct[i];
    sumAbsDM += Math.abs(filtered.dupire_vs_ssvi_pct[i]);
  }
  document.getElementById('diag-stats').innerHTML =
    '<b>' + nf + ' options</b> &nbsp;|&nbsp; ' +
    '<span style="color:#e65100">LSV IV: ME=' + (sumIV/nf).toFixed(1) + ' bp  MAE=' + (sumAbsIV/nf).toFixed(1) + ' bp</span>' +
    ' &nbsp;|&nbsp; LSV/Mkt: MAE=' + (sumAbsLM/nf).toFixed(2) + '%' +
    ' &nbsp;|&nbsp; Dup/Mkt: MAE=' + (sumAbsDM/nf).toFixed(2) + '%';
}

// Render the first tab on load
renderPlot(0);
</script>
</body>
</html>"""

    return html_head + html_tabs + html_script


def main():
    (df, leverage, spot_grid, time_grid, heston, particle_log,
     val_summary, market, iv_surface, log_m_grid, ttm_grid, local_vol) = load_data()

    S = market["S"]

    tab_names = [
        "Leverage Surface",
        "Leverage Slices",
        "Surfaces Overview",
        "Heston + Params",
        "LSV vs Dupire (Moneyness)",
        "LSV vs Dupire (TTM)",
        "LSV vs Dupire (Hist)",
        "Price Scatter",
        "vs SSVI (Moneyness)",
        "Error Diagnostic",
        "Summary Stats",
    ]

    diag_tab_idx = tab_names.index("Error Diagnostic")

    figures = [
        make_leverage_surface(leverage, spot_grid, time_grid, S),
        make_leverage_slices(leverage, spot_grid, time_grid, S),
        make_surfaces_comparison(leverage, spot_grid, time_grid, iv_surface,
                                 local_vol, log_m_grid, ttm_grid, S),
        make_heston_summary(heston, particle_log, val_summary, market),
        make_lsv_vs_dupire_moneyness(df),
        make_lsv_vs_dupire_ttm(df),
        make_lsv_vs_dupire_hist(df),
        make_price_scatter(df),
        make_lsv_vs_ssvi_moneyness(df),
        None,  # diagnostic tab: custom HTML
        make_summary_table(df, val_summary, market),
    ]

    diag_data = prepare_diagnostic_data(df)

    html = build_html(figures, tab_names, TAB_DESCRIPTIONS,
                      diagnostic_data=diag_data, diagnostic_tab_idx=diag_tab_idx)
    os.makedirs(DIR_PLOTS, exist_ok=True)
    out_path = os.path.join(DIR_PLOTS, "lsv_explorer.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path} — open in your browser")


if __name__ == "__main__":
    main()

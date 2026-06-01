#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bergomi LSV Explorer — Interactive Plotly Dashboard
=====================================================
Single HTML file with tabbed plots for exploring the Bergomi LSV model outputs:
leverage surface, forward variance, repricing errors.

Usage:
    python lsv_explorer.py

Output:
    plots/lsv_bergomi_explorer.html — open in any browser, switch between tabs
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Directories (relative to repo root — run from repo root or lsv_bergomi/)
DIR_DATA   = "lsv_bergomi/data"
DIR_PLOTS  = "lsv_bergomi/plots"
DIR_ARRAYS = "lsv_bergomi/arrays"
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

    with open(os.path.join(DIR_DATA, "bergomi_params.json")) as f:
        bergomi = json.load(f)

    with open(os.path.join(DIR_DATA, "particle_log.json")) as f:
        particle_log = json.load(f)

    with open(os.path.join(DIR_DATA, "validation_summary.json")) as f:
        val_summary = json.load(f)

    with open(os.path.join(DUPIRE_DIR_DATA, "market_params.json")) as f:
        market = json.load(f)

    # Forward variance artifacts
    fwd_var = np.load(os.path.join(DIR_ARRAYS, "fwd_var_curve.npy"))
    vs_vol = np.load(os.path.join(DIR_ARRAYS, "vs_vol_curve.npy"))
    vs_vol_fitted = np.load(os.path.join(DIR_ARRAYS, "vs_vol_fitted.npy"))
    ttm_grid_iv = np.load(os.path.join(IV_DIR_ARRAYS, "ttm_grid.npy"))

    with open(os.path.join(DIR_DATA, "fwd_var_fit.json")) as f:
        fwd_var_fit = json.load(f)

    # IV surface and Dupire local vol for context
    iv_surface = np.load(os.path.join(IV_DIR_ARRAYS, "iv_surface.npy"))
    log_m_grid = np.load(os.path.join(IV_DIR_ARRAYS, "log_m_grid.npy"))
    local_vol = np.load(os.path.join(DUPIRE_DIR_ARRAYS, "local_vol_surface.npy"))

    return (df, leverage, spot_grid, time_grid, bergomi, particle_log,
            val_summary, market, fwd_var, vs_vol, vs_vol_fitted, ttm_grid_iv,
            fwd_var_fit, iv_surface, log_m_grid, local_vol)


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


# ── Tab 1: 3D Leverage Surface ──────────────────────────────────────────────

def make_leverage_surface(leverage, spot_grid, time_grid, S):
    log_spot = np.log(spot_grid / S)
    fig = go.Figure(go.Surface(
        x=log_spot, y=time_grid, z=leverage.T,
        colorscale="Inferno", colorbar_title="σ(t,S)",
        contours=dict(
            x=dict(show=True, color="rgba(0,0,0,0.12)", width=1),
            y=dict(show=True, color="rgba(0,0,0,0.12)", width=1),
        ),
        hovertemplate=(
            "ln(S/S₀): %{x:.4f}<br>"
            "t: %{y:.3f} yr<br>"
            "σ(t,S): %{z:.4f}<extra></extra>"
        ),
    ))
    x_range = log_spot.max() - log_spot.min()
    y_range = time_grid.max() - time_grid.min()
    z_range = leverage.max() - leverage.min()
    mx = max(x_range, y_range)
    fig.update_layout(
        title="Bergomi Leverage Function σ(t, S)",
        scene=dict(
            xaxis_title="ln(S/S₀)",
            yaxis_title="Time (years)",
            zaxis_title="σ(t, S)",
            aspectmode="manual",
            aspectratio=dict(x=x_range/mx, y=y_range/mx, z=z_range/mx*1.5),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── Tab 2: Leverage Slices by Time ──────────────────────────────────────────

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
            hovertemplate="ln(S/S₀)=%{x:.4f}<br>σ=%{y:.4f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5,
                  annotation_text="ATM")
    fig.update_layout(
        title="Leverage Function Slices σ(t, S) by Maturity (Bergomi)",
        xaxis_title="ln(S/S₀)",
        yaxis_title="σ(t, S)",
    )
    return fig


# ── Tab 3: Forward Variance Curve ──────────────────────────────────────────

def make_fwd_var_plot(ttm_grid, vs_vol, vs_vol_fitted, fwd_var, fwd_var_fit):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Variance Swap Volatility",
            "Forward Variance ξ<sup>T</sup><sub>0</sub>",
            "Forward Volatility √ξ<sup>T</sup><sub>0</sub>",
        ],
    )

    # Panel 1: VS vol
    fig.add_trace(go.Scatter(
        x=ttm_grid, y=vs_vol, mode="markers", name="VS Vol (integrated)",
        marker=dict(size=4, color="#1f77b4", opacity=0.6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ttm_grid, y=vs_vol_fitted, mode="lines", name="NSS Fit",
        line=dict(color="#d62728", width=2),
        hovertemplate=(
            "TTM: %{x:.3f}<br>σ̂=%{y:.4f}<br>"
            f"β₀={fwd_var_fit['nss_beta_0']:.4f}, "
            f"β₁={fwd_var_fit['nss_beta_1']:+.4f}, "
            f"β₂={fwd_var_fit['nss_beta_2']:+.4f}, "
            f"β₃={fwd_var_fit['nss_beta_3']:+.4f}, "
            f"τ₁={fwd_var_fit['nss_tau_1']:.4f}, "
            f"τ₂={fwd_var_fit['nss_tau_2']:.4f}<extra></extra>"
        ),
    ), row=1, col=1)
    fig.update_xaxes(title_text="TTM (years)", row=1, col=1)
    fig.update_yaxes(title_text="σ̂(T)", row=1, col=1)

    # Panel 2: Forward variance
    fig.add_trace(go.Scatter(
        x=ttm_grid, y=fwd_var, mode="lines", name="ξ<sup>T</sup><sub>0</sub>",
        line=dict(color="darkorange", width=2),
        hovertemplate="TTM: %{x:.3f}<br>ξ=%{y:.6f}<extra></extra>",
    ), row=1, col=2)
    fig.update_xaxes(title_text="TTM (years)", row=1, col=2)
    fig.update_yaxes(title_text="ξ<sup>T</sup><sub>0</sub>", row=1, col=2)

    # Panel 3: Forward volatility
    fwd_vol = np.sqrt(np.maximum(fwd_var, 0))
    fig.add_trace(go.Scatter(
        x=ttm_grid, y=fwd_vol, mode="lines", name="√ξ<sup>T</sup><sub>0</sub>",
        line=dict(color="green", width=2),
        hovertemplate="TTM: %{x:.3f}<br>fwd vol=%{y:.4f}<extra></extra>",
    ), row=1, col=3)
    fig.update_xaxes(title_text="TTM (years)", row=1, col=3)
    fig.update_yaxes(title_text="Forward Vol", row=1, col=3)

    fig.update_layout(
        title="Forward Variance Extraction from SSVI Surface",
        showlegend=True,
    )
    return fig


# ── Tab 4: Surfaces Comparison ─────────────────────────────────────────────

def make_surfaces_comparison(leverage, spot_grid, time_grid, iv_surface,
                             local_vol, log_m_grid, ttm_grid_iv, S):
    log_spot = np.log(spot_grid / S)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["IV Surface", "Dupire Local Vol", "Leverage σ(t,S)"],
    )
    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid_iv, z=iv_surface.T,
        colorscale="Viridis", colorbar=dict(x=0.28, len=0.8, title="IV"),
        hovertemplate="log(K/F): %{x:.4f}<br>TTM: %{y:.3f}<br>IV: %{z:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid_iv, z=local_vol.T,
        colorscale="Plasma", colorbar=dict(x=0.62, len=0.8, title="σ<sub>D</sub>"),
        hovertemplate="log(K/F): %{x:.4f}<br>TTM: %{y:.3f}<br>σ_D: %{z:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Surface(
        x=log_spot, y=time_grid, z=leverage.T,
        colorscale="Inferno", colorbar=dict(x=0.96, len=0.8, title="σ"),
        hovertemplate="ln(S/S₀): %{x:.4f}<br>t: %{y:.3f}<br>σ: %{z:.4f}<extra></extra>",
    ), row=1, col=3)
    fig.update_layout(
        title="Surface Comparison: IV / Dupire / Bergomi Leverage",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── Tab 5: Bergomi Params + Summary ────────────────────────────────────────

def make_bergomi_summary(bergomi, particle_log, val_summary, market, fwd_var_fit):
    """Rich text summary of Bergomi params and validation results."""
    fig = go.Figure()

    # Bergomi params
    lines = [
        "<b>Bergomi Two-Factor Parameters (Wang Set II)</b>",
        f"  ν (vol-of-vol):     {bergomi['nu']}",
        f"  θ (mixing):         {bergomi['theta']}",
        f"  κ₁ (fast reversion): {bergomi['kappa1']}",
        f"  κ₂ (slow reversion): {bergomi['kappa2']}",
        f"  ρ₁₂ (OU corr):      {bergomi.get('rho12', 0.0)}",
        f"  ρ₁ (spot-OU₁):      {bergomi['rho1']}",
        f"  ρ₂ (spot-OU₂):      {bergomi['rho2']}",
        "",
        "<b>Forward Variance Fit (NSS)</b>",
        f"  β₀ = {fwd_var_fit['nss_beta_0']:+.4f}   β₁ = {fwd_var_fit['nss_beta_1']:+.4f}   "
        f"β₂ = {fwd_var_fit['nss_beta_2']:+.4f}   β₃ = {fwd_var_fit['nss_beta_3']:+.4f}",
        f"  τ₁ = {fwd_var_fit['nss_tau_1']:.4f}   τ₂ = {fwd_var_fit['nss_tau_2']:.4f}   "
        f"RMSE = {fwd_var_fit.get('nss_rmse', 0.0) * 1e4:.1f} bp",
        f"  σ̂(T) = β₀ + β₁ f₁(T,τ₁) + β₂ f₂(T,τ₁) + β₃ f₂(T,τ₂)   (Svensson 1994)",
        "",
        "<b>Market</b>",
        f"  S₀ = {market['S']}   r = {market['r']}   q = {market['q']}",
        f"  Date: {market.get('date', 'N/A')}",
    ]

    if particle_log:
        lines += [
            "",
            "<b>Particle Method</b>",
            f"  Particles:   {particle_log.get('N_particles', particle_log.get('N', 'N/A'))}",
            f"  Time steps:  {particle_log.get('n_steps', particle_log.get('n_time_steps', 'N/A'))}",
            f"  dt:          {particle_log.get('dt', 'N/A')}",
        ]

    if val_summary:
        lines += [
            "",
            "<b>Bergomi LSV Validation</b>",
            f"  Options repriced: {val_summary.get('n_valid', 'N/A')}",
            f"  IV Error MAE:     {val_summary.get('lsv_iv_mae_bps', 'N/A'):.1f} bp",
            f"  IV Error ME:      {val_summary.get('lsv_iv_me_bps', 'N/A'):+.1f} bp",
            f"  IV Error RMSE:    {val_summary.get('lsv_iv_rmse_bps', 'N/A'):.1f} bp",
            f"  Price MAE:        {val_summary.get('lsv_vs_ssvi_mae_pct', 'N/A'):.2f}%",
            f"  Dupire MAE:       {val_summary.get('dupire_vs_ssvi_mae_pct', 'N/A'):.2f}%",
        ]

    fig.add_annotation(
        text="<br>".join(lines),
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False, align="left",
        font=dict(family="monospace", size=13),
        xanchor="left", yanchor="top",
        bordercolor="#ccc", borderwidth=1, borderpad=10,
        bgcolor="white",
    )
    fig.update_layout(
        title="Bergomi Model Summary",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ── Tab 6: IV Error by Moneyness ──────────────────────────────────────────

def make_error_by_moneyness(df):
    valid = df.dropna(subset=["lsv_iv_error_bps"]).copy()
    fig = go.Figure()
    _add_cat_traces(
        fig, valid, "log_moneyness", "lsv_iv_error_bps",
        customdata_cols=["strike", "ttm", "option_type", "iv_ssvi", "iv_lsv"],
        hovertemplate=(
            "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
            "Type: %{customdata[2]}<br>"
            "IV Mkt: %{customdata[3]:.4f}  IV LSV: %{customdata[4]:.6f}<br>"
            "IV Error: %{y:+.1f} bp<extra></extra>"
        ),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="grey", opacity=0.5,
                  annotation_text="ATM")
    mae = valid["lsv_iv_error_bps"].abs().mean()
    fig.update_layout(
        title=f"Bergomi LSV IV Error by Moneyness (MAE = {mae:.1f} bp)",
        xaxis_title="Fwd Log-Moneyness ln(K/F)",
        yaxis_title="IV Error (bp)",
    )
    return fig


# ── Tab 7: IV Error by TTM ────────────────────────────────────────────────

def make_error_by_ttm(df):
    valid = df.dropna(subset=["lsv_iv_error_bps"]).copy()
    fig = go.Figure()
    _add_cat_traces(
        fig, valid, "ttm", "lsv_iv_error_bps",
        customdata_cols=["strike", "log_moneyness", "option_type", "iv_ssvi", "iv_lsv"],
        hovertemplate=(
            "K=%{customdata[0]:,.0f}  log(K/F)=%{customdata[1]:.4f}<br>"
            "Type: %{customdata[2]}<br>"
            "IV Mkt: %{customdata[3]:.4f}  IV LSV: %{customdata[4]:.6f}<br>"
            "IV Error: %{y:+.1f} bp<extra></extra>"
        ),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="Bergomi LSV IV Error by Time-to-Maturity",
        xaxis_title="TTM (years)",
        yaxis_title="IV Error (bp)",
    )
    return fig


# ── Tab 8: IV Error Histogram ─────────────────────────────────────────────

def make_error_hist(df):
    valid = df.dropna(subset=["lsv_iv_error_bps"])
    err = valid["lsv_iv_error_bps"]
    mae = err.abs().mean()
    me = err.mean()
    rmse = np.sqrt((err**2).mean())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=err, nbinsx=50, marker_color="steelblue",
        hovertemplate="IV Error: %{x:.1f} bp<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_vline(x=me, line_dash="dot", line_color="orange", line_width=1.5,
                  annotation_text=f"ME={me:+.1f}")
    fig.update_layout(
        title=f"IV Error Distribution (MAE={mae:.1f} bp, RMSE={rmse:.1f} bp)",
        xaxis_title="IV Error (bp)",
        yaxis_title="Count",
    )
    return fig


# ── Tab 9: Price Scatter ──────────────────────────────────────────────────

def make_price_scatter(df):
    valid = df.dropna(subset=["lsv_price"]).copy()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["LSV vs SSVI", "LSV vs Dupire"],
    )
    _add_cat_traces(fig, valid, "ssvi_price", "lsv_price",
                    customdata_cols=["strike", "ttm", "option_type"],
                    hovertemplate=(
                        "K=%{customdata[0]:,.0f} TTM=%{customdata[1]:.3f}<br>"
                        "Mkt: $%{x:.2f}  LSV: $%{y:.2f}<extra></extra>"
                    ), row=1, col=1)
    _add_cat_traces(fig, valid, "dupire_price", "lsv_price",
                    customdata_cols=["strike", "ttm", "option_type"],
                    hovertemplate=(
                        "K=%{customdata[0]:,.0f} TTM=%{customdata[1]:.3f}<br>"
                        "Dupire: $%{x:.2f}  LSV: $%{y:.2f}<extra></extra>"
                    ), row=1, col=2, showlegend=False)

    mx = max(valid["ssvi_price"].max(), valid["lsv_price"].max()) * 1.05
    for c in [1, 2]:
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                                 line=dict(dash="dash", color="black", width=1),
                                 showlegend=False), row=1, col=c)
    fig.update_xaxes(title_text="Market Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="LSV Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Dupire Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="LSV Price ($)", row=1, col=2)
    fig.update_layout(title="Price Scatter: Bergomi LSV")
    return fig


# ── Tab 10: Summary Stats Table ───────────────────────────────────────────

def make_summary_table(df, val_summary, market):
    valid = df.dropna(subset=["lsv_iv_error_bps"])
    err = valid["lsv_iv_error_bps"]

    header_vals = ["Metric", "Value"]
    cell_metric = [
        "Options repriced",
        "IV Error MAE (bp)", "IV Error ME (bp)", "IV Error RMSE (bp)",
        "IV Error Median (bp)", "5th pctl (bp)", "95th pctl (bp)",
        "LSV vs Mkt Price MAE %", "Dupire vs Mkt Price MAE %",
        "S₀", "r", "q",
    ]
    cell_value = [
        str(len(valid)),
        f"{err.abs().mean():.1f}", f"{err.mean():+.1f}",
        f"{np.sqrt((err**2).mean()):.1f}",
        f"{err.abs().median():.1f}",
        f"{err.quantile(0.05):.1f}", f"{err.quantile(0.95):.1f}",
        f"{val_summary.get('lsv_vs_ssvi_mae_pct', 'N/A')}",
        f"{val_summary.get('dupire_vs_ssvi_mae_pct', 'N/A')}",
        str(market["S"]), str(market["r"]), str(market["q"]),
    ]

    fig = go.Figure(go.Table(
        header=dict(values=header_vals, fill_color="#1a73e8",
                    font=dict(color="white", size=13), align="left"),
        cells=dict(values=[cell_metric, cell_value],
                   fill_color=[["white", "#f8f8f8"] * (len(cell_metric) // 2 + 1)],
                   font=dict(size=12), align="left", height=28),
    ))
    fig.update_layout(title="Bergomi LSV — Summary Statistics", margin=dict(t=50))
    return fig


# ── HTML builder ───────────────────────────────────────────────────────────

def build_html(figures, tab_names, descriptions, diagnostic_data=None, diagnostic_tab_idx=None):
    """Build a single HTML file with CSS tabs and Plotly figures."""
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
<title>Bergomi LSV Explorer</title>
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
<h1>Bergomi LSV Model Explorer</h1>
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
  var xRange = filtered.log_moneyness.length > 0 ?
    [Math.min.apply(null, filtered.log_moneyness), Math.max.apply(null, filtered.log_moneyness)] : [-0.3, 0.3];
  traces.push({x: xRange, y: [0, 0], mode: 'lines',
               line: {color: 'black', dash: 'dash', width: 1.2}, showlegend: false});
  var yExt = filtered.lsv_iv_error_bps.length > 0 ?
    [Math.min.apply(null, filtered.lsv_iv_error_bps) * 1.5, Math.max.apply(null, filtered.lsv_iv_error_bps) * 1.5] : [-200, 200];
  traces.push({x: [0, 0], y: yExt, mode: 'lines',
               line: {color: 'grey', dash: 'dot', width: 1}, showlegend: false});

  var layout = {
    template: 'plotly_white',
    title: 'Bergomi IV Error (bp) vs Fwd Log-Moneyness  [' + filtered.strike.length + ' options, price >= $' + pMin.toFixed(0) + ']',
    xaxis: {title: 'Fwd Log-Moneyness ln(K/F)'}, yaxis: {title: 'IV Error (bp)'},
    height: 520, margin: {l: 60, r: 40, t: 50, b: 50},
    legend: {font: {size: 11}}
  };
  Plotly.react('diag-plot', traces, layout, {responsive: true});

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

renderPlot(0);
</script>
</body>
</html>"""

    return html_head + html_tabs + html_script


# ── Data prep for diagnostic tab ───────────────────────────────────────────

def prepare_diagnostic_data(df):
    valid = df.dropna(subset=["lsv_iv_error_bps", "lsv_vs_ssvi_pct",
                               "dupire_vs_ssvi_pct"]).copy()
    cats = classify_options(valid)
    return {
        "strike": valid["strike"].tolist(),
        "ttm": valid["ttm"].tolist(),
        "log_moneyness": valid["log_moneyness"].tolist(),
        "option_type": valid["option_type"].tolist(),
        "iv_ssvi": valid["iv_ssvi"].tolist(),
        "iv_lsv": valid["iv_lsv"].tolist(),
        "lsv_iv_error_bps": valid["lsv_iv_error_bps"].tolist(),
        "lsv_vs_ssvi_pct": valid["lsv_vs_ssvi_pct"].tolist(),
        "dupire_vs_ssvi_pct": valid["dupire_vs_ssvi_pct"].tolist(),
        "dupire_price": valid["dupire_price"].tolist(),
        "category": cats.tolist(),
    }


# ── Tab descriptions ───────────────────────────────────────────────────────

TAB_DESCRIPTIONS = {
    "Leverage Surface": "3D view of the calibrated Bergomi leverage function σ(t, S). Rotate and zoom.",
    "Leverage Slices": "Leverage σ(t, S) at selected maturity slices. Compare shape across time.",
    "Forward Variance": "Forward variance curve ξ<sup>T</sup><sub>0</sub> extracted from the SSVI surface via variance swap vol integration and parametric fit.",
    "Surfaces Overview": "Side-by-side: IV surface, Dupire local vol, and Bergomi leverage surface.",
    "Bergomi + Params": "Bergomi two-factor model parameters (Wang Set II), forward variance fit, and validation summary.",
    "IV Error (Moneyness)": "IV repricing error (bp) vs forward log-moneyness, coloured by option type.",
    "IV Error (TTM)": "IV repricing error (bp) vs time-to-maturity.",
    "IV Error Histogram": "Distribution of IV repricing errors (bp).",
    "Price Scatter": "LSV price vs market and Dupire prices.",
    "Error Diagnostic": "Interactive IV error explorer with adjustable strike, TTM, and min price filters.",
    "Summary Stats": "Summary statistics table.",
}


def main():
    (df, leverage, spot_grid, time_grid, bergomi, particle_log,
     val_summary, market, fwd_var, vs_vol, vs_vol_fitted, ttm_grid_iv,
     fwd_var_fit, iv_surface, log_m_grid, local_vol) = load_data()

    S = market["S"]

    tab_names = [
        "Leverage Surface",
        "Leverage Slices",
        "Forward Variance",
        "Surfaces Overview",
        "Bergomi + Params",
        "IV Error (Moneyness)",
        "IV Error (TTM)",
        "IV Error Histogram",
        "Price Scatter",
        "Error Diagnostic",
        "Summary Stats",
    ]

    diag_tab_idx = tab_names.index("Error Diagnostic")

    figures = [
        make_leverage_surface(leverage, spot_grid, time_grid, S),
        make_leverage_slices(leverage, spot_grid, time_grid, S),
        make_fwd_var_plot(ttm_grid_iv, vs_vol, vs_vol_fitted, fwd_var, fwd_var_fit),
        make_surfaces_comparison(leverage, spot_grid, time_grid, iv_surface,
                                 local_vol, log_m_grid, ttm_grid_iv, S),
        make_bergomi_summary(bergomi, particle_log, val_summary, market, fwd_var_fit),
        make_error_by_moneyness(df),
        make_error_by_ttm(df),
        make_error_hist(df),
        make_price_scatter(df),
        None,  # diagnostic tab — custom HTML
        make_summary_table(df, val_summary, market),
    ]

    diag_data = prepare_diagnostic_data(df)

    html = build_html(figures, tab_names, TAB_DESCRIPTIONS,
                      diagnostic_data=diag_data, diagnostic_tab_idx=diag_tab_idx)
    os.makedirs(DIR_PLOTS, exist_ok=True)
    out_path = os.path.join(DIR_PLOTS, "lsv_bergomi_explorer.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path} — open in your browser")


if __name__ == "__main__":
    main()

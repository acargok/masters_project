#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dupire Explorer — interactive Plotly dashboard.

Single tabbed HTML for exploring the Dupire local vol surface and MC
repricing, from dupire_local_vol.py + iv_surface/ outputs.
Run `python dupire_explorer.py` -> plots/dupire_explorer.html.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

# Directories (project root, matching dupire_local_vol.py)
DIR_DATA   = "dupire_vol/data"
DIR_PLOTS  = "dupire_vol/plots"
DIR_ARRAYS = "dupire_vol/arrays"
IV_DIR_ARRAYS = "iv_surface/arrays"
IV_DIR_DATA   = "iv_surface/data"

# 4-category colours (matches dupire_local_vol.py)
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
    """Load pre-computed data from both pipelines."""
    local_vol = np.load(os.path.join(DIR_ARRAYS, "local_vol_surface.npy"))
    mask = np.load(os.path.join(DIR_ARRAYS, "local_vol_mask.npy"))

    with open(os.path.join(DIR_DATA, "market_params.json")) as f:
        params = json.load(f)

    repricing = pd.read_csv(os.path.join(DIR_DATA, "repricing_errors.csv"))

    iv_surface = np.load(os.path.join(IV_DIR_ARRAYS, "iv_surface.npy"))
    log_m_grid = np.load(os.path.join(IV_DIR_ARRAYS, "log_m_grid.npy"))
    ttm_grid = np.load(os.path.join(IV_DIR_ARRAYS, "ttm_grid.npy"))

    return (local_vol, mask, iv_surface,
            log_m_grid, ttm_grid, repricing, params)


def classify_options(df):
    """Classify into OTM/ITM Call/Put by k = log(K/F)."""
    cat = pd.Series("", index=df.index)
    is_call = df["option_type"] == "call"
    # k >= 0 (K >= F): call OTM, put ITM
    is_otm = ((is_call) & (df["fwd_log_m"] >= 0.0)) | ((~is_call) & (df["fwd_log_m"] < 0.0))
    cat[is_call & is_otm]  = "OTM Call"
    cat[is_call & ~is_otm] = "ITM Call"
    cat[~is_call & is_otm] = "OTM Put"
    cat[~is_call & ~is_otm] = "ITM Put"
    return cat


def _add_cat_traces(fig, df, x_col, y_col, customdata_cols=None,
                    hovertemplate=None, row=None, col=None, size=5, opacity=0.6):
    """Add one scatter trace per category."""
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


# Tab 1: 3D local vol surface

def make_local_vol_surface(local_vol, log_m_grid, ttm_grid):
    fig = go.Figure(go.Surface(
        x=log_m_grid, y=ttm_grid, z=local_vol.T,
        colorscale="Inferno", colorbar_title="Local Vol",
        contours=dict(
            x=dict(show=True, color="rgba(0,0,0,0.12)", width=1),
            y=dict(show=True, color="rgba(0,0,0,0.12)", width=1),
        ),
        hovertemplate=(
            "Log-moneyness: %{x:.4f}<br>"
            "TTM: %{y:.3f} yr<br>"
            "Local Vol: %{z:.4f}<extra></extra>"
        ),
    ))
    x_range = log_m_grid.max() - log_m_grid.min()
    y_range = ttm_grid.max() - ttm_grid.min()
    z_range = local_vol.max() - local_vol.min()
    mx = max(x_range, y_range)
    fig.update_layout(
        title="Dupire Local Volatility Surface",
        scene=dict(
            xaxis_title="Log-Moneyness k = log(K/F)",
            yaxis_title="TTM (years)",
            zaxis_title="Local Vol",
            aspectmode="manual",
            aspectratio=dict(x=x_range/mx, y=y_range/mx, z=z_range/mx*1.5),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# Tab 2: IV vs local vol side-by-side

def make_iv_vs_local_vol(iv_surface, local_vol, log_m_grid, ttm_grid):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["Implied Volatility Surface", "Dupire Local Volatility Surface"],
    )
    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid, z=iv_surface.T,
        colorscale="Viridis", colorbar=dict(x=0.42, title="IV"),
        hovertemplate="x: %{x:.4f}<br>TTM: %{y:.3f}<br>IV: %{z:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid, z=local_vol.T,
        colorscale="Inferno", colorbar=dict(x=1.02, title="LV"),
        hovertemplate="x: %{x:.4f}<br>TTM: %{y:.3f}<br>LV: %{z:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(
        title="Implied Volatility vs Local Volatility",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# Tab 3: local vol smiles by maturity

def make_local_vol_smiles(local_vol, log_m_grid, ttm_grid):
    n_slices = 10
    indices = np.linspace(0, len(ttm_grid) - 1, n_slices, dtype=int)

    fig = go.Figure()
    for idx in indices:
        ttm = ttm_grid[idx]
        lv_slice = local_vol[:, idx]
        fig.add_trace(go.Scatter(
            x=log_m_grid, y=lv_slice,
            mode="lines", name=f"TTM={ttm:.3f}y ({ttm*365:.0f}d)",
            hovertemplate="k=%{x:.4f}<br>LV=%{y:.4f}<extra></extra>",
        ))
    fig.add_vline(x=0.0, line_dash="dash", line_color="grey", opacity=0.5,
                  annotation_text="ATM (k=0)")
    fig.update_layout(
        title="Local Volatility Smiles by Maturity (Forward Log-Moneyness Space)",
        xaxis_title="k = log(K/F)",
        yaxis_title="Local Volatility",
    )
    return fig


# Tab 4: MC vs SSVI scatter (all + liquid)

def make_mc_vs_ssvi_scatter(df):
    liquid = df[df["ssvi_price"] >= 10.0]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["All Options (log scale)",
                                        "Liquid Options (mid >= $10)"])
    hover = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "SSVI: $%{customdata[3]:.2f}<br>"
        "MC: $%{customdata[4]:.2f}<br>"
        "Error: %{customdata[5]:.2f}%<extra></extra>"
    )
    cdata = ["strike", "ttm", "option_type", "ssvi_price", "mc_price", "price_error_pct"]

    # All, log
    _add_cat_traces(fig, df, "ssvi_price", "mc_price",
                    customdata_cols=cdata, hovertemplate=hover,
                    row=1, col=1, size=4, opacity=0.4)
    lo = max(0.01, min(df["ssvi_price"].min(), df["mc_price"].min()))
    hi = max(df["ssvi_price"].max(), df["mc_price"].max()) * 1.1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="black", dash="dash"),
                             name="y=x", showlegend=False), row=1, col=1)
    fig.update_xaxes(type="log", title_text="SSVI BS Price ($)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="MC Price ($)", row=1, col=1)

    # Liquid, linear
    if len(liquid) > 0:
        _add_cat_traces(fig, liquid, "ssvi_price", "mc_price",
                        customdata_cols=cdata, hovertemplate=hover,
                        row=1, col=2, size=4, opacity=0.5)
        hi2 = max(liquid["ssvi_price"].max(), liquid["mc_price"].max()) * 1.05
        fig.add_trace(go.Scatter(x=[0, hi2], y=[0, hi2], mode="lines",
                                 line=dict(color="black", dash="dash"),
                                 name="y=x", showlegend=False), row=1, col=2)
        ss_res = ((liquid["mc_price"] - liquid["ssvi_price"])**2).sum()
        ss_tot = ((liquid["ssvi_price"] - liquid["ssvi_price"].mean())**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = liquid["price_error"].abs().mean()
        calls_l = liquid[liquid["option_type"] == "call"]
        puts_l = liquid[liquid["option_type"] == "put"]
        call_mae = calls_l["price_error"].abs().mean() if len(calls_l) > 0 else 0
        put_mae = puts_l["price_error"].abs().mean() if len(puts_l) > 0 else 0
        fig.add_annotation(
            text=(f"R²={r2:.4f}  MAE=${mae:.2f}<br>"
                  f"Call MAE=${call_mae:.2f}  Put MAE=${put_mae:.2f}"),
            xref="x2", yref="y2", x=0.05*hi2, y=0.92*hi2,
            showarrow=False, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="grey",
        )
    fig.update_xaxes(title_text="SSVI BS Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="MC Price ($)", row=1, col=2)
    fig.update_layout(title="MC Repriced vs SSVI Prices", height=550)
    return fig


# Tab 5: absolute error ($) vs TTM

def make_abs_error(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Absolute Error vs SSVI Price (all options)",
                                        "Absolute Error Distribution"])
    hover = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Error: $%{y:.2f}<extra></extra>"
    )
    cdata = ["strike", "ttm", "option_type"]

    if len(df) > 0:
        _add_cat_traces(fig, df, "ssvi_price", "price_error",
                        customdata_cols=cdata, hovertemplate=hover,
                        row=1, col=1, size=4, opacity=0.45)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

        # Histogram by category
        cats = classify_options(df)
        for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
            vals = df.loc[cats == label, "price_error"].dropna()
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(
                x=vals, name=f"{label} ({len(vals)})", legendgroup=label,
                marker_color=CAT_COLORS[label], opacity=0.6,
                nbinsx=40, showlegend=False,
            ), row=1, col=2)

        # Mean lines
        all_me = df["price_error"].mean()
        call_me = df[df["option_type"]=="call"]["price_error"].mean()
        put_me = df[df["option_type"]=="put"]["price_error"].mean()
        for val, color, name in [
            (all_me, "red", f"All mean=${all_me:.2f}"),
            (call_me, "#1f77b4", f"Call mean=${call_me:.2f}"),
            (put_me, "#d62728", f"Put mean=${put_me:.2f}"),
        ]:
            fig.add_vline(x=val, line_color=color, line_width=2,
                          annotation_text=name, annotation_font_size=10,
                          row=1, col=2)

    fig.update_xaxes(title_text="SSVI BS Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Error: MC - SSVI ($)", row=1, col=1)
    fig.update_xaxes(title_text="Absolute Error ($)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(title="Absolute Repricing Error ($)")
    return fig


# Tab 6: percentage error (%) vs TTM

def make_pct_error(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Percentage Error vs SSVI Price (all options)",
                                        "Percentage Error Distribution"])
    hover = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Error: %{y:.2f}%<extra></extra>"
    )
    cdata = ["strike", "ttm", "option_type"]

    if len(df) > 0:
        all_pct = df.dropna(subset=["price_error_pct"])
        _add_cat_traces(fig, all_pct, "ssvi_price", "price_error_pct",
                        customdata_cols=cdata, hovertemplate=hover,
                        row=1, col=1, size=4, opacity=0.45)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

        cats = classify_options(df)
        for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
            vals = df.loc[cats == label, "price_error_pct"].dropna()
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(
                x=vals, name=f"{label} ({len(vals)})", legendgroup=label,
                marker_color=CAT_COLORS[label], opacity=0.6,
                nbinsx=40, showlegend=False,
            ), row=1, col=2)

        all_me = df["price_error_pct"].dropna().mean()
        call_me = df[df["option_type"]=="call"]["price_error_pct"].dropna().mean()
        put_me = df[df["option_type"]=="put"]["price_error_pct"].dropna().mean()
        for val, color, name in [
            (all_me, "red", f"All mean={all_me:.2f}%"),
            (call_me, "#1f77b4", f"Call mean={call_me:.2f}%"),
            (put_me, "#d62728", f"Put mean={put_me:.2f}%"),
        ]:
            fig.add_vline(x=val, line_color=color, line_width=2,
                          annotation_text=name, annotation_font_size=10,
                          row=1, col=2)

    fig.update_xaxes(title_text="SSVI BS Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Error (%)", row=1, col=1)
    fig.update_xaxes(title_text="Percentage Error (%)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(title="Percentage Repricing Error (%)")
    return fig


# Tab 7: error vs moneyness

def make_error_vs_moneyness(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Abs Error ($) vs Forward Log-Moneyness",
                                        "Pct Error (%) vs Forward Log-Moneyness"])
    hover_abs = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Error: $%{y:.2f}<extra></extra>"
    )
    hover_pct = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Error: %{y:.2f}%<extra></extra>"
    )
    cdata = ["strike", "ttm", "option_type"]

    if len(df) > 0:
        _add_cat_traces(fig, df, "fwd_log_m", "price_error",
                        customdata_cols=cdata, hovertemplate=hover_abs,
                        row=1, col=1, size=4, opacity=0.45)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

        all_pct = df.dropna(subset=["price_error_pct"])
        _add_cat_traces(fig, all_pct, "fwd_log_m", "price_error_pct",
                        customdata_cols=cdata, hovertemplate=hover_pct,
                        row=1, col=2, size=4, opacity=0.45)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

        fig.add_vline(x=0.0, line_dash="dash", line_color="grey", opacity=0.5,
                      annotation_text="ATM", row=1, col=1)
        fig.add_vline(x=0.0, line_dash="dash", line_color="grey", opacity=0.5,
                      annotation_text="ATM", row=1, col=2)

    fig.update_xaxes(title_text="k = log(K/F)", row=1, col=1)
    fig.update_yaxes(title_text="Error ($)", row=1, col=1)
    fig.update_xaxes(title_text="k = log(K/F)", row=1, col=2)
    fig.update_yaxes(title_text="Error (%)", row=1, col=2)
    fig.update_layout(title="Repricing Error vs Forward Log-Moneyness")
    return fig


# Tab 8: percentage error vs liquidity

def make_error_vs_liquidity(df):
    """Percentage error vs liquidity (volume and open interest)."""
    # volume/openInterest live in the source CSV
    src_path = os.path.join(IV_DIR_DATA, "spx_iv_data.csv")
    src = pd.read_csv(src_path)

    # Merge on strike + option_type + rounded TTM (float precision differs)
    if "volume" not in src.columns and "openInterest" not in src.columns:
        fig = go.Figure()
        fig.add_annotation(text="No volume or openInterest data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16))
        return fig

    liq_cols = ["strike", "option_type", "volume", "openInterest"]
    available = [c for c in liq_cols if c in src.columns]
    src_merge = src[available + ["ttm"]].copy()
    src_merge["ttm_round"] = src_merge["ttm"].round(4)
    df_merge = df.copy()
    df_merge["ttm_round"] = df_merge["ttm"].round(4)
    merged = df_merge.merge(
        src_merge.drop(columns=["ttm"]),
        on=["strike", "ttm_round", "option_type"], how="left"
    ).drop(columns=["ttm_round"])

    # NaN volume/OI -> 0
    if "volume" in merged.columns:
        merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce").fillna(0)
    if "openInterest" in merged.columns:
        merged["openInterest"] = pd.to_numeric(merged["openInterest"], errors="coerce").fillna(0)

    liquid = merged[merged["ssvi_price"] >= 10.0].dropna(subset=["price_error_pct"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["% Error vs Volume", "% Error vs Open Interest"])

    hover = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}  M=%{customdata[3]:.3f}<br>"
        "Price: $%{customdata[4]:.2f}<br>"
        "Error: %{y:.2f}%<extra></extra>"
    )
    cdata = ["strike", "ttm", "option_type", "fwd_log_m", "ssvi_price"]

    # Volume
    if "volume" in liquid.columns:
        has_vol = liquid[liquid["volume"] > 0]
        if len(has_vol) > 0:
            _add_cat_traces(fig, has_vol, "volume", "price_error_pct",
                            customdata_cols=cdata, hovertemplate=hover,
                            row=1, col=1, size=4, opacity=0.5)
            fig.add_trace(go.Scatter(
                x=[has_vol["volume"].min(), has_vol["volume"].max()],
                y=[0, 0], mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False), row=1, col=1)
        fig.update_xaxes(type="log", title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="Error (%)", row=1, col=1)

    # Open interest
    if "openInterest" in liquid.columns:
        has_oi = liquid[liquid["openInterest"] > 0]
        if len(has_oi) > 0:
            _add_cat_traces(fig, has_oi, "openInterest", "price_error_pct",
                            customdata_cols=cdata, hovertemplate=hover,
                            row=1, col=2, size=4, opacity=0.5)
            fig.add_trace(go.Scatter(
                x=[has_oi["openInterest"].min(), has_oi["openInterest"].max()],
                y=[0, 0], mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False), row=1, col=2)
        fig.update_xaxes(type="log", title_text="Open Interest", row=1, col=2)
    fig.update_yaxes(title_text="Error (%)", row=1, col=2)

    fig.update_layout(title="Percentage Repricing Error vs Liquidity (liquid, mid >= $10)")
    return fig


# Tab 9: error diagnostic (interactive filtering)

def prepare_diagnostic_data(df):
    """Repricing data as dict-of-lists for the interactive diagnostic tab."""
    cols = ["strike", "ttm", "option_type", "fwd_log_m",
            "ssvi_price", "mc_price", "price_error", "price_error_pct"]
    clean = df[cols].dropna().copy()
    clean["category"] = classify_options(clean).values
    return clean.to_dict(orient="list")


# Tab 10: reliability mask

def make_reliability_mask(mask, log_m_grid, ttm_grid):
    fig = go.Figure(go.Heatmap(
        x=ttm_grid, y=log_m_grid, z=mask.astype(float),
        colorscale=[[0, "#d62728"], [1, "#2ca02c"]],
        zmin=0, zmax=1,
        colorbar=dict(
            tickvals=[0.25, 0.75], ticktext=["Unreliable", "Reliable"],
            title="Status",
        ),
        hovertemplate=(
            "TTM: %{x:.3f} yr<br>"
            "Log-moneyness: %{y:.4f}<br>"
            "Reliable: %{z:.0f}<extra></extra>"
        ),
    ))
    pct = mask.sum() / mask.size * 100
    fig.update_layout(
        title=f"Dupire Local Vol Reliability Mask ({pct:.1f}% reliable)",
        xaxis_title="TTM (years)",
        yaxis_title="k = log(K/F)",
    )
    return fig


# Summary statistics tab

def make_summary_table(df, params):
    liquid = df[df["ssvi_price"] >= 10.0]
    calls = df[df["option_type"] == "call"]
    puts = df[df["option_type"] == "put"]
    liq_calls = liquid[liquid["option_type"] == "call"]
    liq_puts = liquid[liquid["option_type"] == "put"]

    has_iv = "iv_error_bps" in df.columns

    def stats_row(name, subset):
        if len(subset) == 0:
            base = [name, 0, "—", "—", "—", "—"]
            return base + (["—", "—"] if has_iv else [])
        me = subset["price_error"].mean()
        mae = subset["price_error"].abs().mean()
        me_pct = subset["price_error_pct"].dropna().mean()
        mae_pct = subset["price_error_pct"].dropna().abs().mean()
        base = [name, len(subset),
                f"${me:+.2f}", f"${mae:.2f}",
                f"{me_pct:+.2f}%", f"{mae_pct:.2f}%"]
        if has_iv:
            iv_v = subset.dropna(subset=["iv_error_bps"])
            if len(iv_v) == 0:
                base += ["—", "—"]
            else:
                iv_me  = iv_v["iv_error_bps"].mean()
                iv_mae = iv_v["iv_error_bps"].abs().mean()
                base += [f"{iv_me:+.1f} bp", f"{iv_mae:.1f} bp"]
        return base

    rows = [
        stats_row("All", df),
        stats_row("Liquid (mid>=$10)", liquid),
        stats_row("All Calls", calls),
        stats_row("All Puts", puts),
        stats_row("Liquid Calls", liq_calls),
        stats_row("Liquid Puts", liq_puts),
        stats_row("OTM Calls", df[classify_options(df) == "OTM Call"]),
        stats_row("OTM Puts", df[classify_options(df) == "OTM Put"]),
        stats_row("ITM Calls", df[classify_options(df) == "ITM Call"]),
        stats_row("ITM Puts", df[classify_options(df) == "ITM Put"]),
    ]

    headers = ["Subset", "Count", "ME ($)", "MAE ($)", "ME (%)", "MAE (%)"]
    if has_iv:
        headers += ["IV ME (bp)", "IV MAE (bp)"]
    header_vals = [h for h in headers]
    cell_vals = [[r[i] for r in rows] for i in range(len(headers))]

    fig = go.Figure(go.Table(
        header=dict(values=header_vals,
                    fill_color="#4472C4", font=dict(color="white", size=13),
                    align="center"),
        cells=dict(values=cell_vals,
                   fill_color=[["#f0f4ff", "white"] * (len(rows) // 2 + 1)][:len(rows)],
                   font=dict(size=12), align="center", height=28),
    ))
    fig.update_layout(
        title=(f"MC Repricing Summary  |  S={params['S']:,.0f}  "
               f"r={params['r']:.4f}  q={params['q']:.4f}  date={params['date']}"),
        height=450,
    )
    return fig


# IV error (bp) tab

def make_iv_error(df: pd.DataFrame) -> go.Figure:
    """
    Three-panel IV error (MC IV − market IV) in bp: vs k, vs TTM, distribution
    with per-type mean lines.

    Needs iv_error_bps (from dupire_local_vol.py); shows a placeholder if absent.
    """
    if "iv_error_bps" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="iv_error_bps column not found.<br>"
                 "Re-run dupire_local_vol.py to generate IV errors.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig.update_layout(title="IV Error (bp) — data not available")
        return fig

    valid = df.dropna(subset=["iv_error_bps"]).copy()
    cats  = classify_options(valid)
    valid["category"] = cats.values

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "IV Error vs Moneyness (all options)",
            "IV Error vs TTM",
            "IV Error Distribution",
        ],
    )

    hover_scatter = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "Mkt IV: %{customdata[3]:.4f}   MC IV: %{customdata[4]:.4f}<br>"
        "IV Error: %{y:+.1f} bp<extra></extra>"
    )
    cdata_cols = ["strike", "ttm", "option_type", "iv_ssvi", "iv_mc"]

    for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
        sub = valid[valid["category"] == label]
        if len(sub) == 0:
            continue
        color  = CAT_COLORS[label]
        symbol = CAT_SYMBOLS[label]

        # vs moneyness
        fig.add_trace(go.Scatter(
            x=sub["fwd_log_m"], y=sub["iv_error_bps"],
            mode="markers",
            marker=dict(size=4, color=color, symbol=symbol, opacity=0.55),
            name=f"{label} ({len(sub)})", legendgroup=label,
            customdata=sub[cdata_cols].values,
            hovertemplate=hover_scatter,
        ), row=1, col=1)

        # vs TTM
        fig.add_trace(go.Scatter(
            x=sub["ttm"], y=sub["iv_error_bps"],
            mode="markers",
            marker=dict(size=4, color=color, symbol=symbol, opacity=0.55),
            name=label, legendgroup=label, showlegend=False,
            customdata=sub[cdata_cols].values,
            hovertemplate=hover_scatter,
        ), row=1, col=2)

        # histogram
        fig.add_trace(go.Histogram(
            x=sub["iv_error_bps"], nbinsx=50,
            marker_color=color, opacity=0.6,
            name=label, legendgroup=label, showlegend=False,
            hovertemplate="IV Error: %{x:.1f} bp<br>Count: %{y}<extra></extra>",
        ), row=1, col=3)

    # Zero lines
    for c in [1, 2]:
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=c)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=3)

    # Call/put mean lines on histogram
    for opt_type, color, label in [("call", "#1f77b4", "Call mean"),
                                    ("put",  "#d62728", "Put mean")]:
        sub = valid[valid["option_type"] == opt_type]["iv_error_bps"].dropna()
        if len(sub) > 0:
            fig.add_vline(x=sub.mean(), line_dash="dot", line_color=color,
                          opacity=0.9, row=1, col=3,
                          annotation_text=f"{label}: {sub.mean():+.1f}bp",
                          annotation_position="top right")

    # Summary annotation (left panel)
    me   = valid["iv_error_bps"].mean()
    mae  = valid["iv_error_bps"].abs().mean()
    rmse = float(np.sqrt((valid["iv_error_bps"]**2).mean()))
    n_inv = len(valid)
    n_tot = len(df)
    fig.add_annotation(
        text=(f"n={n_inv}/{n_tot} inverted<br>"
              f"ME={me:+.1f} bp<br>"
              f"MAE={mae:.1f} bp<br>"
              f"RMSE={rmse:.1f} bp"),
        xref="x", yref="paper",
        x=valid["fwd_log_m"].max() * 0.98, y=0.97,
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="grey",
        align="right",
    )

    fig.update_xaxes(title_text="k = log(K/F)",     row=1, col=1)
    fig.update_yaxes(title_text="IV Error (bp)",     row=1, col=1)
    fig.update_xaxes(title_text="TTM (years)",       row=1, col=2)
    fig.update_yaxes(title_text="IV Error (bp)",     row=1, col=2)
    fig.update_xaxes(title_text="IV Error (bp)",     row=1, col=3)
    fig.update_yaxes(title_text="Count",             row=1, col=3)

    fig.update_layout(
        title=(f"MC Implied Vol Error  —  ME={me:+.1f} bp  "
               f"MAE={mae:.1f} bp  RMSE={rmse:.1f} bp  "
               f"({n_inv}/{n_tot} options inverted)"),
        barmode="overlay",
        height=560,
    )
    return fig


# Tab 12: Dupire MC vs SSVI repricing

def _bs_price_fwd(F: np.ndarray, K: np.ndarray, T: np.ndarray,
                  r: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorised forward BSM: price = e^{-rT}[F·N(d1) − K·N(d2)]."""
    valid = (sigma > 1e-6) & (T > 0)
    price = np.zeros(len(F))
    v = valid
    sqrtT = np.sqrt(T[v])
    d1 = (np.log(F[v] / K[v]) + 0.5 * sigma[v] ** 2 * T[v]) / (sigma[v] * sqrtT)
    d2 = d1 - sigma[v] * sqrtT
    disc = np.exp(-r * T[v])
    call_px = disc * (F[v] * norm.cdf(d1) - K[v] * norm.cdf(d2))
    put_px  = disc * (K[v] * norm.cdf(-d2) - F[v] * norm.cdf(-d1))
    price[v] = np.where(is_call[v], call_px, put_px)
    return price


def make_dupire_vs_ssvi(repricing: pd.DataFrame, iv_surface: np.ndarray,
                        log_m_grid: np.ndarray, ttm_grid: np.ndarray,
                        params: dict) -> go.Figure:
    """
    Dupire MC vs SSVI-smoothed prices for all priced options.

    SSVI price = forward BSM at the iv_surface IV interpolated at each
    option's (fwd_log_m, ttm). Error = (mc − ssvi)/ssvi × 100%.
    """
    r = params["r"]

    # SSVI IV at every option's (k, T)
    interp_fn = RegularGridInterpolator(
        (log_m_grid, ttm_grid), iv_surface,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    pts = repricing[["fwd_log_m", "ttm"]].values
    iv_ssvi = interp_fn(pts)

    # BSM price using the forward from the repricing file
    F       = repricing["forward"].values
    K       = repricing["strike"].values
    T       = repricing["ttm"].values
    is_call = (repricing["option_type"] == "call").values
    ssvi_price = _bs_price_fwd(F, K, T, r, iv_ssvi, is_call)

    df = repricing.copy()
    df["ssvi_price"] = ssvi_price
    df["iv_ssvi"]    = iv_ssvi

    # Drop options outside the IV surface grid (interpolation failed)
    df = df[(df["ssvi_price"] > 0) & df["ssvi_price"].notna()].copy()
    df["err_pct"] = (df["mc_price"] - df["ssvi_price"]) / df["ssvi_price"] * 100.0

    cats = classify_options(df)
    df["category"] = cats.values

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Dupire MC vs SSVI Price",
                        "(MC − SSVI) / SSVI %  Distribution",
                        "Error % vs SSVI Price"],
    )

    hover_scatter = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "SSVI IV: %{customdata[3]:.4f}<br>"
        "SSVI: $%{x:.2f}  MC: $%{y:.2f}<br>"
        "Err: %{customdata[4]:+.2f}%"
        "<extra></extra>"
    )
    cdata_scatter = ["strike", "ttm", "option_type", "iv_ssvi", "err_pct"]

    hover_err = (
        "K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>"
        "Type: %{customdata[2]}<br>"
        "SSVI: $%{x:.2f}<br>"
        "Err: %{y:+.2f}%"
        "<extra></extra>"
    )
    cdata_err = ["strike", "ttm", "option_type"]

    for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
        sub = df[df["category"] == label]
        if len(sub) == 0:
            continue
        color  = CAT_COLORS[label]
        symbol = CAT_SYMBOLS[label]

        # price scatter
        fig.add_trace(go.Scatter(
            x=sub["ssvi_price"], y=sub["mc_price"],
            mode="markers",
            marker=dict(size=4, color=color, symbol=symbol, opacity=0.55),
            name=f"{label} ({len(sub)})",
            legendgroup=label,
            customdata=sub[cdata_scatter].values,
            hovertemplate=hover_scatter,
        ), row=1, col=1)

        # error vs SSVI price
        fig.add_trace(go.Scatter(
            x=sub["ssvi_price"], y=sub["err_pct"],
            mode="markers",
            marker=dict(size=4, color=color, symbol=symbol, opacity=0.55),
            name=label, legendgroup=label, showlegend=False,
            customdata=sub[cdata_err].values,
            hovertemplate=hover_err,
        ), row=1, col=3)

    # y = x diagonal
    lo = max(0.01, min(df["ssvi_price"].min(), df["mc_price"].min()) * 0.95)
    hi = max(df["ssvi_price"].max(), df["mc_price"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="black", dash="dash", width=1),
        name="y = x", showlegend=False,
    ), row=1, col=1)

    # histogram by category
    for label in ["OTM Call", "ITM Call", "OTM Put", "ITM Put"]:
        sub = df[df["category"] == label]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=sub["err_pct"], nbinsx=50,
            marker_color=CAT_COLORS[label], opacity=0.6,
            name=label, legendgroup=label, showlegend=False,
            hovertemplate="Error: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ), row=1, col=2)

    me   = df["err_pct"].mean()
    mae  = df["err_pct"].abs().mean()
    rmse = float(np.sqrt((df["err_pct"] ** 2).mean()))

    fig.add_vline(x=0,  line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
    fig.add_vline(x=me, line_dash="dot",  line_color="red",   opacity=0.8,
                  annotation_text=f"ME={me:+.2f}%", annotation_position="top right",
                  row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=1, col=3)

    fig.update_xaxes(title_text="SSVI Price ($)",           row=1, col=1)
    fig.update_yaxes(title_text="Dupire MC Price ($)",      row=1, col=1)
    fig.update_xaxes(title_text="(MC − SSVI) / SSVI %",    row=1, col=2)
    fig.update_yaxes(title_text="Count",                    row=1, col=2)
    fig.update_xaxes(title_text="SSVI Price ($)",           row=1, col=3)
    fig.update_yaxes(title_text="(MC − SSVI) / SSVI %",    row=1, col=3)

    fig.update_layout(
        title=(
            f"Dupire MC vs SSVI Repricing  ({len(df)} options)  —  "
            f"MAE={mae:.2f}%  RMSE={rmse:.2f}%  ME={me:+.2f}%"
        ),
        barmode="overlay",
        height=560,
    )
    return fig


# HTML assembly

TAB_DESCRIPTIONS = {
    "Local Vol Surface": (
        "The Dupire local volatility surface computed from the IV surface via "
        "the Dupire formula. Local vol is the instantaneous diffusion coefficient "
        "at each (strike, maturity) point. Rotate and zoom to inspect the shape."
    ),
    "IV vs Local Vol": (
        "Side-by-side comparison: the smooth implied vol surface (input) vs the "
        "local vol surface (output). Local vol is typically spikier because it "
        "captures instantaneous dynamics rather than lifetime-averaged volatility."
    ),
    "LV Smiles": (
        "Local volatility smile slices at selected maturities, plotted in strike "
        "space. Compare with IV smiles — local vol smiles are steeper and more "
        "peaked, reflecting the skew dynamics the Dupire model captures."
    ),
    "MC vs SSVI": (
        "Monte Carlo repriced prices vs actual market mid prices. Left: all options "
        "on a log scale. Right: liquid options (mid >= $10) on a linear scale with "
        "R² and MAE statistics broken down by calls and puts."
    ),
    "Absolute Error ($)": (
        "Absolute repricing error (MC - Market) in dollars. Left: scatter vs TTM "
        "to see if errors are concentrated in specific maturities. Right: histogram "
        "with separate mean lines for calls (blue) and puts (red)."
    ),
    "Percentage Error (%)": (
        "Percentage repricing error for liquid options. This normalises by price, "
        "so cheap options don't dominate. Separate call/put mean lines reveal "
        "any systematic bias by option type."
    ),
    "Error vs Moneyness": (
        "Repricing errors plotted against forward log-moneyness k = log(K/F). "
        "ATM is at k=0. Left wing (k < 0) shows OTM put bias; right wing (k > 0) "
        "shows OTM call bias. Key diagnostic for skew calibration quality."
    ),
    "Error vs Liquidity": (
        "Percentage repricing error plotted against liquidity measures (volume and open "
        "interest) on log scales. This reveals whether repricing errors are concentrated "
        "in illiquid options — where bid-ask spreads are wide and mid prices are noisy — "
        "or whether they persist even for heavily traded contracts."
    ),
    "Error Diagnostic": (
        "Interactive diagnostic tool: filter by strike range and TTM range using "
        "double-ended sliders to isolate specific regions. The plot updates live, "
        "showing MC error vs market price with summary stats for the filtered subset. "
        "Use this to pinpoint which strikes and maturities drive the repricing bias."
    ),
    "Reliability Mask": (
        "Green = grid points where the Dupire formula produced a stable, positive "
        "local vol. Red = points where the denominator was too small or the result "
        "was clipped. Unreliable regions are typically at the boundaries."
    ),
    "Summary Stats": (
        "Table of mean error (ME) and mean absolute error (MAE) in both dollar and "
        "percentage terms, broken down by option type and moneyness category. "
        "This is the quantitative checkpoint for the Dupire surface quality."
    ),
    "IV Error (bp)": (
        "Implied volatility error: MC IV minus market IV, in basis points (1 bp = 0.01%). "
        "MC IV is obtained by inverting the BSM formula on each MC price. "
        "This is the Jacquier 'exact data' diagnostic: on synthetic SSVI input the errors "
        "should be ±5–20 bp (residual MC noise); on real market data larger errors reveal "
        "either genuine model misspecification or bid-ask noise. "
        "Left: errors vs moneyness expose skew-dependent bias. "
        "Middle: errors vs TTM show whether the pipeline degrades at short or long end. "
        "Right: distribution with separate call/put mean lines."
    ),
    "Dupire vs SSVI": (
        "How faithfully does the Dupire MC reprice the SSVI smoothed surface? "
        "Error = (MC price − SSVI price) / SSVI price. SSVI prices are computed "
        "by interpolating the IV surface grid at each option's (k, T) and pricing "
        "via forward-based BSM — covering all priced options, not just the validation "
        "sample. A tight distribution around zero means the Dupire derivation "
        "(finite-difference ∂w/∂T, PCHIP θ interpolation, MC discretisation) is "
        "numerically consistent with the SSVI input. Residual bias by moneyness or "
        "maturity isolates where the pipeline breaks down."
    ),
}


def build_html(figures, tab_names, descriptions, diagnostic_data=None, diagnostic_tab_idx=None):
    """Build one HTML file with CSS tabs and per-tab Plotly figures.

    diagnostic_data: dict-of-lists repricing data embedded as JSON for the
    client-side-filtered diagnostic tab (or None). diagnostic_tab_idx: its
    index in tab_names (custom HTML instead of a Plotly figure).
    """
    fig_json_list = []
    fig_idx_map = {}  # tab index -> figSpecs index
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
                f'  </div>'
                f'  <div id="diag-stats" style="margin-top:8px;font-size:13px;color:#555;"></div>'
                f'</div>'
                f'<div id="diag-plot" style="width:100%;height:560px;"></div>'
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
<title>Dupire Explorer</title>
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
<h1>Dupire Local Volatility Explorer</h1>
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
  if (sLo > sHi) { var tmp = sLo; sLo = sHi; sHi = tmp; }
  if (tLo > tHi) { var tmp = tLo; tLo = tHi; tHi = tmp; }

  document.getElementById('strike-lo-val').textContent = sLo.toFixed(0);
  document.getElementById('strike-hi-val').textContent = sHi.toFixed(0);
  document.getElementById('ttm-lo-val').textContent = tLo.toFixed(3);
  document.getElementById('ttm-hi-val').textContent = tHi.toFixed(3);

  // Filter data
  var n = diagData.strike.length;
  var filtered = {strike:[], ttm:[], option_type:[], fwd_log_m:[],
                  ssvi_price:[], mc_price:[], price_error:[], price_error_pct:[], category:[]};
  for (var i = 0; i < n; i++) {
    if (diagData.strike[i] >= sLo && diagData.strike[i] <= sHi &&
        diagData.ttm[i] >= tLo && diagData.ttm[i] <= tHi) {
      for (var k in filtered) filtered[k].push(diagData[k][i]);
    }
  }

  // Build traces by category
  var traces = [];
  var cats = ["OTM Call", "ITM Call", "OTM Put", "ITM Put"];
  var catCounts = {};
  for (var ci = 0; ci < cats.length; ci++) {
    var cat = cats[ci];
    var x = [], y = [], cd = [];
    var cnt = 0;
    for (var i = 0; i < filtered.strike.length; i++) {
      if (filtered.category[i] === cat) {
        x.push(filtered.ssvi_price[i]);
        y.push(filtered.price_error[i]);
        cd.push([filtered.strike[i], filtered.ttm[i], filtered.option_type[i],
                 filtered.mc_price[i], filtered.price_error_pct[i], filtered.fwd_log_m[i]]);
        cnt++;
      }
    }
    catCounts[cat] = cnt;
    if (cnt === 0) continue;
    traces.push({
      x: x, y: y, customdata: cd, mode: 'markers',
      marker: {size: 5, color: catColors[cat], symbol: catSymbols[cat], opacity: 0.7},
      name: cat + ' (' + cnt + ')',
      hovertemplate: 'K=%{customdata[0]:,.0f}  TTM=%{customdata[1]:.3f}<br>' +
                     'Type: %{customdata[2]}  k=%{customdata[5]:.4f}<br>' +
                     'SSVI: $%{x:.2f}  MC: $%{customdata[3]:.2f}<br>' +
                     'Error: $%{y:.2f} (%{customdata[4]:.2f}%)<extra></extra>'
    });
  }
  // y=0 line
  traces.push({x: [0, Math.max.apply(null, filtered.ssvi_price.length ? filtered.ssvi_price : [1])],
               y: [0, 0], mode: 'lines', line: {color: 'black', dash: 'dash', width: 1.2},
               showlegend: false});

  var layout = {
    template: 'plotly_white',
    title: 'Error Diagnostic: MC-SSVI ($) vs SSVI Price  [' + filtered.strike.length + ' options]',
    xaxis: {title: 'SSVI BS Price ($)'}, yaxis: {title: 'Error: MC - SSVI ($)'},
    height: 560, margin: {l: 60, r: 40, t: 50, b: 50},
    legend: {font: {size: 11}}
  };
  Plotly.react('diag-plot', traces, layout, {responsive: true});

  // Stats
  var nf = filtered.strike.length;
  if (nf === 0) {
    document.getElementById('diag-stats').innerHTML = '<b>No options in range.</b>';
    return;
  }
  var sumE = 0, sumAE = 0, sumEcall = 0, nCall = 0, sumEput = 0, nPut = 0;
  var sumPct = 0, nPct = 0;
  for (var i = 0; i < nf; i++) {
    sumE += filtered.price_error[i];
    sumAE += Math.abs(filtered.price_error[i]);
    if (filtered.option_type[i] === 'call') { sumEcall += filtered.price_error[i]; nCall++; }
    else { sumEput += filtered.price_error[i]; nPut++; }
    if (filtered.price_error_pct[i] != null) { sumPct += filtered.price_error_pct[i]; nPct++; }
  }
  var me = (sumE/nf).toFixed(2), mae = (sumAE/nf).toFixed(2);
  var cme = nCall ? (sumEcall/nCall).toFixed(2) : '—';
  var pme = nPut ? (sumEput/nPut).toFixed(2) : '—';
  var mePct = nPct ? (sumPct/nPct).toFixed(2) + '%' : '—';
  document.getElementById('diag-stats').innerHTML =
    '<b>' + nf + ' options</b> ('+nCall+' calls, '+nPut+' puts) &nbsp;|&nbsp; ' +
    'ME=$' + me + ' &nbsp; MAE=$' + mae + ' &nbsp; ME%=' + mePct +
    ' &nbsp;|&nbsp; <span style="color:#1f77b4">Call ME=$' + cme + '</span>' +
    ' &nbsp; <span style="color:#d62728">Put ME=$' + pme + '</span>';
}

// Render the first tab on load
renderPlot(0);
</script>
</body>
</html>"""

    return html_head + html_tabs + html_script


def main():
    (local_vol, mask, iv_surface,
     log_m_grid, ttm_grid, repricing, params) = load_data()

    S = params["S"]

    tab_names = [
        "Local Vol Surface",
        "IV vs Local Vol",
        "LV Smiles",
        "MC vs SSVI",
        "Absolute Error ($)",
        "Percentage Error (%)",
        "Error vs Moneyness",
        "Error vs Liquidity",
        "Error Diagnostic",
        "Reliability Mask",
        "Summary Stats",
        "IV Error (bp)",
        "Dupire vs SSVI",
    ]

    diag_tab_idx = tab_names.index("Error Diagnostic")

    figures = [
        make_local_vol_surface(local_vol, log_m_grid, ttm_grid),
        make_iv_vs_local_vol(iv_surface, local_vol, log_m_grid, ttm_grid),
        make_local_vol_smiles(local_vol, log_m_grid, ttm_grid),
        make_mc_vs_ssvi_scatter(repricing),
        make_abs_error(repricing),
        make_pct_error(repricing),
        make_error_vs_moneyness(repricing),
        make_error_vs_liquidity(repricing),
        None,  # diagnostic tab: custom HTML
        make_reliability_mask(mask, log_m_grid, ttm_grid),
        make_summary_table(repricing, params),
        make_iv_error(repricing),
        make_dupire_vs_ssvi(repricing, iv_surface, log_m_grid, ttm_grid, params),
    ]

    diag_data = prepare_diagnostic_data(repricing)

    html = build_html(figures, tab_names, TAB_DESCRIPTIONS,
                      diagnostic_data=diag_data, diagnostic_tab_idx=diag_tab_idx)
    os.makedirs(DIR_PLOTS, exist_ok=True)
    out_path = os.path.join(DIR_PLOTS, "dupire_explorer.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path} — open in your browser")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV Explorer — interactive Plotly dashboard for the SSVI fit. Builds a single
tabbed HTML (plots/iv_explorer.html) from the CSVs and .npy arrays produced by
iv_surface_ssvi.py. Tabs: market smiles, total-var smiles, SSVI fit, fitted
surface, θ(T) term structure, no-butterfly check, Dupire diagnostics,
validation, liquidity map, TV surface mesh.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from iv_surface_ssvi import DIR_ARRAYS, DIR_DATA, DIR_PLOTS


# Data loading

def load_data() -> dict:
    iv = pd.read_csv(os.path.join(DIR_DATA, "spx_iv_data.csv"))
    params = pd.read_csv(os.path.join(DIR_DATA, "ssvi_params.csv"))
    val = pd.read_csv(os.path.join(DIR_DATA, "validation_results.csv"))
    fwd = pd.read_csv(os.path.join(DIR_DATA, "implied_forwards.csv"))

    # Compute fwd_log_m and total_var if not already present on disk.
    if "fwd_log_m" not in iv.columns:
        fwd_map = dict(zip(fwd["expiry"], fwd["forward"]))
        iv["fwd_log_m"] = np.log(iv["strike"] / iv["expiry"].map(fwd_map))
    if "total_var" not in iv.columns:
        iv["total_var"] = iv["iv"] ** 2 * iv["ttm"]

    ttm_grid       = np.load(os.path.join(DIR_ARRAYS, "ttm_grid.npy"))
    log_m_grid     = np.load(os.path.join(DIR_ARRAYS, "log_m_grid.npy"))
    iv_surface     = np.load(os.path.join(DIR_ARRAYS, "iv_surface.npy"))
    tv_surface     = np.load(os.path.join(DIR_ARRAYS, "total_var_surface.npy"))

    try:
        dupire_g  = np.load(os.path.join(DIR_ARRAYS, "dupire_g_surface.npy"))
        dupire_lv = np.load(os.path.join(DIR_ARRAYS, "dupire_local_var.npy"))
    except FileNotFoundError:
        dupire_g  = None
        dupire_lv = None

    return dict(
        iv=iv, params=params, val=val, fwd=fwd,
        ttm_grid=ttm_grid, log_m_grid=log_m_grid,
        iv_surface=iv_surface, tv_surface=tv_surface,
        dupire_g=dupire_g, dupire_lv=dupire_lv,
    )


# Helpers

def _ssvi_w(k: np.ndarray, theta: float, phi: float, rho: float) -> np.ndarray:
    """SSVI total variance: w = (θ/2)[1 + ρφk + √((φk+ρ)²+1−ρ²)]."""
    fk = phi * k
    return (theta / 2.0) * (1.0 + rho * fk + np.sqrt((fk + rho) ** 2 + 1.0 - rho ** 2))


def _expiry_colorscale(n: int) -> list:
    """Return n hex colors from blue (short) → red (long)."""
    import colorsys
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(0.67 * (1 - t), 0.8, 0.9)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


# Tab 1 — Market smiles (raw IV by expiry)

def make_market_smiles(iv: pd.DataFrame) -> go.Figure:
    expiries = sorted(iv["expiry"].unique())
    colors = _expiry_colorscale(len(expiries))
    fig = go.Figure()
    for i, exp in enumerate(expiries):
        sub = iv[iv["expiry"] == exp].sort_values("fwd_log_m")
        ttm_val = sub["ttm"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["fwd_log_m"], y=sub["iv"],
            mode="lines+markers",
            marker=dict(size=4, color=colors[i]),
            line=dict(color=colors[i], width=1.5),
            name=f"{exp} ({ttm_val:.2f}y)",
            hovertemplate=(
                "Strike: %{customdata[0]:,.0f}<br>"
                "Type: %{customdata[1]}<br>"
                "k = ln(K/F): %{x:.4f}<br>"
                "IV: %{y:.4f}<br>"
                "Mid: $%{customdata[2]:.2f}"
                "<extra></extra>"
            ),
            customdata=sub[["strike", "option_type", "mid"]].values,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.4,
                  annotation_text="ATM (k=0)")
    fig.update_layout(
        title="Market IV Smiles by Expiry — k = ln(K/F(T))",
        xaxis_title="Forward Log-Moneyness k = ln(K/F)",
        yaxis_title="Implied Volatility",
    )
    return fig


# Tab 2 — Total variance smiles (raw w = σ²·T by expiry)

def make_total_var_smiles(iv: pd.DataFrame) -> go.Figure:
    expiries = sorted(iv["expiry"].unique())
    colors = _expiry_colorscale(len(expiries))
    fig = go.Figure()
    for i, exp in enumerate(expiries):
        sub = iv[iv["expiry"] == exp].sort_values("fwd_log_m")
        ttm_val = sub["ttm"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["fwd_log_m"], y=sub["total_var"],
            mode="lines+markers",
            marker=dict(size=4, color=colors[i]),
            line=dict(color=colors[i], width=1.5),
            name=f"{exp} ({ttm_val:.2f}y)",
            hovertemplate=(
                "Strike: %{customdata[0]:,.0f}<br>"
                "k = ln(K/F): %{x:.4f}<br>"
                "Total Var w: %{y:.6f}<br>"
                "IV: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
            customdata=sub[["strike", "iv"]].values,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.4,
                  annotation_text="ATM (k=0)")
    fig.update_layout(
        title="Total Variance Smiles w = σ²·T — the SSVI fitting target",
        xaxis_title="Forward Log-Moneyness k = ln(K/F)",
        yaxis_title="Total Variance w = σ²·T",
    )
    return fig


# Tab 3 — SSVI fit: per-expiry subplot grid

def make_ssvi_fit_grid(iv: pd.DataFrame, params: pd.DataFrame) -> go.Figure:
    expiries = sorted(params["expiry"].unique())
    n = min(len(expiries), 24)
    expiries = expiries[:n]

    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[
            f"{exp}<br><sup>T={params[params['expiry']==exp]['ttm'].iloc[0]:.2f}y  "
            f"RMSE={params[params['expiry']==exp]['rmse'].iloc[0]*100:.2f}vp</sup>"
            for exp in expiries
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    params_map = {row["expiry"]: row for _, row in params.iterrows()}

    for idx, exp in enumerate(expiries):
        row = (idx // ncols) + 1
        col = (idx % ncols) + 1
        p = params_map[exp]

        sub = iv[iv["expiry"] == exp].sort_values("fwd_log_m")
        k_data = sub["fwd_log_m"].values
        w_data = sub["total_var"].values

        # SSVI model curve over the data range
        k_min, k_max = k_data.min() - 0.02, k_data.max() + 0.02
        k_range = np.linspace(k_min, k_max, 200)
        w_model = _ssvi_w(k_range, float(p["theta"]), float(p["phi"]), float(p["rho"]))

        show_legend = idx == 0
        fig.add_trace(go.Scatter(
            x=k_data, y=w_data,
            mode="markers",
            marker=dict(size=4, color="rgba(50,100,200,0.6)"),
            name="Market w",
            showlegend=show_legend,
            hovertemplate="k: %{x:.4f}<br>w: %{y:.6f}<extra></extra>",
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=k_range, y=w_model,
            mode="lines",
            line=dict(color="rgba(220,50,50,0.9)", width=2),
            name="SSVI model",
            showlegend=show_legend,
            hovertemplate="k: %{x:.4f}<br>w̃: %{y:.6f}<extra></extra>",
        ), row=row, col=col)

    fig.update_layout(
        title="SSVI Fit per Expiry — raw w (blue) vs model curve (red)",
        height=max(300 * nrows, 400),
    )
    return fig


# Tab 4 — Fitted surface (3D)

def make_fitted_surface(
    log_m_grid: np.ndarray,
    ttm_grid: np.ndarray,
    iv_surface: np.ndarray,
    tv_surface: np.ndarray,
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["Total Variance Surface  w(k,T)", "IV Surface  σ(k,T)"],
        horizontal_spacing=0.02,
    )

    common_hover = "k: %{x:.4f}<br>T: %{y:.3f}y<br>%{z:.4f}<extra></extra>"

    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid, z=tv_surface.T,
        colorscale="Blues", showscale=False,
        hovertemplate=common_hover,
    ), row=1, col=1)

    fig.add_trace(go.Surface(
        x=log_m_grid, y=ttm_grid, z=iv_surface.T,
        colorscale="Viridis", showscale=False,
        hovertemplate=common_hover,
    ), row=1, col=2)

    axis_kw = dict(
        xaxis_title="k = ln(K/F)",
        yaxis_title="TTM (years)",
    )
    fig.update_scenes({**axis_kw, "zaxis_title": "Total Variance w"}, row=1, col=1)
    fig.update_scenes({**axis_kw, "zaxis_title": "Implied Vol σ"}, row=1, col=2)
    fig.update_layout(
        title="SSVI Fitted Surface — Total Variance (left) and IV (right)",
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# Tab 5 — θ(T) term structure

def make_term_structure(iv: pd.DataFrame, params: pd.DataFrame) -> go.Figure:
    params = params.sort_values("ttm")

    # ATM total var from raw data (closest k to 0 per expiry)
    atm = iv.copy()
    atm["atm_dist"] = atm["fwd_log_m"].abs()
    atm = atm.sort_values("atm_dist").groupby("expiry").first().reset_index()
    atm = atm.merge(params[["expiry", "ttm"]].rename(columns={"ttm": "ttm_p"}),
                    on="expiry", how="left")
    atm = atm.sort_values("ttm")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # θ(T) — the SSVI ATM total variance
    fig.add_trace(go.Scatter(
        x=params["ttm"], y=params["theta"],
        mode="lines+markers",
        marker=dict(size=7, color="steelblue"),
        line=dict(color="steelblue", width=2),
        name="θ(T) — SSVI ATM total var",
        hovertemplate=(
            "Expiry: %{customdata}<br>"
            "TTM: %{x:.3f}y<br>"
            "θ: %{y:.6f}"
            "<extra></extra>"
        ),
        customdata=params["expiry"].values,
    ), secondary_y=False)

    # Raw ATM total variance from market data
    fig.add_trace(go.Scatter(
        x=atm["ttm"], y=atm["total_var"],
        mode="markers",
        marker=dict(size=8, color="steelblue", symbol="circle-open", line_width=2),
        name="Market ATM total var",
        hovertemplate=(
            "Expiry: %{customdata}<br>"
            "TTM: %{x:.3f}y<br>"
            "w_ATM: %{y:.6f}"
            "<extra></extra>"
        ),
        customdata=atm["expiry"].values,
    ), secondary_y=False)

    # φ(θ) — curvature parameter on secondary axis
    fig.add_trace(go.Scatter(
        x=params["ttm"], y=params["phi"],
        mode="lines+markers",
        marker=dict(size=6, color="darkorange"),
        line=dict(color="darkorange", width=1.5, dash="dot"),
        name="φ(θ) — smile curvature",
        hovertemplate=(
            "Expiry: %{customdata}<br>"
            "TTM: %{x:.3f}y<br>"
            "φ: %{y:.4f}"
            "<extra></extra>"
        ),
        customdata=params["expiry"].values,
    ), secondary_y=True)

    fig.update_yaxes(title_text="Total Variance θ(T)", secondary_y=False)
    fig.update_yaxes(title_text="Curvature φ(θ)", secondary_y=True)
    fig.update_xaxes(title_text="Time to Maturity (years)")

    gamma = params["gamma"].iloc[0]
    p0    = params["p0"].iloc[0]
    p1    = params["p1"].iloc[0]
    p2    = params["p2"].iloc[0]
    fig.update_layout(
        title=(
            f"SSVI Term Structure — θ(T) and φ(θ)   "
            f"[γ={gamma:.4f}  p₀={p0:.4f}, p₁={p1:.4f}, p₂={p2:.4f}]"
        ),
    )
    return fig


# Tab 6 — No-butterfly check

def make_nb_check(params: pd.DataFrame) -> go.Figure:
    params = params.sort_values("ttm").copy()
    abs_rho = params["rho"].abs()
    params["C1"] = params["theta"] * params["phi"] * (1.0 + abs_rho)
    params["C2"] = params["theta"] * params["phi"] ** 2 * (1.0 + abs_rho)

    fig = go.Figure()

    for col, color, name in [
        ("C1", "steelblue", "C1 = θ·φ·(1+|ρ|)  [≤ 4]"),
        ("C2", "darkorange", "C2 = θ·φ²·(1+|ρ|)  [≤ 4]"),
    ]:
        viol = params[col] > 4.0
        fig.add_trace(go.Scatter(
            x=params["ttm"], y=params[col],
            mode="lines+markers",
            marker=dict(
                size=8, color=color,
                symbol=["x" if v else "circle" for v in viol],
                line_width=[2 if v else 0 for v in viol],
            ),
            line=dict(color=color, width=2),
            name=name,
            hovertemplate=(
                "Expiry: %{customdata[0]}<br>"
                "TTM: %{x:.3f}y<br>"
                f"{col}: %{{y:.4f}}<br>"
                "Violated: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=list(zip(params["expiry"].values,
                                viol.astype(str).values)),
        ))

    fig.add_hline(y=4.0, line_dash="dash", line_color="red", opacity=0.6,
                  annotation_text="Limit = 4  (G&J 2014 Thm 4.2)",
                  annotation_position="top right")

    n_viol = int(((params["C1"] > 4) | (params["C2"] > 4)).sum())
    fig.update_layout(
        title=(
            f"No-Butterfly Conditions — G&J (2014) Thm 4.2   "
            f"[{n_viol} expiries with violations]"
        ),
        xaxis_title="Time to Maturity (years)",
        yaxis_title="Condition Value",
    )
    return fig


# Tab 7 — Dupire diagnostics

def make_dupire_diagnostics(
    log_m_grid: np.ndarray,
    ttm_grid: np.ndarray,
    dupire_g: np.ndarray | None,
    dupire_lv: np.ndarray | None,
) -> go.Figure:
    if dupire_g is None or dupire_lv is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Dupire arrays not found — run iv_surface_ssvi.py first.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font_size=16,
        )
        fig.update_layout(title="Dupire Diagnostics (data not available)")
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Gatheral Density g(k,T)  [g<0 = butterfly arb]",
            "Dupire Local Variance σ²_loc(k,T)",
        ],
        horizontal_spacing=0.08,
    )

    # g-surface: diverging colorscale centred at 0 (red where g < 0)
    g_clipped = np.clip(dupire_g.T, -1.0, 2.0)  # shape (n_ttm, n_logm)
    fig.add_trace(go.Heatmap(
        z=g_clipped,
        x=log_m_grid,
        y=ttm_grid,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="g", x=0.46, len=0.9),
        hovertemplate=(
            "k: %{x:.4f}<br>"
            "TTM: %{y:.3f}y<br>"
            "g: %{z:.4f}"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    # Local variance heatmap
    lv = np.clip(dupire_lv.T, 0, None)
    fig.add_trace(go.Heatmap(
        z=lv,
        x=log_m_grid,
        y=ttm_grid,
        colorscale="Viridis",
        colorbar=dict(title="σ²_loc", x=1.0, len=0.9),
        hovertemplate=(
            "k: %{x:.4f}<br>"
            "TTM: %{y:.3f}y<br>"
            "σ²_loc: %{z:.6f}"
            "<extra></extra>"
        ),
    ), row=1, col=2)

    n_neg = int((dupire_g < 0).sum())
    n_total = dupire_g.size
    pct_neg = 100.0 * n_neg / max(n_total, 1)
    fig.update_xaxes(title_text="k = ln(K/F)", row=1, col=1)
    fig.update_xaxes(title_text="k = ln(K/F)", row=1, col=2)
    fig.update_yaxes(title_text="TTM (years)", row=1, col=1)
    fig.update_yaxes(title_text="TTM (years)", row=1, col=2)
    fig.update_layout(
        title=(
            f"Dupire Diagnostics — "
            f"{pct_neg:.1f}% of grid points have g<0 (butterfly arb)"
        ),
        height=500,
    )
    return fig


# Tab 8 — Validation

def make_validation(val: pd.DataFrame) -> go.Figure:
    val = val.copy()
    val["iv_signed_err"] = val["iv_interpolated"] - val["iv_computed"]

    calls = val[val["option_type"] == "call"]
    puts  = val[val["option_type"] == "put"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "IV: Computed vs Interpolated",
            "Signed Error Distribution",
            "Error vs Market Price",
        ],
        horizontal_spacing=0.08,
    )

    # Left: computed vs interpolated
    for sub, name, color, sym in [
        (calls, "Calls", "steelblue", "circle"),
        (puts,  "Puts",  "coral",     "diamond"),
    ]:
        fig.add_trace(go.Scatter(
            x=sub["iv_computed"], y=sub["iv_interpolated"],
            mode="markers",
            marker=dict(size=6, color=color, symbol=sym, opacity=0.7),
            name=name,
            hovertemplate=(
                "Strike: %{customdata[0]:,.0f}<br>"
                "Expiry: %{customdata[1]}<br>"
                "Computed: %{x:.4f}<br>"
                "Interpolated: %{y:.4f}<br>"
                "Signed err: %{customdata[2]:+.4f}"
                "<extra></extra>"
            ),
            customdata=sub[["strike", "expiry", "iv_signed_err"]].values,
        ), row=1, col=1)

    iv_lims = [
        min(val["iv_computed"].min(), val["iv_interpolated"].min()) * 0.95,
        max(val["iv_computed"].max(), val["iv_interpolated"].max()) * 1.05,
    ]
    fig.add_trace(go.Scatter(
        x=iv_lims, y=iv_lims, mode="lines",
        line=dict(color="red", dash="dash", width=1), name="y = x",
        showlegend=True,
    ), row=1, col=1)

    # Middle: signed error histogram
    err_vp = val["iv_signed_err"] * 100  # vol points
    fig.add_trace(go.Histogram(
        x=err_vp,
        nbinsx=30,
        marker_color="mediumpurple", opacity=0.8,
        name="Signed err",
        showlegend=False,
        hovertemplate="Error: %{x:.2f} vp<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)
    fig.add_vline(
        x=0, line_dash="dash", line_color="black", opacity=0.6,
        row=1, col=2,
    )
    me_vp = err_vp.mean()
    fig.add_vline(
        x=me_vp,
        line_dash="dot", line_color="red", opacity=0.8,
        annotation_text=f"ME={me_vp:+.2f}vp",
        annotation_position="top right",
        row=1, col=2,
    )

    # Right: error vs market price (liquidity check)
    for sub, name, color, sym in [
        (calls, "Calls", "steelblue", "circle"),
        (puts,  "Puts",  "coral",     "diamond"),
    ]:
        fig.add_trace(go.Scatter(
            x=sub["market_price"],
            y=sub["iv_signed_err"] * 100,
            mode="markers",
            marker=dict(size=5, color=color, symbol=sym, opacity=0.6),
            name=name,
            showlegend=False,
            hovertemplate=(
                "Market price: $%{x:.2f}<br>"
                "Error: %{y:+.2f} vp"
                "<extra></extra>"
            ),
        ), row=1, col=3)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=1, col=3)

    mae  = val["iv_abs_err"].mean() * 100
    rmse = np.sqrt((val["iv_abs_err"] ** 2).mean()) * 100
    me   = me_vp

    fig.update_xaxes(title_text="Computed IV",        row=1, col=1)
    fig.update_yaxes(title_text="Interpolated IV",    row=1, col=1)
    fig.update_xaxes(title_text="Error (vol pts)",    row=1, col=2)
    fig.update_yaxes(title_text="Count",              row=1, col=2)
    fig.update_xaxes(title_text="Market Price ($)",   row=1, col=3)
    fig.update_yaxes(title_text="Signed Error (vp)",  row=1, col=3)

    fig.update_layout(
        title=(
            f"Surface Validation  ({len(val)} options sampled)  —  "
            f"MAE={mae:.2f}vp  RMSE={rmse:.2f}vp  ME={me:+.2f}vp"
        ),
    )
    return fig


# Tab 9 — Liquidity map

def make_liquidity_map(iv: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    calls = iv[iv["option_type"] == "call"]
    puts  = iv[iv["option_type"] == "put"]

    for sub, name, color in [
        (calls, "Calls", "steelblue"),
        (puts,  "Puts",  "coral"),
    ]:
        oi = sub["openInterest"].fillna(0).values
        fig.add_trace(go.Scatter(
            x=sub["fwd_log_m"], y=sub["ttm"],
            mode="markers",
            marker=dict(
                size=np.clip(np.sqrt(oi) / 5, 2, 30),
                color=sub["iv"].values,
                colorscale="Viridis",
                opacity=0.55,
                showscale=(name == "Calls"),
                colorbar=dict(title="IV") if name == "Calls" else None,
            ),
            name=name,
            hovertemplate=(
                "Strike: %{customdata[0]:,.0f}<br>"
                "Expiry: %{customdata[1]}<br>"
                "k: %{x:.4f}<br>"
                "TTM: %{y:.3f}y<br>"
                "OI: %{customdata[2]:,.0f}<br>"
                "IV: %{customdata[3]:.4f}"
                "<extra></extra>"
            ),
            customdata=sub[["strike", "expiry", "openInterest", "iv"]].values,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.4,
                  annotation_text="ATM (k=0)")
    fig.update_layout(
        title="Liquidity Map — Open Interest by k × TTM (bubble size = √OI, colour = IV)",
        xaxis_title="Forward Log-Moneyness k = ln(K/F)",
        yaxis_title="TTM (years)",
    )
    return fig


# Tab 10 — TV surface mesh (SSVI slices × cubic spline)

def make_tv_mesh(params: pd.DataFrame) -> go.Figure:
    """
    3D total-variance mesh: per-slice SSVI w(k) curves (fitted θ,φ,ρ) as ribs,
    over a filled surface from cubic-spline interpolation of θ,φ,ρ in TTM.
    """
    params = params.sort_values("ttm").copy()
    ttms      = params["ttm"].values
    theta_mkt = params["theta"].values
    phi_mkt   = params["phi"].values
    rho_mkt   = params["rho"].values
    expiries  = params["expiry"].values

    # Fine grids
    k_grid   = np.linspace(-1.2, 0.6, 160)
    ttm_fine = np.linspace(ttms[0], ttms[-1], 300)

    # Cubic-spline interpolation of (θ, φ, ρ) along TTM
    cs_theta = CubicSpline(ttms, theta_mkt)
    cs_phi   = CubicSpline(ttms, phi_mkt)
    cs_rho   = CubicSpline(ttms, rho_mkt)

    theta_fine = np.maximum(cs_theta(ttm_fine), 1e-8)
    phi_fine   = np.maximum(cs_phi(ttm_fine),   1e-8)
    rho_fine   = np.clip(cs_rho(ttm_fine), -0.999, 0.999)

    # Build surface w[k, T] from cubic-spline params
    W = np.zeros((len(k_grid), len(ttm_fine)))
    for j in range(len(ttm_fine)):
        W[:, j] = _ssvi_w(k_grid, theta_fine[j], phi_fine[j], rho_fine[j])

    fig = go.Figure()

    # Filled surface (cubic spline interpolation in T direction)
    fig.add_trace(go.Surface(
        x=ttm_fine,
        y=k_grid,
        z=W,
        colorscale="Blues",
        opacity=0.55,
        showscale=True,
        colorbar=dict(title="w = σ²T", x=1.0),
        name="Cubic spline surface",
        hovertemplate=(
            "TTM: %{x:.3f}y<br>"
            "k: %{y:.3f}<br>"
            "w: %{z:.5f}"
            "<extra>Spline</extra>"
        ),
    ))

    # SSVI slice ribs at market maturities
    colors = _expiry_colorscale(len(ttms))
    for i in range(len(ttms)):
        w_slice = _ssvi_w(k_grid, theta_mkt[i], phi_mkt[i], rho_mkt[i])
        fig.add_trace(go.Scatter3d(
            x=np.full_like(k_grid, ttms[i]),
            y=k_grid,
            z=w_slice,
            mode="lines",
            line=dict(color=colors[i], width=4),
            name=str(expiries[i]),
            hovertemplate=(
                f"Expiry: {expiries[i]}<br>"
                "TTM: %{x:.3f}y<br>"
                "k: %{y:.3f}<br>"
                "w: %{z:.5f}"
                "<extra>SSVI slice</extra>"
            ),
        ))

    fig.update_layout(
        title=(
            "Total Variance Surface — SSVI Slices (coloured ribs) "
            "× Cubic Spline Interpolation (filled mesh)"
        ),
        scene=dict(
            xaxis=dict(title="TTM (years)"),
            yaxis=dict(title="Log Forward Moneyness k"),
            zaxis=dict(title="Total Variance w = σ²T"),
            camera=dict(eye=dict(x=-1.6, y=-1.4, z=0.8)),
        ),
        height=750,
        legend=dict(
            title="Market Expiry",
            font=dict(size=10),
            x=0.0, y=1.0,
        ),
    )
    return fig


# HTML builder

TAB_DESCRIPTIONS = {
    "Market Smiles": (
        "Raw implied volatilities by expiry plotted against forward log-moneyness "
        "k = ln(K/F(T)). Each line is one expiry, coloured from blue (short) to red "
        "(long). This is the cleanest view of the raw smile data entering the SSVI fit."
    ),
    "Total Var Smiles": (
        "Total variance w = σ²·T — the direct SSVI fitting target. The monotone "
        "spacing of these curves by maturity is what the calendar-spread-free θ "
        "parameterisation enforces. Each curve should be convex (no butterfly arb)."
    ),
    "SSVI Fit": (
        "Per-expiry comparison: blue dots are market total variance, red curve is the "
        "SSVI model. RMSE in vol points shown in each panel title. Use this to spot "
        "expiries where the model fit is poor — typically thin slices or unusual skew."
    ),
    "Fitted Surface": (
        "Left: the continuous SSVI total variance surface w(k,T). Right: the implied "
        "volatility surface σ(k,T) = √(w/T). Both are fully arbitrage-free by "
        "construction (SSVI power-law with G&J no-butterfly penalty during fitting)."
    ),
    "θ(T) Term Structure": (
        "θ(T) (blue solid) is the SSVI ATM total variance — the spine of the surface. "
        "Open circles are raw market ATM total variances: gaps indicate fit quality. "
        "φ(θ) (orange) controls smile curvature; it falls as T grows."
    ),
    "No-Butterfly Check": (
        "G&J (2014) Thm 4.2 requires C1 = θ·φ·(1+|ρ|) ≤ 4 and C2 = θ·φ²·(1+|ρ|) ≤ 4. "
        "Points above the red dashed line at 4 are violations — these expiries will "
        "show butterfly arbitrage in the local vol and should be investigated."
    ),
    "Dupire Diagnostics": (
        "Left: Gatheral density g(k,T); red regions (g < 0) signal butterfly arbitrage. "
        "Right: Dupire local variance σ²_loc(k,T) = ∂w/∂T / g. Negative or extreme "
        "local variance values show where the surface is not suitable for the LSV model."
    ),
    "Validation": (
        "How well does the SSVI surface reproduce the market IVs? Left: computed vs "
        "surface-interpolated IV — close to diagonal is good. Middle: signed error "
        "histogram; the mean error (ME) diagnoses systematic bias (ME ≠ 0 means the "
        "surface is biased high or low everywhere). Right: error vs price — liquidity "
        "check (errors should be random, not larger for cheap OTM options)."
    ),
    "Liquidity Map": (
        "Bubble size = √(open interest). Larger bubbles = more liquid options = more "
        "reliable IV data. Colour = IV level. Use this to judge data quality by region: "
        "thin bubbles in the wings mean the smile there is less reliable."
    ),
    "TV Surface Mesh": (
        "Total variance surface w(k,T) = σ²·T decomposed along its two axes. "
        "Coloured ribs: SSVI slices at each market maturity — each rib is the "
        "fitted w(k) curve using that slice's own (θ, φ, ρ). "
        "Filled mesh: cubic spline interpolation of (θ, φ, ρ) between market maturities. "
        "The current pipeline uses PCHIP for θ interpolation; this mesh uses a natural "
        "cubic spline so you can see where the two differ (cubic spline can overshoot "
        "between knots, PCHIP cannot). The gap between ribs and mesh at non-market TTMs "
        "shows the sensitivity of the Dupire ∂w/∂T numerator to the interpolation choice."
    ),
}


def build_html(figures: list, tab_names: list, descriptions: dict) -> str:
    fig_json_list = []
    for fig in figures:
        fig.update_layout(
            template="plotly_white",
            height=fig.layout.height or 650,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        fig_json_list.append(fig.to_json())

    tab_buttons = [
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f'onclick="switchTab({i})">{name}</button>'
        for i, name in enumerate(tab_names)
    ]

    tab_contents = [
        f'<div class="tab-content" id="tab-{i}" '
        f'style="display:{"block" if i == 0 else "none"}">'
        f'<p class="tab-desc">{descriptions.get(name, "")}</p>'
        f'<div id="plot-{i}" style="width:100%;height:{figures[i].layout.height or 650}px;"></div>'
        f'</div>'
        for i, name in enumerate(tab_names)
    ]

    html_head = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IV Explorer — SSVI</title>
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
  .tab-desc { font-size: 14px; line-height: 1.5; color: #555;
               margin: 8px 0 12px 0; max-width: 900px; }
  .tab-content { background: white; border-radius: 0 0 8px 8px; padding: 12px; }
</style>
</head>
<body>
<h1>SPX IV Explorer — SSVI Surface</h1>
"""

    fig_specs_js = ",\n".join(fig_json_list)
    html_script = (
        "\n<script>\nvar figSpecs = ["
        + fig_specs_js
        + """];
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
    )

    return (
        html_head
        + '<div class="tab-bar">\n  ' + "".join(tab_buttons) + "\n</div>\n"
        + "".join(tab_contents)
        + html_script
    )


# Main

def main() -> None:
    print("Loading data…")
    d = load_data()

    tab_names = [
        "Market Smiles",
        "Total Var Smiles",
        "SSVI Fit",
        "Fitted Surface",
        "θ(T) Term Structure",
        "No-Butterfly Check",
        "Dupire Diagnostics",
        "Validation",
        "Liquidity Map",
        "TV Surface Mesh",
    ]

    print("Building figures…")
    figures = [
        make_market_smiles(d["iv"]),
        make_total_var_smiles(d["iv"]),
        make_ssvi_fit_grid(d["iv"], d["params"]),
        make_fitted_surface(d["log_m_grid"], d["ttm_grid"],
                            d["iv_surface"], d["tv_surface"]),
        make_term_structure(d["iv"], d["params"]),
        make_nb_check(d["params"]),
        make_dupire_diagnostics(d["log_m_grid"], d["ttm_grid"],
                                d["dupire_g"], d["dupire_lv"]),
        make_validation(d["val"]),
        make_liquidity_map(d["iv"]),
        make_tv_mesh(d["params"]),
    ]

    html = build_html(figures, tab_names, TAB_DESCRIPTIONS)

    os.makedirs(DIR_PLOTS, exist_ok=True)
    out_path = os.path.join(DIR_PLOTS, "iv_explorer.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path} — open in your browser")


if __name__ == "__main__":
    main()

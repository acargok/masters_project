#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis §4.1 (Surface Fitting) figure pack. Stand-alone: reads cached
artefacts from iv_surface/{data,arrays} and dupire_vol/{data,arrays} and
writes thesis-ready figures + LaTeX tables to iv_surface/results_4.1_plots/.
Multi-panel figures are emitted one PNG per panel (except the SSVI fit grid);
see main() for the list.
"""
import json
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

warnings.filterwarnings("ignore")

# Paths
HERE       = Path(__file__).resolve().parent          # iv_surface/
ROOT       = HERE.parent
IV_DIR     = HERE
DUPIRE_DIR = ROOT / "dupire_vol"
OUT_DIR    = HERE / "results_4.1_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
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

LIQUID_MIN_PRICE = 10.0    # liquidity threshold reused across the pipeline

# Shared layout for the two §4.1.1 (k,T) overview figures so their data canvas
# and image dimensions match (the bubble figure's colourbar slot is invisible
# but consumes the same width as the market-IV-smiles colourbar).
_SMILE_BUBBLE_FIGSIZE = (7.5, 4.8)
_CBAR_SIZE            = "3%"
_CBAR_PAD             = 0.10


# SSVI math
def ssvi_w(k: np.ndarray, theta: float, phi: float, rho: float) -> np.ndarray:
    """w(k; θ, φ, ρ) = (θ/2)[1 + ρφk + √((φk+ρ)² + 1 − ρ²)]."""
    fk = phi * k
    return 0.5 * theta * (1.0 + rho * fk + np.sqrt((fk + rho) ** 2 + 1.0 - rho ** 2))


def ssvi_rho_t(t: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    """ρ(t) = clip(arctan(p₀ t + p₁) + p₂, ±0.999)."""
    return np.clip(np.arctan(p0 * t + p1) + p2, -0.999, 0.999)


# Data
def _maybe_load(path: Path):
    return np.load(path) if path.exists() else None


def load_data() -> dict:
    """Load every artefact §4.1 needs; derives fwd_log_m/total_var if absent."""
    d = {}
    d["iv_df"]   = pd.read_csv(IV_DIR / "data" / "spx_iv_data.csv")
    d["ssvi_df"] = (pd.read_csv(IV_DIR / "data" / "ssvi_params.csv")
                      .sort_values("ttm").reset_index(drop=True))
    d["val_df"]  = pd.read_csv(IV_DIR / "data" / "validation_results.csv")
    d["fwd_df"]  = pd.read_csv(IV_DIR / "data" / "implied_forwards.csv")

    # Derive columns absent on disk
    iv = d["iv_df"]
    if "fwd_log_m" not in iv.columns:
        fwd_map = dict(zip(d["fwd_df"]["expiry"], d["fwd_df"]["forward"]))
        iv["fwd_log_m"] = np.log(iv["strike"] / iv["expiry"].map(fwd_map))
    if "total_var" not in iv.columns:
        iv["total_var"] = iv["iv"] ** 2 * iv["ttm"]
    with open(IV_DIR / "data" / "market_params.json") as f:
        d["market"] = json.load(f)

    arr_iv = IV_DIR / "arrays"
    d["iv_surface"] = np.load(arr_iv / "iv_surface.npy")
    d["tv_surface"] = np.load(arr_iv / "total_var_surface.npy")
    d["log_m_grid"] = np.load(arr_iv / "log_m_grid.npy")
    d["ttm_grid"]   = np.load(arr_iv / "ttm_grid.npy")
    d["g_surface"]  = _maybe_load(arr_iv / "dupire_g_surface.npy")

    # Warn loudly if ssvi/iv expiries disagree (the fit grid would .min() an
    # empty array); tells the user to refresh the data.
    ssvi_exps = set(d["ssvi_df"]["expiry"].unique())
    iv_exps   = set(d["iv_df"]["expiry"].unique())
    inter     = ssvi_exps & iv_exps
    d["expiries_with_iv"] = inter
    if len(inter) < len(ssvi_exps):
        print(
            f"  !!  data inconsistency detected\n"
            f"      ssvi_params.csv : {len(ssvi_exps)} expiries\n"
            f"      spx_iv_data.csv : {len(iv_exps)} expiries\n"
            f"      intersection    : {len(inter)} expiries\n"
            f"      → only the intersection will have market dots; the rest\n"
            f"        of ssvi has no matching option chain rows. To restore\n"
            f"        full coverage rerun the upstream SSVI script so that\n"
            f"        iv, ssvi_params, implied_forwards and market_params\n"
            f"        come out of the same snapshot:\n"
            f"          python \"iv_surface/optionmetrics_iv_surface_ssvi.py\"\n"
        )

    arr_dup = DUPIRE_DIR / "arrays"
    d["local_vol"] = np.load(arr_dup / "local_vol_surface.npy")
    d["dup_rep"]   = pd.read_csv(DUPIRE_DIR / "data" / "repricing_errors.csv")
    return d


def _save(fig, name: str, tight: bool = True) -> Path:
    """Save and close a figure, returning its path. tight=False keeps figsize
    exact (bbox_inches=None) so paired figures match pixel dimensions."""
    out = OUT_DIR / name
    if tight:
        fig.savefig(out)
    else:
        fig.savefig(out, bbox_inches=None)
    plt.close(fig)
    return out


# Figures

# §4.1.1  SSVI fit grid (the one multi-panel kept whole)
def fig_ssvi_fit_grid(d: dict) -> Path:
    """SSVI fit per expiry, 2×3 panels: market total variance (dots) vs SSVI
    w-curve. Every third slice along T (≈uniform sweep); panels with no iv
    rows are hidden.
    """
    ssvi = d["ssvi_df"].sort_values("ttm").reset_index(drop=True)
    iv   = d["iv_df"]

    # Every third slice along T, capped to 6 panels
    idxs = list(range(0, len(ssvi), 3))[:6]
    sub  = ssvi.iloc[idxs].reset_index(drop=True)

    nrows, ncols = 2, 3
    n_panels     = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 7.5))
    axes = axes.flatten()
    n_missing = 0

    for i in range(n_panels):
        ax = axes[i]
        if i >= len(sub):
            ax.set_visible(False)
            continue
        row    = sub.iloc[i]
        expiry = row["expiry"]; ttm = float(row["ttm"])
        theta, phi, rho = float(row["theta"]), float(row["phi"]), float(row["rho"])
        rmse   = float(row["rmse"])

        sl = iv[iv["expiry"] == expiry].sort_values("fwd_log_m")
        if len(sl) == 0:
            # No iv rows — hide the panel (avoids .min() on empty array)
            ax.set_visible(False)
            n_missing += 1
            continue

        k_data = sl["fwd_log_m"].values
        w_data = sl["total_var"].values
        k_min, k_max = float(k_data.min()) - 0.02, float(k_data.max()) + 0.02
        k_range = np.linspace(k_min, k_max, 200)
        w_model = ssvi_w(k_range, theta, phi, rho)

        # market dots below the SSVI curve (fit on top)
        ax.scatter(k_data, w_data, s=6, color="black", alpha=0.7, zorder=2)
        ax.plot(k_range, w_model, color="crimson", lw=1.6, zorder=3)

        ax.set_title(f"{expiry}   T = {ttm:.2f} years", fontsize=10)
        ax.set_xlabel(r"$k$", fontsize=12)
        ax.set_ylabel(r"$w$", fontsize=12)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.text(0.04, 0.96, f"RMSE {rmse*100:.2f} vp",
                transform=ax.transAxes, fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          alpha=0.85, edgecolor="none"))

    if n_missing:
        warnings.warn(
            f"SSVI grid: {n_missing}/{len(sub)} selected slices had no rows "
            f"in spx_iv_data.csv for the matching expiry — those panels are "
            f"hidden. Refresh the upstream SSVI pipeline so iv and ssvi "
            f"come from the same snapshot to restore full coverage."
        )

    plt.tight_layout()
    return _save(fig, "ssvi_fit_grid.png")


# §4.1.1  SSVI parameter + metrics table
def fig_ssvi_params_table(d: dict) -> Path:
    """LaTeX table: shared SSVI params, fit size, and validation metrics.
    n_obs = sum of per-slice n_options (post-cleanup count, not the
    validation subsample n_val); IV MAE/RMSE/max abs (bp) from the random
    repricing sample.
    """
    row0 = d["ssvi_df"].iloc[0]
    val  = d["val_df"]
    eta, gamma = float(row0["eta"]),   float(row0["gamma"])
    p0, p1, p2 = float(row0["p0"]),    float(row0["p1"]),    float(row0["p2"])

    # Fit-level dataset size: sum of per-slice n_options (post-cleanup);
    # fall back to len(spx_iv_data.csv) if n_options is absent.
    ssvi_df = d["ssvi_df"]
    if "n_options" in ssvi_df.columns:
        n_obs = int(ssvi_df["n_options"].sum())
    else:
        n_obs = int(len(d["iv_df"]))
    n_slices = int(len(ssvi_df))

    # Validation diagnostics on the repricing sample
    iv_err_bps  = (val["iv_interpolated"] - val["iv_computed"]).abs() * 1e4
    iv_mae      = float(iv_err_bps.mean())
    iv_rmse     = float(np.sqrt((iv_err_bps ** 2).mean()))
    iv_max      = float(iv_err_bps.max())
    n_val       = int(len(val))

    body = [
        ("$\\eta$",                f"{eta:.4f}"),
        ("$\\gamma$",              f"{gamma:.4f}"),
        ("$p_0$",                  f"{p0:+.4f}"),
        ("$p_1$",                  f"{p1:+.4f}"),
        ("$p_2$",                  f"{p2:+.4f}"),
        ("$n_\\mathrm{slices}$",   f"{n_slices}"),
        ("$n_\\mathrm{obs}$",      f"{n_obs}"),
        ("IV MAE (bp)",            f"{iv_mae:.2f}"),
        ("IV RMSE (bp)",           f"{iv_rmse:.2f}"),
        ("IV max abs (bp)",        f"{iv_max:.2f}"),
        ("$n_\\mathrm{val}$",      f"{n_val}"),
    ]
    lines = [
        "% Auto-generated by results_4_1_figures.py — do not edit by hand.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Calibrated SSVI parameters and fit-vs-validation metrics. "
        "Shared parameters $\\eta, \\gamma, p_0, p_1, p_2$ are joint across "
        "all expiry slices. $n_\\mathrm{obs}$ is the total option count used "
        "in the joint SSVI fit (sum of per-slice option counts); "
        "$n_\\mathrm{val}$ is the random subsample drawn for repricing "
        "validation, and the IV-error metrics are computed on that "
        "subsample.}",
        "\\label{tab:ssvi_params}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Parameter & Value \\\\",
        "\\midrule",
    ]
    for k, v in body:
        if k == "IV MAE (bp)":
            lines.append("\\midrule")
        lines.append(f"{k} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = OUT_DIR / "ssvi_params_table.tex"
    out.write_text("\n".join(lines))
    return out


# Appendix: SSVI parameters — shared block + per-slice
def fig_ssvi_params_table_full(d: dict) -> Path:
    """Appendix LaTeX table: shared params (η,γ,p₀,p₁,p₂) above, per-slice
    block (θ,φ,ρ,RMSE vp,n_options per expiry) below, as two stacked
    tabulars under one caption."""
    row0 = d["ssvi_df"].iloc[0]
    eta, gamma   = float(row0["eta"]),   float(row0["gamma"])
    p0, p1, p2   = float(row0["p0"]),    float(row0["p1"]),    float(row0["p2"])

    shared = [
        (r"$\eta$",    "SSVI level",              f"${eta:.4f}$"),
        (r"$\gamma$",  "SSVI curvature exponent", f"${gamma:.4f}$"),
        (r"$p_0$",     r"$\rho(T)$ slope",        f"${p0:+.4f}$"),
        (r"$p_1$",     r"$\rho(T)$ intercept",    f"${p1:+.4f}$"),
        (r"$p_2$",     r"$\rho(T)$ shift",        f"${p2:+.4f}$"),
    ]

    slices = d["ssvi_df"].sort_values("ttm").reset_index(drop=True)

    lines = [
        "% Auto-generated by results_4_1_figures.py — do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Calibrated SSVI parameters (appendix, full detail). The "
        r"upper block lists the shared parameters $\eta, \gamma, p_0, p_1, "
        r"p_2$ that are joint across every expiry slice. The lower block "
        r"lists the per-slice values $\theta(T_i)$ (ATM total variance), "
        r"$\varphi(T_i)$ (slope, derived from $\eta$ and $\theta(T_i)$ via "
        r"$\varphi(\theta) = \eta\,\theta^{-\gamma}$), and $\rho(T_i)$ "
        r"(skew at maturity $T_i$), together with the per-slice fit RMSE in "
        r"vol points (vp) and the option count used in that slice's "
        r"contribution to the joint fit.}",
        r"\label{tab:ssvi_params_full}",
        # Shared block
        r"\begin{tabular}{l l c}",
        r"\toprule",
        r"\textbf{Symbol} & \textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
    ]
    for sym, name, val in shared:
        lines.append(f"{sym} & {name} & {val} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{1em}",
        r"",
        # Per-slice block
        r"\begin{tabular}{lccccrr}",
        r"\toprule",
        r"\textbf{Expiry} & $T_i$ (yr) & $\theta(T_i)$ & $\varphi(T_i)$ & "
        r"$\rho(T_i)$ & RMSE (vp) & $n_\mathrm{opt}$ \\",
        r"\midrule",
    ]
    for _, r in slices.iterrows():
        lines.append(
            f"{r['expiry']} & "
            f"{float(r['ttm']):.4f} & "
            f"{float(r['theta']):.6f} & "
            f"{float(r['phi']):.4f} & "
            f"{float(r['rho']):+.4f} & "
            f"{float(r['rmse']) * 100:.3f} & "
            f"{int(r['n_options'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = OUT_DIR / "ssvi_params_table_full.tex"
    out.write_text("\n".join(lines))
    return out


# §4.1.1  Two separate 3D surface figures
def fig_iv_surface_3d(d: dict) -> Path:
    """Fitted IV surface σ(k, T) — 3D plot."""
    log_m = d["log_m_grid"]; ttm = d["ttm_grid"]; iv = d["iv_surface"]
    K_mesh, T_mesh = np.meshgrid(log_m, ttm, indexing="ij")

    # Sizing matched to fig_dupire_local_vol_3d (colorbar pad clears z-label)
    fig = plt.figure(figsize=(9.0, 6.0))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, iv, cmap=cm.viridis,
                           edgecolor="none", alpha=0.92, rcount=80, ccount=80)
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.12)
    cb.set_label(r"Implied volatility $\sigma$", fontsize=12)
    ax.set_xlabel(r"Forward log-moneyness $k$",     labelpad=10, fontsize=12)
    ax.set_ylabel(r"Time to maturity $T$ (years)",  labelpad=10, fontsize=12)
    ax.set_zlabel(r"Implied volatility $\sigma$",   labelpad=10, fontsize=12)
    ax.view_init(elev=26, azim=-58)
    ax.set_proj_type("ortho")   # orthographic — no perspective distortion
    plt.tight_layout()
    return _save(fig, "ssvi_iv_surface_3d.png")


def fig_tv_surface_with_slices_3d(d: dict) -> Path:
    """Total-variance surface w(k, T) with each fitted expiry overlaid as a
    slice line, so the per-expiry fits appear as ribs of a single surface."""
    log_m = d["log_m_grid"]; ttm = d["ttm_grid"]; tv = d["tv_surface"]
    K_mesh, T_mesh = np.meshgrid(log_m, ttm, indexing="ij")

    fig = plt.figure(figsize=(9.0, 6.0))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, tv, cmap=cm.viridis,
                           edgecolor="none", alpha=0.55, rcount=80, ccount=80)
    for _, row in d["ssvi_df"].iterrows():
        j = int(np.argmin(np.abs(ttm - float(row["ttm"]))))
        ax.plot(log_m, np.full_like(log_m, ttm[j]), tv[:, j],
                color="C3", lw=1.0, alpha=0.85)
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.12)
    cb.set_label(r"Total variance $w$", fontsize=12)
    ax.set_xlabel(r"Forward log-moneyness $k$",    labelpad=10, fontsize=12)
    ax.set_ylabel(r"Time to maturity $T$ (years)", labelpad=10, fontsize=12)
    ax.set_zlabel(r"Total variance $w$",           labelpad=10, fontsize=12)
    ax.view_init(elev=26, azim=-58)
    ax.set_proj_type("ortho")   # orthographic — no perspective distortion
    plt.tight_layout()
    return _save(fig, "ssvi_tv_surface_with_slices_3d.png")


# §4.1.1  Term structure — single plot, three axes
def fig_term_structure(d: dict) -> Path:
    """ρ(T), θ(T), φ(T) on shared T axis: ρ (C0, parametric + dots),
    θ (C1, per-slice line), φ (C2, parametric + dots). φ(θ)=η·θ^(-γ);
    per-slice φ_i may sit slightly off the curve at optimiser bounds.
    """
    ssvi  = d["ssvi_df"]
    row0  = ssvi.iloc[0]
    p0, p1, p2  = float(row0["p0"]), float(row0["p1"]), float(row0["p2"])
    eta, gamma  = float(row0["eta"]), float(row0["gamma"])

    T_pts     = ssvi["ttm"].values
    theta_pts = ssvi["theta"].values
    rho_pts   = ssvi["rho"].values
    phi_pts   = ssvi["phi"].values
    T_fine    = np.linspace(T_pts.min(), T_pts.max(), 240)
    rho_fine  = ssvi_rho_t(T_fine, p0, p1, p2)

    # φ(θ)=η·θ^(-γ): interpolate θ over T, then map θ→φ via shared (η,γ)
    theta_interp = np.interp(T_fine, T_pts, theta_pts)
    phi_fine     = eta * theta_interp ** (-gamma)

    # Wide figure leaves room for the third (offset) right axis
    fig, ax_rho = plt.subplots(figsize=(9.8, 5.0))
    ax_th  = ax_rho.twinx()
    ax_phi = ax_rho.twinx()
    ax_phi.spines["right"].set_position(("outward", 60))

    # ρ on the inner-left axis (parametric + per-slice dots)
    l_rho_curve, = ax_rho.plot(T_fine, rho_fine, color="C0", lw=1.7,
                               label=r"$\rho(T)$  parametric")
    l_rho_pts = ax_rho.scatter(T_pts, rho_pts, color="C0", s=22, zorder=3,
                               edgecolor="white", linewidth=0.6,
                               label=r"$\rho(T_i)$  per slice")
    ax_rho.axhline(0, color="black", lw=0.5)
    ax_rho.set_xlabel(r"Time to maturity $T$ (years)")
    ax_rho.set_ylabel(r"$\rho(T)$", color="C0")
    ax_rho.tick_params(axis="y", labelcolor="C0")
    ax_rho.grid(True, alpha=0.3)

    # θ on the inner-right axis (per-slice line)
    l_theta, = ax_th.plot(T_pts, theta_pts, color="C1", lw=1.7, marker="o",
                          markersize=4, label=r"$\theta(T_i)$  per slice")
    ax_th.set_ylabel(r"$\theta(T) = w(0, T)$", color="C1")
    ax_th.tick_params(axis="y", labelcolor="C1")

    # φ on the outer-right axis (parametric + per-slice dots)
    l_phi_curve, = ax_phi.plot(T_fine, phi_fine, color="C2", lw=1.7,
                                label=r"$\varphi(T)$  parametric")
    l_phi_pts = ax_phi.scatter(T_pts, phi_pts, color="C2", s=22, zorder=3,
                                edgecolor="white", linewidth=0.6,
                                marker="s",
                                label=r"$\varphi(T_i)$  per slice")
    ax_phi.set_ylabel(r"$\varphi(T) = \eta \, \theta(T)^{-\gamma}$",
                       color="C2")
    ax_phi.tick_params(axis="y", labelcolor="C2")

    ax_rho.set_title(r"Time-varying skew $\rho(T)$, ATM total variance "
                     r"$\theta(T)$, and slope $\varphi(T)$",
                     fontweight="bold")

    handles = [l_rho_curve, l_rho_pts, l_theta, l_phi_curve, l_phi_pts]
    labels  = [h.get_label() for h in handles]
    ax_rho.legend(handles, labels, loc="best", fontsize=8, framealpha=0.9,
                   ncol=2)

    plt.tight_layout()
    return _save(fig, "term_structure.png")


# §4.1.2  SSVI repricing — three separate figures
# From validation_results.csv (one join to implied_forwards.csv for the
# moneyness scatter). Errors in basis points (bp = 1e4·Δσ).

def _val_prepared(d: dict) -> pd.DataFrame:
    val = d["val_df"].copy()
    val["iv_signed_err_bps"] = (val["iv_interpolated"] - val["iv_computed"]) * 1e4
    return val


def fig_ssvi_repricing_iv_scatter(d: dict) -> Path:
    """Panel (a): computed IV vs interpolated IV scatter (calls vs puts),
    with the y = x reference."""
    val = _val_prepared(d)
    calls = val[val["option_type"] == "call"]
    puts  = val[val["option_type"] == "put"]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(calls["iv_computed"], calls["iv_interpolated"],
               s=22, alpha=0.7, color="steelblue", marker="o", label="calls")
    ax.scatter(puts["iv_computed"], puts["iv_interpolated"],
               s=22, alpha=0.7, color="coral", marker="D", label="puts")

    lo = float(min(val["iv_computed"].min(), val["iv_interpolated"].min())) * 0.95
    hi = float(max(val["iv_computed"].max(), val["iv_interpolated"].max())) * 1.05
    ax.plot([lo, hi], [lo, hi], color="red", lw=1.0, ls="--", label=r"$y = x$")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Computed IV")
    ax.set_ylabel("Interpolated IV")
    ax.set_title("SSVI repricing  —  computed vs interpolated IV",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "ssvi_repricing_iv_scatter.png")


# Shared visual contract for the SSVI-repricing trio: same figsize, square
# axes box (set_box_aspect(1)), no titles, calls=blue / puts=red circles.
_REPR_FIGSIZE   = (4.5, 4.5)
_REPR_SCATTER_S = 5
_CALL_COLOR     = "tab:blue"
_PUT_COLOR      = "tab:red"


def _square_axes(ax) -> None:
    """Force the data axes box to be a square regardless of label widths."""
    ax.set_box_aspect(1)


_ERR_LABEL = "Implied volatility error (bp)"


def fig_ssvi_repricing_error_hist(d: dict) -> Path:
    """Histogram of signed IV error in basis points, with ME line."""
    val = _val_prepared(d)
    err_bps = val["iv_signed_err_bps"]
    me_bps  = float(err_bps.mean())

    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.hist(err_bps, bins=30, color="mediumpurple", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(me_bps, color="red", lw=1.1, ls=":",
               label=f"ME {me_bps:+.1f} bp")
    ax.set_xlabel(_ERR_LABEL)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, "ssvi_repricing_error_hist.png")


def fig_ssvi_repricing_error_vs_price(d: dict) -> Path:
    """Signed IV error (bp) vs market price, calls vs puts."""
    val = _val_prepared(d)
    calls = val[val["option_type"] == "call"]
    puts  = val[val["option_type"] == "put"]

    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.scatter(calls["market_price"], calls["iv_signed_err_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_CALL_COLOR, marker="o", label="calls")
    ax.scatter(puts["market_price"], puts["iv_signed_err_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_PUT_COLOR,  marker="o", label="puts")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Market price (USD)")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, "ssvi_repricing_error_vs_price.png")


def fig_ssvi_repricing_error_vs_moneyness(d: dict) -> Path:
    """Signed IV error (bp) vs forward log-moneyness k, calls vs puts.
    Merges implied_forwards.csv on expiry to recover F, then k = ln(K/F).
    """
    val = _val_prepared(d).merge(
        d["fwd_df"][["expiry", "forward"]], on="expiry", how="left",
    )
    val = val.dropna(subset=["forward"])
    val["fwd_log_m"] = np.log(val["strike"] / val["forward"])

    calls = val[val["option_type"] == "call"]
    puts  = val[val["option_type"] == "put"]

    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.scatter(calls["fwd_log_m"], calls["iv_signed_err_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_CALL_COLOR, marker="o", label="calls")
    ax.scatter(puts["fwd_log_m"], puts["iv_signed_err_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_PUT_COLOR,  marker="o", label="puts")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Forward log-moneyness $k$")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, "ssvi_repricing_error_vs_moneyness.png")


# §4.1.3  Gatheral g(k, T) heatmap
def fig_gatheral_g(d: dict):
    g = d.get("g_surface")
    if g is None:
        warnings.warn("dupire_g_surface.npy not found — skipping g figure.")
        return None
    log_m = d["log_m_grid"]; ttm_int = d["ttm_grid"][:-1]
    K_mesh, T_mesh = np.meshgrid(log_m, ttm_int, indexing="ij")

    vmax = max(abs(float(g.min())), abs(float(g.max())), 1e-6)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    pcm = ax.pcolormesh(T_mesh, K_mesh, g, cmap="RdYlGn",
                        vmin=-vmax, vmax=vmax, shading="auto")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(r"Gatheral density function $g(k, T)$", fontsize=12)
    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=12)
    ax.set_ylabel(r"Forward log-moneyness $k$", fontsize=12)
    plt.tight_layout()
    return _save(fig, "arbitrage_g.png")


# §4.1.4  Dupire local-vol — single 3D + heatmap
def fig_dupire_local_vol_3d(d: dict) -> Path:
    lv = d["local_vol"]; log_m = d["log_m_grid"]; ttm = d["ttm_grid"]
    K_mesh, T_mesh = np.meshgrid(log_m, ttm, indexing="ij")

    # Right-pad so the colorbar clears the z-axis label region
    fig = plt.figure(figsize=(9.0, 6.0))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, lv, cmap=cm.viridis,
                           edgecolor="none", alpha=0.92, rcount=80, ccount=80)
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.12)
    cb.set_label(r"Local volatility", fontsize=12)
    ax.set_xlabel(r"Forward log-moneyness $k$",          labelpad=10, fontsize=12)
    ax.set_ylabel(r"Time to maturity $T$ (years)",       labelpad=10, fontsize=12)
    ax.set_zlabel(r"Local volatility", labelpad=10, fontsize=12)
    ax.view_init(elev=26, azim=-58)
    ax.set_proj_type("ortho")   # orthographic — no perspective distortion
    plt.tight_layout()
    return _save(fig, "dupire_local_vol_3d.png")


def fig_dupire_local_var_heatmap(d: dict) -> Path:
    lv = d["local_vol"]; log_m = d["log_m_grid"]; ttm = d["ttm_grid"]
    K_mesh, T_mesh = np.meshgrid(log_m, ttm, indexing="ij")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    pcm = ax.pcolormesh(T_mesh, K_mesh, lv ** 2, cmap="viridis", shading="auto")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Local variance", fontsize=12)
    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=12)
    ax.set_ylabel(r"Forward log-moneyness $k$", fontsize=12)
    plt.tight_layout()
    return _save(fig, "dupire_local_var_heatmap.png")


# §4.1.4  σ_BS vs σ_loc — six separate smile figures
def fig_iv_vs_local_vol_smiles(d: dict) -> list:
    """Per-T overlay σ_BS(k) (SSVI) vs σ_loc(k) (Dupire); six PNGs, one per
    evenly-spaced maturity."""
    iv    = d["iv_surface"]; lv = d["local_vol"]
    log_m = d["log_m_grid"]; ttm = d["ttm_grid"]
    idxs  = np.linspace(0, len(ttm) - 1, 6, dtype=int)

    outs = []
    for n, j in enumerate(idxs, start=1):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(log_m, iv[:, j], color="C0", lw=1.6,
                label=r"$\sigma_\mathrm{BS}(k)$  (SSVI)")
        ax.plot(log_m, lv[:, j], color="C3", lw=1.6, ls="--",
                label=r"$\sigma_\mathrm{loc}(k)$  (Dupire)")
        ax.set_xlabel(r"Forward log-moneyness $k$")
        ax.set_ylabel("vol")
        ax.set_title(rf"$\sigma_\mathrm{{BS}}$ vs $\sigma_\mathrm{{loc}}$   "
                     rf"at $T = {ttm[j]:.3f}$ yr",
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        outs.append(_save(fig, f"iv_vs_local_vol_smile_T{n}.png"))
    return outs


# §4.1.5  Dupire repricing — three separate figures
# Same styling as the SSVI repricing trio. Errors in IV-space bp against the
# SSVI surface: iv_error_bps = (iv_mc − iv_ssvi)×1e4 (from repricing_errors.csv),
# so the benchmark is SSVI, not the raw market mid.
def _dupire_repricing_liquid(d: dict) -> pd.DataFrame:
    df = d["dup_rep"].dropna(subset=["iv_error_bps"])
    return df[df["ssvi_price"] >= LIQUID_MIN_PRICE]


def fig_dupire_repricing_error_hist(d: dict) -> Path:
    """Histogram of Dupire-MC IV error (bp) vs the SSVI surface, with ME line."""
    liq = _dupire_repricing_liquid(d)
    err_bps = liq["iv_error_bps"]
    me_bps  = float(err_bps.mean())

    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.hist(err_bps, bins=30, color="mediumpurple", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(me_bps, color="red", lw=1.1, ls=":",
               label=f"ME {me_bps:+.1f} bp")
    ax.set_xlabel(_ERR_LABEL)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, "dupire_repricing_error_hist.png")


def fig_dupire_repricing_error_vs_price(d: dict) -> Path:
    """Dupire-MC IV error (bp) vs SSVI BS price, calls vs puts."""
    liq = _dupire_repricing_liquid(d)
    calls = liq[liq["option_type"] == "call"]
    puts  = liq[liq["option_type"] == "put"]

    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.scatter(calls["ssvi_price"], calls["iv_error_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_CALL_COLOR, marker="o", label="calls")
    ax.scatter(puts["ssvi_price"], puts["iv_error_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_PUT_COLOR,  marker="o", label="puts")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("SSVI option price (USD)")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, "dupire_repricing_error_vs_price.png")


def fig_dupire_repricing_error_vs_moneyness(d: dict) -> Path:
    """Dupire-MC IV error (bp) vs forward log-moneyness k, calls vs puts
    (repricing_errors.csv already carries fwd_log_m)."""
    liq = _dupire_repricing_liquid(d)
    calls = liq[liq["option_type"] == "call"]
    puts  = liq[liq["option_type"] == "put"]

    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.scatter(calls["fwd_log_m"], calls["iv_error_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_CALL_COLOR, marker="o", label="calls")
    ax.scatter(puts["fwd_log_m"], puts["iv_error_bps"],
               s=_REPR_SCATTER_S, alpha=0.6,
               color=_PUT_COLOR,  marker="o", label="puts")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Forward log-moneyness $k$")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, "dupire_repricing_error_vs_moneyness.png")


# §4.1.5  Dupire MC vs vanilla — six separate figures
def fig_dupire_mc_scatter_log_all(d: dict) -> Path:
    df = d["dup_rep"].dropna(subset=["iv_error_bps"])
    is_c = (df["option_type"] == "call").values
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(df.loc[is_c,  "ssvi_price"], df.loc[is_c,  "mc_price"],
               s=9, alpha=0.5, color="C0", label="call")
    ax.scatter(df.loc[~is_c, "ssvi_price"], df.loc[~is_c, "mc_price"],
               s=9, alpha=0.5, color="C3", label="put")
    lo = max(0.01, float(df["ssvi_price"].min()))
    hi = float(df["ssvi_price"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("SSVI price (USD)"); ax.set_ylabel("Dupire MC price (USD)")
    ax.set_title("Dupire MC vs SSVI  —  all options, log scale",
                 fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    return _save(fig, "dupire_mc_scatter_log_all.png")


def fig_dupire_mc_scatter_linear_liquid(d: dict) -> Path:
    liq = _dupire_repricing_liquid(d)
    is_c = (liq["option_type"] == "call").values
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(liq.loc[is_c,  "ssvi_price"], liq.loc[is_c,  "mc_price"],
               s=14, alpha=0.65, color="C0", label="call")
    ax.scatter(liq.loc[~is_c, "ssvi_price"], liq.loc[~is_c, "mc_price"],
               s=14, alpha=0.65, color="C3", label="put")
    lo = float(liq["ssvi_price"].min())
    hi = float(liq["ssvi_price"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--")
    ax.set_xlabel("SSVI price (USD)"); ax.set_ylabel("Dupire MC price (USD)")
    ax.set_title(rf"Dupire MC vs SSVI  —  liquid (SSVI $\geq$ "
                 rf"{LIQUID_MIN_PRICE:.0f})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "dupire_mc_scatter_linear_liquid.png")


def fig_dupire_mc_abs_error_vs_ttm(d: dict) -> Path:
    liq = _dupire_repricing_liquid(d)
    is_c = (liq["option_type"] == "call").values
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(liq.loc[is_c,  "ttm"], liq.loc[is_c,  "price_error"],
               s=14, alpha=0.65, color="C0", label="call")
    ax.scatter(liq.loc[~is_c, "ttm"], liq.loc[~is_c, "price_error"],
               s=14, alpha=0.65, color="C3", label="put")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"Time to maturity $T$ (years)"); ax.set_ylabel("Absolute error (USD)")
    ax.set_title("Dupire MC  —  absolute error vs maturity",
                 fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "dupire_mc_abs_error_vs_ttm.png")


def fig_dupire_mc_abs_error_hist(d: dict) -> Path:
    liq = _dupire_repricing_liquid(d)
    vals = liq["price_error"].dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(vals, bins=40, color="C0", alpha=0.78, edgecolor="white")
    ax.axvline(0, color="red", lw=0.7, ls="--")
    ax.axvline(vals.mean(), color="black", lw=1.0,
               label=f"mean {vals.mean():+.2f}")
    ax.set_xlabel("Absolute error (USD)"); ax.set_ylabel("Count")
    ax.set_title("Dupire MC  —  absolute-error distribution",
                 fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "dupire_mc_abs_error_hist.png")


def fig_dupire_mc_pct_error_vs_ttm(d: dict) -> Path:
    liq = _dupire_repricing_liquid(d)
    is_c = (liq["option_type"] == "call").values
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(liq.loc[is_c,  "ttm"], liq.loc[is_c,  "price_error_pct"],
               s=14, alpha=0.65, color="C0", label="call")
    ax.scatter(liq.loc[~is_c, "ttm"], liq.loc[~is_c, "price_error_pct"],
               s=14, alpha=0.65, color="C3", label="put")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"Time to maturity $T$ (years)"); ax.set_ylabel("Percentage error (%)")
    ax.set_title("Dupire MC  —  percentage error vs maturity",
                 fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "dupire_mc_pct_error_vs_ttm.png")


# §4.1.1  Full SSVI fit grid — all 18 slices
def fig_full_ssvi_grid(d: dict) -> Path:
    """SSVI fit per expiry, 6×3 = 18 panels (every fitted slice). Same
    per-panel content as fig_ssvi_fit_grid."""
    ssvi = d["ssvi_df"].sort_values("ttm").reset_index(drop=True)
    iv   = d["iv_df"]

    nrows, ncols = 6, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 22.5))
    axes = axes.flatten()
    n_missing = 0

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= len(ssvi):
            ax.set_visible(False)
            continue
        row    = ssvi.iloc[i]
        expiry = row["expiry"]; ttm = float(row["ttm"])
        theta, phi, rho = float(row["theta"]), float(row["phi"]), float(row["rho"])
        rmse   = float(row["rmse"])

        sl = iv[iv["expiry"] == expiry].sort_values("fwd_log_m")
        if len(sl) == 0:
            ax.set_visible(False); n_missing += 1
            continue

        k_data = sl["fwd_log_m"].values
        w_data = sl["total_var"].values
        k_min, k_max = float(k_data.min()) - 0.02, float(k_data.max()) + 0.02
        k_range = np.linspace(k_min, k_max, 200)
        w_model = ssvi_w(k_range, theta, phi, rho)

        ax.scatter(k_data, w_data, s=6, color="black", alpha=0.7, zorder=2)
        ax.plot(k_range, w_model, color="crimson", lw=1.6, zorder=3)

        ax.set_title(f"{expiry}   T = {ttm:.2f} years", fontsize=10)
        ax.set_xlabel(r"$k$", fontsize=12)
        ax.set_ylabel(r"$w$", fontsize=12)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.text(0.04, 0.96, f"RMSE {rmse*100:.2f} vp",
                transform=ax.transAxes, fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          alpha=0.85, edgecolor="none"))

    if n_missing:
        warnings.warn(
            f"Full SSVI grid: {n_missing}/{len(ssvi)} slices had no iv rows "
            f"and were hidden. Refresh upstream SSVI pipeline to restore.")

    plt.tight_layout()
    return _save(fig, "full_ssvi_grid.png")


# §4.1.1  Market IV smiles — single panel, all expiries
def fig_market_iv_smiles(d: dict) -> Path:
    """Raw OptionMetrics market IV per expiry, all maturities on one axes,
    coloured by TTM. Uses impl_volatility (raw OM IV), not the fitted iv
    column, to show the data the surface was calibrated against."""
    iv = d["iv_df"]
    expiries = (iv.groupby("expiry")["ttm"].first()
                  .sort_values().index.tolist())
    ttms = np.array([float(iv.loc[iv["expiry"] == e, "ttm"].iloc[0])
                      for e in expiries])
    norm = mpl.colors.Normalize(vmin=float(ttms.min()), vmax=float(ttms.max()))
    cmap = cm.viridis

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=_SMILE_BUBBLE_FIGSIZE)
    for exp, T in zip(expiries, ttms):
        sl = iv[iv["expiry"] == exp].sort_values("fwd_log_m")
        ax.plot(sl["fwd_log_m"], sl["impl_volatility"],
                color=cmap(norm(T)), lw=1.0, marker="o", ms=2.5,
                alpha=0.85)

    # Fixed-width colourbar slot so this canvas matches the bubble figure
    # (which uses an invisible slot of the same size).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=_CBAR_SIZE, pad=_CBAR_PAD)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"Time to maturity $T$ (years)", fontsize=12)
    ax.set_xlabel(r"Forward log-moneyness $k$", fontsize=12)
    ax.set_ylabel(r"Market implied volatility $\sigma_\mathrm{mkt}$", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "market_iv_smiles.png", tight=False)


# §4.1.1  Liquidity bubble plot
def fig_liquidity_bubble(d: dict) -> Path:
    """(k, T) bubble plot, marker area ∝ √(open interest) (≈ vega-weighted
    emphasis). Calls blue, puts red."""
    iv = d["iv_df"].copy()
    oi = iv["open_interest"].to_numpy(dtype=float)
    # Scale area to ~[5, 250] pts² so one outlier can't dominate
    s_min, s_max = 4.0, 250.0
    rank = np.sqrt(np.maximum(oi, 1.0))
    rank = (rank - rank.min()) / (rank.max() - rank.min() + 1e-12)
    area = s_min + (s_max - s_min) * rank

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=_SMILE_BUBBLE_FIGSIZE)
    for opt, color, label in [
        ("call", "tab:blue", "calls"),
        ("put",  "tab:red",  "puts"),
    ]:
        mask = (iv["option_type"] == opt).to_numpy()
        ax.scatter(iv.loc[mask, "fwd_log_m"], iv.loc[mask, "ttm"],
                   s=area[mask], alpha=0.35, color=color,
                   edgecolors="none", label=label)

    # Legend reference markers (open-interest scale)
    ref_oi = [1_000, 10_000, 100_000]
    ref_rank = np.sqrt(np.array(ref_oi, dtype=float))
    ref_rank = (ref_rank - np.sqrt(max(oi.min(), 1.0))) / \
               (np.sqrt(oi.max()) - np.sqrt(max(oi.min(), 1.0)) + 1e-12)
    ref_area = s_min + (s_max - s_min) * ref_rank
    for r_oi, r_area in zip(ref_oi, ref_area):
        ax.scatter([], [], s=max(r_area, 4.0), color="grey",
                   alpha=0.45, edgecolors="none",
                   label=f"OI {r_oi:,}")

    # Invisible colourbar slot to match the market-IV-smiles canvas
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=_CBAR_SIZE, pad=_CBAR_PAD)
    cax.set_visible(False)

    ax.set_xlabel(r"Forward log-moneyness $k$", fontsize=12)
    ax.set_ylabel(r"Time to maturity $T$ (years)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, scatterpoints=1, labelspacing=1.0,
               borderpad=0.8)
    plt.tight_layout()
    return _save(fig, "liquidity_bubble.png", tight=False)


# §4.1.1  Iso-strike grid over (k, T)
def fig_strike_isolines(d: dict) -> Path:
    """(S,T) → (k,T) coordinate transform: for fixed K,
    k(T) = log(K/S0) − (r−q)T is a line of slope −(r−q). Iso-strike lines at
    5% steps from 50–150% of spot; K=S0 dashed."""
    iv  = d["iv_df"]
    mkt = d["market"]
    S0, r, q = float(mkt["S"]), float(mkt["r"]), float(mkt["q"])

    ttm = iv["ttm"].to_numpy()
    T_lo, T_hi = float(ttm.min()), float(ttm.max())
    T_pad      = 0.02 * (T_hi - T_lo)
    T_line     = np.linspace(T_lo - T_pad, T_hi + T_pad, 200)

    # 5% strike-ratio spacing
    moneyness_grid = np.arange(0.50, 1.50 + 1e-9, 0.05)
    K_grid = S0 * moneyness_grid

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    for K, m_ratio in zip(K_grid, moneyness_grid):
        k_line = np.log(K / S0) - (r - q) * T_line
        ls = "--" if abs(m_ratio - 1.0) < 1e-9 else "-"
        lw = 1.4  if abs(m_ratio - 1.0) < 1e-9 else 1.0
        ax.plot(k_line, T_line, color="black", lw=lw, ls=ls, alpha=0.85)

    ax.set_xlabel(r"Forward log-moneyness $k = \log(K / F(T))$", fontsize=12)
    ax.set_ylabel(r"Time to maturity $T$ (years)", fontsize=12)
    ax.set_xlim(float(iv["fwd_log_m"].min()) - 0.03,
                float(iv["fwd_log_m"].max()) + 0.03)
    ax.set_ylim(T_lo - T_pad, T_hi + T_pad)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "strike_isolines.png")


def fig_dupire_mc_pct_error_hist(d: dict) -> Path:
    liq = _dupire_repricing_liquid(d)
    vals = liq["price_error_pct"].dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(vals, bins=40, color="C0", alpha=0.78, edgecolor="white")
    ax.axvline(0, color="red", lw=0.7, ls="--")
    ax.axvline(vals.mean(), color="black", lw=1.0,
               label=f"mean {vals.mean():+.2f}%")
    ax.set_xlabel("Percentage error (%)"); ax.set_ylabel("Count")
    ax.set_title("Dupire MC  —  percentage-error distribution",
                 fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "dupire_mc_pct_error_hist.png")


# Main
def main():
    print(f"Output dir: {OUT_DIR.resolve()}")
    print("Loading pipeline artefacts ...")
    d = load_data()

    figures = [
        # §4.1.1
        ("§4.1.1  SSVI fit grid",                 fig_ssvi_fit_grid),
        ("§4.1.1  Full SSVI fit grid (18 slices)",fig_full_ssvi_grid),
        ("§4.1.1  Market IV smiles",              fig_market_iv_smiles),
        ("§4.1.1  Liquidity bubble (OI)",         fig_liquidity_bubble),
        ("§4.1.1  Strike iso-lines over (k,T)",   fig_strike_isolines),
        ("§4.1.1  SSVI parameter table",          fig_ssvi_params_table),
        ("appendix SSVI params (shared + slices)",fig_ssvi_params_table_full),
        ("§4.1.1  σ(k,T) 3D surface",             fig_iv_surface_3d),
        ("§4.1.1  w(k,T) 3D surface with slices", fig_tv_surface_with_slices_3d),
        ("§4.1.1  ρ(T), θ(T), φ(T) term struct",  fig_term_structure),
        # §4.1.2 — SSVI repricing diagnostics
        ("§4.1.2  SSVI repr.  —  computed vs interp IV",  fig_ssvi_repricing_iv_scatter),
        ("§4.1.2  SSVI repr.  —  signed-error hist (vp)", fig_ssvi_repricing_error_hist),
        ("§4.1.2  SSVI repr.  —  error vs market price",  fig_ssvi_repricing_error_vs_price),
        ("§4.1.2  SSVI repr.  —  error vs moneyness $k$", fig_ssvi_repricing_error_vs_moneyness),
        # §4.1.3
        ("§4.1.3  Gatheral g(k, T)",              fig_gatheral_g),
        # §4.1.4
        ("§4.1.4  Dupire σ_loc 3D",               fig_dupire_local_vol_3d),
        ("§4.1.4  Dupire σ²_loc heatmap",         fig_dupire_local_var_heatmap),
        ("§4.1.4  σ_BS vs σ_loc (6 smiles)",      fig_iv_vs_local_vol_smiles),
        # §4.1.5
        ("§4.1.5  Dupire repr.  —  IV err hist",        fig_dupire_repricing_error_hist),
        ("§4.1.5  Dupire repr.  —  IV err vs price",    fig_dupire_repricing_error_vs_price),
        ("§4.1.5  Dupire repr.  —  IV err vs moneyness $k$", fig_dupire_repricing_error_vs_moneyness),
        # §4.1.5 detail
        ("§4.1.5  Dupire MC scatter (log)",       fig_dupire_mc_scatter_log_all),
        ("§4.1.5  Dupire MC scatter (linear)",    fig_dupire_mc_scatter_linear_liquid),
        ("§4.1.5  Dupire MC abs err vs T",        fig_dupire_mc_abs_error_vs_ttm),
        ("§4.1.5  Dupire MC abs err hist",        fig_dupire_mc_abs_error_hist),
        ("§4.1.5  Dupire MC pct err vs T",        fig_dupire_mc_pct_error_vs_ttm),
        ("§4.1.5  Dupire MC pct err hist",        fig_dupire_mc_pct_error_hist),
    ]

    print()
    n_ok = 0; n_skip = 0; n_fail = 0
    for label, fn in figures:
        try:
            out = fn(d)
            if out is None:
                print(f"  SKIP  {label}"); n_skip += 1
            elif isinstance(out, list):
                names = ", ".join(o.name for o in out)
                print(f"  OK    {label:<42}  {names}")
                n_ok += len(out)
            else:
                print(f"  OK    {label:<42}  {out.name}"); n_ok += 1
        except Exception as exc:
            print(f"  FAIL  {label:<42}  {exc!r}"); n_fail += 1

    print(f"\n{n_ok} files written, {n_skip} skipped, {n_fail} failed.")
    print(f"All outputs under {OUT_DIR}/")


if __name__ == "__main__":
    main()

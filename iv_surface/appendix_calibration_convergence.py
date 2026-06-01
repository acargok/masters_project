#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
appendix_calibration_convergence.py — thesis appendix figure pack
====================================================================
Depicts the Heston and Bergomi calibration *process* (rather than just the
final fitted parameters): differential-evolution loss decay, per-parameter
convergence traces over generations, 1D objective slices around the
calibrated optimum, and (Bergomi only) the target-vs-model curve evolution
over generations for the two-stage benchmark fit.

Stand-alone script. Re-runs the two production calibrations with callback
hooks that capture the best-of-population state per generation, caches the
traces to CSV, and renders the figures.

Outputs:
    iv_surface/appendix_plots/                # generated PNGs
    iv_surface/appendix_plots/cache/          # per-generation trace CSVs

First run wall-time: ~3 min (Heston DE ~2 min on 1499 options + Bergomi
two-stage DE ~30 s each). Subsequent runs ~5 s if the cache is present.
"""
import json
import sys
import warnings
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import optimize

warnings.filterwarnings("ignore")

# ───────────────────────── Paths ─────────────────────────
HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
LSV_DIR     = ROOT / "lsv"
BERGOMI_DIR = ROOT / "lsv_bergomi"
OUT_DIR     = HERE / "appendix_plots"
CACHE_DIR   = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Both lsv/ and lsv_bergomi/ are added to sys.path so the calibration
# objective functions live at a canonical module name (heston_calibration,
# bergomi_param_calibration). This matters for multiprocessing — DE workers
# need to re-import the objective by its module name when pickling tasks.
for p in (LSV_DIR, BERGOMI_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


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


def _save(fig, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


# ─────────────── Pipeline module loaders ──────────────────
# Standard imports (works because LSV_DIR and BERGOMI_DIR are on sys.path).
import heston_calibration as _heston_module
import bergomi_param_calibration as _bparam_module


def heston_module():
    return _heston_module


def bparam_module():
    return _bparam_module


# ════════════════════ Trace capture ══════════════════════════
# scipy.optimize.differential_evolution will hand the callback either
# (x, convergence)  on the legacy API, or  (intermediate_result,)  on the
# newer one. We support both so the script doesn't break across scipy
# versions.

def _make_de_callback(trace: list, objective: Callable):
    def cb(*args, **kwargs):
        if len(args) >= 1 and hasattr(args[0], "x") and hasattr(args[0], "fun"):
            x   = np.asarray(args[0].x, dtype=float).copy()
            fun = float(args[0].fun)
        else:
            x   = np.asarray(args[0], dtype=float).copy()
            fun = float(objective(x))
        trace.append((x, fun))
        return False
    return cb


def _trace_to_df(trace: List[Tuple[np.ndarray, float]],
                  param_names: List[str]) -> pd.DataFrame:
    rows = []
    for gen, (x, fun) in enumerate(trace, start=1):
        row = {"gen": gen, "objective": fun}
        row.update({n: float(v) for n, v in zip(param_names, x)})
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════ Heston calibration trace ═══════════════
_HESTON_PARAMS = ["kappa", "theta", "xi", "rho", "V0"]
_HESTON_BOUNDS = [
    (0.1, 10.0),      # kappa
    (0.005, 0.50),    # theta
    (0.05, 2.0),      # xi
    (-0.99, 0.10),    # rho
    (0.005, 0.50),    # V0
]
_HESTON_TRACE_CSV = CACHE_DIR / "heston_de_trace.csv"
_HESTON_FINAL_JSON = CACHE_DIR / "heston_final.json"


def _run_heston_de_with_trace():
    """Re-run Heston DE on the production calibration pool, capturing per-
    generation best state. Cached to CSV; subsequent calls read the cache."""
    if _HESTON_TRACE_CSV.exists() and _HESTON_FINAL_JSON.exists():
        return (pd.read_csv(_HESTON_TRACE_CSV),
                json.loads(_HESTON_FINAL_JSON.read_text()))

    print("  Running Heston DE with trace capture (~2 min)…")
    h = heston_module()
    md = h.load_market_data()
    K_arr, T_arr, iv_mkt, mkt_px, vegas, opt_arr = h.prepare_calibration_data(md)
    S, r, q = md["S"], md["r"], md["q"]

    # The objective lives at module level (h.calibration_objective) so
    # multiprocessing workers can pickle it; pass the per-option data via args.
    obj_args = (S, r, q, K_arr, T_arr, mkt_px, vegas, opt_arr)
    eval_obj = lambda x: h.calibration_objective(x, *obj_args)

    trace: list = []
    cb = _make_de_callback(trace, eval_obj)

    de = optimize.differential_evolution(
        h.calibration_objective, _HESTON_BOUNDS, args=obj_args,
        seed=42, maxiter=h.MAX_ITER,
        tol=1e-10, atol=1e-10,
        polish=False, workers=h.N_WORKERS, updating="deferred",
        disp=False, callback=cb,
    )
    # Local NM polish.
    pol = optimize.minimize(
        h.calibration_objective, de.x, args=obj_args, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )

    df = _trace_to_df(trace, _HESTON_PARAMS)
    df.to_csv(_HESTON_TRACE_CSV, index=False)
    final = {
        "x_de": list(map(float, de.x)),
        "fun_de": float(de.fun),
        "x_polish": list(map(float, pol.x)),
        "fun_polish": float(pol.fun),
        "S": float(S), "r": float(r), "q": float(q),
    }
    _HESTON_FINAL_JSON.write_text(json.dumps(final, indent=2))
    return df, final


# ════════════════════ Bergomi Stage-1 trace ══════════════════
_BG1_PARAMS = ["nu", "theta", "kappa1", "kappa2", "rho12"]
_BG1_BOUNDS = [
    (0.5, 2.0),     # nu
    (0.1, 0.5),     # theta
    (3.0, 20.0),    # kappa1
    (0.05, 0.6),    # kappa2
    (-0.99, 0.8),   # rho12
]
_BG1_TRACE_CSV = CACHE_DIR / "bergomi_stage1_de_trace.csv"
_BG1_FINAL_JSON = CACHE_DIR / "bergomi_stage1_final.json"


def _run_bergomi_stage1_with_trace():
    if _BG1_TRACE_CSV.exists() and _BG1_FINAL_JSON.exists():
        return (pd.read_csv(_BG1_TRACE_CSV),
                json.loads(_BG1_FINAL_JSON.read_text()))

    print("  Running Bergomi Stage 1 DE with trace capture (~30 s)…")
    b = bparam_module()
    T_grid = b.T_GRID_VOLOFVOL
    target = b.vol_of_vol_benchmark(T_grid)
    obj_args = (T_grid, target)
    eval_obj = lambda x: b.stage1_objective(x, *obj_args)

    trace: list = []
    cb = _make_de_callback(trace, eval_obj)

    de = optimize.differential_evolution(
        b.stage1_objective, _BG1_BOUNDS, args=obj_args,
        seed=b.SEED, maxiter=b.DE_MAXITER,
        tol=1e-10, atol=1e-10,
        polish=False, workers=b.N_WORKERS, updating="deferred",
        disp=False, callback=cb,
    )
    pol = optimize.minimize(
        b.stage1_objective, de.x, args=obj_args,
        method="L-BFGS-B", bounds=_BG1_BOUNDS,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )

    df = _trace_to_df(trace, _BG1_PARAMS)
    df.to_csv(_BG1_TRACE_CSV, index=False)
    final = {
        "x_de": list(map(float, de.x)),
        "fun_de": float(de.fun),
        "x_polish": list(map(float, pol.x)),
        "fun_polish": float(pol.fun),
        "T_grid":  T_grid.tolist(),
        "target":  target.tolist(),
    }
    _BG1_FINAL_JSON.write_text(json.dumps(final, indent=2))
    return df, final


# ════════════════════ Bergomi Stage-2 trace ══════════════════
_BG2_PARAMS = ["rho1", "chi"]
_BG2_BOUNDS = [
    (-0.99, 0.0),
    (-1.0, 1.0),
]
_BG2_TRACE_CSV = CACHE_DIR / "bergomi_stage2_de_trace.csv"
_BG2_FINAL_JSON = CACHE_DIR / "bergomi_stage2_final.json"


def _run_bergomi_stage2_with_trace(stage1: dict):
    if _BG2_TRACE_CSV.exists() and _BG2_FINAL_JSON.exists():
        return (pd.read_csv(_BG2_TRACE_CSV),
                json.loads(_BG2_FINAL_JSON.read_text()))

    print("  Running Bergomi Stage 2 DE with trace capture (~30 s)…")
    b = bparam_module()

    # Stage 1 result as a dict (matches bparam.calibrate_stage1 return).
    nu, theta, k1, k2, rho12 = stage1["x_polish"]
    s1 = {"nu": nu, "theta": theta, "kappa1": k1, "kappa2": k2, "rho12": rho12}

    # Stage 2 target: SSVI-derived empirical skew on a fixed maturity grid.
    iv_surface = np.load(ROOT / "iv_surface" / "arrays" / "iv_surface.npy")
    log_m_grid = np.load(ROOT / "iv_surface" / "arrays" / "log_m_grid.npy")
    ttm_grid   = np.load(ROOT / "iv_surface" / "arrays" / "ttm_grid.npy")
    T2 = ttm_grid[(ttm_grid >= 0.25) & (ttm_grid <= 2.0)]
    if len(T2) > 25:
        T2 = T2[np.linspace(0, len(T2) - 1, 25, dtype=int)]
    target = b.empirical_atmf_skew(iv_surface, log_m_grid, ttm_grid, T2)

    obj_args = (T2, target, s1)
    eval_obj = lambda x: b.stage2_objective(x, *obj_args)

    trace: list = []
    cb = _make_de_callback(trace, eval_obj)

    de = optimize.differential_evolution(
        b.stage2_objective, _BG2_BOUNDS, args=obj_args,
        seed=b.SEED, maxiter=b.DE_MAXITER,
        tol=1e-10, atol=1e-10,
        polish=False, workers=b.N_WORKERS, updating="deferred",
        disp=False, callback=cb,
    )
    pol = optimize.minimize(
        b.stage2_objective, de.x, args=obj_args,
        method="L-BFGS-B", bounds=_BG2_BOUNDS,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )

    df = _trace_to_df(trace, _BG2_PARAMS)
    df.to_csv(_BG2_TRACE_CSV, index=False)
    final = {
        "x_de":   list(map(float, de.x)),
        "fun_de": float(de.fun),
        "x_polish":   list(map(float, pol.x)),
        "fun_polish": float(pol.fun),
        "T_grid": T2.tolist(),
        "target": target.tolist(),
        "stage1": s1,
    }
    _BG2_FINAL_JSON.write_text(json.dumps(final, indent=2))
    return df, final


# ════════════════════ Figures: Heston ════════════════════════
def fig_heston_loss(df: pd.DataFrame, final: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.semilogy(df["gen"], df["objective"], color="C0", lw=1.5,
                label="Differential Evolution best-of-generation")
    ax.axhline(final["fun_de"], color="C0", lw=0.7, ls=":",
                label=f"Differential Evolution final  {final['fun_de']:.3e}")
    ax.axhline(final["fun_polish"], color="C3", lw=0.9, ls="--",
                label=f"Nelder-Mead polish  {final['fun_polish']:.3e}")
    ax.set_xlabel("Differential Evolution generation")
    ax.set_ylabel("Objective (IV-approx SSE, log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, framealpha=1.0).set_zorder(10)
    plt.tight_layout()
    return _save(fig, "heston_loss_convergence.png")


def fig_heston_param_traces(df: pd.DataFrame, final: dict) -> Path:
    """All five Heston parameters traced over Differential Evolution
    generations on a single axes object. Parameters with similar magnitudes
    share a y-axis: θ and V₀ both sit on the "variance scale" axis since
    they're both small positive variance levels; κ, ξ, ρ each get their own
    axis. The Nelder-Mead polish endpoint for each parameter is marked as a
    thin dashed horizontal in the matching colour."""
    gen   = df["gen"].to_numpy()
    kappa = df["kappa"].to_numpy()
    theta = df["theta"].to_numpy()
    xi    = df["xi"].to_numpy()
    rho   = df["rho"].to_numpy()
    V0    = df["V0"].to_numpy()
    p_kap, p_th, p_xi, p_rho, p_V0 = final["x_polish"]

    # Colours fixed per parameter (consistent across the legend and the
    # polish-endpoint dashed lines).
    c_kappa, c_xi, c_theta, c_V0, c_rho = "C0", "C1", "C2", "C3", "C4"

    fig, ax_k = plt.subplots(figsize=(9.0, 4.5))
    ax_xi  = ax_k.twinx()
    ax_var = ax_k.twinx(); ax_var.spines["right"].set_position(("outward", 60))
    ax_rho = ax_k.twinx(); ax_rho.spines["right"].set_position(("outward", 120))

    l_k,  = ax_k.plot(gen,   kappa, color=c_kappa, lw=1.5, label=r"$\kappa$")
    l_xi, = ax_xi.plot(gen,  xi,    color=c_xi,    lw=1.5, label=r"$\xi$")
    l_th, = ax_var.plot(gen, theta, color=c_theta, lw=1.5, label=r"$\theta$")
    l_v0, = ax_var.plot(gen, V0,    color=c_V0,    lw=1.5, label=r"$V_0$")
    l_r,  = ax_rho.plot(gen, rho,   color=c_rho,   lw=1.5, label=r"$\rho$")

    ax_k.axhline(p_kap, color=c_kappa, lw=0.8, ls="--", alpha=0.7)
    ax_xi.axhline(p_xi, color=c_xi,    lw=0.8, ls="--", alpha=0.7)
    ax_var.axhline(p_th, color=c_theta, lw=0.8, ls="--", alpha=0.7)
    ax_var.axhline(p_V0, color=c_V0,    lw=0.8, ls="--", alpha=0.7)
    ax_rho.axhline(p_rho, color=c_rho,  lw=0.8, ls="--", alpha=0.7)

    ax_k.set_xlabel("Differential Evolution generation")
    ax_k.set_ylabel(r"$\kappa$",          color=c_kappa)
    ax_xi.set_ylabel(r"$\xi$",            color=c_xi)
    ax_var.set_ylabel(r"$\theta,\ V_0$")
    ax_rho.set_ylabel(r"$\rho$",          color=c_rho)
    for ax_, c in [(ax_k, c_kappa), (ax_xi, c_xi), (ax_rho, c_rho)]:
        ax_.tick_params(axis="y", labelcolor=c)
    ax_k.grid(True, alpha=0.3)

    # Combined legend; polish dashed lines explained by a single proxy entry.
    from matplotlib.lines import Line2D
    proxy_polish = Line2D([], [], color="grey", lw=0.8, ls="--",
                           label="Nelder-Mead polish (per param)")
    handles = [l_k, l_xi, l_th, l_v0, l_r, proxy_polish]
    # Anchor the legend to the LAST twinx axis (ax_rho) so it paints above
    # the lines on every twin axis (see Bergomi Stage-1 trace for context).
    _lg = ax_rho.legend(handles, [h.get_label() for h in handles],
                         loc="best", fontsize=9, ncol=2, framealpha=1.0)
    _lg.set_zorder(10)
    plt.tight_layout()
    return _save(fig, "heston_param_traces.png")


def fig_heston_objective_slices(final: dict) -> Path:
    """1D cuts through the objective at the calibrated optimum — sweeps each
    parameter ±25% (within its bounds), holding the others fixed at the
    NM-polished optimum. Curvature shows how tight each parameter is."""
    h = heston_module()
    md = h.load_market_data()
    K_arr, T_arr, iv_mkt, mkt_px, vegas, opt_arr = h.prepare_calibration_data(md)
    S, r, q = md["S"], md["r"], md["q"]

    x_star = np.array(final["x_polish"], dtype=float)
    pretty = {"kappa": r"$\kappa$", "theta": r"$\theta$", "xi": r"$\xi$",
              "rho": r"$\rho$", "V0": r"$V_0$"}

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.4))
    axes = axes.flatten()
    for i, name in enumerate(_HESTON_PARAMS):
        ax = axes[i]
        lo, hi = _HESTON_BOUNDS[i]
        c = x_star[i]
        half = 0.25 * max(abs(c), (hi - lo) * 0.05)
        sweep_lo = max(lo, c - half)
        sweep_hi = min(hi, c + half)
        sweep = np.linspace(sweep_lo, sweep_hi, 41)
        vals = []
        for v in sweep:
            x = x_star.copy(); x[i] = v
            vals.append(h.calibration_objective(
                x, S, r, q, K_arr, T_arr, mkt_px, vegas, opt_arr))
        vals = np.asarray(vals)
        ax.plot(sweep, vals, color="C0", lw=1.5)
        ax.axvline(c, color="C3", lw=0.9, ls="--",
                   label=f"$x^*$ = {c:+.4f}")
        ax.set_title(pretty[name], fontsize=12)
        ax.set_xlabel(pretty[name])
        ax.set_ylabel("Objective")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=1.0).set_zorder(10)
    axes[-1].set_visible(False)
    plt.tight_layout()
    return _save(fig, "heston_objective_slices.png")


# ════════════════════ Figures: Bergomi Stage 1 ═══════════════
def _loss_curve(df, final, title, fname):
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.semilogy(df["gen"], df["objective"], color="C0", lw=1.5,
                label="Differential Evolution best-of-generation")
    ax.axhline(final["fun_de"], color="C0", lw=0.7, ls=":",
                label=f"Differential Evolution final  {final['fun_de']:.3e}")
    ax.axhline(final["fun_polish"], color="C3", lw=0.9, ls="--",
                label=f"Nelder-Mead polish  {final['fun_polish']:.3e}")
    ax.set_xlabel("Differential Evolution generation")
    ax.set_ylabel("Objective (SSE, log scale)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, framealpha=1.0).set_zorder(10)
    plt.tight_layout()
    return _save(fig, fname)


def fig_bergomi_stage1_loss(df, final) -> Path:
    return _loss_curve(df, final,
                       "Bergomi Stage 1 — vol-of-vol fit convergence",
                       "bergomi_stage1_loss.png")


def fig_bergomi_stage1_param_traces(df: pd.DataFrame, final: dict) -> Path:
    """All five Bergomi Stage-1 parameters traced over Differential
    Evolution generations on a single axes object. Grouped axes: κ₁ (large),
    ν (medium 0–2), θ + κ₂ share the "small-positive" scale (both ≲ 0.6),
    and ρ₁₂ gets its own signed-correlation axis."""
    gen    = df["gen"].to_numpy()
    nu     = df["nu"].to_numpy()
    theta  = df["theta"].to_numpy()
    kappa1 = df["kappa1"].to_numpy()
    kappa2 = df["kappa2"].to_numpy()
    rho12  = df["rho12"].to_numpy()
    p_nu, p_th, p_k1, p_k2, p_r12 = final["x_polish"]

    c_k1, c_nu, c_theta, c_k2, c_r12 = "C0", "C1", "C2", "C3", "C4"

    fig, ax_k1 = plt.subplots(figsize=(9.0, 4.5))
    ax_nu   = ax_k1.twinx()
    ax_sm   = ax_k1.twinx(); ax_sm.spines["right"].set_position(("outward", 60))
    ax_r12  = ax_k1.twinx(); ax_r12.spines["right"].set_position(("outward", 120))

    l_k1, = ax_k1.plot(gen,  kappa1, color=c_k1,    lw=1.5, label=r"$\kappa_1$")
    l_nu, = ax_nu.plot(gen,  nu,     color=c_nu,    lw=1.5, label=r"$\nu$")
    l_th, = ax_sm.plot(gen,  theta,  color=c_theta, lw=1.5, label=r"$\theta$")
    l_k2, = ax_sm.plot(gen,  kappa2, color=c_k2,    lw=1.5, label=r"$\kappa_2$")
    l_r,  = ax_r12.plot(gen, rho12,  color=c_r12,   lw=1.5, label=r"$\rho_{12}$")

    ax_k1.axhline(p_k1,  color=c_k1,    lw=0.8, ls="--", alpha=0.7)
    ax_nu.axhline(p_nu,  color=c_nu,    lw=0.8, ls="--", alpha=0.7)
    ax_sm.axhline(p_th,  color=c_theta, lw=0.8, ls="--", alpha=0.7)
    ax_sm.axhline(p_k2,  color=c_k2,    lw=0.8, ls="--", alpha=0.7)
    ax_r12.axhline(p_r12, color=c_r12,  lw=0.8, ls="--", alpha=0.7)

    ax_k1.set_xlabel("Differential Evolution generation")
    ax_k1.set_ylabel(r"$\kappa_1$",     color=c_k1)
    ax_nu.set_ylabel(r"$\nu$",          color=c_nu)
    ax_sm.set_ylabel(r"$\theta,\ \kappa_2$")
    ax_r12.set_ylabel(r"$\rho_{12}$",   color=c_r12)
    for ax_, c in [(ax_k1, c_k1), (ax_nu, c_nu), (ax_r12, c_r12)]:
        ax_.tick_params(axis="y", labelcolor=c)
    ax_k1.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    proxy_polish = Line2D([], [], color="grey", lw=0.8, ls="--",
                           label="Nelder-Mead polish (per param)")
    handles = [l_k1, l_nu, l_th, l_k2, l_r, proxy_polish]
    # Anchor the legend to the LAST twinx axis (ax_r12). Each twinx draws on
    # top of the previously created axes, so a legend on ax_k1 would be
    # painted over by lines living on ax_nu / ax_sm / ax_r12; putting it on
    # the topmost axis keeps it visually above all four traces.
    _lg = ax_r12.legend(handles, [h.get_label() for h in handles],
                         loc="best", fontsize=9, ncol=2, framealpha=1.0)
    _lg.set_zorder(10)
    plt.tight_layout()
    return _save(fig, "bergomi_stage1_param_traces.png")


def fig_bergomi_stage1_fit_evolution(df: pd.DataFrame, final: dict) -> Path:
    """Vol-of-vol model curve at progressive DE generations vs the
    power-law benchmark."""
    b = bparam_module()
    T_grid = np.asarray(final["T_grid"])
    target = np.asarray(final["target"])

    n_gen = len(df)
    # Pick log-spaced snapshot generations.
    gens = sorted(set(np.unique(np.round(np.logspace(
        0, np.log10(n_gen), 8)).astype(int))))
    gens = [g for g in gens if 1 <= g <= n_gen]
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=min(gens), vmax=max(gens))

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.plot(T_grid, target * 100.0, "k-", lw=1.8, label="Target", zorder=4)

    for g in gens:
        row = df.iloc[g - 1]
        model = b.vol_of_vol_model(T_grid,
                                    row["nu"], row["theta"],
                                    row["kappa1"], row["kappa2"], row["rho12"])
        ax.plot(T_grid, model * 100.0, color=cmap(norm(g)), lw=1.1,
                alpha=0.85, label=f"gen {g}")

    # Final (polished) curve on top.
    nu, th, k1, k2, r12 = final["x_polish"]
    model_final = b.vol_of_vol_model(T_grid, nu, th, k1, k2, r12)
    ax.plot(T_grid, model_final * 100.0, color="C3", lw=2.0, ls="--",
            label="Nelder-Mead polish", zorder=5)

    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=12)
    ax.set_ylabel(r"Vol-of-vol $\nu^B(T)$ (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=1.0).set_zorder(10)
    plt.tight_layout()
    return _save(fig, "bergomi_stage1_fit_evolution.png")


# ════════════════════ Figures: Bergomi Stage 2 ═══════════════
def fig_bergomi_stage2_loss(df, final) -> Path:
    return _loss_curve(df, final,
                       "Bergomi Stage 2 — ATMF skew fit convergence",
                       "bergomi_stage2_loss.png")


def fig_bergomi_stage2_param_traces(df: pd.DataFrame, final: dict) -> Path:
    """Both Stage-2 parameters traced on a single shared axis — ρ₁ and χ
    are both signed quantities in [-1, 1], so a single y-axis is natural."""
    gen   = df["gen"].to_numpy()
    rho1  = df["rho1"].to_numpy()
    chi   = df["chi"].to_numpy()
    p_r1, p_chi = final["x_polish"]

    c_r1, c_chi = "C0", "C1"

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    l_r1, = ax.plot(gen, rho1, color=c_r1,  lw=1.5, label=r"$\rho_1$")
    l_ch, = ax.plot(gen, chi,  color=c_chi, lw=1.5, label=r"$\chi$")
    ax.axhline(p_r1,  color=c_r1,  lw=0.8, ls="--", alpha=0.7)
    ax.axhline(p_chi, color=c_chi, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0,     color="black", lw=0.4)

    ax.set_xlabel("Differential Evolution generation")
    ax.set_ylabel(r"$\rho_1,\ \chi$")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    proxy_polish = Line2D([], [], color="grey", lw=0.8, ls="--",
                           label="Nelder-Mead polish (per param)")
    handles = [l_r1, l_ch, proxy_polish]
    _lg = ax.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=9, ncol=2, framealpha=1.0)

    _lg.set_zorder(10)
    plt.tight_layout()
    return _save(fig, "bergomi_stage2_param_traces.png")


def fig_bergomi_stage2_fit_evolution(df: pd.DataFrame, final: dict) -> Path:
    """ATMF skew model curve at progressive DE generations vs the SSVI
    empirical skew."""
    b = bparam_module()
    T_grid = np.asarray(final["T_grid"])
    target = np.asarray(final["target"])
    s1 = final["stage1"]

    n_gen = len(df)
    gens = sorted(set(np.unique(np.round(np.logspace(
        0, np.log10(n_gen), 8)).astype(int))))
    gens = [g for g in gens if 1 <= g <= n_gen]
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=min(gens), vmax=max(gens))

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.plot(T_grid, target, "k-", lw=1.8, label="SSVI ATMF skew", zorder=4)

    for g in gens:
        row = df.iloc[g - 1]
        rho1 = row["rho1"]; chi = row["chi"]
        rho2 = b.derive_rho2(rho1, chi, s1["rho12"])
        model = b.skew_order1_model(
            T_grid, s1["nu"], s1["theta"],
            s1["kappa1"], s1["kappa2"], s1["rho12"], rho1, rho2)
        ax.plot(T_grid, model, color=cmap(norm(g)), lw=1.1,
                alpha=0.85, label=f"gen {g}")

    rho1, chi = final["x_polish"]
    rho2 = b.derive_rho2(rho1, chi, s1["rho12"])
    model_final = b.skew_order1_model(
        T_grid, s1["nu"], s1["theta"],
        s1["kappa1"], s1["kappa2"], s1["rho12"], rho1, rho2)
    ax.plot(T_grid, model_final, color="C3", lw=2.0, ls="--",
            label="Nelder-Mead polish", zorder=5)
    ax.axhline(0.0, color="grey", lw=0.5, ls="--")

    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=12)
    ax.set_ylabel(r"ATMF skew $\partial \sigma / \partial \ln K$",
                   fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=1.0).set_zorder(10)
    plt.tight_layout()
    return _save(fig, "bergomi_stage2_fit_evolution.png")


# ══════════════════════════ Main ══════════════════════════
def main():
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Cache dir:  {CACHE_DIR.resolve()}")
    print()

    print("Heston:")
    h_df, h_final = _run_heston_de_with_trace()
    print(f"  trace rows: {len(h_df)}   final obj: {h_final['fun_polish']:.6e}")

    print("Bergomi Stage 1:")
    bg1_df, bg1_final = _run_bergomi_stage1_with_trace()
    print(f"  trace rows: {len(bg1_df)}   final obj: {bg1_final['fun_polish']:.6e}")

    print("Bergomi Stage 2:")
    bg2_df, bg2_final = _run_bergomi_stage2_with_trace(bg1_final)
    print(f"  trace rows: {len(bg2_df)}   final obj: {bg2_final['fun_polish']:.6e}")
    print()

    figures = [
        ("Heston loss curve",                 lambda: fig_heston_loss(h_df, h_final)),
        ("Heston param traces",               lambda: fig_heston_param_traces(h_df, h_final)),
        ("Heston objective slices",           lambda: fig_heston_objective_slices(h_final)),
        ("Bergomi Stage 1 loss curve",        lambda: fig_bergomi_stage1_loss(bg1_df, bg1_final)),
        ("Bergomi Stage 1 param traces",      lambda: fig_bergomi_stage1_param_traces(bg1_df, bg1_final)),
        ("Bergomi Stage 1 fit evolution",     lambda: fig_bergomi_stage1_fit_evolution(bg1_df, bg1_final)),
        ("Bergomi Stage 2 loss curve",        lambda: fig_bergomi_stage2_loss(bg2_df, bg2_final)),
        ("Bergomi Stage 2 param traces",      lambda: fig_bergomi_stage2_param_traces(bg2_df, bg2_final)),
        ("Bergomi Stage 2 fit evolution",     lambda: fig_bergomi_stage2_fit_evolution(bg2_df, bg2_final)),
    ]
    n_ok = 0; n_fail = 0
    for label, fn in figures:
        try:
            out = fn()
            print(f"  OK    {label:<38}  {out.name}"); n_ok += 1
        except Exception as exc:
            print(f"  FAIL  {label:<38}  {exc!r}"); n_fail += 1

    print(f"\n{n_ok} files written, {n_fail} failed.")
    print(f"All outputs under {OUT_DIR}/")


if __name__ == "__main__":
    main()

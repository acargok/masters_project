#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis §4.2 (Stochastic Volatility Calibration) figure pack. Stand-alone:
reads cached artefacts from lsv_heston/, lsv_bergomi/, iv_surface/,
dupire_vol/ and writes figures + two LaTeX tables to
iv_surface/results_4.2_plots/. Style inherited from results_4_1_figures.py.

Per-option repricing for the Heston/Bergomi pure-SV fits is not persisted
upstream, so the first run re-prices (Heston semi-analytic ~seconds, Bergomi
MC ~minutes) into cache/{heston,bergomi}_repricing_errors.csv; later runs read
the cache.
"""
import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

warnings.filterwarnings("ignore")

# Paths
HERE          = Path(__file__).resolve().parent          # iv_surface/
ROOT          = HERE.parent
IV_DIR        = HERE
DUPIRE_DIR    = ROOT / "dupire_vol"
LSV_DIR       = ROOT / "lsv_heston"
BERGOMI_DIR   = ROOT / "lsv_bergomi"
OUT_DIR       = HERE / "results_4.2_plots"
CACHE_DIR     = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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

# Shared style constants (matches results_4_1_figures.py)
_REPR_FIGSIZE   = (4.5, 4.5)
_REPR_SCATTER_S = 5
_CALL_COLOR     = "tab:blue"
_PUT_COLOR      = "tab:red"
_ERR_LABEL      = "Implied volatility error (bp)"
_FIT_FIGSIZE    = (6.5, 5.0)
_TERM_FIGSIZE   = (7.0, 4.5)   # shared by vs_vol_curve + fwd_var_curve
_TERM_LABEL_FS  = 12           # axis-label fontsize for the two term-structure figs
_TERM_TICK_FS   = 10           # tick-label fontsize for the two term-structure figs


def _square_axes(ax) -> None:
    ax.set_box_aspect(1)


def _save(fig, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


# Pipeline module loaders.
# lsv_heston/ and lsv_bergomi/ share file basenames, so load by path under
# unique module names to avoid sys.modules collisions.
def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_heston_calib = None
_bergomi_pure  = None


def heston_module():
    global _heston_calib
    if _heston_calib is None:
        _heston_calib = _import_by_path(
            "thesis_heston_calibration", LSV_DIR / "heston_calibration.py")
    return _heston_calib


def bergomi_pure_module():
    global _bergomi_pure
    if _bergomi_pure is None:
        _bergomi_pure = _import_by_path(
            "thesis_bergomi_pure_sv", BERGOMI_DIR / "bergomi_pure_sv_fit.py")
    return _bergomi_pure


# Bergomi parametric-form helpers.
# Reproduced here so the §4.2.2 Stage 1/2 figures avoid importing
# lsv_bergomi/bergomi_param_calibration.py (and its __main__ side effects).
def alpha_theta(theta: float, rho12: float) -> float:
    denom = np.sqrt((1.0 - theta) ** 2 + theta ** 2
                    + 2.0 * rho12 * theta * (1.0 - theta))
    return 1.0 / max(denom, 1e-12)


def A_i(kappa: float, T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    small = kappa * T < 1e-8
    out[small] = 1.0
    big = ~small
    out[big] = (1.0 - np.exp(-kappa * T[big])) / (kappa * T[big])
    return out


def vol_of_vol_model(T, nu, theta, kappa1, kappa2, rho12):
    a_th = alpha_theta(theta, rho12)
    a1   = A_i(kappa1, T)
    a2   = A_i(kappa2, T)
    var = ((1 - theta) ** 2 * a1 ** 2
           + theta ** 2 * a2 ** 2
           + 2.0 * rho12 * theta * (1 - theta) * a1 * a2)
    var = np.maximum(var, 0.0)
    return nu * a_th * np.sqrt(var)


def vol_of_vol_benchmark(T, sigma0, tau0, alpha):
    return sigma0 * (tau0 / np.asarray(T, dtype=float)) ** alpha


def skew_g(x):
    x = np.asarray(x, dtype=float)
    return x - (1.0 - np.exp(-x))


def skew_order1_model(T, nu, theta, kappa1, kappa2, rho12, rho1, rho2):
    a_th = alpha_theta(theta, rho12)
    k1T = np.maximum(kappa1 * T, 1e-12)
    k2T = np.maximum(kappa2 * T, 1e-12)
    term1 = (1 - theta) * rho1 * skew_g(k1T) / (k1T ** 2)
    term2 =       theta  * rho2 * skew_g(k2T) / (k2T ** 2)
    return nu * a_th * (term1 + term2)


def empirical_atmf_skew(iv_surface, log_m_grid, ttm_grid, T_query, delta=0.01):
    interp = interpolate.RegularGridInterpolator(
        (log_m_grid, ttm_grid), iv_surface,
        method="linear", bounds_error=False, fill_value=None,
    )
    k_pos = np.log(1.0 + delta)
    k_neg = np.log(1.0 - delta)
    T_q = np.asarray(T_query, dtype=float)
    pts_pos = np.column_stack([np.full_like(T_q, k_pos), T_q])
    pts_neg = np.column_stack([np.full_like(T_q, k_neg), T_q])
    return (interp(pts_pos) - interp(pts_neg)) / (2.0 * delta)


# Stage 1/2 grids — match bergomi_param_calibration.py defaults
_T_VOLOFVOL = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
_T_SKEW_MIN, _T_SKEW_MAX, _T_SKEW_N = 0.25, 2.0, 25


# Bounds for parameter tables — the admissible set the optimisers searched.
HESTON_BOUNDS = {
    "kappa": (0.1, 10.0),
    "theta": (0.005, 0.50),
    "xi":    (0.05, 2.0),
    "rho":   (-0.99, 0.10),
    "V0":    (0.005, 0.50),
}
BERGOMI_BOUNDS = {
    "nu":     (0.5, 2.0),
    "theta":  (0.1, 0.5),
    "kappa1": (3.0, 20.0),
    "kappa2": (0.05, 0.6),
    "rho12":  (-0.99, 0.8),
    "rho1":   (-0.99, 0.0),
    "rho2":   (-0.99, 0.0),   # implied through chi parametrisation
}


# Data loading
def load_data() -> dict:
    d = {}
    # Heston
    with open(LSV_DIR / "data" / "heston_params.json") as f:
        d["heston_params"] = json.load(f)

    # Bergomi
    with open(BERGOMI_DIR / "data" / "bergomi_params.json") as f:
        d["bergomi_params"] = json.load(f)
    with open(BERGOMI_DIR / "data" / "bergomi_fit.json") as f:
        d["bergomi_fit"] = json.load(f)
    with open(BERGOMI_DIR / "data" / "fwd_var_fit.json") as f:
        d["fwd_var_fit"] = json.load(f)

    d["vs_vol_curve"]   = np.load(BERGOMI_DIR / "arrays" / "vs_vol_curve.npy")
    d["vs_vol_fitted"]  = np.load(BERGOMI_DIR / "arrays" / "vs_vol_fitted.npy")
    d["fwd_var_curve"]  = np.load(BERGOMI_DIR / "arrays" / "fwd_var_curve.npy")

    # SSVI surface & grids
    d["iv_surface"] = np.load(IV_DIR / "arrays" / "iv_surface.npy")
    d["log_m_grid"] = np.load(IV_DIR / "arrays" / "log_m_grid.npy")
    d["ttm_grid"]   = np.load(IV_DIR / "arrays" / "ttm_grid.npy")

    # Market params (used by repricing helpers)
    with open(DUPIRE_DIR / "data" / "market_params.json") as f:
        d["market"] = json.load(f)
    return d


# Repricing CSV cache.
# Pipelines persist only summary JSONs; re-run their helpers and cache the
# per-option errors here.

_HESTON_CSV   = CACHE_DIR / "heston_repricing_errors.csv"
_BERGOMI_CSV  = CACHE_DIR / "bergomi_repricing_errors.csv"


def _generate_heston_repricing(d: dict) -> pd.DataFrame:
    print("  building Heston per-option repricing (semi-analytic pricer)…")
    h = heston_module()
    market_data = h.load_market_data()
    # Pipeline default: reprice every option surviving the filter chain.
    K, T, iv_ssvi, ssvi_prices, _vegas, opt_arr = h.prepare_calibration_data(
        market_data)

    p = d["heston_params"]
    S, r, q = market_data["S"], market_data["r"], market_data["q"]
    model_prices = h.heston_call_price_vectorised(
        S, K, T, r, q, p["kappa"], p["theta"], p["xi"], p["rho"], p["V0"],
        N_quad=h.N_QUAD_DIAG, upper_limit=h.UPPER_LIMIT_DIAG)
    put_mask = opt_arr == "put"
    if put_mask.any():
        model_prices = model_prices.copy()
        model_prices[put_mask] -= (
            S * np.exp(-q * T[put_mask]) - K[put_mask] * np.exp(-r * T[put_mask]))
    iv_model = np.array([
        h.bs_implied_vol(model_prices[i], S, K[i], T[i], r, q,
                          option_type=opt_arr[i])
        for i in range(len(K))
    ])
    fwd_log_m = np.log(K / (S * np.exp((r - q) * T)))
    df = pd.DataFrame({
        "strike": K, "ttm": T, "option_type": opt_arr,
        "iv_ssvi": iv_ssvi, "iv_model": iv_model,
        "ssvi_price": ssvi_prices, "model_price": model_prices,
        "fwd_log_m": fwd_log_m,
    })
    df["iv_error_bps"] = (df["iv_model"] - df["iv_ssvi"]) * 1e4
    df.to_csv(_HESTON_CSV, index=False)
    return df


def _generate_bergomi_repricing(d: dict) -> pd.DataFrame:
    print("  building Bergomi pure-SV per-option repricing (MC, this takes a couple of minutes)…")
    b = bergomi_pure_module()
    mkt = d["market"]
    S, r, q = mkt["S"], mkt["r"], mkt["q"]
    bergomi = d["bergomi_params"]

    fwd_var = d["fwd_var_curve"]
    ttm_grid = d["ttm_grid"]
    fwd_var_interp = interpolate.interp1d(
        ttm_grid, fwd_var, kind="linear",
        bounds_error=False, fill_value=(fwd_var[0], fwd_var[-1]))

    # Pipeline default: reprice every option surviving the filter chain.
    pool = b.select_option_pool(S, r, q, seed=b.SEED)
    K = pool["strike"].values.astype(np.float64)
    T = pool["ttm"].values.astype(np.float64)
    iv_ssvi = pool["iv"].values.astype(np.float64)
    opt_arr = pool["option_type"].values

    dt = 1.0 / b.STEPS_PER_YEAR
    maturities = sorted(set(np.round(T, 6)))
    snapshots, step_of = b.simulate_bergomi_no_leverage(
        S, r, q, bergomi, fwd_var_interp, ttm_grid,
        maturities, b.N_PATHS, dt, b.SEED)

    iv_model     = np.full(len(K), np.nan)
    model_prices = np.full(len(K), np.nan)
    ssvi_prices  = np.full(len(K), np.nan)
    for i, (Ki, Ti, opt) in enumerate(zip(K, T, opt_arr)):
        S_T = snapshots[step_of[float(round(Ti, 6))]]
        if opt == "call":
            payoff = np.maximum(S_T - Ki, 0.0)
            ssvi_prices[i] = b.bs_call_price(S, Ki, Ti, r, q, iv_ssvi[i])
        else:
            payoff = np.maximum(Ki - S_T, 0.0)
            ssvi_prices[i] = b.bs_put_price(S, Ki, Ti, r, q, iv_ssvi[i])
        price = np.exp(-r * Ti) * payoff.mean()
        model_prices[i] = price
        iv_model[i]     = b.bs_iv(price, S, Ki, Ti, r, q, opt)

    fwd_log_m = np.log(K / (S * np.exp((r - q) * T)))
    df = pd.DataFrame({
        "strike": K, "ttm": T, "option_type": opt_arr,
        "iv_ssvi": iv_ssvi, "iv_model": iv_model,
        "ssvi_price": ssvi_prices, "model_price": model_prices,
        "fwd_log_m": fwd_log_m,
    })
    df["iv_error_bps"] = (df["iv_model"] - df["iv_ssvi"]) * 1e4
    df.to_csv(_BERGOMI_CSV, index=False)
    return df


def get_heston_repricing(d: dict) -> pd.DataFrame:
    if _HESTON_CSV.exists():
        return pd.read_csv(_HESTON_CSV)
    return _generate_heston_repricing(d)


def get_bergomi_repricing(d: dict) -> pd.DataFrame:
    if _BERGOMI_CSV.exists():
        return pd.read_csv(_BERGOMI_CSV)
    return _generate_bergomi_repricing(d)


# §4.2.1 — Heston

def fig_heston_params_table(d: dict) -> Path:
    """LaTeX table: bounds + calibrated values for the five Heston parameters,
    Feller status, and IV/price fit diagnostics."""
    p = d["heston_params"]
    rows = [
        (r"$\kappa$",  HESTON_BOUNDS["kappa"], p["kappa"]),
        (r"$\theta$",  HESTON_BOUNDS["theta"], p["theta"]),
        (r"$\xi$",     HESTON_BOUNDS["xi"],    p["xi"]),
        (r"$\rho$",    HESTON_BOUNDS["rho"],   p["rho"]),
        (r"$V_0$",     HESTON_BOUNDS["V0"],    p["V0"]),
    ]
    lines = [
        "% Auto-generated by results_4_2_figures.py — do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Calibrated Heston parameters with their differential-evolution"
        r" search bounds, the Feller condition status $2\kappa\theta - \xi^2$, and the"
        r" IV/price fit diagnostics measured against the SSVI BS price.}",
        r"\label{tab:heston_params}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Parameter & Lower & Upper & Calibrated \\",
        r"\midrule",
    ]
    for name, (lo, hi), val in rows:
        lines.append(f"{name} & {lo:+.4f} & {hi:+.4f} & {val:+.6f} \\\\")
    lines.append(r"\midrule")
    lines.append(
        r"Feller $2\kappa\theta - \xi^2$ & & & "
        f"{p['feller_value']:+.4f} "
        f"({'satisfied' if p['feller_satisfied'] else 'violated'}) \\\\")
    lines.append(r"\midrule")
    lines.append(
        r"IV MAE  (bp)  & & & " + f"{p['iv_mae']*1e4:.1f} \\\\")
    lines.append(
        r"IV RMSE (bp)  & & & " + f"{p['iv_rmse']*1e4:.1f} \\\\")
    lines.append(
        r"IV ME   (bp)  & & & " + f"{p['iv_me']*1e4:+.1f} \\\\")
    lines.append(
        r"Price MAE vs SSVI (\%) & & & "
        f"{p['price_vs_ssvi_mae_pct']:.2f} \\\\")
    lines.append(
        r"$n_\mathrm{obs}$ & & & " + f"{p['n_options']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = OUT_DIR / "heston_params_table.tex"
    out.write_text("\n".join(lines))
    return out


def _fit_panel_setup(ax, valid, iv_ssvi, iv_model):
    hi = max(iv_ssvi[valid].max(), iv_model[valid].max()) * 100 * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1.0, label=r"$y = x$")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)


def fig_heston_fit_model_vs_ssvi(d: dict) -> Path:
    repr_df = get_heston_repricing(d)
    valid = repr_df["iv_model"].notna()
    iv_ssvi  = repr_df["iv_ssvi"].values
    iv_model = repr_df["iv_model"].values
    T_arr    = repr_df["ttm"].values

    fig, ax = plt.subplots(figsize=_FIT_FIGSIZE)
    sc = ax.scatter(iv_ssvi[valid] * 100, iv_model[valid] * 100,
                    c=T_arr[valid], cmap="viridis",
                    alpha=0.7, s=5, edgecolors="none")
    _fit_panel_setup(ax, valid.values, iv_ssvi, iv_model)
    ax.set_xlabel("SSVI implied volatility (%)")
    ax.set_ylabel("Heston implied volatility (%)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Time to maturity $T$ (years)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return _save(fig, "heston_fit_model_vs_ssvi.png")


def _fit_iv_error_hist(df: pd.DataFrame, fname: str) -> Path:
    err_bps = ((df["iv_model"] - df["iv_ssvi"]) * 1e4).dropna()
    me_bps  = float(err_bps.mean())
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    ax.hist(err_bps, bins=30, color="mediumpurple", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(me_bps, color="red", lw=1.1, ls=":",
               label=f"ME {me_bps:+.0f} bp")
    ax.set_xlabel(_ERR_LABEL)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def _fit_iv_error_vs_moneyness(df: pd.DataFrame, fname: str) -> Path:
    sub = df.dropna(subset=["iv_model"]).copy()
    sub["err_bps"] = (sub["iv_model"] - sub["iv_ssvi"]) * 1e4
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    for sub_set, color, marker, label in [
        (sub[sub["option_type"] == "call"], _CALL_COLOR, "o", "calls"),
        (sub[sub["option_type"] == "put"],  _PUT_COLOR,  "o", "puts"),
    ]:
        ax.scatter(sub_set["fwd_log_m"], sub_set["err_bps"],
                   s=_REPR_SCATTER_S, alpha=0.6,
                   color=color, marker=marker, label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Forward log-moneyness $k$")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def _fit_iv_error_vs_ttm(df: pd.DataFrame, fname: str) -> Path:
    sub = df.dropna(subset=["iv_model"]).copy()
    sub["err_bps"] = (sub["iv_model"] - sub["iv_ssvi"]) * 1e4
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    for sub_set, color, label in [
        (sub[sub["option_type"] == "call"], _CALL_COLOR, "calls"),
        (sub[sub["option_type"] == "put"],  _PUT_COLOR,  "puts"),
    ]:
        ax.scatter(sub_set["ttm"], sub_set["err_bps"],
                   s=_REPR_SCATTER_S, alpha=0.6,
                   color=color, marker="o", label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Time to maturity $T$ (years)")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def fig_heston_fit_iv_error_hist(d: dict) -> Path:
    return _fit_iv_error_hist(get_heston_repricing(d),
                              "heston_fit_iv_error_hist.png")


def fig_heston_fit_iv_error_vs_moneyness(d: dict) -> Path:
    return _fit_iv_error_vs_moneyness(get_heston_repricing(d),
                                       "heston_fit_iv_error_vs_moneyness.png")


def fig_heston_fit_iv_error_vs_ttm(d: dict) -> Path:
    return _fit_iv_error_vs_ttm(get_heston_repricing(d),
                                 "heston_fit_iv_error_vs_ttm.png")


# Checkpoint trio — square box, bp units, calls/puts circles (matches the
# SSVI/Dupire repricing trio convention).
def _trio_error_hist(df: pd.DataFrame, fname: str) -> Path:
    return _fit_iv_error_hist(df, fname)


def _trio_error_vs_price(df: pd.DataFrame, fname: str) -> Path:
    sub = df.dropna(subset=["iv_model"]).copy()
    sub["err_bps"] = (sub["iv_model"] - sub["iv_ssvi"]) * 1e4
    fig, ax = plt.subplots(figsize=_REPR_FIGSIZE)
    for sub_set, color, label in [
        (sub[sub["option_type"] == "call"], _CALL_COLOR, "calls"),
        (sub[sub["option_type"] == "put"],  _PUT_COLOR,  "puts"),
    ]:
        ax.scatter(sub_set["ssvi_price"], sub_set["err_bps"],
                   s=_REPR_SCATTER_S, alpha=0.6,
                   color=color, marker="o", label=label)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("SSVI option price (USD)")
    ax.set_ylabel(_ERR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _square_axes(ax)
    plt.tight_layout()
    return _save(fig, fname)


def _trio_error_vs_moneyness(df: pd.DataFrame, fname: str) -> Path:
    return _fit_iv_error_vs_moneyness(df, fname)


def fig_heston_repricing_error_hist(d: dict) -> Path:
    return _trio_error_hist(get_heston_repricing(d),
                             "heston_repricing_error_hist.png")


def fig_heston_repricing_error_vs_price(d: dict) -> Path:
    return _trio_error_vs_price(get_heston_repricing(d),
                                 "heston_repricing_error_vs_price.png")


def fig_heston_repricing_error_vs_moneyness(d: dict) -> Path:
    return _trio_error_vs_moneyness(get_heston_repricing(d),
                                     "heston_repricing_error_vs_moneyness.png")


# §4.2.2 — Bergomi

def fig_vs_vol_curve(d: dict) -> Path:
    """Variance-swap vol: SSVI-derived per-maturity points and the
    Nelson-Siegel-Svensson fit (Svensson 1994)."""
    ttm = d["ttm_grid"]
    vs_raw    = d["vs_vol_curve"]
    vs_fitted = d["vs_vol_fitted"]
    f = d["fwd_var_fit"]

    fig, ax = plt.subplots(figsize=_TERM_FIGSIZE)
    ax.plot(ttm, vs_raw * 100, "o", ms=4, alpha=0.75, color="black",
            label="SSVI-derived (Carr–Madan)")
    ax.plot(ttm, vs_fitted * 100, "-", lw=1.8, color="C0",
            label=(rf"NSS fit  $\beta_0$={f['nss_beta_0']:.3f}, "
                   rf"$\beta_1$={f['nss_beta_1']:+.3f}, "
                   rf"$\beta_2$={f['nss_beta_2']:+.3f}, "
                   rf"$\beta_3$={f['nss_beta_3']:+.3f}, "
                   rf"$\tau_1$={f['nss_tau_1']:.3f}, "
                   rf"$\tau_2$={f['nss_tau_2']:.3f}"))
    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=_TERM_LABEL_FS)
    ax.set_ylabel("Variance swap volatility (%)", fontsize=_TERM_LABEL_FS)
    ax.tick_params(labelsize=_TERM_TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "vs_vol_curve.png")


def fig_fwd_var_curve(d: dict) -> Path:
    """Initial forward variance curve  ξ⁰_T = d/dT (σ_VS²(T) · T)."""
    ttm = d["ttm_grid"]
    fwd_var = d["fwd_var_curve"]

    fig, ax = plt.subplots(figsize=_TERM_FIGSIZE)
    ax.plot(ttm, fwd_var, "-", lw=1.8, color="C1")
    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=_TERM_LABEL_FS)
    ax.set_ylabel(r"Forward variance $\xi^{T}_{0}$", fontsize=_TERM_LABEL_FS)
    ax.tick_params(labelsize=_TERM_TICK_FS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "fwd_var_curve.png")


def fig_bergomi_calib_volofvol(d: dict) -> Path:
    """Stage 1 — vol-of-vol fit vs the power-law benchmark in bergomi_params.json."""
    p   = d["bergomi_params"]
    bm  = p["benchmark"]
    T_grid = _T_VOLOFVOL
    target = vol_of_vol_benchmark(T_grid, bm["sigma0"], bm["tau0"], bm["alpha"])
    model  = vol_of_vol_model(T_grid, p["nu"], p["theta"],
                              p["kappa1"], p["kappa2"], p["rho12"])

    T_fine = np.linspace(T_grid.min(), T_grid.max(), 240)
    target_fine = vol_of_vol_benchmark(T_fine, bm["sigma0"], bm["tau0"], bm["alpha"])
    model_fine  = vol_of_vol_model(T_fine, p["nu"], p["theta"],
                                    p["kappa1"], p["kappa2"], p["rho12"])

    fig, ax = plt.subplots(figsize=_TERM_FIGSIZE)
    ax.plot(T_fine, target_fine * 100, "k-", lw=1.5, label="Target")
    ax.plot(T_fine, model_fine * 100, color="C0", lw=1.8,
            label="Bergomi two-factor model")
    ax.scatter(T_grid, target * 100, c="k",  s=22, zorder=3)
    ax.scatter(T_grid, model  * 100, c="C0", marker="x", s=28, zorder=3)
    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=_TERM_LABEL_FS)
    ax.set_ylabel(r"Vol-of-vol  $\nu^B(T)$  (%)", fontsize=_TERM_LABEL_FS)
    ax.tick_params(labelsize=_TERM_TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "bergomi_calib_volofvol.png")


def fig_bergomi_calib_skew(d: dict) -> Path:
    """Stage 2 — SPX ATMF skew (SSVI-derived) vs Bergomi–Guyon order-1 model."""
    p = d["bergomi_params"]
    iv_surface = d["iv_surface"]
    log_m = d["log_m_grid"]
    ttm   = d["ttm_grid"]

    T_skew = ttm[(ttm >= _T_SKEW_MIN) & (ttm <= _T_SKEW_MAX)]
    if len(T_skew) > _T_SKEW_N:
        T_skew = T_skew[np.linspace(0, len(T_skew) - 1, _T_SKEW_N, dtype=int)]
    target = empirical_atmf_skew(iv_surface, log_m, ttm, T_skew)
    model  = skew_order1_model(T_skew, p["nu"], p["theta"],
                                p["kappa1"], p["kappa2"], p["rho12"],
                                p["rho1"], p["rho2"])
    T_fine = np.linspace(T_skew.min(), T_skew.max(), 240)
    model_fine = skew_order1_model(T_fine, p["nu"], p["theta"],
                                    p["kappa1"], p["kappa2"], p["rho12"],
                                    p["rho1"], p["rho2"])

    fig, ax = plt.subplots(figsize=_TERM_FIGSIZE)
    ax.plot(T_fine, model_fine, color="C3", lw=1.8,
            label="Bergomi two-factor model")
    ax.scatter(T_skew, target, c="k", s=22, label="SSVI ATMF skew", zorder=3)
    ax.scatter(T_skew, model,  c="C3", marker="x", s=28, zorder=3)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(r"Time to maturity $T$ (years)", fontsize=_TERM_LABEL_FS)
    ax.set_ylabel(r"ATMF skew  $\partial \sigma / \partial \ln K$", fontsize=_TERM_LABEL_FS)
    ax.tick_params(labelsize=_TERM_TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, "bergomi_calib_skew.png")


def fig_bergomi_params_table(d: dict) -> Path:
    p = d["bergomi_params"]
    fit = d["bergomi_fit"]
    rows = [
        (r"$\nu$",       BERGOMI_BOUNDS["nu"],     p["nu"]),
        (r"$\theta$",    BERGOMI_BOUNDS["theta"],  p["theta"]),
        (r"$\kappa_1$",  BERGOMI_BOUNDS["kappa1"], p["kappa1"]),
        (r"$\kappa_2$",  BERGOMI_BOUNDS["kappa2"], p["kappa2"]),
        (r"$\rho_{12}$", BERGOMI_BOUNDS["rho12"],  p["rho12"]),
        (r"$\rho_1$",    BERGOMI_BOUNDS["rho1"],   p["rho1"]),
        (r"$\rho_2$",    BERGOMI_BOUNDS["rho2"],   p["rho2"]),
    ]
    lines = [
        "% Auto-generated by results_4_2_figures.py — do not edit by hand.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Calibrated two-factor Bergomi parameters with their"
        r" differential-evolution search bounds. Stage~1 fits"
        r" $(\nu, \theta, \kappa_1, \kappa_2, \rho_{12})$ to the power-law"
        r" vol-of-vol benchmark; Stage~2 fits $(\rho_1, \rho_2)$ (via the $\chi$"
        r" parametrisation) to the SSVI-derived ATMF skew. The pure-Bergomi"
        r" IV fit row reports MAE/RMSE against the SSVI surface for the same"
        r" option pool used by the Heston fit.}",
        r"\label{tab:bergomi_params}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Parameter & Lower & Upper & Calibrated \\",
        r"\midrule",
    ]
    for name, (lo, hi), val in rows:
        lines.append(f"{name} & {lo:+.4f} & {hi:+.4f} & {val:+.6f} \\\\")
    lines.append(r"\midrule")
    lines.append(
        r"Vol-of-vol fit RMSE & & & "
        f"{p['vol_of_vol_fit_rmse']*100:.2f}\\% \\\\")
    lines.append(
        r"ATMF skew fit RMSE  & & & "
        f"{p['atmf_skew_fit_rmse']:.4f} \\\\")
    lines.append(r"\midrule")
    lines.append(
        r"Pure-Bergomi IV MAE  (bp) & & & " + f"{fit['iv_mae']*1e4:.1f} \\\\")
    lines.append(
        r"Pure-Bergomi IV RMSE (bp) & & & " + f"{fit['iv_rmse']*1e4:.1f} \\\\")
    lines.append(
        r"Pure-Bergomi IV ME   (bp) & & & " + f"{fit['iv_me']*1e4:+.1f} \\\\")
    lines.append(
        r"Price MAE vs SSVI (\%) & & & "
        f"{fit['price_vs_ssvi_mae_pct']:.2f} \\\\")
    lines.append(
        r"$n_\mathrm{obs}$ & & & "
        f"{fit['n_valid']} / {fit['n_total']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = OUT_DIR / "bergomi_params_table.tex"
    out.write_text("\n".join(lines))
    return out


def fig_bergomi_fit_model_vs_ssvi(d: dict) -> Path:
    repr_df = get_bergomi_repricing(d)
    valid = repr_df["iv_model"].notna()
    iv_ssvi  = repr_df["iv_ssvi"].values
    iv_model = repr_df["iv_model"].values
    T_arr    = repr_df["ttm"].values

    fig, ax = plt.subplots(figsize=_FIT_FIGSIZE)
    sc = ax.scatter(iv_ssvi[valid] * 100, iv_model[valid] * 100,
                    c=T_arr[valid], cmap="viridis",
                    alpha=0.7, s=5, edgecolors="none")
    _fit_panel_setup(ax, valid.values, iv_ssvi, iv_model)
    ax.set_xlabel("SSVI implied volatility (%)")
    ax.set_ylabel("Bergomi implied volatility (%)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Time to maturity $T$ (years)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return _save(fig, "bergomi_fit_model_vs_ssvi.png")


def fig_bergomi_fit_iv_error_hist(d: dict) -> Path:
    return _fit_iv_error_hist(get_bergomi_repricing(d),
                              "bergomi_fit_iv_error_hist.png")


def fig_bergomi_fit_iv_error_vs_moneyness(d: dict) -> Path:
    return _fit_iv_error_vs_moneyness(get_bergomi_repricing(d),
                                       "bergomi_fit_iv_error_vs_moneyness.png")


def fig_bergomi_fit_iv_error_vs_ttm(d: dict) -> Path:
    return _fit_iv_error_vs_ttm(get_bergomi_repricing(d),
                                 "bergomi_fit_iv_error_vs_ttm.png")


def fig_bergomi_repricing_error_hist(d: dict) -> Path:
    return _trio_error_hist(get_bergomi_repricing(d),
                             "bergomi_repricing_error_hist.png")


def fig_bergomi_repricing_error_vs_price(d: dict) -> Path:
    return _trio_error_vs_price(get_bergomi_repricing(d),
                                 "bergomi_repricing_error_vs_price.png")


def fig_bergomi_repricing_error_vs_moneyness(d: dict) -> Path:
    return _trio_error_vs_moneyness(get_bergomi_repricing(d),
                                     "bergomi_repricing_error_vs_moneyness.png")


# Main
def main():
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Cache dir:  {CACHE_DIR.resolve()}")
    print("Loading pipeline artefacts ...")
    d = load_data()

    figures = [
        # §4.2.1 — Heston
        ("§4.2.1  Heston params table",              fig_heston_params_table),
        ("§4.2.1  Heston fit  —  model vs SSVI",     fig_heston_fit_model_vs_ssvi),
        ("§4.2.1  Heston fit  —  IV error hist",     fig_heston_fit_iv_error_hist),
        ("§4.2.1  Heston fit  —  IV err vs k",       fig_heston_fit_iv_error_vs_moneyness),
        ("§4.2.1  Heston fit  —  IV err vs T",       fig_heston_fit_iv_error_vs_ttm),
        ("§4.2.1  Heston repr.  —  hist",            fig_heston_repricing_error_hist),
        ("§4.2.1  Heston repr.  —  err vs price",    fig_heston_repricing_error_vs_price),
        ("§4.2.1  Heston repr.  —  err vs k",        fig_heston_repricing_error_vs_moneyness),
        # §4.2.2 — Bergomi
        ("§4.2.2  Variance swap vol curve",          fig_vs_vol_curve),
        ("§4.2.2  Forward variance curve",           fig_fwd_var_curve),
        ("§4.2.2  Stage 1 — vol-of-vol",             fig_bergomi_calib_volofvol),
        ("§4.2.2  Stage 2 — ATMF skew",              fig_bergomi_calib_skew),
        ("§4.2.2  Bergomi params table",             fig_bergomi_params_table),
        ("§4.2.2  Bergomi fit  —  model vs SSVI",    fig_bergomi_fit_model_vs_ssvi),
        ("§4.2.2  Bergomi fit  —  IV error hist",    fig_bergomi_fit_iv_error_hist),
        ("§4.2.2  Bergomi fit  —  IV err vs k",      fig_bergomi_fit_iv_error_vs_moneyness),
        ("§4.2.2  Bergomi fit  —  IV err vs T",      fig_bergomi_fit_iv_error_vs_ttm),
        ("§4.2.2  Bergomi repr.  —  hist",           fig_bergomi_repricing_error_hist),
        ("§4.2.2  Bergomi repr.  —  err vs price",   fig_bergomi_repricing_error_vs_price),
        ("§4.2.2  Bergomi repr.  —  err vs k",       fig_bergomi_repricing_error_vs_moneyness),
    ]

    print()
    n_ok = 0; n_fail = 0
    for label, fn in figures:
        try:
            out = fn(d)
            if out is None:
                print(f"  SKIP  {label}")
            else:
                print(f"  OK    {label:<46}  {out.name}"); n_ok += 1
        except Exception as exc:
            print(f"  FAIL  {label:<46}  {exc!r}"); n_fail += 1

    print(f"\n{n_ok} files written, {n_fail} failed.")
    print(f"All outputs under {OUT_DIR}/")


if __name__ == "__main__":
    main()

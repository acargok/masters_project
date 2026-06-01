import logging

import numpy as np
import pandas as pd
from scipy import optimize

from config import *

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 4 — SSVI SURFACE CONSTRUCTION
# =============================================================================

def ssvi_phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """
    Power-law curvature function (Gatheral & Jacquier 2014):

        φ(θ) = η / (θ^γ · (1 + θ)^{1−γ})

    Parameters
    ----------
    theta : ATM total variance  (> 0)
    eta   : level parameter  η ∈ (0, 2)
    gamma : power parameter  γ ∈ (0, 0.5)

    Behaviour: φ diverges as θ → 0 (short-maturity skew amplification),
    which allows the model to reproduce the steep SPX short-dated skew.
    """
    th = np.maximum(theta, 1e-10)
    return eta / (th ** gamma * (1.0 + th) ** (1.0 - gamma))


def ssvi_theta_t(t: np.ndarray, alpha0: float, alpha1: float, alpha2: float) -> np.ndarray:
    """
    Parametric ATM total variance term structure (Jacquier 2017).

        θ_t = α₀t + α₁(1 − e^{−α₂t})

    Satisfies θ(0) = 0 and is monotone increasing for α₀, α₁, α₂ > 0.
    The initial slope is α₀ + α₁α₂; the long-run slope is α₀.
    """
    return alpha0 * t + alpha1 * (1.0 - np.exp(-alpha2 * t))


def ssvi_rho_t(t: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    """
    Time-varying skew parameter (Jacquier 2017).

        ρ(t) = clip(arctan(p₀t + p₁) + p₂,  −0.999,  0.999)

    Allows the skew to vary across maturities while staying in (−1, 1).
    For SPX, ρ is typically most negative at short maturities and flattens
    at long maturities (p₀ > 0).
    """
    return np.clip(np.arctan(p0 * t + p1) + p2, -0.999, 0.999)


def ssvi_total_variance(k: np.ndarray, theta: float, phi: float,
                        rho: float) -> np.ndarray:
    """
    SSVI total implied variance (Gatheral & Jacquier 2014).

    FORMULA:
        w(k, θ) = (θ/2) · [1 + ρφk + √((φk + ρ)² + 1 − ρ²)]

    where:
        k   = ln(K/F)          forward log-moneyness
        θ   = ATM total var    w(0, θ) = θ  (exact, by construction)
        φ   = φ(θ)             power-law curvature, evaluated at this θ
        ρ ∈ (−1, 1)            skew  (equity: ρ < 0)

    Note: the at-the-money value is w(0) = θ regardless of φ and ρ, so θ
    is exactly the ATM total variance — not an approximation.
    """
    fk = phi * k
    return (theta / 2.0) * (1.0 + rho * fk + np.sqrt((fk + rho)**2 + 1.0 - rho**2))


def _ssvi_nb_penalty(theta_vals: np.ndarray, eta: float, gamma: float,
                     rho_vals: np.ndarray, weight: float) -> float:
    """
    Penalty for Gatheral & Jacquier (2014) Theorem 4.2 no-butterfly violations.

    Both conditions must hold at every (θ, ρ) pair in the term structure:
        C1(t) = θ(t) · φ(θ(t)) · (1 + |ρ(t)|) ≤ 4
        C2(t) = θ(t) · φ(θ(t))² · (1 + |ρ(t)|) ≤ 4

    rho_vals : np.ndarray, per-slice ρ(T) values.
    Returns the sum of squared exceedances multiplied by `weight`.
    """
    phi_vals = ssvi_phi(theta_vals, eta, gamma)
    abs_rho = np.abs(rho_vals)
    c1 = theta_vals * phi_vals * (1.0 + abs_rho)
    c2 = theta_vals * phi_vals ** 2 * (1.0 + abs_rho)
    sq_viol = np.maximum(0.0, c1 - 4.0) ** 2 + np.maximum(0.0, c2 - 4.0) ** 2
    return weight * float(sq_viol.sum())


def _ssvi_transform(x: np.ndarray, n_slices: int) -> tuple:
    """
    Transform the unconstrained optimisation vector to SSVI parameters.

    Layout: x = [log_η, logit_γ, p₀, p₁, p₂, log_θ₁, …, log_θₙ]

    Transforms (guarantee structural constraints automatically):
        η    = exp(log_η)               → η > 0
        γ    = 0.5·sigmoid(logit_γ)     → γ ∈ (0, 0.5)
        p₀, p₁, p₂  unconstrained      → ρ(t) = clip(arctan(p₀t+p₁)+p₂)
        θᵢ   = exp(log_θᵢ)             → θᵢ > 0  (per-slice ATM total var)
    """
    log_eta, logit_gamma = x[0], x[1]
    p0, p1, p2 = x[2], x[3], x[4]
    log_theta = x[5:5 + n_slices]
    eta   = np.exp(log_eta)
    gamma = 0.5 / (1.0 + np.exp(-logit_gamma))   # sigmoid → (0, 0.5)
    theta = np.exp(log_theta)
    return eta, gamma, p0, p1, p2, theta


def _ssvi_objective_fn(x: np.ndarray, slices: list, penalty_weight: float) -> float:
    """
    Joint SSVI objective: weighted MSE across all slices + no-butterfly penalty.

    slices : list of (ttm, k_data, w_data, weights) tuples.
    Uses per-slice θ and the Jacquier time-varying ρ(t) term structure.
    """
    n_slices = len(slices)
    eta, gamma, p0, p1, p2, theta_arr = _ssvi_transform(x, n_slices)

    ttm_arr = np.array([s[0] for s in slices])
    rho_arr = ssvi_rho_t(ttm_arr, p0, p1, p2)

    total_sse = 0.0
    total_wt  = 0.0
    for i, (_, k_data, w_data, wt) in enumerate(slices):
        phi_i  = float(ssvi_phi(np.array([theta_arr[i]]), eta, gamma)[0])
        w_model = ssvi_total_variance(k_data, float(theta_arr[i]), phi_i, float(rho_arr[i]))
        total_sse += float(np.sum(wt * (w_model - w_data) ** 2))
        total_wt  += float(wt.sum())

    mse     = total_sse / max(total_wt, 1e-10)
    penalty = _ssvi_nb_penalty(theta_arr, eta, gamma, rho_arr, penalty_weight)
    return mse + penalty


def fit_ssvi_surface(slice_data: list) -> dict:
    """
    Jointly fit SSVI w(k,θ) = (θ/2)[1 + ρ(t)φk + √((φk+ρ(t))²+1−ρ(t)²)] using:

        θ(T)   per-slice free parameters (one per expiry)  — observed ATMF total var
        φ(θ)   = η / (θ^γ · (1 + θ)^{1−γ})               power-law curvature (G&J 2014)
        ρ(t)   = clip(arctan(p₀t + p₁) + p₂)              time-varying skew (Jacquier 2017)

    Shared parameters: η, γ, p₀, p₁, p₂  (5 scalars).
    Per-slice parameters: θ₁, …, θₙ  (n scalars).
    Total: 5 + n parameters.  No-butterfly conditions (G&J 2014 Thm 4.2) are
    enforced as a penalty during optimisation.

    Parameters
    ----------
    slice_data : list of dict
        Each element has keys 'ttm', 'expiry', 'k_data', 'w_data'.
        Must be sorted by TTM ascending.

    Returns
    -------
    dict
        Keys: 'eta', 'gamma', 'p0', 'p1', 'p2',
              'theta' (per-slice, shape n), 'rho' (per-slice, shape n),
              'ttm' (shape n), 'rmse_per_slice' (shape n),
              'overall_rmse', 'nb_conditions_ok', 'c1_max', 'c2_max', 'success'.
    """
    n_slices = len(slice_data)
    if n_slices < 3:
        raise ValueError(f"Need ≥ 3 slices for SSVI joint fit; got {n_slices}")

    ttms = np.array([s["ttm"] for s in slice_data])

    # Build (ttm, k, w, weights) tuples; weight = 1/w (inverse total variance)
    slices = []
    for s in slice_data:
        k_data = s["k_data"]
        w_data = s["w_data"]
        wt = 1.0 / np.maximum(w_data, 1e-6)
        wt /= wt.sum()
        slices.append((s["ttm"], k_data, w_data, wt))

    # ── Initial per-slice θ: ATMF total variance for each slice ──
    theta_init = np.zeros(n_slices)
    for i, s in enumerate(slice_data):
        idx_atm = int(np.argmin(np.abs(s["k_data"])))
        theta_init[i] = max(float(s["w_data"][idx_atm]), 1e-4)

    # ── Adaptive penalty weight ──
    _eta0, _gamma0, _rho0 = 1.0, 0.3, -0.7
    _init_mse = 0.0
    for i, s in enumerate(slice_data):
        _phi0 = float(ssvi_phi(np.array([theta_init[i]]), _eta0, _gamma0)[0])
        _w0   = ssvi_total_variance(s["k_data"], theta_init[i], _phi0, _rho0)
        _init_mse += float(np.mean((_w0 - s["w_data"]) ** 2))
    _init_mse = max(_init_mse / n_slices, 1e-10)
    adaptive_penalty = SSVI_NB_PENALTY * _init_mse
    logger.debug(f"  SSVI penalty: init_mse={_init_mse:.2e}  weight={adaptive_penalty:.2e}")

    # logit(γ/0.5) for γ=0.3 → ln(0.6/0.4) ≈ 0.405
    _logit_gamma0 = np.log(0.3 / (0.5 - 0.3))

    # Initial parameter vector: [log_η, logit_γ, p₀, p₁, p₂, log_θ₁, …, log_θₙ]
    x0 = np.concatenate([
        [np.log(1.0),      # log_η   → η = 1.0
         _logit_gamma0,    # logit_γ → γ = 0.3
         0.0,              # p₀ = 0  (flat ρ to start)
         0.0,              # p₁ = 0
         -0.7],            # p₂ → ρ ≈ arctan(0)+(-0.7) = -0.7
        np.log(theta_init),           # log_θᵢ  (per-slice ATM total var)
    ])

    # Bounds on the transformed variables
    bounds = (
        [(-3.0,  1.0)]        # log_η   : η ∈ (0.05, 2.72)
        + [(-10., 10.)]       # logit_γ : γ ∈ (0, 0.5) via sigmoid
        + [(-3.0, 3.0)]       # p₀
        + [(-5.0, 5.0)]       # p₁
        + [(-2.0, 0.5)]       # p₂  (equity: ρ < 0)
        + [(-8.0, 0.0)] * n_slices   # log_θᵢ : θ ∈ (3e-4, 1.0)
    )

    # ── Stage 1: Differential evolution (global search) ──
    best_result = None
    try:
        res_de = optimize.differential_evolution(
            _ssvi_objective_fn,
            bounds=bounds,
            args=(slices, adaptive_penalty),
            seed=42,
            maxiter=600,
            tol=1e-10,
            polish=False,
            popsize=15,
        )
        best_result = res_de
        logger.debug(f"  SSVI DE: fun={res_de.fun:.6e}  nit={res_de.nit}")
    except Exception as exc:
        logger.warning(f"  SSVI DE failed: {exc}")

    # ── Stage 2: L-BFGS-B local refinement ──
    x_start = best_result.x if best_result is not None else x0
    try:
        res_lb = optimize.minimize(
            _ssvi_objective_fn,
            x0=x_start,
            args=(slices, adaptive_penalty),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
        )
        if best_result is None or res_lb.fun < best_result.fun:
            best_result = res_lb
        logger.debug(f"  SSVI L-BFGS-B: fun={res_lb.fun:.6e}  success={res_lb.success}")
    except Exception as exc:
        logger.warning(f"  SSVI L-BFGS-B failed: {exc}")

    if best_result is None:
        return {"success": False}

    eta, gamma, p0, p1, p2, theta_vals = _ssvi_transform(best_result.x, n_slices)

    # Per-slice ρ from parametric form
    rho_vals = ssvi_rho_t(ttms, p0, p1, p2)

    # ── Per-slice RMSE ──
    rmse_per_slice = np.zeros(n_slices)
    for i, s in enumerate(slice_data):
        phi_i  = float(ssvi_phi(np.array([theta_vals[i]]), eta, gamma)[0])
        w_fit  = ssvi_total_variance(s["k_data"], float(theta_vals[i]), phi_i, float(rho_vals[i]))
        rmse_per_slice[i] = float(np.sqrt(np.mean((w_fit - s["w_data"]) ** 2)))
    overall_rmse = float(np.sqrt(np.mean(rmse_per_slice ** 2)))

    # ── Verify no-butterfly conditions ──
    phi_vec = ssvi_phi(theta_vals, eta, gamma)
    abs_rho_vec = np.abs(rho_vals)
    c1_max = float(np.max(theta_vals * phi_vec * (1.0 + abs_rho_vec)))
    c2_max = float(np.max(theta_vals * phi_vec ** 2 * (1.0 + abs_rho_vec)))
    nb_ok = (c1_max <= 4.0 + 1e-6) and (c2_max <= 4.0 + 1e-6)

    if nb_ok:
        logger.info(f"  No-butterfly (G&J): C1_max={c1_max:.4f}, C2_max={c2_max:.4f}  ✓")
    else:
        logger.warning(
            f"  No-butterfly (G&J): C1_max={c1_max:.4f}, C2_max={c2_max:.4f}  "
            f"— residual violation (increase SSVI_NB_PENALTY)"
        )

    logger.info(
        f"  SSVI params: η={eta:.4f}  γ={gamma:.4f}  "
        f"p₀={p0:.3f} p₁={p1:.3f} p₂={p2:.3f}  "
        f"overall_RMSE={overall_rmse:.6f}  n_slices={n_slices}"
    )
    logger.info(
        f"  ρ range: [{rho_vals.min():.4f}, {rho_vals.max():.4f}]  "
        f"θ range: [{theta_vals.min():.4f}, {theta_vals.max():.4f}]"
    )

    return {
        "eta":   float(eta),
        "gamma": float(gamma),
        "p0": float(p0),
        "p1": float(p1),
        "p2": float(p2),
        "theta": theta_vals,
        "rho":   rho_vals,
        "ttm":   ttms,
        "rmse_per_slice": rmse_per_slice,
        "overall_rmse":   overall_rmse,
        "nb_conditions_ok": nb_ok,
        "c1_max": c1_max,
        "c2_max": c2_max,
        "success": True,
    }


def enforce_calendar_arbitrage(total_var_surface: np.ndarray) -> np.ndarray:
    """
    Enforce no-calendar-spread arbitrage: w(k, T₁) ≤ w(k, T₂) for T₁ < T₂.

    METHODOLOGY: Sweep from the earliest to the latest maturity.  At each
    TTM index j, ensure w[:, j] ≥ w[:, j-1] pointwise.  Where violated,
    set w[:, j] = w[:, j-1] (flat-forward total variance).

    This is a simple projection that preserves the per-slice smile shape
    while guaranteeing monotonicity.
    """
    w = total_var_surface.copy()
    n_violations = 0
    for j in range(1, w.shape[1]):
        violation = w[:, j] < w[:, j - 1]
        if violation.any():
            n_violations += violation.sum()
            w[:, j] = np.maximum(w[:, j], w[:, j - 1])
    if n_violations > 0:
        logger.info(f"  Calendar-arb fix: {n_violations} grid points adjusted")
    return w


def build_iv_surface(df: pd.DataFrame, fwd_df: pd.DataFrame,
                     ttm_grid_size: int = TTM_GRID_SIZE,
                     moneyness_grid_size: int = MONEYNESS_GRID_SIZE) -> tuple:
    """
    Build the IV surface via a joint SSVI fit.

    Steps:
      1. For each expiry slice with ≥ MIN_OPTIONS_PER_SLICE options, collect
         (k, w) pairs in forward log-moneyness × total variance space.
      2. Jointly fit SSVI  w(k,θ) = (θ/2)[1 + ρφk + √((φk+ρ)²+1−ρ²)]
         with power-law φ(θ) = η/(θ^γ·(1+θ)^(1−γ)).  Shared parameters
         η, γ, p₀, p₁, p₂; per-slice θ(T); ρ(T) = clip(arctan(p₀T+p₁)+p₂).
         No-butterfly conditions (G&J 2014 Thm 4.2) are enforced as a penalty
         during fitting.
      3. Evaluate SSVI on the (k, T) grid (analytically smooth, no smoothing
         step).
      4. Enforce calendar-spread arbitrage as a safety net for grid rounding.
      5. Convert to IV: σ(k, T) = √(w(k, T) / T).

    Parameters
    ----------
    df : pd.DataFrame
        Options with columns: strike, expiry, ttm, moneyness, iv.
    fwd_df : pd.DataFrame
        Per-expiry forwards from compute_implied_forwards().

    Returns
    -------
    ttm_grid : np.ndarray, shape (ttm_grid_size,)
    log_m_grid : np.ndarray, shape (moneyness_grid_size,)
        Forward log-moneyness k = ln(K/F).
    iv_surface : np.ndarray, shape (moneyness_grid_size, ttm_grid_size)
    total_var_surface : np.ndarray, shape (moneyness_grid_size, ttm_grid_size)
    ssvi_params_df : pd.DataFrame
        Per-slice θ(T) and ρ(T) plus the shared scalars η, γ, p₀, p₁, p₂.
    """
    fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))

    df = df.copy()
    df["forward"] = df["expiry"].map(fwd_map)
    df["fwd_log_m"] = np.log(df["strike"] / df["forward"])
    df["total_var"] = df["iv"]**2 * df["ttm"]

    # ── Step 1: collect per-slice data ──
    sorted_expiries = sorted(df["expiry"].unique(),
                             key=lambda e: df[df["expiry"] == e]["ttm"].iloc[0])

    slice_data = []
    skipped_thin = []
    for expiry in sorted_expiries:
        slice_df = df[df["expiry"] == expiry]
        ttm = float(slice_df["ttm"].iloc[0])
        k_data = slice_df["fwd_log_m"].values
        w_data = slice_df["total_var"].values
        if len(k_data) < MIN_OPTIONS_PER_SLICE:
            skipped_thin.append((expiry, ttm, len(k_data)))
        else:
            slice_data.append({"expiry": expiry, "ttm": ttm,
                                "k_data": k_data, "w_data": w_data})

    n_total = len(sorted_expiries)
    n_skipped = len(skipped_thin)
    n_fit = len(slice_data)
    logger.info(
        f"SSVI surface: {n_total} expiries total | "
        f"{n_skipped} thin (< {MIN_OPTIONS_PER_SLICE} options) | "
        f"{n_fit} for joint fit"
    )
    if skipped_thin:
        logger.info(f"  Thin slices skipped (expiry, TTM, n_options):")
        for exp, t, n in skipped_thin:
            logger.info(f"    {exp}  T={t:.3f}y  n={n}")

    if n_fit < 3:
        raise RuntimeError(
            f"Only {n_fit} slices with ≥ {MIN_OPTIONS_PER_SLICE} options. "
            f"Cannot fit SSVI (need ≥ 3)."
        )

    # ── Step 2: joint SSVI fit ──
    ssvi_result = fit_ssvi_surface(slice_data)
    if not ssvi_result["success"]:
        raise RuntimeError("SSVI joint fit failed — check data quality.")

    eta   = ssvi_result["eta"]
    gamma = ssvi_result["gamma"]
    p0, p1, p2 = ssvi_result["p0"], ssvi_result["p1"], ssvi_result["p2"]
    theta_vals = ssvi_result["theta"]
    rho_vals   = ssvi_result["rho"]
    ttm_vals   = ssvi_result["ttm"]

    # ── Step 3: grid construction ──
    # k-grid: use explicit bounds if set, else fall back to data percentiles.
    fwd_lm = df["fwd_log_m"].values
    lm_min = GRID_K_MIN if GRID_K_MIN is not None else float(np.percentile(fwd_lm, 2))
    lm_max = GRID_K_MAX if GRID_K_MAX is not None else float(np.percentile(fwd_lm, 98))
    log_m_grid = np.linspace(lm_min, lm_max, moneyness_grid_size)

    # TTM grid: use explicit bounds if set, else use observed option TTM range.
    ttm_min = GRID_T_MIN if GRID_T_MIN is not None else float(ttm_vals.min())
    ttm_max = GRID_T_MAX if GRID_T_MAX is not None else float(ttm_vals.max())
    ttm_grid = np.linspace(ttm_min, ttm_max, ttm_grid_size)

    logger.info(f"Grid k-range: [{lm_min:.4f}, {lm_max:.4f}]  "
                f"(data 2–98pct: [{np.percentile(fwd_lm,2):.4f}, {np.percentile(fwd_lm,98):.4f}])")
    logger.info(f"Grid T-range: [{ttm_min:.4f}, {ttm_max:.4f}]  "
                f"(observed TTMs: [{float(ttm_vals.min()):.4f}, {float(ttm_vals.max()):.4f}])")

    # ── Step 4: evaluate SSVI on the grid ──
    # θ(T): PCHIP interpolation of the per-slice values onto the fine TTM grid.
    # PCHIP preserves local monotonicity, avoiding spurious oscillations.
    # ρ(T): evaluated analytically from the parametric form.
    from scipy.interpolate import PchipInterpolator
    _theta_interp = PchipInterpolator(ttm_vals, theta_vals, extrapolate=True)
    theta_on_grid = np.maximum(_theta_interp(ttm_grid), 1e-8)
    phi_on_grid   = ssvi_phi(theta_on_grid, eta, gamma)
    rho_on_grid   = ssvi_rho_t(ttm_grid, p0, p1, p2)

    total_var_surface = np.zeros((moneyness_grid_size, ttm_grid_size))
    for j in range(ttm_grid_size):
        total_var_surface[:, j] = ssvi_total_variance(
            log_m_grid, theta_on_grid[j], phi_on_grid[j], float(rho_on_grid[j])
        )
    total_var_surface = np.maximum(total_var_surface, 1e-8)

    # ── Step 5: calendar-spread arbitrage (safety net) ──
    total_var_surface = enforce_calendar_arbitrage(total_var_surface)

    # ── Step 6: convert to IV ──
    TTM_broadcast = ttm_grid[np.newaxis, :]
    iv_surface = np.sqrt(total_var_surface / TTM_broadcast)
    iv_surface = np.clip(iv_surface, 0.01, 3.0)

    logger.info(
        f"Surface: {moneyness_grid_size}×{ttm_grid_size}  |  "
        f"k [{lm_min:.3f}, {lm_max:.3f}]  |  "
        f"TTM [{ttm_min:.3f}, {ttm_max:.3f}]  |  "
        f"IV [{iv_surface.min():.3f}, {iv_surface.max():.3f}]"
    )

    # ── Build ssvi_params_df ──
    ssvi_records = []
    for i, s in enumerate(slice_data):
        phi_i = float(ssvi_phi(np.array([theta_vals[i]]), eta, gamma)[0])
        ssvi_records.append({
            "expiry":    s["expiry"],
            "ttm":       s["ttm"],
            "n_options": len(s["k_data"]),
            "theta":     float(theta_vals[i]),    # θ(T) — per-slice free parameter
            "phi":       phi_i,                   # φ(θ(T)) — per-slice
            "rho":       float(rho_vals[i]),      # ρ(T) — from parametric ρ(t) form
            "eta":       eta,                     # shared scalar
            "gamma":     gamma,                   # shared scalar
            "p0":        p0,                      # shared ρ term structure params
            "p1":        p1,
            "p2":        p2,
            "rmse":      float(ssvi_result["rmse_per_slice"][i]),
            "success":   True,
        })
    ssvi_params_df = pd.DataFrame(ssvi_records)

    return ttm_grid, log_m_grid, iv_surface, total_var_surface, ssvi_params_df

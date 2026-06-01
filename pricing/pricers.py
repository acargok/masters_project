import logging

import numpy as np

from io_loaders import build_leverage_interpolator
from variance_processes import HestonVariance, BergomiVariance
from simulation import simulate_lsv, compute_returns
from payoffs import payoff_accumulator, payoff_reverse_cliquet, payoff_napoleon

logger = logging.getLogger(__name__)


def bs_cliquet_price(S0, r, q, sigma, reset_dates, payoff_fn, payoff_kwargs,
                     n_paths=500_000, seed=99):
    """Flat Black-Scholes (constant vol, no stoch variance) cliquet price as a
    sanity check. Returns dict: price, se, payoffs."""
    rng = np.random.default_rng(seed)

    # Time grid hitting each reset exactly (weekly sub-stepping)
    all_times = [0.0]
    for T in reset_dates:
        t_prev = all_times[-1]
        gap = T - t_prev
        if gap <= 0:
            continue
        n_sub = max(1, int(np.ceil(gap / (1.0 / 52.0))))
        sub_times = np.linspace(t_prev, T, n_sub + 1)[1:]
        all_times.extend(sub_times.tolist())
    time_grid = np.array(all_times)
    reset_indices = np.searchsorted(time_grid, reset_dates)

    S = np.full(n_paths, S0)
    S_all = np.empty((n_paths, len(time_grid)), dtype=np.float64)
    S_all[:, 0] = S

    for i in range(len(time_grid) - 1):
        dt = time_grid[i + 1] - time_grid[i]
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S_all[:, i + 1] = S

    S_resets = S_all[:, reset_indices]
    returns = compute_returns(S_resets, S0)
    payoffs = payoff_fn(returns, **payoff_kwargs)

    T_final = reset_dates[-1]
    df = np.exp(-r * T_final)
    price = df * payoffs.mean()
    se = df * payoffs.std() / np.sqrt(n_paths)

    return {"price": price, "se": se, "payoffs": payoffs}


def heston_cliquet_price(S0, r, q, heston, reset_dates, payoff_fn,
                         payoff_kwargs, n_paths=500_000, dt_max=1.0/52.0,
                         seed=99, variance_scheme="qe"):
    """Price a cliquet under pure Heston (L=1) for diagnostic comparison."""
    def leverage_one(S_arr, t):
        return np.ones(len(S_arr))

    var_proc = HestonVariance(
        kappa=heston["kappa"], theta=heston["theta"],
        xi=heston["xi"], V0=heston["V0"],
        scheme=variance_scheme, rho=heston["rho"],
    )
    sim = simulate_lsv(S0, r, q, heston["rho"], leverage_one, var_proc,
                       reset_dates, n_paths, dt_max=dt_max, seed=seed)
    returns = compute_returns(sim["S_resets"], S0)
    payoffs = payoff_fn(returns, **payoff_kwargs)
    T_final = reset_dates[-1]
    df = np.exp(-r * T_final)
    return {"price": df * payoffs.mean(),
            "se": df * payoffs.std() / np.sqrt(n_paths),
            "payoffs": payoffs}


def bergomi_cliquet_price(S0, r, q, bergomi, fwd_var, ttm_grid, reset_dates,
                          payoff_fn, payoff_kwargs, n_paths=500_000,
                          dt_max=1.0/52.0, seed=99):
    """Pure Bergomi (L=1) cliquet price, mirroring heston_cliquet_price. With
    L identically 1, the gap vs the calibrated LSV run isolates the leverage
    surface's contribution."""
    def leverage_one(S_arr, t):
        return np.ones(len(S_arr))

    var_proc = BergomiVariance(
        nu=bergomi["nu"], theta=bergomi["theta"],
        kappa1=bergomi["kappa1"], kappa2=bergomi["kappa2"],
        rho1=bergomi["rho1"], rho2=bergomi["rho2"],
        rho12=bergomi.get("rho12", 0.0),
        fwd_var_curve=fwd_var,
        ttm_grid=ttm_grid,
        seed=seed + 5000,
    )
    # Spot-vol correlation handled inside BergomiVariance (3x3 Cholesky); the
    # rho here is unused on the Bergomi branch (uses_spot_noise=True), passed
    # as rho1 for parity with the LSV path.
    rho_eff = bergomi["rho1"]
    sim = simulate_lsv(S0, r, q, rho_eff, leverage_one, var_proc,
                       reset_dates, n_paths, dt_max=dt_max, seed=seed)
    returns = compute_returns(sim["S_resets"], S0)
    payoffs = payoff_fn(returns, **payoff_kwargs)
    T_final = reset_dates[-1]
    df = np.exp(-r * T_final)
    return {"price": df * payoffs.mean(),
            "se": df * payoffs.std() / np.sqrt(n_paths),
            "payoffs": payoffs}


def price_cliquet(inputs, payoff_fn, payoff_kwargs, payoff_name,
                  n_paths=500_000, dt_max=1.0 / 52.0, seed=42,
                  variance_scheme="qe"):
    """Price a cliquet under the calibrated Heston LSV model.

    inputs: load_pricing_inputs() output. payoff_fn(returns, **kwargs) ->
    per-path payoffs; payoff_name labels logging. Returns dict with price, se,
    ci_half, baselines, payoffs/returns and sample paths."""
    S0 = inputs["S"]
    r = inputs["r"]
    q = inputs["q"]
    heston = inputs["heston"]

    leverage_fn = build_leverage_interpolator(
        inputs["leverage"], inputs["spot_grid"], inputs["time_grid"]
    )

    var_process = HestonVariance(
        kappa=heston["kappa"],
        theta=heston["theta"],
        xi=heston["xi"],
        V0=heston["V0"],
        scheme=variance_scheme,
        rho=heston["rho"],
    )

    # Monthly resets over 1 year
    n_resets = 12
    reset_dates = np.array([(i + 1) / 12.0 for i in range(n_resets)])

    logger.info(f"Pricing {payoff_name}: {n_paths:,} paths, "
                f"{n_resets} monthly resets, dt_max={dt_max:.4f}")

    sim = simulate_lsv(
        S0, r, q, heston["rho"], leverage_fn, var_process,
        reset_dates, n_paths, dt_max=dt_max, seed=seed,
    )

    returns = compute_returns(sim["S_resets"], S0)
    payoffs = payoff_fn(returns, **payoff_kwargs)

    T_final = reset_dates[-1]
    df = np.exp(-r * T_final)
    discounted = df * payoffs

    price = discounted.mean()
    se = discounted.std() / np.sqrt(n_paths)
    ci_half = 1.96 * se

    logger.info(f"  {payoff_name}: price = {price:.6f}, SE = {se:.6f}, "
                f"95% CI = [{price - ci_half:.6f}, {price + ci_half:.6f}]")

    # BS baseline (flat ATM vol ~ sqrt(V0))
    atm_vol = np.sqrt(heston["V0"])
    bs_result = bs_cliquet_price(
        S0, r, q, atm_vol, reset_dates, payoff_fn, payoff_kwargs,
        n_paths=n_paths, seed=seed + 1000,
    )
    logger.info(f"  BS baseline ({payoff_name}, vol={atm_vol:.4f}): "
                f"price = {bs_result['price']:.6f}")

    # Pure Heston baseline (L=1)
    heston_result = heston_cliquet_price(
        S0, r, q, heston, reset_dates, payoff_fn, payoff_kwargs,
        n_paths=n_paths, dt_max=dt_max, seed=seed + 2000,
        variance_scheme=variance_scheme,
    )
    logger.info(f"  Heston baseline ({payoff_name}): "
                f"price = {heston_result['price']:.6f}")

    # Pure Bergomi baseline (L=1) — only if Bergomi inputs present.
    bergomi_params = inputs.get("bergomi")
    fwd_var = inputs.get("fwd_var")
    if bergomi_params is not None and fwd_var is not None:
        bergomi_result = bergomi_cliquet_price(
            S0, r, q, bergomi_params, fwd_var, inputs["ttm_grid"],
            reset_dates, payoff_fn, payoff_kwargs,
            n_paths=n_paths, dt_max=dt_max, seed=seed + 3000,
        )
        logger.info(f"  Bergomi baseline ({payoff_name}): "
                    f"price = {bergomi_result['price']:.6f}")
    else:
        bergomi_result = {"price": float("nan"), "se": float("nan")}

    return {
        "price": float(price),
        "se": float(se),
        "ci_half": float(ci_half),
        "bs_price": float(bs_result["price"]),
        "bs_se": float(bs_result["se"]),
        "heston_price": float(heston_result["price"]),
        "heston_se": float(heston_result["se"]),
        "bergomi_price": float(bergomi_result["price"]),
        "bergomi_se": float(bergomi_result["se"]),
        "atm_vol": float(atm_vol),
        "payoffs": payoffs,
        "returns": returns,
        "S_resets": sim["S_resets"],
        "S_all": sim["S_all"],
        "V_all": sim["V_all"],
        "time_grid": sim["time_grid"],
        "reset_indices": sim["reset_indices"],
        "reset_dates": reset_dates,
        "n_paths": n_paths,
        "payoff_name": payoff_name,
        "payoff_kwargs": payoff_kwargs,
        "discount_factor": float(df),
    }


def price_cliquet_bergomi(inputs, payoff_fn, payoff_kwargs, payoff_name,
                          n_paths=500_000, dt_max=1.0 / 52.0, seed=42):
    """Price a cliquet under the calibrated Bergomi LSV model. Same interface
    as price_cliquet() but uses BergomiVariance and a weighted spot-vol rho."""
    S0 = inputs["S"]
    r = inputs["r"]
    q = inputs["q"]
    bergomi = inputs["bergomi"]
    heston = inputs["heston"]

    leverage_fn = build_leverage_interpolator(
        inputs["leverage"], inputs["spot_grid"], inputs["time_grid"]
    )

    var_process = BergomiVariance(
        nu=bergomi["nu"], theta=bergomi["theta"],
        kappa1=bergomi["kappa1"], kappa2=bergomi["kappa2"],
        rho1=bergomi["rho1"], rho2=bergomi["rho2"],
        rho12=bergomi.get("rho12", 0.0),
        fwd_var_curve=inputs["fwd_var"],
        ttm_grid=inputs["ttm_grid"],
        seed=seed + 5000,
    )

    # Effective spot-vol rho = rho1 (dominant factor); full 3D correlation
    # handled inside BergomiVariance.step().
    rho_eff = bergomi["rho1"]

    n_resets = 12
    reset_dates = np.array([(i + 1) / 12.0 for i in range(n_resets)])

    logger.info(f"Pricing {payoff_name} (Bergomi): {n_paths:,} paths, "
                f"{n_resets} monthly resets")

    sim = simulate_lsv(
        S0, r, q, rho_eff, leverage_fn, var_process,
        reset_dates, n_paths, dt_max=dt_max, seed=seed,
    )

    returns = compute_returns(sim["S_resets"], S0)
    payoffs = payoff_fn(returns, **payoff_kwargs)

    T_final = reset_dates[-1]
    df = np.exp(-r * T_final)
    discounted = df * payoffs
    price = discounted.mean()
    se = discounted.std() / np.sqrt(n_paths)
    ci_half = 1.96 * se

    logger.info(f"  {payoff_name} (Bergomi): price = {price:.6f}, SE = {se:.6f}")

    # BS baseline (ATM vol from fwd variance at T=0.5)
    atm_vol = np.sqrt(max(float(var_process.fwd_var_interp(0.5)), 1e-6))
    bs_result = bs_cliquet_price(
        S0, r, q, atm_vol, reset_dates, payoff_fn, payoff_kwargs,
        n_paths=n_paths, seed=seed + 1000,
    )

    # Heston baseline
    heston_result = heston_cliquet_price(
        S0, r, q, heston, reset_dates, payoff_fn, payoff_kwargs,
        n_paths=n_paths, dt_max=dt_max, seed=seed + 2000,
    )
    logger.info(f"  Heston baseline ({payoff_name}): "
                f"price = {heston_result['price']:.6f}")

    # Pure Bergomi baseline (L=1), same Bergomi inputs as the LSV run.
    bergomi_result = bergomi_cliquet_price(
        S0, r, q, bergomi, inputs["fwd_var"], inputs["ttm_grid"],
        reset_dates, payoff_fn, payoff_kwargs,
        n_paths=n_paths, dt_max=dt_max, seed=seed + 3000,
    )
    logger.info(f"  Bergomi baseline ({payoff_name}): "
                f"price = {bergomi_result['price']:.6f}")

    return {
        "price": float(price),
        "se": float(se),
        "ci_half": float(ci_half),
        "bs_price": float(bs_result["price"]),
        "bs_se": float(bs_result["se"]),
        "heston_price": float(heston_result["price"]),
        "heston_se": float(heston_result["se"]),
        "bergomi_price": float(bergomi_result["price"]),
        "bergomi_se": float(bergomi_result["se"]),
        "atm_vol": float(atm_vol),
        "payoffs": payoffs,
        "returns": returns,
        "S_resets": sim["S_resets"],
        "S_all": sim["S_all"],
        "V_all": sim["V_all"],
        "time_grid": sim["time_grid"],
        "reset_indices": sim["reset_indices"],
        "reset_dates": reset_dates,
        "n_paths": n_paths,
        "payoff_name": payoff_name,
        "payoff_kwargs": payoff_kwargs,
        "discount_factor": float(df),
    }

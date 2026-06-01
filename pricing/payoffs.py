import numpy as np


def payoff_accumulator(returns, cap=0.01, floor=0.0):
    """Accumulator: max(0, sum_i max(min(r_i, cap), floor)).
    returns: (n_paths, n_resets) -> (n_paths,)."""
    capped = np.minimum(returns, cap)
    floored = np.maximum(capped, floor)
    return np.maximum(floored.sum(axis=1), 0.0)


def payoff_reverse_cliquet(returns, coupon=0.15):
    """Reverse cliquet: max(0, coupon + sum_i min(r_i, 0)).
    returns: (n_paths, n_resets) -> (n_paths,)."""
    neg_returns = np.minimum(returns, 0.0)
    return np.maximum(coupon + neg_returns.sum(axis=1), 0.0)


def payoff_napoleon(returns, coupon=0.08):
    """Napoleon: max(0, coupon + min_i r_i).
    returns: (n_paths, n_resets) -> (n_paths,)."""
    return np.maximum(coupon + returns.min(axis=1), 0.0)

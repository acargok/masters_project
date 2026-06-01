import numpy as np


def payoff_accumulator(returns, cap=0.01, floor=0.0):
    """
    Accumulator cliquet payoff.

    max(0, sum_i max(min(r_i, cap), floor))

    Parameters
    ----------
    returns : np.ndarray, shape (n_paths, n_resets)
    cap : float
    floor : float

    Returns
    -------
    np.ndarray, shape (n_paths,)
    """
    capped = np.minimum(returns, cap)
    floored = np.maximum(capped, floor)
    return np.maximum(floored.sum(axis=1), 0.0)


def payoff_reverse_cliquet(returns, coupon=0.15):
    """
    Reverse cliquet payoff.

    max(0, C + sum_i min(r_i, 0))

    Parameters
    ----------
    returns : np.ndarray, shape (n_paths, n_resets)
    coupon : float

    Returns
    -------
    np.ndarray, shape (n_paths,)
    """
    neg_returns = np.minimum(returns, 0.0)
    return np.maximum(coupon + neg_returns.sum(axis=1), 0.0)


def payoff_napoleon(returns, coupon=0.08):
    """
    Napoleon cliquet payoff.

    max(0, C + min_i r_i)

    Parameters
    ----------
    returns : np.ndarray, shape (n_paths, n_resets)
    coupon : float

    Returns
    -------
    np.ndarray, shape (n_paths,)
    """
    return np.maximum(coupon + returns.min(axis=1), 0.0)

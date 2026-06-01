#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black-Scholes helpers for LSV validation (Checkpoint 2).

Extracted from lsv_validation.py: bs_call_price, bs_put_price, bs_iv. Imported
by the lsv_validation facade via `from validation_bs import *`.
"""

import numpy as np
from scipy.stats import norm


# =============================================================================
# Black-Scholes helper
# =============================================================================

def bs_call_price(S, K, T, r, q, sigma):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    """Black-Scholes European put price via put-call parity."""
    return bs_call_price(S, K, T, r, q, sigma) - S * np.exp(-q * T) + K * np.exp(-r * T)


def bs_iv(price, S, K, T, r, q, option_type, tol=1e-6):
    """Invert Black-Scholes to implied vol. Returns nan if no solution."""
    from scipy.optimize import brentq
    if T <= 0 or price <= 0:
        return np.nan
    disc_fwd = S * np.exp(-q * T)
    disc_str = K * np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(disc_fwd - disc_str, 0.0)
        fn = lambda sig: bs_call_price(S, K, T, r, q, sig) - price
    else:
        intrinsic = max(disc_str - disc_fwd, 0.0)
        fn = lambda sig: bs_put_price(S, K, T, r, q, sig) - price
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(fn, 1e-6, 5.0, xtol=tol, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan

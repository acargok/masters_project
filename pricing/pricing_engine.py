#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliquet Pricing Engine — Monte Carlo under Calibrated LSV
==========================================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Provides a Monte Carlo simulation engine that prices path-dependent cliquet
payoffs under the calibrated Heston LSV dynamics:

    dS = (r - q) S dt + L(t, S) sqrt(V) S dW^S
    dV = kappa (theta - V) dt + xi sqrt(V) dW^V
    d<W^S, W^V> = rho dt

The engine is structured around two abstractions:
    1. A *variance process* object that steps V forward given (V, S, dt, Z).
       The Heston implementation is `HestonVariance`; a Bergomi two-factor
       implementation can be swapped in by exposing the same interface.
    2. A *payoff function* that maps a vector of per-reset returns to a scalar
       payoff.  Three cliquet payoffs are provided: accumulator, reverse
       cliquet, and napoleon.

Inputs (from previous steps):
    lsv/arrays/leverage_surface.npy       — L(t, S) on a grid
    lsv/arrays/leverage_spot_grid.npy     — spot grid for L
    lsv/arrays/leverage_time_grid.npy     — time grid for L
    lsv/data/heston_params.json           — calibrated Heston parameters
    dupire_vol/data/market_params.json    — S, r, q
    iv_surface/arrays/forward_curve.npy   — forward curve F(0, T)
    iv_surface/arrays/ttm_grid.npy        — TTM grid for forward curve

Usage:
    from pricing_engine import (
        load_pricing_inputs, HestonVariance, build_leverage_interpolator,
        simulate_lsv, payoff_accumulator, payoff_reverse_cliquet,
        payoff_napoleon, price_cliquet,
    )
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy import interpolate
from scipy.stats import norm

logger = logging.getLogger("pricing_engine")

from config import *
from variance_schemes import *
from io_loaders import *
from variance_processes import *
from simulation import *
from payoffs import *
from pricers import *

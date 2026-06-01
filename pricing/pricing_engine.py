#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliquet Pricing Engine — Monte Carlo under Calibrated LSV
=========================================================
Aggregator module: re-exports the cliquet MC pricing engine for the LSV
thesis. Prices path-dependent cliquets (accumulator, reverse cliquet,
napoleon) under Heston LSV dynamics:

    dS = (r - q) S dt + L(t, S) sqrt(V) S dW^S
    dV = kappa (theta - V) dt + xi sqrt(V) dW^V,   d<W^S, W^V> = rho dt

Built on two abstractions: a variance-process object stepping V forward
(HestonVariance, or a swap-in two-factor BergomiVariance with the same
interface), and a payoff function mapping per-reset returns to a scalar.

Inputs: lsv_heston/arrays/leverage_* + data/heston_params.json,
dupire_vol/data/market_params.json (S, r, q),
iv_surface/arrays/{forward_curve,ttm_grid}.npy.
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

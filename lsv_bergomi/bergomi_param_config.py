#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level constants for the Bergomi two-factor parameter calibration."""

# ===== PATHS =====
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
BERGOMI_DIR = ROOT / "lsv_bergomi"
DATA_DIR = BERGOMI_DIR / "data"
PLOT_DIR = BERGOMI_DIR / "plots"

# ===== DEFAULTS (configurable) =====
# Vol-of-vol benchmark: nu^B(T) = sigma0 * (tau0 / T)^alpha
DEFAULT_SIGMA0 = 1.00     # 100% vol-of-vol level at the reference maturity
DEFAULT_TAU0   = 0.25     # reference maturity (3 months)
DEFAULT_ALPHA  = 0.40     # power-law exponent

# ATMF skew strike offset for empirical extraction
SKEW_DELTA_K = 0.01       # +/-1% of forward

# Optimisation
SEED = 42
DE_MAXITER = 500
N_WORKERS = -1            # all cores

# Maturity grids for the two stages (years)
T_GRID_VOLOFVOL = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0,
                              1.5, 2.0])

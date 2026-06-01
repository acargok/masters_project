#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heston calibration config (Step 3a): paths, optimiser, quadrature."""

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
LSV_DIR = ROOT / "lsv_heston"
DATA_DIR = LSV_DIR / "data"
PLOT_DIR = LSV_DIR / "plots"
ARRAY_DIR = LSV_DIR / "arrays"

logger = logging.getLogger("heston_calibration")

# Optimiser
MAX_ITER = 500          # differential evolution iterations
SEED = 42
N_WORKERS = -1          # -1 = all cores
FELLER_PENALTY = 10.0   # soft Feller-violation weight, scaled for IV-approx SSE ~0.01-0.05

# Quadrature (GL nodes)
N_QUAD_OPT  = 100       # in optimisation objective
N_QUAD_DIAG = 200       # final diagnostic pricing
UPPER_LIMIT_OPT  = 100.0
UPPER_LIMIT_DIAG = 200.0

# Vega floor for IV-approx normalisation (absolute dollar vega, SPX scale)
VEGA_FLOOR = 1.0

# Generated-output dirs (git-ignored, recreated each run).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

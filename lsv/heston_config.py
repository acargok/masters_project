#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration constants for Heston calibration (Step 3a).

Holds module-level constants extracted from heston_calibration.py: directory
paths and optimiser/quadrature settings. Imported by the heston_calibration
facade and pricing module via `from heston_config import *`.
"""

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
LSV_DIR = ROOT / "lsv"
DATA_DIR = LSV_DIR / "data"
PLOT_DIR = LSV_DIR / "plots"
ARRAY_DIR = LSV_DIR / "arrays"

logger = logging.getLogger("heston_calibration")

# Optimiser settings
MAX_ITER = 500          # max iterations for differential evolution
SEED = 42               # reproducibility
N_WORKERS = -1          # parallel workers (-1 = all cores)
FELLER_PENALTY = 10.0   # soft penalty weight for Feller condition violation,
                        # scaled for the IV-approx objective (SSE ~ 0.01-0.05).

# Quadrature settings
N_QUAD_OPT  = 100       # GL nodes inside the optimisation objective
N_QUAD_DIAG = 200       # GL nodes for final diagnostic pricing
UPPER_LIMIT_OPT  = 100.0
UPPER_LIMIT_DIAG = 200.0

# Vega floor for IV-approximation normalisation (absolute dollar vega, SPX scale)
VEGA_FLOOR = 1.0

# Ensure generated-output directories exist (git-ignored, recreated each run).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

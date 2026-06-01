#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Particle method config (Step 3b): paths, particle/step settings, clipping, recording grid."""

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
LSV_DIR = ROOT / "lsv_heston"
DATA_DIR = LSV_DIR / "data"
PLOT_DIR = LSV_DIR / "plots"
ARRAY_DIR = LSV_DIR / "arrays"

logger = logging.getLogger("particle_method")

# Convergence analysis params
N_PARTICLES = 5_000
DT = 1.0 / 504.0           # years
BANDWIDTH_OVERRIDE = None   # float overrides NW CV bandwidth selection
L_SQUARED_CLIP = (1e-4, 25.0)   # L^2 clip range
SEED = 42
VARIANCE_SCHEME = "qe"     # "euler" (full-truncation) or "qe" (Andersen 2008)
QE_PSI_C = 1.5             # Andersen QE switching threshold

# L(t, S) recording grid
N_SPOT_GRID = 200
SPOT_GRID_RANGE = (0.70, 1.30)  # moneyness range

# Generated-output dirs (git-ignored, recreated each run).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

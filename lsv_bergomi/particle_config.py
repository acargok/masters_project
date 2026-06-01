#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level constants for the Bergomi particle-method leverage calibration (Step 3b)."""

# Paths
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
BERGOMI_DIR = ROOT / "lsv_bergomi"
DATA_DIR = BERGOMI_DIR / "data"
PLOT_DIR = BERGOMI_DIR / "plots"
ARRAY_DIR = BERGOMI_DIR / "arrays"

# Parameters
N_PARTICLES = 5_000
DT = 1.0 / 504.0
BANDWIDTH_OVERRIDE = None
L_SQUARED_CLIP = (1e-4, 25.0)
SEED = 42

N_SPOT_GRID = 200
SPOT_GRID_RANGE = (0.70, 1.30)

# Create output dirs (git-ignored).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

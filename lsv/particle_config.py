#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration constants for the particle method (Step 3b).

Holds module-level constants extracted from particle_method.py: directory paths,
particle/time-step settings, clipping ranges, and the recording grid. Imported
by the particle_method facade and helper modules via `from particle_config
import *`.
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

logger = logging.getLogger("particle_method")

# Top-level parameters for the convergence analysis.
N_PARTICLES = 5_000        # number of particles
DT = 1.0 / 504.0           # time step in years
BANDWIDTH_OVERRIDE = None   # set to a float to override NW CV bandwidth selection
L_SQUARED_CLIP = (1e-4, 25.0)   # clipping range for L^2
SEED = 42                  # random seed
VARIANCE_SCHEME = "qe"     # "euler" (full-truncation) or "qe" (Andersen 2008)
QE_PSI_C = 1.5             # Andersen QE switching threshold

# Grid for recording L(t, S)
N_SPOT_GRID = 200           # number of spot grid points for leverage surface
SPOT_GRID_RANGE = (0.70, 1.30)  # moneyness range for spot grid

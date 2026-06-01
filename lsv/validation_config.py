#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration constants for LSV validation (Checkpoint 2).

Holds module-level constants extracted from lsv_validation.py: directory paths
and Monte Carlo settings. Imported by the lsv_validation facade via
`from validation_config import *`.
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

logger = logging.getLogger("lsv_validation")

MC_N_PATHS = 100_000       # number of MC paths for validation
MC_STEPS_PER_YEAR = 252    # daily time steps
MC_N_REPRICE = 0           # number of options to reprice (0 = all in-bounds)
MC_SEED = 42

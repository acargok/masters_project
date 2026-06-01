#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSV validation config (Checkpoint 2): paths, Monte Carlo settings."""

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
LSV_DIR = ROOT / "lsv_heston"
DATA_DIR = LSV_DIR / "data"
PLOT_DIR = LSV_DIR / "plots"
ARRAY_DIR = LSV_DIR / "arrays"

logger = logging.getLogger("lsv_validation")

MC_N_PATHS = 100_000
MC_STEPS_PER_YEAR = 252    # daily
MC_N_REPRICE = 0           # options to reprice (0 = all in-bounds)
MC_SEED = 42

# Generated-output dirs (git-ignored, recreated each run).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level constants for the Bergomi LSV validation (Checkpoint 2)."""

# ===== PATHS =====
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
BERGOMI_DIR = ROOT / "lsv_bergomi"
DATA_DIR = BERGOMI_DIR / "data"
PLOT_DIR = BERGOMI_DIR / "plots"
ARRAY_DIR = BERGOMI_DIR / "arrays"

# ===== CONFIGURATION =====
MC_N_PATHS = 100_000
MC_STEPS_PER_YEAR = 252
MC_N_REPRICE = 0           # number of options to reprice (0 = all in-bounds)
MC_SEED = 42

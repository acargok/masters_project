#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level constants for the Bergomi pure-SV fit diagnostic (Step 3a'')."""

# ===== PATHS =====
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
BERGOMI_DIR = ROOT / "lsv_bergomi"
DATA_DIR = BERGOMI_DIR / "data"
PLOT_DIR = BERGOMI_DIR / "plots"
ARRAY_DIR = BERGOMI_DIR / "arrays"

# ===== DEFAULTS =====
N_PATHS = 100_000
STEPS_PER_YEAR = 252
MAX_OPTIONS = 2000
SEED = 42

# Ensure generated-output directories exist (git-ignored, recreated each run).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

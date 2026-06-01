#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level constants for the Bergomi forward-variance calibration (Step 3a)."""

# Paths
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
BERGOMI_DIR = ROOT / "lsv_bergomi"
DATA_DIR = BERGOMI_DIR / "data"
PLOT_DIR = BERGOMI_DIR / "plots"
ARRAY_DIR = BERGOMI_DIR / "arrays"

# VS vol extraction method:
#   "carr_madan": Carr-Madan / DDKZ model-free replication (default)
#   "proxy":      uniform smile-average of total variance (approximation)
VS_METHOD_DEFAULT = "carr_madan"

_NSS_T_SMALL = 1e-8

# Create output dirs (git-ignored).
for _d in (DATA_DIR, PLOT_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

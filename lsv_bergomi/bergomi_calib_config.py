#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level constants for the Bergomi forward-variance calibration (Step 3a)."""

# ===== PATHS =====
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "iv_surface"
DUPIRE_DIR = ROOT / "dupire_vol"
BERGOMI_DIR = ROOT / "lsv_bergomi"
DATA_DIR = BERGOMI_DIR / "data"
PLOT_DIR = BERGOMI_DIR / "plots"
ARRAY_DIR = BERGOMI_DIR / "arrays"

# ===== Variance swap vol extraction defaults =====
# Two methods are provided:
#   "carr_madan": Carr-Madan (1998) / Demeterfi-Derman-Kamal-Zou (1999)
#                 model-free replication. The default.
#   "proxy":      uniform smile-average of total variance over the log-moneyness
#                 grid; a quick approximation.

VS_METHOD_DEFAULT = "carr_madan"

_NSS_T_SMALL = 1e-8

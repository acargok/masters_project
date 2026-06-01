# -*- coding: utf-8 -*-
"""Configuration constants for the Dupire local volatility pipeline (Step 2)."""

# ===== PARAMETERS =====
# Input directories (Step 1 outputs)
IV_DIR_ARRAYS = "iv_surface/arrays"
IV_DIR_DATA   = "iv_surface/data"

# Output directories
DIR_DATA   = "dupire_vol/data"
DIR_PLOTS  = "dupire_vol/plots"
DIR_ARRAYS = "dupire_vol/arrays"

# Dupire stability parameters
MIN_G_VALUE     = 1e-6   # floor for the Gatheral g-function (prevents 1/0 blow-up)
LOCAL_VOL_FLOOR = 0.01   # minimum local vol (1% annualised)
LOCAL_VOL_CAP   = 3.0    # maximum local vol (300% annualised)

# Monte Carlo parameters
MC_N_PATHS        = 1000000   # number of simulated paths
MC_STEPS_PER_YEAR = 1000       # Euler-Maruyama time steps per year
MC_N_REPRICE      = 0         # 0 = reprice ALL in-bounds options
MC_SEED           = 42

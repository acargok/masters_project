# -*- coding: utf-8 -*-
"""Configuration constants for the Dupire local volatility pipeline (Step 2)."""

# Step 1 inputs
IV_DIR_ARRAYS = "iv_surface/arrays"
IV_DIR_DATA   = "iv_surface/data"

# Outputs
DIR_DATA   = "dupire_vol/data"
DIR_PLOTS  = "dupire_vol/plots"
DIR_ARRAYS = "dupire_vol/arrays"

# Dupire stability
MIN_G_VALUE     = 1e-6   # g floor, prevents 1/0
LOCAL_VOL_FLOOR = 0.01   # 1% annualised
LOCAL_VOL_CAP   = 3.0    # 300% annualised

# Monte Carlo
MC_N_PATHS        = 1000000
MC_STEPS_PER_YEAR = 1000       # Euler-Maruyama steps/year
MC_N_REPRICE      = 0         # 0 = all in-bounds options
MC_SEED           = 42

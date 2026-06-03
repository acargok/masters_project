#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward-skew decomposition — shared paths and constants.

Isolates the forward-skew contribution to the Heston-vs-Bergomi cliquet gap by
re-running each backbone with spot-variance correlation forced to zero
(Heston rho = 0; Bergomi rho1 = rho2 = 0), recalibrating the leverage function,
and re-pricing the cliquets. The forward-skew contribution is
G = (Δ_baseline − Δ_zero) / Δ_baseline, with Δ = P_Heston − P_Bergomi.

All output lives under this folder; mainline stage outputs are never modified.
Requires the mainline pipeline and cliquet pricing to have been run first.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECOMP_DIR = Path(__file__).resolve().parent

# Mainline stage folders (read-only sources)
LSV_HESTON_DIR = ROOT / "lsv_heston"
LSV_BERGOMI_DIR = ROOT / "lsv_bergomi"
PRICING_DIR = ROOT / "pricing"

# Experiment outputs
HESTON0_DIR = DECOMP_DIR / "heston_rho0"
BERGOMI0_DIR = DECOMP_DIR / "bergomi_rho0"
PLOTS_DIR = DECOMP_DIR / "plots"
DATA_DIR = DECOMP_DIR / "data"
REPORT_PATH = DECOMP_DIR / "report.md"

# Cliquet pricing settings (match pricing/run_pricing.py)
SEED = 42
N_PATHS = 500_000
DT_MAX = 1.0 / 52.0

# Cliquet payoffs priced under each backbone
PAYOFF_NAMES = ["accumulator", "reverse_cliquet", "napoleon"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward-skew decomposition driver.

Runs each backbone's rho = 0 re-run as its own subprocess (so the two stage
folders never share a module cache), then reprices the cliquets in-process and
writes the report quantifying the forward-skew contribution
G = (Δ_baseline − Δ_zero) / Δ_baseline.

Pipeline:
  1. lsv_heston ρ = 0     — particle method + vanilla validation  (subprocess)
  2. lsv_bergomi ρ₁=ρ₂=0  — particle method + vanilla validation  (subprocess)
  3. cliquet pricing under both ρ = 0 leverages
  4. report + plots vs the baseline (production) cliquet prices

Run from the repo root. Requires the mainline pipeline and cliquet pricing to
have been run first (it reads each stage's calibrated outputs and the baseline
pricing_results.json). Production outputs are never modified.

Usage:
    python forward_skew_decomposition/run_decomposition.py
        [--skip-heston-particle] [--skip-bergomi-particle] [--skip-pricing]
"""
import argparse
import json
import logging
import subprocess
import sys
import time

import decomp_config as cfg
import cliquet_pricing as cp
import report as rp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("decomposition")


def _run_backbone(script_name):
    """Run a backbone ρ = 0 script as an isolated subprocess from the repo root."""
    script = cfg.DECOMP_DIR / script_name
    logger.info(f">>> subprocess: {script_name}")
    rc = subprocess.run([sys.executable, "-u", str(script)], cwd=str(cfg.ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"{script_name} failed (rc={rc})")


def _load_summary(data_dir):
    with open(data_dir / "validation_summary.json") as f:
        return json.load(f)


def main(skip_heston_particle=False, skip_bergomi_particle=False, skip_pricing=False):
    for d in (cfg.DATA_DIR, cfg.PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not skip_heston_particle:
        _run_backbone("heston_rho0.py")
    if not skip_bergomi_particle:
        _run_backbone("bergomi_rho0.py")

    h_val = _load_summary(cfg.HESTON0_DIR / "data")
    b_val = _load_summary(cfg.BERGOMI0_DIR / "data")

    results_path = cfg.DATA_DIR / "pricing_results.json"
    if not skip_pricing:
        zero_corr = {
            "n_paths": cfg.N_PATHS, "dt_max": cfg.DT_MAX, "seed": cfg.SEED,
            "experiment": "decomposition_rho_zero",
            "heston_options": cp.price_heston(),
            "bergomi_options": cp.price_bergomi(),
        }
        with open(results_path, "w") as f:
            json.dump(zero_corr, f, indent=2)
    else:
        with open(results_path) as f:
            zero_corr = json.load(f)
        logger.info(">>> Loaded cached zero-corr pricing results")

    baseline = rp.load_baseline()
    rp.write_report(baseline, zero_corr, h_val, b_val)
    rp.generate_plots(baseline, zero_corr)

    logger.info("=" * 70)
    logger.info(f"  Decomposition complete in {time.time() - t0:.1f}s "
                f"→ {cfg.REPORT_PATH}")
    logger.info("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Forward-skew decomposition driver")
    p.add_argument("--skip-heston-particle", action="store_true",
                   help="Reuse the cached Heston ρ = 0 leverage/validation")
    p.add_argument("--skip-bergomi-particle", action="store_true",
                   help="Reuse the cached Bergomi ρ₁=ρ₂=0 leverage/validation")
    p.add_argument("--skip-pricing", action="store_true",
                   help="Reuse the cached zero-corr pricing_results.json")
    a = p.parse_args()
    main(a.skip_heston_particle, a.skip_bergomi_particle, a.skip_pricing)

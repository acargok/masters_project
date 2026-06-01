#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSV Model — Full Pipeline Runner
================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Runs the Step 3 pipeline:
    3a) Heston calibration
    3b) Particle method for leverage function L(t, S)
    Checkpoint 2) LSV validation via MC repricing

Convergence-analysis parameters (Chapter 4) are centralised at module level.

Usage:
    python run_lsv.py                   # full pipeline with defaults
    python run_lsv.py --particles 10000 # override N
    python run_lsv.py --skip-heston     # reuse existing Heston params
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_lsv")

# Parameters varied in the convergence analysis; override via CLI arguments.
N_PARTICLES = 5_000              # number of particles
DT = 1.0 / 252.0                # daily time step
BANDWIDTH_OVERRIDE = None        # None = Silverman's rule; set float to override
MAX_CALIBRATION_OPTIONS = 500    # Heston calibration: max options
MC_VALIDATION_PATHS = 100_000   # MC repricing: number of paths
MC_VALIDATION_OPTIONS = 200      # MC repricing: number of options (0 = all)
SEED = 42


def main():
    """Run the full LSV pipeline."""
    parser = argparse.ArgumentParser(description="LSV Model — Step 3 Pipeline")
    parser.add_argument("--particles", "-N", type=int, default=N_PARTICLES,
                        help="Number of particles (default: 5000)")
    parser.add_argument("--dt", type=float, default=DT,
                        help="Time step in years (default: 1/252)")
    parser.add_argument("--bandwidth", "-bw", type=float, default=None,
                        help="Bandwidth override (default: Silverman's rule)")
    parser.add_argument("--mc-paths", type=int, default=MC_VALIDATION_PATHS,
                        help="MC validation paths (default: 100000)")
    parser.add_argument("--mc-options", type=int, default=MC_VALIDATION_OPTIONS,
                        help="Options to reprice (default: 200, 0 = all)")
    parser.add_argument("--max-cal-options", type=int, default=MAX_CALIBRATION_OPTIONS,
                        help="Max options for Heston calibration (default: 500)")
    parser.add_argument("--skip-heston", action="store_true",
                        help="Skip Heston calibration (reuse existing params)")
    parser.add_argument("--skip-particles", action="store_true",
                        help="Skip particle method (reuse existing leverage)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  LSV MODEL — STEP 3 PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Particles:   {args.particles:,}")
    logger.info(f"  dt:          {args.dt:.6f} ({1/args.dt:.0f} steps/year)")
    logger.info(f"  Bandwidth:   {'Silverman' if args.bandwidth is None else args.bandwidth}")
    logger.info(f"  MC paths:    {args.mc_paths:,}")
    logger.info(f"  Seed:        {args.seed}")
    logger.info("=" * 70)

    # ---- Step 3a: Heston calibration ----
    if not args.skip_heston:
        logger.info("")
        logger.info(">>> STEP 3a: Heston Calibration")
        import heston_calibration
        heston_params = heston_calibration.run(max_options=args.max_cal_options)
        logger.info(f">>> Step 3a complete in {time.time() - t_start:.1f}s")
    else:
        logger.info(">>> Skipping Step 3a (--skip-heston)")

    # ---- Step 3b: Particle method ----
    t_particles = time.time()
    if not args.skip_particles:
        logger.info("")
        logger.info(">>> STEP 3b: Particle Method")
        import particle_method
        particle_results = particle_method.run(
            N=args.particles,
            dt=args.dt,
            bandwidth_override=args.bandwidth,
            seed=args.seed,
        )
        logger.info(f">>> Step 3b complete in {time.time() - t_particles:.1f}s")
    else:
        logger.info(">>> Skipping Step 3b (--skip-particles)")

    # ---- Checkpoint 2: Validation ----
    t_val = time.time()
    logger.info("")
    logger.info(">>> CHECKPOINT 2: LSV Validation")
    import lsv_validation
    result_df, summary = lsv_validation.run(
        n_paths=args.mc_paths,
        n_reprice=args.mc_options,
        seed=args.seed,
    )
    logger.info(f">>> Checkpoint 2 complete in {time.time() - t_val:.1f}s")

    # ---- Summary ----
    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    logger.info("=" * 70)
    if summary:
        logger.info(f"  LSV repricing MAE: {summary.get('mae_price_pct', 'N/A')}%")
        if "dupire_mae_pct" in summary:
            logger.info(f"  Dupire MAE:        {summary['dupire_mae_pct']}% (comparison)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

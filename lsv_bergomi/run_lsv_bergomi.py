#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bergomi LSV Model — Full Pipeline Runner
==========================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Runs the complete Step 3 (Bergomi) pipeline:
    3a) Forward variance extraction from SSVI surface
    3b) Particle method for leverage function sigma(t, S)
    Checkpoint 2) LSV validation via MC repricing

Top-level parameters for convergence analysis (Chapter 4) are centralised here
and can be easily varied.

Usage:
    python run_lsv_bergomi.py                   # run full pipeline with defaults
    python run_lsv_bergomi.py --particles 10000 # override N
    python run_lsv_bergomi.py --skip-fwd-var    # skip forward variance, reuse existing
"""

# ===== IMPORTS =====
import argparse
import logging
import sys
import time
from pathlib import Path

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_lsv_bergomi")

# ===== TOP-LEVEL PARAMETERS =====
N_PARTICLES = 5_000              # number of particles
DT = 1.0 / 504.0                # daily time step
BANDWIDTH_OVERRIDE = None        # None = LOO-CV; set float to override
MC_VALIDATION_PATHS = 100_000   # MC repricing: number of paths
MC_VALIDATION_OPTIONS = 0      # MC repricing: number of options (0 = all)
SEED = 42


def main():
    """Run the full Bergomi LSV pipeline."""
    parser = argparse.ArgumentParser(description="Bergomi LSV Model — Step 3 Pipeline")
    parser.add_argument("--particles", "-N", type=int, default=N_PARTICLES,
                        help="Number of particles (default: 5000)")
    parser.add_argument("--dt", type=float, default=DT,
                        help="Time step in years (default: 1/252)")
    parser.add_argument("--bandwidth", "-bw", type=float, default=None,
                        help="Bandwidth override (default: LOO-CV)")
    parser.add_argument("--mc-paths", type=int, default=MC_VALIDATION_PATHS,
                        help="MC validation paths (default: 100000)")
    parser.add_argument("--mc-options", type=int, default=MC_VALIDATION_OPTIONS,
                        help="Options to reprice (default: 200, 0 = all)")
    parser.add_argument("--skip-fwd-var", action="store_true",
                        help="Skip forward variance extraction (reuse existing)")
    parser.add_argument("--skip-particles", action="store_true",
                        help="Skip particle method (reuse existing leverage)")
    parser.add_argument("--skip-param-calib", action="store_true",
                        help="Skip Bergomi parameter calibration (reuse existing JSON)")
    parser.add_argument("--skip-pure-sv-fit", action="store_true",
                        help="Skip pure-Bergomi vanilla repricing diagnostic (Step 3a'')")
    parser.add_argument("--use-handpicked", action="store_true",
                        help="Restore Wang Set II handpicked parameters before running "
                             "(reads from data/bergomi_params_handpicked_backup.json)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  BERGOMI LSV MODEL — STEP 3 PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Particles:   {args.particles:,}")
    logger.info(f"  dt:          {args.dt:.6f} ({1/args.dt:.0f} steps/year)")
    logger.info(f"  Bandwidth:   {'LOO-CV' if args.bandwidth is None else args.bandwidth}")
    logger.info(f"  MC paths:    {args.mc_paths:,}")
    logger.info(f"  Seed:        {args.seed}")
    logger.info("=" * 70)

    # Add this directory to path for imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    HERE = Path(__file__).resolve().parent
    PARAMS_PATH = HERE / "data" / "bergomi_params.json"
    HANDPICKED_BACKUP = HERE / "data" / "bergomi_params_handpicked_backup.json"

    # ---- Step 3a: Forward variance extraction ----
    if not args.skip_fwd_var:
        logger.info("")
        logger.info(">>> STEP 3a: Forward Variance Extraction")
        import bergomi_calibration
        fit_params, fwd_var, vs_vol = bergomi_calibration.run()
        logger.info(f">>> Step 3a complete in {time.time() - t_start:.1f}s")
    else:
        logger.info(">>> Skipping Step 3a (--skip-fwd-var)")

    # ---- Step 3a': Bergomi parameter calibration (Wang separate calibration) ----
    if args.use_handpicked:
        if HANDPICKED_BACKUP.exists():
            import shutil
            shutil.copy(HANDPICKED_BACKUP, PARAMS_PATH)
            logger.info(">>> Restored handpicked Wang Set II parameters "
                        f"from {HANDPICKED_BACKUP.name}")
        else:
            logger.warning(f">>> --use-handpicked requested but {HANDPICKED_BACKUP} "
                            "does not exist; proceeding with whatever is in "
                            f"{PARAMS_PATH.name}")
    elif not args.skip_param_calib:
        t_calib = time.time()
        logger.info("")
        logger.info(">>> STEP 3a': Bergomi parameter calibration (Wang §5.2.3)")
        import bergomi_param_calibration
        bergomi_param_calibration.run()
        logger.info(f">>> Step 3a' complete in {time.time() - t_calib:.1f}s")
    else:
        logger.info(">>> Skipping Step 3a' parameter calibration (--skip-param-calib)")

    # ---- Step 3a'': Pure-Bergomi vanilla repricing diagnostic ----
    # Shows how well the calibrated pure stochastic-volatility backbone fits
    # market vanillas before the LSV leverage function is applied.
    if not args.skip_pure_sv_fit:
        t_fit = time.time()
        logger.info("")
        logger.info(">>> STEP 3a'': Pure-Bergomi Vanilla Fit Diagnostic")
        import bergomi_pure_sv_fit
        bergomi_pure_sv_fit.run(seed=args.seed)
        logger.info(f">>> Step 3a'' complete in {time.time() - t_fit:.1f}s")
    else:
        logger.info(">>> Skipping Step 3a'' pure-SV fit (--skip-pure-sv-fit)")

    # ---- Step 3b: Particle method ----
    t_particles = time.time()
    if not args.skip_particles:
        logger.info("")
        logger.info(">>> STEP 3b: Particle Method (Bergomi backbone)")
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
    logger.info(">>> CHECKPOINT 2: Bergomi LSV Validation")
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
    logger.info(f"  BERGOMI LSV PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    logger.info("=" * 70)
    if summary:
        logger.info(f"  Vol-space RMSE:  {summary.get('lsv_iv_rmse_bps', 'N/A'):.1f} bp")
        logger.info(f"  Vol-space MAE:   {summary.get('lsv_iv_mae_bps', 'N/A'):.1f} bp")
        logger.info(f"  Vol-space ME:    {summary.get('lsv_iv_me_bps', 'N/A'):+.1f} bp")
        if "lsv_vs_ssvi_mae_pct" in summary:
            logger.info(f"  Price vs SSVI MAE: {summary['lsv_vs_ssvi_mae_pct']:.2f}%")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

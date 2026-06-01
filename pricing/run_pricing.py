#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliquet Pricing — Main Runner
================================
Part of an LSV (Local Stochastic Volatility) model for pricing Asian options.
Master's Thesis, Imperial College London.

Runs the cliquet pricing step under both Heston LSV and Bergomi LSV:
    1) Accumulator cliquet (cap=1%, floor=0%)
    2) Reverse cliquet (coupon=15%)
    3) Napoleon cliquet (coupon=8%)

All three use monthly resets, 1-year maturity, real forward curve and
discount factors from saved data.

Usage:
    python run_pricing.py
    python run_pricing.py --paths 1000000
    python run_pricing.py --skip-accumulator --skip-reverse
    python run_pricing.py --skip-bergomi
"""

# ===== IMPORTS =====
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pricing")

# ===== PATHS =====
PRICING_DIR = Path(__file__).resolve().parent
DATA_DIR = PRICING_DIR / "data"
ARRAY_DIR = PRICING_DIR / "arrays"
for _d in (DATA_DIR, ARRAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ===== TOP-LEVEL PARAMETERS =====
N_PATHS = 500_000
DT_MAX = 1.0 / 52.0    # weekly sub-stepping
SEED = 42
N_SAMPLE_PATHS = 200    # sample paths saved for explorer diagnostics


def main():
    parser = argparse.ArgumentParser(description="Cliquet Pricing — Step 4")
    parser.add_argument("--paths", "-N", type=int, default=N_PATHS,
                        help="Number of MC paths (default: 500000)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip-accumulator", action="store_true")
    parser.add_argument("--skip-reverse", action="store_true")
    parser.add_argument("--skip-napoleon", action="store_true")
    parser.add_argument("--skip-bergomi", action="store_true",
                        help="Skip Bergomi LSV pricing (Heston only)")
    parser.add_argument("--skip-heston", action="store_true",
                        help="Skip Heston LSV pricing (Bergomi only)")
    args = parser.parse_args()

    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  CLIQUET PRICING — STEP 4")
    logger.info("=" * 70)
    logger.info(f"  Paths:     {args.paths:,}")
    logger.info(f"  dt_max:    {DT_MAX:.4f} (weekly)")
    logger.info(f"  Seed:      {args.seed}")
    logger.info(f"  Models:    {'Heston LSV' if not args.skip_heston else ''}"
                f"{'  +  Bergomi LSV' if not args.skip_bergomi else ''}")
    logger.info("=" * 70)

    # Add pricing dir to path for imports
    sys.path.insert(0, str(PRICING_DIR))
    import pricing_engine as pe

    options = [
        ("accumulator", pe.payoff_accumulator, {"cap": 0.01, "floor": 0.0},
         args.skip_accumulator),
        ("reverse_cliquet", pe.payoff_reverse_cliquet, {"coupon": 0.15},
         args.skip_reverse),
        ("napoleon", pe.payoff_napoleon, {"coupon": 0.08},
         args.skip_napoleon),
    ]

    # Results container
    results_summary = {
        "n_paths": args.paths,
        "dt_max": DT_MAX,
        "seed": args.seed,
        "n_resets": 12,
        "maturity": 1.0,
        "heston_options": {},
        "bergomi_options": {},
    }

    # ================================================================
    # Phase 1: Heston LSV
    # ================================================================
    if not args.skip_heston:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PHASE 1: HESTON LSV PRICING")
        logger.info("=" * 70)

        inputs = pe.load_pricing_inputs()
        results_summary["S0"] = inputs["S"]
        results_summary["r"] = inputs["r"]
        results_summary["q"] = inputs["q"]
        results_summary["date"] = inputs["date"]

        first_sim_saved = False

        for name, payoff_fn, kwargs, skip in options:
            if skip:
                logger.info(f">>> Skipping {name}")
                continue

            logger.info("")
            logger.info(f">>> Pricing (Heston): {name}")
            t_opt = time.time()

            result = pe.price_cliquet(
                inputs, payoff_fn, kwargs, name,
                n_paths=args.paths, dt_max=DT_MAX, seed=args.seed,
            )

            elapsed = time.time() - t_opt
            logger.info(f">>> {name} (Heston) complete in {elapsed:.1f}s")

            # Save per-path data
            np.save(ARRAY_DIR / f"{name}_payoffs.npy", result["payoffs"])
            np.save(ARRAY_DIR / f"{name}_returns.npy", result["returns"])
            np.save(ARRAY_DIR / f"{name}_S_resets.npy", result["S_resets"])

            if not first_sim_saved:
                np.save(ARRAY_DIR / "sim_time_grid.npy", result["time_grid"])
                np.save(ARRAY_DIR / "reset_dates.npy", result["reset_dates"])
                n_save = min(N_SAMPLE_PATHS, args.paths)
                np.save(ARRAY_DIR / "sample_paths_S.npy",
                        result["S_all"][:n_save, :])
                np.save(ARRAY_DIR / "sample_paths_V.npy",
                        result["V_all"][:n_save, :])
                np.save(ARRAY_DIR / "reset_indices.npy", result["reset_indices"])
                first_sim_saved = True
                logger.info(f"  Saved {n_save} sample paths for explorer")

            results_summary["heston_options"][name] = {
                "price": result["price"],
                "se": result["se"],
                "ci_half": result["ci_half"],
                "ci_lower": result["price"] - result["ci_half"],
                "ci_upper": result["price"] + result["ci_half"],
                "bs_price": result["bs_price"],
                "bs_se": result["bs_se"],
                "heston_price": result["heston_price"],
                "heston_se": result["heston_se"],
                "bergomi_price": result.get("bergomi_price", float("nan")),
                "bergomi_se": result.get("bergomi_se", float("nan")),
                "atm_vol": result["atm_vol"],
                "payoff_kwargs": result["payoff_kwargs"],
                "discount_factor": result["discount_factor"],
                "mean_payoff": float(result["payoffs"].mean()),
                "std_payoff": float(result["payoffs"].std()),
                "pct_zero": float((result["payoffs"] == 0).mean() * 100),
            }

    # ================================================================
    # Phase 2: Bergomi LSV
    # ================================================================
    if not args.skip_bergomi:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PHASE 2: BERGOMI LSV PRICING")
        logger.info("=" * 70)

        bergomi_inputs = pe.load_bergomi_pricing_inputs()
        results_summary["S0"] = bergomi_inputs["S"]
        results_summary["r"] = bergomi_inputs["r"]
        results_summary["q"] = bergomi_inputs["q"]
        results_summary["date"] = bergomi_inputs.get("date", "unknown")

        first_bergomi_saved = False

        for name, payoff_fn, kwargs, skip in options:
            if skip:
                logger.info(f">>> Skipping {name}")
                continue

            logger.info("")
            logger.info(f">>> Pricing (Bergomi): {name}")
            t_opt = time.time()

            result = pe.price_cliquet_bergomi(
                bergomi_inputs, payoff_fn, kwargs, name,
                n_paths=args.paths, dt_max=DT_MAX, seed=args.seed,
            )

            elapsed = time.time() - t_opt
            logger.info(f">>> {name} (Bergomi) complete in {elapsed:.1f}s")

            # Save per-path data with bergomi_ prefix
            np.save(ARRAY_DIR / f"bergomi_{name}_payoffs.npy", result["payoffs"])
            np.save(ARRAY_DIR / f"bergomi_{name}_returns.npy", result["returns"])
            np.save(ARRAY_DIR / f"bergomi_{name}_S_resets.npy", result["S_resets"])

            if not first_bergomi_saved:
                n_save = min(N_SAMPLE_PATHS, args.paths)
                np.save(ARRAY_DIR / "bergomi_sample_paths_S.npy",
                        result["S_all"][:n_save, :])
                np.save(ARRAY_DIR / "bergomi_sample_paths_V.npy",
                        result["V_all"][:n_save, :])
                np.save(ARRAY_DIR / "bergomi_sim_time_grid.npy",
                        result["time_grid"])
                np.save(ARRAY_DIR / "bergomi_reset_indices.npy",
                        result["reset_indices"])
                first_bergomi_saved = True
                logger.info(f"  Saved {n_save} Bergomi sample paths for explorer")

            results_summary["bergomi_options"][name] = {
                "price": result["price"],
                "se": result["se"],
                "ci_half": result["ci_half"],
                "ci_lower": result["price"] - result["ci_half"],
                "ci_upper": result["price"] + result["ci_half"],
                "bs_price": result["bs_price"],
                "bs_se": result["bs_se"],
                "heston_price": result["heston_price"],
                "heston_se": result["heston_se"],
                "bergomi_price": result.get("bergomi_price", float("nan")),
                "bergomi_se": result.get("bergomi_se", float("nan")),
                "atm_vol": result["atm_vol"],
                "payoff_kwargs": result["payoff_kwargs"],
                "discount_factor": result["discount_factor"],
                "mean_payoff": float(result["payoffs"].mean()),
                "std_payoff": float(result["payoffs"].std()),
                "pct_zero": float((result["payoffs"] == 0).mean() * 100),
            }

    # An "options" key aliases the Heston results (Bergomi if no Heston).
    if results_summary["heston_options"]:
        results_summary["options"] = results_summary["heston_options"]
    elif results_summary["bergomi_options"]:
        results_summary["options"] = results_summary["bergomi_options"]

    # Save summary
    with open(DATA_DIR / "pricing_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"\nSaved pricing results -> {DATA_DIR / 'pricing_results.json'}")

    # ================================================================
    # Print comparison table
    # ================================================================
    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 90)
    logger.info(f"  PRICING COMPLETE — Total time: {total_time:.1f}s")
    logger.info("=" * 90)

    opt_names = list(set(
        list(results_summary["heston_options"].keys()) +
        list(results_summary["bergomi_options"].keys())
    ))
    opt_names.sort()

    header = (f"  {'Option':<20} {'Heston LSV':>12} {'Bergomi LSV':>12} "
              f"{'Pure Heston':>12} {'Pure Bergomi':>13} {'BS':>12} "
              f"{'H-B Diff':>10}")
    logger.info(header)
    logger.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*13} {'-'*12} {'-'*10}")

    for name in opt_names:
        h = results_summary["heston_options"].get(name, {})
        b = results_summary["bergomi_options"].get(name, {})

        h_price = h.get("price", float("nan"))
        b_price = b.get("price", float("nan"))
        pure_h = h.get("heston_price", b.get("heston_price", float("nan")))
        pure_b = b.get("bergomi_price", h.get("bergomi_price", float("nan")))
        bs = h.get("bs_price", b.get("bs_price", float("nan")))
        diff = h_price - b_price if h and b else float("nan")

        logger.info(f"  {name:<20} {h_price:>12.6f} {b_price:>12.6f} "
                    f"{pure_h:>12.6f} {pure_b:>13.6f} {bs:>12.6f} "
                    f"{diff:>+10.6f}")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()

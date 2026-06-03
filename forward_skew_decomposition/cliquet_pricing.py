#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cliquet pricing under the rho = 0 leverage surfaces.

Reprices the three cliquets under each zero-correlation backbone using the
production pricing engine, matching the production MC settings. Only
pricing_engine is imported, so this runs in-process alongside the orchestrator
without any module-name clash.
"""
import json
import logging
import sys
import time

import numpy as np

import decomp_config as cfg

sys.path.insert(0, str(cfg.PRICING_DIR))
import pricing_engine as pe   # noqa: E402

logger = logging.getLogger("decomp_pricing")

PAYOFFS = [
    ("accumulator",     pe.payoff_accumulator,     {"cap": 0.01, "floor": 0.0}),
    ("reverse_cliquet", pe.payoff_reverse_cliquet, {"coupon": 0.15}),
    ("napoleon",        pe.payoff_napoleon,        {"coupon": 0.08}),
]


def _summary(r):
    """JSON-serialisable price summary (drops the raw path arrays)."""
    return {
        "price": float(r["price"]), "se": float(r["se"]),
        "ci_half": float(r["ci_half"]), "bs_price": float(r["bs_price"]),
        "heston_price": float(r["heston_price"]),
        "bergomi_price": float(r.get("bergomi_price", float("nan"))),
        "atm_vol": float(r["atm_vol"]),
        "discount_factor": float(r["discount_factor"]),
    }


def _patch_leverage(inputs, params_path, lev_dir):
    """Point pricing inputs at the rho = 0 leverage; return the rho = 0 params."""
    with open(params_path) as f:
        params = json.load(f)
    inputs["leverage"] = np.load(lev_dir / "leverage_surface.npy")
    inputs["spot_grid"] = np.load(lev_dir / "leverage_spot_grid.npy")
    inputs["time_grid"] = np.load(lev_dir / "leverage_time_grid.npy")
    return params


def price_heston():
    """Re-price the three cliquets under Heston rho = 0 LSV."""
    logger.info(">>> Heston rho = 0 cliquet pricing")
    inputs = pe.load_pricing_inputs()
    inputs["heston"] = _patch_leverage(
        inputs, cfg.HESTON0_DIR / "data" / "heston_params.json",
        cfg.HESTON0_DIR / "arrays")
    out = {}
    for name, fn, kwargs in PAYOFFS:
        t0 = time.time()
        r = pe.price_cliquet(inputs, fn, kwargs, name,
                             n_paths=cfg.N_PATHS, dt_max=cfg.DT_MAX, seed=cfg.SEED)
        logger.info(f"    {name}: {r['price']:.6f} ± {r['ci_half']:.6f} "
                    f"({time.time() - t0:.1f}s)")
        out[name] = _summary(r)
    return out


def price_bergomi():
    """Re-price the three cliquets under Bergomi rho1 = rho2 = 0 LSV."""
    logger.info(">>> Bergomi rho1 = rho2 = 0 cliquet pricing")
    inputs = pe.load_bergomi_pricing_inputs()
    inputs["bergomi"] = _patch_leverage(
        inputs, cfg.BERGOMI0_DIR / "data" / "bergomi_params.json",
        cfg.BERGOMI0_DIR / "arrays")
    out = {}
    for name, fn, kwargs in PAYOFFS:
        t0 = time.time()
        r = pe.price_cliquet_bergomi(inputs, fn, kwargs, name,
                                     n_paths=cfg.N_PATHS, dt_max=cfg.DT_MAX,
                                     seed=cfg.SEED)
        logger.info(f"    {name}: {r['price']:.6f} ± {r['ci_half']:.6f} "
                    f"({time.time() - t0:.1f}s)")
        out[name] = _summary(r)
    return out

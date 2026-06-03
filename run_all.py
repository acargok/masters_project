#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Pipeline Runner
=====================
Runs the entire Master's-thesis pipeline in dependency order:

    1.  iv_surface/iv_surface_ssvi.py        — SSVI surface fit
    2.  iv_surface/iv_explorer.py            — IV surface diagnostics
    3.  dupire_vol/dupire_local_vol.py       — Dupire local-vol surface
    4.  dupire_vol/dupire_explorer.py        — Dupire diagnostics
    5.  lsv_heston/run_lsv_heston.py         — Heston calibration + particle method + validation
    6.  lsv_bergomi/run_lsv_bergomi.py       — Bergomi forward-variance + particle method + validation
    7.  lsv_bergomi/lsv_explorer.py          — Bergomi LSV diagnostics
    8.  lsv_heston/lsv_explorer.py           — Heston LSV diagnostics
    9.  pricing/run_pricing.py               — cliquet MC pricing
    10. pricing/pricing_explorer.py          — cliquet diagnostics

Each script runs from the repo root; Python adds the script's own folder to
sys.path so sibling imports resolve while root-relative data paths stay valid.
Output streams to stdout and is tee'd to run_logs/.

Usage:
    python run_all.py [--from N] [--to N] [--only N ...] [--dry-run]
                      [--skip-heston] [--skip-particles] [--extra 'N args']
Exit code is 0 only if every requested step succeeds.
"""

import argparse
import datetime as dt
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "run_logs"

# Pipeline: (step_id, working_dir, script_filename, label)
STEPS = [
    (1,  "iv_surface",  "iv_surface_ssvi.py",   "SSVI fit"),
    (2,  "iv_surface",  "iv_explorer.py",       "IV explorer"),
    (3,  "dupire_vol",  "dupire_local_vol.py",  "Dupire local vol"),
    (4,  "dupire_vol",  "dupire_explorer.py",   "Dupire explorer"),
    (5,  "lsv_heston",  "run_lsv_heston.py",    "Heston LSV pipeline"),
    (6,  "lsv_bergomi", "run_lsv_bergomi.py",   "Bergomi LSV pipeline"),
    (7,  "lsv_bergomi", "lsv_explorer.py",      "Bergomi LSV explorer"),
    (8,  "lsv_heston",  "lsv_explorer.py",      "Heston LSV explorer"),
    (9,  "pricing",     "run_pricing.py",       "Cliquet pricing"),
    (10, "pricing",     "pricing_explorer.py",  "Pricing explorer"),
]


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")


def parse_extra(extras):
    """Parse repeated --extra flags `<step_id> <args...>` -> {step_id: [args]}."""
    out = {}
    if not extras:
        return out
    for entry in extras:
        parts = shlex.split(entry)
        if not parts:
            continue
        try:
            step_id = int(parts[0])
        except ValueError:
            raise SystemExit(f"--extra must start with a step id, got {entry!r}")
        out[step_id] = parts[1:]
    return out


def run_step(step_id, cwd, script, label, extra_args, log_path, dry_run):
    """Run one script as a subprocess with live output and tee to a log file."""
    script_path = ROOT / cwd / script
    if not script_path.exists():
        logger.error(f"  step {step_id}: script not found at {script_path}")
        return False, 0.0

    cmd = [sys.executable, "-u", str(script_path)] + list(extra_args)
    logger.info("=" * 70)
    logger.info(f"  STEP {step_id}: {label}")
    logger.info(f"    cwd:  {ROOT}")
    logger.info(f"    cmd:  {' '.join(cmd)}")
    logger.info(f"    log:  {log_path}")
    logger.info("=" * 70)

    if dry_run:
        return True, 0.0

    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rc = -1
    with open(log_path, "w", buffering=1) as logf:
        logf.write(f"# step {step_id}: {label}\n")
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.write(f"# cwd: {ROOT}\n")
        logf.write(f"# started: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            logger.error(f"  step {step_id} interrupted by user")
            raise

        logf.write(f"\n# finished: {dt.datetime.now().isoformat(timespec='seconds')} "
                   f"(rc={rc})\n")

    elapsed = time.time() - t0
    if rc == 0:
        logger.info(f"  step {step_id} OK  ({elapsed:.1f}s)")
        return True, elapsed
    logger.error(f"  step {step_id} FAILED  rc={rc}  ({elapsed:.1f}s)")
    return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="First step to run (default: 1)")
    parser.add_argument("--to", dest="to_step", type=int, default=len(STEPS),
                        help="Last step to run, inclusive (default: last)")
    parser.add_argument("--only", type=int, nargs="+",
                        help="Run only these step ids (overrides --from/--to)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep running after a failure (default: stop)")
    parser.add_argument("--extra", action="append", default=[],
                        help="Extra args for one step, e.g. "
                             "--extra '5 --skip-heston'. May be repeated.")
    parser.add_argument("--skip-heston", action="store_true",
                        help="Forward --skip-heston to step 5 (run_lsv_heston.py)")
    parser.add_argument("--skip-particles", action="store_true",
                        help="Forward --skip-particles to step 5 (run_lsv_heston.py)")
    args = parser.parse_args()

    # Steps to execute
    if args.only:
        wanted = set(args.only)
        plan = [s for s in STEPS if s[0] in wanted]
    else:
        plan = [s for s in STEPS
                if args.from_step <= s[0] <= args.to_step]

    if not plan:
        logger.error("No steps selected.")
        sys.exit(2)

    # Per-step extra args
    extras = parse_extra(args.extra)
    if args.skip_heston:
        extras.setdefault(5, []).append("--skip-heston")
    if args.skip_particles:
        extras.setdefault(5, []).append("--skip-particles")

    # Timestamped log subfolder
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = LOG_DIR / stamp
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plan.txt").write_text(
            "\n".join(f"{s[0]:>2}  {s[1]}/{s[2]}  ({s[3]})  extra={extras.get(s[0], [])}"
                      for s in plan)
        )

    logger.info("=" * 70)
    logger.info(f"  PIPELINE START — {stamp}")
    logger.info(f"  Steps to run: {[s[0] for s in plan]}")
    if args.dry_run:
        logger.info("  (dry-run mode — no execution)")
    logger.info("=" * 70)

    overall_t0 = time.time()
    results = []
    for step_id, cwd, script, label in plan:
        log_path = run_dir / f"step_{step_id:02d}_{script.replace('.py', '')}.log"
        ok, elapsed = run_step(
            step_id, cwd, script, label, extras.get(step_id, []),
            log_path, args.dry_run,
        )
        results.append((step_id, label, ok, elapsed))
        if not ok and not args.continue_on_error:
            logger.error("Stopping pipeline due to step failure "
                         "(use --continue-on-error to override).")
            break

    # Summary
    total = time.time() - overall_t0
    logger.info("=" * 70)
    logger.info(f"  PIPELINE SUMMARY  ({total:.1f}s wall)")
    logger.info("=" * 70)
    logger.info(f"  {'Step':<5}{'Label':<32}{'Status':<10}{'Time (s)':>10}")
    logger.info("  " + "-" * 65)
    for step_id, label, ok, elapsed in results:
        status = "OK" if ok else "FAIL"
        logger.info(f"  {step_id:<5}{label:<32}{status:<10}{elapsed:>10.1f}")
    logger.info("=" * 70)
    if not args.dry_run:
        logger.info(f"  Logs: {run_dir}")

    failed = [r for r in results if not r[2]]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

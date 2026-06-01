# Local Stochastic Volatility Pricing of Cliquet Options

Master's thesis code, Imperial College London. Calibrates a Local Stochastic
Volatility (LSV) model to the SPX implied-volatility surface and uses it to
price path-dependent cliquet options. Two stochastic-volatility backbones are
calibrated in parallel — **Heston** and a **two-factor Bergomi** forward-variance
model — each combined with a Dupire local-volatility leverage function via the
particle (Gyöngy projection) method.

## Pipeline

| Stage | Folder | Entry point | Produces |
|-------|--------|-------------|----------|
| 1. IV surface | `iv_surface/` | `iv_surface_ssvi.py` | SSVI total-variance / implied-vol surface from the SPX option chain |
| 2. Local vol | `dupire_vol/` | `dupire_local_vol.py` | Dupire local-volatility surface (Gatheral formula) |
| 3. Heston LSV | `lsv/` | `run_lsv.py` | Heston calibration → leverage function → MC validation |
| 4. Bergomi LSV | `lsv_bergomi/` | `run_lsv_bergomi.py` | Bergomi forward-variance calibration → leverage function → MC validation |
| 5. Pricing | `pricing/` | `run_pricing.py` | Cliquet prices (accumulator, reverse cliquet, Napoleon) under both models |

Each stage has a `*_explorer.py` Plotly dashboard for diagnostics. The
`iv_surface/results_4_*_figures.py` and `iv_surface/appendix_*.py` scripts
regenerate the thesis figures and tables.

## Structure

Each stage folder is a flat package: an entry script (same name as the stage,
acts as a thin orchestrator and public-API facade) plus focused modules. For
example `iv_surface/`:

```
iv_surface_ssvi.py     # entry: config import, re-exports, main() pipeline
config.py              # paths, snapshot date, r, q, thresholds, grids
market_data.py         # OptionMetrics chain loading, spot/rate/dividend
data_cleaning.py       # liquidity / no-arbitrage filters, implied forwards
black_scholes.py       # BSM pricing, vega, implied-vol inversion
ssvi.py                # SSVI fit, calendar-arbitrage, surface construction
dupire_diagnostics.py  # surface Dupire-compatibility checks
plotting.py            # surface / smile / fit plots
validation.py          # in-sample repricing validation
```

The other stage folders follow the same pattern (e.g. `pricing/` splits into
`variance_processes.py`, `simulation.py`, `payoffs.py`, `pricers.py`, …).

## Data

The pipeline input is `iv_surface/spx_raw_data.csv`, an OptionMetrics historical
SPX option-chain extract. The snapshot date is selected by `SNAPSHOT_DATE` in
`iv_surface/config.py` (default: latest date in the file); `r` and `q` are
module-level constants entered from the same OptionMetrics dataset.

All `arrays/`, `data/`, and `plots/` folders are **generated** by the pipeline
and are git-ignored — they are recreated when you run the stages.

## Usage

```bash
pip install -r requirements.txt          # Python 3.10+

# Run the whole pipeline (from the repo root)
python run_all.py
python run_all.py --from 3               # start at the Heston LSV stage
python run_all.py --only 1 3 5           # run selected stages
python run_all.py --dry-run              # print the plan only

# Run a single stage directly (always from the repo root)
python iv_surface/iv_surface_ssvi.py
python lsv/run_lsv.py --particles 10000

# Tests
cd iv_surface && python -m pytest test_iv_surface.py -v
```

Scripts must be launched from the repository root: Python adds the script's own
folder to `sys.path` (so sibling-module imports resolve) while the working
directory stays at the root (so the root-relative data paths are correct).

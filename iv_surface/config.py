# =============================================================================
# PARAMETERS
# =============================================================================
# ── Data source ──
# Path to the OptionMetrics raw extract.  Schema: standard OptionMetrics
# `secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
#  open_interest, impl_volatility, …`. Strikes are stored as
# `strike_price = strike_in_dollars * 1000` (OptionMetrics convention).
RAW_CSV_PATH = "iv_surface/spx_raw_data.csv"

# Snapshot date to filter the file on (YYYY-MM-DD).  Set to None to use
# the latest date present in the CSV.
SNAPSHOT_DATE = "2025-08-29"

# ── Market parameters for SNAPSHOT_DATE ──
# Continuously-compounded risk-free rate and dividend yield.
RISK_FREE_RATE = 0.0432   # e.g. 13-week T-bill secondary-market rate / 100
DIVIDEND_YIELD = 0.0125   # SPX TTM dividend yield

# ── Spot price ──
# If None, the spot is inferred from put-call parity at the shortest available
# expiry (median of `K + e^{rT}(C-P) - q-discount` across near-ATM pairs).
# Set to a float to override (e.g. 6460.26 for the SPX close on the snapshot
# date — same source as RISK_FREE_RATE / DIVIDEND_YIELD).
OVERRIDE_SPOT_PRICE = 6460.25

# ── Liquidity filters ──
# MIN_OPEN_INTEREST=100 retains the most actively traded contracts.
# MAX_BID_ASK_SPREAD_PCT=0.20 (20% of mid) is permissive enough for SPX
# (typical spread ≈ 2–5% of mid for liquid strikes) while removing stale or
# illiquid quotes.
MIN_OPEN_INTEREST = 100
MAX_BID_ASK_SPREAD_PCT = 0.20

# ── Moneyness and maturity bounds ──
# Moneyness bounds [0.70, 1.30] in K/S space (≈ ±26% log-moneyness) capture the
# liquid part of the smile; deep-OTM SPX options beyond this have negligible
# open interest and unreliable IVs.  TTM bounds: short end (avoid pin risk /
# gamma explosion) to 2 years.
MIN_MONEYNESS = 0.70
MAX_MONEYNESS = 1.30
MIN_TTM = 0.2 #7 / 365.0
MAX_TTM = 2.0

# ── Surface grid resolution ──
# 50 × 60 gives sufficient density for Dupire finite-difference derivatives
# (∂w/∂T needs ~50 T points, ∂²w/∂k² needs ~60 k points).
TTM_GRID_SIZE = 50
MONEYNESS_GRID_SIZE = 60

# ── Surface grid evaluation boundaries ──
# These are INDEPENDENT of the option data filters above.  The SSVI model is
# fit to observable options (moneyness 0.70–1.30) but evaluated on a wider
# k-grid so that MC paths wandering deep OTM/ITM are not clamped to the edge.
# Rule of thumb: cover ±3σ√T_max (σ≈0.20, T_max=2) → ±3×0.20×√2 ≈ ±0.85.
# Set None to fall back to data-derived bounds (2nd–98th percentile of k).
GRID_K_MIN = None   # forward log-moneyness lower bound for surface evaluation
GRID_K_MAX = None   # forward log-moneyness upper bound for surface evaluation
# TTM grid bounds — set None to use min/max of observed option TTMs.
GRID_T_MIN = None    # e.g. set 0.05 to extend below the shortest expiry
GRID_T_MAX = None    # e.g. set 2.5 to extend beyond the longest expiry

# ── SSVI fitting ──
# SSVI (Surface SVI, Gatheral & Jacquier 2014) fit jointly across all expiry
# slices.  Total-variance surface:
#
#   w(k, θ) = (θ/2) · [1 + ρφ(θ)k + √((φ(θ)k + ρ)² + 1 − ρ²)]
#
# with power-law  φ(θ) = η / (θ^γ · (1 + θ)^(1−γ)).  Surface parameters
# η > 0, γ ∈ (0, 0.5], ρ ∈ (−1, 1) are shared across maturities; θ(T) = ATM
# total variance is one value per expiry, monotone increasing (calendar-spread
# arb).  No-butterfly conditions (G&J 2014, Thm 4.2) are enforced during
# fitting via a penalty (weight SSVI_NB_PENALTY):
#   C1 = θ · φ(θ) · (1 + |ρ|) ≤ 4
#   C2 = θ · φ(θ)² · (1 + |ρ|) ≤ 4
# The fit is analytically smooth by construction, so no separate smoothing
# step is needed.
MIN_OPTIONS_PER_SLICE = 25     # need > 5 for a credible smile shape
SSVI_NB_PENALTY = 10.0        # dimensionless multiplier for no-butterfly penalty (adaptive scaling applied at runtime)

# ── Implied forward ──
# Per-expiry implied forwards are always computed from put-call parity. This is
# more accurate than assuming constant q because:
#   (a) SPX dividends are lumpy (concentrated in certain months).
#   (b) The borrowing rate embedded in options differs from the T-Bill rate.
#   (c) Put-call parity is model-free (holds under any no-arb dynamics).
# Pair near-ATM calls and puts (|K/F_approx - 1| < 10%) and take the median F
# across strikes.  The constant-q dividend yield is a fallback when too few
# call-put pairs exist.
NEAR_ATM_BAND = 0.10

# ── Option type selection ──
# True  → keep all calls (OTM and ITM); no puts.
# False → keep OTM only, forward-based (calls with K ≥ F, puts with K < F).
USE_CALLS_ONLY = False

# ── Validation ──
N_VALIDATION = 600

# ── Output directories ──
DIR_DATA   = "iv_surface/data"
DIR_PLOTS  = "iv_surface/plots"
DIR_ARRAYS = "iv_surface/arrays"

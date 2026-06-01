# Parameters

# ── Data source ──
# OptionMetrics raw extract (strike_price = strike_in_dollars * 1000).
RAW_CSV_PATH = "iv_surface/spx_raw_data.csv"

# Snapshot date (YYYY-MM-DD); None → latest date in CSV.
SNAPSHOT_DATE = "2025-08-29"

# ── Market parameters (continuously compounded) ──
RISK_FREE_RATE = 0.0432   # 13-week T-bill rate
DIVIDEND_YIELD = 0.0125   # SPX TTM dividend yield

# ── Spot price ──
# None → infer from put-call parity at shortest expiry; else override.
OVERRIDE_SPOT_PRICE = 6460.25

# ── Liquidity filters ──
# Spread cap = 20% of mid (SPX liquid spreads ≈ 2–5%), removing stale quotes.
MIN_OPEN_INTEREST = 100
MAX_BID_ASK_SPREAD_PCT = 0.20

# ── Moneyness and maturity bounds ──
# [0.70, 1.30] in K/S (≈ ±26% log-moneyness) covers the liquid smile.
MIN_MONEYNESS = 0.70
MAX_MONEYNESS = 1.30
MIN_TTM = 0.2 #7 / 365.0
MAX_TTM = 2.0

# ── Surface grid resolution ──
# 50×60 supports Dupire FD derivatives (∂w/∂T, ∂²w/∂k²).
TTM_GRID_SIZE = 50
MONEYNESS_GRID_SIZE = 60

# ── Surface grid evaluation boundaries ──
# Independent of data filters: fit on liquid options but evaluate on a wider
# k-grid (≈ ±3σ√T_max) so MC paths deep OTM/ITM aren't clamped.
# None → data-derived bounds (2nd–98th percentile of k / observed TTMs).
GRID_K_MIN = None   # forward log-moneyness lower bound for surface evaluation
GRID_K_MAX = None   # forward log-moneyness upper bound for surface evaluation
GRID_T_MIN = None    # e.g. set 0.05 to extend below the shortest expiry
GRID_T_MAX = None    # e.g. set 2.5 to extend beyond the longest expiry

# ── SSVI fitting (Gatheral & Jacquier 2014) ──
# Total variance w(k,θ) = (θ/2)·[1 + ρφ(θ)k + √((φ(θ)k+ρ)² + 1−ρ²)],
# power-law φ(θ) = η/(θ^γ·(1+θ)^(1−γ)). Surface params η>0, γ∈(0,0.5],
# ρ∈(−1,1) shared; θ(T)=ATM total variance, monotone (calendar-arb).
# No-butterfly (G&J Thm 4.2) enforced via penalty SSVI_NB_PENALTY:
#   C1 = θ·φ·(1+|ρ|) ≤ 4,  C2 = θ·φ²·(1+|ρ|) ≤ 4.
MIN_OPTIONS_PER_SLICE = 25     # need > 5 for a credible smile shape
SSVI_NB_PENALTY = 10.0        # no-butterfly penalty weight (adaptively scaled at runtime)

# ── Implied forward ──
# Per-expiry forward from put-call parity (model-free; handles lumpy SPX
# dividends and option-implied borrow). Median F over near-ATM call-put
# pairs (|K/F-1| < band); constant-q is fallback when too few pairs.
NEAR_ATM_BAND = 0.10

# ── Option type selection ──
# True → all calls; False → OTM only, forward-based (calls K≥F, puts K<F).
USE_CALLS_ONLY = False

# ── Validation ──
N_VALIDATION = 600

# ── Output directories ──
DIR_DATA   = "iv_surface/data"
DIR_PLOTS  = "iv_surface/plots"
DIR_ARRAYS = "iv_surface/arrays"

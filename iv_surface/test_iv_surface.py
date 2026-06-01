#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for iv_surface_ssvi.py
=================================
Tests the pure computational functions (BSM pricing, IV inversion, filtering,
surface interpolation, validation) without requiring the raw data extract.

Run:
    cd iv_surface
    python -m pytest test_iv_surface.py -v
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

# Import the module under test
import iv_surface_ssvi as iv


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def market_params():
    """Typical SPX market parameters."""
    return dict(S=5000.0, r=0.045, q=0.012)


@pytest.fixture
def sample_option_df(market_params):
    """A small synthetic option chain for filter/IV tests."""
    S = market_params["S"]
    strikes = np.array([4000, 4500, 4750, 5000, 5250, 5500, 6000])
    ttm = 0.5
    r, q = market_params["r"], market_params["q"]

    rows = []
    for K in strikes:
        moneyness = K / S
        # Generate realistic mid prices from BSM with known vol
        sigma = 0.20 + 0.10 * (1.0 - moneyness)  # slight skew
        sigma = max(sigma, 0.05)

        for opt_type in ["call", "put"]:
            mid = iv.bs_price(S, K, ttm, r, q, sigma, opt_type)
            if mid < 0.01:
                continue
            rows.append({
                "strike": K,
                "expiry": "2026-06-15",
                "ttm": ttm,
                "option_type": opt_type,
                "bid": mid * 0.98,
                "ask": mid * 1.02,
                "mid": mid,
                "openInterest": 500,
                "moneyness": moneyness,
                "iv_yf": sigma,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def iv_df_for_surface(market_params):
    """A larger synthetic IV dataset suitable for SSVI surface building.

    The current build_iv_surface() fits SSVI jointly across expiry slices and
    requires:
      - at least 3 expiry slices, each with >= MIN_OPTIONS_PER_SLICE options,
      - per-option ``iv`` (used to form total variance w = iv²·T),
      - forward log-moneyness, computed internally from a forward table.

    We therefore generate 5 expiries with 30 strikes each (well above the
    25-option threshold).  Option type is assigned relative to the per-expiry
    forward F(T) = S·e^{(r-q)T} so the chain is consistent with the
    forward-based OTM convention used downstream.  Expiry labels are unique per
    slice so each maps to its own forward.
    """
    S, r, q = market_params["S"], market_params["r"], market_params["q"]
    rng = np.random.default_rng(42)

    rows = []
    for i, ttm in enumerate(np.linspace(0.25, 1.5, 5)):
        F = S * np.exp((r - q) * ttm)
        for moneyness in np.linspace(0.80, 1.20, 30):
            K = S * moneyness
            # Synthetic IV: flat 20% + slight skew + slight term structure
            iv_val = 0.20 + 0.08 * (1.0 - moneyness) + 0.02 * np.sqrt(ttm)
            iv_val += rng.normal(0, 0.002)  # tiny noise
            iv_val = max(iv_val, 0.05)
            opt_type = "call" if K >= F else "put"
            rows.append({
                "strike": K,
                "expiry": f"2026-SLICE{i:02d}",
                "ttm": ttm,
                "option_type": opt_type,
                "moneyness": moneyness,
                "iv": iv_val,
                "mid": max(1.0, iv.bs_price(S, K, ttm, r, q, iv_val, opt_type)),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def fwd_df_for_surface(iv_df_for_surface, market_params):
    """Per-expiry forward curve matching ``iv_df_for_surface``.

    build_iv_surface() and validate_surface() both take a forward table
    (expiry, ttm, forward, q_eff) — the output schema of
    compute_implied_forwards().  The synthetic chain is generated from a
    constant (r, q), so the exact forward is F(T) = S·e^{(r-q)T} with
    q_eff = q; we build the table directly to keep the fixture deterministic.
    """
    S, r, q = market_params["S"], market_params["r"], market_params["q"]
    recs = []
    for expiry, grp in iv_df_for_surface.groupby("expiry"):
        ttm = float(grp["ttm"].iloc[0])
        recs.append({
            "expiry": expiry,
            "ttm": ttm,
            "forward": S * np.exp((r - q) * ttm),
            "q_eff": q,
            "n_pairs": len(grp),
        })
    return pd.DataFrame(recs).sort_values("ttm").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: BSM Price
# ══════════════════════════════════════════════════════════════════════════════

class TestBSPrice:
    """Tests for bs_price()."""

    def test_call_put_parity(self, market_params):
        """Call − Put = S·e^{-qT} − K·e^{-rT}."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, sigma = 5000.0, 0.5, 0.20

        call = iv.bs_price(S, K, T, r, q, sigma, "call")
        put = iv.bs_price(S, K, T, r, q, sigma, "put")
        parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        assert call - put == pytest.approx(parity_rhs, rel=1e-10)

    def test_atm_call_brenner_subrahmanyam(self):
        """ATM call ≈ S·σ·√T / √(2π) when r=q (Brenner-Subrahmanyam approx)."""
        S, T, sigma = 5000.0, 1.0, 0.20
        r = q = 0.03  # r=q so the drift term vanishes

        call = iv.bs_price(S, S, T, r, q, sigma, "call")
        approx = S * sigma * np.sqrt(T) / np.sqrt(2 * np.pi) * np.exp(-q * T)
        # Approximation is rough — within 5%
        assert call == pytest.approx(approx, rel=0.05)

    def test_deep_itm_call_near_intrinsic(self, market_params):
        """Deep ITM call should be close to S·e^{-qT} − K·e^{-rT}."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, sigma = 3000.0, 0.5, 0.20

        call = iv.bs_price(S, K, T, r, q, sigma, "call")
        intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert call == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_call_near_zero(self, market_params):
        """Deep OTM call should be near zero."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        call = iv.bs_price(S, 8000.0, 0.1, r, q, 0.15, "call")
        assert call < 0.01

    def test_call_positive(self, market_params):
        """Call price must be non-negative."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        for K in [3000, 4000, 5000, 6000, 7000]:
            for T in [0.01, 0.1, 0.5, 1.0, 2.0]:
                for sigma in [0.05, 0.20, 0.50, 1.0]:
                    price = iv.bs_price(S, K, T, r, q, sigma, "call")
                    assert price >= 0.0

    def test_put_positive(self, market_params):
        """Put price must be non-negative."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        for K in [3000, 4000, 5000, 6000, 7000]:
            for T in [0.01, 0.1, 0.5, 1.0, 2.0]:
                for sigma in [0.05, 0.20, 0.50, 1.0]:
                    price = iv.bs_price(S, K, T, r, q, sigma, "put")
                    assert price >= 0.0

    def test_zero_vol_returns_intrinsic(self, market_params):
        """sigma=0 should return intrinsic value."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T = 4500.0, 0.5
        call = iv.bs_price(S, K, T, r, q, 0.0, "call")
        expected = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        assert call == pytest.approx(expected, abs=1e-10)

    def test_zero_ttm_returns_intrinsic(self, market_params):
        """T=0 should return intrinsic value."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        call = iv.bs_price(S, 4800.0, 0.0, r, q, 0.20, "call")
        # At T=0 with the formula: max(0, S*exp(0) - K*exp(0)) = max(0, S-K)
        expected = max(0.0, S - 4800.0)
        assert call == pytest.approx(expected, abs=1e-8)

    def test_call_increases_with_vol(self, market_params):
        """Call price should increase with volatility (positive vega)."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T = 5000.0, 0.5
        p1 = iv.bs_price(S, K, T, r, q, 0.15, "call")
        p2 = iv.bs_price(S, K, T, r, q, 0.25, "call")
        assert p2 > p1

    def test_put_increases_with_vol(self, market_params):
        """Put price should increase with volatility."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T = 5000.0, 0.5
        p1 = iv.bs_price(S, K, T, r, q, 0.15, "put")
        p2 = iv.bs_price(S, K, T, r, q, 0.25, "put")
        assert p2 > p1

    def test_known_value(self):
        """Verify against a hand-computed BSM value."""
        # S=100, K=100, T=1, r=0.05, q=0, sigma=0.2
        # d1 = (ln(1) + (0.05 + 0.02)*1) / 0.2 = 0.35
        # d2 = 0.35 - 0.2 = 0.15
        # C = 100*N(0.35) - 100*exp(-0.05)*N(0.15)
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.20
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        expected = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        result = iv.bs_price(S, K, T, r, q, sigma, "call")
        assert result == pytest.approx(expected, rel=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: BSM Vega
# ══════════════════════════════════════════════════════════════════════════════

class TestBSVega:
    """Tests for bs_vega()."""

    def test_vega_positive(self, market_params):
        """Vega must be positive for T > 0, sigma > 0."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        for K in [4000, 5000, 6000]:
            for T in [0.1, 0.5, 1.0]:
                v = iv.bs_vega(S, K, T, r, q, 0.20)
                assert v > 0

    def test_vega_peaks_near_atm(self, market_params):
        """Vega should be largest near ATM."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        T, sigma = 0.5, 0.20
        v_atm = iv.bs_vega(S, S, T, r, q, sigma)
        v_otm = iv.bs_vega(S, S * 1.2, T, r, q, sigma)
        v_itm = iv.bs_vega(S, S * 0.8, T, r, q, sigma)
        assert v_atm > v_otm
        assert v_atm > v_itm

    def test_vega_zero_when_expired(self, market_params):
        """Vega should be near zero at T=0."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        v = iv.bs_vega(S, S, 0.0, r, q, 0.20)
        assert v < 1e-10

    def test_vega_numerical_derivative(self, market_params):
        """Vega should match finite-difference ∂C/∂σ."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, sigma = 5000.0, 0.5, 0.20
        ds = 1e-5
        c_up = iv.bs_price(S, K, T, r, q, sigma + ds, "call")
        c_dn = iv.bs_price(S, K, T, r, q, sigma - ds, "call")
        numerical_vega = (c_up - c_dn) / (2 * ds)
        analytical_vega = iv.bs_vega(S, K, T, r, q, sigma)
        assert analytical_vega == pytest.approx(numerical_vega, rel=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Implied Vol Inversion
# ══════════════════════════════════════════════════════════════════════════════

class TestImpliedVol:
    """Tests for implied_vol_single()."""

    def test_roundtrip_call(self, market_params):
        """BSM price → IV inversion should recover the original vol."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, true_vol = 5000.0, 0.5, 0.22
        price = iv.bs_price(S, K, T, r, q, true_vol, "call")
        recovered = iv.implied_vol_single(price, S, K, T, r, q, "call")
        assert recovered == pytest.approx(true_vol, abs=1e-5)

    def test_roundtrip_put(self, market_params):
        """Put IV inversion roundtrip."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, true_vol = 5200.0, 0.8, 0.18
        price = iv.bs_price(S, K, T, r, q, true_vol, "put")
        recovered = iv.implied_vol_single(price, S, K, T, r, q, "put")
        assert recovered == pytest.approx(true_vol, abs=1e-5)

    def test_roundtrip_deep_otm_call(self, market_params):
        """Deep OTM call IV roundtrip (tests Brent fallback)."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, true_vol = 6500.0, 0.3, 0.25
        price = iv.bs_price(S, K, T, r, q, true_vol, "call")
        if price > 0.01:  # only test if price is non-trivial
            recovered = iv.implied_vol_single(price, S, K, T, r, q, "call")
            assert recovered == pytest.approx(true_vol, abs=1e-4)

    def test_roundtrip_various_vols(self, market_params):
        """Roundtrip across a range of volatilities."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T = 5000.0, 0.5
        for true_vol in [0.05, 0.10, 0.20, 0.40, 0.80, 1.50]:
            price = iv.bs_price(S, K, T, r, q, true_vol, "call")
            recovered = iv.implied_vol_single(price, S, K, T, r, q, "call")
            assert recovered == pytest.approx(true_vol, abs=1e-4), \
                f"Failed at true_vol={true_vol}"

    def test_nan_for_below_intrinsic(self, market_params):
        """Price below intrinsic should return NaN."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T = 4500.0, 0.5
        intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        result = iv.implied_vol_single(intrinsic * 0.5, S, K, T, r, q, "call")
        assert np.isnan(result)

    def test_nan_for_negative_price(self, market_params):
        """Negative price should return NaN."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        result = iv.implied_vol_single(-1.0, S, 5000.0, 0.5, r, q, "call")
        assert np.isnan(result)

    def test_high_vol_option(self, market_params):
        """Should handle very high vol (e.g. meme stock scenarios)."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        K, T, true_vol = 5000.0, 0.5, 2.5
        price = iv.bs_price(S, K, T, r, q, true_vol, "call")
        recovered = iv.implied_vol_single(price, S, K, T, r, q, "call")
        assert recovered == pytest.approx(true_vol, abs=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestFiltering:
    """Tests for filter_liquidity, filter_no_arbitrage, filter_option_type_forward."""

    def test_liquidity_removes_low_oi(self, sample_option_df):
        """Options with low OI should be removed."""
        df = sample_option_df.copy()
        df.loc[0, "openInterest"] = 10  # below MIN_OPEN_INTEREST (100)
        result = iv.filter_liquidity(df)
        assert len(result) < len(df)
        # The row with OI=10 should be gone
        assert 10 not in result["openInterest"].values

    def test_liquidity_removes_wide_spread(self, sample_option_df):
        """Options with wide bid-ask should be removed."""
        df = sample_option_df.copy()
        # Make one option have a 50% spread
        idx = df.index[0]
        mid = df.loc[idx, "mid"]
        df.loc[idx, "bid"] = mid * 0.70
        df.loc[idx, "ask"] = mid * 1.30
        result = iv.filter_liquidity(df)
        assert len(result) <= len(df)

    def test_liquidity_removes_out_of_moneyness_range(self, sample_option_df):
        """Options outside [MIN_MONEYNESS, MAX_MONEYNESS] should be removed."""
        df = sample_option_df.copy()
        # Add an option with extreme moneyness
        extreme = df.iloc[0:1].copy()
        extreme["moneyness"] = 2.0  # way outside [0.70, 1.30]
        extreme["strike"] = 10000.0
        df = pd.concat([df, extreme], ignore_index=True)
        result = iv.filter_liquidity(df)
        assert all(result["moneyness"] >= iv.MIN_MONEYNESS)
        assert all(result["moneyness"] <= iv.MAX_MONEYNESS)

    def test_no_arb_removes_violating_calls(self, market_params):
        """Calls priced above the forward should be removed."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        T = 0.5
        K = 5000.0
        disc_S = S * np.exp(-q * T)

        df = pd.DataFrame([{
            "strike": K, "ttm": T, "option_type": "call",
            "mid": disc_S * 1.5,  # violates upper bound
            "moneyness": K / S,
        }])
        result = iv.filter_no_arbitrage(df, S, r, q)
        assert len(result) == 0

    def test_no_arb_keeps_valid_options(self, market_params, sample_option_df):
        """Valid BSM-priced options should all pass no-arb filter."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        result = iv.filter_no_arbitrage(sample_option_df, S, r, q)
        # All synthetic options are generated from BSM, so they should pass
        assert len(result) == len(sample_option_df)

    def test_filter_otm(self, sample_option_df, market_params):
        """OTM filter (forward-based): keep calls with K>=F(T), puts with K<F(T).

        The current API filters relative to the per-expiry forward F(T)
        (from put-call parity), not the spot S.  We build the forward table
        with compute_implied_forwards() and assert that every kept option is
        OTM on the forward.
        """
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        fwd_df = iv.compute_implied_forwards(sample_option_df, S, r, q_fallback=q)
        fwd_map = dict(zip(fwd_df["expiry"], fwd_df["forward"]))

        result = iv.filter_option_type_forward(sample_option_df, fwd_df)
        assert len(result) > 0
        for _, row in result.iterrows():
            F = fwd_map[row["expiry"]]
            if row["option_type"] == "call":
                assert row["strike"] >= F
            else:
                assert row["strike"] < F

    def test_filter_call_only(self, sample_option_df, market_params):
        """calls_only=True should keep only calls (OTM + ITM)."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        fwd_df = iv.compute_implied_forwards(sample_option_df, S, r, q_fallback=q)
        result = iv.filter_option_type_forward(sample_option_df, fwd_df, calls_only=True)
        assert all(result["option_type"] == "call")
        # Every call in the chain is retained (no moneyness restriction).
        n_calls = (sample_option_df["option_type"] == "call").sum()
        assert len(result) == n_calls

    def test_filter_otm_keeps_otm_puts(self, sample_option_df, market_params):
        """OTM filter retains the OTM puts (K < F) and drops ITM puts (K >= F)."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        fwd_df = iv.compute_implied_forwards(sample_option_df, S, r, q_fallback=q)
        F = fwd_df["forward"].iloc[0]
        result = iv.filter_option_type_forward(sample_option_df, fwd_df)
        kept_puts = result[result["option_type"] == "put"]
        # All kept puts are strictly OTM (K < F)...
        assert all(kept_puts["strike"] < F)
        # ...and they are exactly the OTM puts from the original chain.
        otm_puts = sample_option_df[
            (sample_option_df["option_type"] == "put")
            & (sample_option_df["strike"] < F)
        ]
        assert len(kept_puts) == len(otm_puts)

    def test_filter_otm_union_is_all_otm(self, sample_option_df, market_params):
        """OTM (calls K>=F) ∪ (puts K<F) partitions the chain by the forward.

        The forward-based default filter keeps exactly the OTM half of the
        chain: OTM calls plus OTM puts.  This replaces the old "all" mode,
        which kept every row regardless of moneyness.
        """
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        fwd_df = iv.compute_implied_forwards(sample_option_df, S, r, q_fallback=q)
        F = fwd_df["forward"].iloc[0]
        result = iv.filter_option_type_forward(sample_option_df, fwd_df)

        expected = sample_option_df[
            ((sample_option_df["option_type"] == "call")
             & (sample_option_df["strike"] >= F))
            | ((sample_option_df["option_type"] == "put")
               & (sample_option_df["strike"] < F))
        ]
        assert len(result) == len(expected)

    def test_filter_missing_forward_drops_option(self, sample_option_df, market_params):
        """Options whose expiry is absent from fwd_df get no forward → dropped.

        The forward-based filter maps each row's expiry to F via fwd_df.  An
        expiry missing from fwd_df yields NaN forward, so the OTM comparison is
        False and the option is excluded.  (The old API raised ValueError on an
        unknown string mode; the forward-based API has no string mode at all,
        so the equivalent "unrecognised input" behaviour is silent exclusion.)
        """
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        fwd_df = iv.compute_implied_forwards(sample_option_df, S, r, q_fallback=q)
        # Forward table that does not cover the option's expiry.
        bad_fwd = fwd_df.copy()
        bad_fwd["expiry"] = "1999-01-01"
        result = iv.filter_option_type_forward(sample_option_df, bad_fwd)
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: compute_implied_vols
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeImpliedVols:
    """Tests for compute_implied_vols()."""

    def test_recovers_known_vols(self, market_params, sample_option_df):
        """IV computation on BSM-priced options should recover the input vols."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        result = iv.compute_implied_vols(sample_option_df, S, r, q)
        assert "iv" in result.columns
        assert len(result) > 0
        # All recovered IVs should be in a reasonable range
        assert all(result["iv"] >= 0.01)
        assert all(result["iv"] <= 3.0)

    def test_iv_range_filtering(self, market_params):
        """IVs outside [0.01, 3.0] should be removed."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        # Create an option whose "mid" is so close to intrinsic that IV → 0
        K, T = 4000.0, 0.5
        intrinsic = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        df = pd.DataFrame([{
            "strike": K, "expiry": "2026-06-15", "ttm": T,
            "option_type": "call", "mid": intrinsic + 0.001,
            "moneyness": K / S,
        }])
        result = iv.compute_implied_vols(df, S, r, q)
        # Should either have a very small IV or be filtered out
        if len(result) > 0:
            assert all(result["iv"] >= 0.01)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Surface Building
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildIVSurface:
    """Tests for build_iv_surface()."""

    # NOTE: build_iv_surface() now returns a 5-tuple
    #   (ttm_grid, log_m_grid, iv_surface, total_var_surface, ssvi_params_df)
    # and requires a forward table (fwd_df) to map each option to its forward
    # log-moneyness k = ln(K/F(T)).  The surface is built via a joint SSVI fit.

    def test_output_shapes(self, iv_df_for_surface, fwd_df_for_surface):
        """Surface arrays should have correct shapes."""
        ttm_grid, log_m_grid, surface, total_var, params = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid_size=20, moneyness_grid_size=25
        )
        assert ttm_grid.shape == (20,)
        assert log_m_grid.shape == (25,)
        assert surface.shape == (25, 20)
        assert total_var.shape == (25, 20)

    def test_surface_values_positive(self, iv_df_for_surface, fwd_df_for_surface):
        """All surface IV values should be positive (clipped to >= 0.01)."""
        _, _, surface, _, _ = iv.build_iv_surface(iv_df_for_surface, fwd_df_for_surface)
        assert np.all(surface >= 0.01)
        assert np.all(surface <= 3.0)

    def test_surface_values_finite(self, iv_df_for_surface, fwd_df_for_surface):
        """Surface should contain no NaN or Inf."""
        _, _, surface, _, _ = iv.build_iv_surface(iv_df_for_surface, fwd_df_for_surface)
        assert np.all(np.isfinite(surface))

    def test_grid_monotonic(self, iv_df_for_surface, fwd_df_for_surface):
        """Grid arrays should be strictly increasing."""
        ttm_grid, log_m_grid, _, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface)
        assert np.all(np.diff(ttm_grid) > 0)
        assert np.all(np.diff(log_m_grid) > 0)

    def test_surface_contains_atm(self, iv_df_for_surface, fwd_df_for_surface):
        """The forward-log-moneyness grid should span ATM (contain 0)."""
        _, log_m_grid, _, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface)
        assert log_m_grid.min() < 0 < log_m_grid.max()

    def test_atm_iv_reasonable(self, iv_df_for_surface, fwd_df_for_surface):
        """ATM IV at the surface should be close to input ATM IV (~0.20)."""
        ttm_grid, log_m_grid, surface, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface)
        # Find grid index closest to log_m = 0 (ATM)
        atm_idx = np.argmin(np.abs(log_m_grid))
        mid_t_idx = len(ttm_grid) // 2
        atm_iv = surface[atm_idx, mid_t_idx]
        # Synthetic data has ATM IV around 0.20-0.22
        assert 0.15 < atm_iv < 0.30

    def test_surface_has_skew(self, iv_df_for_surface, fwd_df_for_surface):
        """OTM puts (negative log-moneyness) should have higher IV than OTM calls."""
        ttm_grid, log_m_grid, surface, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface)
        mid_t = len(ttm_grid) // 2
        # Average IV for negative log-m vs positive log-m
        neg_mask = log_m_grid < -0.05
        pos_mask = log_m_grid > 0.05
        if neg_mask.any() and pos_mask.any():
            avg_otm_put_iv = surface[neg_mask, mid_t].mean()
            avg_otm_call_iv = surface[pos_mask, mid_t].mean()
            # Our synthetic data has a negative skew built in
            assert avg_otm_put_iv > avg_otm_call_iv


# ══════════════════════════════════════════════════════════════════════════════
# Section 4b: Implied Forwards
# ══════════════════════════════════════════════════════════════════════════════

class TestImpliedForwards:
    """Tests for compute_implied_forwards()."""

    def test_implied_forward_from_synthetic_parity(self, market_params):
        """Forward from BSM call-put pairs should recover F = S·exp((r-q)T)."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        T = 0.5
        F_true = S * np.exp((r - q) * T)
        sigma = 0.20

        rows = []
        for K in np.linspace(0.95 * S, 1.05 * S, 10):
            call_mid = iv.bs_price(S, K, T, r, q, sigma, "call")
            put_mid = iv.bs_price(S, K, T, r, q, sigma, "put")
            for opt_type, mid in [("call", call_mid), ("put", put_mid)]:
                rows.append({
                    "strike": K, "expiry": "2026-06-15", "ttm": T,
                    "option_type": opt_type, "mid": mid,
                    "moneyness": K / S,
                })

        df = pd.DataFrame(rows)
        # compute_implied_forwards now takes an explicit q_fallback (used only
        # when too few near-ATM call/put pairs exist; here all 10 strikes are
        # within NEAR_ATM_BAND, so the forward is recovered from parity).
        fwd_df = iv.compute_implied_forwards(df, S, r, q_fallback=q)
        assert len(fwd_df) == 1
        assert fwd_df["forward"].iloc[0] == pytest.approx(F_true, rel=1e-6)

    def test_implied_q_eff(self, market_params):
        """Effective dividend yield should be close to the true q."""
        S, r, q = market_params["S"], market_params["r"], market_params["q"]
        T = 0.5
        sigma = 0.20

        rows = []
        for K in np.linspace(0.92 * S, 1.08 * S, 15):
            call_mid = iv.bs_price(S, K, T, r, q, sigma, "call")
            put_mid = iv.bs_price(S, K, T, r, q, sigma, "put")
            for opt_type, mid in [("call", call_mid), ("put", put_mid)]:
                rows.append({
                    "strike": K, "expiry": "2026-06-15", "ttm": T,
                    "option_type": opt_type, "mid": mid,
                    "moneyness": K / S,
                })

        df = pd.DataFrame(rows)
        fwd_df = iv.compute_implied_forwards(df, S, r, q_fallback=q)
        assert fwd_df["q_eff"].iloc[0] == pytest.approx(q, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    """Tests for validate_surface()."""

    # NOTE: validate_surface() now takes the forward table (fwd_df) so it can
    # interpolate the surface at each option's forward log-moneyness
    # k = ln(K/F(T)) and reprice with the per-expiry q_eff.  Its signature is
    # validate_surface(df, fwd_df, ttm_grid, log_m_grid, iv_surface, S, r,
    #                  n_sample=...)  — q is sourced from fwd_df's q_eff, not
    # passed separately.

    def test_validation_output_columns(self, iv_df_for_surface,
                                       fwd_df_for_surface, market_params):
        """Validation should return expected columns."""
        S, r = market_params["S"], market_params["r"]
        ttm_grid, log_m_grid, surface, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid_size=20, moneyness_grid_size=25
        )
        val_df = iv.validate_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid, log_m_grid, surface, S, r,
            n_sample=10
        )
        expected_cols = {"strike", "expiry", "option_type", "market_price",
                         "iv_computed", "iv_interpolated", "iv_abs_err",
                         "repriced", "error_pct"}
        assert expected_cols.issubset(set(val_df.columns))

    def test_validation_iv_errors_small(self, iv_df_for_surface,
                                        fwd_df_for_surface, market_params):
        """IV interpolation errors should be small for smooth synthetic data."""
        S, r = market_params["S"], market_params["r"]
        ttm_grid, log_m_grid, surface, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid_size=30, moneyness_grid_size=40
        )
        val_df = iv.validate_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid, log_m_grid, surface, S, r,
            n_sample=30
        )
        # For smooth synthetic data, IV MAE should be under 2 vol pts
        iv_mae = val_df["iv_abs_err"].mean()
        assert iv_mae < 0.02, f"IV MAE too large: {iv_mae:.4f}"

    def test_validation_sample_size(self, iv_df_for_surface,
                                    fwd_df_for_surface, market_params):
        """Should respect the n_sample parameter."""
        S, r = market_params["S"], market_params["r"]
        ttm_grid, log_m_grid, surface, _, _ = iv.build_iv_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid_size=20, moneyness_grid_size=25
        )
        val_df = iv.validate_surface(
            iv_df_for_surface, fwd_df_for_surface,
            ttm_grid, log_m_grid, surface, S, r,
            n_sample=5
        )
        assert len(val_df) == 5


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases and integration
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_bs_price_symmetry(self):
        """Call with S=K should equal put with S=K when r=q."""
        S = K = 100.0
        T, sigma = 0.5, 0.20
        r = q = 0.03
        call = iv.bs_price(S, K, T, r, q, sigma, "call")
        put = iv.bs_price(S, K, T, r, q, sigma, "put")
        assert call == pytest.approx(put, rel=1e-10)

    def test_iv_inversion_near_boundary_vol(self):
        """IV inversion should work near the low end of vol (sigma ~ 0.05)."""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
        true_vol = 0.05
        price = iv.bs_price(S, K, T, r, q, true_vol, "call")
        recovered = iv.implied_vol_single(price, S, K, T, r, q, "call")
        assert recovered == pytest.approx(true_vol, abs=1e-4)

    def test_bs_price_high_rate(self):
        """BSM should still work with high interest rates."""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.20, 0.0, 0.30
        price = iv.bs_price(S, K, T, r, q, sigma, "call")
        assert price > 0
        assert np.isfinite(price)

    def test_bs_price_long_maturity(self):
        """BSM should handle long maturities."""
        S, K, T, r, q, sigma = 5000.0, 5000.0, 5.0, 0.04, 0.01, 0.20
        call = iv.bs_price(S, K, T, r, q, sigma, "call")
        put = iv.bs_price(S, K, T, r, q, sigma, "put")
        assert call > 0
        assert put > 0
        # Parity should still hold
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, rel=1e-8)

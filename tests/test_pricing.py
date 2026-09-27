"""Cross-validation tests: BS vs binomial vs Monte Carlo, parity, Greeks, IV."""

import math

import pytest

from optpricing import black_scholes as bs
from optpricing.binomial import binomial_price
from optpricing.greeks_fd import (
    default_pricer, fd_delta, fd_gamma, fd_rho, fd_theta, fd_vega,
)
from optpricing.implied_vol import implied_vol
from optpricing.monte_carlo import mc_price

# A representative set of parameters.
S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


# ----------------------------- Black-Scholes ----------------------------- #
def test_bs_known_atm_call():
    # Textbook value for these parameters ~ 10.4506.
    assert bs.bs_price(S, K, T, r, sigma, q, "call") == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity():
    call = bs.bs_price(S, K, T, r, sigma, q, "call")
    put = bs.bs_price(S, K, T, r, sigma, q, "put")
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_bs_intrinsic_at_expiry():
    assert bs.bs_price(120, 100, 0, r, sigma, q, "call") == pytest.approx(20.0)
    assert bs.bs_price(80, 100, 0, r, sigma, q, "put") == pytest.approx(20.0)


# ------------------------ Convergence: BS vs tree ------------------------ #
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_bs(option_type):
    bs_val = bs.bs_price(S, K, T, r, sigma, q, option_type)
    tree = binomial_price(S, K, T, r, sigma, q, option_type, american=False, steps=2000)
    assert tree == pytest.approx(bs_val, abs=2e-2)


def test_american_call_no_dividend_equals_european():
    # American call on a non-dividend stock = European call.
    eu = binomial_price(S, K, T, r, sigma, 0.0, "call", american=False, steps=1000)
    am = binomial_price(S, K, T, r, sigma, 0.0, "call", american=True, steps=1000)
    assert am == pytest.approx(eu, abs=1e-6)


def test_american_put_premium_positive():
    # American put should be worth at least the European put.
    eu = binomial_price(S, K, T, r, sigma, q, "put", american=False, steps=1000)
    am = binomial_price(S, K, T, r, sigma, q, "put", american=True, steps=1000)
    assert am >= eu - 1e-9
    assert am > eu  # early-exercise premium is strictly positive here


# ------------------------ Convergence: BS vs MC ------------------------- #
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_converges_to_bs(option_type):
    bs_val = bs.bs_price(S, K, T, r, sigma, q, option_type)
    mc = mc_price(S, K, T, r, sigma, q, option_type, n_paths=200_000, seed=42)
    # Within ~3 standard errors of the analytic price.
    assert abs(mc.price - bs_val) < 4 * mc.std_error + 1e-2


def test_monte_carlo_variance_reduction_helps():
    plain = mc_price(S, K, T, r, sigma, q, "call", n_paths=50_000,
                     antithetic=False, control_variate=False, seed=7)
    reduced = mc_price(S, K, T, r, sigma, q, "call", n_paths=50_000,
                       antithetic=True, control_variate=True, seed=7)
    assert reduced.std_error < plain.std_error


# ------------------------------- Greeks -------------------------------- #
def test_analytic_greeks_match_finite_difference():
    g = bs.all_greeks(S, K, T, r, sigma, q, "call")
    assert g.delta == pytest.approx(fd_delta(default_pricer, S, K, T, r, sigma, q, "call"), abs=1e-4)
    assert g.gamma == pytest.approx(fd_gamma(default_pricer, S, K, T, r, sigma, q, "call"), abs=1e-4)
    assert g.vega == pytest.approx(fd_vega(default_pricer, S, K, T, r, sigma, q, "call"), abs=1e-2)
    assert g.theta == pytest.approx(fd_theta(default_pricer, S, K, T, r, sigma, q, "call"), abs=1e-2)
    assert g.rho == pytest.approx(fd_rho(default_pricer, S, K, T, r, sigma, q, "call"), abs=1e-2)


def test_call_delta_bounds():
    assert 0.0 <= bs.delta(S, K, T, r, sigma, q, "call") <= 1.0
    assert -1.0 <= bs.delta(S, K, T, r, sigma, q, "put") <= 0.0


def test_gamma_positive():
    assert bs.gamma(S, K, T, r, sigma, q) > 0


# ----------------------------- Implied vol ----------------------------- #
@pytest.mark.parametrize("true_vol", [0.05, 0.15, 0.30, 0.60])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_roundtrip(true_vol, option_type):
    price = bs.bs_price(S, K, T, r, true_vol, q, option_type)
    iv = implied_vol(price, S, K, T, r, q, option_type)
    assert iv == pytest.approx(true_vol, abs=1e-5)


def test_implied_vol_out_of_bounds_returns_nan():
    # A price above the no-arb upper bound has no solution.
    iv = implied_vol(1e6, S, K, T, r, q, "call")
    assert math.isnan(iv)

"""Finite-difference Greeks.

These compute Greeks by bumping the relevant input and repricing, using a
central-difference scheme for accuracy. They serve as an independent check on
the analytic Greeks in :mod:`optpricing.black_scholes`, and they work for *any*
pricer (binomial, Monte Carlo) — not just closed-form models.
"""

from __future__ import annotations

from typing import Callable

from . import black_scholes as bs


def fd_delta(pricer: Callable, S, K, T, r, sigma, q=0.0, option_type="call", h=1e-2):
    """Central-difference delta: (V(S+h) - V(S-h)) / 2h."""
    up = pricer(S + h, K, T, r, sigma, q, option_type)
    dn = pricer(S - h, K, T, r, sigma, q, option_type)
    return (up - dn) / (2.0 * h)


def fd_gamma(pricer: Callable, S, K, T, r, sigma, q=0.0, option_type="call", h=1e-2):
    """Central-difference gamma: (V(S+h) - 2V(S) + V(S-h)) / h^2."""
    up = pricer(S + h, K, T, r, sigma, q, option_type)
    mid = pricer(S, K, T, r, sigma, q, option_type)
    dn = pricer(S - h, K, T, r, sigma, q, option_type)
    return (up - 2.0 * mid + dn) / (h ** 2)


def fd_vega(pricer: Callable, S, K, T, r, sigma, q=0.0, option_type="call", h=1e-4):
    """Central-difference vega w.r.t. sigma (per 1.00 vol change)."""
    up = pricer(S, K, T, r, sigma + h, q, option_type)
    dn = pricer(S, K, T, r, sigma - h, q, option_type)
    return (up - dn) / (2.0 * h)


def fd_theta(pricer: Callable, S, K, T, r, sigma, q=0.0, option_type="call", h=1e-4):
    """Theta by bumping time to expiry (per year, sign matches analytic)."""
    up = pricer(S, K, T + h, r, sigma, q, option_type)
    dn = pricer(S, K, T - h, r, sigma, q, option_type)
    return -(up - dn) / (2.0 * h)


def fd_rho(pricer: Callable, S, K, T, r, sigma, q=0.0, option_type="call", h=1e-4):
    """Central-difference rho w.r.t. r (per 1.00 rate change)."""
    up = pricer(S, K, T, r + h, sigma, q, option_type)
    dn = pricer(S, K, T, r - h, sigma, q, option_type)
    return (up - dn) / (2.0 * h)


def default_pricer(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Convenience wrapper so finite-difference funcs can default to BS."""
    return bs.bs_price(S, K, T, r, sigma, q, option_type)

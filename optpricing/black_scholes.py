r"""Black-Scholes-Merton analytic pricing and Greeks.

The Black-Scholes-Merton model prices a European option on an asset paying a
continuous dividend yield :math:`q` under geometric Brownian motion:

.. math::

    dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t

The closed-form price of a European call is

.. math::

    C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)

.. math::

    d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}},
    \qquad d_2 = d_1 - \sigma\sqrt{T}

with the put obtained via put-call parity. All Greeks below are the analytic
partial derivatives of this formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    """Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    S, K : float
        Spot and strike.
    T : float
        Time to expiry in years.
    r : float
        Continuously-compounded risk-free rate.
    sigma : float
        Volatility (annualised).
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    """
    option_type = option_type.lower()
    # Handle the degenerate expiry case with the intrinsic value.
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def delta(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    """Sensitivity of price to the underlying (dV/dS)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0) -> float:
    """Second derivative of price w.r.t. spot (same for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0) -> float:
    """Sensitivity to volatility, per 1.00 (100 vol points) change.

    Divide by 100 for the change per 1 vol point.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * math.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    """Time decay per year (divide by 365 for per-calendar-day)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    term1 = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if option_type == "call":
        return (
            term1
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
    return (
        term1
        + r * K * disc_r * norm.cdf(-d2)
        - q * S * disc_q * norm.cdf(-d1)
    )


def rho(S, K, T, r, sigma, q=0.0, option_type="call") -> float:
    """Sensitivity to the risk-free rate, per 1.00 (100 bps = 0.01) change."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    if option_type == "call":
        return K * T * disc_r * norm.cdf(d2)
    return -K * T * disc_r * norm.cdf(-d2)


@dataclass
class Greeks:
    """Container for the full set of option Greeks."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def all_greeks(S, K, T, r, sigma, q=0.0, option_type="call") -> Greeks:
    """Compute price and all analytic Greeks at once."""
    return Greeks(
        price=bs_price(S, K, T, r, sigma, q, option_type),
        delta=delta(S, K, T, r, sigma, q, option_type),
        gamma=gamma(S, K, T, r, sigma, q),
        vega=vega(S, K, T, r, sigma, q),
        theta=theta(S, K, T, r, sigma, q, option_type),
        rho=rho(S, K, T, r, sigma, q, option_type),
    )

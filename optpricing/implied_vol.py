r"""Implied-volatility solver.

Given a market option price, the implied volatility is the :math:`\sigma` that
makes the Black-Scholes price match the market:

.. math:: \text{BS}(S, K, T, r, \sigma_{\text{imp}}, q) = V_{\text{market}}.

Because vega is strictly positive, the map :math:`\sigma \mapsto \text{BS}` is
monotone increasing, so the root is unique when the price is arbitrage-free.

The solver uses **Newton-Raphson** (quadratic convergence using the analytic
vega) and falls back to **Brent's method** (bracketing, guaranteed convergence)
if Newton steps leave the valid region or fail to converge.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .black_scholes import bs_price, vega


def _price_bounds(S, K, T, r, q, option_type):
    """No-arbitrage price bounds for the option."""
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_vol(
    price,
    S,
    K,
    T,
    r,
    q=0.0,
    option_type="call",
    tol=1e-8,
    max_iter=100,
    initial_guess=0.2,
):
    """Solve for Black-Scholes implied volatility.

    Returns
    -------
    float
        The implied volatility, or ``nan`` if the price violates no-arbitrage
        bounds (no solution exists).
    """
    option_type = option_type.lower()
    lower, upper = _price_bounds(S, K, T, r, q, option_type)
    # Allow a tiny tolerance at the boundaries for floating point.
    if price < lower - 1e-10 or price > upper + 1e-10:
        return float("nan")

    # --- Newton-Raphson ---
    sigma = max(initial_guess, 1e-4)
    for _ in range(max_iter):
        model = bs_price(S, K, T, r, sigma, q, option_type)
        diff = model - price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma, q)
        if v < 1e-10:
            break  # vega too small -> fall back to Brent
        sigma_new = sigma - diff / v
        if sigma_new <= 0 or sigma_new > 10 or math.isnan(sigma_new):
            break
        sigma = sigma_new

    # --- Brent fallback (robust bracketing) ---
    def objective(vol):
        return bs_price(S, K, T, r, vol, q, option_type) - price

    lo, hi = 1e-6, 10.0
    try:
        if objective(lo) * objective(hi) > 0:
            return float("nan")
        return brentq(objective, lo, hi, xtol=tol, maxiter=max_iter)
    except (ValueError, RuntimeError):
        return float("nan")

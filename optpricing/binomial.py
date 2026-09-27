r"""Cox-Ross-Rubinstein (CRR) binomial tree pricing.

The CRR tree discretises the underlying's evolution into ``n`` steps of length
:math:`\Delta t = T/n`. Over each step the price moves up by a factor

.. math:: u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u

and the risk-neutral up-probability is

.. math:: p = \frac{e^{(r-q)\Delta t} - d}{u - d}.

The tree is rolled back from expiry to today, discounting expected values at the
risk-free rate. Unlike Black-Scholes, the tree naturally handles **American**
exercise by taking ``max(continuation, intrinsic)`` at every node. As
``n -> infinity`` the European price converges to Black-Scholes.
"""

from __future__ import annotations

import math

import numpy as np


def binomial_price(
    S, K, T, r, sigma, q=0.0, option_type="call", american=False, steps=500
) -> float:
    """Price a European or American option with a CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard option parameters (see :mod:`optpricing.black_scholes`).
    option_type : {"call", "put"}
    american : bool
        If ``True``, allow early exercise at every node.
    steps : int
        Number of tree steps (higher = more accurate, slower).
    """
    option_type = option_type.lower()
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "Risk-neutral probability out of [0,1]; increase steps or check inputs"
        )

    # Terminal asset prices: S * u^j * d^(steps-j)
    j = np.arange(steps + 1)
    prices = S * (u ** j) * (d ** (steps - j))
    if option_type == "call":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # Roll back through the tree.
    for i in range(steps, 0, -1):
        values = disc * (p * values[1:i + 1] + (1.0 - p) * values[0:i])
        if american:
            j = np.arange(i)
            prices = S * (u ** j) * (d ** (i - 1 - j))
            if option_type == "call":
                values = np.maximum(values, prices - K)
            else:
                values = np.maximum(values, K - prices)

    return float(values[0])

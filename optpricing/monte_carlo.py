r"""Monte Carlo pricing with variance reduction.

Under the risk-neutral measure the terminal asset price is

.. math:: S_T = S_0 \exp\!\big((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}\,Z\big),
   \quad Z \sim N(0,1).

The option price is the discounted expected payoff
:math:`e^{-rT}\,\mathbb{E}[\text{payoff}(S_T)]`, estimated by sampling many
paths. Two variance-reduction techniques are implemented:

* **Antithetic variates** - for each draw ``Z`` also use ``-Z``, cancelling
  much of the sampling noise for monotone payoffs.
* **Control variates** - use the known analytic expectation of the terminal
  price :math:`\mathbb{E}[S_T] = S_0 e^{(r-q)T}` as a control to reduce
  variance further.

The estimator returns both the price and its standard error, so convergence can
be quantified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """Result of a Monte Carlo pricing run."""

    price: float
    std_error: float
    n_paths: int

    @property
    def confidence_interval_95(self):
        half = 1.96 * self.std_error
        return (self.price - half, self.price + half)


def mc_price(
    S,
    K,
    T,
    r,
    sigma,
    q=0.0,
    option_type="call",
    n_paths=100_000,
    antithetic=True,
    control_variate=True,
    seed=None,
) -> MCResult:
    """Monte Carlo price of a European option with variance reduction."""
    option_type = option_type.lower()
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return MCResult(intrinsic, 0.0, n_paths)

    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    ST = S * np.exp(drift + vol * z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    discounted = disc * payoff

    if control_variate:
        # Control: discounted terminal price has known mean S*exp(-q*T).
        control = disc * ST
        expected_control = S * math.exp(-q * T)
        cov = np.cov(discounted, control, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        discounted = discounted - beta * (control - expected_control)

    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / math.sqrt(len(discounted)))
    return MCResult(price, std_error, len(discounted))

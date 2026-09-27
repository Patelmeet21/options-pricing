"""Plots for the options library: volatility smile and Greeks vs spot."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import black_scholes as bs  # noqa: E402
from .implied_vol import implied_vol  # noqa: E402


def plot_vol_smile(S, T, r, out_path, q=0.0):
    """Synthesise a skewed market and recover its implied-vol smile."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    strikes = np.linspace(0.7 * S, 1.3 * S, 25)
    # Impose a realistic skew: OTM puts richer than OTM calls.
    true_vol = 0.20 + 0.15 * ((strikes - S) / S) ** 2 - 0.05 * (strikes - S) / S
    market_prices = [bs.bs_price(S, k, T, r, v, q, "call") for k, v in zip(strikes, true_vol)]
    recovered = [implied_vol(p, S, k, T, r, q, "call") for p, k in zip(market_prices, strikes)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(strikes, np.array(true_vol) * 100, "o-", label="Input vol", alpha=0.5)
    ax.plot(strikes, np.array(recovered) * 100, "x--", label="Recovered implied vol")
    ax.axvline(S, color="grey", ls=":", label="Spot")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Volatility Smile / Skew (implied-vol solver round-trip)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_greeks_vs_spot(K, T, r, sigma, out_path, q=0.0, option_type="call"):
    """Plot the five Greeks as a function of spot price."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spots = np.linspace(0.5 * K, 1.5 * K, 100)
    delta = [bs.delta(s, K, T, r, sigma, q, option_type) for s in spots]
    gamma = [bs.gamma(s, K, T, r, sigma, q) for s in spots]
    vega = [bs.vega(s, K, T, r, sigma, q) / 100 for s in spots]
    theta = [bs.theta(s, K, T, r, sigma, q, option_type) / 365 for s in spots]
    rho = [bs.rho(s, K, T, r, sigma, q, option_type) / 100 for s in spots]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    series = [
        ("Delta", delta), ("Gamma", gamma), ("Vega (per 1%)", vega),
        ("Theta (per day)", theta), ("Rho (per 1%)", rho),
    ]
    for ax, (name, vals) in zip(axes.ravel(), series):
        ax.plot(spots, vals, color="#1f77b4")
        ax.axvline(K, color="grey", ls=":", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Spot")
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"{option_type.title()} Greeks vs Spot (K={K}, T={T}, sigma={sigma})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

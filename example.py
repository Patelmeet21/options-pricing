"""Command-line demo of the options-pricing library.

Prices a single option three ways (Black-Scholes, binomial tree, Monte Carlo),
prints the full Greeks (analytic vs finite-difference), recovers implied vol,
and saves the vol-smile and Greeks plots to ``results/``.

Usage
-----
    python example.py --spot 100 --strike 100 --tenor 1 --rate 0.05 --vol 0.2
    python example.py --type put --american
"""

from __future__ import annotations

import argparse

from optpricing import black_scholes as bs
from optpricing.binomial import binomial_price
from optpricing.greeks_fd import (
    default_pricer, fd_delta, fd_gamma, fd_rho, fd_theta, fd_vega,
)
from optpricing.implied_vol import implied_vol
from optpricing.monte_carlo import mc_price
from optpricing.plotting import plot_greeks_vs_spot, plot_vol_smile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spot", type=float, default=100.0)
    p.add_argument("--strike", type=float, default=100.0)
    p.add_argument("--tenor", type=float, default=1.0, help="years to expiry")
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--vol", type=float, default=0.20)
    p.add_argument("--div", type=float, default=0.0, help="dividend yield")
    p.add_argument("--type", choices=["call", "put"], default="call")
    p.add_argument("--american", action="store_true")
    p.add_argument("--paths", type=int, default=200_000)
    args = p.parse_args()

    S, K, T, r, sigma, q, ot = (
        args.spot, args.strike, args.tenor, args.rate, args.vol, args.div, args.type,
    )

    print(f"\n{'='*60}")
    print(f"Option: {ot.upper()}  S={S} K={K} T={T} r={r} sigma={sigma} q={q}")
    print(f"{'='*60}")

    bs_val = bs.bs_price(S, K, T, r, sigma, q, ot)
    eu_tree = binomial_price(S, K, T, r, sigma, q, ot, american=False, steps=1000)
    am_tree = binomial_price(S, K, T, r, sigma, q, ot, american=True, steps=1000)
    mc = mc_price(S, K, T, r, sigma, q, ot, n_paths=args.paths, seed=1)

    print("\n--- Prices ---")
    print(f"  Black-Scholes (European)   : {bs_val:.4f}")
    print(f"  Binomial tree (European)   : {eu_tree:.4f}")
    print(f"  Binomial tree (American)   : {am_tree:.4f}")
    print(f"  Monte Carlo (European)     : {mc.price:.4f} +/- {mc.std_error:.4f}")
    print(f"  MC 95% CI                  : [{mc.confidence_interval_95[0]:.4f}, "
          f"{mc.confidence_interval_95[1]:.4f}]")

    g = bs.all_greeks(S, K, T, r, sigma, q, ot)
    print("\n--- Greeks (analytic vs finite-difference) ---")
    rows = [
        ("Delta", g.delta, fd_delta(default_pricer, S, K, T, r, sigma, q, ot)),
        ("Gamma", g.gamma, fd_gamma(default_pricer, S, K, T, r, sigma, q, ot)),
        ("Vega",  g.vega,  fd_vega(default_pricer, S, K, T, r, sigma, q, ot)),
        ("Theta", g.theta, fd_theta(default_pricer, S, K, T, r, sigma, q, ot)),
        ("Rho",   g.rho,   fd_rho(default_pricer, S, K, T, r, sigma, q, ot)),
    ]
    print(f"  {'Greek':<8}{'Analytic':>14}{'Finite-diff':>16}")
    for name, a, fd in rows:
        print(f"  {name:<8}{a:>14.6f}{fd:>16.6f}")

    iv = implied_vol(bs_val, S, K, T, r, q, ot)
    print(f"\n--- Implied vol round-trip ---")
    print(f"  Priced at sigma={sigma}, recovered implied vol = {iv:.6f}")

    smile = plot_vol_smile(S, T, r, "results/vol_smile.png", q)
    greeks = plot_greeks_vs_spot(K, T, r, sigma, "results/greeks_vs_spot.png", q, ot)
    print(f"\nSaved plots:\n  {smile}\n  {greeks}\n")


if __name__ == "__main__":
    main()

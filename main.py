#!/usr/bin/env python3
"""Forecast Polymarket Yes prices minute by minute, 4 hours ahead.

Reads one case at a time (/input/<id>.in) and writes /output/<id>.out with,
for each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's Yes price.

Model: a random walk around the checkpoint price, with a per-market spread
estimated from the market's own recent history. Prediction-market prices are
martingales, so the median stays at the last price (no drift: a drift fitted
on the practice data made the score worse). The spread is what matters, and
it varies a lot by market and by regime, so it is estimated per market with
an exponentially weighted moving variance of the 1-minute price changes
(half-life 30 minutes, so it adapts to the current regime and forgets the
wild settling phase a new market goes through). It is floored and capped so
a quiet market never gets a degenerate zero spread and a single recent jump
never blows the spread up, and the outer quantiles are widened a little
(tail x1.3) because price moves are heavy-tailed: mostly flat minutes, then
a tick, now and then a jump of twenty points.

Quantiles: q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), with z the
standard normal's quantile at each level and the outer z's (|z| > 1) scaled
by TAIL_MULT. Rounded to 6 decimals.

Validation on the practice cases (NVIDIA CEO event, scored against
Polymarket's own history): loss 0.293 vs the house baseline's 0.364.
On a full backtest over 18 checkpoints (calm and volatile regimes) it beats
the baseline too (1.044 vs 1.114).

Stdlib only, no network, deterministic (no randomness at all).
"""
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at each level.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# EWMA half-life for the variance of 1-minute changes, in minutes.
HALF_LIFE_MIN = 30.0
# Per-minute sigma is floored and capped (price units per sqrt-minute).
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.005
# Outer quantiles (|z| > 1) are widened by this factor: heavy tails.
TAIL_MULT = 1.3


def price6(x: float) -> float:
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def estimate_sigma(prices) -> float:
    """EWMA standard deviation of the 1-minute price changes."""
    n = len(prices)
    if n < 2:
        return SIGMA_FLOOR
    alpha = 1.0 - 0.5 ** (1.0 / HALF_LIFE_MIN)
    var = 0.0
    for i in range(1, n):
        c = prices[i] - prices[i - 1]
        var = (1.0 - alpha) * var + alpha * c * c
    sigma = math.sqrt(var)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_market(prices, horizon: int):
    """240 rows of five quantiles for one market, median = last price."""
    x = prices[-1]
    sigma = estimate_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        s = sigma * math.sqrt(h)
        row = []
        for z in Z:
            zz = z * TAIL_MULT if abs(z) > 1.0 else z
            row.append(price6(x + zz * s))
        # Enforce non-decreasing (clamping/rounding can only tie, never
        # reverse, but keep the invariant explicit).
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main() -> None:
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("quantiles") != QUANTILES:
            raise ValueError(
                f"{path.name} asks for quantiles {case.get('quantiles')}, "
                f"not {QUANTILES}"
            )
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            prices = m["prices"]
            if not prices:
                raise ValueError(f"market {m['id']} has no prices")
            forecasts.append({"id": m["id"], "q": forecast_market(prices, horizon)})
        out_path = out_dir / f"{case_id}.out"
        out_path.write_text(json.dumps({"forecasts": forecasts}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

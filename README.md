# Polymarket FAA outage attribution forecaster

Entry for make.fun's **"Forecast Polymarket's FAA outage attribution prices
minute by minute, 4 hours ahead"** (polymarket-faa-attribution-02).

`main.py` (Python 3.13, stdlib only) reads one case at a time
(`/input/<id>.in`) and writes `/output/<id>.out`: for each market and each of
the 240 minutes after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%)
of that minute's Yes price.

## Model

A random walk around the checkpoint price, with a per-market spread estimated
from the market's own recent history:

- **Median = last price, no drift.** Prediction-market prices are martingales;
  a drift fitted on the practice data made the score worse.
- **Spread = EWMA volatility** of the 1-minute price changes (half-life
  30 min), floored at 0.0005 and capped at 0.005 price units per
  square-root minute. It adapts to the current regime and forgets the wild
  settling phase a new market goes through; the floor keeps a quiet market
  from getting a degenerate zero spread and the cap keeps one recent jump
  from blowing the spread up.
- **Outer quantiles widened x1.3.** Price moves are heavy-tailed: mostly flat
  minutes, then a tick, now and then a jump of twenty points.

Quantiles: `q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1)`, rounded to
6 decimals, with the outer `z`'s scaled by 1.3.

## Validation

Practice cases (NVIDIA CEO event, scored against Polymarket's own price
history, fetched from the public CLOB API):

| | loss (pinball, pp) |
|---|---|
| house baseline | 0.363589 |
| this forecaster | **0.292990** |

On a full backtest over 18 checkpoints (calm and volatile regimes) it beats
the baseline too (1.044 vs 1.114). Deterministic (no randomness), runs all
6 practice cases in ~0.3s.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

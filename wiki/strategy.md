# Strategy Architecture

Current best: exp #109, score 3.456. Config: `LSTM_h384_3L_drop03_target24h_discrete_funding_1H`.

## Model: SignalLSTM

```
Input:  (batch, 168, 23)  -- 7 days of 1H candles, 23 features each
Layer 1: BatchNorm1d on features dimension
Layer 2: LSTM(input=23, hidden=384, layers=3, dropout=0.3, bidirectional=False)
Layer 3: Head = BatchNorm1d(384) -> GELU -> Dropout(0.2) -> Linear(384,96) -> GELU -> Dropout(0.1) -> Linear(96,1) -> Tanh
Output: confidence in [-1, +1]
```

Training: 300 epochs, AdamW (lr=0.002, wd=0.02), CosineAnnealingLR, batch=512, seed=42.
Train split: first 80% of data. Predictions generated only on last 20% of train (not full OOS).

## Features (23 total)

| Category | Features | Count |
|----------|----------|-------|
| Trend | EMA50 distance, EMA200 distance, Ichimoku (tenkan/kijun/senkou_a/senkou_b), chikou span | 7 |
| Momentum | MACD standard (12/26/9), MACD fast (8/17/9), RSI(14) | 5 |
| Volatility | ATR(14), Bollinger bandwidth(20) | 2 |
| Volume | Volume vs 20-day MA | 1 |
| Multi-period returns | 1h, 3h, 6h, 12h, 24h pct_change | 5 |
| Macro | SPY trend, UUP trend, market_funding (avg FR BTC+ETH+SOL), VIXY trend | 3 |

Note: `market_funding` uses averaged funding rates (BTC+ETH+SOL) -- per-asset funding failed because XMR/TAO lack own futures (exp #100).

## Signal Generation (Discrete Thresholds)

```
prediction > +0.55  ->  position = +1.0   (full long)
prediction > +0.35  ->  position = +0.75
prediction > +0.15  ->  position = +0.5
prediction < -0.55  ->  position = -1.0   (full short)
prediction < -0.15  ->  position = -0.5
else                 ->  position = 0      (flat)
```

## ATR Trailing Stop

| Parameter | Value | Notes |
|-----------|-------|-------|
| Multiplier | 1.9 | Stop distance = 1.9 x ATR |
| Profit target | 3.0 x ATR | Half position closed at 3x ATR profit |
| Cooldown | 24 bars | No re-entry for 24h after stop-out |

## Scoring Formula

`score = 0.35*sharpe + 0.20*sortino + 0.20*(1-maxDD_pct) + 0.15*return + 0.10*winrate`

Averaged across 5 assets on validation period. Score is NOT bounded above -- the return component
can explode with short targets (see [decisions.md](decisions.md)).

## Per-Asset Results (exp #109, validation)

| Asset | Val Sharpe | Val Return | Buy & Hold | Score |
|-------|-----------|-----------|-----------|-------|
| TAO | 6.976 | +1702% | +12% | 5.66 |
| XMR | 7.707 | +1402% | +76% | 3.64 |
| ETH | 5.559 | +368% | +22% | 3.79 |
| SOL | 4.147 | +254% | -27% | 2.41 |
| BTC | 3.415 | +74% | -11% | 1.79 |

BTC is weakest -- most efficient/liquid market, hardest to predict. Val period was a BTC bear market.

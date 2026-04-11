# Experiments

123 experiments run from 2026-03-17 to 2026-03-19. Full log in `results.tsv`.

## Score Progression (Phase Evolution)

| Phase | Best Score | Experiment | Key Change |
|-------|-----------|------------|------------|
| Rule-based (Ichimoku+EMA) | 0.408 | #39 | dualMACD + chikou span |
| MLP neural network | 1.064 | #85 | NN512 + BatchNorm |
| LSTM 2L h=256 | 1.903 | #93 | LSTM on 168-candle lookback |
| LSTM h=384 2L | 1.948 | #101 | Wider hidden layer |
| LSTM h=384 3L | 2.116 | #103 | Added 3rd LSTM layer |
| +dropout=0.4 | 2.188 | #105 | Regularization |
| +target=24h | 3.341 | #108 | Switched from 48h to 24h forward return |
| +dropout=0.3 | **3.456** | **#109** | Less regularization for shorter target |

## Top 5 Best Scores (realistic, target >= 24h)

| # | Score | Sharpe (val) | Return (val) | MaxDD | Trades | Description |
|---|-------|-------------|-------------|-------|--------|-------------|
| 109 | 3.456 | 5.561 | 760% | -10.4% | 1263 | LSTM h384 3L drop0.3 target24h |
| 112 | 3.447 | 5.469 | 723% | -10.8% | 1267 | Same + ensemble 3 seeds (no improvement) |
| 114 | 3.425 | 5.469 | 648% | -10.8% | 1265 | 4 layers (marginal regression) |
| 118 | 3.381 | 5.430 | 761% | -10.2% | 1260 | wd=0.05 (marginal regression) |
| 108 | 3.341 | 5.307 | 682% | -11.2% | 1238 | dropout=0.4, target=24h |

## Top 5 Worst Failures

| # | Score | What Went Wrong |
|---|-------|----------------|
| 88 | -0.545 | First LSTM attempt (lb=50, broken scaling) -- negative Sharpe on all assets |
| 89 | -0.454 | LSTM lb=50 with scaled params -- still fundamentally broken |
| 1 | -0.030 | Very first experiment: Ichimoku+RSI histereza on 4H |
| 23 | 0.163 | Ensemble voting of 4 indicators -- diluted signals |
| 49 | 0.055 | First MLP hybrid -- 27 trades total, model collapsed to flat |

## Unrealistic Scores (Short Target Artifacts)

| # | Score | Target | Why Unrealistic |
|---|-------|--------|----------------|
| 111 | 4,982 | 6h | SOL train return 141 trillion %. 2836 trades/yr = 280% in fees. |
| 110 | 27.4 | 12h | 1948 trades/yr. Compounding artifact in zero-fee backtest. |

Sensible boundary: target >= 24h (max ~1200 trades/asset/year).

## Key Technical Discoveries

1. **BatchNorm is mandatory** -- score jump 0.60 -> 0.80, LSTM does not stabilize without it
2. **LSTM >> MLP** on crypto time series (+79% score improvement)
3. **1H >> 4H** timeframe -- 4x more training data, better signal-to-noise
4. **predict_lstm_confidence() >> predict_on_data()** -- OOS predictions are too noisy
5. **dropout x target interaction** -- shorter target needs less regularization
6. **h=384 sweet spot** -- h=256 too small, h=512 destabilizes multi-asset training
7. **Averaged funding > per-asset funding** -- XMR/TAO lack own futures, per-asset adds noise

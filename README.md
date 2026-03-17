# Autoquant — autonomous trading strategy optimizer

Autonomous agent loop: Claude Code modifies `strategy.py`, backtests on SPY+BTC+ETH daily data, evaluates composite `score`, keeps improvements, discards regressions. All commits pushed for full transparency. Telegram notifications after each experiment.

Same pattern as [karpathy/autoresearch](https://github.com/karpathy/autoresearch), applied to trading strategies instead of ML training.

## Architecture

| Component | File | Role |
|-----------|------|------|
| Strategy | `strategy.py` | Agent modifies — trading signals |
| Engine | `prepare.py` | Read-only — AV data download, vectorized backtest, composite scoring |
| Agent loop | `program.md` | Instructions for Claude |
| Notifications | `notify.sh` | Telegram alerts |

**Scoring:** composite of Sharpe, Sortino, max drawdown, return, win rate × trade_penalty (min 20 trades) × consistency (train/val Sharpe similarity). Train: 2019-01 to 2023-06, Val: 2023-07 to 2025-06, Holdout: 2025-07+.

**Data:** SPY (stocks), BTC, ETH (crypto) — daily candles from Alpha Vantage. Multi-asset averaging resists overfitting.

## Results summary (2026-03-17, 40 experiments)

### Score evolution

```
#0  0.015  baseline SMA crossover (1 trade, penalty for <20 trades)
#5  0.314  ADX trend strength filter
#8  0.448  DI directional confirmation
#14 0.546  BREAKTHROUGH: long-only (removed shorts)
#19 0.591  DI spread filter
#25 0.620  SMA50 + ADX>20 + DI>12 + BB
#27 0.630  vol regime filter (current best)
```

### Key discoveries

- **Long-only >> long+short** — score jumped 0.45→0.55 after removing shorts. SPY has upward bias, shorting BTC/ETH is dangerous
- **Bollinger Bands essential** — experiment #31 without BB dropped to 0.497, agent noted "BB essential for trade count"
- **DI spread 10-12 is optimal** — DI>15 too restrictive (too few trades), DI>5 too loose
- **Vol regime filter** gave marginal +0.01 — go flat during extreme volatility

### Current best strategy (score 0.630)

| Metric | Value |
|--------|-------|
| Score | 0.630 |
| Sharpe | 0.65 |
| Max drawdown | -17% |
| Strategy | long-only, SMA50 trend, ADX>20, DI>12, BB dip-buy, vol regime filter |

### Status

Agent entering plateau — last ~15 experiments are minor variants with marginal or zero improvement. Sharpe still below 1.0. Agent tried: Keltner, RSI, MACD, adaptive DI, ADX period tuning, dual SMA — nothing broke through.

## Full experiment log

See [`results.tsv`](results.tsv) for all experiments with commit hashes. Each commit on this branch contains the exact `strategy.py` that produced the result.

## Setup

Source code (Dockerfile, docker-compose, entrypoint): [`auditmos/autoresearch`](https://github.com/auditmos/autoresearch/tree/plans/autoquant) branch `plans/autoquant`.

```bash
git clone https://github.com/auditmos/autoresearch.git -b plans/autoquant
cd autoresearch

# .env
ALPHA_VANTAGE_API_KEY=your_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GIT_REMOTE_URL=https://github_pat_xxx@github.com/auditmos/autoquant.git

docker compose build
docker compose run autoquant login        # one-time auth
docker compose run -d autoquant agent     # launch headless
```

## License

MIT

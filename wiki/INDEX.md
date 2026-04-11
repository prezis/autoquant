# Autoquant Wiki

Autonomous trading strategy optimization for crypto using the Karpathy autoresearch pattern.
An LSTM neural network trained on 1H candles predicts 24h forward returns for BTC, ETH, XMR, SOL, TAO.

## Project Summary

| Metric | Value |
|--------|-------|
| Experiments run | 123 (as of 2026-03-19) |
| Best score | 3.456 (exp #109) |
| Score progression | 0.408 (rule-based) -> 3.456 (LSTM 3L) |
| Architecture | SignalLSTM, hidden=384, 3 layers, BatchNorm, GELU |
| Assets | BTC, ETH, XMR, SOL, TAO |
| Timeframe | 1H candles, 24h forward target |
| Training data | March 2023 - March 2025 (~17,500 candles/asset) |
| Validation | March 2025 - March 2026 |
| GPU | RTX 5090 (local, via Ollama for autoresearch) |

## Key Files

| File | Purpose |
|------|---------|
| `strategy.py` | The single mutable file (~850 lines). Model + features + signals + ATR stops |
| `prepare.py` | Read-only evaluation harness. Calls `strategy(df, context)` for 5 assets x 2 periods |
| `program.md` | Agent protocol: experiment rules, record, rejected ideas, next ideas |
| `results.tsv` | 123-row experiment log (score, sharpe, return, drawdown, trades, description) |
| `local_autoresearch.py` | Ollama-powered autonomous loop (qwen3.5:27b, search/replace diffs) |
| `live_signals.py` | Production signal generator (loads cached models, optional Telegram) |

## Wiki Pages

- [strategy.md](strategy.md) -- LSTM architecture, features, scoring formula
- [experiments.md](experiments.md) -- Top results, worst failures, score progression
- [methods.md](methods.md) -- Autoresearch loop, how program.md structures the agent
- [decisions.md](decisions.md) -- Design rationale, rejected alternatives, future directions

# Decisions

Design rationale, rejected alternatives, and future directions.

## Why LSTM Over Transformer

Transformer was tried in experiments #95-#96. Results:

| Model | Score | Trades | Issue |
|-------|-------|--------|-------|
| Transformer (continuous) | 1.695 | 3949 | 5x more trades, continuous positioning |
| Transformer (discrete) | 1.673 | 1049 | Still worse than LSTM 3L |
| LSTM 3L h=384 | 3.456 | 1263 | Best overall |

Transformer generates noisier confidence outputs, leading to more frequent position changes.
LSTM's recurrent memory handles the sequential nature of candle data more naturally.
The Transformer also requires positional encoding and more careful hyperparameter tuning
for relatively short sequences (168 timesteps).

## Why Search/Replace Diffs

`strategy.py` is ~850 lines. Options for LLM-driven modification:

| Method | Tokens/iter | Failure mode |
|--------|------------|--------------|
| Full file output | ~3000 | Truncation, context window pressure |
| Search/replace diffs | 200-500 | Search block mismatch |
| Line-number edits | ~300 | Line drift after prior edits |

Search/replace with fuzzy fallback was chosen. The fuzzy matcher normalizes whitespace
and adjusts indentation, handling ~90% of LLM indentation errors. Failed diffs trigger
auto-revert, so worst case is a wasted iteration (not a corrupted file).

## Why This Score Formula

`score = 0.35*sharpe + 0.20*sortino + 0.20*(1-maxDD) + 0.15*return + 0.10*winrate`

- **Sharpe (35%)**: primary quality metric, risk-adjusted return
- **Sortino (20%)**: penalizes downside volatility specifically
- **Max drawdown (20%)**: capital preservation -- a 50% DD needs 100% to recover
- **Return (15%)**: raw performance, but capped by other metrics to prevent overfit
- **Win rate (10%)**: tie-breaker, consistency indicator

**Known issue**: the return component is unbounded. With short targets (6h, 12h),
compounding in a zero-fee backtest creates fantasy scores (exp #111: score 4982).
Constraint: target >= 24h keeps scores realistic.

## Why h=384 (Not Smaller or Larger)

| Hidden | Score | Issue |
|--------|-------|-------|
| 256 | 1.903 | Underfitting -- not enough capacity for 5-asset multi-pattern learning |
| 384 | 3.456 | Sweet spot |
| 512 | 1.754-2.867 | Destabilizes at least 1 asset (ETH or XMR collapse, overfit) |
| 768 | 1.039 | Severe overfit |

384 provides enough capacity for 23 features x 168 timesteps without overfitting
on the ~14,000 training samples per asset.

## Rejected Approaches (from program.md)

| Idea | Exp | Result | Why It Failed |
|------|-----|--------|--------------|
| Per-asset ensemble (different seeds) | #94 | 1.600 | Destroys cross-asset correlations |
| Ensemble 3 seeds (same config) | #112 | 3.447 | No improvement, 3x GPU cost |
| Walkforward training | #97 | 1.479 | Weaker 70% model + threshold oscillation |
| Full OOS prediction | #98 | 1.526 | Too noisy outside train window |
| Continuous positioning (no thresholds) | #95 | 3949 trades | 5x more trades, fee-destroyed in production |
| LOOKBACK=96 | #107 | 1.474 | BTC overfit (train Sharpe 3.77 vs val 0.27) |
| LOOKBACK=240 | #119 | 3.086 | SOL/XMR degraded |
| 500 epochs | #116 | 3.207 | Overfit |
| lr=0.001 | #115 | 2.236 | Underfitting in 300 epochs |

## What to Try Next (from program.md)

**High priority (architecture):**
- BiLSTM -- bidirectional may capture longer dependencies
- GRU -- fewer parameters, often comparable to LSTM
- h=256 3L -- smaller model, stronger regularization

**Medium priority (features):**
- OBV (On-Balance Volume), Stochastic RSI, Williams %R
- Treasury yield curve spread (10Y-2Y, data available in context)
- News sentiment (NEWS_BTC, NEWS_ETH -- available but unused)

**Medium priority (risk management):**
- ATR multiplier=2.5 (wider stop), profit_target=4.0-5.0, cooldown=12

**Low priority:**
- target=36h (between 24h and 48h)
- Different seed than 42

## Ecosystem Learnings (2026-04-11)

From autoresearch ecosystem analysis (see [research.md](research.md)):

- **Feature deletion experiments needed** — Nunchi's biggest improvements came from REMOVING features, not adding. Try systematic feature ablation on our 23 features.
- **Judge agent needed** — review proposed changes before execution (autohypothesis observer pattern). Reduces wasted iterations on bad diffs.
- **Walk-forward validation needed** — our fixed 80/20 split is weak. Rolling windows catch regime changes and reduce overfit risk.
- **xLSTM worth trying** — drop-in LSTM replacement with exponential gating. No architecture rewrite needed.
- **PBO (Probability of Backtest Overfitting) as secondary metric** — CPCV-based metric that quantifies how likely our strategy is overfit. Add alongside score.

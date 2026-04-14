# Fibo Autoresearch — Agent Rules

## MODIFY ONLY strategy.py
- `prepare.py` is READ-ONLY (backtest engine, do not touch)
- `orchestrator.py` is READ-ONLY (experiment management)
- `program.md` is READ-ONLY (your instructions)

## Metric
The ONLY metric is `score` printed by `python strategy.py`.
Higher = better. Current best is in `results.db`.

## Workflow
1. Read `program.md` for domain knowledge and ideas
2. Read current `strategy.py` and `results.db` (what worked before)
3. Make ONE improvement — change ONE thing at a time
4. Run: `python strategy.py`
5. If `score` improved → commit with descriptive message
6. If `score` dropped or stayed same → revert with `git checkout strategy.py`
7. Repeat forever — NEVER STOP

## Constraints
- Max runtime per experiment: 2 minutes
- strategy.py must export a `strategy(df, fib_levels) -> pd.Series` function
- Signals must be -1, 0, or +1 only
- No imports of external paid APIs
- torch is available for GPU (RTX 5090) if you want neural approaches
- No per-asset tuning (`if symbol == 'X'` is overfitting)
- Log hypothesis + mutation in every commit message

## One-Change Discipline
Before each experiment, think:
1. **Hypothesis**: "I expect [X] to improve score because [Y]"
2. **Variable**: exactly ONE thing changed
3. **Prediction**: "Score should go from A to ~B"
After: record actual score, verdict (keep/discard/crash), learning

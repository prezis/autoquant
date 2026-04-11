# Methods

## The Autoresearch Pattern

Inspired by Karpathy's autonomous research loop. An LLM proposes strategy changes,
runs backtests, keeps improvements, reverts failures. Zero human intervention per iteration.

Two implementations exist:

1. **Claude Code** (original) -- Claude modifies `strategy.py` directly via tool use.
   Used for experiments #1-#123. Expensive (API tokens).
2. **local_autoresearch.py** -- Ollama on local RTX 5090, zero API cost.
   Uses qwen3.5:27b (60 tok/s). Same loop, runs autonomously.

## How program.md Structures the Agent

`program.md` is the agent's protocol document. It contains:

- **Rules**: only modify `strategy.py`, no new dependencies, log everything
- **Experiment protocol**: read state -> propose change -> run -> compare -> keep/revert
- **Current record**: score 3.456, full config, per-asset breakdown
- **Rejected ideas** ("NIE POWTARZAJ"): 15+ experiments that failed, with explanations
- **Future ideas** ("Co warto probowac dalej"): prioritized list of untried approaches
- **Data context**: available features in `context` dict (funding rates, macro, news)
- **Constraints**: target >= 24h, single seed=42, train=80% split

The agent reads this before every iteration to avoid repeating failed experiments.

## How local_autoresearch.py Works

### Loop (per iteration):
1. Read `strategy.py`, last 30 lines of `results.tsv`, full `program.md`
2. Build prompt with system instructions + user context (line-numbered strategy code)
3. Send to Ollama (qwen3.5:27b, temperature=0.7, max 4096 tokens)
4. Parse response: `<thinking>`, `<new_opis>`, `<diff>` blocks
5. Apply search/replace diffs to strategy.py
6. Validate (syntax check, required functions present, no blocked patterns)
7. Run `python3 strategy.py` (20min timeout)
8. Parse score from output
9. If score > current best: keep. Otherwise: revert from backup.
10. Log to `logi/autoresearch.jsonl`

### Search/Replace Diffs (not full file output)

The LLM outputs diffs in this format:
```
<<<SEARCH
exact lines from strategy.py
===
replacement lines
>>>
```

Why diffs instead of full file: strategy.py is ~850 lines. Full output would consume
~3000 tokens per iteration. Search/replace typically uses 200-500 tokens. This is critical
for local models with limited context windows.

### Fuzzy Matching

When exact search text is not found (LLM misremembers indentation), the engine falls back to
stripped-whitespace line matching. It then adjusts indentation of the replacement block
to match the original. This handles ~90% of indentation mismatches from the LLM.

### Safety

- Backup before every modification (`logi/autoresearch_backups/`)
- Blocked patterns: `subprocess`, `shutil.rmtree`, `__import__`, and OS exec calls
- Syntax validation via `compile()` before running
- Required function checks: `strategy()`, imports, `OPIS`, `evaluate()`
- Auto-revert on any failure (backtest error, parse error, no improvement)

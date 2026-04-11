# Research

External research, ecosystem analysis, and competitive intelligence.

## Autoresearch Ecosystem Analysis (2026-04-11)

### Ecosystem Map

- **karpathy/autoresearch** (70K stars) — spawned the entire ecosystem
- **auditmos/autoquant** (tkowalczyk) — score 0.630, first quant fork
- **popek1990/autoquant** — score 3.456, 5.5x improvement over auditmos baseline

### Top Repos

| Repo | Stars | Key Innovation | Result |
|------|-------|----------------|--------|
| **atlas-gic** | 1327 | Darwinian multi-agent evolution | Prompt-as-weights paradigm |
| **dietmarwo** | — | Split Brain (AI + optimizer) | AI proposes, optimizer validates |
| **Nunchi** | — | Feature deletion > addition | Sharpe 21.4, biggest gains from REMOVING features |
| **autohypothesis** | — | Observer + worker pattern | Beat Karpathy in 13 runs |

### Key Innovations

- **Feature deletion > addition** (Nunchi) — biggest improvements came from removing features, not adding them
- **Observer + worker pattern** (autohypothesis) — separate agent observes what works, worker executes
- **Prompt-as-weights** (ATLAS) — treat LLM prompts as evolvable weights in Darwinian selection
- **Split Brain** (dietmarwo) — AI generates hypotheses, classical optimizer validates/tunes
- **Walk-forward validation** — rolling train/test windows instead of fixed split
- **CPCV overfitting detection** — Combinatorially Purged Cross-Validation catches lookahead bias
- **xLSTM** — drop-in LSTM replacement with exponential gating, worth testing

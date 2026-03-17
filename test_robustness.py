"""Test robustności — ten sam model LSTM z 5 różnymi seedami."""
import sys
sys.path.insert(0, '.')
import strategy as strat
from prepare import evaluate
import numpy as np

seeds = [42, 137, 271, 404, 999]
results_all = []

print("=" * 70)
print("TEST ROBUSTNOŚCI — LSTM lb168 z 5 seedami")
print("=" * 70)

for seed in seeds:
    print(f"\n{'─'*70}")
    print(f"  SEED = {seed}")
    print(f"{'─'*70}\n")
    
    strat.GLOBAL_SEED = seed
    strat.OPIS = f"robustness_seed_{seed}"
    
    results = evaluate(strat.strategy, timeframe="1h")
    avg_score = results["_avg_score"]
    
    # Per-asset scores
    assets = [k for k in results if not k.startswith("_")]
    per_asset = {a: results[a]["score"] for a in assets}
    
    results_all.append({
        "seed": seed,
        "score": avg_score,
        "per_asset": per_asset,
        "val_trades": {a: results[a]["val"]["num_trades"] for a in assets},
    })
    
    print(f"\n  → Seed {seed}: score = {avg_score:.4f}")

print("\n" + "=" * 70)
print("PODSUMOWANIE ROBUSTNOŚCI")
print("=" * 70)

scores = [r["score"] for r in results_all]
print(f"\nScores: {[f'{s:.4f}' for s in scores]}")
print(f"Średnia:  {np.mean(scores):.4f}")
print(f"Std:     {np.std(scores):.4f}")
print(f"Min:     {np.min(scores):.4f}")
print(f"Max:     {np.max(scores):.4f}")
print(f"Rozrzut: {np.max(scores) - np.min(scores):.4f}")

# Per-asset
all_assets = list(results_all[0]["per_asset"].keys())
print(f"\nPer-asset (średnia ± std):")
for asset in all_assets:
    asset_scores = [r["per_asset"][asset] for r in results_all]
    trades = [r["val_trades"][asset] for r in results_all]
    print(f"  {asset:12s}: {np.mean(asset_scores):>7.3f} ± {np.std(asset_scores):.3f}  "
          f"(trades: {int(np.mean(trades))})")

# Stabilność
cv = np.std(scores) / np.mean(scores) * 100
print(f"\nWspółczynnik zmienności (CV): {cv:.1f}%")
if cv < 10:
    print("✅ STABILNY — CV < 10%")
elif cv < 25:
    print("⚠️  UMIARKOWANY — CV 10-25%")
else:
    print("❌ NIESTABILNY — CV > 25%")

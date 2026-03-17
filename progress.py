"""
progress.py — wizualizacja progresu eksperymentów z results.tsv
Uruchomienie: python3 progress.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

RESULTS_FILE = Path(__file__).parent / "results.tsv"


def load_results() -> pd.DataFrame:
    """Wczytuje results.tsv."""
    if not RESULTS_FILE.exists():
        print("Brak pliku results.tsv — uruchom najpierw strategy.py")
        sys.exit(1)

    df = pd.read_csv(RESULTS_FILE, sep="\t")
    df["data"] = pd.to_datetime(df["data"])

    if len(df) == 0:
        print("Plik results.tsv jest pusty — uruchom strategy.py")
        sys.exit(1)

    return df


def plot_progress(df: pd.DataFrame):
    """Rysuje wykres progresu eksperymentów."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Autoquant — progres eksperymentów", fontsize=14, fontweight="bold")

    nr = df["nr"]

    # ─── 1. Score (główna metryka) ───
    ax = axes[0]
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in df["score"]]
    ax.bar(nr, df["score"], color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Linia trendu (średnia krocząca)
    if len(df) >= 3:
        rolling = df["score"].rolling(3, min_periods=1).mean()
        ax.plot(nr, rolling, color="#2196F3", linewidth=2, label="Średnia krocząca (3)")
        ax.legend()

    # Najlepszy wynik
    best_idx = df["score"].idxmax()
    best = df.loc[best_idx]
    ax.annotate(
        f"Najlepszy: {best['score']:.4f}\n(#{int(best['nr'])})",
        xy=(best["nr"], best["score"]),
        xytext=(best["nr"], best["score"] + 0.05),
        ha="center", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2196F3"),
        color="#2196F3",
    )

    ax.set_ylabel("Score")
    ax.set_title("Score (wyższy = lepiej)")
    ax.grid(True, alpha=0.3)

    # ─── 2. Sharpe train vs val ───
    ax = axes[1]
    ax.plot(nr, df["sharpe_train"], "o-", color="#2196F3", label="Sharpe train", markersize=4)
    ax.plot(nr, df["sharpe_val"], "s-", color="#FF9800", label="Sharpe val", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Strefa overfittingu: train wysoko, val nisko
    ax.fill_between(nr, df["sharpe_train"], df["sharpe_val"],
                     where=(df["sharpe_train"] > df["sharpe_val"]),
                     alpha=0.1, color="red", label="Overfit gap")

    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe — train vs validation (mały gap = dobra generalizacja)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ─── 3. Return i Max Drawdown ───
    ax = axes[2]
    ax.bar(nr - 0.2, df["return_val"].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x),
           width=0.4, color="#4CAF50", alpha=0.7, label="Return val")
    ax.bar(nr + 0.2, df["max_dd_val"].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x),
           width=0.4, color="#F44336", alpha=0.7, label="Max DD val")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Wartość")
    ax.set_xlabel("Numer eksperymentu")
    ax.set_title("Validation — return vs max drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(__file__).parent / "progress.png"
    plt.savefig(save_path, dpi=120)
    print(f"Wykres zapisano → {save_path}")
    plt.close()

    # ─── Tabela podsumowania w terminalu ───
    print()
    print("─" * 80)
    print(f"{'Nr':>4}  {'Data':>16}  {'Score':>8}  {'Sharpe T':>9}  "
          f"{'Sharpe V':>9}  {'Return V':>10}  {'MaxDD V':>9}  Opis")
    print("─" * 80)

    for _, r in df.iterrows():
        score_color = "\033[92m" if r["score"] > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{int(r['nr']):>4}  {r['data'].strftime('%Y-%m-%d %H:%M'):>16}  "
              f"{score_color}{r['score']:>8.4f}{reset}  "
              f"{r['sharpe_train']:>9.3f}  {r['sharpe_val']:>9.3f}  "
              f"{r['return_val']:>10}  {r['max_dd_val']:>9}  "
              f"{r['opis'][:40]}")

    print("─" * 80)
    print(f"\nNajlepszy wynik: #{int(best['nr'])} — score {best['score']:.4f} "
          f"({best['opis'][:50]})")
    print(f"Łącznie eksperymentów: {len(df)}")


if __name__ == "__main__":
    df = load_results()
    plot_progress(df)

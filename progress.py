"""
progress.py — wizualizacja progresu eksperymentów z results.tsv
Uruchomienie: python3 progress.py
Monitoring:   watch --color -n 30 python3 progress.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import os

RESULTS_FILE = Path(__file__).parent / "results.tsv"

# Kolory ANSI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
BG_GREEN = "\033[42;30m"
BG_RED = "\033[41;97m"
BG_CYAN = "\033[46;30m"


def _pct(val) -> float:
    """Konwertuje string '12.34%' lub float na float."""
    if isinstance(val, str):
        return float(val.strip('%')) / 100
    return val


def _get_system_stats() -> dict:
    """Pobiera CPU, RAM i GPU stats."""
    stats = {"cpu": 0.0, "ram_used": 0, "ram_total": 0,
             "gpu_util": 0, "gpu_mem_used": 0, "gpu_mem_total": 0,
             "gpu_temp": 0, "gpu_power": 0, "gpu_name": "?"}

    # CPU
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = list(map(int, line.split()[1:]))
        idle = vals[3]
        total = sum(vals)
        # Potrzebujemy dwóch odczytów — użyjemy loadavg jako przybliżenie
        with open("/proc/loadavg") as f:
            load1 = float(f.read().split()[0])
        ncpu = len(os.sched_getaffinity(0))
        stats["cpu"] = min(load1 / ncpu * 100, 100)
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                mem[parts[0].rstrip(":")] = int(parts[1])
        total = mem.get("MemTotal", 0)
        avail = mem.get("MemAvailable", 0)
        stats["ram_total"] = total // 1024  # MB
        stats["ram_used"] = (total - avail) // 1024
    except Exception:
        pass

    # GPU (nvidia-smi)
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            stats["gpu_util"] = int(parts[0])
            stats["gpu_mem_used"] = int(parts[1])
            stats["gpu_mem_total"] = int(parts[2])
            stats["gpu_temp"] = int(parts[3])
            stats["gpu_power"] = float(parts[4])
            stats["gpu_name"] = parts[5]
    except Exception:
        pass

    return stats


def _bar(pct: float, width: int = 15, fill_color: str = GREEN) -> str:
    """Rysuje pasek postępu ASCII."""
    filled = int(pct / 100 * width)
    empty = width - filled

    if pct > 80:
        fill_color = RED
    elif pct > 50:
        fill_color = YELLOW

    return f"{fill_color}{'█' * filled}{DIM}{'░' * empty}{RESET} {pct:>5.1f}%"


def _is_agent_running() -> bool:
    """Sprawdza czy strategy.py jest aktualnie uruchomiony."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "strategy.py"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _time_ago(dt: datetime) -> str:
    """Zwraca czytelny czas od podanej daty."""
    diff = datetime.now() - dt
    minutes = int(diff.total_seconds() / 60)
    if minutes < 1:
        return "przed chwilą"
    elif minutes < 60:
        return f"{minutes} min temu"
    else:
        hours = minutes // 60
        return f"{hours}h {minutes % 60}min temu"


def load_results() -> pd.DataFrame:
    """Wczytuje results.tsv."""
    if not RESULTS_FILE.exists():
        print("Brak pliku results.tsv — uruchom najpierw strategy.py")
        sys.exit(1)

    df = pd.read_csv(RESULTS_FILE, sep="\t")
    df["data"] = pd.to_datetime(df["data"])

    for col in ["return_train", "return_val", "max_dd_val"]:
        df[col] = df[col].apply(_pct)

    if len(df) == 0:
        print("Plik results.tsv jest pusty — uruchom strategy.py")
        sys.exit(1)

    return df


def plot_progress(df: pd.DataFrame):
    """Rysuje wykres progresu eksperymentów."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Autoquant — progres eksperymentów", fontsize=14, fontweight="bold")

    nr = df["nr"]

    # ─── 1. Score ───
    ax = axes[0]
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in df["score"]]
    ax.bar(nr, df["score"], color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    if len(df) >= 3:
        rolling = df["score"].rolling(3, min_periods=1).mean()
        ax.plot(nr, rolling, color="#2196F3", linewidth=2, label="Srednia krocząca (3)")
        ax.legend()

    best_idx = df["score"].idxmax()
    best = df.loc[best_idx]
    ax.annotate(
        f"Najlepszy: {best['score']:.4f} (#{int(best['nr'])})",
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
    ax.fill_between(nr, df["sharpe_train"], df["sharpe_val"],
                     where=(df["sharpe_train"] > df["sharpe_val"]),
                     alpha=0.1, color="red", label="Overfit gap")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe — train vs validation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ─── 3. Return i Max Drawdown ───
    ax = axes[2]
    ax.bar(nr - 0.2, df["return_val"], width=0.4, color="#4CAF50", alpha=0.7, label="Return val")
    ax.bar(nr + 0.2, df["max_dd_val"], width=0.4, color="#F44336", alpha=0.7, label="Max DD val")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Wartość")
    ax.set_xlabel("Numer eksperymentu")
    ax.set_title("Validation — return vs max drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / "progress.png"
    plt.savefig(save_path, dpi=120)
    plt.close()


def print_dashboard(df: pd.DataFrame):
    """Wyświetla dashboard w terminalu."""
    best_idx = df["score"].idxmax()
    best = df.loc[best_idx]
    last = df.iloc[-1]
    first = df.iloc[0]

    W = 95

    # ─── Status agenta ───
    agent_running = _is_agent_running()
    last_time = _time_ago(last["data"])

    if agent_running:
        status = f"{BG_GREEN} AGENT PRACUJE {RESET}  Ostatni eksperyment: {last_time}"
    else:
        status = f"{BG_RED} AGENT ZATRZYMANY {RESET}  Ostatni eksperyment: {last_time}"

    # ─── System stats ───
    sys_stats = _get_system_stats()

    cpu_bar = _bar(sys_stats["cpu"])
    ram_pct = sys_stats["ram_used"] / max(sys_stats["ram_total"], 1) * 100
    ram_bar = _bar(ram_pct)
    gpu_bar = _bar(sys_stats["gpu_util"])
    vram_pct = sys_stats["gpu_mem_used"] / max(sys_stats["gpu_mem_total"], 1) * 100
    vram_bar = _bar(vram_pct)

    # ─── Nagłówek ───
    print()
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}")
    print(f"{BOLD}{CYAN}  AUTOQUANT — DASHBOARD{RESET}                                    {status}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}")

    # ─── System ───
    print(f"  {DIM}CPU{RESET}  {cpu_bar}   "
          f"{DIM}RAM{RESET}  {ram_bar}  "
          f"({sys_stats['ram_used']//1024:.0f}/{sys_stats['ram_total']//1024:.0f} GB)")

    print(f"  {DIM}GPU{RESET}  {gpu_bar}   "
          f"{DIM}VRAM{RESET} {vram_bar}  "
          f"({sys_stats['gpu_mem_used']//1024:.0f}/{sys_stats['gpu_mem_total']//1024:.0f} GB)  "
          f"{DIM}{sys_stats['gpu_temp']}°C  {sys_stats['gpu_power']:.0f}W{RESET}")

    # ─── Statystyki główne ───
    score_change = last["score"] - first["score"]
    score_arrow = "▲" if score_change > 0 else "▼"
    score_color = GREEN if score_change > 0 else RED

    # Ile było pozytywnych
    positive = (df["score"] > 0).sum()
    win_rate = positive / len(df)

    # Średni score z ostatnich 5
    last5_avg = df["score"].tail(5).mean()

    print(f"""
  {BOLD}Eksperymenty:{RESET} {len(df)}     {BOLD}Pozytywnych:{RESET} {positive}/{len(df)} ({win_rate:.0%})     {BOLD}Średnia (ost. 5):{RESET} {last5_avg:+.4f}

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  {BOLD}NAJLEPSZY{RESET}   {BG_GREEN} #{int(best['nr']):>3}  score {best['score']:+.4f}  Sharpe {best['sharpe_val']:+.3f}  ret {best['return_val']:+.1%}  dd {best['max_dd_val']:+.1%} {RESET}  │
  │  {BOLD}OSTATNI {RESET}    {score_color} #{int(last['nr']):>3}  score {last['score']:+.4f}  Sharpe {last['sharpe_val']:+.3f}  ret {last['return_val']:+.1%}  dd {last['max_dd_val']:+.1%} {RESET}  │
  │  {BOLD}ZMIANA  {RESET}    {score_color}{score_arrow} {score_change:+.4f} od startu{RESET}                                                      │
  └─────────────────────────────────────────────────────────────────────────────────┘
""")

    # ─── Top 5 ───
    top5 = df.nlargest(5, "score")
    print(f"  {BOLD}{YELLOW}TOP 5 NAJLEPSZYCH{RESET}")
    print(f"  {'─' * (W - 4)}")
    print(f"  {'#':>4}  {'Score':>8}    {'Sharpe V':>9}  {'Return V':>10}  {'MaxDD V':>9}  {'Trades':>6}  Opis")
    print(f"  {'─' * (W - 4)}")

    for _, r in top5.iterrows():
        is_best = r["nr"] == best["nr"]
        marker = f"{BG_GREEN}" if is_best else "  "
        end = f"{RESET}" if is_best else ""

        ret_c = GREEN if r["return_val"] > 0 else RED
        dd_c = RED if r["max_dd_val"] < -0.3 else YELLOW if r["max_dd_val"] < -0.2 else GREEN

        print(f"{marker}{int(r['nr']):>4}  {GREEN}{r['score']:>+8.4f}{RESET}    "
              f"{r['sharpe_val']:>+9.3f}  "
              f"{ret_c}{r['return_val']:>+9.1%}{RESET}   "
              f"{dd_c}{r['max_dd_val']:>+8.1%}{RESET}  "
              f"{r['trades_val']:>6}  "
              f"{r['opis'][:40]}{end}")

    # ─── Ostatnie eksperymenty ───
    print()
    print(f"  {BOLD}OSTATNIE EKSPERYMENTY{RESET}")
    print(f"  {'─' * (W - 4)}")
    print(f"  {'#':>4}  {'Czas':>5}  {'Score':>8}  {'':>2}  "
          f"{'ShT':>5}  {'ShV':>5}  {'RetV':>7}  {'DDV':>7}  {'Tr':>4}  Opis")
    print(f"  {'─' * (W - 4)}")

    tail = df.tail(15)
    for _, r in tail.iterrows():
        sc = GREEN if r["score"] > 0.2 else YELLOW if r["score"] > 0 else RED
        ret_c = GREEN if r["return_val"] > 0 else RED
        dd_c = RED if r["max_dd_val"] < -0.3 else YELLOW

        # Strzałka vs poprzedni
        idx = df.index[df["nr"] == r["nr"]].tolist()
        if idx and idx[0] > 0:
            prev_score = df.iloc[idx[0] - 1]["score"]
            diff = r["score"] - prev_score
            if diff > 0.01:
                arrow = f" {GREEN}▲{RESET}"
            elif diff < -0.01:
                arrow = f" {RED}▼{RESET}"
            else:
                arrow = f" {DIM}={RESET}"
        else:
            arrow = "  "

        star = f"{YELLOW}*{RESET}" if r["nr"] == best["nr"] else " "
        time_str = r["data"].strftime("%H:%M")

        print(f" {star}{int(r['nr']):>3}  {DIM}{time_str}{RESET}  "
              f"{sc}{r['score']:>+7.4f}{RESET} {arrow} "
              f"{r['sharpe_train']:>+5.2f}  {r['sharpe_val']:>+5.2f}  "
              f"{ret_c}{r['return_val']:>+6.1%}{RESET}  "
              f"{dd_c}{r['max_dd_val']:>+6.1%}{RESET}  "
              f"{r['trades_val']:>4}  "
              f"{DIM}{r['opis'][:35]}{RESET}")

    # ─── Stopka ───
    print(f"  {'─' * (W - 4)}")
    print(f"  {DIM}* = najlepszy   ▲ poprawa  ▼ pogorszenie  = bez zmian{RESET}")
    print(f"  {DIM}ShT = Sharpe train   ShV = Sharpe val   RetV = return val   DDV = max drawdown   Tr = transakcje{RESET}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}")
    print()


if __name__ == "__main__":
    df = load_results()
    plot_progress(df)
    print_dashboard(df)

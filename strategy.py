"""
strategy.py — Hybrid: NN (PyTorch MLP) + wskaźniki techniczne + ATR trailing stop (4H)
Sieć neuronowa uczy się optymalnego łączenia wskaźników w sygnał tradingowy.
GPU RTX 4090 wykorzystane do treningu i inferencji.
Agent modyfikuje TEN plik. prepare.py jest read-only.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
pd.set_option('future.no_silent_downcasting', True)
from prepare import evaluate, plot_equity

RESULTS_FILE = Path(__file__).parent / "results.tsv"
OPIS = "NN_confidence_scaler+rule_based+ATR1.9+cd6+PT3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Wskaźniki ───────────────────────────────────────────────────

def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high, low, close = df["high"], df["low"], df["close"]
    t = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    sa = ((t + k) / 2).shift(kijun)
    sb = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return {"tenkan": t, "kijun": k, "senkou_a": sa, "senkou_b": sb}

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1/period, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, 1e-9)))

def macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, min_periods=fast).mean()
    es = series.ewm(span=slow, min_periods=slow).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, min_periods=signal).mean()
    return {"macd": ml, "signal": sl, "hist": ml - sl}

def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()

def atr_trailing_stop(close, atr_values, positions, multiplier=2.0,
                      cooldown=0, profit_target_atr=0):
    result = positions.copy()
    trail_price = entry_price = np.nan
    direction = cooldown_remaining = 0
    profit_taken = False
    for i in range(len(close)):
        pos, price, av = result.iloc[i], close.iloc[i], atr_values.iloc[i]
        if np.isnan(av): continue
        if cooldown_remaining > 0:
            result.iloc[i] = 0; cooldown_remaining -= 1; direction = 0; continue
        sd = multiplier * av
        if pos > 0:
            if direction != 1:
                trail_price, entry_price, direction, profit_taken = price-sd, price, 1, False
            else:
                trail_price = max(trail_price, price-sd)
                if profit_target_atr > 0 and not profit_taken and price > entry_price + profit_target_atr*av:
                    result.iloc[i] = pos*0.5; profit_taken = True
                if price < trail_price:
                    result.iloc[i] = 0; direction = 0; cooldown_remaining = cooldown
        elif pos < 0:
            if direction != -1:
                trail_price, entry_price, direction, profit_taken = price+sd, price, -1, False
            else:
                trail_price = min(trail_price, price+sd)
                if profit_target_atr > 0 and not profit_taken and price < entry_price - profit_target_atr*av:
                    result.iloc[i] = pos*0.5; profit_taken = True
                if price > trail_price:
                    result.iloc[i] = 0; direction = 0; cooldown_remaining = cooldown
        else: direction = 0
    return result


# ─── Sieć neuronowa ──────────────────────────────────────────────

class SignalMLP(nn.Module):
    def __init__(self, n_features, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, 1), nn.Tanh())
    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_features(df, context):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    ema50 = close.ewm(span=50, min_periods=50).mean()
    ema200 = close.ewm(span=200, min_periods=200).mean()
    ichi = ichimoku(df)
    ct = pd.concat([ichi["senkou_a"], ichi["senkou_b"]], axis=1).max(axis=1)
    cb = pd.concat([ichi["senkou_a"], ichi["senkou_b"]], axis=1).min(axis=1)
    rv = rsi(close, 14)
    ms = macd(close); mf = macd(close, fast=8, slow=17, signal=9)
    av = atr(df, 14)
    vsma = vol.rolling(20).mean()

    features = pd.DataFrame({
        "ema50_dist": (close-ema50)/close, "ema200_dist": (close-ema200)/close,
        "cloud_top_dist": (close-ct)/close, "cloud_bottom_dist": (close-cb)/close,
        "tk_diff": (ichi["tenkan"]-ichi["kijun"])/close,
        "rsi_norm": (rv-50)/50, "macd_hist": ms["hist"]/close,
        "macd_fast_hist": mf["hist"]/close,
        "chikou": (close-close.shift(26))/close.shift(26).replace(0,1e-9),
        "atr_pct": av/close, "vol_ratio": (vol/vsma.replace(0,1e-9)-1).clip(-2,2),
        "ret_1": close.pct_change(1).clip(-0.1,0.1),
        "ret_6": close.pct_change(6).clip(-0.3,0.3),
        "ret_24": close.pct_change(24).clip(-0.5,0.5),
        "hl_range": (high-low)/close,
    }, index=df.index)

    spy_d = uup_d = pd.Series(0.0, index=df.index)
    if "SPY" in context and len(context["SPY"]) > 50:
        s = context["SPY"]["close"]; sm = s.rolling(50).mean()
        spy_d = ((s-sm)/sm).reindex(df.index, method="ffill").fillna(0)
    if "UUP" in context and len(context["UUP"]) > 20:
        u = context["UUP"]["close"]; um = u.rolling(20).mean()
        uup_d = ((u-um)/um).reindex(df.index, method="ffill").fillna(0)
    features["spy_trend"] = spy_d.clip(-0.1,0.1)
    features["uup_trend"] = uup_d.clip(-0.05,0.05)
    return features


def train_model(features, targets, n_epochs=150, lr=0.001):
    valid = features.dropna().index.intersection(targets.dropna().index)
    X = features.loc[valid].values.astype(np.float32)
    y = targets.loc[valid].values.astype(np.float32)
    yc = np.clip(y, np.percentile(y,2), np.percentile(y,98))
    ym = max(abs(yc.max()), abs(yc.min()), 1e-9)
    yn = np.clip(yc/ym, -1, 1)

    Xt = torch.tensor(X, device=DEVICE)
    yt = torch.tensor(yn, device=DEVICE)

    model = SignalMLP(n_features=X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    bs = min(256, len(Xt))
    best_loss, patience, no_imp = float("inf"), 20, 0

    model.train()
    for ep in range(n_epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        Xs, ys = Xt[perm], yt[perm]
        el, nb = 0.0, 0
        for i in range(0, len(Xt), bs):
            xb, yb = Xs[i:i+bs], ys[i:i+bs]
            pred = model(xb)
            mse = ((pred-yb)**2).mean()
            loss = mse - 0.01*pred.abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += mse.item(); nb += 1
        sched.step()
        al = el/max(nb,1)
        if al < best_loss-1e-6: best_loss, no_imp = al, 0
        else: no_imp += 1
        if no_imp >= patience: break

    model.eval()
    return model


@torch.no_grad()
def predict_nn_confidence(model, features):
    """Surowa predykcja NN (-1 do 1)."""
    valid = features.dropna()
    X = torch.tensor(valid.values.astype(np.float32), device=DEVICE)
    raw = model(X).cpu().numpy()
    result = pd.Series(0.0, index=features.index)
    result.loc[valid.index] = raw
    return result


def nn_confidence_to_scale(nn_pred, base_signals):
    """Konwertuje NN predykcję na skaler pozycji (0.3 do 1.3)."""
    scale = pd.Series(1.0, index=nn_pred.index)

    # Gdy base jest long (>0) i NN zgadza się (>0) → wzmocnij
    # Gdy base jest long (>0) i NN nie zgadza się (<0) → osłab
    for_long = base_signals > 0
    for_short = base_signals < 0

    # Skalowanie: nn_pred 0→1.0, +0.5→1.3, -0.5→0.3
    scale[for_long] = (1.0 + nn_pred[for_long] * 0.6).clip(0.3, 1.3)
    # Dla short: nn_pred negatywna = zgadza się z short → wzmocnij
    scale[for_short] = (1.0 - nn_pred[for_short] * 0.6).clip(0.3, 1.3)

    return scale


# ─── Strategia ───────────────────────────────────────────────────

def rule_based_signals(df, context):
    """Strategia bazowa: Ichimoku + Dual MACD + EMA200 + Chikou (sprawdzona, score 0.408)."""
    close = df["close"]

    ema200 = close.ewm(span=200, min_periods=200).mean()
    trend_up = close > ema200
    trend_down = close < ema200

    ichi = ichimoku(df)
    tenkan, kijun = ichi["tenkan"], ichi["kijun"]
    span_a, span_b = ichi["senkou_a"], ichi["senkou_b"]
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    chikou_bull = close > close.shift(26)

    rsi_val = rsi(close, 14)
    macd_std = macd(close)
    macd_fast_data = macd(close, fast=8, slow=17, signal=9)
    mb = macd_std["hist"] > 0
    mfb = macd_fast_data["hist"] > 0
    dual_bull = mb & mfb
    dual_bear = (macd_std["hist"] < 0) & (macd_fast_data["hist"] < 0)

    spy_bearish = pd.Series(False, index=df.index)
    dxy_rising = pd.Series(False, index=df.index)
    if "SPY" in context and len(context["SPY"]) > 200:
        spy = context["SPY"]["close"]
        spy_bearish = (~(spy.rolling(50).mean() > spy.rolling(200).mean())).reindex(df.index, method="ffill").fillna(False)
    if "UUP" in context and len(context["UUP"]) > 50:
        uup = context["UUP"]["close"]
        dxy_rising = (uup.rolling(20).mean() > uup.rolling(50).mean()).reindex(df.index, method="ffill").fillna(False)
    macro_bearish = spy_bearish | dxy_rising

    signals = pd.Series(0.0, index=df.index)
    long_base = (close > cloud_top) & (tenkan > kijun) & trend_up

    signals[long_base & dual_bull & chikou_bull] = 1.0
    signals[long_base & dual_bull & ~chikou_bull] = 0.75
    signals[long_base & (mb | mfb) & ~dual_bull & chikou_bull] = 0.75
    signals[long_base & (mb | mfb) & ~dual_bull & ~chikou_bull] = 0.5
    signals[long_base & ~mb & ~mfb] = 0.5

    short_cond = ((close < cloud_bottom) & (tenkan < kijun) & (rsi_val < 40)
                  & macro_bearish & dual_bear & trend_down)
    signals[short_cond] = -1.0

    return signals


def strategy(df, context):
    """
    Hybrid: rule-based signals × NN confidence scaler + ATR trailing stop.
    NN uczy się kiedy strategia bazowa działa dobrze (wzmocnij) vs źle (osłab).
    """
    close = df["close"]

    # 1. Strategia bazowa (sprawdzona)
    base_signals = rule_based_signals(df, context)

    # 2. Features dla NN
    features = build_features(df, context)
    # Dodaj sygnał bazowy jako feature
    features["base_signal"] = base_signals

    # 3. Forward returns jako target
    fwd_ret = close.pct_change(6).shift(-6)

    # 4. Trening NN na 80% danych
    te = int(len(df) * 0.8)
    model = train_model(features.iloc[:te], fwd_ret.iloc[:te], n_epochs=200, lr=0.002)

    # 5. NN predykcja → confidence scaler
    nn_pred = predict_nn_confidence(model, features)

    # 6. Łączenie: base_signal × nn_confidence
    # NN confidence: 0.3 do 1.5 (nie może odwrócić znaku, tylko skalować)
    signals = base_signals * nn_confidence_to_scale(nn_pred, base_signals)

    # 7. ATR trailing stop
    atr_val = atr(df, period=14)
    signals = atr_trailing_stop(close, atr_val, signals,
                                multiplier=1.9, cooldown=6, profit_target_atr=3.0)
    return signals


# ─── Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"AUTOQUANT — NN Hybrid (PyTorch MLP na {DEVICE})")
    print("=" * 60 + "\n")

    results = evaluate(strategy, timeframe="4h")
    avg_score = results["_avg_score"]

    print(f"\n{'='*60}\nWYNIK KOŃCOWY (avg score): {avg_score}\n{'='*60}")
    plot_equity(results)

    assets = [k for k in results if not k.startswith("_")]
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write("nr\tdata\tscore\tsharpe_train\tsharpe_val\t"
                    "return_train\treturn_val\tmax_dd_val\ttrades_val\topis\n")
    with open(RESULTS_FILE, "r") as f:
        nr = max(len(f.readlines())-1, 0)+1

    vs = [results[a]["val"]["sharpe"] for a in assets]
    ts = [results[a]["train"]["sharpe"] for a in assets]
    vr = [results[a]["val"]["total_return"] for a in assets]
    tr = [results[a]["train"]["total_return"] for a in assets]
    vd = [results[a]["val"]["max_drawdown"] for a in assets]
    vt = [results[a]["val"]["num_trades"] for a in assets]

    row = (f"{nr}\t{datetime.now().strftime('%Y-%m-%d %H:%M')}\t{avg_score:.4f}\t"
           f"{np.mean(ts):.3f}\t{np.mean(vs):.3f}\t{np.mean(tr):.2%}\t"
           f"{np.mean(vr):.2%}\t{np.mean(vd):.2%}\t{int(np.mean(vt))}\t{OPIS}")
    with open(RESULTS_FILE, "a") as f:
        f.write(row + "\n")
    print(f"\n📊 Zapisano wynik #{nr} → {RESULTS_FILE}")

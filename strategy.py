"""
strategy.py — Hybrid: LSTM (PyTorch) + wskaźniki techniczne + ATR trailing stop (1H)
LSTM z lookback 50 świec uczy się sekwencyjnych wzorców w time series krypto.
GPU RTX 4090 wykorzystane do treningu i inferencji.
Agent modyfikuje TEN plik. prepare.py jest read-only.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timezone
from pathlib import Path
pd.set_option('future.no_silent_downcasting', True)
from prepare import evaluate, plot_equity

RESULTS_FILE = Path(__file__).parent / "results.tsv"
OPIS = "LSTM_h384_3L_drop03_target24h_discrete_funding_1H"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOKBACK = 168  # 168 świec lookback (7 dni na 1H)
SINGLE_SEED = 42  # jednolity seed dla wszystkich assetów (bez per-asset ensemble)

MODEL_RETRAIN_HOURS = 168  # retrenuj model jeśli starszy niż 7 dni
BEST_MODEL_DIR = Path.home() / ".cache" / "autoquant" / "best_model"

# Modele wytrenowane w aktualnej sesji (wypełniane przez strategy())
_SESSION_MODELS: dict = {}  # asset_id -> [(model_info, seed), ...]


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


# ─── Sieci neuronowe ──────────────────────────────────────────────

class SignalLSTM(nn.Module):
    """LSTM do przetwarzania sekwencji wskaźników technicznych.
    Input: (batch, seq_len=50, n_features) → Output: (batch,) confidence [-1,1]
    """
    def __init__(self, n_features, hidden=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
            bidirectional=False)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 4, 1),
            nn.Tanh())

    def forward(self, x):
        # x: (batch, seq_len, features)
        # BatchNorm na features — transpose do (batch, features, seq_len) i z powrotem
        b, s, f = x.shape
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last = out[:, -1, :]   # ostatni timestep: (batch, hidden)
        return self.head(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidalne kodowanie pozycji — pozwala Transformerowi rozumieć kolejność."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SignalTransformer(nn.Module):
    """Transformer encoder do time series krypto.
    Input: (batch, seq_len=168, n_features) → Output: (batch,) confidence [-1,1]
    Self-attention pozwala modelowi patrzeć na WSZYSTKIE timestepy naraz
    i sam decydować które momenty w historii są najważniejsze.
    """
    def __init__(self, n_features, d_model=128, n_heads=4, n_layers=4, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Tanh())

    def forward(self, x):
        # x: (batch, seq_len, features)
        b, s, f = x.shape
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)    # (batch, seq_len, d_model)
        # Mean pooling po sekwencji (stabilniejsze niż [CLS] token)
        x = x.mean(dim=1)      # (batch, d_model)
        return self.head(x).squeeze(-1)


class SignalMLP(nn.Module):
    """Fallback MLP (dla porównania / BTC)."""
    def __init__(self, n_features, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(hidden//2, hidden//4), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden//4, 1), nn.Tanh())
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Features ─────────────────────────────────────────────────────

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
        "ret_3": close.pct_change(3).clip(-0.2,0.2),
        "ret_6": close.pct_change(6).clip(-0.3,0.3),
        "ret_12": close.pct_change(12).clip(-0.4,0.4),
        "ret_24": close.pct_change(24).clip(-0.5,0.5),
        "hl_range": (high-low)/close,
        "bb_width": (close.rolling(20).std()*2)/close,
        "vol_regime": (av/av.rolling(50).mean()).clip(0.3, 3.0) - 1,
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

    # Funding rate rynku krypto (Binance Futures, co 8H → forward-fill na 1H)
    # Wysoki funding (>0.01%) = rynek przegrzany, longs przepłacają → ryzyko reversal
    def _strip_tz(series):
        """Usuwa timezone z indeksu Series (funding rate ma UTC, OHLCV nie ma TZ)."""
        if series.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_localize(None)
        return series

    fr_btc = fr_eth = fr_sol = pd.Series(0.0, index=df.index)
    if "FR_BTC_" in context and len(context["FR_BTC_"]) > 10:
        fr_btc = _strip_tz(context["FR_BTC_"]["close"]).reindex(df.index, method="ffill").fillna(0)
    if "FR_ETH_" in context and len(context["FR_ETH_"]) > 10:
        fr_eth = _strip_tz(context["FR_ETH_"]["close"]).reindex(df.index, method="ffill").fillna(0)
    if "FR_SOL_" in context and len(context["FR_SOL_"]) > 10:
        fr_sol = _strip_tz(context["FR_SOL_"]["close"]).reindex(df.index, method="ffill").fillna(0)
    fr_count = (fr_btc != 0).astype(int) + (fr_eth != 0).astype(int) + (fr_sol != 0).astype(int)
    market_fr = (fr_btc + fr_eth + fr_sol) / fr_count.replace(0, 1)
    features["market_funding"] = market_fr.clip(-0.005, 0.005)

    # VIXY (ETF proxy VIX) — rośnie przy risk-off, spada przy risk-on
    vixy_d = pd.Series(0.0, index=df.index)
    if "VIXY" in context and len(context["VIXY"]) > 20:
        vixy = context["VIXY"]["close"]
        vixy_ma = vixy.rolling(20).mean()
        vixy_d = ((vixy - vixy_ma) / vixy_ma.replace(0, 1e-9)).reindex(
            df.index, method="ffill").fillna(0)
    features["vixy_trend"] = vixy_d.clip(-0.5, 0.5)

    return features


# ─── Trening LSTM ─────────────────────────────────────────────────

def make_sequences(X, y, lookback):
    """Tworzy sekwencje (sliding window) z macierzy features i targetów."""
    seqs, targets = [], []
    for i in range(lookback, len(X)):
        seqs.append(X[i-lookback:i])
        targets.append(y[i])
    return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_lstm(features, targets, lookback=LOOKBACK, n_epochs=300, lr=0.002, seed=42):
    """Trenuje LSTM na sekwencjach features → forward return."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Przygotuj dane — usuń NaN, zrób sekwencje
    valid = features.dropna().index.intersection(targets.dropna().index)
    feat_df = features.loc[valid]
    tgt_series = targets.loc[valid]

    X_raw = feat_df.values.astype(np.float32)
    y_raw = tgt_series.values.astype(np.float32)

    # Normalizacja targetów
    yc = np.clip(y_raw, np.percentile(y_raw, 2), np.percentile(y_raw, 98))
    ym = max(abs(yc.max()), abs(yc.min()), 1e-9)
    yn = np.clip(yc / ym, -1, 1)

    # Sekwencje
    X_seq, y_seq = make_sequences(X_raw, yn, lookback)
    if len(X_seq) < 100:
        return None  # za mało danych

    Xt = torch.tensor(X_seq, device=DEVICE)
    yt = torch.tensor(y_seq, device=DEVICE)

    n_features = X_seq.shape[2]
    model = SignalLSTM(n_features=n_features, hidden=384, n_layers=3, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    bs = min(512, len(Xt))
    best_loss, patience, no_imp = float("inf"), 25, 0

    model.train()
    for ep in range(n_epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        Xs, ys = Xt[perm], yt[perm]
        el, nb = 0.0, 0
        for i in range(0, len(Xt), bs):
            xb, yb = Xs[i:i+bs], ys[i:i+bs]
            pred = model(xb)
            mse = ((pred - yb) ** 2).mean()
            loss = mse - 0.01 * pred.abs().mean()  # zachęta do silnych sygnałów
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += mse.item(); nb += 1
        sched.step()
        al = el / max(nb, 1)
        if al < best_loss - 1e-6: best_loss, no_imp = al, 0
        else: no_imp += 1
        if no_imp >= patience: break

    model.eval()
    # Zwróć model + indeksy valid (do mapowania predykcji)
    return model, valid, lookback


@torch.no_grad()
def predict_lstm_confidence(model_info, features):
    """Predykcja LSTM na pełnym zbiorze features."""
    if model_info is None:
        return pd.Series(0.0, index=features.index)

    model, valid_idx, lookback = model_info
    feat_df = features.loc[features.index.isin(valid_idx)].dropna()
    X_raw = feat_df.values.astype(np.float32)

    result = pd.Series(0.0, index=features.index)

    if len(X_raw) <= lookback:
        return result

    # Sekwencje dla predykcji
    seqs = []
    seq_indices = []
    for i in range(lookback, len(X_raw)):
        seqs.append(X_raw[i-lookback:i])
        seq_indices.append(feat_df.index[i])

    X_seq = np.array(seqs, dtype=np.float32)
    Xt = torch.tensor(X_seq, device=DEVICE)

    # Predykcja w batchach (oszczędność pamięci GPU)
    preds = []
    bs = 2048
    for i in range(0, len(Xt), bs):
        batch = Xt[i:i+bs]
        pred = model(batch).cpu().numpy()
        preds.append(pred)
    raw = np.concatenate(preds)

    for idx, val in zip(seq_indices, raw):
        result.loc[idx] = val

    return result


def predict_live(model_info, features):
    """Predykcja LSTM na PEŁNYM zakresie features (bez filtru valid_idx).
    Używana przez live_signals.py — przewiduje dla każdej świecy łącznie z najnowszymi."""
    if model_info is None:
        return pd.Series(0.0, index=features.index)

    model, _, lookback = model_info
    feat_df = features.dropna()
    X_raw = feat_df.values.astype(np.float32)

    result = pd.Series(0.0, index=features.index)
    if len(X_raw) <= lookback:
        return result

    seqs, seq_indices = [], []
    for i in range(lookback, len(X_raw)):
        seqs.append(X_raw[i - lookback:i])
        seq_indices.append(feat_df.index[i])

    X_seq = np.array(seqs, dtype=np.float32)
    Xt = torch.tensor(X_seq, device=DEVICE)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xt), 2048):
            preds.append(model(Xt[i:i + 2048]).cpu().numpy())
    raw = np.concatenate(preds)

    for idx, val in zip(seq_indices, raw):
        result.loc[idx] = val
    return result


# ─── Trening Transformer ──────────────────────────────────────────

def train_transformer(features, targets, lookback=LOOKBACK, n_epochs=300, lr=0.001, seed=42):
    """Trenuje Transformer na sekwencjach features → forward return."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    valid = features.dropna().index.intersection(targets.dropna().index)
    feat_df = features.loc[valid]
    tgt_series = targets.loc[valid]

    X_raw = feat_df.values.astype(np.float32)
    y_raw = tgt_series.values.astype(np.float32)

    yc = np.clip(y_raw, np.percentile(y_raw, 2), np.percentile(y_raw, 98))
    ym = max(abs(yc.max()), abs(yc.min()), 1e-9)
    yn = np.clip(yc / ym, -1, 1)

    X_seq, y_seq = make_sequences(X_raw, yn, lookback)
    if len(X_seq) < 100:
        return None

    Xt = torch.tensor(X_seq, device=DEVICE)
    yt = torch.tensor(y_seq, device=DEVICE)

    n_features = X_seq.shape[2]
    model = SignalTransformer(n_features=n_features, d_model=128, n_heads=4,
                              n_layers=4, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    bs = min(256, len(Xt))  # mniejszy batch — Transformer je więcej pamięci
    best_loss, patience, no_imp = float("inf"), 25, 0

    model.train()
    for ep in range(n_epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        Xs, ys = Xt[perm], yt[perm]
        el, nb = 0.0, 0
        for i in range(0, len(Xt), bs):
            xb, yb = Xs[i:i+bs], ys[i:i+bs]
            pred = model(xb)
            mse = ((pred - yb) ** 2).mean()
            loss = mse - 0.01 * pred.abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += mse.item(); nb += 1
        sched.step()
        al = el / max(nb, 1)
        if al < best_loss - 1e-6: best_loss, no_imp = al, 0
        else: no_imp += 1
        if no_imp >= patience: break

    model.eval()
    return model, valid, lookback


# ─── Trening MLP (fallback) ──────────────────────────────────────

def train_mlp(features, targets, n_epochs=500, lr=0.003, seed=42):
    """MLP trening (dla BTC lub fallback)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    valid = features.dropna().index.intersection(targets.dropna().index)
    X = features.loc[valid].values.astype(np.float32)
    y = targets.loc[valid].values.astype(np.float32)
    yc = np.clip(y, np.percentile(y, 2), np.percentile(y, 98))
    ym = max(abs(yc.max()), abs(yc.min()), 1e-9)
    yn = np.clip(yc / ym, -1, 1)

    Xt = torch.tensor(X, device=DEVICE)
    yt = torch.tensor(yn, device=DEVICE)

    model = SignalMLP(n_features=X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    bs = min(256, len(Xt))
    best_loss, patience, no_imp = float("inf"), 25, 0

    model.train()
    for ep in range(n_epochs):
        perm = torch.randperm(len(Xt), device=DEVICE)
        Xs, ys = Xt[perm], yt[perm]
        el, nb = 0.0, 0
        for i in range(0, len(Xt), bs):
            xb, yb = Xs[i:i+bs], ys[i:i+bs]
            pred = model(xb)
            mse = ((pred - yb) ** 2).mean()
            loss = mse - 0.01 * pred.abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += mse.item(); nb += 1
        sched.step()
        al = el / max(nb, 1)
        if al < best_loss - 1e-6: best_loss, no_imp = al, 0
        else: no_imp += 1
        if no_imp >= patience: break

    model.eval()
    return model


@torch.no_grad()
def predict_mlp_confidence(model, features):
    """Predykcja MLP."""
    valid = features.dropna()
    X = torch.tensor(valid.values.astype(np.float32), device=DEVICE)
    raw = model(X).cpu().numpy()
    result = pd.Series(0.0, index=features.index)
    result.loc[valid.index] = raw
    return result


@torch.no_grad()
def predict_on_data(model_info, features_all):
    """Predykcja na PEŁNYM zakresie features (nie tylko train_valid_idx).
    Walkforward: model trenowany na [0:t] może predykować na [t:].
    """
    if model_info is None:
        return pd.Series(0.0, index=features_all.index)
    model, valid_idx, lookback = model_info
    feat_df = features_all.dropna()
    X_raw = feat_df.values.astype(np.float32)
    result = pd.Series(0.0, index=features_all.index)
    if len(X_raw) <= lookback:
        return result
    seqs, seq_indices = [], []
    for i in range(lookback, len(X_raw)):
        seqs.append(X_raw[i-lookback:i])
        seq_indices.append(feat_df.index[i])
    X_seq = np.array(seqs, dtype=np.float32)
    Xt = torch.tensor(X_seq, device=DEVICE)
    preds = []
    for i in range(0, len(Xt), 2048):
        preds.append(model(Xt[i:i+2048]).cpu().numpy())
    raw = np.concatenate(preds)
    for idx, val in zip(seq_indices, raw):
        result.loc[idx] = val
    return result


def nn_confidence_to_scale(nn_pred, base_signals):
    """Konwertuje NN predykcję na skaler pozycji (0.1 do 1.8)."""
    scale = pd.Series(1.0, index=nn_pred.index)
    for_long = base_signals > 0
    for_short = base_signals < 0
    scale[for_long] = (1.0 + nn_pred[for_long] * 1.0).clip(0.1, 1.8)
    scale[for_short] = (1.0 - nn_pred[for_short] * 1.0).clip(0.1, 1.8)
    return scale


# ─── Strategia ───────────────────────────────────────────────────

def rule_based_signals(df, context):
    """Strategia bazowa: Ichimoku + Dual MACD + EMA200 + Chikou."""
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


def btc_simple_strategy(df, context):
    """BTC: prosta strategia trend-following — EMA50/200 + MACD + szeroki ATR."""
    close = df["close"]
    ema50 = close.ewm(span=50, min_periods=50).mean()
    ema200 = close.ewm(span=200, min_periods=200).mean()
    trend_up = close > ema200
    trend_down = close < ema200

    macd_std = macd(close)
    macd_fast_data = macd(close, fast=8, slow=17, signal=9)
    mb = macd_std["hist"] > 0
    mfb = macd_fast_data["hist"] > 0
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
    btc_long = trend_up & (ema50 > ema200) & (mb | mfb)
    signals[btc_long] = 1.0
    btc_long_weak = trend_up & (ema50 > ema200) & ~mb & ~mfb
    signals[btc_long_weak] = 0.5
    btc_short = trend_down & (ema50 < ema200) & dual_bear & macro_bearish
    signals[btc_short] = -0.5

    atr_val = atr(df, period=14)
    signals = atr_trailing_stop(close, atr_val, signals, multiplier=2.5, cooldown=16, profit_target_atr=4.0)
    return signals


# ─── Cache modeli ────────────────────────────────────────────────

def _asset_id(df) -> str:
    """Identyfikuje asset po cenie mediany i zmienności.
    Progi dobrane empirycznie na danych 2023-2026 (wszystkie okresy: full/train/val):
      BTC  median > 10 000
      ETH  median > 1 000
      XMR  vol < 0.011 AND median > 155  (train=162, val=324 — zawsze > 155)
      TAO  median > 150  (val=248, train=~300 — zawsze > SOL)
      SOL  reszta  (val median=148, train=40 — zawsze < 150)
    Uwaga: SOL w val period ma vol=0.0088 (niższy niż XMR!), dlatego sam vol nie wystarczy.
    """
    median_price = df["close"].median()
    vol = df["close"].pct_change().std()
    if median_price > 10000:
        return "btc"
    elif median_price > 1000:
        return "eth"
    elif vol < 0.011 and median_price > 155:
        return "xmr"
    elif median_price > 150:
        return "tao"
    return "sol"


def save_model(model_info, path: Path):
    """Zapisuje model LSTM do pliku .pt"""
    if model_info is None:
        return
    model, valid_idx, lookback = model_info
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": model.input_bn.num_features,
        "hidden": model.lstm.hidden_size,
        "n_layers": model.lstm.num_layers,
        "dropout": model.lstm.dropout,
        "lookback": lookback,
        "valid_idx": valid_idx,
    }, path)


def load_model(path: Path):
    """Ładuje model LSTM z pliku .pt — zwraca model_info lub None przy błędzie."""
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        model = SignalLSTM(
            n_features=data["n_features"],
            hidden=data["hidden"],
            n_layers=data["n_layers"],
            dropout=data.get("dropout", 0.3),
        ).to(DEVICE)
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model, data["valid_idx"], data["lookback"]
    except Exception:
        return None


def _model_fresh(path: Path, max_hours: float = MODEL_RETRAIN_HOURS) -> bool:
    """Sprawdza czy model jest świeższy niż max_hours."""
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < max_hours


def save_best_models(model_dir: Path = BEST_MODEL_DIR):
    """Zapisuje modele z aktualnej sesji jako najlepsze."""
    model_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for asset, models in _SESSION_MODELS.items():
        for model_info, seed in models:
            save_model(model_info, model_dir / f"lstm_{asset}_s{seed}.pt")
            count += 1
    print(f"  🏆 Zapisano {count} modeli → {model_dir}")


def strategy(df, context, model_cache_dir=None, model_retrain_hours=MODEL_RETRAIN_HOURS):
    """
    LSTM 3L h=384 + BatchNorm + dropout=0.3, target=24h forward return.
    Jeśli model_cache_dir podany i model jest świeży → ładuje z dysku (tryb live).
    W przeciwnym razie trenuje od zera i zapisuje do cache (tryb backtest).
    """
    close = df["close"]
    features = build_features(df, context)
    n = len(df)
    asset = _asset_id(df)
    live_mode = False

    # ─── Próba załadowania z cache ───
    model_info = None
    if model_cache_dir is not None:
        model_path = Path(model_cache_dir) / f"lstm_{asset}_s{SINGLE_SEED}.pt"
        if _model_fresh(model_path, max_hours=model_retrain_hours):
            model_info = load_model(model_path)
            if model_info is not None:
                live_mode = True
                print(f"  [cache] {asset}: załadowano model z dysku ({model_path.name})")

    # ─── Trening (gdy brak cache lub za stary) ───
    if model_info is None:
        fwd_ret = close.pct_change(24).shift(-24)
        te = int(n * 0.80)
        model_info = train_lstm(
            features.iloc[:te], fwd_ret.iloc[:te],
            lookback=LOOKBACK, n_epochs=300, lr=0.002, seed=SINGLE_SEED
        )
        # Zapisz do _SESSION_MODELS (potrzebne do save_best_models)
        if asset not in _SESSION_MODELS:
            _SESSION_MODELS[asset] = []
        _SESSION_MODELS[asset].append((model_info, SINGLE_SEED))
        # Zapisz od razu do cache (jeśli podano)
        if model_cache_dir is not None and model_info is not None:
            Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
            save_model(model_info, Path(model_cache_dir) / f"lstm_{asset}_s{SINGLE_SEED}.pt")

    # ─── Predykcja ───
    # Live mode: na pełnych danych (w tym świeże świece poza oknem treningowym)
    # Backtest mode: tylko na train_valid_idx (unikamy lookahead)
    if live_mode:
        nn_pred = predict_live(model_info, features)
    else:
        nn_pred = predict_lstm_confidence(model_info, features)

    # ─── Dyskretne progi (jak w rekordowym #93) ───
    # Ciągłe pozycjonowanie (#95) tworzyło 3949 trades vs 785 w #93 → ogromne koszty
    signals = pd.Series(0.0, index=df.index)
    signals[nn_pred > 0.15] = 0.5
    signals[nn_pred > 0.35] = 0.75
    signals[nn_pred > 0.55] = 1.0
    signals[nn_pred < -0.15] = -0.5
    signals[nn_pred < -0.55] = -1.0

    atr_val = atr(df, period=14)
    signals = atr_trailing_stop(close, atr_val, signals,
                                multiplier=1.9, cooldown=24, profit_target_atr=3.0)
    return signals


# ─── Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"AUTOQUANT — LSTM Hybrid (PyTorch na {DEVICE})")
    print(f"  Lookback: {LOOKBACK} świec, Timeframe: 1H")
    print("=" * 60 + "\n")

    results = evaluate(strategy, timeframe="1h")
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

    # Zawsze zapisz modele — live_signals.py ładuje je bez ponownego treningu
    save_best_models(BEST_MODEL_DIR)

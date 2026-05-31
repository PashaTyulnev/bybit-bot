"""
Technische Indikatoren als reine numpy/pandas Funktionen.
Kein Lookahead: Wert an Index i nutzt nur Daten bis i.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> np.ndarray:
    return series.ewm(span=period, adjust=False).mean().to_numpy(float)


def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder-ATR (EWM mit alpha=1/period)."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    prev_c = np.empty(n)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    alpha = 1.0 / period
    return pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().to_numpy(float)


def adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder-ADX — delegiert an vorhandene Implementierung."""
    from src.strategy_backtester import compute_adx
    return compute_adx(df, period)


def bollinger(df: pd.DataFrame, period: int = 20,
              std_dev: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gibt (upper, mid, lower) zurück."""
    c   = df["close"]
    mid = c.rolling(period, min_periods=period).mean()
    std = c.rolling(period, min_periods=period).std(ddof=0)
    upper = (mid + std_dev * std).to_numpy(float)
    lower = (mid - std_dev * std).to_numpy(float)
    mid   = mid.to_numpy(float)
    return upper, mid, lower


def volatility_ratio(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ATR / close — normalisierte Volatilität."""
    atr_v  = atr(df, period)
    close  = df["close"].to_numpy(float)
    safe_c = np.where(close > 0, close, np.nan)
    return atr_v / safe_c


def align_htf_trend(
    base_df:    pd.DataFrame,
    htf_df:     pd.DataFrame,
    ema_period: int = 50,
) -> np.ndarray:
    """
    Berechnet HTF-EMA-Trend und aligned ihn auf das Base-TF.
    Gibt Array zurück: 1 = HTF bullish, -1 = bearish, 0 = unbekannt.
    Nutzt pd.merge_asof (kein Lookahead — immer der zuletzt bekannte HTF-Wert).
    """
    htf_close = htf_df["close"]
    htf_ema   = htf_close.ewm(span=ema_period, adjust=False).mean()
    trend     = np.where(htf_close.to_numpy() >= htf_ema.to_numpy(), 1, -1).astype(int)

    htf_times = pd.to_datetime(htf_df["datetime"]).reset_index(drop=True)
    base_times = pd.to_datetime(base_df["datetime"]).reset_index(drop=True)

    htf_frame  = pd.DataFrame({"time": htf_times,  "trend": trend})
    base_frame = pd.DataFrame({"time": base_times})

    merged = pd.merge_asof(
        base_frame.sort_values("time"),
        htf_frame.sort_values("time"),
        on="time",
        direction="backward",
    ).sort_values("time").reset_index(drop=True)

    return merged["trend"].fillna(0).to_numpy(int)

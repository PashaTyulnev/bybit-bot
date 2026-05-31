"""
CSV-Loader für Anti-Randomness Backtests.

Normalisiert Symbolnamen, findet passende CSV-Dateien, resampled wenn nötig
(z.B. 4h aus 1h), filtert nach Datum.
"""
from __future__ import annotations

import os
import pandas as pd

_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "raw",
)

# Bekannte Timeframes sortiert (niedrigste zuerst) — für Resampling-Fallback
_TF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Resample-Regeln für pd.resample()
_RESAMPLE_MAP = {
    "1m":  "1min",  "5m":  "5min",  "15m": "15min",
    "30m": "30min", "1h":  "1h",    "4h":  "4h",    "1d": "1D",
}


def _normalize_symbol(symbol: str) -> str:
    """'BTCUSDT', 'BTC/USDT:USDT', 'BTC_USDT', 'BTC' → 'BTC'"""
    s = symbol.upper()
    for suffix in ("/USDT:USDT", "_USDT:USDT", "USDT:USDT", "/USDT", "_USDT", "USDT"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s.strip("/_: ")


def find_csv(symbol: str, timeframe: str) -> str | None:
    """Gibt den Pfad zur CSV zurück, oder None wenn nicht vorhanden."""
    sym   = _normalize_symbol(symbol)
    fname = f"{sym}_USDT_USDT_{timeframe}.csv"
    path  = os.path.join(_RAW_DIR, fname)
    return path if os.path.exists(path) else None


def _resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resampled ein OHLCV-DataFrame auf das Ziel-Timeframe."""
    rule = _RESAMPLE_MAP.get(target_tf)
    if not rule:
        raise ValueError(f"Unbekanntes Timeframe für Resampling: {target_tf}")

    df = df.set_index(pd.to_datetime(df["datetime"]))
    resampled = df.resample(rule, closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    resampled = resampled.reset_index().rename(columns={"index": "datetime"})
    resampled["timestamp"] = resampled["datetime"].astype("int64") // 10**6
    return resampled


def load_csv(symbol: str, timeframe: str,
             start: str = "", end: str = "") -> pd.DataFrame:
    """
    Lädt OHLCV-CSV. Wenn timeframe-CSV nicht existiert, wird aus dem
    nächst-kleineren verfügbaren Timeframe resampled.

    Raises FileNotFoundError wenn keine geeignete CSV gefunden wird.
    """
    path = find_csv(symbol, timeframe)

    if path is None:
        # Fallback: aus kleinerem TF resampled
        tf_idx = _TF_ORDER.index(timeframe) if timeframe in _TF_ORDER else -1
        if tf_idx <= 0:
            raise FileNotFoundError(
                f"Keine CSV für {symbol} {timeframe} gefunden und kein Fallback möglich."
            )
        for smaller_tf in reversed(_TF_ORDER[:tf_idx]):
            fallback = find_csv(symbol, smaller_tf)
            if fallback:
                df_raw = _read_and_clean(fallback)
                df_raw = _filter_dates(df_raw, start, end)
                resampled = _resample_ohlcv(df_raw, timeframe)
                return resampled.reset_index(drop=True)
        raise FileNotFoundError(
            f"Keine CSV für {symbol} {timeframe} und kein kleineres Timeframe verfügbar."
        )

    df = _read_and_clean(path)
    return _filter_dates(df, start, end).reset_index(drop=True)


def _read_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime")
    return df


def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if start:
        ts = pd.Timestamp(start, tz="UTC")
        df = df[df["datetime"] >= ts]
    if end:
        te = pd.Timestamp(end, tz="UTC")
        df = df[df["datetime"] <= te]
    return df


def available_symbols() -> list[str]:
    """Gibt alle Symbole zurück für die mindestens eine CSV existiert."""
    seen: set[str] = set()
    for fname in os.listdir(_RAW_DIR):
        if fname.endswith(".csv"):
            sym = fname.split("_")[0]
            seen.add(sym)
    return sorted(seen)


def available_timeframes(symbol: str) -> list[str]:
    sym = _normalize_symbol(symbol)
    result = []
    for tf in _TF_ORDER:
        if find_csv(sym, tf):
            result.append(tf)
    return result

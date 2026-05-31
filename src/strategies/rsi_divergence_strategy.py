"""RSI Divergence Strategy – Scalping-Fokus."""
from __future__ import annotations

import pandas as pd

from src.strategies.base_strategy import BaseStrategy


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


class RSIDivergenceStrategy(BaseStrategy):
    """
    Bullish Div:  Kurs bei/nahe dem rollenden Tief  UND RSI über seinem rollenden Tief.
    Bearish Div:  Kurs bei/nahe dem rollenden Hoch  UND RSI unter seinem rollenden Hoch.

    Logik ohne Lookahead: Rolling-Werte basieren nur auf vergangenen Kerzen.
    """
    name = "rsi_divergence"

    def __init__(
        self,
        period:     int   = 14,
        lookback:   int   = 14,
        oversold:   float = 35.0,
        overbought: float = 65.0,
    ):
        self.period     = period
        self.lookback   = lookback
        self.oversold   = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        low   = df["low"]
        high  = df["high"]
        lb    = self.lookback

        rsi = _rsi(close, self.period)

        price_lo = low.rolling(lb,  min_periods=lb).min()
        price_hi = high.rolling(lb, min_periods=lb).max()
        rsi_lo   = rsi.rolling(lb,  min_periods=lb).min()
        rsi_hi   = rsi.rolling(lb,  min_periods=lb).max()

        # Bullish: Kurs berührt/nahe Rolling-Tief, RSI divergiert aufwärts
        bull = (
            (low <= price_lo * 1.001) &
            (rsi > rsi_lo + 2.0) &
            (rsi <= self.oversold + 20)
        )

        # Bearish: Kurs berührt/nahe Rolling-Hoch, RSI divergiert abwärts
        bear = (
            (high >= price_hi * 0.999) &
            (rsi < rsi_hi - 2.0) &
            (rsi >= self.overbought - 20)
        )

        signals = pd.Series(0, index=df.index)
        signals[bull] =  1
        signals[bear] = -1
        return signals

    def params_str(self) -> str:
        return (
            f"RSIDiv(p={self.period},lb={self.lookback},"
            f"os={self.oversold:.0f},ob={self.overbought:.0f})"
        )

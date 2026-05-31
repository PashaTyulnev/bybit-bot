from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast >= slow:
            raise ValueError("fast muss kleiner als slow sein.")
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def __str__(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    def params_str(self) -> str:
        return f"macd_{self.fast}_{self.slow}_{self.signal}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close       = df["close"]
        ema_fast    = close.ewm(span=self.fast,   adjust=False).mean()
        ema_slow    = close.ewm(span=self.slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        prev_macd   = macd_line.shift(1)
        prev_signal = signal_line.shift(1)

        signals = pd.Series(0, index=df.index, dtype=int)
        # Long:  MACD crosses above signal line
        signals[(prev_macd <= prev_signal) & (macd_line > signal_line)] = 1
        # Short: MACD crosses below signal line
        signals[(prev_macd >= prev_signal) & (macd_line < signal_line)] = -1

        warmup = self.slow + self.signal
        signals.iloc[:warmup] = 0
        return signals

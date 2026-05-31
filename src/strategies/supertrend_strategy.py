from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class SupertrendStrategy(BaseStrategy):
    name = "supertrend"

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0):
        self.atr_period  = atr_period
        self.multiplier  = multiplier

    def __str__(self) -> str:
        return f"Supertrend({self.atr_period},{self.multiplier})"

    def params_str(self) -> str:
        return f"supertrend_{self.atr_period}_{self.multiplier}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n     = len(df)

        # True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low,
             np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        # Wilder ATR
        atr    = np.zeros(n)
        period = self.atr_period
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        hl2         = (high + low) / 2.0
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        direction   = np.ones(n, dtype=int)   # 1 = bullish
        signals_arr = np.zeros(n, dtype=int)

        for i in range(period + 1, n):
            # Adjust final bands (carry-forward if not broken)
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # New direction
            prev_dir = direction[i - 1]
            if prev_dir == -1 and close[i] > final_upper[i]:
                direction[i] = 1
            elif prev_dir == 1 and close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = prev_dir

            # Emit signal only on flip
            if direction[i] != direction[i - 1]:
                signals_arr[i] = direction[i]

        return pd.Series(signals_arr, index=df.index, dtype=int)

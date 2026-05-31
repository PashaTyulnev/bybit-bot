import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class EMACrossStrategy(BaseStrategy):
    """
    EMA-Crossover:
      fast EMA > slow EMA  =>  Long
      fast EMA < slow EMA  =>  Short
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        if fast_period >= slow_period:
            raise ValueError("fast_period muss kleiner als slow_period sein.")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"EMA_{fast_period}_{slow_period}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close    = df["close"]
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        prev_fast = ema_fast.shift(1)
        prev_slow = ema_slow.shift(1)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(prev_fast <= prev_slow) & (ema_fast > ema_slow)] =  1   # crossover aufwärts
        signals[(prev_fast >= prev_slow) & (ema_fast < ema_slow)] = -1   # crossover abwärts

        signals.iloc[: self.slow_period] = 0
        return signals

    def params_str(self) -> str:
        return f"ema_cross_{self.fast_period}_{self.slow_period}"

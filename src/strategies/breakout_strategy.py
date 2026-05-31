import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    Breakout-Strategie:
      Close > höchstes High der letzten N Kerzen  =>  Long
      Close < tiefstes Low  der letzten N Kerzen  =>  Short

    shift(1) stellt sicher, dass die aktuelle Kerze nicht in die Berechnung einfliesst
    (kein Lookahead).
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.name     = f"Breakout_{lookback}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # shift(1): letzten N Kerzen vor der aktuellen
        roll_high = high.shift(1).rolling(window=self.lookback).max()
        roll_low  = low.shift(1).rolling(window=self.lookback).min()

        cond_long  = close > roll_high
        cond_short = close < roll_low
        prev_long  = cond_long.shift(1).fillna(False)
        prev_short = cond_short.shift(1).fillna(False)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[cond_long  & ~prev_long]  =  1   # erste Kerze des Ausbruchs nach oben
        signals[cond_short & ~prev_short] = -1   # erste Kerze des Ausbruchs nach unten

        signals.iloc[: self.lookback + 1] = 0
        return signals

    def params_str(self) -> str:
        return f"breakout_{self.lookback}"

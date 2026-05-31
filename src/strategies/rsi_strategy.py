import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI-Strategie (Wilder's RSI):
      RSI < oversold   =>  Long
      RSI > overbought =>  Short
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought
        self.name       = f"RSI_{period}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing (equivalent to EMA with com = period-1)
        avg_gain = gain.ewm(com=self.period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.period - 1, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)   # neutral wenn avg_loss = 0

        prev_rsi = rsi.shift(1).fillna(50)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(rsi < self.oversold)   & (prev_rsi >= self.oversold)]   =  1   # RSI kreuzt oversold
        signals[(rsi > self.overbought) & (prev_rsi <= self.overbought)] = -1   # RSI kreuzt overbought

        signals.iloc[: self.period + 1] = 0
        return signals

    def params_str(self) -> str:
        return f"rsi_{self.period}_{int(self.oversold)}_{int(self.overbought)}"

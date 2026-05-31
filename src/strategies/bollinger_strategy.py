import numpy as np
import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class BollingerStrategy(BaseStrategy):
    """
    Bollinger-Bands-Strategie mit optionalem Supertrend-Filter:
      Preis <= unteres Band  =>  Long  (mean-reversion: überverkauft)
      Preis >= oberes Band   =>  Short (mean-reversion: überkauft)

    Mit use_supertrend_filter=True: Long-Signale nur wenn Supertrend bullish,
    Short-Signale nur wenn Supertrend bearish.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 use_supertrend_filter: bool = True,
                 st_atr_period: int = 10, st_multiplier: float = 3.0):
        self.period  = period
        self.std_dev = std_dev
        self.use_supertrend_filter = use_supertrend_filter
        self.st_atr_period  = st_atr_period
        self.st_multiplier  = st_multiplier
        self.name    = f"BB_{period}_{std_dev}"

    def __str__(self) -> str:
        st = f"+ST({self.st_atr_period},{self.st_multiplier})" if self.use_supertrend_filter else ""
        return f"Bollinger({self.period},{self.std_dev}){st}"

    @staticmethod
    def _compute_supertrend_direction(df: pd.DataFrame, atr_period: int, multiplier: float) -> np.ndarray:
        """Berechnet Supertrend-Richtung. Gibt Array zurück: 1 = bullish, -1 = bearish."""
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n     = len(df)

        prev_close    = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low,
             np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        atr = np.zeros(n)
        if n >= atr_period:
            atr[atr_period - 1] = tr[:atr_period].mean()
            for i in range(atr_period, n):
                atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        hl2         = (high + low) / 2.0
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        direction   = np.ones(n, dtype=int)

        for i in range(atr_period + 1, n):
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

            prev_dir = direction[i - 1]
            if prev_dir == -1 and close[i] > final_upper[i]:
                direction[i] = 1
            elif prev_dir == 1 and close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = prev_dir

        return direction

    def live_st_direction(self, df: pd.DataFrame) -> int:
        """
        Supertrend-Richtung auf dem aktuellen (noch offenen) Candle.
        Wird vom LiveTrader als letzter Check direkt vor dem Order-Entry aufgerufen.
        Gibt 1 (bullish) oder -1 (bearish) zurück.
        """
        direction = self._compute_supertrend_direction(df, self.st_atr_period, self.st_multiplier)
        return int(direction[-1])

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        sma   = close.rolling(window=self.period).mean()
        std   = close.rolling(window=self.period).std(ddof=0)

        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std

        cond_long  = close <= lower
        cond_short = close >= upper
        prev_long  = cond_long.shift(1).fillna(False)
        prev_short = cond_short.shift(1).fillna(False)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[cond_long  & ~prev_long]  =  1   # erste Berührung des unteren Bands
        signals[cond_short & ~prev_short] = -1   # erste Berührung des oberen Bands

        signals.iloc[: self.period] = 0

        if self.use_supertrend_filter:
            direction = self._compute_supertrend_direction(df, self.st_atr_period, self.st_multiplier)
            st_dir    = pd.Series(direction, index=df.index)
            # Nur Signale durchlassen die mit Supertrend-Richtung übereinstimmen
            aligned = ((signals == 1) & (st_dir == 1)) | ((signals == -1) & (st_dir == -1))
            signals = signals.where(aligned, other=0)

        return signals

    def params_str(self) -> str:
        return f"bollinger_{self.period}_{self.std_dev}"

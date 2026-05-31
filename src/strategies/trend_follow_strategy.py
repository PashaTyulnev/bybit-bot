import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.strategy_backtester import compute_adx


class TrendFollowStrategy(BaseStrategy):
    """
    Trend-Following: EMA-Crossover nur bei starkem Trend (ADX >= threshold).
    Long  bei bullishem EMA-Cross wenn Trend stark genug.
    Short bei bearishem EMA-Cross wenn Trend stark genug.
    """
    name = "TrendFollow"

    def __init__(self, fast_ema: int = 20, slow_ema: int = 100, adx_threshold: float = 25.0):
        if fast_ema >= slow_ema:
            raise ValueError("fast_ema muss kleiner als slow_ema sein.")
        self.fast_ema      = fast_ema
        self.slow_ema      = slow_ema
        self.adx_threshold = adx_threshold

    def __str__(self) -> str:
        return f"TrendFollow(EMA{self.fast_ema}/{self.slow_ema}, ADX≥{self.adx_threshold})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close    = df["close"]
        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        prev_fast = ema_fast.shift(1)
        prev_slow = ema_slow.shift(1)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(prev_fast <= prev_slow) & (ema_fast > ema_slow)] =  1
        signals[(prev_fast >= prev_slow) & (ema_fast < ema_slow)] = -1
        signals.iloc[:self.slow_ema] = 0

        adx_arr       = compute_adx(df)
        trending_mask = pd.Series(adx_arr >= self.adx_threshold, index=df.index)
        signals       = signals.where(trending_mask, other=0)
        return signals

    def params_str(self) -> str:
        return f"trend_follow_{self.fast_ema}_{self.slow_ema}_adx{self.adx_threshold}"

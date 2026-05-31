import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.strategy_backtester import compute_adx


class MeanRevStrategy(BaseStrategy):
    """
    Mean-Reversion: BB-Touches (erste Berührung) nur im Ranging-Markt (ADX < threshold).
    Long  wenn Preis <= unteres Band und ADX zu niedrig für Trend.
    Short wenn Preis >= oberes  Band und ADX zu niedrig für Trend.
    """
    name = "MeanRev"

    def __init__(self, bb_period: int = 10, bb_std: float = 2.0, adx_threshold: float = 20.0):
        self.bb_period     = bb_period
        self.bb_std        = bb_std
        self.adx_threshold = adx_threshold

    def __str__(self) -> str:
        return f"MeanRev(BB{self.bb_period}/{self.bb_std}σ, ADX<{self.adx_threshold})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        sma   = close.rolling(self.bb_period).mean()
        std   = close.rolling(self.bb_period).std(ddof=0)
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        cond_long  = close <= lower
        cond_short = close >= upper
        prev_long  = cond_long.shift(1).fillna(False)
        prev_short = cond_short.shift(1).fillna(False)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[cond_long  & ~prev_long]  =  1
        signals[cond_short & ~prev_short] = -1
        signals.iloc[:self.bb_period] = 0

        adx_arr      = compute_adx(df)
        ranging_mask = pd.Series(adx_arr < self.adx_threshold, index=df.index)
        signals      = signals.where(ranging_mask, other=0)
        return signals

    def params_str(self) -> str:
        return f"mean_rev_{self.bb_period}_{self.bb_std}_adx{self.adx_threshold}"

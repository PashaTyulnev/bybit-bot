"""
Anti-Randomness Strategien.

Alle Strategien erben von BaseStrategy und geben Signale auf dem Base-TF zurück.
HTF-Filter und Volatilitäts-Filter werden in runner.py extern angewandt.

Strategien:
  1. TrendFollowStrategy       – EMA-Cross + Trend-Confirmation
  2. MeanReversionRegime       – BB in Ranging-Markt (ADX-basiertes Regime)
  3. RegimeSwitchStrategy      – EMA-Cross wenn Trend, BB wenn Range
  4. VolatilityFilterStrategy  – Wrapper: beliebige Strategie + Vol-Fenster
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy
from src.anti_randomness.indicators import ema, bollinger, adx as calc_adx


# ── 1. Trend-Following ─────────────────────────────────────────────────────────

class TrendFollowStrategy(BaseStrategy):
    """
    EMA-Cross auf dem Base-TF.
    Long:  fast EMA kreuzt slow EMA von unten (bullish cross)
    Short: fast EMA kreuzt slow EMA von oben (bearish cross)

    HTF-Filter und Volatilitäts-Filter werden vom Runner extern angewandt.
    """

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) muss kleiner als slow ({slow}) sein.")
        self.fast = fast
        self.slow = slow
        self.name = f"TrendFollow(EMA{fast}/{slow})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close     = df["close"]
        ema_fast  = pd.Series(ema(close, self.fast), index=df.index)
        ema_slow  = pd.Series(ema(close, self.slow),  index=df.index)

        above     = ema_fast > ema_slow
        prev_above = above.shift(1).fillna(False)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[above  & ~prev_above] =  1   # bullish cross
        signals[~above &  prev_above] = -1   # bearish cross

        # Warm-up: keine Signale vor slow-Periode
        signals.iloc[: self.slow] = 0
        return signals

    def params_str(self) -> str:
        return f"trend_ema{self.fast}_{self.slow}"


# ── 2. Mean-Reversion im Ranging-Markt ────────────────────────────────────────

class MeanReversionRegime(BaseStrategy):
    """
    Bollinger-Band Mean-Reversion, aber NUR in Seitwärts-Phasen.

    Regime-Erkennung: ADX(adx_period) < adx_threshold → ranging
    Signale:
      Long:  close <= lower BB  UND  ADX < threshold
      Short: close >= upper BB  UND  ADX < threshold
    Nur auf erste Berührung (nicht wenn Vorgänger schon außerhalb lag).
    """

    def __init__(
        self,
        bb_period:     int   = 20,
        bb_std:        float = 2.0,
        adx_period:    int   = 14,
        adx_threshold: float = 25.0,
    ) -> None:
        self.bb_period     = bb_period
        self.bb_std        = bb_std
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold
        self.name = (
            f"MeanRevRegime(BB{bb_period}/{bb_std}σ,"
            f"ADX{adx_period}<{adx_threshold})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, _, lower = bollinger(df, self.bb_period, self.bb_std)
        adx_v           = calc_adx(df, self.adx_period)
        close           = df["close"].to_numpy(float)

        ranging   = adx_v < self.adx_threshold
        at_lower  = close <= lower
        at_upper  = close >= upper

        # Nur erste Berührung (vorheriger Balken war nicht außerhalb)
        prev_lower = np.roll(at_lower, 1); prev_lower[0] = False
        prev_upper = np.roll(at_upper, 1); prev_upper[0] = False

        long_sig  = at_lower & ~prev_lower & ranging
        short_sig = at_upper & ~prev_upper & ranging

        signals = np.zeros(len(df), dtype=int)
        signals[long_sig]  =  1
        signals[short_sig] = -1

        warmup = max(self.bb_period, self.adx_period * 2)
        signals[:warmup] = 0
        return pd.Series(signals, index=df.index)

    def params_str(self) -> str:
        return f"mean_rev_bb{self.bb_period}_{self.bb_std}_adx{self.adx_threshold}"


# ── 3. Regime-Switch ──────────────────────────────────────────────────────────

class RegimeSwitchStrategy(BaseStrategy):
    """
    Kombinierte Strategie: wechselt Ansatz je nach Marktregime.
      Trending  (ADX ≥ threshold): EMA-Cross (Trend-Following)
      Ranging   (ADX <  threshold): BB-Touch  (Mean-Reversion)

    Signale aus dem je passenden Sub-Modus werden kombiniert,
    aber nie gleichzeitig aktiv.
    """

    def __init__(
        self,
        fast_ema:      int   = 20,
        slow_ema:      int   = 100,
        bb_period:     int   = 20,
        bb_std:        float = 2.0,
        adx_period:    int   = 14,
        adx_threshold: float = 25.0,
    ) -> None:
        self.fast_ema      = fast_ema
        self.slow_ema      = slow_ema
        self.bb_period     = bb_period
        self.bb_std        = bb_std
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold
        self.name = (
            f"RegimeSwitch(EMA{fast_ema}/{slow_ema}|"
            f"BB{bb_period}/{bb_std}σ|ADX{adx_threshold})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"]
        adx_v  = calc_adx(df, self.adx_period)
        trend  = adx_v >= self.adx_threshold   # True = trending

        # EMA-Cross Signale
        ef = pd.Series(ema(close, self.fast_ema), index=df.index)
        es = pd.Series(ema(close, self.slow_ema),  index=df.index)
        above      = ef > es
        prev_above = above.shift(1).fillna(False)
        ema_long  = (above  & ~prev_above).to_numpy()
        ema_short = (~above &  prev_above).to_numpy()

        # BB-Touch Signale
        upper, _, lower = bollinger(df, self.bb_period, self.bb_std)
        c = close.to_numpy(float)
        at_lower  = c <= lower
        at_upper  = c >= upper
        prev_lower = np.roll(at_lower, 1); prev_lower[0] = False
        prev_upper = np.roll(at_upper, 1); prev_upper[0] = False
        bb_long  = at_lower & ~prev_lower
        bb_short = at_upper & ~prev_upper

        # Regime-abhängig zusammenführen
        sigs = np.zeros(len(df), dtype=int)
        sigs[ema_long  & trend]  =  1
        sigs[ema_short & trend]  = -1
        sigs[bb_long   & ~trend] =  1
        sigs[bb_short  & ~trend] = -1

        warmup = max(self.slow_ema, self.bb_period, self.adx_period * 2)
        sigs[:warmup] = 0
        return pd.Series(sigs, index=df.index)

    def params_str(self) -> str:
        return (
            f"regime_switch_ema{self.fast_ema}_{self.slow_ema}"
            f"_bb{self.bb_period}_adx{self.adx_threshold}"
        )


# ── 4. Volatility-Filter Wrapper ──────────────────────────────────────────────

class VolatilityFilterStrategy(BaseStrategy):
    """
    Wrapper-Strategie: Filtert Signale einer Basis-Strategie nach Volatilitäts-Fenster.
    Signale werden nur durchgelassen wenn ATR/close ∈ [vol_min, vol_max].

    Verhindert Trades in extremer Volatilität (News-Spikes) und totem Markt.
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        vol_period:    int   = 14,
        vol_min:       float = 0.002,
        vol_max:       float = 0.03,
    ) -> None:
        self.base         = base_strategy
        self.vol_period   = vol_period
        self.vol_min      = vol_min
        self.vol_max      = vol_max
        self.name = (
            f"VolFilter({base_strategy.name},"
            f"vol∈[{vol_min},{vol_max}])"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        base_sigs = self.base.generate_signals(df).to_numpy(int)

        from src.anti_randomness.indicators import volatility_ratio
        vol = volatility_ratio(df, self.vol_period)
        in_window = (vol >= self.vol_min) & (vol <= self.vol_max)

        filtered = np.where(in_window, base_sigs, 0).astype(int)
        return pd.Series(filtered, index=df.index)

    def params_str(self) -> str:
        return f"vol_filter_{self.base.params_str()}_{self.vol_min}_{self.vol_max}"

"""
Strategie-Optimizer: Grid-Search über Strategie-Parameter + Trade-Einstellungen.

Verwendung:
    python -m src.strategy_optimizer --strategy ema --top 10
    python -m src.strategy_optimizer --strategy rsi --leverages 5 10 20 --tp 1 2 3
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.strategy_backtester import (
    StrategyConfig, compute_adx, compute_market_condition_arr, run_strategy_backtest_fast,
)
from src.strategies import (
    BollingerStrategy, BreakoutStrategy, EMACrossStrategy,
    MACDStrategy, RSIStrategy, SupertrendStrategy, RSIDivergenceStrategy,
)
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# ── Standard-Suchräume pro Strategie ──────────────────────────────────────────

EMA_GRID = {
    "fast_periods": [5, 10, 20, 50],
    "slow_periods": [20, 50, 100, 200],
}
RSI_GRID = {
    "periods":     [7, 14, 21],
    "oversolds":   [20, 25, 30, 35],
    "overboughts": [65, 70, 75, 80],
}
RSI_DIV_GRID = {
    "periods":     [7, 14, 21],
    "lookbacks":   [10, 14, 20],
    "oversolds":   [25, 30, 35],
    "overboughts": [65, 70, 75],
}
BB_GRID = {
    "periods":  [10, 20, 30, 50],
    "std_devs": [1.5, 2.0, 2.5, 3.0],
}
BO_GRID = {
    "lookbacks": [10, 20, 30, 50, 100],
}
MACD_GRID = {
    "fasts":   [8, 12, 16],
    "slows":   [21, 26, 35],
    "signals": [7, 9, 12],
}
ST_GRID = {
    "atr_periods": [7, 10, 14, 21],
    "multipliers": [2.0, 3.0, 4.0],
}

DEFAULT_LEVERAGES       = [3, 5, 10, 20]
DEFAULT_POSITION_SIZES  = [0.05, 0.10, 0.20]
DEFAULT_TP_PCTS         = [0.5, 1.0, 2.0, 3.0, 5.0]
DEFAULT_SL_PCTS         = [0.3, 0.5, 1.0, 1.5, 2.0]


@dataclass
class StrategyOptConfig:
    # EMA Cross
    ema_fast_periods:    list[int]   | None = None
    ema_slow_periods:    list[int]   | None = None
    # RSI
    rsi_periods:         list[int]   | None = None
    rsi_oversolds:       list[float] | None = None
    rsi_overboughts:     list[float] | None = None
    # RSI Divergence
    rsi_div_periods:     list[int]   | None = None
    rsi_div_lookbacks:   list[int]   | None = None
    rsi_div_oversolds:   list[float] | None = None
    rsi_div_overboughts: list[float] | None = None
    # Bollinger
    bb_periods:          list[int]   | None = None
    bb_std_devs:         list[float] | None = None
    # Breakout
    bo_lookbacks:        list[int]   | None = None
    # MACD
    macd_fasts:          list[int]   | None = None
    macd_slows:          list[int]   | None = None
    macd_signals:        list[int]   | None = None
    # Supertrend
    st_atr_periods:      list[int]   | None = None
    st_multipliers:      list[float] | None = None
    # Trade-Parameter
    leverages:           list[int]   = field(default_factory=lambda: DEFAULT_LEVERAGES)
    position_sizes:      list[float] = field(default_factory=lambda: DEFAULT_POSITION_SIZES)
    tp_pcts:             list[float] = field(default_factory=lambda: DEFAULT_TP_PCTS)
    sl_pcts:             list[float] = field(default_factory=lambda: DEFAULT_SL_PCTS)
    # Allgemein
    initial_capital:     float = 1000.0
    fee_rate:            float = 0.00055
    max_hold_candles:    int   = 1440
    exit_on_signal:      bool  = True
    min_trades:          int   = 5
    score_metric:        str   = "composite"
    # MTF-Filter
    mtf_ema_period:      int   = 50
    # ADX-Filter
    adx_thresholds:      list[float] = field(default_factory=lambda: [25.0])
    adx_mode:            str         = "none"   # "trending" | "ranging" | "none"


# ── ADX-Filter ─────────────────────────────────────────────────────────────────

def apply_adx_filter(
    signals:   np.ndarray,
    adx_arr:   np.ndarray,
    threshold: float,
    mode:      str,
) -> np.ndarray:
    """
    Unterdrückt Signale basierend auf ADX-Bedingung.
    mode='trending': Signal nur wenn ADX >= threshold  (Bollinger-Modus)
    mode='ranging':  Signal nur wenn ADX <  threshold  (RSI-Div-Modus)
    """
    if mode == "trending":
        mask = adx_arr >= threshold
    elif mode == "ranging":
        mask = adx_arr < threshold
    else:
        return signals
    return np.where(mask, signals, 0).astype(int)


# ── MTF-Filter ─────────────────────────────────────────────────────────────────

def apply_mtf_filter(
    primary_df: pd.DataFrame,
    val_dfs:    list[pd.DataFrame],
    signals:    np.ndarray,
    ema_period: int = 50,
) -> np.ndarray:
    """
    Unterdrückt Signale, die nicht mit dem EMA-Trend aller höheren Timeframes übereinstimmen.
    Long-Signal (+1) nur erlaubt, wenn alle höheren TF bullish sind (Close >= EMA).
    Short-Signal (-1) nur erlaubt, wenn alle höheren TF bearish sind.
    """
    if not val_dfs:
        return signals

    signals = signals.copy()

    # Stelle sicher, dass primary_df einen datetime-Index hat
    primary_times = pd.to_datetime(primary_df["datetime"]).reset_index(drop=True)

    for val_df in val_dfs:
        if val_df is None or len(val_df) == 0:
            continue

        val_times = pd.to_datetime(val_df["datetime"]).reset_index(drop=True)
        ema       = val_df["close"].ewm(span=ema_period, adjust=False).mean()
        trend     = np.where(val_df["close"].to_numpy() >= ema.to_numpy(), 1, -1)

        htf = pd.DataFrame({"time": val_times, "trend": trend})
        pri = pd.DataFrame({"time": primary_times})

        merged = pd.merge_asof(
            pri.sort_values("time"),
            htf.sort_values("time"),
            on="time",
            direction="backward",
        ).sort_values("time").reset_index(drop=True)

        htf_trend = merged["trend"].fillna(0).to_numpy(int)

        # Nur Signale durchlassen, die mit dem höheren TF übereinstimmen
        mask = (
            ((signals == 1)  & (htf_trend == 1)) |
            ((signals == -1) & (htf_trend == -1))
        )
        signals = np.where(mask, signals, 0).astype(int)

    return signals


# ── Score-Funktionen ───────────────────────────────────────────────────────────

def _score(r: dict, metric: str) -> float:
    if not r:
        return -999.0
    pf  = r.get("profit_factor", 0)
    pnl = r.get("total_pnl_pct", 0)
    dd  = r.get("max_drawdown_pct", 100)
    wr  = r.get("winrate_pct", 0)

    if metric == "profit_factor":
        return min(pf, 99.0)
    if metric == "total_pnl_pct":
        return pnl
    if metric == "winrate":
        return wr
    if metric == "sharpe":
        return r.get("sharpe_ratio", 0)
    if metric == "scalping":
        # Harte Constraints: PF > 1.5, Trades >= 50, Max-DD <= 20 %
        if pf < 1.5 or r.get("num_trades", 0) < 50 or dd > 20:
            return -999.0
        return pnl * min(pf, 10.0) * max(0.1, 1 - dd / 100)
    # composite (Standard)
    return pnl * min(pf, 10.0) * max(0.1, 1 - dd / 100)


# ── Strategy-Job-Builder ───────────────────────────────────────────────────────

def _build_ema_jobs(cfg: StrategyOptConfig):
    for fast, slow in itertools.product(
        cfg.ema_fast_periods or EMA_GRID["fast_periods"],
        cfg.ema_slow_periods or EMA_GRID["slow_periods"],
    ):
        if fast < slow:
            yield EMACrossStrategy(fast, slow)


def _build_rsi_jobs(cfg: StrategyOptConfig):
    for p, os, ob in itertools.product(
        cfg.rsi_periods     or RSI_GRID["periods"],
        cfg.rsi_oversolds   or RSI_GRID["oversolds"],
        cfg.rsi_overboughts or RSI_GRID["overboughts"],
    ):
        if os < ob:
            yield RSIStrategy(p, os, ob)


def _build_rsi_div_jobs(cfg: StrategyOptConfig):
    for p, lb, os, ob in itertools.product(
        cfg.rsi_div_periods     or RSI_DIV_GRID["periods"],
        cfg.rsi_div_lookbacks   or RSI_DIV_GRID["lookbacks"],
        cfg.rsi_div_oversolds   or RSI_DIV_GRID["oversolds"],
        cfg.rsi_div_overboughts or RSI_DIV_GRID["overboughts"],
    ):
        if os < ob:
            yield RSIDivergenceStrategy(p, lb, os, ob)


def _build_bb_jobs(cfg: StrategyOptConfig):
    for p, s in itertools.product(
        cfg.bb_periods  or BB_GRID["periods"],
        cfg.bb_std_devs or BB_GRID["std_devs"],
    ):
        yield BollingerStrategy(p, s)


def _build_bo_jobs(cfg: StrategyOptConfig):
    for lb in (cfg.bo_lookbacks or BO_GRID["lookbacks"]):
        yield BreakoutStrategy(lb)


def _build_macd_jobs(cfg: StrategyOptConfig):
    for fast, slow, sig in itertools.product(
        cfg.macd_fasts   or MACD_GRID["fasts"],
        cfg.macd_slows   or MACD_GRID["slows"],
        cfg.macd_signals or MACD_GRID["signals"],
    ):
        if fast < slow:
            yield MACDStrategy(fast, slow, sig)


def _build_st_jobs(cfg: StrategyOptConfig):
    for atr, mult in itertools.product(
        cfg.st_atr_periods or ST_GRID["atr_periods"],
        cfg.st_multipliers or ST_GRID["multipliers"],
    ):
        yield SupertrendStrategy(atr, mult)


_STRATEGY_BUILDERS = {
    "ema":           _build_ema_jobs,
    "rsi":           _build_rsi_jobs,
    "rsi_divergence": _build_rsi_div_jobs,
    "bollinger":     _build_bb_jobs,
    "breakout":      _build_bo_jobs,
    "macd":          _build_macd_jobs,
    "supertrend":    _build_st_jobs,
}


# ── Haupt-Optimierungsfunktion ─────────────────────────────────────────────────

def run_strategy_optimization(
    df:           pd.DataFrame,
    strategy_key: str,
    opt_cfg:      StrategyOptConfig,
    progress_cb:  Callable[[int, int], None] | None = None,
    val_dfs:      list[pd.DataFrame] | None = None,
) -> list[dict]:
    """
    Testet alle Kombos aus Strategie-Parametern × Trade-Parametern.
    Gibt eine nach Score sortierte Liste zurück.

    val_dfs: optionale Liste höherer-TF DataFrames für MTF-Filter (z.B. [15m_df, 1h_df]).
    """
    if strategy_key not in _STRATEGY_BUILDERS:
        raise ValueError(
            f"Unbekannte Strategie: {strategy_key}. "
            f"Wähle aus: {list(_STRATEGY_BUILDERS.keys())}"
        )

    opens  = df["open"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    # Marktbedingungen + roher ADX einmalig berechnen
    condition_arr = compute_market_condition_arr(df)
    adx_raw       = compute_adx(df) if opt_cfg.adx_mode != "none" else None
    adx_th_list   = (opt_cfg.adx_thresholds
                     if (opt_cfg.adx_mode != "none" and adx_raw is not None)
                     else [None])

    strategies   = list(_STRATEGY_BUILDERS[strategy_key](opt_cfg))
    trade_combos = list(itertools.product(
        opt_cfg.leverages,
        opt_cfg.position_sizes,
        opt_cfg.tp_pcts,
        opt_cfg.sl_pcts,
    ))

    total = len(strategies) * len(adx_th_list) * len(trade_combos)
    logger.info(
        "Strategie-Optimizer: %d Varianten × %d ADX-Th × %d Trade-Kombos = %d Jobs",
        len(strategies), len(adx_th_list), len(trade_combos), total,
    )

    results: list[dict] = []
    done = 0

    for strategy in strategies:
        try:
            sig_series = strategy.generate_signals(df)
            signals    = sig_series.to_numpy(dtype=int)
        except Exception as exc:
            logger.warning("Signal-Fehler bei %s: %s", strategy, exc)
            done += len(adx_th_list) * len(trade_combos)
            if progress_cb:
                progress_cb(done, total)
            continue

        # MTF-Filter einmalig pro Strategie-Variante anwenden
        if val_dfs:
            signals = apply_mtf_filter(df, val_dfs, signals, opt_cfg.mtf_ema_period)

        for adx_th in adx_th_list:
            # ADX-Filter pro Schwellwert
            if adx_th is not None and adx_raw is not None:
                cur_signals = apply_adx_filter(signals, adx_raw, adx_th, opt_cfg.adx_mode)
            else:
                cur_signals = signals

            for lev, size, tp, sl in trade_combos:
                cfg = StrategyConfig(
                    initial_capital  = opt_cfg.initial_capital,
                    leverage         = lev,
                    position_size    = size,
                    fee_rate         = opt_cfg.fee_rate,
                    take_profit_pct  = tp / 100,
                    stop_loss_pct    = sl / 100,
                    exit_on_signal   = opt_cfg.exit_on_signal,
                    max_hold_candles = opt_cfg.max_hold_candles,
                )

                r = run_strategy_backtest_fast(
                    opens, highs, lows, closes, cur_signals, cfg,
                    condition_arr=condition_arr,
                )

                if r and r.get("num_trades", 0) >= opt_cfg.min_trades:
                    r["score"]         = _score(r, opt_cfg.score_metric)
                    r["strategy"]      = strategy
                    r["config"]        = cfg
                    r["adx_threshold"] = adx_th
                    r["adx_mode"]      = opt_cfg.adx_mode
                    results.append(r)

                done += 1
                if progress_cb:
                    progress_cb(done, total)

    results.sort(key=lambda x: x["score"], reverse=True)
    profitable = sum(1 for r in results if r.get("total_pnl_pct", 0) > 0)
    logger.info("Optimizer fertig: %d Ergebnisse, %d profitable.", len(results), profitable)
    return results


def format_strategy_results_df(results: list[dict], top_n: int = 20) -> pd.DataFrame:
    rows = []
    for r in results[:top_n]:
        cfg   = r["config"]
        strat = r["strategy"]
        adx_th = r.get("adx_threshold")
        adx_mo = r.get("adx_mode", "none")
        adx_str = (f"{adx_th:.0f} ({'T' if adx_mo=='trending' else 'R'})"
                   if adx_th is not None else "–")
        rows.append({
            "Strategie":     str(strat),
            "ADX Th":        adx_str,
            "Hebel":         cfg.leverage,
            "Pos %":         round(cfg.position_size * 100, 0),
            "TP %":          round(cfg.take_profit_pct * 100, 2) if cfg.take_profit_pct else 0,
            "SL %":          round(cfg.stop_loss_pct   * 100, 2) if cfg.stop_loss_pct   else 0,
            "PnL %":         r.get("total_pnl_pct", 0),
            "Profit Factor": r.get("profit_factor", 0),
            "Winrate %":     r.get("winrate_pct", 0),
            "Max DD %":      r.get("max_drawdown_pct", 0),
            "Trades":        r.get("num_trades", 0),
            "L-WR %":        r.get("long_winrate", 0),
            "S-WR %":        r.get("short_winrate", 0),
            "Score":         round(r.get("score", 0), 4),
        })
    return pd.DataFrame(rows)

"""
Strategie-basierter Backtester.

Entry-Logik:
  Signal  1 an Kerze i  =>  Long-Entry  am Open von Kerze i+1
  Signal -1 an Kerze i  =>  Short-Entry am Open von Kerze i+1

Exit-Logik (in Priorität):
  1. TP/Trailing-SL/fixer SL intra-candle (High/Low)
  2. Gegenrichtungs-Signal: Exit am Open der Folgekerze (wenn exit_on_signal=True)
  3. Fallback nach max_hold_candles: Exit am Close

Zusatzfeatures:
  - trailing_stop_pct:  Trailing Stop (folgt dem Preis)
  - volume_filter_period: Signale bei niedrigem Volumen unterdrücken
  - circuit_breaker_pct:  Handelstopp wenn Equity X% unter Startkapital fällt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtester import ExitReason, Side, Trade, _compute_stats, _sharpe


def compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder-geglätteter ADX. Gibt Array mit ADX-Werten zurück."""
    high  = df["high"].to_numpy(float)
    low   = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n     = len(close)

    tr  = np.zeros(n)
    dmp = np.zeros(n)
    dmm = np.zeros(n)

    for i in range(1, n):
        hl     = high[i] - low[i]
        hc     = abs(high[i]  - close[i - 1])
        lc     = abs(low[i]   - close[i - 1])
        tr[i]  = max(hl, hc, lc)
        up     = high[i]  - high[i - 1]
        dn     = low[i - 1] - low[i]
        dmp[i] = up if up > dn and up > 0 else 0.0
        dmm[i] = dn if dn > up and dn > 0 else 0.0

    alpha = 1.0 / period
    tr_s  = pd.Series(tr).ewm(alpha=alpha,  adjust=False).mean().to_numpy()
    dmp_s = pd.Series(dmp).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    dmm_s = pd.Series(dmm).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    safe_tr = np.where(tr_s > 0, tr_s, 1.0)
    di_p    = np.where(tr_s > 0, 100 * dmp_s / safe_tr, 0.0)
    di_m    = np.where(tr_s > 0, 100 * dmm_s / safe_tr, 0.0)
    denom   = di_p + di_m
    dx      = np.where(denom > 0, 100 * np.abs(di_p - di_m) / np.where(denom > 0, denom, 1.0), 0.0)
    adx  = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return adx


def compute_market_condition_arr(
    df: pd.DataFrame,
    period: int    = 14,
    threshold: float = 25.0,
) -> np.ndarray:
    """Gibt 1 (trending, ADX >= threshold) oder 0 (ranging) pro Kerze zurück."""
    adx = compute_adx(df, period)
    return (adx >= threshold).astype(np.int8)
from src.config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_LEVERAGE,
    BACKTEST_MAX_HOLD_CANDLES,
    BACKTEST_POSITION_SIZE,
    BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT,
    BACKTEST_TAKER_FEE,
)
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    initial_capital:      float        = BACKTEST_INITIAL_CAPITAL
    leverage:             int          = BACKTEST_LEVERAGE
    position_size:        float        = BACKTEST_POSITION_SIZE
    fee_rate:             float        = BACKTEST_TAKER_FEE
    take_profit_pct:      float | None = BACKTEST_TAKE_PROFIT_PCT
    stop_loss_pct:        float | None = BACKTEST_STOP_LOSS_PCT
    trailing_stop_pct:    float | None = None   # Trailing Stop, ersetzt/ergänzt fixen SL
    exit_on_signal:       bool         = True
    max_hold_candles:     int          = BACKTEST_MAX_HOLD_CANDLES
    volume_filter_period: int   | None = None   # 0/None = kein Filter
    circuit_breaker_pct:  float | None = None   # z.B. 0.20 = Stopp bei -20%
    max_notional:         float | None = None   # z.B. 1000 = max. 1000 USDT Notional pro Trade


def _apply_volume_filter(signals: np.ndarray, volumes: np.ndarray, period: int) -> np.ndarray:
    """Setzt Signale auf 0 wenn das aktuelle Volumen unter dem Rolling-Durchschnitt liegt."""
    vol_ma   = pd.Series(volumes).rolling(period, min_periods=1).mean().to_numpy()
    vol_mask = volumes >= vol_ma
    return (signals * vol_mask).astype(int)


def _find_exit_strategy(
    opens:          np.ndarray,
    highs:          np.ndarray,
    lows:           np.ndarray,
    closes:         np.ndarray,
    signals:        np.ndarray,
    entry_idx:      int,
    side:           Side,
    tp_price:       float | None,
    sl_price:       float | None,
    trailing_pct:   float | None,
    max_hold:       int,
    exit_on_signal: bool,
) -> tuple[ExitReason, float, int]:
    n   = len(highs)
    end = min(entry_idx + max_hold, n)

    entry_price  = opens[entry_idx]
    best_price   = entry_price
    trail_sl     = None

    if trailing_pct is not None:
        # Startet beim fixen SL — trailing verbessert ihn nur wenn Preis steigt.
        # Kein fixer SL → startet beim Trailing-Abstand vom Entry.
        if sl_price is not None:
            trail_sl = sl_price
        else:
            trail_sl = (entry_price * (1 - trailing_pct) if side == "long"
                        else entry_price * (1 + trailing_pct))

    for j in range(entry_idx, end):
        h = highs[j]
        l = lows[j]

        # ── Trailing Stop aktualisieren (nur verbessern, nie verschlechtern) ──
        if trailing_pct is not None:
            if side == "long" and h > best_price:
                best_price = h
                candidate  = best_price * (1 - trailing_pct)
                if candidate > trail_sl:
                    trail_sl = candidate
            elif side == "short" and l < best_price:
                best_price = l
                candidate  = best_price * (1 + trailing_pct)
                if candidate < trail_sl:
                    trail_sl = candidate

        # ── Effektiven SL ermitteln ───────────────────────────────────────────
        eff_sl = sl_price
        if trail_sl is not None:
            if eff_sl is None:
                eff_sl = trail_sl
            elif side == "long":
                eff_sl = max(eff_sl, trail_sl)
            else:
                eff_sl = min(eff_sl, trail_sl)

        # ── TP/SL prüfen (Priorität 1) ────────────────────────────────────────
        tp_hit = (tp_price is not None) and (h >= tp_price if side == "long" else l <= tp_price)
        sl_hit = (eff_sl   is not None) and (l <= eff_sl   if side == "long" else h >= eff_sl)

        if sl_hit and tp_hit:           # konservativ: SL gewinnt
            return "sl", eff_sl, j
        if tp_hit:
            return "tp", tp_price, j
        if sl_hit:
            return "sl", eff_sl, j

        # ── Signal-basierter Exit (Priorität 2) ───────────────────────────────
        if exit_on_signal and j < n - 1:
            sig = int(signals[j])
            if (side == "long" and sig == -1) or (side == "short" and sig == 1):
                return "signal", float(opens[j + 1]), j + 1

    last = end - 1
    return "timeout", float(closes[last]), last


def run_strategy_backtest(
    df:       pd.DataFrame,
    strategy: BaseStrategy,
    cfg:      StrategyConfig,
) -> dict:
    """Vollständiger Backtest mit Trade-Objekten."""
    if len(df) < 3:
        raise ValueError("Zu wenig Kerzen fuer einen Backtest.")

    logger.info("Strategie-Backtest: %s  |  %dx  |  TP %s  SL %s  Trail %s",
                strategy, cfg.leverage,
                f"{cfg.take_profit_pct*100:.1f}%"   if cfg.take_profit_pct   else "–",
                f"{cfg.stop_loss_pct*100:.1f}%"     if cfg.stop_loss_pct     else "–",
                f"{cfg.trailing_stop_pct*100:.1f}%" if cfg.trailing_stop_pct else "–")

    signals_series = strategy.generate_signals(df)

    opens   = df["open"].to_numpy(dtype=float)
    highs   = df["high"].to_numpy(dtype=float)
    lows    = df["low"].to_numpy(dtype=float)
    closes  = df["close"].to_numpy(dtype=float)
    times   = df["datetime"].astype(str).to_numpy()
    signals = signals_series.to_numpy(dtype=int)

    # Volume-Filter
    if cfg.volume_filter_period and "volume" in df.columns:
        volumes = df["volume"].to_numpy(dtype=float)
        signals = _apply_volume_filter(signals, volumes, cfg.volume_filter_period)
        signals_series = pd.Series(signals, index=df.index)

    equity       = cfg.initial_capital
    trades:       list[Trade] = []
    equity_curve: list[float] = [equity]
    n = len(df)
    i = 0

    while i < n - 1:
        # Circuit Breaker
        if cfg.circuit_breaker_pct and equity < cfg.initial_capital * (1 - cfg.circuit_breaker_pct):
            logger.info("Circuit Breaker ausgelöst: Equity %.2f < Limit %.2f",
                        equity, cfg.initial_capital * (1 - cfg.circuit_breaker_pct))
            break

        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side: Side  = "long" if sig == 1 else "short"
        entry_price = opens[i + 1]
        if entry_price <= 0:
            i += 1
            continue

        tp_price = sl_price = None
        if cfg.take_profit_pct:
            tp_price = (entry_price * (1 + cfg.take_profit_pct) if side == "long"
                        else entry_price * (1 - cfg.take_profit_pct))
        if cfg.stop_loss_pct:
            sl_price = (entry_price * (1 - cfg.stop_loss_pct) if side == "long"
                        else entry_price * (1 + cfg.stop_loss_pct))

        exit_reason, exit_price, exit_idx = _find_exit_strategy(
            opens, highs, lows, closes, signals,
            entry_idx      = i + 1,
            side           = side,
            tp_price       = tp_price,
            sl_price       = sl_price,
            trailing_pct   = cfg.trailing_stop_pct,
            max_hold       = cfg.max_hold_candles,
            exit_on_signal = cfg.exit_on_signal,
        )

        margin   = equity * cfg.position_size
        notional = margin * cfg.leverage
        if cfg.max_notional:
            notional = min(notional, cfg.max_notional)
        raw_pnl  = ((exit_price - entry_price) / entry_price * notional
                    if side == "long"
                    else (entry_price - exit_price) / entry_price * notional)
        fees    = notional * cfg.fee_rate * 2
        net_pnl = raw_pnl - fees
        pnl_pct = net_pnl / margin * 100
        equity += net_pnl

        trades.append(Trade(
            index        = len(trades),
            side         = side,
            entry_time   = times[i + 1],
            exit_time    = times[exit_idx] if exit_idx < len(times) else times[-1],
            entry_price  = entry_price,
            exit_price   = round(exit_price, 6),
            exit_reason  = exit_reason,
            size_usdt    = notional,
            pnl          = net_pnl,
            pnl_pct      = pnl_pct,
            equity_after = equity,
        ))
        equity_curve.append(equity)

        # Signal-Exit: i = exit_idx - 1 damit das Reversal-Signal (das den Exit
        # ausgelöst hat) im nächsten Loop-Durchlauf als Entry geprüft wird.
        # Ohne diesen Fix überspringt die Schleife das -1/+1 Signal und nimmt
        # strukturell nur Long- ODER nur Short-Trades (je nach erstem Signal).
        if exit_reason == "signal":
            i = exit_idx - 1
        else:
            i = exit_idx

        if equity <= 0:
            logger.warning("Kapital aufgebraucht nach %d Trades.", len(trades))
            break

    result = _compute_stats(trades, equity_curve, cfg)  # type: ignore[arg-type]
    result["strategy"] = strategy
    result["signals"]  = signals_series
    return result


def _side_and_condition_stats(
    long_pnls:  list[float],
    short_pnls: list[float],
    trend_pnls: list[float],
    range_pnls: list[float],
) -> dict:
    def _s(pnls: list[float]) -> tuple[int, float, float, float]:
        if not pnls:
            return 0, 0.0, 0.0, 0.0
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p <= 0]
        wr   = len(wins) / len(pnls) * 100
        sl_s = sum(loss)
        pf   = round(abs(sum(wins) / sl_s) if sl_s != 0 else (99.0 if wins else 0.0), 4)
        return len(pnls), round(wr, 2), min(pf, 99.0), round(sum(pnls), 4)

    ln, lwr, lpf, lpnl = _s(long_pnls)
    sn, swr, spf, spnl = _s(short_pnls)
    tn, twr, _,   _    = _s(trend_pnls)
    rn, rwr, _,   _    = _s(range_pnls)
    return {
        "long_trades":      ln,
        "long_winrate":     lwr,
        "long_pf":          lpf,
        "long_total_pnl":   lpnl,
        "short_trades":     sn,
        "short_winrate":    swr,
        "short_pf":         spf,
        "short_total_pnl":  spnl,
        "trending_trades":  tn,
        "trending_winrate": twr,
        "ranging_trades":   rn,
        "ranging_winrate":  rwr,
    }


def run_strategy_backtest_fast(
    opens:         np.ndarray,
    highs:         np.ndarray,
    lows:          np.ndarray,
    closes:        np.ndarray,
    signals:       np.ndarray,
    cfg:           StrategyConfig,
    volumes:       np.ndarray | None = None,
    condition_arr: np.ndarray | None = None,
) -> dict:
    """Schnelle Variante ohne Trade-Objekte – für den Strategie-Optimizer.

    condition_arr: optionales Array mit 1 (trending) / 0 (ranging) pro Kerze.
    """
    from src.backtester import _stats_from_raw

    if cfg.volume_filter_period and volumes is not None:
        signals = _apply_volume_filter(signals, volumes, cfg.volume_filter_period)

    equity       = cfg.initial_capital
    equity_curve = [equity]
    pnls:        list[float] = []
    long_pnls:   list[float] = []
    short_pnls:  list[float] = []
    trend_pnls:  list[float] = []
    range_pnls:  list[float] = []
    tp_c = sl_c = to_c = 0
    n = len(opens)
    i = 0

    while i < n - 1:
        if cfg.circuit_breaker_pct and equity < cfg.initial_capital * (1 - cfg.circuit_breaker_pct):
            break

        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side: Side  = "long" if sig == 1 else "short"
        entry_price = opens[i + 1]
        if entry_price <= 0:
            i += 1
            continue

        tp_price = sl_price = None
        if cfg.take_profit_pct:
            tp_price = (entry_price * (1 + cfg.take_profit_pct) if side == "long"
                        else entry_price * (1 - cfg.take_profit_pct))
        if cfg.stop_loss_pct:
            sl_price = (entry_price * (1 - cfg.stop_loss_pct) if side == "long"
                        else entry_price * (1 + cfg.stop_loss_pct))

        exit_reason, exit_price, exit_idx = _find_exit_strategy(
            opens, highs, lows, closes, signals,
            entry_idx=i + 1, side=side,
            tp_price=tp_price, sl_price=sl_price,
            trailing_pct=cfg.trailing_stop_pct,
            max_hold=cfg.max_hold_candles,
            exit_on_signal=cfg.exit_on_signal,
        )

        margin   = equity * cfg.position_size
        notional = margin * cfg.leverage
        if cfg.max_notional:
            notional = min(notional, cfg.max_notional)
        raw_pnl  = ((exit_price - entry_price) / entry_price * notional
                    if side == "long"
                    else (entry_price - exit_price) / entry_price * notional)
        net_pnl  = raw_pnl - notional * cfg.fee_rate * 2
        equity  += net_pnl
        pnls.append(net_pnl)
        equity_curve.append(equity)

        if side == "long":
            long_pnls.append(net_pnl)
        else:
            short_pnls.append(net_pnl)

        if condition_arr is not None:
            cond = int(condition_arr[i]) if i < len(condition_arr) else 0
            if cond == 1:
                trend_pnls.append(net_pnl)
            else:
                range_pnls.append(net_pnl)

        if exit_reason == "tp":   tp_c += 1
        elif exit_reason == "sl": sl_c += 1
        else:                     to_c += 1

        if exit_reason == "signal":
            i = exit_idx - 1
        else:
            i = exit_idx
        if equity <= 0:
            break

    if not pnls:
        return {}
    result = _stats_from_raw(pnls, equity_curve, cfg, tp_c, sl_c, to_c)  # type: ignore[arg-type]
    result.update(_side_and_condition_stats(long_pnls, short_pnls, trend_pnls, range_pnls))
    return result

"""
Futures-Backtester auf OHLCV-Daten (1m Kerzen).

Regeln:
- Immer nur 1 offene Position gleichzeitig.
- Sequenz (z.B. long, short) wird zyklisch wiederholt.
- Entry:  Open der naechsten Kerze nach dem Signal.
- Exit:   Sobald High/Low einer Kerze den TP oder SL beruehrt.
          Werden TP und SL in derselben Kerze beruehrt, gewinnt SL (konservativ).
- Fallback: nach max_hold_candles wird zum Close-Preis geschlossen.
- Gebuehren auf Nominalwert (Taker Entry + Taker Exit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_LEVERAGE,
    BACKTEST_MAX_HOLD_CANDLES,
    BACKTEST_POSITION_SIZE,
    BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT,
    BACKTEST_TAKER_FEE,
)

logger = logging.getLogger(__name__)

Side = Literal["long", "short"]
ExitReason = Literal["tp", "sl", "timeout", "signal"]


@dataclass
class BacktestConfig:
    initial_capital:       float         = BACKTEST_INITIAL_CAPITAL
    leverage:              int           = BACKTEST_LEVERAGE
    position_size:         float         = BACKTEST_POSITION_SIZE
    fee_rate:              float         = BACKTEST_TAKER_FEE
    sequence:              list[Side]    = field(default_factory=lambda: ["long", "short"])
    take_profit_pct:       float         = BACKTEST_TAKE_PROFIT_PCT
    stop_loss_pct:         float         = BACKTEST_STOP_LOSS_PCT
    max_hold_candles:      int           = BACKTEST_MAX_HOLD_CANDLES
    # Breakeven: SL -> Entry sobald X% des TP-Abstands erreicht (z.B. 0.5 = 50%)
    breakeven_trigger_pct: Optional[float] = None
    # Trailing SL: folgt dem Kurs mit festem Abstand in % (z.B. 0.008 = 0.8%)
    trailing_sl_pct:       Optional[float] = None


@dataclass
class Trade:
    index:        int
    side:         Side
    entry_time:   str
    exit_time:    str
    entry_price:  float
    exit_price:   float
    exit_reason:  ExitReason
    size_usdt:    float
    pnl:          float
    pnl_pct:      float
    equity_after: float


def _find_exit(
    highs:       np.ndarray,
    lows:        np.ndarray,
    closes:      np.ndarray,
    start_idx:   int,
    entry_price: float,
    side:        Side,
    tp_price:    float,
    sl_price:    float,
    max_hold:    int,
) -> tuple[ExitReason, float, int]:
    end = min(start_idx + max_hold, len(highs))
    h = highs[start_idx:end]
    l = lows[start_idx:end]

    if side == "long":
        tp_hits = np.where(h >= tp_price)[0]
        sl_hits = np.where(l <= sl_price)[0]
    else:
        tp_hits = np.where(l <= tp_price)[0]
        sl_hits = np.where(h >= sl_price)[0]

    tp_i = int(tp_hits[0]) if len(tp_hits) else max_hold
    sl_i = int(sl_hits[0]) if len(sl_hits) else max_hold

    if sl_i <= tp_i:
        if sl_i < len(h):
            return "sl", sl_price, start_idx + sl_i
    else:
        if tp_i < len(h):
            return "tp", tp_price, start_idx + tp_i

    last = end - 1
    return "timeout", float(closes[last]), last


def _find_exit_dynamic(
    highs:                 np.ndarray,
    lows:                  np.ndarray,
    closes:                np.ndarray,
    start_idx:             int,
    entry_price:           float,
    side:                  Side,
    tp_price:              float,
    sl_price:              float,
    max_hold:              int,
    breakeven_trigger_pct: Optional[float],
    trailing_sl_pct:       Optional[float],
) -> tuple[ExitReason, float, int]:
    """
    Kerze-fuer-Kerze Exit-Suche mit beweglichem SL.
    Reihenfolge pro Kerze: Trailing update → Breakeven trigger → Exit-Check (SL gewinnt bei Gleichstand).
    """
    end = min(start_idx + max_hold, len(highs))
    current_sl  = sl_price
    be_triggered = False
    best_price   = entry_price

    if breakeven_trigger_pct is not None:
        if side == "long":
            be_price = entry_price + (tp_price - entry_price) * breakeven_trigger_pct
        else:
            be_price = entry_price - (entry_price - tp_price) * breakeven_trigger_pct
    else:
        be_price = None

    for i in range(start_idx, end):
        h = highs[i]
        l = lows[i]

        # 1) Trailing SL nachziehen
        if trailing_sl_pct is not None:
            if side == "long" and h > best_price:
                best_price = h
                new_sl = best_price * (1 - trailing_sl_pct)
                if new_sl > current_sl:
                    current_sl = new_sl
            elif side == "short" and l < best_price:
                best_price = l
                new_sl = best_price * (1 + trailing_sl_pct)
                if new_sl < current_sl:
                    current_sl = new_sl

        # 2) Breakeven aktivieren (nur einmal)
        if be_price is not None and not be_triggered:
            if side == "long" and h >= be_price:
                be_triggered = True
                if entry_price > current_sl:
                    current_sl = entry_price
            elif side == "short" and l <= be_price:
                be_triggered = True
                if entry_price < current_sl:
                    current_sl = entry_price

        # 3) Exit pruefen – SL gewinnt bei Gleichstand (konservativ)
        if side == "long":
            sl_hit = l <= current_sl
            tp_hit = h >= tp_price
        else:
            sl_hit = h >= current_sl
            tp_hit = l <= tp_price

        if sl_hit:
            return "sl", current_sl, i
        if tp_hit:
            return "tp", tp_price, i

    last = end - 1
    return "timeout", float(closes[last]), last


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if len(df) < 3:
        raise ValueError("Zu wenig Kerzen fuer einen Backtest.")

    opens  = df["open"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times  = df["datetime"].astype(str).to_numpy()

    equity     = cfg.initial_capital
    trades: list[Trade] = []
    equity_curve: list[float] = [equity]

    seq_len   = len(cfg.sequence)
    trade_idx = 0
    i = 0

    while i + 1 < len(df):
        side: Side = cfg.sequence[trade_idx % seq_len]
        entry_price = opens[i + 1]
        if entry_price <= 0:
            i += 1
            continue

        if side == "long":
            tp_price = entry_price * (1 + cfg.take_profit_pct)
            sl_price = entry_price * (1 - cfg.stop_loss_pct)
        else:
            tp_price = entry_price * (1 - cfg.take_profit_pct)
            sl_price = entry_price * (1 + cfg.stop_loss_pct)

        use_dynamic = cfg.breakeven_trigger_pct is not None or cfg.trailing_sl_pct is not None
        if use_dynamic:
            exit_reason, exit_price, exit_idx = _find_exit_dynamic(
                highs, lows, closes,
                start_idx             = i + 2,
                entry_price           = entry_price,
                side                  = side,
                tp_price              = tp_price,
                sl_price              = sl_price,
                max_hold              = cfg.max_hold_candles,
                breakeven_trigger_pct = cfg.breakeven_trigger_pct,
                trailing_sl_pct       = cfg.trailing_sl_pct,
            )
        else:
            exit_reason, exit_price, exit_idx = _find_exit(
                highs, lows, closes,
                start_idx   = i + 2,
                entry_price = entry_price,
                side        = side,
                tp_price    = tp_price,
                sl_price    = sl_price,
                max_hold    = cfg.max_hold_candles,
            )

        margin   = equity * cfg.position_size
        notional = margin * cfg.leverage

        if side == "long":
            raw_pnl = (exit_price - entry_price) / entry_price * notional
        else:
            raw_pnl = (entry_price - exit_price) / entry_price * notional

        fees    = notional * cfg.fee_rate * 2
        net_pnl = raw_pnl - fees
        pnl_pct = net_pnl / margin * 100
        equity += net_pnl

        trades.append(Trade(
            index       = len(trades),
            side        = side,
            entry_time  = times[i + 1],
            exit_time   = times[exit_idx] if exit_idx < len(times) else times[-1],
            entry_price = entry_price,
            exit_price  = round(exit_price, 6),
            exit_reason = exit_reason,
            size_usdt   = notional,
            pnl         = net_pnl,
            pnl_pct     = pnl_pct,
            equity_after= equity,
        ))
        equity_curve.append(equity)

        i = exit_idx
        trade_idx += 1

        if equity <= 0:
            logger.warning("Kapital aufgebraucht nach %d Trades.", len(trades))
            break

    return _compute_stats(trades, equity_curve, cfg)


def run_backtest_fast(
    opens: np.ndarray,
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    cfg: BacktestConfig,
) -> dict:
    """
    Schnelle Variante ohne Trade-Objekte – nur fuer den Optimizer.
    Gibt nur die aggregierten Stats zurueck.
    """
    equity     = cfg.initial_capital
    equity_curve: list[float] = [equity]
    pnls: list[float] = []
    tp_count = sl_count = timeout_count = 0

    seq_len   = len(cfg.sequence)
    trade_idx = 0
    i = 0
    n = len(opens)

    while i + 1 < n:
        side: Side = cfg.sequence[trade_idx % seq_len]
        entry_price = opens[i + 1]
        if entry_price <= 0:
            i += 1
            continue

        if side == "long":
            tp_price = entry_price * (1 + cfg.take_profit_pct)
            sl_price = entry_price * (1 - cfg.stop_loss_pct)
        else:
            tp_price = entry_price * (1 - cfg.take_profit_pct)
            sl_price = entry_price * (1 + cfg.stop_loss_pct)

        use_dynamic = cfg.breakeven_trigger_pct is not None or cfg.trailing_sl_pct is not None
        if use_dynamic:
            exit_reason, exit_price, exit_idx = _find_exit_dynamic(
                highs, lows, closes,
                start_idx             = i + 2,
                entry_price           = entry_price,
                side                  = side,
                tp_price              = tp_price,
                sl_price              = sl_price,
                max_hold              = cfg.max_hold_candles,
                breakeven_trigger_pct = cfg.breakeven_trigger_pct,
                trailing_sl_pct       = cfg.trailing_sl_pct,
            )
        else:
            exit_reason, exit_price, exit_idx = _find_exit(
                highs, lows, closes,
                start_idx   = i + 2,
                entry_price = entry_price,
                side        = side,
                tp_price    = tp_price,
                sl_price    = sl_price,
                max_hold    = cfg.max_hold_candles,
            )

        margin   = equity * cfg.position_size
        notional = margin * cfg.leverage

        if side == "long":
            raw_pnl = (exit_price - entry_price) / entry_price * notional
        else:
            raw_pnl = (entry_price - exit_price) / entry_price * notional

        net_pnl = raw_pnl - notional * cfg.fee_rate * 2
        equity += net_pnl
        pnls.append(net_pnl)
        equity_curve.append(equity)

        if exit_reason == "tp":      tp_count += 1
        elif exit_reason == "sl":    sl_count += 1
        else:                        timeout_count += 1

        i = exit_idx
        trade_idx += 1

        if equity <= 0:
            break

    return _stats_from_raw(pnls, equity_curve, cfg, tp_count, sl_count, timeout_count)


def _stats_from_raw(
    pnls: list[float],
    equity_curve: list[float],
    cfg: BacktestConfig,
    tp_count: int,
    sl_count: int,
    timeout_count: int,
) -> dict:
    if not pnls:
        return {}

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    final_eq  = equity_curve[-1]
    total_pct = (final_eq - cfg.initial_capital) / cfg.initial_capital * 100
    winrate   = len(wins) / len(pnls) * 100

    peak = max_dd = 0.0
    peak = equity_curve[0]
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    sum_wins   = sum(wins)
    sum_losses = sum(losses)
    pf = abs(sum_wins / sum_losses) if sum_losses != 0 else float("inf")

    return {
        "final_balance":    round(final_eq, 4),
        "total_pnl_pct":    round(total_pct, 2),
        "num_trades":       len(pnls),
        "winrate_pct":      round(winrate, 2),
        "profit_factor":    round(pf, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "tp_count":         tp_count,
        "sl_count":         sl_count,
        "timeout_count":    timeout_count,
    }


def _sharpe(pnl_pcts: list[float]) -> float:
    n = len(pnl_pcts)
    if n < 2:
        return 0.0
    mean = sum(pnl_pcts) / n
    var  = sum((p - mean) ** 2 for p in pnl_pcts) / (n - 1)
    std  = var ** 0.5
    return round(mean / std, 4) if std > 0 else 0.0


def _compute_stats(trades: list[Trade], equity_curve: list[float], cfg: BacktestConfig) -> dict:
    if not trades:
        return {"error": "Keine Trades ausgefuehrt."}

    pnls     = [t.pnl     for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    tp_c  = sum(1 for t in trades if t.exit_reason == "tp")
    sl_c  = sum(1 for t in trades if t.exit_reason == "sl")
    to_c  = sum(1 for t in trades if t.exit_reason in ("timeout", "signal"))

    stats  = _stats_from_raw(pnls, equity_curve, cfg, tp_c, sl_c, to_c)
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        **stats,
        "initial_capital": cfg.initial_capital,
        "total_pnl":       round(equity_curve[-1] - cfg.initial_capital, 4),
        "avg_win":         round(sum(wins)   / len(wins)   if wins   else 0, 4),
        "avg_loss":        round(sum(losses) / len(losses) if losses else 0, 4),
        "sharpe_ratio":    _sharpe(pnl_pcts),
        "equity_curve":    equity_curve,
        "trades":          trades,
        "config":          cfg,
    }

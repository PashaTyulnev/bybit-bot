"""
ATR-basierter Backtest-Engine für Anti-Randomness Tests.

Unterschied zu strategy_backtester.py:
  - TP und SL werden pro Trade dynamisch aus dem ATR berechnet
    (ATR[i] × Multiplikator), nicht als fixer Prozentsatz.
  - Slippage wird auf Einstiegs- und Ausstiegspreis angewandt.
  - Ergebnis-Dict ist kompatibel mit dem bestehenden Format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from src.strategy_backtester import _find_exit_strategy


@dataclass
class AtrBacktestConfig:
    initial_capital: float = 1000.0
    risk_per_trade:  float = 0.01       # Anteil des Equity als Margin (1 %)
    leverage:        int   = 1
    fee_rate:        float = 0.0006     # Taker-Fee
    slippage_rate:   float = 0.0002     # einseitiger Slippage (× 2 = Round-Trip)
    atr_multiplier:  float = 1.5        # SL-Abstand = ATR × Multiplikator
    rr_ratio:        float = 2.0        # TP-Abstand = SL × RR
    max_hold_candles: int  = 500        # Zwangs-Exit-Limit
    exit_on_signal:  bool  = True


def run_atr_backtest(
    df:      pd.DataFrame,
    signals: np.ndarray,
    atr_arr: np.ndarray,
    cfg:     AtrBacktestConfig,
) -> dict:
    """
    Führt einen ATR-basierten Backtest durch.

    signals:  Array mit 1 (Long), -1 (Short), 0 (neutral) — gleiche Länge wie df.
    atr_arr:  ATR-Array, gleiche Länge wie df. ATR[i] wird bei Entry an Kerze i verwendet.
    """
    opens  = df["open"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    n      = len(opens)

    equity       = cfg.initial_capital
    equity_curve = [equity]
    pnls:   list[float] = []
    tp_c = sl_c = to_c = sig_c = 0

    # Für Sharpe/Sortino
    long_pnls:  list[float] = []
    short_pnls: list[float] = []

    i = 0
    while i < n - 1:
        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side        = "long" if sig == 1 else "short"
        entry_raw   = opens[i + 1]
        if entry_raw <= 0:
            i += 1
            continue

        # Slippage auf Entry-Preis
        slip        = entry_raw * cfg.slippage_rate
        entry_price = entry_raw + slip if side == "long" else entry_raw - slip

        # ATR-basierte TP/SL
        atr_val  = float(atr_arr[i]) if i < len(atr_arr) else 0.0
        if atr_val <= 0:
            i += 1
            continue
        sl_dist  = atr_val * cfg.atr_multiplier
        tp_dist  = sl_dist * cfg.rr_ratio
        tp_price = entry_price + tp_dist if side == "long" else entry_price - tp_dist
        sl_price = entry_price - sl_dist if side == "long" else entry_price + sl_dist

        exit_reason, exit_raw, exit_idx = _find_exit_strategy(
            opens, highs, lows, closes, signals,
            entry_idx      = i + 1,
            side           = side,
            tp_price       = tp_price,
            sl_price       = sl_price,
            trailing_pct   = None,
            max_hold       = cfg.max_hold_candles,
            exit_on_signal = cfg.exit_on_signal,
        )

        # Slippage auf Exit-Preis (TP/SL haben bereits korrekten Preis, nur Timeout/Signal)
        if exit_reason in ("timeout", "signal"):
            exit_price = exit_raw - slip if side == "long" else exit_raw + slip
        else:
            exit_price = exit_raw   # TP/SL: Bybit füllt am gesetzten Preis

        margin   = equity * cfg.risk_per_trade
        notional = margin * cfg.leverage
        raw_pnl  = (
            (exit_price - entry_price) / entry_price * notional
            if side == "long"
            else (entry_price - exit_price) / entry_price * notional
        )
        fees     = notional * cfg.fee_rate * 2
        net_pnl  = raw_pnl - fees
        equity  += net_pnl
        pnls.append(net_pnl)
        equity_curve.append(equity)

        if side == "long":  long_pnls.append(net_pnl)
        else:               short_pnls.append(net_pnl)

        if   exit_reason == "tp":     tp_c  += 1
        elif exit_reason == "sl":     sl_c  += 1
        elif exit_reason == "signal": sig_c += 1
        else:                         to_c  += 1

        i = (exit_idx - 1) if exit_reason == "signal" else exit_idx
        if equity <= 0:
            break

    if not pnls:
        return {}

    return _compute_result(pnls, equity_curve, cfg, tp_c, sl_c, to_c, sig_c,
                           long_pnls, short_pnls)


def _compute_result(
    pnls:        list[float],
    eq_curve:    list[float],
    cfg:         AtrBacktestConfig,
    tp_c:        int,
    sl_c:        int,
    to_c:        int,
    sig_c:       int,
    long_pnls:   list[float],
    short_pnls:  list[float],
) -> dict:
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    final_eq  = eq_curve[-1]
    total_pct = (final_eq - cfg.initial_capital) / cfg.initial_capital * 100
    winrate   = len(wins) / len(pnls) * 100 if pnls else 0.0

    peak = cfg.initial_capital
    max_dd = 0.0
    for e in eq_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    sum_w = sum(wins)
    sum_l = sum(losses)
    pf    = abs(sum_w / sum_l) if sum_l != 0 else (99.0 if wins else 0.0)
    pf    = min(pf, 99.0)

    pnl_pcts = [p / (cfg.initial_capital * cfg.risk_per_trade) * 100 for p in pnls]
    sharpe   = _sharpe(pnl_pcts)
    sortino  = _sortino(pnl_pcts)

    return {
        "final_balance":    round(final_eq, 4),
        "total_pnl":        round(final_eq - cfg.initial_capital, 4),
        "total_pnl_pct":    round(total_pct, 2),
        "num_trades":       len(pnls),
        "winrate_pct":      round(winrate, 2),
        "profit_factor":    round(pf, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio":     round(sharpe, 4),
        "sortino_ratio":    round(sortino, 4),
        "tp_count":         tp_c,
        "sl_count":         sl_c,
        "timeout_count":    to_c,
        "signal_exit_count": sig_c,
        "avg_win":   round(np.mean(wins)   if wins   else 0.0, 4),
        "avg_loss":  round(np.mean(losses) if losses else 0.0, 4),
        "long_trades":  len(long_pnls),
        "short_trades": len(short_pnls),
        "long_pf":  round(
            abs(sum(w for w in long_pnls if w > 0) /
                sum(l for l in long_pnls if l <= 0))
            if any(l <= 0 for l in long_pnls) else 99.0, 4
        ),
        "short_pf": round(
            abs(sum(w for w in short_pnls if w > 0) /
                sum(l for l in short_pnls if l <= 0))
            if any(l <= 0 for l in short_pnls) else 99.0, 4
        ),
    }


def _sharpe(pnl_pcts: list[float]) -> float:
    arr = np.array(pnl_pcts)
    if len(arr) < 2:
        return 0.0
    std = arr.std()
    return float(arr.mean() / std) if std > 0 else 0.0


def _sortino(pnl_pcts: list[float]) -> float:
    arr      = np.array(pnl_pcts)
    if len(arr) < 2:
        return 0.0
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 99.0
    ds_std   = np.std(downside)
    return float(arr.mean() / ds_std) if ds_std > 0 else 0.0

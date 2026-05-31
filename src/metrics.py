"""
Erweiterte Performance-Metriken für Backtest-Ergebnisse.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtester import Trade


def sortino_ratio(pnl_pcts: list[float]) -> float:
    """Mean / Downside-Std (nur negative Returns)."""
    arr = np.array(pnl_pcts)
    if len(arr) < 2:
        return 0.0
    mean     = arr.mean()
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 99.0
    ds_std = np.std(downside)
    return float(mean / ds_std) if ds_std > 0 else 0.0


def calmar_ratio(total_pnl_pct: float, max_dd_pct: float) -> float:
    """Gesamt-PnL / Max-Drawdown (vereinfacht, nicht annualisiert)."""
    if max_dd_pct <= 0:
        return 0.0
    return round(total_pnl_pct / max_dd_pct, 4)


def win_loss_streaks(pnl_list: list[float]) -> dict:
    """Maximale Gewinns- und Verlust-Serie."""
    max_win = max_loss = cur_win = cur_loss = 0
    for p in pnl_list:
        if p > 0:
            cur_win  += 1;  cur_loss = 0
            max_win   = max(max_win, cur_win)
        elif p < 0:
            cur_loss += 1;  cur_win  = 0
            max_loss  = max(max_loss, cur_loss)
        else:
            cur_win = cur_loss = 0
    return {"max_win_streak": max_win, "max_loss_streak": max_loss}


def monthly_pnl(trades: list["Trade"]) -> pd.DataFrame:
    """Monatliche PnL-Zusammenfassung aus Trade-Objekten."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        try:
            month = str(t.entry_time)[:7]   # YYYY-MM
            rows.append({"Monat": month, "PnL": t.pnl, "Gewinn": t.pnl > 0})
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("Monat").agg(
        PnL_USDT   = ("PnL",    "sum"),
        Trades     = ("PnL",    "count"),
        Gewinne    = ("Gewinn", "sum"),
    ).reset_index()
    agg["Winrate %"] = (agg["Gewinne"] / agg["Trades"] * 100).round(1)
    agg["PnL_USDT"]  = agg["PnL_USDT"].round(4)
    return agg


def extended_metrics(result: dict) -> dict:
    """
    Berechnet Sortino, Calmar, Streaks aus einem Backtest-Result-Dict.
    Fügt die Werte dem Result-Dict hinzu und gibt es zurück.
    """
    trades = result.get("trades", [])
    pnl_pcts = [t.pnl_pct for t in trades] if trades else []
    pnls     = [t.pnl     for t in trades] if trades else []

    result["sortino_ratio"] = sortino_ratio(pnl_pcts)
    result["calmar_ratio"]  = calmar_ratio(
        result.get("total_pnl_pct", 0),
        result.get("max_drawdown_pct", 0),
    )
    streaks = win_loss_streaks(pnls)
    result["max_win_streak"]  = streaks["max_win_streak"]
    result["max_loss_streak"] = streaks["max_loss_streak"]
    result["monthly_pnl"]     = monthly_pnl(trades)
    return result

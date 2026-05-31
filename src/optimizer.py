"""
Grid-Search-Optimizer: testet alle Parameterkombinationen und liefert die besten.

Verwendung:
    python -m src.optimizer --csv data/raw/BTC_USDT_USDT_1m.csv --top 10
"""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.backtester import BacktestConfig, run_backtest_fast

logger = logging.getLogger(__name__)

Side = Literal["long", "short"]

# ── Standard-Suchraum ─────────────────────────────────────────────────────────

DEFAULT_SEQUENCES: list[list[Side]] = [
    ["long"],
    ["short"],
    ["long", "short"],
    ["short", "long"],
    ["long", "long", "short"],
    ["short", "short", "long"],
    ["long", "short", "short"],
]

DEFAULT_LEVERAGES:       list[int]   = [3, 5, 10, 20]
DEFAULT_POSITION_SIZES:  list[float] = [0.05, 0.10, 0.20]
DEFAULT_TP_PCTS:         list[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
DEFAULT_SL_PCTS:         list[float] = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


@dataclass
class OptimizeConfig:
    sequences:        list[list[Side]] = field(default_factory=lambda: DEFAULT_SEQUENCES)
    leverages:        list[int]        = field(default_factory=lambda: DEFAULT_LEVERAGES)
    position_sizes:   list[float]      = field(default_factory=lambda: DEFAULT_POSITION_SIZES)
    tp_pcts:          list[float]      = field(default_factory=lambda: DEFAULT_TP_PCTS)
    sl_pcts:          list[float]      = field(default_factory=lambda: DEFAULT_SL_PCTS)
    initial_capital:  float            = 1000.0
    fee_rate:         float            = 0.00055
    max_hold_candles: int              = 1440
    min_trades:       int              = 10     # weniger Trades → ignoriert
    score_metric:     str              = "composite"  # composite | profit_factor | total_pnl_pct


def _score(result: dict, metric: str) -> float:
    if not result or result.get("num_trades", 0) == 0:
        return -999.0
    pf  = result["profit_factor"]
    pnl = result["total_pnl_pct"]
    dd  = result["max_drawdown_pct"]
    wr  = result["winrate_pct"]

    if metric == "profit_factor":
        return pf if pf != float("inf") else 99.0
    if metric == "total_pnl_pct":
        return pnl
    if metric == "winrate":
        return wr
    # composite: balanciert PnL, PF und bestraft hohen Drawdown
    pf_capped = min(pf, 10.0)
    return pnl * pf_capped * max(0.1, 1 - dd / 100)


def _run_single(args: tuple) -> dict | None:
    """Wird in Worker-Prozessen ausgefuehrt."""
    opens, highs, lows, closes, cfg, score_metric, min_trades = args
    try:
        result = run_backtest_fast(opens, highs, lows, closes, cfg)
        if not result or result.get("num_trades", 0) < min_trades:
            return None
        result["score"]  = _score(result, score_metric)
        result["config"] = cfg
        return result
    except Exception:
        return None


def run_optimization(
    df: pd.DataFrame,
    opt_cfg: OptimizeConfig,
    progress_cb: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> list[dict]:
    """
    Fuehrt den Grid-Search durch und gibt eine nach Score sortierte Liste zurueck.

    progress_cb(done, total) wird nach jedem abgeschlossenen Job aufgerufen.
    workers=1 fuer sequenziellen Betrieb (Streamlit-kompatibel).
    """
    opens  = df["open"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    # Alle Kombinationen aufbauen
    combos = list(itertools.product(
        opt_cfg.sequences,
        opt_cfg.leverages,
        opt_cfg.position_sizes,
        opt_cfg.tp_pcts,
        opt_cfg.sl_pcts,
    ))
    total = len(combos)
    logger.info("Optimizer: %d Kombinationen", total)

    jobs = []
    for seq, lev, size, tp, sl in combos:
        cfg = BacktestConfig(
            initial_capital  = opt_cfg.initial_capital,
            leverage         = lev,
            position_size    = size,
            fee_rate         = opt_cfg.fee_rate,
            sequence         = list(seq),
            take_profit_pct  = tp / 100,
            stop_loss_pct    = sl / 100,
            max_hold_candles = opt_cfg.max_hold_candles,
        )
        jobs.append((opens, highs, lows, closes, cfg, opt_cfg.score_metric, opt_cfg.min_trades))

    results: list[dict] = []
    done = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_single, j): j for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)
                done += 1
                if progress_cb:
                    progress_cb(done, total)
    else:
        for j in jobs:
            r = _run_single(j)
            if r:
                results.append(r)
            done += 1
            if progress_cb:
                progress_cb(done, total)

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info("Optimizer: %d profitable Konfigurationen gefunden.", sum(1 for r in results if r["total_pnl_pct"] > 0))
    return results


def format_results_df(results: list[dict], top_n: int = 20) -> pd.DataFrame:
    rows = []
    for r in results[:top_n]:
        cfg = r["config"]
        rows.append({
            "Sequenz":        " → ".join(cfg.sequence),
            "Hebel":          cfg.leverage,
            "Pos-Groesse %":  round(cfg.position_size * 100, 0),
            "TP %":           round(cfg.take_profit_pct * 100, 2),
            "SL %":           round(cfg.stop_loss_pct * 100, 2),
            "PnL %":          r["total_pnl_pct"],
            "Profit Factor":  r["profit_factor"],
            "Winrate %":      r["winrate_pct"],
            "Max DD %":       r["max_drawdown_pct"],
            "Trades":         r["num_trades"],
            "TP Hits":        r["tp_count"],
            "SL Hits":        r["sl_count"],
            "Score":          round(r["score"], 4),
        })
    return pd.DataFrame(rows)

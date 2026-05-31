"""
Vergleicht alle Strategien auf denselben Daten.

Verwendung:
    python -m src.compare_strategies
    python -m src.compare_strategies --csv data/raw/BTC_USDT_USDT_1m.csv --leverage 10 --charts
"""

import argparse
import logging
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import pandas as pd

from src.config import (
    BACKTEST_INITIAL_CAPITAL, BACKTEST_LEVERAGE,
    BACKTEST_POSITION_SIZE, BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT, BACKTEST_TAKER_FEE,
)
from src.reporting import compare_to_df, export_trades_csv, save_equity_chart
from src.strategies import (
    BollingerStrategy, BreakoutStrategy, EMACrossStrategy, RSIStrategy,
)
from src.strategy_backtester import StrategyConfig, run_strategy_backtest


def default_strategies():
    return [
        EMACrossStrategy(fast_period=20, slow_period=50),
        EMACrossStrategy(fast_period=50, slow_period=200),
        RSIStrategy(period=14, oversold=30, overbought=70),
        RSIStrategy(period=14, oversold=25, overbought=75),
        BollingerStrategy(period=20, std_dev=2.0),
        BollingerStrategy(period=20, std_dev=1.5),
        BreakoutStrategy(lookback=50),
        BreakoutStrategy(lookback=20),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Alle Strategien vergleichen.")
    p.add_argument("--csv",      default="data/raw/BTC_USDT_USDT_1m.csv")
    p.add_argument("--leverage", type=int,   default=BACKTEST_LEVERAGE)
    p.add_argument("--size",     type=float, default=BACKTEST_POSITION_SIZE)
    p.add_argument("--capital",  type=float, default=BACKTEST_INITIAL_CAPITAL)
    p.add_argument("--tp",       type=float, default=BACKTEST_TAKE_PROFIT_PCT * 100)
    p.add_argument("--sl",       type=float, default=BACKTEST_STOP_LOSS_PCT * 100)
    p.add_argument("--charts",   action="store_true", help="Charts fuer alle Strategien speichern")
    p.add_argument("--export",   action="store_true", help="Trades fuer alle Strategien als CSV speichern")
    args = p.parse_args()

    logger.info("Lade CSV: %s", args.csv)
    df = pd.read_csv(args.csv)

    cfg = StrategyConfig(
        initial_capital = args.capital,
        leverage        = args.leverage,
        position_size   = args.size,
        fee_rate        = BACKTEST_TAKER_FEE,
        take_profit_pct = args.tp / 100,
        stop_loss_pct   = args.sl / 100,
    )

    strategies = default_strategies()
    results: dict[str, dict] = {}

    for strat in strategies:
        name = str(strat)
        logger.info("Teste: %s", name)
        r = run_strategy_backtest(df, strat, cfg)
        results[name] = r

        stem = f"{strat.params_str()}_{cfg.leverage}x"
        if args.export and "error" not in r:
            export_trades_csv(r, stem)
        if args.charts and "error" not in r:
            title = f"{strat}  |  {cfg.leverage}x  |  TP {cfg.take_profit_pct*100:.1f}%  SL {cfg.stop_loss_pct*100:.1f}%"
            save_equity_chart(r, title, stem)

    # ── Vergleichstabelle ausgeben ─────────────────────────────────────────────
    comp = compare_to_df(results)

    sep = "=" * 90
    print(f"\n{sep}")
    print("STRATEGIE-VERGLEICH")
    print(f"Leverage: {cfg.leverage}x  |  TP: {cfg.take_profit_pct*100:.1f}%  |  SL: {cfg.stop_loss_pct*100:.1f}%  |  Kapital: {cfg.initial_capital:,.0f} USDT")
    print(sep)
    print(comp.to_string(index=False,
        formatters={
            "PnL %":         lambda x: f"{x:+.2f}%",
            "Final Balance": lambda x: f"{x:,.2f}",
            "Winrate %":     lambda x: f"{x:.1f}%",
            "Profit Factor": lambda x: f"{x:.4f}",
            "Sharpe":        lambda x: f"{x:.4f}",
            "Max DD %":      lambda x: f"{x:.2f}%",
            "Avg Win":       lambda x: f"{x:+.4f}",
            "Avg Loss":      lambda x: f"{x:+.4f}",
        }
    ))
    print(sep)

    best = comp.iloc[0]
    print(f"\nBeste Strategie: {best['Strategie']}  =>  PnL {best['PnL %']:+.2f}%")

    if args.charts:
        print(f"Charts gespeichert unter: data/charts/")
    if args.export:
        print(f"CSVs gespeichert unter:   data/results/")


if __name__ == "__main__":
    main()

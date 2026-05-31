"""
Strategie-Backtest per CLI.

Beispiele:
    python -m src.run_strategy --strategy ema --fast 20 --slow 50 --leverage 10
    python -m src.run_strategy --strategy rsi --period 14 --oversold 30 --overbought 70
    python -m src.run_strategy --strategy bollinger --period 20 --stddev 2.0
    python -m src.run_strategy --strategy breakout --lookback 50
    python -m src.run_strategy --strategy ema --fast 20 --slow 50 --export --chart
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
    BACKTEST_MAX_HOLD_CANDLES, BACKTEST_POSITION_SIZE,
    BACKTEST_STOP_LOSS_PCT, BACKTEST_TAKE_PROFIT_PCT, BACKTEST_TAKER_FEE,
)
from src.reporting import export_trades_csv, save_equity_chart
from src.strategies import BollingerStrategy, BreakoutStrategy, EMACrossStrategy, RSIStrategy
from src.strategy_backtester import StrategyConfig, run_strategy_backtest


def build_strategy(args):
    s = args.strategy.lower()
    if s in ("ema", "ema_cross", "emacross"):
        return EMACrossStrategy(fast_period=args.fast, slow_period=args.slow)
    if s == "rsi":
        return RSIStrategy(period=args.period, oversold=args.oversold, overbought=args.overbought)
    if s in ("bb", "bollinger"):
        return BollingerStrategy(period=args.period, std_dev=args.stddev)
    if s in ("bo", "breakout"):
        return BreakoutStrategy(lookback=args.lookback)
    raise ValueError(f"Unbekannte Strategie: {args.strategy}")


def print_result(result: dict) -> None:
    sep = "=" * 58
    s   = result.get("strategy")
    cfg = result["config"]
    print(f"\n{sep}")
    print(f"STRATEGIE-BACKTEST:  {s}")
    print(sep)
    print(f"Hebel:         {cfg.leverage}x")
    print(f"Positionsgroesse: {cfg.position_size*100:.0f}%")
    if cfg.take_profit_pct:
        print(f"Take Profit:   {cfg.take_profit_pct*100:.2f}%")
    if cfg.stop_loss_pct:
        print(f"Stop Loss:     {cfg.stop_loss_pct*100:.2f}%")
    print(f"Exit on Signal:{cfg.exit_on_signal}")
    print(sep)
    print(f"Startkapital:  {result['initial_capital']:>12,.2f} USDT")
    print(f"Endkapital:    {result['final_balance']:>12,.4f} USDT")
    print(f"Gesamt-PnL:    {result['total_pnl']:>+12,.4f} USDT  ({result['total_pnl_pct']:+.2f}%)")
    print(sep)
    tc = result.get("tp_count", 0)
    sc = result.get("sl_count", 0)
    oc = result.get("timeout_count", 0)
    print(f"Trades:        {result['num_trades']:>6}  (TP:{tc}  SL:{sc}  Other:{oc})")
    print(f"Winrate:       {result['winrate_pct']:>6.2f}%")
    print(f"Avg Win:       {result['avg_win']:>+12,.4f} USDT")
    print(f"Avg Loss:      {result['avg_loss']:>+12,.4f} USDT")
    print(f"Profit Factor: {result['profit_factor']:>8.4f}")
    print(f"Sharpe Ratio:  {result.get('sharpe_ratio', 0):>8.4f}")
    print(f"Max Drawdown:  {result['max_drawdown_pct']:>6.2f}%")
    print(sep)


def main() -> None:
    p = argparse.ArgumentParser(description="Strategie-Backtest auf OHLCV-Daten.")
    p.add_argument("--csv",        default="data/raw/BTC_USDT_USDT_1m.csv")
    p.add_argument("--strategy",   default="ema",
                   choices=["ema", "ema_cross", "emacross", "rsi", "bb",
                            "bollinger", "bo", "breakout"])
    # EMA
    p.add_argument("--fast",       type=int,   default=20)
    p.add_argument("--slow",       type=int,   default=50)
    # RSI + Bollinger
    p.add_argument("--period",     type=int,   default=14)
    p.add_argument("--oversold",   type=float, default=30.0)
    p.add_argument("--overbought", type=float, default=70.0)
    p.add_argument("--stddev",     type=float, default=2.0)
    # Breakout
    p.add_argument("--lookback",   type=int,   default=50)
    # Standard
    p.add_argument("--leverage",   type=int,   default=BACKTEST_LEVERAGE)
    p.add_argument("--size",       type=float, default=BACKTEST_POSITION_SIZE)
    p.add_argument("--capital",    type=float, default=BACKTEST_INITIAL_CAPITAL)
    p.add_argument("--fee",        type=float, default=BACKTEST_TAKER_FEE)
    p.add_argument("--tp",         type=float, default=BACKTEST_TAKE_PROFIT_PCT * 100)
    p.add_argument("--sl",         type=float, default=BACKTEST_STOP_LOSS_PCT * 100)
    p.add_argument("--no-tp",      action="store_true", help="Kein Take-Profit")
    p.add_argument("--no-sl",      action="store_true", help="Kein Stop-Loss")
    p.add_argument("--no-signal-exit", action="store_true",
                   help="Nur SL/TP als Exit, kein Signal-Exit")
    p.add_argument("--max-hold",   type=int,   default=BACKTEST_MAX_HOLD_CANDLES)
    p.add_argument("--export",     action="store_true", help="Trades als CSV speichern")
    p.add_argument("--chart",      action="store_true", help="Equity-Chart als PNG speichern")
    args = p.parse_args()

    strategy = build_strategy(args)

    cfg = StrategyConfig(
        initial_capital  = args.capital,
        leverage         = args.leverage,
        position_size    = args.size,
        fee_rate         = args.fee,
        take_profit_pct  = None if args.no_tp else args.tp / 100,
        stop_loss_pct    = None if args.no_sl else args.sl / 100,
        exit_on_signal   = not args.no_signal_exit,
        max_hold_candles = args.max_hold,
    )

    logger.info("Lade CSV: %s", args.csv)
    df = pd.read_csv(args.csv)

    result = run_strategy_backtest(df, strategy, cfg)
    if "error" in result:
        logger.error(result["error"])
        raise SystemExit(1)

    print_result(result)

    stem = f"{strategy.params_str()}_{cfg.leverage}x"
    if args.export:
        path = export_trades_csv(result, stem)
        print(f"CSV:   {path}")
    if args.chart:
        title = f"{strategy}  |  {cfg.leverage}x  |  TP {cfg.take_profit_pct*100:.1f}%  SL {cfg.stop_loss_pct*100:.1f}%" \
                if cfg.take_profit_pct and cfg.stop_loss_pct else str(strategy)
        path  = save_equity_chart(result, title, stem)
        print(f"Chart: {path}")


if __name__ == "__main__":
    main()

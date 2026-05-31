"""
Backtest ausfuehren – mit optionalem Vergleich Baseline vs. Breakeven vs. Trailing vs. Beides.

Verwendung:
    python -m src.run_backtest
    python -m src.run_backtest --breakeven 0.5 --trailing-sl 0.8
    python -m src.run_backtest --compare
    python -m src.run_backtest --sequence long short --leverage 10 --tp 2.0 --sl 1.0 --compare
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
from src.backtester import BacktestConfig, run_backtest
from src.config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_LEVERAGE,
    BACKTEST_MAX_HOLD_CANDLES,
    BACKTEST_POSITION_SIZE,
    BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT,
    BACKTEST_TAKER_FEE,
)


def print_results(r: dict, label: str = "BACKTEST RESULTS") -> None:
    sep = "=" * 60
    cfg = r["config"]
    print(f"\n{sep}")
    print(label)
    print(sep)
    print(f"Sequence:       {' -> '.join(cfg.sequence)}  (zyklisch)")
    print(f"Leverage:       {cfg.leverage}x")
    print(f"Position size:  {cfg.position_size*100:.0f}% of equity per trade")
    print(f"Take Profit:    {cfg.take_profit_pct*100:.2f}%")
    print(f"Stop Loss:      {cfg.stop_loss_pct*100:.2f}%")
    if cfg.breakeven_trigger_pct is not None:
        print(f"Breakeven SL:   bei {cfg.breakeven_trigger_pct*100:.0f}% des TP-Abstands -> SL auf Entry")
    if cfg.trailing_sl_pct is not None:
        print(f"Trailing SL:    {cfg.trailing_sl_pct*100:.3f}% Abstand zum Hoechstkurs")
    print(f"Fee rate:       {cfg.fee_rate*100:.4f}% (taker, entry+exit)")
    print(f"Max hold:       {cfg.max_hold_candles} candles (Fallback)")
    print(sep)
    print(f"Initial capital:  {r['initial_capital']:>12,.2f} USDT")
    print(f"Final balance:    {r['final_balance']:>12,.4f} USDT")
    print(f"Total PnL:        {r['total_pnl']:>+12,.4f} USDT  ({r['total_pnl_pct']:+.2f}%)")
    print(sep)
    print(f"Trades:           {r['num_trades']:>6}  (TP: {r['tp_count']}  SL: {r['sl_count']}  Timeout: {r['timeout_count']})")
    print(f"Winrate:          {r['winrate_pct']:>6.2f}%")
    print(f"Avg win:          {r['avg_win']:>+12,.4f} USDT")
    print(f"Avg loss:         {r['avg_loss']:>+12,.4f} USDT")
    print(f"Profit factor:    {r['profit_factor']:>8.4f}")
    print(f"Max drawdown:     {r['max_drawdown_pct']:>6.2f}%")
    print(sep)

    trades = r["trades"]
    if trades:
        t = trades[0]
        print(f"First trade:  {t.entry_time}  {t.side.upper():<5}  "
              f"entry {t.entry_price:.2f}  exit {t.exit_price:.2f}  "
              f"[{t.exit_reason.upper()}]  PnL {t.pnl:+.4f}")
        t = trades[-1]
        print(f"Last  trade:  {t.entry_time}  {t.side.upper():<5}  "
              f"entry {t.entry_price:.2f}  exit {t.exit_price:.2f}  "
              f"[{t.exit_reason.upper()}]  PnL {t.pnl:+.4f}")


def print_comparison(results: list[tuple[str, dict]]) -> None:
    sep  = "=" * 80
    line = "-" * 80
    print(f"\n{sep}")
    print("VERGLEICH: Baseline vs. Breakeven SL vs. Trailing SL vs. Beides")
    print(sep)

    hdrs = ["Metrik"] + [name for name, _ in results]
    col0 = 26
    coln = 13
    print(f"{'Metrik':<{col0}}" + "".join(f"{h:>{coln}}" for h in hdrs[1:]))
    print(line)

    def row(label: str, vals: list[str]) -> None:
        print(f"{label:<{col0}}" + "".join(f"{v:>{coln}}" for v in vals))

    metrics = [
        ("PnL gesamt (%)",     lambda r: f"{r['total_pnl_pct']:+.2f}%"),
        ("Final Balance",      lambda r: f"{r['final_balance']:,.2f}"),
        ("Trades",             lambda r: str(r["num_trades"])),
        ("Winrate (%)",        lambda r: f"{r['winrate_pct']:.2f}%"),
        ("TP-Hits",            lambda r: str(r["tp_count"])),
        ("SL-Hits",            lambda r: str(r["sl_count"])),
        ("Timeouts",           lambda r: str(r["timeout_count"])),
        ("Profit Factor",      lambda r: f"{r['profit_factor']:.4f}"),
        ("Max Drawdown (%)",   lambda r: f"{r['max_drawdown_pct']:.2f}%"),
        ("Avg Win (USDT)",     lambda r: f"{r['avg_win']:+.4f}"),
        ("Avg Loss (USDT)",    lambda r: f"{r['avg_loss']:+.4f}"),
    ]

    for label, fn in metrics:
        vals = [fn(r) for _, r in results]
        row(label, vals)

    print(sep)

    # Verbesserung gegenueber Baseline
    base_pnl = results[0][1]["total_pnl_pct"]
    print("Verbesserung ggue. Baseline:")
    for name, r in results[1:]:
        diff = r["total_pnl_pct"] - base_pnl
        print(f"  {name:<30}  {diff:+.2f}% PnL")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest auf OHLCV-CSV-Daten.")
    parser.add_argument("--csv",        default="data/raw/BTC_USDT_USDT_1m.csv")
    parser.add_argument("--sequence",   nargs="+", default=["long", "short"],
                        choices=["long", "short"])
    parser.add_argument("--leverage",   type=int,   default=BACKTEST_LEVERAGE)
    parser.add_argument("--size",       type=float, default=BACKTEST_POSITION_SIZE)
    parser.add_argument("--capital",    type=float, default=BACKTEST_INITIAL_CAPITAL)
    parser.add_argument("--fee",        type=float, default=BACKTEST_TAKER_FEE)
    parser.add_argument("--tp",         type=float, default=BACKTEST_TAKE_PROFIT_PCT * 100,
                        help="Take-Profit in %% (z.B. 2.0 fuer 2%%)")
    parser.add_argument("--sl",         type=float, default=BACKTEST_STOP_LOSS_PCT * 100,
                        help="Stop-Loss in %% (z.B. 1.0 fuer 1%%)")
    parser.add_argument("--max-hold",   type=int,   default=BACKTEST_MAX_HOLD_CANDLES,
                        help="Fallback-Exit nach N Kerzen")
    parser.add_argument("--breakeven",  type=float, default=None,
                        help="Breakeven-Trigger: Anteil des TP-Abstands (0.0-1.0), z.B. 0.5")
    parser.add_argument("--trailing-sl", type=float, default=None,
                        help="Trailing-SL-Abstand in %% (z.B. 0.8 fuer 0.8%%)")
    parser.add_argument("--compare",    action="store_true",
                        help="Vergleich: Baseline / Breakeven / Trailing / Beides")
    args = parser.parse_args()

    logger.info("Lade CSV: %s", args.csv)
    df = pd.read_csv(args.csv)

    base_cfg = dict(
        initial_capital  = args.capital,
        leverage         = args.leverage,
        position_size    = args.size,
        fee_rate         = args.fee,
        sequence         = args.sequence,
        take_profit_pct  = args.tp / 100,
        stop_loss_pct    = args.sl / 100,
        max_hold_candles = args.max_hold,
    )

    if args.compare:
        be_trigger   = args.breakeven   if args.breakeven   is not None else 0.5
        trail_pct    = (args.trailing_sl / 100) if args.trailing_sl is not None else (args.sl / 100) * 0.8

        variants = [
            ("Baseline",               {"breakeven_trigger_pct": None,       "trailing_sl_pct": None}),
            (f"Breakeven {be_trigger*100:.0f}%", {"breakeven_trigger_pct": be_trigger, "trailing_sl_pct": None}),
            (f"Trailing {trail_pct*100:.3f}%",   {"breakeven_trigger_pct": None,       "trailing_sl_pct": trail_pct}),
            ("Beides",                 {"breakeven_trigger_pct": be_trigger, "trailing_sl_pct": trail_pct}),
        ]

        results = []
        for name, extra in variants:
            cfg = BacktestConfig(**base_cfg, **extra)
            logger.info("Starte: %s", name)
            r = run_backtest(df, cfg)
            if "error" in r:
                logger.error("%s: %s", name, r["error"])
            else:
                results.append((name, r))

        if results:
            print_comparison(results)
            for name, r in results:
                print_results(r, label=name)
    else:
        cfg = BacktestConfig(
            **base_cfg,
            breakeven_trigger_pct = args.breakeven,
            trailing_sl_pct       = (args.trailing_sl / 100) if args.trailing_sl is not None else None,
        )

        logger.info(
            "Starte Backtest  |  %d Kerzen  |  %s  |  %dx  |  TP %.2f%%  SL %.2f%%",
            len(df), " -> ".join(cfg.sequence), cfg.leverage,
            cfg.take_profit_pct * 100, cfg.stop_loss_pct * 100,
        )

        result = run_backtest(df, cfg)

        if "error" in result:
            logger.error(result["error"])
            raise SystemExit(1)

        print_results(result)


if __name__ == "__main__":
    main()

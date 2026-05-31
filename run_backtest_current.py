"""
Backtest der aktuell laufenden Strategie.

Konfiguration (aus live_state / Session 2026-05-30):
  Strategie:    SupertrendStrategy(atr_period=20, multiplier=2.0)
  Timeframe:    1h
  Leverage:     3x
  Position:     5%
  Trailing SL:  0.5%
  TP / fixer SL: None
"""
from __future__ import annotations

import os
import sys
import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.strategies.supertrend_strategy import SupertrendStrategy
from src.strategy_backtester import StrategyConfig, run_strategy_backtest

# ── Strategie-Konfiguration (aktuelles Live-Setup) ──────────────────────────
STRATEGY      = SupertrendStrategy(atr_period=20, multiplier=2.0)
LEVERAGE      = 3
POSITION_SIZE = 0.05
TRAILING_SL   = 0.005    # 0.5%
TAKE_PROFIT   = None
STOP_LOSS     = None
CAPITAL       = 35_000   # Demo-Balance ~35k USDT

CFG = StrategyConfig(
    initial_capital   = CAPITAL,
    leverage          = LEVERAGE,
    position_size     = POSITION_SIZE,
    trailing_stop_pct = TRAILING_SL,
    take_profit_pct   = TAKE_PROFIT,
    stop_loss_pct     = STOP_LOSS,
    exit_on_signal    = True,
    fee_rate          = 0.00055,
)

# ── Alle 1h-Datensätze laden ─────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_1h.csv")))

print(f"Strategie: {STRATEGY}")
print(f"Config:    {LEVERAGE}x Leverage | Pos {POSITION_SIZE*100:.0f}% | Trailing {TRAILING_SL*100:.1f}% | TP {TAKE_PROFIT} | SL {STOP_LOSS}")
print(f"Kapital:   {CAPITAL:,} USDT")
print(f"Datasets:  {len(csv_files)} × 1h\n")
print(f"{'Coin':<18} {'Trades':>6} {'Win%':>6} {'PnL USDT':>10} {'PnL%':>7} {'Sharpe':>7} {'MaxDD%':>8}")
print("-" * 70)

results = []
for fpath in csv_files:
    fname = os.path.basename(fpath)
    # Symbol aus Dateiname: BTC_USDT_USDT_1h.csv → BTC
    coin = fname.split("_")[0]

    df = pd.read_csv(fpath)
    df.columns = [c.lower() for c in df.columns]

    if len(df) < 50:
        continue

    try:
        r = run_strategy_backtest(df, STRATEGY, CFG)
    except Exception as e:
        print(f"{coin:<18}  ERROR: {e}")
        continue

    trades    = r.get("num_trades", 0) or 0
    if trades == 0:
        continue

    win_pct   = r.get("winrate_pct", 0) or 0
    pnl_usdt  = r.get("total_pnl", 0) or 0
    pnl_pct   = (pnl_usdt / CAPITAL) * 100
    sharpe    = r.get("sharpe_ratio", 0) or 0
    max_dd    = (r.get("max_drawdown_pct", 0) or 0) * 100

    results.append({
        "coin": coin, "trades": trades, "win_pct": win_pct,
        "pnl_usdt": pnl_usdt, "pnl_pct": pnl_pct,
        "sharpe": sharpe, "max_dd": max_dd,
    })
    flag = " ★" if pnl_usdt > 0 else ""
    print(f"{coin:<18} {trades:>6} {win_pct:>5.1f}% {pnl_usdt:>+10.1f} {pnl_pct:>+6.1f}% {sharpe:>7.2f} {max_dd:>7.1f}%{flag}")

if results:
    total_pnl = sum(r["pnl_usdt"] for r in results)
    positive  = sum(1 for r in results if r["pnl_usdt"] > 0)
    print("-" * 70)
    print(f"{'GESAMT':<18} {'':>6} {'':>6} {total_pnl:>+10.1f} {total_pnl/CAPITAL*100:>+6.1f}%  ({positive}/{len(results)} profitabel)")
    print()

    # Top 5 nach PnL
    top5 = sorted(results, key=lambda x: x["pnl_usdt"], reverse=True)[:5]
    print("Top 5 Coins:")
    for r in top5:
        print(f"  {r['coin']:<16} {r['pnl_usdt']:>+10.1f} USDT  Win {r['win_pct']:.1f}%  {r['trades']} Trades")

    # Schlechteste 5
    bot5 = sorted(results, key=lambda x: x["pnl_usdt"])[:5]
    print("\nSchlechteste 5 Coins:")
    for r in bot5:
        print(f"  {r['coin']:<16} {r['pnl_usdt']:>+10.1f} USDT  Win {r['win_pct']:.1f}%  {r['trades']} Trades")

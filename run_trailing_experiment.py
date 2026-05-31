"""
Trailing-Stop Experiment — vergleicht verschiedene trailing_pct-Werte
sowie Aktivierungs-Schwellen (activation_pct) und optionalen Notfall-SL.

Coins:  alle 10 aktuellen Live-Coins (1h-Daten)
Params: je Coin die per_symbol_strategies_data Params aus live_state.json
"""
from __future__ import annotations

import os, sys, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.strategies.supertrend_strategy import SupertrendStrategy

# ── Konfiguration ────────────────────────────────────────────────────────────
DATA_DIR   = "data/raw"
LEVERAGE   = 3
POS_SIZE   = 0.05
CAPITAL    = 35_000
FEE        = 0.00055
MAX_HOLD   = 500          # max Kerzen pro Trade
TIMEFRAME  = "1h"

# Live-Coins + ihre ST-Parameter aus live_state.json
COIN_PARAMS = {
    "RIVER":      (20, 2.0),
    "LAB":        (20, 2.0),
    "SIREN":      (10, 2.0),
    "PLAYSOUT":   (14, 2.0),
    "LYN":        (20, 2.0),
    "PIPPIN":     (20, 2.0),
    "CLO":        (20, 2.0),
    "JELLYJELLY": (20, 2.0),
    "ESPORTS":    (14, 2.0),
    "PIEVERSE":   (14, 2.0),
}

# ── Exit-Funktion mit Activation-Threshold ───────────────────────────────────
def backtest_one(
    opens, highs, lows, closes, signals,
    trailing_pct: float,
    activation_pct: float | None,   # None = sofort aktiv, >0 = erst nach X% Gewinn
    sl_pct: float | None,           # None = kein Notfall-SL
    leverage: int,
    pos_size: float,
    capital_start: float,
) -> dict:
    equity = capital_start
    pnls, hold_times = [], []
    exits = {"trail": 0, "signal": 0, "sl": 0, "timeout": 0}
    n = len(opens)
    i = 0

    while i < n - 1:
        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side  = "long" if sig == 1 else "short"
        entry = opens[i + 1]
        if entry <= 0:
            i += 1
            continue

        notional = equity * pos_size * leverage

        # Fixer Notfall-SL (falls konfiguriert)
        if sl_pct is not None:
            pm_sl = sl_pct / leverage
            sl_price = (entry * (1 - pm_sl) if side == "long"
                        else entry * (1 + pm_sl))
        else:
            sl_price = None

        # Trail-State
        best_price = entry
        if activation_pct is None:
            # Sofort aktiv
            trail_sl = (entry * (1 - trailing_pct) if side == "long"
                        else entry * (1 + trailing_pct))
            trail_active = True
        else:
            trail_sl     = None
            trail_active = False

        reason = "timeout"
        exit_price = closes[min(i + MAX_HOLD, n - 1)]
        exit_idx   = min(i + 1 + MAX_HOLD, n - 1)

        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            h = highs[j]
            l = lows[j]

            # Aktivierungs-Check
            if not trail_active and activation_pct is not None:
                if side == "long":
                    profit_pct = (h - entry) / entry
                else:
                    profit_pct = (entry - l) / entry
                if profit_pct >= activation_pct:
                    trail_active = True
                    # Trailing startet vom besten Preis bisher
                    trail_sl = (best_price * (1 - trailing_pct) if side == "long"
                                else best_price * (1 + trailing_pct))

            # Trailing aktualisieren (nur wenn aktiv)
            if trail_active and trail_sl is not None:
                if side == "long" and h > best_price:
                    best_price = h
                    cand = best_price * (1 - trailing_pct)
                    if cand > trail_sl:
                        trail_sl = cand
                elif side == "short" and l < best_price:
                    best_price = l
                    cand = best_price * (1 + trailing_pct)
                    if cand < trail_sl:
                        trail_sl = cand

            # Effektiver SL
            eff_sl = sl_price
            if trail_sl is not None and trail_active:
                if eff_sl is None:
                    eff_sl = trail_sl
                elif side == "long":
                    eff_sl = max(eff_sl, trail_sl)
                else:
                    eff_sl = min(eff_sl, trail_sl)

            # SL-Check
            sl_hit = (eff_sl is not None) and (
                (l <= eff_sl if side == "long" else h >= eff_sl)
            )
            if sl_hit:
                exit_price = eff_sl
                exit_idx   = j
                reason     = "trail" if (trail_sl is not None and trail_active and
                                          eff_sl == trail_sl) else "sl"
                break

            # Signal-Exit
            if j < n - 1:
                s = int(signals[j])
                if (side == "long" and s == -1) or (side == "short" and s == 1):
                    exit_price = opens[j + 1]
                    exit_idx   = j + 1
                    reason     = "signal"
                    break
        else:
            exit_idx = min(i + 1 + MAX_HOLD, n - 1)

        raw_pnl = ((exit_price - entry) / entry * notional if side == "long"
                   else (entry - exit_price) / entry * notional)
        net_pnl = raw_pnl - notional * FEE * 2
        equity += net_pnl
        pnls.append(net_pnl)
        hold_times.append(exit_idx - (i + 1))
        exits[reason] += 1

        i = exit_idx - 1 if reason == "signal" else exit_idx
        if equity <= 0:
            break

    if not pnls:
        return {}

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gl     = abs(sum(losses))
    pf     = sum(wins) / gl if gl > 0 else (99.0 if wins else 0.0)

    eq = capital_start; peak = eq; max_dd = 0.0
    for p in pnls:
        eq += p; peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100)

    return {
        "trades":    len(pnls),
        "pnl_usdt":  equity - capital_start,
        "pnl_pct":   (equity - capital_start) / capital_start * 100,
        "winrate":   len(wins) / len(pnls) * 100,
        "pf":        min(pf, 99.0),
        "max_dd":    max_dd,
        "avg_hold":  np.mean(hold_times) if hold_times else 0,
        "exit_trail":   exits["trail"],
        "exit_signal":  exits["signal"],
        "exit_sl":      exits["sl"],
        "exit_timeout": exits["timeout"],
    }


def run_scenario(label, trailing_pct, activation_pct=None, sl_pct=None):
    """Läuft alle Coins durch und aggregiert Ergebnisse."""
    all_pnl, all_trades = 0.0, 0
    all_wins, all_losses = 0, 0
    all_trail, all_signal, all_sl_exits, all_timeout = 0, 0, 0, 0
    all_hold = []
    all_dd = []

    for coin, (atr_p, mult) in COIN_PARAMS.items():
        fpath = os.path.join(DATA_DIR, f"{coin}_USDT_USDT_{TIMEFRAME}.csv")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        df.columns = [c.lower() for c in df.columns]
        if len(df) < 50:
            continue

        strat   = SupertrendStrategy(atr_period=atr_p, multiplier=mult)
        signals = strat.generate_signals(df).to_numpy(int)
        opens   = df["open"].to_numpy(float)
        highs   = df["high"].to_numpy(float)
        lows    = df["low"].to_numpy(float)
        closes  = df["close"].to_numpy(float)

        r = backtest_one(opens, highs, lows, closes, signals,
                         trailing_pct=trailing_pct,
                         activation_pct=activation_pct,
                         sl_pct=sl_pct,
                         leverage=LEVERAGE,
                         pos_size=POS_SIZE,
                         capital_start=CAPITAL)
        if not r:
            continue

        all_pnl     += r["pnl_usdt"]
        all_trades  += r["trades"]
        n_wins       = round(r["trades"] * r["winrate"] / 100)
        all_wins    += n_wins
        all_losses  += r["trades"] - n_wins
        all_trail   += r["exit_trail"]
        all_signal  += r["exit_signal"]
        all_sl_exits+= r["exit_sl"]
        all_timeout += r["exit_timeout"]
        all_hold.append(r["avg_hold"])
        all_dd.append(r["max_dd"])

    if all_trades == 0:
        return None

    wr = all_wins / all_trades * 100
    avg_hold = np.mean(all_hold)
    avg_dd   = np.mean(all_dd)

    return {
        "label":       label,
        "trailing":    trailing_pct,
        "activation":  activation_pct,
        "sl_pct":      sl_pct,
        "trades":      all_trades,
        "pnl_usdt":    all_pnl,
        "pnl_pct":     all_pnl / CAPITAL * 100,
        "winrate":     wr,
        "avg_hold_h":  avg_hold,
        "trail_exits": all_trail,
        "signal_exits":all_signal,
        "sl_exits":    all_sl_exits,
        "timeout_exits":all_timeout,
        "trail_pct_of_exits": all_trail / all_trades * 100,
        "signal_pct_of_exits": all_signal / all_trades * 100,
        "avg_max_dd":  avg_dd,
    }


# ── Szenarien ────────────────────────────────────────────────────────────────
print("=" * 90)
print("TEIL 1: Verschiedene trailing_pct (kein Activation-Threshold, kein Notfall-SL)")
print("=" * 90)

scenarios_1 = []
for t in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020]:
    r = run_scenario(f"Trail {t*100:.2f}%", t)
    if r:
        scenarios_1.append(r)

hdr = f"{'Szenario':<22} {'Trades':>6} {'PnL USDT':>10} {'PnL%':>7} {'WR%':>6} {'AvgHold':>8} {'Trail%':>8} {'Signal%':>8} {'MaxDD%':>7}"
print(hdr)
print("-" * 90)
for r in scenarios_1:
    print(f"{r['label']:<22} {r['trades']:>6} {r['pnl_usdt']:>+10.0f} {r['pnl_pct']:>+6.1f}% "
          f"{r['winrate']:>5.1f}% {r['avg_hold_h']:>7.1f}h "
          f"{r['trail_pct_of_exits']:>7.1f}% {r['signal_pct_of_exits']:>7.1f}% "
          f"{r['avg_max_dd']:>6.1f}%")

print()
print("Exits aufgeschlüsselt:")
print(f"{'Szenario':<22} {'Trail':>7} {'Signal':>7} {'FixSL':>7} {'Timeout':>8}")
print("-" * 55)
for r in scenarios_1:
    print(f"{r['label']:<22} {r['trail_exits']:>7} {r['signal_exits']:>7} "
          f"{r['sl_exits']:>7} {r['timeout_exits']:>8}")


print()
print("=" * 90)
print("TEIL 2: Activation-Threshold (trailing erst nach X% Gewinn, trailing=0.3%)")
print("=" * 90)

scenarios_2 = []
# Baseline ohne Threshold
r = run_scenario("Trail 0.3% (sofort)", 0.003, activation_pct=None)
if r: scenarios_2.append(r)

for act in [0.003, 0.005, 0.010, 0.015]:
    r = run_scenario(f"Trail 0.3% act={act*100:.1f}%", 0.003, activation_pct=act)
    if r: scenarios_2.append(r)

# Auch mit 0.5% Trailing + verschiedene Activation
print()
print("  — mit 0.5% Trailing:")
for act in [None, 0.003, 0.005, 0.010]:
    lbl = f"Trail 0.5% act={'sofort' if act is None else f'{act*100:.1f}%'}"
    r = run_scenario(lbl, 0.005, activation_pct=act)
    if r: scenarios_2.append(r)

print(hdr)
print("-" * 90)
for r in scenarios_2:
    act_str = "sofort" if r["activation"] is None else f"{r['activation']*100:.1f}%"
    lbl = f"T={r['trailing']*100:.1f}% A={act_str}"
    print(f"{lbl:<22} {r['trades']:>6} {r['pnl_usdt']:>+10.0f} {r['pnl_pct']:>+6.1f}% "
          f"{r['winrate']:>5.1f}% {r['avg_hold_h']:>7.1f}h "
          f"{r['trail_pct_of_exits']:>7.1f}% {r['signal_pct_of_exits']:>7.1f}% "
          f"{r['avg_max_dd']:>6.1f}%")


print()
print("=" * 90)
print("TEIL 3: Notfall-SL (sl_pct=0.05 / leverage=3 → 1.67% unter Entry)")
print("        Vergleich: kein SL  vs.  fixer SL  vs.  fixer SL + Activation")
print("=" * 90)

scenarios_3 = []
combos = [
    ("Trail 0.3% / kein SL",       0.003, None,  None),
    ("Trail 0.3% / SL 1.67%",      0.003, None,  0.05),
    ("Trail 0.5% / kein SL",       0.005, None,  None),
    ("Trail 0.5% / SL 1.67%",      0.005, None,  0.05),
    ("Trail 0.3% act0.5% / kein SL",  0.003, 0.005, None),
    ("Trail 0.3% act0.5% / SL 1.67%", 0.003, 0.005, 0.05),
    ("Trail 0.5% act0.5% / kein SL",  0.005, 0.005, None),
    ("Trail 0.5% act0.5% / SL 1.67%", 0.005, 0.005, 0.05),
    ("Trail 0.75% act1.0% / kein SL", 0.0075,0.010, None),
    ("Trail 0.75% act1.0% / SL 1.67%",0.0075,0.010, 0.05),
    ("Trail 1.0% act1.0% / kein SL",  0.010, 0.010, None),
    ("Trail 1.0% act1.0% / SL 1.67%", 0.010, 0.010, 0.05),
    ("Trail 1.5% act1.5% / kein SL",  0.015, 0.015, None),
    ("Trail 2.0% act2.0% / kein SL",  0.020, 0.020, None),
]
for lbl, t, act, sl in combos:
    r = run_scenario(lbl, t, activation_pct=act, sl_pct=sl)
    if r: scenarios_3.append(r)

print(f"{'Szenario':<38} {'Trades':>6} {'PnL%':>7} {'WR%':>6} {'AvgHold':>7} {'Trail%':>8} {'Sig%':>6} {'SL%':>6} {'MaxDD%':>7}")
print("-" * 98)
for r in scenarios_3:
    lbl = r['label'][:38]
    sl_pct_exits = r['sl_exits'] / r['trades'] * 100 if r['trades'] else 0
    print(f"{lbl:<38} {r['trades']:>6} {r['pnl_pct']:>+6.1f}% "
          f"{r['winrate']:>5.1f}% {r['avg_hold_h']:>6.1f}h "
          f"{r['trail_pct_of_exits']:>7.1f}% {r['signal_pct_of_exits']:>5.1f}% "
          f"{sl_pct_exits:>5.1f}% {r['avg_max_dd']:>6.1f}%")

# ── Top-3 nach PnL ──────────────────────────────────────────────────────────
all_results = scenarios_1 + scenarios_2 + scenarios_3
all_results.sort(key=lambda x: x["pnl_usdt"], reverse=True)
print()
print("=" * 90)
print("TOP 5 Szenarien nach PnL:")
print("=" * 90)
for r in all_results[:5]:
    act_s = "sofort" if r["activation"] is None else f"act={r['activation']*100:.1f}%"
    sl_s  = f"SL={r['sl_pct']*100:.0f}%" if r["sl_pct"] else "kein SL"
    print(f"  Trail={r['trailing']*100:.2f}%  {act_s}  {sl_s}")
    print(f"    PnL={r['pnl_pct']:+.1f}%  WR={r['winrate']:.1f}%  Trades={r['trades']}  "
          f"AvgHold={r['avg_hold_h']:.1f}h  "
          f"Trail%={r['trail_pct_of_exits']:.0f}%  Sig%={r['signal_pct_of_exits']:.0f}%  "
          f"MaxDD={r['avg_max_dd']:.1f}%")
    print()

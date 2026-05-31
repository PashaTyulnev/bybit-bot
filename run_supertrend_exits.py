"""
SuperTrend Exit-Modus Test – 1h Timeframe, alle Coins.

Testet 4 Exit-Varianten:
  1. Fest        – fixer TP + fixer SL, kein Trailing
  2. Trailing    – nur Trailing SL (kein fixer SL/TP)
  3. Trail+TP    – Trailing SL + fixer TP
  4. ATR-basiert – TP = ATR × N, SL = ATR × M (dynamisch je Trade)
  5. Signal-only – kein TP/SL, nur Gegenrichtungs-Signal oder Timeout

SuperTrend Parameter: ATR=20, Mult=2.0 (beste Kombi aus Vortest)
Timeframe: 1h (beste Timeframe aus Vortest)

Starten:
    python run_supertrend_exits.py
"""

from __future__ import annotations

import os
import sys
import time
import logging
import json
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.strategies import SupertrendStrategy
from src.strategy_backtester import StrategyConfig, _find_exit_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konstanten ────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "supertrend_test")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "data", "supertrend_exits")
os.makedirs(OUT_DIR, exist_ok=True)

TF       = "1h"
LEVERAGE = 3
POS_SIZE = 0.05
FEE      = 0.00055
MIN_TRADES = 8
MAX_HOLD   = 500   # ~20 Tage auf 1h

# SuperTrend Parameter die getestet werden
ST_PARAMS = [
    (20, 2.0),
    (14, 2.0),
    (10, 2.0),
]

# ── Exit-Modi ─────────────────────────────────────────────────────────────────
# Format: (label, tp_pct, sl_pct, trailing_pct)
# None = deaktiviert

FIXED_EXITS = [
    ("Fest_TP1.5/SL0.75", 0.015, 0.0075, None),
    ("Fest_TP2/SL1",      0.020, 0.010,  None),
    ("Fest_TP3/SL1",      0.030, 0.010,  None),
    ("Fest_TP3/SL1.5",    0.030, 0.015,  None),
    ("Fest_TP4/SL1.5",    0.040, 0.015,  None),
    ("Fest_TP4/SL2",      0.040, 0.020,  None),
    ("Fest_TP5/SL2",      0.050, 0.020,  None),
    ("Fest_TP5/SL2.5",    0.050, 0.025,  None),
    ("Fest_TP2/SL2",      0.020, 0.020,  None),
    ("Fest_TP3/SL3",      0.030, 0.030,  None),
]

TRAILING_EXITS = [
    ("Trail0.3",  None,  None,  0.003),
    ("Trail0.5",  None,  None,  0.005),
    ("Trail0.8",  None,  None,  0.008),
    ("Trail1.0",  None,  None,  0.010),
    ("Trail1.5",  None,  None,  0.015),
    ("Trail2.0",  None,  None,  0.020),
]

TRAIL_TP_EXITS = [
    ("Trail0.5+TP2",  0.020, None, 0.005),
    ("Trail0.5+TP3",  0.030, None, 0.005),
    ("Trail0.8+TP3",  0.030, None, 0.008),
    ("Trail0.8+TP4",  0.040, None, 0.008),
    ("Trail1.0+TP3",  0.030, None, 0.010),
    ("Trail1.0+TP4",  0.040, None, 0.010),
    ("Trail1.0+TP5",  0.050, None, 0.010),
    ("Trail1.5+TP4",  0.040, None, 0.015),
    ("Trail1.5+TP5",  0.050, None, 0.015),
]

SIGNAL_ONLY_EXIT = [
    ("Signal-only", None, None, None),
]

# ATR-Multiplikatoren für ATR-basierte Exits
ATR_EXIT_PARAMS = [
    (1.0, 0.5),
    (1.5, 0.75),
    (2.0, 1.0),
    (2.5, 1.0),
    (3.0, 1.5),
    (3.0, 2.0),
    (4.0, 2.0),
    (4.0, 1.5),
]  # (atr_tp_mult, atr_sl_mult)


# ── ATR berechnen ─────────────────────────────────────────────────────────────

def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr  = np.maximum(highs - lows,
          np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = np.zeros(n)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ── Standard-Backtest (fest/trailing) ─────────────────────────────────────────

def backtest_standard(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, signals: np.ndarray,
    tp_pct: float | None, sl_pct: float | None, trailing_pct: float | None,
    capital: float,
) -> dict:
    equity = capital
    pnls: list[float] = []
    tp_c = sl_c = to_c = 0
    n = len(opens)
    i = 0

    while i < n - 1:
        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side = "long" if sig == 1 else "short"
        entry = opens[i + 1]
        if entry <= 0:
            i += 1
            continue

        tp_price = sl_price = None
        if tp_pct:
            tp_price = entry * (1 + tp_pct) if side == "long" else entry * (1 - tp_pct)
        if sl_pct:
            sl_price = entry * (1 - sl_pct) if side == "long" else entry * (1 + sl_pct)

        reason, exit_price, exit_idx = _find_exit_strategy(
            opens, highs, lows, closes, signals,
            entry_idx=i + 1, side=side,
            tp_price=tp_price, sl_price=sl_price,
            trailing_pct=trailing_pct,
            max_hold=MAX_HOLD,
            exit_on_signal=True,
        )

        margin   = equity * POS_SIZE
        notional = margin * LEVERAGE
        raw_pnl  = ((exit_price - entry) / entry * notional if side == "long"
                    else (entry - exit_price) / entry * notional)
        net_pnl  = raw_pnl - notional * FEE * 2
        equity  += net_pnl
        pnls.append(net_pnl)

        if reason == "tp":   tp_c += 1
        elif reason == "sl": sl_c += 1
        else:                to_c += 1

        i = exit_idx - 1 if reason == "signal" else exit_idx
        if equity <= 0:
            break

    return _stats(pnls, capital, equity, tp_c, sl_c, to_c)


# ── ATR-basierter Backtest ────────────────────────────────────────────────────

def backtest_atr(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, signals: np.ndarray, atr: np.ndarray,
    atr_tp_mult: float, atr_sl_mult: float,
    capital: float,
) -> dict:
    equity = capital
    pnls: list[float] = []
    tp_c = sl_c = to_c = 0
    n = len(opens)
    i = 0

    while i < n - 1:
        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side = "long" if sig == 1 else "short"
        entry = opens[i + 1]
        if entry <= 0 or atr[i] <= 0:
            i += 1
            continue

        atr_val  = atr[i]
        tp_price = (entry + atr_val * atr_tp_mult if side == "long"
                    else entry - atr_val * atr_tp_mult)
        sl_price = (entry - atr_val * atr_sl_mult if side == "long"
                    else entry + atr_val * atr_sl_mult)

        reason, exit_price, exit_idx = _find_exit_strategy(
            opens, highs, lows, closes, signals,
            entry_idx=i + 1, side=side,
            tp_price=tp_price, sl_price=sl_price,
            trailing_pct=None,
            max_hold=MAX_HOLD,
            exit_on_signal=True,
        )

        margin   = equity * POS_SIZE
        notional = margin * LEVERAGE
        raw_pnl  = ((exit_price - entry) / entry * notional if side == "long"
                    else (entry - exit_price) / entry * notional)
        net_pnl  = raw_pnl - notional * FEE * 2
        equity  += net_pnl
        pnls.append(net_pnl)

        if reason == "tp":   tp_c += 1
        elif reason == "sl": sl_c += 1
        else:                to_c += 1

        i = exit_idx - 1 if reason == "signal" else exit_idx
        if equity <= 0:
            break

    return _stats(pnls, capital, equity, tp_c, sl_c, to_c)


# ── Statistiken ───────────────────────────────────────────────────────────────

def _stats(pnls: list[float], start_cap: float, end_cap: float,
           tp_c: int, sl_c: int, to_c: int) -> dict:
    if not pnls:
        return {}
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total  = sum(pnls)
    wr     = len(wins) / len(pnls) * 100
    gl     = abs(sum(losses))
    pf     = (sum(wins) / gl) if gl > 0 else (99.0 if wins else 0.0)

    # Max Drawdown
    equity = start_cap
    peak   = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    return {
        "num_trades":       len(pnls),
        "total_pnl_pct":    (end_cap - start_cap) / start_cap * 100,
        "winrate_pct":      round(wr, 1),
        "profit_factor":    round(min(pf, 99.0), 3),
        "max_drawdown_pct": round(max_dd, 2),
        "tp_count":         tp_c,
        "sl_count":         sl_c,
        "timeout_count":    to_c,
    }


def score(r: dict) -> float:
    pnl = r.get("total_pnl_pct", 0)
    pf  = min(r.get("profit_factor", 0) or 0, 3.0)
    n   = max(r.get("num_trades", 0), 1)
    return pnl * pf * (n ** 0.5)


# ── Haupt-Backtest ────────────────────────────────────────────────────────────

def run_all(capital: float = 10_000.0) -> list[dict]:
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(f"_{TF}.csv"))
    log.info("Gefundene %s Dateien: %d", TF, len(files))

    results = []
    coin_cap = capital / max(len(files), 1)

    for fi, fname in enumerate(files):
        coin = fname.split("/")[-1].replace(f"_{TF}.csv", "").split("_")[0]
        df   = pd.read_csv(os.path.join(DATA_DIR, fname))
        if len(df) < 150:
            continue

        opens  = df["open"].to_numpy(float)
        highs  = df["high"].to_numpy(float)
        lows   = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)

        # Signale für alle ST-Parameter vorab berechnen
        st_signals: dict[tuple, np.ndarray] = {}
        st_atrs:    dict[tuple, np.ndarray] = {}
        for period, mult in ST_PARAMS:
            strat = SupertrendStrategy(period, mult)
            try:
                st_signals[(period, mult)] = strat.generate_signals(df).to_numpy(int)
                st_atrs[(period, mult)]    = compute_atr(highs, lows, closes, period)
            except Exception:
                pass

        if not st_signals:
            continue

        best: dict | None = None
        all_res: list[dict] = []

        for period, mult in ST_PARAMS:
            if (period, mult) not in st_signals:
                continue
            signals = st_signals[(period, mult)]
            atr_arr = st_atrs[(period, mult)]
            st_label = f"ST({period},{mult})"

            # 1. Feste Exits
            for label, tp, sl, trail in FIXED_EXITS:
                r = backtest_standard(opens, highs, lows, closes, signals,
                                      tp, sl, trail, coin_cap)
                if not r or r.get("num_trades", 0) < MIN_TRADES:
                    continue
                r["exit_mode"]  = "Fest"
                r["exit_label"] = label
                r["st_label"]   = st_label
                r["_score"]     = score(r)
                all_res.append(r)

            # 2. Nur Trailing
            for label, tp, sl, trail in TRAILING_EXITS:
                r = backtest_standard(opens, highs, lows, closes, signals,
                                      tp, sl, trail, coin_cap)
                if not r or r.get("num_trades", 0) < MIN_TRADES:
                    continue
                r["exit_mode"]  = "Trailing"
                r["exit_label"] = label
                r["st_label"]   = st_label
                r["_score"]     = score(r)
                all_res.append(r)

            # 3. Trailing + TP
            for label, tp, sl, trail in TRAIL_TP_EXITS:
                r = backtest_standard(opens, highs, lows, closes, signals,
                                      tp, sl, trail, coin_cap)
                if not r or r.get("num_trades", 0) < MIN_TRADES:
                    continue
                r["exit_mode"]  = "Trail+TP"
                r["exit_label"] = label
                r["st_label"]   = st_label
                r["_score"]     = score(r)
                all_res.append(r)

            # 4. ATR-basiert
            for atr_tp, atr_sl in ATR_EXIT_PARAMS:
                r = backtest_atr(opens, highs, lows, closes, signals, atr_arr,
                                 atr_tp, atr_sl, coin_cap)
                if not r or r.get("num_trades", 0) < MIN_TRADES:
                    continue
                r["exit_mode"]  = "ATR"
                r["exit_label"] = f"ATR_TP{atr_tp}/SL{atr_sl}"
                r["st_label"]   = st_label
                r["_score"]     = score(r)
                all_res.append(r)

            # 5. Signal-only
            for label, tp, sl, trail in SIGNAL_ONLY_EXIT:
                r = backtest_standard(opens, highs, lows, closes, signals,
                                      tp, sl, trail, coin_cap)
                if not r or r.get("num_trades", 0) < MIN_TRADES:
                    continue
                r["exit_mode"]  = "Signal-only"
                r["exit_label"] = label
                r["st_label"]   = st_label
                r["_score"]     = score(r)
                all_res.append(r)

        if not all_res:
            continue

        # Bestes Ergebnis je Coin
        best = max(all_res, key=lambda x: x["_score"])
        results.append({
            "coin":       coin,
            "st":         best["st_label"],
            "exit_mode":  best["exit_mode"],
            "exit":       best["exit_label"],
            "pnl_pct":    round(best.get("total_pnl_pct", 0), 3),
            "trades":     best.get("num_trades", 0),
            "winrate":    round(best.get("winrate_pct", 0), 1),
            "pf":         round(best.get("profit_factor", 0) or 0, 3),
            "max_dd":     round(best.get("max_drawdown_pct", 0), 2),
            "tp_count":   best.get("tp_count", 0),
            "sl_count":   best.get("sl_count", 0),
            "score":      round(best["_score"], 2),
            "_all":       all_res,
        })
        log.info("[%d/%d] %s  %s  %s  PnL %+.2f%%  PF %.2f  WR %.0f%%",
                 fi + 1, len(files), coin,
                 best["st_label"], best["exit_label"],
                 best.get("total_pnl_pct", 0),
                 best.get("profit_factor", 0) or 0,
                 best.get("winrate_pct", 0))

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], elapsed: float) -> None:
    if not results:
        log.warning("Keine Ergebnisse.")
        return

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_all"} for r in results])
    df_sorted = df.sort_values("pnl_pct", ascending=False)

    prof = df[df["pnl_pct"] > 0]

    # ── Exit-Modus Aggregation ────────────────────────────────────────────────
    # Alle Einzelergebnisse flachklopfen
    all_flat: list[dict] = []
    for r in results:
        for sub in r.get("_all", []):
            all_flat.append({
                "coin":      r["coin"],
                "exit_mode": sub["exit_mode"],
                "exit":      sub["exit_label"],
                "st":        sub["st_label"],
                "pnl_pct":   round(sub.get("total_pnl_pct", 0), 3),
                "trades":    sub.get("num_trades", 0),
                "winrate":   round(sub.get("winrate_pct", 0), 1),
                "pf":        round(sub.get("profit_factor", 0) or 0, 3),
                "max_dd":    round(sub.get("max_drawdown_pct", 0), 2),
                "score":     round(sub.get("_score", 0), 2),
            })

    df_all = pd.DataFrame(all_flat) if all_flat else pd.DataFrame()

    lines = []
    lines.append("# SuperTrend – Exit-Modus Vergleich\n")
    lines.append(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Laufzeit:** {elapsed/60:.1f} min  ")
    lines.append(f"**Timeframe:** 1h · 180 Tage  ")
    lines.append(f"**SuperTrend:** {ST_PARAMS}  ")
    lines.append(f"**Exit-Modi:** Fest ({len(FIXED_EXITS)}) · Trailing ({len(TRAILING_EXITS)}) · Trail+TP ({len(TRAIL_TP_EXITS)}) · ATR ({len(ATR_EXIT_PARAMS)}) · Signal-only (1)  ")
    lines.append(f"**Setup:** Leverage {LEVERAGE}× · Pos {POS_SIZE*100:.0f}% · Fee {FEE*100:.4f}%\n")

    lines.append("---\n")
    lines.append("## Zusammenfassung Beste Ergebnisse je Coin\n")
    lines.append(f"- **Coins gesamt:** {len(df)} | **Profitable:** {len(prof)} ({len(prof)/len(df)*100:.0f}%)")
    lines.append(f"- **Ø PnL:** {df['pnl_pct'].mean():+.2f}% | **Ø PF:** {df['pf'].mean():.2f}")

    # Exit-Modus Häufigkeit (als bestes Ergebnis je Coin)
    mode_counts = df["exit_mode"].value_counts()
    lines.append("\n### Welcher Exit-Modus gewinnt am häufigsten?\n")
    lines.append("| Exit-Modus | Anzahl Coins als Bester |")
    lines.append("|-----------|------------------------|")
    for mode, cnt in mode_counts.items():
        lines.append(f"| **{mode}** | {cnt} ({cnt/len(df)*100:.0f}%) |")

    # ── Vergleich aller Exit-Modi über alle Coins ────────────────────────────
    if not df_all.empty:
        lines.append("\n---\n")
        lines.append("## Exit-Modus Vergleich (Ø über alle Coins)\n")
        lines.append("| Exit-Modus | Coins | Ø PnL% | Median PnL | Ø PF | Ø WR% | % Profitabel | Ø MaxDD% |")
        lines.append("|-----------|-------|--------|-----------|------|-------|------------|---------|")
        for mode in ["Fest", "Trailing", "Trail+TP", "ATR", "Signal-only"]:
            g = df_all[df_all["exit_mode"] == mode]
            if g.empty:
                continue
            # Best per coin per mode
            best_per_coin = g.loc[g.groupby("coin")["score"].idxmax()]
            prof_m = best_per_coin[best_per_coin["pnl_pct"] > 0]
            lines.append(
                f"| **{mode}** | {len(best_per_coin)} | "
                f"**{best_per_coin['pnl_pct'].mean():+.2f}%** | "
                f"{best_per_coin['pnl_pct'].median():+.2f}% | "
                f"{best_per_coin['pf'].mean():.2f} | "
                f"{best_per_coin['winrate'].mean():.1f}% | "
                f"{len(prof_m)/len(best_per_coin)*100:.0f}% | "
                f"{best_per_coin['max_dd'].mean():.2f}% |"
            )

        # Bestes spezifisches Exit-Label je Modus
        lines.append("\n### Bestes Exit-Setting je Modus (Ø PnL über alle Coins)\n")
        for mode in ["Fest", "Trailing", "Trail+TP", "ATR"]:
            g = df_all[df_all["exit_mode"] == mode]
            if g.empty:
                continue
            by_exit = g.groupby("exit")["pnl_pct"].mean().sort_values(ascending=False)
            lines.append(f"#### {mode} – Top-5 Exits\n")
            lines.append("| Exit-Setting | Ø PnL% | Ø PF | Ø WR% |")
            lines.append("|-------------|--------|------|-------|")
            for ex, avg_pnl in by_exit.head(5).items():
                eg = g[g["exit"] == ex]
                lines.append(
                    f"| `{ex}` | **{avg_pnl:+.2f}%** | "
                    f"{eg['pf'].mean():.2f} | {eg['winrate'].mean():.1f}% |"
                )
            lines.append("")

    # ── Top-40 Coins (bestes Exit je Coin) ───────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Top-40 Coins nach PnL (bester Exit je Coin)\n")
    lines.append("| # | Coin | ST | Exit-Modus | Exit-Setting | PnL% | WR% | PF | MaxDD% | TP/SL |")
    lines.append("|---|------|-----|-----------|-------------|------|-----|-----|--------|-------|")
    for i, row in enumerate(df_sorted.head(40).itertuples(), 1):
        tp_sl = f"{row.tp_count}/{row.sl_count}"
        lines.append(
            f"| {i} | **{row.coin}** | `{row.st}` | **{row.exit_mode}** | "
            f"`{row.exit}` | **{row.pnl_pct:+.2f}%** | {row.winrate}% | "
            f"{row.pf} | {row.max_dd}% | {tp_sl} |"
        )

    # ── Detailvergleich: ST(20,2.0) mit allen Exit-Modi, Top-Coins ───────────
    if not df_all.empty:
        lines.append("\n---\n")
        lines.append("## Detail: ST(20,2.0) – Exit-Modi im Direktvergleich (Top-Coins)\n")
        lines.append("*Bestes Ergebnis je Exit-Modus pro Coin*\n")
        focus_coins = df_sorted.head(15)["coin"].tolist()
        g20 = df_all[(df_all["st"] == "ST(20,2.0)") & (df_all["coin"].isin(focus_coins))]
        if not g20.empty:
            best_by_coin_mode = g20.loc[g20.groupby(["coin", "exit_mode"])["score"].idxmax()]
            pivot = best_by_coin_mode.pivot(index="coin", columns="exit_mode", values="pnl_pct")
            pivot = pivot.fillna("–")
            modes = [m for m in ["Fest", "Trailing", "Trail+TP", "ATR", "Signal-only"] if m in pivot.columns]
            lines.append("| Coin | " + " | ".join(modes) + " |")
            lines.append("|------|" + "|".join(["-----"] * len(modes)) + "|")
            for coin in focus_coins:
                if coin not in pivot.index:
                    continue
                row_vals = []
                for m in modes:
                    v = pivot.at[coin, m] if coin in pivot.index and m in pivot.columns else "–"
                    row_vals.append(f"**{v:+.2f}%**" if isinstance(v, float) else str(v))
                lines.append(f"| {coin} | " + " | ".join(row_vals) + " |")

    lines.append("\n---\n")
    lines.append("## Bottom-10 Coins\n")
    lines.append("| Coin | ST | Exit | PnL% | Trades | PF |")
    lines.append("|------|-----|------|------|--------|-----|")
    for row in df_sorted.tail(10).itertuples():
        lines.append(f"| {row.coin} | `{row.st}` | `{row.exit}` | {row.pnl_pct:+.2f}% | {row.trades} | {row.pf} |")

    report = "\n".join(lines)
    report_path = os.path.join(OUT_DIR, "report_exits.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report: %s", report_path)


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_all"} for r in results])
    df_sorted = df.sort_values("pnl_pct", ascending=False)
    prof = df[df["pnl_pct"] > 0]

    # Exit-Modus Häufigkeit
    mode_counts = df["exit_mode"].value_counts()

    print("\n" + "="*70)
    print("  SUPERTREND EXIT-MODUS VERGLEICH – 1h")
    print("="*70)
    print(f"\n  Coins: {len(df)} | Profitabel: {len(prof)} ({len(prof)/len(df)*100:.0f}%)")
    print(f"  Ø PnL: {df['pnl_pct'].mean():+.2f}% | Ø PF: {df['pf'].mean():.2f}\n")

    print("  Welcher Exit-Modus gewinnt am häufigsten:")
    for mode, cnt in mode_counts.items():
        print(f"    {mode:<15} {cnt:>3} Coins ({cnt/len(df)*100:.0f}%)")

    print("\n" + "-"*70)
    print("  TOP-20 COINS (bester Exit je Coin)")
    print("-"*70)
    print(f"  {'Coin':<10} {'ST':<12} {'Modus':<12} {'Exit':<20} {'PnL%':>8}  {'PF':>5}  {'WR%':>5}")
    print("  " + "-"*68)
    for row in df_sorted.head(20).itertuples():
        print(f"  {row.coin:<10} {row.st:<12} {row.exit_mode:<12} {row.exit:<20} {row.pnl_pct:>+8.2f}%  {row.pf:>5.2f}  {row.winrate:>5.1f}%")

    print("\n" + "="*70)
    print(f"  Report: data/supertrend_exits/report_exits.md")
    print("="*70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  SuperTrend – Exit-Modus Vergleich · 1h")
    log.info("  Fest / Trailing / Trail+TP / ATR / Signal-only")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    results = run_all(capital=10_000.0)
    elapsed = time.time() - t0

    print_summary(results)
    write_report(results, elapsed)
    log.info("Fertig in %.1f min", elapsed / 60)

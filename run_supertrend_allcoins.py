"""
SuperTrend-Only Test – ALLE Bybit USDT-Perp Coins, alle Timeframes ab 15m.

Timeframes: 15m, 30m, 1h, 2h, 4h, 1d
SuperTrend-Parameter-Grid:
  ATR-Perioden: 7, 10, 14, 20
  Multiplikatoren: 2.0, 2.5, 3.0, 3.5
  Hebel: 1×, 2×, 3×

Starten:
    python run_supertrend_allcoins.py
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

from src.exchange import get_public_exchange
from src.download_ohlcv import fetch_ohlcv
from src.strategies import SupertrendStrategy
from src.strategy_backtester import StrategyConfig, run_strategy_backtest_fast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konstanten ────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "supertrend_test")

TIMEFRAME_DAYS: dict[str, int] = {
    "15m": 60,
    "30m": 90,
    "1h":  180,
}

ATR_PERIODS   = [7, 10, 14, 20]
MULTIPLIERS   = [2.0, 2.5, 3.0, 3.5]
LEVERAGES     = [1, 2, 3]
TP_PCT        = 3.0 / 100
SL_PCT        = 1.5 / 100
TRAILING      = 0.008
FEE           = 0.00055
POS_SIZE      = 0.05
MIN_TRADES    = 8
MIN_VOL_USDT  = 1_000_000

_MEME_BLACKLIST = {
    "1000PEPE", "1000SHIB", "1000BONK", "1000RATS", "1000TURBO", "1000LUNC", "1000BTT",
    "1000XEC", "1000FLOKI", "1000WHY", "1000MOG", "1000CAT", "1000SATS", "10000AIDOGE",
    "FARTCOIN", "BONK", "WIF", "MEME", "BABYDOGE", "DOGECAT", "TURBO", "PEPE",
    "FLOKI", "SHIB", "COW", "COQ", "BOME", "MYRO", "POPCAT", "PNUT", "MOG",
    "NEIRO", "BRETT", "SUNDOG", "PONKE", "MANEKI", "GIGA", "SLERF", "SILLY",
    "WIENER", "MIGGLES", "LADYS", "WOJAK", "TOAD", "MAGA", "TRUMP", "MELANIA",
    "FIDA", "LOTTO", "BEAT", "BILL", "BSB", "CL", "NIL", "GRASS", "BCUT",
    "ORCA", "ALPINE", "ACM", "CITY", "PORTO", "SANTOS", "ATM", "BAR", "JUV",
    "USDC", "USDT", "BUSD", "DAI", "TUSD", "FRAX", "USDP", "GUSD", "LUSD",
    "WBTC", "WETH", "STETH", "CBETH", "RETH",
}

os.makedirs(OUT_DIR, exist_ok=True)


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def csv_path(symbol: str, tf: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(OUT_DIR, f"{safe}_{tf}.csv")


def _is_quality_coin(symbol: str, vol_usdt: float) -> bool:
    base = symbol.split("/")[0].upper()
    if base in _MEME_BLACKLIST:
        return False
    if base.startswith("1000") or base.startswith("10000"):
        return False
    if vol_usdt < MIN_VOL_USDT:
        return False
    return True


def fetch_all_symbols() -> list[str]:
    log.info("Bybit – alle USDT-Perp Symbole laden…")
    ex = get_public_exchange()
    ex.load_markets()
    candidates = [
        s for s, m in ex.markets.items()
        if m.get("type") == "swap" and m.get("quote") == "USDT"
        and m.get("active", True) and ":" in s
    ]
    log.info("  %d USDT-Perp Symbole gefunden (roh)", len(candidates))

    tickers: dict = {}
    batch_size = 500
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        try:
            t = ex.fetch_tickers(batch)
            tickers.update(t)
        except Exception as e:
            log.warning("  Ticker-Batch fehlgeschlagen: %s", e)
        time.sleep(0.3)

    scored: list[tuple[str, float]] = []
    for s in candidates:
        vol = (tickers.get(s) or {}).get("quoteVolume") or 0.0
        if _is_quality_coin(s, vol):
            scored.append((s, vol))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = [s for s, _ in scored]
    log.info("  Nach Qualitätsfilter: %d Symbole", len(result))
    return result


def download_data(symbols: list[str], tf: str, days: int) -> list[str]:
    ok = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        p = csv_path(sym, tf)
        if os.path.exists(p):
            ok.append(sym)
            continue
        try:
            log.info("[%s] [%d/%d] ⬇ %s …", tf, i + 1, total, sym)
            df = fetch_ohlcv(sym, tf, days)
            df.to_csv(p, index=False)
            ok.append(sym)
        except Exception as e:
            log.warning("[%s] %s: Fehler %s", tf, sym, e)
    log.info("[%s] %d / %d Coins verfügbar", tf, len(ok), total)
    return ok


def score(r: dict) -> float:
    pnl = r.get("total_pnl_pct", 0)
    pf  = min(r.get("profit_factor", 0) or 0, 3.0)
    n   = max(r.get("num_trades", 0), 1)
    return pnl * pf * (n ** 0.5)


def backtest_coin_st(df: pd.DataFrame, capital: float) -> dict | None:
    opens  = df["open"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)

    best: dict | None = None

    for period, mult, lev in product(ATR_PERIODS, MULTIPLIERS, LEVERAGES):
        strat = SupertrendStrategy(period, mult)
        try:
            signals = strat.generate_signals(df).to_numpy(int)
        except Exception:
            continue

        cfg = StrategyConfig(
            initial_capital   = capital,
            leverage          = lev,
            position_size     = POS_SIZE,
            fee_rate          = FEE,
            take_profit_pct   = TP_PCT,
            stop_loss_pct     = SL_PCT,
            trailing_stop_pct = TRAILING,
            max_hold_candles  = 1440,
        )
        r = run_strategy_backtest_fast(opens, highs, lows, closes, signals, cfg)
        if not r or r.get("num_trades", 0) < MIN_TRADES:
            continue

        r["_period"] = period
        r["_mult"]   = mult
        r["_lev"]    = lev
        r["_score"]  = score(r)

        if best is None or r["_score"] > best["_score"]:
            best = r

    return best


# ── Haupt-Backtest ────────────────────────────────────────────────────────────

def run_timeframe(symbols: list[str], tf: str, capital: float = 10_000.0) -> list[dict]:
    coin_cap = capital / max(len(symbols), 1)
    results  = []

    for i, sym in enumerate(symbols):
        p = csv_path(sym, tf)
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)
        min_candles = 100
        if len(df) < min_candles:
            continue

        best = backtest_coin_st(df, coin_cap)
        if best is None:
            continue

        coin = sym.split("/")[0]
        results.append({
            "symbol":    sym,
            "coin":      coin,
            "timeframe": tf,
            "period":    best["_period"],
            "mult":      best["_mult"],
            "leverage":  best["_lev"],
            "pnl_pct":   round(best.get("total_pnl_pct", 0), 3),
            "trades":    best.get("num_trades", 0),
            "winrate":   round(best.get("winrate_pct", 0), 1),
            "pf":        round(best.get("profit_factor", 0) or 0, 3),
            "max_dd":    round(best.get("max_drawdown_pct", 0), 2),
            "score":     round(best["_score"], 2),
        })
        log.info(
            "[%s] %s  ST(%d,%.1f) %dx  PnL %+.2f%%  PF %.2f  Trades %d",
            tf, coin, best["_period"], best["_mult"], best["_lev"],
            best.get("total_pnl_pct", 0),
            best.get("profit_factor", 0) or 0,
            best.get("num_trades", 0),
        )

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results: dict[str, list[dict]], elapsed: float) -> str:
    lines = []
    lines.append("# SuperTrend – Alle Coins & Timeframes\n")
    lines.append(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Laufzeit:** {elapsed/60:.1f} min  ")
    lines.append(f"**Strategie:** SuperTrend · ATR {ATR_PERIODS} · Mult {MULTIPLIERS} · Hebel 1-3×  ")
    lines.append(f"**Timeframes:** {', '.join(TIMEFRAME_DAYS.keys())}  ")
    lines.append(f"**Setup:** Pos 5% · TP 3% / SL 1.5% · Trailing SL 0.8% · Fee 0.055%\n")

    # ── Gesamt-Übersicht je Timeframe ─────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Übersicht je Timeframe\n")
    lines.append("| Timeframe | Coins | Profitable | Ø PnL% | Ø PF | Ø Trades | Bester Coin | Bestes PnL% |")
    lines.append("|-----------|-------|------------|--------|------|----------|-------------|-------------|")

    for tf in TIMEFRAME_DAYS:
        res = all_results.get(tf, [])
        if not res:
            lines.append(f"| {tf} | 0 | – | – | – | – | – | – |")
            continue
        df = pd.DataFrame(res)
        prof = df[df["pnl_pct"] > 0]
        best_row = df.loc[df["pnl_pct"].idxmax()]
        lines.append(
            f"| **{tf}** | {len(df)} | {len(prof)} ({len(prof)/len(df)*100:.0f}%) | "
            f"**{df['pnl_pct'].mean():+.2f}%** | {df['pf'].mean():.2f} | "
            f"{df['trades'].mean():.0f} | **{best_row['coin']}** | **{best_row['pnl_pct']:+.2f}%** |"
        )

    # ── Je Timeframe: Top-30 Coins ────────────────────────────────────────────
    for tf in TIMEFRAME_DAYS:
        res = all_results.get(tf, [])
        if not res:
            continue

        df = pd.DataFrame(res).sort_values("pnl_pct", ascending=False)
        prof = df[df["pnl_pct"] > 0]
        best_params = df.groupby(["period", "mult"]).size().idxmax() if len(df) > 0 else ("–", "–")

        lines.append(f"\n---\n")
        lines.append(f"## Timeframe: {tf}\n")
        lines.append(f"**Coins gesamt:** {len(df)} | **Profitable:** {len(prof)} ({len(prof)/len(df)*100:.0f}%)  ")
        lines.append(f"**Ø PnL:** {df['pnl_pct'].mean():+.2f}% | **Ø PF:** {df['pf'].mean():.2f} | **Ø MaxDD:** {df['max_dd'].mean():.2f}%  ")
        lines.append(f"**Häufigste Parameter:** ATR={best_params[0]}, Mult={best_params[1]}\n")

        # Parameter-Verteilung
        lines.append("### Beste Parameter-Kombination (nach Häufigkeit)\n")
        param_counts = df.groupby(["period", "mult"]).size().reset_index(name="count").sort_values("count", ascending=False).head(8)
        lines.append("| ATR-Periode | Multiplikator | Anzahl Coins |")
        lines.append("|-------------|---------------|-------------|")
        for _, row in param_counts.iterrows():
            lines.append(f"| {int(row['period'])} | {row['mult']} | {int(row['count'])} |")

        lines.append(f"\n### Top-30 Coins – {tf}\n")
        lines.append("| # | Coin | ATR | Mult | Hebel | PnL% | Trades | WR% | PF | MaxDD% |")
        lines.append("|---|------|-----|------|-------|------|--------|-----|-----|--------|")
        for i, row in enumerate(df.head(30).itertuples(), 1):
            flag = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
            lines.append(
                f"| {flag} | **{row.coin}** | {row.period} | {row.mult} | {row.leverage}× | "
                f"**{row.pnl_pct:+.2f}%** | {row.trades} | {row.winrate}% | {row.pf} | {row.max_dd}% |"
            )

        # Bottom 10
        lines.append(f"\n### Bottom-10 Coins – {tf}\n")
        lines.append("| Coin | ATR | Mult | Hebel | PnL% | Trades | PF |")
        lines.append("|------|-----|------|-------|------|--------|-----|")
        for row in df.tail(10).itertuples():
            lines.append(
                f"| {row.coin} | {row.period} | {row.mult} | {row.leverage}× | "
                f"{row.pnl_pct:+.2f}% | {row.trades} | {row.pf} |"
            )

    # ── Coin-übergreifende Analyse: bestes TF je Coin ─────────────────────────
    lines.append("\n---\n")
    lines.append("## Bester Timeframe je Coin (Top-40 nach PnL)\n")
    all_flat = []
    for tf, res in all_results.items():
        all_flat.extend(res)

    if all_flat:
        df_all = pd.DataFrame(all_flat)
        best_per_coin = df_all.loc[df_all.groupby("coin")["pnl_pct"].idxmax()].sort_values("pnl_pct", ascending=False)
        lines.append("| # | Coin | Bester TF | ATR | Mult | Hebel | PnL% | Trades | WR% | PF |")
        lines.append("|---|------|-----------|-----|------|-------|------|--------|-----|-----|")
        for i, row in enumerate(best_per_coin.head(40).itertuples(), 1):
            lines.append(
                f"| {i} | **{row.coin}** | `{row.timeframe}` | {row.period} | {row.mult} | "
                f"{row.leverage}× | **{row.pnl_pct:+.2f}%** | {row.trades} | {row.winrate}% | {row.pf} |"
            )

    # JSON-Rohdaten
    lines.append("\n---\n")
    lines.append("## Rohdaten (JSON, erste 5000 Zeichen)\n")
    lines.append("```json")
    lines.append(json.dumps(all_flat[:50], indent=2, default=str)[:5000])
    lines.append("```")

    report = "\n".join(lines)
    report_path = os.path.join(OUT_DIR, "report_supertrend.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report gespeichert: %s", report_path)
    return report


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Gibt eine kompakte Zusammenfassung auf der Konsole aus."""
    print("\n" + "="*70)
    print("  SUPERTREND – ERGEBNISSE NACH TIMEFRAME")
    print("="*70)

    all_flat = []
    for tf, res in all_results.items():
        all_flat.extend(res)
        if not res:
            print(f"\n  {tf}: keine Ergebnisse")
            continue
        df = pd.DataFrame(res)
        prof = df[df["pnl_pct"] > 0]
        best = df.loc[df["pnl_pct"].idxmax()]
        print(f"\n  {tf} ({TIMEFRAME_DAYS[tf]} Tage)")
        print(f"    Coins: {len(df)} | Profitabel: {len(prof)} ({len(prof)/len(df)*100:.0f}%)")
        print(f"    Ø PnL: {df['pnl_pct'].mean():+.2f}% | Ø PF: {df['pf'].mean():.2f}")
        print(f"    Bester: {best['coin']}  ST({int(best['period'])},{best['mult']}) {best['leverage']}×  PnL {best['pnl_pct']:+.2f}%  PF {best['pf']:.2f}")

    if all_flat:
        print("\n" + "-"*70)
        print("  TOP-20 COINS (bester Timeframe je Coin)")
        print("-"*70)
        df_all = pd.DataFrame(all_flat)
        best_per_coin = df_all.loc[df_all.groupby("coin")["pnl_pct"].idxmax()].sort_values("pnl_pct", ascending=False)
        print(f"  {'Coin':<10} {'TF':<5} {'ATR':<5} {'Mult':<6} {'Lev':<4} {'PnL%':>8}  {'PF':>5}  {'Trades':>7}  {'WR%':>5}")
        print("  " + "-"*65)
        for row in best_per_coin.head(20).itertuples():
            print(f"  {row.coin:<10} {row.timeframe:<5} {row.period:<5} {row.mult:<6} {row.leverage}×  {row.pnl_pct:>+8.2f}%  {row.pf:>5.2f}  {row.trades:>7}  {row.winrate:>5.1f}%")

    print("\n" + "="*70)
    print(f"  Report: data/supertrend_test/report_supertrend.md")
    print("="*70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  SuperTrend – ALLE Coins × ALLE Timeframes ab 15m")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 1. Symbole laden
    symbols = fetch_all_symbols()
    log.info("Gesamt Symbole: %d", len(symbols))

    # 2. Je Timeframe: Daten laden + Backtest
    all_results: dict[str, list[dict]] = {}

    for tf, days in TIMEFRAME_DAYS.items():
        log.info("\n══ Timeframe: %s  (%d Tage) ══", tf, days)

        # Daten downloaden (nur fehlende)
        available = download_data(symbols, tf, days)

        # Backtest
        log.info("[%s] Starte Backtest für %d Coins…", tf, len(available))
        results = run_timeframe(available, tf, capital=10_000.0)
        all_results[tf] = results
        log.info("[%s] %d Coins mit Ergebnis", tf, len(results))

    # 3. Zusammenfassung + Report
    elapsed = time.time() - t0
    print_summary(all_results)
    write_report(all_results, elapsed)

    log.info("Gesamt-Laufzeit: %.1f Minuten", elapsed / 60)

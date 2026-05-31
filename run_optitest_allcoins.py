"""
OptiTest – ALLE Bybit USDT-Perp Coins, letzte 6 Monate (180 Tage, 15m).

Führt durch:
  1. Alle qualifizierten USDT-Perp Symbole von Bybit laden (kein Limit)
  2. 180-Tage Daten herunterladen (15m)
  3. Alle Strategien × Hebel backtesten, bestes Setup pro Coin wählen
  4. Ergebnisse in data/optitest_6m/report_all_coins_6m.md speichern

Starten:
    python run_optitest_allcoins.py
"""

from __future__ import annotations

import os
import sys
import time
import logging
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.exchange import get_public_exchange
from src.download_ohlcv import fetch_ohlcv
from src.strategies import (
    SupertrendStrategy, MACDStrategy, EMACrossStrategy,
    BreakoutStrategy, BollingerStrategy, TrendFollowStrategy,
)
from src.strategy_backtester import StrategyConfig, run_strategy_backtest_fast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konstanten ────────────────────────────────────────────────────────────────

OT_DIR         = os.path.join(os.path.dirname(__file__), "data", "optitest_6m")
COIN_LIMIT     = 9999          # Kein effektives Limit – alle qualifizierten Coins
TIMEFRAME      = "15m"
DAYS           = 180           # 6 Monate
FEE            = 0.00055
POS_SIZE       = 0.05
TRAILING       = 0.008
MIN_TRADES     = 10            # Etwas niedriger wegen kürzerem Zeitraum
LEVERAGES      = [1, 2, 3]
MIN_VOL_USDT   = 1_000_000     # Min. 1M USDT 24h-Volumen

_MEME_BLACKLIST = {
    # 1000x-Derivate
    "1000PEPE", "1000SHIB", "1000BONK", "1000RATS", "1000TURBO", "1000LUNC", "1000BTT",
    "1000XEC", "1000FLOKI", "1000WHY", "1000MOG", "1000CAT", "1000SATS", "10000AIDOGE",
    # Meme / Joke-Coins
    "FARTCOIN", "BONK", "WIF", "MEME", "BABYDOGE", "DOGECAT", "TURBO", "PEPE",
    "FLOKI", "SHIB", "COW", "COQ", "BOME", "MYRO", "POPCAT", "PNUT", "MOG",
    "NEIRO", "BRETT", "SUNDOG", "PONKE", "MANEKI", "GIGA", "SLERF", "SILLY",
    "WIENER", "MIGGLES", "LADYS", "WOJAK", "TOAD", "MAGA", "TRUMP", "MELANIA",
    # Extrem illiquide / unbekannte Nischen-Coins
    "FIDA", "LOTTO", "BEAT", "BILL", "BSB", "CL", "NIL", "GRASS", "BCUT",
    "ORCA", "ALPINE", "ACM", "CITY", "PORTO", "SANTOS", "ATM", "BAR", "JUV",
    # Stablecoins & wrapped
    "USDC", "USDT", "BUSD", "DAI", "TUSD", "FRAX", "USDP", "GUSD", "LUSD",
    "WBTC", "WETH", "STETH", "CBETH", "RETH",
}

STRAT_DEFS: list[tuple] = [
    ("ST_ATR14/3.0",       SupertrendStrategy(14, 3.0),          3.0,  1.5),
    ("MACD_12/21/7",       MACDStrategy(12, 21, 7),              3.0,  1.5),
    ("MACD_12/26/9",       MACDStrategy(12, 26, 9),              3.0,  1.5),
    ("EMA_20/50",          EMACrossStrategy(20, 50),             3.0,  1.0),
    ("BO_50",              BreakoutStrategy(50),                 3.0,  1.5),
    ("BB_50/2.5",          BollingerStrategy(50, 2.5, False),    2.0,  1.5),
    ("TF_EMA20/100_ADX25", TrendFollowStrategy(20, 100, 25.0),  3.0,  1.5),
]

os.makedirs(OT_DIR, exist_ok=True)


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def csv_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(OT_DIR, f"{safe}_{TIMEFRAME}.csv")


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
    """Lädt ALLE qualifizierten USDT-Perp Symbole von Bybit (kein Coin-Limit)."""
    log.info("Bybit – alle USDT-Perp Symbole laden…")
    ex = get_public_exchange()
    ex.load_markets()
    candidates = [
        s for s, m in ex.markets.items()
        if m.get("type") == "swap" and m.get("quote") == "USDT"
           and m.get("active", True) and ":" in s
    ]
    log.info("  %d USDT-Perp Symbole gefunden (roh)", len(candidates))

    # Ticker in Batches laden (Bybit-Limit ~600 pro Anfrage)
    tickers: dict = {}
    batch_size = 500
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        try:
            t = ex.fetch_tickers(batch)
            tickers.update(t)
            log.info("  Ticker-Batch %d/%d geladen (%d Symbole)",
                     i // batch_size + 1, (len(candidates) - 1) // batch_size + 1, len(batch))
        except Exception as e:
            log.warning("  Ticker-Batch fehlgeschlagen: %s", e)
        time.sleep(0.3)

    # Filtern + nach Volumen sortieren
    scored: list[tuple[str, float]] = []
    for s in candidates:
        vol = (tickers.get(s) or {}).get("quoteVolume") or 0.0
        if _is_quality_coin(s, vol):
            scored.append((s, vol))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = [s for s, _ in scored]

    log.info(
        "  Nach Qualitätsfilter (>= %.0fM Vol, keine Meme/Blacklist): %d Symbole",
        MIN_VOL_USDT / 1_000_000, len(result),
    )
    log.info("  Top-20 nach Volumen:")
    for sym, vol in scored[:20]:
        log.info("    %s  Vol: %.0fM USDT", sym, vol / 1_000_000)
    return result


def download_missing(symbols: list[str]) -> list[str]:
    ok = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        p = csv_path(sym)
        if os.path.exists(p):
            log.info("[%d/%d] ✓ %s (gecacht)", i + 1, total, sym)
            ok.append(sym)
            continue
        try:
            log.info("[%d/%d] ⬇ %s …", i + 1, total, sym)
            df = fetch_ohlcv(sym, TIMEFRAME, DAYS)
            df.to_csv(p, index=False)
            ok.append(sym)
            log.info("       %d Kerzen gespeichert", len(df))
        except Exception as e:
            log.warning("       ✗ Fehler: %s", e)
    return ok


def score(r: dict) -> float:
    pnl = r.get("total_pnl_pct", 0)
    pf  = min(r.get("profit_factor", 0) or 0, 3.0)
    n   = max(r.get("num_trades", 0), 1)
    return pnl * pf * (n ** 0.5)


def backtest_coin(df: pd.DataFrame, coin_capital: float) -> dict | None:
    opens  = df["open"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)

    best: dict | None = None
    all_results: list[dict] = []

    for label, strat, tp, sl in STRAT_DEFS:
        try:
            signals = strat.generate_signals(df).to_numpy(int)
        except Exception as e:
            log.debug("Signal-Fehler %s: %s", label, e)
            continue
        for lev in LEVERAGES:
            cfg = StrategyConfig(
                initial_capital   = coin_capital,
                leverage          = lev,
                position_size     = POS_SIZE,
                fee_rate          = FEE,
                take_profit_pct   = tp / 100,
                stop_loss_pct     = sl / 100,
                trailing_stop_pct = TRAILING,
                max_hold_candles  = 1440,
            )
            r = run_strategy_backtest_fast(opens, highs, lows, closes, signals, cfg)
            if not r or r.get("num_trades", 0) < MIN_TRADES:
                continue
            r["_label"] = label
            r["_lev"]   = lev
            r["_tp"]    = tp
            r["_sl"]    = sl
            r["_score"] = score(r)
            all_results.append(r)
            if best is None or r["_score"] > best["_score"]:
                best = r

    if best:
        best["_all"] = sorted(all_results, key=lambda x: x["_score"], reverse=True)[:3]
    return best


# ── Haupt-Backtest ────────────────────────────────────────────────────────────

def run(symbols: list[str], initial_capital: float = 10_000.0) -> list[dict]:
    coin_cap = initial_capital / len(symbols)
    results  = []

    for i, sym in enumerate(symbols):
        p = csv_path(sym)
        if not os.path.exists(p):
            log.warning("[%d/%d] %s: keine Daten, übersprungen", i + 1, len(symbols), sym)
            continue

        df = pd.read_csv(p)
        if len(df) < 300:
            log.warning("[%d/%d] %s: zu wenig Kerzen (%d)", i + 1, len(symbols), sym, len(df))
            continue

        log.info("[%d/%d] Backtest %s  (%d Kerzen)…", i + 1, len(symbols), sym, len(df))
        best = backtest_coin(df, coin_cap)
        if best is None:
            log.info("       kein gültiges Ergebnis (< %d Trades)", MIN_TRADES)
            continue

        coin = sym.split("/")[0]
        results.append({
            "symbol":   sym,
            "coin":     coin,
            "strategy": best["_label"],
            "leverage": best["_lev"],
            "tp":       best["_tp"],
            "sl":       best["_sl"],
            "pnl_pct":  round(best.get("total_pnl_pct", 0), 3),
            "trades":   best.get("num_trades", 0),
            "winrate":  round(best.get("winrate_pct", 0), 1),
            "pf":       round(best.get("profit_factor", 0) or 0, 3),
            "max_dd":   round(best.get("max_drawdown_pct", 0), 2),
            "score":    round(best["_score"], 2),
            "top3":     [
                {
                    "strategy": r["_label"],
                    "lev":      r["_lev"],
                    "pnl":      round(r.get("total_pnl_pct", 0), 2),
                    "pf":       round(r.get("profit_factor", 0) or 0, 2),
                    "trades":   r.get("num_trades", 0),
                }
                for r in best.get("_all", [])
            ],
        })
        log.info(
            "       Beste: %s  %dx  PnL %+.2f%%  PF %.2f  Trades %d  Score %.1f",
            best["_label"], best["_lev"],
            best.get("total_pnl_pct", 0),
            best.get("profit_factor", 0) or 0,
            best.get("num_trades", 0),
            best["_score"],
        )

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def strategy_frequency(results: list[dict]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for r in results:
        freq[r["strategy"]] = freq.get(r["strategy"], 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def leverage_frequency(results: list[dict]) -> dict[int, int]:
    freq: dict[int, int] = {}
    for r in results:
        freq[r["leverage"]] = freq.get(r["leverage"], 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def write_report(results: list[dict], symbols_total: int, elapsed: float) -> str:
    if not results:
        return "Keine Ergebnisse."

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("pnl_pct", ascending=False)

    profitable   = df[df["pnl_pct"] > 0]
    unprofitable = df[df["pnl_pct"] <= 0]
    avg_pnl      = df["pnl_pct"].mean()
    med_pnl      = df["pnl_pct"].median()
    avg_pf       = df["pf"].mean()
    avg_trades   = df["trades"].mean()
    total_trades = df["trades"].sum()
    avg_dd       = df["max_dd"].mean()
    strat_freq   = strategy_frequency(results)
    lev_freq     = leverage_frequency(results)

    n             = len(results)
    coin_cap      = 10_000.0 / n
    portfolio_end = sum(coin_cap * (1 + r["pnl_pct"] / 100) for r in results)
    portfolio_pnl = (portfolio_end - 10_000.0) / 10_000.0 * 100

    # Top-Performer für den Bot-Einsatz (PnL > 0, PF > 1.2, mind. 15 Trades)
    bot_candidates = df[
        (df["pnl_pct"] > 0) & (df["pf"] > 1.2) & (df["trades"] >= 15)
    ].sort_values("score", ascending=False)

    lines = []
    lines.append(f"# OptiTest – ALLE Bybit Coins · 6 Monate\n")
    lines.append(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Laufzeit:** {elapsed/60:.1f} min  ")
    lines.append(f"**Timeframe:** 15m · 180 Tage (6 Monate) · Bybit USDT-Perps  ")
    lines.append(f"**Setup:** Leverage 1×/2×/3× · Pos 5% · Trailing SL 0.8% · Fee 0.055%  ")
    lines.append(f"**Getestete Symbole:** {symbols_total} → {n} mit Ergebnis\n")

    lines.append("---\n")
    lines.append("## Portfolio-Zusammenfassung\n")
    lines.append("| Kennzahl | Wert |")
    lines.append("|---------|------|")
    lines.append(f"| Coins mit Ergebnis | {n} / {symbols_total} |")
    lines.append(f"| Profitable Coins | **{len(profitable)}** ({len(profitable)/n*100:.0f}%) |")
    lines.append(f"| Unprofitable Coins | {len(unprofitable)} ({len(unprofitable)/n*100:.0f}%) |")
    lines.append(f"| Bot-Kandidaten (PnL>0, PF>1.2, ≥15 Trades) | **{len(bot_candidates)}** |")
    lines.append(f"| Ø PnL pro Coin | **{avg_pnl:+.2f}%** |")
    lines.append(f"| Median PnL | {med_pnl:+.2f}% |")
    lines.append(f"| Ø Profit Factor | {avg_pf:.2f} |")
    lines.append(f"| Ø Max Drawdown | {avg_dd:.2f}% |")
    lines.append(f"| Ø Trades / Coin | {avg_trades:.0f} |")
    lines.append(f"| Trades gesamt | {total_trades:,} |")
    lines.append(f"| **Portfolio Return** (10.000 USDT, gleich aufgeteilt) | **{portfolio_pnl:+.2f}%** |")
    lines.append(f"| End-Kapital | **{portfolio_end:,.2f} USDT** |")

    # Bot-Kandidaten-Tabelle (Top 50 für den Bot geeignet)
    lines.append("\n---\n")
    lines.append("## Bot-Kandidaten (beste Coins für Live-Trading)\n")
    lines.append("*Kriterien: PnL > 0%, Profit Factor > 1.2, mind. 15 Trades in 6 Monaten*\n")
    lines.append("| # | Coin | Strategie | Hebel | TP/SL | PnL% | Trades | WR% | PF | MaxDD% | Score |")
    lines.append("|---|------|-----------|-------|-------|------|--------|-----|-----|--------|-------|")
    for i, row in enumerate(bot_candidates.head(50).itertuples(), 1):
        lines.append(
            f"| {i} | **{row.coin}** | `{row.strategy}` | {row.leverage}× | "
            f"{row.tp}%/{row.sl}% | **{row.pnl_pct:+.2f}%** | "
            f"{row.trades} | {row.winrate}% | {row.pf} | {row.max_dd}% | {row.score} |"
        )

    lines.append("\n---\n")
    lines.append("## Strategie-Verteilung (beste Strategie je Coin)\n")
    lines.append("| Strategie | Anzahl Coins | Anteil |")
    lines.append("|-----------|-------------|--------|")
    for s, c in strat_freq.items():
        lines.append(f"| `{s}` | {c} | {c/n*100:.0f}% |")

    lines.append("\n## Hebel-Verteilung\n")
    lines.append("| Hebel | Anzahl Coins |")
    lines.append("|-------|-------------|")
    for lv, c in lev_freq.items():
        lines.append(f"| {lv}× | {c} ({c/n*100:.0f}%) |")

    lines.append("\n---\n")
    lines.append("## Top-30 Coins nach PnL\n")
    lines.append("| # | Coin | Strategie | Hebel | TP/SL | PnL% | Trades | WR% | PF | MaxDD% |")
    lines.append("|---|------|-----------|-------|-------|------|--------|-----|-----|--------|")
    for i, row in enumerate(df_sorted.head(30).itertuples(), 1):
        flag = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        lines.append(
            f"| {flag} | **{row.coin}** | `{row.strategy}` | {row.leverage}× | "
            f"{row.tp}%/{row.sl}% | **{row.pnl_pct:+.2f}%** | "
            f"{row.trades} | {row.winrate}% | {row.pf} | {row.max_dd}% |"
        )

    lines.append("\n## Bottom-10 Coins nach PnL\n")
    lines.append("| Coin | Strategie | Hebel | PnL% | Trades | PF | MaxDD% |")
    lines.append("|------|-----------|-------|------|--------|-----|--------|")
    for row in df_sorted.tail(10).itertuples():
        lines.append(
            f"| {row.coin} | `{row.strategy}` | {row.leverage}× | "
            f"{row.pnl_pct:+.2f}% | {row.trades} | {row.pf} | {row.max_dd}% |"
        )

    lines.append("\n---\n")
    lines.append("## Strategie-Performance-Analyse\n")
    lines.append("| Strategie | Coins | Ø PnL% | Ø PF | Ø Trades | % Profitabel |")
    lines.append("|-----------|-------|--------|------|----------|--------------|")
    for s in strat_freq:
        sg = df[df["strategy"] == s]
        sp = sg[sg["pnl_pct"] > 0]
        lines.append(
            f"| `{s}` | {len(sg)} | {sg['pnl_pct'].mean():+.2f}% | "
            f"{sg['pf'].mean():.2f} | {sg['trades'].mean():.0f} | "
            f"{len(sp)/len(sg)*100:.0f}% |"
        )

    lines.append("\n---\n")
    lines.append("## Hebel-Performance-Analyse\n")
    lines.append("| Hebel | Coins | Ø PnL% | Ø PF | Ø MaxDD% |")
    lines.append("|-------|-------|--------|------|---------|")
    for lv in sorted(lev_freq.keys()):
        lg = df[df["leverage"] == lv]
        if len(lg) == 0:
            continue
        lines.append(
            f"| {lv}× | {len(lg)} | {lg['pnl_pct'].mean():+.2f}% | "
            f"{lg['pf'].mean():.2f} | {lg['max_dd'].mean():.2f}% |"
        )

    lines.append("\n---\n")
    lines.append("## Vollständige Coin-Tabelle (nach PnL sortiert)\n")
    lines.append("| Coin | Strategie | Hebel | PnL% | Trades | WR% | PF | MaxDD% | Score |")
    lines.append("|------|-----------|-------|------|--------|-----|-----|--------|-------|")
    for row in df_sorted.itertuples():
        lines.append(
            f"| {row.coin} | `{row.strategy}` | {row.leverage}× | "
            f"{row.pnl_pct:+.2f}% | {row.trades} | {row.winrate}% | "
            f"{row.pf} | {row.max_dd}% | {row.score} |"
        )

    lines.append("\n---\n")
    lines.append("## Rohdaten (JSON)\n")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2, default=str)[:15000])
    lines.append("```")

    report = "\n".join(lines)

    report_path = os.path.join(OT_DIR, "report_all_coins_6m.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report gespeichert: %s", report_path)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  OptiTest – ALLE Bybit Coins · 6 Monate · 15m")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 1. Alle Symbole laden
    symbols = fetch_all_symbols()
    log.info("Gesamt qualifizierte Symbole: %d", len(symbols))

    # 2. Fehlende Downloads
    log.info("\n── Download fehlender Coins (180 Tage, 15m) ──")
    available = download_missing(symbols)
    log.info("\n%d / %d Coins verfügbar", len(available), len(symbols))

    # 3. Backtest
    log.info("\n── Backtest startet ──")
    results = run(available, initial_capital=10_000.0)
    log.info("\nFertig: %d Coins mit Ergebnis", len(results))

    # 4. Report
    elapsed = time.time() - t0
    write_report(results, len(available), elapsed)

    log.info("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Gesamt-Laufzeit: %.1f Minuten", elapsed / 60)
    log.info("  Report: data/optitest_6m/report_all_coins_6m.md")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

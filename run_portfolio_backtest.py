"""
Portfolio-Backtest: 6 Coins × 3 Timeframes × 3 Varianten

Portfolio-Strategie (wie Live-Trader):
  MeanRev(BB10/2σ, ADX<20):  ADA, AVAX, DOGE, BNB
  TrendFollow(EMA20/100):     XRP, DOT

Varianten:
  A) Baseline   — fixer SL 1%, TP 2%
  B) Trailing   — Trailing SL 0.8%, TP 2%
  C) Trail+BE   — Trailing 0.8% + Breakeven bei 50% des TP-Abstands, TP 2%

Timeframes: 1m (90 Tage), 5m (180 Tage), 15m (180 Tage)
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Konfiguration ──────────────────────────────────────────────────────────────

COINS = ["ADA/USDT:USDT", "XRP/USDT:USDT", "DOT/USDT:USDT",
         "AVAX/USDT:USDT", "DOGE/USDT:USDT", "BNB/USDT:USDT"]

# Portfolio-Strategie-Zuweisung (wie im Live-Trader)
MEANREV_COINS    = set()   # alle auf TrendFollow
TRENDFOL_COINS   = set(COINS)

TIMEFRAMES = {
    "1m":  90,   # 90 Tage (1m = sehr viele Kerzen)
    "5m":  180,
    "15m": 180,
}

# TP/SL/Trailing (als % des Einstiegspreises)
TP_PCT       = 0.02    # 2 %
SL_PCT       = 0.01    # 1 %
TRAIL_PCT    = 0.008   # 0.8 %
BE_TRIGGER   = 0.50    # 50 % des TP-Abstands → SL auf Entry

LEVERAGE     = 10
POS_SIZE     = 0.10
CAPITAL      = 1000.0
FEE          = 0.00055
MAX_HOLD = {"1m": 1440, "5m": 288, "15m": 96}   # 1 Tag pro TF


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def coin_to_filename(coin: str, tf: str) -> str:
    return f"data/raw/{coin.replace('/', '_').replace(':', '_')}_{tf}.csv"


def coin_label(coin: str) -> str:
    return coin.split("/")[0]


def data_covers_days(coin: str, tf: str, days: int) -> bool:
    """True wenn die CSV-Datei ≥ days Tage abdeckt."""
    path = coin_to_filename(coin, tf)
    try:
        df = pd.read_csv(path, usecols=["datetime"])
        if len(df) < 10:
            return False
        start = pd.to_datetime(df["datetime"].iloc[0])
        end   = pd.to_datetime(df["datetime"].iloc[-1])
        return (end - start).days >= days - 2
    except Exception:
        return False


def download_one(coin: str, tf: str, days: int) -> tuple[str, str, bool]:
    """Lädt Daten für ein Symbol/Timeframe herunter. Gibt (coin, tf, success) zurück."""
    from src.download_ohlcv import fetch_ohlcv, save_csv
    try:
        logger.info("[DL] %s %s (%d Tage)…", coin_label(coin), tf, days)
        df = fetch_ohlcv(coin, tf, days)
        save_csv(df, coin, tf)
        logger.info("[DL OK] %s %s — %d Kerzen", coin_label(coin), tf, len(df))
        return coin, tf, True
    except Exception as e:
        logger.error("[DL FEHLER] %s %s: %s", coin_label(coin), tf, e)
        return coin, tf, False


def download_missing(jobs: list[tuple[str, str, int]], max_workers: int = 3) -> None:
    """Lädt fehlende Daten parallel herunter (max max_workers gleichzeitig)."""
    if not jobs:
        logger.info("Alle Daten vorhanden — kein Download nötig.")
        return
    logger.info("Starte Download: %d Jobs (%d parallel)…", len(jobs), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_one, c, tf, d): (c, tf) for c, tf, d in jobs}
        for fut in as_completed(futs):
            coin, tf, ok = fut.result()
            if not ok:
                logger.warning("Download fehlgeschlagen: %s %s", coin, tf)


def load_csv(coin: str, tf: str) -> pd.DataFrame | None:
    path = coin_to_filename(coin, tf)
    try:
        return pd.read_csv(path)
    except Exception:
        logger.warning("Datei nicht lesbar: %s", path)
        return None


# ── Breakeven-Erweiterung für strategy_backtester ────────────────────────────

from src.strategy_backtester import StrategyConfig, _find_exit_strategy
from src.backtester import ExitReason, Side, Trade, _compute_stats
from src.strategies.base_strategy import BaseStrategy
import numpy as np


def _find_exit_with_breakeven(
    opens, highs, lows, closes, signals,
    entry_idx: int, side: Side,
    tp_price: float | None, sl_price: float | None,
    trailing_pct: float | None, be_trigger_pct: float | None,
    max_hold: int, exit_on_signal: bool,
) -> tuple[ExitReason, float, int]:
    """Erweitert _find_exit_strategy um Breakeven-SL-Logik."""
    n            = len(highs)
    end          = min(entry_idx + max_hold, n)
    entry_price  = opens[entry_idx]
    best_price   = entry_price
    trail_sl     = None
    be_triggered = False

    if trailing_pct is not None:
        # Startet beim fixen SL — trailing verbessert ihn nur wenn Preis steigt.
        if sl_price is not None:
            trail_sl = sl_price
        else:
            trail_sl = (entry_price * (1 - trailing_pct) if side == "long"
                        else entry_price * (1 + trailing_pct))

    # Breakeven-Trigger-Kurs berechnen
    be_price = None
    if be_trigger_pct is not None and tp_price is not None:
        if side == "long":
            be_price = entry_price + (tp_price - entry_price) * be_trigger_pct
        else:
            be_price = entry_price - (entry_price - tp_price) * be_trigger_pct

    for j in range(entry_idx, end):
        h = highs[j]
        l = lows[j]

        # Trailing aktualisieren (nur verbessern, nie verschlechtern)
        if trailing_pct is not None:
            if side == "long" and h > best_price:
                best_price = h
                candidate  = best_price * (1 - trailing_pct)
                if candidate > trail_sl:
                    trail_sl = candidate
            elif side == "short" and l < best_price:
                best_price = l
                candidate  = best_price * (1 + trailing_pct)
                if candidate < trail_sl:
                    trail_sl = candidate

        # Breakeven aktivieren
        if be_price is not None and not be_triggered:
            if (side == "long"  and h >= be_price) or \
               (side == "short" and l <= be_price):
                be_triggered = True
                if sl_price is None:
                    sl_price = entry_price
                elif side == "long"  and entry_price > sl_price:
                    sl_price = entry_price
                elif side == "short" and entry_price < sl_price:
                    sl_price = entry_price

        # Effektiven SL ermitteln
        eff_sl = sl_price
        if trail_sl is not None:
            if eff_sl is None:
                eff_sl = trail_sl
            elif side == "long":
                eff_sl = max(eff_sl, trail_sl)
            else:
                eff_sl = min(eff_sl, trail_sl)

        # Exit prüfen
        tp_hit = (tp_price is not None) and (h >= tp_price if side == "long" else l <= tp_price)
        sl_hit = (eff_sl   is not None) and (l <= eff_sl   if side == "long" else h >= eff_sl)

        if sl_hit and tp_hit:
            return "sl", eff_sl, j
        if tp_hit:
            return "tp", tp_price, j
        if sl_hit:
            return "sl", eff_sl, j

        if exit_on_signal and j < n - 1:
            sig = int(signals[j])
            if (side == "long" and sig == -1) or (side == "short" and sig == 1):
                return "signal", float(opens[j + 1]), j + 1

    last = end - 1
    return "timeout", float(closes[last]), last


def run_backtest_with_be(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    cfg: StrategyConfig,
    be_trigger_pct: float | None = None,
) -> dict:
    """Vollständiger Backtest mit optionalem Breakeven-SL."""
    signals = strategy.generate_signals(df).to_numpy(dtype=int)
    opens   = df["open"].to_numpy(float)
    highs   = df["high"].to_numpy(float)
    lows    = df["low"].to_numpy(float)
    closes  = df["close"].to_numpy(float)
    times   = df["datetime"].astype(str).to_numpy()

    equity       = cfg.initial_capital
    trades:       list[Trade] = []
    equity_curve: list[float] = [equity]
    n = len(df)
    i = 0

    while i < n - 1:
        if cfg.circuit_breaker_pct and equity < cfg.initial_capital * (1 - cfg.circuit_breaker_pct):
            break
        sig = int(signals[i])
        if sig not in (1, -1):
            i += 1
            continue

        side: Side  = "long" if sig == 1 else "short"
        entry_price = opens[i + 1]
        if entry_price <= 0:
            i += 1
            continue

        tp_price = sl_price = None
        if cfg.take_profit_pct:
            tp_price = (entry_price * (1 + cfg.take_profit_pct) if side == "long"
                        else entry_price * (1 - cfg.take_profit_pct))
        if cfg.stop_loss_pct:
            sl_price = (entry_price * (1 - cfg.stop_loss_pct) if side == "long"
                        else entry_price * (1 + cfg.stop_loss_pct))

        exit_reason, exit_price, exit_idx = _find_exit_with_breakeven(
            opens, highs, lows, closes, signals,
            entry_idx=i + 1, side=side,
            tp_price=tp_price, sl_price=sl_price,
            trailing_pct=cfg.trailing_stop_pct,
            be_trigger_pct=be_trigger_pct,
            max_hold=cfg.max_hold_candles,
            exit_on_signal=cfg.exit_on_signal,
        )

        margin   = equity * cfg.position_size
        notional = margin * cfg.leverage
        raw_pnl  = ((exit_price - entry_price) / entry_price * notional if side == "long"
                    else (entry_price - exit_price) / entry_price * notional)
        net_pnl  = raw_pnl - notional * cfg.fee_rate * 2
        equity  += net_pnl

        trades.append(Trade(
            index=len(trades), side=side,
            entry_time=times[i + 1],
            exit_time=times[exit_idx] if exit_idx < len(times) else times[-1],
            entry_price=entry_price, exit_price=round(exit_price, 6),
            exit_reason=exit_reason, size_usdt=notional,
            pnl=net_pnl, pnl_pct=net_pnl / margin * 100,
            equity_after=equity,
        ))
        equity_curve.append(equity)

        i = exit_idx - 1 if exit_reason == "signal" else exit_idx
        if equity <= 0:
            break

    result = _compute_stats(trades, equity_curve, cfg)  # type: ignore
    result["strategy"] = strategy
    return result


# ── Haupt-Backtest-Funktion ───────────────────────────────────────────────────

from src.strategies import STRATEGY_REGISTRY

MeanRevStrategy    = STRATEGY_REGISTRY["MeanRev"]
TrendFollowStrategy = STRATEGY_REGISTRY["TrendFollow"]

VARIANTS = [
    # stop_loss_pct bleibt IMMER gesetzt — Trailing zieht ihn nur nach oben (wie im BTC-Test)
    ("Baseline",    dict(stop_loss_pct=SL_PCT, trailing_stop_pct=None),      None),
    ("Trailing",    dict(stop_loss_pct=SL_PCT, trailing_stop_pct=TRAIL_PCT), None),
    ("Trail+BE",    dict(stop_loss_pct=SL_PCT, trailing_stop_pct=TRAIL_PCT), BE_TRIGGER),
]


def run_coin_tf(coin: str, tf: str) -> dict:
    """Backtest für alle 3 Varianten für ein Coin/Timeframe."""
    df = load_csv(coin, tf)
    if df is None or len(df) < 100:
        return {}

    strat = (MeanRevStrategy(bb_period=10, bb_std=2.0, adx_threshold=20.0)
             if coin in MEANREV_COINS
             else TrendFollowStrategy(fast_ema=20, slow_ema=100, adx_threshold=25.0))

    results = {}
    for variant_name, extra_cfg, be_trigger in VARIANTS:
        cfg = StrategyConfig(
            initial_capital  = CAPITAL,
            leverage         = LEVERAGE,
            position_size    = POS_SIZE,
            fee_rate         = FEE,
            take_profit_pct  = TP_PCT,
            max_hold_candles = MAX_HOLD[tf],
            exit_on_signal   = True,
            **extra_cfg,
        )
        try:
            r = run_backtest_with_be(df, strat, cfg, be_trigger_pct=be_trigger)
            results[variant_name] = r
        except Exception as e:
            logger.warning("Fehler %s %s %s: %s", coin_label(coin), tf, variant_name, e)
            results[variant_name] = {}
    return results


# ── Ausgabe ───────────────────────────────────────────────────────────────────

def fmt(val, fmt_str=""):
    if val is None or val == "" or (isinstance(val, float) and val != val):
        return "–"
    try:
        return format(val, fmt_str)
    except Exception:
        return str(val)


def print_tf_table(tf: str, all_results: dict) -> None:
    sep  = "=" * 100
    line = "-" * 100
    print(f"\n{sep}")
    print(f"  TIMEFRAME: {tf}   |   TP {TP_PCT*100:.1f}%  SL {SL_PCT*100:.1f}%  Trailing {TRAIL_PCT*100:.1f}%  BE {BE_TRIGGER*100:.0f}%")
    print(sep)

    col0 = 8   # Coin
    col1 = 10  # Strategie
    cols = 13  # Spalten pro Variante

    header = f"{'Coin':<{col0}}{'Strat':<{col1}}"
    for vname, _, _ in VARIANTS:
        header += f"{'PnL%':>{cols}}{'WR%':>{cols}}{'Trades':>{cols}}{'PF':>{cols}}{'MaxDD%':>{cols}}"
    print(header)

    subhdr = " " * (col0 + col1)
    for vname, _, _ in VARIANTS:
        subhdr += f"  [{vname:<10}]{'':>{cols*5 - 14}}"
    print(line)

    # Spalten-Header mit Variantennamen
    h2 = f"{'':>{col0+col1}}"
    for vname, _, _ in VARIANTS:
        h2 += f"{vname:^{cols*5}}"
    print(h2)

    h3 = f"{'Coin':<{col0}}{'Strat':<{col1}}"
    for _ in VARIANTS:
        h3 += f"{'PnL%':>{cols}}{'WR%':>{cols}}{'Trades':>{cols}}{'PF':>{cols}}{'MaxDD%':>{cols}}"
    print(h3)
    print(line)

    summary_rows = []
    for coin in COINS:
        res = all_results.get(coin, {}).get(tf, {})
        if not res:
            continue
        strat_name = "MeanRev" if coin in MEANREV_COINS else "TrendFol"
        row = f"{coin_label(coin):<{col0}}{strat_name:<{col1}}"
        for vname, _, _ in VARIANTS:
            r = res.get(vname, {})
            pnl  = r.get("total_pnl_pct", None)
            wr   = r.get("winrate_pct",   None)
            n    = r.get("num_trades",    None)
            pf   = r.get("profit_factor", None)
            dd   = r.get("max_drawdown_pct", None)
            row += (f"{fmt(pnl, '+.2f'):>{cols}}"
                    f"{fmt(wr, '.1f'):>{cols}}"
                    f"{fmt(n, 'd') if n is not None else '–':>{cols}}"
                    f"{fmt(pf, '.3f'):>{cols}}"
                    f"{fmt(dd, '.1f'):>{cols}}")
            summary_rows.append((vname, pnl or 0.0))
        print(row)

    print(line)

    # Durchschnitt pro Variante
    from collections import defaultdict
    sums: dict[str, list[float]] = defaultdict(list)
    for coin in COINS:
        res = all_results.get(coin, {}).get(tf, {})
        for vname, _, _ in VARIANTS:
            r   = res.get(vname, {})
            pnl = r.get("total_pnl_pct")
            if pnl is not None:
                sums[vname].append(pnl)

    avg_row = f"{'∅ Alle':<{col0+col1}}"
    for vname, _, _ in VARIANTS:
        vals = sums[vname]
        avg  = sum(vals) / len(vals) if vals else 0.0
        avg_row += f"{avg:>+{cols}.2f}{'':>{cols*4}}"
    print(avg_row)
    print(sep)


def print_cross_tf_summary(all_results: dict) -> None:
    sep  = "=" * 90
    line = "-" * 90
    print(f"\n{sep}")
    print("  CROSS-TIMEFRAME ZUSAMMENFASSUNG  —  Durchschnittlicher PnL % pro Variante")
    print(sep)

    col0 = 16
    cols = 14

    print(f"{'Variante':<{col0}}" + "".join(f"{'1m':>{cols}}{'5m':>{cols}}{'15m':>{cols}}"))
    print(f"{'':>{col0}}" + "".join(f"{'(90d)':>{cols}}{'(180d)':>{cols}}{'(180d)':>{cols}}"))
    print(line)

    for vname, _, _ in VARIANTS:
        row = f"{vname:<{col0}}"
        for tf in ["1m", "5m", "15m"]:
            vals = []
            for coin in COINS:
                r = all_results.get(coin, {}).get(tf, {}).get(vname, {})
                p = r.get("total_pnl_pct")
                if p is not None:
                    vals.append(p)
            avg = sum(vals) / len(vals) if vals else None
            row += f"{fmt(avg, '+.2f'):>{cols}}"
        print(row)

    print(line)
    # Bestes Coin gesamt
    print("\n  BESTES COIN je Timeframe + Variante:")
    for tf in ["1m", "5m", "15m"]:
        best_coin, best_var, best_pnl = None, None, -999.0
        for coin in COINS:
            for vname, _, _ in VARIANTS:
                p = all_results.get(coin, {}).get(tf, {}).get(vname, {}).get("total_pnl_pct", None)
                if p is not None and p > best_pnl:
                    best_pnl, best_coin, best_var = p, coin, vname
        if best_coin:
            strat = "MeanRev" if best_coin in MEANREV_COINS else "TrendFol"
            print(f"  {tf:>4}:  {coin_label(best_coin):<6} ({strat})  [{best_var}]  {best_pnl:+.2f}%")
    print(sep)


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("\n" + "=" * 100)
    print("  PORTFOLIO BACKTEST — 6 Coins × 3 Timeframes × 3 Varianten")
    print(f"  Coins: {', '.join(coin_label(c) for c in COINS)}")
    print(f"  Strategie: MeanRev (ADA/AVAX/DOGE/BNB)  |  TrendFollow (XRP/DOT)")
    print(f"  Kapital: {CAPITAL:.0f} USDT  |  Hebel: {LEVERAGE}x  |  Pos-Größe: {POS_SIZE*100:.0f}%")
    print("=" * 100)

    # ── 1) Downloads prüfen und fehlende nachladen ────────────────────────────
    print("\n[1/3] Prüfe Datenverfügbarkeit…")
    missing: list[tuple[str, str, int]] = []
    for coin in COINS:
        for tf, days in TIMEFRAMES.items():
            if not data_covers_days(coin, tf, days):
                missing.append((coin, tf, days))
                logger.info("  Fehlt: %s %s (%d Tage)", coin_label(coin), tf, days)
            else:
                logger.info("  OK:    %s %s", coin_label(coin), tf)

    if missing:
        print(f"\n  → {len(missing)} Datei(en) werden heruntergeladen…")
        # Zuerst 5m/15m (schnell), dann 1m (dauert länger)
        fast = [(c, tf, d) for c, tf, d in missing if tf != "1m"]
        slow = [(c, tf, d) for c, tf, d in missing if tf == "1m"]
        download_missing(fast, max_workers=4)
        download_missing(slow, max_workers=2)   # 1m: weniger parallel (Rate-Limit)
    else:
        print("  → Alle Daten vorhanden.")

    # ── 2) Backtests ausführen ────────────────────────────────────────────────
    print("\n[2/3] Starte Backtests…")
    all_results: dict = {}

    for coin in COINS:
        all_results[coin] = {}
        for tf in TIMEFRAMES:
            logger.info("  Backtest %s %s…", coin_label(coin), tf)
            all_results[coin][tf] = run_coin_tf(coin, tf)

    # ── 3) Ergebnisse ausgeben ────────────────────────────────────────────────
    print("\n[3/3] Ergebnisse:\n")

    for tf in ["1m", "5m", "15m"]:
        print_tf_table(tf, all_results)

    print_cross_tf_summary(all_results)

    elapsed = time.time() - t0
    print(f"\n  Gesamtlaufzeit: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

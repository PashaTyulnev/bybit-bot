#!/usr/bin/env python3
"""
Anti-Randomness Backtest CLI

Testet mehrere Strategien mit unterschiedlichen Parametern und vergleicht sie
gegen eine Random-Baseline, um statistisch robuste Setups zu identifizieren.

Verwendung:
  python run_anti_randomness.py                          # Alle Strategien, .env-Konfiguration
  python run_anti_randomness.py --symbol ETHUSDT         # Anderes Symbol
  python run_anti_randomness.py --base-tf 5m             # Anderes Basis-Timeframe
  python run_anti_randomness.py --strategy trend         # Nur Trend-Following
  python run_anti_randomness.py --strategy meanrev regime # Mehrere auswählen
  python run_anti_randomness.py --top 30                 # Top 30 anzeigen
  python run_anti_randomness.py --detail 3               # Detail-Report für Top 3
  python run_anti_randomness.py --quick                  # Schnell-Modus (weniger Combos)
  python run_anti_randomness.py --no-random              # Keine Random-Baseline
  python run_anti_randomness.py --no-save                # Kein Datei-Export
  python run_anti_randomness.py --list-symbols           # Verfügbare Symbole anzeigen

Verfügbare Strategien (--strategy):
  trend       Trend-Following  (EMA-Cross + HTF-Filter)
  meanrev     Mean-Reversion   (Bollinger Bands, nur Ranging-Markt)
  regime      Regime-Switch    (EMA in Trend, BB in Range)
  volfilter   Volatility-Filter (Trend-Follow + ATR-Volatilitätsfenster)
"""

import argparse
import logging
import os
import sys

# Projektpfad
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anti-Randomness Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--symbol",       default=None,
                   help="Symbol (z.B. BTCUSDT, ETHUSDT). Standard: aus .env")
    p.add_argument("--base-tf",      default=None, dest="base_tf",
                   help="Basis-Timeframe (z.B. 15m, 1h). Standard: aus .env")
    p.add_argument("--trend-tf",     default=None, dest="trend_tf",
                   help="Trend-Filter Timeframe (z.B. 4h, 1h). Standard: aus .env")
    p.add_argument("--start",        default=None,
                   help="Start-Datum YYYY-MM-DD. Standard: aus .env")
    p.add_argument("--end",          default=None,
                   help="End-Datum YYYY-MM-DD. Standard: aus .env")
    p.add_argument("--strategy",     nargs="*", default=None,
                   choices=["trend", "meanrev", "regime", "volfilter"],
                   help="Strategie(n) auswählen. Standard: alle")
    p.add_argument("--top",          type=int, default=20,
                   help="Anzahl der Top-Ergebnisse in der Tabelle (Standard: 20)")
    p.add_argument("--detail",       type=int, default=0,
                   help="Detail-Report für die Top-N Ergebnisse (Standard: 0 = kein Detail)")
    p.add_argument("--quick",        action="store_true",
                   help="Schnell-Modus: reduzierter Parameter-Grid")
    p.add_argument("--no-random",    action="store_true", dest="no_random",
                   help="Random-Baseline deaktivieren")
    p.add_argument("--no-save",      action="store_true", dest="no_save",
                   help="Ergebnisse nicht als Datei speichern")
    p.add_argument("--output-dir",   default=None, dest="output_dir",
                   help="Ausgabe-Verzeichnis. Standard: aus .env")
    p.add_argument("--list-symbols", action="store_true", dest="list_symbols",
                   help="Verfügbare Symbole und Timeframes anzeigen")
    return p.parse_args()


def _cmd_list_symbols() -> None:
    from src.anti_randomness.data_loader import available_symbols, available_timeframes
    syms = available_symbols()
    if not syms:
        print("  Keine CSV-Dateien in data/raw/ gefunden.")
        return
    print(f"\n  {len(syms)} Symbole verfügbar:\n")
    for sym in syms:
        tfs = available_timeframes(sym)
        print(f"  {sym:<10}  Timeframes: {', '.join(tfs)}")
    print()


def main() -> int:
    args = _parse_args()

    if args.list_symbols:
        _cmd_list_symbols()
        return 0

    # .env-Overrides
    import src.anti_randomness.config as cfg_mod

    if args.no_random:
        cfg_mod.ENABLE_RANDOM = False
    if args.output_dir:
        cfg_mod.OUTPUT_DIR = args.output_dir

    from src.anti_randomness import runner, report

    def _progress(msg: str) -> None:
        print(msg, flush=True)

    try:
        results = runner.run_all(
            symbol     = args.symbol,
            base_tf    = args.base_tf,
            trend_tf   = args.trend_tf,
            start      = args.start,
            end        = args.end,
            strategies = args.strategy,
            quick      = args.quick,
            progress_cb= _progress,
        )
    except FileNotFoundError as e:
        print(f"\n  FEHLER: {e}")
        print("  Tipp: python run_anti_randomness.py --list-symbols")
        return 1
    except ValueError as e:
        print(f"\n  FEHLER: {e}")
        return 1

    sym    = args.symbol or cfg_mod.SYMBOL
    btf    = args.base_tf  or cfg_mod.BASE_TF
    ttf    = args.trend_tf or cfg_mod.TREND_TF

    report.print_summary_header(
        symbol         = sym,
        base_tf        = btf,
        trend_tf       = ttf,
        n_results      = len(results),
        n_combos_tested= len(results),   # echte Anzahl aus runner (inkl. leere)
    )

    if not results:
        print("  Keine Ergebnisse mit ausreichend Trades gefunden.")
        print("  Tipps:")
        print("   • Datumsgrenzen weiter fassen (--start / --end)")
        print("   • Anderes Symbol oder Timeframe wählen")
        print("   • --quick deaktivieren für breiteren Grid")
        return 0

    report.print_results_table(results, top_n=args.top)

    if not cfg_mod.ENABLE_RANDOM:
        print("  (Random-Baseline deaktiviert)\n")
    else:
        report.print_random_baseline_note()

    # Detail-Reports
    if args.detail > 0:
        print(_section_header("Detail-Reports"))
        for r in results[:args.detail]:
            report.print_strategy_detail(r)

    # Beste Strategie hervorheben
    best = results[0]
    pv   = best.get("p_value", float("nan"))
    pv_s = f"  p-Wert: {pv:.3f}" if pv == pv else ""
    print(f"  Beste Strategie: {best['strategy']}")
    print(f"  PF={best['profit_factor']:.3f}  Return={best['total_pnl_pct']:+.1f}%"
          f"  MaxDD={best['max_drawdown_pct']:.1f}%  Sharpe={best['sharpe_ratio']:.2f}"
          + pv_s)
    print()

    # Datei-Export
    if not args.no_save:
        try:
            jpath, cpath = report.save_results(
                results, cfg_mod.OUTPUT_DIR, sym, btf
            )
            print(f"  Gespeichert:")
            print(f"    JSON: {jpath}")
            print(f"    CSV:  {cpath}")
            print()
        except Exception as e:
            print(f"  [Warn] Speichern fehlgeschlagen: {e}")

    return 0


def _section_header(title: str) -> str:
    return f"\n  ── {title} " + "─" * max(0, 60 - len(title))


if __name__ == "__main__":
    sys.exit(main())

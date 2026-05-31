"""
Ausgabe und Export der Anti-Randomness Backtest-Ergebnisse.

Terminal: farbige ASCII-Tabelle (keine externen Libs nötig).
Dateien:  JSON (vollständig) + CSV (Vergleichstabelle).
"""
from __future__ import annotations

import csv
import json
import os
import textwrap
from datetime import datetime
from typing import Any


# ── ANSI-Farben (werden deaktiviert wenn kein TTY) ─────────────────────────────
import sys

_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text

def _green(t: str)  -> str: return _c("32", t)
def _red(t: str)    -> str: return _c("31", t)
def _yellow(t: str) -> str: return _c("33", t)
def _bold(t: str)   -> str: return _c("1",  t)
def _dim(t: str)    -> str: return _c("2",  t)


# ── Terminal-Tabelle ───────────────────────────────────────────────────────────

_COL_DEFS = [
    # (header,          width, key,               fmt,        color_fn)
    ("Rang",            5,  None,                  "d",        None),
    ("Strategie",       52, "strategy",             "s",        None),
    ("ATR×",            5,  "atr_mult",             ".1f",      None),
    ("RR",              4,  "rr",                   ".1f",      None),
    ("Trades",          7,  "num_trades",           "d",        None),
    ("WR%",             6,  "winrate_pct",          ".1f",      None),
    ("PF",              6,  "profit_factor",        ".3f",      _green),
    ("Return%",         9,  "total_pnl_pct",        "+.1f",     lambda v: _green(v) if float(v.strip("+")) > 0 else _red(v)),
    ("MaxDD%",          8,  "max_drawdown_pct",     ".1f",      _red),
    ("Sharpe",          7,  "sharpe_ratio",         ".2f",      None),
    ("Sortino",         8,  "sortino_ratio",        ".2f",      None),
    ("p-Wert",          8,  "p_value",              ".3f",      None),
    ("Rand-PF",         8,  "rand_mean_pf",         ".3f",      _dim),
]


def _fmt_cell(col: tuple, rank: int, row: dict) -> str:
    header, width, key, fmt, color_fn = col
    if key is None:
        val_str = str(rank)
    elif key == "rand_mean_pf":
        bl = row.get("random_baseline", {})
        v  = bl.get("mean_pf", float("nan"))
        val_str = f"{v:{fmt}}" if v == v else " n/a"
    elif key == "p_value":
        v = row.get("p_value", float("nan"))
        val_str = f"{v:{fmt}}" if v == v else " n/a"
    else:
        v = row.get(key)
        if v is None:
            val_str = "–"
        else:
            try:
                val_str = format(v, fmt)
            except (TypeError, ValueError):
                val_str = str(v)

    if color_fn:
        colored = color_fn(val_str)
    else:
        colored = val_str

    return colored.ljust(width) if not _COLOR else val_str.ljust(width).replace(val_str, colored, 1)


def print_results_table(results: list[dict], top_n: int = 20) -> None:
    shown = results[:top_n]
    if not shown:
        print("  Keine Ergebnisse.")
        return

    # Header
    header_parts = [_bold(h.ljust(w)) for h, w, *_ in _COL_DEFS]
    sep_parts    = ["-" * w for h, w, *_ in _COL_DEFS]
    print("  " + "  ".join(header_parts))
    print("  " + "  ".join(sep_parts))

    for rank, row in enumerate(shown, 1):
        cells = [_fmt_cell(col, rank, row) for col in _COL_DEFS]
        line  = "  " + "  ".join(cells)
        print(line)

    print()


def print_strategy_detail(result: dict) -> None:
    """Ausführliche Ausgabe für eine einzelne Strategie."""
    bl  = result.get("random_baseline", {})
    pv  = result.get("p_value", float("nan"))
    pv_str = f"{pv:.3f}" if pv == pv else "n/a"

    lines = [
        _bold(f"  ── {result['strategy']} ──"),
        f"  ATR-Mult={result.get('atr_mult')}  RR={result.get('rr')}  "
        f"HTF-Filter={'ja' if result.get('htf_filtered') else 'nein'}",
        "",
        f"  Trades:      {result.get('num_trades', 0):>6}",
        f"  Winrate:     {result.get('winrate_pct', 0):>6.1f} %",
        f"  Profit Factor: {result.get('profit_factor', 0):>6.3f}",
        f"  Return:      {result.get('total_pnl_pct', 0):>+7.1f} %",
        f"  Max Drawdown:{result.get('max_drawdown_pct', 0):>6.1f} %",
        f"  Sharpe:      {result.get('sharpe_ratio', 0):>6.2f}",
        f"  Sortino:     {result.get('sortino_ratio', 0):>6.2f}",
        f"  Long Trades: {result.get('long_trades', 0):>6}   PF={result.get('long_pf', 0):.3f}",
        f"  Short Trades:{result.get('short_trades', 0):>6}   PF={result.get('short_pf', 0):.3f}",
        f"  Exits: TP={result.get('tp_count', 0)}  SL={result.get('sl_count', 0)}"
        f"  Signal={result.get('signal_exit_count', 0)}  Timeout={result.get('timeout_count', 0)}",
    ]

    if bl.get("n_runs", 0) > 0:
        lines += [
            "",
            _bold("  ── Random Baseline ──"),
            f"  Runs:        {bl['n_runs']}",
            f"  Mean PF:     {bl.get('mean_pf', 0):.3f}  ±{bl.get('std_pf', 0):.3f}",
            f"  PF Percentiles: p5={bl.get('p5_pf', 0):.3f}  p50={bl.get('p50_pf', 0):.3f}"
            f"  p95={bl.get('p95_pf', 0):.3f}",
            f"  Mean Return: {bl.get('mean_return_pct', 0):+.2f} %",
            f"  p-Wert:      {pv_str}  "
            + (_green("★ statistisch signifikant") if pv != pv or pv < 0.05
               else _yellow("nicht signifikant (p≥0.05)")),
        ]

    for line in lines:
        print(line)
    print()


# ── Datei-Export ───────────────────────────────────────────────────────────────

def _safe_json(obj: Any) -> Any:
    """Konvertiert numpy/pandas Typen für JSON."""
    import numpy as np
    if isinstance(obj, (np.integer,)):      return int(obj)
    if isinstance(obj, (np.floating,)):     return float(obj)
    if isinstance(obj, (np.ndarray,)):      return obj.tolist()
    if hasattr(obj, "item"):                return obj.item()
    return str(obj)


def save_results(
    results:   list[dict],
    output_dir: str,
    symbol:    str = "",
    base_tf:   str = "",
) -> tuple[str, str]:
    """
    Speichert Ergebnisse als JSON und CSV.
    Gibt (json_path, csv_path) zurück.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym = symbol.replace("/", "").replace(":", "").upper() or "UNKNOWN"

    json_path = os.path.join(output_dir, f"{ts}_{sym}_{base_tf}_results.json")
    csv_path  = os.path.join(output_dir, f"{ts}_{sym}_{base_tf}_summary.csv")

    # JSON — vollständige Ergebnisse (ohne numpy-Arrays)
    clean = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("strategy",)}
        row["strategy_str"] = str(r.get("strategy", ""))
        clean.append(row)

    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2, default=_safe_json)

    # CSV — Vergleichstabelle (Schlüssel-Metriken)
    csv_cols = [
        "tag", "strategy_str", "atr_period", "atr_mult", "rr",
        "num_trades", "winrate_pct", "profit_factor", "total_pnl_pct",
        "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
        "long_pf", "short_pf", "tp_count", "sl_count",
        "htf_filtered", "p_value", "score",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in clean:
            row = {k: r.get(k, "") for k in csv_cols}
            writer.writerow(row)

    return json_path, csv_path


def print_summary_header(symbol: str, base_tf: str, trend_tf: str,
                         n_results: int, n_combos_tested: int) -> None:
    print()
    print(_bold("╔══════════════════════════════════════════════════════════════╗"))
    print(_bold("║       Anti-Randomness Backtest Report                        ║"))
    print(_bold("╚══════════════════════════════════════════════════════════════╝"))
    print(f"  Symbol    : {_bold(symbol)}   Base TF: {base_tf}   Trend TF: {trend_tf}")
    print(f"  Ergebnisse: {n_results}/{n_combos_tested} Kombinationen mit ≥3 Trades")
    print()


def print_random_baseline_note() -> None:
    note = (
        "  p-Wert: Wahrscheinlichkeit dass ein zufälliger Trader mit gleicher "
        "Trade-Frequenz das Ergebnis erreicht.\n"
        "  p < 0.05 = statistisch signifikant (grün ★)."
    )
    print(_dim(note))
    print()

"""
Export-Funktionen: CSV-Trades und Matplotlib-Charts.
"""

from __future__ import annotations

import os
import logging

import pandas as pd

from src.config import CHARTS_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def export_trades_csv(result: dict, filename_stem: str) -> str:
    """
    Speichert alle Trades als CSV unter data/results/<filename_stem>.csv.
    Gibt den Dateipfad zurueck.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"{filename_stem}.csv")

    trades = result.get("trades", [])
    if not trades:
        logger.warning("Keine Trades zum Exportieren.")
        return ""

    rows = [{
        "entry_time":    t.entry_time,
        "exit_time":     t.exit_time,
        "side":          t.side,
        "entry_price":   t.entry_price,
        "exit_price":    t.exit_price,
        "exit_reason":   t.exit_reason,
        "pnl_usdt":      round(t.pnl, 4),
        "pnl_pct":       round(t.pnl_pct, 4),
        "balance_after": round(t.equity_after, 4),
    } for t in trades]

    pd.DataFrame(rows).to_csv(filepath, index=False)
    logger.info("Trades exportiert: %s  (%d Zeilen)", filepath, len(rows))
    return filepath


def save_equity_chart(result: dict, title: str, filename_stem: str) -> str:
    """
    Speichert Equity-Curve + PnL-Balken als PNG unter data/charts/<filename_stem>.png.
    Gibt den Dateipfad zurueck.
    """
    import matplotlib
    matplotlib.use("Agg")   # kein GUI-Backend nötig
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, f"{filename_stem}.png")

    equity_curve = result.get("equity_curve", [])
    trades       = result.get("trades", [])

    if not equity_curve:
        logger.warning("Keine Equity-Daten fuer Chart.")
        return ""

    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 0.8], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#333355")
        ax.grid(True, color="#333355", linewidth=0.5, alpha=0.7)

    # ── Equity Curve ──────────────────────────────────────────────────────────
    initial = equity_curve[0]
    final   = equity_curve[-1]
    color   = "#2ecc71" if final >= initial else "#e74c3c"
    x_eq    = list(range(len(equity_curve)))

    ax1.plot(x_eq, equity_curve, color=color, linewidth=1.8, zorder=3)
    ax1.fill_between(x_eq, initial, equity_curve,
                     where=[e >= initial for e in equity_curve],
                     alpha=0.15, color="#2ecc71")
    ax1.fill_between(x_eq, initial, equity_curve,
                     where=[e < initial for e in equity_curve],
                     alpha=0.15, color="#e74c3c")
    ax1.axhline(initial, color="#888888", linestyle="--", linewidth=0.8)
    ax1.set_title(title, color="#ffffff", fontsize=13, pad=10)
    ax1.set_ylabel("Equity (USDT)", color="#aaaaaa", fontsize=9)
    ax1.set_xlabel("Trade #",       color="#aaaaaa", fontsize=9)

    stats_txt = (
        f"PnL: {result.get('total_pnl_pct', 0):+.2f}%  |  "
        f"Trades: {result.get('num_trades', 0)}  |  "
        f"Winrate: {result.get('winrate_pct', 0):.1f}%  |  "
        f"PF: {result.get('profit_factor', 0):.3f}  |  "
        f"Sharpe: {result.get('sharpe_ratio', 0):.3f}  |  "
        f"Max DD: {result.get('max_drawdown_pct', 0):.1f}%"
    )
    ax1.text(0.01, 0.02, stats_txt, transform=ax1.transAxes,
             color="#cccccc", fontsize=7.5, verticalalignment="bottom")

    # ── PnL-Balken ────────────────────────────────────────────────────────────
    if trades:
        pnls   = [t.pnl for t in trades]
        colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.85, width=0.8)
        ax2.axhline(0, color="#888888", linewidth=0.8)
        ax2.set_ylabel("PnL (USDT)", color="#aaaaaa", fontsize=9)
        ax2.set_xlabel("Trade #",    color="#aaaaaa", fontsize=9)

    # ── Exit-Reason Pie ───────────────────────────────────────────────────────
    if trades:
        from collections import Counter
        counts  = Counter(t.exit_reason for t in trades)
        labels  = list(counts.keys())
        sizes   = list(counts.values())
        pie_colors = {"tp": "#2ecc71", "sl": "#e74c3c",
                      "signal": "#3498db", "timeout": "#f39c12"}
        c_list = [pie_colors.get(l, "#888888") for l in labels]
        ax3.pie(sizes, labels=labels, colors=c_list, autopct="%1.0f%%",
                textprops={"color": "#cccccc", "fontsize": 8},
                startangle=90)
        ax3.set_title("Exit-Gründe", color="#cccccc", fontsize=8, pad=4)

    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    logger.info("Chart gespeichert: %s", filepath)
    return filepath


def compare_to_df(results: dict[str, dict]) -> pd.DataFrame:
    """
    Konvertiert ein {name: result_dict} Dict in einen sortierten Vergleichs-DataFrame.
    """
    rows = []
    for name, r in results.items():
        if "error" in r:
            continue
        rows.append({
            "Strategie":      name,
            "PnL %":          r.get("total_pnl_pct", 0),
            "Final Balance":  r.get("final_balance", 0),
            "Trades":         r.get("num_trades", 0),
            "Winrate %":      r.get("winrate_pct", 0),
            "Profit Factor":  r.get("profit_factor", 0),
            "Sharpe":         r.get("sharpe_ratio", 0),
            "Max DD %":       r.get("max_drawdown_pct", 0),
            "Avg Win":        r.get("avg_win", 0),
            "Avg Loss":       r.get("avg_loss", 0),
        })
    return pd.DataFrame(rows).sort_values("PnL %", ascending=False).reset_index(drop=True)

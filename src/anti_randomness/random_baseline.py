"""
Random-Baseline für Anti-Randomness Tests.

Idee: gleiche Trade-Frequenz wie die echte Strategie,
aber zufällige Richtung (Long/Short gleichverteilt).
Wenn die echte Strategie den Random-Baseline statistisch schlägt,
ist sie nicht zufällig.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.anti_randomness.engine import AtrBacktestConfig


def _randomize_directions(signals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Behält die Einstiegszeitpunkte, randomisiert aber Long/Short."""
    active = signals != 0
    rand   = signals.copy()
    n_entries = int(active.sum())
    if n_entries == 0:
        return rand
    rand[active] = rng.choice([-1, 1], size=n_entries)
    return rand


def run_random_baseline(
    df:        pd.DataFrame,
    signals:   np.ndarray,
    atr_arr:   np.ndarray,
    cfg:       "AtrBacktestConfig",
    n_runs:    int = 100,
    seed:      int = 42,
) -> dict:
    """
    Führt n_runs Backtest-Durchläufe mit zufälligen Signalrichtungen durch.
    Gibt Verteilungs-Statistiken zurück.

    Hinweis: Gleiche Zeitpunkte wie echte Signale — unterschiedliche Richtungen.
    So messen wir, ob die Strategie auf die Richtungswahl ankommt.
    """
    from src.anti_randomness.engine import run_atr_backtest

    rng     = np.random.default_rng(seed)
    results = []

    for _ in range(n_runs):
        rand_sigs = _randomize_directions(signals, rng)
        r = run_atr_backtest(df, rand_sigs, atr_arr, cfg)
        if r:
            results.append(r)

    if not results:
        return {"n_runs": 0}

    pfs      = np.array([r["profit_factor"]    for r in results])
    returns  = np.array([r["total_pnl_pct"]    for r in results])
    dds      = np.array([r["max_drawdown_pct"] for r in results])
    sharpes  = np.array([r["sharpe_ratio"]     for r in results])

    return {
        "n_runs":          len(results),
        "mean_pf":         round(float(pfs.mean()),    4),
        "std_pf":          round(float(pfs.std()),     4),
        "p5_pf":           round(float(np.percentile(pfs,  5)), 4),
        "p25_pf":          round(float(np.percentile(pfs, 25)), 4),
        "p50_pf":          round(float(np.percentile(pfs, 50)), 4),
        "p75_pf":          round(float(np.percentile(pfs, 75)), 4),
        "p95_pf":          round(float(np.percentile(pfs, 95)), 4),
        "mean_return_pct": round(float(returns.mean()), 2),
        "mean_dd_pct":     round(float(dds.mean()),     2),
        "mean_sharpe":     round(float(sharpes.mean()), 4),
    }


def p_value_vs_random(strategy_pf: float, baseline: dict) -> float:
    """
    Anteil der Random-Runs die die Strategie schlagen (Profit Factor).
    Niedrig = Strategie ist statistisch besser als Zufall.

    Gibt Wert ∈ [0, 1] zurück.
    0.05 bedeutet: nur 5 % der Zufalls-Runs schlagen die Strategie → p=0.05.
    """
    if baseline.get("n_runs", 0) == 0:
        return float("nan")
    n_runs = baseline["n_runs"]
    # Rekonstruiert approximativ aus Verteilung (kein Zugriff auf Rohdaten hier)
    # Konservative Abschätzung via Normalverteilung
    mean = baseline.get("mean_pf", 1.0)
    std  = baseline.get("std_pf", 0.1)
    if std <= 0:
        return 0.0 if strategy_pf > mean else 1.0
    # P(random >= strategy_pf) ≈ 1 - Φ((pf - mean) / std)
    from math import erfc, sqrt
    z = (strategy_pf - mean) / std
    p = 0.5 * erfc(z / sqrt(2))
    return round(float(p), 4)

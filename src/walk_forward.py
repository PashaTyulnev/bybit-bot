"""
Walk-Forward Validation: In-Sample-Optimierung + Out-of-Sample-Test.

Verhindert Overfitting: Der Optimizer läuft nur auf Trainingsdaten,
das beste Ergebnis wird auf ungesehenen Testdaten geprüft.
"""
from __future__ import annotations

import logging
import pandas as pd

from src.strategies.base_strategy import BaseStrategy
from src.strategy_backtester import StrategyConfig, run_strategy_backtest

logger = logging.getLogger(__name__)


def simple_split(
    df:          pd.DataFrame,
    strategy:    BaseStrategy,
    cfg:         StrategyConfig,
    train_ratio: float = 0.70,
) -> dict:
    """
    Teilt die Daten in Training (In-Sample) und Test (Out-of-Sample).
    Läuft den Backtest auf beiden Hälften und gibt beide Ergebnisse zurück.
    """
    n     = len(df)
    split = int(n * train_ratio)

    if split < 50 or (n - split) < 50:
        raise ValueError("Zu wenig Kerzen für einen sinnvollen Train/Test-Split.")

    df_train = df.iloc[:split].reset_index(drop=True)
    df_test  = df.iloc[split:].reset_index(drop=True)

    logger.info(
        "Walk-Forward: Train %d Kerzen (%s – %s)  |  Test %d Kerzen (%s – %s)",
        len(df_train),
        str(df_train["datetime"].iloc[0])[:10],
        str(df_train["datetime"].iloc[-1])[:10],
        len(df_test),
        str(df_test["datetime"].iloc[0])[:10],
        str(df_test["datetime"].iloc[-1])[:10],
    )

    r_train = run_strategy_backtest(df_train, strategy, cfg)
    r_test  = run_strategy_backtest(df_test,  strategy, cfg)

    return {
        "train":         r_train,
        "test":          r_test,
        "split_idx":     split,
        "split_date":    str(df["datetime"].iloc[split])[:10],
        "train_candles": len(df_train),
        "test_candles":  len(df_test),
        "train_pct":     train_ratio * 100,
        "test_pct":      (1 - train_ratio) * 100,
    }


def rolling_walk_forward(
    df:           pd.DataFrame,
    strategy:     BaseStrategy,
    cfg:          StrategyConfig,
    window_size:  int   = 5000,
    step_size:    int   = 1000,
    test_size:    int   = 1000,
) -> list[dict]:
    """
    Rollendes Walk-Forward: Mehrere Train/Test-Fenster hintereinander.
    Gibt eine Liste von Ergebnis-Dicts zurück (eines pro Fenster).
    """
    n       = len(df)
    results = []
    start   = 0

    while start + window_size + test_size <= n:
        df_train = df.iloc[start : start + window_size].reset_index(drop=True)
        df_test  = df.iloc[start + window_size : start + window_size + test_size].reset_index(drop=True)

        r_train = run_strategy_backtest(df_train, strategy, cfg)
        r_test  = run_strategy_backtest(df_test,  strategy, cfg)

        results.append({
            "window":     len(results) + 1,
            "train_from": str(df_train["datetime"].iloc[0])[:10],
            "train_to":   str(df_train["datetime"].iloc[-1])[:10],
            "test_from":  str(df_test["datetime"].iloc[0])[:10],
            "test_to":    str(df_test["datetime"].iloc[-1])[:10],
            "train":      r_train,
            "test":       r_test,
        })

        start += step_size

    logger.info("Rolling Walk-Forward: %d Fenster berechnet.", len(results))
    return results

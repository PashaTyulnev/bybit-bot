"""
Kombinierte Strategie: Verknüpft mehrere Strategien mit AND / OR / MAJORITY-Logik.

AND:      Alle Strategien müssen dasselbe Signal geben (alle Long oder alle Short).
OR:       Mindestens eine Strategie gibt ein Signal (erste Übereinstimmung zählt).
MAJORITY: Mehrheitsentscheid – mehr Longs als Shorts → Long, umgekehrt → Short.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategies.base_strategy import BaseStrategy


class CombinedStrategy(BaseStrategy):
    name = "combined"

    def __init__(self, strategies: list[BaseStrategy], logic: str = "AND"):
        if len(strategies) < 2:
            raise ValueError("Mindestens 2 Strategien für CombinedStrategy erforderlich.")
        if logic not in ("AND", "OR", "MAJORITY"):
            raise ValueError("logic muss AND, OR oder MAJORITY sein.")
        self.strategies = strategies
        self.logic      = logic

    def __str__(self) -> str:
        parts = " + ".join(str(s) for s in self.strategies)
        return f"Combined[{self.logic}]({parts})"

    def params_str(self) -> str:
        parts = "_".join(s.params_str() for s in self.strategies)
        return f"combined_{self.logic.lower()}_{parts}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Stack all signal arrays: shape (n_candles, n_strategies)
        arrays = np.stack(
            [s.generate_signals(df).to_numpy(dtype=int) for s in self.strategies],
            axis=1,
        )

        if self.logic == "AND":
            # All strategies must give the same non-zero signal
            all_nonzero = np.all(arrays != 0, axis=1)
            all_equal   = np.all(arrays == arrays[:, [0]], axis=1)
            result      = np.where(all_nonzero & all_equal, arrays[:, 0], 0)

        elif self.logic == "OR":
            # First non-zero signal wins (left-to-right priority)
            result = np.zeros(len(df), dtype=int)
            for col in range(arrays.shape[1] - 1, -1, -1):
                mask         = arrays[:, col] != 0
                result[mask] = arrays[:, col][mask]

        else:  # MAJORITY
            votes  = arrays.sum(axis=1)
            result = np.where(votes > 0, 1, np.where(votes < 0, -1, 0))

        return pd.Series(result.astype(int), index=df.index)

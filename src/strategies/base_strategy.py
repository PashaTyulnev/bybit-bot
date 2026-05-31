from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstrakte Basis für alle Strategien.

    generate_signals() liefert pro Kerze:
        1  = Long-Signal
       -1  = Short-Signal
        0  = Neutral (keine Position öffnen)

    Kein Lookahead: Signal an Kerze i darf nur Daten bis einschließlich Kerze i nutzen.
    Trade-Entry erfolgt am Open der Folgekerze (i+1).
    """
    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ...

    def params_str(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.params_str()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params_str()})"

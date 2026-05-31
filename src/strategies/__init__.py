from src.strategies.base_strategy import BaseStrategy
from src.strategies.ema_cross import EMACrossStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.bollinger_strategy import BollingerStrategy
from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.supertrend_strategy import SupertrendStrategy
from src.strategies.combined_strategy import CombinedStrategy
from src.strategies.rsi_divergence_strategy import RSIDivergenceStrategy
from src.strategies.mean_rev_strategy import MeanRevStrategy
from src.strategies.trend_follow_strategy import TrendFollowStrategy

__all__ = [
    "BaseStrategy",
    "EMACrossStrategy",
    "RSIStrategy",
    "BollingerStrategy",
    "BreakoutStrategy",
    "MACDStrategy",
    "SupertrendStrategy",
    "CombinedStrategy",
    "RSIDivergenceStrategy",
    "MeanRevStrategy",
    "TrendFollowStrategy",
]

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "EMA Cross":      EMACrossStrategy,
    "RSI":            RSIStrategy,
    "RSI Divergence": RSIDivergenceStrategy,
    "Bollinger":      BollingerStrategy,
    "Breakout":       BreakoutStrategy,
    "MACD":           MACDStrategy,
    "Supertrend":     SupertrendStrategy,
    "MeanRev":        MeanRevStrategy,
    "TrendFollow":    TrendFollowStrategy,
}

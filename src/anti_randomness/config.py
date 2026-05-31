"""
Anti-Randomness Backtest Konfiguration — wird aus .env geladen.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

def _floats(key: str, default: str) -> list[float]:
    return [float(x.strip()) for x in os.getenv(key, default).split(",") if x.strip()]

def _ints(key: str, default: str) -> list[int]:
    return [int(x.strip()) for x in os.getenv(key, default).split(",") if x.strip()]

def _bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("1", "true", "yes")


# ── Marktdaten ─────────────────────────────────────────────────────────────────
SYMBOL       = os.getenv("BACKTEST_SYMBOL",         "BTCUSDT")
BASE_TF      = os.getenv("BACKTEST_BASE_TIMEFRAME", "15m")
TREND_TF     = os.getenv("BACKTEST_TREND_TIMEFRAME","4h")
START_DATE   = os.getenv("BACKTEST_START_DATE",     "")
END_DATE     = os.getenv("BACKTEST_END_DATE",       "")

# ── Trade-Parameter ────────────────────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "1000"))
RISK_PER_TRADE  = float(os.getenv("BACKTEST_RISK_PER_TRADE",  "0.01"))
FEE_RATE        = float(os.getenv("BACKTEST_FEE_RATE",        "0.0006"))
SLIPPAGE_RATE   = float(os.getenv("BACKTEST_SLIPPAGE_RATE",   "0.0002"))
LEVERAGE        = int(  os.getenv("BACKTEST_LEVERAGE",        "1"))

# Effektive Gebühr pro Seite: Taker-Fee + Slippage (× 2 = Round-Trip in Engine)
EFFECTIVE_FEE_RATE = FEE_RATE + SLIPPAGE_RATE

# ── Grid-Parameter ─────────────────────────────────────────────────────────────
ATR_PERIODS     = _ints  ("ANTI_RANDOMNESS_ATR_PERIODS",      "14,21")
ATR_MULTIPLIERS = _floats("ANTI_RANDOMNESS_ATR_MULTIPLIERS",  "1.0,1.5,2.0,3.0")
RR_VALUES       = _floats("ANTI_RANDOMNESS_RR_VALUES",        "1.0,1.5,2.0,3.0")

FAST_EMAS       = _ints  ("ANTI_RANDOMNESS_FAST_EMA",         "20,50")
SLOW_EMAS       = _ints  ("ANTI_RANDOMNESS_SLOW_EMA",         "100,200")

VOL_MIN_VALUES  = _floats("ANTI_RANDOMNESS_VOLATILITY_MIN",   "0.002,0.004")
VOL_MAX_VALUES  = _floats("ANTI_RANDOMNESS_VOLATILITY_MAX",   "0.03,0.05")

# ── Random-Baseline ────────────────────────────────────────────────────────────
ENABLE_RANDOM   = _bool  ("ANTI_RANDOMNESS_ENABLE_RANDOM_BASELINE", "true")
RANDOM_RUNS     = int(    os.getenv("ANTI_RANDOMNESS_RANDOM_RUNS",   "100"))

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = os.getenv(
    "ANTI_RANDOMNESS_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "storage", "backtests", "anti_randomness"),
)

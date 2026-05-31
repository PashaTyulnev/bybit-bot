import os
from dotenv import load_dotenv

load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "")  # leer = Mainnet

DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_DAYS = 7

_BASE = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR     = os.path.join(_BASE, "data", "raw")
RESULTS_DIR      = os.path.join(_BASE, "data", "results")
CHARTS_DIR       = os.path.join(_BASE, "data", "charts")

# ── Backtest defaults ──────────────────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL  = 1000.0   # USDT
BACKTEST_LEVERAGE         = 10       # Hebel
BACKTEST_POSITION_SIZE    = 0.10     # Anteil des Kapitals pro Trade (10 %)
BACKTEST_TAKER_FEE        = 0.00055  # Bybit Taker-Fee (0.055 %)
BACKTEST_MAKER_FEE        = 0.00020  # Bybit Maker-Fee (0.020 %)
BACKTEST_TAKE_PROFIT_PCT  = 0.02     # 2 % vom Entry-Preis
BACKTEST_STOP_LOSS_PCT    = 0.01     # 1 % vom Entry-Preis
BACKTEST_MAX_HOLD_CANDLES = 1440     # Zwangs-Exit nach 1 Tag (Safety-Fallback)

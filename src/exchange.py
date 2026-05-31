import logging
import socket
import ccxt
from src.config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_API_URL

# Hartes Socket-Timeout für alle neuen Verbindungen — verhindert SSL-Shutdown-Deadlocks
socket.setdefaulttimeout(12)

logger = logging.getLogger(__name__)

MAINNET_URL = "https://api.bybit.com"
DEMO_URL    = "https://api-demo.bybit.com"

_ACTIVE_URL = (BYBIT_API_URL or DEMO_URL).rstrip("/")
_IS_DEMO    = _ACTIVE_URL == DEMO_URL


def get_public_exchange() -> ccxt.bybit:
    """Marktdaten immer vom Mainnet."""
    return ccxt.bybit({
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "linear"},
        "urls": {"api": {"public": MAINNET_URL, "private": MAINNET_URL}},
    })


def get_exchange() -> ccxt.bybit:
    """Trading-Exchange: Demo oder Mainnet je nach BYBIT_API_URL."""
    if not _IS_DEMO:
        logger.warning("BYBIT_API_URL zeigt nicht auf Demo (%s) — bitte pruefen!", _ACTIVE_URL)

    params: dict = {
        "enableRateLimit": True,
        "timeout": 20000,   # 20s — verhindert ewiges Hängen auf Demo-API
        "options": {
            "defaultType": "linear",
            "adjustForTimeDifference": True,
            "recvWindow": 10000,
        },
        "urls": {"api": {"public": _ACTIVE_URL, "private": _ACTIVE_URL}},
    }
    if BYBIT_API_KEY and BYBIT_API_SECRET:
        params["apiKey"] = BYBIT_API_KEY
        params["secret"] = BYBIT_API_SECRET
    else:
        logger.warning("Bybit: Kein API-Key in .env — private Endpoints nicht verfuegbar.")

    ex = ccxt.bybit(params)

    # Demo-API unterstuetzt fetch_currencies und query-api nicht
    ex.has["fetchCurrencies"] = False
    ex.is_unified_enabled = lambda: (True, True)

    logger.info("Bybit Demo-Exchange aktiv  (%s)", _ACTIVE_URL)
    return ex

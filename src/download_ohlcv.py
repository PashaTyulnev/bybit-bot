"""
Lädt historische OHLCV-Daten von Bybit und speichert sie als CSV.

Verwendung:
    python -m src.download_ohlcv --symbol BTC/USDT:USDT --timeframe 1m --days 7
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from src.config import DEFAULT_DAYS, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, RAW_DATA_DIR
from src.exchange import get_public_exchange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CANDLES_PER_REQUEST = 1000  # Bybit-Limit pro Request
PAUSE_BETWEEN_REQUESTS = 0.5  # Sekunden


def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    exchange = get_public_exchange()

    tf_seconds = exchange.parse_timeframe(timeframe)
    now_ms = exchange.milliseconds()
    since_ms = now_ms - days * 24 * 3600 * 1000

    logger.info(
        "Lade %s  |  Timeframe: %s  |  Zeitraum: letzte %d Tage",
        symbol,
        timeframe,
        days,
    )
    logger.info(
        "Von: %s  bis: %s",
        datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )

    all_candles: list[list] = []
    current_since = since_ms

    while current_since < now_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=CANDLES_PER_REQUEST,
            )
        except ccxt.NetworkError as exc:
            logger.error("Netzwerkfehler: %s – warte 5 s und versuche es erneut.", exc)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange-Fehler: %s", exc)
            raise

        if not candles:
            logger.debug("Keine weiteren Kerzen – Abbruch der Schleife.")
            break

        all_candles.extend(candles)

        last_ts = candles[-1][0]
        next_since = last_ts + tf_seconds * 1000

        logger.info(
            "  %d Kerzen geladen  |  bis %s  |  gesamt: %d",
            len(candles),
            datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            len(all_candles),
        )

        if len(candles) < CANDLES_PER_REQUEST:
            break

        current_since = next_since
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not all_candles:
        raise ValueError(f"Keine Daten empfangen für {symbol} / {timeframe}.")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]

    # Duplikate entfernen und sortieren
    df = df.drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Insgesamt %d eindeutige Kerzen geladen.", len(df))
    return df


def save_csv(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    filename = f"{safe_symbol}_{timeframe}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)

    df.to_csv(filepath, index=False)
    logger.info("CSV gespeichert: %s  (%d Zeilen)", filepath, len(df))
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="Historische OHLCV-Daten von Bybit laden.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="z. B. BTC/USDT:USDT")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="z. B. 1m, 5m, 1h")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Anzahl Tage zurück")
    args = parser.parse_args()

    try:
        df = fetch_ohlcv(args.symbol, args.timeframe, args.days)
        save_csv(df, args.symbol, args.timeframe)
    except Exception as exc:
        logger.error("Fehler beim Laden der Daten: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

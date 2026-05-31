"""
Tests für den MarkPrice-Sanity-Check in LiveTrader._place_trigger_entry.

Reproduziert exakt die Fehlerfälle aus den Live-Logs:
  - retCode 110092: trigger_price <= MarkPrice (DOGE, ADA)
  - Stellt sicher, dass ticker["info"]["markPrice"] (Bybit-Raw-Response)
    für den Check verwendet wird, nicht ticker["last"].
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd

from src.live_trader import LiveTrader


def _make_df(n: int = 50, high: float = 0.0995, low: float = 0.095,
             close: float = 0.098) -> pd.DataFrame:
    """Minimales OHLCV-DataFrame das generate_signals akzeptiert."""
    data = {
        "timestamp": list(range(n)),
        "open":      [close] * n,
        "high":      [high]  * n,
        "low":       [low]   * n,
        "close":     [close] * n,
        "volume":    [1000.0] * n,
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def _make_trader(symbol: str = "DOGE/USDT:USDT") -> LiveTrader:
    trader = LiveTrader.__new__(LiveTrader)
    trader._lock               = __import__("threading").Lock()
    trader._stop_event         = __import__("threading").Event()
    trader._thread             = None
    trader.strategy            = None
    trader.symbols             = [symbol]
    trader.symbol              = symbol
    trader.timeframe           = "15m"
    trader.leverage            = 3
    trader.position_size       = 0.10
    trader.tp_pct              = None
    trader.sl_pct              = None
    trader.atr_mode            = False
    trader.atr_period          = 14
    trader.atr_sl_mult         = 1.5
    trader.atr_rr              = 2.0
    trader.use_trailing        = False
    trader.mtf_enabled         = False
    trader.mtf_ema_period      = 50
    trader.adx_enabled         = False
    trader.adx_threshold       = 25.0
    trader.adx_require_trend   = True
    trader.per_symbol_strategies = {}
    trader.use_trigger_entry   = True
    trader.trigger_buffer_long  = 0.005
    trader.trigger_buffer_short = 0.005
    trader.trigger_expiry_min  = 15
    trader.use_atr_trigger     = False
    trader.atr_trigger_mult    = 0.2
    trader.running             = False
    trader.positions           = {symbol: None}
    trader.equity              = 1000.0
    trader.initial_equity      = 1000.0
    trader.last_signal         = 0
    trader.last_tick           = None
    trader.error               = None
    trader.trades              = []
    trader.log                 = []
    trader.candles             = {}
    trader.pending_triggers    = {}
    trader.manual_positions    = []
    trader.manual_trades       = []

    from src.live_trader import TriggerOrderManager
    trader._trigger_mgr = TriggerOrderManager(trader)
    return trader


class TestMarkPriceSanityCheck(unittest.TestCase):
    """
    Sichert, dass _place_trigger_entry den echten Bybit-MarkPrice
    aus ticker["info"]["markPrice"] liest — nicht ticker["last"].
    """

    def _run_trigger(self, mark_price: float, last_price: float,
                     candle_high: float = 0.0995, side: str = "long",
                     symbol: str = "DOGE/USDT:USDT") -> dict:
        """
        Führt _place_trigger_entry aus und gibt ein Dict mit
        {skipped, placed, log_entries} zurück.
        """
        trader   = _make_trader(symbol)
        df       = _make_df(high=candle_high, low=candle_high * 0.995, close=candle_high * 0.998)
        df_closed = df.copy()

        mock_exchange = MagicMock()
        mock_exchange.markets = {"dummy": True}
        mock_exchange.fetch_ticker.return_value = {
            "last":  last_price,
            "info":  {"markPrice": str(mark_price), "lastPrice": str(last_price)},
        }
        mock_exchange.amount_to_precision.side_effect = lambda sym, amt: str(round(amt, 4))

        placed  = []
        skipped = []

        original_place  = trader._trigger_mgr.place
        original_log    = trader._log

        def fake_place(*args, **kwargs):
            placed.append(kwargs or args)
            return True

        def fake_log(msg, level="INFO"):
            trader.log.insert(0, f"[{level}] {msg}")
            if "TRIGGER SKIP" in msg:
                skipped.append(msg)

        trader._trigger_mgr.place = fake_place
        trader._log = fake_log

        current_price = float(df["close"].iloc[-1])
        trader._place_trigger_entry(mock_exchange, symbol, side, df_closed, current_price, 1000.0)

        return {"skipped": skipped, "placed": placed, "log": trader.log}

    # ── Hauptfehler aus den Logs nachgestellt ─────────────────────────────────

    def test_doge_markprice_above_trigger_long_skips(self):
        """
        DOGE: trigger_price ≈ 0.1000 (candle_high 0.0995 * 1.005)
              MarkPrice = 0.1019 → Bybit würde retCode 110092 liefern
              Erwartet: Order wird NICHT platziert.
        """
        result = self._run_trigger(
            mark_price=0.1019, last_price=0.0998,
            candle_high=0.0995, side="long"
        )
        self.assertEqual(len(result["placed"]), 0,
                         "Order darf nicht platziert werden wenn MarkPrice > trigger")
        self.assertEqual(len(result["skipped"]), 1,
                         "TRIGGER SKIP muss geloggt werden")

    def test_ada_markprice_above_trigger_long_skips(self):
        """
        ADA: trigger_price = 0.2400, MarkPrice = 0.2428 → Skip.
        """
        result = self._run_trigger(
            mark_price=0.2428, last_price=0.2395,
            candle_high=0.239, side="long",
            symbol="ADA/USDT:USDT"
        )
        self.assertEqual(len(result["placed"]), 0)
        self.assertTrue(any("TRIGGER SKIP" in s for s in result["skipped"]))

    # ── Normaler Fall: MarkPrice unter Trigger → Order soll platziert werden ─

    def test_markprice_below_trigger_long_places(self):
        """
        MarkPrice = 0.0985, Last = 0.0983, trigger ≈ 0.1000
        → MarkPrice < trigger → Order platzieren.
        """
        result = self._run_trigger(
            mark_price=0.0985, last_price=0.0983,
            candle_high=0.0995, side="long"
        )
        self.assertEqual(len(result["placed"]), 1,
                         "Order MUSS platziert werden wenn MarkPrice < trigger")
        self.assertEqual(len(result["skipped"]), 0)

    def test_short_markprice_below_trigger_skips(self):
        """
        SHORT: trigger = candle_low * (1 - 0.005) ≈ 0.0990
               MarkPrice = 0.0980 (schon tiefer als trigger) → Skip.
        """
        result = self._run_trigger(
            mark_price=0.0980, last_price=0.0988,
            candle_high=0.1005, side="short",
            symbol="DOGE/USDT:USDT"
        )
        # candle_low wird in _make_df als candle_high * 0.995 = 0.0999975 gesetzt
        # trigger_short = 0.0999975 * (1 - 0.005) ≈ 0.0995
        # markPrice 0.0980 < trigger 0.0995 → Skip
        self.assertEqual(len(result["placed"]), 0)
        self.assertEqual(len(result["skipped"]), 1)

    def test_short_markprice_above_trigger_places(self):
        """
        SHORT: trigger ≈ 0.0995, MarkPrice = 0.1010 (noch über trigger) → platzieren.
        """
        result = self._run_trigger(
            mark_price=0.1010, last_price=0.1008,
            candle_high=0.1005, side="short",
        )
        self.assertEqual(len(result["placed"]), 1)
        self.assertEqual(len(result["skipped"]), 0)

    # ── Fallback-Verhalten wenn MarkPrice in info fehlt ───────────────────────

    def test_fallback_to_last_when_markprice_missing(self):
        """
        Wenn info["markPrice"] nicht vorhanden → Fallback auf ticker["last"].
        last < trigger → platzieren.
        """
        trader   = _make_trader()
        df       = _make_df(high=0.0995)
        df_closed = df.copy()

        mock_exchange = MagicMock()
        mock_exchange.markets = {"dummy": True}
        mock_exchange.fetch_ticker.return_value = {
            "last": 0.0983,
            "info": {},          # kein markPrice
        }
        mock_exchange.amount_to_precision.side_effect = lambda sym, amt: str(round(amt, 4))

        placed = []
        trader._trigger_mgr.place = lambda *a, **kw: placed.append(1) or True
        trader._log = lambda msg, level="INFO": None

        trader._place_trigger_entry(mock_exchange, "DOGE/USDT:USDT", "long",
                                    df_closed, 0.0983, 1000.0)
        self.assertEqual(len(placed), 1,
                         "Ohne markPrice in info soll Fallback auf last greifen")

    def test_fallback_to_current_price_when_ticker_fails(self):
        """
        Wenn fetch_ticker eine Exception wirft → Fallback auf current_price.
        current_price < trigger → platzieren (kein Skip).
        """
        trader   = _make_trader()
        df       = _make_df(high=0.0995)
        df_closed = df.copy()

        mock_exchange = MagicMock()
        mock_exchange.markets = {"dummy": True}
        mock_exchange.fetch_ticker.side_effect = RuntimeError("Network error")
        mock_exchange.amount_to_precision.side_effect = lambda sym, amt: str(round(amt, 4))

        placed = []
        trader._trigger_mgr.place = lambda *a, **kw: placed.append(1) or True
        trader._log = lambda msg, level="INFO": None

        current_price = 0.0983  # < trigger (0.0995*1.005 ≈ 0.1000)
        trader._place_trigger_entry(mock_exchange, "DOGE/USDT:USDT", "long",
                                    df_closed, current_price, 1000.0)
        self.assertEqual(len(placed), 1)

    # ── Sicherstellen dass last-only (ohne markPrice) bei überschrittenem Preis auch greift ─

    def test_last_above_trigger_skips_when_no_markprice(self):
        """
        Wenn info kein markPrice hat aber last bereits über trigger liegt → Skip.
        """
        trader   = _make_trader()
        df       = _make_df(high=0.0995)
        df_closed = df.copy()

        mock_exchange = MagicMock()
        mock_exchange.markets = {"dummy": True}
        mock_exchange.fetch_ticker.return_value = {
            "last": 0.1019,    # > trigger 0.1000
            "info": {},
        }
        mock_exchange.amount_to_precision.side_effect = lambda sym, amt: str(round(amt, 4))

        placed  = []
        skipped = []
        def fake_log(msg, level="INFO"):
            if "TRIGGER SKIP" in msg:
                skipped.append(msg)
        trader._trigger_mgr.place = lambda *a, **kw: placed.append(1) or True
        trader._log = fake_log

        trader._place_trigger_entry(mock_exchange, "DOGE/USDT:USDT", "long",
                                    df_closed, 0.0983, 1000.0)
        self.assertEqual(len(placed), 0)
        self.assertEqual(len(skipped), 1)


class TestMarkPriceFieldExtraction(unittest.TestCase):
    """Stellt sicher, dass der richtige Ticker-Pfad für MarkPrice genutzt wird."""

    def _extract(self, ticker: dict) -> float:
        """Spiegelt exakt die Logik in live_trader._place_trigger_entry."""
        _info     = ticker.get("info") or {}
        _mark_str = _info.get("markPrice") or _info.get("lastPrice")
        _mark_val = float(_mark_str) if _mark_str else 0.0
        return _mark_val if _mark_val > 0 else float(ticker.get("last") or 0)

    def test_info_markprice_as_string(self):
        """Bybit liefert markPrice als String in info."""
        ticker = {"last": 0.0983, "info": {"markPrice": "0.1019"}}
        self.assertAlmostEqual(self._extract(ticker), 0.1019)

    def test_info_markprice_as_float(self):
        ticker = {"last": 0.0983, "info": {"markPrice": 0.1019}}
        self.assertAlmostEqual(self._extract(ticker), 0.1019)

    def test_missing_info_falls_back_to_last(self):
        ticker = {"last": 0.0983, "info": {}}
        self.assertAlmostEqual(self._extract(ticker), 0.0983)

    def test_no_info_key_falls_back_to_last(self):
        ticker = {"last": 0.0983}
        self.assertAlmostEqual(self._extract(ticker), 0.0983)

    def test_zero_markprice_falls_back_to_last(self):
        """
        markPrice = '0' → float("0") = 0.0, nicht > 0
        → Fallback auf ticker["last"].
        """
        ticker     = {"last": 0.0983, "info": {"markPrice": "0", "lastPrice": "0.0990"}}
        _info      = ticker.get("info") or {}
        _mark_str  = _info.get("markPrice") or _info.get("lastPrice")
        _mark_val  = float(_mark_str) if _mark_str else 0.0
        result     = _mark_val if _mark_val > 0 else float(ticker.get("last") or 0)
        # "0" → 0.0 → nicht > 0 → Fallback auf last = 0.0983
        self.assertAlmostEqual(result, 0.0983)


if __name__ == "__main__":
    unittest.main(verbosity=2)

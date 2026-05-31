"""
Live-Trader für Bybit Testnet.

Läuft als Daemon-Thread im Hintergrund, fetcht geschlossene Kerzen,
generiert Signale via technischer Strategie und platziert Market-Orders.

Architektur:
  - LiveTrader-Singleton (via st.cache_resource persistent über Rerenders)
  - Background-Thread wartet auf Kerzen-Close, dann _tick()
  - State wird in data/live_state.json persistiert
  - Thread-sichere Kommunikation via Lock
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STATE_FILE    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "data", "live_state.json")
_MAX_LOG       = 300
_MAX_TRADES    = 200

# Thread-Pool für alle Exchange-API-Calls mit hartem Wall-Clock-Timeout.
# Verhindert, dass ein hängender SSL-Shutdown den GIL blockiert und alles einfriert.
_API_POOL    = ThreadPoolExecutor(max_workers=16, thread_name_prefix="bybit-api")
_API_TIMEOUT = 12   # Sekunden pro API-Call (hard deadline)


def _api(fn, *args, **kwargs):
    """Exchange-API-Call mit hartem 12s-Timeout. Wirft RuntimeError bei Timeout."""
    fut = _API_POOL.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=_API_TIMEOUT)
    except _FutureTimeout:
        raise RuntimeError(f"API-Timeout ({_API_TIMEOUT}s): {fn.__name__ if hasattr(fn, '__name__') else fn}")

# Timeframe → Sekunden
_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}


# ── Datenklassen ───────────────────────────────────────────────────────────────

@dataclass
class LivePosition:
    symbol:              str
    side:                str          # "long" | "short"
    entry_price:         float
    amount:              float        # in Basis-Währung (z.B. BTC)
    notional:            float        # in USDT
    leverage:            int
    tp_price:            Optional[float]
    sl_price:            Optional[float]
    opened_at:           str
    trailing_distance:   Optional[float] = None  # nativer Bybit Trailing Stop (USDT-Abstand)
    breakeven_triggered: bool           = False   # True wenn SL bereits auf Entry verschoben
    tpsl_on_exchange:    bool           = False   # True wenn TP/SL auf Bybit bestätigt gesetzt


# ── Trigger-Order Manager ──────────────────────────────────────────────────────

class TriggerOrderManager:
    """
    Verwaltet Bybit-native Conditional Stop-Market Entry Orders (V5 API).

    Pro Symbol darf maximal eine aktive Entry-Trigger-Order existieren.
    Bybit überwacht den Preis serverseitig — kein Bot-seitiges Pollen nötig.

    Pending-Trigger-State wird in LiveTrader.pending_triggers gehalten und
    via live_state.json über Neustarts hinweg persistiert.
    """

    # Bybit Fehlercodes für "Order nicht gefunden / bereits ausgeführt"
    _GONE_CODES = {"110001", "110004", "20001", "Order does not exist"}

    def __init__(self, trader: "LiveTrader") -> None:
        self._t = trader

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def get_pending(self, symbol: str) -> Optional[dict]:
        with self._t._lock:
            return self._t.pending_triggers.get(symbol)

    @staticmethod
    def _bybit_sym(symbol: str) -> str:
        return symbol.replace("/", "").split(":")[0]

    def _is_gone(self, exc: Exception) -> bool:
        s = str(exc)
        return any(c in s for c in self._GONE_CODES)

    # ── Trigger platzieren ────────────────────────────────────────────────────

    def place(
        self, exchange, symbol: str, side: str,
        trigger_price: float, amount: float, notional: float,
        tp: Optional[float], sl: Optional[float],
        expiry_seconds: int,
    ) -> bool:
        """Platziert eine Conditional Stop-Market Entry Order. Gibt True zurück wenn erfolgreich."""
        # triggerDirection: 1 = Preis steigt auf Trigger (Long BreakOut über Hoch)
        #                   2 = Preis fällt auf Trigger (Short BreakOut unter Tief)
        if not exchange.markets:
            exchange.load_markets()
        params: dict = {
            "category":         "linear",
            "symbol":           self._bybit_sym(symbol),
            "orderType":        "Market",
            "side":             "Buy" if side == "long" else "Sell",
            "qty":              exchange.amount_to_precision(symbol, amount),
            "triggerPrice":     str(round(trigger_price, 2)),
            "triggerDirection": 1 if side == "long" else 2,
            "triggerBy":        "MarkPrice",
            "timeInForce":      "GTC",
            "positionIdx":      0,
            "reduceOnly":       False,
        }
        if tp:
            params["takeProfit"]  = str(round(tp, 2))
            params["tpTriggerBy"] = "MarkPrice"
        if sl:
            params["stopLoss"]    = str(round(sl, 2))
            params["slTriggerBy"] = "MarkPrice"

        try:
            resp     = exchange.private_post_v5_order_create(params)
            order_id = (resp.get("result") or {}).get("orderId", "")
            if not order_id:
                self._t._log(f"[{symbol}] TRIGGER: Keine orderId in Response: {resp}", "ERROR")
                return False

            now     = datetime.now(timezone.utc)
            expires = datetime.fromtimestamp(now.timestamp() + expiry_seconds, tz=timezone.utc)
            pending: dict = {
                "symbol":        symbol,
                "order_id":      order_id,
                "side":          side,
                "trigger_price": trigger_price,
                "amount":        amount,
                "notional":      notional,
                "tp_price":      tp,
                "sl_price":      sl,
                "placed_at":     now.strftime("%Y-%m-%d %H:%M:%S"),
                "expires_at":    expires.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with self._t._lock:
                self._t.pending_triggers[symbol] = pending

            tp_str = f"{tp:.2f}" if tp else "–"
            sl_str = f"{sl:.2f}" if sl else "–"
            self._t._log(
                f"[{symbol}] TRIGGER PLATZIERT  {side.upper():<5}  "
                f"@ {trigger_price:.2f}  |  {notional:.2f} USDT  "
                f"|  TP {tp_str}  SL {sl_str}  "
                f"|  ID {order_id}  läuft bis {expires.strftime('%H:%M:%S')} UTC"
            )
            return True
        except Exception as e:
            self._t._log(f"[{symbol}] TRIGGER PLATZIEREN FEHLER: {e}", "ERROR")
            return False

    # ── Trigger canceln ───────────────────────────────────────────────────────

    def cancel(self, exchange, symbol: str, reason: str = "MANUELL") -> bool:
        """Cancelt die offene Trigger Order für dieses Symbol. Gibt True zurück wenn gecancelt."""
        with self._t._lock:
            pending = self._t.pending_triggers.get(symbol)
        if not pending:
            return False

        try:
            exchange.private_post_v5_order_cancel({
                "category":    "linear",
                "symbol":      self._bybit_sym(symbol),
                "orderId":     pending["order_id"],
                "orderFilter": "StopOrder",
            })
        except Exception as e:
            if not self._is_gone(e):
                self._t._log(f"[{symbol}] TRIGGER CANCEL FEHLER: {e}", "ERROR")
            # Fehlercode = bereits weg → State trotzdem bereinigen

        with self._t._lock:
            self._t.pending_triggers.pop(symbol, None)
        self._t._log(
            f"[{symbol}] TRIGGER GECANCELT  [{reason}]  "
            f"ID {pending['order_id']}  war @ {pending['trigger_price']:.2f}"
        )
        return True

    # ── Ablauf prüfen ─────────────────────────────────────────────────────────

    def check_expiry(self, exchange, symbol: str) -> bool:
        """Cancelt Trigger wenn abgelaufen. Gibt True zurück wenn abgelaufen."""
        with self._t._lock:
            pending = self._t.pending_triggers.get(symbol)
        if not pending:
            return False
        now     = datetime.now(timezone.utc)
        expires = datetime.strptime(
            pending["expires_at"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
        if now < expires:
            return False
        self._t._log(
            f"[{symbol}] TRIGGER ABGELAUFEN  "
            f"ID {pending['order_id']}  war @ {pending['trigger_price']:.2f}"
        )
        self.cancel(exchange, symbol, reason="ABGELAUFEN")
        return True

    # ── Startup-Sync ──────────────────────────────────────────────────────────

    def startup_sync(self, exchange, symbols: list) -> None:
        """
        Beim Bot-Start: Alle offenen Stop-Orders vom Exchange laden.
        - Bekannte Orders (in pending_triggers gespeichert) → wiederhergestellt.
        - Orders die nicht mehr offen sind (fired/cancelled while bot was down) → State bereinigt.
        - Verwaiste fremde Orders → auf Exchange gecancelt.
        """
        self._t._log("TRIGGER SYNC: Starte Exchange-Abgleich für offene Stop-Orders…")
        for sym in symbols:
            bybit_sym = self._bybit_sym(sym)
            try:
                resp = exchange.private_get_v5_order_realtime({
                    "category":    "linear",
                    "symbol":      bybit_sym,
                    "orderFilter": "StopOrder",
                    "openOnly":    1,
                    "limit":       10,
                })
                open_ids: set = {
                    o["orderId"]
                    for o in (resp.get("result") or {}).get("list", [])
                    if o.get("orderId")
                }

                with self._t._lock:
                    pending = self._t.pending_triggers.get(sym)

                if pending:
                    known_id = pending["order_id"]
                    if known_id in open_ids:
                        self._t._log(
                            f"[{sym}] TRIGGER SYNC: Order {known_id} noch aktiv — wiederhergestellt."
                        )
                        open_ids.discard(known_id)
                    else:
                        # Trigger hat ausgelöst oder ist abgelaufen während Bot offline war
                        with self._t._lock:
                            self._t.pending_triggers.pop(sym, None)
                        self._t._log(
                            f"[{sym}] TRIGGER SYNC: Gespeicherte Order {known_id} nicht mehr aktiv "
                            f"(ausgelöst/abgelaufen während Offline) — State bereinigt."
                        )

                # Verwaiste Orders (nicht von uns) canceln
                for oid in open_ids:
                    try:
                        exchange.private_post_v5_order_cancel({
                            "category":    "linear",
                            "symbol":      bybit_sym,
                            "orderId":     oid,
                            "orderFilter": "StopOrder",
                        })
                        self._t._log(f"[{sym}] TRIGGER SYNC: Verwaiste Order {oid} gecancelt.")
                    except Exception as ce:
                        if self._is_gone(ce):
                            self._t._log(
                                f"[{sym}] TRIGGER SYNC: Verwaiste Order {oid} bereits ausgeführt/abgelaufen — übersprungen."
                            )
                        else:
                            self._t._log(
                                f"[{sym}] TRIGGER SYNC: Cancel verwaister Order {oid} fehlgeschlagen: {ce}",
                                "ERROR",
                            )
            except Exception as e:
                self._t._log(f"[{sym}] TRIGGER SYNC: Fehler beim Laden der Stop-Orders: {e}", "ERROR")
        self._t._log("TRIGGER SYNC: Abgeschlossen.")


# ── Haupt-Engine ───────────────────────────────────────────────────────────────

class LiveTrader:
    """Singleton – immer via `get_live_trader()` holen."""

    def __init__(self) -> None:
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Konfiguration (wird via configure() gesetzt)
        self.strategy      = None
        self.symbols:  list[str] = ["BTC/USDT:USDT"]   # Multi-Symbol-Liste
        self.symbol        = "BTC/USDT:USDT"            # Compat: erstes Symbol
        self.timeframe     = "15m"
        self.leverage      = 5
        self.position_size = 0.10
        self.tp_pct: Optional[float] = 0.02
        self.sl_pct: Optional[float] = 0.01
        # ATR-basierte TP/SL + Trailing Stop
        self.atr_mode:    bool  = False
        self.atr_period:  int   = 14
        self.atr_sl_mult: float = 1.5
        self.atr_rr:      float = 2.0
        self.use_trailing: bool = False
        # Breakeven SL + Trailing SL (Prozent-basiert, unabhängig von ATR)
        self.breakeven_trigger_pct:    Optional[float] = None  # z.B. 0.5 = 50% des TP-Abstands
        self.trailing_sl_pct:          Optional[float] = None  # z.B. 0.003 = 0.3% Trailing-Abstand
        self.trailing_activation_pct:  Optional[float] = None  # Trailing erst nach X% Gewinn aktiv
        self.trailing_warmup_candles:  int             = 0     # Anzahl Kerzen nach Entry vor Trail-Start
        # MTF + ADX Filter
        self.mtf_enabled:    bool  = False
        self.mtf_ema_period: int   = 50
        self.adx_enabled:      bool  = False
        self.adx_threshold:    float = 25.0
        self.adx_require_trend: bool  = True   # True=Trending(≥), False=Ranging(<)
        # Per-Symbol Strategien (überschreiben die globale Strategie wenn gesetzt)
        self.per_symbol_strategies: dict[str, object] = {}
        # Conditional Trigger Entry
        self.use_trigger_entry:     bool  = True
        self.trigger_buffer_long:   float = 0.005   # 0.5 % über Candle-High
        self.trigger_buffer_short:  float = 0.005   # 0.5 % unter Candle-Low
        self.trigger_expiry_min:    int   = 5        # Ablauf nach 1 Candle (Minuten)
        self.use_atr_trigger:       bool  = False
        self.atr_trigger_mult:      float = 0.2

        # Laufzeit-State
        self.running       = False
        self.positions: dict[str, Optional[LivePosition]] = {}   # sym → Position
        self.equity: Optional[float] = None
        self.initial_equity: Optional[float] = None
        self.last_signal   = 0
        self.last_tick: Optional[str] = None
        self.error: Optional[str] = None
        self.trades: list[dict] = []
        self.log:    list[str]  = []
        self.candles: dict[str, Optional[pd.DataFrame]] = {}     # sym → letztes OHLCV-DF
        self.pending_triggers: dict[str, Optional[dict]] = {}    # sym → Trigger-Info

        # Manuelle Isolated-Positionen
        self.manual_positions: list[LivePosition] = []
        self.manual_trades:    list[dict]         = []

        self._trigger_mgr = TriggerOrderManager(self)
        self._load_state()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO") -> None:
        ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {level}  {msg}"
        with self._lock:
            self.log.insert(0, entry)
            if len(self.log) > _MAX_LOG:
                self.log = self.log[:_MAX_LOG]
        if level == "ERROR":
            logger.error(msg)
        else:
            logger.info(msg)

    # ── State-Persistenz ──────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
            with self._lock:
                data = {
                    "running":        self.running,
                    "symbols":        self.symbols,
                    "symbol":         self.symbol,
                    "timeframe":      self.timeframe,
                    "equity":         self.equity,
                    "initial_equity": self.initial_equity,
                    "last_tick":      self.last_tick,
                    "last_signal":    self.last_signal,
                    "positions":        {sym: asdict(pos) if pos else None
                                         for sym, pos in self.positions.items()},
                    "trades":           self.trades[:_MAX_TRADES],
                    "log":              self.log[:100],
                    "manual_positions": [asdict(p) for p in self.manual_positions],
                    "manual_trades":    self.manual_trades[:_MAX_TRADES],
                    "mtf_enabled":    self.mtf_enabled,
                    "mtf_ema_period": self.mtf_ema_period,
                    "adx_enabled":       self.adx_enabled,
                    "adx_threshold":     self.adx_threshold,
                    "adx_require_trend": self.adx_require_trend,
                    "leverage":        self.leverage,
                    "position_size":   self.position_size,
                    "tp_pct":          self.tp_pct,
                    "sl_pct":          self.sl_pct,
                    "atr_mode":        self.atr_mode,
                    "atr_period":      self.atr_period,
                    "atr_sl_mult":     self.atr_sl_mult,
                    "atr_rr":          self.atr_rr,
                    "use_trailing":               self.use_trailing,
                    "breakeven_trigger_pct":      self.breakeven_trigger_pct,
                    "trailing_sl_pct":            self.trailing_sl_pct,
                    "trailing_activation_pct":    self.trailing_activation_pct,
                    "trailing_warmup_candles":    self.trailing_warmup_candles,
                    "strategy_class":  type(self.strategy).__name__ if self.strategy else None,
                    "strategy_params": (
                        {k: v for k, v in vars(self.strategy).items()
                         if not k.startswith("_") and k != "name"}
                        if self.strategy else {}
                    ),
                    "per_symbol_strategies_data": {
                        sym: {
                            "class": type(s).__name__,
                            "params": {k: v for k, v in vars(s).items()
                                       if not k.startswith("_") and k != "name"},
                        }
                        for sym, s in self.per_symbol_strategies.items()
                    },
                    "use_trigger_entry":    self.use_trigger_entry,
                    "trigger_buffer_long":  self.trigger_buffer_long,
                    "trigger_buffer_short": self.trigger_buffer_short,
                    "trigger_expiry_min":   self.trigger_expiry_min,
                    "use_atr_trigger":      self.use_atr_trigger,
                    "atr_trigger_mult":     self.atr_trigger_mult,
                    "pending_triggers":     {
                        sym: p for sym, p in self.pending_triggers.items() if p
                    },
                }
            with open(_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("State-Save fehlgeschlagen: %s", e)

    def _load_state(self) -> None:
        try:
            if not os.path.exists(_STATE_FILE):
                return
            with open(_STATE_FILE) as f:
                data = json.load(f)
            # Symbols: neues Format zuerst, dann altes "symbol"-Feld
            old_sym      = data.get("symbol", self.symbol)
            self.symbols = data.get("symbols", [old_sym])
            self.symbol  = self.symbols[0] if self.symbols else old_sym
            self.timeframe      = data.get("timeframe", self.timeframe)
            self.equity         = data.get("equity")
            self.initial_equity = data.get("initial_equity")
            self.last_tick      = data.get("last_tick")
            self.last_signal    = data.get("last_signal", 0)
            self.trades         = data.get("trades", [])
            self.log            = data.get("log", [])
            self.manual_trades  = data.get("manual_trades", [])
            self.mtf_enabled    = data.get("mtf_enabled",    False)
            self.mtf_ema_period = data.get("mtf_ema_period", 50)
            self.adx_enabled       = data.get("adx_enabled",       False)
            self.adx_threshold     = data.get("adx_threshold",     25.0)
            self.adx_require_trend = data.get("adx_require_trend", True)
            self.use_trigger_entry    = data.get("use_trigger_entry",    True)
            self.trigger_buffer_long  = data.get("trigger_buffer_long",  0.005)
            self.trigger_buffer_short = data.get("trigger_buffer_short", 0.005)
            self.trigger_expiry_min   = data.get("trigger_expiry_min",   5)
            self.use_atr_trigger      = data.get("use_atr_trigger",      False)
            self.atr_trigger_mult     = data.get("atr_trigger_mult",     0.2)
            self.pending_triggers     = data.get("pending_triggers",     {})
            self.leverage        = data.get("leverage",      self.leverage)
            self.position_size   = data.get("position_size", self.position_size)
            self.tp_pct          = data.get("tp_pct",        self.tp_pct)
            self.sl_pct          = data.get("sl_pct",        self.sl_pct)
            self.atr_mode        = data.get("atr_mode",      self.atr_mode)
            self.atr_period      = data.get("atr_period",    self.atr_period)
            self.atr_sl_mult     = data.get("atr_sl_mult",   self.atr_sl_mult)
            self.atr_rr          = data.get("atr_rr",        self.atr_rr)
            self.use_trailing              = data.get("use_trailing",              self.use_trailing)
            self.breakeven_trigger_pct     = data.get("breakeven_trigger_pct",     None)
            self.trailing_sl_pct           = data.get("trailing_sl_pct",           None)
            self.trailing_activation_pct   = data.get("trailing_activation_pct",   None)
            self.trailing_warmup_candles   = int(data.get("trailing_warmup_candles", 0))
            # Reconstruct strategy objects from saved class name + params
            try:
                from src.strategies import STRATEGY_REGISTRY
                _cls_map = {cls.__name__: cls for cls in STRATEGY_REGISTRY.values()}
                _sc = data.get("strategy_class")
                _sp = data.get("strategy_params", {})
                if _sc and _sc in _cls_map and _sp is not None:
                    self.strategy = _cls_map[_sc](**_sp)
                _pss_data = data.get("per_symbol_strategies_data", {})
                self.per_symbol_strategies = {}
                for _sym, _sd in _pss_data.items():
                    _dc = _sd.get("class")
                    _dp = _sd.get("params", {})
                    if _dc and _dc in _cls_map and _dp is not None:
                        self.per_symbol_strategies[_sym] = _cls_map[_dc](**_dp)
            except Exception as _e:
                logger.warning("Strategie-Wiederherstellung fehlgeschlagen: %s", _e)
            self.manual_positions = []
            for _p in data.get("manual_positions", []):
                _p.setdefault("trailing_distance", None)
                _p.setdefault("breakeven_triggered", False)
                _p.setdefault("tpsl_on_exchange", False)
                self.manual_positions.append(LivePosition(**_p))
            # Auto-Positionen NICHT wiederherstellen → immer von Exchange lesen
            self.positions = {sym: None for sym in self.symbols}
        except Exception as e:
            logger.warning("State-Load fehlgeschlagen: %s", e)

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def configure(self, strategy, symbols: list[str], timeframe: str,
                  leverage: int, position_size: float,
                  tp_pct: Optional[float], sl_pct: Optional[float],
                  atr_mode: bool = False, atr_period: int = 14,
                  atr_sl_mult: float = 1.5, atr_rr: float = 2.0,
                  use_trailing: bool = False,
                  breakeven_trigger_pct: Optional[float] = None,
                  trailing_sl_pct: Optional[float] = None,
                  trailing_activation_pct: Optional[float] = None,
                  trailing_warmup_candles: int = 0,
                  mtf_enabled: bool = False, mtf_ema_period: int = 50,
                  adx_enabled: bool = False, adx_threshold: float = 25.0,
                  adx_require_trend: bool = True,
                  per_symbol_strategies: dict | None = None,
                  use_trigger_entry: bool = True,
                  trigger_buffer_long: float = 0.005,
                  trigger_buffer_short: float = 0.005,
                  trigger_expiry_min: int = 5,
                  use_atr_trigger: bool = False,
                  atr_trigger_mult: float = 0.2) -> None:
        with self._lock:
            self.strategy      = strategy
            self.symbols       = symbols if symbols else ["BTC/USDT:USDT"]
            self.symbol        = self.symbols[0]
            self.timeframe     = timeframe
            self.leverage      = leverage
            self.position_size = position_size
            self.tp_pct        = tp_pct
            self.sl_pct        = sl_pct
            self.atr_mode      = atr_mode
            self.atr_period    = atr_period
            self.atr_sl_mult   = atr_sl_mult
            self.atr_rr        = atr_rr
            self.use_trailing              = use_trailing
            self.breakeven_trigger_pct     = breakeven_trigger_pct
            self.trailing_sl_pct           = trailing_sl_pct
            self.trailing_activation_pct   = trailing_activation_pct
            self.trailing_warmup_candles   = trailing_warmup_candles
            self.mtf_enabled    = mtf_enabled
            self.mtf_ema_period = mtf_ema_period
            self.adx_enabled       = adx_enabled
            self.adx_threshold     = adx_threshold
            self.adx_require_trend = adx_require_trend
            self.per_symbol_strategies = per_symbol_strategies or {}
            self.use_trigger_entry    = use_trigger_entry
            self.trigger_buffer_long  = trigger_buffer_long
            self.trigger_buffer_short = trigger_buffer_short
            self.trigger_expiry_min   = trigger_expiry_min
            self.use_atr_trigger      = use_atr_trigger
            self.atr_trigger_mult     = atr_trigger_mult
            # Neue Symbole in positions-Dict eintragen (bestehende behalten)
            for sym in self.symbols:
                if sym not in self.positions:
                    self.positions[sym] = None
            # Symbole die nicht mehr konfiguriert sind entfernen (wenn keine Position)
            for sym in list(self.positions):
                if sym not in self.symbols and self.positions[sym] is None:
                    del self.positions[sym]

    def start(self) -> None:
        # Prüfe ob Thread wirklich läuft — nicht nur das Flag (kann durch SIGKILL stuck sein)
        if self._thread is not None and self._thread.is_alive():
            return
        self.running = False   # Reset falls State-Datei running=True hatte
        if self.strategy is None:
            raise ValueError("Keine Strategie konfiguriert.")
        if not self.symbols:
            raise ValueError("Mindestens ein Symbol muss konfiguriert sein.")
        self.running = True
        self.error   = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LiveTrader"
        )
        self._thread.start()
        self._log(f"Live-Trader GESTARTET  ·  {self.strategy}  "
                  f"{', '.join(self.symbols)}  {self.timeframe}  {self.leverage}x")
        self._save_state()

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()
        self._log("Live-Trader GESTOPPT.")
        self._save_state()

    def get_status(self) -> dict:
        with self._lock:
            strat_cls    = type(self.strategy).__name__ if self.strategy else None
            strat_params = {}
            if self.strategy:
                strat_params = {k: v for k, v in vars(self.strategy).items()
                                if not k.startswith("_") and k != "name"}
            positions_out = {
                sym: asdict(pos) if pos else None
                for sym, pos in self.positions.items()
            }
            return {
                "running":         self.running,
                "symbols":         list(self.symbols),
                "symbol":          self.symbol,
                "timeframe":       self.timeframe,
                "strategy":        str(self.strategy) if self.strategy else "–",
                "strategy_class":  strat_cls,
                "strategy_params": strat_params,
                "leverage":        self.leverage,
                "position_size":   self.position_size,
                "tp_pct":          self.tp_pct,
                "sl_pct":          self.sl_pct,
                "equity":            self.equity,
                "initial_equity":    self.initial_equity,
                "positions":         positions_out,
                "last_signal":       self.last_signal,
                "last_tick":         self.last_tick,
                "trades":            list(self.trades),
                "log":               list(self.log[:50]),
                "error":             self.error,
                "manual_positions":  [asdict(p) for p in self.manual_positions],
                "manual_trades":     list(self.manual_trades),
                "atr_mode":          self.atr_mode,
                "atr_period":        self.atr_period,
                "atr_sl_mult":       self.atr_sl_mult,
                "atr_rr":            self.atr_rr,
                "use_trailing":               self.use_trailing,
                "breakeven_trigger_pct":      self.breakeven_trigger_pct,
                "trailing_sl_pct":            self.trailing_sl_pct,
                "trailing_activation_pct":    self.trailing_activation_pct,
                "trailing_warmup_candles":    self.trailing_warmup_candles,
                "mtf_enabled":          self.mtf_enabled,
                "mtf_ema_period":       self.mtf_ema_period,
                "adx_enabled":          self.adx_enabled,
                "adx_threshold":        self.adx_threshold,
                "adx_require_trend":    self.adx_require_trend,
                "per_symbol_strategies": {
                    sym: {"class": type(s).__name__, "str": str(s)}
                    for sym, s in self.per_symbol_strategies.items()
                },
                "use_trigger_entry":    self.use_trigger_entry,
                "trigger_buffer_long":  self.trigger_buffer_long,
                "trigger_buffer_short": self.trigger_buffer_short,
                "trigger_expiry_min":   self.trigger_expiry_min,
                "use_atr_trigger":      self.use_atr_trigger,
                "atr_trigger_mult":     self.atr_trigger_mult,
                "pending_triggers":     dict(self.pending_triggers),
            }

    def sync_manual_positions(self) -> int:
        """
        Gleicht manual_positions mit dem Exchange ab.
        Positionen die auf dem Exchange nicht mehr offen sind werden als
        extern geschlossen markiert und in manual_trades verschoben.
        Gibt die Anzahl entfernter Positionen zurück.
        """
        if not self.manual_positions:
            return 0
        try:
            exchange = self._get_exchange()
            symbols  = list({p.symbol for p in self.manual_positions})

            # Alle offenen Positionen vom Exchange holen
            open_on_exchange: set[tuple[str, str]] = set()
            failed_symbols: set[str] = set()
            for sym in symbols:
                try:
                    positions = exchange.fetch_positions([sym])
                    for p in positions:
                        contracts = float(p.get("contracts") or 0)
                        if contracts > 0:
                            side = "long" if p.get("side") == "long" else "short"
                            open_on_exchange.add((p["symbol"], side))
                except Exception as e:
                    self._log(f"fetch_positions({sym}) fehlgeschlagen: {e}", "ERROR")
                    failed_symbols.add(sym)

            removed = 0
            with self._lock:
                still_open = []
                for pos in self.manual_positions:
                    # Bei API-Fehler für dieses Symbol → Position behalten
                    if pos.symbol in failed_symbols:
                        still_open.append(pos)
                        continue
                    if (pos.symbol, pos.side) in open_on_exchange:
                        still_open.append(pos)
                    else:
                        # Extern geschlossen — als Trade eintragen ohne Preis
                        trade = {
                            "symbol":    pos.symbol,
                            "side":      pos.side.upper(),
                            "entry":     round(pos.entry_price, 4),
                            "exit":      None,
                            "notional":  round(pos.notional, 2),
                            "leverage":  pos.leverage,
                            "pnl_usdt":  None,
                            "pnl_pct":   None,
                            "opened_at": pos.opened_at,
                            "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        self.manual_trades.insert(0, trade)
                        removed += 1
                        self._log(f"SYNC: {pos.side.upper()} {pos.symbol} extern geschlossen.")
                self.manual_positions = still_open

            if removed:
                self._save_state()
            return removed
        except Exception as e:
            self._log(f"Sync fehlgeschlagen: {e}", "ERROR")
            return 0

    def close_position_now(self, symbol: Optional[str] = None) -> None:
        """Bot-Positionen manuell schließen. symbol=None → alle Symbole."""
        syms = [symbol] if symbol else list(self.symbols)
        exchange = self._get_exchange()
        for sym in syms:
            pos = self.positions.get(sym)
            if not pos:
                continue
            try:
                df    = self._fetch_candles_for(exchange, sym)
                price = float(df["close"].iloc[-1])
                self._close_position(exchange, sym, pos, price, "MANUAL")
            except Exception as e:
                if self._is_already_closed(e):
                    with self._lock:
                        self.positions[sym] = None
                    self._log(f"close_position_now [{sym}]: bereits geschlossen. State bereinigt.")
                else:
                    self._log(f"Manuelles Schließen [{sym}] fehlgeschlagen: {e}", "ERROR")
        self._save_state()

    def fetch_equity(self) -> Optional[float]:
        """Balance vom Exchange holen und cachen."""
        try:
            exchange = self._get_exchange()
            eq = self._get_balance(exchange)
            with self._lock:
                self.equity = eq
            return eq
        except Exception as e:
            self._log(f"Balance-Abruf fehlgeschlagen: {e}", "ERROR")
            return None

    def open_manual_position(
        self,
        symbol: str,
        side: str,
        pct_of_balance: float,
        leverage: int,
        tp_pct: Optional[float],
        sl_pct: Optional[float],
        atr_mode: bool = False,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_rr: float = 2.0,
        use_trailing: bool = False,
    ) -> tuple[bool, str]:
        """Öffnet eine manuelle Isolated-Futures-Position."""
        try:
            exchange = self._get_exchange()
            equity   = self._get_balance(exchange)
            if equity is None or equity < 0:
                return False, "Balance-Abruf fehlgeschlagen."

            margin   = equity * pct_of_balance          # USDT als Margin
            notional = margin * leverage                 # Positionsgröße in USDT

            # Preis holen
            ticker = exchange.fetch_ticker(symbol)
            price  = float(ticker["last"])
            amount = round(notional / price, 6)

            if amount <= 0:
                return False, "Berechnete Menge = 0."

            # Isolated Margin + Hebel setzen
            try:
                exchange.set_margin_mode(
                    "isolated", symbol,
                    params={"leverage": leverage, "buy_leverage": leverage, "sell_leverage": leverage},
                )
            except Exception:
                pass  # Manche Symbole ignorieren den Aufruf wenn schon gesetzt

            try:
                exchange.set_leverage(leverage, symbol,
                                      params={"positionIdx": 0})
            except Exception:
                pass

            # TP/SL VOR der Order berechnen
            tp         = None
            sl         = None
            trail_dist = None

            if atr_mode:
                df_atr    = self._fetch_candles(exchange)
                atr_val   = self._calc_atr(df_atr, atr_period)
                sl_dist   = atr_val * atr_sl_mult
                tp_dist   = sl_dist * atr_rr
                tp = round(price + tp_dist if side == "long" else price - tp_dist, 4)
                sl = round(price - sl_dist if side == "long" else price + sl_dist, 4)
                if use_trailing:
                    trail_dist = sl_dist
            else:
                if tp_pct:
                    pm = tp_pct / leverage
                    tp = round(price * (1 + pm) if side == "long" else price * (1 - pm), 4)
                if sl_pct:
                    pm = sl_pct / leverage
                    sl = round(price * (1 - pm) if side == "long" else price * (1 + pm), 4)

            # TP/SL direkt in der Order mitschicken
            def _mprice_str(p: float) -> str:
                decimals = 4 if price < 1.0 else 2
                return str(round(p, decimals))

            order_params: dict = {"positionIdx": 0, "reduceOnly": False}
            if tp:
                order_params["takeProfit"]  = _mprice_str(tp)
                order_params["tpTriggerBy"] = "MarkPrice"
            if sl:
                order_params["stopLoss"]    = _mprice_str(sl)
                order_params["slTriggerBy"] = "MarkPrice"

            order = exchange.create_market_order(
                symbol=symbol,
                side="buy" if side == "long" else "sell",
                amount=amount,
                params=order_params,
            )
            ep = float(order.get("average") or order.get("price") or price)

            if trail_dist:
                self._set_trailing_stop_on_exchange(exchange, symbol, trail_dist)

            pos = LivePosition(
                symbol=symbol, side=side,
                entry_price=ep, amount=amount, notional=notional,
                leverage=leverage, tp_price=tp, sl_price=sl,
                opened_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                trailing_distance=trail_dist,
            )
            with self._lock:
                self.manual_positions.append(pos)
                self.equity = equity

            _tp_str  = f"{tp:.4f}"         if tp         else "-"
            _sl_str  = f"{sl:.4f}"         if sl         else "-"
            _trl_str = f"{trail_dist:.2f}" if trail_dist else "-"
            self._log(
                f"MANUAL OPEN  {side.upper():<5}  {symbol}  @ {ep:.4f}"
                f"  |  {notional:.2f} USDT  {leverage}x ISOLATED"
                f"  |  TP {_tp_str}  SL {_sl_str}  Trail {_trl_str}"
            )
            self._save_state()
            return True, f"{side.upper()} {symbol} @ {ep:.4f} — {notional:.2f} USDT ({leverage}x)"

        except Exception as e:
            self._log(f"MANUAL OPEN FEHLER: {e}", "ERROR")
            return False, str(e)

    def close_manual_position(self, symbol: str, side: str) -> tuple[bool, str]:
        """Schließt eine manuelle Position anhand Symbol+Seite."""
        with self._lock:
            pos = next(
                (p for p in self.manual_positions if p.symbol == symbol and p.side == side),
                None,
            )
        if not pos:
            return False, f"Keine offene {side.upper()}-Position für {symbol}."

        try:
            exchange = self._get_exchange()
            ticker   = exchange.fetch_ticker(symbol)
            price    = float(ticker["last"])

            order = exchange.create_market_order(
                symbol=symbol,
                side="sell" if pos.side == "long" else "buy",
                amount=pos.amount,
                params={"positionIdx": 0, "reduceOnly": True},
            )
            ep = float(order.get("average") or order.get("price") or price)

            if pos.side == "long":
                raw_pnl = (ep - pos.entry_price) / pos.entry_price * pos.notional
            else:
                raw_pnl = (pos.entry_price - ep) / pos.entry_price * pos.notional
            fees    = pos.notional * 0.00055 * 2
            net_pnl = raw_pnl - fees
            margin  = pos.notional / pos.leverage
            pnl_pct = (net_pnl / margin * 100) if margin else 0.0

            trade = {
                "symbol":    pos.symbol,
                "side":      pos.side.upper(),
                "entry":     round(pos.entry_price, 4),
                "exit":      round(ep, 4),
                "notional":  round(pos.notional, 2),
                "leverage":  pos.leverage,
                "pnl_usdt":  round(net_pnl, 4),
                "pnl_pct":   round(pnl_pct, 3),
                "opened_at": pos.opened_at,
                "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }
            with self._lock:
                self.manual_positions = [
                    p for p in self.manual_positions
                    if not (p.symbol == symbol and p.side == side)
                ]
                self.manual_trades.insert(0, trade)

            self._log(
                f"MANUAL CLOSE {pos.side.upper():<5}  {symbol}  @ {ep:.4f}"
                f"  |  PnL {net_pnl:+.4f} USDT ({pnl_pct:+.2f}%)"
            )
            self._save_state()
            return True, f"PnL: {net_pnl:+.4f} USDT ({pnl_pct:+.2f}%)"

        except Exception as e:
            if self._is_already_closed(e):
                # Position wurde extern geschlossen (TP/SL getriggert) — State bereinigen
                with self._lock:
                    self.manual_positions = [
                        p for p in self.manual_positions
                        if not (p.symbol == symbol and p.side == side)
                    ]
                self._log(f"MANUAL CLOSE: Position {side.upper()} {symbol} war bereits auf Exchange geschlossen (TP/SL). State bereinigt.")
                self._save_state()
                return True, "Position war bereits auf Exchange geschlossen (TP/SL getriggert)."
            self._log(f"MANUAL CLOSE FEHLER: {e}", "ERROR")
            return False, str(e)

    # ── Exchange-Operationen ──────────────────────────────────────────────────

    @staticmethod
    def _is_already_closed(e: Exception) -> bool:
        """True wenn Bybit meldet dass keine Position mehr offen ist (z.B. via TP/SL)."""
        s = str(e)
        return "110017" in s or "110025" in s or "position is zero" in s.lower()

    def _get_exchange(self):
        from src.exchange import get_exchange
        return get_exchange()

    def _set_tp_sl_on_exchange(
        self, exchange, symbol: str,
        tp_price: Optional[float], sl_price: Optional[float],
    ) -> None:
        """Setzt TP/SL direkt in der Bybit-Position (V5 trading-stop API)."""
        if not tp_price and not sl_price:
            return
        bybit_sym = symbol.replace("/", "").split(":")[0]
        params: dict = {"category": "linear", "symbol": bybit_sym, "positionIdx": 0}
        if tp_price:
            params["takeProfit"]   = str(round(tp_price, 2))
            params["tpTriggerBy"]  = "MarkPrice"
        if sl_price:
            params["stopLoss"]     = str(round(sl_price, 2))
            params["slTriggerBy"]  = "MarkPrice"
        try:
            exchange.private_post_v5_position_trading_stop(params)
            self._log(
                f"TP/SL auf Exchange gesetzt  "
                f"TP={round(tp_price,2) if tp_price else '-'}  "
                f"SL={round(sl_price,2) if sl_price else '-'}"
            )
        except Exception as e:
            self._log(f"TP/SL Exchange-Setzen fehlgeschlagen: {e}", "ERROR")

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else float(tr.mean())

    def _clear_trading_stops(self, exchange, symbol: str) -> None:
        """Löscht alle gespeicherten TP/SL/Trailing-Stop-Defaults auf Bybit für dieses Symbol.
        Bybit speichert diese Werte nach Positionsschließung als "Position-Level Default" und
        wendet sie automatisch auf neue Positionen an — auch wenn die Order selbst keine TP/SL hat.
        Das führt zu Ablehnungen wenn die neue Position in entgegengesetzter Richtung läuft.
        """
        bybit_sym = symbol.replace("/", "").split(":")[0]
        params = {
            "category":     "linear",
            "symbol":       bybit_sym,
            "positionIdx":  0,
            "takeProfit":   "0",
            "stopLoss":     "0",
            "trailingStop": "0",
        }
        try:
            exchange.private_post_v5_position_trading_stop(params)
        except Exception:
            pass  # Fehler ignorieren — kein Stop vorhanden ist kein Fehler

    def _set_trailing_stop_on_exchange(
        self, exchange, symbol: str, trail_distance: float,
        active_price: float | None = None,
    ) -> bool:
        """Setzt nativen Bybit Trailing Stop. Gibt True zurück wenn erfolgreich, sonst False."""
        bybit_sym = symbol.replace("/", "").split(":")[0]
        trail_val = round(trail_distance, 6)
        if trail_val <= 0:
            self._log(f"Trailing Stop übersprungen: Abstand {trail_distance} ≤ 0", "ERROR")
            return False
        params = {
            "category":     "linear",
            "symbol":       bybit_sym,
            "positionIdx":  0,
            "trailingStop": str(trail_val),
        }
        if active_price is not None:
            decimals = 4 if active_price < 1.0 else 2
            params["activePrice"] = str(round(active_price, decimals))
        try:
            exchange.private_post_v5_position_trading_stop(params)
            act_str = f"  activePrice={params['activePrice']}" if active_price else ""
            self._log(f"Trailing Stop gesetzt: {trail_val} USDT Abstand{act_str}")
            return True
        except Exception as e:
            self._log(f"Trailing Stop setzen fehlgeschlagen: {e}", "ERROR")
            return False

    def _fetch_candles(self, exchange) -> pd.DataFrame:
        return self._fetch_candles_for(exchange, self.symbol)

    def _fetch_candles_for(self, exchange, symbol: str) -> pd.DataFrame:
        """Direkter Exchange-Call — kein _api()-Wrapper, da Timeout vom Aufrufer verwaltet wird."""
        ohlcv = exchange.fetch_ohlcv(symbol, self.timeframe, limit=400)
        df = pd.DataFrame(ohlcv,
                          columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def _fetch_candles_tf(self, exchange, symbol: str, timeframe: str) -> pd.DataFrame:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=300)
        df = pd.DataFrame(ohlcv,
                          columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def _fetch_symbol_data(self, exchange, symbol: str):
        """Pool-Worker: holt Candles (+ HTF für MTF-Filter) + offene Positionen."""
        df        = self._fetch_candles_for(exchange, symbol)
        df_15m    = self._fetch_candles_tf(exchange, symbol, "15m") if self.mtf_enabled else None
        df_1h     = self._fetch_candles_tf(exchange, symbol, "1h")  if self.mtf_enabled else None
        positions = exchange.fetch_positions([symbol])
        return symbol, df, positions, df_15m, df_1h

    def _get_balance(self, exchange) -> float:
        try:
            bal = _api(exchange.fetch_balance)
            # Verschiedene Bybit-API-Versionen / ccxt-Versionen
            usdt = (bal.get("USDT", {}).get("free")
                    or bal.get("free", {}).get("USDT")
                    or bal.get("total", {}).get("USDT")
                    or 0.0)
            return float(usdt)
        except Exception as e:
            self._log(f"Balance-Abruf fehlgeschlagen: {e}", "ERROR")
            return 0.0

    @staticmethod
    def _fetch_positions_for(exchange, sym: str) -> tuple[str, list]:
        """Statische Hilfsfunktion für Pool-Worker — Exchange wird NICHT hier erstellt."""
        return sym, exchange.fetch_positions([sym])

    def _sync_position_state(self, exchange, sym: str, live_positions: list) -> None:
        """Gleicht lokalen State mit den Exchange-Live-Daten ab.
        Erkennt:
        - Manuelle Schließungen und TP/SL-Auslösungen (Position verschwunden)
        - Trigger-Ausführungen (Position erschienen ohne lokalen State)
        """
        open_on_exchange = [
            p for p in live_positions
            if p.get("contracts", 0) and float(p["contracts"]) > 0
        ]
        msgs = []
        tp_sl_to_set: Optional[tuple] = None  # (tp, sl) nach dem Lock setzen
        with self._lock:
            current = self.positions.get(sym)

            if current and not open_on_exchange:
                # Position wurde extern geschlossen (TP/SL oder manuell)
                trade = {
                    "index":     len(self.trades),
                    "symbol":    sym,
                    "side":      current.side.upper(),
                    "entry":     round(current.entry_price, 4),
                    "exit":      None,
                    "notional":  round(current.notional, 2),
                    "pnl_usdt":  None,
                    "pnl_pct":   None,
                    "reason":    "EXTERN",
                    "opened_at": current.opened_at,
                    "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.trades.insert(0, trade)
                self.positions[sym] = None
                msgs.append(f"[{sym}] Position extern geschlossen (manuell/TP/SL). State bereinigt.")

            elif open_on_exchange and not current:
                # Position erschienen — Trigger ausgelöst oder extern eröffnet
                p    = open_on_exchange[0]
                side = "long" if p.get("side") == "long" else "short"
                ep   = float(p.get("entryPrice") or p.get("averagePrice") or 0)
                amt  = float(p.get("contracts") or 0)
                # TP/SL aus pending_trigger übernehmen falls vorhanden
                pending = self.pending_triggers.get(sym)
                tp_p    = pending["tp_price"] if pending else None
                sl_p    = pending["sl_price"] if pending else None
                self.positions[sym] = LivePosition(
                    symbol=sym, side=side,
                    entry_price=ep, amount=amt, notional=round(amt * ep, 2),
                    leverage=self.leverage, tp_price=tp_p, sl_price=sl_p,
                    opened_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )
                if pending:
                    self.pending_triggers.pop(sym, None)
                    msgs.append(
                        f"[{sym}] TRIGGER AUSGELÖST: {side.upper()} {amt:.4f} @ {ep:.2f}  "
                        f"|  TP {f'{tp_p:.2f}' if tp_p else '–'}  SL {f'{sl_p:.2f}' if sl_p else '–'}"
                    )
                    # TP/SL nochmal explizit auf Exchange setzen — Bybit hängt sie bei
                    # Conditional Orders nicht immer zuverlässig an die resultierende Position
                    if tp_p or sl_p:
                        tp_sl_to_set = (tp_p, sl_p)
                    # trailing_distance noch nicht gesetzt → _ensure_trailing_stop erledigt es
                else:
                    msgs.append(
                        f"[{sym}] Position extern eröffnet (unbekannte Herkunft): "
                        f"{side.upper()} {amt:.4f} @ {ep:.2f}"
                    )
        for msg in msgs:
            self._log(msg)
        if tp_sl_to_set:
            tp_p, sl_p = tp_sl_to_set
            try:
                self._set_tp_sl_on_exchange(exchange, sym, tp_p, sl_p)
                self._log(f"[{sym}] TP/SL nach Trigger-Entry auf Exchange bestätigt: TP={tp_p:.4f}  SL={sl_p:.4f}")
                with self._lock:
                    if self.positions.get(sym):
                        self.positions[sym].tpsl_on_exchange = True
            except Exception as e:
                self._log(f"[{sym}] TP/SL-Bestätigung nach Trigger fehlgeschlagen: {e}", "ERROR")
        # Trailing Stop und TP/SL sofort nach Erkennung setzen (nicht erst beim nächsten Tick)
        self._ensure_trailing_stop(exchange, sym)
        self._ensure_tp_sl(exchange, sym)

    def _reconcile_position(self, _exchange=None) -> None:
        """Gleicht lokalen State mit Exchange ab — alle Symbole parallel.
        Exchange-Instanzen werden im aufrufenden Thread erstellt (nicht im Pool),
        um GC-bedingten SSL-Shutdown im Pool-Thread zu verhindern.
        """
        syms = list(self.symbols)
        # Exchanges im LiveTrader-Thread erstellen — nicht im Pool-Worker!
        exchanges = {sym: self._get_exchange() for sym in syms}
        futs = {
            sym: _API_POOL.submit(self._fetch_positions_for, exchanges[sym], sym)
            for sym in syms
        }
        deadline = time.monotonic() + _API_TIMEOUT + 5
        for sym, fut in futs.items():
            remaining = max(0.5, deadline - time.monotonic())
            try:
                _, positions = fut.result(timeout=remaining)
                open_pos = [p for p in positions
                            if p.get("contracts", 0) and float(p["contracts"]) > 0]
                msg_to_log = None
                with self._lock:
                    current = self.positions.get(sym)
                    if open_pos and not current:
                        p    = open_pos[0]
                        side = "long" if p.get("side") == "long" else "short"
                        ep   = float(p.get("entryPrice") or p.get("averagePrice") or 0)
                        amt  = float(p.get("contracts") or 0)
                        # TP/SL direkt aus Exchange-Antwort lesen (Bybit liefert sie im info-Dict)
                        _info  = p.get("info") or {}
                        tp_raw = _info.get("takeProfit") or _info.get("tp") or ""
                        sl_raw = _info.get("stopLoss")   or _info.get("sl") or ""
                        tp_ex  = float(tp_raw) if tp_raw and float(tp_raw) > 0 else None
                        sl_ex  = float(sl_raw) if sl_raw and float(sl_raw) > 0 else None
                        self.positions[sym] = LivePosition(
                            symbol=sym, side=side,
                            entry_price=ep, amount=amt, notional=amt * ep,
                            leverage=self.leverage, tp_price=tp_ex, sl_price=sl_ex,
                            opened_at="(aus Exchange wiederhergestellt)",
                            tpsl_on_exchange=(tp_ex is not None or sl_ex is not None),
                        )
                        tp_info = f"TP={tp_ex:.4f}" if tp_ex else "TP=–"
                        sl_info = f"SL={sl_ex:.4f}" if sl_ex else "SL=–"
                        msg_to_log = f"[{sym}] Position wiederhergestellt: {side.upper()} {amt} @ {ep}  {tp_info}  {sl_info}"
                    elif not open_pos and current:
                        self.positions[sym] = None
                        msg_to_log = f"[{sym}] Lokale Position gelöscht (Exchange: keine offene Position)."
                if msg_to_log:
                    self._log(msg_to_log)
            except _FutureTimeout:
                self._log(f"[{sym}] Positions-Abgleich Timeout — übersprungen.", "ERROR")
            except Exception as e:
                self._log(f"[{sym}] Positions-Abgleich fehlgeschlagen: {e}", "ERROR")
        del exchanges

    def _set_isolated_leverage(self, exchange, symbol: str) -> None:
        try:
            exchange.set_margin_mode(
                "isolated", symbol,
                params={"leverage": self.leverage,
                        "buy_leverage": self.leverage,
                        "sell_leverage": self.leverage},
            )
        except Exception:
            pass
        try:
            exchange.set_leverage(self.leverage, symbol, params={"positionIdx": 0})
        except Exception as e:
            self._log(f"[{symbol}] Leverage-Fehler (ignoriert): {e}")

    def _open_position(self, exchange, symbol: str, side: str,
                       price: float, equity: float,
                       df: Optional[pd.DataFrame] = None) -> None:
        # Safety-Check: Position nur eröffnen wenn mindestens eine Exit-Absicherung konfiguriert
        has_protection = (
            self.tp_pct or self.sl_pct or self.trailing_sl_pct or self.atr_mode
        )
        if not has_protection:
            self._log(
                f"[{symbol}] ABGEBROCHEN: Keine Exit-Absicherung konfiguriert "
                f"(kein TP, SL oder Trailing Stop). Position wird NICHT eröffnet.",
                "ERROR",
            )
            return

        notional = equity * self.position_size * self.leverage
        amount   = round(notional / price, 6)
        if amount <= 0:
            self._log(f"[{symbol}] Berechnete Menge = 0, Trade übersprungen.", "ERROR")
            return

        try:
            self._set_isolated_leverage(exchange, symbol)
            # Stale TP/SL/Trailing-Stop-Defaults löschen — Bybit behält diese nach
            # Positionsschließung und wendet sie automatisch auf neue Positionen an.
            # Bei Richtungswechsel liegen sie auf der falschen Seite → Order-Fehler.
            self._clear_trading_stops(exchange, symbol)

            # TP/SL VOR der Order berechnen (auf Basis Marktpreis)
            tp         = None
            sl         = None
            trail_dist = None

            if self.atr_mode and df is not None:
                atr_val  = self._calc_atr(df, self.atr_period)
                sl_dist  = atr_val * self.atr_sl_mult
                tp_dist  = sl_dist * self.atr_rr
                tp = round(price + tp_dist if side == "long" else price - tp_dist, 4)
                sl = round(price - sl_dist if side == "long" else price + sl_dist, 4)
                if self.use_trailing:
                    trail_dist = sl_dist
            else:
                if self.tp_pct:
                    pm = self.tp_pct / self.leverage
                    tp = round(price * (1 + pm) if side == "long" else price * (1 - pm), 4)
                if self.sl_pct:
                    pm = self.sl_pct / self.leverage
                    sl = round(price * (1 - pm) if side == "long" else price * (1 + pm), 4)
                if self.trailing_sl_pct:
                    # Floor = 0.01% des Kurses statt fixer 0.01 USDT.
                    # 0.01 USDT wäre für Coins < 0.10$ (z.B. JELLYJELLY @ 0.057)
                    # ein 17%+ Abstand statt der gewünschten 0.5%.
                    trail_dist = max(round(price * self.trailing_sl_pct, 6),
                                     round(price * 0.0001, 6))

            # TP/SL direkt in der Order mitschicken (atomisch, kein separater API-Call nötig)
            # Dezimalstellen dynamisch: mind. 4 für Coins < 1 USDT, sonst 2.
            def _price_str(p: float) -> str:
                decimals = 4 if price < 1.0 else 2
                return str(round(p, decimals))

            order_params: dict = {"positionIdx": 0, "reduceOnly": False}
            if tp:
                order_params["takeProfit"]  = _price_str(tp)
                order_params["tpTriggerBy"] = "MarkPrice"
            if sl:
                # Sicherheitscheck: SL muss auf der richtigen Seite des Entry liegen.
                sl_valid = (sl < price) if side == "long" else (sl > price)
                if not sl_valid:
                    self._log(f"[{symbol}] SL {sl:.6f} auf falscher Seite von Entry {price:.6f} — SL ignoriert.", "ERROR")
                    sl = None
                else:
                    order_params["stopLoss"]    = _price_str(sl)
                    order_params["slTriggerBy"] = "MarkPrice"

            order = exchange.create_market_order(
                symbol=symbol,
                side="buy" if side == "long" else "sell",
                amount=amount,
                params=order_params,
            )
            ep = float(order.get("average") or order.get("price") or price)

            trail_set = False
            if trail_dist and self.trailing_warmup_candles == 0:
                # Nur sofort setzen wenn kein Warmup konfiguriert.
                # Mit Warmup: _ensure_trailing_stop setzt Trail nach N Kerzen.
                act_price = None
                if self.trailing_activation_pct:
                    act_price = (ep * (1 + self.trailing_activation_pct) if side == "long"
                                 else ep * (1 - self.trailing_activation_pct))
                trail_set = self._set_trailing_stop_on_exchange(exchange, symbol, trail_dist, act_price)
                if not trail_set:
                    self._log(
                        f"[{symbol}] WARNUNG: Trailing Stop konnte nicht gesetzt werden — "
                        f"wird beim nächsten Tick automatisch erneut versucht.",
                        "ERROR",
                    )

            # trailing_distance nur im State setzen wenn der API-Call erfolgreich war.
            # Bei None bleibt _ensure_trailing_stop aktiv und retried beim nächsten Tick.
            with self._lock:
                self.positions[symbol] = LivePosition(
                    symbol=symbol, side=side,
                    entry_price=ep, amount=amount, notional=notional,
                    leverage=self.leverage, tp_price=tp, sl_price=sl,
                    opened_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    trailing_distance=trail_dist if trail_set else None,
                    tpsl_on_exchange=(tp is not None or sl is not None),
                )
            tp_str  = f"{tp:.2f}"         if tp         else "–"
            sl_str  = f"{sl:.2f}"         if sl         else "–"
            trl_str = f"{trail_dist:.2f}" if trail_dist else "–"
            self._log(f"OPEN  {side.upper():<5}  [{symbol}]  "
                      f"@ {ep:.2f}  |  {notional:.2f} USDT  "
                      f"|  TP {tp_str}  SL {sl_str}  Trail {trl_str}")
        except Exception as e:
            self._log(f"ORDER FEHLER (open) [{symbol}]: {e}", "ERROR")
            raise

    def _place_trigger_entry(
        self, exchange, symbol: str, side: str,
        df_closed: pd.DataFrame, current_price: float, equity: float,
    ) -> None:
        """
        Berechnet Trigger-Preis und platziert eine Conditional Stop-Market Entry Order.
        Der Trigger liegt über dem Hoch (Long) bzw. unter dem Tief (Short) der Signal-Kerze.
        TP/SL werden auf Basis des Trigger-Preises berechnet und direkt in der Order gesetzt.
        """
        # Safety-Check: keine Trigger-Order ohne Exit-Absicherung
        has_protection = (
            self.tp_pct or self.sl_pct or self.trailing_sl_pct or self.atr_mode
        )
        if not has_protection:
            self._log(
                f"[{symbol}] TRIGGER ABGEBROCHEN: Keine Exit-Absicherung konfiguriert "
                f"(kein TP, SL oder Trailing Stop).",
                "ERROR",
            )
            return

        signal_candle = df_closed.iloc[-1]

        # ── Trigger-Preis ─────────────────────────────────────────────────────
        if self.use_atr_trigger:
            atr_val      = self._calc_atr(df_closed, self.atr_period)
            dist         = atr_val * self.atr_trigger_mult
            trigger_price = (
                round(current_price + dist, 4) if side == "long"
                else round(current_price - dist, 4)
            )
        else:
            if side == "long":
                trigger_price = round(
                    float(signal_candle["high"]) * (1 + self.trigger_buffer_long), 4
                )
            else:
                trigger_price = round(
                    float(signal_candle["low"]) * (1 - self.trigger_buffer_short), 4
                )

        # ── Sanity-Check: Echtzeit-MarkPrice abrufen und Trigger prüfen ─────────
        # current_price aus OHLCV kann bei schnellen Moves veraltet sein →
        # live Ticker holen um Race Condition gegen Bybit MarkPrice zu vermeiden.
        # WICHTIG: ccxt legt MarkPrice für Bybit nicht als Top-Level-Feld ab →
        # ticker["info"]["markPrice"] nutzen (Raw-API-Response).
        try:
            ticker     = exchange.fetch_ticker(symbol)
            _info      = ticker.get("info") or {}
            _mark_str  = _info.get("markPrice") or _info.get("lastPrice")
            _mark_val  = float(_mark_str) if _mark_str else 0.0
            live_price = _mark_val if _mark_val > 0 else float(ticker.get("last") or current_price)
        except Exception:
            live_price = current_price

        if side == "long" and live_price >= trigger_price:
            self._log(
                f"[{symbol}] TRIGGER SKIP: Live-Kurs {live_price:.4f} bereits über "
                f"Trigger {trigger_price:.4f} (Candle-High + Buffer) — kein Entry."
            )
            return
        if side == "short" and live_price <= trigger_price:
            self._log(
                f"[{symbol}] TRIGGER SKIP: Live-Kurs {live_price:.4f} bereits unter "
                f"Trigger {trigger_price:.4f} (Candle-Low - Buffer) — kein Entry."
            )
            return

        # ── Notional und Menge (auf Basis Trigger-Preis) ─────────────────────
        notional = equity * self.position_size * self.leverage
        amount   = round(notional / trigger_price, 6)
        if amount <= 0:
            self._log(f"[{symbol}] TRIGGER: Berechnete Menge = 0 — übersprungen.", "ERROR")
            return

        # ── TP/SL auf Basis des Trigger-Preises berechnen ────────────────────
        tp = sl = None
        if self.atr_mode:
            atr_val  = self._calc_atr(df_closed, self.atr_period)
            sl_dist  = atr_val * self.atr_sl_mult
            tp_dist  = sl_dist * self.atr_rr
            tp = round(
                trigger_price + tp_dist if side == "long" else trigger_price - tp_dist, 4
            )
            sl = round(
                trigger_price - sl_dist if side == "long" else trigger_price + sl_dist, 4
            )
        else:
            if self.tp_pct:
                pm = self.tp_pct / self.leverage
                tp = round(
                    trigger_price * (1 + pm) if side == "long"
                    else trigger_price * (1 - pm), 4
                )
            if self.sl_pct:
                pm = self.sl_pct / self.leverage
                sl = round(
                    trigger_price * (1 - pm) if side == "long"
                    else trigger_price * (1 + pm), 4
                )

        # ── Leverage/Margin-Modus vor der Order setzen ────────────────────────
        self._set_isolated_leverage(exchange, symbol)
        self._clear_trading_stops(exchange, symbol)

        expiry_secs = self.trigger_expiry_min * 60
        self._trigger_mgr.place(
            exchange, symbol, side,
            trigger_price, amount, notional,
            tp, sl, expiry_secs,
        )

    def _ensure_trailing_stop(self, exchange, symbol: str) -> None:
        """Setzt Trailing Stop falls noch nicht gesetzt — respektiert Warmup-Periode.
        Self-Healing: Gespeicherter Wert wird gegen Sollwert geprüft. Bei >10% Abweichung
        wird der Trailing Stop als veraltet behandelt und automatisch auf Bybit korrigiert.
        trailing_distance im State wird NUR gesetzt wenn der API-Call erfolgreich war.
        Bei Fehler bleibt trailing_distance=None → nächster Tick versucht es erneut.
        """
        if not self.trailing_sl_pct or self.atr_mode:
            return
        pos = self.positions.get(symbol)
        if not pos:
            return

        # Sollwert aus aktueller Strategie berechnen
        expected_trail = max(
            round(pos.entry_price * self.trailing_sl_pct, 6),
            round(pos.entry_price * 0.0001, 6),
        )

        # Self-Healing: gespeicherten Wert gegen Sollwert prüfen
        is_stale = False
        if pos.trailing_distance is not None:
            deviation = abs(pos.trailing_distance - expected_trail) / expected_trail
            if deviation <= 0.10:
                return  # Wert passt — kein API-Call nötig
            is_stale = True
            self._log(
                f"[{symbol}] Trailing Stop SELF-HEAL: "
                f"gespeichert={pos.trailing_distance:.6f}  "
                f"erwartet={expected_trail:.6f}  "
                f"Abweichung={deviation*100:.1f}% → wird korrigiert"
            )

        # Warmup: Trail erst nach N abgeschlossenen Kerzen setzen
        # (gilt nur für Erst-Setup, nicht für Self-Heal eines bereits aktiven Stops)
        if not is_stale and self.trailing_warmup_candles > 0:
            try:
                from datetime import timezone as _tz
                opened  = datetime.fromisoformat(pos.opened_at).replace(tzinfo=_tz.utc)
                elapsed = (datetime.now(_tz.utc) - opened).total_seconds()
                tf_secs = _TF_SECONDS.get(self.timeframe, 3600)
                if elapsed < self.trailing_warmup_candles * tf_secs:
                    return  # Warmup noch nicht abgelaufen
            except Exception:
                pass  # Parsing-Fehler: Trail sofort setzen

        ok = self._set_trailing_stop_on_exchange(exchange, symbol, expected_trail)
        if ok:
            with self._lock:
                if self.positions.get(symbol):
                    self.positions[symbol].trailing_distance = expected_trail
        else:
            self._log(
                f"[{symbol}] Trailing Stop nicht gesetzt — wird beim nächsten Tick erneut versucht.",
                "ERROR",
            )

    def _ensure_tp_sl(self, exchange, symbol: str) -> None:
        """Stellt TP/SL auf Bybit wieder her wenn sie fehlen.

        Verwendet tpsl_on_exchange-Flag um sicher zu verfolgen ob Bybit die Werte hat.
        Nutzt gespeicherte Preise aus dem Positions-State direkt (nicht nur Config-Prozente).
        Prüft vor dem Setzen ob Werte nicht sofort triggern würden (Sicherheitscheck
        gegen aktuellen MarkPrice aus Cache — kein extra API-Call nötig).
        """
        if self.atr_mode:
            return
        pos = self.positions.get(symbol)
        if not pos:
            return
        if pos.tpsl_on_exchange:
            return  # Bybit-Zustand bestätigt — kein API-Call nötig

        # ── Kandidaten-Preise: erst aus State, dann aus Config berechnen ──────
        tp_candidate = pos.tp_price
        sl_candidate = pos.sl_price
        ep = pos.entry_price

        if tp_candidate is None and self.tp_pct and not self.atr_mode:
            pm = self.tp_pct / self.leverage
            tp_candidate = round(
                ep * (1 + pm) if pos.side == "long" else ep * (1 - pm), 4
            )
        if sl_candidate is None and self.sl_pct and not self.atr_mode:
            pm = self.sl_pct / self.leverage
            sl_raw = round(
                ep * (1 - pm) if pos.side == "long" else ep * (1 + pm), 4
            )
            sl_valid = (sl_raw < ep) if pos.side == "long" else (sl_raw > ep)
            sl_candidate = sl_raw if sl_valid else None

        if not tp_candidate and not sl_candidate:
            return

        # ── Sicherheitsprüfung: kein Wert der sofort triggern würde ──────────
        # Aktuellen Preis aus gecachten Candles holen — kein extra API-Call.
        current_price: Optional[float] = None
        df_cached = self.candles.get(symbol)
        if df_cached is not None and len(df_cached) > 0:
            current_price = float(df_cached["close"].iloc[-1])

        tp_to_set = tp_candidate
        sl_to_set = sl_candidate

        if current_price is not None:
            if pos.side == "long":
                # LONG TP triggert wenn MarkPrice >= TP
                if tp_to_set is not None and current_price >= tp_to_set:
                    self._log(
                        f"[{symbol}] ENSURE-TP übersprungen: "
                        f"MarkPrice {current_price:.6f} ≥ TP {tp_to_set:.6f} "
                        f"(würde sofort triggern)"
                    )
                    tp_to_set = None
                # LONG SL triggert wenn MarkPrice <= SL
                if sl_to_set is not None and current_price <= sl_to_set:
                    self._log(
                        f"[{symbol}] ENSURE-SL übersprungen: "
                        f"MarkPrice {current_price:.6f} ≤ SL {sl_to_set:.6f} "
                        f"(würde sofort triggern)"
                    )
                    sl_to_set = None
            else:  # short
                # SHORT TP triggert wenn MarkPrice <= TP
                if tp_to_set is not None and current_price <= tp_to_set:
                    self._log(
                        f"[{symbol}] ENSURE-TP übersprungen: "
                        f"MarkPrice {current_price:.6f} ≤ TP {tp_to_set:.6f} "
                        f"(würde sofort triggern)"
                    )
                    tp_to_set = None
                # SHORT SL triggert wenn MarkPrice >= SL
                if sl_to_set is not None and current_price >= sl_to_set:
                    self._log(
                        f"[{symbol}] ENSURE-SL übersprungen: "
                        f"MarkPrice {current_price:.6f} ≥ SL {sl_to_set:.6f} "
                        f"(würde sofort triggern)"
                    )
                    sl_to_set = None

        if not tp_to_set and not sl_to_set:
            # Alle Kandidaten ungültig — Flag setzen damit kein permanentes Retrying entsteht
            with self._lock:
                if self.positions.get(symbol):
                    self.positions[symbol].tpsl_on_exchange = True
            return

        try:
            self._set_tp_sl_on_exchange(exchange, symbol, tp_to_set, sl_to_set)
            with self._lock:
                pos2 = self.positions.get(symbol)
                if pos2:
                    if tp_to_set is not None:
                        pos2.tp_price = tp_to_set
                    if sl_to_set is not None:
                        pos2.sl_price = sl_to_set
                    pos2.tpsl_on_exchange = True
            tp_str = f"{tp_to_set:.6f}" if tp_to_set else "–"
            sl_str = f"{sl_to_set:.6f}" if sl_to_set else "–"
            self._log(f"[{symbol}] TP/SL wiederhergestellt: TP {tp_str}  SL {sl_str}")
        except Exception as e:
            self._log(f"[{symbol}] TP/SL Wiederherstellung fehlgeschlagen: {e}", "ERROR")

    def _check_breakeven_for(
        self, exchange, symbol: str, current_high: float, current_low: float
    ) -> None:
        """Verschiebt SL auf Entry sobald der Breakeven-Trigger-Kurs erreicht wird."""
        if self.breakeven_trigger_pct is None:
            return
        pos = self.positions.get(symbol)
        if not pos or pos.breakeven_triggered:
            return
        if pos.tp_price is None:
            return

        tp_dist      = abs(pos.tp_price - pos.entry_price)
        trigger_dist = tp_dist * self.breakeven_trigger_pct

        if pos.side == "long":
            trigger_price = pos.entry_price + trigger_dist
            reached       = current_high >= trigger_price
        else:
            trigger_price = pos.entry_price - trigger_dist
            reached       = current_low  <= trigger_price

        if not reached:
            return

        # SL schon auf Entry oder besser → nur Flag setzen
        already_safe = (
            (pos.side == "long"  and pos.sl_price is not None and pos.sl_price >= pos.entry_price) or
            (pos.side == "short" and pos.sl_price is not None and pos.sl_price <= pos.entry_price)
        )
        if not already_safe:
            self._set_tp_sl_on_exchange(exchange, symbol, None, pos.entry_price)
            with self._lock:
                if self.positions.get(symbol):
                    self.positions[symbol].sl_price = pos.entry_price

        with self._lock:
            if self.positions.get(symbol):
                self.positions[symbol].breakeven_triggered = True

        self._log(
            f"[{symbol}] BREAKEVEN SL: SL → Entry @ {pos.entry_price:.2f}  "
            f"(Trigger {trigger_price:.2f} = {self.breakeven_trigger_pct*100:.0f}% des TP-Abstands erreicht)"
        )
        self._save_state()

    def _close_position(self, exchange, symbol: str, pos: LivePosition,
                        price: float, reason: str) -> None:
        try:
            order = exchange.create_market_order(
                symbol=symbol,
                side="sell" if pos.side == "long" else "buy",
                amount=pos.amount,
                params={"positionIdx": 0, "reduceOnly": True},
            )
            ep = float(order.get("average") or order.get("price") or price)

            if pos.side == "long":
                raw_pnl = (ep - pos.entry_price) / pos.entry_price * pos.notional
            else:
                raw_pnl = (pos.entry_price - ep) / pos.entry_price * pos.notional
            fees    = pos.notional * 0.00055 * 2
            net_pnl = raw_pnl - fees
            margin  = pos.notional / pos.leverage
            pnl_pct = (net_pnl / margin * 100) if margin else 0.0

            trade = {
                "index":       len(self.trades),
                "symbol":      symbol,
                "side":        pos.side.upper(),
                "entry":       round(pos.entry_price, 2),
                "exit":        round(ep, 2),
                "notional":    round(pos.notional, 2),
                "pnl_usdt":    round(net_pnl, 4),
                "pnl_pct":     round(pnl_pct, 3),
                "reason":      reason.upper(),
                "opened_at":   pos.opened_at,
                "closed_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }
            with self._lock:
                self.trades.insert(0, trade)
                self.positions[symbol] = None

            self._log(f"CLOSE {pos.side.upper():<5}  [{symbol}]  "
                      f"@ {ep:.2f}  |  PnL {net_pnl:+.4f} USDT "
                      f"({pnl_pct:+.2f}%)  [{reason}]")
        except Exception as e:
            if self._is_already_closed(e):
                with self._lock:
                    self.positions[symbol] = None
                self._log(f"CLOSE [{symbol}]: bereits auf Exchange geschlossen (TP/SL). State bereinigt.")
                self._save_state()
                return
            self._log(f"ORDER FEHLER (close) [{symbol}]: {e}", "ERROR")
            raise

    def _check_tp_sl(self, pos: Optional[LivePosition], price: float) -> Optional[str]:
        if not pos:
            return None
        if pos.side == "long":
            if pos.tp_price and price >= pos.tp_price:
                return "tp"
            if pos.sl_price and price <= pos.sl_price:
                return "sl"
        else:
            if pos.tp_price and price <= pos.tp_price:
                return "tp"
            if pos.sl_price and price >= pos.sl_price:
                return "sl"
        return None

    # ── Haupt-Loop ────────────────────────────────────────────────────────────

    def _seconds_to_next_candle(self) -> int:
        s   = _TF_SECONDS.get(self.timeframe, 60)
        now = int(time.time())
        return s - (now % s) + 3   # +3s Puffer

    def _run(self) -> None:
        self._log("Hintergrund-Thread gestartet.")
        try:
            self._reconcile_position()
        except Exception as e:
            self._log(f"Verbindungstest fehlgeschlagen: {e}", "ERROR")
        try:
            _ex_sync = self._get_exchange()
            self._trigger_mgr.startup_sync(_ex_sync, list(self.symbols))
        except Exception as e:
            self._log(f"Trigger-Startup-Sync fehlgeschlagen: {e}", "ERROR")

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                self.error = str(e)
                self._log(f"TICK FEHLER: {e}", "ERROR")
                self._stop_event.wait(30)
                continue

            wait = self._seconds_to_next_candle()
            self._log(f"Warte {wait}s auf nächste Kerze…")
            self._stop_event.wait(wait)

        with self._lock:
            self.running = False
        self._save_state()
        self._log("Hintergrund-Thread beendet.")

    def _tick(self) -> None:
        exchange = self._get_exchange()
        equity   = self._get_balance(exchange)

        with self._lock:
            self.equity    = equity
            self.last_tick = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if self.initial_equity is None:
                self.initial_equity = equity

        # Pro Symbol: Candles + aktuelle Positionen parallel fetchen.
        # Exchanges im LiveTrader-Thread erstellen (nicht im Pool-Worker) — verhindert
        # GC-bedingten SSL-Shutdown im Pool-Thread, der den GIL blockieren würde.
        syms = list(self.symbols)
        sym_exchanges = {sym: self._get_exchange() for sym in syms}
        futs = {
            sym: _API_POOL.submit(self._fetch_symbol_data, sym_exchanges[sym], sym)
            for sym in syms
        }
        deadline = time.monotonic() + _API_TIMEOUT + 5
        for sym, fut in futs.items():
            remaining = max(0.5, deadline - time.monotonic())
            try:
                _, df, live_positions, df_15m, df_1h = fut.result(timeout=remaining)
                self._sync_position_state(exchange, sym, live_positions)
                self._process_tick_symbol(exchange, sym, df, equity, df_15m, df_1h)
            except _FutureTimeout:
                self._log(f"[{sym}] Fetch Timeout — übersprungen.", "ERROR")
            except Exception as e:
                self._log(f"[{sym}] Fetch-Fehler: {e}", "ERROR")
        del sym_exchanges

        self._save_state()

    @staticmethod
    def _compute_adx_arr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        n = len(c)
        adx_out = np.zeros(n)
        if n < period * 2:
            return adx_out
        tr  = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        pdm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0.0), 0.0)
        ndm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0.0), 0.0)
        atr14 = np.zeros(n - 1); pdi14 = np.zeros(n - 1); ndi14 = np.zeros(n - 1)
        atr14[period - 1] = tr[:period].sum()
        pdi14[period - 1] = pdm[:period].sum()
        ndi14[period - 1] = ndm[:period].sum()
        for i in range(period, n - 1):
            atr14[i] = atr14[i - 1] - atr14[i - 1] / period + tr[i]
            pdi14[i] = pdi14[i - 1] - pdi14[i - 1] / period + pdm[i]
            ndi14[i] = ndi14[i - 1] - ndi14[i - 1] / period + ndm[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(atr14 > 0, pdi14 / atr14 * 100, 0.0)
            ndi = np.where(atr14 > 0, ndi14 / atr14 * 100, 0.0)
            dx  = np.where((pdi + ndi) > 0, np.abs(pdi - ndi) / (pdi + ndi) * 100, 0.0)
        adx_v = np.zeros(n - 1)
        adx_v[2 * period - 2] = dx[period - 1:2 * period - 1].mean()
        for i in range(2 * period - 1, n - 1):
            adx_v[i] = (adx_v[i - 1] * (period - 1) + dx[i]) / period
        adx_out[1:] = adx_v
        return adx_out

    @staticmethod
    def _apply_mtf_filter(primary_df: pd.DataFrame, htf_dfs: list,
                          signals: np.ndarray, ema_period: int) -> np.ndarray:
        primary_times = pd.to_datetime(primary_df["datetime"]).reset_index(drop=True)
        signals = signals.copy()
        for val_df in htf_dfs:
            if val_df is None or len(val_df) == 0:
                continue
            val_times = pd.to_datetime(val_df["datetime"]).reset_index(drop=True)
            ema       = val_df["close"].ewm(span=ema_period, adjust=False).mean()
            trend     = np.where(val_df["close"].to_numpy() >= ema.to_numpy(), 1, -1)
            htf = pd.DataFrame({"time": val_times, "trend": trend})
            pri = pd.DataFrame({"time": primary_times})
            merged = pd.merge_asof(
                pri.sort_values("time"), htf.sort_values("time"),
                on="time", direction="backward",
            ).sort_values("time").reset_index(drop=True)
            htf_trend = merged["trend"].fillna(0).to_numpy(int)
            mask = ((signals == 1) & (htf_trend == 1)) | ((signals == -1) & (htf_trend == -1))
            signals = np.where(mask, signals, 0).astype(int)
        return signals

    def _process_tick_symbol(self, exchange, symbol: str, df: pd.DataFrame, equity: float,
                             df_15m: Optional[pd.DataFrame] = None,
                             df_1h:  Optional[pd.DataFrame] = None) -> None:
        # ── Signal auf letzter ABGESCHLOSSENER Kerze ──────────────────────────
        df_closed  = df.iloc[:-1].copy()
        # Per-Symbol-Strategie hat Vorrang vor globaler Strategie
        _active_strategy = self.per_symbol_strategies.get(symbol, self.strategy)
        sigs_s     = _active_strategy.generate_signals(df_closed)
        sigs       = sigs_s.to_numpy().astype(int)
        raw_signal = int(sigs[-1])

        # MTF-Filter
        if self.mtf_enabled:
            htf_dfs = [d.iloc[:-1] for d in [df_15m, df_1h]
                       if d is not None and len(d) > 1]
            if htf_dfs:
                sigs = self._apply_mtf_filter(df_closed, htf_dfs, sigs, self.mtf_ema_period)
        mtf_signal = int(sigs[-1])

        # ADX-Filter
        adx_now = 0.0
        if self.adx_enabled:
            adx_v   = self._compute_adx_arr(df_closed)
            adx_now = float(adx_v[-1]) if len(adx_v) else 0.0
            if self.adx_require_trend:
                sigs = np.where(adx_v >= self.adx_threshold, sigs, 0).astype(int)
            else:
                sigs = np.where(adx_v < self.adx_threshold, sigs, 0).astype(int)
        adx_signal = int(sigs[-1])

        signal = adx_signal

        # ── Signal-Pipeline Logging (nur wenn irgendwo was gefiltert wurde) ──
        if raw_signal != 0 or mtf_signal != 0 or adx_signal != 0:
            parts = [f"raw={raw_signal}"]
            if self.mtf_enabled:
                blocked_mtf = "✗" if raw_signal != 0 and mtf_signal == 0 else "✓"
                parts.append(f"MTF={mtf_signal}({blocked_mtf})")
            if self.adx_enabled:
                blocked_adx = "✗" if mtf_signal != 0 and adx_signal == 0 else "✓"
                mode_sym = "≥" if self.adx_require_trend else "<"
                parts.append(f"ADX={adx_signal}({blocked_adx})  ADX={adx_now:.1f}{mode_sym}{self.adx_threshold:.0f}")
            self._log(f"[{symbol}]  Signal: {'  →  '.join(parts)}")

        with self._lock:
            self.last_signal = signal
            self.candles[symbol] = df.copy()

        current_price = float(df["close"].iloc[-1])
        pos = self.positions.get(symbol)

        # ── Trailing Stop und TP/SL sicherstellen (z.B. nach Trigger-Entry oder Extern-Position) ──
        self._ensure_trailing_stop(exchange, symbol)
        self._ensure_tp_sl(exchange, symbol)

        # ── Breakeven SL prüfen ───────────────────────────────────────────────
        if pos:
            last_high = float(df["high"].iloc[-1])
            last_low  = float(df["low"].iloc[-1])
            self._check_breakeven_for(exchange, symbol, last_high, last_low)

        # ── Abgelaufene Trigger-Orders canceln ───────────────────────────────
        self._trigger_mgr.check_expiry(exchange, symbol)

        # ── Signal-basiertes Exit (Market Order — immer sofort) ───────────────
        if pos:
            pos_sig = 1 if pos.side == "long" else -1
            if signal != 0 and signal != pos_sig:
                # Evtl. offene Trigger-Order in gleicher Richtung zuerst canceln
                self._trigger_mgr.cancel(exchange, symbol, reason="SIGNAL-FLIP")
                self._close_position(exchange, symbol, pos, current_price, "SIGNAL")
                pos = None

        # ── Entry ──────────────────────────────────────────────────────────────
        if pos is None and signal in (1, -1):
            side = "long" if signal == 1 else "short"

            # Live-Supertrend-Check (auf aktuellem, noch offenem Candle)
            # WICHTIG: Absichtlich df (mit Live-Candle), NICHT df_closed!
            # Analyse zeigt: Trades die hier geblockt werden haben nur 7.6% Win-Rate
            # und -0.783R avg → der Check rettet ~+113R pro Jahr.
            if hasattr(_active_strategy, "live_st_direction"):
                live_dir = _active_strategy.live_st_direction(df)
                if live_dir != signal:
                    self._log(
                        f"[{symbol}] ST-Live-Check GEBLOCKT: "
                        f"Signal {'+1' if signal == 1 else '-1'} ({side.upper()}) "
                        f"aber Supertrend zeigt {'BULLISH' if live_dir == 1 else 'BEARISH'} "
                        f"auf aktuellem Candle — Entry abgebrochen."
                    )
                    self._trigger_mgr.cancel(exchange, symbol, reason="ST-LIVE-BLOCK")
                    return

            if self.use_trigger_entry:
                # ── Conditional Trigger Entry ─────────────────────────────────
                pending = self._trigger_mgr.get_pending(symbol)
                if pending:
                    if pending["side"] == side:
                        # Gleiche Richtung: Trigger läuft noch → warten
                        remaining = pending["expires_at"]
                        self._log(
                            f"[{symbol}] TRIGGER aktiv ({side.upper()} "
                            f"@ {pending['trigger_price']:.2f}) — läuft bis {remaining[11:19]} UTC."
                        )
                        return
                    else:
                        # Richtungswechsel: alten Trigger canceln, neuen platzieren
                        self._trigger_mgr.cancel(exchange, symbol, reason="SIGNAL-WECHSEL")

                self._place_trigger_entry(exchange, symbol, side, df_closed, current_price, equity)
            else:
                # ── Fallback: Sofortige Market Order (alter Modus) ────────────
                self._open_position(exchange, symbol, side, current_price, equity, df)

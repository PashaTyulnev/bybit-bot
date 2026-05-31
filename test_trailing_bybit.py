#!/usr/bin/env python3
"""
Test: Prüft ob Bybit-Demo Trailing Stop und Breakeven SL (via trading-stop API) unterstützt.

Erwartete Ergebnisse:
  retCode 0          → Feature unterstützt + Positions war offen (Einstellung gesetzt)
  retCode 110017     → Feature unterstützt, aber keine offene Position
  retCode 110025     → Feature unterstützt, aber Position already closed
  anderer retCode    → Möglicher Fehler (Parameter abgelehnt)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import ccxt
from src.config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_API_URL

SYMBOL    = "BTC/USDT:USDT"
BYBIT_SYM = "BTCUSDT"
DEMO_URL  = "https://api-demo.bybit.com"

active_url = (BYBIT_API_URL or DEMO_URL).rstrip("/")
print(f"[INFO] Exchange URL: {active_url}")

ex = ccxt.bybit({
    "apiKey":          BYBIT_API_KEY,
    "secret":          BYBIT_API_SECRET,
    "enableRateLimit": True,
    "timeout":         20000,
    "options": {
        "defaultType":             "linear",
        "adjustForTimeDifference": True,
        "recvWindow":              10000,
    },
    "urls": {"api": {"public": active_url, "private": active_url}},
})
ex.has["fetchCurrencies"] = False
ex.is_unified_enabled = lambda: (True, True)

# Aktuellen Preis holen
ticker = ex.fetch_ticker(SYMBOL)
price  = float(ticker["last"])
print(f"[INFO] BTC aktueller Kurs: {price:.2f} USDT")


# "Keine Position" Fehlermeldungen — der Parameter ist bekannt, nur kein Trade offen
_NO_POS_HINTS = {
    "zero position", "position size is zero", "position does not exist",
    "no position", "can not set tp/sl/ts for zero position",
}

def _is_no_position(msg: str) -> bool:
    m = msg.lower()
    return any(h in m for h in _NO_POS_HINTS)


def check_feature(name: str, params: dict) -> bool:
    """
    Gibt True zurück wenn das Feature unterstützt wird.
    retCode 0                 → Einstellung gesetzt (Position war offen)
    retCode 110017/110025     → Bybit-native "keine Position" Codes
    retCode 10001 + no-pos    → Parameter erkannt, nur keine Position vorhanden
    anderes                   → echter Fehler / Parameter nicht unterstützt
    """
    try:
        resp     = ex.private_post_v5_position_trading_stop(params)
        ret_code = int(resp.get("retCode", -1))
        ret_msg  = resp.get("retMsg", "")
        if ret_code == 0:
            print(f"  [OK – GESETZT]   retCode=0")
            return True
        if ret_code in {110017, 110025} or _is_no_position(ret_msg):
            print(f"  [OK – KEINE POS] retCode={ret_code}  retMsg={ret_msg}")
            return True
        print(f"  [FEHLER]         retCode={ret_code}  retMsg={ret_msg}")
        return False
    except Exception as e:
        s = str(e)
        # ccxt wirft Exception mit der JSON-Antwort als String
        if _is_no_position(s):
            print(f"  [OK – KEINE POS] {s}")
            return True
        print(f"  [EXCEPTION]      {s}")
        return False


print("\n" + "=" * 60)
print("TEST 1: Bybit Trailing Stop (native, server-seitig)")
print("=" * 60)
print(f"  Parameter: trailingStop = {round(price * 0.008, 2)} USDT (~0.8% des Kurses)")
r1 = check_feature("Trailing Stop", {
    "category":     "linear",
    "symbol":       BYBIT_SYM,
    "positionIdx":  0,
    "trailingStop": str(round(price * 0.008, 2)),
})

print("\n" + "=" * 60)
print("TEST 2: Breakeven SL — SL auf Entry-Preis verschieben (via trading-stop)")
print("=" * 60)
fake_entry = round(price * 0.995, 2)   # simulierter Entry 0.5% unter Kurs
print(f"  Parameter: stopLoss = {fake_entry} USDT (simulierter Entry-Preis)")
r2 = check_feature("Breakeven (SL-Update)", {
    "category":    "linear",
    "symbol":      BYBIT_SYM,
    "positionIdx": 0,
    "stopLoss":    str(fake_entry),
    "slTriggerBy": "MarkPrice",
})

print("\n" + "=" * 60)
print("TEST 3: Beide gleichzeitig (Trailing + neuer SL in einem Call)")
print("=" * 60)
r3 = check_feature("Trailing + SL kombiniert", {
    "category":     "linear",
    "symbol":       BYBIT_SYM,
    "positionIdx":  0,
    "trailingStop": str(round(price * 0.008, 2)),
    "stopLoss":     str(round(price * 0.99, 2)),
    "slTriggerBy":  "MarkPrice",
})

print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print(f"  Trailing Stop (native Bybit):  {'JA ✓' if r1 else 'NEIN ✗'}")
print(f"  Breakeven SL (via API):        {'JA ✓' if r2 else 'NEIN ✗'}")
print(f"  Beides kombiniert:             {'JA ✓' if r3 else 'NEIN ✗'}")

if r1 and r2:
    print("\n[OK] Beide Features werden von Bybit unterstützt.")
    print("     Trailing Stop: Bybit verwaltet ihn server-seitig automatisch.")
    print("     Breakeven SL:  Bot verschiebt SL via trading-stop API wenn Trigger erreicht.")
else:
    print("\n[WARNUNG] Mindestens ein Feature nicht verfügbar — Details oben prüfen.")

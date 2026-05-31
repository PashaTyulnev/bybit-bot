#!/usr/bin/env python3
"""
Test: Conditional Stop-Market Order für BNB/USDT:USDT
Gleiche Logik wie live_trader.py – platziert einen Trigger weit vom Kurs
und cancelt ihn sofort wieder.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import ccxt
from src.config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_API_URL

SYMBOL      = "BNB/USDT:USDT"
BYBIT_SYM   = "BNBUSDT"
EQUITY      = 200.0      # Testwert
POS_SIZE    = 0.10
LEVERAGE    = 5

DEMO_URL = "https://api-demo.bybit.com"
active_url = (BYBIT_API_URL or DEMO_URL).rstrip("/")

print(f"[INFO] Exchange URL: {active_url}")

ex = ccxt.bybit({
    "apiKey":          BYBIT_API_KEY,
    "secret":          BYBIT_API_SECRET,
    "enableRateLimit": True,
    "timeout":         20000,
    "options": {
        "defaultType":              "linear",
        "adjustForTimeDifference":  True,
        "recvWindow":               10000,
    },
    "urls": {"api": {"public": active_url, "private": active_url}},
})
ex.has["fetchCurrencies"] = False
ex.is_unified_enabled = lambda: (True, True)

# 1) Märkte laden (wichtig für amount_to_precision!)
print("[INFO] Lade Märkte ...")
ex.load_markets()
print(f"[INFO] Märkte geladen. BNB/USDT:USDT vorhanden: {SYMBOL in ex.markets}")

# 2) Aktuellen Preis holen
ticker = ex.fetch_ticker(SYMBOL)
current_price = float(ticker["last"])
print(f"[INFO] BNB aktueller Kurs: {current_price:.2f} USDT")

# 3) Menge berechnen – gleiche Logik wie live_trader.py Zeile 1262
notional = EQUITY * POS_SIZE * LEVERAGE
amount   = notional / current_price

# Mindestmenge aus Marktinfo lesen
market  = ex.markets[SYMBOL]
min_amt = (market.get("limits") or {}).get("amount", {}).get("min") or 0.01
amt_step = (market.get("precision") or {}).get("amount") or 0.01

print(f"[INFO] Notional: {notional:.2f} USDT | Rohbetrag: {amount:.6f} BNB")
print(f"[INFO] Markt Min-Qty: {min_amt} | Qty-Step: {amt_step}")

# Präzision über ccxt anwenden
qty_str = ex.amount_to_precision(SYMBOL, amount)
print(f"[INFO] qty nach amount_to_precision: '{qty_str}'")

qty_float = float(qty_str)
if qty_float < min_amt:
    print(f"[FEHLER] qty {qty_float} < Minimum {min_amt} — Order würde fehlschlagen!")
    print(f"[FIX]   Erhöhe EQUITY oder reduziere Symbole, damit Notional > {min_amt * current_price / (POS_SIZE * LEVERAGE):.2f} USDT")
    sys.exit(1)

# 4) Trigger weit vom Kurs setzen (Long-Trigger 50% über Kurs → feuert nie)
side          = "long"
trigger_price = round(current_price * 1.50, 2)
print(f"\n[INFO] Platziere Trigger: {side.upper()} @ {trigger_price:.2f} (50% über Kurs — feuert nicht)")

params = {
    "category":         "linear",
    "symbol":           BYBIT_SYM,
    "orderType":        "Market",
    "side":             "Buy",
    "qty":              qty_str,
    "triggerPrice":     str(trigger_price),
    "triggerDirection": 1,
    "triggerBy":        "MarkPrice",
    "timeInForce":      "GTC",
    "positionIdx":      0,
    "reduceOnly":       False,
}
print(f"[DEBUG] Request-Params: {params}")

try:
    resp     = ex.private_post_v5_order_create(params)
    order_id = (resp.get("result") or {}).get("orderId", "")
    ret_code = resp.get("retCode")
    ret_msg  = resp.get("retMsg")

    if int(ret_code) != 0 or not order_id:
        print(f"[FEHLER] retCode={ret_code}  retMsg={ret_msg}")
        print(f"[FEHLER] Volle Response: {resp}")
        sys.exit(1)

    print(f"[OK] Trigger platziert! orderId={order_id}")

    # 5) Sofort wieder canceln
    cancel_resp = ex.private_post_v5_order_cancel({
        "category":    "linear",
        "symbol":      BYBIT_SYM,
        "orderId":     order_id,
        "orderFilter": "StopOrder",
    })
    cancel_code = cancel_resp.get("retCode")
    print(f"[OK] Trigger gecancelt! retCode={cancel_code}")
    print("\n[ERFOLG] Test bestanden — Trigger-Order funktioniert korrekt.")

except Exception as e:
    print(f"[EXCEPTION] {e}")
    sys.exit(1)

"""
Startet den LiveTrader direkt via Konsole ohne Streamlit-UI.
Konfiguration: 5 Assets, Supertrend, 20x Hebel, ATR-Modus.
"""
import sys
import os
import time
import signal

# Pfad setzen damit src-Imports funktionieren
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from src.live_trader import LiveTrader
from src.strategies.supertrend_strategy import SupertrendStrategy

SYMBOLS    = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
              "BNB/USDT:USDT", "XRP/USDT:USDT"]
TIMEFRAME  = "1m"
LEVERAGE   = 20
POS_SIZE   = 0.10      # 10% vom Konto
ATR_PERIOD = 14
ATR_SL_MULT = 0.5
ATR_RR     = 0.5
TRAILING   = True

trader = LiveTrader()
strategy = SupertrendStrategy(atr_period=10, multiplier=3.0)

trader.configure(
    strategy      = strategy,
    symbols       = SYMBOLS,
    timeframe     = TIMEFRAME,
    leverage      = LEVERAGE,
    position_size = POS_SIZE,
    tp_pct        = None,
    sl_pct        = None,
    atr_mode      = True,
    atr_period    = ATR_PERIOD,
    atr_sl_mult   = ATR_SL_MULT,
    atr_rr        = ATR_RR,
    use_trailing  = TRAILING,
)

def stop(sig, frame):
    print("\n[STOP] Bot wird gestoppt...")
    trader.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

trader.start()
print(f"[START] Bot läuft: {SYMBOLS}  {TIMEFRAME}  {LEVERAGE}x  ATR SL={ATR_SL_MULT}  RR={ATR_RR}  Trailing={TRAILING}")
print("Ctrl+C zum Stoppen\n")

last_log_count = 0
while True:
    time.sleep(3)
    status = trader.get_status()
    logs   = status.get("log", [])
    if not status["running"]:
        print("[FEHLER] Bot ist nicht mehr aktiv!")
        break
    # Neue Log-Einträge ausgeben
    new_entries = logs[:len(logs) - last_log_count] if len(logs) > last_log_count else logs[:(len(logs) - last_log_count) or None]
    new_count = len(logs) - last_log_count
    if new_count > 0:
        for entry in reversed(logs[:new_count]):
            print(entry)
        last_log_count = len(logs)

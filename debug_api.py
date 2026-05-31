"""Minimaler Test: fetch_positions + fetch_ohlcv für 5 Symbole parallel."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FT
from src.exchange import get_exchange

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
           "BNB/USDT:USDT", "XRP/USDT:USDT"]

def test_positions(sym):
    t0 = time.time()
    ex = get_exchange()
    try:
        r = ex.fetch_positions([sym])
        return sym, "OK", round(time.time()-t0,2), len(r)
    except Exception as e:
        return sym, "ERR", round(time.time()-t0,2), str(e)[:80]

def test_ohlcv(sym):
    t0 = time.time()
    ex = get_exchange()
    try:
        r = ex.fetch_ohlcv(sym, "1m", limit=10)
        return sym, "OK", round(time.time()-t0,2), len(r)
    except Exception as e:
        return sym, "ERR", round(time.time()-t0,2), str(e)[:80]

pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="test")

print("=== TEST fetch_positions (5 Symbole parallel) ===")
t0 = time.time()
futs = {pool.submit(test_positions, sym): sym for sym in SYMBOLS}
try:
    for fut in as_completed(futs, timeout=20):
        sym, status, dt, extra = fut.result(timeout=1)
        print(f"  {sym:25s} {status}  {dt}s  {extra}")
except FT:
    print("  TIMEOUT nach 20s!")
print(f"  Gesamt: {round(time.time()-t0,2)}s\n")

print("=== TEST fetch_ohlcv (5 Symbole parallel) ===")
t0 = time.time()
futs = {pool.submit(test_ohlcv, sym): sym for sym in SYMBOLS}
try:
    for fut in as_completed(futs, timeout=20):
        sym, status, dt, extra = fut.result(timeout=1)
        print(f"  {sym:25s} {status}  {dt}s  {extra}")
except FT:
    print("  TIMEOUT nach 20s!")
print(f"  Gesamt: {round(time.time()-t0,2)}s")

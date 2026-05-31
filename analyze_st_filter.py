"""
ST-Live-Filter Analyse
======================
Fragestellung: Welche Trades wurden durch den alten ST-Live-Check geblockt
(ST auf dem FOLGENDEN Candle geprüft), und wären diese gut oder schlecht gewesen?

Methodik:
  1. Signale mit BollingerStrategy(use_supertrend_filter=True) generieren
     → entspricht dem, was als raw=±1 im Live-Log erscheint
  2. Jeden Signal-Candle klassifizieren:
     PASSIERT  = ST auf nächstem Candle stimmt mit Signal überein
     GEBLOCKT  = ST auf nächstem Candle widerspricht Signal  (alter Bug)
  3. Trade-Outcome simulieren (ATR-basierter SL/TP, wie im Live-Bot)
  4. Statistik: Block-Rate, Win-Rate, Avg-R, Equity-Kurve
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.strategies.bollinger_strategy import BollingerStrategy

# ── Konfiguration ─────────────────────────────────────────────────────────────
SYMBOLS = [
    "ADA_USDT_USDT", "XRP_USDT_USDT", "DOT_USDT_USDT",
    "AVAX_USDT_USDT", "DOGE_USDT_USDT", "BNB_USDT_USDT",
    "BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT",
    "LINK_USDT_USDT",
]
TF          = "15m"
DATA_DIR    = "data/raw"

# BB-Parameter (aus Memory: beste Strategie BB(10, 2.5))
BB_PERIOD   = 10
BB_STD      = 2.5
ST_ATR      = 10
ST_MULT     = 3.0

# Trade-Simulation
ATR_SL_MULT  = 1.5    # SL = entry ± ATR * mult
ATR_TP_MULT  = 3.0    # TP = entry ± ATR * mult  (RR = 2:1)
ATR_PERIOD   = 14
MAX_HOLD     = 32     # max Candles (~8h auf 15m) bevor Force-Close


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def load_csv(symbol: str, tf: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp_dt"})
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True)
    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    close = df["close"].to_numpy()
    prev  = np.roll(close, 1); prev[0] = close[0]
    tr    = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr   = np.zeros(len(tr))
    if len(tr) >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: int,
                   atr: np.ndarray) -> dict:
    """
    Simuliert einen Trade ab entry_idx (Entry am Close des Signal-Candles).
    direction: +1=Long, -1=Short
    Gibt dict zurück mit: pnl_r (in R), outcome, hold_candles
    """
    n = len(df)
    entry_price = float(df["close"].iloc[entry_idx])
    atr_val     = atr[entry_idx]
    if atr_val <= 0:
        return {"pnl_r": 0.0, "outcome": "skip", "hold_candles": 0}

    sl = entry_price - direction * ATR_SL_MULT * atr_val
    tp = entry_price + direction * ATR_TP_MULT * atr_val

    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, n)):
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])

        if direction == 1:
            sl_hit = lo <= sl
            tp_hit = hi >= tp
        else:
            sl_hit = hi >= sl
            tp_hit = lo <= tp

        hold = i - entry_idx

        if sl_hit and tp_hit:
            # Beide im gleichen Candle → konservativ: SL zuerst
            return {"pnl_r": -1.0, "outcome": "sl", "hold_candles": hold}
        if tp_hit:
            return {"pnl_r": ATR_TP_MULT / ATR_SL_MULT, "outcome": "tp", "hold_candles": hold}
        if sl_hit:
            return {"pnl_r": -1.0, "outcome": "sl", "hold_candles": hold}

    # Force-Close nach MAX_HOLD
    exit_price = float(df["close"].iloc[min(entry_idx + MAX_HOLD, n - 1)])
    pnl_pct    = direction * (exit_price - entry_price) / entry_price
    pnl_r      = pnl_pct / (ATR_SL_MULT * atr_val / entry_price)
    return {"pnl_r": pnl_r, "outcome": "timeout", "hold_candles": MAX_HOLD}


# ── Analyse pro Symbol ────────────────────────────────────────────────────────

def analyse_symbol(symbol: str) -> pd.DataFrame:
    try:
        df = load_csv(symbol, TF)
    except FileNotFoundError:
        return pd.DataFrame()

    strategy = BollingerStrategy(
        period=BB_PERIOD, std_dev=BB_STD,
        use_supertrend_filter=True,
        st_atr_period=ST_ATR, st_multiplier=ST_MULT
    )

    atr = compute_atr(df, ATR_PERIOD)

    # ST-Richtung auf JEDEM Candle (für den "nächsten Candle"-Check)
    st_dir_all = strategy._compute_supertrend_direction(df, ST_ATR, ST_MULT)

    records = []
    # Über rollende Fenster: Signal auf df[0:i], dann Check auf Candle i
    # (= wie LiveTrader es macht: df_closed = df[:-1], dann df[-1] = live candle)
    #
    # Effizienter: Einmal generate_signals auf dem gesamten df berechnen,
    # dann für jeden Signal-Candle i prüfen ob st_dir_all[i+1] == signal.

    sigs = strategy.generate_signals(df).to_numpy().astype(int)

    for i in range(len(df) - MAX_HOLD - 2):
        sig = sigs[i]
        if sig == 0:
            continue

        # ST auf dem NÄCHSTEN Candle (= was der alte Live-Check sah)
        next_st = int(st_dir_all[i + 1])
        blocked_old = (next_st != sig)

        # Trade simulieren (Entry am Close des Signal-Candles)
        result = simulate_trade(df, i, sig, atr)
        if result["outcome"] == "skip":
            continue

        records.append({
            "symbol":       symbol,
            "candle_time":  df["timestamp_dt"].iloc[i],
            "direction":    "LONG" if sig == 1 else "SHORT",
            "old_blocked":  blocked_old,           # Geblockt vom alten Live-Check?
            "pnl_r":        result["pnl_r"],
            "outcome":      result["outcome"],
            "hold_candles": result["hold_candles"],
        })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_dfs = []
    for sym in SYMBOLS:
        df_r = analyse_symbol(sym)
        if not df_r.empty:
            all_dfs.append(df_r)
            print(f"  {sym:25s}: {len(df_r):4d} Signale")

    if not all_dfs:
        print("Keine Daten gefunden.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    total_signals = len(df)

    passed  = df[~df["old_blocked"]]
    blocked = df[ df["old_blocked"]]

    print(f"\n{'='*65}")
    print(f"  ST-Live-Filter Analyse  ({total_signals} Signale gesamt, {df['symbol'].nunique()} Symbole)")
    print(f"{'='*65}")

    print(f"\n  {'Gruppe':<22} {'Anzahl':>7}  {'Anteil':>7}  {'Win%':>7}  {'Avg-R':>8}  {'Gesamt-R':>10}")
    print(f"  {'-'*62}")

    for label, sub in [("PASSIERT (wäre getraded)", passed), ("GEBLOCKT (alter Check)", blocked)]:
        if sub.empty:
            print(f"  {label:<22}  —")
            continue
        n       = len(sub)
        pct     = 100 * n / total_signals
        wins    = (sub["pnl_r"] > 0).sum()
        win_pct = 100 * wins / n
        avg_r   = sub["pnl_r"].mean()
        tot_r   = sub["pnl_r"].sum()
        print(f"  {label:<22} {n:>7}  {pct:>6.1f}%  {win_pct:>6.1f}%  {avg_r:>+8.3f}R  {tot_r:>+10.2f}R")

    print(f"\n  {'Outcome-Aufschlüsselung':}")
    for label, sub in [("PASSIERT", passed), ("GEBLOCKT", blocked)]:
        if sub.empty:
            continue
        oc = sub["outcome"].value_counts()
        parts = "  ".join(f"{k}={v}" for k,v in oc.items())
        print(f"    {label:<12}: {parts}")

    # ── Block-Rate nach Richtung ──────────────────────────────────────────────
    print(f"\n  {'Block-Rate nach Richtung':}")
    for direction in ["LONG", "SHORT"]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        bl = sub["old_blocked"].sum()
        print(f"    {direction:<8}: {bl}/{len(sub)} geblockt ({100*bl/len(sub):.1f}%)")

    # ── Equity-Vergleich: Passiert vs. Alles (ohne Filter) ───────────────────
    print(f"\n  {'Equity-Vergleich (kumuliert, in R-Vielfachen)':}")
    cum_passed  = passed["pnl_r"].sum()
    cum_blocked = blocked["pnl_r"].sum()
    cum_all     = df["pnl_r"].sum()
    print(f"    Ohne Filter (alle nehmen):   {cum_all:+.2f}R")
    print(f"    Nur PASSIERTe:               {cum_passed:+.2f}R")
    print(f"    Nur GEBLOCKTe:               {cum_blocked:+.2f}R")
    delta = cum_passed - cum_all
    print(f"    Filter-Effekt (Δ):           {delta:+.2f}R  "
          f"({'✅ Filter hilft' if delta > 0 else '❌ Filter schadet'})")

    # ── Pro-Symbol Aufschlüsselung ────────────────────────────────────────────
    print(f"\n  {'Pro-Symbol (nur GEBLOCKTe)':}")
    print(f"  {'Symbol':<25} {'Geblockt':>9}  {'Win%':>7}  {'Avg-R':>8}  {'Gesamt-R':>10}")
    print(f"  {'-'*62}")
    for sym, grp in blocked.groupby("symbol"):
        n    = len(grp)
        wp   = 100 * (grp["pnl_r"] > 0).sum() / n
        ar   = grp["pnl_r"].mean()
        tr   = grp["pnl_r"].sum()
        flag = "✅" if ar > 0 else "❌"
        print(f"  {sym:<25} {n:>9}  {wp:>6.1f}%  {ar:>+8.3f}R  {tr:>+10.2f}R  {flag}")

    # ── Block-Rate über Zeit ──────────────────────────────────────────────────
    if not blocked.empty:
        df["month"] = df["candle_time"].dt.to_period("M")
        monthly = df.groupby("month").apply(
            lambda x: pd.Series({
                "total": len(x),
                "blocked": x["old_blocked"].sum(),
                "block_pct": 100 * x["old_blocked"].mean(),
                "avg_r_passiert": x[~x["old_blocked"]]["pnl_r"].mean() if (~x["old_blocked"]).any() else np.nan,
                "avg_r_geblockt": x[ x["old_blocked"]]["pnl_r"].mean() if (  x["old_blocked"]).any() else np.nan,
            })
        ).reset_index()
        print(f"\n  {'Monatliche Block-Rate & Performance':}")
        print(f"  {'Monat':<10} {'Total':>6}  {'Geblockt':>9}  {'Block%':>7}  {'AvgR-Pass':>10}  {'AvgR-Block':>11}")
        print(f"  {'-'*60}")
        for _, row in monthly.iterrows():
            rp = f"{row['avg_r_passiert']:+.3f}R" if not pd.isna(row['avg_r_passiert']) else "   —   "
            rb = f"{row['avg_r_geblockt']:+.3f}R"  if not pd.isna(row['avg_r_geblockt'])  else "   —   "
            print(f"  {str(row['month']):<10} {int(row['total']):>6}  {int(row['blocked']):>9}  "
                  f"{row['block_pct']:>6.1f}%  {rp:>10}  {rb:>11}")

    print(f"\n{'='*65}")
    print(f"  Fazit:")
    if cum_blocked < -0.5 * abs(cum_all) or (len(blocked) > 0 and blocked['pnl_r'].mean() < -0.1):
        print(f"  ❌ Geblockte Trades wären SCHLECHT gewesen → Filter war hilfreich!")
    elif len(blocked) > 0 and blocked['pnl_r'].mean() > 0.05:
        print(f"  ✅ Geblockte Trades wären GUT gewesen → Filter hat Gewinne verhindert!")
    else:
        print(f"  ➡  Kein eindeutiges Signal — geblockten Trades sind neutral.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

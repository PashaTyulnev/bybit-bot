#!/usr/bin/env python3
"""
Walk-Forward Anti-Overfitting Analyse – BTC Scalping (Round 3)
==============================================================
Neu: ADX-Filter (Bollinger=Trending, RSI-Div=Ranging), erweiterte TP/SL,
     Long/Short-Auswertung pro Fold, Top-5-Tabelle, Live-Ready-Bewertung.

Ausführen:
    cd /home/pashacryptotrader/app
    python3 -m src.research_btc_scalping
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
import numpy as np
import pandas as pd

from src.strategy_backtester import (
    StrategyConfig, compute_adx, compute_market_condition_arr,
    run_strategy_backtest_fast,
)
from src.strategy_optimizer import (
    StrategyOptConfig, apply_adx_filter, apply_mtf_filter,
    run_strategy_optimization,
)

RAW   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
N_DAY = 288   # 5m-Kerzen/Tag


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATEN
# ══════════════════════════════════════════════════════════════════════════════

def _load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(f"{RAW}/BTC_USDT_USDT_{tf}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def _clip(df: pd.DataFrame, t0, t1) -> pd.DataFrame:
    return df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].reset_index(drop=True)

print("=" * 68)
print("BTC SCALPING – WALK-FORWARD ROUND 3 (ADX-Filter + erw. TP/SL)")
print("=" * 68)
print("\nLade Daten …")
df5  = _load("5m");  df15 = _load("15m");  df1h = _load("1h")
start = max(df5["datetime"].iloc[0],  df15["datetime"].iloc[0], df1h["datetime"].iloc[0])
end   = min(df5["datetime"].iloc[-1], df15["datetime"].iloc[-1], df1h["datetime"].iloc[-1])
df5   = _clip(df5,  start, end);  df15 = _clip(df15, start, end);  df1h = _clip(df1h, start, end)
n5    = len(df5)
print(f"5m : {n5} Kerzen ({n5//N_DAY}d) | {start.date()} → {end.date()}")
print(f"15m: {len(df15)}  |  1h: {len(df1h)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FOLDS (gleich wie bisher: 4 × 60d OOS, Expanding Window)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DAYS, TEST_DAYS, STEP = 120, 60, 60
folds: list[tuple[int,int,int,int]] = []
d = TRAIN_DAYS
while d + TEST_DAYS <= n5 // N_DAY:
    te = d * N_DAY
    folds.append((0, te, te, min(te + TEST_DAYS * N_DAY, n5)))
    d += STEP
    if len(folds) >= 4: break

print(f"\n{len(folds)} Walk-Forward Folds:")
for i, (ts, te, qs, qe) in enumerate(folds):
    print(f"  Fold {i+1}: Train {(te-ts)//N_DAY}d "
          f"[{df5['datetime'].iloc[ts].date()}→{df5['datetime'].iloc[te-1].date()}] | "
          f"Test {(qe-qs)//N_DAY}d "
          f"[{df5['datetime'].iloc[qs].date()}→{df5['datetime'].iloc[qe-1].date()}]")


# ══════════════════════════════════════════════════════════════════════════════
# 3. STRATEGIE-KONFIGURATIONEN
#    Bollinger  → adx_mode="trending",  ADX-Schwellen [20, 25, 30]
#    RSI-Div    → adx_mode="ranging",   ADX-Schwellen [20, 25, 30]
#    TP/SL: erweitert (Ziel PF > 1.5)
# ══════════════════════════════════════════════════════════════════════════════

ADX_THS = [20.0, 25.0, 30.0]

_BASE = dict(
    leverages=[10, 20],
    position_sizes=[0.10],
    tp_pcts=[0.75, 1.0, 1.5],        # erweitert
    sl_pcts=[0.4, 0.5],              # TP 1.0/SL 0.4 und TP 1.5/SL 0.5 enthalten
    min_trades=20,
    score_metric="composite",
    mtf_ema_period=50,
)

STRATEGIES = [
    ("bollinger", StrategyOptConfig(
        bb_periods=[10, 15, 20],
        bb_std_devs=[2.0, 2.5],
        adx_thresholds=ADX_THS,
        adx_mode="trending",
        **_BASE,
    )),
    ("rsi_divergence", StrategyOptConfig(
        rsi_div_periods=[7, 14],
        rsi_div_lookbacks=[8, 14],
        rsi_div_oversolds=[30.0, 35.0],
        rsi_div_overboughts=[65.0, 70.0],
        adx_thresholds=ADX_THS,
        adx_mode="ranging",
        **_BASE,
    )),
]

print("\nKombinationen pro Fold:")
from src.strategy_optimizer import _STRATEGY_BUILDERS
for skey, scfg in STRATEGIES:
    n_s   = sum(1 for _ in _STRATEGY_BUILDERS[skey](scfg))
    n_adx = len(scfg.adx_thresholds)
    n_t   = len(scfg.leverages)*len(scfg.position_sizes)*len(scfg.tp_pcts)*len(scfg.sl_pcts)
    print(f"  {skey:16s}: {n_s} Var × {n_adx} ADX-Th × {n_t} Trade = {n_s*n_adx*n_t}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HILFS-FUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def _config_key(strat_str: str, adx_th, cfg: StrategyConfig) -> str:
    tp = (cfg.take_profit_pct or 0) * 100
    sl = (cfg.stop_loss_pct   or 0) * 100
    return (f"{strat_str}|ADX{adx_th or '–'}|"
            f"{cfg.leverage}x|{cfg.position_size*100:.0f}p|"
            f"TP{tp:.2f}|SL{sl:.2f}")

def _oos_test(
    qdf: pd.DataFrame,
    te15: pd.DataFrame, te1h: pd.DataFrame,
    strategy, trade_cfg: StrategyConfig,
    adx_th, adx_mode: str, mtf_period: int,
    adx_raw_oos: np.ndarray,
    cond_oos: np.ndarray,
) -> dict | None:
    try:
        sigs = strategy.generate_signals(qdf).to_numpy(int)
    except Exception:
        return None
    if len(te15) > 0 and len(te1h) > 0:
        sigs = apply_mtf_filter(qdf, [te15, te1h], sigs, mtf_period)
    if adx_th is not None and adx_mode != "none":
        sigs = apply_adx_filter(sigs, adx_raw_oos, adx_th, adx_mode)
    r = run_strategy_backtest_fast(
        qdf["open"].to_numpy(float), qdf["high"].to_numpy(float),
        qdf["low"].to_numpy(float),  qdf["close"].to_numpy(float),
        sigs, trade_cfg, condition_arr=cond_oos,
    )
    return r if r and r.get("num_trades", 0) >= 5 else None

def _sc(r: dict) -> float:
    """Composite OOS-Score."""
    pf  = min(r.get("profit_factor", 0), 10.0)
    pnl = r.get("total_pnl_pct", 0)
    dd  = r.get("max_drawdown_pct", 100)
    return pnl * pf * max(0.1, 1 - dd / 100)

def _fmt_ls(r: dict) -> str:
    """Kurzformat Long/Short."""
    ln, lwr, lpf = r.get("long_trades",0), r.get("long_winrate",0), r.get("long_pf",0)
    sn, swr, spf = r.get("short_trades",0), r.get("short_winrate",0), r.get("short_pf",0)
    return (f"L:{ln}T WR{lwr:.0f}% PF{lpf:.2f} | "
            f"S:{sn}T WR{swr:.0f}% PF{spf:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD HAUPTSCHLEIFE
# ══════════════════════════════════════════════════════════════════════════════

all_oos: list[dict] = []

for fi, (ts, te, qs, qe) in enumerate(folds):
    fn   = fi + 1
    tdf  = df5.iloc[ts:te].reset_index(drop=True)
    qdf  = df5.iloc[qs:qe].reset_index(drop=True)
    t_d  = (qe - qs) // N_DAY
    t0, t1 = tdf["datetime"].iloc[0],  tdf["datetime"].iloc[-1]
    q0, q1 = qdf["datetime"].iloc[0],  qdf["datetime"].iloc[-1]
    tr15 = _clip(df15, t0, t1);  tr1h = _clip(df1h, t0, t1)
    te15 = _clip(df15, q0, q1);  te1h = _clip(df1h, q0, q1)

    # ADX + Market-Condition für OOS-Daten vorberechnen
    adx_raw_oos = compute_adx(qdf)
    cond_oos    = compute_market_condition_arr(qdf)

    print(f"\n{'─'*68}")
    print(f"FOLD {fn}  |  OOS: {q0.date()} → {q1.date()}  ({t_d}d)")
    print(f"{'─'*68}")

    for skey, scfg in STRATEGIES:
        adx_label = "Trending" if scfg.adx_mode == "trending" else "Ranging"
        print(f"\n  [{skey.upper()} – ADX {adx_label}]")

        # Optimierung auf Trainingsdaten
        train_res = run_strategy_optimization(
            tdf, skey, scfg,
            val_dfs=[tr15, tr1h] if len(tr15) > 0 else None,
        )
        if not train_res:
            print("    – keine Trainingsergebnisse")
            continue

        # Top-10 Trainingskandidaten auf OOS testen
        tested: list[dict] = []
        for r_tr in train_res[:10]:
            oos = _oos_test(
                qdf, te15, te1h,
                r_tr["strategy"], r_tr["config"],
                r_tr.get("adx_threshold"), scfg.adx_mode, scfg.mtf_ema_period,
                adx_raw_oos, cond_oos,
            )
            if oos:
                oos.update(
                    strategy_str = str(r_tr["strategy"]),
                    strategy_key = skey,
                    adx_mode     = scfg.adx_mode,
                    adx_threshold= r_tr.get("adx_threshold"),
                    config       = r_tr["config"],
                    fold         = fn,
                    test_days    = t_d,
                    train_pnl    = r_tr.get("total_pnl_pct", 0),
                    cfg_key      = _config_key(
                        str(r_tr["strategy"]),
                        r_tr.get("adx_threshold"),
                        r_tr["config"],
                    ),
                )
                tested.append(oos)
                all_oos.append(oos)

        if not tested:
            print("    – keine OOS-Ergebnisse")
            continue

        # Bestes + Alle anzeigen (nach OOS-Score sortiert)
        tested_sorted = sorted(tested, key=_sc, reverse=True)
        for rank, r in enumerate(tested_sorted[:5], 1):
            cfg   = r["config"]
            tp    = (cfg.take_profit_pct or 0) * 100
            sl_   = (cfg.stop_loss_pct   or 0) * 100
            tpd   = r["num_trades"] / t_d
            adx_s = f"ADX{'≥' if scfg.adx_mode=='trending' else '<'}{r['adx_threshold']:.0f}"
            pf_ok = "✓PF" if r["profit_factor"] >= 1.5 else "  "
            dd_ok = "✓DD" if r["max_drawdown_pct"] <= 15 else "  "
            print(
                f"    #{rank} {adx_s} {r['strategy_str']:<22}"
                f"  {cfg.leverage}x TP{tp:.2f}/SL{sl_:.2f}"
                f"  PnL{r['total_pnl_pct']:>+6.2f}%"
                f"  PF{r['profit_factor']:.3f}{pf_ok}"
                f"  WR{r['winrate_pct']:.0f}%"
                f"  DD{r['max_drawdown_pct']:.1f}%{dd_ok}"
                f"  {r['num_trades']}T({tpd:.1f}/d)"
            )
            print(f"       {_fmt_ls(r)}"
                  f"  |  Trend-WR {r.get('trending_winrate',0):.0f}%"
                  f"  Range-WR {r.get('ranging_winrate',0):.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 6. AGGREGATION – TOP 5 ROBUSTE KONFIGURATIONEN
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print("TOP 5 ROBUSTE KONFIGURATIONEN (konsistent über alle Folds)")
print(f"{'='*68}")

# Gruppieren nach Config-Key
by_key: dict[str, list[dict]] = defaultdict(list)
for r in all_oos:
    by_key[r["cfg_key"]].append(r)

# Aggregieren
agg_rows = []
for key, rs in by_key.items():
    pnls  = [x["total_pnl_pct"]    for x in rs]
    pfs   = [x["profit_factor"]     for x in rs]
    wrs   = [x["winrate_pct"]       for x in rs]
    dds   = [x["max_drawdown_pct"]  for x in rs]
    tpds  = [x["num_trades"] / x["test_days"] for x in rs]
    n     = len(rs)
    prof  = sum(1 for p in pnls if p > 0) / n
    pf15  = sum(1 for p in pfs if p >= 1.5) / n
    dd15  = sum(1 for d in dds if d <= 15)  / n
    # Kombinierter Qualitätsscore: Konsistenz × Ø PnL × Ø PF
    q_score = prof * np.mean(pnls) * min(np.mean(pfs), 10.0) * max(0.1, 1 - np.mean(dds)/100)
    agg_rows.append(dict(
        key=key, n_folds=n,
        avg_pnl=np.mean(pnls),  avg_pf=np.mean(pfs),
        avg_wr=np.mean(wrs),    avg_dd=np.mean(dds),
        avg_tpd=np.mean(tpds),
        prof_pct=prof*100, pf15_pct=pf15*100, dd15_pct=dd15*100,
        q_score=q_score,
        sample=rs[0],  # für Details
    ))

agg_rows.sort(key=lambda x: x["q_score"], reverse=True)
top5 = agg_rows[:5]

# Tabellenausgabe
HDR = (f"{'#':>2}  {'Strategie':<24}{'ADX':>7}{'Cfg':>18}"
       f"{'Folds':>6}{'ØPnL%':>8}{'ØPF':>7}{'ØWR%':>7}"
       f"{'ØDD%':>7}{'ØT/d':>7}{'PF>1.5':>8}{'DD≤15':>7}")
print(f"\n{HDR}")
print("─" * len(HDR))

for rank, row in enumerate(top5, 1):
    r   = row["sample"]
    cfg = r["config"]
    tp  = (cfg.take_profit_pct or 0) * 100
    sl_ = (cfg.stop_loss_pct   or 0) * 100
    adx_sym = "≥" if r["adx_mode"] == "trending" else "<"
    adx_str = f"ADX{adx_sym}{r['adx_threshold']:.0f}" if r["adx_threshold"] else "–"
    cfg_str = f"{cfg.leverage}x TP{tp:.2f}/SL{sl_:.2f}"
    pf_ok   = "✓" if row["pf15_pct"] >= 75 else ("~" if row["pf15_pct"] >= 50 else "✗")
    dd_ok   = "✓" if row["dd15_pct"] >= 75 else ("~" if row["dd15_pct"] >= 50 else "✗")
    print(
        f"{rank:>2}  {r['strategy_str']:<24}{adx_str:>7}{cfg_str:>18}"
        f"{row['n_folds']:>6}{row['avg_pnl']:>+8.2f}{row['avg_pf']:>7.3f}"
        f"{row['avg_wr']:>7.1f}{row['avg_dd']:>7.1f}{row['avg_tpd']:>7.1f}"
        f"  {pf_ok}({row['pf15_pct']:.0f}%)  {dd_ok}({row['dd15_pct']:.0f}%)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. LONG vs SHORT – BESTE KONFIGURATION (vollständig)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print("LONG vs SHORT – BESTE KONFIGURATION (#1)")
print(f"{'='*68}")

if top5:
    best_row = top5[0]
    best_rs  = by_key[best_row["key"]]

    # Fold-by-Fold Long/Short
    print(f"\nStrategie: {best_rs[0]['strategy_str']}")
    cfg0 = best_rs[0]["config"]
    tp0  = (cfg0.take_profit_pct or 0) * 100
    sl0  = (cfg0.stop_loss_pct   or 0) * 100
    adx0 = best_rs[0].get("adx_threshold")
    adxm = best_rs[0].get("adx_mode", "–")
    print(f"Config    : {cfg0.leverage}x | Pos {cfg0.position_size*100:.0f}% | "
          f"TP {tp0:.2f}% | SL {sl0:.2f}%")
    print(f"ADX-Filter: {adxm} ADX{'≥' if adxm=='trending' else '<'}{adx0:.0f}"
          if adx0 else "ADX-Filter: keiner")

    print(f"\n{'Fold':>5}{'Zeitraum':>24}{'PnL%':>7}{'PF':>6}  "
          f"{'Long':>22}  {'Short':>22}")
    print("─" * 90)

    total_lt = total_lw = total_st = total_sw = 0
    long_pfs = []; short_pfs = []
    for r in sorted(best_rs, key=lambda x: x["fold"]):
        fn_  = r["fold"]
        t_d  = r["test_days"]
        lt   = r.get("long_trades",  0);  lwr = r.get("long_winrate",  0)
        lpf  = r.get("long_pf",      0);  lp  = r.get("long_total_pnl", 0)
        st   = r.get("short_trades", 0);  swr = r.get("short_winrate", 0)
        spf  = r.get("short_pf",     0);  sp  = r.get("short_total_pnl", 0)
        total_lt += lt; total_lw += int(lt * lwr / 100)
        total_st += st; total_sw += int(st * swr / 100)
        long_pfs.append(lpf);  short_pfs.append(spf)
        ts_start = df5["datetime"].iloc[folds[fn_-1][2]].date()
        ts_end   = df5["datetime"].iloc[folds[fn_-1][3]-1].date()
        print(f"{fn_:>5}  {str(ts_start):>11}→{str(ts_end):<11}"
              f"  {r['total_pnl_pct']:>+6.2f}% {r['profit_factor']:>5.3f}  "
              f"  {lt:>3}T WR{lwr:.0f}% PF{lpf:.2f} PnL{lp:>+7.2f}"
              f"  {st:>3}T WR{swr:.0f}% PF{spf:.2f} PnL{sp:>+7.2f}")

    # Aggregat Long/Short
    print("─" * 90)
    l_agg_wr = total_lw / total_lt * 100 if total_lt > 0 else 0
    s_agg_wr = total_sw / total_st * 100 if total_st > 0 else 0
    print(f"{'GESAMT':>5}  {'':>24}"
          f"  {'':>13}"
          f"  {total_lt:>3}T WR{l_agg_wr:.0f}% ØPF{np.mean(long_pfs):.2f}"
          f"  {total_st:>3}T WR{s_agg_wr:.0f}% ØPF{np.mean(short_pfs):.2f}")

    bias = "Long-Bias" if l_agg_wr > s_agg_wr + 5 else (
           "Short-Bias" if s_agg_wr > l_agg_wr + 5 else "Ausgewogen")
    print(f"\n  Richtungsanalyse: {bias} "
          f"(Long WR {l_agg_wr:.0f}% vs Short WR {s_agg_wr:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. LIVE-READY BEWERTUNG
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print("LIVE-READY BEWERTUNG")
print(f"{'='*68}")

# Mindestbedingungen: PF > 1.5, ≥ 30 Trades, DD ≤ 15%
MIN_PF = 1.5;  MIN_T = 30;  MAX_DD = 15.0

print(f"\nMindestbedingungen: PF > {MIN_PF} | Trades ≥ {MIN_T} | DD ≤ {MAX_DD}%\n")

for rank, row in enumerate(top5, 1):
    rs = by_key[row["key"]]
    n  = len(rs)
    ok = [r for r in rs
          if r["profit_factor"]    > MIN_PF
          and r["num_trades"]      >= MIN_T
          and r["max_drawdown_pct"] <= MAX_DD]
    pct = len(ok) / n * 100

    r0  = rs[0]
    cfg = r0["config"]
    tp  = (cfg.take_profit_pct or 0) * 100
    sl_ = (cfg.stop_loss_pct   or 0) * 100
    adx_s = (f"ADX{'≥' if r0['adx_mode']=='trending' else '<'}"
             f"{r0['adx_threshold']:.0f}" if r0.get("adx_threshold") else "kein ADX")

    if pct >= 75:
        status = "✅  LIVE READY"
    elif pct >= 50:
        status = "⚡  BEDINGT READY (PF-Grenze prüfen)"
    else:
        status = "❌  NICHT LIVE READY"

    print(f"#{rank}  {r0['strategy_str']} | {adx_s} | "
          f"{cfg.leverage}x TP{tp:.2f}/SL{sl_:.2f}")
    print(f"    Constraints erfüllt: {len(ok)}/{n} Folds ({pct:.0f}%)  → {status}")
    print(f"    Ø PnL {row['avg_pnl']:+.2f}%/60d | Ø PF {row['avg_pf']:.3f} | "
          f"Ø {row['avg_tpd']:.1f}T/Tag\n")

# Gesamt-Empfehlung
live_ready = [row for row in top5
              if sum(1 for r in by_key[row["key"]]
                     if r["profit_factor"] > MIN_PF
                     and r["num_trades"] >= MIN_T
                     and r["max_drawdown_pct"] <= MAX_DD) / len(by_key[row["key"]]) >= 0.75]

print(f"{'='*68}")
if live_ready:
    best = live_ready[0]
    rs   = by_key[best["key"]]
    r0   = rs[0]; cfg = r0["config"]
    tp   = (cfg.take_profit_pct or 0) * 100
    sl_  = (cfg.stop_loss_pct   or 0) * 100
    pf_f = sum(1 for r in rs if r["profit_factor"] > MIN_PF)
    print(f"\n✅  EMPFEHLUNG FÜR LIVE-EINSATZ")
    print(f"   Strategie : {r0['strategy_str']}")
    adx_s = (f"ADX {'≥' if r0['adx_mode']=='trending' else '<'} "
             f"{r0['adx_threshold']:.0f}" if r0.get("adx_threshold") else "kein ADX")
    print(f"   ADX-Filter: {adx_s} (Signal-TF 5m + MTF 15m+1h)")
    print(f"   Config    : {cfg.leverage}x | Pos {cfg.position_size*100:.0f}% | "
          f"TP {tp:.2f}% | SL {sl_:.2f}%")
    print(f"   PF > 1.5  : {pf_f}/{len(rs)} Folds ({'✓' if pf_f >= 3 else '~'})")
    print(f"   Ø PnL     : {best['avg_pnl']:+.2f}%/60-Tage-Periode")
    print(f"   Ø Trades  : {best['avg_tpd']:.1f}/Tag")
    print(f"   Live-Setup : 5m Chart + Bollinger({r0['strategy_str'].split('_')[1]},...")
else:
    best_part = min(top5, key=lambda x: -sum(
        1 for r in by_key[x["key"]]
        if r["profit_factor"] > MIN_PF
        and r["num_trades"] >= MIN_T
        and r["max_drawdown_pct"] <= MAX_DD
    ) if by_key[x["key"]] else 0)
    print(f"\n⚠️   KEINE KONFIGURATION ERFÜLLT VOLLSTÄNDIG ALLE BEDINGUNGEN")
    print(f"   Bester Kandidat: #{1} oben – PF-Schwelle weiter anpassen")
    print(f"   Empfehlung: TP-SL-Verhältnis > 2:1 testen (z.B. TP 1.5% / SL 0.5%)")

print(f"\n{'='*68}")
print("Analyse abgeschlossen.")
print(f"{'='*68}\n")

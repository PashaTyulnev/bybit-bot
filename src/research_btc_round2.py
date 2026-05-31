#!/usr/bin/env python3
"""
Round 2 – BTC Scalping: Bollinger (mit/ohne MTF) + RSI Divergence
Fokus: Signalfrequenz erhöhen, MTF-Einfluss messen
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import numpy as np
import pandas as pd
from src.strategy_backtester import compute_market_condition_arr, run_strategy_backtest_fast
from src.strategy_optimizer import StrategyOptConfig, apply_mtf_filter, run_strategy_optimization

RAW   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
N_DAY = 288

def _load(tf):
    df = pd.read_csv(f"{RAW}/BTC_USDT_USDT_{tf}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def _slice(df, t0, t1):
    return df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].reset_index(drop=True)

print("=" * 65)
print("BTC SCALPING ROUND 2 – Bollinger Feinoptimierung + MTF-Vergleich")
print("=" * 65)

df5  = _load("5m"); df15 = _load("15m"); df1h = _load("1h")
start = max(df5["datetime"].iloc[0], df15["datetime"].iloc[0], df1h["datetime"].iloc[0])
end   = min(df5["datetime"].iloc[-1], df15["datetime"].iloc[-1], df1h["datetime"].iloc[-1])
for dfx in (df5, df15, df1h):
    mask = (dfx["datetime"] >= start) & (dfx["datetime"] <= end)
df5  = df5[(df5["datetime"]   >= start) & (df5["datetime"]   <= end)].reset_index(drop=True)
df15 = df15[(df15["datetime"] >= start) & (df15["datetime"]  <= end)].reset_index(drop=True)
df1h = df1h[(df1h["datetime"] >= start) & (df1h["datetime"]  <= end)].reset_index(drop=True)
n5   = len(df5)

TRAIN_DAYS, TEST_DAYS, STEP = 120, 60, 60
folds = []
d = TRAIN_DAYS
while d + TEST_DAYS <= n5 // N_DAY:
    te = d * N_DAY
    folds.append((0, te, te, min(te + TEST_DAYS * N_DAY, n5)))
    d += STEP
    if len(folds) >= 4:
        break

print(f"\n{len(folds)} Folds | {start.date()} → {end.date()}\n")

# ── Kleine, effiziente Suchräume ────────────────────────────────────────────
_TRADE = dict(
    leverages=[5, 10, 20],
    position_sizes=[0.05, 0.10],
    tp_pcts=[0.3, 0.5, 0.75, 1.0],
    sl_pcts=[0.2, 0.3, 0.5],
    min_trades=15,
    score_metric="composite",
    mtf_ema_period=50,
)

STRATS = [
    # Bollinger mit MTF (15m+1h)
    ("bb_mtf",    "bollinger", True,  StrategyOptConfig(
        bb_periods=[5, 7, 10, 15, 20], bb_std_devs=[1.5, 2.0, 2.5], **_TRADE)),
    # Bollinger ohne MTF (Baseline)
    ("bb_nomtf",  "bollinger", False, StrategyOptConfig(
        bb_periods=[5, 7, 10, 15, 20], bb_std_devs=[1.5, 2.0, 2.5], **_TRADE)),
    # RSI Divergence (kleiner Suchraum für Geschwindigkeit)
    ("rsidiv",    "rsi_divergence", True, StrategyOptConfig(
        rsi_div_periods=[7, 14],
        rsi_div_lookbacks=[8, 14],
        rsi_div_oversolds=[30.0, 35.0],
        rsi_div_overboughts=[65.0, 70.0],
        **_TRADE)),
    # EMA scalping
    ("ema_scalp", "ema", True, StrategyOptConfig(
        ema_fast_periods=[3, 5, 8],
        ema_slow_periods=[13, 21, 34],
        **_TRADE)),
]

print("Kombinationen pro Fold:")
from src.strategy_optimizer import _STRATEGY_BUILDERS
for label, skey, _, scfg in STRATS:
    n_s = sum(1 for _ in _STRATEGY_BUILDERS[skey](scfg))
    n_t = len(scfg.leverages) * len(scfg.position_sizes) * len(scfg.tp_pcts) * len(scfg.sl_pcts)
    print(f"  {label:<12}: {n_s} × {n_t} = {n_s*n_t}")

def _oos(qdf, te15, te1h, strat, cfg, mtf_per, use_mtf):
    try: sigs = strat.generate_signals(qdf).to_numpy(int)
    except: return None
    if use_mtf and len(te15) > 0 and len(te1h) > 0:
        sigs = apply_mtf_filter(qdf, [te15, te1h], sigs, mtf_per)
    cond = compute_market_condition_arr(qdf)
    r = run_strategy_backtest_fast(
        qdf["open"].to_numpy(float), qdf["high"].to_numpy(float),
        qdf["low"].to_numpy(float),  qdf["close"].to_numpy(float),
        sigs, cfg, condition_arr=cond,
    )
    return r if r and r.get("num_trades", 0) >= 5 else None

all_oos = []

for fi, (ts, te, qs, qe) in enumerate(folds):
    fn = fi + 1
    tdf = df5.iloc[ts:te].reset_index(drop=True)
    qdf = df5.iloc[qs:qe].reset_index(drop=True)
    t_d = (qe - qs) // N_DAY
    t0, t1 = tdf["datetime"].iloc[0], tdf["datetime"].iloc[-1]
    q0, q1 = qdf["datetime"].iloc[0], qdf["datetime"].iloc[-1]
    tr15 = _slice(df15, t0, t1); tr1h = _slice(df1h, t0, t1)
    te15 = _slice(df15, q0, q1); te1h = _slice(df1h, q0, q1)

    print(f"\n{'─'*65}")
    print(f"FOLD {fn} | OOS: {q0.date()} → {q1.date()}")
    print(f"{'─'*65}")

    for label, skey, use_mtf, scfg in STRATS:
        vdfs = [tr15, tr1h] if (use_mtf and len(tr15) > 0) else None
        tr = run_strategy_optimization(tdf, skey, scfg, val_dfs=vdfs)
        if not tr:
            print(f"  {label:<12}: –");  continue

        best = None
        for r_tr in tr[:5]:
            r = _oos(qdf, te15, te1h, r_tr["strategy"], r_tr["config"],
                     scfg.mtf_ema_period, use_mtf)
            if r:
                r.update(label=label, fold=fn, config=r_tr["config"],
                         strat_str=str(r_tr["strategy"]),
                         test_days=t_d, train_pnl=r_tr.get("total_pnl_pct",0),
                         use_mtf=use_mtf)
                if best is None or (
                    r["total_pnl_pct"] * min(r["profit_factor"],10) * max(0.1, 1-r["max_drawdown_pct"]/100)
                    > best["total_pnl_pct"] * min(best["profit_factor"],10) * max(0.1, 1-best["max_drawdown_pct"]/100)
                ):
                    best = r

        if best:
            all_oos.append(best)
            tpd = best["num_trades"] / t_d
            m   = "MTF✓" if use_mtf else "noMTF"
            print(f"  {label:<12} [{m}]  "
                  f"PnL {best['total_pnl_pct']:>+6.2f}%  "
                  f"PF {best['profit_factor']:.3f}  "
                  f"WR {best['winrate_pct']:.1f}%  "
                  f"DD {best['max_drawdown_pct']:.1f}%  "
                  f"{best['num_trades']}T ({tpd:.1f}/d)  {best['strat_str']}")
        else:
            print(f"  {label:<12}: keine OOS-Ergebnisse")

# ── Aggregation ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("AGGREGIERTE OOS-ERGEBNISSE")
print(f"{'='*65}")

by = defaultdict(list)
for r in all_oos: by[r["label"]].append(r)

summ = []
for label, rs in by.items():
    pnls = [x["total_pnl_pct"] for x in rs]
    n    = len(rs)
    prof = sum(1 for p in pnls if p > 0) / n * 100
    scok = sum(1 for x in rs
               if x["profit_factor"] >= 1.5
               and x["num_trades"] >= 50
               and x["max_drawdown_pct"] <= 20) / n * 100
    summ.append(dict(
        label=label, mtf=rs[0]["use_mtf"],
        avg_pnl=np.mean(pnls),  avg_pf=np.mean([x["profit_factor"] for x in rs]),
        avg_wr=np.mean([x["winrate_pct"] for x in rs]),
        avg_dd=np.mean([x["max_drawdown_pct"] for x in rs]),
        avg_tpd=np.mean([x["num_trades"]/x["test_days"] for x in rs]),
        twr=np.mean([x.get("trending_winrate",0) for x in rs]),
        rwr=np.mean([x.get("ranging_winrate", 0) for x in rs]),
        prof=prof, scok=scok, n=n,
    ))

summ.sort(key=lambda x: (x["prof"], x["avg_pnl"]), reverse=True)
print(f"\n{'Label':<14}{'MTF':>5}{'ØPnL%':>8}{'ØPF':>7}{'ØWR':>7}{'ØDD':>7}"
      f"{'ØT/d':>7}{'TrWR':>7}{'RaWR':>7}{'Prof':>8}{'SC-OK':>7}")
print("-" * 85)
for s in summ:
    print(f"{s['label']:<14}{'✓' if s['mtf'] else '✗':>5}"
          f"{s['avg_pnl']:>+8.2f}{s['avg_pf']:>7.3f}"
          f"{s['avg_wr']:>7.1f}{s['avg_dd']:>7.1f}"
          f"{s['avg_tpd']:>7.1f}{s['twr']:>7.1f}{s['rwr']:>7.1f}"
          f"{s['prof']:>7.0f}%{s['scok']:>6.0f}%")

# ── Details je Label ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("DETAIL BESTE KONFIGURATION PRO STRATEGIE")
print(f"{'='*65}")

for s in summ:
    rs   = by[s["label"]]
    best = max(rs, key=lambda x: x["total_pnl_pct"])
    cfg  = best["config"]
    tp   = (cfg.take_profit_pct or 0) * 100
    sl   = (cfg.stop_loss_pct   or 0) * 100
    tpd  = best["num_trades"] / best["test_days"]
    lwr  = best.get("long_winrate",    0)
    swr_ = best.get("short_winrate",   0)
    twr  = best.get("trending_winrate",0)
    rwr  = best.get("ranging_winrate", 0)
    ok   = " ".join(f for f, c in [("PF>1.5", best["profit_factor"]>=1.5),
                                    ("≥50T",   best["num_trades"]>=50),
                                    ("DD≤20",  best["max_drawdown_pct"]<=20)] if c)
    print(f"\n▶ {s['label'].upper()} | {'MTF aktiv' if s['mtf'] else 'kein MTF'}")
    print(f"  {best['strat_str']}  |  {cfg.leverage}x  Pos {cfg.position_size*100:.0f}%  TP {tp:.2f}%  SL {sl:.2f}%")
    print(f"  PnL {best['total_pnl_pct']:+.2f}%  PF {best['profit_factor']:.3f}  "
          f"WR {best['winrate_pct']:.1f}%  DD {best['max_drawdown_pct']:.1f}%  "
          f"{best['num_trades']}T ({tpd:.1f}/d)")
    print(f"  Long WR {lwr:.1f}%  Short WR {swr_:.1f}%  "
          f"Trending WR {twr:.1f}%  Ranging WR {rwr:.1f}%")
    print(f"  Scalping-Checks: {ok if ok else '–'}")

    # Fold-by-Fold Performance
    print(f"  Fold-Performance:")
    for r in sorted(rs, key=lambda x: x["fold"]):
        tpd_f = r["num_trades"] / r["test_days"]
        print(f"    Fold {r['fold']}: PnL {r['total_pnl_pct']:>+6.2f}%  "
              f"PF {r['profit_factor']:.3f}  {r['num_trades']}T ({tpd_f:.1f}/d)")

# ── MTF-Einfluss ─────────────────────────────────────────────────────────────
if "bb_mtf" in by and "bb_nomtf" in by:
    print(f"\n{'='*65}")
    print("MTF-EINFLUSS: Bollinger MIT vs OHNE MTF-Filter")
    print(f"{'='*65}")
    def _agg(label):
        rs = by[label]
        return {
            "pnl": np.mean([x["total_pnl_pct"] for x in rs]),
            "pf":  np.mean([x["profit_factor"]  for x in rs]),
            "wr":  np.mean([x["winrate_pct"]     for x in rs]),
            "tpd": np.mean([x["num_trades"]/x["test_days"] for x in rs]),
            "dd":  np.mean([x["max_drawdown_pct"] for x in rs]),
        }
    m  = _agg("bb_mtf")
    nm = _agg("bb_nomtf")
    print(f"\n{'':20}{'MTF':>10}{'kein MTF':>12}{'Δ':>8}")
    print("-" * 52)
    for k, label in [("pnl","Ø PnL %"),("pf","Ø PF"),("wr","Ø WR %"),
                     ("tpd","Ø T/d"),("dd","Ø DD %")]:
        d = m[k] - nm[k]
        print(f"  {label:<18}{m[k]:>10.3f}{nm[k]:>12.3f}{d:>+8.3f}")

print(f"\n{'='*65}")
print("EMPFEHLUNG FÜR LIVE-EINSATZ")
print(f"{'='*65}")
best_s = summ[0] if summ else None
if best_s:
    rs = by[best_s["label"]]
    # Finde die Config die am häufigsten alle Constraints erfüllt
    scalping_ok = [x for x in rs
                   if x["profit_factor"] >= 1.5
                   and x["num_trades"] >= 50
                   and x["max_drawdown_pct"] <= 20]
    if scalping_ok:
        best = max(scalping_ok, key=lambda x: x["total_pnl_pct"])
        cfg  = best["config"]
        tp   = (cfg.take_profit_pct or 0) * 100
        sl   = (cfg.stop_loss_pct   or 0) * 100
        print(f"\n✅  {best_s['label'].upper()} – alle Scalping-Constraints erfüllt")
        print(f"    Strategie : {best['strat_str']}")
        print(f"    Config    : {cfg.leverage}x | Pos {cfg.position_size*100:.0f}% | TP {tp:.2f}% | SL {sl:.2f}%")
        print(f"    OOS-PnL   : {best['total_pnl_pct']:+.2f}%  PF {best['profit_factor']:.3f}  "
              f"WR {best['winrate_pct']:.1f}%")
    else:
        best = max(rs, key=lambda x: x["total_pnl_pct"])
        cfg  = best["config"]
        tp   = (cfg.take_profit_pct or 0) * 100
        sl   = (cfg.stop_loss_pct   or 0) * 100
        print(f"\n⚡  {best_s['label'].upper()} – bester Kompromiss")
        print(f"    Strategie : {best['strat_str']}")
        print(f"    Config    : {cfg.leverage}x | Pos {cfg.position_size*100:.0f}% | TP {tp:.2f}% | SL {sl:.2f}%")
        print(f"    Ø PnL     : {best_s['avg_pnl']:+.2f}%/Periode | {best_s['avg_tpd']:.1f} Trades/Tag")
        nok = [f for f, c in [("PF>1.5", best_s["avg_pf"]>=1.5),
                               ("≥50T",   best_s["avg_tpd"]*TEST_DAYS>=50),
                               ("DD≤20",  best_s["avg_dd"]<=20)] if not c]
        if nok:
            print(f"    Fehlt     : {', '.join(nok)} – Parameter weiter verfeinern")

print(f"\n{'='*65}")
print("Analyse abgeschlossen.")
print(f"{'='*65}\n")

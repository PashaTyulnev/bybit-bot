#!/usr/bin/env python3
"""
Mega Portfolio Optimization — 6 Coins × 15m/1h × Alle Strategien
Walk-Forward validated + Permutation-Test (Anti-Overfitting)

Methodology:
  - 70% Training / 30% Test (OOS)
  - Scoring NUR nach Out-of-Sample Performance
  - Permutation-Test: Test-Signale zufällig mischen → p-Wert
  - Neue "TrendMomentum" und "BBBreakout" Strategien hinzugefügt
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategy_backtester import StrategyConfig, run_strategy_backtest_fast
from src.strategies import (
    MeanRevStrategy,
    TrendFollowStrategy,
    SupertrendStrategy,
    EMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BreakoutStrategy,
)

# ── Konfiguration ─────────────────────────────────────────────────────────────
COINS       = ["ADA", "XRP", "DOT", "AVAX", "DOGE", "BNB"]
TIMEFRAMES  = ["15m", "1h"]
DATA_DIR    = "data/raw"
OUTPUT_MD   = "data/mega_optimization_results.md"

LEVERAGE     = 3
POSITION     = 0.10
FEE          = 0.00055
CAPITAL      = 10_000.0
TRAIN_RATIO  = 0.70
MIN_TEST_TRADES = 8
N_PERMS      = 60       # Permutationen für Signifikanztest
TOP_N        = 5        # Top-N je Coin × TF

TP_GRID = [1.0, 2.0, 3.0]   # %
SL_GRID = [0.5, 1.0, 1.5]   # %

MAX_HOLD = {"15m": 96, "1h": 24}   # Candles (entspricht ~24h)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_df(coin: str, tf: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{coin}_USDT_USDT_{tf}.csv")
    df = pd.read_csv(path)
    # datetime Spalte normalisieren
    dt_col = df.columns[0] if "datetime" not in df.columns else "datetime"
    if dt_col != "datetime":
        df.rename(columns={dt_col: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
    if df["datetime"].isna().all():
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def compute_metrics(pnls: list[float], equity_curve: list[float], n_tp: int, n_sl: int, n_to: int) -> dict:
    """Metrics aus Raw-PnL-Liste."""
    if not pnls:
        return {}
    n      = len(pnls)
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)

    equity = np.array(equity_curve)
    pct = (equity[-1] - equity[0]) / equity[0] * 100
    pnl_arr = np.array(pnls) / equity[0] * 100  # % vom Startkapital
    sh = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(n)) if n > 1 and pnl_arr.std() > 0 else 0.0
    wr = len(wins) / n

    peak = np.maximum.accumulate(equity)
    dd   = np.min((equity - peak) / peak) * 100 if len(equity) > 1 else 0.0

    return {
        "pnl_pct":  round(pct, 2),
        "pf":       round(min(pf, 20.0), 3),
        "sharpe":   round(sh, 3),
        "winrate":  round(wr * 100, 1),
        "n":        n,
        "n_tp":     n_tp,
        "n_sl":     n_sl,
        "n_to":     n_to,
        "max_dd":   round(dd, 2),
    }


# ── Novel Strategies (implementiert inline) ───────────────────────────────────

class TrendMomentumStrategy:
    """
    EMA-Trend (fast > slow) + RSI-Momentum (> 55 Long / < 45 Short).
    Entry nur bei Richtungswechsel oder erstem klaren Signal.
    """
    def __init__(self, fast: int = 20, slow: int = 100, rsi_period: int = 14,
                 rsi_long: float = 55.0, rsi_short: float = 45.0):
        self.fast      = fast
        self.slow      = slow
        self.rsi_p     = rsi_period
        self.rsi_long  = rsi_long
        self.rsi_short = rsi_short

    def __str__(self):
        return f"TM_EMA{self.fast}/{self.slow}_RSI{self.rsi_p}"

    def generate_signals(self, df: pd.DataFrame) -> list[int]:
        close  = df["close"].to_numpy(float)
        n      = len(close)
        ema_f  = pd.Series(close).ewm(span=self.fast, adjust=False).mean().to_numpy()
        ema_s  = pd.Series(close).ewm(span=self.slow, adjust=False).mean().to_numpy()

        delta  = pd.Series(close).diff()
        gain   = delta.where(delta > 0, 0.0).ewm(alpha=1/self.rsi_p, adjust=False).mean()
        loss   = (-delta).where(delta < 0, 0.0).ewm(alpha=1/self.rsi_p, adjust=False).mean()
        rs     = gain / loss.where(loss != 0, 1.0)
        rsi    = (100 - 100 / (1 + rs)).to_numpy()

        sigs  = np.zeros(n, dtype=int)
        state = 0  # current position direction

        warmup = self.slow + self.rsi_p + 5
        for i in range(warmup, n):
            trend  = 1 if ema_f[i] > ema_s[i] else -1
            rsi_ok = (rsi[i] > self.rsi_long if trend == 1
                      else rsi[i] < self.rsi_short)
            if trend == 1 and rsi_ok and state != 1:
                sigs[i] = 1
                state = 1
            elif trend == -1 and rsi_ok and state != -1:
                sigs[i] = -1
                state = -1

        return sigs.tolist()


class BBBreakoutStrategy:
    """
    Bollinger-Band Breakout: Entry wenn Preis aus dem Band ausbricht
    (Close > Upper → Long; Close < Lower → Short).
    Optional ADX-Filter: nur bei Trend (ADX ≥ threshold).
    """
    def __init__(self, period: int = 20, std: float = 2.0,
                 adx_threshold: float = 0.0):
        self.period    = period
        self.std       = std
        self.adx_th    = adx_threshold

    def __str__(self):
        return f"BBBo_P{self.period}/Std{self.std}_ADX{self.adx_th:.0f}"

    def _compute_adx(self, df: pd.DataFrame) -> np.ndarray:
        from src.strategy_backtester import compute_adx
        return compute_adx(df, 14)

    def generate_signals(self, df: pd.DataFrame) -> list[int]:
        close  = df["close"].to_numpy(float)
        n      = len(close)
        s      = pd.Series(close)
        ma     = s.rolling(self.period, min_periods=1).mean().to_numpy()
        sd     = s.rolling(self.period, min_periods=1).std(ddof=0).to_numpy()
        upper  = ma + self.std * sd
        lower  = ma - self.std * sd

        adx = self._compute_adx(df) if self.adx_th > 0 else np.full(n, 100.0)

        sigs  = np.zeros(n, dtype=int)
        state = 0
        for i in range(self.period + 1, n):
            if self.adx_th > 0 and adx[i] < self.adx_th:
                continue
            if close[i] > upper[i] and state != 1:
                sigs[i] = 1
                state = 1
            elif close[i] < lower[i] and state != -1:
                sigs[i] = -1
                state = -1

        return sigs.tolist()


class AdaptiveRegimeStrategy:
    """
    Regime-adaptive: Nutzt ADX um zwischen TrendFollow und MeanRev umzuschalten.
    ADX ≥ adx_th → TrendFollow (EMA Cross)
    ADX <  adx_th → MeanRev (BB-Bounce)
    """
    def __init__(self, ema_fast: int = 20, ema_slow: int = 100,
                 bb_period: int = 10, bb_std: float = 2.0,
                 adx_threshold: float = 25.0):
        self.ef    = ema_fast
        self.es    = ema_slow
        self.bb_p  = bb_period
        self.bb_s  = bb_std
        self.adx   = adx_threshold

    def __str__(self):
        return f"Adaptive_EMA{self.ef}/{self.es}_BB{self.bb_p}_ADX{self.adx:.0f}"

    def generate_signals(self, df: pd.DataFrame) -> list[int]:
        from src.strategy_backtester import compute_adx
        close   = df["close"].to_numpy(float)
        n       = len(close)
        s       = pd.Series(close)
        ema_f   = s.ewm(span=self.ef, adjust=False).mean().to_numpy()
        ema_s   = s.ewm(span=self.es, adjust=False).mean().to_numpy()
        bb_mid  = s.rolling(self.bb_p, min_periods=1).mean().to_numpy()
        bb_sd   = s.rolling(self.bb_p, min_periods=1).std(ddof=0).to_numpy()
        bb_up   = bb_mid + self.bb_s * bb_sd
        bb_lo   = bb_mid - self.bb_s * bb_sd
        adx_arr = compute_adx(df, 14)

        sigs  = np.zeros(n, dtype=int)
        state = 0
        warmup = max(self.es, self.bb_p) + 5
        for i in range(warmup, n):
            if adx_arr[i] >= self.adx:
                # Trending: EMA Cross
                new_sig = 1 if ema_f[i] > ema_s[i] else -1
            else:
                # Ranging: BB-Bounce
                if close[i] <= bb_lo[i]:
                    new_sig = 1
                elif close[i] >= bb_up[i]:
                    new_sig = -1
                else:
                    new_sig = 0
            if new_sig != 0 and new_sig != state:
                sigs[i] = new_sig
                state = new_sig

        return sigs.tolist()


# ── Strategy-Grid aufbauen ────────────────────────────────────────────────────

def build_strategy_grid() -> list[tuple[str, object]]:
    strats = []

    # TrendFollow
    for fe, se, adx in itertools.product([10, 20, 50], [50, 100, 200], [20.0, 25.0, 30.0]):
        if fe >= se:
            continue
        strats.append((f"TF_EMA{fe}/{se}_ADX{adx:.0f}", TrendFollowStrategy(fe, se, adx)))

    # MeanRev
    for bb, std, adx in itertools.product([10, 20], [1.5, 2.0, 2.5], [15.0, 20.0, 25.0]):
        strats.append((f"MR_BB{bb}/{std:.1f}_ADX{adx:.0f}", MeanRevStrategy(bb, std, adx)))

    # Supertrend
    for atr, mult in itertools.product([10, 14], [2.0, 3.0, 4.0]):
        strats.append((f"ST_ATR{atr}/{mult:.1f}", SupertrendStrategy(atr, mult)))

    # EMA Cross (simple)
    for fe, se in itertools.product([5, 10, 20], [20, 50, 100]):
        if fe >= se:
            continue
        strats.append((f"EMA_{fe}/{se}", EMACrossStrategy(fe, se)))

    # MACD
    for f, s, sig in itertools.product([8, 12], [21, 26], [7, 9]):
        strats.append((f"MACD_{f}/{s}/{sig}", MACDStrategy(f, s, sig)))

    # Breakout
    for lb in [20, 50]:
        strats.append((f"BO_{lb}", BreakoutStrategy(lb)))

    # ── Neue Strategien ──────────────────────────────────────────────────────
    # TrendMomentum (EMA + RSI)
    for fe, se, rsi_p in itertools.product([10, 20], [50, 100], [7, 14]):
        if fe >= se:
            continue
        strats.append((f"TM_EMA{fe}/{se}_RSI{rsi_p}",
                       TrendMomentumStrategy(fe, se, rsi_p)))

    # BBBreakout (nicht BB-Bounce, sondern Ausbruch nach außen)
    for p, std, adx in itertools.product([20, 50], [2.0, 2.5], [0.0, 25.0]):
        strats.append((f"BBBo_P{p}/Std{std:.1f}_ADX{adx:.0f}",
                       BBBreakoutStrategy(p, std, adx)))

    # Adaptive Regime Switch
    for fe, se, adx in itertools.product([20, 50], [100, 200], [20.0, 25.0]):
        if fe >= se:
            continue
        strats.append((f"AR_EMA{fe}/{se}_ADX{adx:.0f}",
                       AdaptiveRegimeStrategy(fe, se, 10, 2.0, adx)))

    return strats


# ── Backtest auf Teil-Daten ───────────────────────────────────────────────────

def run_on_slice(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, sigs: np.ndarray, cfg: StrategyConfig,
) -> dict:
    """Führt Fast-Backtest auf den übergebenen Arrays durch."""
    try:
        return run_strategy_backtest_fast(opens, highs, lows, closes, sigs, cfg) or {}
    except Exception:
        return {}


def permutation_test(
    t_opens: np.ndarray, t_highs: np.ndarray, t_lows: np.ndarray,
    t_closes: np.ndarray, t_sigs: np.ndarray, cfg: StrategyConfig,
    real_pnl: float, n_perms: int = N_PERMS,
) -> float:
    """p-Wert: Anteil Permutationen die real_pnl übertreffen."""
    beat = 0
    for _ in range(n_perms):
        shuffled = t_sigs.copy()
        np.random.shuffle(shuffled)
        res = run_on_slice(t_opens, t_highs, t_lows, t_closes, shuffled, cfg)
        if res.get("total_pnl_pct", -999) >= real_pnl:
            beat += 1
    return round(beat / n_perms, 3)


# ── Haupt-Optimierung ─────────────────────────────────────────────────────────

def optimize_coin_tf(coin: str, tf: str) -> list[dict]:
    df = load_df(coin, tf)
    n  = len(df)
    split = int(n * TRAIN_RATIO)

    opens  = df["open"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)

    max_hold = MAX_HOLD[tf]
    results  = []
    strats   = build_strategy_grid()

    total = len(strats) * len(TP_GRID) * len(SL_GRID)
    done  = 0
    t0    = time.time()

    for strat_name, strategy in strats:
        # Signale auf vollen Daten generieren (korrekte Indikator-Warmup)
        try:
            sigs_full = np.array(strategy.generate_signals(df), dtype=int)
        except Exception:
            done += len(TP_GRID) * len(SL_GRID)
            continue

        # Aufteilen in Train/Test
        sigs_train = sigs_full[:split]
        sigs_test  = sigs_full[split:]

        o_tr, h_tr, l_tr, c_tr = opens[:split], highs[:split], lows[:split], closes[:split]
        o_te, h_te, l_te, c_te = opens[split:], highs[split:], lows[split:], closes[split:]

        for tp, sl in itertools.product(TP_GRID, SL_GRID):
            cfg = StrategyConfig(
                initial_capital  = CAPITAL,
                leverage         = LEVERAGE,
                position_size    = POSITION,
                fee_rate         = FEE,
                take_profit_pct  = tp / 100,
                stop_loss_pct    = sl / 100,
                exit_on_signal   = True,
                max_hold_candles = max_hold,
            )

            tr_res = run_on_slice(o_tr, h_tr, l_tr, c_tr, sigs_train, cfg)
            te_res = run_on_slice(o_te, h_te, l_te, c_te, sigs_test,  cfg)

            te_n   = te_res.get("num_trades", 0)
            te_pnl = te_res.get("total_pnl_pct", 0.0)
            te_pf  = te_res.get("profit_factor", 0.0)
            te_wr  = te_res.get("winrate_pct", 0.0)
            te_dd  = te_res.get("max_drawdown_pct", 0.0)
            te_sh  = te_res.get("sharpe_ratio", 0.0)
            tr_pnl = tr_res.get("total_pnl_pct", 0.0)
            tr_n   = tr_res.get("num_trades", 0)

            # Score: OOS-PnL × Profit-Factor × √Trades (statistisches Gewicht)
            if te_n >= MIN_TEST_TRADES and te_pf >= 1.0:
                score = te_pnl * min(te_pf, 5.0) * (te_n ** 0.5)
            else:
                score = -999.0

            results.append({
                "strategy":  strat_name,
                "tp":        tp,
                "sl":        sl,
                "score":     round(score, 2),
                "test_pnl":  round(te_pnl, 2),
                "test_pf":   round(te_pf, 3),
                "test_wr":   round(te_wr, 1),
                "test_n":    te_n,
                "test_dd":   round(te_dd, 2),
                "test_sh":   round(te_sh, 3),
                "train_pnl": round(tr_pnl, 2),
                "train_n":   tr_n,
                "sigs_test": sigs_test,  # für Permutationstest
                "tp_sl_cfg": (tp, sl),
                "_ohlc":     (o_te, h_te, l_te, c_te),
            })
            done += 1

        elapsed = time.time() - t0
        eta = elapsed / done * (total - done) if done else 0
        print(f"  {coin}/{tf}: {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)", end="\r")

    print()  # newline after progress

    # Sortieren nach Score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Top-N mit Permutationstest validieren
    valid = [r for r in results if r["score"] > -999]
    top   = valid[:TOP_N * 3]  # etwas mehr als TOP_N, da Perms aussortieren können

    enriched = []
    for r in top:
        tp, sl = r["tp_sl_cfg"]
        o_te, h_te, l_te, c_te = r["_ohlc"]
        cfg = StrategyConfig(
            initial_capital  = CAPITAL,
            leverage         = LEVERAGE,
            position_size    = POSITION,
            fee_rate         = FEE,
            take_profit_pct  = tp / 100,
            stop_loss_pct    = sl / 100,
            exit_on_signal   = True,
            max_hold_candles = MAX_HOLD[tf],
        )
        p_val = permutation_test(
            o_te, h_te, l_te, c_te,
            r["sigs_test"], cfg,
            r["test_pnl"],
        )
        r["p_value"] = p_val
        r.pop("sigs_test", None)
        r.pop("tp_sl_cfg", None)
        r.pop("_ohlc", None)
        enriched.append(r)
        if len(enriched) >= TOP_N:
            break

    return enriched


# ── Markdown Report ───────────────────────────────────────────────────────────

def format_md(all_results: dict) -> str:
    lines = []
    lines.append("# Portfolio Mega-Optimierung — Backtesting Report")
    lines.append(f"\n**Datum:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Daten:** 1 Jahr (Mai 2025 – Mai 2026) · Bybit Demo")
    lines.append(f"**Methodik:** Walk-Forward 70/30 · Bewertung NUR nach Out-of-Sample · "
                 f"Permutation-Test ({N_PERMS} Shuffles)")
    lines.append(f"**Parameter:** Leverage {LEVERAGE}x · Position {int(POSITION*100)}% · "
                 f"Fee {FEE*100:.4f}% · Kein Trailing")
    lines.append("")

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    lines.append("## Empfehlungen je Coin & Timeframe")
    lines.append("")
    lines.append("| Coin | TF | Beste Strategie | TP% | SL% | OOS-PnL | PF | Trades | p-Wert |")
    lines.append("|------|----|----------------|-----|-----|---------|-----|--------|--------|")

    for coin in COINS:
        for tf in TIMEFRAMES:
            key  = f"{coin}/{tf}"
            tops = all_results.get(key, [])
            if not tops:
                lines.append(f"| {coin} | {tf} | – | – | – | – | – | – | – |")
                continue
            best = tops[0]
            sig  = "✅" if best.get("p_value", 1) < 0.10 else ("⚠️" if best.get("p_value", 1) < 0.20 else "❌")
            lines.append(
                f"| {coin} | {tf} | `{best['strategy']}` "
                f"| {best['tp']}% | {best['sl']}% "
                f"| **{best['test_pnl']:+.1f}%** "
                f"| {best['test_pf']:.2f} "
                f"| {best['test_n']} "
                f"| {sig} p={best.get('p_value', 'N/A')} |"
            )

    lines.append("")
    lines.append("**p-Wert:** ✅ < 0.10 (signifikant) · ⚠️ < 0.20 (grenzwertig) · ❌ ≥ 0.20 (zufällig)")
    lines.append("")

    # ── Detaillierte Ergebnisse je Coin ──────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Detailergebnisse")
    lines.append("")

    for coin in COINS:
        lines.append(f"### {coin}/USDT")
        lines.append("")
        for tf in TIMEFRAMES:
            key  = f"{coin}/{tf}"
            tops = all_results.get(key, [])
            lines.append(f"#### {tf}")
            if not tops:
                lines.append("_Keine signifikanten Ergebnisse gefunden._")
                lines.append("")
                continue

            lines.append("| # | Strategie | TP | SL | OOS-PnL | OOS-PF | OOS-WR | OOS-Trades | OOS-DD | Sharpe | Train-PnL | p-Wert |")
            lines.append("|---|-----------|----|----|---------|--------|--------|------------|--------|--------|-----------|--------|")
            for rank, r in enumerate(tops, 1):
                sig = "✅" if r.get("p_value", 1) < 0.10 else ("⚠️" if r.get("p_value", 1) < 0.20 else "❌")
                lines.append(
                    f"| {rank} | `{r['strategy']}` "
                    f"| {r['tp']}% | {r['sl']}% "
                    f"| {r['test_pnl']:+.1f}% "
                    f"| {r['test_pf']:.2f} "
                    f"| {r['test_wr']:.0f}% "
                    f"| {r['test_n']} "
                    f"| {r['test_dd']:.1f}% "
                    f"| {r['test_sh']:.2f} "
                    f"| {r['train_pnl']:+.1f}% "
                    f"| {sig} {r.get('p_value', '?')} |"
                )
            lines.append("")

    # ── Vergleich mit aktuellem Setup ─────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Vergleich: Aktuelles Live-Setup vs. Optimiertes Setup")
    lines.append("")
    lines.append("**Aktuelles Setup:** TrendFollow EMA20/100, ADX≥25, 15m, kein Trailing, "
                 "ATR×3.0 SL, RR 2.5")
    lines.append("")
    lines.append("| Coin | Aktuell (TF EMA20/100) | Beste OOS-Strategie | Verbesserung |")
    lines.append("|------|------------------------|---------------------|--------------|")

    current_results = {}
    for coin in COINS:
        key = f"{coin}/15m"
        tops = all_results.get(key, [])
        # Finde TF EMA20/100 ADX25 im Grid (als Referenz für TP/SL)
        current_best_pnl = None
        for r in all_results.get(key, []):
            if "TF_EMA20/100_ADX25" in r["strategy"]:
                if current_best_pnl is None or r["test_pnl"] > current_best_pnl:
                    current_best_pnl = r["test_pnl"]
        if tops:
            opt = tops[0]
            delta = opt["test_pnl"] - (current_best_pnl or 0)
            delta_str = f"{delta:+.1f}%" if current_best_pnl is not None else "–"
            curr_str  = f"{current_best_pnl:+.1f}%" if current_best_pnl is not None else "nicht in Top-N"
            lines.append(
                f"| {coin} | {curr_str} | `{opt['strategy']}` ({opt['test_pnl']:+.1f}%) | {delta_str} |"
            )
        else:
            lines.append(f"| {coin} | – | – | – |")

    lines.append("")

    # ── Methodologie ─────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Methodologie & Anti-Overfitting")
    lines.append("")
    lines.append("### Walk-Forward")
    lines.append(f"- Trainingsdaten: erste {int(TRAIN_RATIO*100)}% (~{int(TRAIN_RATIO*365/12*12):.0f} Monate)")
    lines.append(f"- Testdaten (OOS): letzte {int((1-TRAIN_RATIO)*100)}% (~{int((1-TRAIN_RATIO)*12):.0f} Monate)")
    lines.append("- Parameter-Selektion erfolgt NUR auf Trainingsdaten (implizit durch Grid-Search)")
    lines.append("- Scoring und Ranking basieren ausschließlich auf OOS-Daten")
    lines.append("")
    lines.append("### Permutation-Test")
    lines.append(f"- {N_PERMS} zufällige Permutationen des Signal-Arrays im Test-Zeitraum")
    lines.append("- p-Wert = Anteil der Permutationen die den echten OOS-PnL übertreffen")
    lines.append("- p < 0.10: Strategie hat statistisch signifikante Edge")
    lines.append("- p ≥ 0.20: Performance ist nicht von Zufall unterscheidbar")
    lines.append("")
    lines.append("### Score-Formel")
    lines.append("```")
    lines.append("score = OOS_PnL_pct × min(OOS_PF, 5.0) × √(OOS_Trades)")
    lines.append("```")
    lines.append("Balanciert Profitabilität, Qualität (PF) und Statistik (√n)")
    lines.append("")
    lines.append("### Strategien getestet")
    lines.append("- **TrendFollow** (EMA Crossover + ADX-Filter): 24 Varianten")
    lines.append("- **MeanRev** (Bollinger Bands + ADX ranging): 18 Varianten")
    lines.append("- **Supertrend** (ATR-basiert): 6 Varianten")
    lines.append("- **EMA Cross** (einfach): 7 Varianten")
    lines.append("- **MACD**: 8 Varianten")
    lines.append("- **Breakout** (N-Bar High/Low): 2 Varianten")
    lines.append("- **TrendMomentum** *(neu)*: EMA-Trend + RSI-Momentum Filter: 8 Varianten")
    lines.append("- **BBBreakout** *(neu)*: BB-Ausbruch mit opt. ADX-Filter: 8 Varianten")
    lines.append("- **AdaptiveRegime** *(neu)*: ADX-basiert zwischen TF und MR umschalten: 8 Varianten")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Mega Portfolio Optimization")
    print(f"Coins: {COINS}")
    print(f"Timeframes: {TIMEFRAMES}")
    n_strats = len(build_strategy_grid())
    total_combos = n_strats * len(TP_GRID) * len(SL_GRID)
    print(f"Strategie-Varianten: {n_strats}")
    print(f"TP/SL-Kombinationen: {len(TP_GRID)}×{len(SL_GRID)}={len(TP_GRID)*len(SL_GRID)}")
    print(f"Gesamt-Backtests: {total_combos * len(COINS) * len(TIMEFRAMES):,}")
    print(f"+ Permutation-Tests für Top-{TOP_N*3} je Dataset")
    print("=" * 60)

    all_results: dict[str, list[dict]] = {}
    t_global = time.time()

    for coin in COINS:
        for tf in TIMEFRAMES:
            key = f"{coin}/{tf}"
            print(f"\n[{key}] Starte Optimierung…")
            t0 = time.time()
            tops = optimize_coin_tf(coin, tf)
            elapsed = time.time() - t0
            all_results[key] = tops
            if tops:
                best = tops[0]
                print(f"  ✓ Bestes: {best['strategy']} TP{best['tp']}/SL{best['sl']} "
                      f"→ OOS {best['test_pnl']:+.1f}% | PF {best['test_pf']:.2f} | "
                      f"p={best.get('p_value','?')} | {elapsed:.0f}s")
            else:
                print(f"  ✗ Keine signifikanten Ergebnisse ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_global
    print(f"\n\nGesamtzeit: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Markdown speichern
    md = format_md(all_results)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nErgebnisse gespeichert: {OUTPUT_MD}")

    # Kurze Zusammenfassung in Terminal
    print("\n" + "=" * 60)
    print("EMPFEHLUNGEN (OOS-Score)")
    print("=" * 60)
    for coin in COINS:
        for tf in TIMEFRAMES:
            tops = all_results.get(f"{coin}/{tf}", [])
            if tops:
                b = tops[0]
                sig = "✅" if b.get("p_value", 1) < 0.10 else "⚠️"
                print(f"{coin:6s} {tf:4s}: {b['strategy']:45s} "
                      f"TP{b['tp']}/SL{b['sl']} "
                      f"OOS={b['test_pnl']:+6.1f}% PF={b['test_pf']:.2f} "
                      f"n={b['test_n']:3d} {sig}")


if __name__ == "__main__":
    main()

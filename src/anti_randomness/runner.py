"""
Grid-Sweep und Orchestrierung der Anti-Randomness Backtests.

Führt alle Strategie × Parameter-Kombinationen durch und sammelt Ergebnisse.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Callable

import numpy as np
import pandas as pd

from src.anti_randomness import config as C
from src.anti_randomness.data_loader import load_csv
from src.anti_randomness.indicators import atr as calc_atr, align_htf_trend
from src.anti_randomness.engine import AtrBacktestConfig, run_atr_backtest
from src.anti_randomness.strategies import (
    TrendFollowStrategy, MeanReversionRegime,
    RegimeSwitchStrategy, VolatilityFilterStrategy,
)
from src.anti_randomness.random_baseline import (
    run_random_baseline, p_value_vs_random,
)

logger = logging.getLogger(__name__)


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def _make_cfg(atr_mult: float, rr: float) -> AtrBacktestConfig:
    return AtrBacktestConfig(
        initial_capital  = C.INITIAL_CAPITAL,
        risk_per_trade   = C.RISK_PER_TRADE,
        leverage         = C.LEVERAGE,
        fee_rate         = C.FEE_RATE,
        slippage_rate    = C.SLIPPAGE_RATE,
        atr_multiplier   = atr_mult,
        rr_ratio         = rr,
    )


def _score(r: dict) -> float:
    """Ranking-Score: PF × Sharpe × (1 - MaxDD/100)."""
    pf     = min(r.get("profit_factor",    0.0), 10.0)
    sharpe = r.get("sharpe_ratio",          0.0)
    dd     = r.get("max_drawdown_pct",    100.0)
    trades = r.get("num_trades",              0)
    if trades < 5:
        return -1.0
    return pf * max(sharpe, 0.01) * max(0.01, 1.0 - dd / 100.0)


def _run_one(
    tag:      str,
    strategy,
    df_base:  pd.DataFrame,
    atr_arr:  np.ndarray,
    df_htf:   pd.DataFrame | None,
    htf_ema:  int,
    atr_mult: float,
    rr:       float,
    progress: Callable[[str], None] | None = None,
) -> dict | None:
    """Ein einzelner Combo-Backtest. Gibt None zurück wenn zu wenig Trades."""
    signals = strategy.generate_signals(df_base).to_numpy(int)

    # HTF-Trend-Filter anwenden
    if df_htf is not None and len(df_htf) > htf_ema + 2:
        htf_trend = align_htf_trend(df_base, df_htf, ema_period=htf_ema)
        mask = ((signals == 1) & (htf_trend == 1)) | ((signals == -1) & (htf_trend == -1))
        signals = np.where(mask, signals, 0).astype(int)

    cfg = _make_cfg(atr_mult, rr)
    result = run_atr_backtest(df_base, signals, atr_arr, cfg)
    if not result or result.get("num_trades", 0) < 3:
        return None

    result.update({
        "tag":          tag,
        "strategy":     str(strategy),
        "atr_mult":     atr_mult,
        "rr":           rr,
        "htf_filtered": df_htf is not None,
        "score":        _score(result),
    })

    # Random-Baseline
    if C.ENABLE_RANDOM:
        baseline = run_random_baseline(
            df_base, signals, atr_arr, cfg, n_runs=C.RANDOM_RUNS
        )
        result["random_baseline"] = baseline
        result["p_value"] = p_value_vs_random(result["profit_factor"], baseline)
    else:
        result["random_baseline"] = {}
        result["p_value"] = float("nan")

    if progress:
        n  = result["num_trades"]
        pf = result["profit_factor"]
        p  = result.get("p_value", float("nan"))
        p_str = f"{p:.3f}" if p == p else "n/a"
        progress(f"  {tag:<54}  trades={n:>3}  PF={pf:.3f}  p={p_str}")

    return result


# ── Haupt-Funktion ─────────────────────────────────────────────────────────────

def run_all(
    symbol:      str | None = None,
    base_tf:     str | None = None,
    trend_tf:    str | None = None,
    start:       str | None = None,
    end:         str | None = None,
    strategies:  list[str] | None = None,  # None = alle
    quick:       bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Führt alle Strategie × Parameter-Kombinationen durch.

    Gibt Liste von Result-Dicts zurück, sortiert nach Score (bestes zuerst).
    """
    sym       = symbol   or C.SYMBOL
    btf       = base_tf  or C.BASE_TF
    ttf       = trend_tf or C.TREND_TF
    s_date    = start    or C.START_DATE
    e_date    = end      or C.END_DATE

    t0 = time.perf_counter()

    if progress_cb:
        progress_cb(f"[Lade Daten] Symbol={sym}  Base-TF={btf}  Trend-TF={ttf}")

    # Marktdaten laden
    df_base = load_csv(sym, btf, s_date, e_date)
    if len(df_base) < 200:
        raise ValueError(
            f"Zu wenig Daten ({len(df_base)} Kerzen) für {sym} {btf}. "
            "Bitte Datumsgrenzen oder Symbol prüfen."
        )

    try:
        df_htf = load_csv(sym, ttf, s_date, e_date)
    except FileNotFoundError:
        df_htf = None
        if progress_cb:
            progress_cb(f"  [Warn] Kein HTF-Data für {ttf} — HTF-Filter deaktiviert.")

    if progress_cb:
        progress_cb(
            f"  Base: {len(df_base):,} Kerzen  "
            f"({df_base['datetime'].iloc[0].strftime('%Y-%m-%d')} – "
            f"{df_base['datetime'].iloc[-1].strftime('%Y-%m-%d')})"
        )
        if df_htf is not None:
            progress_cb(f"  HTF:  {len(df_htf):,} Kerzen ({ttf})")

    # Grid-Parameter (Quick-Mode reduziert Suchraum)
    atr_periods  = [14]          if quick else C.ATR_PERIODS
    atr_mults    = [1.5, 2.0]    if quick else C.ATR_MULTIPLIERS
    rr_vals      = [1.5, 2.0]    if quick else C.RR_VALUES
    fast_emas    = [20]          if quick else C.FAST_EMAS
    slow_emas    = [100]         if quick else C.SLOW_EMAS
    vol_mins     = [C.VOL_MIN_VALUES[0]] if quick else C.VOL_MIN_VALUES
    vol_maxs     = [C.VOL_MAX_VALUES[-1]] if quick else C.VOL_MAX_VALUES
    htf_ema_pds  = [50]

    # Pre-compute ATR arrays (ein mal pro atr_period)
    atr_arrays: dict[int, np.ndarray] = {
        p: calc_atr(df_base, p) for p in atr_periods
    }

    want = set(strategies) if strategies else None

    results: list[dict] = []
    total_combos = 0

    def _run(tag, strategy, atr_p, atr_m, rr, htf_ema=50):
        nonlocal total_combos
        total_combos += 1
        r = _run_one(
            tag, strategy,
            df_base, atr_arrays[atr_p], df_htf,
            htf_ema, atr_m, rr,
            progress=progress_cb,
        )
        if r:
            r["atr_period"] = atr_p
            results.append(r)

    # ── Strategie 1: Trend-Following ──────────────────────────────────────────
    if not want or "trend" in want:
        if progress_cb:
            progress_cb("\n[Strategie 1/4] Trend-Following (EMA-Cross + HTF-Filter)")
        for fast, slow, atr_p, atr_m, rr in itertools.product(
            fast_emas, slow_emas, atr_periods, atr_mults, rr_vals,
        ):
            if fast >= slow:
                continue
            for htf_ema in htf_ema_pds:
                strat = TrendFollowStrategy(fast=fast, slow=slow)
                tag   = f"TrendFollow(EMA{fast}/{slow}) ATR{atr_p}×{atr_m} RR{rr} HTF-EMA{htf_ema}"
                _run(tag, strat, atr_p, atr_m, rr, htf_ema)

    # ── Strategie 2: Mean-Reversion ───────────────────────────────────────────
    if not want or "meanrev" in want:
        if progress_cb:
            progress_cb("\n[Strategie 2/4] Mean-Reversion (BB, nur Ranging-Markt)")
        bb_params  = [(10, 2.0), (20, 2.0)] if quick else [(10, 1.5), (10, 2.0), (20, 2.0), (20, 2.5)]
        adx_thrs   = [25.0] if quick else [20.0, 25.0, 30.0]
        for (bb_p, bb_std), adx_th, atr_p, atr_m, rr in itertools.product(
            bb_params, adx_thrs, atr_periods, atr_mults, rr_vals,
        ):
            strat = MeanReversionRegime(
                bb_period=bb_p, bb_std=bb_std, adx_threshold=adx_th
            )
            tag = (
                f"MeanRev(BB{bb_p}/{bb_std}σ,ADX<{adx_th}) "
                f"ATR{atr_p}×{atr_m} RR{rr}"
            )
            _run(tag, strat, atr_p, atr_m, rr)

    # ── Strategie 3: Regime-Switch ────────────────────────────────────────────
    if not want or "regime" in want:
        if progress_cb:
            progress_cb("\n[Strategie 3/4] Regime-Switch (EMA wenn Trend, BB wenn Range)")
        adx_thrs = [25.0] if quick else [20.0, 25.0]
        for fast, slow, adx_th, atr_p, atr_m, rr in itertools.product(
            fast_emas, slow_emas, adx_thrs, atr_periods, atr_mults, rr_vals,
        ):
            if fast >= slow:
                continue
            strat = RegimeSwitchStrategy(
                fast_ema=fast, slow_ema=slow, adx_threshold=adx_th
            )
            tag = (
                f"RegimeSwitch(EMA{fast}/{slow},ADX{adx_th}) "
                f"ATR{atr_p}×{atr_m} RR{rr}"
            )
            _run(tag, strat, atr_p, atr_m, rr)

    # ── Strategie 4: Volatilitäts-Filter ─────────────────────────────────────
    if not want or "volfilter" in want:
        if progress_cb:
            progress_cb("\n[Strategie 4/4] Volatility-Filter (Trend-Follow + Vol-Fenster)")
        base_pairs  = [(20, 100)] if quick else list(itertools.product(fast_emas, slow_emas))
        vol_combos  = list(itertools.product(vol_mins, vol_maxs))
        for (fast, slow), (v_min, v_max), atr_p, atr_m, rr in itertools.product(
            base_pairs, vol_combos, atr_periods, atr_mults, rr_vals,
        ):
            if fast >= slow or v_min >= v_max:
                continue
            base_strat = TrendFollowStrategy(fast=fast, slow=slow)
            strat      = VolatilityFilterStrategy(
                base_strategy=base_strat,
                vol_min=v_min, vol_max=v_max,
            )
            tag = (
                f"VolFilter(EMA{fast}/{slow},vol[{v_min},{v_max}]) "
                f"ATR{atr_p}×{atr_m} RR{rr}"
            )
            _run(tag, strat, atr_p, atr_m, rr)

    elapsed = time.perf_counter() - t0
    if progress_cb:
        progress_cb(
            f"\n✓ {total_combos} Kombinationen getestet, "
            f"{len(results)} mit Trades  ({elapsed:.1f}s)"
        )

    results.sort(key=lambda r: r.get("score", -1), reverse=True)
    return results

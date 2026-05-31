"""
Streamlit GUI fuer Backtester + Optimizer.

Starten:
    streamlit run src/app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.backtester import BacktestConfig, Trade, run_backtest
from src.config import RAW_DATA_DIR
from src.download_ohlcv import fetch_ohlcv, save_csv
from src.reporting import compare_to_df, export_trades_csv, save_equity_chart
from src.strategies import (
    BollingerStrategy, BreakoutStrategy, EMACrossStrategy,
    MACDStrategy, RSIStrategy, SupertrendStrategy,
    CombinedStrategy, MeanRevStrategy, TrendFollowStrategy,
    STRATEGY_REGISTRY,
)
from src.strategy_backtester import StrategyConfig, run_strategy_backtest, run_strategy_backtest_fast, compute_adx
from src.metrics import extended_metrics
from src.walk_forward import simple_split
from src.live_trader import LiveTrader
from src.strategy_optimizer import (
    StrategyOptConfig, format_strategy_results_df, run_strategy_optimization,
    DEFAULT_LEVERAGES as SO_LEVERAGES,
    DEFAULT_POSITION_SIZES as SO_SIZES,
    DEFAULT_TP_PCTS as SO_TP,
    DEFAULT_SL_PCTS as SO_SL,
    EMA_GRID, RSI_GRID, RSI_DIV_GRID, BB_GRID, BO_GRID,
    apply_mtf_filter, apply_adx_filter,
)
from src.optimizer import (
    DEFAULT_LEVERAGES,
    DEFAULT_POSITION_SIZES,
    DEFAULT_SEQUENCES,
    DEFAULT_SL_PCTS,
    DEFAULT_TP_PCTS,
    OptimizeConfig,
    format_results_df,
    run_optimization,
)

st.set_page_config(page_title="Crypto Backtester", page_icon="📈", layout="wide")
st.title("📈 Crypto Backtester")
st.caption("Lokale Simulation auf historischen OHLCV-Daten")

# ── CSV-Dateien ────────────────────────────────────────────────────────────────
csv_files = []
if os.path.isdir(RAW_DATA_DIR):
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]

_no_csv = not csv_files
_csv_opts = csv_files if csv_files else ["(keine Datei – bitte Daten laden)"]

# ── Sidebar-Navigation ────────────────────────────────────────────────────────
_NAV_ITEMS = [
    "⚡ SuperTrend Live",
    "🤖 Live Trading", "🧪 OptiTest",
    "🔁 Backtest", "🔍 Optimizer", "📈 Strategien", "📊 Vergleich",
    "🔬 Strategie-Optimizer", "🎯 Multi-Symbol", "📥 Daten laden",
]

if "_page" not in st.session_state:
    st.session_state["_page"] = _NAV_ITEMS[0]

with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("""
<style>
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent;
    border: none;
    text-align: left;
    color: inherit;
    opacity: 0.75;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08);
    opacity: 1;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    background: rgba(255,255,255,0.12);
    border: none;
    text-align: left;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)
    for _nav_item in _NAV_ITEMS:
        _is_active = st.session_state["_page"] == _nav_item
        if st.button(
            _nav_item,
            key=f"nav_{_nav_item}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state["_page"] = _nav_item
            st.rerun()

_page = st.session_state["_page"]

if _no_csv:
    st.info("Keine Daten vorhanden. Bitte unter **📥 Daten laden** zuerst historische Daten herunterladen.", icon="📥")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# OPTITEST – Hilfs-Funktionen (Modul-Ebene)
# ══════════════════════════════════════════════════════════════════════════════

_OT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "optitest")

# Beste Strategien aus mega_optimization_results.md
_OT_STRAT_DEFS: list[tuple] = [
    # (label,             strategy,                              tp%,  sl%)
    ("ST_ATR14/3.0",      SupertrendStrategy(14, 3.0),          3.0,  1.5),
    ("MACD_12/21/7",      MACDStrategy(12, 21, 7),              3.0,  1.5),
    ("MACD_12/26/9",      MACDStrategy(12, 26, 9),              3.0,  1.5),
    ("EMA_20/50",         EMACrossStrategy(20, 50),             3.0,  1.0),
    ("BO_50",             BreakoutStrategy(50),                 3.0,  1.5),
    ("BB_50/2.5",         BollingerStrategy(50, 2.5, False),    2.0,  1.5),
    ("TF_EMA20/100_ADX25", TrendFollowStrategy(20, 100, 25.0), 3.0,  1.5),
]
_OT_LEVERAGES = [1, 2, 3]
_OT_FEE       = 0.00055  # Bybit Taker-Fee


_OT_MIN_VOL    = 1_000_000   # 1M USDT Mindest-Tagesvolumen
_OT_BLACKLIST  = {
    "1000PEPE","1000SHIB","1000BONK","1000RATS","1000TURBO","1000LUNC","1000BTT",
    "1000XEC","1000FLOKI","1000WHY","1000MOG","1000CAT","1000SATS","10000AIDOGE",
    "FARTCOIN","BONK","WIF","MEME","BABYDOGE","DOGECAT","TURBO","PEPE",
    "FLOKI","SHIB","COW","COQ","BOME","MYRO","POPCAT","PNUT","MOG",
    "NEIRO","BRETT","SUNDOG","PONKE","MANEKI","GIGA","SLERF","SILLY",
    "WIENER","MIGGLES","LADYS","WOJAK","TOAD","MAGA","TRUMP","MELANIA",
    "FIDA","LOTTO","BEAT","BILL","BSB","CL","NIL","GRASS","BCUT",
    "ORCA","ALPINE","ACM","CITY","PORTO","SANTOS","ATM","BAR","JUV",
    "USDC","USDT","BUSD","DAI","TUSD","FRAX","USDP","GUSD","LUSD",
    "WBTC","WETH","STETH","CBETH","RETH",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _ot_fetch_symbols(limit: int) -> list[str]:
    """Holt qualitative USDT-Perp Symbole von Bybit: kein Meme/Scam, min. 1M Vol/Tag."""
    from src.exchange import get_public_exchange
    ex = get_public_exchange()
    try:
        ex.load_markets()
    except Exception:
        return []
    candidates = [
        s for s, m in ex.markets.items()
        if m.get("type") == "swap" and m.get("quote") == "USDT"
           and m.get("active", True) and ":" in s
    ]
    try:
        tickers = ex.fetch_tickers(candidates[:600])
    except Exception:
        tickers = {}

    scored = []
    for s in candidates:
        base = s.split("/")[0].upper()
        if base in _OT_BLACKLIST or base.startswith("1000") or base.startswith("10000"):
            continue
        vol = (tickers.get(s) or {}).get("quoteVolume") or 0.0
        if vol < _OT_MIN_VOL:
            continue
        scored.append((s, vol))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:limit]]


def _ot_csv_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(_OT_DIR, f"{safe}_15m.csv")


def _ot_load_df(symbol: str) -> pd.DataFrame | None:
    p = _ot_csv_path(symbol)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def _ot_score(r: dict) -> float:
    pnl = r.get("total_pnl_pct", 0)
    pf  = min(r.get("profit_factor", 0) or 0, 3.0)
    n   = max(r.get("num_trades", 0), 1)
    return pnl * pf * (n ** 0.5)


def _ot_fast_sweep(
    df:           pd.DataFrame,
    trailing_pct: float,
    pos_size:     float,
    min_trades:   int,
    coin_capital: float,
) -> dict | None:
    """Testet alle Strategie × Hebel-Kombos auf einem Coin (fast). Gibt bestes Ergebnis zurück."""
    opens  = df["open"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)

    best: dict | None = None
    for label, strat, tp, sl in _OT_STRAT_DEFS:
        try:
            signals = strat.generate_signals(df).to_numpy(int)
        except Exception:
            continue
        for lev in _OT_LEVERAGES:
            cfg = StrategyConfig(
                initial_capital   = coin_capital,
                leverage          = lev,
                position_size     = pos_size,
                fee_rate          = _OT_FEE,
                take_profit_pct   = tp / 100,
                stop_loss_pct     = sl / 100,
                trailing_stop_pct = trailing_pct if trailing_pct > 0 else None,
                max_hold_candles  = 1440,
            )
            r = run_strategy_backtest_fast(opens, highs, lows, closes, signals, cfg)
            if not r or r.get("num_trades", 0) < min_trades:
                continue
            r["_label"]    = label
            r["_strategy"] = strat
            r["_lev"]      = lev
            r["_tp"]       = tp
            r["_sl"]       = sl
            r["_cfg"]      = cfg
            r["_score"]    = _ot_score(r)
            if best is None or r["_score"] > best["_score"]:
                best = r
    return best


def _ot_equity_series(df: pd.DataFrame, strat, cfg: StrategyConfig) -> pd.Series:
    """Vollständiger Backtest → zeitindizierte Equity-Serie (nach Trade-Exit)."""
    try:
        result = run_strategy_backtest(df, strat, cfg)
    except Exception:
        return pd.Series(dtype=float)
    trades = result.get("trades", [])
    if not trades:
        return pd.Series(dtype=float)
    idx  = pd.to_datetime([t.exit_time for t in trades], utc=True, errors="coerce")
    vals = [float(t.equity_after) for t in trades]
    return pd.Series(vals, index=idx, dtype=float).dropna()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
if _page == "🔁 Backtest":

    _bt_c1, _bt_c2, _bt_c3 = st.columns(3)

    with _bt_c1:
        st.subheader("Daten & Sequenz")
        selected_csv = st.selectbox("Datei", _csv_opts, key="bt_csv")
        csv_path = os.path.join(RAW_DATA_DIR, selected_csv)
        _bn_c1, _bn_c2 = st.columns(2)
        with _bn_c1:
            n_long  = st.number_input("Long",  min_value=0, max_value=10, value=1, step=1, key="bt_long")
        with _bn_c2:
            n_short = st.number_input("Short", min_value=0, max_value=10, value=1, step=1, key="bt_short")
        sequence = ["long"] * int(n_long) + ["short"] * int(n_short)
        if not sequence:
            st.warning("Mindestens 1 Long oder Short.")
            st.stop()
        st.caption(f"**{' → '.join(sequence)}** (zyklisch)")

    with _bt_c2:
        st.subheader("Exit-Strategie")
        tp_pct   = st.slider("Take Profit (%)", 0.1, 20.0, 2.0, 0.1, format="%.1f%%", key="bt_tp")
        sl_pct   = st.slider("Stop Loss (%)",   0.1, 20.0, 1.0, 0.1, format="%.1f%%", key="bt_sl")
        max_hold = st.number_input("Max. Haltedauer (Kerzen)", 1, 10080, 1440, 60, key="bt_hold",
                                   help="Fallback-Exit nach N Kerzen (1440 = 1 Tag)")

    with _bt_c3:
        st.subheader("Trade & Kapital")
        leverage        = st.slider("Hebel", 1, 50, 10, 1, key="bt_lev")
        position_size   = st.slider("Positionsgroesse (%)", 1, 100, 10, 1, key="bt_size") / 100
        initial_capital = st.number_input("Startkapital (USDT)", 10.0, 1_000_000.0, 1000.0, 100.0, key="bt_cap")
        fee_rate        = st.number_input("Taker-Fee (%)", 0.0, 1.0, 0.055, 0.001,
                                          format="%.4f", key="bt_fee") / 100

    run_btn = st.button("▶  Backtest starten", type="primary", key="bt_run", disabled=_no_csv)
    st.divider()

    if run_btn:
        with st.spinner("Berechne..."):
            df = pd.read_csv(csv_path)
            cfg = BacktestConfig(
                initial_capital  = initial_capital,
                leverage         = leverage,
                position_size    = position_size,
                fee_rate         = fee_rate,
                sequence         = sequence,
                take_profit_pct  = tp_pct / 100,
                stop_loss_pct    = sl_pct / 100,
                max_hold_candles = int(max_hold),
            )
            result = run_backtest(df, cfg)

        if "error" in result:
            st.error(result["error"])
            st.stop()

        trades: list[Trade] = result["trades"]
        equity_curve: list[float] = result["equity_curve"]

        st.subheader("Ergebnis")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        pnl_col = "normal" if result["total_pnl"] >= 0 else "inverse"
        m1.metric("Startkapital",  f"{initial_capital:,.2f} USDT")
        m2.metric("Endkapital",    f"{result['final_balance']:,.2f} USDT",
                  f"{result['total_pnl_pct']:+.2f}%", delta_color=pnl_col)
        m3.metric("Trades",        f"{result['num_trades']}  (TP {result['tp_count']} / SL {result['sl_count']} / TO {result['timeout_count']})")
        m4.metric("Winrate",       f"{result['winrate_pct']:.2f}%")
        m5.metric("Profit Factor", f"{result['profit_factor']:.4f}")
        m6.metric("Max Drawdown",  f"{result['max_drawdown_pct']:.2f}%", delta_color="inverse")

        # Charts
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=False,
            row_heights=[0.50, 0.25, 0.25], vertical_spacing=0.07,
            subplot_titles=("Equity Curve", "PnL pro Trade (USDT)", "BTC Close + Entries"),
        )
        eq_color = "#2ecc71" if equity_curve[-1] >= equity_curve[0] else "#e74c3c"
        fig.add_trace(go.Scatter(x=list(range(len(equity_curve))), y=equity_curve,
            mode="lines", line=dict(color=eq_color, width=2), name="Equity",
            hovertemplate="Trade #%{x}<br>%{y:,.2f} USDT<extra></extra>"), row=1, col=1)
        fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)

        reason_colors = {"tp": "#2ecc71", "sl": "#e74c3c", "timeout": "#f39c12"}
        fig.add_trace(go.Bar(
            x=list(range(len(trades))), y=[t.pnl for t in trades],
            marker_color=[reason_colors.get(t.exit_reason, "#aaa") for t in trades],
            name="PnL", customdata=[[t.exit_reason.upper()] for t in trades],
            hovertemplate="Trade #%{x} [%{customdata[0]}]<br>%{y:+.4f} USDT<extra></extra>"),
            row=2, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(df))), y=df["close"].tolist(),
            mode="lines", line=dict(color="#95a5a6", width=1), name="BTC Close",
            hovertemplate="Kerze %{x}<br>%{y:,.2f}<extra></extra>"), row=3, col=1)

        time_to_idx = {t: i for i, t in enumerate(df["datetime"].astype(str).tolist())}
        lx, ly, sx, sy = [], [], [], []
        for t in trades:
            idx = time_to_idx.get(t.entry_time)
            if idx is None:
                continue
            if t.side == "long":
                lx.append(idx); ly.append(t.entry_price)
            else:
                sx.append(idx); sy.append(t.entry_price)
        if lx:
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers",
                marker=dict(symbol="triangle-up", size=7, color="#2ecc71"),
                name="Long", hovertemplate="%{y:,.2f}<extra>Long</extra>"), row=3, col=1)
        if sx:
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                marker=dict(symbol="triangle-down", size=7, color="#e74c3c"),
                name="Short", hovertemplate="%{y:,.2f}<extra>Short</extra>"), row=3, col=1)

        fig.update_layout(height=700, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=60, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 TP  |  🔴 SL  |  🟠 Timeout")

        with st.expander("Alle Trades anzeigen"):
            tdf = pd.DataFrame([{
                "#": t.index + 1, "Side": t.side.upper(),
                "Entry Time": t.entry_time, "Entry": round(t.entry_price, 2),
                "Exit": round(t.exit_price, 2), "Grund": t.exit_reason.upper(),
                "PnL (USDT)": round(t.pnl, 4), "PnL (%)": round(t.pnl_pct, 3),
                "Equity": round(t.equity_after, 4),
            } for t in trades])
            st.dataframe(
                tdf.style.map(
                    lambda v: "color: #2ecc71" if isinstance(v, (int, float)) and v > 0
                    else ("color: #e74c3c" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["PnL (USDT)", "PnL (%)"],
                ),
                use_container_width=True, hide_index=True)
    else:
        st.info("Parameter oben einstellen und **▶ Backtest starten** klicken.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "🔍 Optimizer":

    st.subheader("🔍 Parameter-Optimizer")
    st.caption("Testet alle Kombinationen und findet die besten Konfigurationen.")

    oc1, oc2 = st.columns([2, 1])

    with oc1:
        st.markdown("**Datei**")
        opt_csv = st.selectbox("CSV", _csv_opts, key="opt_csv", label_visibility="collapsed")
        opt_csv_path = os.path.join(RAW_DATA_DIR, opt_csv)

        st.markdown("**Sequenzen testen**")
        seq_labels = {
            "Long": ["long"],
            "Short": ["short"],
            "Long → Short": ["long", "short"],
            "Short → Long": ["short", "long"],
            "Long Long → Short": ["long", "long", "short"],
            "Short Short → Long": ["short", "short", "long"],
            "Long → Short Short": ["long", "short", "short"],
        }
        selected_seqs = st.multiselect(
            "Sequenzen", list(seq_labels.keys()),
            default=["Long", "Short", "Long → Short", "Short → Long"],
            key="opt_seqs",
        )
        chosen_sequences = [seq_labels[s] for s in selected_seqs]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Hebel-Werte**")
            opt_leverages = st.multiselect("Hebel", [1, 2, 3, 5, 10, 15, 20, 30, 50],
                                           default=[3, 5, 10, 20], key="opt_lev")
            st.markdown("**Take-Profit-Werte (%)**")
            opt_tp = st.multiselect("TP %", [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0],
                                    default=[0.5, 1.0, 2.0, 3.0, 5.0], key="opt_tp")
        with col_b:
            st.markdown("**Positionsgroessen (%)**")
            opt_sizes = st.multiselect("Pos %", [2, 5, 10, 15, 20, 30, 50],
                                       default=[5, 10, 20], key="opt_size")
            st.markdown("**Stop-Loss-Werte (%)**")
            opt_sl = st.multiselect("SL %", [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                                    default=[0.3, 0.5, 1.0, 1.5], key="opt_sl")

    with oc2:
        st.markdown("**Score-Metrik**")
        score_metric = st.radio(
            "Bewertung nach", ["composite", "profit_factor", "total_pnl_pct", "winrate"],
            format_func=lambda x: {
                "composite":      "Composite (PnL × PF × -DD)",
                "profit_factor":  "Profit Factor",
                "total_pnl_pct":  "Gesamt-PnL %",
                "winrate":        "Winrate %",
            }[x],
            key="opt_metric",
        )
        st.markdown("**Filter**")
        min_trades = st.number_input("Mindest-Trades", 1, 200, 10, 1, key="opt_min_trades",
                                     help="Konfigurationen mit weniger Trades werden ignoriert")
        opt_capital = st.number_input("Startkapital (USDT)", 10.0, 1_000_000.0, 1000.0, 100.0, key="opt_cap")
        opt_fee = st.number_input("Taker-Fee (%)", 0.0, 1.0, 0.055, 0.001,
                                  format="%.4f", key="opt_fee") / 100
        top_n = st.slider("Top-N anzeigen", 5, 50, 20, 5, key="opt_topn")

        st.divider()
        # Kombinationsanzahl live berechnen
        n_combos = (len(chosen_sequences) * len(opt_leverages) *
                    len(opt_sizes) * len(opt_tp) * len(opt_sl))
        st.metric("Kombinationen", f"{n_combos:,}")
        est_sec = max(1, int(n_combos * 0.003))
        st.caption(f"Geschaetzte Dauer: ~{est_sec}s")

        opt_btn = st.button("🔍 Optimierung starten", type="primary",
                            use_container_width=True, key="opt_run",
                            disabled=n_combos == 0 or _no_csv)

    if opt_btn:
        if not chosen_sequences or not opt_leverages or not opt_sizes or not opt_tp or not opt_sl:
            st.error("Bitte mindestens einen Wert pro Parameter auswaehlen.")
            st.stop()

        df_opt = pd.read_csv(opt_csv_path)

        opt_cfg = OptimizeConfig(
            sequences       = chosen_sequences,
            leverages       = [int(x) for x in opt_leverages],
            position_sizes  = [x / 100 for x in opt_sizes],
            tp_pcts         = list(opt_tp),
            sl_pcts         = list(opt_sl),
            initial_capital = opt_capital,
            fee_rate        = opt_fee,
            max_hold_candles= 1440,
            min_trades      = int(min_trades),
            score_metric    = score_metric,
        )

        progress_bar = st.progress(0, text="Starte Optimierung…")
        status_text  = st.empty()

        done_counter = [0]

        def update_progress(done: int, total: int) -> None:
            done_counter[0] = done
            pct = done / total
            progress_bar.progress(pct, text=f"{done}/{total} Kombinationen getestet…")

        results = run_optimization(df_opt, opt_cfg, progress_cb=update_progress, workers=1)

        progress_bar.progress(1.0, text="Fertig!")
        profitable = sum(1 for r in results if r["total_pnl_pct"] > 0)
        status_text.success(
            f"✅ {len(results)} Konfigurationen ausgewertet | "
            f"**{profitable} mit positivem PnL** | "
            f"Top-Score: {results[0]['score']:.4f}" if results else "Keine Ergebnisse."
        )

        if not results:
            st.warning("Keine Konfiguration erfuellt die Mindest-Trades-Anforderung.")
            st.stop()

        # Ergebnisse in session_state speichern damit Selectbox-Änderungen
        # keine Neu-Berechnung auslösen
        st.session_state["opt_results"] = results
        st.session_state["opt_df"]      = df_opt
        st.session_state["opt_capital"] = opt_capital
        st.session_state["opt_top_n"]   = top_n

    # ── Ergebnisse anzeigen (auch nach erneutem Rendern ohne Button-Klick) ────
    if "opt_results" in st.session_state:
        results    = st.session_state["opt_results"]
        df_opt     = st.session_state["opt_df"]
        opt_capital= st.session_state["opt_capital"]
        top_n      = st.session_state.get("opt_top_n", 20)

        # ── Ergebnistabelle ────────────────────────────────────────────────────
        st.subheader(f"Top {min(top_n, len(results))} Konfigurationen")
        res_df = format_results_df(results, top_n)

        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:   return "color: #2ecc71"
                if val < 0:   return "color: #e74c3c"
            return ""

        st.dataframe(
            res_df.style
                .map(color_pnl, subset=["PnL %", "Profit Factor", "Score"])
                .format({
                    "PnL %":         "{:+.2f}",
                    "Profit Factor": "{:.4f}",
                    "Winrate %":     "{:.2f}",
                    "Max DD %":      "{:.2f}",
                    "Score":         "{:.4f}",
                }),
            use_container_width=True,
            hide_index=True,
            height=min(35 * min(top_n, len(results)) + 60, 600),
        )

        # ── Grafik-Auswahl ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("Grafik ansehen")

        n_show = min(top_n, len(results))

        def _cfg_label(rank: int) -> str:
            r   = results[rank]
            cfg = r["config"]
            return (f"#{rank+1}  {' → '.join(cfg.sequence)}"
                    f"  |  {cfg.leverage}x"
                    f"  |  TP {cfg.take_profit_pct*100:.1f}% / SL {cfg.stop_loss_pct*100:.1f}%"
                    f"  |  Pos {cfg.position_size*100:.0f}%"
                    f"  |  PnL {r['total_pnl_pct']:+.2f}%"
                    f"  |  PF {r['profit_factor']:.3f}")

        sel_col, btn_col = st.columns([4, 1])
        with sel_col:
            selected_rank = st.selectbox(
                "Konfiguration wählen",
                options=list(range(n_show)),
                format_func=_cfg_label,
                key="opt_sel_rank",
                label_visibility="collapsed",
            )
        with btn_col:
            show_chart_btn = st.button("📈 Grafik laden",
                                       use_container_width=True, key="opt_show_chart")

        if show_chart_btn or st.session_state.get("opt_chart_rank") == selected_rank:
            st.session_state["opt_chart_rank"] = selected_rank
            chosen_cfg = results[selected_rank]["config"]

            with st.spinner("Lade vollständige Trade-Details..."):
                from src.backtester import run_backtest as _rb
                full = _rb(df_opt, chosen_cfg)

            trades_full = full["trades"]
            eq_full     = full["equity_curve"]

            # Header-Info
            st.markdown(
                f"**{_cfg_label(selected_rank)}**  "
                f"&nbsp;·&nbsp; Winrate **{full['winrate_pct']:.1f}%**"
                f"&nbsp;·&nbsp; Trades **{full['num_trades']}**"
                f"&nbsp;·&nbsp; Max DD **{full['max_drawdown_pct']:.2f}%**"
                f"&nbsp;·&nbsp; Sharpe **{full.get('sharpe_ratio', 0):.4f}**"
            )

            # Equity + PnL Chart
            fig_detail = make_subplots(
                rows=2, cols=1, shared_xaxes=False,
                row_heights=[0.65, 0.35], vertical_spacing=0.08,
                subplot_titles=("Equity Curve", "PnL pro Trade"),
            )

            eq_color = "#2ecc71" if eq_full[-1] >= eq_full[0] else "#e74c3c"
            fig_detail.add_trace(
                go.Scatter(x=list(range(len(eq_full))), y=eq_full,
                    mode="lines", line=dict(color=eq_color, width=2), name="Equity",
                    hovertemplate="Trade #%{x}<br>%{y:,.4f} USDT<extra></extra>"),
                row=1, col=1)
            fig_detail.add_hline(y=opt_capital, line_dash="dash",
                                  line_color="gray", line_width=1, row=1, col=1)

            reason_colors = {"tp": "#2ecc71", "sl": "#e74c3c",
                             "signal": "#3498db", "timeout": "#f39c12"}
            fig_detail.add_trace(
                go.Bar(
                    x=list(range(len(trades_full))),
                    y=[t.pnl for t in trades_full],
                    marker_color=[reason_colors.get(t.exit_reason, "#aaa") for t in trades_full],
                    customdata=[[t.exit_reason.upper(), t.side.upper(),
                                 t.entry_time, round(t.entry_price, 2), round(t.exit_price, 2)]
                                for t in trades_full],
                    name="PnL",
                    hovertemplate=(
                        "Trade #%{x}  [%{customdata[0]} / %{customdata[1]}]<br>"
                        "Entry: %{customdata[3]}  →  Exit: %{customdata[4]}<br>"
                        "PnL: %{y:+.4f} USDT<extra></extra>"
                    ),
                ),
                row=2, col=1)

            fig_detail.update_layout(
                height=550, showlegend=False,
                margin=dict(l=50, r=30, t=50, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_detail.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            fig_detail.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            st.plotly_chart(fig_detail, use_container_width=True)
            st.caption("🟢 TP  |  🔴 SL  |  🔵 Signal  |  🟠 Timeout")

            with st.expander("Alle Trades dieser Konfiguration"):
                tdf_detail = pd.DataFrame([{
                    "#":          t.index + 1,
                    "Side":       t.side.upper(),
                    "Entry Time": t.entry_time,
                    "Entry":      round(t.entry_price, 2),
                    "Exit":       round(t.exit_price, 2),
                    "Grund":      t.exit_reason.upper(),
                    "PnL (USDT)": round(t.pnl, 4),
                    "PnL (%)":    round(t.pnl_pct, 3),
                    "Equity":     round(t.equity_after, 4),
                } for t in trades_full])
                st.dataframe(
                    tdf_detail.style.map(
                        lambda v: "color: #2ecc71" if isinstance(v, (int, float)) and v > 0
                        else ("color: #e74c3c" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=["PnL (USDT)", "PnL (%)"],
                    ),
                    use_container_width=True, hide_index=True,
                )

        # ── Beste Konfiguration Info ───────────────────────────────────────────
        st.divider()
        best = results[0]["config"]
        st.info(
            f"**Beste Konfiguration:**  "
            f"{' → '.join(best.sequence)}  |  {best.leverage}x Hebel  |  "
            f"TP {best.take_profit_pct*100:.1f}%  |  SL {best.stop_loss_pct*100:.1f}%  |  "
            f"Pos-Groesse {best.position_size*100:.0f}%  →  "
            f"PnL **{results[0]['total_pnl_pct']:+.2f}%**"
        )

    else:
        st.info("Parameter auswaehlen und **🔍 Optimierung starten** klicken.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – STRATEGIEN
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "📈 Strategien":

    st.subheader("📈 Strategie backtesten")

    sc1, sc2 = st.columns([1.3, 0.7])

    with sc1:
        ALL_STRAT_NAMES = ["EMA Cross", "RSI", "Bollinger", "Breakout", "MACD", "Supertrend", "Kombiniert",
                           "MeanRev (BB+ADX Ranging)", "TrendFollow (EMA+ADX Trending)"]
        strat_name = st.selectbox("Strategie", ALL_STRAT_NAMES, key="st_name")

        st.markdown("**Strategie-Parameter**")
        if strat_name == "EMA Cross":
            st_fast = st.number_input("Fast EMA", 2, 500, 20, key="st_ema_fast")
            st_slow = st.number_input("Slow EMA", 3, 1000, 50, key="st_ema_slow")
        elif strat_name == "RSI":
            st_period     = st.number_input("RSI Period", 2, 200, 14, key="st_rsi_p")
            st_oversold   = st.slider("Oversold",   1, 49, 30, key="st_rsi_os")
            st_overbought = st.slider("Overbought", 51, 99, 70, key="st_rsi_ob")
        elif strat_name == "Bollinger":
            st_period = st.number_input("Period", 2, 500, 20, key="st_bb_p")
            st_stddev = st.slider("Std Dev", 0.5, 5.0, 2.0, 0.1, key="st_bb_std")
        elif strat_name == "Breakout":
            st_lookback = st.number_input("Lookback", 2, 1000, 50, key="st_bo_lb")
        elif strat_name == "MACD":
            st_macd_fast = st.number_input("Fast EMA", 2, 100, 12, key="st_macd_fast")
            st_macd_slow = st.number_input("Slow EMA", 3, 200, 26, key="st_macd_slow")
            st_macd_sig  = st.number_input("Signal",   2, 50,   9, key="st_macd_sig")
        elif strat_name == "Supertrend":
            st_st_atr  = st.number_input("ATR Periode", 2, 100, 10, key="st_st_atr")
            st_st_mult = st.slider("Multiplikator", 1.0, 6.0, 3.0, 0.1, key="st_st_mult")
        elif strat_name == "MeanRev (BB+ADX Ranging)":
            _m1, _m2, _m3 = st.columns(3)
            mr_bb_p   = _m1.number_input("BB Period", 2, 500, 10, key="mr_bb_p")
            mr_bb_std = _m2.number_input("BB Std Dev", 0.5, 5.0, 2.0, step=0.1, key="mr_bb_std")
            mr_adx_th = _m3.number_input("ADX Schwelle (<)", 5, 50, 20, key="mr_adx_th")
            st.caption("Signale nur wenn ADX < Schwelle (Ranging-Markt). Optimales Setup: BB10/2.0σ, ADX<20, ATR×3.0, RR2.0.")
        elif strat_name == "TrendFollow (EMA+ADX Trending)":
            _t1, _t2, _t3 = st.columns(3)
            tf_fast   = _t1.number_input("Fast EMA", 2, 500, 20, key="tf_fast")
            tf_slow   = _t2.number_input("Slow EMA", 3, 1000, 100, key="tf_slow")
            tf_adx_th = _t3.number_input("ADX Schwelle (≥)", 5, 50, 25, key="tf_adx_th")
            st.caption("Signale nur wenn ADX ≥ Schwelle (Trending-Markt). Optimales Setup: EMA20/100, ADX≥25, ATR×3.0, RR3.0.")
        else:  # Kombiniert
            comb_logic = st.radio("Logik", ["AND", "OR", "MAJORITY"], horizontal=True, key="comb_logic",
                help="AND: alle stimmen überein | OR: mindestens eine | MAJORITY: Mehrheit")
            st.markdown("**Strategien aktivieren**")
            c_ema_on  = st.checkbox("EMA Cross",  value=True,  key="comb_ema_on")
            if c_ema_on:
                _c1, _c2 = st.columns(2)
                comb_ema_fast = _c1.number_input("Fast", 2, 500, 20,   key="comb_ema_fast")
                comb_ema_slow = _c2.number_input("Slow", 3, 1000, 50,  key="comb_ema_slow")
            c_rsi_on  = st.checkbox("RSI",        value=True,  key="comb_rsi_on")
            if c_rsi_on:
                _c1, _c2, _c3 = st.columns(3)
                comb_rsi_p  = _c1.number_input("Period", 2, 200, 14, key="comb_rsi_p")
                comb_rsi_os = _c2.number_input("OS",     1, 49,  30, key="comb_rsi_os")
                comb_rsi_ob = _c3.number_input("OB",    51, 99,  70, key="comb_rsi_ob")
            c_bb_on   = st.checkbox("Bollinger",  value=False, key="comb_bb_on")
            if c_bb_on:
                _c1, _c2 = st.columns(2)
                comb_bb_p   = _c1.number_input("Period", 2, 500, 20,  key="comb_bb_p")
                comb_bb_std = _c2.number_input("Std",  0.5, 5.0, 2.0, key="comb_bb_std")
            c_macd_on = st.checkbox("MACD",       value=False, key="comb_macd_on")
            if c_macd_on:
                _c1, _c2, _c3 = st.columns(3)
                comb_macd_f  = _c1.number_input("Fast",   2, 100, 12, key="comb_macd_f")
                comb_macd_s  = _c2.number_input("Slow",   3, 200, 26, key="comb_macd_s")
                comb_macd_si = _c3.number_input("Signal", 2,  50,  9, key="comb_macd_si")
            c_st_on   = st.checkbox("Supertrend", value=False, key="comb_st_on")
            if c_st_on:
                _c1, _c2 = st.columns(2)
                comb_st_atr  = _c1.number_input("ATR",  2, 100, 10,  key="comb_st_atr")
                comb_st_mult = _c2.number_input("Mult", 1.0, 6.0, 3.0, key="comb_st_mult")
            c_bo_on   = st.checkbox("Breakout",   value=False, key="comb_bo_on")
            if c_bo_on:
                comb_bo_lb = st.number_input("Lookback", 2, 1000, 50, key="comb_bo_lb")

        st.divider()
        st.markdown("**Exit-Strategie**")
        st_tp = st.slider("Take Profit (%)", 0.0, 20.0, 2.0, 0.1, format="%.1f%%", key="st_tp",
                          help="0 = kein TP")
        st_sl = st.slider("Stop Loss (%)",   0.0, 20.0, 1.0, 0.1, format="%.1f%%", key="st_sl",
                          help="0 = kein SL")
        st_exit_signal = st.checkbox("Exit bei Gegenrichtungs-Signal", value=True, key="st_sig")
        st_max_hold    = st.number_input("Max. Haltedauer (Kerzen)", 1, 10080, 1440, key="st_hold")

        st.markdown("**Optionale Features**")
        _f1, _f2 = st.columns(2)
        with _f1:
            use_trail = st.checkbox("Trailing Stop", key="st_use_trail")
            trail_pct = (st.slider("Trail %", 0.1, 10.0, 1.0, 0.1, key="st_trail_pct") / 100
                         if use_trail else None)
            use_vf    = st.checkbox("Volume-Filter", key="st_use_vf")
            vf_period = (int(st.number_input("VF Periode", 5, 200, 20, key="st_vf_period"))
                         if use_vf else None)
        with _f2:
            use_cb    = st.checkbox("Circuit Breaker", key="st_use_cb",
                                    help="Stoppt Handel wenn Equity zu weit fällt")
            cb_pct    = (st.slider("Stopp bei -%", 5, 80, 30, key="st_cb_pct") / 100
                         if use_cb else None)

    with sc2:
        st.markdown("**Trade-Einstellungen**")
        st_csv     = st.selectbox("Datei", _csv_opts, key="st_csv")
        st_lev     = st.slider("Hebel", 1, 50, 10, key="st_lev")
        st_size    = st.slider("Positionsgroesse (%)", 1, 100, 10, key="st_size") / 100
        st_capital = st.number_input("Startkapital (USDT)", 10.0, 1_000_000.0, 1000.0, 100.0, key="st_cap")
        st_fee     = st.number_input("Taker-Fee (%)", 0.0, 1.0, 0.055, 0.001,
                                     format="%.4f", key="st_fee") / 100
        use_max_not = st.checkbox("Max. Notional pro Trade", key="st_use_maxnot",
                                   help="Begrenzt die Positionsgröße unabhängig vom Kapitalwachstum")
        max_not_val = (st.number_input("Max. Notional (USDT)", 10.0, 1_000_000.0, 1000.0, 100.0,
                                        key="st_maxnot")
                       if use_max_not else None)
        st.markdown("**Export**")
        st_export = st.checkbox("Trades als CSV speichern", key="st_exp")
        st_chart  = st.checkbox("Equity-Chart als PNG speichern", key="st_cht")
        st.markdown("&nbsp;")
        st_run = st.button("▶  Backtest starten", type="primary",
                           use_container_width=True, key="st_run",
                           disabled=_no_csv)

    if st_run:
        try:
            if strat_name == "EMA Cross":
                strategy = EMACrossStrategy(fast_period=int(st_fast), slow_period=int(st_slow))
            elif strat_name == "RSI":
                strategy = RSIStrategy(int(st_period), float(st_oversold), float(st_overbought))
            elif strat_name == "Bollinger":
                strategy = BollingerStrategy(int(st_period), float(st_stddev))
            elif strat_name == "Breakout":
                strategy = BreakoutStrategy(int(st_lookback))
            elif strat_name == "MACD":
                strategy = MACDStrategy(int(st_macd_fast), int(st_macd_slow), int(st_macd_sig))
            elif strat_name == "Supertrend":
                strategy = SupertrendStrategy(int(st_st_atr), float(st_st_mult))
            elif strat_name == "MeanRev (BB+ADX Ranging)":
                strategy = MeanRevStrategy(int(mr_bb_p), float(mr_bb_std), float(mr_adx_th))
            elif strat_name == "TrendFollow (EMA+ADX Trending)":
                strategy = TrendFollowStrategy(int(tf_fast), int(tf_slow), float(tf_adx_th))
            else:  # Kombiniert
                sub_strats = []
                if c_ema_on:  sub_strats.append(EMACrossStrategy(int(comb_ema_fast), int(comb_ema_slow)))
                if c_rsi_on:  sub_strats.append(RSIStrategy(int(comb_rsi_p), float(comb_rsi_os), float(comb_rsi_ob)))
                if c_bb_on:   sub_strats.append(BollingerStrategy(int(comb_bb_p), float(comb_bb_std)))
                if c_macd_on: sub_strats.append(MACDStrategy(int(comb_macd_f), int(comb_macd_s), int(comb_macd_si)))
                if c_st_on:   sub_strats.append(SupertrendStrategy(int(comb_st_atr), float(comb_st_mult)))
                if c_bo_on:   sub_strats.append(BreakoutStrategy(int(comb_bo_lb)))
                if len(sub_strats) < 2:
                    st.error("Mindestens 2 Strategien aktivieren.")
                    st.stop()
                strategy = CombinedStrategy(sub_strats, comb_logic)
        except ValueError as e:
            st.error(str(e)); st.stop()

        st_cfg = StrategyConfig(
            initial_capital      = st_capital,
            leverage             = st_lev,
            position_size        = st_size,
            fee_rate             = st_fee,
            take_profit_pct      = st_tp / 100 if st_tp > 0 else None,
            stop_loss_pct        = st_sl / 100 if st_sl > 0 else None,
            trailing_stop_pct    = trail_pct,
            exit_on_signal       = st_exit_signal,
            max_hold_candles     = int(st_max_hold),
            volume_filter_period = vf_period,
            circuit_breaker_pct  = cb_pct,
            max_notional         = max_not_val,
        )

        with st.spinner("Berechne Signale und Trades..."):
            df_st = pd.read_csv(os.path.join(RAW_DATA_DIR, st_csv))
            res   = run_strategy_backtest(df_st, strategy, st_cfg)

        if "error" in res:
            st.error(res["error"]); st.stop()

        res = extended_metrics(res)

        st.subheader(f"Ergebnis: {strategy}")
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        pnl_col = "normal" if res["total_pnl"] >= 0 else "inverse"
        m1.metric("Startkapital",  f"{st_capital:,.0f}")
        m2.metric("Endkapital",    f"{res['final_balance']:,.2f}",
                  f"{res['total_pnl_pct']:+.2f}%", delta_color=pnl_col)
        m3.metric("Trades",        res["num_trades"])
        m4.metric("Winrate",       f"{res['winrate_pct']:.1f}%")
        m5.metric("Profit Factor", f"{res['profit_factor']:.4f}")
        m6.metric("Sharpe",        f"{res.get('sharpe_ratio', 0):.4f}")
        m7.metric("Max Drawdown",  f"{res['max_drawdown_pct']:.2f}%", delta_color="inverse")

        try:
            _t0 = pd.to_datetime(df_st["datetime"].iloc[0])
            _t1 = pd.to_datetime(df_st["datetime"].iloc[-1])
            _hours = (_t1 - _t0).total_seconds() / 3600
            _trades_per_h = res["num_trades"] / _hours if _hours > 0 else 0
        except Exception:
            _trades_per_h = 0

        em1, em2, em3, em4, em5, em6 = st.columns(6)
        em1.metric("Sortino",          f"{res.get('sortino_ratio', 0):.4f}")
        em2.metric("Calmar",           f"{res.get('calmar_ratio',  0):.4f}")
        em3.metric("Max Win-Serie",    res.get("max_win_streak",  0))
        em4.metric("Max Loss-Serie",   res.get("max_loss_streak", 0))
        em5.metric("TP / SL / TO",     f"{res.get('tp_count',0)} / {res.get('sl_count',0)} / {res.get('timeout_count',0)}")
        em6.metric("Trades / Stunde",  f"{_trades_per_h:.3f}")

        trades_st      = res["trades"]
        eq_curve_st    = res["equity_curve"]
        signals_series = res.get("signals", pd.Series(dtype=int))

        # ── Indikatoren für den Candlestick-Chart ─────────────────────────────
        df_ind_st = df_st.copy()
        try:
            df_ind_st["datetime"] = pd.to_datetime(df_ind_st["datetime"], utc=True, errors="coerce")
            df_ind_st = df_ind_st.dropna(subset=["datetime"]).reset_index(drop=True)
        except Exception:
            pass

        _is_mr  = strat_name == "MeanRev (BB+ADX Ranging)"
        _is_tf  = strat_name == "TrendFollow (EMA+ADX Trending)"
        _is_bb  = strat_name == "Bollinger"

        if _is_mr or _is_bb:
            _bp  = mr_bb_p  if _is_mr else st_period
            _bsd = mr_bb_std if _is_mr else st_stddev
            _bm  = df_ind_st["close"].rolling(int(_bp)).mean()
            _bs  = df_ind_st["close"].rolling(int(_bp)).std(ddof=0)
            df_ind_st["bb_upper"] = _bm + float(_bsd) * _bs
            df_ind_st["bb_mid"]   = _bm
            df_ind_st["bb_lower"] = _bm - float(_bsd) * _bs

        if _is_tf:
            df_ind_st["ema_fast"] = df_ind_st["close"].ewm(span=int(tf_fast), adjust=False).mean()
            df_ind_st["ema_slow"] = df_ind_st["close"].ewm(span=int(tf_slow), adjust=False).mean()
        elif strat_name == "EMA Cross":
            df_ind_st["ema_fast"] = df_ind_st["close"].ewm(span=int(st_fast), adjust=False).mean()
            df_ind_st["ema_slow"] = df_ind_st["close"].ewm(span=int(st_slow), adjust=False).mean()

        _has_dt = "datetime" in df_ind_st.columns and pd.api.types.is_datetime64_any_dtype(df_ind_st["datetime"])
        _x_cs   = df_ind_st["datetime"] if _has_dt else list(range(len(df_ind_st)))

        fig_st = make_subplots(rows=3, cols=1,
            shared_xaxes=True if _has_dt else False,
            row_heights=[0.55, 0.20, 0.25], vertical_spacing=0.04,
            subplot_titles=(f"Candlestick  ·  {strategy}", "PnL pro Trade", "Equity Curve"))
        eq_c = "#2ecc71" if eq_curve_st[-1] >= eq_curve_st[0] else "#e74c3c"

        # ─ Panel 1: Candlestick ───────────────────────────────────────────────
        fig_st.add_trace(go.Candlestick(
            x=_x_cs,
            open=df_ind_st["open"], high=df_ind_st["high"],
            low=df_ind_st["low"],   close=df_ind_st["close"],
            name="OHLC",
            increasing_line_color="#2ecc71", decreasing_line_color="#e74c3c",
            showlegend=False,
        ), row=1, col=1)

        if _is_mr or _is_bb:
            fig_st.add_trace(go.Scatter(x=_x_cs, y=df_ind_st["bb_upper"],
                mode="lines", name="BB Oben",
                line=dict(color="rgba(52,152,219,0.65)", width=1, dash="dot")), row=1, col=1)
            fig_st.add_trace(go.Scatter(x=_x_cs, y=df_ind_st["bb_lower"],
                mode="lines", name="BB Unten",
                line=dict(color="rgba(52,152,219,0.65)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(52,152,219,0.05)", showlegend=False), row=1, col=1)
            fig_st.add_trace(go.Scatter(x=_x_cs, y=df_ind_st["bb_mid"],
                mode="lines", name="BB Mitte",
                line=dict(color="rgba(52,152,219,0.4)", width=1)), row=1, col=1)

        if _is_tf or strat_name == "EMA Cross":
            fig_st.add_trace(go.Scatter(x=_x_cs, y=df_ind_st["ema_fast"],
                mode="lines", name=f"EMA Fast",
                line=dict(color="rgba(241,196,15,0.9)", width=1.5)), row=1, col=1)
            fig_st.add_trace(go.Scatter(x=_x_cs, y=df_ind_st["ema_slow"],
                mode="lines", name=f"EMA Slow",
                line=dict(color="rgba(155,89,182,0.9)", width=1.5)), row=1, col=1)

        # Trade-Marker auf Panel 1
        _dt_map = {str(v): v for v in df_ind_st["datetime"]} if _has_dt else {}
        _idx_map = {str(v): i for i, v in enumerate(df_ind_st["datetime"] if _has_dt else range(len(df_ind_st)))}
        lx_cs, ly_cs, sx_cs, sy_cs = [], [], [], []
        lx_ex, ly_ex, sx_ex, sy_ex = [], [], [], []
        l_hover_cs, s_hover_cs = [], []
        for t in trades_st:
            _et = _dt_map.get(str(t.entry_time)) if _has_dt else _idx_map.get(str(t.entry_time))
            _xt = _dt_map.get(str(getattr(t, "exit_time", None))) if _has_dt else _idx_map.get(str(getattr(t, "exit_time", None)))
            if _et is None:
                _ei = _idx_map.get(str(t.entry_time))
                _et = _ei
            _ht = (f"Entry: {t.entry_price:,.4f}<br>"
                   f"Exit:  {t.exit_price:,.4f} ({t.exit_reason.upper()})<br>"
                   f"PnL:   {t.pnl_pct:+.2f}%")
            if t.side == "long":
                lx_cs.append(_et); ly_cs.append(t.entry_price); l_hover_cs.append(_ht)
                if _xt: lx_ex.append(_xt); ly_ex.append(t.exit_price)
            else:
                sx_cs.append(_et); sy_cs.append(t.entry_price); s_hover_cs.append(_ht)
                if _xt: sx_ex.append(_xt); sy_ex.append(t.exit_price)

        if lx_cs:
            fig_st.add_trace(go.Scatter(x=lx_cs, y=ly_cs, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#2ecc71",
                            line=dict(color="white", width=1)),
                name="Long Entry", hovertext=l_hover_cs, hoverinfo="text"), row=1, col=1)
        if sx_cs:
            fig_st.add_trace(go.Scatter(x=sx_cs, y=sy_cs, mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#e74c3c",
                            line=dict(color="white", width=1)),
                name="Short Entry", hovertext=s_hover_cs, hoverinfo="text"), row=1, col=1)
        if lx_ex:
            fig_st.add_trace(go.Scatter(x=lx_ex, y=ly_ex, mode="markers",
                marker=dict(symbol="x", size=8, color="rgba(46,204,113,0.6)"),
                name="Long Exit", showlegend=False), row=1, col=1)
        if sx_ex:
            fig_st.add_trace(go.Scatter(x=sx_ex, y=sy_ex, mode="markers",
                marker=dict(symbol="x", size=8, color="rgba(231,76,60,0.6)"),
                name="Short Exit", showlegend=False), row=1, col=1)

        # ─ Panel 2: PnL pro Trade ─────────────────────────────────────────────
        reason_c = {"tp": "#2ecc71", "sl": "#e74c3c", "signal": "#3498db", "timeout": "#f39c12"}
        fig_st.add_trace(go.Bar(
            x=list(range(len(trades_st))), y=[t.pnl for t in trades_st],
            marker_color=[reason_c.get(t.exit_reason, "#aaa") for t in trades_st],
            customdata=[[t.exit_reason.upper()] for t in trades_st], name="PnL",
            hovertemplate="Trade #%{x} [%{customdata[0]}]<br>%{y:+.4f}<extra></extra>"),
            row=2, col=1)

        # ─ Panel 3: Equity ────────────────────────────────────────────────────
        fig_st.add_trace(go.Scatter(x=list(range(len(eq_curve_st))), y=eq_curve_st,
            mode="lines", line=dict(color=eq_c, width=2), name="Equity"), row=3, col=1)
        fig_st.add_hline(y=st_capital, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)

        fig_st.update_layout(height=900, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=60, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False)
        fig_st.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", rangeslider_visible=False)
        fig_st.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        st.plotly_chart(fig_st, use_container_width=True)
        st.caption("▲ Long Entry  |  ▼ Short Entry  |  × Exit  ·  🟢 TP  |  🔴 SL  |  🔵 Signal-Exit  |  🟠 Timeout")

        # Monatliche Auswertung
        monthly_df = res.get("monthly_pnl", pd.DataFrame())
        if not monthly_df.empty:
            with st.expander("📅 Monatliche PnL-Auswertung"):
                fig_mo = go.Figure(go.Bar(
                    x=monthly_df["Monat"], y=monthly_df["PnL_USDT"],
                    marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly_df["PnL_USDT"]],
                    text=[f"{v:+.2f}" for v in monthly_df["PnL_USDT"]], textposition="outside",
                ))
                fig_mo.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"))
                st.plotly_chart(fig_mo, use_container_width=True)
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        # Walk-Forward
        with st.expander("🔀 Walk-Forward Analyse (Overfitting-Test)"):
            st.caption("Optimale Strategie nur auf Trainingsdaten → Test auf ungesehenen Daten.")
            wf_ratio = st.slider("Trainings-Anteil", 50, 90, 70, 5, key="st_wf_ratio") / 100
            wf_run   = st.button("🔀 Walk-Forward starten", key="st_wf_run")
            if wf_run:
                with st.spinner("Berechne Train/Test-Split..."):
                    try:
                        wf   = simple_split(df_st, strategy, st_cfg, train_ratio=wf_ratio)
                        r_tr = wf["train"];  r_te = wf["test"]
                        st.caption(f"Split: **{wf['split_date']}**  |  "
                                   f"Train {wf['train_candles']:,} Kerzen  |  "
                                   f"Test {wf['test_candles']:,} Kerzen")
                        wf_c1, wf_c2 = st.columns(2)
                        for wf_col, r, label in [(wf_c1, r_tr, "🔵 Training"), (wf_c2, r_te, "🟠 Test")]:
                            with wf_col:
                                st.markdown(f"**{label}**")
                                dc = "normal" if r.get("total_pnl_pct", 0) >= 0 else "inverse"
                                st.metric("PnL %",         f"{r.get('total_pnl_pct',0):+.2f}%", delta_color=dc)
                                st.metric("Trades",        r.get("num_trades", 0))
                                st.metric("Winrate",       f"{r.get('winrate_pct',0):.1f}%")
                                st.metric("Profit Factor", f"{r.get('profit_factor',0):.4f}")
                                st.metric("Max DD %",      f"{r.get('max_drawdown_pct',0):.2f}%")
                        fig_wf = go.Figure()
                        for ec, name, col in [
                            (r_tr.get("equity_curve",[]), "Training", "#3498db"),
                            (r_te.get("equity_curve",[]), "Test",     "#f39c12"),
                        ]:
                            fig_wf.add_trace(go.Scatter(x=list(range(len(ec))), y=ec,
                                mode="lines", name=name, line=dict(color=col, width=2)))
                        fig_wf.add_hline(y=st_capital, line_dash="dash", line_color="gray")
                        fig_wf.update_layout(height=280, showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=40, r=20, t=10, b=30),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_wf, use_container_width=True)
                        gap = r_te.get("total_pnl_pct",0) - r_tr.get("total_pnl_pct",0)
                        if gap > -15:
                            st.success(f"✅ Kein starkes Overfitting (Gap: {gap:+.2f}%)")
                        else:
                            st.warning(f"⚠️ Mögliches Overfitting: Test {gap:+.2f}% schlechter als Training")
                    except Exception as e:
                        st.error(str(e))

        with st.expander("Alle Trades"):
            tdf = pd.DataFrame([{
                "#": t.index+1, "Side": t.side.upper(),
                "Entry Time": t.entry_time, "Entry": round(t.entry_price, 2),
                "Exit": round(t.exit_price, 2), "Grund": t.exit_reason.upper(),
                "PnL (USDT)": round(t.pnl, 4), "PnL (%)": round(t.pnl_pct, 3),
                "Equity": round(t.equity_after, 4),
            } for t in trades_st])
            st.dataframe(tdf.style.map(
                lambda v: "color: #2ecc71" if isinstance(v, (int, float)) and v > 0
                else ("color: #e74c3c" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["PnL (USDT)", "PnL (%)"]),
                use_container_width=True, hide_index=True)

        stem = f"{strategy.params_str()}_{st_lev}x"
        if st_export:
            p = export_trades_csv(res, stem); st.success(f"CSV gespeichert: `{p}`")
        if st_chart:
            title = f"{strategy}  {st_lev}x"
            p = save_equity_chart(res, title, stem)
            st.success(f"Chart gespeichert: `{p}`")
    else:
        st.info("Strategie und Parameter wählen, dann **▶ Backtest starten**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MULTI-COIN VERGLEICH
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "📊 Vergleich":

    st.subheader("📊 Multi-Coin Vergleich")
    st.caption("Eine Strategie auf mehreren Coins / Datensätzen gleichzeitig testen.")

    cv1, cv2 = st.columns([2, 1])

    with cv1:
        comp_csvs = st.multiselect("CSV-Dateien (mehrere wählbar)", csv_files,
                                   default=csv_files[:min(3, len(csv_files))], key="cv_csvs",
                                   disabled=_no_csv)
        comp_strat_name = st.selectbox("Strategie", list(STRATEGY_REGISTRY.keys()), key="cv_strat_name")

        st.markdown("**Strategie-Parameter**")
        if comp_strat_name == "EMA Cross":
            cv_ema_fast = st.number_input("Fast EMA", 2, 500, 20, key="cv_ema_fast")
            cv_ema_slow = st.number_input("Slow EMA", 3, 1000, 50, key="cv_ema_slow")
        elif comp_strat_name == "RSI":
            cv_rsi_p  = st.number_input("RSI Period", 2, 200, 14, key="cv_rsi_p")
            cv_rsi_os = st.slider("Oversold",   1, 49, 30, key="cv_rsi_os")
            cv_rsi_ob = st.slider("Overbought", 51, 99, 70, key="cv_rsi_ob")
        elif comp_strat_name == "Bollinger":
            cv_bb_p   = st.number_input("Period", 2, 500, 20, key="cv_bb_p")
            cv_bb_std = st.slider("Std Dev", 0.5, 5.0, 2.0, 0.1, key="cv_bb_std")
        elif comp_strat_name == "Breakout":
            cv_bo_lb  = st.number_input("Lookback", 2, 1000, 50, key="cv_bo_lb")
        elif comp_strat_name == "MACD":
            cv_macd_f  = st.number_input("Fast", 2, 100, 12, key="cv_macd_f")
            cv_macd_s  = st.number_input("Slow", 3, 200, 26, key="cv_macd_s")
            cv_macd_si = st.number_input("Signal", 2, 50, 9,  key="cv_macd_si")
        elif comp_strat_name == "Supertrend":
            cv_st_atr  = st.number_input("ATR", 2, 100, 10, key="cv_st_atr")
            cv_st_mult = st.slider("Mult", 1.0, 6.0, 3.0, 0.1, key="cv_st_mult")

    with cv2:
        comp_lev    = st.slider("Hebel",             1, 50, 10, key="cv_lev")
        comp_size   = st.slider("Positionsgroesse (%)", 1, 100, 10, key="cv_size") / 100
        comp_tp     = st.slider("Take Profit (%)",   0.0, 20.0, 2.0, 0.1, key="cv_tp")
        comp_sl     = st.slider("Stop Loss (%)",     0.0, 20.0, 1.0, 0.1, key="cv_sl")
        comp_cap    = st.number_input("Kapital (USDT)", 10.0, 1_000_000.0, 1000.0, 100.0, key="cv_cap")
        comp_charts = st.checkbox("Charts speichern (PNG)", key="cv_charts")
        comp_run    = st.button("▶  Vergleich starten", type="primary",
                                use_container_width=True, key="cv_run",
                                disabled=len(comp_csvs) == 0 or _no_csv)

    if comp_run:
        try:
            if comp_strat_name == "EMA Cross":
                comp_strategy = EMACrossStrategy(int(cv_ema_fast), int(cv_ema_slow))
            elif comp_strat_name == "RSI":
                comp_strategy = RSIStrategy(int(cv_rsi_p), float(cv_rsi_os), float(cv_rsi_ob))
            elif comp_strat_name == "Bollinger":
                comp_strategy = BollingerStrategy(int(cv_bb_p), float(cv_bb_std))
            elif comp_strat_name == "Breakout":
                comp_strategy = BreakoutStrategy(int(cv_bo_lb))
            elif comp_strat_name == "MACD":
                comp_strategy = MACDStrategy(int(cv_macd_f), int(cv_macd_s), int(cv_macd_si))
            else:
                comp_strategy = SupertrendStrategy(int(cv_st_atr), float(cv_st_mult))
        except ValueError as e:
            st.error(str(e)); st.stop()

        comp_cfg = StrategyConfig(
            initial_capital = comp_cap,
            leverage        = comp_lev,
            position_size   = comp_size,
            fee_rate        = 0.00055,
            take_profit_pct = comp_tp / 100 if comp_tp > 0 else None,
            stop_loss_pct   = comp_sl / 100 if comp_sl > 0 else None,
        )

        prog      = st.progress(0, text="Berechne Coins...")
        mc_res: dict[str, dict] = {}

        for idx, csv_f in enumerate(comp_csvs):
            label = csv_f.replace(".csv", "")
            df_c  = pd.read_csv(os.path.join(RAW_DATA_DIR, csv_f))
            r     = run_strategy_backtest(df_c, comp_strategy, comp_cfg)
            r     = extended_metrics(r)
            mc_res[label] = r
            prog.progress((idx + 1) / len(comp_csvs), text=f"{label} fertig")
            if comp_charts and "error" not in r:
                save_equity_chart(r, f"{comp_strategy}  {comp_lev}x  {label}",
                                  f"{comp_strategy.params_str()}_{label}_{comp_lev}x")
        prog.progress(1.0, text="Fertig!")

        mc_rows = []
        for coin, r in mc_res.items():
            if "error" in r: continue
            mc_rows.append({
                "Coin / Datei":   coin,
                "PnL %":          round(r.get("total_pnl_pct", 0), 2),
                "Final USDT":     round(r.get("final_balance", comp_cap), 2),
                "Trades":         r.get("num_trades", 0),
                "Winrate %":      round(r.get("winrate_pct", 0), 1),
                "Profit Factor":  round(r.get("profit_factor", 0), 4),
                "Sharpe":         round(r.get("sharpe_ratio", 0), 4),
                "Sortino":        round(r.get("sortino_ratio", 0), 4),
                "Calmar":         round(r.get("calmar_ratio", 0), 4),
                "Max DD %":       round(r.get("max_drawdown_pct", 0), 2),
                "Win-Serie":      r.get("max_win_streak", 0),
                "Loss-Serie":     r.get("max_loss_streak", 0),
                "TP / SL / TO":   f"{r.get('tp_count',0)}/{r.get('sl_count',0)}/{r.get('timeout_count',0)}",
            })

        mc_df = pd.DataFrame(mc_rows).sort_values("PnL %", ascending=False)
        st.subheader(f"Ergebnisse: {comp_strategy}  |  {comp_lev}x  |  TP {comp_tp:.1f}%  SL {comp_sl:.1f}%")

        def _color_mc(v):
            if isinstance(v, (int, float)):
                if v > 0: return "color: #2ecc71"
                if v < 0: return "color: #e74c3c"
            return ""

        st.dataframe(
            mc_df.style
                .map(_color_mc, subset=["PnL %", "Profit Factor", "Sharpe", "Sortino", "Calmar"])
                .format({"PnL %": "{:+.2f}", "Final USDT": "{:,.2f}",
                         "Profit Factor": "{:.4f}", "Sharpe": "{:.4f}",
                         "Sortino": "{:.4f}", "Calmar": "{:.4f}",
                         "Winrate %": "{:.1f}", "Max DD %": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

        # Equity Curves
        st.subheader("Equity Curves Overlay")
        colors_mc = ["#2ecc71","#3498db","#f39c12","#e74c3c","#9b59b6",
                     "#1abc9c","#e67e22","#95a5a6","#34495e","#c0392b"]
        fig_mc = go.Figure()
        for idx, (coin, r) in enumerate(mc_res.items()):
            if "error" in r: continue
            ec    = r.get("equity_curve", [])
            label = f"{coin}  ({r.get('total_pnl_pct',0):+.1f}%)"
            fig_mc.add_trace(go.Scatter(
                x=list(range(len(ec))), y=ec, mode="lines", name=label,
                line=dict(color=colors_mc[idx % len(colors_mc)], width=2),
            ))
        fig_mc.add_hline(y=comp_cap, line_dash="dash", line_color="gray", line_width=1)
        fig_mc.update_layout(height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Trade #", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
            yaxis=dict(title="Equity (USDT)", showgrid=True, gridcolor="rgba(128,128,128,0.15)"))
        st.plotly_chart(fig_mc, use_container_width=True)

        if comp_charts:
            st.success("Charts gespeichert unter `data/charts/`")
    else:
        st.info("CSV-Dateien und Strategie wählen, dann **▶ Vergleich starten**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – STRATEGIE-OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "🔬 Strategie-Optimizer":

    st.subheader("🔬 Strategie-Optimizer")
    st.caption("Testet alle Kombinationen aus Strategie-Parametern × Hebel × TP/SL.")

    so1, so2, so3 = st.columns([1, 1, 1])

    with so1:
        st.markdown("**Strategie (Scalping-Fokus)**")
        so_strategy = st.selectbox(
            "Strategie wählen",
            ["EMA Cross", "RSI Divergence", "Supertrend", "RSI", "Bollinger", "Breakout", "MACD"],
            key="so_strat", label_visibility="collapsed",
        )
        so_csv = st.selectbox("Primär-CSV (Signal-TF)", _csv_opts, key="so_csv")

        st.markdown("**Strategie-Parameter (Suchraum)**")
        if so_strategy == "EMA Cross":
            so_fast = st.multiselect("Fast EMA Perioden",
                [3, 5, 8, 10, 20, 50], default=[5, 10, 20], key="so_ema_fast")
            so_slow = st.multiselect("Slow EMA Perioden",
                [20, 50, 100, 200], default=[20, 50, 200], key="so_ema_slow")
        elif so_strategy == "RSI Divergence":
            so_rdiv_p  = st.multiselect("RSI Perioden",
                RSI_DIV_GRID["periods"],    default=[14], key="so_rdiv_p")
            so_rdiv_lb = st.multiselect("Lookback",
                RSI_DIV_GRID["lookbacks"],  default=[10, 14], key="so_rdiv_lb")
            so_rdiv_os = st.multiselect("Oversold-Zone",
                RSI_DIV_GRID["oversolds"],  default=[30, 35], key="so_rdiv_os")
            so_rdiv_ob = st.multiselect("Overbought-Zone",
                RSI_DIV_GRID["overboughts"],default=[65, 70], key="so_rdiv_ob")
        elif so_strategy == "RSI":
            so_rsi_p  = st.multiselect("RSI Perioden",    [7, 14, 21], default=[7, 14], key="so_rsi_p")
            so_rsi_os = st.multiselect("Oversold-Werte",  [20, 25, 30, 35], default=[25, 30], key="so_rsi_os")
            so_rsi_ob = st.multiselect("Overbought-Werte",[65, 70, 75, 80], default=[70, 75], key="so_rsi_ob")
        elif so_strategy == "Bollinger":
            so_bb_p = st.multiselect("Perioden",  [10, 20, 30, 50], default=[10, 20], key="so_bb_p")
            so_bb_s = st.multiselect("Std Dev",   [1.5, 2.0, 2.5, 3.0], default=[1.5, 2.0], key="so_bb_s")
        elif so_strategy == "Breakout":
            so_bo_lb = st.multiselect("Lookback", [10, 20, 30, 50, 100], default=[20, 50], key="so_bo_lb")
        elif so_strategy == "MACD":
            so_macd_f  = st.multiselect("Fast EMA",  [8, 10, 12, 16], default=[8, 12], key="so_macd_f")
            so_macd_s  = st.multiselect("Slow EMA",  [21, 26, 35],    default=[21, 26], key="so_macd_s")
            so_macd_si = st.multiselect("Signal",    [7, 9, 12],      default=[7, 9],   key="so_macd_si")
        else:  # Supertrend
            so_st_atr  = st.multiselect("ATR Periode", [7, 10, 14, 21],   default=[7, 10], key="so_st_atr")
            so_st_mult = st.multiselect("Multiplikator", [2.0, 3.0, 4.0], default=[2.0, 3.0], key="so_st_mult")

        # ── Multi-Timeframe-Filter ─────────────────────────────────────────────
        st.divider()
        st.markdown("**Multi-Timeframe-Filter**")
        so_mtf_on = st.checkbox("MTF aktivieren", value=False, key="so_mtf_on")
        so_val_df1 = so_val_df2 = None
        if so_mtf_on:
            so_tf_combo = st.selectbox(
                "TF-Kombination",
                ["5m (Signal) + 15m + 1h", "1m (Signal) + 5m + 15m"],
                key="so_tf_combo",
            )
            so_val_csv1 = st.selectbox(
                "Validierungs-CSV 1 (mittlerer TF)", _csv_opts, key="so_val_csv1",
            )
            so_val_csv2 = st.selectbox(
                "Validierungs-CSV 2 (höherer TF)",  _csv_opts, key="so_val_csv2",
            )
            so_mtf_ema  = st.number_input(
                "MTF EMA Periode", 10, 200, 50, key="so_mtf_ema",
            )

        # ── ADX-Filter ────────────────────────────────────────────────────────
        st.divider()
        st.markdown("**ADX-Filter (Signal-TF)**")
        so_adx_mode = st.selectbox(
            "ADX-Modus",
            ["none", "trending", "ranging"],
            format_func=lambda x: {
                "none":     "Kein ADX-Filter",
                "trending": "Trending (ADX ≥ Schwelle) – für Bollinger",
                "ranging":  "Ranging  (ADX < Schwelle) – für RSI Div",
            }[x],
            key="so_adx_mode",
        )
        so_adx_ths: list[float] = [25.0]
        if so_adx_mode != "none":
            so_adx_ths = st.multiselect(
                "ADX-Schwellen testen",
                [15, 20, 25, 30, 35],
                default=[20, 25, 30],
                key="so_adx_ths",
            )
            if not so_adx_ths:
                so_adx_ths = [25.0]

    with so2:
        st.markdown("**Trade-Parameter (Suchraum)**")
        so_lev   = st.multiselect("Hebel",               [1,2,3,5,10,15,20,30,50],
                                  default=[5, 10, 20], key="so_lev")
        so_sizes = st.multiselect("Positionsgroesse (%)", [2,5,10,15,20,30,50],
                                  default=[5, 10], key="so_sizes")
        so_tp    = st.multiselect("Take Profit (%)",
                                  [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
                                  default=[0.5, 1.0, 2.0], key="so_tp")
        so_sl    = st.multiselect("Stop Loss (%)",
                                  [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                                  default=[0.3, 0.5, 1.0], key="so_sl")

    with so3:
        st.markdown("**Einstellungen**")
        so_capital    = st.number_input("Startkapital (USDT)", 10.0, 1_000_000.0,
                                        1000.0, 100.0, key="so_cap")
        so_fee        = st.number_input("Taker-Fee (%)", 0.0, 1.0, 0.055,
                                        0.001, format="%.4f", key="so_fee") / 100
        so_min_trades = st.number_input("Mindest-Trades", 1, 200, 5, key="so_min")
        so_metric     = st.radio(
            "Score-Metrik",
            ["scalping", "composite", "profit_factor", "total_pnl_pct", "sharpe"],
            format_func=lambda x: {
                "scalping":      "Scalping (PF>1.5, 50T, DD<20%)",
                "composite":     "Composite",
                "profit_factor": "Profit Factor",
                "total_pnl_pct": "PnL %",
                "sharpe":        "Sharpe",
            }[x],
            key="so_metric",
        )
        so_topn       = st.slider("Top-N", 5, 50, 20, key="so_topn")

        # Kombinationsanzahl anzeigen
        def _count_combos() -> int:
            if so_strategy == "EMA Cross":
                n_strat = sum(1 for f in (so_fast or []) for s in (so_slow or []) if f < s)
            elif so_strategy == "RSI Divergence":
                n_strat = sum(
                    1 for p in (so_rdiv_p or []) for lb in (so_rdiv_lb or [])
                    for o in (so_rdiv_os or []) for b in (so_rdiv_ob or []) if o < b
                )
            elif so_strategy == "RSI":
                n_strat = sum(1 for p in (so_rsi_p or []) for o in (so_rsi_os or [])
                              for b in (so_rsi_ob or []) if o < b)
            elif so_strategy == "Bollinger":
                n_strat = len(so_bb_p or []) * len(so_bb_s or [])
            elif so_strategy == "Breakout":
                n_strat = len(so_bo_lb or [])
            elif so_strategy == "MACD":
                n_strat = sum(1 for f in (so_macd_f or []) for s in (so_macd_s or [])
                              for si in (so_macd_si or []) if f < s)
            else:  # Supertrend
                n_strat = len(so_st_atr or []) * len(so_st_mult or [])
            n_adx = len(so_adx_ths) if so_adx_mode != "none" else 1
            return n_strat * n_adx * len(so_lev or []) * len(so_sizes or []) * len(so_tp or []) * len(so_sl or [])

        n_combos = _count_combos()
        st.metric("Kombinationen", f"{n_combos:,}")
        st.caption(f"~{max(1, int(n_combos * 0.004))}s geschätzt")

        st.divider()
        so_run = st.button("🔬 Optimierung starten", type="primary",
                           use_container_width=True, key="so_run",
                           disabled=n_combos == 0 or _no_csv)

    if so_run:
        # OptConfig bauen
        so_opt_cfg = StrategyOptConfig(
            leverages        = [int(x) for x in so_lev],
            position_sizes   = [x / 100 for x in so_sizes],
            tp_pcts          = list(so_tp),
            sl_pcts          = list(so_sl),
            initial_capital  = so_capital,
            fee_rate         = so_fee,
            min_trades       = int(so_min_trades),
            score_metric     = so_metric,
            adx_mode         = so_adx_mode,
            adx_thresholds   = [float(x) for x in so_adx_ths],
        )
        if so_strategy == "EMA Cross":
            so_opt_cfg.ema_fast_periods = [int(x) for x in so_fast]
            so_opt_cfg.ema_slow_periods = [int(x) for x in so_slow]
            so_key = "ema"
        elif so_strategy == "RSI Divergence":
            so_opt_cfg.rsi_div_periods     = [int(x) for x in so_rdiv_p]
            so_opt_cfg.rsi_div_lookbacks   = [int(x) for x in so_rdiv_lb]
            so_opt_cfg.rsi_div_oversolds   = list(so_rdiv_os)
            so_opt_cfg.rsi_div_overboughts = list(so_rdiv_ob)
            so_key = "rsi_divergence"
        elif so_strategy == "RSI":
            so_opt_cfg.rsi_periods     = [int(x) for x in so_rsi_p]
            so_opt_cfg.rsi_oversolds   = list(so_rsi_os)
            so_opt_cfg.rsi_overboughts = list(so_rsi_ob)
            so_key = "rsi"
        elif so_strategy == "Bollinger":
            so_opt_cfg.bb_periods  = [int(x) for x in so_bb_p]
            so_opt_cfg.bb_std_devs = list(so_bb_s)
            so_key = "bollinger"
        elif so_strategy == "Breakout":
            so_opt_cfg.bo_lookbacks = [int(x) for x in so_bo_lb]
            so_key = "breakout"
        elif so_strategy == "MACD":
            so_opt_cfg.macd_fasts   = [int(x) for x in so_macd_f]
            so_opt_cfg.macd_slows   = [int(x) for x in so_macd_s]
            so_opt_cfg.macd_signals = [int(x) for x in so_macd_si]
            so_key = "macd"
        else:  # Supertrend
            so_opt_cfg.st_atr_periods = [int(x) for x in so_st_atr]
            so_opt_cfg.st_multipliers = list(so_st_mult)
            so_key = "supertrend"

        so_df = pd.read_csv(os.path.join(RAW_DATA_DIR, so_csv))

        # MTF-Validierungs-CSVs laden
        so_val_dfs: list[pd.DataFrame] = []
        if so_mtf_on:
            so_opt_cfg.mtf_ema_period = int(so_mtf_ema)
            _v1 = pd.read_csv(os.path.join(RAW_DATA_DIR, so_val_csv1))
            _v2 = pd.read_csv(os.path.join(RAW_DATA_DIR, so_val_csv2))
            so_val_dfs = [_v1, _v2]
            st.info(f"MTF aktiv: {so_tf_combo}  |  EMA {so_mtf_ema}")

        so_prog = st.progress(0, text="Starte Strategie-Optimizer…")

        def _so_progress(done: int, total: int) -> None:
            so_prog.progress(done / total,
                             text=f"{done}/{total} Kombinationen getestet…")

        so_results = run_strategy_optimization(
            so_df, so_key, so_opt_cfg, _so_progress,
            val_dfs=so_val_dfs if so_val_dfs else None,
        )
        so_prog.progress(1.0, text="Fertig!")

        profitable_so = sum(1 for r in so_results if r.get("total_pnl_pct", 0) > 0)
        st.success(
            f"✅ {len(so_results)} Ergebnisse  |  "
            f"**{profitable_so} profitable**  |  "
            f"Bester Score: {so_results[0]['score']:.4f}" if so_results else "Keine Ergebnisse."
        )

        if not so_results:
            st.warning("Keine Konfiguration erfüllt die Mindest-Trades-Anforderung.")
            st.stop()

        # Session-State speichern
        st.session_state["so_results"] = so_results
        st.session_state["so_df"]      = so_df
        st.session_state["so_capital"] = so_capital

    # ── Ergebnisse anzeigen ────────────────────────────────────────────────────
    if "so_results" in st.session_state:
        so_results  = st.session_state["so_results"]
        so_df       = st.session_state["so_df"]
        so_capital  = st.session_state["so_capital"]
        so_topn     = st.session_state.get("so_topn", 20)

        st.subheader(f"Top {min(so_topn, len(so_results))} Konfigurationen")

        so_res_df = format_strategy_results_df(so_results, so_topn)

        def _color_so(val):
            if isinstance(val, (int, float)):
                if val > 0: return "color: #2ecc71"
                if val < 0: return "color: #e74c3c"
            return ""

        st.dataframe(
            so_res_df.style
                .map(_color_so, subset=["PnL %", "Profit Factor", "Score", "L-WR %", "S-WR %"])
                .format({
                    "PnL %":         "{:+.2f}",
                    "Profit Factor": "{:.4f}",
                    "Winrate %":     "{:.2f}",
                    "Max DD %":      "{:.2f}",
                    "L-WR %":        "{:.1f}",
                    "S-WR %":        "{:.1f}",
                    "Score":         "{:.4f}",
                }),
            use_container_width=True, hide_index=True,
            height=min(35 * min(so_topn, len(so_results)) + 60, 600),
        )

        # ── Long vs Short Aufschlüsselung ──────────────────────────────────────
        with st.expander("📊 Long vs Short Aufschlüsselung (Top Ergebnisse)"):
            ls_rows = []
            for rank, r in enumerate(so_results[:min(so_topn, len(so_results))]):
                cfg_r = r["config"]
                ls_rows.append({
                    "#":          rank + 1,
                    "Strategie":  str(r["strategy"]),
                    "L-Trades":   r.get("long_trades",  0),
                    "L-WR %":     r.get("long_winrate",  0),
                    "L-PF":       r.get("long_pf",       0),
                    "S-Trades":   r.get("short_trades",  0),
                    "S-WR %":     r.get("short_winrate", 0),
                    "S-PF":       r.get("short_pf",      0),
                })
            ls_df = pd.DataFrame(ls_rows)
            st.dataframe(
                ls_df.style.format({
                    "L-WR %": "{:.1f}", "L-PF": "{:.3f}",
                    "S-WR %": "{:.1f}", "S-PF": "{:.3f}",
                }),
                use_container_width=True, hide_index=True,
            )

        # ── Marktbedingungen (Trending vs Ranging) ─────────────────────────────
        with st.expander("📈 Marktbedingungen: Trending vs Ranging (ADX 14, Schwellwert 25)"):
            mc_rows = []
            for rank, r in enumerate(so_results[:min(so_topn, len(so_results))]):
                tn = r.get("trending_trades", 0)
                rn = r.get("ranging_trades",  0)
                if tn + rn == 0:
                    continue
                mc_rows.append({
                    "#":          rank + 1,
                    "Strategie":  str(r["strategy"]),
                    "Trend-Trades":  tn,
                    "Trend-WR %":    r.get("trending_winrate", 0),
                    "Range-Trades":  rn,
                    "Range-WR %":    r.get("ranging_winrate",  0),
                    "Trend-Anteil %": round(tn / (tn + rn) * 100, 1) if tn + rn > 0 else 0,
                })
            if mc_rows:
                mc_df = pd.DataFrame(mc_rows)
                st.dataframe(
                    mc_df.style.format({
                        "Trend-WR %": "{:.1f}", "Range-WR %": "{:.1f}",
                        "Trend-Anteil %": "{:.1f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("Keine Marktbedingungs-Daten verfügbar.")

        # ── Grafik-Auswahl ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("Grafik ansehen")

        def _so_label(rank: int) -> str:
            r   = so_results[rank]
            cfg = r["config"]
            tp  = cfg.take_profit_pct * 100 if cfg.take_profit_pct else 0
            sl  = cfg.stop_loss_pct   * 100 if cfg.stop_loss_pct   else 0
            return (f"#{rank+1}  {r['strategy']}"
                    f"  |  {cfg.leverage}x"
                    f"  |  TP {tp:.1f}% / SL {sl:.1f}%"
                    f"  |  Pos {cfg.position_size*100:.0f}%"
                    f"  |  PnL {r.get('total_pnl_pct',0):+.2f}%"
                    f"  |  PF {r.get('profit_factor',0):.3f}")

        so_sel_col, so_btn_col = st.columns([4, 1])
        with so_sel_col:
            so_sel = st.selectbox(
                "Konfiguration wählen",
                options=list(range(min(so_topn, len(so_results)))),
                format_func=_so_label,
                key="so_sel_rank",
                label_visibility="collapsed",
            )
        with so_btn_col:
            so_show = st.button("📈 Grafik laden",
                                use_container_width=True, key="so_show_chart")

        if so_show or st.session_state.get("so_chart_rank") == so_sel:
            st.session_state["so_chart_rank"] = so_sel
            chosen_result = so_results[so_sel]
            chosen_strat  = chosen_result["strategy"]
            chosen_cfg    = chosen_result["config"]
            adx_th_chart  = float(chosen_result.get("adx_threshold") or 25.0)

            # Cache backtest result; only re-run when button pressed or config changed
            _bt_key = f"__so_full_{so_sel}"
            if so_show or _bt_key not in st.session_state:
                with st.spinner("Lade vollständige Trade-Details..."):
                    st.session_state[_bt_key] = run_strategy_backtest(
                        so_df, chosen_strat, chosen_cfg
                    )
            full_so = st.session_state[_bt_key]

            t_so = full_so["trades"]
            e_so = full_so["equity_curve"]

            st.markdown(
                f"**{_so_label(so_sel)}**"
                f"  ·  Sharpe **{full_so.get('sharpe_ratio', 0):.4f}**"
                f"  ·  Max DD **{full_so['max_drawdown_pct']:.2f}%**"
            )

            # ── Indikatoren berechnen ──────────────────────────────────────────
            df_ind = so_df.copy()
            df_ind["datetime"] = pd.to_datetime(
                df_ind["datetime"], utc=True, errors="coerce"
            )
            df_ind = df_ind.dropna(subset=["datetime"]).reset_index(drop=True)
            df_ind["ema50"] = df_ind["close"].ewm(span=50, adjust=False).mean()
            df_ind["adx"]   = compute_adx(df_ind)

            _is_bb = isinstance(chosen_strat, BollingerStrategy)
            if _is_bb:
                _bp, _bsd = chosen_strat.period, chosen_strat.std_dev
                _bm = df_ind["close"].rolling(_bp).mean()
                _bs = df_ind["close"].rolling(_bp).std()
                df_ind["bb_mid"]   = _bm
                df_ind["bb_upper"] = _bm + _bsd * _bs
                df_ind["bb_lower"] = _bm - _bsd * _bs

            # ── Fold-Auswahl ───────────────────────────────────────────────────
            _nc  = len(df_ind)
            _fsz = max(_nc // 4, 1)
            _dts = df_ind["datetime"]
            _fold_opts = ["Alle Perioden"] + [
                f"Fold {i+1}  ({_dts.iloc[i*_fsz].strftime('%d.%m.%y')} – "
                f"{_dts.iloc[min((i+1)*_fsz-1, _nc-1)].strftime('%d.%m.%y')})"
                for i in range(4)
            ]
            fold_sel = st.selectbox(
                "Zeitraum wählen",
                range(len(_fold_opts)),
                format_func=lambda i: _fold_opts[i],
                key="so_fold_sel",
            )

            if fold_sel == 0:
                df_f = df_ind
            else:
                _fi = fold_sel - 1
                df_f = df_ind.iloc[_fi*_fsz : min((_fi+1)*_fsz, _nc)].reset_index(drop=True)

            fold_t0 = df_f["datetime"].iloc[0]
            fold_t1 = df_f["datetime"].iloc[-1]

            def _pt(s):
                if not s:
                    return None
                try:
                    return pd.to_datetime(s, utc=True)
                except Exception:
                    return None

            def _in_fold(t) -> bool:
                et = _pt(getattr(t, "entry_time", None))
                return et is not None and fold_t0 <= et <= fold_t1

            trades_f  = [t for t in t_so if _in_fold(t)]
            _so_id_map = {id(t): i for i, t in enumerate(t_so)}
            fold_idxs  = [_so_id_map[id(t)] for t in trades_f]

            # ── Chart: 3 Panels mit shared x-Achse ────────────────────────────
            _ctitle = (
                f"OHLCV  ·  {chosen_strat}"
                + (f"  ·  BB({_bp},{_bsd}σ)  +  EMA50" if _is_bb else "  ·  EMA50")
            )
            fig_r = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.55, 0.25, 0.20],
                vertical_spacing=0.04,
                subplot_titles=(
                    _ctitle,
                    "Equity  (🟢 Long  🟠 Short)",
                    f"ADX(14)  ·  Schwelle {adx_th_chart:.0f}",
                ),
            )

            # ─ Panel 1: Candlestick ───────────────────────────────────────────
            fig_r.add_trace(go.Candlestick(
                x=df_f["datetime"],
                open=df_f["open"], high=df_f["high"],
                low=df_f["low"],   close=df_f["close"],
                name="OHLC",
                increasing_line_color="#2ecc71",
                decreasing_line_color="#e74c3c",
                showlegend=False,
            ), row=1, col=1)

            # Bollinger Bands
            if _is_bb:
                fig_r.add_trace(go.Scatter(
                    x=df_f["datetime"], y=df_f["bb_upper"],
                    mode="lines", name="BB Oben",
                    line=dict(color="rgba(52,152,219,0.65)", width=1, dash="dot"),
                ), row=1, col=1)
                # fill="tonexty" fills from bb_lower UP to previously-added bb_upper
                fig_r.add_trace(go.Scatter(
                    x=df_f["datetime"], y=df_f["bb_lower"],
                    mode="lines", name="BB Unten",
                    line=dict(color="rgba(52,152,219,0.65)", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(52,152,219,0.05)",
                    showlegend=False,
                ), row=1, col=1)
                fig_r.add_trace(go.Scatter(
                    x=df_f["datetime"], y=df_f["bb_mid"],
                    mode="lines", name="BB Mitte",
                    line=dict(color="rgba(52,152,219,0.4)", width=1),
                ), row=1, col=1)

            # EMA 50
            fig_r.add_trace(go.Scatter(
                x=df_f["datetime"], y=df_f["ema50"],
                mode="lines", name="EMA 50",
                line=dict(color="rgba(241,196,15,0.9)", width=1.5),
            ), row=1, col=1)

            # Collect trade coordinates
            lx, ly, sx, sy = [], [], [], []
            tx, ty, rx, ry = [], [], [], []
            l_lines_x, l_lines_y = [], []
            s_lines_x, s_lines_y = [], []
            l_hover, s_hover     = [], []

            for t in trades_f:
                et  = _pt(t.entry_time)
                ext = _pt(t.exit_time) if getattr(t, "exit_time", None) else et
                if et is None:
                    continue
                ep, xp  = t.entry_price, t.exit_price
                ht = (f"Entry: {ep:,.2f}<br>"
                      f"Exit: {xp:,.2f} ({t.exit_reason.upper()})<br>"
                      f"PnL: {t.pnl_pct:+.2f}%")
                if t.side == "long":
                    lx.append(et); ly.append(ep); l_hover.append(ht)
                    if ext:
                        l_lines_x += [et, ext, None]
                        l_lines_y += [ep, xp, None]
                else:
                    sx.append(et); sy.append(ep); s_hover.append(ht)
                    if ext:
                        s_lines_x += [et, ext, None]
                        s_lines_y += [ep, xp, None]
                if t.exit_reason == "tp" and ext:
                    tx.append(ext); ty.append(xp)
                elif t.exit_reason == "sl" and ext:
                    rx.append(ext); ry.append(xp)

            # Entry→exit connector lines (faint)
            for _lx, _ly, _lc in [
                (l_lines_x, l_lines_y, "rgba(46,204,113,0.25)"),
                (s_lines_x, s_lines_y, "rgba(231,76,60,0.25)"),
            ]:
                if _lx:
                    fig_r.add_trace(go.Scatter(
                        x=_lx, y=_ly, mode="lines",
                        line=dict(color=_lc, width=1),
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=1)

            if lx:
                fig_r.add_trace(go.Scatter(
                    x=lx, y=ly, mode="markers", name="Long Entry",
                    marker=dict(symbol="triangle-up", size=10, color="#2ecc71",
                                line=dict(color="white", width=1)),
                    text=l_hover, hovertemplate="%{text}<extra>Long</extra>",
                ), row=1, col=1)
            if sx:
                fig_r.add_trace(go.Scatter(
                    x=sx, y=sy, mode="markers", name="Short Entry",
                    marker=dict(symbol="triangle-down", size=10, color="#e74c3c",
                                line=dict(color="white", width=1)),
                    text=s_hover, hovertemplate="%{text}<extra>Short</extra>",
                ), row=1, col=1)
            if tx:
                fig_r.add_trace(go.Scatter(
                    x=tx, y=ty, mode="markers", name="TP",
                    marker=dict(symbol="circle", size=8, color="#27ae60",
                                line=dict(color="white", width=1)),
                    hovertemplate="TP: %{y:,.2f}<extra></extra>",
                ), row=1, col=1)
            if rx:
                fig_r.add_trace(go.Scatter(
                    x=rx, y=ry, mode="markers", name="SL",
                    marker=dict(symbol="x", size=9, color="#c0392b",
                                line=dict(color="#c0392b", width=2)),
                    hovertemplate="SL: %{y:,.2f}<extra></extra>",
                ), row=1, col=1)

            # ─ Panel 2: Equity ────────────────────────────────────────────────
            eq_start = e_so[fold_idxs[0]] if fold_idxs else so_capital
            eq_t  = [fold_t0]
            eq_v  = [eq_start]
            prev_eq = eq_start
            leq_x, leq_y = [], []
            seq_x, seq_y = [], []

            for t, ti in zip(trades_f, fold_idxs):
                entry_t = _pt(t.entry_time)
                exit_t  = _pt(t.exit_time) if getattr(t, "exit_time", None) else entry_t
                new_eq  = (e_so[ti + 1] if ti + 1 < len(e_so)
                           else prev_eq + t.pnl)
                eq_t.append(exit_t or fold_t1)
                eq_v.append(new_eq)
                if t.side == "long":
                    leq_x += [entry_t, exit_t, None]
                    leq_y += [prev_eq, new_eq, None]
                else:
                    seq_x += [entry_t, exit_t, None]
                    seq_y += [prev_eq, new_eq, None]
                prev_eq = new_eq

            fig_r.add_trace(go.Scatter(
                x=eq_t, y=eq_v, mode="lines",
                line=dict(color="rgba(200,200,200,0.35)", width=1),
                showlegend=False,
                hovertemplate="Equity: %{y:,.2f}<extra></extra>",
            ), row=2, col=1)
            fig_r.add_hline(
                y=eq_start, line_dash="dash",
                line_color="rgba(150,150,150,0.5)", row=2, col=1,
            )
            for _ex, _ey, _ec, _el in [
                (leq_x, leq_y, "#2ecc71", "Long Equity"),
                (seq_x, seq_y, "#e67e22", "Short Equity"),
            ]:
                if _ex:
                    fig_r.add_trace(go.Scatter(
                        x=_ex, y=_ey, mode="lines", name=_el,
                        line=dict(color=_ec, width=2),
                    ), row=2, col=1)

            # ─ Panel 3: ADX ───────────────────────────────────────────────────
            _adx_v = df_f["adx"].fillna(0).to_numpy()
            # Gray fill when ADX < threshold (= no-trading zone)
            _gray  = np.where(_adx_v < adx_th_chart, float(adx_th_chart), float("nan"))
            fig_r.add_trace(go.Scatter(
                x=df_f["datetime"], y=_gray,
                mode="none", fill="tozeroy",
                fillcolor="rgba(150,150,150,0.12)",
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ), row=3, col=1)
            fig_r.add_trace(go.Scatter(
                x=df_f["datetime"], y=df_f["adx"],
                mode="lines", name="ADX(14)",
                line=dict(color="#9b59b6", width=1.5),
                hovertemplate="ADX: %{y:.1f}<extra></extra>",
            ), row=3, col=1)
            fig_r.add_hline(
                y=adx_th_chart,
                line_dash="dash",
                line_color="rgba(255,200,0,0.9)",
                line_width=1.5,
                annotation_text=f"  {adx_th_chart:.0f}",
                annotation_position="right",
                row=3, col=1,
            )

            # ─ Layout ─────────────────────────────────────────────────────────
            fig_r.update_layout(
                height=950,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11),
                ),
                margin=dict(l=60, r=60, t=80, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_r.update_xaxes(
                showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                rangeslider_visible=False,
            )
            fig_r.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            # Hide x-tick labels on top two rows
            fig_r.update_xaxes(showticklabels=False, row=1, col=1)
            fig_r.update_xaxes(showticklabels=False, row=2, col=1)

            st.plotly_chart(fig_r, use_container_width=True)
            st.caption(
                "▲ Long Entry  ▼ Short Entry  ● TP-Exit  × SL-Exit  |  "
                "Equity: 🟢 Long-Trades  🟠 Short-Trades  |  "
                "ADX-Panel: Grau = kein Trading (ADX < Schwelle)"
            )

            with st.expander("Alle Trades"):
                tdf_so = pd.DataFrame([{
                    "#": t.index+1, "Side": t.side.upper(),
                    "Entry": round(t.entry_price,2), "Exit": round(t.exit_price,2),
                    "Grund": t.exit_reason.upper(),
                    "PnL (USDT)": round(t.pnl,4), "PnL (%)": round(t.pnl_pct,3),
                    "Equity": round(t.equity_after,4),
                } for t in t_so])
                st.dataframe(tdf_so.style.map(
                    lambda v: "color: #2ecc71" if isinstance(v,(int,float)) and v>0
                    else ("color: #e74c3c" if isinstance(v,(int,float)) and v<0 else ""),
                    subset=["PnL (USDT)","PnL (%)"]),
                    use_container_width=True, hide_index=True)
    else:
        st.info("Parameter wählen und **🔬 Optimierung starten** klicken.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – MULTI-SYMBOL SCALPING
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "🎯 Multi-Symbol":

    st.subheader("🎯 Multi-Symbol Bollinger Scalping")
    st.caption(
        "Bollinger Bands + Multi-Timeframe Filter (15m/1h) + ADX-Filter  ·  "
        "Walk-Forward über alle gewählten Symbole"
    )

    # ── Auto-detect Symbole die 5m + 15m + 1h haben ───────────────────────────
    _ms_avail: list[str] = []
    for _f in sorted(csv_files):
        if _f.endswith("_USDT_USDT_5m.csv"):
            _s = _f.replace("_USDT_USDT_5m.csv", "")
            if (f"{_s}_USDT_USDT_15m.csv" in csv_files and
                    f"{_s}_USDT_USDT_1h.csv" in csv_files):
                _ms_avail.append(_s)

    _ms_defaults = [s for s in ["ADA", "BTC", "DOT", "ETH", "SOL"] if s in _ms_avail]

    mc1, mc2, mc3 = st.columns([1.3, 1.1, 0.8])

    with mc1:
        st.markdown("**Symbole**")
        ms_syms = st.multiselect(
            "Symbole",
            _ms_avail,
            default=_ms_defaults or _ms_avail[:5],
            key="ms_syms",
            label_visibility="collapsed",
            disabled=_no_csv,
        )
        st.divider()
        st.markdown("**Bollinger Bands**")
        ms_bb_p   = st.number_input("Periode",  2, 100,  10, key="ms_bb_p")
        ms_bb_std = st.slider("Std Dev", 0.5, 5.0, 2.5, 0.1, key="ms_bb_std")
        st.divider()
        st.markdown("**Multi-Timeframe Filter**")
        ms_mtf     = st.checkbox("MTF aktivieren (15m + 1h)", value=True, key="ms_mtf")
        ms_mtf_ema = (st.number_input("EMA Periode", 10, 200, 50, key="ms_mtf_ema")
                      if ms_mtf else 50)

    with mc2:
        st.markdown("**ADX Filter**")
        ms_adx_mode = st.selectbox(
            "ADX Modus",
            ["trending", "none"],
            format_func=lambda x: {
                "trending": "Trending  (ADX ≥ Schwelle)",
                "none":     "Kein Filter",
            }[x],
            key="ms_adx_mode",
            label_visibility="collapsed",
        )
        ms_adx_th = (
            st.slider("ADX Schwelle", 15, 40, 25, 1, key="ms_adx_th")
            if ms_adx_mode != "none" else 25
        )
        st.divider()
        st.markdown("**Trade-Parameter**")
        ms_lev  = st.slider("Hebel",            1,  50, 20, key="ms_lev")
        ms_pos  = st.slider("Position %",       1,  50, 10, key="ms_pos") / 100
        ms_tp   = st.slider("Take Profit %", 0.10, 5.0, 0.75, 0.05,
                             format="%.2f%%", key="ms_tp") / 100
        ms_sl   = st.slider("Stop Loss %",   0.10, 3.0, 0.50, 0.05,
                             format="%.2f%%", key="ms_sl") / 100
        ms_fee  = st.number_input("Taker-Fee (%)", 0.0, 1.0, 0.055, 0.001,
                                   format="%.4f", key="ms_fee") / 100

    with mc3:
        st.markdown("**Walk-Forward**")
        ms_wf       = st.checkbox("Walk-Forward (4 Folds)", value=True, key="ms_wf")
        ms_train_d  = st.number_input("Train Tage", 30, 300, 120, key="ms_train") if ms_wf else 120
        ms_test_d   = st.number_input("Test Tage",  10, 120,  60, key="ms_test")  if ms_wf else 60

        st.divider()
        _ms_n_sym = len(ms_syms)
        _ms_n_folds = 4 if ms_wf else 1
        st.metric("Symbole",      _ms_n_sym)
        st.metric("Folds gesamt", _ms_n_sym * _ms_n_folds)

        st.markdown("&nbsp;")
        ms_run = st.button(
            "▶  Analyse starten",
            type="primary",
            use_container_width=True,
            key="ms_run",
            disabled=_ms_n_sym == 0 or _no_csv,
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    if ms_run:
        from src.strategies.bollinger_strategy import BollingerStrategy as _BB

        _N_DAY = 288  # 5m candles per day

        def _ms_load(sym: str, tf: str) -> pd.DataFrame:
            _p = os.path.join(RAW_DATA_DIR, f"{sym}_USDT_USDT_{tf}.csv")
            _d = pd.read_csv(_p)
            _d["datetime"] = pd.to_datetime(_d["datetime"], utc=True)
            return _d.sort_values("datetime").reset_index(drop=True)

        def _ms_clip(df: pd.DataFrame, t0, t1) -> pd.DataFrame:
            return df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].reset_index(drop=True)

        _EMPTY_R = {"num_trades": 0, "profit_factor": 0.0,
                    "total_pnl_pct": 0.0, "max_drawdown_pct": 0.0,
                    "long_trades": 0, "short_trades": 0,
                    "long_winrate": 0.0, "short_winrate": 0.0}

        _strat_bb   = _BB(period=int(ms_bb_p), std_dev=float(ms_bb_std))
        _trade_cfg  = StrategyConfig(
            leverage=int(ms_lev), position_size=float(ms_pos),
            take_profit_pct=float(ms_tp), stop_loss_pct=float(ms_sl),
            fee_rate=float(ms_fee),
        )

        ms_prog = st.progress(0, text="Starte…")
        all_sym_results: dict[str, list[dict]] = {}
        _total_steps = _ms_n_sym * _ms_n_folds
        _done = 0

        for sym in ms_syms:
            try:
                _df5  = _ms_load(sym, "5m")
                _df15 = _ms_load(sym, "15m")
                _df1h = _ms_load(sym, "1h")
            except FileNotFoundError:
                st.warning(f"{sym}: Datei fehlt – übersprungen")
                continue

            _st  = max(_df5["datetime"].iloc[0],  _df15["datetime"].iloc[0], _df1h["datetime"].iloc[0])
            _en  = min(_df5["datetime"].iloc[-1], _df15["datetime"].iloc[-1], _df1h["datetime"].iloc[-1])
            _df5  = _ms_clip(_df5,  _st, _en)
            _df15 = _ms_clip(_df15, _st, _en)
            _df1h = _ms_clip(_df1h, _st, _en)
            _n5   = len(_df5)

            # Build folds
            if ms_wf:
                _folds: list[tuple[int,int,int,int]] = []
                _d = int(ms_train_d)
                while _d + int(ms_test_d) <= _n5 // _N_DAY:
                    _te = _d * _N_DAY
                    _folds.append((0, _te, _te, min(_te + int(ms_test_d) * _N_DAY, _n5)))
                    _d += int(ms_test_d)
                    if len(_folds) >= 4:
                        break
                if not _folds:
                    st.warning(f"{sym}: zu wenig Daten ({_n5//_N_DAY}d) – übersprungen")
                    continue
            else:
                _folds = [(0, _n5, 0, _n5)]

            _sigs_full = _strat_bb.generate_signals(_df5).to_numpy(int)
            _opens     = _df5["open"].to_numpy(float)
            _highs     = _df5["high"].to_numpy(float)
            _lows      = _df5["low"].to_numpy(float)
            _closes    = _df5["close"].to_numpy(float)

            fold_results: list[dict] = []
            for _ts, _te, _qs, _qe in _folds:
                _qdf  = _df5.iloc[_qs:_qe].reset_index(drop=True)
                _q0, _q1 = _qdf["datetime"].iloc[0], _qdf["datetime"].iloc[-1]
                _te15 = _ms_clip(_df15, _q0, _q1)
                _te1h = _ms_clip(_df1h, _q0, _q1)

                _sigs = _sigs_full[_qs:_qe].copy()
                if ms_mtf and len(_te15) > 0 and len(_te1h) > 0:
                    _sigs = apply_mtf_filter(_qdf, [_te15, _te1h], _sigs, int(ms_mtf_ema))
                if ms_adx_mode != "none":
                    _adx = compute_adx(_qdf)
                    _sigs = apply_adx_filter(_sigs, _adx, float(ms_adx_th), "trending")

                _r = run_strategy_backtest_fast(
                    _opens[_qs:_qe], _highs[_qs:_qe],
                    _lows[_qs:_qe],  _closes[_qs:_qe],
                    _sigs, _trade_cfg,
                )
                _r = _r if _r else _EMPTY_R.copy()

                _days = (_qe - _qs) // _N_DAY or 1
                fold_results.append({
                    **_r,
                    "days":   _days,
                    "tpd":    _r["num_trades"] / _days,
                    "t0":     str(_q0.date()),
                    "t1":     str(_q1.date()),
                })
                _done += 1
                ms_prog.progress(
                    _done / _total_steps,
                    text=f"{sym} Fold {len(fold_results)}/{len(_folds)}…",
                )

            all_sym_results[sym] = fold_results

        ms_prog.progress(1.0, text="Fertig!")
        st.session_state["ms_results"]  = all_sym_results
        st.session_state["ms_n_folds"]  = _ms_n_folds
        st.session_state["ms_test_days"] = int(ms_test_d)

    # ── Ergebnisse anzeigen ────────────────────────────────────────────────────
    if "ms_results" in st.session_state:
        _res     = st.session_state["ms_results"]
        _nf      = st.session_state.get("ms_n_folds", 4)
        _td      = st.session_state.get("ms_test_days", 60)
        _syms    = list(_res.keys())
        _n_syms  = len(_syms)

        if not _syms:
            st.warning("Keine Ergebnisse – alle Symbole hatten zu wenig Daten.")
        else:
            # ── Kacheln: Portfolio-Gesamt ──────────────────────────────────────
            _all_folds: list[dict] = [f for fds in _res.values() for f in fds]
            _total_trades  = sum(f["num_trades"] for f in _all_folds)
            _total_tpd     = sum(f["tpd"]        for f in _all_folds) / _nf
            _avg_pf        = sum(f["profit_factor"] for f in _all_folds) / max(len(_all_folds), 1)
            _avg_pnl_sym   = sum(
                sum(f["total_pnl_pct"] for f in fds) / max(len(fds), 1)
                for fds in _res.values()
            ) / _n_syms
            _n_prof_folds  = sum(
                1 for fds in _res.values()
                for f in fds if f["profit_factor"] >= 1.0
            )
            _n_total_folds = sum(len(fds) for fds in _res.values())

            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            kc1.metric("Symbole",           _n_syms)
            kc2.metric("Trades / Tag",       f"{_total_tpd:.2f}",
                       help="Summe aller Symbole pro Tag")
            kc3.metric("Ø Profit Factor",    f"{_avg_pf:.3f}",
                       delta=f"{_avg_pf - 1:.3f}",
                       delta_color="normal" if _avg_pf >= 1 else "inverse")
            kc4.metric("Ø PnL/Symbol/Fold",  f"{_avg_pnl_sym:+.2f}%",
                       delta_color="normal" if _avg_pnl_sym >= 0 else "inverse")
            kc5.metric("Profitable Folds",   f"{_n_prof_folds}/{_n_total_folds}",
                       help="Folds mit PF ≥ 1.0")

            st.divider()

            # ── Per-Symbol Tabelle ─────────────────────────────────────────────
            st.markdown("#### Ergebnisse pro Symbol")

            def _fold_cell(f: dict) -> str:
                pf = f["profit_factor"]
                t  = f["num_trades"]
                m  = "✓" if (pf >= 1.5 and t >= 5) else ("~" if pf >= 1.0 else "✗")
                return f"{m} PF{pf:.2f} ({t}T)"

            _rows = []
            for sym, fds in _res.items():
                _pfs    = [f["profit_factor"]  for f in fds]
                _pnls   = [f["total_pnl_pct"]  for f in fds]
                _trades = [f["num_trades"]      for f in fds]
                _row = {
                    "Symbol":    sym,
                    "Ø PF":      round(sum(_pfs)  / len(_pfs), 3),
                    "Ø PnL %":   round(sum(_pnls) / len(_pnls), 2),
                    "Ø T/Tag":   round(sum(f["tpd"] for f in fds) / len(fds), 2),
                    "Folds ✓":   f"{sum(1 for pf,t in zip(_pfs,_trades) if pf>=1.5 and t>=5)}/{len(fds)}",
                }
                for i, f in enumerate(fds):
                    _row[f"F{i+1}\n{f['t0']}"] = _fold_cell(f)
                _rows.append(_row)

            _sym_df = pd.DataFrame(_rows)

            def _color_pf(val):
                if isinstance(val, float):
                    if val >= 1.5: return "color: #2ecc71"
                    if val >= 1.0: return "color: #f39c12"
                    return "color: #e74c3c"
                return ""

            def _color_pnl(val):
                if isinstance(val, float):
                    return "color: #2ecc71" if val >= 0 else "color: #e74c3c"
                return ""

            st.dataframe(
                _sym_df.style
                    .map(_color_pf,  subset=["Ø PF"])
                    .map(_color_pnl, subset=["Ø PnL %"])
                    .format({"Ø PF": "{:.3f}", "Ø PnL %": "{:+.2f}", "Ø T/Tag": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )

            # ── Portfolio Fold-Tabelle ─────────────────────────────────────────
            if _nf > 1:
                st.markdown("#### Portfolio pro Fold")

                _max_folds = max(len(fds) for fds in _res.values())
                _fold_rows = []
                for fi in range(_max_folds):
                    _fold_syms = [
                        (sym, fds[fi]) for sym, fds in _res.items() if fi < len(fds)
                    ]
                    _f_trades  = sum(f["num_trades"]    for _, f in _fold_syms)
                    _f_tpd     = sum(f["tpd"]           for _, f in _fold_syms)
                    _f_pnl_avg = sum(f["total_pnl_pct"] for _, f in _fold_syms) / max(len(_fold_syms), 1)
                    _f_pf_avg  = sum(f["profit_factor"] for _, f in _fold_syms) / max(len(_fold_syms), 1)
                    _f_pos     = sum(1 for _, f in _fold_syms if f["profit_factor"] >= 1.0)
                    _t0 = _fold_syms[0][1]["t0"] if _fold_syms else ""
                    _t1 = _fold_syms[0][1]["t1"] if _fold_syms else ""
                    _fold_rows.append({
                        "Fold":          f"F{fi+1}  ({_t0} → {_t1})",
                        "Trades Σ":      _f_trades,
                        "Trades / Tag":  round(_f_tpd, 2),
                        "Ø PF":          round(_f_pf_avg, 3),
                        "Ø PnL %":       round(_f_pnl_avg, 2),
                        "Sym. profitabel": f"{_f_pos}/{len(_fold_syms)}",
                    })

                _fold_df = pd.DataFrame(_fold_rows)
                st.dataframe(
                    _fold_df.style
                        .map(_color_pf,  subset=["Ø PF"])
                        .map(_color_pnl, subset=["Ø PnL %"]),
                    use_container_width=True, hide_index=True,
                )

            # ── Equity-Chart ───────────────────────────────────────────────────
            with st.expander("📈 Equity-Kurve pro Symbol"):
                _eq_fig = go.Figure()
                _colors = [
                    "#2ecc71","#3498db","#e74c3c","#f39c12","#9b59b6",
                    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
                ]
                for _ci, (sym, fds) in enumerate(_res.items()):
                    _cum_pnl = [0.0]
                    for f in fds:
                        _cum_pnl.append(_cum_pnl[-1] + f["total_pnl_pct"])
                    _eq_fig.add_trace(go.Scatter(
                        x=list(range(len(_cum_pnl))),
                        y=_cum_pnl,
                        mode="lines+markers",
                        name=sym,
                        line=dict(color=_colors[_ci % len(_colors)], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"{sym}<br>nach Fold %{{x}}: %{{y:+.2f}}%<extra></extra>",
                    ))
                _eq_fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                _eq_fig.update_layout(
                    height=350, showlegend=True,
                    xaxis_title="Fold", yaxis_title="Kumulierter PnL %",
                    legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
                    margin=dict(l=50, r=20, t=30, b=40),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                _eq_fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
                _eq_fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
                st.plotly_chart(_eq_fig, use_container_width=True)

            # ── Live-Ready Bewertung ───────────────────────────────────────────
            st.divider()
            st.markdown("#### Live-Ready Bewertung")

            _lr_cols = st.columns(min(_n_syms, 6))
            for _ci2, (sym, fds) in enumerate(_res.items()):
                _pfs2   = [f["profit_factor"] for f in fds]
                _trad2  = [f["num_trades"]    for f in fds]
                _n_ok   = sum(1 for pf, t in zip(_pfs2, _trad2) if pf >= 1.5 and t >= 5)
                _n_pos  = sum(1 for pf in _pfs2 if pf >= 1.0)
                _folds_total = len(fds)
                _ready  = _n_ok >= max(1, round(0.75 * _folds_total))
                with _lr_cols[_ci2 % 6]:
                    _icon = "✅" if _ready else ("⚠️" if _n_pos >= _folds_total else "❌")
                    st.metric(
                        label=f"{_icon} {sym}",
                        value=f"{_n_ok}/{_folds_total} Folds ✓",
                        delta=f"Ø PF {sum(_pfs2)/len(_pfs2):.2f}",
                        delta_color="normal" if sum(_pfs2)/len(_pfs2) >= 1 else "inverse",
                    )

    else:
        st.info("Symbole und Parameter wählen, dann **▶ Analyse starten** klicken.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – DATEN LADEN
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "📥 Daten laden":
    st.subheader("📥 Historische Daten laden")
    st.caption("Lädt OHLCV-Daten von Bybit und speichert sie als CSV in `data/raw/`.")

    KNOWN_SYMBOLS = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "BNB/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "MATIC/USDT:USDT",
        "LINK/USDT:USDT",
    ]
    TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]

    tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }

    dl_col, info_col = st.columns([1, 1])

    with dl_col:
        st.markdown("**Datensätze herunterladen**")

        dl_symbols = st.multiselect(
            "Coins / Symbole (mehrere wählbar)",
            options=KNOWN_SYMBOLS,
            default=["BTC/USDT:USDT"],
            key="dl_symbols",
        )
        dl_custom = st.text_input(
            "Eigene Symbole (kommagetrennt, z. B. `OP/USDT:USDT, ARB/USDT:USDT`)",
            value="", key="dl_custom",
        )

        # Alle Symbole zusammenführen
        extra = [s.strip() for s in dl_custom.split(",") if s.strip()]
        all_dl_symbols = list(dict.fromkeys(dl_symbols + extra))  # unique, order preserved

        dl_timeframes = st.multiselect(
            "Timeframes (mehrere wählbar)",
            TIMEFRAMES, default=["15m"], key="dl_tfs",
        )
        dl_days = st.slider("Zeitraum (Tage)", 1, 365, 30, key="dl_days")

        # Vorschau
        n_jobs = len(all_dl_symbols) * len(dl_timeframes)
        if n_jobs > 0 and dl_timeframes:
            approx = int(dl_days * 24 * 60 / tf_minutes.get(dl_timeframes[0], 1))
            st.caption(
                f"**{n_jobs} Download(s)**  ·  ~{approx:,} Kerzen pro Datei  "
                f"·  ~{max(1, n_jobs * dl_days // 10)}s geschätzt"
            )

        dl_btn = st.button("⬇️ Alle herunterladen", type="primary",
                           use_container_width=True, key="dl_run",
                           disabled=n_jobs == 0)

        if dl_btn:
            if not all_dl_symbols or not dl_timeframes:
                st.error("Mindestens ein Symbol und ein Timeframe auswählen.")
            else:
                dl_prog  = st.progress(0, text="Starte Downloads...")
                dl_log   = st.empty()
                done_cnt = 0
                errors   = []
                successes = []

                for sym in all_dl_symbols:
                    for tf in dl_timeframes:
                        label = f"{sym} / {tf}"
                        dl_log.info(f"Lade {label}…")
                        try:
                            df_dl = fetch_ohlcv(sym, tf, dl_days)
                            path  = save_csv(df_dl, sym, tf)
                            successes.append(f"✅ `{os.path.basename(path)}` ({len(df_dl):,} Kerzen)")
                        except Exception as exc:
                            errors.append(f"❌ {label}: {exc}")
                        done_cnt += 1
                        dl_prog.progress(done_cnt / n_jobs,
                                         text=f"{done_cnt}/{n_jobs} — {label}")

                dl_prog.progress(1.0, text="Fertig!")
                dl_log.empty()
                for msg in successes:
                    st.success(msg)
                for msg in errors:
                    st.error(msg)
                if successes:
                    st.rerun()

    with info_col:
        st.markdown("**Vorhandene Datensätze**")

        if os.path.isdir(RAW_DATA_DIR):
            csv_list = sorted(
                [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
            )
        else:
            csv_list = []

        if not csv_list:
            st.info("Noch keine CSV-Dateien vorhanden.")
        else:
            rows_info = []
            for fname in csv_list:
                fpath = os.path.join(RAW_DATA_DIR, fname)
                try:
                    df_info = pd.read_csv(fpath, usecols=["datetime"])
                    n_rows  = len(df_info)
                    d_from  = df_info["datetime"].iloc[0][:10]
                    d_to    = df_info["datetime"].iloc[-1][:10]
                    size_kb = os.path.getsize(fpath) // 1024
                    rows_info.append({
                        "Datei":       fname,
                        "Kerzen":      f"{n_rows:,}",
                        "Von":         d_from,
                        "Bis":         d_to,
                        "Größe (KB)": size_kb,
                    })
                except Exception:
                    rows_info.append({
                        "Datei": fname, "Kerzen": "?",
                        "Von": "?", "Bis": "?", "Größe (KB)": "?",
                    })

            st.dataframe(
                pd.DataFrame(rows_info),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – LIVE TRADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_live_trader() -> LiveTrader:
    trader = LiveTrader()
    # Auto-resume: wenn State-Datei running=True hatte und Strategie rekonstruiert wurde
    if trader.running and trader.strategy is not None:
        if not (trader._thread is not None and trader._thread.is_alive()):
            try:
                trader.start()
            except Exception as _ae:
                trader._log(f"Auto-Start fehlgeschlagen: {_ae}", "ERROR")
    return trader


_LT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "BNB/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT",
    # OptiTest Top-Coins (BB_50/2.5 · 3×)
    "H/USDT:USDT", "RIVER/USDT:USDT", "RAVE/USDT:USDT",
    "ZEREBRO/USDT:USDT", "LAB/USDT:USDT", "PLAYSOUT/USDT:USDT",
    "UB/USDT:USDT", "SOON/USDT:USDT", "ESPORTS/USDT:USDT",
    "STABLE/USDT:USDT", "HANA/USDT:USDT", "MITO/USDT:USDT",
    "TRUST/USDT:USDT", "XAN/USDT:USDT", "ZEC/USDT:USDT",
]
_LT_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_live_candles(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Auf Modul-Ebene definiert damit @st.cache_data stabil cached."""
    from src.exchange import get_public_exchange
    ex = get_public_exchange()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUPERTREND LIVE – Helper-Funktionen
# ══════════════════════════════════════════════════════════════════════════════
import json as _json
import threading as _threading
from datetime import datetime as _dt, timedelta as _td, timezone as _tz

_STL_DIR        = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "supertrend_live")
_STL_STATE_FILE = os.path.join(_STL_DIR, "state.json")
_STL_DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "supertrend_test")
_STL_OPT_LOCK   = _threading.Lock()

# Beste Params aus Backtest – automatisch gesetzt
_STL_LEVERAGE    = 3
_STL_POS_SIZE    = 0.05
_STL_FEE         = 0.00055
_STL_TIMEFRAME   = "1h"
_STL_ST_PARAMS   = [(20, 2.0), (14, 2.0), (10, 2.0)]
_STL_TOP_N       = 10
# Exit-Logik Variante C (kein Lookahead-Bias, echter 3-Kerzen-Edge):
# Trail startet erst nach 3 abgeschlossenen Kerzen. Kein Intrabar-Bias.
# Signal-Flip exitiert weiterhin sofort. Notfall-SL 1.67% während Warmup.
_STL_ATR_MODE         = False
_STL_ATR_PERIOD       = 14
_STL_ATR_SL_MULT      = 1.5
_STL_ATR_RR           = 2.0
_STL_TRAILING         = 0.003   # 0.3% Trail-Abstand (aktiv ab Kerze 4)
_STL_ACTIVATION       = None    # kein Activation-Threshold
_STL_SL_PCT           = None    # kein fixer SL — 1.67% schneidet 57% aller Trades ab
_STL_WARMUP_CANDLES   = 3       # Trail erst ab 4. Kerze nach Entry


def _stl_load_state() -> dict:
    try:
        with open(_STL_STATE_FILE, encoding="utf-8") as _f:
            return _json.load(_f)
    except Exception:
        return {"last_optimized": None, "top_coins": [], "optimizing": False}


def _stl_save_state(state: dict) -> None:
    os.makedirs(_STL_DIR, exist_ok=True)
    with open(_STL_STATE_FILE, "w", encoding="utf-8") as _f:
        _json.dump(state, _f, indent=2, default=str)


def _stl_compute_supertrend_arr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                 period: int, mult: float):
    """Berechnet SuperTrend-Linie und Richtungs-Array. Gibt (st_line, direction) zurück."""
    n = len(closes)
    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (highs + lows) / 2.0
    bu  = hl2 + mult * atr
    bl  = hl2 - mult * atr
    fu  = bu.copy(); fl = bl.copy()
    direction = np.ones(n, dtype=int)

    for i in range(period + 1, n):
        fu[i] = bu[i] if (bu[i] < fu[i-1] or closes[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = bl[i] if (bl[i] > fl[i-1] or closes[i-1] < fl[i-1]) else fl[i-1]
        prev_d = direction[i-1]
        if   prev_d == -1 and closes[i] > fu[i]: direction[i] =  1
        elif prev_d ==  1 and closes[i] < fl[i]: direction[i] = -1
        else:                                     direction[i] = prev_d

    st_line = np.where(direction == 1, fl, fu)
    return st_line, direction


def _stl_backtest_fast(opens, highs, lows, closes, signals, capital):
    """Backtest Variante C: Trail=0.3% erst ab Kerze 4, Notfall-SL=1.67% waehrend Warmup.
    Kein Intrabar-Bias: Trail startet nach abgeschlossenen Warmup-Kerzen.
    Backtest-Logik identisch mit Live (_ensure_trailing_stop nach N Ticks).
    """
    equity   = capital
    pnls: list = []
    WARMUP   = _STL_WARMUP_CANDLES
    n = len(opens); i = 0

    while i < n - 1:
        sig = int(signals[i])
        if sig not in (1, -1): i += 1; continue
        side  = "long" if sig == 1 else "short"
        entry = opens[i + 1]
        if entry <= 0: i += 1; continue

        notional  = equity * _STL_POS_SIZE * _STL_LEVERAGE
        best      = entry
        trail_sl  = None
        reason    = "timeout"
        exit_price = closes[min(i + 500, n - 1)]
        exit_idx   = min(i + 1 + 500, n - 1)

        for j in range(i + 1, min(i + 501, n)):
            h = highs[j]; l = lows[j]
            candle_num = j - (i + 1)

            # Signal-Flip: immer erlaubt
            if j < n - 1:
                s = int(signals[j])
                if (side == "long" and s == -1) or (side == "short" and s == 1):
                    exit_price = opens[j + 1]; exit_idx = j + 1; reason = "signal"; break

            # best_price kontinuierlich tracken
            if side == "long" and h > best:
                best = h
            elif side == "short" and l < best:
                best = l

            # Warmup: nur Signal-Exit erlaubt, kein SL/Trail
            if candle_num < WARMUP:
                continue

            # Ab Kerze WARMUP: Trail einmalig initialisieren
            if trail_sl is None:
                trail_sl = (best * (1 - _STL_TRAILING) if side == "long"
                            else best * (1 + _STL_TRAILING))

            # Trail verschieben
            if side == "long":
                cand = best * (1 - _STL_TRAILING)
                if cand > trail_sl: trail_sl = cand
            else:
                cand = best * (1 + _STL_TRAILING)
                if cand < trail_sl: trail_sl = cand

            # Trail-Exit
            if (side == "long" and l <= trail_sl) or                (side == "short" and h >= trail_sl):
                exit_price = trail_sl; exit_idx = j; reason = "trail"; break

        raw_pnl = ((exit_price - entry) / entry * notional if side == "long"
                   else (entry - exit_price) / entry * notional)
        net_pnl = raw_pnl - notional * _STL_FEE * 2
        equity += net_pnl
        pnls.append(net_pnl)
        i = exit_idx - 1 if reason == "signal" else exit_idx
        if equity <= 0: break

    if not pnls: return {}
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gl     = abs(sum(losses))
    pf     = sum(wins) / gl if gl > 0 else (99.0 if wins else 0.0)
    eq = capital; peak = eq; max_dd = 0.0
    for p in pnls:
        eq += p; peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100)
    return {
        "num_trades":       len(pnls),
        "total_pnl_pct":    (equity - capital) / capital * 100,
        "winrate_pct":      len(wins) / len(pnls) * 100,
        "profit_factor":    min(pf, 99.0),
        "max_drawdown_pct": max_dd,
    }

def _stl_score(r: dict) -> float:
    pnl = r.get("total_pnl_pct", 0)
    pf  = min(r.get("profit_factor", 0) or 0, 3.0)
    n   = max(r.get("num_trades", 0), 1)
    return pnl * pf * (n ** 0.5)


def _stl_run_optimization(max_days_cache: int = 7, progress_fn=None) -> list[dict]:
    """
    Optimierungsloop: liest gecachte 1h-CSVs (re-download wenn > max_days_cache alt),
    testet ST-Params × Trail0.3%, gibt Top-N Coins zurück.
    """
    from src.download_ohlcv import fetch_ohlcv
    import time as _t

    os.makedirs(_STL_DATA_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(_STL_DATA_DIR) if f.endswith(f"_{_STL_TIMEFRAME}.csv"))
    if not files:
        return []

    results: list[dict] = []
    total = len(files)

    for fi, fname in enumerate(files):
        coin = fname.replace(f"_{_STL_TIMEFRAME}.csv", "").split("_")[0]
        fpath = os.path.join(_STL_DATA_DIR, fname)

        # Re-download wenn Datei zu alt
        import time as _time_mod
        age_days = (_time_mod.time() - os.path.getmtime(fpath)) / 86400
        if age_days > max_days_cache:
            sym_raw = fname.replace(f"_{_STL_TIMEFRAME}.csv", "")
            symbol  = sym_raw.replace("_", "/", 1).replace("_", ":")
            try:
                df_new = fetch_ohlcv(symbol, _STL_TIMEFRAME, 180)
                df_new.to_csv(fpath, index=False)
            except Exception:
                pass

        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if len(df) < 100:
            continue

        opens  = df["open"].to_numpy(float)
        highs  = df["high"].to_numpy(float)
        lows   = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        capital = 10_000.0 / max(total, 1)

        best_r = None; best_score = -1e9
        best_period = 20; best_mult = 2.0

        for period, mult in _STL_ST_PARAMS:
            strat = SupertrendStrategy(period, mult)
            try:
                sigs = strat.generate_signals(df).to_numpy(int)
            except Exception:
                continue
            r = _stl_backtest_fast(opens, highs, lows, closes, sigs, capital)
            if not r or r.get("num_trades", 0) < 5:
                continue
            s = _stl_score(r)
            if s > best_score:
                best_score = s; best_r = r
                best_period = period; best_mult = mult

        if best_r and best_r.get("total_pnl_pct", 0) > 0:
            sym_parts = fname.replace(f"_{_STL_TIMEFRAME}.csv", "").split("_")
            symbol = f"{sym_parts[0]}/USDT:USDT"
            results.append({
                "symbol":   symbol,
                "coin":     coin,
                "period":   best_period,
                "mult":     best_mult,
                "pnl_pct":  round(best_r["total_pnl_pct"], 2),
                "winrate":  round(best_r["winrate_pct"], 1),
                "pf":       round(best_r["profit_factor"], 3),
                "max_dd":   round(best_r["max_drawdown_pct"], 2),
                "trades":   best_r["num_trades"],
                "score":    round(best_score, 2),
                "trailing": _STL_TRAILING,
            })

        if progress_fn:
            progress_fn((fi + 1) / total)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:_STL_TOP_N]


def _stl_build_chart(df: pd.DataFrame, period: int, mult: float,
                     symbol: str, trader_trades: list) -> go.Figure:
    """Baut Plotly-Chart: Kerzen + SuperTrend-Linie + Entry/Exit-Marker."""
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    opens  = df["open"].to_numpy(float)

    ts_ms  = df["timestamp"].astype(int).tolist()
    ts_sec = [t // 1000 for t in ts_ms]
    dt_idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    st_line, direction = _stl_compute_supertrend_arr(highs, lows, closes, period, mult)

    # Signale berechnen
    strat = SupertrendStrategy(period, mult)
    try:
        sigs = strat.generate_signals(df).to_numpy(int)
    except Exception:
        sigs = np.zeros(len(df), dtype=int)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.02,
        subplot_titles=("", "Volumen"),
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dt_idx, open=opens, high=highs, low=lows, close=closes,
        name="Preis",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── SuperTrend-Linie (grün = bullish, rot = bearish) ─────────────────────
    # Segmente nach Richtung splitten
    _prev_dir = direction[0]
    _seg_start = 0
    for _i in range(1, len(direction) + 1):
        _cur_dir = direction[_i] if _i < len(direction) else None
        if _cur_dir != _prev_dir or _i == len(direction):
            _color = "#26a69a" if _prev_dir == 1 else "#ef5350"
            _name  = "ST Bullish" if _prev_dir == 1 else "ST Bearish"
            fig.add_trace(go.Scatter(
                x=dt_idx[_seg_start:_i],
                y=st_line[_seg_start:_i],
                mode="lines",
                line=dict(color=_color, width=2),
                name=_name,
                showlegend=(_seg_start == 0),
            ), row=1, col=1)
            _seg_start = _i - 1
            _prev_dir  = _cur_dir

    # ── Entry-Signale ─────────────────────────────────────────────────────────
    long_idx  = np.where(sigs == 1)[0]
    short_idx = np.where(sigs == -1)[0]

    if len(long_idx):
        fig.add_trace(go.Scatter(
            x=dt_idx.iloc[long_idx],
            y=lows[long_idx] * 0.998,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#26a69a",
                        line=dict(color="white", width=1)),
            name="Long Signal",
        ), row=1, col=1)

    if len(short_idx):
        fig.add_trace(go.Scatter(
            x=dt_idx.iloc[short_idx],
            y=highs[short_idx] * 1.002,
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef5350",
                        line=dict(color="white", width=1)),
            name="Short Signal",
        ), row=1, col=1)

    # ── Live-Trade Marker aus dem Trader ─────────────────────────────────────
    for _tr in (trader_trades or []):
        if _tr.get("symbol") != symbol:
            continue
        try:
            _et = pd.to_datetime(_tr["entry_time"])
            _xt = pd.to_datetime(_tr.get("exit_time") or _tr.get("timestamp"))
            _ep = float(_tr["entry_price"])
            _xp = float(_tr.get("exit_price") or _tr.get("entry_price"))
            _side = _tr.get("side", "long")
            _pnl  = _tr.get("pnl_pct", 0) or 0
            _col  = "#00e676" if _pnl > 0 else "#ff1744"
            fig.add_trace(go.Scatter(
                x=[_et, _xt], y=[_ep, _xp],
                mode="lines+markers",
                line=dict(color=_col, width=1.5, dash="dot"),
                marker=dict(size=6, color=_col),
                name=f"Trade {_side.upper()} ({_pnl:+.1f}%)",
                showlegend=False,
            ), row=1, col=1)
        except Exception:
            pass

    # ── Volumen ───────────────────────────────────────────────────────────────
    if "volume" in df.columns:
        _vol_colors = ["#26a69a" if c >= o else "#ef5350"
                       for c, o in zip(closes, opens)]
        fig.add_trace(go.Bar(
            x=dt_idx, y=df["volume"].astype(float),
            marker_color=_vol_colors, name="Volumen",
        ), row=2, col=1)

    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=24, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,17,0.9)",
        font=dict(color="#e0e0e0"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)
    return fig


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_manual_balance() -> float:
    import time as _t, hmac as _hmac, hashlib as _hl
    import requests as _req
    from src.config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_API_URL
    _url = (BYBIT_API_URL or "https://api-demo.bybit.com").rstrip("/")
    _ts  = str(int(_t.time() * 1000))
    _rw  = "5000"
    _q   = "accountType=UNIFIED"
    _pre = f"{_ts}{BYBIT_API_KEY}{_rw}{_q}"
    _sig = _hmac.new(BYBIT_API_SECRET.encode(), _pre.encode(), _hl.sha256).hexdigest()
    _hdr = {"X-BAPI-API-KEY": BYBIT_API_KEY, "X-BAPI-TIMESTAMP": _ts,
            "X-BAPI-SIGN": _sig, "X-BAPI-RECV-WINDOW": _rw}
    _r = _req.get(f"{_url}/v5/account/wallet-balance",
                  params={"accountType": "UNIFIED"}, headers=_hdr, timeout=8)
    _d = _r.json()
    if _d.get("retCode") != 0:
        return 0.0
    _coins = _d["result"]["list"][0]["coin"] if _d["result"]["list"] else []
    _usdt  = next((c for c in _coins if c["coin"] == "USDT"), None)
    def _f(v):
        try: return float(v) if v not in (None, "", "N/A") else 0.0
        except: return 0.0
    return _f(_usdt["walletBalance"]) if _usdt else 0.0


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_ticker_price(symbol: str) -> float:
    try:
        import requests as _r
        # Bybit public ticker — kein Auth, kein ccxt-Overhead
        _sym = symbol.replace("/", "").replace(":USDT", "")
        _resp = _r.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": _sym},
            timeout=3,
        )
        _data = _resp.json()
        return float(_data["result"]["list"][0]["lastPrice"])
    except Exception:
        return 0.0


def _render_manual_tab(trader, status):
    """Inhalt des Manuell-Handeln-Tabs."""
    balance = _fetch_manual_balance()

    # ── Balance-Anzeige ───────────────────────────────────────────────────────
    bm1, bm2, bm3 = st.columns(3)
    bm1.metric("Kontostand (USDT)", f"{balance:,.2f}" if balance else "Laden...")
    manual_positions = status.get("manual_positions", [])
    total_margin_used = sum(p["notional"] / p["leverage"] for p in manual_positions)
    bm2.metric("Margin in Positionen", f"{total_margin_used:,.2f} USDT")
    bm3.metric("Offene Positionen", len(manual_positions))

    rb1, rb2 = st.columns(2)
    if rb1.button("Aktualisieren", key="man_refresh_bal"):
        st.cache_data.clear()
        st.rerun()
    if rb2.button("Sync mit Exchange", key="man_sync", help="Prüft welche Positionen auf Bybit noch offen sind"):
        removed = trader.sync_manual_positions()
        st.cache_data.clear()
        if removed:
            st.success(f"{removed} extern geschlossene Position(en) entfernt.")
        else:
            st.info("Alles synchron.")
        st.rerun()

    st.divider()

    # ── Order-Formular ────────────────────────────────────────────────────────
    st.markdown("**Neue Position eröffnen**")
    mf1, mf2 = st.columns([1, 1])

    with mf1:
        man_symbol = st.selectbox(
            "Symbol", _LT_SYMBOLS, key="man_sym"
        )
        man_side = st.radio(
            "Richtung", ["Long", "Short"], horizontal=True, key="man_side"
        )
        man_pct = st.slider(
            "Margin (% vom Kontostand)", 1, 50, 10, key="man_pct",
            help="Wieviel % deines Kontostands als Margin einsetzen"
        )
        man_lev = st.slider("Hebel (x)", 1, 50, 5, key="man_lev")

    with mf2:
        man_mode = st.radio(
            "TP/SL Modus", ["ROI %", "ATR Auto"], horizontal=True, key="man_mode",
        )

        if man_mode == "ROI %":
            man_tp = st.slider(
                "Take Profit (% ROI)", 0.0, 500.0, 10.0, 1.0, key="man_tp",
                help="ROI auf die eingesetzte Margin. 0 = kein TP"
            )
            man_sl = st.slider(
                "Stop Loss (% ROI)", 0.0, 500.0, 5.0, 1.0, key="man_sl",
                help="ROI-Verlust auf die eingesetzte Margin. 0 = kein SL"
            )
            man_atr_mode    = False
            man_atr_period  = 14
            man_atr_sl_mult = 1.5
            man_atr_rr      = 2.0
            man_trailing    = False
        else:  # ATR Auto
            man_atr_period  = st.number_input("ATR Periode", 5, 50, 14, key="man_atr_p")
            man_atr_sl_mult = st.slider("SL Multiplikator", 0.5, 5.0, 1.5, 0.1, key="man_atr_slm",
                                        help="SL Abstand = ATR × Multiplikator")
            man_atr_rr      = st.slider("R:R Verhältnis", 0.5, 5.0, 2.0, 0.1, key="man_atr_rr",
                                        help="TP Abstand = SL Abstand × R:R")
            man_trailing    = st.checkbox(
                "Moving Take Profit (Trailing Stop)", key="man_trailing", value=True,
                help="Bybit-nativer Trailing Stop — SL bewegt sich automatisch mit dem Preis mit"
            )
            man_atr_mode    = True
            man_tp          = 0.0
            man_sl          = 0.0

        # Vorschau-Rechnung
        margin_usdt   = balance * man_pct / 100
        notional_usdt = margin_usdt * man_lev
        cur_price     = _fetch_ticker_price(man_symbol)
        amount_base   = notional_usdt / cur_price if cur_price > 0 else 0.0

        st.markdown("**Vorschau**")
        if man_mode == "ROI %":
            _tp_price_prev = cur_price * (1 + (man_tp / 100) / man_lev) if cur_price > 0 and man_tp > 0 else None
            _sl_price_prev = cur_price * (1 - (man_sl / 100) / man_lev) if cur_price > 0 and man_sl > 0 else None
            _tp_prev_str  = f"`{_tp_price_prev:,.2f} USDT`" if _tp_price_prev else "`–`"
            _sl_prev_str  = f"`{_sl_price_prev:,.2f} USDT`" if _sl_price_prev else "`–`"
            _trl_prev_str = "`–`"
        else:
            _tp_prev_str  = "`bei Order berechnet`"
            _sl_prev_str  = "`bei Order berechnet`"
            _trl_prev_str = "`aktiv`" if man_trailing else "`–`"

        st.markdown(f"""
| | |
|---|---|
| Aktueller Preis | `{cur_price:,.4f} USDT` |
| Margin | `{margin_usdt:,.2f} USDT` |
| Positionsgröße | `{notional_usdt:,.2f} USDT` |
| Menge | `{amount_base:.6f}` |
| Modus | `Isolated  {man_lev}x` |
| TP-Preis | {_tp_prev_str} |
| SL-Preis | {_sl_prev_str} |
| Trailing Stop | {_trl_prev_str} |
""")

    bc1, bc2 = st.columns([1, 3])
    with bc1:
        open_btn = st.button(
            f"{'📈 Long' if man_side == 'Long' else '📉 Short'} eröffnen",
            type="primary", use_container_width=True, key="man_open"
        )

    if open_btn:
        if balance <= 0:
            st.error("Balance konnte nicht abgerufen werden.")
        else:
            with st.spinner("Order wird gesendet…"):
                ok, msg = trader.open_manual_position(
                    symbol         = man_symbol,
                    side           = man_side.lower(),
                    pct_of_balance = man_pct / 100,
                    leverage       = man_lev,
                    tp_pct         = man_tp / 100 if man_tp > 0 else None,
                    sl_pct         = man_sl / 100 if man_sl > 0 else None,
                    atr_mode       = man_atr_mode,
                    atr_period     = int(man_atr_period),
                    atr_sl_mult    = float(man_atr_sl_mult),
                    atr_rr         = float(man_atr_rr),
                    use_trailing   = man_trailing,
                )
            if ok:
                st.success(f"Position eröffnet: {msg}")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Fehler: {msg}")

    st.divider()

    # ── Offene Positionen ─────────────────────────────────────────────────────
    st.markdown("**Offene Positionen**")
    manual_positions = status.get("manual_positions", [])

    if not manual_positions:
        st.info("Keine offenen Positionen.")
    else:
        for _pos_idx, pos in enumerate(manual_positions):
            cur = _fetch_ticker_price(pos["symbol"])
            if pos["side"] == "long":
                upnl = (cur - pos["entry_price"]) / pos["entry_price"] * pos["notional"] if cur else 0.0
            else:
                upnl = (pos["entry_price"] - cur) / pos["entry_price"] * pos["notional"] if cur else 0.0
            margin = pos["notional"] / pos["leverage"]
            upnl_pct = (upnl / margin * 100) if margin else 0.0
            side_icon = "📈" if pos["side"] == "long" else "📉"
            pnl_color = "green" if upnl >= 0 else "red"

            with st.container(border=True):
                pc1, pc2, pc3, pc4, pc5 = st.columns([2, 2, 2, 2, 1])
                pc1.metric("Symbol", f"{side_icon} {pos['symbol'].split('/')[0]}")
                pc2.metric("Seite / Hebel", f"{pos['side'].upper()}  {pos['leverage']}x")
                pc3.metric("Entry", f"{pos['entry_price']:,.4f}")
                pc4.metric("uPnL", f"{upnl:+,.2f} USDT",
                           delta=f"{upnl_pct:+.2f}%",
                           delta_color="normal" if upnl >= 0 else "inverse")
                with pc5:
                    close_key = f"man_close_{pos['symbol']}_{pos['side']}_{_pos_idx}"
                    if st.button("Schliessen", key=close_key, use_container_width=True):
                        with st.spinner("Schliesse Position…"):
                            ok, msg = trader.close_manual_position(pos["symbol"], pos["side"])
                        if ok:
                            st.success(msg)
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(msg)

                tp_str  = f"{pos['tp_price']:,.4f}"       if pos.get("tp_price")       else "–"
                sl_str  = f"{pos['sl_price']:,.4f}"       if pos.get("sl_price")       else "–"
                trl_str = f"{pos['trailing_distance']:,.2f}" if pos.get("trailing_distance") else "–"
                st.caption(
                    f"Notional: {pos['notional']:,.2f} USDT  |  "
                    f"TP: {tp_str}  |  SL: {sl_str}  |  Trailing: {trl_str}  |  Eröffnet: {pos['opened_at']}"
                )

    st.divider()

    # ── Trade-Verlauf (manuell) ───────────────────────────────────────────────
    st.markdown("**Abgeschlossene Trades**")
    manual_trades = status.get("manual_trades", [])
    if manual_trades:
        mt_df = pd.DataFrame(manual_trades)
        shown = ["symbol","side","entry","exit","notional","leverage","pnl_usdt","pnl_pct","opened_at","closed_at"]
        mt_df = mt_df[[c for c in shown if c in mt_df.columns]]
        mt_df.columns = [c.replace("_", " ").title() for c in mt_df.columns]

        def _mc(v):
            if isinstance(v, (int, float)):
                if v > 0: return "color: #2ecc71"
                if v < 0: return "color: #e74c3c"
            return ""

        pnl_cols = [c for c in mt_df.columns if "Pnl" in c]
        st.dataframe(
            mt_df.style.map(_mc, subset=pnl_cols).format({c: "{:+.4f}" for c in pnl_cols}),
            use_container_width=True, hide_index=True, height=260,
        )
    else:
        st.info("Noch keine abgeschlossenen Trades.")

def _render_bot_tab_impl(trader, status, is_running):

    # ── Session-State aus aktiver Konfiguration vorbelegen ────────────────────
    _LT_CLS_MAP = {
        "EMACrossStrategy":    "EMA Cross",
        "RSIStrategy":         "RSI",
        "BollingerStrategy":   "Bollinger",
        "BreakoutStrategy":    "Breakout",
        "MACDStrategy":        "MACD",
        "SupertrendStrategy":  "Supertrend",
        "MeanRevStrategy":     "Bollinger",
        "TrendFollowStrategy": "EMA Cross",
    }
    if (is_running or not st.session_state.get("_lt_initialized")) and status.get("strategy_class"):
        st.session_state["_lt_initialized"] = True
        _cls    = status["strategy_class"]
        _params = status.get("strategy_params", {})
        st.session_state["lt_strat"] = _LT_CLS_MAP.get(_cls, "Supertrend")
        if _cls == "EMACrossStrategy":
            st.session_state["lt_ema_fast"] = int(_params.get("fast_period", 20))
            st.session_state["lt_ema_slow"] = int(_params.get("slow_period", 50))
        elif _cls == "RSIStrategy":
            st.session_state["lt_rsi_p"]  = int(_params.get("period", 14))
            st.session_state["lt_rsi_os"] = int(_params.get("oversold", 30))
            st.session_state["lt_rsi_ob"] = int(_params.get("overbought", 70))
        elif _cls == "BollingerStrategy":
            st.session_state["lt_bb_p"]   = int(_params.get("period", 20))
            st.session_state["lt_bb_std"] = float(_params.get("std_dev", 2.0))
            st.session_state["lt_bb_st"]  = bool(_params.get("use_supertrend_filter", False))
        elif _cls == "BreakoutStrategy":
            st.session_state["lt_bo_lb"]  = int(_params.get("lookback", 50))
        elif _cls == "MACDStrategy":
            st.session_state["lt_macd_f"]  = int(_params.get("fast", 12))
            st.session_state["lt_macd_s"]  = int(_params.get("slow", 26))
            st.session_state["lt_macd_si"] = int(_params.get("signal", 9))
        else:
            st.session_state["lt_st_atr"]  = int(_params.get("atr_period", 10))
            st.session_state["lt_st_mult"] = float(_params.get("multiplier", 3.0))
        _syms = status.get("symbols", [status.get("symbol", "BTC/USDT:USDT")])
        _tf   = status.get("timeframe", "15m")
        st.session_state["lt_syms"] = [s for s in _syms if s in _LT_SYMBOLS] or ["BTC/USDT:USDT"]
        if _tf in _LT_TIMEFRAMES:
            st.session_state["lt_tf"] = _tf
        st.session_state["lt_lev"]  = int(status.get("leverage", 5))
        st.session_state["lt_size"] = int((status.get("position_size") or 0.10) * 100)
        st.session_state["lt_tp"]   = round((status.get("tp_pct") or 0.0) * 100, 1)
        st.session_state["lt_sl"]   = round((status.get("sl_pct") or 0.0) * 100, 1)
        if status.get("atr_mode"):
            st.session_state["lt_mode"]    = "ATR Auto"
            st.session_state["lt_atr_p"]   = int(status.get("atr_period", 14))
            st.session_state["lt_atr_slm"] = float(status.get("atr_sl_mult", 1.5))
            st.session_state["lt_atr_rr"]  = float(status.get("atr_rr", 2.0))
            st.session_state["lt_trailing"] = bool(status.get("use_trailing", True))
        else:
            st.session_state["lt_mode"] = "ROI %"
        st.session_state["lt_mtf_on"]  = bool(status.get("mtf_enabled", False))
        st.session_state["lt_mtf_ema"] = int(status.get("mtf_ema_period", 50))
        st.session_state["lt_adx_on"]      = bool(status.get("adx_enabled", False))
        st.session_state["lt_adx_th"]      = int(status.get("adx_threshold", 25))
        st.session_state["lt_adx_mode"]    = "Trending (ADX ≥)" if status.get("adx_require_trend", True) else "Ranging (ADX <)"
        st.session_state["lt_trigger_on"]    = bool(status.get("use_trigger_entry", True))
        st.session_state["lt_trigger_buf_l"] = float(status.get("trigger_buffer_long",  0.1))
        st.session_state["lt_trigger_buf_s"] = float(status.get("trigger_buffer_short", 0.1))
        st.session_state["lt_trigger_exp"]   = int(status.get("trigger_expiry_min", 5))

    # ── Preset-Buttons ────────────────────────────────────────────────────────
    _BB_SCALPING_SYMBOLS = [
        "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
        "ADA/USDT:USDT", "DOT/USDT:USDT", "XRP/USDT:USDT",
    ]
    _TREND_SYMBOLS = ["XRP/USDT:USDT", "DOT/USDT:USDT"]

    if not is_running:
        _pc1, _pc2 = st.columns(2)
        with _pc1:
            if st.button("📈 MeanRev Preset  (ADA · 1h · BB10/2.0σ · ADX<20)",
                         use_container_width=True, key="lt_preset_mr"):
                st.session_state["lt_strat"]    = "Bollinger"
                st.session_state["lt_bb_p"]     = 10
                st.session_state["lt_bb_std"]   = 2.0
                st.session_state["lt_bb_st"]    = False
                st.session_state["lt_syms"]     = [s for s in ["ADA/USDT:USDT"] if s in _LT_SYMBOLS]
                st.session_state["lt_tf"]       = "1h"
                st.session_state["lt_lev"]      = 3
                st.session_state["lt_size"]     = 10
                st.session_state["lt_mode"]     = "ATR Auto"
                st.session_state["lt_atr_p"]    = 14
                st.session_state["lt_atr_slm"]  = 3.0
                st.session_state["lt_atr_rr"]   = 2.0
                st.session_state["lt_trailing"] = False
                st.session_state["lt_mtf_on"]   = False
                st.session_state["lt_adx_on"]   = True
                st.session_state["lt_adx_th"]   = 20
                st.session_state["lt_adx_mode"] = "Ranging (ADX <)"
                st.session_state["lt_portfolio_mode"] = False
                st.rerun()
        with _pc2:
            if st.button("🚀 TrendFollow Preset  (XRP+DOT · 15m · EMA20/100 · ADX≥25)",
                         use_container_width=True, key="lt_preset_tf"):
                st.session_state["lt_strat"]    = "EMA Cross"
                st.session_state["lt_ema_fast"] = 20
                st.session_state["lt_ema_slow"] = 100
                st.session_state["lt_syms"]     = [s for s in _TREND_SYMBOLS if s in _LT_SYMBOLS]
                st.session_state["lt_tf"]       = "15m"
                st.session_state["lt_lev"]      = 3
                st.session_state["lt_size"]     = 10
                st.session_state["lt_mode"]     = "ATR Auto"
                st.session_state["lt_atr_p"]    = 14
                st.session_state["lt_atr_slm"]  = 3.0
                st.session_state["lt_atr_rr"]   = 3.0
                st.session_state["lt_trailing"] = False
                st.session_state["lt_mtf_on"]   = False
                st.session_state["lt_adx_on"]   = True
                st.session_state["lt_adx_th"]   = 25
                st.session_state["lt_adx_mode"] = "Trending (ADX ≥)"
                st.session_state["lt_portfolio_mode"] = False
                st.rerun()

        # ── Portfolio-Preset: alle optimalen Strategien auf 15m ──────────────
        if st.button(
            "🎯 Portfolio Preset  (6 Coins · 15m · TrendFollow · ATR×3.0)",
            use_container_width=True, key="lt_preset_portfolio",
            help="Alle 6 Coins → TrendFollow EMA20/100 ADX≥25 (kein Trailing — Baseline ist bestes Setup laut Backtest)",
        ):
            _portfolio_syms = [
                "ADA/USDT:USDT", "XRP/USDT:USDT", "DOT/USDT:USDT",
                "AVAX/USDT:USDT", "DOGE/USDT:USDT", "BNB/USDT:USDT",
            ]
            st.session_state["lt_syms"]           = [s for s in _portfolio_syms if s in _LT_SYMBOLS]
            st.session_state["lt_tf"]             = "15m"
            st.session_state["lt_lev"]            = 3
            st.session_state["lt_size"]           = 10
            st.session_state["lt_mode"]           = "ATR Auto"
            st.session_state["lt_atr_p"]          = 14
            st.session_state["lt_atr_slm"]        = 3.0
            st.session_state["lt_atr_rr"]         = 2.5
            st.session_state["lt_trailing"]       = False
            st.session_state["lt_mtf_on"]         = False
            st.session_state["lt_adx_on"]         = False
            st.session_state["lt_portfolio_mode"] = True
            st.rerun()

    # ── Layout ────────────────────────────────────────────────────────────────
    lt_left, lt_right = st.columns([1, 1])

    # ── Linke Spalte: Konfiguration ───────────────────────────────────────────
    with lt_left:
        st.markdown("**Strategie & Parameter**")

        cfg_dis = is_running   # Felder sperren wenn Trader läuft

        lt_strat_name = st.selectbox(
            "Strategie",
            ["EMA Cross", "RSI", "Bollinger", "Breakout", "MACD", "Supertrend"],
            key="lt_strat", disabled=cfg_dis,
        )

        if lt_strat_name == "EMA Cross":
            lt_fast = st.number_input("Fast EMA", 2, 500, 20, key="lt_ema_fast", disabled=cfg_dis)
            lt_slow = st.number_input("Slow EMA", 3, 1000, 50, key="lt_ema_slow", disabled=cfg_dis)
        elif lt_strat_name == "RSI":
            lt_rsi_p  = st.number_input("RSI Period",  2,  200, 14, key="lt_rsi_p",  disabled=cfg_dis)
            lt_rsi_os = st.slider("Oversold",  1, 49, 30, key="lt_rsi_os", disabled=cfg_dis)
            lt_rsi_ob = st.slider("Overbought", 51, 99, 70, key="lt_rsi_ob", disabled=cfg_dis)
        elif lt_strat_name == "Bollinger":
            lt_bb_p   = st.number_input("Period", 2, 500, 20,  key="lt_bb_p",   disabled=cfg_dis)
            lt_bb_std = st.slider("Std Dev", 0.5, 5.0, 2.0, 0.1, key="lt_bb_std", disabled=cfg_dis)
            lt_bb_st_filter = st.checkbox(
                "Supertrend-Filter (intern)", value=False, key="lt_bb_st", disabled=cfg_dis,
                help="Nur BB-Signale die mit dem Supertrend übereinstimmen. "
                     "Deaktiviert lassen wenn MTF+ADX aktiv sind — der Live-ST-Check bleibt immer aktiv.",
            )
        elif lt_strat_name == "Breakout":
            lt_bo_lb  = st.number_input("Lookback", 2, 1000, 50, key="lt_bo_lb", disabled=cfg_dis)
        elif lt_strat_name == "MACD":
            lt_mf  = st.number_input("Fast EMA", 2, 100, 12, key="lt_macd_f",  disabled=cfg_dis)
            lt_ms  = st.number_input("Slow EMA", 3, 200, 26, key="lt_macd_s",  disabled=cfg_dis)
            lt_msi = st.number_input("Signal",   2,  50,  9, key="lt_macd_si", disabled=cfg_dis)
        else:  # Supertrend
            lt_st_atr  = st.number_input("ATR Periode",   2, 100, 10,  key="lt_st_atr",  disabled=cfg_dis)
            lt_st_mult = st.slider("Multiplikator", 1.0, 6.0, 3.0, 0.1, key="lt_st_mult", disabled=cfg_dis)

        # Fallback für Nicht-Bollinger-Strategien
        if lt_strat_name != "Bollinger":
            lt_bb_st_filter = False

        st.divider()
        st.markdown("**Handels-Einstellungen**")
        lt_symbols = st.multiselect(
            "Symbole", _LT_SYMBOLS, default=["BTC/USDT:USDT"], key="lt_syms", disabled=cfg_dis,
            help="Strategie läuft gleichzeitig auf allen gewählten Symbolen",
        )
        if not lt_symbols:
            st.warning("Mindestens ein Symbol wählen.")
            lt_symbols = ["BTC/USDT:USDT"]
        lt_tf      = st.selectbox("Timeframe", _LT_TIMEFRAMES,  key="lt_tf",   disabled=cfg_dis, index=3)
        lt_lev     = st.slider("Hebel",            1, 20, 5,  key="lt_lev",  disabled=cfg_dis)
        lt_size    = st.slider("Positionsgröße (%)", 1, 50, 10, key="lt_size", disabled=cfg_dis) / 100
        lt_mode = st.radio(
            "TP/SL Modus", ["ROI %", "ATR Auto"], horizontal=True,
            key="lt_mode", disabled=cfg_dis,
        )
        if lt_mode == "ROI %":
            lt_tp = st.slider("Take Profit (% ROI)", 0.0, 200.0, 20.0, 1.0, key="lt_tp", disabled=cfg_dis,
                               help="ROI auf die Margin. 0 = kein TP")
            lt_sl = st.slider("Stop Loss (% ROI)",   0.0, 200.0, 10.0, 1.0, key="lt_sl", disabled=cfg_dis,
                               help="ROI-Verlust auf die Margin. 0 = kein SL")
            _be_col, _tr_col = st.columns(2)
            with _be_col:
                lt_use_be = st.checkbox(
                    "Breakeven SL", key="lt_use_be", disabled=cfg_dis,
                    help="SL wird auf Entry verschoben sobald X% des TP-Abstands erreicht sind",
                )
            with _tr_col:
                lt_use_trail_pct = st.checkbox(
                    "Trailing SL (%)", key="lt_use_trail_pct", disabled=cfg_dis,
                    help="Bybit-nativer Trailing Stop als % des Einstiegspreises",
                )
            if lt_use_be:
                lt_be_trigger = st.slider(
                    "Breakeven-Trigger (% zum TP)", 10, 90, 50, 5,
                    key="lt_be_trigger", disabled=cfg_dis,
                    help="Sobald der Preis X% des Weges zum TP zurückgelegt hat → SL → Entry",
                ) / 100
            else:
                lt_be_trigger = None
            if lt_use_trail_pct:
                lt_trail_pct = st.number_input(
                    "Trailing-Abstand (% des Kurses)", min_value=0.1, max_value=5.0,
                    value=float(st.session_state.get("lt_trail_pct_val", 0.8)),
                    step=0.1, format="%.1f", key="lt_trail_pct_val", disabled=cfg_dis,
                    help="z.B. 0.8 = SL bewegt sich 0.8% unter dem Höchstkurs",
                ) / 100
            else:
                lt_trail_pct = None
            lt_atr_mode     = False
            lt_atr_period   = 14
            lt_atr_sl_mult  = 1.5
            lt_atr_rr       = 2.0
            lt_use_trailing = False
        else:  # ATR Auto
            lt_tp           = 0.0
            lt_sl           = 0.0
            lt_atr_mode     = True
            lt_be_trigger   = None
            lt_trail_pct    = None
            lt_atr_period   = st.number_input("ATR Periode", 5, 50, 14, key="lt_atr_p",   disabled=cfg_dis)
            lt_atr_sl_mult  = st.slider("SL Multiplikator", 0.5, 5.0, 1.5, 0.1, key="lt_atr_slm", disabled=cfg_dis,
                                         help="SL Abstand = ATR × Multiplikator")
            lt_atr_rr       = st.slider("R:R Verhältnis",   0.5, 5.0, 2.0, 0.1, key="lt_atr_rr",  disabled=cfg_dis,
                                         help="TP Abstand = SL Abstand × R:R")
            lt_use_trailing = st.checkbox(
                "Moving Take Profit (Trailing Stop)", key="lt_trailing", disabled=cfg_dis, value=True,
                help="Bybit-nativer Trailing Stop — SL bewegt sich automatisch mit dem Preis mit"
            )

        st.divider()
        st.markdown("**Signal-Filter**")
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            lt_mtf_on = st.checkbox("MTF-Filter (15m + 1h EMA)", value=False,
                                    key="lt_mtf_on", disabled=cfg_dis,
                                    help="Signale nur in Richtung des höheren Timeframe-Trends")
        with _fc2:
            lt_adx_on = st.checkbox("ADX-Filter", value=False,
                                    key="lt_adx_on", disabled=cfg_dis,
                                    help="Signale nach ADX-Bedingung filtern (Trending oder Ranging)")
        if lt_mtf_on:
            lt_mtf_ema = st.slider("MTF EMA Periode", 10, 200, 50, key="lt_mtf_ema", disabled=cfg_dis)
        else:
            lt_mtf_ema = st.session_state.get("lt_mtf_ema", 50)
        if lt_adx_on:
            _adx_cols = st.columns(2)
            with _adx_cols[0]:
                lt_adx_mode_sel = st.selectbox(
                    "ADX Modus",
                    ["Ranging (ADX <)", "Trending (ADX ≥)"],
                    index=0 if st.session_state.get("lt_adx_mode", "Ranging (ADX <)") == "Ranging (ADX <)" else 1,
                    key="lt_adx_mode", disabled=cfg_dis,
                    help="Ranging: Signale nur wenn kein Trend (MeanRev) · Trending: Signale nur bei starkem Trend (TrendFollow)",
                )
            with _adx_cols[1]:
                lt_adx_th = st.slider("ADX Schwelle", 15, 40, 20, key="lt_adx_th", disabled=cfg_dis)
            lt_adx_require_trend = (lt_adx_mode_sel == "Trending (ADX ≥)")
        else:
            lt_adx_th = st.session_state.get("lt_adx_th", 20)
            lt_adx_require_trend = st.session_state.get("lt_adx_mode", "Ranging (ADX <)") == "Trending (ADX ≥)"

        st.divider()
        st.markdown("**Entry-Ausführung**")
        lt_trigger_on = st.checkbox(
            "Conditional Trigger Entry",
            value=True, key="lt_trigger_on", disabled=cfg_dis,
            help=(
                "Statt sofortiger Market Order wird eine Stop-Market Order gesetzt. "
                "Bybit triggert den Entry automatisch wenn der Preis das Hoch (Long) "
                "oder Tief (Short) der Signalkerze + Buffer erreicht. "
                "Verhindert schlechte Entries bei bereits gelaufenen Moves."
            ),
        )
        if lt_trigger_on:
            _tc1, _tc2, _tc3 = st.columns(3)
            with _tc1:
                lt_trigger_buf_l = st.number_input(
                    "Buffer Long (%)", min_value=0.0, max_value=2.0,
                    value=st.session_state.get("lt_trigger_buf_l", 0.1),
                    step=0.05, format="%.2f", key="lt_trigger_buf_l",
                    disabled=cfg_dis,
                    help="Trigger = Candle-High × (1 + Buffer/100)",
                ) / 100
            with _tc2:
                lt_trigger_buf_s = st.number_input(
                    "Buffer Short (%)", min_value=0.0, max_value=2.0,
                    value=st.session_state.get("lt_trigger_buf_s", 0.1),
                    step=0.05, format="%.2f", key="lt_trigger_buf_s",
                    disabled=cfg_dis,
                    help="Trigger = Candle-Low × (1 - Buffer/100)",
                ) / 100
            with _tc3:
                lt_trigger_exp = st.number_input(
                    "Ablauf (min)", min_value=1, max_value=60,
                    value=st.session_state.get("lt_trigger_exp", 5),
                    step=1, key="lt_trigger_exp",
                    disabled=cfg_dis,
                    help="Trigger wird nach dieser Zeit automatisch gecancelt (1 Candle = 5 min)",
                )
        else:
            lt_trigger_buf_l = st.session_state.get("lt_trigger_buf_l", 0.1) / 100
            lt_trigger_buf_s = st.session_state.get("lt_trigger_buf_s", 0.1) / 100
            lt_trigger_exp   = st.session_state.get("lt_trigger_exp", 5)

        st.divider()

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_c1, btn_c2, btn_c3 = st.columns(3)

        with btn_c1:
            start_btn = st.button(
                "▶ Starten", type="primary",
                use_container_width=True, key="lt_start",
                disabled=is_running,
            )
        with btn_c2:
            stop_btn = st.button(
                "⏹ Stoppen",
                use_container_width=True, key="lt_stop",
                disabled=not is_running,
            )
        with btn_c3:
            _any_bot_pos = any(p for p in status.get("positions", {}).values() if p)
            close_btn = st.button(
                "🔴 Alle schließen",
                use_container_width=True, key="lt_close",
                disabled=(not is_running or not _any_bot_pos),
            )

        # ── Button-Aktionen ───────────────────────────────────────────────────
        if start_btn:
            try:
                if lt_strat_name == "EMA Cross":
                    lt_strategy = EMACrossStrategy(int(lt_fast), int(lt_slow))
                elif lt_strat_name == "RSI":
                    lt_strategy = RSIStrategy(int(lt_rsi_p), float(lt_rsi_os), float(lt_rsi_ob))
                elif lt_strat_name == "Bollinger":
                    lt_strategy = BollingerStrategy(
                        int(lt_bb_p), float(lt_bb_std),
                        use_supertrend_filter=lt_bb_st_filter,
                    )
                elif lt_strat_name == "Breakout":
                    lt_strategy = BreakoutStrategy(int(lt_bo_lb))
                elif lt_strat_name == "MACD":
                    lt_strategy = MACDStrategy(int(lt_mf), int(lt_ms), int(lt_msi))
                else:
                    lt_strategy = SupertrendStrategy(int(lt_st_atr), float(lt_st_mult))

                # Portfolio-Modus: per-Symbol-Strategien aus Backtest-Erkenntnissen
                _portfolio_mode = st.session_state.get("lt_portfolio_mode", False)
                _per_sym_strats: dict = {}
                if _portfolio_mode:
                    for _sym in lt_symbols:
                        _per_sym_strats[_sym] = TrendFollowStrategy(20, 100, 25.0)

                trader.configure(
                    strategy               = lt_strategy,
                    symbols                = lt_symbols,
                    timeframe              = lt_tf,
                    leverage               = lt_lev,
                    position_size          = lt_size,
                    tp_pct                 = lt_tp / 100 if lt_tp > 0 else None,
                    sl_pct                 = lt_sl / 100 if lt_sl > 0 else None,
                    atr_mode               = lt_atr_mode,
                    atr_period             = int(lt_atr_period),
                    atr_sl_mult            = float(lt_atr_sl_mult),
                    atr_rr                 = float(lt_atr_rr),
                    use_trailing           = lt_use_trailing,
                    breakeven_trigger_pct  = lt_be_trigger,
                    trailing_sl_pct        = lt_trail_pct,
                    mtf_enabled            = lt_mtf_on,
                    mtf_ema_period         = int(lt_mtf_ema),
                    adx_enabled            = lt_adx_on,
                    adx_threshold          = float(lt_adx_th),
                    adx_require_trend      = lt_adx_require_trend,
                    per_symbol_strategies  = _per_sym_strats if _portfolio_mode else None,
                    use_trigger_entry      = lt_trigger_on,
                    trigger_buffer_long    = float(lt_trigger_buf_l),
                    trigger_buffer_short   = float(lt_trigger_buf_s),
                    trigger_expiry_min     = int(lt_trigger_exp),
                )
                trader.start()
                st.success("Live-Trader gestartet.")
                st.rerun()
            except Exception as e:
                st.error(f"Start fehlgeschlagen: {e}")

        if stop_btn:
            trader.stop()
            st.warning("Live-Trader gestoppt.")
            st.rerun()

        if close_btn:
            with st.spinner("Schließe Position…"):
                trader.close_position_now()
            st.success("Position geschlossen.")
            st.rerun()

    # ── Rechte Spalte: Live-Status ────────────────────────────────────────────
    with lt_right:
        st.markdown("**Live-Status**")

        if is_running:
            st.success("⚡ **AKTIV**")
        else:
            st.warning("⏹ **GESTOPPT**")

        # Metriken-Reihe 1
        s1, s2, s3 = st.columns(3)
        eq     = status.get("equity")
        eq_ini = status.get("initial_equity")
        eq_str = f"{eq:,.2f} USDT" if eq else "–"
        if eq and eq_ini and eq_ini > 0:
            eq_delta = f"{(eq - eq_ini) / eq_ini * 100:+.2f}%"
            eq_color = "normal" if eq >= eq_ini else "inverse"
        else:
            eq_delta = None
            eq_color = "off"
        s1.metric("Kapital",    eq_str, eq_delta, delta_color=eq_color)
        _pss = status.get("per_symbol_strategies", {})
        if _pss:
            s2.metric("Strategie", f"Portfolio ({len(_pss)} Symbole)")
        else:
            s2.metric("Strategie",  status.get("strategy", "–"))
        s3.metric("Letzte Tick", status.get("last_tick") or "–")

        # Portfolio-Strategie-Übersicht
        if _pss:
            with st.expander("Portfolio-Strategien je Symbol"):
                for _s, _sc in _pss.items():
                    st.caption(f"`{_s}` → {_sc.get('str', _sc.get('class', '?'))}")

        # Signal
        sig_val = status.get("last_signal", 0)
        sig_str = "⬆ LONG" if sig_val == 1 else ("⬇ SHORT" if sig_val == -1 else "⬜ Neutral")
        st.info(f"Letztes Signal: **{sig_str}**")

        # Fehler
        if status.get("error"):
            st.error(f"Fehler: {status['error']}")

        # Offene Position
        st.divider()
        st.markdown("**Offene Positionen**")
        _bot_positions = status.get("positions", {})
        _any_open = any(p for p in _bot_positions.values() if p)
        if not _any_open:
            st.info("Keine offene Position.")
        else:
            for _bsym, _bpos in _bot_positions.items():
                if not _bpos:
                    continue
                with st.container(border=True):
                    _bp1, _bp2 = st.columns([3, 1])
                    _bp1.markdown(
                        f"**{'📈' if _bpos['side'] == 'long' else '📉'} "
                        f"{_bsym.split('/')[0]}** — "
                        f"{_bpos['side'].upper()}  {_bpos['leverage']}x"
                    )
                    with _bp2:
                        if st.button("Schliessen", key=f"lt_close_sym_{_bsym}",
                                     use_container_width=True):
                            with st.spinner(f"Schliesse {_bsym}…"):
                                trader.close_position_now(symbol=_bsym)
                            st.rerun()
                    _bc1, _bc2 = st.columns(2)
                    _bc1.metric("Entry", f"{_bpos['entry_price']:,.2f}")
                    _bc2.metric("Notional", f"{_bpos['notional']:,.2f} USDT")
                    _bc3, _bc4 = st.columns(2)
                    _bc3.metric("TP", f"{_bpos['tp_price']:,.2f}" if _bpos.get("tp_price") else "–")
                    _bc4.metric("SL", f"{_bpos['sl_price']:,.2f}" if _bpos.get("sl_price") else "–")
                    if _bpos.get("trailing_distance"):
                        st.caption(f"Trailing: {_bpos['trailing_distance']:,.2f} USDT Abstand")

        # Auto-Refresh
        st.divider()
        auto_ref = st.checkbox(
            "Auto-Refresh (alle 15s)", key="lt_autoref",
            help="Seite wird automatisch aktualisiert wenn der Trader läuft.",
        )
        if st.session_state.get("_lt_autoref_prev") != auto_ref:
            st.session_state["_lt_autoref_prev"] = auto_ref
            st.rerun()  # Full-Rerun → _ri in tab_live wird neu berechnet
        manual_ref = st.button("🔄 Jetzt aktualisieren", key="lt_refresh",
                               use_container_width=True)
        if manual_ref:
            st.rerun()

    # ── Trade-Verlauf ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Trade-Verlauf")

    trades_lt = status.get("trades", [])
    if trades_lt:
        lt_df = pd.DataFrame(trades_lt)
        shown_cols = ["index","symbol","side","entry","exit","notional","pnl_usdt","pnl_pct","reason","opened_at","closed_at"]
        lt_df = lt_df[[c for c in shown_cols if c in lt_df.columns]]
        lt_df.columns = [c.replace("_", " ").title() for c in lt_df.columns]

        def _lt_color(v):
            if isinstance(v, (int, float)):
                if v > 0: return "color: #2ecc71"
                if v < 0: return "color: #e74c3c"
            return ""

        pnl_cols = [c for c in lt_df.columns if "Pnl" in c]
        st.dataframe(
            lt_df.style.map(_lt_color, subset=pnl_cols)
                       .format({c: "{:+.4f}" for c in pnl_cols}),
            use_container_width=True, hide_index=True, height=280,
        )
    else:
        st.info("Noch keine abgeschlossenen Trades.")

    # ── Live-Chart ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Live-Chart")

    _lt_has_cfg = status.get("strategy_class") is not None
    if not _lt_has_cfg:
        st.info("Strategie konfigurieren und starten um den Chart zu sehen.")
    else:
        try:
            from streamlit_lightweight_charts import renderLightweightCharts

            _lc_syms_all = status.get("symbols", [status.get("symbol", "BTC/USDT:USDT")])
            _lc_col_sym, _lc_col_gap = st.columns([2, 5])
            _lc_sym = _lc_col_sym.selectbox(
                "Symbol", _lc_syms_all, key="lt_chart_sym",
            )
            _lc_tf     = status.get("timeframe", "15m")
            _lc_cls    = status["strategy_class"]
            _lc_params = status.get("strategy_params", {})

            # Kerzen aus Background-Thread (kein Netzwerkcall im UI-Thread)
            _lc_df = trader.candles.get(_lc_sym)
            if _lc_df is None:
                # Fallback: einmaliger Fetch wenn Bot noch nicht gelaufen ist
                with st.spinner("Lade Kerzen…"):
                    _lc_df = _fetch_live_candles(_lc_sym, _lc_tf, 500)

            # Unix-Timestamp in Sekunden (lightweight-charts Format)
            _lc_ts = (_lc_df["timestamp"] // 1000).astype(int).tolist()

            # Signale (mit denselben MTF- und ADX-Filtern wie der Bot)
            _lc_strategy = trader.strategy
            _lc_sigs = _lc_strategy.generate_signals(_lc_df).to_numpy().astype(int) if _lc_strategy else None

            if _lc_sigs is not None and status.get("mtf_enabled"):
                _mtf_period   = status.get("mtf_ema_period", 50)
                _primary_dt   = pd.to_datetime(_lc_df["datetime"]).reset_index(drop=True)
                _htf_tfs      = [("15m", 300), ("1h", 100)]
                _sigs_mtf     = _lc_sigs.copy()
                for _htf_tf, _htf_lim in _htf_tfs:
                    _hdf = _fetch_live_candles(_lc_sym, _htf_tf, _htf_lim)
                    if _hdf is None or len(_hdf) < 2:
                        continue
                    _hdt   = pd.to_datetime(_hdf["datetime"]).reset_index(drop=True)
                    _hema  = _hdf["close"].ewm(span=_mtf_period, adjust=False).mean()
                    _htrnd = np.where(_hdf["close"].to_numpy() >= _hema.to_numpy(), 1, -1)
                    _htf_frame   = pd.DataFrame({"time": _hdt,        "trend": _htrnd})
                    _pri_frame   = pd.DataFrame({"time": _primary_dt})
                    _merged      = pd.merge_asof(
                        _pri_frame.sort_values("time"), _htf_frame.sort_values("time"),
                        on="time", direction="backward",
                    ).sort_values("time").reset_index(drop=True)
                    _htf_trend   = _merged["trend"].fillna(0).to_numpy(int)
                    _mask        = ((_sigs_mtf ==  1) & (_htf_trend ==  1)) | \
                                   ((_sigs_mtf == -1) & (_htf_trend == -1))
                    _sigs_mtf    = np.where(_mask, _sigs_mtf, 0).astype(int)
                _lc_sigs = _sigs_mtf

            if _lc_sigs is not None and status.get("adx_enabled"):
                _adx_thr  = float(status.get("adx_threshold", 25))
                _adx_vals = compute_adx(_lc_df)
                _lc_sigs  = np.where(_adx_vals >= _adx_thr, _lc_sigs, 0).astype(int)

            # ── Kerzen-Daten ──────────────────────────────────────────────────
            _candles = [
                {"time": t, "open": float(o), "high": float(h),
                 "low": float(l), "close": float(c)}
                for t, o, h, l, c in zip(
                    _lc_ts,
                    _lc_df["open"], _lc_df["high"],
                    _lc_df["low"],  _lc_df["close"],
                )
            ]

            # ── Signal-Marker ─────────────────────────────────────────────────
            _markers = []
            if _lc_sigs is not None:
                for _i in range(len(_lc_sigs) - 1):
                    if _lc_sigs[_i] == 1:
                        _markers.append({"time": _lc_ts[_i + 1], "position": "belowBar",
                            "color": "#2ecc71", "shape": "arrowUp", "text": "L"})
                    elif _lc_sigs[_i] == -1:
                        _markers.append({"time": _lc_ts[_i + 1], "position": "aboveBar",
                            "color": "#e74c3c", "shape": "arrowDown", "text": "S"})

            # ── Trade-Marker (vergangene abgeschlossene Trades) ───────────────
            _ts_arr = np.array(_lc_ts)
            def _snap_to_candle(dt_str):
                if not dt_str:
                    return None
                try:
                    _ut = int(pd.Timestamp(dt_str, tz="UTC").timestamp())
                    _i  = int(np.searchsorted(_ts_arr, _ut, side="left"))
                    _i  = min(_i, len(_ts_arr) - 1)
                    if _i > 0 and abs(_ts_arr[_i - 1] - _ut) < abs(_ts_arr[_i] - _ut):
                        _i -= 1
                    return int(_ts_arr[_i])
                except Exception:
                    return None

            for _tr in trades_lt:
                if _tr.get("symbol") != _lc_sym:
                    continue
                _is_long  = _tr.get("side", "").upper() == "LONG"
                _entry_t  = _snap_to_candle(_tr.get("opened_at"))
                _exit_t   = _snap_to_candle(_tr.get("closed_at"))
                if _entry_t:
                    _markers.append({
                        "time":     _entry_t,
                        "position": "belowBar" if _is_long else "aboveBar",
                        "color":    "#2ecc71"  if _is_long else "#e74c3c",
                        "shape":    "arrowUp"  if _is_long else "arrowDown",
                        "text":     "E",
                    })
                if _exit_t:
                    _pnl = _tr.get("pnl_usdt")
                    _xc  = ("#2ecc71" if (_pnl is not None and _pnl > 0)
                            else "#aaaaaa" if _pnl is None
                            else "#e74c3c")
                    _markers.append({
                        "time":     _exit_t,
                        "position": "aboveBar" if _is_long else "belowBar",
                        "color":    _xc,
                        "shape":    "circle",
                        "text":     _tr.get("reason", "X")[:3],
                    })

            # lightweight-charts requires markers sorted by time
            _markers.sort(key=lambda m: m["time"])

            # ── Gemeinsame Chart-Optionen ─────────────────────────────────────
            _chart_opts = {
                "layout": {
                    "background": {"type": "solid", "color": "#0e1117"},
                    "textColor": "#d1d4dc",
                    "fontSize": 12,
                },
                "grid": {
                    "vertLines": {"color": "rgba(128,128,128,0.15)"},
                    "horzLines": {"color": "rgba(128,128,128,0.15)"},
                },
                "crosshair": {"mode": 1},
                "rightPriceScale": {"borderColor": "rgba(128,128,128,0.3)"},
                "timeScale": {
                    "borderColor": "rgba(128,128,128,0.3)",
                    "timeVisible": True,
                    "secondsVisible": False,
                    "rightOffset": 5,
                },
                "height": 480,
            }
            _candle_opts = {
                "upColor":         "#2ecc71",
                "downColor":       "#e74c3c",
                "borderUpColor":   "#2ecc71",
                "borderDownColor": "#e74c3c",
                "wickUpColor":     "#2ecc71",
                "wickDownColor":   "#e74c3c",
            }

            # ── Preischart-Serien aufbauen ────────────────────────────────────
            _price_series = [
                {"type": "Candlestick", "data": _candles,
                 "options": _candle_opts, "markers": _markers}
            ]

            # Position für das ausgewählte Chart-Symbol (Entry / TP / SL Linien)
            pos = status.get("positions", {}).get(_lc_sym)

            # Entry / TP / SL als Preislinien
            if pos:
                _ep_c = "#2ecc71" if pos["side"] == "long" else "#e74c3c"
                _price_series.append({"type": "Line", "data": [
                        {"time": _lc_ts[0], "value": pos["entry_price"]},
                        {"time": _lc_ts[-1], "value": pos["entry_price"]},
                    ], "options": {"color": _ep_c, "lineWidth": 1,
                                   "lineStyle": 2, "title": f"Entry {pos['entry_price']:,.0f}",
                                   "crosshairMarkerVisible": False, "lastValueVisible": True}})
                if pos.get("tp_price"):
                    _price_series.append({"type": "Line", "data": [
                            {"time": _lc_ts[0], "value": pos["tp_price"]},
                            {"time": _lc_ts[-1], "value": pos["tp_price"]},
                        ], "options": {"color": "#27ae60", "lineWidth": 1,
                                       "lineStyle": 1, "title": f"TP {pos['tp_price']:,.0f}",
                                       "crosshairMarkerVisible": False, "lastValueVisible": True}})
                if pos.get("sl_price"):
                    _price_series.append({"type": "Line", "data": [
                            {"time": _lc_ts[0], "value": pos["sl_price"]},
                            {"time": _lc_ts[-1], "value": pos["sl_price"]},
                        ], "options": {"color": "#c0392b", "lineWidth": 1,
                                       "lineStyle": 1, "title": f"SL {pos['sl_price']:,.0f}",
                                       "crosshairMarkerVisible": False, "lastValueVisible": True}})

            # ── Strategie-Overlays (Preischart) ──────────────────────────────
            def _line(data, color, width=1, title=""):
                clean = [{"time": t, "value": float(v)}
                         for t, v in zip(_lc_ts, data) if v == v and v != 0]
                return {"type": "Line", "data": clean,
                        "options": {"color": color, "lineWidth": width,
                                    "title": title, "crosshairMarkerVisible": False,
                                    "lastValueVisible": False}}

            _indicator_series = []   # für zweiten Chart (RSI/MACD)
            _has_indicator    = False

            if _lc_cls == "EMACrossStrategy":
                _fp = int(_lc_params.get("fast_period", 20))
                _sp = int(_lc_params.get("slow_period", 50))
                _price_series.append(_line(
                    _lc_df["close"].ewm(span=_fp, adjust=False).mean(),
                    "#3498db", 1, f"EMA{_fp}"))
                _price_series.append(_line(
                    _lc_df["close"].ewm(span=_sp, adjust=False).mean(),
                    "#e67e22", 1, f"EMA{_sp}"))

            elif _lc_cls == "BollingerStrategy":
                _bp  = int(_lc_params.get("period", 20))
                _bsd = float(_lc_params.get("std_dev", 2.0))
                _mid = _lc_df["close"].rolling(_bp).mean()
                _dev = _lc_df["close"].rolling(_bp).std(ddof=0)
                _price_series.append(_line(_mid + _bsd * _dev, "#9b59b6", 1, "BB+"))
                _price_series.append(_line(_mid,               "#9b59b6", 1, "BB"))
                _price_series.append(_line(_mid - _bsd * _dev, "#9b59b6", 1, "BB-"))

            elif _lc_cls == "SupertrendStrategy":
                _stp  = int(_lc_params.get("atr_period", 10))
                _stm  = float(_lc_params.get("multiplier", 3.0))
                _hi   = _lc_df["high"].to_numpy(float)
                _lo   = _lc_df["low"].to_numpy(float)
                _cl   = _lc_df["close"].to_numpy(float)
                _n    = len(_cl)
                _tr   = np.maximum(_hi - _lo,
                        np.maximum(np.abs(_hi - np.roll(_cl, 1)),
                                   np.abs(_lo  - np.roll(_cl, 1))))
                _atr  = np.zeros(_n)
                _atr[_stp - 1] = _tr[:_stp].mean()
                for _k in range(_stp, _n):
                    _atr[_k] = (_atr[_k-1] * (_stp - 1) + _tr[_k]) / _stp
                _hl2  = (_hi + _lo) / 2
                _bu   = _hl2 + _stm * _atr
                _bl   = _hl2 - _stm * _atr
                _dir  = np.ones(_n, dtype=int)
                _fu   = _bu.copy(); _fl = _bl.copy()
                _line_vals = np.zeros(_n)
                for _k in range(1, _n):
                    _fu[_k] = min(_bu[_k], _fu[_k-1]) if _cl[_k-1] <= _fu[_k-1] else _bu[_k]
                    _fl[_k] = max(_bl[_k], _fl[_k-1]) if _cl[_k-1] >= _fl[_k-1] else _bl[_k]
                    if   _dir[_k-1] ==  1 and _cl[_k] < _fl[_k]: _dir[_k] = -1
                    elif _dir[_k-1] == -1 and _cl[_k] > _fu[_k]: _dir[_k] =  1
                    else:                                           _dir[_k] =  _dir[_k-1]
                    _line_vals[_k] = _fl[_k] if _dir[_k] == 1 else _fu[_k]
                # Bullish/Bearish getrennt (NaN für Lücken)
                _bull = [float(v) if d == 1  else None for v, d in zip(_line_vals, _dir)]
                _bear = [float(v) if d == -1 else None for v, d in zip(_line_vals, _dir)]
                _price_series.append({"type": "Line",
                    "data": [{"time": t, "value": v} for t, v in zip(_lc_ts, _bull) if v is not None],
                    "options": {"color": "#2ecc71", "lineWidth": 2, "title": "ST↑",
                                "crosshairMarkerVisible": False, "lastValueVisible": False}})
                _price_series.append({"type": "Line",
                    "data": [{"time": t, "value": v} for t, v in zip(_lc_ts, _bear) if v is not None],
                    "options": {"color": "#e74c3c", "lineWidth": 2, "title": "ST↓",
                                "crosshairMarkerVisible": False, "lastValueVisible": False}})

            elif _lc_cls == "BreakoutStrategy":
                _lb = int(_lc_params.get("lookback", 50))
                _price_series.append(_line(
                    _lc_df["high"].shift(1).rolling(_lb).max(), "#27ae60", 1, f"H{_lb}"))
                _price_series.append(_line(
                    _lc_df["low"].shift(1).rolling(_lb).min(),  "#c0392b", 1, f"L{_lb}"))

            elif _lc_cls == "MACDStrategy":
                _mf  = int(_lc_params.get("fast", 12))
                _ms  = int(_lc_params.get("slow", 26))
                _msi = int(_lc_params.get("signal", 9))
                _mfast = _lc_df["close"].ewm(span=_mf,  adjust=False).mean()
                _mslow = _lc_df["close"].ewm(span=_ms,  adjust=False).mean()
                _mline = _mfast - _mslow
                _msig  = _mline.ewm(span=_msi, adjust=False).mean()
                _mhist = (_mline - _msig).tolist()
                _indicator_series = [
                    {"type": "Histogram", "data": [
                        {"time": t, "value": float(v),
                         "color": "#2ecc71" if v >= 0 else "#e74c3c"}
                        for t, v in zip(_lc_ts, _mhist) if v == v],
                     "options": {"priceScaleId": "macd"}},
                    _line(_mline, "#3498db", 1, "MACD"),
                    _line(_msig,  "#e67e22", 1, "Signal"),
                ]
                _has_indicator = True

            elif _lc_cls == "RSIStrategy":
                _rp  = int(_lc_params.get("period", 14))
                _ros = float(_lc_params.get("oversold", 30))
                _rob = float(_lc_params.get("overbought", 70))
                _rdelta = _lc_df["close"].diff()
                _rg  = _rdelta.clip(lower=0).ewm(com=_rp-1, adjust=False).mean()
                _rl  = (-_rdelta).clip(lower=0).ewm(com=_rp-1, adjust=False).mean()
                _rsi = 100 - 100 / (1 + _rg / _rl.replace(0, float("nan")))
                _indicator_series = [
                    _line(_rsi, "#3498db", 1, "RSI"),
                    # Overbought/Oversold als flache Linien
                    {"type": "Line", "data": [
                        {"time": _lc_ts[0], "value": _rob},
                        {"time": _lc_ts[-1], "value": _rob}],
                     "options": {"color": "#e74c3c", "lineWidth": 1, "lineStyle": 1,
                                 "crosshairMarkerVisible": False, "lastValueVisible": False}},
                    {"type": "Line", "data": [
                        {"time": _lc_ts[0], "value": _ros},
                        {"time": _lc_ts[-1], "value": _ros}],
                     "options": {"color": "#2ecc71", "lineWidth": 1, "lineStyle": 1,
                                 "crosshairMarkerVisible": False, "lastValueVisible": False}},
                ]
                _has_indicator = True

            # ── Rendern ───────────────────────────────────────────────────────
            _charts = [{"chart": _chart_opts, "series": _price_series}]
            if _has_indicator:
                _ind_opts = dict(_chart_opts)
                _ind_opts["height"] = 180
                _charts.append({"chart": _ind_opts, "series": _indicator_series})

            renderLightweightCharts(_charts, key="live_chart")
            st.caption("▲ Long-Signal  |  ▼ Short-Signal  |  Scroll & Zoom mit Maus/Trackpad")

        except Exception as _lc_err:
            st.warning(f"Chart konnte nicht geladen werden: {_lc_err}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_lines = status.get("log", [])
    with st.expander(f"📋 Log ({len(log_lines)} Einträge)", expanded=True):
        if log_lines:
            st.code("\n".join(log_lines), language=None)
        else:
            st.info("Kein Log-Eintrag vorhanden.")

    # Auto-Refresh wird durch run_every=_ri im Bot-Fragment in tab_live gesteuert


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – LIVE TRADING (Haupt-Container)
# ══════════════════════════════════════════════════════════════════════════════
if _page == "🤖 Live Trading":
    st.subheader("🖐 Manuell Handeln — Bybit Demo")

    @st.fragment
    def _manual_fragment():
        _t = _get_live_trader()
        _render_manual_tab(_t, _t.get_status())
    _manual_fragment()

# ══════════════════════════════════════════════════════════════════════════════
# OPTITEST – Multi-Coin Auto-Backtest Seite
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "🧪 OptiTest":
    st.subheader("🧪 OptiTest — Top-Coins Simultaneous Backtest")
    st.caption(
        "Lädt automatisch die meistgehandelten Bybit Perp-Coins · "
        "365 Tage 15m · 7 Strategien × 3 Hebel · automatische SL/TP + Trailing SL"
    )

    # ── Einstellungen ─────────────────────────────────────────────────────────
    _oc1, _oc2, _oc3, _oc4 = st.columns(4)
    with _oc1:
        _ot_limit      = st.slider("Max. Coins", 10, 500, 50, 10, key="ot_limit")
    with _oc2:
        _ot_pos_pct    = st.slider("Pos.-Größe (%)", 1, 20, 5, 1, key="ot_pos")
        _ot_pos_size   = _ot_pos_pct / 100
    with _oc3:
        _ot_trail_pct  = st.slider("Trailing SL (%)", 0.0, 3.0, 0.8, 0.1,
                                   format="%.1f%%", key="ot_trail")
        _ot_trailing   = _ot_trail_pct / 100
    with _oc4:
        _ot_min_trades = st.number_input("Min. Trades", 5, 100, 10, 5, key="ot_min")

    _ot_init_cap = st.number_input(
        "Startkapital gesamt (USDT)", 100.0, 1_000_000.0, 10_000.0, 1000.0, key="ot_cap"
    )

    _oa, _ob, _oc_ = st.columns([1, 1, 6])
    _ot_dl_btn  = _oa.button("⬇ Coins laden",  key="ot_dl",  use_container_width=True)
    _ot_run_btn = _ob.button("▶ Backtesten",   key="ot_run", use_container_width=True,
                              type="primary",
                              disabled="ot_syms" not in st.session_state)

    # Info welche Strategien getestet werden
    with st.expander("ℹ️ Getestete Strategien", expanded=False):
        st.markdown("\n".join(
            f"- **{lbl}** — TP {tp}% / SL {sl}% · Hebel 1×, 2×, 3×"
            for lbl, _, tp, sl in _OT_STRAT_DEFS
        ))

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    if _ot_dl_btn:
        os.makedirs(_OT_DIR, exist_ok=True)
        with st.spinner("Symbole von Bybit laden..."):
            _ot_symbols = _ot_fetch_symbols(_ot_limit)
        if not _ot_symbols:
            st.error("Keine Symbole von Bybit gefunden. Verbindung prüfen.")
        else:
            _ot_prog    = st.progress(0.0, text="Lade Kerzen...")
            _ot_status  = st.empty()
            _ot_ok: list[str] = []
            for _oti, _sym in enumerate(_ot_symbols):
                _csv_p = _ot_csv_path(_sym)
                if os.path.exists(_csv_p):
                    _ot_status.text(f"✓ {_sym} (gecacht)")
                    _ot_ok.append(_sym)
                else:
                    try:
                        _ot_status.text(f"⬇ {_sym} …")
                        _df_dl = fetch_ohlcv(_sym, "15m", 365)
                        _df_dl.to_csv(_csv_p, index=False)
                        _ot_ok.append(_sym)
                    except Exception as _e:
                        _ot_status.text(f"✗ {_sym}: {_e}")
                _ot_prog.progress((_oti + 1) / len(_ot_symbols),
                                  text=f"{_oti + 1}/{len(_ot_symbols)} Coins")
            st.session_state["ot_syms"] = _ot_ok
            _ot_prog.empty()
            _ot_status.empty()
            st.success(f"✅ {len(_ot_ok)} Coins heruntergeladen – jetzt Backtest starten.")
            st.rerun()

    # ── Backtest ──────────────────────────────────────────────────────────────
    if _ot_run_btn and "ot_syms" in st.session_state:
        _ot_syms       = st.session_state["ot_syms"]
        _ot_coin_cap   = _ot_init_cap / max(len(_ot_syms), 1)
        _ot_rows: list[dict]         = []
        _ot_eq_series: list[pd.Series] = []

        _ot_prog2  = st.progress(0.0, text="Backtest läuft…")
        _ot_status2 = st.empty()

        for _oti2, _sym2 in enumerate(_ot_syms):
            _ot_status2.text(f"⚙️  {_sym2} ({_oti2+1}/{len(_ot_syms)})")
            _df2 = _ot_load_df(_sym2)
            if _df2 is None or len(_df2) < 200:
                _ot_prog2.progress((_oti2 + 1) / len(_ot_syms))
                continue

            _best = _ot_fast_sweep(
                _df2, _ot_trailing, _ot_pos_size, int(_ot_min_trades), _ot_coin_cap
            )
            if _best is None:
                _ot_prog2.progress((_oti2 + 1) / len(_ot_syms))
                continue

            # Zeilen für Ergebnis-Tabelle
            _coin_short = _sym2.split("/")[0]
            _ot_rows.append({
                "Coin":       _coin_short,
                "Symbol":     _sym2,
                "Strategie":  _best["_label"],
                "Hebel":      f"{_best['_lev']}×",
                "TP / SL":    f"{_best['_tp']}% / {_best['_sl']}%",
                "OOS-PnL":    round(_best.get("total_pnl_pct", 0), 2),
                "Trades":     _best.get("num_trades", 0),
                "WR%":        round(_best.get("winrate_pct", 0), 1),
                "PF":         round(_best.get("profit_factor", 0) or 0, 2),
                "MaxDD%":     round(_best.get("max_drawdown_pct", 0), 2),
            })

            # Equity-Zeitreihe für kombinierten Chart (Full-Backtest)
            _eq_s = _ot_equity_series(_df2, _best["_strategy"], _best["_cfg"])
            if not _eq_s.empty:
                _ot_eq_series.append(_eq_s)

            _ot_prog2.progress((_oti2 + 1) / len(_ot_syms),
                               text=f"{_oti2+1}/{len(_ot_syms)}: {_coin_short}")

        _ot_prog2.empty()
        _ot_status2.empty()

        st.session_state["ot_results"]    = _ot_rows
        st.session_state["ot_eq_series"]  = _ot_eq_series
        st.session_state["ot_init_cap"]   = _ot_init_cap
        st.session_state["ot_coin_cap"]   = _ot_coin_cap
        st.rerun()

    # ── Ergebnisse anzeigen ───────────────────────────────────────────────────
    if "ot_results" in st.session_state and st.session_state["ot_results"]:
        _ot_rows2      = st.session_state["ot_results"]
        _ot_eq_all     = st.session_state.get("ot_eq_series", [])
        _ot_cap_used   = st.session_state.get("ot_init_cap", 10_000.0)
        _ot_coin_c     = st.session_state.get("ot_coin_cap", 500.0)
        _n_coins       = len(_ot_rows2)

        # ── Summary-Metriken ──────────────────────────────────────────────
        _avg_pnl  = sum(r["OOS-PnL"]  for r in _ot_rows2) / _n_coins
        _avg_pf   = sum(r["PF"]       for r in _ot_rows2) / _n_coins
        _avg_dd   = sum(r["MaxDD%"]   for r in _ot_rows2) / _n_coins
        _tot_tr   = sum(r["Trades"]   for r in _ot_rows2)
        _pos_coins= sum(1 for r in _ot_rows2 if r["OOS-PnL"] > 0)

        _sm1, _sm2, _sm3, _sm4, _sm5 = st.columns(5)
        _sm1.metric("Coins getestet",   _n_coins)
        _sm2.metric("Profitable Coins", f"{_pos_coins}/{_n_coins}")
        _sm3.metric("Ø PnL / Coin",     f"{_avg_pnl:+.2f}%",
                    delta_color="normal" if _avg_pnl >= 0 else "inverse")
        _sm4.metric("Ø Profit Factor",  f"{_avg_pf:.2f}")
        _sm5.metric("Trades gesamt",    f"{_tot_tr:,}")

        # ── Kombinierter Equity-Chart ──────────────────────────────────────
        if _ot_eq_all:
            st.subheader("📈 Kombiniertes Portfolio-Kapital")
            st.caption(
                f"Jeder Coin startet mit {_ot_coin_c:,.0f} USDT · "
                f"Gesamt: {_ot_cap_used:,.0f} USDT · "
                f"Alle Coins simultan gehandelt"
            )

            # Tagesdurchschnitt je Coin, dann summieren
            _daily_frames = []
            for _s in _ot_eq_all:
                _d = _s.resample("D").last().ffill()
                _daily_frames.append(_d)

            _combined_eq = pd.concat(_daily_frames, axis=1)
            # Start-Kapital pro Coin vorausfüllen (für Coins die noch keinen Trade hatten)
            _combined_eq = _combined_eq.fillna(_ot_coin_c)
            _portfolio   = _combined_eq.sum(axis=1)

            _delta_total = _portfolio.iloc[-1] - _ot_cap_used
            _delta_pct   = _delta_total / _ot_cap_used * 100

            _fig_eq = go.Figure()
            _eq_color = "#2ecc71" if _portfolio.iloc[-1] >= _ot_cap_used else "#e74c3c"
            _fig_eq.add_trace(go.Scatter(
                x=_portfolio.index,
                y=_portfolio.values,
                mode="lines",
                line=dict(color=_eq_color, width=2),
                fill="tozeroy",
                fillcolor=_eq_color.replace(")", ", 0.08)").replace("rgb", "rgba"),
                name="Portfolio",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f} USDT<extra></extra>",
            ))
            _fig_eq.add_hline(y=_ot_cap_used, line_dash="dash",
                              line_color="gray", line_width=1)
            _fig_eq.update_layout(
                height=350,
                margin=dict(l=50, r=30, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="USDT"),
            )
            _fig_eq.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            _fig_eq.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            st.plotly_chart(_fig_eq, use_container_width=True)

            _eq_m1, _eq_m2 = st.columns(2)
            _eq_m1.metric("End-Kapital", f"{_portfolio.iloc[-1]:,.2f} USDT",
                          f"{_delta_pct:+.2f}%",
                          delta_color="normal" if _delta_pct >= 0 else "inverse")
            _eq_m2.metric("Absoluter P/L", f"{_delta_total:+,.2f} USDT")

        # ── Per-Coin Ergebnisse ────────────────────────────────────────────
        st.subheader("📊 Ergebnisse je Coin")

        # Bar-Chart: PnL je Coin
        _df_rows = pd.DataFrame(_ot_rows2).sort_values("OOS-PnL", ascending=False)
        _fig_bar = go.Figure(go.Bar(
            x=_df_rows["Coin"],
            y=_df_rows["OOS-PnL"],
            marker_color=[
                "#2ecc71" if v >= 0 else "#e74c3c" for v in _df_rows["OOS-PnL"]
            ],
            hovertemplate="%{x}<br>PnL: %{y:+.2f}%<extra></extra>",
        ))
        _fig_bar.update_layout(
            height=280, margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="PnL %", zeroline=True, zerolinecolor="gray"),
        )
        _fig_bar.update_xaxes(showgrid=False)
        _fig_bar.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        st.plotly_chart(_fig_bar, use_container_width=True)

        # Detail-Tabelle
        _tbl = _df_rows.copy()
        _tbl["OOS-PnL"] = _tbl["OOS-PnL"].apply(lambda v: f"{v:+.2f}%")
        _tbl["MaxDD%"]  = _tbl["MaxDD%"].apply(lambda v: f"{v:.2f}%")
        st.dataframe(
            _tbl[["Coin", "Strategie", "Hebel", "TP / SL", "OOS-PnL",
                  "Trades", "WR%", "PF", "MaxDD%"]],
            use_container_width=True,
            hide_index=True,
        )

    elif "ot_results" in st.session_state and not st.session_state["ot_results"]:
        st.warning("Keine verwertbaren Ergebnisse. Min. Trades evtl. zu hoch oder Daten fehlen.")
    else:
        st.info("1️⃣ **Coins laden** → wählt automatisch die meistgehandelten Bybit-Perps\n\n"
                "2️⃣ **Backtesten** → testet alle Strategien × Hebel, wählt bestes Setup pro Coin\n\n"
                "3️⃣ Equity-Chart + Tabelle zeigen simuliertes simultanes Trading aller Coins")


# ══════════════════════════════════════════════════════════════════════════════
# SUPERTREND LIVE – Auto-optimiertes Live Trading
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "⚡ SuperTrend Live":
    import time as _stl_time

    st.subheader("⚡ SuperTrend Live")
    st.caption(
        "Automatisch optimierte Parameter · 1h Timeframe · Trailing SL 0.3% · 3× Hebel · "
        "Wöchentlicher Backtest wählt Top-10 Coins"
    )

    _stl_trader = _get_live_trader()
    _stl_status = _stl_trader.get_status()
    _stl_running = _stl_status.get("running", False)
    _stl_state   = _stl_load_state()
    _stl_coins   = _stl_state.get("top_coins", [])
    _stl_last_opt = _stl_state.get("last_optimized")

    # ── Status-Banner ─────────────────────────────────────────────────────────
    if _stl_running:
        st.success("⚡ **AKTIV** — SuperTrend Live Trading läuft")
    else:
        st.warning("⏹ **GESTOPPT** — Trading inaktiv")

    # ── Haupt-Layout ──────────────────────────────────────────────────────────
    _stl_col_ctrl, _stl_col_stats = st.columns([1, 1])

    with _stl_col_ctrl:
        st.markdown("### Steuerung")

        # Optimierungs-Status
        if _stl_last_opt:
            _opt_dt = _dt.fromisoformat(_stl_last_opt) if isinstance(_stl_last_opt, str) else _stl_last_opt
            _opt_age = (_dt.now() - _opt_dt.replace(tzinfo=None)).days if hasattr(_opt_dt, 'tzinfo') else 0
            _next_opt = _opt_dt.replace(tzinfo=None) + _td(days=7)
            st.info(
                f"🔬 Letzter Backtest: **{_opt_dt.strftime('%d.%m.%Y %H:%M') if hasattr(_opt_dt, 'strftime') else _stl_last_opt}**  \n"
                f"Nächster: **{_next_opt.strftime('%d.%m.%Y')}**  \n"
                f"Top-{len(_stl_coins)} Coins geladen"
            )
            if _opt_age >= 7:
                st.warning("⚠ Backtest-Daten älter als 7 Tage — bitte re-optimieren.")
        else:
            st.warning("⚠ Noch kein Backtest durchgeführt. Bitte zuerst optimieren.")

        # Optimieren-Button
        _stl_do_opt = st.button(
            "🔄 Jetzt optimieren (wählt Top-10 Coins neu)",
            disabled=_stl_running,
            use_container_width=True,
            key="stl_optimize",
        )

        st.divider()

        # Start / Stop
        _stl_b1, _stl_b2 = st.columns(2)
        _stl_start = _stl_b1.button(
            "▶ Trading starten", type="primary",
            disabled=_stl_running or not _stl_coins,
            use_container_width=True, key="stl_start",
        )
        _stl_stop = _stl_b2.button(
            "⏹ Stoppen",
            disabled=not _stl_running,
            use_container_width=True, key="stl_stop",
        )

        if not _stl_coins:
            st.caption("Erst optimieren um Coins zu laden.")

        # Konfigurationsinfo
        st.markdown("**Automatische Parameter:**")
        st.markdown(
            f"- Timeframe: `{_STL_TIMEFRAME}`  \n"
            f"- **TP:** keiner (Trailing übernimmt Exit)  \n"
            f"- **Trailing SL:** `{_STL_TRAILING*100:.1f}%` des Preises  \n"
            f"- **Notfall-SL:** `{'keiner' if _STL_SL_PCT is None else f'{_STL_SL_PCT*100:.0f}%'}`  \n"
            f"- Leverage: `{_STL_LEVERAGE}×`  \n"
            f"- Position: `{_STL_POS_SIZE*100:.0f}%` des Kapitals"
        )

    with _stl_col_stats:
        st.markdown("### Portfolio-Status")

        _stl_eq   = _stl_status.get("equity")
        _stl_ieq  = _stl_status.get("initial_equity")
        if _stl_eq and _stl_ieq:
            _stl_delta = (_stl_eq - _stl_ieq) / _stl_ieq * 100
            st.metric("Kapital", f"{_stl_eq:,.2f} USDT",
                      f"{_stl_delta:+.2f}%",
                      delta_color="normal" if _stl_delta >= 0 else "inverse")
        else:
            st.metric("Kapital", "–")

        _stl_positions = _stl_status.get("positions", {})
        _stl_open_pos  = sum(1 for v in _stl_positions.values() if v)
        st.metric("Offene Positionen", f"{_stl_open_pos} / {len(_stl_coins)}")

        _stl_trades_all = _stl_status.get("trades", [])
        _stl_trades_today = [
            t for t in _stl_trades_all
            if (t.get("timestamp") or t.get("exit_time") or "")[:10] == _dt.now().strftime("%Y-%m-%d")
        ]
        _stl_today_pnl = sum(t.get("pnl_pct", 0) or 0 for t in _stl_trades_today)
        st.metric("Trades heute", len(_stl_trades_today),
                  f"{_stl_today_pnl:+.2f}%" if _stl_trades_today else None)

        _stl_last_tick = _stl_status.get("last_tick")
        st.metric("Letzter Tick", _stl_last_tick or "–")

        # Aktive Positionen anzeigen
        if _stl_open_pos:
            st.divider()
            st.markdown("**Offene Positionen:**")
            for _sym, _pos in _stl_positions.items():
                if not _pos:
                    continue
                _coin_n = _sym.split("/")[0]
                _side_i = "🟢 LONG" if _pos["side"] == "long" else "🔴 SHORT"
                _upnl   = _pos.get("unrealized_pnl", 0) or 0
                st.markdown(
                    f"`{_coin_n}` **{_side_i}**  "
                    f"Entry: `{_pos['entry_price']:,.4f}`  "
                    f"uPnL: `{_upnl:+.2f} USDT`"
                )

    # ── Optimierung ausführen ─────────────────────────────────────────────────
    if _stl_do_opt:
        with st.spinner("🔄 Optimierung läuft — teste SuperTrend auf allen 1h-Coins…"):
            _stl_prog = st.progress(0.0, text="Starte…")
            def _stl_prog_cb(v):
                _stl_prog.progress(v, text=f"{int(v*100)}% abgeschlossen")
            try:
                _stl_new_coins = _stl_run_optimization(progress_fn=_stl_prog_cb)
                _stl_new_state = {
                    "last_optimized": _dt.now().isoformat(),
                    "top_coins":      _stl_new_coins,
                }
                _stl_save_state(_stl_new_state)
                st.success(f"✅ Optimierung fertig — {len(_stl_new_coins)} Top-Coins gefunden.")
                _stl_coins = _stl_new_coins
                _stl_prog.empty()
                st.rerun()
            except Exception as _e:
                st.error(f"Optimierung fehlgeschlagen: {_e}")

    # ── Trading starten ───────────────────────────────────────────────────────
    if _stl_start and _stl_coins:
        try:
            import inspect as _stl_ins
            _stl_symbols    = [c["symbol"] for c in _stl_coins]
            _stl_per_sym    = {
                c["symbol"]: SupertrendStrategy(c["period"], c["mult"])
                for c in _stl_coins
            }
            _stl_main_strat = SupertrendStrategy(20, 2.0)
            _stl_cfg_kw: dict = dict(
                strategy              = _stl_main_strat,
                symbols               = _stl_symbols,
                timeframe             = _STL_TIMEFRAME,
                leverage              = _STL_LEVERAGE,
                position_size         = _STL_POS_SIZE,
                tp_pct                = None,
                sl_pct                = _STL_SL_PCT,
                atr_mode                 = _STL_ATR_MODE,
                atr_period               = _STL_ATR_PERIOD,
                atr_sl_mult              = _STL_ATR_SL_MULT,
                atr_rr                   = _STL_ATR_RR,
                use_trailing             = False,
                trailing_warmup_candles  = _STL_WARMUP_CANDLES,
                mtf_enabled              = False,
                adx_enabled              = False,
                adx_require_trend        = False,
                per_symbol_strategies    = _stl_per_sym,
                use_trigger_entry        = False,
            )
            if "trailing_sl_pct" in _stl_ins.signature(_stl_trader.configure).parameters:
                _stl_cfg_kw["trailing_sl_pct"] = _STL_TRAILING
            _stl_trader.configure(**_stl_cfg_kw)
            _stl_trader.start()
            st.success(
                f"▶ SuperTrend Live gestartet — "
                f"{len(_stl_symbols)} Coins · Trail {_STL_TRAILING*100:.1f}% ab K+{_STL_WARMUP_CANDLES+1} · "
                f"{'kein SL' if _STL_SL_PCT is None else f'SL {_STL_SL_PCT/_STL_LEVERAGE*100:.2f}%'} · kein TP"
            )
            st.rerun()
        except Exception as _e:
            st.error(f"Start fehlgeschlagen: {_e}")

    if _stl_stop:
        _stl_trader.stop()
        st.warning("⏹ Trading gestoppt.")
        st.rerun()

    # ── Top-Coins Tabelle ─────────────────────────────────────────────────────
    if _stl_coins:
        st.divider()
        st.markdown("### Top-10 Coins (letzter Backtest)")
        _stl_tbl_data = []
        for _i, _c in enumerate(_stl_coins, 1):
            _pos_c = _stl_positions.get(_c["symbol"])
            _pos_str = (
                f"🟢 LONG" if (_pos_c and _pos_c["side"] == "long") else
                f"🔴 SHORT" if (_pos_c and _pos_c["side"] == "short") else
                "—"
            )
            _stl_tbl_data.append({
                "#":          _i,
                "Coin":       _c["coin"],
                "ST Params":  f"ATR={_c['period']}, Mult={_c['mult']}",
                "PnL (BT)%":  f"{_c['pnl_pct']:+.1f}%",
                "WR%":        f"{_c['winrate']:.0f}%",
                "PF":         f"{_c['pf']:.2f}",
                "MaxDD%":     f"{_c['max_dd']:.1f}%",
                "Position":   _pos_str,
            })
        st.dataframe(
            pd.DataFrame(_stl_tbl_data),
            use_container_width=True, hide_index=True,
        )

    # ── Coin-Tabs mit Charts ──────────────────────────────────────────────────
    if _stl_coins:
        st.divider()
        st.markdown("### Charts & Positionen")

        _stl_tab_labels = [c["coin"] for c in _stl_coins]
        _stl_tabs = st.tabs(_stl_tab_labels)

        for _ti, (_stl_tab, _stl_coin_cfg) in enumerate(zip(_stl_tabs, _stl_coins)):
            with _stl_tab:
                _sym    = _stl_coin_cfg["symbol"]
                _period = _stl_coin_cfg["period"]
                _mult   = _stl_coin_cfg["mult"]
                _coin_n = _stl_coin_cfg["coin"]

                # Header-Zeile
                _hc1, _hc2, _hc3, _hc4 = st.columns(4)
                _hc1.metric("Symbol", _coin_n)
                _hc2.metric("ST Params", f"ATR={_period}, Mult={_mult}")
                _hc3.metric("Trailing SL", f"{_STL_TRAILING*100:.1f}%")

                _pos_this = _stl_positions.get(_sym)
                if _pos_this:
                    _upnl = _pos_this.get("unrealized_pnl", 0) or 0
                    _hc4.metric(
                        "Position",
                        f"{'LONG' if _pos_this['side'] == 'long' else 'SHORT'}  @{_pos_this['entry_price']:,.4f}",
                        f"{_upnl:+.2f} USDT",
                        delta_color="normal" if _upnl >= 0 else "inverse",
                    )
                else:
                    _hc4.metric("Position", "Keine")

                # Chart
                _stl_candles = _stl_trader.candles.get(_sym)
                if _stl_candles is None:
                    with st.spinner(f"Lade {_coin_n} Kerzen…"):
                        try:
                            _stl_candles = _fetch_live_candles(_sym, _STL_TIMEFRAME, 200)
                        except Exception as _e:
                            st.error(f"Fehler beim Laden: {_e}")
                            _stl_candles = None

                if _stl_candles is not None and len(_stl_candles) >= 20:
                    _stl_fig = _stl_build_chart(
                        _stl_candles, _period, _mult, _sym, _stl_trades_all
                    )
                    st.plotly_chart(_stl_fig, use_container_width=True,
                                    key=f"stl_chart_{_ti}")
                else:
                    st.info("Noch keine Kerzen verfügbar.")

                # Letzte Trades dieses Coins
                _coin_trades = [t for t in _stl_trades_all if t.get("symbol") == _sym]
                if _coin_trades:
                    st.markdown(f"**Letzte Trades ({len(_coin_trades)} gesamt)**")
                    _ct_df = pd.DataFrame(_coin_trades[-8:][::-1])
                    _ct_shown = [c for c in ["side", "entry_price", "exit_price",
                                              "pnl_pct", "exit_reason", "timestamp"]
                                 if c in _ct_df.columns]
                    if _ct_shown:
                        _ct_df = _ct_df[_ct_shown].rename(columns={
                            "side": "Side", "entry_price": "Entry",
                            "exit_price": "Exit", "pnl_pct": "PnL %",
                            "exit_reason": "Grund", "timestamp": "Zeit",
                        })
                        st.dataframe(
                            _ct_df.style.map(
                                lambda v: ("color: #2ecc71" if isinstance(v, (int, float)) and v > 0
                                           else ("color: #e74c3c" if isinstance(v, (int, float)) and v < 0 else "")),
                                subset=["PnL %"] if "PnL %" in _ct_df.columns else [],
                            ),
                            use_container_width=True, hide_index=True, height=200,
                        )
                else:
                    st.caption("Noch keine abgeschlossenen Trades für diesen Coin.")

    elif not _stl_coins:
        st.info("👆 Bitte zuerst **'Jetzt optimieren'** klicken um Top-10 Coins zu berechnen.")

    # ── Log ───────────────────────────────────────────────────────────────────
    _stl_log = _stl_status.get("log", [])
    if _stl_log and _stl_running:
        with st.expander(f"Bot-Log ({min(15, len(_stl_log))} Einträge)", expanded=False):
            st.code("\n".join(_stl_log[:15]), language=None)

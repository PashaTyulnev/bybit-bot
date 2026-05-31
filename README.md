# Crypto Trading Bot

A systematic crypto futures trading bot with a Streamlit dashboard, strategy backtester, and walk-forward optimizer — running on Bybit (Demo & Mainnet).

---

## Features

- **Live Trading** — multi-symbol background engine with market orders, TP/SL, breakeven logic, and native Bybit trailing stops
- **Strategy Backtester** — signal-driven backtester with walk-forward split (70/30), permutation tests, and extended metrics
- **Optimizer** — grid-search over strategy parameters, leverage, position size, TP/SL; ranks by out-of-sample PnL
- **Portfolio Mode** — 6-coin portfolio (ADA, XRP, DOT, AVAX, DOGE, BNB) at 3× leverage, strategy per coin
- **9 Strategies** — EMA Cross, RSI, RSI Divergence, Bollinger, Breakout, MACD, Supertrend, MeanRev, TrendFollow
- **Data Loader** — downloads historical OHLCV data from Bybit (all timeframes, up to 1 year)
- **State Persistence** — positions, trades, and config survive restarts via `data/live_state.json`

---

## Project Structure

```
app/
├── src/
│   ├── app.py                  # Streamlit UI (all pages)
│   ├── live_trader.py          # Live engine — background thread, multi-symbol
│   ├── exchange.py             # ccxt Bybit wrapper (Demo / Mainnet)
│   ├── backtester.py           # OHLCV backtester (blind long/short sequences)
│   ├── strategy_backtester.py  # Signal-driven backtester
│   ├── strategy_optimizer.py   # Strategy grid-search optimizer
│   ├── optimizer.py            # General optimizer (sequences)
│   ├── metrics.py              # Extended metrics (Sharpe, drawdown, PF, …)
│   ├── walk_forward.py         # Train/test split helpers
│   ├── download_ohlcv.py       # Historical data fetcher
│   ├── reporting.py            # CSV export, equity charts
│   └── strategies/
│       ├── __init__.py         # STRATEGY_REGISTRY
│       ├── ema_cross.py
│       ├── rsi_strategy.py
│       ├── bollinger_strategy.py
│       ├── breakout_strategy.py
│       ├── macd_strategy.py
│       ├── supertrend_strategy.py
│       ├── mean_rev_strategy.py
│       ├── trend_follow_strategy.py
│       └── …
├── data/
│   ├── live_state.json         # Persisted live trading state
│   ├── raw/                    # Downloaded OHLCV CSVs
│   └── optitest_6m/            # 6-month optimization dataset
└── README.md
```

---

## Setup

**1. Install dependencies**
```bash
pip install streamlit ccxt pandas numpy plotly
```

**2. Configure API keys**

Create a `.env` file or set environment variables:
```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_API_URL=https://api-demo.bybit.com   # omit for mainnet
```

**3. Run the dashboard**
```bash
streamlit run src/app.py
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| ⚡ SuperTrend Live | Quick Supertrend live view |
| 🤖 Live Trading | Start/stop bot, monitor open positions and trade log |
| 🧪 OptiTest | Optimized strategy test on historical data |
| 🔁 Backtest | Single strategy backtest with equity curve |
| 🔍 Optimizer | Parameter grid search (sequences, leverage, TP/SL) |
| 📈 Strategies | Signal-driven strategy backtest |
| 📊 Vergleich | Compare multiple backtest runs |
| 🔬 Strategie-Optimizer | Strategy-level optimizer with ADX/MTF filters |
| 🎯 Multi-Symbol | Run backtests across multiple symbols at once |
| 📥 Daten laden | Download OHLCV data from Bybit |

---

## Live Trading Architecture

```
Streamlit UI
     │
     ▼
LiveTrader (singleton, st.cache_resource)
     │
     ├── Background Thread  ← waits for candle close → _tick()
     │       │
     │       ├── fetch OHLCV  (ccxt, 12s timeout per call)
     │       ├── compute signals (strategy)
     │       ├── place / manage orders (TP, SL, trailing stop)
     │       └── save state → data/live_state.json
     │
     └── TriggerOrderManager  ← Bybit-native conditional entry orders
```

- All exchange calls run in a thread pool with a hard 12-second wall-clock timeout to prevent SSL/network deadlocks from freezing the UI.
- Trailing stop floor is percentage-based (e.g. 0.01% of price), not a fixed USDT amount.

---

## Portfolio Configuration

| Coin | Strategy | TP | SL | OOS PnL |
|------|----------|----|----|---------|
| ADA | MeanRev (BB 10/2σ, ADX < 20) | 3.0% | 1.5% | +10.2% |
| XRP | TrendFollow (EMA 20/100, ADX ≥ 25) | 3.0% | 1.5% | +9.3% |
| DOT | TrendFollow (EMA 20/100, ADX ≥ 25) | 2.0% | 1.5% | +9.4% |
| AVAX | MeanRev (BB 10/2σ, ADX < 20) | 2.0% | 1.0% | +5.5% |
| DOGE | MeanRev (BB 10/2σ, ADX < 20) | 3.0% | 1.0% | +7.0% |
| BNB | MeanRev (BB 10/2σ, ADX < 20) | 3.0% | 0.5% | +5.9% |

Parameters derived from walk-forward optimization over 9,720 backtests on 1 year of 15m data. OOS PnL is out-of-sample (30% test split) at 3× leverage, 10% position size, 0.055% fee.

---

## Backtesting Methodology

- **Walk-forward split** — 70% in-sample for parameter fitting, 30% out-of-sample for evaluation
- **Permutation test** — 60 shuffles to measure statistical significance (target p < 0.10)
- **Metrics** — PnL, Profit Factor, Win Rate, Max Drawdown, Sharpe Ratio
- Strategies are only considered valid if out-of-sample results are significant

---

## Disclaimer

This bot is for educational and research purposes. Running it with real funds carries significant financial risk. Always test thoroughly on the Demo API before going live.

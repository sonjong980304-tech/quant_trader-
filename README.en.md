# Quant Automated Trading System

**🌐 Language:** [한국어](README.md) | English

> **Investment Disclaimer**: This program is developed for educational and research purposes only.
> The user bears full responsibility for any gains or losses from actual investments.
> Past performance does not guarantee future returns.

---

## Project Overview

An automated trading system for the KRX Korean stock market that runs two independent agents (Mean Reversion + Trend Following) in parallel with separated slot management. Signals, exits, and reports are delivered in real-time via Telegram bot.

---

## Current Status

```
LIVE_TRADING = False  (paper trading)
Agents: reversion (ML-based) + trend following (rule-based)
Slots: reversion 10 stocks + trend 10 stocks = max 20 simultaneous positions
Paper test period: started 2026-06-19 (2-week target)
```

---

## Strategy Architecture

### Agent 1 — Mean Reversion (ML-based)

- XGBoost (raw probability, no calibration)
- Triple-Barrier labeling: TP=+15%, SL=-8%, hold=10 days
- Walk-Forward Expanding Window (4-Fold WF, 2023~2026)
- Universe: Point-in-Time dynamic top-200 by market cap (survivorship bias removed)
- Features: atr_pct, kospi_relative_20d, beta_60d, ma200_deviation, ret_60d, ret_20d, high52_pct, kospi_relative_5d

### Agent 2 — Trend Following (rule-based)

- ADX≥25 + MA alignment (MA5>MA20>MA60>MA200) + volume>1.3×
- ATR-based trailing stop: 2.0×ATR
- Exit on MA20 downward breach

### Portfolio Management

- Separated slots: reversion 10 / trend 10 (no cross-contamination)
- Position sizing: Half-Kelly + ATR (calculated independently per agent)
- Max single position: 20%
- Regime filter: trend agent only — entry allowed only when KOSPI close > KOSPI MA200
  (reversion agent has no filter — designed to capture oversold bounces even in downtrends / SL -8% handles downside risk)

---

## Strategy Rationale (Research-based)

### Mean Reversion

- De Bondt & Thaler (1985), "Does the Stock Market Overreact?", Journal of Finance — empirically demonstrates that investor overreaction causes sharp declines to rebound.
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", Review of Financial Studies — shows non-linear tree models like XGBoost dominate linear models in return prediction.
- arXiv:2601.19504 (2026), "Generating Alpha: A Hybrid AI-Driven Trading System", Springer LNNS — RSI/Bollinger Band mean-reversion + XGBoost + regime filter combination achieves +135% over 24 months.
- López de Prado (2018), "Advances in Financial Machine Learning", Wiley — source of Triple-Barrier labeling methodology. Combines time barrier, TP, and SL to generate non-linear labels; now a standard technique in financial ML.

### Trend Following

- Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers", Journal of Finance — first empirical proof of momentum strategy profitability.
- Moskowitz, Ooi & Pedersen (2012) — optimal formation/holding period research for trend-following strategies.

---

## Differences from Prior Research

| Item | Paper (arXiv:2601.19504) | This System |
|------|--------------------------|-------------|
| Labeling | Next-day direction (simple) | Triple-Barrier (precise) |
| Validation | Simple 7:3 split | Walk-Forward (strict time-series) |
| Probability calibration | None | Platt Scaling |
| Evaluation metric | Accuracy 63% | AUC (robust to class imbalance) |
| Order type | Market order | Next-day open limit order |
| Target market | S&P 500 | KRX Korean stocks |
| Agents | Single strategy | reversion + trend dual agents |

---

## Backtest Results

### Slot Configuration Comparison (2024-01-01 ~ 2026-06-19)

| Config | Return | Sharpe | MDD | Trades |
|--------|--------|--------|-----|--------|
| Shared 10 slots (old) | +84.67% | 1.210 | -25.69% | 1,284 |
| Separated 5+5 | +130.33% | 1.933 | -15.62% | 493 |
| **Separated 10+10 (adopted)** | **+159.31%** | **1.847** | **-20.01%** | **970** |

### Per-Agent Performance (2024~2026.6)

| Agent | Return | Sharpe | MDD | Trades | Win Rate |
|-------|--------|--------|-----|--------|----------|
| Reversion (ML) | +29.09% | 0.745 | -18.54% | 469 | 48.8% |
| Trend Following | +157.87% | 1.928 | -15.59% | 501 | 43.5% |

### Separated 10+10 Full Metrics

- Total return: +159.31% / Sharpe: 1.847 / MDD: -20.01% / Trades: 970 / Win rate: 46.1% / P/L ratio: 2.02

### Annual Returns (Separated 10+10)

| Year | Combined | Reversion | Trend |
|------|----------|-----------|-------|
| 2024 | +17.24% | +13.15% | +15.35% |
| 2025 | +47.18% | +1.93% | +45.09% |
| 2026 | +47.68% | +10.10% | +52.87% |

Agent ratio: reversion 469 trades (48%) / trend 501 trades (52%)
Monthly correlation (reversion vs trend): 0.309
Cash ratio (separated 10+10): 2024 31.7%, 2025 48.0%, 2026 32.0%

---

## Trend Agent Grid Search Results (27 combinations)

Parameters: ADX threshold [20,25,30] × trailing stop [1.5,2.0,2.5 ATR] × volume [1.0,1.3,1.5×]

Top 5 combinations:

| ADX | Trail | Vol | Return | Sharpe | MDD | Trades |
|-----|-------|-----|--------|--------|-----|--------|
| ≥25 | 2.0× | 1.3× | +144.72% | 1.586 | -15.98% | 662 |
| ≥30 | 2.5× | 1.3× | +143.52% | 1.574 | -14.43% | 542 |
| ≥25 | 2.5× | 1.3× | +125.73% | 1.459 | -16.94% | 565 |
| ≥30 | 2.0× | 1.3× | +118.46% | 1.463 | -13.83% | 648 |
| ≥25 | 2.5× | 1.0× | +114.27% | 1.468 | -19.62% | 615 |

**Adopted: ADX≥25 / trail=2.0ATR / vol>1.3×**

---

## Reversion Agent Feature Importance (8 features)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | atr_pct | 0.1709 |
| 2 | ret_20d | 0.1348 |
| 3 | kospi_relative_20d | 0.1296 |
| 4 | beta_60d | 0.1253 |
| 5 | ma200_deviation | 0.1236 |
| 6 | high52_pct | 0.1234 |
| 7 | ret_60d | 0.1191 |
| 8 | kospi_relative_5d | 0.0732 |

Walk-Forward AUC (OOF): 0.5270 (TP=15%/SL=8%/hold=10d)

---

## Strategy Details

### Reversion Agent — Signal Pipeline (3 stages)

**Stage 1: KRX Universe Screening**

FinanceDataReader scans all KOSPI+KOSDAQ stocks → filter change rate > 0% + top 100 by trading value

**Stage 2: Technical Trigger Detection**

| Signal | Condition |
|--------|-----------|
| BB Lower Bounce | Close breaks below Bollinger Band lower band then re-enters |
| RSI Oversold Escape | RSI crosses above 30 from below |
| EMA Deviation Low | Price ≥ 5% below EMA20 |

**Stage 3: XGBoost ML Prediction**

Features (8): `atr_pct`, `kospi_relative_20d`, `beta_60d`, `ma200_deviation`, `ret_60d`, `ret_20d`, `high52_pct`, `kospi_relative_5d`

Triple-Barrier Labeling (López de Prado):

| Barrier | Condition | Result |
|---------|-----------|--------|
| Upper TP | Intraday High ≥ entry × 1.15 (+15%) | label=1 (success) |
| Lower SL | Intraday Low ≤ entry × 0.92 (−8%) | label=0 (failure) |
| Time | Close after 10 trading days | Close ≥ entry → 1, below → 0 |

### Trend Agent — Entry Conditions

| Condition | Threshold |
|-----------|-----------|
| ADX | ≥ 25 |
| MA Alignment | MA5 > MA20 > MA60 > MA200 |
| Volume | ≥ 1.3× 20-day average |
| Regime filter | KOSPI close > KOSPI MA200 (blocks entry in downtrends) |
| Exit | ATR×2.0 trailing stop or MA20 downward breach |

### Position Sizing (Half-Kelly + ATR)

```
Full Kelly:  f* = (p × b - q) / b
Half Kelly:  f = f* × 0.5

p = ML predicted win probability (reversion) / historical win rate (trend)
b = avg win / avg loss (risk/reward ratio)
```

```
Risk Parity qty = total_assets × 1% ÷ (2 × ATR(14))
Final qty       = min(Half-Kelly qty, Risk Parity qty)
```

---

## Automation Schedule

| Time | Action |
|------|--------|
| 07:30 (trading days) | XGBoost parallel retraining on universe filtered by change rate > 0% + top 100 by trading value |
| 08:00 (trading days) | Morning briefing — AI market overview + news |
| 09:00 (trading days) | KR pending order execution — next-day open buy based on EOD signal |
| **09:05 (trading days)** | **KR paper open price confirmation** — `update_entry_prices("KR")` |
| Every 5 min (market hours) | ML position TP/SL/forced close check + paper TP/SL evaluation |
| **15:31 (trading days)** | **EOD signal scan** — signal detection on completed daily candles → next-day open reservation |
| **15:30 (trading days)** | **KR EOD evaluation** — trade_days+1 + TP/SL check |
| **15:35 (daily)** | **KR paper trading daily report** (sent via Telegram) |
| Sunday 20:00 | Paper trading weekly summary |

---

## Bot Activation Gate

If existing holdings are present, the bot requires **manually selling all of them** before automated trading begins.

```
state.json: {"bot_active": false, "legacy_tickers": ["XXXX"], ...}
  ↓ manual sell completed
state.json: {"bot_active": true, ...}  →  automated trading starts
```

**Telegram Controls**

| Command | Action |
|---------|--------|
| `/stop` | Pause automated trading |
| `/start` | Resume automated trading |

---

## Paper Trading (`paper_trader.py`)

2-week paper validation phase before live trading. Operates only when `LIVE_TRADING=False` — no real API calls.

**Separated Slot Management**
- reversion dedicated 10 slots / trend dedicated 10 slots (fully isolated)
- `can_add_position(agent)` checks slot availability before entry

**Circuit Breaker Conditions (P3)**

| CB | Condition | Trigger |
|----|-----------|---------|
| CB1 | Paper EV | n≥30: EV ≤ −0.5% |
| CB2 | CI lower bound | n≥50: 95% CI lower < −1.0% |
| CB3 | Consecutive losses | Max consecutive losses ≥ 8 |
| CB4 | Backtest gap | n≥30: paper EV − backtest EV ≤ −1.0%pt |
| CB5 | Slippage | Measured avg slippage > 0.50% |
| CB6 | AUC | Quarterly avg AUC < 0.45 |

**P4 Live Trading Gate Criteria**

| Item | Threshold |
|------|-----------|
| Paper operation period | ≥ 60 trading days |
| Cumulative closed trades | ≥ 50 |
| After-cost EV | ≥ +0.30% |
| 95% CI lower bound | > 0% |
| Win rate | ≥ 52% |
| Measured slippage | < 0.40% |
| Stock concentration | < 30% |
| Max consecutive losses | ≤ 5 |
| Regime AUC average | ≥ 0.55 |

---

## Trade History Management

All automated trades are automatically recorded in `trade_history.csv`.

| Column | Description |
|--------|-------------|
| trade_id | Unique trade ID |
| ticker / name | Stock code / stock name |
| entry_date / entry_price | Buy date / buy price |
| exit_date / exit_price | Sell date / sell price |
| qty | Quantity |
| pnl_amount / pnl_pct | P&L (KRW) / P&L rate (%) |
| win | Success (1) / Failure (0) |
| strategy | Strategy used |
| win_prob | ML predicted win probability at entry (%) |
| avg_win_pct | Model training average win rate (%) |
| avg_loss_pct | Model training average loss rate (%) |
| model_auc | Model OOF AUC at entry |

---

## GPT AI Assistant (LangGraph ReAct)

Built with `create_react_agent` (langgraph.prebuilt) + `MemorySaver` checkpointer for per-user `thread_id` conversation history isolation.

| Tool | Purpose |
|------|---------|
| `get_naver_finance` | Korean stock financials (PER, PBR, EPS, etc.) |
| `get_naver_news` | Naver latest news search |
| `get_stock_signal` | Technical indicators + buy/sell signal analysis |
| `get_historical_price` | Historical closing price on a specific date |
| `get_account_balance` | Domestic balance |
| `get_portfolio_status` | reversion/trend slot status |
| `set_conditional_order` | Register conditional order |
| `list_conditional_orders` | List conditional orders |
| `cancel_conditional_order` | Cancel conditional order |
| `list_trade_records` | Trade history query (open/closed/all) |
| `edit_trade_record` | Edit trade record |

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| Free text | LangChain AI assistant auto-reply |
| `/ask <question>` | Explicit GPT question |
| `/reset` | Clear conversation history |
| `/status` | Signal query for all stocks |
| `/balance` | Domestic balance |
| `/portfolio` | Portfolio status |
| `/scanstocks` | Manual ML signal scan for growth stocks |
| `/buysignal_TICKER` | Confirm buy for scanned signal |
| `/skipsignal` | Skip pending signal |
| `/trainmodel` | Full ML model retraining |
| `/tradestats` | Trade history statistics + CSV send |
| `/backtest` | 45-day intraday ML backtest |
| `/stocks` | Watchlist |
| `/addstock CODE NAME` | Add stock to watchlist |
| `/removestock CODE` | Remove stock from watchlist |
| `/buy CODE QTY` | Manual buy |
| `/sell CODE QTY` | Manual sell |
| `/sellall CODE` | Sell all shares |

---

## Overall Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│              macOS launchd — 3 daemons running continuously           │
├──────────────┬───────────────────────────┬───────────────────────────┤
│  runner.py   │  telegram_bot.py          │  dashboard.py             │
│  (Scheduler) │  (User Interface)         │  (Streamlit Monitoring)   │
└──────┬───────┴─────────────┬─────────────┴───────────────────────────┘
       │                     │
       │    ┌────────────────▼──────────────────────────────────────┐
       │    │     langchain_agent.py  (LangGraph ReAct + gpt-5.5)   │
       │    │   MemorySaver checkpointer — per-user thread_id       │
       │    └───────────────────────────────────────────────────────┘
       │
       ├── [07:30] ml/trainer.py — XGBoost parallel retraining (change rate > 0% + top 100 by trading value)
       │
       ├── [08:00] morning_briefer.py — holdings news + market overview
       │
       ├── [every 5 min] check_ml_positions() + _run_paper_evaluate_kr()
       │     → ML position TP · ATR-SL · trailing stop · forced close check
       │     → Paper position TP/SL evaluation (KR market hours)
       │
       ├── [15:31] scan_growth_signals_eod()
       │     KOSPI MA200 regime filter (trend agent only)
       │     signals/krx_universe.py → FinanceDataReader KOSPI+KOSDAQ screening
       │     signals/signal_graph.py — signal detection pipeline
       │       ├── reversion agent: BB Lower Bounce · RSI Oversold · EMA Deviation
       │       └── trend agent: ADX≥25 + MA alignment + volume>1.3×
       │     → slot check → paper record or pending_orders registration
       │
       ├── [15:30] _run_paper_evaluate_kr_eod() — KR EOD trade_days+1
       ├── [15:35] paper_trader.daily_report(market="KR") — KR paper daily report
       └── [Sunday 20:00] paper_trader.weekly_summary()

Common layer:
  KIS API (KR real-time price · orders · balance)  ·  yfinance ≥1.2
  FinanceDataReader (KRX universe screening)  ·  ml/models/*.pkl
  trade_history.csv  ·  state.json
```

---

## Installation

```bash
git clone https://github.com/sonjong980304-tech/quant_trader-.git
cd quant_trader
bash install.sh
```

---

## API Key Setup (.env)

```
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...        # Account number (e.g., 12345678-01)
KIS_MOCK=true             # Paper trading: true / Live trading: false
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
OPENAI_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
```

---

## Running

```bash
# Train ML models (first run or /trainmodel command)
python3 ml/trainer.py

# Market hours scheduler (includes 07:30 auto-retraining)
python3 runner.py

# Telegram bot
python3 telegram_bot.py

# Slot-separated combined backtest
python3 combined_backtest.py
```

---

## Project Structure

```
quant_trader/
├── config.py               # Strategy parameters / API configuration
├── stocks.py               # Watchlist (STOCKS)
├── runner.py               # Scheduler
├── telegram_bot.py         # Telegram bot
├── langchain_agent.py      # LangGraph ReAct AI assistant
├── pending_confirmations.py # EOD buy signal confirmation queue
├── trader.py               # KIS API (domestic)
├── trade_logger.py         # Trade history CSV recorder + Telegram sender
├── backtest_ml.py          # 45-day intraday ML backtest
├── backtest_walkforward.py # Walk-forward backtest (cost-adjusted)
├── combined_backtest.py    # Slot-separated combined backtest
├── paper_trader.py         # Paper trading engine (separated 10+10, Circuit Breaker)
├── position_manager.py     # ML position tracking and bot activation state
├── trend_agent.py          # Trend Following agent
├── tests/
│   ├── test_triple_barrier.py      # Triple-Barrier labeling unit tests
│   ├── test_paper_trader.py        # Paper trading engine unit tests
│   └── test_position_manager.py    # ML position tracking unit tests
├── morning_briefer.py      # Morning briefing (LangGraph quality retry loop)
├── data_fetcher.py         # yfinance daily + KIS intraday
├── indicators.py           # MA / RSI / Bollinger Bands
├── strategy.py             # MA/RSI buy · sell signals
├── notifier.py             # Telegram message builder
├── news_fetcher.py         # Naver News API
├── naver_finance.py        # Naver Finance fundamentals scraper
├── conditional_orders.py   # Conditional orders (price/return conditions)
├── market_calendar.py      # KRX trading day cache
├── market_regime.py        # KOSPI market regime filter
├── gpt_agent.py            # GPT tool functions
├── signals/
│   ├── signal_graph.py     # LangGraph StateGraph signal detection pipeline
│   ├── scanner.py          # Technical trigger detection + ML agent evaluation
│   ├── krx_universe.py     # KRX full universe first-pass screening
│   └── alert.py            # Growth stock signal alert message
├── state.json              # Bot activation gate
├── trade_history.csv       # Trade history
├── ml/
│   ├── features.py         # Feature engineering + Triple-Barrier labeling
│   ├── model.py            # XGBoost training & prediction
│   ├── trainer.py          # KRX universe parallel retraining
│   └── models/             # {ticker}_reversion.pkl
├── portfolio/
│   └── kelly.py            # Kelly Criterion position sizing
├── logs/
│   └── trader.log
├── com.quant.trader.plist
├── com.quant.telegrambot.plist
└── com.quant.dashboard.plist
```

---

## Important Notes

1. Never commit the `.env` file to GitHub.
2. If `KIS_APP_KEY` is not set, the system automatically runs in simulation mode.
3. ML models must be trained before first run via `/trainmodel` or `python3 ml/trainer.py`.
4. Thoroughly validate with paper trading before switching to live trading.

---

## License

MIT License — For personal educational and research purposes only.

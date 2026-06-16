# Quant Automated Trading System

**🌐 Language:** [한국어](README.md) | English

> **Investment Disclaimer**: This program is developed for educational and research purposes only.  
> The user bears full responsibility for any gains or losses from actual investments.  
> Past performance does not guarantee future returns.

---

## Portfolio Structure

### 70% — Safe Assets (Monte Carlo Simulation Optimized)

| Asset | Weight | Description |
|-------|--------|-------------|
| QQQ | 22.3% | NASDAQ 100 ETF |
| Samsung Electronics (005930.KS) | 27.3% | Korean large-cap |
| TLT | 0.2% | U.S. Long-Term Treasury ETF |
| ACE KRX Gold Spot (411060.KS) | 50.3% | Gold ETF |

- 100,000 Monte Carlo simulations on the latest 5-year data → derive **maximum Sharpe ratio** weights
- Automatic rebalancing at 08:30 on the 1st of every month (LLM decides share quantities)

### 30% — Growth Stocks (Dual-Agent ML Strategy)

- Two XGBoost models run in parallel: **Breakout Agent** (`_momentum.pkl`) + **Pullback Agent** (`_reversion.pkl`)
- Each agent filters only the historical rows matching its trigger type for training → situation-specific predictions
- Signal detection uses **EOD (end-of-day) completed candlesticks** → automatic buy order scheduled for the next day's open price (eliminates Train/Serve Skew)
- On signal: **register in `pending_orders`** → auto-execute at market open: 09:00 (KR) / 22:30·23:30 (US)
- Entry conditions (5-layer filter):
  1. **Market regime filter** — new buys fully blocked when KOSPI is in a downtrend (MA5 < MA20)
  2. At least **1 technical trigger** satisfied
  3. Relevant trigger-type agent **AUC ≥ 0.58** (blocks coin-flip models)
  4. **Win rate ≥ 60%** (Platt Scaling calibration applied)
  5. **Expected value risk/reward ≥ 1.5** — in bear markets (KOSPI < MA20 OR RSI < 35), automatically tightened by applying `avg_win × 0.4`
- When both agents satisfy conditions → **select agent with higher win_prob**
- Position sizing: **Half-Kelly + Risk Parity** (`min(Kelly qty, total_assets × 1% ÷ 2×ATR)`)
- Daily retraining at 07:30 on **KRX top-200 universe** / 22:30·23:30 on **US top-100 universe** (auto-branching for DST/standard time)

---

## ML Strategy Details

### Universe Screening (Step 1)

| Market | Method | Criteria |
|--------|--------|----------|
| Korea | **FinanceDataReader** full KOSPI+KOSDAQ | Change rate > 0% + top 100 by trading value |
| U.S. | Full S&P 500 (503 stocks) | Change rate > 0% + top 50 by volume ratio ≥ 1.5× |

- Volume is normalized to a full-day estimate based on market hours: **Korea (KST 09:00~15:30) / U.S. (ET 09:30~16:00)**
- Structurally declining stocks are permanently excluded via **blacklist** (`BLACKLIST`)

### Technical Triggers (Step 2)

| Signal | Condition |
|--------|-----------|
| Volume Explosion | Projected daily volume > 20-day average × 2.0 + bullish candle |
| BB Lower Bounce | Close breaks below Bollinger Band lower band then re-enters |
| RSI Oversold Escape | RSI crosses above 30 from below |
| EMA Deviation Low | Price ≥ 5% below EMA20 |
| BB Squeeze Breakout | Band contraction (60-day low) followed by upper band breakout |

> At least **1 of 5** triggers must be met to proceed to ML prediction. Primary quality filters are AUC ≥ 0.58 + win rate ≥ 60%.  
> Triggers are detected on **EOD completed candlesticks** (no intraday bar synthesis — same features as training time, eliminating Train/Serve Skew)

### Dual-Agent ML Prediction (Step 3)

The agent model is routed based on the trigger type detected.

| Agent | Assigned Triggers | Training Data |
|-------|-------------------|---------------|
| Breakout Agent (`_momentum.pkl`) | Volume Explosion, BB Squeeze Breakout | Past 5 years filtered to days when those triggers fired |
| Pullback Agent (`_reversion.pkl`) | BB Lower Bounce, RSI Oversold Escape, EMA Deviation Low | Past 5 years filtered to days when those triggers fired |

When both agents satisfy conditions → **select the one with higher win_prob and proceed to buy**

```
Trigger Detected
  ├── Volume Explosion / BB Squeeze Breakout → Breakout Agent prediction
  ├── BB Lower Bounce / RSI Oversold Escape / EMA Deviation Low → Pullback Agent prediction
  └── Both triggered → select highest win_prob from both predictions
```

**XGBoost Features (16)**

`change_rate`, `volume_change`, `rsi`, `ema_deviation_20`, `bb_width_20`, `bb_pct_20`, `bb_std_20`, `volume_ratio`, `candle_body`, `candle_upper_wick`, `candle_lower_wick`, `ret_3d`, `ret_5d`, `ret_10d`, `volatility_10d`, `atr_pct`

> `atr_pct` = ATR(14) / Close — price-normalized volatility feature. Lets the model directly learn each stock's volatility regime.

**Labeling**: Triple-Barrier (López de Prado method) — G1 grid adopted (2026-06-10)

| Barrier | Condition | Result |
|---------|-----------|--------|
| Upper TP | Intraday High ≥ entry × 1.15 (+15%) | label=1 (success) |
| Lower SL | Intraday Low ≤ entry × 0.94 (−6%) | label=0 (failure) |
| Time | Close after 7 trading days | Close ≥ entry → 1, below → 0 |

> Same-day simultaneous TP+SL touch → SL takes priority (conservative assumption).  
> G1 grid (20-combination walk-forward) result: KR backtest EV **+1.468%** (at 0.05% slippage).

**Training Data**: 5 years of daily candlesticks / KRX 200 + US 100 universe parallelized daily retraining at 07:30 (separate momentum and reversion agents)

**Signal Confirmation Conditions (6-layer filter)**

| Filter | Threshold | Description |
|--------|-----------|-------------|
| Market Regime | KOSPI MA5 > MA20 | Downtrend (MA5 < MA20) fully blocks new buys |
| MA200 Trend | Close ≥ 200-day MA | Blocks buy signals on downtrending stocks (Phase 4) |
| Model AUC | ≥ 0.58 | OOF AUC 0.5 = coin flip; only trust ≥ 0.58 |
| Win Rate | ≥ 60% | Triple-Barrier success probability after Platt Scaling calibration |
| EV Risk/Reward | ≥ 1.5 | `(avg_win_eff × win_prob) / (avg_loss × (1−win_prob))` — bear market applies avg_win × 0.4 |
| Trigger Count | ≥ 1 | AUC + win rate are the primary filters; trigger is the entry condition |

The old simple risk/reward (`avg_win / avg_loss`) was a fixed value independent of win rate.  
The **EV risk/reward** automatically tightens as win rate decreases.

**Platt Scaling Probability Calibration**  
XGBoost's `predict_proba()` internally compresses probabilities toward 0.5, causing it to under-predict a true 60% win rate as 0.55. After training, fitting a sigmoid regression (Platt Scaling) on the last TimeSeriesSplit fold's validation data aligns predicted probabilities with the actual outcome distribution.

### Half-Kelly + Risk Parity Position Sizing

```
Full Kelly:  f* = (p × b - q) / b
Half Kelly:  f = f* × 0.5  ← applied value

p = ML predicted win probability
b = avg_win / avg_loss (risk/reward ratio)
q = 1 - p
```

Full Kelly is sensitive to estimation errors in inputs (win rate and risk/reward), so Half Kelly acts as a buffer.

**Risk Parity Integration**

Since ATR-based stop widths vary by stock, position size is back-calculated so that a loss always equals exactly 1% of total assets.

```
Risk Parity qty = total_assets × 1% ÷ (2 × ATR(14))
Final qty       = min(Half-Kelly qty, Risk Parity qty)
```

Low-volatility stocks (small ATR) → more shares; high-volatility stocks (large ATR) → fewer shares, equalizing real risk across positions.

---

## Algorithm Selection Rationale

### XGBoost + TimeSeriesSplit

#### Why XGBoost

XGBoost (Extreme Gradient Boosting) is a boosting-family model that sequentially ensembles decision trees.  
Three reasons it suits price data:

1. **Non-linear pattern capture** — Technical signals like volume explosions, RSI oversold escapes, and Bollinger Band contractions are not linear. Tree-based models naturally learn these threshold conditions.
2. **Feature scale invariance** — 16 features with different units (MA deviation %, RSI 0~100, volume ratio multiplier) can be used directly without normalization.
3. **Class imbalance handling** — Cases yielding +3% within 7 days represent a minority of all data. `scale_pos_weight = (non-positive ratio) / (positive ratio)` reweights the minority class to correct bias.

KR and US retraining are separated and run just before each market opens. KRX runs at 07:30; US is registered at both 22:30 and 23:30 but internally checks the ET 09:30~10:00 window to run exactly once regardless of DST. XGBoost can process 150 stocks in tens of minutes using 8-thread parallelism without a GPU.

#### Why TimeSeriesSplit

Standard k-fold cross-validation **randomly shuffles data** to build train/validation sets.  
Applying this to price data causes **data leakage** — future data predicts the past — making validation metrics overly optimistic.

```
Standard k-fold (incorrect)
  Fold 1:  [──val──][──────train──────][──────train──────]
  Fold 2:  [──────train──────][──val──][──────train──────]
                                          ↑ future included in past training

TimeSeriesSplit (correct)
  Fold 1:  [──────train──────][──val──]
  Fold 2:  [────────────train────────][──val──]
  Fold 3:  [──────────────────train──────────][──val──]
                     past → future direction only
```

`TimeSeriesSplit(n_splits=5)` expands the training window sequentially each fold while always placing the validation window in the future. OOF (Out-of-Fold) metrics reliably reflect actual predictive power under the same conditions as live trading.

---

### Markowitz Efficient Frontier and Monte Carlo Simulation

#### What is the Markowitz Efficient Frontier

![Markowitz Efficient Frontier](docs/efficient_frontier.png)

Harry Markowitz's Modern Portfolio Theory (MPT, 1952) mathematically proved that **diversification can achieve the same expected return with lower risk**.

Considering expected returns, variances, and correlations of assets, the feasible portfolio space has two boundaries:

- **Minimum Variance Frontier**: the set of portfolios with the smallest variance at each expected return level
- **Efficient Frontier**: the upper half of the minimum variance frontier where expected returns are higher — only portfolios on this frontier are rational choices

Among these, the **tangency portfolio with the maximum Sharpe ratio** (excess return / volatility given the risk-free rate) is the theoretically optimal risky asset allocation.

#### Why Monte Carlo Instead of the Analytical Solution

The analytical solution to MPT requires inverting the covariance matrix. While theoretically complete, it has practical limitations:

| Limitation | Detail |
|-----------|--------|
| Return distribution assumption | The analytical solution assumes normal distributions, but stock returns show fat tails and asymmetry |
| Non-linear constraints | Adding constraints like weights sum to 1 and weights ≥ 0 (no short selling) makes the convex optimization complex |
| Small-sample instability | Covariance matrix estimation errors are amplified in the inverse, potentially producing extreme weights |

Monte Carlo simulation **samples 100,000 random weight combinations**, computes each Sharpe ratio, and takes the maximum.

```python
# rebalancer.py core logic
returns = prices.pct_change().dropna()
mean_ret = returns.mean()
cov = returns.cov()

best_sharpe, best_weights = -np.inf, None
for _ in range(100_000):
    w = np.random.dirichlet(np.ones(n))          # weights sum to 1 automatically
    port_ret = mean_ret @ w * 252                # annualized return
    port_vol = np.sqrt(w @ cov @ w * 252)        # annualized volatility
    sharpe   = (port_ret - RISK_FREE_RATE) / port_vol
    if sharpe > best_sharpe:
        best_sharpe, best_weights = sharpe, w    # track efficient frontier tangency
```

This approach reflects the actual return distribution without a normality assumption, naturally satisfies no-short-selling constraints via Dirichlet sampling, and has low implementation complexity suitable for monthly re-execution.

---

### Full Kelly vs Half Kelly

#### What is the Kelly Criterion

The Kelly Criterion derives the optimal bet fraction that **maximizes the geometric mean growth rate of capital** over repeated bets.

```
Full Kelly:  f* = (p × b - q) / b

  p = win probability
  b = risk/reward = avg win rate / avg loss rate
  q = 1 - p (loss probability)
```

For example, with win rate 60% and risk/reward 2.0: `f* = (0.6 × 2 - 0.4) / 2 = 0.4` → bet 40% of assets.

#### The Problem with Full Kelly

Full Kelly is theoretically optimal but relies on the premise that **input values are precisely known**.

In this system, `p` (win rate) and `b` (risk/reward) are **estimates** that XGBoost learned from historical data. There is no guarantee these numbers will be reproduced exactly in future markets.

| Scenario | Result |
|----------|--------|
| Estimated win rate 0.60 → actual 0.52 | Full Kelly overbets → drawdown spikes |
| Risk/reward overestimated | Portfolio drops sharply on losses |
| Streak of consecutive losses | Full Kelly is geometric-mean optimal but exceeds psychological tolerance |

#### Why Half Kelly

```
Half Kelly:  f = f* × 0.5
```

Half Kelly has the following mathematical properties:

- **Drawdown** size: reduced to approximately **75%** of Full Kelly
- **Long-term growth rate**: maintained at approximately **75%** of Full Kelly
- **Input error sensitivity**: loss magnitude is significantly reduced even when estimates differ from reality

In other words, sacrificing 25% of growth in exchange for a 25% reduction in drawdown risk. In an environment where ML model estimation error is inevitable, this buffer is critical for strategy continuity.

A **20% cap** is also applied in practice (`min(f_half, 0.20)`).

```python
# kelly.py
MAX_KELLY = 0.20                 # single position maximum 20% cap
f_full = (win_prob * b - q) / b  # Full Kelly
f_half = f_full * 0.5            # Half Kelly applied
return round(min(MAX_KELLY, max(0.0, f_half)), 4)
```

---

## Automation Schedule

| Time | Action |
|------|--------|
| 07:30 (trading days) | Parallel ML model retraining on KRX top-200 universe (outside market hours, 5-year daily data) |
| 22:30 / 23:30 (weekdays) | S&P 500 top-100 universe ML model retraining (auto-branching DST/standard, runs only in ET 09:30~10:00 window) |
| 08:00 (trading days) | Morning briefing (news on holdings + market overview) |
| 08:30 (1st of month, trading days) | Safe asset Monte Carlo rebalancing |
| 09:00 (trading days) | KR pending order execution — next-day open buy based on EOD signal |
| **09:05 (trading days)** | **KR paper open price confirmation** — `update_entry_prices("KR")` queries actual Open via FinanceDataReader; confirms `entry_price=None` positions |
| Every 5 min (all market hours) | ML position TP · ATR-SL · trailing stop · 7-trading-day forced close check + paper TP/SL evaluation |
| **15:31 (trading days)** | **EOD signal scan** — signal detection on completed daily candles after close → next-day open order scheduled (GATE B, Train/Serve Skew eliminated) |
| 22:30 / 23:30 (weekdays) | US pending order execution — next-day open buy based on EOD signal (auto-branching DST/standard) |
| **22:35 / 23:35 (weekdays)** | **US paper open price confirmation** — `update_entry_prices("US")` (auto-branching DST/standard) |
| 15:00 (trading days) | Daily technical analysis report |
| **15:35 (daily)** | **KR paper trading daily report** (immediately after Korean market close, sent via Telegram) |
| **05:30 / 06:30 (daily)** | **US paper trading daily report** (immediately after U.S. market close, auto-branching DST/standard, once per day) |

> **Trading day auto-detection**: `market_calendar.py` caches KRX annual trading days via pykrx, automatically excluding both weekends and public holidays.

---

## Bot Activation Gate

If existing holdings are present, the bot requires **manually selling all of them** before automated trading begins.

```
state.json: {"bot_active": false, "legacy_tickers": ["XXXX"], ...}
  ↓ manual sell completed
state.json: {"bot_active": true, ...}  →  automated trading starts
```

Telegram LLM features (Q&A, briefings, etc.) always work regardless of bot activation status.

**Telegram Controls**

| Command | Action |
|---------|--------|
| `/stop` | Pause automated trading (`user_stopped=true` flag — persists across bot restarts) |
| `/start` | Resume automated trading (clears `user_stopped=false`) |

> After `/stop`, even if the bot restarts due to a `.py` file edit, the `user_stopped` flag blocks automatic re-activation.

---

## U.S. Stocks — Integrated Margin Service

U.S. stocks such as QQQ and TLT are traded via **KIS Integrated Margin Service**.  
KRW balance is automatically converted to USD for settlement — no manual currency exchange required.

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

The CSV file is automatically sent via Telegram on every trade update.

---

## GPT AI Assistant (LangGraph ReAct)

Built with `create_react_agent` (langgraph.prebuilt) + `MemorySaver` checkpointer for per-user `thread_id` conversation history isolation.  
`/reset` increments the generation counter to switch to a new thread (immediate memory reset).

| Tool | Purpose |
|------|---------|
| `get_naver_finance` | Korean stock financials (PER, PBR, EPS, etc.) |
| `get_yahoo_finance` | U.S. stock financials |
| `get_naver_news` | Naver latest news search |
| `get_stock_signal` | Technical indicators + buy/sell signal analysis |
| `get_historical_price` | Historical closing price on a specific date |
| `get_account_balance` | Domestic + U.S. stock balance |
| `get_portfolio_status` | Safe asset portfolio status + rebalancing check |
| `get_paper_status` | Paper trading status — open positions + session cumulative stats |
| `set_conditional_order` | Register conditional order |
| `list_conditional_orders` | List conditional orders |
| `cancel_conditional_order` | Cancel conditional order |
| `list_trade_records` | Trade history query (open/closed/all) |
| `edit_trade_record` | Edit trade record (entry price, exit price, quantity, etc.) |

Responses using tools automatically display a **Context Recall score** (0.0~1.0) evaluated by gpt-5.4-mini.

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| Free text | LangChain AI assistant auto-reply |
| `/ask <question>` | Explicit GPT question |
| `/reset` | Clear conversation history |
| `/status` | Signal query for all stocks |
| `/balance` | Domestic + U.S. stock balance |
| `/portfolio` | Safe asset 70% portfolio status |
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
┌─────────────────────────────────────────────────────────────────────┐
│              macOS launchd — 3 daemons running continuously          │
├──────────────┬──────────────────────────┬───────────────────────────┤
│  runner.py   │  telegram_bot.py         │  dashboard.py             │
│  (Scheduler) │  (User Interface)        │  (Streamlit Monitoring)   │
└──────┬───────┴────────────┬─────────────┴───────────────────────────┘
       │                    │
       │    ┌───────────────▼───────────────────────────────────────┐
       │    │   langchain_agent.py  (LangGraph ReAct + gpt-5.5)     │
       │    │   MemorySaver checkpointer — per-user thread_id       │
       │    │   13 tools auto-called + Context Recall (gpt-5.4-mini)│
       │    └───────────────────────────────────────────────────────┘
       │
       ├── [07:30] ml/trainer.py — KRX 200+US 100 universe XGBoost parallel retraining
       │                          (momentum & reversion agents separately)
       │
       ├── [08:00] morning_briefer.py — holdings news + market overview
       │     LangGraph quality retry loop: regenerate up to 3× if score < 0.7
       │
       ├── [08:30/1st of month] portfolio/rebalancer.py
       │     100k Monte Carlo → max Sharpe weights
       │     → gpt-5.5 decides quantities → KIS order
       │
       ├── [every 5 min] check_ml_positions() + _run_paper_evaluate_kr()
       │     → ML position TP · ATR-SL · trailing stop · 7-trading-day forced close
       │     → Paper position TP/SL evaluation (KR market hours)
       │
       ├── [15:31] scan_growth_signals_eod()  ← B1 strategy: intraday → EOD scan
       │     market_regime.py → KOSPI MA5/MA20/RSI14 market regime
       │       ├── Downtrend (MA5 < MA20): all new buys blocked (only exits continue)
       │       └── Bear market (Close < MA20 OR RSI < 35): avg_win × 0.4 penalty
       │     signals/krx_universe.py → FinanceDataReader KOSPI+KOSDAQ screening (200 stocks)
       │       ↓ EOD completed candles only (no intraday synthesis — same features as training)
       │     signals/signal_graph.py (LangGraph StateGraph) — signal detection pipeline
       │       trigger_detect → (momentum/reversion/both) → select_best → END
       │       ├── Volume Explosion · BB Squeeze Breakout → Breakout Agent (_momentum.pkl)
       │       ├── BB Lower Bounce · RSI Oversold Escape · EMA Deviation Low → Pullback Agent (_reversion.pkl)
       │       └── Both conditions met → select agent with highest win_prob
       │       ↓ Platt Scaling (AUC≥0.58 AND win rate≥60% AND EV risk/reward≥1.5)
       │     → Telegram inline keyboard buy confirmation (✅Confirm / ❌Cancel)
       │       ✅ Confirmed → register in pending_orders → next-day 09:00 open buy
       │       ❌ Cancelled → order discarded  (sells handled automatically by check_ml_positions)
       │
       ├── [15:00] send_daily_summary() — daily technical analysis report
       ├── [15:35] paper_trader.daily_report(market="KR") — KR paper daily report
       └── [05:30·06:30] paper_trader.daily_report(market="US") — US paper daily report (once/day)

Common layer:
  KIS API (KR real-time price · orders · balance)  ·  yfinance ≥1.2 (daily · US 5-min, MultiIndex support)
  FinanceDataReader (KRX universe screening)  ·  ml/models/*.pkl
  trade_history.csv  ·  state.json
```

> **yfinance 1.2+ compatibility**: `group_by` parameter removed; MultiIndex columns accessed via `xs(ticker, level=1)`.  
> Duplicate column/index defense code (`duplicated()` removal) applied to daily and intraday DataFrames.

---

## Project Structure

```
quant_trader/
├── config.py               # Strategy parameters / API configuration
├── stocks.py               # Watchlist (STOCKS, US_STOCKS)
├── runner.py               # Scheduler (07:30 retraining · scan · rebalancing · 09:05/22:35/23:35 open price update)
├── telegram_bot.py         # Telegram bot
├── langchain_agent.py      # LangGraph ReAct AI assistant (MemorySaver, per-thread conversation history)
├── pending_confirmations.py # EOD buy signal confirmation queue (inline keyboard ✅/❌)
├── trader.py               # KIS API (domestic + U.S. stocks)
├── trade_logger.py         # Trade history CSV recorder + Telegram sender
├── backtest_ml.py          # 45-day intraday ML backtest
├── backtest_walkforward.py # Walk-forward backtest (cost-adjusted, live trading gate)
├── paper_trader.py         # Paper trading engine (KR+US, 2-stage fill, Circuit Breaker, P4 gate, daily report)
├── position_manager.py     # ML position tracking and bot activation state management
├── tests/
│   ├── test_triple_barrier.py      # Triple-Barrier labeling unit tests
│   ├── test_paper_trader.py        # Paper trading engine unit tests (60 cases, V8 2-stage fill)
│   └── test_position_manager.py    # ML position tracking unit tests (20 cases)
├── morning_briefer.py      # Morning briefing (LangGraph quality retry loop)
├── data_fetcher.py         # yfinance daily + KIS intraday
├── indicators.py           # MA / RSI / Bollinger Bands
├── strategy.py             # MA/RSI buy · sell signals
├── notifier.py             # Telegram message builder
├── news_fetcher.py         # Naver News API
├── naver_finance.py        # Naver Finance fundamentals scraper
├── conditional_orders.py   # Conditional orders (price/return conditions)
├── market_calendar.py      # KRX trading day cache (weekends + public holidays auto-detected)
├── market_regime.py        # KOSPI market regime filter (downtrend block / bear market detection)
├── gpt_agent.py            # GPT tool functions (called from langchain_agent)
├── signals/
│   ├── signal_graph.py     # LangGraph StateGraph signal detection pipeline
│   ├── scanner.py          # Technical trigger detection + ML agent evaluation (called from signal_graph)
├── state.json              # Bot activation gate
├── trade_history.csv       # Trade history
├── ml/
│   ├── features.py         # Feature engineering (16 features, incl. atr_pct) + Triple-Barrier labeling + agent trigger filter
│   ├── model.py            # XGBoost training & prediction (agent="" | "momentum" | "reversion")
│   ├── trainer.py          # KRX+US universe parallel retraining (8 threads, 5-year data, per-agent)
│   └── models/             # {ticker}_momentum.pkl / {ticker}_reversion.pkl
├── signals/
│   ├── scanner.py          # Technical triggers + ML prediction (blacklist applied)
│   ├── krx_universe.py     # KRX full universe first-pass screening
│   ├── us_universe.py      # S&P 500 full universe first-pass screening (ET volume normalization)
│   └── alert.py            # Growth stock signal alert message
├── portfolio/
│   ├── kelly.py            # Kelly Criterion position sizing
│   ├── safe_portfolio.py   # Safe asset weight tracking
│   └── rebalancer.py       # Monte Carlo rebalancing
├── logs/
│   └── trader.log
├── com.quant.trader.plist
├── com.quant.telegrambot.plist
└── com.quant.dashboard.plist
```

---

## Walk-forward Backtest (Cost-Adjusted)

`backtest_walkforward.py` — pre-live gate validation. Live trading only resumes when expected value is positive after cost deduction.

**Structure**: 2-year training window / 3-month test window / 3-month sliding step

**Cost Model (round-trip)**

| Item | Rate |
|------|------|
| Commission (buy + sell) | 0.03% |
| Slippage (next-day open limit order) | 0.05% |
| Securities transaction tax (KRX only) | 0.18% |
| **KRX total cost** | **0.26%** |

**G1 Grid Adoption Result** (KR, 2026-06-10)

| Item | Phase 4 (TP=7%/SL=7%) | **G1 Adopted (TP=15%/SL=6%)** |
|------|----------------------|-------------------------------|
| Slippage | 0.25% | **0.05%** (next-day open limit order) |
| KRX total cost | 0.46% | **0.26%** |
| After-cost EV | +1.019% | **+1.468%** |
| TP close ratio | — | 27% (time-expiry profit improvement is the main EV driver) |
| Time-expiry avg return | +0.140% | **+2.529%** |

> 20-combination walk-forward grid search → G1 (TP=15%/SL=6%) adopted as highest EV combination.  
> US backtest is unvalidated — exploratory operation (`PAPER_BACKTEST_EV_US = None`).

```bash
python3 backtest_walkforward.py
```

Sample output:
```
✅ Gate passed — live trading may resume
❌ Gate failed — no edge after cost deduction. Live trading not allowed
```

---

## Paper Trading (`paper_trader.py`)

2-week paper validation phase before live trading. Operates only when `LIVE_TRADING=False` — no real API calls.

**Design Principles**
- Fill price and costs both shared with `backtest_walkforward._apply_costs()` → exact numerical parity with backtests
- Signal timing, fill price assumption, and exit timing must not be arbitrarily adjusted (bias-free measurement)
- **2-stage fill structure**: On signal, record `entry_price=None` + `eod_close` (prior closing price) → after next-day market open, query actual open price via FinanceDataReader and confirm. Skip exit checks until confirmed.

**Key Features**

| Feature | Description |
|---------|-------------|
| P1 Signal recording | `log_paper_signal()` — records ticker + EOD close on signal (`entry_price=None` for 2-stage wait) |
| P1-2 Open price confirmation | `update_entry_prices(market)` — queries actual Open at 09:05 KST (KR) / ET 09:35 (US), updates `entry_price`, computes gap slippage |
| P2 Metrics calculation | `get_metrics(market=None)` — KR/US/total separated EV · win rate · CI · consecutive loss |
| P3 Circuit Breaker | Auto-monitors 6 conditions (EV ≤ −0.5%, CI lower < −1%, 8 consecutive losses, etc.) |
| P4 Gate evaluation | `evaluate_live_gate()` — 60-day · 50-trade · EV≥0.3% · AUC≥0.55 live trading entry checklist |
| Daily report | KR: **15:35**, US: **05:30/06:30** auto-sent via Telegram |
| Weekly summary | Auto-sent Sundays at 20:00 |

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

> Current status: **2-week paper trading in progress** (started 2026-06-10, G1 parameters applied).  
> KR backtest EV = **+1.468%** / US = exploratory operation (no benchmark)

---

## Backtest Results (Reference — Pre-G1 Parameters)

> ⚠️ **Note**: The results below are based on **pre-G1 parameters (before 2026-06-10)**.  
> Differences from the current G1 strategy: SL `-7%→-6%`, TP `dynamic avg_win→+15% fixed`, signal basis `intraday bars→EOD completed candles`.  
> The actual G1 performance is measured by the `backtest_walkforward.py` walk-forward result: **net EV +1.468% per trade**.

**45-day intraday backtest** (2026-04-11 ~ 2026-05-26, KRX 200 + S&P 500 100 = 300 stocks)  
Conditions: AUC ≥ 0.58, win rate ≥ 0.60, EV risk/reward ≥ 1.5 (Platt Scaling calibration not applied)

| Stock | Trades | Win Rate | Risk/Reward | Avg Return |
|-------|--------|----------|-------------|------------|
| Wonik IPS (240810.KQ) | 1 | 100.0% | — | +14.46% |
| Hanjin KAL (180640.KS) | 1 | 100.0% | — | +10.10% |
| HD Hyundai (267250.KS) | 1 | 100.0% | — | +4.01% |
| KEPCO (015760.KS) | 2 | 50.0% | 2.09 | +2.70% |
| Taesung (323280.KQ) | 4 | 25.0% | 8.92 | +7.13% |

**Total: 38 trades | 16 stocks with signals (KRX-weighted)**

> More signals expected after applying Platt Scaling calibration with win rate ≥ 0.60 threshold  
> Intraday signal detection → buy at next bar open → auto-exit on the conditions below

### Auto-Exit Conditions (checked every 5 minutes)

| Condition | Threshold | Alert |
|-----------|-----------|-------|
| ✅ Take Profit | Current price ≥ buy price × (1 + avg_win) | "Take Profit" |
| 🔴 ATR Stop Loss | Current price ≤ buy price − 2 × ATR(14) | "ATR Stop Loss" |
| 📉 Trailing Stop | After entering profit zone, −2.5% from high or −1×ATR drop | "Trailing Stop" |
| ⏰ Time Exit | 7 **KRX trading days** elapsed since buy (excluding holidays) | "Time Exit" |

> **ATR Stop Loss**: Custom stop based on stock-specific volatility instead of a fixed −7%. Wider stop for high-ATR stocks reduces whipsaw exits.  
> **Trailing Stop**: Tracks the high after entering profit territory to lock in gains.

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

# 45-day intraday backtest
python3 backtest_ml.py
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

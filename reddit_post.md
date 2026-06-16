# Reddit Post — r/algotrading & r/quant

---

## Title

**Built a dual-market (KRX + S&P 500) algo trader in Python: MPT rebalancing + XGBoost signals + Half-Kelly sizing — backtest EV +1.47% per trade after costs**

---

## Body

Hey everyone. Been quietly building this as a side project for the past few months and finally feel like it's at a point worth sharing.

**The problem I wanted to solve:**
Most retail algo resources focus exclusively on US markets. I trade both KRX (Korean Exchange) and S&P 500 and couldn't find a clean open-source setup that handles both markets, different trading hours, DST switching, and Korean public holidays — so I built one.

**The approach:**

The portfolio splits into two layers:

- **70% safe assets** — QQQ, Samsung, TLT, Gold ETF. Weights optimized monthly via Monte Carlo simulation (100k iterations) targeting max Sharpe. No analytical MPT inversion — avoids covariance estimation blowup on small samples.

- **30% growth stocks** — Dual XGBoost agents: a *breakout agent* and a *pullback agent*, each trained only on historical bars where their respective triggers fired (volume explosion, BB squeeze, RSI oversold escape, etc.). Signals are detected on **EOD completed candles only** to eliminate train/serve skew. Position sizing uses Half-Kelly capped at 20%, with ATR-based Risk Parity as a secondary constraint.

Entry requires passing a 6-layer filter: market regime check, MA200 trend, model AUC ≥ 0.58, win rate ≥ 60% (Platt-calibrated), EV risk/reward ≥ 1.5, and at least one technical trigger.

**Results (walk-forward backtest, KRX, G1 grid):**
- Net EV per trade: **+1.47%** after 0.26% round-trip costs (0.05% slippage on next-day open limit)
- TP=15% / SL=6% / 7-trading-day time barrier
- Currently in a 2-week live paper trading phase before going live

**Repo:** https://github.com/sonjong980304-tech/quant_trader-

Happy to discuss any of the design decisions — especially the dual-agent routing and the EOD-only signal detection. Would love feedback from anyone who's dealt with train/serve skew in live deployments.

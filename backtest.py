"""
backtest.py - vectorbt를 이용한 전략 A vs 전략 B 백테스트
결과 차트는 results/ 폴더에 PNG로 저장
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import vectorbt as vbt

from data_fetcher import fetch_ohlcv
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals
from config import STOCKS, STRATEGY_A, STRATEGY_B, BACKTEST_PERIOD_YEARS, BACKTEST_INIT_CASH

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# macOS 한글 폰트 설정
plt.rcParams["axes.unicode_minus"] = False
for font in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
    if font in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams["font.family"] = font
        break


def run_backtest_for_strategy(ticker: str, stock_name: str, strategy: dict) -> dict:
    """
    단일 종목 + 단일 전략에 대해 vectorbt 백테스트 수행.
    반환: 성과 지표 딕셔너리
    """
    short = strategy["short_window"]
    long_ = strategy["long_window"]

    # 데이터 수집 및 지표 계산
    df = fetch_ohlcv(ticker, period_years=BACKTEST_PERIOD_YEARS)
    df = add_all_indicators(df, short=short, long=long_,
                            rsi_period=strategy["rsi_period"])
    df = detect_crossover(df, short=short, long=long_)
    df = generate_signals(df, strategy=strategy)

    close  = df["Close"]
    entries = df["buy_signal"].astype(bool)
    # 전량 매도 OR 분할 매도를 둘 다 exits로 처리
    exits   = (df["sell_full"] | df["sell_partial"]).astype(bool)

    # vectorbt 포트폴리오 시뮬레이션
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=BACKTEST_INIT_CASH,
        freq="D",
        sl_stop=None,
    )

    stats = pf.stats()

    # 핵심 지표 추출
    total_return = pf.total_return()
    years        = BACKTEST_PERIOD_YEARS
    cagr         = (1 + total_return) ** (1 / years) - 1
    mdd          = pf.max_drawdown()
    sharpe       = pf.sharpe_ratio()

    # 승률: 수익 거래 수 / 전체 거래 수
    trades = pf.trades.records_readable
    if len(trades) > 0:
        win_rate    = (trades["PnL"] > 0).mean()
        total_trades = len(trades)
    else:
        win_rate     = 0.0
        total_trades = 0

    return {
        "strategy":      strategy["name"],
        "ticker":        ticker,
        "stock_name":    stock_name,
        "cagr":          cagr,
        "mdd":           mdd,
        "sharpe":        sharpe,
        "win_rate":      win_rate,
        "total_trades":  total_trades,
        "cumulative":    pf.cumulative_returns(),
        "portfolio":     pf,
    }


def print_summary(results_a: list, results_b: list):
    """
    전략 A vs B 성과 비교표를 터미널에 출력.
    """
    header = f"{'종목':<12} {'전략':<18} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'승률':>7} {'매매수':>7}"
    print("\n" + "=" * 75)
    print(" 전략 A vs 전략 B 백테스트 결과 (최근 3년)")
    print("=" * 75)
    print(header)
    print("-" * 75)

    for r in results_a + results_b:
        print(
            f"{r['stock_name']:<12} "
            f"{r['strategy']:<18} "
            f"{r['cagr']:>7.1%} "
            f"{r['mdd']:>7.1%} "
            f"{r['sharpe']:>8.2f} "
            f"{r['win_rate']:>6.1%} "
            f"{r['total_trades']:>7}"
        )

    print("=" * 75)


def save_chart(results_a: list, results_b: list):
    """
    종목별 누적수익 곡선을 PNG로 저장.
    """
    n_stocks = len(STOCKS)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(12, 4 * n_stocks))
    if n_stocks == 1:
        axes = [axes]

    tickers = list(STOCKS.keys())
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        stock_name = STOCKS[ticker]

        ra = next((r for r in results_a if r["ticker"] == ticker), None)
        rb = next((r for r in results_b if r["ticker"] == ticker), None)

        if ra:
            cum = ra["cumulative"] * 100
            ax.plot(cum.index, cum.values, label=f"전략A({ra['cagr']:.1%})", color="steelblue", linewidth=1.5)
        if rb:
            cum = rb["cumulative"] * 100
            ax.plot(cum.index, cum.values, label=f"전략B({rb['cagr']:.1%})", color="tomato", linewidth=1.5, linestyle="--")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"{stock_name} 누적 수익률 (%)", fontsize=12)
        ax.set_ylabel("누적 수익률 (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = "results/backtest_result.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"\n  차트 저장 완료 → {chart_path}")


def main():
    print("\n[백테스트 시작] 데이터 수집 중...")

    results_a = []
    results_b = []

    for ticker, name in STOCKS.items():
        print(f"\n  {name}({ticker}) 처리 중...")
        try:
            ra = run_backtest_for_strategy(ticker, name, STRATEGY_A)
            results_a.append(ra)
            print(f"    전략A: CAGR={ra['cagr']:.1%}, MDD={ra['mdd']:.1%}, Sharpe={ra['sharpe']:.2f}")
        except Exception as e:
            print(f"    전략A 실패: {e}")

        try:
            rb = run_backtest_for_strategy(ticker, name, STRATEGY_B)
            results_b.append(rb)
            print(f"    전략B: CAGR={rb['cagr']:.1%}, MDD={rb['mdd']:.1%}, Sharpe={rb['sharpe']:.2f}")
        except Exception as e:
            print(f"    전략B 실패: {e}")

    print_summary(results_a, results_b)
    save_chart(results_a, results_b)
    print("\n[백테스트 완료]\n")


if __name__ == "__main__":
    main()

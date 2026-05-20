"""
backtest.py - vectorbt를 이용한 전략 백테스트
결과 차트는 results/ 폴더에 PNG로 저장
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import vectorbt as vbt

from data_fetcher import fetch_ohlcv
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals
from config import (
    STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD,
    BACKTEST_PERIOD_YEARS, BACKTEST_INIT_CASH,
    TRAILING_STOP_RATIO,
)

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# macOS 한글 폰트 설정
plt.rcParams["axes.unicode_minus"] = False
for font in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
    if font in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams["font.family"] = font
        break


def run_backtest(ticker: str, stock_name: str, eval_days: int = None) -> dict:
    """
    단일 종목 vectorbt 백테스트.
    eval_days: 평가 기간 일수 (None이면 BACKTEST_PERIOD_YEARS 전체 사용).
    지표 계산을 위해 항상 최소 6개월 데이터를 수집한 뒤 슬라이싱.
    손절(-5%) / 익절(+15%) vectorbt 파라미터 적용.
    """
    fetch_years = max(BACKTEST_PERIOD_YEARS, 0.5) if eval_days else BACKTEST_PERIOD_YEARS
    df = fetch_ohlcv(ticker, period_years=fetch_years)
    df = add_all_indicators(df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD)
    df = detect_crossover(df, short=MA_SHORT, long=MA_LONG)
    df = generate_signals(df)

    if eval_days is not None:
        df = df.iloc[-eval_days:]

    close   = df["Close"]
    entries = df["buy_signal"].astype(bool)
    exits   = (df["sell_full"] | df["sell_partial"]).astype(bool)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=BACKTEST_INIT_CASH,
        freq="D",
        sl_stop=abs(TRAILING_STOP_RATIO),
        sl_trail=True,
    )

    total_return = pf.total_return()
    years        = BACKTEST_PERIOD_YEARS
    cagr         = (1 + total_return) ** (1 / years) - 1
    mdd          = pf.max_drawdown()
    sharpe       = pf.sharpe_ratio()

    trades = pf.trades.records_readable
    if len(trades) > 0:
        win_rate     = (trades["PnL"] > 0).mean()
        total_trades = len(trades)
    else:
        win_rate     = 0.0
        total_trades = 0

    return {
        "ticker":       ticker,
        "stock_name":   stock_name,
        "cagr":         cagr,
        "mdd":          mdd,
        "sharpe":       sharpe,
        "win_rate":     win_rate,
        "total_trades": total_trades,
        "cumulative":   pf.cumulative_returns(),
    }


def print_summary(results: list):
    """백테스트 성과표를 터미널에 출력"""
    header = f"{'종목':<12} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'승률':>7} {'매매수':>7}"
    print("\n" + "=" * 60)
    print(f" 백테스트 결과 (MA{MA_SHORT}/MA{MA_LONG} + 거래량/캔들 전략, 최근 {BACKTEST_PERIOD_YEARS}년)")
    print("=" * 60)
    print(header)
    print("-" * 60)

    for r in results:
        print(
            f"{r['stock_name']:<12} "
            f"{r['cagr']:>7.1%} "
            f"{r['mdd']:>7.1%} "
            f"{r['sharpe']:>8.2f} "
            f"{r['win_rate']:>6.1%} "
            f"{r['total_trades']:>7}"
        )

    print("=" * 60)


def save_chart(results: list):
    """종목별 누적수익 곡선을 PNG로 저장"""
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, r in enumerate(results):
        ax  = axes[i]
        cum = r["cumulative"] * 100
        ax.plot(cum.index, cum.values,
                label=f"전략 (CAGR {r['cagr']:.1%})",
                color="steelblue", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"{r['stock_name']} 누적 수익률 (%)", fontsize=12)
        ax.set_ylabel("누적 수익률 (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = "results/backtest_result.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"\n  차트 저장 완료 → {chart_path}")


def main(eval_days: int = None):
    """
    eval_days: 평가 기간 일수. None이면 BACKTEST_PERIOD_YEARS 전체.
    예) eval_days=22 → 최근 약 1개월(22거래일) 평가
    """
    period_label = f"최근 {eval_days}거래일" if eval_days else f"최근 {BACKTEST_PERIOD_YEARS}년"
    print(f"\n[백테스트 시작] {period_label}")

    results = []
    for ticker, name in STOCKS.items():
        print(f"\n  {name}({ticker}) 처리 중...")
        try:
            r = run_backtest(ticker, name, eval_days=eval_days)
            results.append(r)
            print(f"    CAGR={r['cagr']:.1%}, MDD={r['mdd']:.1%}, Sharpe={r['sharpe']:.2f}, 매매수={r['total_trades']}")
        except Exception as e:
            print(f"    실패: {e}")

    print_summary(results)
    save_chart(results)
    print("\n[백테스트 완료]\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
rebalancer.py - 월 1회 자동 몬테카를로 리밸런싱

흐름:
  1. 안전자산 4종목 최신 5개년 데이터 다운로드
  2. 몬테카를로 시뮬레이션 100,000번 → 최대 샤프비율 비중 산출
  3. 현재 보유 비중과 비교
  4. LLM(GPT)이 주식 가격 감안해 실제 매매 수량 결정
  5. 텔레그램으로 리밸런싱 결과 전송
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import yfinance as yf
from openai import OpenAI

from config import OPENAI_API_KEY
from notifier import send_telegram

logger = logging.getLogger(__name__)

SAFE_TICKERS = {
    "QQQ":       "QQQ (나스닥 ETF)",
    "005930.KS": "삼성전자",
    "TLT":       "TLT (미국 장기채)",
    "411060.KS": "ACE KRX금현물",
}

N_SIMULATIONS = 100_000
RISK_FREE_RATE = 0.045  # 연 무위험수익률


# ─────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────

def _fetch_5y_prices() -> pd.DataFrame:
    """안전자산 4종목 최신 5개년 종가 데이터."""
    end   = date.today()
    start = end - relativedelta(years=5)
    data  = {}
    for ticker, name in SAFE_TICKERS.items():
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        if not df.empty:
            data[ticker] = df["Close"].squeeze()
            logger.info("  %s: %d일 데이터", name, len(df))
        else:
            logger.warning("  %s: 데이터 없음", name)

    prices = pd.DataFrame(data).dropna()
    logger.info("공통 거래일: %d일 (%s ~ %s)",
                len(prices), prices.index[0].date(), prices.index[-1].date())
    return prices


# ─────────────────────────────────────────────
# 몬테카를로 시뮬레이션
# ─────────────────────────────────────────────

def run_monte_carlo(prices: pd.DataFrame) -> dict:
    """
    100,000번 몬테카를로 시뮬레이션으로 최대 샤프비율 포트폴리오 산출.

    반환:
      weights      : {ticker: 최적 비중}
      annual_return: 예상 연간 수익률
      annual_vol   : 예상 연간 변동성
      sharpe       : 샤프비율
      corr         : 상관계수 행렬 (dict)
    """
    returns     = prices.pct_change().dropna()
    mean_ret    = returns.mean()
    cov_matrix  = returns.cov()
    n_assets    = len(returns.columns)
    tickers     = list(returns.columns)

    results     = np.zeros((3, N_SIMULATIONS))
    weights_all = np.zeros((N_SIMULATIONS, n_assets))

    np.random.seed(int(datetime.now().strftime("%Y%m")))  # 월별 시드 고정

    for i in range(N_SIMULATIONS):
        w = np.random.random(n_assets)
        w = w / w.sum()
        weights_all[i] = w

        port_ret = float(np.sum(mean_ret * w) * 252)
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w))))
        sharpe   = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0

        results[0, i] = port_ret
        results[1, i] = port_vol
        results[2, i] = sharpe

    best_idx = results[2].argmax()
    best_w   = weights_all[best_idx]

    weights  = {t: round(float(w), 4) for t, w in zip(tickers, best_w)}
    corr     = returns.corr().round(3).to_dict()

    return {
        "weights":       weights,
        "annual_return": round(results[0, best_idx], 4),
        "annual_vol":    round(results[1, best_idx], 4),
        "sharpe":        round(results[2, best_idx], 4),
        "corr":          corr,
        "period_start":  str(prices.index[0].date()),
        "period_end":    str(prices.index[-1].date()),
    }


# ─────────────────────────────────────────────
# LLM 수량 조정
# ─────────────────────────────────────────────

def _llm_decide_rebalance(
    result: dict,
    total_safe_budget: float,
    current_holdings: dict,
    current_prices: dict,
) -> str:
    """
    GPT가 최적 비중 + 현재 주가 기준 실제 매매 수량을 결정.

    current_holdings : {ticker: {"qty": int, "avg_price": float}}
    current_prices   : {ticker: float}
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    weights_text = "\n".join(
        f"  {SAFE_TICKERS.get(t, t)}: {w*100:.1f}%"
        for t, w in result["weights"].items()
    )
    holdings_text = "\n".join(
        f"  {SAFE_TICKERS.get(t, t)}: {h['qty']}주 × {current_prices.get(t, 0):,.0f}원 "
        f"= {h['qty'] * current_prices.get(t, 0):,.0f}원"
        for t, h in current_holdings.items()
    ) or "  (보유 없음)"
    prices_text = "\n".join(
        f"  {SAFE_TICKERS.get(t, t)}: {p:,.0f}원"
        for t, p in current_prices.items()
    )

    prompt = f"""당신은 포트폴리오 리밸런싱 전문가입니다.

몬테카를로 시뮬레이션(10만번) 결과 최적 비중 (최대 샤프비율):
{weights_text}

안전자산 총 예산: {total_safe_budget:,.0f}원
예상 수익률: {result['annual_return']*100:.1f}% | 변동성: {result['annual_vol']*100:.1f}% | 샤프: {result['sharpe']:.3f}

현재 보유 현황:
{holdings_text}

현재 주가:
{prices_text}

위 정보를 바탕으로:
1. 각 종목별 목표 금액 = 예산 × 최적비중
2. 현재 보유 금액과의 차이 계산
3. 주가가 애매해서 정확한 비중이 불가능한 경우, 목표에 가장 근접한 수량으로 조정
4. 매수/매도가 필요한 종목과 수량을 구체적으로 알려주세요

형식:
- [매수/매도/유지] 종목명: X주 (목표 Y원 → 현재 Z원, 차이 ±W원)
- 최종 예상 비중 및 샤프비율 달성 여부 코멘트
"""

    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "포트폴리오 리밸런싱 전문가. 간결하고 정확하게 답변."},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────

def run_monthly_rebalance(total_asset: float, current_holdings: dict):
    """
    월 1회 리밸런싱 실행.

    total_asset       : 전체 자산 (현금 + 평가액)
    current_holdings  : {ticker: {"qty": int, "avg_price": float}}
    """
    from config import SAFE_ASSET_RATIO

    now = datetime.now()
    logger.info("월간 리밸런싱 시작 (%s)", now.strftime("%Y-%m"))

    send_telegram(
        f"🔄 <b>월간 포트폴리오 리밸런싱 시작</b>\n"
        f"{now.strftime('%Y년 %m월')} — 최신 5개년 데이터 기준"
    )

    try:
        # 1. 데이터 수집
        logger.info("  데이터 수집 중...")
        prices_df = _fetch_5y_prices()

        # 2. 몬테카를로
        logger.info("  몬테카를로 시뮬레이션 %d회 실행 중...", N_SIMULATIONS)
        result = run_monte_carlo(prices_df)

        # 3. 현재가 조회
        current_prices = {}
        for ticker in SAFE_TICKERS:
            try:
                t = yf.Ticker(ticker)
                current_prices[ticker] = float(t.fast_info.last_price or 0)
            except Exception:
                current_prices[ticker] = 0.0

        # 4. LLM 수량 결정
        safe_budget = total_asset * SAFE_ASSET_RATIO
        logger.info("  LLM 리밸런싱 수량 결정 중...")
        llm_advice = _llm_decide_rebalance(
            result, safe_budget, current_holdings, current_prices
        )

        # 5. 결과 전송
        weights_lines = "\n".join(
            f"  {SAFE_TICKERS.get(t, t)}: {w*100:.1f}%"
            for t, w in result["weights"].items()
        )
        send_telegram(
            f"📊 <b>몬테카를로 최적 비중 ({now.strftime('%Y.%m')})</b>\n"
            f"데이터: {result['period_start']} ~ {result['period_end']}\n\n"
            f"{weights_lines}\n\n"
            f"예상 수익률: {result['annual_return']*100:.1f}%\n"
            f"변동성:     {result['annual_vol']*100:.1f}%\n"
            f"샤프비율:   {result['sharpe']:.3f}"
        )
        send_telegram(
            f"🤖 <b>LLM 리밸런싱 결정</b>\n\n{llm_advice}"
        )

        # 6. config의 SAFE_WEIGHTS 갱신 (런타임 반영)
        _update_safe_weights(result["weights"])

        logger.info("  월간 리밸런싱 완료")

    except Exception as e:
        logger.error("월간 리밸런싱 실패: %s", e)
        send_telegram(f"⚠️ 월간 리밸런싱 오류: {e}")


def _update_safe_weights(new_weights: dict):
    """config.py의 SAFE_WEIGHTS를 새 비중으로 업데이트."""
    import re

    config_path = "/Users/gyuyeong/quant_trader/config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = ["SAFE_WEIGHTS = {\n"]
    for ticker, w in new_weights.items():
        lines.append(f'    "{ticker}": {w},\n')
    lines.append("}\n")
    new_block = "".join(lines)

    content = re.sub(
        r"SAFE_WEIGHTS\s*=\s*\{[^}]*\}",
        new_block.rstrip("\n"),
        content,
        flags=re.DOTALL,
    )
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("config.py SAFE_WEIGHTS 업데이트 완료: %s", new_weights)

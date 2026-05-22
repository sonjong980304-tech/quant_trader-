"""
safe_portfolio.py - 안전자산 70% 포트폴리오 관리

몬테카를로 최적화 결과 (최대 샤프비율):
  QQQ          22.3%
  삼성전자      27.3%
  TLT           0.2%
  ACE KRX금현물 50.3%

전체 자산의 70%를 위 비중으로 배분하고,
현재 보유 비중과 목표 비중의 괴리를 계산해 리밸런싱 신호를 제공.
"""

import logging
import yfinance as yf

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 목표 비중 (안전자산 70% 내부 비중)
# ─────────────────────────────────────────────
SAFE_WEIGHTS = {
    "QQQ":        0.223,   # 미국 나스닥 ETF
    "005930.KS":  0.273,   # 삼성전자
    "TLT":        0.002,   # 미국 장기채 ETF
    "411060.KS":  0.503,   # ACE KRX 금현물
}

SAFE_RATIO = 0.70   # 전체 자산 중 안전자산 비율
GROWTH_RATIO = 0.30  # 전체 자산 중 급등주 비율

REBALANCE_THRESHOLD = 0.05  # 목표 대비 ±5% 이상 괴리 시 리밸런싱 권고


def get_current_prices() -> dict:
    """안전자산 4종목 현재가 조회 (yfinance)."""
    prices = {}
    for ticker in SAFE_WEIGHTS:
        try:
            t = yf.Ticker(ticker)
            price = t.fast_info.last_price
            prices[ticker] = float(price) if price else 0.0
        except Exception as e:
            logger.warning("현재가 조회 실패 (%s): %s", ticker, e)
            prices[ticker] = 0.0
    return prices


def calc_target_amounts(total_asset: float) -> dict:
    """
    전체 자산 기준 각 종목의 목표 투자금액 계산.

    반환: {ticker: 목표금액(원)}
    """
    safe_budget = total_asset * SAFE_RATIO
    return {ticker: safe_budget * w for ticker, w in SAFE_WEIGHTS.items()}


def calc_rebalance(holdings: dict, total_asset: float) -> list:
    """
    현재 보유 현황과 목표 비중 비교 → 리밸런싱 필요 종목 반환.

    holdings : {ticker: {"qty": int, "avg_price": float}}
    total_asset : 전체 자산 (현금 + 평가액)

    반환: [{"ticker", "name", "target_pct", "current_pct", "diff_pct", "action", "amount"}]
    """
    prices = get_current_prices()
    targets = calc_target_amounts(total_asset)

    names = {
        "QQQ":       "QQQ",
        "005930.KS": "삼성전자",
        "TLT":       "TLT",
        "411060.KS": "ACE KRX금현물",
    }

    result = []
    for ticker, target_amount in targets.items():
        price = prices.get(ticker, 0)
        qty = holdings.get(ticker, {}).get("qty", 0)
        current_amount = qty * price

        target_pct  = SAFE_WEIGHTS[ticker] * SAFE_RATIO * 100
        current_pct = (current_amount / total_asset * 100) if total_asset > 0 else 0
        diff_pct    = current_pct - target_pct

        action = None
        if abs(diff_pct) >= REBALANCE_THRESHOLD * 100:
            action = "매도 (비중 과다)" if diff_pct > 0 else "매수 (비중 부족)"

        result.append({
            "ticker":      ticker,
            "name":        names.get(ticker, ticker),
            "price":       price,
            "qty":         qty,
            "target_pct":  round(target_pct, 2),
            "current_pct": round(current_pct, 2),
            "diff_pct":    round(diff_pct, 2),
            "action":      action,
            "gap_amount":  round(target_amount - current_amount, 0),
        })

    return result


def format_rebalance_report(holdings: dict, total_asset: float) -> str:
    """텔레그램 전송용 리밸런싱 리포트 문자열 생성."""
    rows = calc_rebalance(holdings, total_asset)
    safe_budget = total_asset * SAFE_RATIO
    growth_budget = total_asset * GROWTH_RATIO

    lines = [
        "<b>📊 안전자산 포트폴리오 현황</b>",
        f"전체 자산: {total_asset:,.0f}원",
        f"  안전자산(70%): {safe_budget:,.0f}원",
        f"  급등주(30%):   {growth_budget:,.0f}원\n",
        "<b>종목별 비중</b>",
    ]

    needs_rebalance = False
    for r in rows:
        icon = "⚠️" if r["action"] else "✅"
        lines.append(
            f"{icon} {r['name']} ({r['ticker']})\n"
            f"   목표 {r['target_pct']:.1f}% | 현재 {r['current_pct']:.1f}%"
            f" ({r['diff_pct']:+.1f}%)"
        )
        if r["action"]:
            needs_rebalance = True
            lines.append(f"   → {r['action']} | 괴리금액: {abs(r['gap_amount']):,.0f}원")

    if needs_rebalance:
        lines.append("\n⚠️ 리밸런싱 권고 종목이 있습니다.")
    else:
        lines.append("\n✅ 모든 종목이 목표 비중 이내입니다.")

    return "\n".join(lines)

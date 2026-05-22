"""
kelly.py - 켈리 공식 기반 포지션 사이징 (급등주 30% 자산 대상)

f* = (p * b - q) / b
  p = 승률 (win probability)
  b = 손익비 (avg_win / avg_loss)
  q = 1 - p
"""


def kelly_fraction(win_prob: float, avg_win: float, avg_loss: float, max_fraction: float = 0.25) -> float:
    """
    켈리 비율 계산.

    win_prob : 승률 (0~1)
    avg_win  : 성공 시 평균 수익률 (예: 0.072 → 7.2%)
    avg_loss : 실패 시 평균 손실률, 양수로 전달 (예: 0.038 → 3.8%)
    max_fraction : 안전장치 — 이 비율을 초과하지 않도록 캡 (기본 25%)

    반환: 투입 비율 (0~max_fraction)
    """
    if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = avg_win / avg_loss          # 손익비
    q = 1.0 - win_prob
    f = (win_prob * b - q) / b      # 켈리 비율

    return round(max(0.0, min(f, max_fraction)), 4)


def position_size(
    growth_cash: float,
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    current_price: float,
    max_fraction: float = 0.25,
) -> tuple[int, float, float]:
    """
    켈리 비율로 매수 수량 계산.

    growth_cash   : 급등주 전용 현금 (전체 자산의 30%)
    current_price : 현재가

    반환: (매수 수량, 켈리 비율, 투자 금액)
    """
    f = kelly_fraction(win_prob, avg_win, avg_loss, max_fraction)
    amount = growth_cash * f
    qty = int(amount / current_price) if current_price > 0 else 0
    return qty, f, amount

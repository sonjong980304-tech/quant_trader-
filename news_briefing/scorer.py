"""
news_briefing/scorer.py — 예보(forecast) 채점

runner.py 등 호출부가 예측(direction: up/flat/down)과 실제 지수 등락률(actual_pct)을
비교해 hit/miss를 판정하고 db.forecasts를 갱신한다. 시장 데이터 조회가 일시적으로
실패(휴장·API 지연 등)할 수 있어 재시도 로직을 포함하며, 재시도가 모두 소진되면
해당 forecast는 pending 상태로 남겨 다음 실행에서 다시 시도한다.

판정 기준(경계값 ±0.3%는 hit 포함):
  up   : actual_pct >= 0.3        → hit
  down : actual_pct <= -0.3       → hit
  flat : -0.3 <= actual_pct <= 0.3 → hit
  (그 외 miss)

같은 실행 내에서 동일 시장(KOSPI/KOSDAQ)에 속한 forecast가 여러 건이어도
market_data_fn은 시장당 1회만 호출한다(딕셔너리 캐시).
"""
import logging
import time

from news_briefing import db
from news_briefing import market_data as _market_data_module

logger = logging.getLogger(__name__)

_HIT_THRESHOLD = 0.3


def _judge_verdict(direction, actual_pct):
    """direction·actual_pct(등락률 %)로 hit/miss를 판정한다(경계값은 hit 포함)."""
    if direction == "up":
        return "hit" if actual_pct >= _HIT_THRESHOLD else "miss"
    if direction == "down":
        return "hit" if actual_pct <= -_HIT_THRESHOLD else "miss"
    if direction == "flat":
        return "hit" if -_HIT_THRESHOLD <= actual_pct <= _HIT_THRESHOLD else "miss"
    return "miss"


def _fetch_with_retry(market, market_data_fn, sleep_fn, max_retries, retry_interval_sec):
    """market_data_fn(market)을 호출하고 None이면 최대 max_retries회 재시도한다."""
    data = market_data_fn(market)
    attempts = 0
    while data is None and attempts < max_retries:
        sleep_fn(retry_interval_sec)
        data = market_data_fn(market)
        attempts += 1
    return data


def score_pending_forecasts(now_str, db_path=None, market_data_fn=None, sleep_fn=None,
                             max_retries=3, retry_interval_sec=600):
    """
    verdict='pending'인 forecasts를 조회해 실제 지수 등락률과 비교·채점한다.

    market_data_fn: market(str) -> {"close","change_pct","asof"} | None
                     (기본은 news_briefing.market_data.get_kr_index_change)
    sleep_fn:        재시도 대기 함수(기본 time.sleep). 테스트에서는 mock 주입.
    반환: [{"forecast_id","market","verdict","actual_pct"}, ...]
          채점하지 못해 pending으로 남은 건도 verdict="pending", actual_pct=None으로 포함한다.
    """
    if market_data_fn is None:
        market_data_fn = _market_data_module.get_kr_index_change
    if sleep_fn is None:
        sleep_fn = time.sleep

    pending = db.get_pending_forecasts(db_path=db_path)
    if not pending:
        logger.info("채점 대상 없음")
        return []

    results = []
    market_cache = {}

    for forecast in pending:
        market = forecast["market"]
        forecast_id = forecast["id"]

        if market not in market_cache:
            market_cache[market] = _fetch_with_retry(
                market, market_data_fn, sleep_fn, max_retries, retry_interval_sec
            )
        data = market_cache[market]

        if data is None:
            logger.warning(
                "채점 스킵(시장 데이터 미가용, pending 유지): forecast_id=%s market=%s",
                forecast_id, market,
            )
            results.append({
                "forecast_id": forecast_id,
                "market": market,
                "verdict": "pending",
                "actual_pct": None,
            })
            continue

        actual_pct = data["change_pct"]
        verdict = _judge_verdict(forecast["direction"], actual_pct)
        db.update_forecast_verdict(forecast_id, actual_pct, verdict, now_str, db_path=db_path)
        results.append({
            "forecast_id": forecast_id,
            "market": market,
            "verdict": verdict,
            "actual_pct": actual_pct,
        })

    return results


def get_hit_rate(db_path=None):
    """db.get_hit_rate 위임 래퍼."""
    return db.get_hit_rate(db_path=db_path)

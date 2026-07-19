"""
catchup_eod.py - 맥미니를 낮에 꺼서 놓친 KR 오후 EOD 작업을 밤에 수동 실행.

낮에 봇(runner)이 꺼져 있어 15:00~15:35 스케줄이 실행되지 않았을 때 사용한다.
장 마감(15:30) 이후 종가가 확정된 뒤 실행해야 정상 동작한다.

실행 순서(스케줄러와 동일):
  1. 15:30 EOD 평가   — 보유 포지션 청산 체크(TP/SL/기간), 보유일 +1
  2. 15:31 EOD 스캔   — 오늘 종가 기준 신호 → 내일 시초가 진입 예약
  3. 15:00 일일 리포트  (텔레그램)
  4. 15:35 페이퍼 리포트 (텔레그램)

사용법:
  python3 catchup_eod.py          # 하루 한 번만 실행(중복 방지 가드)
  python3 catchup_eod.py --force  # 가드 무시하고 강제 재실행

⚠️ 주의:
  - 반드시 당일 장 마감(15:30) 이후, 가급적 자정 전에 실행할 것.
  - EOD 평가는 보유일을 +1 하므로 하루 두 번 실행하면 안 됨(가드가 기본 차단).
  - 오늘 잡은 신호는 '내일' 09:05에 시초가로 확정된다. 내일도 봇을 못 켜면
    내일 아침 catch-up이 별도로 필요하다(시초가 확정용).
"""

import os
import sys
import json
import logging
from datetime import datetime

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo-root: scripts/에서 직접 실행 시 runner import 보장

import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("catchup")

KST     = pytz.timezone("Asia/Seoul")
_MARKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".catchup_last.json")


def _today() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _already_ran_today() -> bool:
    """오늘 이미 catch-up을 돌렸는지(중복 EOD 평가 방지)."""
    if not os.path.exists(_MARKER):
        return False
    try:
        return json.load(open(_MARKER, encoding="utf-8")).get("date") == _today()
    except Exception:
        return False


def _mark_ran():
    try:
        json.dump({"date": _today(), "at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")},
                  open(_MARKER, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception as e:
        logger.warning("실행 마커 기록 실패: %s", e)


def _step(label: str, fn):
    logger.info("▶ %s", label)
    try:
        fn()
        logger.info("  ✓ %s 완료", label)
    except Exception as e:
        logger.error("  ✗ %s 실패: %s", label, e)


def main():
    force = "--force" in sys.argv

    if _already_ran_today() and not force:
        logger.warning("오늘(%s) 이미 catch-up을 실행했습니다. 재실행하려면 --force 를 붙이세요.", _today())
        return

    import runner

    logger.info("=== EOD catch-up 시작 (%s) ===", datetime.now(KST).strftime("%Y-%m-%d %H:%M"))

    _step("[1/3] EOD 평가 (청산 체크 + 보유일 +1)", runner._run_paper_evaluate_kr_eod)
    _step("[2/3] EOD 신호 스캔 (내일 시초가 진입 예약)", runner.scan_growth_signals_eod)
    _step("[3/3] 페이퍼 리포트 (텔레그램)", runner._run_paper_daily_report_kr)

    _mark_ran()
    logger.info("=== EOD catch-up 완료 ===")


if __name__ == "__main__":
    main()

"""
runner.py - 장중 주기 실행 스케줄러
매일 09:05 ~ 15:30 사이에 30분 간격으로 graph.py를 실행합니다.
launchd로 부팅 시 자동 시작되며 상시 대기 상태를 유지합니다.
"""

import schedule
import time
import subprocess
import logging
import logging.handlers
from datetime import datetime
import pytz

KST = pytz.timezone("Asia/Seoul")
LOG_FILE = "logs/trader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("runner")


def is_market_hours() -> bool:
    """현재 시각이 한국 주식 장 시간(09:00~15:30, 평일)인지 확인"""
    now = datetime.now(KST)
    # 주말 제외
    if now.weekday() >= 5:
        return False
    # 09:00 ~ 15:30
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def run_agent():
    """graph.py 실행 — 장 시간 외에는 건너뜀"""
    now = datetime.now(KST)

    if not is_market_hours():
        logger.info("장 시간 외 — 실행 건너뜀 (%s)", now.strftime("%H:%M"))
        return

    logger.info("=" * 50)
    logger.info("에이전트 실행 시작 (%s)", now.strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 50)

    result = subprocess.run(
        ["python3", "graph.py"],
        cwd="/Users/gyuyeong/quant_trader",
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode == 0:
        logger.info("에이전트 실행 완료")
    else:
        logger.error("에이전트 오류:\n%s", result.stderr[-500:])


def main():
    logger.info("스케줄러 시작 — 30분 간격 장중 실행")

    # 30분마다 실행 (09:05, 09:35, 10:05, ..., 15:05)
    times = []
    h, m = 9, 5
    while (h, m) <= (15, 5):
        times.append(f"{h:02d}:{m:02d}")
        m += 30
        if m >= 60:
            m -= 60
            h += 1

    schedule.clear()  # 재시작 시 중복 등록 방지
    for t in times:
        schedule.every().day.at(t).do(run_agent)
        logger.info("  등록: %s", t)

    logger.info("총 %d개 시간대 등록 완료", len(times))

    # 스케줄러 루프
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()

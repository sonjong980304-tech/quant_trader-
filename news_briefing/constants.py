"""
news_briefing/constants.py — DB 절대경로 및 텔레그램 피드백 콜백 데이터 스키마

DB_PATH를 절대경로로 고정하는 이유: runner.py/telegram_bot.py/dashboard/app.py
3개 프로세스의 cwd가 서로 달라 상대경로를 쓰면 파일이 어긋난다(커밋 e20ab2e 전례).

콜백 데이터는 텔레그램 InlineKeyboardButton의 callback_data로 왕복하는 문자열이며,
텔레그램 사양상 64바이트를 초과할 수 없다.
"""
import os
from typing import Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_briefing.db")

CALLBACK_PREFIX_UP = "fb_up_"
CALLBACK_PREFIX_DOWN = "fb_down_"

_MAX_CALLBACK_DATA_BYTES = 64


def make_callback_data(vote, briefing_id):
    # type: (str, int) -> str
    """
    피드백 투표(vote: 'up' 또는 'down')와 briefing_id로 callback_data 문자열을 만든다.
    결과가 64바이트를 초과하면 ValueError를 발생시킨다.
    """
    if vote == "up":
        prefix = CALLBACK_PREFIX_UP
    elif vote == "down":
        prefix = CALLBACK_PREFIX_DOWN
    else:
        raise ValueError("vote는 'up' 또는 'down'이어야 합니다: {!r}".format(vote))

    data = "{}{}".format(prefix, briefing_id)
    if len(data.encode("utf-8")) > _MAX_CALLBACK_DATA_BYTES:
        raise ValueError(
            "callback_data가 {}바이트를 초과합니다: {!r}".format(_MAX_CALLBACK_DATA_BYTES, data)
        )
    return data


def parse_callback_data(data):
    # type: (Optional[str]) -> Optional[Tuple[str, int]]
    """
    callback_data 문자열을 (vote, briefing_id) 튜플로 파싱한다.
    형식이 스키마와 맞지 않으면 None을 반환한다(예외를 던지지 않음 — 핸들러가
    다른 prefix의 콜백과 뒤섞여 들어올 수 있으므로 방어적으로 처리).
    """
    if not data:
        return None

    if data.startswith(CALLBACK_PREFIX_UP):
        vote = "up"
        rest = data[len(CALLBACK_PREFIX_UP):]
    elif data.startswith(CALLBACK_PREFIX_DOWN):
        vote = "down"
        rest = data[len(CALLBACK_PREFIX_DOWN):]
    else:
        return None

    try:
        briefing_id = int(rest)
    except ValueError:
        return None

    return (vote, briefing_id)

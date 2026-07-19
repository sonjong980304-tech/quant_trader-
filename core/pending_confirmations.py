"""
pending_confirmations.py - EOD 매수 신호 텔레그램 확인 대기 목록

runner.py가 EOD 신호를 발견하면 이 모듈에 저장하고 인라인 키보드로 확인 요청.
telegram_bot.py의 CallbackQueryHandler가 ✅/❌ 응답을 처리.
"""
from __future__ import annotations  # dict | None 등 PEP604 표기를 Python 3.9(실제 운영 인터프리터)에서도 쓰기 위함

import json
import os
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "buy_confirmations.json")


def _load() -> dict:
    if not os.path.exists(_PATH):
        return {}
    try:
        with open(_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict):
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_confirmation(
    ticker: str,
    name: str,
    qty: int,
    code: str,
    is_us: bool,
    ml_meta: dict,
    note: str,
) -> str:
    """확인 대기 항목 추가 후 conf_id 반환."""
    conf_id = uuid.uuid4().hex[:8]
    data = _load()
    data[conf_id] = {
        "ticker":     ticker,
        "name":       name,
        "qty":        qty,
        "code":       code,
        "is_us":      is_us,
        "ml_meta":    ml_meta,
        "note":       note,
        "created_at": datetime.now().isoformat(),
    }
    _save(data)
    logger.info("매수 확인 대기 등록: %s (conf_id=%s)", ticker, conf_id)
    return conf_id


def get_confirmation(conf_id: str) -> dict | None:
    """conf_id로 확인 대기 항목 조회. 없으면 None."""
    return _load().get(conf_id)


def remove_confirmation(conf_id: str):
    """처리 완료된 항목 삭제."""
    data = _load()
    if conf_id in data:
        data.pop(conf_id)
        _save(data)
        logger.info("매수 확인 항목 삭제: conf_id=%s", conf_id)

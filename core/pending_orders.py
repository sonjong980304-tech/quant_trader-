"""
pending_orders.py — 다음 장 예약 주문 관리

/buynext, /sellnext 명령으로 등록 → 장 시작 시 runner가 자동 실행
저장: pending_orders.json
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime

import pytz

logger = logging.getLogger(__name__)

KST          = pytz.timezone("Asia/Seoul")
_ORDERS_PATH = os.path.join(os.path.dirname(__file__), "pending_orders.json")


def _read() -> list[dict]:
    if not os.path.exists(_ORDERS_PATH):
        return []
    try:
        with open(_ORDERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write(orders: list[dict]):
    with open(_ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)


def add_pending_order(action: str, ticker: str, code: str, qty: int,
                      is_us: bool, note: str = "",
                      ml_meta: dict | None = None) -> str:
    """예약 주문 추가. order_id 반환.

    ml_meta: ML 신호 메타데이터 (avg_win, avg_loss, atr 등).
             execute_pending_orders 실행 후 save_ml_position 호출에 사용.
    """
    orders = _read()
    order_id = str(uuid.uuid4())[:8]
    entry: dict = {
        "id":       order_id,
        "action":   action.upper(),   # "BUY" or "SELL"
        "ticker":   ticker,
        "code":     code,
        "qty":      qty,
        "is_us":    is_us,
        "market":   "US" if is_us else "KR",
        "added_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "note":     note,
    }
    if ml_meta:
        entry["ml_meta"] = ml_meta
    orders.append(entry)
    _write(orders)
    logger.info("[PendingOrder] 등록: %s %s %s %d주", order_id, action, ticker, qty)
    return order_id


def list_pending_orders(market: str | None = None) -> list[dict]:
    """예약 주문 목록. market='KR'|'US'|None(전체)"""
    orders = _read()
    if market:
        orders = [o for o in orders if o.get("market") == market]
    return orders


def remove_pending_order(order_id: str) -> bool:
    """특정 order_id 삭제."""
    orders = _read()
    before = len(orders)
    orders = [o for o in orders if o["id"] != order_id]
    _write(orders)
    return len(orders) < before


def pop_pending_orders(market: str) -> list[dict]:
    """해당 마켓의 예약 주문을 모두 꺼내고 파일에서 제거."""
    orders = _read()
    due    = [o for o in orders if o.get("market") == market]
    rest   = [o for o in orders if o.get("market") != market]
    _write(rest)
    return due


def clear_all_pending_orders() -> int:
    """전체 예약 주문 초기화. 삭제된 건수 반환."""
    orders = _read()
    count  = len(orders)
    _write([])
    logger.info("[PendingOrder] 전체 초기화: %d건 삭제", count)
    return count

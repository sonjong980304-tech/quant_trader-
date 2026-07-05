"""
conditional_orders.py - 조건부 주문 관리

조건 타입:
  price_below  : 현재가 < 기준가 → 매수/매도
  price_above  : 현재가 > 기준가 → 매수/매도
  profit_above : 수익률 > X% → 매도 (익절)
  profit_below : 수익률 < X% → 매도 (손절)
"""

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_ORDERS_FILE = Path("/Users/gyuyeong/projects/quant_trader/conditional_orders.json")


def _load() -> list:
    if not _ORDERS_FILE.exists():
        return []
    try:
        return json.loads(_ORDERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(orders: list) -> None:
    _ORDERS_FILE.write_text(
        json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_order(
    ticker: str,
    stock_name: str,
    stock_code: str,
    condition_type: str,
    condition_value: float,
    action: str,
    quantity: int,
) -> dict:
    orders = _load()
    order = {
        "id":              str(uuid.uuid4())[:8],
        "ticker":          ticker,
        "stock_name":      stock_name,
        "stock_code":      stock_code,
        "condition_type":  condition_type,
        "condition_value": condition_value,
        "action":          action,
        "quantity":        quantity,
        "created_at":      datetime.now().isoformat(timespec="seconds"),
    }
    orders.append(order)
    _save(orders)
    logger.info("조건부 주문 등록: %s", order)
    return order


def cancel_order(order_id: str) -> bool:
    orders = _load()
    new = [o for o in orders if o["id"] != order_id]
    if len(new) == len(orders):
        return False
    _save(new)
    return True


def list_orders() -> list:
    return _load()


def check_and_execute(
    ticker: str, stock_code: str, current_price: float, trader=None
) -> list:
    """해당 ticker의 조건부 주문을 체크하고, 충족된 주문을 실행 후 결과 메시지 목록 반환."""
    orders = _load()
    executed_ids = []
    messages = []

    for order in orders:
        if order["stock_code"] != stock_code:
            continue

        ctype  = order["condition_type"]
        cvalue = float(order["condition_value"])

        triggered = False

        if ctype == "price_below" and current_price < cvalue:
            triggered = True
        elif ctype == "price_above" and current_price > cvalue:
            triggered = True
        elif ctype in ("profit_above", "profit_below"):
            avg_price = None
            if trader:
                try:
                    balance  = trader.get_balance()
                    holding  = next((b for b in balance if b["stock_code"] == stock_code), None)
                    if holding:
                        avg_price = holding["avg_price"]
                except Exception:
                    pass
            if avg_price and avg_price > 0:
                profit_pct = (current_price - avg_price) / avg_price * 100
                if ctype == "profit_above" and profit_pct >= cvalue:
                    triggered = True
                elif ctype == "profit_below" and profit_pct <= cvalue:
                    triggered = True

        if not triggered:
            continue

        msg = _do_execute(order, current_price, trader)
        messages.append(msg)
        executed_ids.append(order["id"])
        logger.info("조건부 주문 실행: %s → %s", order["id"], msg)

    if executed_ids:
        _save([o for o in orders if o["id"] not in executed_ids])

    return messages


def _do_execute(order: dict, current_price: float, trader) -> str:
    from config import KIS_APP_KEY
    stock_code   = order["stock_code"]
    stock_name   = order["stock_name"]
    action       = order["action"]
    qty          = order.get("quantity", 0)
    oid          = order["id"]
    action_label = {"buy": "매수", "sell": "매도", "sellall": "전량 매도"}.get(action, action)

    if not KIS_APP_KEY or not trader:
        return (
            f"[조건부주문 #{oid}] {stock_name} {action_label} "
            f"— 시뮬레이션 모드 (KIS 미설정)"
        )

    try:
        if action == "buy":
            trader.buy(stock_code, qty)
        elif action == "sell":
            trader.sell(stock_code, qty)
        elif action == "sellall":
            balance = trader.get_balance()
            holding = next((b for b in balance if b["stock_code"] == stock_code), None)
            if holding and holding["qty"] > 0:
                qty = holding["qty"]
                trader.sell(stock_code, qty)
            else:
                return f"[조건부주문 #{oid}] {stock_name} 전량 매도 실패 — 보유 수량 없음"

        return (
            f"✅ [조건부주문 #{oid}] 조건 충족 → {stock_name}({stock_code})\n"
            f"{action_label} {qty}주 @ 현재가 {current_price:,.0f}원"
        )
    except Exception as e:
        return f"⚠️ [조건부주문 #{oid}] {stock_name} {action_label} 실패: {e}"

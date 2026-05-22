"""
trade_logger.py - 매매 이력 CSV 기록 + 텔레그램 전송

CSV 컬럼:
  trade_id, ticker, name, side(BUY/SELL),
  entry_date, entry_price, exit_date, exit_price,
  qty, pnl_amount, pnl_pct, win(1/0), strategy, notes

매수 시 → 행 추가 (exit 컬럼 비워둠)
매도 시 → 해당 행 업데이트 (exit 정보 + 손익 계산)
업데이트마다 CSV 파일을 텔레그램으로 전송.
"""

import os
import csv
import uuid
import logging
from datetime import datetime

import pytz
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

KST      = pytz.timezone("Asia/Seoul")
CSV_PATH = "/Users/gyuyeong/quant_trader/trade_history.csv"

_FIELDNAMES = [
    "trade_id", "ticker", "name", "side",
    "entry_date", "entry_price",
    "exit_date",  "exit_price",
    "qty", "pnl_amount", "pnl_pct", "win",
    "strategy", "notes",
]


# ─────────────────────────────────────────────
# CSV 초기화
# ─────────────────────────────────────────────

def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()


def _read_all() -> list[dict]:
    _ensure_csv()
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict]):
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────
# 매수 기록
# ─────────────────────────────────────────────

def log_buy(
    ticker: str,
    name: str,
    entry_price: float,
    qty: int,
    strategy: str = "",
    notes: str = "",
) -> str:
    """
    매수 시 호출. trade_id 반환.
    """
    _ensure_csv()
    trade_id   = str(uuid.uuid4())[:8]
    entry_date = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

    row = {
        "trade_id":   trade_id,
        "ticker":     ticker,
        "name":       name,
        "side":       "BUY",
        "entry_date": entry_date,
        "entry_price": entry_price,
        "exit_date":  "",
        "exit_price": "",
        "qty":        qty,
        "pnl_amount": "",
        "pnl_pct":    "",
        "win":        "",
        "strategy":   strategy,
        "notes":      notes,
    }

    rows = _read_all()
    rows.append(row)
    _write_all(rows)

    logger.info("[TradeLog] 매수 기록: %s %s %d주 @ %s원", trade_id, ticker, qty, f"{entry_price:,.0f}")
    _send_csv_to_telegram(f"📥 매수 기록 추가: {name} ({ticker}) {qty}주")
    return trade_id


# ─────────────────────────────────────────────
# 매도 기록 (기존 행 업데이트)
# ─────────────────────────────────────────────

def log_sell(
    ticker: str,
    exit_price: float,
    qty: int = None,
    notes: str = "",
) -> bool:
    """
    매도 시 호출. ticker 기준으로 열린 포지션(exit_date 비어있는 것) 업데이트.
    """
    rows     = _read_all()
    exit_date = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    updated  = False

    for row in rows:
        if row["ticker"] == ticker and row["exit_date"] == "":
            entry_price = float(row["entry_price"])
            trade_qty   = int(row["qty"]) if not qty else qty
            pnl_amount  = (exit_price - entry_price) * trade_qty
            pnl_pct     = (exit_price - entry_price) / entry_price * 100
            win         = 1 if pnl_pct > 0 else 0

            row["exit_date"]  = exit_date
            row["exit_price"] = exit_price
            row["qty"]        = trade_qty
            row["pnl_amount"] = round(pnl_amount, 0)
            row["pnl_pct"]    = round(pnl_pct, 2)
            row["win"]        = win
            if notes:
                row["notes"] = notes
            updated = True
            break

    if updated:
        _write_all(rows)
        closed = next(r for r in rows if r["ticker"] == ticker and r["exit_date"] == exit_date)
        pnl    = float(closed["pnl_amount"])
        pct    = float(closed["pnl_pct"])
        icon   = "✅" if pct > 0 else "❌"
        logger.info(
            "[TradeLog] 매도 기록: %s %s %.2f%%",
            ticker, f"{pnl:+,.0f}원", pct
        )
        _send_csv_to_telegram(
            f"{icon} 매도 완료: {closed['name']} ({ticker})\n"
            f"수익: {pnl:+,.0f}원 ({pct:+.2f}%)"
        )

    return updated


# ─────────────────────────────────────────────
# 통계 요약
# ─────────────────────────────────────────────

def get_stats() -> dict:
    """전체 청산 거래 기준 승률 / 손익비 / 평균 수익률 반환."""
    rows   = _read_all()
    closed = [r for r in rows if r["exit_date"]]

    if not closed:
        return {"total": 0, "win_rate": 0, "avg_pnl_pct": 0, "risk_reward": 0}

    total    = len(closed)
    wins     = [r for r in closed if int(r["win"]) == 1]
    losses   = [r for r in closed if int(r["win"]) == 0]
    win_rate = len(wins) / total * 100

    avg_win  = sum(float(r["pnl_pct"]) for r in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(float(r["pnl_pct"]) for r in losses) / len(losses) if losses else 0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    avg_pnl  = sum(float(r["pnl_pct"]) for r in closed) / total

    return {
        "total":      total,
        "win_rate":   round(win_rate, 1),
        "avg_pnl_pct": round(avg_pnl, 2),
        "risk_reward": round(rr, 2),
        "avg_win":    round(avg_win, 2),
        "avg_loss":   round(avg_loss, 2),
    }


def format_stats_message() -> str:
    s = get_stats()
    if s["total"] == 0:
        return "📋 아직 청산된 거래가 없습니다."
    return (
        f"📋 <b>매매 이력 통계</b>\n"
        f"총 거래: {s['total']}건\n"
        f"승률:   {s['win_rate']:.1f}%\n"
        f"손익비: {s['risk_reward']:.2f}\n"
        f"평균 수익: {s['avg_pnl_pct']:+.2f}%\n"
        f"  (성공: +{s['avg_win']:.2f}% / 실패: {s['avg_loss']:.2f}%)"
    )


# ─────────────────────────────────────────────
# 텔레그램으로 CSV 파일 전송
# ─────────────────────────────────────────────

def _send_csv_to_telegram(caption: str = ""):
    """CSV 파일을 텔레그램 문서로 전송."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    _ensure_csv()
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(CSV_PATH, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"document": ("trade_history.csv", f, "text/csv")},
                timeout=15,
            )
        if resp.status_code != 200:
            logger.warning("CSV 전송 실패: %s", resp.text[:200])
    except Exception as e:
        logger.error("CSV 텔레그램 전송 오류: %s", e)

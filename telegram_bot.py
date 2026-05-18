"""
telegram_bot.py - 텔레그램 봇 명령어 처리
폰에서 명령어로 신호·잔고 조회 및 에이전트 수동 실행

사용 가능한 명령어:
  /start   - 봇 소개
  /status  - 전 종목 현재 신호 조회
  /balance - 계좌 잔고 조회
  /run     - 에이전트 수동 실행
  /help    - 도움말
"""

import asyncio
import subprocess
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from config import TELEGRAM_BOT_TOKEN, STOCKS, ACTIVE_STRATEGY
from data_fetcher import fetch_ohlcv
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# /start
# ─────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📈 <b>퀀트 자동매매 봇</b>\n\n"
        "사용 가능한 명령어:\n"
        "/status  — 전 종목 신호 조회\n"
        "/balance — 계좌 잔고 조회\n"
        "/run     — 에이전트 수동 실행\n"
        "/help    — 도움말"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


# ─────────────────────────────────────────────
# /status — 전 종목 신호 조회
# ─────────────────────────────────────────────
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ 신호 조회 중...")

    s = ACTIVE_STRATEGY
    lines = [f"<b>📊 종목별 신호 ({s['name']})</b>\n"]

    for ticker, name in STOCKS.items():
        try:
            df = fetch_ohlcv(ticker, period_years=1)
            df = add_all_indicators(df, short=s["short_window"], long=s["long_window"], rsi_period=s["rsi_period"])
            df = detect_crossover(df, short=s["short_window"], long=s["long_window"])
            df = generate_signals(df, strategy=s)
            sig = get_latest_signal(df)

            price = df["Close"].iloc[-1]

            if sig["buy"]:
                icon = "🟢"
                signal_txt = "매수"
            elif sig["sell_full"]:
                icon = "🔴"
                signal_txt = "전량매도"
            elif sig["sell_partial"]:
                icon = "🟡"
                signal_txt = "분할매도"
            else:
                icon = "⚪"
                signal_txt = "없음"

            lines.append(
                f"{icon} <b>{name}</b>\n"
                f"   현재가: {price:,.0f}원 | RSI: {sig['rsi']}\n"
                f"   MA{s['short_window']}: {sig['ma_short']:,.0f} / MA{s['long_window']}: {sig['ma_long']:,.0f}\n"
                f"   신호: {signal_txt}\n"
            )
        except Exception as e:
            lines.append(f"⚠️ <b>{name}</b>: 조회 실패 ({e})\n")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ─────────────────────────────────────────────
# /balance — 계좌 잔고 조회
# ─────────────────────────────────────────────
async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ 잔고 조회 중...")

    try:
        from trader import KISTrader
        from config import KIS_APP_KEY, IS_MOCK
        if not KIS_APP_KEY:
            await update.message.reply_text("⚠️ KIS_APP_KEY가 설정되지 않았습니다.")
            return

        t = KISTrader()
        balance = t.get_balance()
        cash    = t.get_available_cash()
        mode    = "모의투자" if IS_MOCK else "실전투자"

        lines = [f"<b>💰 계좌 잔고 ({mode})</b>\n"]
        lines.append(f"주문 가능 현금: <b>{cash:,}원</b>\n")

        if balance:
            lines.append("─────────────────")
            for h in balance:
                lines.append(
                    f"• {h['name']} {h['qty']}주\n"
                    f"  평균단가: {h['avg_price']:,}원 | 평가손익: {h['eval_profit']:+,}원"
                )
        else:
            lines.append("보유 종목 없음")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        await update.message.reply_text(f"⚠️ 잔고 조회 실패: {e}")


# ─────────────────────────────────────────────
# /run — 에이전트 수동 실행
# ─────────────────────────────────────────────
async def cmd_run(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 에이전트 실행 시작...")

    try:
        result = subprocess.run(
            ["python3", "graph.py"],
            cwd="/Users/gyuyeong/quant_trader",
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            await update.message.reply_text("✅ 에이전트 실행 완료!\n결과는 위 메시지를 확인하세요.")
        else:
            await update.message.reply_text(f"⚠️ 오류 발생:\n{result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏱️ 실행 시간 초과 (120초)")
    except Exception as e:
        await update.message.reply_text(f"⚠️ 실행 실패: {e}")


# ─────────────────────────────────────────────
# /help
# ─────────────────────────────────────────────
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = (
        "<b>📖 도움말</b>\n\n"
        "/status  — 전 종목의 현재 MA·RSI·신호 조회\n"
        "/balance — KIS 계좌 잔고 및 주문 가능 금액\n"
        "/run     — 에이전트 즉시 실행 (신호감지→AI→주문)\n"
        "/help    — 이 도움말\n\n"
        f"전략: MA{ACTIVE_STRATEGY['short_window']}/{ACTIVE_STRATEGY['long_window']} 골든크로스 + RSI≥{ACTIVE_STRATEGY['rsi_buy_threshold']}\n"
        f"종목: {', '.join(STOCKS.values())}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


# ─────────────────────────────────────────────
# 봇 실행
# ─────────────────────────────────────────────
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("run",     cmd_run))
    app.add_handler(CommandHandler("help",    cmd_help))

    logger.info("텔레그램 봇 시작 (polling...)")
    app.run_polling()


if __name__ == "__main__":
    main()

"""
telegram_bot.py - 텔레그램 봇 명령어 처리

사용 가능한 명령어:
  /start        - 봇 소개
  /status       - 전 종목 현재 신호 조회
  /balance      - 계좌 잔고 조회
  /run          - 에이전트 수동 실행
  /addstock     - 종목 추가 예) /addstock 005930 삼성전자
  /removestock  - 종목 삭제 예) /removestock 005930
  /stocks       - 현재 종목 목록
  /help         - 도움말
"""

import asyncio
import re
import subprocess
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from config import TELEGRAM_BOT_TOKEN, STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD
from langchain_agent import ask as gpt_ask, clear_history as gpt_clear
from data_fetcher import fetch_ohlcv
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal

# 급등주 신호 대기 중인 주문 {user_id: signal_dict}
_pending_signals: dict[int, dict] = {}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "/Users/gyuyeong/quant_trader/config.py"


# ─────────────────────────────────────────────
# config.py STOCKS 딕셔너리 읽기/쓰기 헬퍼
# ─────────────────────────────────────────────
def read_stocks_from_config() -> dict:
    """config.py에서 현재 STOCKS 딕셔너리를 읽어 반환"""
    import importlib, sys
    # 모듈 캐시 초기화 후 재로드
    if "config" in sys.modules:
        del sys.modules["config"]
    import config as cfg
    return dict(cfg.STOCKS)


def write_stocks_to_config(stocks: dict):
    """config.py의 STOCKS 블록을 새 딕셔너리로 교체"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    lines = ['STOCKS = {\n']
    for ticker, name in stocks.items():
        lines.append(f'    "{ticker}": "{name}",\n')
    lines.append('}\n')
    new_block = "".join(lines)

    # STOCKS = { ... } 블록 통째로 교체
    content = re.sub(
        r'STOCKS\s*=\s*\{[^}]*\}',
        new_block.rstrip('\n'),
        content,
        flags=re.DOTALL,
    )

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────
# /start
# ─────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📈 <b>퀀트 자동매매 봇</b>\n\n"
        "안녕하세요! 주식 관련 질문은 <b>그냥 말씀해주시면</b> 바로 답변드립니다 😊\n\n"
        "예) <i>\"삼성전자 PER 얼마야?\"</i>\n"
        "예) <i>\"매수 2원칙 조건이 뭐야?\"</i>\n"
        "예) <i>\"지금 내 포지션 뭐 있어?\"</i>\n\n"
        "재무정보(PER, 영업이익 등)는 Naver 증권에서 자동 검색해 답변합니다.\n\n"
        "자동매매 기능이 필요할 땐 /help 를 입력하세요."
    )
    await update.message.reply_text(msg, parse_mode="HTML")


# ─────────────────────────────────────────────
# /status — 전 종목 신호 조회
# ─────────────────────────────────────────────
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ 신호 조회 중...")
    stocks = read_stocks_from_config()
    lines = [f"<b>📊 종목별 신호 (MA{MA_SHORT}/MA{MA_LONG} + 거래량/캔들 전략)</b>\n"]

    # KIS API 가용 여부 확인 (실시간 현재가 조회용)
    trader = None
    try:
        from config import KIS_APP_KEY
        if KIS_APP_KEY:
            from trader import KISTrader
            trader = KISTrader()
    except Exception:
        pass

    for ticker, name in stocks.items():
        try:
            from data_fetcher import get_minute_data
            from runner import _append_today_bar

            raw_df    = fetch_ohlcv(ticker, period_years=1)
            stock_code = ticker.replace(".KS", "").replace(".KQ", "")

            # 실시간 현재가 + 분봉 데이터로 오늘 바 구성
            price     = float(raw_df["Close"].iloc[-1])
            minute_df = None
            if trader:
                try:
                    price_info = trader.get_current_price(stock_code)
                    price      = float(price_info["price"])
                except Exception:
                    pass
                try:
                    minute_df = get_minute_data(ticker, interval_min=1)
                except Exception:
                    pass

            df  = _append_today_bar(raw_df, minute_df)
            df  = add_all_indicators(df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD)
            df  = detect_crossover(df, short=MA_SHORT, long=MA_LONG)
            df  = generate_signals(df)
            sig = get_latest_signal(df)

            if sig["buy"]:
                principles = "/".join(sig.get("buy_which", []))
                icon, signal_txt = "🟢", f"매수({principles})"
            elif sig["sell_full"]:
                icon, signal_txt = "🔴", "매도(2원칙-전량)"
            elif sig["sell_partial"]:
                icon, signal_txt = "🟡", "매도(1원칙-부분)"
            else:
                icon, signal_txt = "⚪", "없음"

            lines.append(
                f"{icon} <b>{name}</b>\n"
                f"   현재가: {price:,.0f}원 | RSI: {sig['rsi']}\n"
                f"   MA{MA_SHORT}: {sig['ma_short']:,.0f} / MA{MA_LONG}: {sig['ma_long']:,.0f}\n"
                f"   신호: {signal_txt}\n"
            )
        except Exception as e:
            lines.append(f"⚠️ <b>{name}</b>: 조회 실패\n")

    price_label = "실시간" if trader else "전일 종가"
    lines.append(f"<i>현재가 기준: {price_label}</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ─────────────────────────────────────────────
# /balance — 계좌 잔고 조회
# ─────────────────────────────────────────────
async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ 잔고 조회 중...")
    try:
        from trader import KISTrader
        from config import IS_MOCK
        t = KISTrader()
        balance = t.get_balance()
        cash    = t.get_available_cash()
        mode    = "모의투자" if IS_MOCK else "실전투자"

        lines = [f"<b>💰 계좌 잔고 ({mode})</b>\n", f"주문 가능 현금: <b>{cash:,}원</b>\n"]
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
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            await update.message.reply_text("✅ 에이전트 실행 완료!\n신호가 있으면 위에 메시지가 왔을 거예요.")
        else:
            await update.message.reply_text(f"⚠️ 오류:\n{result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏱️ 실행 시간 초과 (120초)")
    except Exception as e:
        await update.message.reply_text(f"⚠️ 실행 실패: {e}")


# ─────────────────────────────────────────────
# /stocks — 현재 종목 목록
# ─────────────────────────────────────────────
async def cmd_stocks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stocks = read_stocks_from_config()
    lines = ["<b>📋 현재 매매 종목 목록</b>\n"]
    for ticker, name in stocks.items():
        lines.append(f"• {name} ({ticker})")
    lines.append(f"\n총 {len(stocks)}개 종목")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ─────────────────────────────────────────────
# /addstock — 종목 추가
# 사용법: /addstock 005930 삼성전자
# KS/KQ 자동 판별 (KOSDAQ은 .KQ, 나머지 .KS)
# ─────────────────────────────────────────────
async def cmd_addstock(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "사용법: /addstock <종목코드> <종목명>\n"
            "예) /addstock 005930 삼성전자\n"
            "KOSDAQ은 자동 감지하며, 직접 지정도 가능:\n"
            "예) /addstock 432720.KQ 퀄리타스반도체"
        )
        return

    code = args[0].upper()
    name = " ".join(args[1:])

    # 티커 형식 결정
    if "." in code:
        ticker = code  # .KS/.KQ 직접 지정
    else:
        # 6자리 코드 기준으로 시장 자동 판별
        # KOSDAQ 종목은 일반적으로 코드가 0으로 시작하지 않거나 특정 범위
        # yfinance로 .KS 먼저 시도, 실패하면 .KQ
        import yfinance as yf
        df_ks = yf.download(f"{code}.KS", period="5d", progress=False, auto_adjust=True)
        if not df_ks.empty:
            ticker = f"{code}.KS"
        else:
            df_kq = yf.download(f"{code}.KQ", period="5d", progress=False, auto_adjust=True)
            if not df_kq.empty:
                ticker = f"{code}.KQ"
            else:
                await update.message.reply_text(
                    f"⚠️ 종목코드 <b>{code}</b>로 데이터를 찾을 수 없습니다.\n"
                    f"코드를 다시 확인하거나 시장을 직접 지정해보세요:\n"
                    f"/addstock {code}.KQ {name}",
                    parse_mode="HTML"
                )
                return

    stocks = read_stocks_from_config()
    if ticker in stocks:
        await update.message.reply_text(f"⚠️ <b>{name}</b>({ticker})은 이미 목록에 있습니다.", parse_mode="HTML")
        return

    stocks[ticker] = name
    write_stocks_to_config(stocks)
    await update.message.reply_text(
        f"✅ <b>{name}</b> ({ticker}) 추가 완료!\n현재 {len(stocks)}개 종목",
        parse_mode="HTML"
    )


# ─────────────────────────────────────────────
# /removestock — 종목 삭제
# 사용법: /removestock 005930
# ─────────────────────────────────────────────
async def cmd_removestock(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "사용법: /removestock <종목코드>\n"
            "예) /removestock 005930\n"
            "현재 목록 확인: /stocks"
        )
        return

    code = args[0].upper()
    stocks = read_stocks_from_config()

    # .KS/.KQ 없이 입력한 경우 자동 탐색
    matched_ticker = None
    if code in stocks:
        matched_ticker = code
    else:
        for t in stocks:
            if t.startswith(code):
                matched_ticker = t
                break

    if not matched_ticker:
        await update.message.reply_text(
            f"⚠️ <b>{code}</b>를 종목 목록에서 찾을 수 없습니다.\n/stocks 로 목록을 확인하세요.",
            parse_mode="HTML"
        )
        return

    removed_name = stocks.pop(matched_ticker)
    write_stocks_to_config(stocks)
    await update.message.reply_text(
        f"🗑️ <b>{removed_name}</b> ({matched_ticker}) 삭제 완료!\n남은 종목: {len(stocks)}개",
        parse_mode="HTML"
    )


# ─────────────────────────────────────────────
# GPT 어시스턴트
# ─────────────────────────────────────────────
async def _gpt_reply(update: Update, text: str):
    """GPT에게 질문하고 답변을 텔레그램으로 전송 (공통 처리)."""
    if not text.strip():
        await update.message.reply_text("질문 내용을 입력해주세요.")
        return

    await update.message.reply_text("🤔 생각 중...")
    user_id = update.effective_user.id

    try:
        loop   = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, gpt_ask, user_id, text.strip())
        # 텔레그램 메시지 최대 4096자 제한
        if len(answer) > 4000:
            for i in range(0, len(answer), 4000):
                await update.message.reply_text(answer[i:i + 4000])
        else:
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error("GPT 핸들러 오류: %s", e)
        await update.message.reply_text(f"⚠️ 오류: {e}")


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/ask <질문> — 명시적 GPT 질문."""
    question = " ".join(ctx.args) if ctx.args else ""
    await _gpt_reply(update, question)


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/reset — 대화 기록 초기화."""
    gpt_clear(update.effective_user.id)
    await update.message.reply_text("🗑️ 대화 기록을 초기화했습니다.")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """일반 텍스트 메시지 → GPT로 전달."""
    await _gpt_reply(update, update.message.text or "")


# ─────────────────────────────────────────────
# /help
# ─────────────────────────────────────────────
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stocks = read_stocks_from_config()
    msg = (
        "<b>📖 도움말</b>\n\n"
        "<b>[자동매매]</b>\n"
        "/status              — 전 종목 신호 조회\n"
        "/balance             — 계좌 잔고\n"
        "/run                 — 에이전트 즉시 실행\n"
        "/stocks              — 종목 목록 보기\n"
        "/addstock 코드 이름  — 종목 추가\n"
        "/removestock 코드    — 종목 삭제\n\n"
        "<b>[수동 매매]</b>\n"
        "/buy 코드 수량       — 수동 매수 (국내/미국 자동 감지)\n"
        "  예) /buy 005930 1  또는  /buy AAPL 5\n"
        "/sell 코드 수량      — 수동 매도\n"
        "  예) /sell 010140 5  또는  /sell AAPL 3\n"
        "/sellall 코드        — 전량 매도\n"
        "  예) /sellall 010140\n\n"
        "<b>[다음 장 예약 주문]</b>\n"
        "/buynext 코드 수량   — 다음 장 시작 시 매수\n"
        "/sellnext 코드 수량  — 다음 장 시작 시 매도\n"
        "/pendingorders       — 예약 주문 조회/취소\n\n"
        "<b>[AI 어시스턴트]</b>\n"
        "메시지 전송          — GPT 자동 답변\n"
        "/ask 질문            — 명시적 GPT 질문\n"
        "/reset               — 대화 기록 초기화\n\n"
        "💡 재무 관련 질문(PER, 영업이익 등)은\n"
        "   Naver 증권에서 자동 검색해서 답변합니다.\n\n"
        f"현재 전략: MA{MA_SHORT}/MA{MA_LONG} + 거래량/캔들 전략\n"
        f"현재 종목 수: {len(stocks)}개"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


# ─────────────────────────────────────────────
# /buy — 수동 매수
# 사용법: /buy 010140 10
# ─────────────────────────────────────────────
async def cmd_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "사용법: /buy <종목코드> <수량>\n"
            "예) /buy 005930 1  또는  /buy 005930.KS 1"
        )
        return

    raw  = args[0].upper()
    code = raw.replace(".KS", "").replace(".KQ", "")
    suffix = ".KS" if ".KS" in raw else (".KQ" if ".KQ" in raw else "")
    ticker = code + suffix
    is_us  = not (raw.endswith(".KS") or raw.endswith(".KQ")) and any(c.isalpha() for c in code)

    try:
        qty = int(args[1])
        if qty <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("⚠️ 수량은 1 이상의 정수여야 합니다.")
        return

    market_tag = "🇺🇸 미국" if is_us else "🇰🇷 국내"
    await update.message.reply_text(f"⏳ [{market_tag}] {ticker} {qty}주 시장가 매수 주문 중...")

    try:
        from trader import KISTrader
        t = KISTrader()

        if is_us:
            price_info = t.get_us_current_price(code)
            price      = price_info["price"]
            name       = code
            total_usd  = price * qty
            t.buy_us(code, qty)
            try:
                from trade_logger import log_buy
                log_buy(ticker, name, price, qty, strategy="수동")
            except Exception as log_e:
                logger.warning("[TradeLog] 수동 매수 기록 실패 [%s]: %s", ticker, log_e)
            await update.message.reply_text(
                f"✅ <b>매수 주문 완료 (미국)</b>\n"
                f"종목: {name} ({ticker})\n"
                f"수량: {qty}주\n"
                f"현재가: ${price:,.2f}\n"
                f"총 금액: ${total_usd:,.2f}",
                parse_mode="HTML"
            )
        else:
            price_info = t.get_current_price(code)
            price      = price_info["price"]
            name       = price_info.get("name", ticker)
            total      = price * qty
            cash = t.get_available_cash()
            if cash < total:
                await update.message.reply_text(
                    f"⚠️ 주문 가능 금액 부족\n필요: {total:,}원 | 가능: {cash:,}원"
                )
                return
            t.buy(code, qty)
            try:
                from trade_logger import log_buy
                log_buy(ticker, name, price, qty, strategy="수동")
            except Exception as log_e:
                logger.warning("[TradeLog] 수동 매수 기록 실패 [%s]: %s", ticker, log_e)
            logger.info("수동 매수 완료: %s %d주 @ %d원", ticker, qty, price)
            await update.message.reply_text(
                f"✅ <b>매수 주문 완료</b>\n"
                f"종목: {name} ({ticker})\n"
                f"수량: {qty}주\n"
                f"현재가: {price:,}원\n"
                f"총 금액: {total:,}원",
                parse_mode="HTML"
            )
    except Exception as e:
        logger.error("수동 매수 실패 [%s]: %s", ticker, e)
        await update.message.reply_text(f"⚠️ 매수 주문 실패: {e}")


# ─────────────────────────────────────────────
# /sell — 수동 매도
# 사용법: /sell 010140 5
# ─────────────────────────────────────────────
async def cmd_sell(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "사용법: /sell <종목코드> <수량>\n"
            "예) /sell 010140 5"
        )
        return

    raw    = args[0].upper()
    code   = raw.replace(".KS", "").replace(".KQ", "")
    suffix = ".KS" if ".KS" in raw else (".KQ" if ".KQ" in raw else "")
    ticker = code + suffix
    is_us  = not (raw.endswith(".KS") or raw.endswith(".KQ")) and any(c.isalpha() for c in code)

    try:
        qty = int(args[1])
        if qty <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("⚠️ 수량은 1 이상의 정수여야 합니다.")
        return

    market_tag = "🇺🇸 미국" if is_us else "🇰🇷 국내"
    await update.message.reply_text(f"⏳ [{market_tag}] {ticker} {qty}주 시장가 매도 주문 중...")

    try:
        from trader import KISTrader
        t = KISTrader()

        if is_us:
            us_bal  = t.get_us_balance()
            holding = next((b for b in us_bal if b["symbol"] == code), None)
            if not holding or holding["qty"] < qty:
                held = holding["qty"] if holding else 0
                await update.message.reply_text(f"⚠️ {code} 보유 수량 부족 (보유: {held}주)")
                return
            price_info = t.get_us_current_price(code)
            price      = price_info["price"]
            t.sell_us(code, qty)
            try:
                from trade_logger import log_sell
                log_sell(ticker, price, qty=qty, notes="수동매도")
            except Exception as log_e:
                logger.warning("[TradeLog] 수동 매도 기록 실패 [%s]: %s", ticker, log_e)
            await update.message.reply_text(
                f"✅ <b>매도 주문 완료 (미국)</b>\n"
                f"종목: {code}\n수량: {qty}주\n현재가: ${price:,.2f}",
                parse_mode="HTML"
            )
        else:
            balance = t.get_balance()
            holding = next((b for b in balance if b["stock_code"] == code), None)
            if not holding:
                await update.message.reply_text(f"⚠️ {code} 보유 종목이 없습니다.")
                return
            if holding["qty"] < qty:
                await update.message.reply_text(
                    f"⚠️ 보유 수량 부족\n요청: {qty}주 | 보유: {holding['qty']}주"
                )
                return
            price_info = t.get_current_price(code)
            price      = price_info["price"]
            t.sell(code, qty)
            try:
                from trade_logger import log_sell
                log_sell(ticker, price, qty=qty, notes="수동매도")
            except Exception as log_e:
                logger.warning("[TradeLog] 수동 매도 기록 실패 [%s]: %s", ticker, log_e)
            await update.message.reply_text(
                f"✅ <b>매도 주문 완료</b>\n"
                f"종목: {code}\n수량: {qty}주\n현재가: {price:,}원\n예상 금액: {price * qty:,}원",
                parse_mode="HTML"
            )
    except Exception as e:
        await update.message.reply_text(f"⚠️ 매도 주문 실패: {e}")


# ─────────────────────────────────────────────
# /sellall — 전량 매도
# 사용법: /sellall 010140
# ─────────────────────────────────────────────
async def cmd_sellall(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "사용법: /sellall <종목코드>\n"
            "예) /sellall 010140\n"
            "보유한 수량 전체를 시장가 매도합니다."
        )
        return

    raw    = args[0].upper()
    code   = raw.replace(".KS", "").replace(".KQ", "")
    suffix = ".KS" if ".KS" in raw else (".KQ" if ".KQ" in raw else "")
    ticker = code + suffix
    is_us  = not (raw.endswith(".KS") or raw.endswith(".KQ")) and any(c.isalpha() for c in code)

    market_tag = "🇺🇸 미국" if is_us else "🇰🇷 국내"
    await update.message.reply_text(f"⏳ [{market_tag}] {ticker} 전량 매도 주문 중...")

    try:
        from trader import KISTrader
        t = KISTrader()

        if is_us:
            us_bal  = t.get_us_balance()
            holding = next((b for b in us_bal if b["symbol"] == code), None)
            if not holding or holding["qty"] == 0:
                await update.message.reply_text(f"⚠️ {code} 미국주식 보유 없음")
                return
            qty        = holding["qty"]
            price_info = t.get_us_current_price(code)
            price      = price_info["price"]
            t.sell_us(code, qty)
            try:
                from trade_logger import log_sell
                log_sell(ticker, price, qty=qty, notes="수동전량매도")
            except Exception as log_e:
                logger.warning("[TradeLog] 수동 매도 기록 실패 [%s]: %s", ticker, log_e)
            await update.message.reply_text(
                f"✅ <b>전량 매도 완료 (미국)</b>\n"
                f"종목: {code}\n수량: {qty}주\n현재가: ${price:,.2f}",
                parse_mode="HTML"
            )
        else:
            balance = t.get_balance()
            holding = next((b for b in balance if b["stock_code"] == code), None)
            if not holding or holding["qty"] == 0:
                await update.message.reply_text(f"⚠️ {code} 보유 종목이 없습니다.")
                return
            qty        = holding["qty"]
            price_info = t.get_current_price(code)
            price      = price_info["price"]
            t.sell(code, qty)
            try:
                from trade_logger import log_sell
                log_sell(ticker, price, qty=qty, notes="수동전량매도")
            except Exception as log_e:
                logger.warning("[TradeLog] 수동 매도 기록 실패 [%s]: %s", ticker, log_e)
            await update.message.reply_text(
                f"✅ <b>전량 매도 주문 완료</b>\n"
                f"종목: {code} ({holding.get('name', code)})\n"
                f"수량: {qty}주 (전량)\n"
                f"현재가: {price:,}원\n"
                f"예상 금액: {price * qty:,}원",
                parse_mode="HTML"
            )
    except Exception as e:
        await update.message.reply_text(f"⚠️ 전량 매도 실패: {e}")


# ─────────────────────────────────────────────
# /buynext /sellnext — 다음 장 예약 주문
# ─────────────────────────────────────────────
async def cmd_buynext(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """다음 장 시작 시 매수 예약. 사용법: /buynext <코드> <수량>"""
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "사용법: /buynext <종목코드> <수량>\n"
            "예) /buynext 005930 10  또는  /buynext AAPL 5\n"
            "다음 장 시작 시 자동 실행됩니다."
        )
        return
    raw    = args[0].upper()
    code   = raw.replace(".KS", "").replace(".KQ", "")
    suffix = ".KS" if ".KS" in raw else (".KQ" if ".KQ" in raw else "")
    ticker = code + suffix
    is_us  = not (raw.endswith(".KS") or raw.endswith(".KQ")) and any(c.isalpha() for c in code)
    try:
        qty = int(args[1])
        if qty <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("⚠️ 수량은 1 이상의 정수여야 합니다.")
        return
    from pending_orders import add_pending_order, list_pending_orders
    order_id = add_pending_order("BUY", ticker, code, qty, is_us)
    market_tag = "🇺🇸 미국장" if is_us else "🇰🇷 한국장"
    pending = list_pending_orders()
    await update.message.reply_text(
        f"✅ <b>다음 장 매수 예약 완료</b>\n"
        f"종목: {ticker}  수량: {qty}주\n"
        f"실행 시점: {market_tag} 시작 시\n"
        f"예약 ID: {order_id}\n\n"
        f"📋 전체 예약 {len(pending)}건 대기 중",
        parse_mode="HTML"
    )


async def cmd_sellnext(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """다음 장 시작 시 매도 예약. 사용법: /sellnext <코드> <수량>"""
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "사용법: /sellnext <종목코드> <수량>\n"
            "예) /sellnext 005930 10  또는  /sellnext AAPL 5\n"
            "다음 장 시작 시 자동 실행됩니다."
        )
        return
    raw    = args[0].upper()
    code   = raw.replace(".KS", "").replace(".KQ", "")
    suffix = ".KS" if ".KS" in raw else (".KQ" if ".KQ" in raw else "")
    ticker = code + suffix
    is_us  = not (raw.endswith(".KS") or raw.endswith(".KQ")) and any(c.isalpha() for c in code)
    try:
        qty = int(args[1])
        if qty <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("⚠️ 수량은 1 이상의 정수여야 합니다.")
        return
    from pending_orders import add_pending_order, list_pending_orders
    order_id = add_pending_order("SELL", ticker, code, qty, is_us)
    market_tag = "🇺🇸 미국장" if is_us else "🇰🇷 한국장"
    pending = list_pending_orders()
    await update.message.reply_text(
        f"✅ <b>다음 장 매도 예약 완료</b>\n"
        f"종목: {ticker}  수량: {qty}주\n"
        f"실행 시점: {market_tag} 시작 시\n"
        f"예약 ID: {order_id}\n\n"
        f"📋 전체 예약 {len(pending)}건 대기 중",
        parse_mode="HTML"
    )


async def cmd_pendingorders(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """예약 주문 목록 조회."""
    from pending_orders import list_pending_orders, remove_pending_order
    args = ctx.args
    if args and args[0].lower() == "cancel" and len(args) > 1:
        from pending_orders import remove_pending_order
        ok = remove_pending_order(args[1])
        await update.message.reply_text(
            f"✅ 예약 취소 완료 ({args[1]})" if ok else f"⚠️ ID {args[1]} 예약 없음"
        )
        return
    orders = list_pending_orders()
    if not orders:
        await update.message.reply_text("📋 대기 중인 예약 주문 없음")
        return
    lines = ["📋 <b>예약 주문 목록</b>"]
    for o in orders:
        flag = "🇺🇸" if o["is_us"] else "🇰🇷"
        lines.append(
            f"{flag} [{o['id']}] {o['action']} {o['ticker']} {o['qty']}주  "
            f"({o['added_at']})"
        )
    lines.append("\n취소: /pendingorders cancel <ID>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# /portfolio — 안전자산 포트폴리오 현황
# ─────────────────────────────────────────────
async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ 포트폴리오 현황 조회 중...")
    try:
        from portfolio.safe_portfolio import format_rebalance_report
        from trader import KISTrader
        from config import KIS_APP_KEY

        holdings = {}
        total_asset = 0.0
        if KIS_APP_KEY:
            t = KISTrader()
            balance = t.get_balance()
            cash = t.get_available_cash()
            for h in balance:
                holdings[h["stock_code"]] = {"qty": h["qty"], "avg_price": h["avg_price"]}
                total_asset += h["qty"] * h["avg_price"]
            total_asset += cash

        if total_asset <= 0:
            total_asset = 10_000_000  # 시뮬레이션 기본값

        report = format_rebalance_report(holdings, total_asset)
        await update.message.reply_text(report, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ 포트폴리오 조회 실패: {e}")


# ─────────────────────────────────────────────
# /scanstocks — 급등주 수동 스캔
# ─────────────────────────────────────────────
async def cmd_scanstocks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 급등주 스캔 시작... (잠시 후 결과를 보내드립니다)")
    try:
        loop = asyncio.get_event_loop()

        def _scan():
            from signals.scanner import scan_all
            from signals.alert import send_signal_alert
            from config import KIS_APP_KEY, GROWTH_ASSET_RATIO
            from data_fetcher import fetch_ohlcv

            growth_cash = 3_000_000  # 시뮬레이션 기본값
            if KIS_APP_KEY:
                try:
                    from trader import KISTrader
                    t = KISTrader()
                    cash = t.get_available_cash()
                    balance = t.get_balance()
                    total = cash + sum(h["qty"] * h["avg_price"] for h in balance)
                    growth_cash = total * GROWTH_ASSET_RATIO
                except Exception:
                    pass

            stocks = read_stocks_from_config()
            signals = scan_all(stocks, lambda t: fetch_ohlcv(t, period_years=1))
            return signals, growth_cash

        signals, growth_cash = await loop.run_in_executor(None, _scan)

        if not signals:
            await update.message.reply_text("✅ 스캔 완료 — 조건 충족 신호 없음")
        else:
            await update.message.reply_text(f"🚨 <b>{len(signals)}개 신호 발견!</b>", parse_mode="HTML")
            for sig in signals:
                from signals.alert import build_signal_message
                msg = build_signal_message(sig, growth_cash)
                _pending_signals[update.effective_user.id] = sig
                await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ 스캔 실패: {e}")


# ─────────────────────────────────────────────
# /buysignal_{TICKER} — 급등주 매수 확정
# ─────────────────────────────────────────────
async def cmd_buysignal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    signal = _pending_signals.get(user_id)
    if not signal:
        await update.message.reply_text("⚠️ 대기 중인 신호가 없습니다. /scanstocks 로 다시 스캔하세요.")
        return

    await update.message.reply_text(f"⏳ {signal.get('name', signal['ticker'])} 매수 중...")
    try:
        from config import KIS_APP_KEY, GROWTH_ASSET_RATIO
        from portfolio.kelly import position_size

        growth_cash = 3_000_000
        if KIS_APP_KEY:
            from trader import KISTrader
            t = KISTrader()
            cash = t.get_available_cash()
            balance = t.get_balance()
            total = cash + sum(h["qty"] * h["avg_price"] for h in balance)
            growth_cash = total * GROWTH_ASSET_RATIO

        price = signal["current_price"]
        qty, kelly_f, amount = position_size(
            growth_cash, signal["win_prob"], signal["avg_win"], signal["avg_loss"], price
        )

        if qty <= 0:
            await update.message.reply_text("⚠️ 켈리 공식 기준 매수 수량 0주 — 자금 부족 또는 조건 미달")
            return

        ticker = signal["ticker"]
        is_us = not (ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker.isdigit())
        stock_code = ticker.replace(".KS", "").replace(".KQ", "")

        if KIS_APP_KEY:
            from trader import KISTrader
            t = KISTrader()
            if is_us:
                t.buy_us(stock_code, qty)
            else:
                t.buy(stock_code.replace(".", ""), qty)

        from signals.alert import build_buy_confirm_message
        from trade_logger import log_buy
        log_buy(ticker, signal.get("name", ticker), price, qty, strategy="ML급등주")
        msg = build_buy_confirm_message(signal, qty, price)
        await update.message.reply_text(msg, parse_mode="HTML")
        _pending_signals.pop(user_id, None)

    except Exception as e:
        await update.message.reply_text(f"⚠️ 매수 실패: {e}")


# ─────────────────────────────────────────────
# /skipsignal — 신호 패스
# ─────────────────────────────────────────────
async def cmd_skipsignal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    signal = _pending_signals.pop(user_id, None)
    name = signal.get("name", signal["ticker"]) if signal else "신호"
    await update.message.reply_text(f"❌ {name} 신호 패스했습니다.")


# ─────────────────────────────────────────────
# /tradestats — 매매 이력 통계
# ─────────────────────────────────────────────
async def cmd_tradestats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    from trade_logger import format_stats_message, _send_csv_to_telegram
    await update.message.reply_text(format_stats_message(), parse_mode="HTML")
    _send_csv_to_telegram("📎 전체 매매 이력")


# ─────────────────────────────────────────────
# /backtest — 최근 1개월 ML 백테스트
# ─────────────────────────────────────────────
async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stocks = read_stocks_from_config()
    await update.message.reply_text(
        f"⏳ 1개월 ML 백테스트 시작 ({len(stocks)}종목)\n"
        f"수 분이 소요됩니다..."
    )
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: __import__("backtest_ml").run_backtest(stocks)
        )
        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000], parse_mode="HTML")
        else:
            await update.message.reply_text(result, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ 백테스트 실패: {e}")


# ─────────────────────────────────────────────
# /trainmodel — ML 모델 학습
# ─────────────────────────────────────────────
async def cmd_trainmodel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stocks = read_stocks_from_config()
    await update.message.reply_text(
        f"🤖 ML 모델 학습 시작 ({len(stocks)}개 종목)\n"
        f"10년치 데이터 다운로드 + XGBoost 학습 중...\n"
        f"완료까지 수 분이 소요됩니다."
    )

    def _train():
        from ml.trainer import train_all
        return train_all(list(stocks.keys()))

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _train)
        success = {k: v for k, v in results.items() if v}
        failed  = [k for k, v in results.items() if not v]
        lines   = [f"✅ <b>모델 학습 완료</b> ({len(success)}/{len(results)}개 성공)\n"]
        for ticker, m in success.items():
            name = stocks.get(ticker, ticker)
            lines.append(
                f"• {name}: acc={m['accuracy']:.3f} | "
                f"avg_win=+{m['avg_win']*100:.1f}% | "
                f"avg_loss=-{m['avg_loss']*100:.1f}%"
            )
        if failed:
            lines.append(f"\n❌ 실패: {', '.join(failed)}")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"⚠️ 모델 학습 실패: {e}")


# ─────────────────────────────────────────────
# 봇 실행
# ─────────────────────────────────────────────
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",        cmd_start))
    app.add_handler(CommandHandler("status",       cmd_status))
    app.add_handler(CommandHandler("balance",      cmd_balance))
    app.add_handler(CommandHandler("run",          cmd_run))
    app.add_handler(CommandHandler("stocks",       cmd_stocks))
    app.add_handler(CommandHandler("addstock",     cmd_addstock))
    app.add_handler(CommandHandler("removestock",  cmd_removestock))
    app.add_handler(CommandHandler("buy",           cmd_buy))
    app.add_handler(CommandHandler("sell",          cmd_sell))
    app.add_handler(CommandHandler("sellall",       cmd_sellall))
    app.add_handler(CommandHandler("buynext",       cmd_buynext))
    app.add_handler(CommandHandler("sellnext",      cmd_sellnext))
    app.add_handler(CommandHandler("pendingorders", cmd_pendingorders))
    app.add_handler(CommandHandler("help",         cmd_help))
    app.add_handler(CommandHandler("ask",          cmd_ask))
    app.add_handler(CommandHandler("reset",        cmd_reset))
    # 퀀트 / ML 커맨드
    app.add_handler(CommandHandler("portfolio",    cmd_portfolio))
    app.add_handler(CommandHandler("scanstocks",   cmd_scanstocks))
    app.add_handler(CommandHandler("skipsignal",   cmd_skipsignal))
    app.add_handler(CommandHandler("trainmodel",   cmd_trainmodel))
    app.add_handler(CommandHandler("tradestats",   cmd_tradestats))
    app.add_handler(CommandHandler("backtest",     cmd_backtest))
    # /buysignal_{TICKER} — 패턴 매칭으로 처리
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r"^/buysignal_"),
        cmd_buysignal,
    ))
    # 커맨드가 아닌 일반 텍스트 → GPT (CommandHandler보다 나중에 등록해야 함)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("텔레그램 봇 시작 (polling...)")
    app.run_polling()


if __name__ == "__main__":
    main()

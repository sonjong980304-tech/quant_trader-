"""
dashboard.py - Streamlit 퀀트 자동매매 대시보드
실행: streamlit run dashboard.py
"""

import os
import subprocess
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD, LOG_FILE, KIS_APP_KEY
from data_fetcher import fetch_ohlcv
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="퀀트 자동매매 대시보드",
    page_icon="📈",
    layout="wide",
)

st.title("📈 퀀트 자동매매 대시보드")
st.caption(f"전략: MA{MA_SHORT}/MA{MA_LONG} + 거래량/캔들 전략 | 마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─────────────────────────────────────────────
# 사이드바 — 설정 및 수동 실행
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    # 자동 새로고침
    auto_refresh = st.checkbox("자동 새로고침 (60초)", value=False)

    st.divider()

    # 수동 에이전트 실행
    st.subheader("🤖 에이전트 수동 실행")
    if st.button("▶ 지금 실행", type="primary", use_container_width=True):
        with st.spinner("에이전트 실행 중..."):
            result = subprocess.run(
                ["python3", "graph.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=120,
            )
        if result.returncode == 0:
            st.success("실행 완료!")
        else:
            st.error(f"오류 발생:\n{result.stderr[-500:]}")

    st.divider()

    # 종목 선택
    st.subheader("📋 종목 목록")
    for ticker, name in STOCKS.items():
        st.write(f"• {name} ({ticker})")


# ─────────────────────────────────────────────
# 데이터 로드 (캐시 60초)
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker: str):
    df = fetch_ohlcv(ticker, period_years=1)
    df = add_all_indicators(df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD)
    df = detect_crossover(df, short=MA_SHORT, long=MA_LONG)
    df = generate_signals(df)
    return df


@st.cache_data(ttl=60)
def load_balance():
    if not KIS_APP_KEY:
        return [], 0
    try:
        from trader import KISTrader
        t = KISTrader()
        return t.get_balance(), t.get_available_cash()
    except Exception:
        return [], 0


def load_logs(n: int = 30) -> list:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


# ─────────────────────────────────────────────
# 종목별 탭
# ─────────────────────────────────────────────
tickers = list(STOCKS.keys())
tabs = st.tabs([STOCKS[t] for t in tickers])

for i, ticker in enumerate(tickers):
    with tabs[i]:
        stock_name = STOCKS[ticker]

        try:
            with st.spinner(f"{stock_name} 데이터 로딩..."):
                df = load_data(ticker)
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            continue

        sig = get_latest_signal(df)

        # ── 상단 지표 카드 ──
        col1, col2, col3, col4, col5 = st.columns(5)

        current_price = df["Close"].iloc[-1]
        prev_price    = df["Close"].iloc[-2]
        price_change  = current_price - prev_price
        price_pct     = price_change / prev_price * 100

        col1.metric("현재가", f"{current_price:,.0f}원", f"{price_change:+,.0f} ({price_pct:+.2f}%)")
        col2.metric(f"MA{MA_SHORT}", f"{sig['ma_short']:,.0f}원")
        col3.metric(f"MA{MA_LONG}", f"{sig['ma_long']:,.0f}원")
        col4.metric("RSI", f"{sig['rsi']}")

        # 신호 상태 배지
        if sig["buy"]:
            principles = "/".join(sig.get("buy_which", []))
            col5.metric("신호", f"🟢 매수({principles})")
        elif sig["sell_full"]:
            col5.metric("신호", "🔴 매도(1원칙)")
        elif sig["sell_partial"]:
            col5.metric("신호", "🟡 매도(2원칙)")
        else:
            col5.metric("신호", "⚪ 없음")

        st.divider()

        # ── 차트 (가격 + RSI) ──
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )

        # 종가 라인
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            name="종가", line=dict(color="#1f77b4", width=1.5),
        ), row=1, col=1)

        # MA 라인
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA_{MA_SHORT}"],
            name=f"MA{MA_SHORT}", line=dict(color="#ff7f0e", width=1.2, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA_{MA_LONG}"],
            name=f"MA{MA_LONG}", line=dict(color="#2ca02c", width=1.2, dash="dot"),
        ), row=1, col=1)

        # 매수 / 매도 신호 마커
        buy_marks  = df[df["buy_signal"]]
        sell_marks = df[df["sell_full"] | df["sell_partial"]]
        if not buy_marks.empty:
            fig.add_trace(go.Scatter(
                x=buy_marks.index, y=buy_marks["Close"],
                mode="markers", name="매수 신호",
                marker=dict(symbol="triangle-up", size=12, color="lime"),
            ), row=1, col=1)
        if not sell_marks.empty:
            fig.add_trace(go.Scatter(
                x=sell_marks.index, y=sell_marks["Close"],
                mode="markers", name="매도 신호",
                marker=dict(symbol="triangle-down", size=12, color="red"),
            ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", line=dict(color="#9467bd", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=75, line_dash="dash", line_color="red",   annotation_text="75", row=2, col=1)
        fig.add_hline(y=55, line_dash="dash", line_color="green",  annotation_text="55", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray",   annotation_text="30", row=2, col=1)

        fig.update_layout(
            height=550,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            xaxis2=dict(gridcolor="#333"),
            yaxis2=dict(gridcolor="#333", range=[0, 100]),
        )

        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# 하단 — 잔고 + 로그
# ─────────────────────────────────────────────
st.divider()
col_bal, col_log = st.columns([1, 2])

# 잔고 섹션
with col_bal:
    st.subheader("💰 계좌 잔고")
    with st.spinner("잔고 조회 중..."):
        balance, cash = load_balance()

    st.metric("주문 가능 현금", f"{cash:,}원")

    if balance:
        df_bal = pd.DataFrame(balance)
        df_bal.columns = ["종목코드", "종목명", "수량", "평균단가", "평가손익"]
        df_bal["평균단가"] = df_bal["평균단가"].apply(lambda x: f"{x:,}원")
        df_bal["평가손익"] = df_bal["평가손익"].apply(lambda x: f"{x:+,}원")
        st.dataframe(df_bal, use_container_width=True, hide_index=True)
    else:
        st.info("보유 종목 없음")

# 로그 섹션
with col_log:
    st.subheader("📋 최근 실행 로그")
    logs = load_logs(30)
    if logs:
        log_text = "\n".join(logs)
        st.code(log_text, language=None)
    else:
        st.info("로그가 없습니다. 에이전트를 실행해주세요.")

# ─────────────────────────────────────────────
# 자동 새로고침
# ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()

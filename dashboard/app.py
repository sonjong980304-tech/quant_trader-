"""
app.py - Quant Trader 대시보드 (Streamlit)

페이퍼 트레이딩 / 실제 매매를 탭으로 분리해 시각화한다.
실행:  streamlit run dashboard/app.py

주의: 봇 코드는 읽기 전용으로만 사용하며, 이 폴더(dashboard/) 안에서만 동작한다.
"""

import os
import sys
import time

import pandas as pd
import streamlit as st

# dashboard/ 내부 모듈 import 경로
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl   # noqa: E402
import charts as ch        # noqa: E402
import kis_live as kl      # noqa: E402
import news_briefing.db as nb_db                       # noqa: E402
from news_briefing.constants import DB_PATH as NB_DB_PATH  # noqa: E402

st.set_page_config(page_title="Quant Trader 대시보드", page_icon="📊", layout="wide")


# ─────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────
def _fmt_pct(v):
    """% 포맷(없으면 '-')."""
    return f"{v:.2f}%" if v is not None and pd.notna(v) else "-"


def _fmt_won(v):
    """원화 포맷."""
    return f"{v:,.0f}원" if v is not None and pd.notna(v) else "-"


def _fmt_score(v):
    """신뢰 점수(0~1 등 raw 실수) 포맷(없으면 '-')."""
    return f"{v:.3f}" if v is not None and pd.notna(v) else "-"


def _color_signed(v):
    """양수 초록 / 음수 빨강 (pandas Styler용)."""
    try:
        f = float(str(v).replace("%", "").replace(",", "").replace("원", ""))
    except Exception:
        return ""
    if f > 0:
        return f"color: {ch.GREEN}; font-weight:600;"
    if f < 0:
        return f"color: {ch.RED}; font-weight:600;"
    return ""


# ─────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
auto = st.sidebar.checkbox("30초 자동 새로고침", value=False)
if st.sidebar.button("🔄 지금 새로고침"):
    st.cache_data.clear()
    st.rerun()

_live_ok = kl.is_live_available()
st.sidebar.caption(
    f"KIS 라이브: {'🟢 연결됨' if _live_ok else '🔴 미연결(폴백 사용)'}"
)
_start = dl.get_paper_start_date()
if _start:
    st.sidebar.caption(f"페이퍼 시작일: {_start}")

st.title("📊 Quant Trader 대시보드")
tab_paper, tab_live, tab_news = st.tabs(["🧪 페이퍼 트레이딩", "💰 실제 매매", "📰 뉴스 브리핑"])


# ═════════════════════════════════════════════════════════════════════════
# 탭 A: 페이퍼 트레이딩
# ═════════════════════════════════════════════════════════════════════════
with tab_paper:
    df = dl.load_paper_trades()

    # ── 요약 카드 ─────────────────────────────────────────────────────
    s = dl.paper_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 신호 수", f"{s['총신호']}건")
    c2.metric("평균 승률", _fmt_pct(s["평균승률"]))
    c3.metric("누적 수익률", _fmt_pct(s["누적수익률"]))
    c4.metric("진행 중 포지션", f"{s['진행중']}건")

    if df.empty:
        st.info("페이퍼 거래 기록이 아직 없습니다.")
    else:
        st.divider()

        # ── 전종목 신호 테이블 ────────────────────────────────────────
        st.subheader("📋 전종목 신호")

        # AUC는 모델 메타에서(G1), trend은 win_prob N/A(G3)
        auc_map = {a: dl.get_model_auc(a) for a in df["agent"].unique()}

        # 오픈 포지션 향후수익률 = 현재가 평가(라이브), 청산은 net_pnl_pct
        def _future_ret(r):
            if r["status"] == "closed":
                return r.get("net_pnl_pct")
            ep = r.get("진입가")
            if ep and pd.notna(ep) and ep > 0:
                cur = kl.get_price(r["ticker"])
                if cur:
                    return (cur - ep) / ep * 100
            return None

        view = pd.DataFrame({
            "종목명": df["name"],
            "에이전트": df["agent"],
            "트리거": df["trigger_str"],
            "win_prob": df.apply(
                lambda r: "N/A" if r["agent"] == "trend"
                else (f"{r['win_prob']*100:.1f}%" if pd.notna(r["win_prob"]) else "-"),
                axis=1,
            ),
            "RR": df["rr"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "-"),
            "AUC": df["agent"].apply(
                lambda a: f"{auc_map.get(a):.3f}" if auc_map.get(a) else "N/A"
            ),
            "진입일": df["timestamp"].astype(str).str.slice(0, 16),
            "진입가": df["진입가"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "-"),
            "향후수익률": df.apply(_future_ret, axis=1).apply(_fmt_pct),
            "상태": df["status"],
        })

        # 필터/정렬
        f1, f2, f3 = st.columns([2, 2, 2])
        agents = sorted(df["agent"].dropna().unique().tolist())
        sel_agent = f1.multiselect("에이전트 필터", agents, default=agents)
        statuses = sorted(df["status"].dropna().unique().tolist())
        sel_status = f2.multiselect("상태 필터", statuses, default=statuses)
        sort_key = f3.selectbox(
            "정렬", ["진입일(최신)", "향후수익률(높은순)", "win_prob(높은순)", "RR(높은순)"]
        )

        mask = view["에이전트"].isin(sel_agent) & view["상태"].isin(sel_status)
        vshow = view[mask].copy()

        # 정렬용 숫자 보조키
        if sort_key == "향후수익률(높은순)":
            vshow["_k"] = pd.to_numeric(
                vshow["향후수익률"].str.replace("%", "").replace("-", None), errors="coerce")
            vshow = vshow.sort_values("_k", ascending=False, na_position="last").drop(columns="_k")
        elif sort_key == "win_prob(높은순)":
            vshow["_k"] = pd.to_numeric(
                vshow["win_prob"].str.replace("%", "").replace({"N/A": None, "-": None}),
                errors="coerce")
            vshow = vshow.sort_values("_k", ascending=False, na_position="last").drop(columns="_k")
        elif sort_key == "RR(높은순)":
            vshow["_k"] = pd.to_numeric(vshow["RR"].replace("-", None), errors="coerce")
            vshow = vshow.sort_values("_k", ascending=False, na_position="last").drop(columns="_k")
        else:
            vshow = vshow.sort_values("진입일", ascending=False)

        styled = vshow.style.applymap(_color_signed, subset=["향후수익률"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()

        # ── 누적수익 곡선 ─────────────────────────────────────────────
        eq = dl.paper_equity_curve(df)
        st.subheader("📈 누적 수익 곡선 (청산 기준)")
        if eq.empty:
            st.caption("청산된 거래가 아직 없어 곡선을 표시할 수 없습니다.")
        else:
            st.plotly_chart(
                ch.equity_curve(eq["청산시각"], eq["누적수익률"], "페이퍼 누적 수익률"),
                use_container_width=True,
            )

        # ── 에이전트별 / 트리거별 성과 ────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🤖 에이전트별 성과")
            ap = dl.paper_agent_perf(df)
            if ap.empty:
                st.caption("청산 거래 없음")
            else:
                st.plotly_chart(
                    ch.grouped_bar(
                        ap["에이전트"],
                        {"평균수익률(%)": ap["평균수익률"], "승률(%)": ap["승률"].fillna(0)},
                        "에이전트별 평균수익률 · 승률", "값",
                    ),
                    use_container_width=True,
                )
                st.dataframe(ap, use_container_width=True, hide_index=True)
        with col_b:
            st.subheader("🎯 트리거별 성과")
            tp = dl.paper_trigger_perf(df)
            if tp.empty:
                st.caption("청산 거래 없음")
            else:
                st.plotly_chart(
                    ch.bar_compare(tp["트리거"], tp["평균수익률"],
                                   "트리거별 평균수익률", "평균수익률 (%)"),
                    use_container_width=True,
                )
                st.dataframe(tp, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════
# 탭 B: 실제 매매
# ═════════════════════════════════════════════════════════════════════════
with tab_live:
    df_hist = dl.load_trade_history()
    df_pos = dl.load_live_positions()

    # ── 요약 카드 ─────────────────────────────────────────────────────
    ls = dl.live_summary(df_hist, df_pos)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("보유 종목 수", f"{ls['보유종목']}건")
    c3.metric("실현손익 누계", _fmt_won(ls["실현손익"]))
    c4.metric("승률", _fmt_pct(ls["승률"]))

    # 평가손익(보유 포지션 현재가 기준) 계산
    eval_pnl = 0.0
    pos_rows = []
    if not df_pos.empty:
        for _, p in df_pos.iterrows():
            entry = float(p.get("entry_price", 0) or 0)
            qty = int(p.get("qty", 0) or 0)
            stop = float(p.get("stop_price", 0) or 0)
            cur = kl.get_price_with_fallback(p["ticker"], entry)
            pnl_pct = (cur - entry) / entry * 100 if entry > 0 else 0.0
            pnl_amt = (cur - entry) * qty
            eval_pnl += pnl_amt
            # 손절 근접(-5% 이내) 경고
            near_stop = ""
            if stop > 0 and cur <= stop * 1.05:
                near_stop = "⚠️"
            pos_rows.append({
                "종목명": p.get("name", ""),
                "에이전트": p.get("agent", "-"),  # ml_positions엔 없음 → '-'
                "진입가": f"{entry:,.0f}",
                "현재가": f"{cur:,.0f}",
                "손익률": f"{pnl_pct:.2f}%",
                "손익금액": f"{pnl_amt:,.0f}",
                "손절가": f"{stop:,.0f}",
                "상태": f"{near_stop} 보유" if near_stop else "보유",
            })
    c2.metric("평가손익(보유)", _fmt_won(eval_pnl) if not df_pos.empty else "-")

    st.divider()

    # ── 보유 포지션 실시간 손익 ───────────────────────────────────────
    st.subheader("📦 보유 포지션 (실시간)")
    if not pos_rows:
        st.info("현재 보유 중인 실매매 포지션이 없습니다. (LIVE_TRADING 비활성 또는 미보유)")
    else:
        pos_df = pd.DataFrame(pos_rows)
        styled = pos_df.style.applymap(_color_signed, subset=["손익률", "손익금액"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── 매수/매도 이력 ────────────────────────────────────────────────
    st.subheader("🧾 매수/매도 이력")
    if df_hist.empty:
        st.info("매매 이력이 없습니다.")
    else:
        hist_view = pd.DataFrame({
            "날짜": df_hist.get("entry_date", pd.Series(dtype=str)).astype(str).str.slice(0, 16),
            "종목": df_hist.get("name", ""),
            "구분": df_hist.get("side", "-"),
            "수량": df_hist.get("qty", pd.Series(dtype=float)),
            "진입가": df_hist.get("entry_price", pd.Series(dtype=float)),
            "청산가": df_hist.get("exit_price", pd.Series(dtype=float)),
            "전략": df_hist.get("strategy", "-"),  # G2: agent 없음 → 전략
            "트리거": df_hist.get("notes", "-").replace("", "-"),  # G2: 트리거 미저장
            "win_prob": df_hist.get("win_prob", pd.Series(dtype=float)).apply(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-"),
            "AUC": df_hist.get("model_auc", pd.Series(dtype=float)).apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else "-"),
            "실현손익": df_hist.get("pnl_amount", pd.Series(dtype=float)).apply(
                lambda v: f"{v:,.0f}" if pd.notna(v) else "-"),
        })
        hist_view = hist_view.iloc[::-1]  # 최신 먼저
        styled = hist_view.style.applymap(_color_signed, subset=["실현손익"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── 누적 실현손익 곡선 ────────────────────────────────────────────
    rc = dl.live_realized_curve(df_hist)
    st.subheader("📉 누적 실현손익 곡선")
    if rc.empty:
        st.caption("청산된 실매매 거래가 없습니다.")
    else:
        st.plotly_chart(
            ch.equity_curve(rc["청산일"], rc["누적실현손익"],
                            "실매매 누적 실현손익", "누적 실현손익 (원)"),
            use_container_width=True,
        )

    # ── 전략(에이전트 대체)별 실현손익 ───────────────────────────────
    st.subheader("🤖 전략별 실현손익")
    sp = dl.live_strategy_perf(df_hist)
    if sp.empty:
        st.caption("청산 거래 없음")
    else:
        st.plotly_chart(
            ch.bar_compare(sp["전략"], sp["실현손익합"],
                           "전략별 누적 실현손익", "실현손익 (원)", value_fmt=",.0f"),
            use_container_width=True,
        )
        st.dataframe(sp, use_container_width=True, hide_index=True)
        st.caption("※ 실매매 CSV에는 에이전트/트리거 컬럼이 없어 strategy로 그룹핑합니다 (G2).")


# ═════════════════════════════════════════════════════════════════════════
# 탭 C: 뉴스 브리핑
# ═════════════════════════════════════════════════════════════════════════
with tab_news:
    if not os.path.exists(NB_DB_PATH):
        st.info("아직 브리핑 데이터가 없습니다.")
    else:
        try:
            nb_morning = nb_db.get_latest_briefing(kind="morning")
            nb_evening = nb_db.get_latest_briefing(kind="evening")
            nb_score_hist = nb_db.get_score_history(limit=90)
            nb_forecast_hist = nb_db.get_forecast_history(limit=90)
            nb_feedback_ratio = nb_db.get_feedback_ratio()
            nb_hit_rate = nb_db.get_hit_rate()
        except Exception:
            nb_morning = nb_evening = None
            nb_score_hist = nb_forecast_hist = []
            nb_feedback_ratio = nb_hit_rate = None
            st.info("아직 브리핑 데이터가 없습니다.")
        else:
            # ── 최신 브리핑 전문 ─────────────────────────────────────
            st.subheader("📝 최신 브리핑 전문")

            def _render_briefing(label, briefing):
                st.markdown(f"#### {label}")
                if briefing is None:
                    st.info(f"{label} 이력이 없습니다.")
                    return
                st.caption(
                    f"발송: {briefing.get('sent_at') or '-'}  |  "
                    f"fact_score: {_fmt_score(briefing.get('fact_score'))}  |  "
                    f"grounding_score: {_fmt_score(briefing.get('grounding_score'))}"
                )
                st.markdown(briefing.get("body_html") or "", unsafe_allow_html=True)

            col_m, col_e = st.columns(2)
            with col_m:
                _render_briefing("🌅 아침 브리핑", nb_morning)
            with col_e:
                _render_briefing("🌆 저녁 브리핑", nb_evening)

            st.divider()

            # ── 신뢰 점수 4축 추이 ───────────────────────────────────
            st.subheader("📈 신뢰 점수 추이")
            if not nb_score_hist:
                st.caption("추이를 표시할 브리핑 이력이 없습니다.")
            else:
                sdf = pd.DataFrame(nb_score_hist)
                sdf["x"] = sdf["sent_at"].fillna(sdf["id"].astype(str))
                st.plotly_chart(
                    ch.line_series(
                        sdf["x"],
                        {"fact_score": sdf["fact_score"], "grounding_score": sdf["grounding_score"]},
                        "신뢰 점수 추이 (fact_score / grounding_score)", "점수",
                    ),
                    use_container_width=True,
                )

            m1, m2 = st.columns(2)
            recent_scored = [f for f in nb_forecast_hist if f.get("verdict") in ("hit", "miss")]
            if recent_scored:
                recent_hits = sum(1 for f in recent_scored if f["verdict"] == "hit")
                m1.metric("최근 90건 기준 적중률", f"{recent_hits / len(recent_scored) * 100:.1f}%")
            else:
                m1.metric("최근 90건 기준 적중률", "채점 이력 없음")
            m2.metric(
                "피드백 👍 비율",
                f"{nb_feedback_ratio * 100:.1f}%" if nb_feedback_ratio is not None else "피드백 없음",
            )

            st.divider()

            # ── 전망 채점 상세 ───────────────────────────────────────
            st.subheader("🎯 전망 채점 상세")
            st.metric(
                "누적 적중률",
                f"{nb_hit_rate * 100:.1f}%" if nb_hit_rate is not None else "채점 이력 없음",
            )
            if not nb_forecast_hist:
                st.info("전망 채점 이력이 없습니다.")
            else:
                fdf = pd.DataFrame(nb_forecast_hist)
                _direction_ko = {"up": "상승", "flat": "보합", "down": "하락"}
                _verdict_ko = {"hit": "적중", "miss": "실패", "pending": "대기중"}
                fdf_view = pd.DataFrame({
                    "시장": fdf["market"],
                    "방향": fdf["direction"].map(_direction_ko).fillna(fdf["direction"]),
                    "근거": fdf["rationale"],
                    "실제등락률": fdf["actual_pct"].apply(_fmt_pct),
                    "판정": fdf["verdict"].map(_verdict_ko).fillna(fdf["verdict"]),
                    "채점시각": fdf["scored_at"].fillna("-"),
                })
                fdf_view = fdf_view.iloc[::-1]  # 최신 먼저
                st.dataframe(fdf_view, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# 자동 새로고침 (체크 시 30초 후 재실행). 현재가는 ttl=10 캐싱이라 API 폭주 없음.
# ─────────────────────────────────────────────────────────────────────────
if auto:
    time.sleep(30)
    st.rerun()

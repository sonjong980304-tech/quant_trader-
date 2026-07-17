"""
news_briefing/service.py — 아침/저녁 브리핑 오케스트레이션
(플랜: .omc/plans/news-briefing-revamp-plan.md §3 service.py)

run_morning: RSS/네이버 수집 → DB 저장 → 선별 → 원문 보강 → 시장 스냅샷 병합 →
             작성 → 검증 게이트 → HTML 조립 → DB 저장(인용·전망 포함) → 발송.
run_evening: 대기 중인 전망 채점 → 보유종목 마감 뉴스 수집 → 간단 HTML 조립 →
             DB 저장 → 발송.

하위 모듈은 항상 모듈 자체를 임포트해 `module.func(...)` 형태로 호출한다
(gate.py의 관례와 동일 — 테스트가 `service.db.insert_articles` 식으로 monkeypatch하기 위함).

datetime.now()를 직접 쓰지 않고 now_fn 파라미터로 주입 가능하게 한다
(기본값은 datetime.now 자체, 테스트는 고정값 반환 함수를 주입).
"""
import html
import json
import logging
from datetime import datetime

from news_briefing import db, formatter, gate, market_data, positions, scorer, selector, sources, writer

logger = logging.getLogger(__name__)


def _now_str(now_fn):
    """now_fn()이 반환한 값을 ISO 문자열로 변환한다(datetime 객체·문자열 둘 다 허용)."""
    now = now_fn()
    return now.isoformat() if hasattr(now, "isoformat") else str(now)


def _make_counting_llm_call(llm_call):
    """llm_call을 감싸 호출 횟수를 세는 래퍼를 만든다(AC17: selector/writer/gate 일괄 전달용).
    llm_call이 None이면 래핑하지 않는다(하위 모듈이 각자 기본 LLM 경로를 쓰므로 카운트 불가)."""
    if llm_call is None:
        return None, [0]

    count = [0]

    def _wrapped(system, user):
        count[0] += 1
        return llm_call(system, user)

    return _wrapped, count


def run_morning(now_fn=None, llm_call=None, send_fn=None, db_path=None):
    """
    아침 브리핑 파이프라인.
    반환: 정상 시 {"briefing_id","fact_score","grounding_score","regen_count","llm_call_count"}
          치명적 실패(선별/작성 LLM 파싱 최종 실패 등) 시 {"error": str} (예외 전파 없음)
    """
    if now_fn is None:
        now_fn = datetime.now

    # 1. RSS 수집 (실패해도 계속)
    rss_articles, failed_sources = sources.fetch_rss()
    if failed_sources:
        logger.warning("[Service] RSS 수집 실패 소스: %s", failed_sources)

    # 2. 보유종목
    holdings = positions.get_holdings()

    # 3. 보유종목별 네이버 뉴스 (종목별 예외 격리 — 한 종목 실패가 전체를 막지 않는다)
    naver_articles = []
    for h in holdings:
        try:
            query = h.get("name") or h.get("ticker")
            naver_articles.extend(sources.fetch_naver(query))
        except Exception as e:
            logger.warning("[Service] 보유종목 뉴스 수집 실패 (%s): %s", h.get("ticker"), e)
            continue

    wrapped_llm_call, llm_call_count = _make_counting_llm_call(llm_call)

    try:
        # 4. 후보 기사 합침 + DB 삽입 + id 채우기
        candidates = list(rss_articles) + naver_articles
        ids = db.insert_articles(candidates, db_path=db_path)
        for candidate, candidate_id in zip(candidates, ids):
            candidate["id"] = candidate_id

        # 5. 선별 (매크로 + 종목뉴스)
        selection = selector.select_articles(candidates, holdings, wrapped_llm_call)
        macro_ids = set(selection.get("macro_article_ids", []))
        stock_news = selection.get("stock_news", [])
        selected_ids = macro_ids | {sn["article_id"] for sn in stock_news}
        selected = [c for c in candidates if c.get("id") in selected_ids]

        # 6. 선택된 기사만 원문 보강
        sources.fetch_fulltext_batch(selected, limit=12)

        # 7. 시장 스냅샷 병합 (nasdaq/sp500/asof + kospi/kosdaq을 최상위로)
        us_snapshot = market_data.get_us_snapshot()
        kr_snapshot = market_data.get_kr_index_snapshot()
        market_snapshot = dict(us_snapshot)
        market_snapshot["kospi"] = kr_snapshot.get("kospi")
        market_snapshot["kosdaq"] = kr_snapshot.get("kosdaq")

        # 8. closed-book 작성
        draft = writer.write_briefing(selected, market_snapshot, stock_news, wrapped_llm_call)

        # 9. 검증 게이트
        gate_result = gate.run_gate(draft, selected, market_snapshot, wrapped_llm_call)
        final_draft = gate_result["final_draft"]
        fact_score = gate_result["fact_score"]
        grounding_score = gate_result["grounding_score"]
        regen_count = gate_result["regen_count"]
        warn_flags = gate_result["warn_flags"]

        # 10. HTML 조립 (일부 소스 수집 실패 시 본문 말미에 누락 표기 — AC15)
        articles_by_id = {a["id"]: a for a in selected}
        body_html = formatter.format_briefing_html(
            final_draft, 0, articles_by_id, fact_score, grounding_score, warn_flags
        )
        if failed_sources:
            body_html += "\n\n⚠️ 일부 소스 누락: {}".format(html.escape(", ".join(failed_sources)))

        # 11. 브리핑 저장(id 확보)
        briefing_id = db.insert_briefing("morning", body_html, db_path=db_path)

        # 12. 인용 저장 (market_data 태그 제외)
        citations = []
        for sentence in final_draft.get("sentences", []):
            sentence_idx = sentence.get("idx")
            for article_id in sentence.get("article_ids") or []:
                if article_id == writer.MARKET_DATA_TAG:
                    continue
                citations.append((sentence_idx, article_id))
        db.insert_citations(briefing_id, citations, db_path=db_path)

        # 13. 전망 저장 (KOSPI·KOSDAQ 각 1건)
        for forecast in final_draft.get("forecasts", []):
            db.insert_forecast(
                briefing_id,
                forecast.get("market"),
                forecast.get("direction"),
                forecast.get("rationale"),
                db_path=db_path,
            )

        # 14. 피드백 키보드
        keyboard = formatter.build_feedback_keyboard_payload(briefing_id)

        # 15. 텔레그램 길이 제한 분할
        chunks = formatter.split_html_by_length(body_html)

        # 16. 발송 (send_fn 미주입 시 로그만 — 실제 텔레그램 연결은 US-012)
        if send_fn is not None:
            send_fn(chunks, keyboard)
        else:
            logger.info("[Service] send_fn 미주입 — 발송 스킵(briefing_id=%s)", briefing_id)

        # 17. 발송 완료 기록
        sent_at = _now_str(now_fn)
        db.update_briefing_sent(
            briefing_id, sent_at, fact_score, grounding_score, regen_count,
            json.dumps(warn_flags), db_path=db_path,
        )

        # 18. LLM 호출 횟수 로그 (AC17)
        logger.info("[Service] run_morning LLM 호출 횟수: %d", llm_call_count[0])

        return {
            "briefing_id": briefing_id,
            "fact_score": fact_score,
            "grounding_score": grounding_score,
            "regen_count": regen_count,
            "llm_call_count": llm_call_count[0],
        }
    except Exception as e:
        # 19. 치명적 실패(선별/작성 LLM 파싱 최종 실패 등) — 예외를 전파하지 않고 발송 스킵
        logger.error("[Service] run_morning 치명적 실패 — 발송 스킵: %s", e)
        return {"error": str(e)}


def run_evening(now_fn=None, market_data_fn=None, sleep_fn=None, llm_call=None, send_fn=None,
                 db_path=None):
    """
    저녁 브리핑 파이프라인: 대기 중인 전망 채점 → 보유종목 마감 뉴스 수집 →
    간단 HTML 조립 → DB 저장 → 발송.
    반환: {"briefing_id","scored_count","holdings_news_count"}
    """
    if now_fn is None:
        now_fn = datetime.now
    sent_at = _now_str(now_fn)

    # 1. 대기 중인 전망 채점 (없으면 스킵)
    pending = db.get_pending_forecasts(db_path=db_path)
    if not pending:
        logger.info("[Service] 채점 대상 없음")
        scored = []
    else:
        scored = scorer.score_pending_forecasts(
            sent_at, db_path=db_path, market_data_fn=market_data_fn, sleep_fn=sleep_fn,
        )

    # 2. 보유종목 마감 뉴스 수집 (종목별 예외 격리)
    holdings = positions.get_holdings()
    naver_articles = []
    for h in holdings:
        try:
            query = h.get("name") or h.get("ticker")
            naver_articles.extend(sources.fetch_naver(query))
        except Exception as e:
            logger.warning("[Service] 저녁 보유종목 뉴스 수집 실패 (%s): %s", h.get("ticker"), e)
            continue

    ids = db.insert_articles(naver_articles, db_path=db_path)
    for article, article_id in zip(naver_articles, ids):
        article["id"] = article_id

    selection = selector.select_articles(naver_articles, holdings, llm_call)
    stock_news = selection.get("stock_news", [])

    selected_ids = {sn["article_id"] for sn in stock_news}
    selected_articles = [a for a in naver_articles if a.get("id") in selected_ids]
    sources.fetch_fulltext_batch(selected_articles, limit=12)
    articles_by_id = {a["id"]: a for a in selected_articles}

    # 3. 저녁 본문 직접 조립 (format_briefing_html은 4섹션 구조라 저녁 포맷과 안 맞음)
    _VERDICT_LABELS = {"hit": "적중", "miss": "미적중", "pending": "채점 보류"}
    parts = ["<b>채점 결과</b>"]
    if scored:
        for s in scored:
            actual_pct = s.get("actual_pct")
            pct_text = "{:.2f}".format(actual_pct) if actual_pct is not None else "N/A"
            verdict = s.get("verdict", "")
            parts.append(
                "{}: {} ({}%)".format(
                    html.escape(str(s.get("market", ""))),
                    html.escape(_VERDICT_LABELS.get(verdict, verdict)),
                    pct_text,
                )
            )
    else:
        parts.append("채점 대상 없음")

    parts.append("<b>보유종목 마감 뉴스</b>")
    if stock_news:
        for sn in stock_news:
            article = articles_by_id.get(sn.get("article_id"))
            url = article.get("url", "") if article else ""
            parts.append(
                '{}: {} <a href="{}">[link]</a>'.format(
                    html.escape(str(sn.get("ticker", ""))),
                    html.escape(str(sn.get("summary", ""))),
                    html.escape(url),
                )
            )
    else:
        parts.append("보유종목 마감 뉴스 없음")

    body_html = "\n".join(parts)

    briefing_id = db.insert_briefing("evening", body_html, db_path=db_path)

    if send_fn is not None:
        send_fn([body_html], None)

    db.update_briefing_sent(briefing_id, sent_at, None, None, 0, "[]", db_path=db_path)

    return {
        "briefing_id": briefing_id,
        "scored_count": len(scored),
        "holdings_news_count": len(stock_news),
    }

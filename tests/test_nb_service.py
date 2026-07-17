"""
tests/test_nb_service.py — news_briefing.service 오케스트레이션 TDD

run_morning/run_evening이 9개 하위 모듈(db, sources, positions, market_data, selector,
writer, gate, scorer, formatter)을 올바른 순서·인자로 호출하는지 검증한다.
모든 하위 모듈 함수는 monkeypatch로 대체 — 실제 API·DB·네트워크 호출 없음.
db_path는 실제로 쓰이지 않는다(모든 db.* 함수가 monkeypatch로 대체되므로 ":memory:"는
단순 placeholder).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from news_briefing import service  # noqa: E402


# ─────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────

def _make_draft():
    return {
        "sentences": [
            {"idx": 0, "section": "nasdaq", "text": "나스닥 상승 마감", "article_ids": ["market_data"]},
            {"idx": 1, "section": "cause", "text": "기술주 강세가 견인", "article_ids": [1]},
        ],
        "forecasts": [
            {"market": "KOSPI", "direction": "up", "rationale": "미국 강세 동조"},
            {"market": "KOSDAQ", "direction": "flat", "rationale": "혼조"},
        ],
    }


def _install_common_mocks(monkeypatch, log, llm_calls_from=("selector", "writer", "gate")):
    """run_morning happy-path 하위 모듈을 전부 monkeypatch한다.
    log에 호출 순서(문자열)를 append하고, refs 딕셔너리에 주요 호출 인자를 담아 반환한다.
    llm_calls_from에 지정된 단계의 fake는 전달받은 llm_call을 1회 호출해, 실제 selector/writer/
    gate(grounding) 내부가 LLM을 각 1회씩 호출하는 상황을 흉내낸다(AC17 카운트 검증용)."""
    refs = {
        "forecast_markets": [],
        "citations": None,
        "keyboard_briefing_id": None,
        "market_snapshot": None,
    }

    rss_articles = [
        {"url": "https://cnbc.com/a", "domain": "cnbc.com", "source_lang": "en",
         "title": "t1", "body": "b1", "published_at": "p1", "http_status": None},
    ]
    monkeypatch.setattr(service.sources, "fetch_rss", lambda: (list(rss_articles), []))
    monkeypatch.setattr(
        service.positions, "get_holdings",
        lambda: [{"ticker": "005930", "name": "삼성전자", "source": "paper"}],
    )
    monkeypatch.setattr(service.sources, "fetch_naver", lambda q: [])

    def fake_insert_articles(articles, db_path=None):
        log.append("insert_articles")
        return list(range(1, len(articles) + 1))

    monkeypatch.setattr(service.db, "insert_articles", fake_insert_articles)

    def fake_select_articles(articles, holdings, llm_call=None):
        log.append("select_articles")
        if llm_call is not None and "selector" in llm_calls_from:
            llm_call("sys", "user")
        return {"macro_article_ids": [1], "stock_news": []}

    monkeypatch.setattr(service.selector, "select_articles", fake_select_articles)

    def fake_fetch_fulltext_batch(articles, limit=12, timeout=5):
        log.append("fetch_fulltext_batch")
        return articles

    monkeypatch.setattr(service.sources, "fetch_fulltext_batch", fake_fetch_fulltext_batch)

    monkeypatch.setattr(
        service.market_data, "get_us_snapshot",
        lambda: {"nasdaq": {"close": 18000.0, "change_pct": 1.0},
                 "sp500": {"close": 5000.0, "change_pct": 0.5}, "asof": "2026-07-10"},
    )
    monkeypatch.setattr(
        service.market_data, "get_kr_index_snapshot",
        lambda: {"kospi": {"close": 3000.0, "change_pct": 0.2, "asof": "2026-07-10"}, "kosdaq": None},
    )

    draft = _make_draft()

    def fake_write_briefing(articles, market_snapshot, stock_news, llm_call=None):
        log.append("write_briefing")
        refs["market_snapshot"] = market_snapshot
        if llm_call is not None and "writer" in llm_calls_from:
            llm_call("sys", "user")
        return draft

    monkeypatch.setattr(service.writer, "write_briefing", fake_write_briefing)

    gate_result = {
        "final_draft": draft,
        "fact_score": 1.0,
        "grounding_score": 1.0,
        "regen_count": 0,
        "warn_flags": [],
    }

    def fake_run_gate(d, articles, market_snapshot, llm_call=None, max_regen=2):
        log.append("run_gate")
        if llm_call is not None and "gate" in llm_calls_from:
            llm_call("sys", "user")
        return gate_result

    monkeypatch.setattr(service.gate, "run_gate", fake_run_gate)

    monkeypatch.setattr(
        service.formatter, "format_briefing_html",
        lambda *a, **k: "<html>body</html>",
    )

    def fake_insert_briefing(kind, body_html=None, db_path=None):
        log.append("insert_briefing")
        return 99

    monkeypatch.setattr(service.db, "insert_briefing", fake_insert_briefing)

    def fake_insert_citations(briefing_id, citations, db_path=None):
        log.append("insert_citations")
        refs["citations"] = citations

    monkeypatch.setattr(service.db, "insert_citations", fake_insert_citations)

    def fake_insert_forecast(briefing_id, market, direction, rationale, db_path=None):
        log.append("insert_forecast")
        refs["forecast_markets"].append(market)
        return len(refs["forecast_markets"])

    monkeypatch.setattr(service.db, "insert_forecast", fake_insert_forecast)

    def fake_build_feedback_keyboard_payload(briefing_id):
        log.append("build_feedback_keyboard_payload")
        refs["keyboard_briefing_id"] = briefing_id
        return [[{"text": "up"}]]

    monkeypatch.setattr(
        service.formatter, "build_feedback_keyboard_payload", fake_build_feedback_keyboard_payload
    )
    monkeypatch.setattr(
        service.formatter, "split_html_by_length", lambda html_text, max_len=4096: [html_text]
    )

    def fake_update_briefing_sent(briefing_id, sent_at, fact_score, grounding_score,
                                   regen_count, warn_flags, db_path=None):
        log.append("update_briefing_sent")

    monkeypatch.setattr(service.db, "update_briefing_sent", fake_update_briefing_sent)

    return refs


# ─────────────────────────────────────────────
# 1. run_morning 호출 순서 + 데이터 결선
# ─────────────────────────────────────────────

class TestRunMorningHappyPath:
    def test_call_order_and_payload(self, monkeypatch):
        log = []
        refs = _install_common_mocks(monkeypatch, log)

        send_calls = []

        def fake_send_fn(chunks, keyboard):
            log.append("send_fn")
            send_calls.append((chunks, keyboard))

        result = service.run_morning(
            llm_call=lambda system, user: "ok",
            send_fn=fake_send_fn,
            db_path=":memory:",
        )

        assert log == [
            "insert_articles",
            "select_articles",
            "fetch_fulltext_batch",
            "write_briefing",
            "run_gate",
            "insert_briefing",
            "insert_citations",
            "insert_forecast",
            "insert_forecast",
            "build_feedback_keyboard_payload",
            "send_fn",
            "update_briefing_sent",
        ]
        assert refs["forecast_markets"] == ["KOSPI", "KOSDAQ"]
        # market_data 태그 문장(idx=0)은 인용에서 제외, idx=1만 (idx, article_id) 쌍으로 남는다
        assert refs["citations"] == [(1, 1)]
        assert refs["keyboard_briefing_id"] == 99
        # market_snapshot 병합: nasdaq/sp500(미국) + kospi/kosdaq(한국)가 최상위 키
        assert refs["market_snapshot"]["nasdaq"]["close"] == 18000.0
        assert refs["market_snapshot"]["sp500"]["close"] == 5000.0
        assert refs["market_snapshot"]["kospi"]["close"] == 3000.0
        assert refs["market_snapshot"]["kosdaq"] is None
        assert refs["market_snapshot"]["asof"] == "2026-07-10"

        assert result["briefing_id"] == 99
        assert result["fact_score"] == 1.0
        assert result["grounding_score"] == 1.0
        assert result["regen_count"] == 0
        assert send_calls == [(["<html>body</html>"], [[{"text": "up"}]])]

    def test_failed_sources_appended_as_notice(self, monkeypatch):
        """RSS 일부 소스가 실패해도 발송은 계속되고, 본문 말미에 누락 소스가 표기된다(AC15)."""
        log = []
        _install_common_mocks(monkeypatch, log)
        monkeypatch.setattr(
            service.sources, "fetch_rss",
            lambda: ([{"url": "https://cnbc.com/a", "domain": "cnbc.com", "source_lang": "en",
                       "title": "t1", "body": "b1", "published_at": "p1", "http_status": None}],
                      ["reuters_markets", "ap_business"]),
        )

        send_calls = []
        result = service.run_morning(
            llm_call=lambda system, user: "ok",
            send_fn=lambda chunks, keyboard: send_calls.append(chunks),
            db_path=":memory:",
        )

        assert "error" not in result
        assert send_calls, "일부 소스 실패에도 발송은 이뤄져야 한다"
        sent_text = "".join(send_calls[0])
        assert "일부 소스 누락" in sent_text
        assert "reuters_markets" in sent_text
        assert "ap_business" in sent_text


# ─────────────────────────────────────────────
# 2. run_evening — 채점 스킵/실행 분기
# ─────────────────────────────────────────────

class TestRunEveningScoring:
    def _install(self, monkeypatch, pending, scored):
        monkeypatch.setattr(service.db, "get_pending_forecasts", lambda db_path=None: pending)

        score_calls = []

        def fake_score_pending_forecasts(now_str, db_path=None, market_data_fn=None, sleep_fn=None):
            score_calls.append(now_str)
            return scored

        monkeypatch.setattr(service.scorer, "score_pending_forecasts", fake_score_pending_forecasts)

        monkeypatch.setattr(service.positions, "get_holdings", lambda: [])
        monkeypatch.setattr(service.sources, "fetch_naver", lambda q: [])
        monkeypatch.setattr(service.db, "insert_articles", lambda articles, db_path=None: [])
        monkeypatch.setattr(
            service.selector, "select_articles",
            lambda articles, holdings, llm_call=None: {"macro_article_ids": [], "stock_news": []},
        )
        monkeypatch.setattr(
            service.sources, "fetch_fulltext_batch", lambda articles, limit=12, timeout=5: articles
        )
        monkeypatch.setattr(
            service.db, "insert_briefing", lambda kind, body_html=None, db_path=None: 42
        )
        monkeypatch.setattr(
            service.db, "update_briefing_sent",
            lambda briefing_id, sent_at, fact_score, grounding_score, regen_count, warn_flags,
            db_path=None: None,
        )
        return score_calls

    def test_no_pending_skips_scoring(self, monkeypatch):
        score_calls = self._install(monkeypatch, pending=[], scored=[])

        result = service.run_evening(db_path=":memory:")

        assert score_calls == []
        assert result["scored_count"] == 0
        assert result["briefing_id"] == 42

    def test_pending_triggers_scoring(self, monkeypatch):
        pending = [{"id": 1, "market": "KOSPI", "direction": "up"}]
        scored = [{"forecast_id": 1, "market": "KOSPI", "verdict": "hit", "actual_pct": 1.0}]
        score_calls = self._install(monkeypatch, pending=pending, scored=scored)

        result = service.run_evening(db_path=":memory:")

        assert len(score_calls) == 1
        assert result["scored_count"] == 1

    def test_scored_body_formats_percent_and_korean_verdict(self, monkeypatch):
        """actual_pct는 소수점 2자리로 반올림, verdict는 한글 라벨로 표시된다
        (실제 발송에서 2.523755778664297% 같은 미가공 float가 노출됐던 회귀 방지)."""
        pending = [{"id": 1, "market": "KOSPI", "direction": "up"}]
        scored = [
            {"forecast_id": 1, "market": "KOSPI", "verdict": "hit", "actual_pct": 2.523755778664297},
        ]
        self._install(monkeypatch, pending=pending, scored=scored)

        captured = {}

        def fake_insert_briefing(kind, body_html=None, db_path=None):
            captured["body_html"] = body_html
            return 42

        monkeypatch.setattr(service.db, "insert_briefing", fake_insert_briefing)

        service.run_evening(db_path=":memory:")

        assert "2.52%" in captured["body_html"]
        assert "2.523755778664297" not in captured["body_html"]
        assert "적중" in captured["body_html"]
        assert "hit" not in captured["body_html"]


# ─────────────────────────────────────────────
# 3. LLM 호출 계수 (AC17)
# ─────────────────────────────────────────────

class TestRunMorningLlmCallCount:
    def test_llm_call_count_is_three_on_happy_path(self, monkeypatch):
        log = []
        _install_common_mocks(monkeypatch, log)

        result = service.run_morning(
            llm_call=lambda system, user: "ok",
            send_fn=lambda chunks, keyboard: None,
            db_path=":memory:",
        )

        assert result["llm_call_count"] == 3


# ─────────────────────────────────────────────
# 4. 치명적 실패 경로 — writer 예외 시 발송 스킵 + error 반환
# ─────────────────────────────────────────────

class TestRunMorningFatalFailure:
    def test_writer_value_error_returns_error_dict_without_sending(self, monkeypatch):
        log = []
        _install_common_mocks(monkeypatch, log)

        def raising_write_briefing(articles, market_snapshot, stock_news, llm_call=None):
            log.append("write_briefing")
            raise ValueError("closed-book 스키마 파싱 실패 (재요청 포함 2회)")

        monkeypatch.setattr(service.writer, "write_briefing", raising_write_briefing)

        send_calls = []

        def fake_send_fn(chunks, keyboard):
            send_calls.append((chunks, keyboard))

        result = service.run_morning(
            llm_call=lambda system, user: "ok",
            send_fn=fake_send_fn,
            db_path=":memory:",
        )

        assert "error" in result
        assert send_calls == []
        assert "run_gate" not in log
        assert "insert_briefing" not in log
        assert "update_briefing_sent" not in log

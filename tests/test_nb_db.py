"""
tests/test_nb_db.py — news_briefing.db 스키마·CRUD·동시성 검증

플랜(.omc/plans/news-briefing-revamp-plan.md) §3의 동시성 정책(WAL + busy_timeout)이
runner/telegram_bot 두 개의 별도 OS 프로세스가 동시에 써도 무손실·무잠금인지까지 검증한다.
"""
import multiprocessing
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_briefing import db  # noqa: E402


def _article(url, title="title"):
    """테스트용 기사 dict를 만든다."""
    return {
        "url": url,
        "domain": "example.com",
        "source_lang": "ko",
        "title": title,
        "body": "본문",
        "published_at": "2026-07-10T08:00:00",
        "fetched_at": "2026-07-10T08:00:00",
    }


def _mp_insert_worker(db_path, prefix, count):
    """
    멀티프로세싱 워커: db_path에 http://{prefix}.example.com/{i} 형태의
    고유 URL로 기사 count건을 삽입한다. (spawn 방식으로 별도 프로세스에서 실행)
    """
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from news_briefing import db as nb_db

    articles = [
        {
            "url": "http://{}.example.com/{}".format(prefix, i),
            "domain": "{}.example.com".format(prefix),
            "source_lang": "ko",
            "title": "title-{}-{}".format(prefix, i),
            "body": "본문",
            "published_at": "2026-07-10T00:00:00",
            "fetched_at": "2026-07-10T00:00:00",
        }
        for i in range(count)
    ]
    nb_db.insert_articles(articles, db_path=db_path)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "news_briefing_test.db")


@pytest.fixture
def initialized_db(db_path):
    db.init_db(db_path)
    return db_path


class TestInitDb:
    def test_creates_five_tables(self, db_path):
        db.init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            names = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        expected = {"articles", "briefings", "briefing_citations", "forecasts", "feedback"}
        assert expected.issubset(names)

    def test_idempotent_when_called_twice(self, db_path):
        db.init_db(db_path)
        db.init_db(db_path)  # IF NOT EXISTS라 재호출해도 에러 없음

    def test_wal_journal_mode(self, initialized_db):
        with sqlite3.connect(initialized_db) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


class TestArticles:
    def test_insert_new_articles_returns_distinct_ids(self, initialized_db):
        ids = db.insert_articles(
            [_article("http://a.com/1"), _article("http://a.com/2")],
            db_path=initialized_db,
        )
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_duplicate_url_reuses_existing_id(self, initialized_db):
        first = db.insert_articles([_article("http://dup.com/1")], db_path=initialized_db)
        second = db.insert_articles(
            [_article("http://dup.com/1", title="다른 제목")], db_path=initialized_db
        )
        assert first == second

    def test_mixed_new_and_duplicate_in_one_call(self, initialized_db):
        db.insert_articles([_article("http://mix.com/1")], db_path=initialized_db)
        ids = db.insert_articles(
            [_article("http://mix.com/1"), _article("http://mix.com/2")],
            db_path=initialized_db,
        )
        assert ids[0] != ids[1]


class TestBriefings:
    def test_insert_and_get_latest(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>hello</p>", db_path=initialized_db)
        assert isinstance(bid, int)
        latest = db.get_latest_briefing(db_path=initialized_db)
        assert latest["id"] == bid
        assert latest["kind"] == "morning"

    def test_get_latest_filters_by_kind(self, initialized_db):
        db.insert_briefing("morning", "<p>m</p>", db_path=initialized_db)
        eid = db.insert_briefing("evening", "<p>e</p>", db_path=initialized_db)
        latest_evening = db.get_latest_briefing(kind="evening", db_path=initialized_db)
        assert latest_evening["id"] == eid

    def test_update_briefing_sent(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        db.update_briefing_sent(
            bid, "2026-07-10T08:00:00", 0.95, 0.9, 0, None, db_path=initialized_db
        )
        latest = db.get_latest_briefing(db_path=initialized_db)
        assert latest["sent_at"] == "2026-07-10T08:00:00"
        assert latest["fact_score"] == 0.95
        assert latest["grounding_score"] == 0.9
        assert latest["regen_count"] == 0

    def test_insert_citations_roundtrip(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        aids = db.insert_articles([_article("http://cite.com/1")], db_path=initialized_db)
        db.insert_citations(bid, [(0, aids[0]), (1, aids[0])], db_path=initialized_db)
        with sqlite3.connect(initialized_db) as conn:
            rows = conn.execute(
                "SELECT sentence_idx, article_id FROM briefing_citations WHERE briefing_id = ? "
                "ORDER BY sentence_idx",
                (bid,),
            ).fetchall()
        assert rows == [(0, aids[0]), (1, aids[0])]


class TestForecasts:
    def test_insert_forecast_defaults_pending(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        fid = db.insert_forecast(bid, "KOSPI", "up", "근거", db_path=initialized_db)
        pending = db.get_pending_forecasts(db_path=initialized_db)
        assert any(f["id"] == fid and f["verdict"] == "pending" for f in pending)

    def test_update_forecast_verdict_removes_from_pending(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        fid = db.insert_forecast(bid, "KOSPI", "up", "근거", db_path=initialized_db)
        db.update_forecast_verdict(fid, 0.5, "hit", "2026-07-10T15:40:00", db_path=initialized_db)
        pending = db.get_pending_forecasts(db_path=initialized_db)
        assert all(f["id"] != fid for f in pending)

    def test_hit_rate_none_when_no_scored_forecasts(self, initialized_db):
        assert db.get_hit_rate(db_path=initialized_db) is None

    def test_hit_rate_computes_ratio(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        f1 = db.insert_forecast(bid, "KOSPI", "up", "r1", db_path=initialized_db)
        f2 = db.insert_forecast(bid, "KOSPI", "down", "r2", db_path=initialized_db)
        f3 = db.insert_forecast(bid, "KOSDAQ", "flat", "r3", db_path=initialized_db)
        db.update_forecast_verdict(f1, 0.5, "hit", "t", db_path=initialized_db)
        db.update_forecast_verdict(f2, 0.5, "hit", "t", db_path=initialized_db)
        db.update_forecast_verdict(f3, -0.5, "miss", "t", db_path=initialized_db)
        assert db.get_hit_rate(db_path=initialized_db) == pytest.approx(2 / 3)

    def test_forecast_history_limit_and_chronological_order(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        for i in range(3):
            db.insert_forecast(bid, "KOSPI", "up", "r{}".format(i), db_path=initialized_db)
        history = db.get_forecast_history(limit=2, db_path=initialized_db)
        assert len(history) == 2
        assert history[0]["id"] < history[1]["id"]


class TestFeedback:
    def test_feedback_ratio_none_when_empty(self, initialized_db):
        assert db.get_feedback_ratio(db_path=initialized_db) is None

    def test_record_feedback_and_ratio(self, initialized_db):
        bid = db.insert_briefing("morning", "<p>x</p>", db_path=initialized_db)
        db.record_feedback(bid, "up", "t1", db_path=initialized_db)
        db.record_feedback(bid, "up", "t2", db_path=initialized_db)
        db.record_feedback(bid, "up", "t3", db_path=initialized_db)
        db.record_feedback(bid, "down", "t4", db_path=initialized_db)
        assert db.get_feedback_ratio(db_path=initialized_db) == pytest.approx(0.75)


class TestScoreHistory:
    def test_score_history_limit_and_chronological_order(self, initialized_db):
        ids = []
        for i in range(3):
            bid = db.insert_briefing("morning", "<p>{}</p>".format(i), db_path=initialized_db)
            db.update_briefing_sent(
                bid, "t{}".format(i), 0.1 * i, 0.2 * i, 0, None, db_path=initialized_db
            )
            ids.append(bid)
        history = db.get_score_history(limit=2, db_path=initialized_db)
        assert [h["id"] for h in history] == ids[-2:]


class TestConcurrency:
    def test_concurrent_insert_no_lock_and_no_data_loss(self, db_path):
        db.init_db(db_path)
        p1 = multiprocessing.Process(target=_mp_insert_worker, args=(db_path, "procA", 50))
        p2 = multiprocessing.Process(target=_mp_insert_worker, args=(db_path, "procB", 50))
        p1.start()
        p2.start()
        p1.join(timeout=60)
        p2.join(timeout=60)
        assert p1.exitcode == 0, "프로세스A가 'database is locked' 등 예외 없이 종료해야 한다"
        assert p2.exitcode == 0, "프로세스B가 'database is locked' 등 예외 없이 종료해야 한다"
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        assert count == 100

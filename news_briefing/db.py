"""
news_briefing/db.py — SQLite 스키마 및 CRUD

동시성 정책(플랜 §3, 1급 설계): runner.py·telegram_bot.py는 별도 launchd
프로세스 2개, dashboard/app.py(수동 구동)까지 news_briefing.db에 동시 접근한다.
모든 커넥션은 PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000을 설정하고,
공개 함수는 호출당 커넥션을 열고 닫는다(장기 보유 커넥션 없음 → 크로스프로세스
쓰기 경합은 WAL + busy_timeout 재시도로 흡수, 예외 없이 대기).

시간 문자열(sent_at/voted_at/scored_at 등)은 항상 호출자가 넘긴다 — 이 모듈
내부에서 datetime.now()를 고정 사용하지 않는다(테스트 가능성 확보).
"""
import sqlite3
from contextlib import closing

from news_briefing import constants


def _connect(db_path=None):
    """
    sqlite3 커넥션을 생성하고 WAL 모드·busy_timeout(5000ms)을 설정해 반환한다.
    db_path가 None이면 constants.DB_PATH(고정 절대경로)를 사용한다.
    row_factory는 sqlite3.Row로 설정해 컬럼명으로 접근 가능하게 한다.
    """
    path = db_path if db_path is not None else constants.DB_PATH
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _row_to_dict(row):
    return dict(row) if row is not None else None


def _rows_to_dicts(rows):
    return [dict(row) for row in rows]


def init_db(db_path=None):
    """5개 테이블을 생성한다(CREATE TABLE IF NOT EXISTS — 재호출해도 안전)."""
    with closing(_connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    domain TEXT,
                    source_lang TEXT,
                    title TEXT,
                    body TEXT,
                    published_at TEXT,
                    fetched_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS briefings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT CHECK(kind IN ('morning', 'evening')),
                    sent_at TEXT,
                    body_html TEXT,
                    fact_score REAL,
                    grounding_score REAL,
                    regen_count INTEGER DEFAULT 0,
                    warn_flags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS briefing_citations (
                    briefing_id INTEGER,
                    sentence_idx INTEGER,
                    article_id INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    briefing_id INTEGER,
                    market TEXT CHECK(market IN ('KOSPI', 'KOSDAQ')),
                    direction TEXT CHECK(direction IN ('up', 'flat', 'down')),
                    rationale TEXT,
                    actual_pct REAL,
                    verdict TEXT DEFAULT 'pending' CHECK(verdict IN ('hit', 'miss', 'pending')),
                    scored_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    briefing_id INTEGER,
                    vote TEXT CHECK(vote IN ('up', 'down')),
                    voted_at TEXT
                )
                """
            )


def insert_articles(articles, db_path=None):
    """
    기사 목록(dict: url, domain, source_lang, title, body, published_at, fetched_at)을
    삽입한다. url이 이미 존재하면 새로 만들지 않고 기존 id를 재사용한다.
    반환값은 입력 순서에 대응하는 id 목록.

    행 하나당 트랜잭션을 커밋해 동시 쓰기 시 잠금 보유 시간을 최소화한다
    (WAL + busy_timeout 5000ms와 조합해 크로스프로세스 경합을 흡수).
    """
    ids = []
    with closing(_connect(db_path)) as conn:
        for article in articles:
            url = article.get("url")
            with conn:
                row = conn.execute(
                    "SELECT id FROM articles WHERE url = ?", (url,)
                ).fetchone()
                if row is not None:
                    ids.append(row["id"])
                    continue
                try:
                    cur = conn.execute(
                        """
                        INSERT INTO articles
                            (url, domain, source_lang, title, body, published_at, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            url,
                            article.get("domain"),
                            article.get("source_lang"),
                            article.get("title"),
                            article.get("body"),
                            article.get("published_at"),
                            article.get("fetched_at"),
                        ),
                    )
                    ids.append(cur.lastrowid)
                except sqlite3.IntegrityError:
                    # 동시 삽입 경합으로 다른 프로세스가 먼저 같은 URL을 넣은 경우 재조회
                    row = conn.execute(
                        "SELECT id FROM articles WHERE url = ?", (url,)
                    ).fetchone()
                    ids.append(row["id"] if row is not None else None)
    return ids


def insert_briefing(kind, body_html=None, db_path=None):
    """briefings 행을 만들고 id를 반환한다. sent_at·점수류는 update_briefing_sent에서 채운다."""
    with closing(_connect(db_path)) as conn:
        with conn:
            cur = conn.execute(
                "INSERT INTO briefings (kind, body_html) VALUES (?, ?)",
                (kind, body_html),
            )
            return cur.lastrowid


def update_briefing_sent(briefing_id, sent_at, fact_score, grounding_score, regen_count,
                          warn_flags, db_path=None):
    """발송 완료 후 sent_at·fact_score·grounding_score·regen_count·warn_flags를 갱신한다."""
    with closing(_connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                UPDATE briefings
                SET sent_at = ?, fact_score = ?, grounding_score = ?, regen_count = ?, warn_flags = ?
                WHERE id = ?
                """,
                (sent_at, fact_score, grounding_score, regen_count, warn_flags, briefing_id),
            )


def insert_citations(briefing_id, citations, db_path=None):
    """briefing_citations에 (sentence_idx, article_id) 목록을 일괄 삽입한다."""
    with closing(_connect(db_path)) as conn:
        with conn:
            conn.executemany(
                """
                INSERT INTO briefing_citations (briefing_id, sentence_idx, article_id)
                VALUES (?, ?, ?)
                """,
                [(briefing_id, sentence_idx, article_id) for sentence_idx, article_id in citations],
            )


def insert_forecast(briefing_id, market, direction, rationale, db_path=None):
    """forecasts 행을 만든다(verdict 기본값 'pending'). id를 반환한다."""
    with closing(_connect(db_path)) as conn:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO forecasts (briefing_id, market, direction, rationale)
                VALUES (?, ?, ?, ?)
                """,
                (briefing_id, market, direction, rationale),
            )
            return cur.lastrowid


def get_pending_forecasts(db_path=None):
    """verdict='pending'인 forecasts 행을 id 오름차순으로 반환한다(dict 목록)."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM forecasts WHERE verdict = 'pending' ORDER BY id ASC"
        ).fetchall()
    return _rows_to_dicts(rows)


def update_forecast_verdict(forecast_id, actual_pct, verdict, scored_at, db_path=None):
    """채점 결과(actual_pct·verdict·scored_at)로 forecasts 행을 갱신한다."""
    with closing(_connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                UPDATE forecasts
                SET actual_pct = ?, verdict = ?, scored_at = ?
                WHERE id = ?
                """,
                (actual_pct, verdict, scored_at, forecast_id),
            )


def record_feedback(briefing_id, vote, voted_at, db_path=None):
    """피드백(👍/👎) 한 건을 기록하고 새 행의 id를 반환한다."""
    with closing(_connect(db_path)) as conn:
        with conn:
            cur = conn.execute(
                "INSERT INTO feedback (briefing_id, vote, voted_at) VALUES (?, ?, ?)",
                (briefing_id, vote, voted_at),
            )
            return cur.lastrowid


def get_latest_briefing(kind=None, db_path=None):
    """
    가장 최근 briefings 행을 dict로 반환한다. kind가 주어지면 해당 종류로 필터링.
    행이 없으면 None.
    """
    with closing(_connect(db_path)) as conn:
        if kind is None:
            row = conn.execute(
                "SELECT * FROM briefings ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM briefings WHERE kind = ? ORDER BY id DESC LIMIT 1",
                (kind,),
            ).fetchone()
    return _row_to_dict(row)


def get_score_history(limit=90, db_path=None):
    """최근 briefings 최대 limit건을 시간순(오름차순)으로 반환한다."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM briefings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = _rows_to_dicts(rows)
    result.reverse()
    return result


def get_forecast_history(limit=90, db_path=None):
    """최근 forecasts 최대 limit건을 시간순(오름차순)으로 반환한다."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM forecasts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = _rows_to_dicts(rows)
    result.reverse()
    return result


def get_hit_rate(db_path=None):
    """
    채점 완료(hit/miss)된 forecasts 기준 적중률(hit / (hit+miss))을 반환한다.
    채점된 건이 하나도 없으면 None.
    """
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT verdict, COUNT(*) AS cnt FROM forecasts "
            "WHERE verdict IN ('hit', 'miss') GROUP BY verdict"
        ).fetchall()
    counts = {row["verdict"]: row["cnt"] for row in rows}
    hits = counts.get("hit", 0)
    misses = counts.get("miss", 0)
    total = hits + misses
    if total == 0:
        return None
    return hits / total


def get_feedback_ratio(db_path=None):
    """
    피드백 중 👍 비율(up / (up+down))을 반환한다. 피드백이 하나도 없으면 None.
    """
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT vote, COUNT(*) AS cnt FROM feedback GROUP BY vote"
        ).fetchall()
    counts = {row["vote"]: row["cnt"] for row in rows}
    up = counts.get("up", 0)
    down = counts.get("down", 0)
    total = up + down
    if total == 0:
        return None
    return up / total

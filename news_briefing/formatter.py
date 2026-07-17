"""
news_briefing/formatter.py — 텔레그램 HTML 브리핑 조립
(플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5 10단계)

writer.py가 만든 BriefingDraft를 텔레그램에 보낼 HTML 문자열로 조립한다.
- format_briefing_html: 섹션별 문장 + 인용 링크 + 신뢰 점수 배지 + 전망을 HTML로 조립 (html.escape 적용)
- split_html_by_length: 4096자(텔레그램 한도) 단위로 분할하되 <a>...</a> 앵커 태그를 절단하지 않는다
- build_feedback_keyboard_payload: 👍👎 인라인 키보드 payload
"""
import html
import re
from typing import List, Optional

from news_briefing.constants import make_callback_data
from news_briefing.writer import SECTIONS, MARKET_DATA_TAG

_SECTION_TITLES = {
    "nasdaq": "🇺🇸 나스닥 시황",
    "cause": "📊 등락 원인",
    "kr_impact": "🇰🇷 한국 증시 영향",
    "holdings": "💼 보유종목 뉴스",
}

_DIRECTION_EMOJI = {
    "up": "📈",
    "flat": "➡️",
    "down": "📉",
}

# <a ...>...</a> 앵커 전체를 하나의 원자로 매칭 (분할 시 절단 금지 대상)
_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.DOTALL)


def _render_sentence(sentence: dict, articles_by_id: dict) -> str:
    """문장 텍스트를 escape하고 article_ids를 실제 기사 링크로 치환해 붙인다.
    market_data 태그·매핑 없는 기사 id는 링크 없이 건너뛴다(크래시하지 않음)."""
    text = html.escape(sentence.get("text", ""))
    article_ids = sentence.get("article_ids") or []

    if article_ids == [MARKET_DATA_TAG]:
        return text

    links = []
    for aid in article_ids:
        if aid == MARKET_DATA_TAG:
            continue
        article = articles_by_id.get(aid)
        if article is None:
            continue
        url = html.escape(article.get("url", ""))
        links.append('<a href="{}">[{}]</a>'.format(url, aid))

    if links:
        return text + " " + " ".join(links)
    return text


def format_briefing_html(
    draft: dict,
    briefing_id: int,
    articles_by_id: dict,
    fact_score: float,
    grounding_score: float,
    warn_idxs: Optional[List[int]] = None,
) -> str:
    """BriefingDraft를 텔레그램 발송용 HTML 문자열로 조립한다(4096자 분할 전 단계).
    briefing_id는 향후 확장(예: 딥링크) 대비 시그니처에 포함하되 현재 본문 조립에는 쓰지 않는다."""
    warn_set = set(warn_idxs or [])
    sentences = draft.get("sentences", [])
    forecasts = draft.get("forecasts", [])

    by_section = {}
    for s in sentences:
        by_section.setdefault(s.get("section"), []).append(s)

    lines = [
        "🔎 사실정확성 {:.0%} · 근거연결 {:.0%}".format(fact_score, grounding_score),
        "",
    ]

    for section in SECTIONS:
        section_sentences = by_section.get(section)
        if not section_sentences:
            continue
        lines.append(_SECTION_TITLES.get(section, section))
        for s in section_sentences:
            line = _render_sentence(s, articles_by_id)
            if s.get("idx") in warn_set:
                line = "⚠️ " + line
            lines.append(line)
        lines.append("")

    lines.append("📅 전망")
    for f in forecasts:
        emoji = _DIRECTION_EMOJI.get(f.get("direction"), "")
        market = html.escape(str(f.get("market", "")))
        rationale = html.escape(str(f.get("rationale", "")))
        lines.append("{} {}: {}".format(emoji, market, rationale))

    return "\n".join(lines)


def _tokenize(html_text: str) -> List[str]:
    """html_text를 원자 단위 리스트로 쪼갠다. <a>...</a> 앵커는 통째로 하나의 원자,
    그 외 구간은 문자 하나하나가 원자(경계를 자유롭게 잡을 수 있음)."""
    atoms = []
    pos = 0
    for m in _ANCHOR_RE.finditer(html_text):
        if m.start() > pos:
            atoms.extend(list(html_text[pos:m.start()]))
        atoms.append(m.group(0))
        pos = m.end()
    if pos < len(html_text):
        atoms.extend(list(html_text[pos:]))
    return atoms


def split_html_by_length(html_text: str, max_len: int = 4096) -> List[str]:
    """HTML 문자열을 max_len 이하 조각으로 분할하되 <a>...</a> 앵커 태그는
    조각 경계에서 절단하지 않는다(원자 단위로 취급, 항상 한 조각 안에 온전히 포함).
    원자 하나(주로 매우 긴 앵커)가 그 자체로 max_len보다 크면 그 원자만 단독
    조각으로 배치한다 — 이 경우에 한해 그 조각은 예외적으로 max_len을 넘을 수 있다
    (앵커를 잘라서 억지로 맞추는 것보다 안전).
    """
    if html_text == "":
        return [""]

    atoms = _tokenize(html_text)

    chunks = []
    current = ""
    for atom in atoms:
        if current and len(current) + len(atom) > max_len:
            chunks.append(current)
            current = ""
        current += atom
    if current:
        chunks.append(current)

    return chunks


def build_feedback_keyboard_payload(briefing_id: int) -> List[List[dict]]:
    """텔레그램 InlineKeyboardMarkup 구조에 대응하는 👍👎 피드백 버튼 payload를 만든다."""
    return [[
        {"text": "👍", "callback_data": make_callback_data("up", briefing_id)},
        {"text": "👎", "callback_data": make_callback_data("down", briefing_id)},
    ]]

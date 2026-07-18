"""
charts.py - plotly 인터랙티브 그래프 모음

- 누적수익 곡선 : 라인차트 + 0 기준선
- 에이전트/전략/트리거 비교 : 그룹 막대차트(양수 초록 / 음수 빨강)
- 색상 : 수익 초록(#26a69a) / 손실 빨강(#ef5350)
"""

import plotly.graph_objects as go

GREEN = "#26a69a"   # 수익(양수)
RED = "#ef5350"     # 손실(음수)
NEUTRAL = "#5c6bc0"  # 부호 없는 크기 지표(승률·거래수 등 — 이익/손실 색과 절대 겹치지 않게 분리)
_LAYOUT = dict(
    template="plotly_white",
    height=360,
    margin=dict(l=10, r=10, t=46, b=10),
    bargap=0.35,       # 막대 사이 간격 — 없으면 막대가 두껍고 뭉툭해 보임
    bargroupgap=0.15,  # 그룹 막대 안에서 계열끼리의 간격
)


def equity_curve(x, y, title: str, y_title: str = "누적수익률 (%)"):
    """
    누적수익(또는 실현손익) 곡선. 0선 기준선 표시.
    선 색은 마지막 값의 부호를 따름(양수=초록/음수=빨강) — 값의 상태를 색으로도 즉시 읽히게.
    """
    y = list(y)
    color = RED if (y and y[-1] < 0) else GREEN
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x), y=y,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor=_hex_alpha(color, 0.12),
        name=y_title,
        hovertemplate="%{x}<br>" + y_title + ": %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, xaxis_title="시간", yaxis_title=y_title, **_LAYOUT)
    return fig


def _hex_alpha(hex_color: str, alpha: float) -> str:
    """#rrggbb → rgba(r,g,b,alpha) 문자열 변환(면적 채우기용 옅은 색)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def bar_compare(labels, values, title: str, y_title: str, value_fmt: str = ".2f"):
    """
    그룹 막대차트. 값 부호에 따라 색상 자동(양수 초록 / 음수 빨강).
    labels : x축 항목, values : 막대 높이
    """
    labels = list(labels)
    values = [float(v) if v is not None else 0.0 for v in values]
    colors = [GREEN if v >= 0 else RED for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[format(v, value_fmt) for v in values],
        textposition="auto",
    ))
    fig.update_layout(title=title, xaxis_title="", yaxis_title=y_title, **_LAYOUT)
    return fig


def line_series(x, series: dict, title: str, y_title: str):
    """
    여러 계열의 시계열 라인차트(예: fact_score/grounding_score 추이).
    series : {계열명: [값,...]} — x 와 길이 동일
    """
    fig = go.Figure()
    palette = [GREEN, "#5c6bc0", "#ffa726", RED]
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=list(x), y=list(vals),
            mode="lines+markers",
            name=name,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(title=title, xaxis_title="시간", yaxis_title=y_title, **_LAYOUT)
    return fig


def grouped_bar(labels, series: dict, title: str, y_title: str):
    """
    여러 지표를 묶은 그룹 막대(예: 평균수익률 vs 승률).
    series : {계열명: [값,...]} — labels 와 길이 동일
    색은 계열 구분용(범주형)이라 GREEN/RED(다른 차트의 손익 부호 색)와
    헷갈리지 않도록 의도적으로 배제한 팔레트를 쓴다.
    """
    fig = go.Figure()
    palette = [NEUTRAL, "#ffa726", "#8d6e63", "#26c6da"]
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=list(labels), y=list(vals), name=name,
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(title=title, barmode="group", yaxis_title=y_title, **_LAYOUT)
    return fig

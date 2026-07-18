"""
charts.py - plotly 인터랙티브 그래프 모음

- 누적수익 곡선 : 라인차트 + 0 기준선
- 에이전트/전략/트리거 비교 : 그룹 막대차트(양수 초록 / 음수 빨강)
- 색상 : 수익 초록(#26a69a) / 손실 빨강(#ef5350)
"""

import plotly.graph_objects as go

GREEN = "#26a69a"   # 수익
RED = "#ef5350"     # 손실
_LAYOUT = dict(
    template="plotly_white",
    height=360,
    margin=dict(l=10, r=10, t=46, b=10),
)


def equity_curve(x, y, title: str, y_title: str = "누적수익률 (%)"):
    """누적수익(또는 실현손익) 곡선. 0선 기준선 표시."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y),
        mode="lines+markers",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6),
        name=y_title,
        hovertemplate="%{x}<br>" + y_title + ": %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, xaxis_title="시간", yaxis_title=y_title, **_LAYOUT)
    return fig


def equity_vs_benchmark(strat_x, strat_y, strat_name, bench_x, bench_y, bench_name,
                         title: str, y_title: str = "누적수익률 (%)"):
    """전략 누적수익률(청산 이벤트 기준) vs 시장 벤치마크(일별) 오버레이."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(strat_x), y=list(strat_y),
        mode="lines+markers",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6),
        name=strat_name,
    ))
    fig.add_trace(go.Scatter(
        x=list(bench_x), y=list(bench_y),
        mode="lines",
        line=dict(color="#9e9e9e", width=2, dash="dot"),
        name=bench_name,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, xaxis_title="시간", yaxis_title=y_title, **_LAYOUT)
    return fig


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
    """
    fig = go.Figure()
    palette = [GREEN, "#5c6bc0", "#ffa726", RED]
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=list(labels), y=list(vals), name=name,
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(title=title, barmode="group", yaxis_title=y_title, **_LAYOUT)
    return fig

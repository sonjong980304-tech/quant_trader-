"""
data_fetcher.py - yfinance를 이용한 주가 데이터 수집
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import STOCKS, BACKTEST_PERIOD_YEARS


def fetch_ohlcv(ticker: str, period_years: int = 1) -> pd.DataFrame:
    """
    종목 티커와 기간(년)을 받아 OHLCV 데이터프레임 반환.
    컬럼: Open, High, Low, Close, Volume
    """
    end   = datetime.today()
    start = end - timedelta(days=period_years * 365)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"[{ticker}] 데이터를 가져올 수 없습니다.")

    # 멀티인덱스 컬럼 평탄화 (yfinance 0.2+ 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_all_stocks(period_years: int = 1) -> dict:
    """
    config.STOCKS에 정의된 모든 종목의 OHLCV 데이터를 딕셔너리로 반환.
    반환: {ticker: DataFrame}
    """
    result = {}
    for ticker, name in STOCKS.items():
        try:
            df = fetch_ohlcv(ticker, period_years)
            result[ticker] = df
            print(f"  ✓ {name}({ticker}) {len(df)}일치 데이터 수집 완료")
        except Exception as e:
            print(f"  ✗ {name}({ticker}) 수집 실패: {e}")
    return result


def get_minute_data(ticker: str, interval_min: int = 1) -> pd.DataFrame:
    """
    KIS API를 통해 당일 분봉 데이터 수집.
    반환 컬럼: datetime(index), open, high, low, close, volume
    interval_min: 1 또는 5 (분)

    KIS API 엔드포인트: 주식당일분봉조회
    URL: /uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice
    기존 trader.py의 KIS API 토큰/헤더 방식 그대로 재사용.
    """
    import requests
    from trader import KISTrader
    from config import KIS_BASE_URL

    stock_code = ticker.replace(".KS", "").replace(".KQ", "")

    try:
        trader  = KISTrader()
        token   = trader.get_access_token()
        now_str = datetime.now().strftime("%H%M%S")

        headers = {
            "Content-Type":  "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey":        trader.app_key,
            "appsecret":     trader.app_secret,
            "tr_id":         "FHKST03010200",
        }
        params = {
            "FID_ETC_CLS_CODE":       "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD":         stock_code,
            "FID_INPUT_HOUR_1":       now_str,
            "FID_PW_DATA_INCU_YN":    "Y",
        }

        resp = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            headers=headers, params=params, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"분봉 조회 실패: {data.get('msg1')}")

        output = data.get("output2", [])
        if not output:
            return pd.DataFrame()

        rows = []
        for item in reversed(output):  # 오래된 순서로 정렬
            dt_str = item.get("stck_bsop_date", "") + item.get("stck_cntg_hour", "")
            if len(dt_str) < 14:
                continue
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            rows.append({
                "datetime": dt,
                "open":     int(item.get("stck_oprc", 0)),
                "high":     int(item.get("stck_hgpr", 0)),
                "low":      int(item.get("stck_lwpr", 0)),
                "close":    int(item.get("stck_prpr", 0)),
                "volume":   int(item.get("cntg_vol", 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.set_index("datetime", inplace=True)
        return df

    except Exception as e:
        print(f"  ✗ [{ticker}] 분봉 데이터 조회 실패: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=== 데이터 수집 테스트 ===")
    data = fetch_all_stocks(period_years=1)
    for ticker, df in data.items():
        print(f"{ticker}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}행")

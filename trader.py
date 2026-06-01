"""
trader.py - 한국투자증권 KIS API 연동
OAuth 토큰 발급, 주문, 잔고 조회 등 실제 매매 기능
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from config import (
    KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_BASE_URL, IS_MOCK,
    TAKE_PROFIT_HALF, TAKE_PROFIT_FULL, ORDER_RATIO,
)

logger = logging.getLogger(__name__)

# 토큰 캐시 파일 경로
TOKEN_CACHE_FILE = ".kis_token_cache.json"


class KISTrader:
    """한국투자증권 REST API 클라이언트"""

    def __init__(self):
        self.base_url    = KIS_BASE_URL
        self.app_key     = KIS_APP_KEY
        self.app_secret  = KIS_APP_SECRET
        self.account_no  = KIS_ACCOUNT_NO
        # 계좌번호 파싱: "12345678-01" → cano=12345678, prdt=01 / "3579" → cano=3579, prdt=""
        if "-" in KIS_ACCOUNT_NO:
            parts = KIS_ACCOUNT_NO.split("-")
            self.cano     = parts[0]
            self.acnt_prdt = parts[1]
        else:
            self.cano     = KIS_ACCOUNT_NO
            self.acnt_prdt = ""
        self.access_token = None
        self.token_expire = None
        self._load_token_cache()

    # ─────────────────────────────────────────────
    # 토큰 관리
    # ─────────────────────────────────────────────

    def _load_token_cache(self):
        """저장된 토큰 캐시를 불러와 유효하면 재사용"""
        if not os.path.exists(TOKEN_CACHE_FILE):
            return
        try:
            with open(TOKEN_CACHE_FILE) as f:
                cache = json.load(f)
            expire = datetime.fromisoformat(cache["expire"])
            if datetime.now() < expire - timedelta(minutes=10):
                self.access_token = cache["token"]
                self.token_expire = expire
                logger.info("기존 토큰 재사용 (만료: %s)", expire)
        except Exception as e:
            logger.warning("토큰 캐시 로드 실패: %s", e)

    def _save_token_cache(self):
        """발급된 토큰을 파일에 캐시"""
        try:
            with open(TOKEN_CACHE_FILE, "w") as f:
                json.dump({
                    "token":  self.access_token,
                    "expire": self.token_expire.isoformat(),
                }, f)
        except Exception as e:
            logger.warning("토큰 캐시 저장 실패: %s", e)

    def get_access_token(self) -> str:
        """
        OAuth 2.0 토큰 발급.
        유효한 토큰이 있으면 재사용, 없으면 새로 발급.
        """
        if self.access_token and self.token_expire and datetime.now() < self.token_expire - timedelta(minutes=10):
            return self.access_token

        url  = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey":     self.app_key,
            "appsecret":  self.app_secret,
        }

        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self.access_token = data["access_token"]
        # 만료 시간: 응답의 expires_in(초) 또는 기본 24시간
        expires_in = int(data.get("expires_in", 86400))
        self.token_expire = datetime.now() + timedelta(seconds=expires_in)

        self._save_token_cache()
        logger.info("신규 토큰 발급 완료 (만료: %s)", self.token_expire)
        return self.access_token

    def _headers(self, tr_id: str) -> dict:
        """공통 요청 헤더 생성"""
        return {
            "Content-Type":  "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey":        self.app_key,
            "appsecret":     self.app_secret,
            "tr_id":         tr_id,
            "custtype":      "P",  # 개인
        }

    # ─────────────────────────────────────────────
    # 시세 조회
    # ─────────────────────────────────────────────

    def get_current_price(self, stock_code: str) -> dict:
        """
        주식 현재가 조회.
        stock_code: 6자리 종목코드 (예: "005930")
        반환: {"price": int, "change_rate": float, ...}
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._headers("FHKST01010100")
        params  = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD":         stock_code,
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"현재가 조회 실패: {data.get('msg1')}")

        output = data["output"]
        return {
            "stock_code":  stock_code,
            "price":       int(float(output["stck_prpr"])),   # 현재가
            "open":        int(float(output.get("stck_oprc", 0))),  # 당일 시가
            "high":        int(float(output.get("stck_hgpr", 0))),  # 당일 고가
            "low":         int(float(output.get("stck_lwpr", 0))),  # 당일 저가
            "change":      int(float(output["prdy_vrss"])),   # 전일 대비
            "change_rate": float(output["prdy_ctrt"]),        # 등락률 (%)
            "volume":      int(float(output["acml_vol"])),    # 누적 거래량
        }

    # ─────────────────────────────────────────────
    # 주문
    # ─────────────────────────────────────────────

    def _place_order(self, stock_code: str, qty: int, order_type: str) -> dict:
        """
        시장가 주문 내부 처리.
        order_type: "BUY" 또는 "SELL"
        모의: VTTC0802U(매수) / VTTC0801U(매도)
        실투: TTTC0802U(매수) / TTTC0801U(매도)
        """
        if IS_MOCK:
            tr_id = "VTTC0802U" if order_type == "BUY" else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if order_type == "BUY" else "TTTC0801U"

        url  = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        body = {
            "CANO":       self.cano,  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": self.acnt_prdt,  # 계좌 상품코드
            "PDNO":       stock_code,
            "ORD_DVSN":   "01",   # 01: 시장가
            "ORD_QTY":    str(qty),
            "ORD_UNPR":   "0",    # 시장가는 0
        }

        resp = requests.post(url, headers=self._headers(tr_id), json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"주문 실패: {data.get('msg1')}")

        logger.info("%s 주문 완료: %s %d주", order_type, stock_code, qty)
        return data["output"]

    def buy(self, stock_code: str, qty: int) -> dict:
        """시장가 매수 주문"""
        return self._place_order(stock_code, qty, "BUY")

    def sell(self, stock_code: str, qty: int) -> dict:
        """시장가 매도 주문"""
        return self._place_order(stock_code, qty, "SELL")

    # ─────────────────────────────────────────────
    # 잔고 조회
    # ─────────────────────────────────────────────

    def get_balance(self) -> list:
        """
        주식 잔고 조회.
        반환: [{"stock_code": str, "name": str, "qty": int, "avg_price": int, "eval_profit": int}, ...]
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if IS_MOCK else "TTTC8434R"

        params = {
            "CANO":            self.cano,
            "ACNT_PRDT_CD":    self.acnt_prdt,
            "AFHR_FLPR_YN":    "N",
            "OFL_YN":          "",
            "INQR_DVSN":       "02",
            "UNPR_DVSN":       "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN":       "01",
            "CTX_AREA_FK100":  "",
            "CTX_AREA_NK100":  "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"잔고 조회 실패: {data.get('msg1')}")

        holdings = []
        for item in data.get("output1", []):
            qty = int(float(item.get("hldg_qty", 0)))
            if qty > 0:
                holdings.append({
                    "stock_code":  item["pdno"],
                    "name":        item["prdt_name"],
                    "qty":         qty,
                    "avg_price":   int(float(item.get("pchs_avg_pric", 0))),
                    "eval_profit": int(float(item.get("evlu_pfls_amt", 0))),
                })
        return holdings

    def get_total_eval_amt(self) -> int:
        """
        계좌 총평가금액 조회 (예수금 + 보유주식 평가금액).
        매도 후 미결제 대금도 포함되어 실제 운용 가능 자산에 더 가깝다.
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if IS_MOCK else "TTTC8434R"

        params = {
            "CANO":            self.cano,
            "ACNT_PRDT_CD":    self.acnt_prdt,
            "AFHR_FLPR_YN":    "N",
            "OFL_YN":          "",
            "INQR_DVSN":       "02",
            "UNPR_DVSN":       "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN":       "01",
            "CTX_AREA_FK100":  "",
            "CTX_AREA_NK100":  "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"총평가금액 조회 실패: {data.get('msg1')}")

        output2 = data.get("output2", [{}])
        row = output2[0] if output2 else {}
        return int(float(row.get("tot_evlu_amt", 0)))

    def get_available_cash(self) -> int:
        """주문 가능 현금 조회 (원)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        tr_id = "VTTC8908R" if IS_MOCK else "TTTC8908R"

        params = {
            "CANO":            self.cano,
            "ACNT_PRDT_CD":    self.acnt_prdt,
            "PDNO":            "005930",  # 조회용 임시 종목
            "ORD_UNPR":        "0",
            "ORD_DVSN":        "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN":    "N",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"주문 가능 금액 조회 실패: {data.get('msg1')}")

        return int(float(data["output"].get("ord_psbl_cash", 0)))

    # ─────────────────────────────────────────────
    # 미국 주식 (해외주식, 통합증거금서비스)
    # ─────────────────────────────────────────────

    @staticmethod
    def _us_exchange(symbol: str) -> tuple[str, str]:
        """
        심볼 → (주문용 OVRS_EXCG_CD, 시세용 EXCD) 반환.
        기본 NASDAQ. NYSE/AMEX 종목은 직접 지정 필요.
        """
        _nyse = {"BRK-B", "JPM", "GS", "BAC", "WMT", "XOM", "CVX"}
        _amex = {"SPY"}  # SPY는 NYSE Arca지만 AMEX로도 조회 가능
        sym = symbol.upper().replace("-", ".")
        if sym in _nyse:
            return "NYSE", "NYS"
        if sym in _amex:
            return "AMEX", "AMS"
        return "NASD", "NAS"

    def get_us_current_price(self, symbol: str) -> dict:
        """
        미국 주식 현재가 조회. NASD → NYSE → AMEX 순서로 거래소 자동 감지.
        symbol: Yahoo Finance 심볼 (예: 'QQQ', 'TLT', 'AAPL')
        반환: {"symbol": str, "price": float, "currency": "USD", "_excg_cd": str}
        """
        sym = symbol.upper()
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
        last_err = None
        for excd, excg_cd in [("NAS", "NASD"), ("NYS", "NYSE"), ("AMS", "AMEX")]:
            params = {"AUTH": "", "EXCD": excd, "SYMB": sym}
            try:
                resp = requests.get(url, headers=self._headers("HHDFS00000300"), params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("rt_cd") == "0":
                    output = data["output"]
                    price = float(output.get("last", output.get("stck_prpr", 0)))
                    if price > 0:
                        return {
                            "symbol":      sym,
                            "price":       price,
                            "change_rate": float(output.get("diff", 0)),
                            "currency":    "USD",
                            "_excg_cd":    excg_cd,
                        }
                last_err = data.get("msg1", "가격 0")
            except Exception as e:
                last_err = str(e)
        raise ValueError(f"미국주식 현재가 조회 실패 [{symbol}]: {last_err}")

    def _place_us_order(self, symbol: str, qty: int, order_type: str) -> dict:
        """
        미국 주식 주문 (통합증거금서비스 — KIS가 원화→달러 자동 환전).
        order_type: "BUY" 또는 "SELL"
        모의: VTTT1002U(매수) / VTTT1001U(매도)
        실투: TTTT1002U(매수) / TTTT1006U(매도)
        """
        if IS_MOCK:
            tr_id = "VTTT1002U" if order_type == "BUY" else "VTTT1001U"
        else:
            tr_id = "TTTT1002U" if order_type == "BUY" else "TTTT1006U"

        # 현재가 조회로 거래소도 자동 감지
        price_info   = self.get_us_current_price(symbol)
        ovrs_excg_cd = price_info.get("_excg_cd", "NASD")
        limit_price  = str(round(price_info["price"], 2))

        url  = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        body = {
            "CANO":           self.cano,
            "ACNT_PRDT_CD":   self.acnt_prdt,
            "OVRS_EXCG_CD":   ovrs_excg_cd,
            "PDNO":           symbol.upper(),
            "ORD_DVSN":       "00",          # 지정가
            "ORD_QTY":        str(qty),
            "OVRS_ORD_UNPR":  limit_price,   # 현재가로 지정 → 시장가 효과
            "ORD_SVR_DVSN_CD": "0",
        }

        resp = requests.post(url, headers=self._headers(tr_id), json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"미국주식 {order_type} 주문 실패 [{symbol}]: {data.get('msg1')}")

        logger.info("미국주식 %s 주문 완료: %s %d주 @ $%s", order_type, symbol, qty, limit_price)
        return data.get("output", {})

    def buy_us(self, symbol: str, qty: int) -> dict:
        """미국 주식 시장가(지정가) 매수 — 통합증거금서비스"""
        return self._place_us_order(symbol, qty, "BUY")

    def sell_us(self, symbol: str, qty: int) -> dict:
        """미국 주식 시장가(지정가) 매도 — 통합증거금서비스"""
        return self._place_us_order(symbol, qty, "SELL")

    def get_us_balance(self) -> list:
        """
        미국 주식 잔고 조회.
        반환: [{"symbol": str, "qty": int, "avg_price": float, "eval_profit": float, "currency": "USD"}, ...]
        """
        url   = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        tr_id = "VTTS3012R" if IS_MOCK else "TTTS3012R"

        params = {
            "CANO":          self.cano,
            "ACNT_PRDT_CD":  self.acnt_prdt,
            "OVRS_EXCG_CD":  "NASD",  # 전체 조회 시 NASD로 전체 반환
            "TR_CRCY_CD":    "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise ValueError(f"미국주식 잔고 조회 실패: {data.get('msg1')}")

        holdings = []
        for item in data.get("output1", []):
            qty = int(float(item.get("cblc_qty", 0)))
            if qty > 0:
                holdings.append({
                    "symbol":      item.get("pdno", ""),
                    "name":        item.get("prdt_name", ""),
                    "qty":         qty,
                    "avg_price":   float(item.get("pchs_avg_pric", 0)),
                    "eval_profit": float(item.get("evlu_pfls_amt", 0)),
                    "currency":    "USD",
                })
        return holdings


# ─────────────────────────────────────────────
# 포지션 관리 (단일 딕셔너리, 함수를 통해서만 접근)
# ─────────────────────────────────────────────

positions: dict = {}
# {
#   "티커": {
#       "avg_price":     매수 평균가,
#       "highest_price": 장중 고점 (매 루프마다 갱신),
#       "quantity":      보유 수량,
#       "half_sold":     1차 익절(50%) 완료 여부,
#   }
# }

_trader_instance: "KISTrader | None" = None


def _get_trader() -> "KISTrader":
    """KISTrader 싱글턴 반환"""
    global _trader_instance
    if _trader_instance is None:
        _trader_instance = KISTrader()
    return _trader_instance


def register_position(ticker: str, avg_price: float, quantity: int) -> None:
    """포지션 등록 (신규 매수 완료 후 호출)"""
    positions[ticker] = {
        "avg_price": avg_price,
        "quantity":  quantity,
        "half_sold": False,
    }


def clear_position(ticker: str) -> None:
    """포지션 초기화 (매도 완료 후 호출)"""
    positions.pop(ticker, None)




def check_take_profit(ticker: str, current_price: float) -> "str | None":
    """
    2단계 익절 판단.

    반환값:
      - "half" : +8% 도달, 보유량 50% 매도 (half_sold=False 일 때만)
      - "full" : +15% 도달, 잔량 전량 매도
      - None   : 익절 조건 없음
    """
    if ticker not in positions:
        return None

    avg_price  = positions[ticker]["avg_price"]
    half_sold  = positions[ticker]["half_sold"]
    gain_ratio = (current_price - avg_price) / avg_price

    if gain_ratio >= TAKE_PROFIT_FULL:
        return "full"
    if gain_ratio >= TAKE_PROFIT_HALF and not half_sold:
        return "half"
    return None


def calc_position_size(ticker: str, total_asset: float, current_price: float) -> int:
    """
    1종목당 매수 가능 수량 계산.
    - 이미 보유 중인 종목은 추가 매수 금지 (0 반환)
    - 총 자산의 ORDER_RATIO(40%) 한도로 매수
    - 총 자산 조회 실패(0) 또는 예산 부족 시 0 반환 (1주 강제 매수 금지)
    반환값: 매수 가능 수량 (주)
    """
    if ticker in positions or current_price <= 0 or total_asset <= 0:
        return 0
    budget = total_asset * ORDER_RATIO
    return int(budget / current_price)


# ─────────────────────────────────────────────
# 주문 실행 헬퍼 (KISTrader 래핑)
# ─────────────────────────────────────────────

def execute_buy(stock_code: str, qty: int) -> "dict | None":
    """시장가 매수 실행"""
    if not KIS_APP_KEY:
        logger.info("[시뮬레이션] 매수: %s %d주", stock_code, qty)
        return {"simulated": True}
    try:
        return _get_trader().buy(stock_code, qty)
    except Exception as e:
        logger.error("매수 실행 실패 [%s %d주]: %s", stock_code, qty, e)
        return None


def execute_sell_all(ticker: str) -> "dict | None":
    """보유 전량 시장가 매도 실행"""
    stock_code = ticker.replace(".KS", "").replace(".KQ", "")
    if not KIS_APP_KEY:
        logger.info("[시뮬레이션] 전량 매도: %s", stock_code)
        return {"simulated": True}
    try:
        t       = _get_trader()
        balance = t.get_balance()
        holding = next((b for b in balance if b["stock_code"] == stock_code), None)
        if holding and holding["qty"] > 0:
            return t.sell(stock_code, holding["qty"])
        logger.warning("전량 매도 실패: %s 보유 수량 없음", stock_code)
        return None
    except Exception as e:
        logger.error("전량 매도 실패 [%s]: %s", stock_code, e)
        return None


def execute_sell_half(ticker: str) -> "dict | None":
    """보유량 50% 시장가 매도 실행"""
    stock_code = ticker.replace(".KS", "").replace(".KQ", "")
    if not KIS_APP_KEY:
        logger.info("[시뮬레이션] 50%% 매도: %s", stock_code)
        return {"simulated": True}
    try:
        t       = _get_trader()
        balance = t.get_balance()
        holding = next((b for b in balance if b["stock_code"] == stock_code), None)
        if holding and holding["qty"] > 0:
            qty = max(1, holding["qty"] // 2)
            return t.sell(stock_code, qty)
        logger.warning("분할 매도 실패: %s 보유 수량 없음", stock_code)
        return None
    except Exception as e:
        logger.error("분할 매도 실패 [%s]: %s", stock_code, e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trader = KISTrader()
    print(f"[모의투자: {IS_MOCK}] BASE URL: {KIS_BASE_URL}")
    if KIS_APP_KEY:
        try:
            token = trader.get_access_token()
            print(f"토큰 발급 성공: {token[:20]}...")
        except Exception as e:
            print(f"토큰 발급 실패: {e}")
    else:
        print("KIS_APP_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

"""
tests/conftest.py — pytest 공통 설정

ml.features를 pytest 수집 전에 미리 import해서 sys.modules에 등록.
test_scanner.py의 setdefault mock이 실제 모듈을 덮어쓰는 것을 방지.
"""
import importlib
importlib.import_module("ml.features")

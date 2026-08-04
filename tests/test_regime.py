"""regime 模块单元测试(合成数据)。"""
import numpy as np
import pandas as pd

from strategy.regime import (
    _has_run, enso_regime_by_month, enso_regime_by_year,
    market_regime_by_month, liquidity_regime_by_year,
)


def test_has_run():
    assert _has_run(np.array([1, 1, 1, 0, 1]), 3)
    assert not _has_run(np.array([1, 1, 0, 1, 1]), 3)


def test_enso_regime_by_month():
    oni = pd.Series([0.6, 0.6, -0.6, 0.0],
                    index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]))
    fm = enso_regime_by_month(oni, threshold=0.5)
    assert list(fm.values) == [1, 1, -1, 0]


def test_enso_regime_by_year_episode():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    oni = pd.Series([0.6] * 5 + [0.0] * 7, index=dates)
    ry = enso_regime_by_year(oni, threshold=0.5, min_run=5)
    assert ry.iloc[0] == 1                      # 连续5月超阈 → 厄尔尼诺


def test_market_regime_uptrend_is_bull():
    dates = pd.date_range("2018-01-31", periods=24, freq="ME")
    s = pd.Series(np.linspace(100, 150, 24), index=dates)
    fm = market_regime_by_month(s, ma=6)
    assert (fm == 1).mean() > 0.7               # 单调上涨多数判为牛


def test_liquidity_regime_rising_is_loose():
    # 3 年序列(首年无前置→0, 后两年上行→宽); 至少 2/3 年判宽
    dates = pd.date_range("2018-01-31", periods=36, freq="ME")
    s = pd.Series(np.linspace(8, 12, 36), index=dates)   # M2 同比持续上行
    ry = liquidity_regime_by_year(s, lookback=6, direction="rise")
    assert (ry == 1).mean() >= 2 / 3

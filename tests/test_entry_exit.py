"""确定性进出场(v2 趋势跟踪)单测: 合成数据, 确定性, 无网络/无数据文件。

覆盖: 效率比/RSI/MACD/ATR 纯函数性质 + backtest_trend_follow 在"上涨→崩跌"合成序列上
能进场并用移动止损离场, 且不出现未来函数(入场时间 <= 离场时间)。
"""
import numpy as np
import pandas as pd

from strategy.entry_exit import (
    efficiency_ratio, rsi, macd, atr, backtest_trend_follow, summarize_trades,
)


# ---------- 纯函数 ----------
def test_efficiency_ratio_trend_vs_chop():
    up = pd.Series(np.linspace(100, 200, 60))
    er_up = efficiency_ratio(up, 30).iloc[-1]
    assert er_up > 0.95                       # 单边趋势 ≈ 1
    chop = pd.Series(100 + np.tile([0, 1, 0, -1], 30))
    er_chop = efficiency_ratio(chop, 30).iloc[-1]
    assert er_chop < er_up                    # 震荡 < 趋势


def test_rsi_bounds_and_extremes():
    base = np.linspace(100, 300, 60)
    base[::7] -= 1.0                          # 偶有小回撤, 避免"零亏损"退化(纯单调时 avg_loss=0→NaN)
    up = pd.Series(base)
    r = rsi(up, 14)
    valid = r.dropna()
    assert (valid <= 100).all() and (valid >= 0).all()
    assert valid.iloc[-1] > 70                # 强势上涨 RSI 偏高


def test_macd_shapes():
    close = pd.Series(np.linspace(100, 150, 60))
    dif, dea, hist = macd(close)
    assert len(dif) == len(dea) == len(hist) == 60
    assert np.isfinite(hist.iloc[-1])


def test_atr_positive():
    df = pd.DataFrame({
        "high": [10, 11, 12, 11, 13, 12, 14, 13] * 3,
        "low":  [9, 10, 11, 10, 12, 11, 13, 12] * 3,
        "close": [9.5, 10.5, 11.5, 10.5, 12.5, 11.5, 13.5, 12.5] * 3,
    })
    a = atr(df, 14)
    assert (a.dropna() > 0).all()


# ---------- 合成 60min: 上涨(可进) → 崩跌(移动止损离场) ----------
def _make_min(n_days=130, crash_frac=0.90):
    ts = []
    for d in pd.date_range("2024-01-01", periods=n_days, freq="B"):
        for hh in ("10:30", "11:30", "14:00", "15:00"):
            ts.append(pd.Timestamp(d.date()) + pd.Timedelta(hh + ":00"))
    n = len(ts)
    crash = int(n * crash_frac)
    rng = np.random.default_rng(0)
    # 先单边涨(带小噪声, 留足日线MA60生效后的进场窗口), 后急跌触发移动止损
    rets = list(0.0018 + rng.normal(0, 0.0006, crash)) + [-0.014] * (n - crash)
    close = 100 * np.cumprod([1 + r for r in rets])
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = np.full(n, 1e6)
    return pd.DataFrame({"date": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


_CFG = {"entry_exit": {
    "daily_ma_short": 20, "daily_ma_long": 60,
    "ma_short": 20, "ma_long": 60, "breakout_lookback": 20,
    "vol_mult": 1.5, "rsi_period": 14, "rsi_lo": 50, "rsi_hi": 82,
    "atr_period": 14, "atr_stop_mult": 2.0, "cooldown_bars": 8, "overextend_mult": 0.06,
    "donchian": 20, "er_n": 30, "er_min": 0.25, "atr_trail_mult": 3.0,
}}


def test_trend_follow_enters_and_stops_out():
    m = _make_min()
    res = backtest_trend_follow(m, m, _CFG)          # market 用自身(同为上涨, 闸门通过)
    tr = res["trades"]
    assert len(tr) >= 1, "应在上涨段产生至少一笔进场"
    # 崩跌后应有移动止损离场
    assert (tr["reason"].isin(["trail", "dailyweak", "eod"])).all()
    # 无未来: 每笔入场 <= 离场
    assert (pd.to_datetime(tr["entry_date"]) <= pd.to_datetime(tr["exit_date"])).all()


def test_trend_follow_no_entry_without_trend():
    # 纯震荡: ER 低 + 无方向 -> 不应有 A 级进场(或极少)
    ts = []
    for d in pd.date_range("2024-01-01", periods=80, freq="B"):
        for hh in ("10:30", "11:30", "14:00", "15:00"):
            ts.append(pd.Timestamp(d.date()) + pd.Timedelta(hh + ":00"))
    n = len(ts)
    close = 100 + np.tile([0, 0.5, 0, -0.5], n // 4 + 1)[:n]     # 来回震荡
    open_ = np.concatenate([[100.0], close[:-1]])
    df = pd.DataFrame({"date": ts, "open": open_, "high": np.maximum(open_, close) * 1.001,
                       "low": np.minimum(open_, close) * 0.999, "close": close,
                       "volume": np.full(n, 1e6)})
    res = backtest_trend_follow(df, df, _CFG)
    s = summarize_trades(res["trades"])
    # 震荡市要么不进, 进了也应是少数(抗whipsaw)
    assert s["n"] <= 3

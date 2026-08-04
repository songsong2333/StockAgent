"""ETF 策略核心纯函数单测(合成数据, 确定性, 无网络/无数据文件)。

覆盖: 最大回撤 _mdd、夏普 _sharpe、信号防抖状态机 _debounce、情绪仓位上限 apply_sentiment_cap。
这些是 ETF 策略里最易回归又最难肉眼的逻辑(尤其防抖状态机的未来函数/抖动问题)。
"""
import numpy as np
import pandas as pd

from strategy.etf import _mdd, _sharpe, _debounce, apply_sentiment_cap


# ---------- _mdd ----------
def test_mdd_known_drawdown():
    curve = np.array([1.0, 2.0, 1.5, 3.0])
    # 运行峰值 [1,2,2,3] -> 回撤最小 1.5/2-1 = -0.25
    assert abs(_mdd(curve) - (-0.25)) < 1e-9


def test_mdd_monotonic_up_is_zero():
    assert _mdd(np.array([1.0, 2.0, 3.0])) == 0.0


def test_mdd_short_curve():
    assert _mdd(np.array([1.0])) == 0.0


# ---------- _sharpe ----------
def test_sharpe_flat_curve_zero():
    assert _sharpe(np.array([1.0, 1.0, 1.0])) == 0.0


def test_sharpe_upward_positive():
    curve = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
    assert _sharpe(curve) > 0


# ---------- _debounce: 单日噪声不翻转 ----------
def _const_ma(n, val=100.0):
    idx = pd.RangeIndex(n)
    return pd.Series(val, index=idx)


def test_debounce_single_day_dip_does_not_exit():
    # raw 里单日 0  blip, confirm_days=2 需要连续两日 0 才退出 -> 应保持多头
    raw = pd.Series([1, 1, 1, 0, 1, 1, 1])
    hold = _debounce(raw, confirm_days=2, band_filter=0.0,
                     ma_short_s=_const_ma(7), ma_long_s=_const_ma(7))
    # 第0日 rolling 未满 -> 空仓; 之后进入并保持多头(含 blip 处)
    assert hold.iloc[0] == 0
    assert (hold.iloc[1:] == 1).all()


def test_debounce_enters_only_after_confirm_days():
    raw = pd.Series([1, 1, 1, 1])
    hold = _debounce(raw, confirm_days=2, band_filter=0.0,
                     ma_short_s=_const_ma(4), ma_long_s=_const_ma(4))
    # 需连续2日确认: 第0日不满窗=0, 第1日起=1
    assert hold.tolist() == [0, 1, 1, 1]


def test_debounce_exits_after_consecutive_bear():
    raw = pd.Series([1, 1, 0, 0, 0])
    hold = _debounce(raw, confirm_days=2, band_filter=0.0,
                     ma_short_s=_const_ma(5), ma_long_s=_const_ma(5))
    # 第1日进入多头; 连续两个0后(第3日)退出
    assert hold.iloc[1] == 1
    assert hold.iloc[3] == 0
    assert hold.iloc[4] == 0


def test_debounce_long_ma_not_ready_stays_flat():
    ma_long = pd.Series([np.nan, np.nan, 100.0, 100.0])
    raw = pd.Series([1, 1, 1, 1])
    hold = _debounce(raw, confirm_days=2, band_filter=0.0,
                     ma_short_s=_const_ma(4), ma_long_s=ma_long)
    # 长均线 NaN 处强制空仓
    assert hold.iloc[0] == 0 and hold.iloc[1] == 0


# ---------- apply_sentiment_cap ----------
def _top_df(n):
    return pd.DataFrame({"code": [f"C{i}" for i in range(n)],
                         "score": np.linspace(1, 0, n)})


def test_sentiment_cap_halves_top():
    sig = {"top": _top_df(5)}
    out = apply_sentiment_cap(sig, {"position_cap": 0.5})
    assert len(out["top"]) == 2


def test_sentiment_cap_full_keeps_all():
    sig = {"top": _top_df(5)}
    out = apply_sentiment_cap(sig, {"position_cap": 1.0})
    assert len(out["top"]) == 5


def test_sentiment_cap_zero_flattens():
    sig = {"top": _top_df(5)}
    out = apply_sentiment_cap(sig, {"position_cap": 0.0})
    assert out["top"].empty

"""季节性模块单元测试(合成数据, 确定性, 无网络)。

验证: 窗口检测(Kadane/连阳)、月度统计、bootstrap CI、锚点 event-study。
"""
import numpy as np
import pandas as pd

from strategy.seasonality import (
    _max_subarray, _longest_pos_run, detect_windows, window_distribution,
    seasonality_stats, anchor_event_study, bootstrap_ci,
)


def _monthly_level(returns_by_month, years=10, base=100.0, seed=0, noise=0.03):
    """构造月末水平序列: 每年同月用固定收益模式 + 高斯噪声。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{2024 - years}-01-31", periods=12 * years, freq="ME")
    rets = [returns_by_month.get(d.month, 0.0) + rng.normal(0, noise) for d in dates]
    level = base * np.cumprod([1 + r for r in rets])
    return pd.Series(level, index=dates)


def test_max_subarray_best_window():
    s, e, m = _max_subarray(np.array([0.1, 0.1, -0.5, 0.2, 0.2]))
    assert (s, e) == (3, 5)
    assert abs(m - 0.4) < 1e-9


def test_max_subarray_all_negative():
    s, e, m = _max_subarray(np.array([-0.1, -0.2, -0.1]))
    assert (s, e) == (-1, -1) and np.isnan(m)


def test_longest_pos_run():
    s, e, _ = _longest_pos_run(np.array([0.1, 0.1, -0.1, 0.1]))
    assert (s, e) == (0, 2)


def test_detect_windows_finds_summer_rally():
    # 非行情月明确为负, 使 Kadane 不向前扩展 → 窗口紧贴 7-8 月
    rets = {m: -0.02 for m in range(1, 13)}
    rets.update({7: 0.08, 8: 0.08})
    lvl = _monthly_level(rets, years=10, noise=0.01)
    w = detect_windows(lvl, method="peak", cross_validate=False)
    assert len(w) == 10
    # 大多数年份窗口起点落在 6-8 月
    assert sum(6 <= s <= 8 for s in w["start"]) >= 7


def test_seasonality_stats_july_high():
    lvl = _monthly_level({7: 0.06, 8: 0.06}, years=12)
    st = seasonality_stats(lvl, min_n=5)
    assert st.loc[7, "mean"] > 0.03
    assert st.loc[7, "hit_rate"] >= 0.8
    # 无信号月份均值应近 0
    assert abs(st.loc[3, "mean"]) < 0.03


def test_bootstrap_ci_covers_mean():
    arr = np.array([0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03, 0.02])
    lo, hi = bootstrap_ci(arr, iters=500)
    assert lo < arr.mean() < hi


def test_anchor_event_study_positive_after_event():
    """构造: 每年5月 anchor 跨阈, 之后 2 月 target 注入正收益 → 事件后累计均值>0。"""
    dates = pd.date_range("2015-01-31", periods=12 * 10, freq="ME")
    rng = np.random.default_rng(1)
    anchor = pd.Series(0.0, index=dates)
    target_ret = pd.Series(rng.normal(0, 0.02, len(dates)), index=dates)
    for y in range(2015, 2025):
        m5 = pd.Timestamp(year=y, month=5, day=1) + pd.offsets.MonthEnd(0)
        anchor.loc[m5] = 0.6
        target_ret.loc[m5 + pd.DateOffset(months=1)] = 0.05
        target_ret.loc[m5 + pd.DateOffset(months=2)] = 0.05
    target_level = (1 + target_ret).cumprod() * 100
    res = anchor_event_study(target_level, anchor, threshold=0.5,
                             direction="above", pre=2, post=4)
    assert res["n_events"] >= 8
    post_mean = res["curve"].loc[res["curve"]["offset"] >= 3, "mean"].iloc[-1]
    assert post_mean > 0.03


def test_window_distribution_summary():
    lvl = _monthly_level({6: 0.05, 7: 0.05, 8: 0.05}, years=10)
    dist = window_distribution(lvl, method="peak", cross_validate=False)
    assert dist["n_years"] == 10
    assert dist["magnitude"]["median"] > 0.05
    assert dist["hit_rate"] >= 0.8

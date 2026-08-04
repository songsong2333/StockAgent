"""发现引擎单元测试(合成数据, 确定性)。"""
import numpy as np
import pandas as pd

from strategy.discovery import (
    _pair_lead_lag, lead_lag_matrix, pairwise_corr, cluster_series,
    discover, corr_pvalue,
)


def _mk(n=120, seed=0):
    """A 领先 B 滞后 2 期(强相关); C 独立。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    a = pd.Series(rng.normal(0, 0.05, n), index=dates)
    b = a.shift(2) * 0.8 + pd.Series(rng.normal(0, 0.03, n), index=dates)
    c = pd.Series(rng.normal(0, 0.05, n), index=dates)
    return pd.DataFrame({"A": a, "B": b, "C": c}).fillna(0.0)


def test_corr_pvalue_bounds():
    assert corr_pvalue(0.0, 100) > 0.9
    assert corr_pvalue(0.9, 100) < 0.001


def test_pair_lead_lag_detects_lag():
    df = _mk()
    lag, corr = _pair_lead_lag(df["A"], df["B"], max_lag=4)
    assert lag == 2
    assert corr > 0.5


def test_lead_lag_matrix_sign():
    df = _mk()
    ll = lead_lag_matrix(df, max_lag=4)
    assert ll["lag"].loc["A", "B"] > 0       # A 领先 B
    assert ll["lag"].loc["B", "A"] < 0


def test_cluster_groups_correlated():
    df = _mk()
    cl = cluster_series(df)
    order = cl["order"]
    # A 与 B 强相关 → 聚类排序中相邻
    assert abs(order.index("A") - order.index("B")) <= 1


def test_pairwise_corr_returns_fdr():
    df = _mk()
    pw = pairwise_corr(df)
    assert "q" in pw
    # 对角线自相关 = 1
    for c in df.columns:
        assert abs(pw["corr"].loc[c, c] - 1.0) < 1e-9
    # q 值在 [0,1]
    q = pw["q"].stack().dropna()
    assert q.between(0, 1).all()
    # 注: A/B 是 lead-lag(滞后2期)关系, 同期相关≈0, 不在此断言


def test_discover_finds_leader():
    # A 领先 B 2 期 → discover(target=B) 应检出 A 是 B 的先行指标
    df = _mk()
    meta = {c: {"category": "x", "kind": "price"} for c in df.columns}
    names = {c: c for c in df.columns}
    res = discover("B", df, meta, names, max_lag=4)
    assert "error" not in res
    t = res["table"]
    a_row = t[t["id"] == "A"].iloc[0]
    assert a_row["best_lag"] == 2              # A 领先 B 2 期
    assert bool(a_row["leads_target"]) is True

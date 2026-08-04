"""周期探索 —— regime 标注 + 条件分布。

给每个年/月打标(ENSO/牛熊/流动性/气温), 所有分析可"按 regime 分组看条件分布":
  - 验证"厄尔尼诺年电力窗口从7月延到9-10月"这类断言 → conditional_window_stats
    按 regime 分组对比窗口起止/持续/幅度(带 N + CI)
  - medium_term_signal: 月线 regime 状态(牛/熊/震荡), 与日线 trend_signal 互补

统计诚实: regime 子样本(厄尔尼诺年)在 10-15 年历史里仅 2-4 次 → 条件结论是
"假设生成器"不是定论; 所有条件输出显式标注各 regime 的 N。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from common import setup_logger, load_config
from strategy.seasonality import detect_windows, window_distribution

log = setup_logger("strategy.regime")


# ============== ENSO regime(厄尔尼诺/拉尼娜/中性) ==============
def enso_regime_by_month(oni: pd.Series, threshold: float = 0.5) -> pd.Series:
    """逐月 ENSO 状态: ONI>=thr → 1(El Niño); <=-thr → -1(La Niña); 否则 0(Neutral)。"""
    s = oni.dropna().sort_index()
    flag = pd.Series(0, index=s.index, dtype=int)
    flag[s >= threshold] = 1
    flag[s <= -threshold] = -1
    return flag


def enso_regime_by_year(oni: pd.Series, threshold: float = 0.5,
                        min_run: int = 5) -> pd.Series:
    """逐年 ENSO 状态(需连续 min_run 月超阈才认定 episode, 近似 NOAA 定义)。

    返回 Series(index=年末, 值=1/-1/0)。1=该年有厄尔尼诺 episode, -1=拉尼娜, 0=中性。
    """
    fm = enso_regime_by_month(oni, threshold)
    out = {}
    for year, grp in fm.groupby(fm.index.year):
        vals = grp.values
        # 找连续 min_run 段
        el_nino = _has_run(vals >= 1, min_run)
        la_nina = _has_run(vals <= -1, min_run)
        if el_nino and not la_nina:
            out[year] = 1
        elif la_nina and not el_nino:
            out[year] = -1
        elif el_nino and la_nina:
            out[year] = out.get(year, 1)   # 两者都有 → 取厄尔尼诺(强信号)
        else:
            out[year] = 0
    idx = [pd.Timestamp(year=int(y), month=12, day=31) for y in sorted(out)]
    return pd.Series([out[y] for y in sorted(out)], index=idx)


def _has_run(bool_arr: np.ndarray, min_run: int) -> bool:
    """数组中是否存在连续 True 段长度 >= min_run。"""
    best, cur = 0, 0
    for v in bool_arr:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best >= min_run


# ============== 市场 regime(月线牛熊震荡) ==============
def market_regime_by_month(index_level: pd.Series, ma: int = 12) -> pd.Series:
    """逐月市场状态: 价>MA且MA上行→牛(1); 价<MA且MA下行→熊(-1); 否则震荡(0)。

    用 Donchian(高/低=近 ma 个月) 辅助: 价在上沿附近强化多头判定。
    """
    s = index_level.dropna().sort_index().resample("ME").last()
    if len(s) < ma + 2:
        return pd.Series(dtype=int)
    m = s.rolling(ma).mean()
    m_slope = m.diff() > 0
    upper = s.rolling(ma).max()
    state = pd.Series(0, index=s.index, dtype=int)
    bull = (s > m) & m_slope
    bear = (s < m) & ~m_slope
    state[bull] = 1
    state[bear] = -1
    return state


def market_regime_by_year(index_level: pd.Series, ma: int = 12) -> pd.Series:
    """逐年市场状态: 该年牛/熊/震荡月的占比定调。"""
    fm = market_regime_by_month(index_level, ma=ma)
    if fm.empty:
        return pd.Series(dtype=int)
    out = {}
    for year, grp in fm.groupby(fm.index.year):
        score = grp.mean()
        out[year] = 1 if score > 0.33 else (-1 if score < -0.33 else 0)
    idx = [pd.Timestamp(year=int(y), month=12, day=31) for y in sorted(out)]
    return pd.Series([out[y] for y in sorted(out)], index=idx)


# ============== 流动性 regime ==============
def liquidity_regime_by_year(rate_series: pd.Series, lookback: int = 12,
                             direction: str = "fall") -> pd.Series:
    """逐年流动性状态。rate_series 取月频。

    direction="fall": 利率下行=宽(1), 上行=紧(-1)(适用于 LPR/DR007)。
    direction="rise": 序列上升=宽(适用于 M2同比, 增速走高=宽)。
    判据: 该年均值 vs 前 lookback 月均值。
    """
    s = rate_series.dropna().sort_index().resample("ME").last()
    out = {}
    # rise: 序列上行=宽(+1); fall: 序列下行=宽(+1)
    sign = 1 if direction == "rise" else -1
    for year, grp in s.groupby(s.index.year):
        prev = s.loc[:grp.index[0] - pd.Timedelta(days=1)].tail(lookback)
        if prev.empty:
            out[year] = 0
            continue
        diff = grp.mean() - prev.mean()
        out[year] = int(sign * np.sign(diff))   # 符合 direction 的变动 → 1=宽
    idx = [pd.Timestamp(year=int(y), month=12, day=31) for y in sorted(out)]
    return pd.Series([out[y] for y in sorted(out)], index=idx)


# ============== 条件窗口分布(核心: 验证"厄尔尼诺年窗口漂移") ==============
def conditional_window_stats(level: pd.Series, regime_by_year: pd.Series,
                             method: str = "peak", cross_validate: bool = True) -> dict:
    """按 regime 分组的窗口分布对比。

    regime_by_year: Series(index=年末, 值=离散 regime 标签, 如 1/-1/0)。
    返回 {regime_label: window_distribution(...), ..., "_regimes": {label: n_years}}。
    """
    windows = detect_windows(level, method=method, cross_validate=cross_validate)
    if windows.empty:
        return {"error": "无窗口检出"}
    reg = regime_by_year.copy()
    reg.index = reg.index.year                       # 用年份对齐
    windows["regime"] = windows["year"].map(reg).fillna("未知")
    out = {"_regimes": {}}
    for label, grp in windows.groupby("regime"):
        out[str(label)] = _windows_to_dist(grp)
        out["_regimes"][str(label)] = int(len(grp))
    out["warning"] = ("按 regime 分组; 各组 N 标注在 _regimes。某组 N<5 时其结论仅作假设, "
                      "不可当定论(条件样本常很小)。")
    return out


def _windows_to_dist(w: pd.DataFrame) -> dict:
    """从窗口 DataFrame(子集)算分布摘要(轻量版 window_distribution)。"""
    if w.empty:
        return {"n_years": 0}
    mag = w["magnitude"].values
    from strategy.seasonality import bootstrap_ci
    lo, hi = bootstrap_ci(mag)
    return {
        "n_years": int(len(w)),
        "start": {"median": float(w["start"].median()), "mean": float(w["start"].mean()),
                  "std": float(w["start"].std(ddof=1)) if len(w) > 1 else 0.0},
        "end": {"median": float(w["end"].median()), "mean": float(w["end"].mean()),
                "std": float(w["end"].std(ddof=1)) if len(w) > 1 else 0.0},
        "duration": {"median": float(w["duration"].median())},
        "magnitude": {"median": float(np.median(mag)), "mean": float(mag.mean()),
                      "ci_lo": lo, "ci_hi": hi},
        "windows": w.reset_index(drop=True),
    }


# ============== 月级趋势信号(与日线 trend_signal 互补) ==============
def medium_term_signal(cfg: dict, code: str) -> dict:
    """月线 regime + MA 状态(该不该做这个方向)。

    返回 {code, name, regime(牛/熊/震荡), price, ma3/ma6/ma12, advice}。
    """
    from collector.store import universe_panel
    panel, names, meta = universe_panel(cfg, freq="M")
    if code not in panel.columns:
        return {"code": code, "error": f"{code} 不在面板"}
    s = panel[code].dropna()
    if len(s) < 13:
        return {"code": code, "name": names.get(code, code), "error": "月线样本不足(<13)"}
    ma3 = s.rolling(3).mean().iloc[-1]
    ma6 = s.rolling(6).mean().iloc[-1]
    ma12 = s.rolling(12).mean().iloc[-1]
    px = s.iloc[-1]
    state = market_regime_by_month(panel[code], ma=12)
    regime_label = {1: "牛市", -1: "熊市", 0: "震荡"}.get(int(state.iloc[-1]) if not state.empty else 0, "震荡")
    aligned = px > ma3 > ma6 > ma12
    advice = ("月线多头排列 + 牛市 → 方向许可, 可用日线信号进场" if aligned and regime_label == "牛市"
              else ("熊市 → 方向不许可, 日线信号忽略或不做多" if regime_label == "熊市"
                    else "震荡 → 日线信号谨慎, 控制仓位"))
    return {"code": code, "name": names.get(code, code), "date": s.index[-1].strftime("%Y-%m-%d"),
            "regime": regime_label, "price": round(float(px), 2),
            "ma3": round(float(ma3), 2), "ma6": round(float(ma6), 2), "ma12": round(float(ma12), 2),
            "bull_aligned": bool(aligned), "advice": advice}


if __name__ == "__main__":
    cfg = load_config()
    from collector.store import universe_panel
    panel, names, meta = universe_panel(cfg)
    # 厄尔尼诺 vs 中性年: 电力ETF 窗口是否不同?
    oni = panel["enso_oni"]
    enso_year = enso_regime_by_year(oni, threshold=cfg["cycle"]["regime"]["enso_threshold"])
    print("ENSO 年度分布:", enso_year.value_counts().to_dict())
    cond = conditional_window_stats(panel["159611"], enso_year)
    print("\n电力ETF 窗口(按 ENSO regime 分组):")
    for label in cond:
        if label.startswith("_") or label == "warning":
            continue
        d = cond[label]
        if d.get("n_years", 0) == 0:
            continue
        print(f"  [{label}] N={d['n_years']}  起月中位{d['start']['median']:.0f} 止月中位{d['end']['median']:.0f} "
              f"幅度中位{d['magnitude']['median']:+.1%}")
    print("\n", cond.get("warning", ""))

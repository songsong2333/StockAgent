"""周期探索 —— 事件窗口季节性(替代固定日历月分桶)。

核心理念: 行情窗口是会漂移的(电力"6-8月有行情"不代表每月都收阳, 有的年份6-7月、
有的7-8月, 起止由外因驱动)。所以不按"日历月分桶"看均值, 而是:
  1) detect_windows: 逐年检出真实主升浪窗口(runlength / peak两种方法交叉验证)
  2) window_distribution: 跨年统计窗口起止/持续/幅度分布 → 直接刻画"漂移"
  3) anchor_event_study: 把各年对齐到外部锚点(ENSO跨阈/气温破X度), 看行情是否
     系统性地在锚点后启动 → 检验"夏季温度何时上来决定行情何时启动"
  4) seasonality_stats / matrix: 保留固定月分桶作粗览(带 N + bootstrap CI, 诚实)

统计诚实: 每个输出带样本量 N + bootstrap 置信区间; 窗口方法交叉验证, 不一致标注;
多重检验警告(N年×12月为事后筛选, 单看最高月会高估)。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from common import setup_logger, load_config

log = setup_logger("strategy.seasonality")


# ============== 通用工具 ==============
def _monthly_returns(level: pd.Series) -> pd.Series:
    """月末水平序列 → 月收益(pct_change)。"""
    s = level.dropna().sort_index()
    return s.pct_change(fill_method=None).dropna()


def bootstrap_ci(values, iters: int = 1000, ci: float = 0.95, seed: int = 42):
    """bootstrap 均值的置信区间。values 为 array-like。返回 (lo, hi); 空返回 (nan, nan)。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(iters)
    for i in range(iters):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    alpha = (1 - ci) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def _summary_row(values) -> dict:
    """一组数值的统计摘要。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
                "hit_rate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    lo, hi = bootstrap_ci(arr)
    return {"n": int(n), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "hit_rate": float((arr > 0).mean()), "ci_lo": lo, "ci_hi": hi}


# ============== 固定日历月分桶(粗览, 带诚实区间) ==============
def seasonality_stats(level: pd.Series, min_n: int = 5,
                      iters: int = 1000, ci: float = 0.95) -> pd.DataFrame:
    """每只标的每个日历月(1-12)的收益分布统计。

    返回 DataFrame(index=月份1-12, 列: n/mean/median/std/hit_rate/ci_lo/ci_hi/min/max)。
    n<min_n 的行 mean/hit_rate 仍计算但标注样本不足(UI 应标灰)。
    """
    r = _monthly_returns(level)
    if r.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"month": r.index.month, "ret": r.values})
    rows = []
    for m in range(1, 13):
        vals = df.loc[df["month"] == m, "ret"].values
        s = _summary_row(vals)
        s["month"] = m
        s["sufficient"] = s["n"] >= min_n
        if len(vals):
            s["min"], s["max"] = float(vals.min()), float(vals.max())
        else:
            s["min"], s["max"] = np.nan, np.nan
        rows.append(s)
    out = pd.DataFrame(rows).set_index("month")
    out.attrs["warning"] = (f"12月×N标的为事后筛选; 单看最高月收益会高估(多重检验)。"
                            f"请关注 hit_rate 高且 CI 不跨 0 的月份。共 {r.index.year.nunique()} 年样本。")
    return out


def seasonality_matrix(panel: pd.DataFrame, min_n: int = 5) -> tuple:
    """多标的 × 月份 的均值矩阵 + 胜率矩阵(供 heatmap)。

    返回 (mean_matrix, hit_matrix, n_matrix), 行=id, 列=月份1-12。
    n<min_n 的格子置 NaN(标灰)。
    """
    means, hits, ns = {}, {}, {}
    for sid in panel.columns:
        st = seasonality_stats(panel[sid], min_n=min_n)
        if st.empty:
            continue
        means[sid] = st["mean"]
        hits[sid] = st["hit_rate"]
        ns[sid] = st["n"]
    if not means:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    M = pd.DataFrame(means).T
    H = pd.DataFrame(hits).T
    N = pd.DataFrame(ns).T
    # 样本不足的格子置 NaN(不参与"最强月"排序/着色)
    M = M.where(N >= min_n)
    H = H.where(N >= min_n)
    return M, H, N


# ============== 事件窗口检测(核心: 捕捉漂移) ==============
def _max_subarray(rets: np.ndarray) -> tuple:
    """Kadane: 返回 (start, end_exclusive, max_sum) —— 最大累计收益的连续段。"""
    best_sum, best_s, best_e = -np.inf, 0, 0
    cur_sum, cur_s = 0.0, 0
    for i, r in enumerate(rets):
        if cur_sum + r > r:
            cur_sum += r
        else:
            cur_sum, cur_s = r, i
        if cur_sum > best_sum:
            best_sum, best_s, best_e = cur_sum, cur_s, i + 1
    if best_sum <= 0:           # 全年无正收益窗口
        return (-1, -1, np.nan)
    return (best_s, best_e, float(best_sum))


def _longest_pos_run(rets: np.ndarray) -> tuple:
    """最长连续正月段。返回 (start, end_exclusive, sum)。"""
    best_len, best_s, best_e, best_sum = 0, -1, -1, np.nan
    cur_s, cur_len, cur_sum = 0, 0, 0.0
    for i, r in enumerate(rets):
        if r > 0:
            if cur_len == 0:
                cur_s, cur_sum = i, 0.0
            cur_len += 1
            cur_sum += r
            if cur_len > best_len:
                best_len, best_s, best_e, best_sum = cur_len, cur_s, i + 1, cur_sum
        else:
            cur_len = 0
    if best_len == 0:
        return (-1, -1, np.nan)
    return (best_s, best_e, float(best_sum))


def detect_windows(level: pd.Series, method: str = "peak",
                  cross_validate: bool = True) -> pd.DataFrame:
    """逐年检测真实行情窗口。

    method: "peak"(最大累计收益连续段, Kadane) / "runlength"(最长连阳段)。
    cross_validate=True 时两种都跑, 起止月份差异>1 的年份标 inconsistent=True。
    返回 DataFrame: year, start(月份1-12), end, duration(月数), magnitude, method, inconsistent。
    """
    r = _monthly_returns(level)
    if r.empty:
        return pd.DataFrame()
    rows = []
    for year, grp in r.groupby(r.index.year):
        rets = grp.values
        dates = grp.index
        if len(rets) < 2:
            continue
        s_p, e_p, m_p = _max_subarray(rets)
        s_r, e_r, m_r = _longest_pos_run(rets)
        if method == "runlength":
            s, e, mag = s_r, e_r, m_r
        else:                       # 默认 peak
            s, e, mag = s_p, e_p, m_p
        if s < 0:
            continue                # 全年无正窗口
        start_m = dates[s].month
        end_m = dates[e - 1].month
        # 交叉验证一致性(两种方法起止月份差异)
        inconsist = False
        if cross_validate and s_r >= 0 and s_p >= 0:
            inconsist = (abs(dates[s_r].month - dates[s_p].month) > 1
                         or abs(dates[e_r - 1].month - dates[e_p - 1].month) > 1)
        rows.append({"year": int(year), "start": int(start_m), "end": int(end_m),
                     "duration": int(e - s), "magnitude": mag,
                     "method": method, "inconsistent": bool(inconsist)})
    return pd.DataFrame(rows)


def window_distribution(level: pd.Series, method: str = "peak",
                        cross_validate: bool = True, min_n: int = 5) -> dict:
    """跨年统计窗口分布(刻画"漂移")。

    返回 {start: {mean,median,std,min,max}, end:..., duration:..., magnitude: {...,ci},
         hit_rate, n_years, n_consistent, warning}。
    """
    w = detect_windows(level, method=method, cross_validate=cross_validate)
    if w.empty:
        return {"n_years": 0, "warning": "无正收益窗口检出(该标的历年无连续上涨段)"}
    out = {}
    for col in ("start", "end", "duration"):
        out[col] = {"mean": float(w[col].mean()), "median": float(w[col].median()),
                    "std": float(w[col].std(ddof=1)) if len(w) > 1 else 0.0,
                    "min": int(w[col].min()), "max": int(w[col].max())}
    mag = w["magnitude"].values
    lo, hi = bootstrap_ci(mag)
    out["magnitude"] = {"mean": float(mag.mean()), "median": float(np.median(mag)),
                        "ci_lo": lo, "ci_hi": hi}
    out["hit_rate"] = float((mag > 0.05).mean())   # 窗口幅度>5%算"有行情"的年份占比
    out["n_years"] = int(len(w))
    out["n_consistent"] = int((~w["inconsistent"]).sum())
    out["windows"] = w
    out["warning"] = (f"基于 {len(w)} 年样本; start/end std刻画窗口漂移幅度。"
                      f"交叉验证一致 {out['n_consistent']}/{len(w)} 年。")
    return out


# ============== 锚点 event-study(杀手功能) ==============
def anchor_event_study(target_level: pd.Series, anchor_level: pd.Series,
                       threshold: float = 0.5, direction: str = "above",
                       pre: int = 6, post: int = 6) -> dict:
    """把各年对齐到锚点事件(如 ONI 首跨 +0.5), 叠加 target 累计收益曲线 ± 分位带。

    检验"锚点事件 → 行情是否系统性在之后启动"。target/anchor 均为月末水平序列(对齐月频)。
    threshold + direction 定义锚点事件: direction="above" 取首月 anchor>=threshold,
    "below" 取 anchor<=threshold。返回 {curve: DataFrame(offset, mean, lo, hi, n_events)}。
    """
    # 月收益
    tr = _monthly_returns(target_level)
    ar = anchor_level.dropna() if anchor_level is not None else None
    if ar is None or tr.empty:
        return {"n_events": 0, "curve": pd.DataFrame()}
    ar = ar.reindex(tr.index).ffill()        # 对齐到 target 月份
    # 逐年找锚点事件月
    event_dates = []
    for year, grp in ar.groupby(ar.index.year):
        if direction == "above":
            hit = grp[grp >= threshold]
        else:
            hit = grp[grp <= threshold]
        if not hit.empty:
            event_dates.append(hit.index[0])    # 该年首个事件月
    if not event_dates:
        return {"n_events": 0, "curve": pd.DataFrame(),
                "warning": f"未找到 {direction} {threshold} 的锚点事件年"}

    # 对每个事件月, 抽 [t-pre, t+post] 窗口的累计收益(相对事件月归零)
    offsets = np.arange(-pre, post + 1)
    curves = []
    for ed in event_dates:
        idx_pos = tr.index.get_loc(ed) if ed in tr.index else None
        if idx_pos is None:
            continue
        lo_i, hi_i = idx_pos - pre, idx_pos + post
        if lo_i < 0 or hi_i >= len(tr):
            continue
        window = tr.iloc[lo_i:hi_i + 1].values
        # 累计收益: 从 t-pre 开始 cumprod, 再相对事件月(idx=pre)归零
        cum = np.cumprod(1.0 + window) - 1.0
        base = cum[pre]                        # 事件月处的累计值
        rel = cum - base                       # 相对事件月: 之前为运行, 之后为反应
        curves.append(rel)
    if not curves:
        return {"n_events": 0, "curve": pd.DataFrame()}
    C = np.vstack(curves)
    curve = pd.DataFrame({
        "offset": offsets,
        "mean": C.mean(axis=0),
        "lo": np.quantile(C, 0.25, axis=0),
        "hi": np.quantile(C, 0.75, axis=0),
    })
    return {"curve": curve, "n_events": len(curves),
            "event_dates": [pd.Timestamp(d).strftime("%Y-%m") for d in event_dates],
            "warning": (f"{len(curves)} 个事件年; 曲线=事件月归零后的累计收益均值+四分位带。"
                        f"事件后均值显著>0 提示行情系统性跟在锚点后启动(相关非因果)。")}


# ============== Markdown 输出 ==============
def stats_to_markdown(stats: pd.DataFrame, name: str = "") -> str:
    if stats is None or stats.empty:
        return f"### {name} 季节性: 无数据\n"
    L = [f"### 📅 {name} 月度季节性(固定分桶, 粗览)\n"]
    warn = stats.attrs.get("warning", "")
    if warn:
        L.append(f"> ⚠️ {warn}\n")
    L.append("| 月份 | N | 均值收益 | 中位 | 胜率 | 95%CI | 样本足 |")
    L.append("|---|---|---|---|---|---|---|")
    for m, r in stats.iterrows():
        flag = "✅" if r["sufficient"] else "⚠️不足"
        ci = f"[{r['ci_lo']:+.1%},{r['ci_hi']:+.1%}]" if pd.notna(r["ci_lo"]) else "—"
        L.append(f"| {m} | {r['n']} | {r['mean']:+.1%} | {r['median']:+.1%} | "
                 f"{r['hit_rate']:.0%} | {ci} | {flag} |")
    return "\n".join(L)


def window_to_markdown(dist: dict, name: str = "") -> str:
    if not dist or dist.get("n_years", 0) == 0:
        return f"### {name} 事件窗口: {dist.get('warning', '无')}\n"
    s, e, d, mg = dist["start"], dist["end"], dist["duration"], dist["magnitude"]
    L = [f"### 🌊 {name} 行情窗口分布(逐年检出, 刻画漂移)\n"]
    L.append(f"> {dist.get('warning', '')}\n")
    L.append(f"- **起月**: 中位 {s['median']:.0f} (均值{s['mean']:.1f}±{s['std']:.1f}, 范围 {s['min']}-{s['max']})")
    L.append(f"- **止月**: 中位 {e['median']:.0f} (均值{e['mean']:.1f}±{e['std']:.1f}, 范围 {e['min']}-{e['max']})")
    L.append(f"- **持续**: 中位 {d['median']:.0f} 个月")
    L.append(f"- **幅度**: 中位 {mg['median']:+.1%} (均值{mg['mean']:+.1%}, 95%CI[{mg['ci_lo']:+.1%},{mg['ci_hi']:+.1%}])")
    L.append(f"- **有行情年占比** (窗口>5%): {dist['hit_rate']:.0%} ({dist['n_years']}年样本)")
    return "\n".join(L)


def seasonality_signal(cfg: dict, code: str, method: Optional[str] = None) -> dict:
    """单标的事件窗口季节性信号(UI 用)。返回 {code, name, stats, windows, distribution}。"""
    cc = cfg.get("cycle", {})
    method = method or cc.get("window", {}).get("method", "peak")
    cross = cc.get("window", {}).get("cross_validate", True)
    min_n = cc.get("min_n", 5)
    from collector.store import universe_panel
    panel, names, meta = universe_panel(cfg)
    if code not in panel.columns:
        return {"code": code, "error": f"{code} 不在已加载序列中(未采集?)"}
    level = panel[code]
    return {
        "code": code, "name": names.get(code, code), "method": method,
        "stats": seasonality_stats(level, min_n=min_n),
        "distribution": window_distribution(level, method=method, cross_validate=cross, min_n=min_n),
    }


if __name__ == "__main__":
    cfg = load_config()
    sig = seasonality_signal(cfg, "159611")   # 电力ETF
    if "error" in sig:
        print(sig["error"], "(先跑 init_cycle.py)")
    else:
        print(stats_to_markdown(sig["stats"], sig["name"]))
        print("\n" + window_to_markdown(sig["distribution"], sig["name"]))

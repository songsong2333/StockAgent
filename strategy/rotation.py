"""板块/个股轮动策略 (散户版)。

思路: 在关注池(如17只龙头)中, 定期(每周)持有"动量最强+结构健康"的 N 只,
弱势的换出。内置两层防护:
  1. 大盘择时: 沪深300 < MA60 时空仓(规避系统性大跌, 利用散户可空仓优势)。
  2. 追高控制: 只选"上升趋势中、回踩均线、未过度乖离"的票(不裸追突破)。

评分 = 中期动量(60日) / 波动率(风险调整), 趋势+回踩过滤后取 topN。
附带历史回测: 2024-07~今, 周频轮动 vs 一直持有基准。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from common import setup_logger, load_config, to_qlib_code, PROJECT_ROOT

log = setup_logger("strategy.rotation")


def _universe_file(cfg: dict) -> Path:
    """轮动候选宇宙文件。默认板块龙头池, 可在 config 里覆盖。"""
    uni = cfg.get("rotation", {}).get("universe", "config/sector_leaders.yaml")
    p = Path(uni)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """加载候选池收盘价面板 + 收益面板。列=股票, 行=日期。"""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    with open(_universe_file(cfg), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 兼容 watchlist(stocks) 和 sector_leaders(leaders) 两种格式
    items = data.get("leaders") or data.get("stocks") or []
    name_map = {to_qlib_code(s["code"]): s["name"] for s in items}
    sector_map = {to_qlib_code(s["code"]): s.get("sector", "") for s in items}

    closes = {}
    for qc in name_map:
        p = raw_dir / f"{qc}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].astype(float)
        closes[qc] = s
    close = pd.DataFrame(closes).sort_index()
    # 对齐基准
    bench_p = raw_dir / "SH000300.parquet"
    if bench_p.exists():
        b = pd.read_parquet(bench_p)
        b["date"] = pd.to_datetime(b["date"])
        close["__bench__"] = b.set_index("date")["close"].astype(float)
    close = close.dropna(how="all")
    ret = close.pct_change()
    return close, ret, name_map, sector_map


def _top_by_sector(scored: pd.DataFrame, topn: int, max_per_sector: int) -> pd.DataFrame:
    """从已打分的候选里选 topn, 限制每个板块最多 max_per_sector 只(真板块轮动)。"""
    picked, used = [], {}
    for _, r in scored.iterrows():
        sec = r["sector"]
        if used.get(sec, 0) >= max_per_sector:
            continue
        picked.append(r)
        used[sec] = used.get(sec, 0) + 1
        if len(picked) >= topn:
            break
    return pd.DataFrame(picked)


def rotation_signal(cfg: dict, topn: int = 3, mom_win: int = 60,
                    bench_ma: int = 60, bias_max: float = 0.20, bias_min: float = -0.05,
                    max_per_sector: int = 1) -> dict:
    """生成最新一轮动信号。max_per_sector: 每板块最多选几只(默认1=纯板块轮动)。"""
    close, ret, names, sectors = load_panel(cfg)
    stocks = [c for c in close.columns if c != "__bench__"]
    last = close.index[-1]

    # 大盘择时
    bench = close["__bench__"]
    bench_ma_val = bench.rolling(bench_ma).mean()
    market_bull = bench.iloc[-1] > bench_ma_val.iloc[-1]

    rows = []
    for c in stocks:
        s = close[c].dropna()
        if len(s) < mom_win + 5:
            continue
        ma20 = s.rolling(20).mean().iloc[-1]
        ma60 = s.rolling(60).mean().iloc[-1]
        mom60 = s.iloc[-1] / s.iloc[- mom_win] - 1
        vol60 = ret[c].rolling(60).std().iloc[-1]
        bias = s.iloc[-1] / ma20 - 1
        uptrend = ma20 > ma60
        score = mom60 / vol60 if vol60 and not np.isnan(vol60) and vol60 > 0 else np.nan
        # 过滤: 上升趋势 + 回踩均线区间(不裸追高)
        ok = uptrend and (bias_min <= bias <= bias_max)
        rows.append({
            "code": c[2:], "name": names.get(c, c), "sector": sectors.get(c, ""),
            "close": round(s.iloc[-1], 2),
            "mom60": mom60, "vol60": vol60, "score": score,
            "bias_MA20": bias, "uptrend": uptrend, "pass_filter": ok,
        })
    df = pd.DataFrame(rows)
    # 候选 = 通过过滤的, 按风险调整动量排序
    cand = df[df["pass_filter"]].sort_values("score", ascending=False)
    top = _top_by_sector(cand, topn, max_per_sector).copy()

    return {
        "date": last.strftime("%Y-%m-%d"), "market_bull": market_bull,
        "bench_close": round(bench.iloc[-1], 1),
        "bench_ma": round(bench_ma_val.iloc[-1], 1),
        "candidates": df, "top": top,
    }


def backtest_rotation(cfg: dict, topn: int = 3, freq: int = 5, mom_win: int = 60,
                      bench_ma: int = 60, cost_rate: float = 0.0015,
                      start: str = "2024-10-01", max_per_sector: int = 1,
                      exposure: float = 1.0, bias_max: float = 0.20) -> dict:
    """历史回测: 周频轮动 + 大盘择时 + 追高过滤, vs 基准。

    exposure: 仓位比例(0~1), 用于回撤预算控制(其余持现金)。回撤≈exposure×满仓回撤。
    bias_max: 乖离过滤上限(越低越不追高)。
    """
    close, ret, names, sectors = load_panel(cfg)
    close = close[close.index >= start]
    ret = ret[ret.index >= start]
    stocks = [c for c in close.columns if c != "__bench__"]
    bench = close["__bench__"]
    ma20_all = close[stocks].rolling(20).mean()
    ma60_all = close[stocks].rolling(60).mean()
    bench_ma = bench.rolling(bench_ma).mean()
    mom = close[stocks] / close[stocks].shift(mom_win) - 1
    vol = ret[stocks].rolling(60).std()
    score = mom / vol

    dates = close.index.tolist()
    holdings: List[str] = []      # qlib代码
    eq = 1.0
    bench_eq = 1.0
    curve, bench_curve = [], []

    for i in range(1, len(dates)):
        # 每 freq 天调仓
        if i % freq == 0:
            market_bull = bench.iloc[i - 1] > bench_ma.iloc[i - 1]
            if market_bull:
                bias = close[stocks].iloc[i - 1] / ma20_all.iloc[i - 1] - 1
                uptrend = ma20_all.iloc[i - 1] > ma60_all.iloc[i - 1]
                ok = (uptrend & (bias >= -0.05) & (bias <= bias_max))
                sc = score.iloc[i - 1].where(ok).dropna().sort_values(ascending=False)
                # 每板块最多 max_per_sector 只
                new_hold, used = [], {}
                for code in sc.index:
                    sec = sectors.get(code, "")
                    if used.get(sec, 0) >= max_per_sector:
                        continue
                    new_hold.append(code)
                    used[sec] = used.get(sec, 0) + 1
                    if len(new_hold) >= topn:
                        break
            else:
                new_hold = []  # 空仓
            # 换手成本 (按仓位比例缩放)
            old_set, new_set = set(holdings), set(new_hold)
            turnover = len(new_set - old_set) + len(old_set - new_set)
            eq *= (1 - cost_rate * turnover / max(topn, 1) * exposure)
            holdings = new_hold

        # 当日收益 (用昨天的持仓, 在今天实现); 仓位 exposure, 其余现金(0收益)
        if holdings:
            r = ret[stocks].iloc[i][holdings]
            daily = r.mean() if len(r) else 0.0
        else:
            daily = 0.0
        if np.isnan(daily):
            daily = 0.0
        eq *= (1 + exposure * daily)
        bench_eq *= (1 + (bench.iloc[i] / bench.iloc[i - 1] - 1))
        curve.append(eq)
        bench_curve.append(bench_eq)

    curve = np.array(curve)
    bench_curve = np.array(bench_curve)
    def mdd(x):
        return float((x / np.maximum.accumulate(x) - 1).min())
    def sharpe(x):
        r = np.diff(np.log(x))
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    return {
        "cum_ret": float(curve[-1] - 1), "bench_cum": float(bench_curve[-1] - 1),
        "max_dd": mdd(curve), "bench_dd": mdd(bench_curve),
        "sharpe": sharpe(curve), "bench_sharpe": sharpe(bench_curve),
        "n_days": len(curve), "curve": curve, "bench_curve": bench_curve,
        "dates": dates[1:],
        # 满仓(exposure=1)日收益序列, 供 App 即时缩仓重算
        "daily_ret": np.diff(curve) / curve[:-1] if exposure == 1.0 else None,
        "bench_daily": np.diff(bench_curve) / bench_curve[:-1],
    }


def metrics_from_returns(daily_ret, exposure: float = 1.0, periods: int = 252):
    """由满仓日收益序列 + 仓位比例, 即时算 (累计, 回撤, 夏普, 净值)。"""
    daily = np.asarray(daily_ret, dtype=float)
    eq = np.cumprod(1 + exposure * daily)
    cum = float(eq[-1] - 1)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    r = np.diff(np.log(eq))
    sharpe = float(r.mean() / r.std() * np.sqrt(periods)) if len(r) > 1 and r.std() > 0 else 0.0
    return {"cum_ret": cum, "max_dd": mdd, "sharpe": sharpe, "curve": eq}


def signal_to_markdown(sig: dict) -> str:
    L = [f"# 🔁 轮动信号 ({sig['date']})\n"]
    L.append(f"**大盘择时**: 沪深300 {sig['bench_close']} {'>' if sig['market_bull'] else '<'} MA60({sig['bench_ma']}) "
             f"→ {'✅ 多头市场, 可持仓' if sig['market_bull'] else '⚠️ 空头市场, 建议空仓观望'}\n")
    top = sig["top"]
    if not sig["market_bull"]:
        L.append("> 大盘在MA60之下, 系统性风险高, 本期**空仓**。等大盘重回MA60再入场。\n")
    elif top.empty:
        L.append("> 无股票同时满足「上升趋势+回踩均线+不过度乖离」, 本期**观望**(说明都在追高或下跌)。\n")
    else:
        L.append("## 本期应持仓 (风险调整动量 topN, 已过趋势+回踩+乖离过滤)")
        L.append("| 代码 | 名称 | 板块 | 现价 | 60日动量 | 乖离MA20 | 动量/波动分 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in top.iterrows():
            L.append(f"| {r['code']} | {r['name']} | {r['sector']} | {r['close']} | {r['mom60']:+.1%} | "
                     f"{r['bias_MA20']:+.1%} | {r['score']:.2f} |")
        L.append(f"\n**等权持有 {len(top)} 只**。下周再评估, 跌出 topN 或跌破MA60 则换出。")
    # 被过滤的(供参考)
    df = sig["candidates"]
    dropped = df[~df["pass_filter"]].sort_values("mom60", ascending=False).head(6)
    if not dropped.empty:
        L.append("\n## 动量强但被过滤的 (追高/下跌中, 不入选)")
        L.append("| 代码 | 名称 | 板块 | 60日动量 | 乖离MA20 | 原因 |")
        L.append("|---|---|---|---|---|---|")
        for _, r in dropped.iterrows():
            why = "乖离过大(追高)" if r["bias_MA20"] > 0.20 else ("破位(下跌)" if not r["uptrend"] else "回踩不够")
            L.append(f"| {r['code']} | {r['name']} | {r['sector']} | {r['mom60']:+.1%} | {r['bias_MA20']:+.1%} | {why} |")
    return "\n".join(L)


def backtest_to_markdown(bt: dict) -> str:
    L = [f"## 轮动策略回测 (2024-10 至今, 周频, top3 + 大盘择时 + 追高过滤)\n"]
    L.append("| 指标 | 轮动策略 | 一直持有沪深300 |")
    L.append("|---|---|---|")
    L.append(f"| 累计收益 | {bt['cum_ret']:+.1%} | {bt['bench_cum']:+.1%} |")
    L.append(f"| 最大回撤 | {bt['max_dd']:.1%} | {bt['bench_dd']:.1%} |")
    L.append(f"| 夏普(年化) | {bt['sharpe']:.2f} | {bt['bench_sharpe']:.2f} |")
    dd_diff = bt["max_dd"] - bt["bench_dd"]   # 负=策略回撤更大(恶化), 正=更小(改善)
    dd_word = f"回撤改善 {-dd_diff:.1%}" if dd_diff > 0 else f"回撤恶化 {-dd_diff:.1%}(策略波动更大)"
    L.append(f"\n超额收益: **{bt['cum_ret']-bt['bench_cum']:+.1%}** | {dd_word}")
    L.append("\n说明: 大盘跌破MA60时空仓避险; 个股只在'上升趋势+回踩均线+乖离<20%'时入选, 避免裸追高。")
    return "\n".join(L)


# ===================== 回撤预算 (风险控制) =====================
def rotation_risk_control(cfg: dict, max_dd_target: float, **kw) -> dict:
    """按目标最大回撤自动定仓位比例。

    满仓回测得 natural_dd; exposure = min(1, target/natural_dd);
    再以该 exposure 回测, 回撤≈目标, 收益按比例缩小。
    """
    bt_full = backtest_rotation(cfg, exposure=1.0, **kw)
    natural_dd = abs(bt_full["max_dd"])
    scale = min(1.0, max_dd_target / natural_dd) if natural_dd > 0 else 1.0
    bt_scaled = backtest_rotation(cfg, exposure=scale, **kw)
    return {
        "target": max_dd_target, "natural_dd": bt_full["max_dd"],
        "exposure": round(scale, 3),
        "full": bt_full, "scaled": bt_scaled,
    }


def risk_control_to_markdown(rc: dict, topn: int) -> str:
    f, s = rc["full"], rc["scaled"]
    L = [f"## 回撤预算控制 (目标最大回撤 {rc['target']:.0%}, 持仓 top{topn})\n"]
    L.append(f"满仓回撤 {rc['natural_dd']:.1%} → 推荐**仓位 {rc['exposure']:.0%}**(其余持现金)\n")
    L.append("| 指标 | 满仓轮动 | 按目标缩仓 | 一直持有沪深300 |")
    L.append("|---|---|---|---|")
    L.append(f"| 累计收益 | {f['cum_ret']:+.1%} | {s['cum_ret']:+.1%} | {s['bench_cum']:+.1%} |")
    L.append(f"| 最大回撤 | {f['max_dd']:.1%} | **{s['max_dd']:.1%}** | {s['bench_dd']:.1%} |")
    L.append(f"| 夏普 | {f['sharpe']:.2f} | {s['sharpe']:.2f} | {s['bench_sharpe']:.2f} |")
    L.append(f"\n**含义**: 你设定最多接受 {rc['target']:.0%} 回撤 → 系统只用 {rc['exposure']:.0%} 资金做轮动, "
             f"{1-rc['exposure']:.0%} 持现金/货基。换来回撤从 {f['max_dd']:.1%} 压到 {s['max_dd']:.1%}, "
             f"代价是收益从 {f['cum_ret']:+.1%} 降到 {s['cum_ret']:+.1%}。")
    L.append("\n> 注: 回撤预算靠仓位缩放实现, 可预测但简化; 真实极端行情(熔断/跌停)回撤可能略超目标, "
             "建议目标留缓冲(如想要15%实设12%)。")
    return "\n".join(L)


def risk_control_tradeoff(cfg: dict, targets=(0.10, 0.15, 0.20, 0.25), topn=5, **kw) -> str:
    """不同回撤目标下的收益/夏普权衡表。"""
    L = ["## 回撤目标权衡表 (持仓 top%d)\n" % topn]
    L.append("| 目标回撤 | 仓位比例 | 实际回撤 | 累计收益 | 夏普 |")
    L.append("|---|---|---|---|---|")
    for t in targets:
        rc = rotation_risk_control(cfg, t, topn=topn, **kw)
        s = rc["scaled"]
        L.append(f"| {t:.0%} | {rc['exposure']:.0%} | {s['max_dd']:.1%} | {s['cum_ret']:+.1%} | {s['sharpe']:.2f} |")
    L.append("\n**读法**: 回撤目标越低 → 仓位越低 → 收益越低、夏普未必更差(波动也降)。按你的风险承受力选一档。")
    return "\n".join(L)


if __name__ == "__main__":
    cfg = load_config()
    topn = cfg.get("rotation", {}).get("topn", 5)
    target = cfg.get("rotation", {}).get("max_dd_target", 0.15)
    print(signal_to_markdown(rotation_signal(cfg, topn=topn, max_per_sector=1)))
    print("\n" + risk_control_to_markdown(rotation_risk_control(cfg, target, topn=topn), topn))
    print("\n" + risk_control_tradeoff(cfg, topn=topn))

"""单股未来一周操作单。

现有多因子模型是"截面排名器"(预测相对收益排名, 非单股价格预测), 因此单股周计划
= 模型态度(打分/排名) + 技术面(均线/支撑阻力/ATR) + 量价启动特征, 综合成一周操作建议:
  入场区间 / 止损 / 目标位(T1/T2) / 仓位(基于ATR的风险定价) / 情境应对。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import setup_logger, to_qlib_code, load_config, PROJECT_ROOT

log = setup_logger("live.stock_plan")


def load_stock(qlib_code: str, cfg: dict) -> pd.DataFrame:
    p = Path(cfg["paths"]["raw_dir"]) / f"{qlib_code}.parquet"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def model_attitude(qlib_code: str, cfg: dict) -> dict:
    """获取该股在最新日的模型打分与排名。"""
    from backtest.qlib_runner import train_and_predict
    _, model, pred = train_and_predict(cfg)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("score")
    if isinstance(pred.index, pd.MultiIndex):
        lvl = "datetime" if "datetime" in pred.index.names else 0
        last_date = pred.index.get_level_values(lvl).max()
        latest = pred.xs(last_date, level=lvl).dropna().sort_values(ascending=False)
    else:
        latest = pred.dropna().sort_values(ascending=False)
    rank = list(latest.index).index(qlib_code) + 1 if qlib_code in latest.index else None
    return {
        "score": float(latest.get(qlib_code, np.nan)),
        "rank": rank, "total": len(latest),
        "pct": rank / len(latest) if rank else None,
    }


def swing_levels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 lookback: int = 30, band: float = 0.25, cluster_pct: float = 0.015) -> dict:
    """找近期支撑/阻力: 近 lookback 日的盘中高/低点, 距现价 ±band 内, 相近价位合并(cluster_pct内合一)。

    返回:
      resistance: 升序(最近阻力在前); resistance_touches: 各阻力触及次数(越大越关键)
      support: 降序(最近支撑在前); support_touches: 各支撑触及次数
    """
    h = high[-lookback:]; l = low[-lookback:]; c = float(close[-1])
    lo_bound, hi_bound = c * (1 - band), c * (1 + band)
    r_raw = sorted(round(float(x), 2) for x in h.tolist() if c < x <= hi_bound)
    s_raw = sorted(round(float(x), 2) for x in l.tolist() if lo_bound <= x < c)

    def cluster(levels):
        """合并相差 ≤cluster_pct 的相邻价位为一簇, 返回 [(价位均值, 触及次数)]。"""
        if not levels:
            return []
        groups = [[levels[0]]]
        for x in levels[1:]:
            if x <= groups[-1][-1] * (1 + cluster_pct):
                groups[-1].append(x)
            else:
                groups.append([x])
        return [(round(sum(g) / len(g), 2), len(g)) for g in groups]

    res = cluster(r_raw)                 # 升序: 最低阻力(最近)在前
    sup = list(reversed(cluster(s_raw)))  # 降序: 最高支撑(最近)在前
    return {
        "resistance": [p for p, _ in res[:3]],
        "resistance_touches": [t for _, t in res[:3]],
        "support": [p for p, _ in sup[:3]],
        "support_touches": [t for _, t in sup[:3]],
    }


def price_levels(code: str, cfg: dict, capital: float = 1_000_000,
                 risk_pct: float = 0.02, live_px: float = None) -> dict:
    """单股技术价位(不含模型, 快): 现价/加仓区间/减仓位/止损/止盈 + 趋势。

    live_px: 传入实时最新价时, 覆盖末根K, 使所有计算(均线/支撑/压力/止损)以实时现价为基准。
    供持仓页批量算每只持仓的加仓/减仓价位, 不调模型训练。
    """
    qc = to_qlib_code(code)
    df = load_stock(qc, cfg)
    close = df["close"].to_numpy(float).copy()
    high = df["high"].to_numpy(float).copy()
    low = df["low"].to_numpy(float).copy()
    vol = df["volume"].to_numpy(float) if "volume" in df else None
    # 用实时价覆盖最后一根 → 所有后续计算以实时现价为基准, 压力位必在现价之上
    if live_px:
        lp = float(live_px)
        close[-1] = lp
        high[-1] = max(high[-1], lp)
        low[-1] = min(low[-1], lp)
    last_close = float(close[-1])
    last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

    def ma(n):
        return float(np.mean(close[-n:])) if len(close) >= n else np.nan
    ma5, ma10, ma20, ma60, ma120, ma250 = ma(5), ma(10), ma(20), ma(60), ma(120), ma(250)

    tr = np.maximum.reduce([
        high[-15:-1] - low[-15:-1],
        np.abs(high[-15:-1] - close[-16:-2]),
        np.abs(low[-15:-1] - close[-16:-2]),
    ])
    atr = float(np.mean(tr))

    hi250, lo250 = float(np.max(close[-250:])), float(np.min(close[-250:]))
    pos_high = last_close / hi250 - 1
    range_pos = (last_close - lo250) / (hi250 - lo250)
    m5 = close[-1] / close[-6] - 1 if len(close) > 6 else np.nan
    m20 = close[-1] / close[-21] - 1 if len(close) > 21 else np.nan
    m60 = close[-1] / close[-61] - 1 if len(close) > 61 else np.nan
    v20 = float(np.mean(vol[-20:])) if vol is not None else np.nan
    v60 = float(np.mean(vol[-60:])) if vol is not None else np.nan
    vol_ratio = v20 / v60 - 1 if v60 else np.nan

    lv = swing_levels(high, low, close, lookback=30, band=0.25)
    # 兜底: 压力位必在现价之上, 支撑位必在现价之下(突破无近端高点时给百分比位)
    if not lv["resistance"]:
        lv["resistance"] = [round(last_close * 1.05, 2), round(last_close * 1.12, 2)]
        lv["resistance_touches"] = [0, 0]
    if not lv["support"]:
        lv["support"] = [round(last_close * 0.95, 2)]
        lv["support_touches"] = [0]
    extended = (last_close > ma20 * 1.10) or (m5 > 0.12)
    if extended:
        trend = f"短线急涨偏离 (5日{m5:+.0%}, 高于MA20 {(last_close/ma20-1):+.0%})"
    elif ma5 > ma10 > ma20 > ma60:
        trend = "多头排列"
    elif ma5 < ma10 < ma20 < ma60:
        trend = "空头排列"
    else:
        trend = "震荡/纠缠"

    near_support = lv["support"][0] if lv["support"] else ma20
    if extended:
        entry_low = min(ma10, ma20) * 0.99
        entry_high = max(ma10, ma20) * 1.01
        entry_hint = "急涨偏离, 忌追高! 等回踩MA10/MA20再低吸"
    else:
        entry_low = min(near_support, ma20) * 0.995
        entry_high = last_close * 1.01
        entry_hint = "回踩支撑/MA20低吸"
    stop = entry_low - 1.5 * atr
    resists = lv["resistance"] if lv["resistance"] else [last_close * 1.07, last_close * 1.15]
    t1 = resists[0]
    t2 = resists[1] if len(resists) > 1 else resists[0] * 1.10
    if t2 <= t1:
        t2 = t1 * 1.10
    risk_amt = capital * risk_pct
    entry_ref = (entry_low + entry_high) / 2
    per_share_risk = entry_ref - stop
    shares = int(risk_amt / per_share_risk / 100) * 100 if per_share_risk > 0 else 0

    return {
        "code": code, "qc": qc, "last_date": last_date, "last_close": last_close,
        "ma": {"MA5": ma5, "MA10": ma10, "MA20": ma20, "MA60": ma60, "MA120": ma120, "MA250": ma250},
        "atr": atr, "hi250": hi250, "lo250": lo250, "pos_high": pos_high, "range_pos": range_pos,
        "mom": {"m5": m5, "m20": m20, "m60": m60}, "vol_ratio": vol_ratio,
        "support": lv["support"], "resistance": lv["resistance"],
        "resistance_touches": lv.get("resistance_touches", []),
        "support_touches": lv.get("support_touches", []),
        "trend": trend, "extended": extended,
        "entry_low": entry_low, "entry_high": entry_high, "entry_hint": entry_hint,
        "stop": stop, "t1": t1, "t2": t2,
        "shares": shares, "pos_value": shares * entry_ref, "risk_amt": risk_amt, "capital": capital,
    }


def build_week_plan(code: str, cfg: dict, capital: float = 1_000_000,
                    risk_pct: float = 0.02) -> dict:
    """生成单股一周操作计划(技术价位 + 模型态度)。code: 6位代码。"""
    p = price_levels(code, cfg, capital, risk_pct)
    qc = p["qc"]
    try:
        att = model_attitude(qc, cfg)
    except Exception as e:
        log.warning(f"模型打分失败: {e}")
        att = {"score": np.nan, "rank": None, "total": 17, "pct": None}
    if att["rank"] is not None:
        if att["pct"] <= 0.3:
            att_txt = f"看好 (17只中排第{att['rank']})"
        elif att["pct"] <= 0.6:
            att_txt = f"中性偏多 (排第{att['rank']})"
        else:
            att_txt = f"暂不看好 (排第{att['rank']}/17, 打分偏低)"
    else:
        att_txt = "无打分"
    p["attitude"] = att_txt
    p["att"] = att
    return p


def to_markdown(p: dict, week: str) -> str:
    L = [f"# 📅 {p['code']} 一周操作单 ({week})\n"]
    L.append(f"**最新收盘**: {p['last_close']} ({p['last_date']}) | "
             f"距250日高 {p['pos_high']:+.1%} | 区间位置 {p['range_pos']:.0%} | 趋势: {p['trend']}\n")

    L.append("## 模型与技术面")
    L.append(f"- **模型态度**: {p['attitude']}（多因子排名器, 非价格预测）")
    L.append(f"- **均线**: MA5 {p['ma']['MA5']:.2f} / MA10 {p['ma']['MA10']:.2f} / "
             f"MA20 {p['ma']['MA20']:.2f} / MA60 {p['ma']['MA60']:.2f} / MA120 {p['ma']['MA120']:.2f}")
    L.append(f"- **动量**: 5日 {p['mom']['m5']:+.1%} | 20日 {p['mom']['m20']:+.1%} | 60日 {p['mom']['m60']:+.1%}")
    L.append(f"- **量能**: 20/60日量比 {p['vol_ratio']:+.1%} | ATR(14) {p['atr']:.2f}")

    L.append("\n## 关键价位")
    sup = " / ".join(f"{x:.2f}" for x in p["support"][:3]) or "-"
    res = " / ".join(f"{x:.2f}" for x in p["resistance"][:3]) or "-"
    L.append(f"- **支撑位**: {sup}")
    L.append(f"- **阻力位**: {res}")

    L.append("\n## 一周操作建议 (基于总资金 ¥{:,.0f}, 单笔风险 {:.0%})".format(p["capital"], p["risk_amt"]/p["capital"]))
    L.append(f"| 项目 | 价位 | 说明 |")
    L.append(f"|---|---|---|")
    L.append(f"| 入场区间 | {p['entry_low']:.2f} ~ {p['entry_high']:.2f} | {p['entry_hint']} |")
    L.append(f"| 止损价 | {p['stop']:.2f} | 入场下方1.5×ATR, 跌破无条件离场 |")
    L.append(f"| 目标T1 | {p['t1']:.2f} | 第一阻力, 到价减半仓 |")
    L.append(f"| 目标T2 | {p['t2']:.2f} | 第二阻力, 清仓或上移止损保盈利 |")
    L.append(f"| 建议仓位 | {p['shares']:,}股 ≈ ¥{p['pos_value']:,.0f} | 风险定价: 每股风险{(p['entry_low']+p['entry_high'])/2 - p['stop']:.2f}元 |")

    if p["extended"]:
        L.append("\n> ⚠️ **该股短线急涨偏离均线, 追高风险大。原则上等回踩, 不在现价追涨; 已持仓者上移止损保护利润。**")

    L.append("\n## 情境应对")
    L.append(f"- **情境A 突破**: 放量站上阻力 {p['t1']:.2f} → 加仓, 止损上移至 {p['t1']*0.98:.2f}, 看 {p['t2']:.2f}")
    L.append(f"- **情境B 震荡/回踩**: 在支撑 {p['entry_low']:.2f}~阻力 {p['t1']:.2f} 间 → 回踩支撑低吸, 反弹到阻力高抛")
    L.append(f"- **情境C 破位**: 跌破 {p['stop']:.2f} → 止损离场, 不补仓不扛单")

    L.append("\n⚠️ 现有模型是截面排名器(选股用), 单股周计划以技术面为主、模型态度为辅。仅供参考, 不构成投资建议。")
    return "\n".join(L)


if __name__ == "__main__":
    cfg = load_config()
    p = build_week_plan("603629", cfg)
    print(to_markdown(p, "2026-06-23~06-27"))

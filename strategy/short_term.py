"""短线激进模块: 情绪温度计 + 打板连板梯队 + 强势股。

给"想博短线"的人用, 但内置风控与现实提醒(打板是负和游戏, 新手是食物链底端)。
数据: akshare 东财涨停池/炸板/跌停/连板/强势股 (日级, 收盘后取当日, 盘中取实时)。

⚠️ 本模块仅供短线博弈参考, 不构成投资建议。短线打板对新手是高风险负和游戏。
"""
from __future__ import annotations

from typing import Optional

import akshare as ak
import pandas as pd

from common import setup_logger, to_qlib_code
from collector.aux_collector import latest_trade_date

log = setup_logger("strategy.short_term")


def _zt_pool(date: str) -> pd.DataFrame:
    return ak.stock_zt_pool_em(date=date)

def _zbgc(date: str) -> pd.DataFrame:
    return ak.stock_zt_pool_zbgc_em(date=date)

def _dtgc(date: str) -> pd.DataFrame:
    try:
        return ak.stock_zt_pool_dtgc_em(date=date)
    except Exception:
        return pd.DataFrame()

def _strong(date: str) -> pd.DataFrame:
    return ak.stock_zt_pool_strong_em(date=date)


def market_sentiment(date: Optional[str] = None) -> dict:
    """情绪温度计: 涨停/跌停/炸板/连板高度/温度0-100 + 操作建议。"""
    date = date or latest_trade_date().replace("-", "")
    zt = _zt_pool(date)
    zbgc = _zbgc(date)
    dt = _dtgc(date)
    n_zt, n_zbgc, n_dt = len(zt), len(zbgc), len(dt)
    # 连板高度: 找含'连板'的列取最大
    max_streak = 1
    ladder_col = None
    for c in zt.columns:
        if "连板" in str(c):
            ladder_col = c
            max_streak = int(pd.to_numeric(zt[c], errors="coerce").max())
            break
    # 炸板率 = 炸板 / (涨停+炸板)
    zha_rate = n_zbgc / (n_zt + n_zbgc) if (n_zt + n_zbgc) else 0

    # 温度(0-100): 涨停多+连板高+炸板率低 → 高温
    zt_score = min(n_zt / 100, 1.0)            # 100只涨停=满分
    streak_score = min(max_streak / 6, 1.0)    # 6连=满分
    zha_penalty = zha_rate                      # 炸板率越高越扣
    temp = 100 * (0.45 * zt_score + 0.30 * streak_score + 0.25 * (1 - zha_penalty))

    # 操作建议
    if temp >= 75:
        advice = "🔥情绪高潮: 赚钱效应爆棚, 龙头可打但高位票随时核按钮。只做最强龙头, 破板必走。"
        can_play = "可"
    elif temp >= 50:
        advice = "😊情绪偏热: 正常操作, 选有辨识度的连板/首板, 控制仓位。"
        can_play = "可"
    elif temp >= 30:
        advice = "😐情绪一般: 结构性行情, 谨慎打板, 优选低位首板, 别追高连板。"
        can_play = "谨慎"
    else:
        advice = "🧊情绪冰点/退潮: 炸板率高、高度塌, 打板大概率吃面。建议**观望不做**, 等情绪回暖。"
        can_play = "否(别下嘴)"

    return {
        "date": date, "n_zt": n_zt, "n_zbgc": n_zbgc, "n_dt": n_dt,
        "zha_rate": zha_rate, "max_streak": max_streak, "ladder_col": ladder_col,
        "temp": round(temp), "can_play": can_play, "advice": advice,
        "zt_df": zt,
    }


def limit_up_ladder(date: Optional[str] = None) -> pd.DataFrame:
    """连板梯队: 按连板数分组, 每组列股票。"""
    date = date or latest_trade_date().replace("-", "")
    zt = _zt_pool(date)
    col = None
    for c in zt.columns:
        if "连板" in str(c):
            col = c
            break
    if col is None:
        zt = zt.copy()
        zt["连板数"] = 1
        col = "连板数"
    zt = zt.copy()
    zt[col] = pd.to_numeric(zt[col], errors="coerce").fillna(1).astype(int)
    # 每只股票一行: 代码/名称/连板数/涨停时
    name_col = "名称" if "名称" in zt.columns else zt.columns[2]
    code_col = "代码" if "代码" in zt.columns else zt.columns[1]
    out = zt[[code_col, name_col, col]].rename(columns={code_col: "代码", name_col: "名称", col: "连板数"})
    out = out.sort_values(["连板数", "代码"], ascending=[False, True])
    return out


def strong_pool(date: Optional[str] = None, topn: int = 15) -> pd.DataFrame:
    """强势股池: 接近涨停/大涨的, 按成交额排序(含最新价, 供仓位计算)。"""
    date = date or latest_trade_date().replace("-", "")
    s = _strong(date).copy()
    code_col = "代码" if "代码" in s.columns else s.columns[1]
    name_col = "名称" if "名称" in s.columns else s.columns[2]
    amt_col = "成交额" if "成交额" in s.columns else None
    chg_col = next((c for c in s.columns if "涨跌幅" in str(c)), None)
    price_col = next((c for c in s.columns if "最新价" in str(c) or "现价" in str(c)), None)
    keep = [c for c in [code_col, name_col, price_col, chg_col, amt_col] if c]
    out = s[keep].copy()
    out.columns = (["代码", "名称", "现价", "涨跌幅", "成交额"])[: len(keep)]
    if "现价" in out.columns:
        out["现价"] = pd.to_numeric(out["现价"], errors="coerce")
    if amt_col:
        out = out.sort_values("成交额", ascending=False)
    return out.head(topn)


def to_markdown(sent: dict, ladder: pd.DataFrame, strong: pd.DataFrame) -> str:
    L = [f"# 🎰 短线情绪温度计 ({sent['date']})\n"]
    L.append(f"## 情绪温度: **{sent['temp']}/100** → 能否下嘴: **{sent['can_play']}**\n")
    L.append(f"- 涨停 {sent['n_zt']} 只 | 跌停 {sent['n_dt']} 只 | 炸板 {sent['n_zbgc']} 只(炸板率 {sent['zha_rate']:.0%})")
    L.append(f"- 最高连板高度: **{sent['max_streak']}连**")
    L.append(f"\n> {sent['advice']}")
    # 连板梯队
    L.append("\n## 打板连板梯队 (高度龙在顶部)")
    top_ladder = ladder.head(20)
    L.append("| 连板数 | 代码 | 名称 |")
    L.append("|---|---|---|")
    for _, r in top_ladder.iterrows():
        L.append(f"| {int(r['连板数'])}连 | {r['代码']} | {r['名称']} |")
    # 强势股
    L.append(f"\n## 强势股 (成交额前列, 接近涨停)")
    L.append("| 代码 | 名称 | 涨跌幅 |")
    L.append("|---|---|---|")
    for _, r in strong.head(10).iterrows():
        chg = f"{r['涨跌幅']:+.1f}%" if pd.notna(r.get("涨跌幅")) else "-"
        L.append(f"| {r['代码']} | {r['名称']} | {chg} |")
    # 风控红线
    L.append("\n## 🔴 短线风控红线 (不守必死)")
    L.append("- 单只 ≤ 总资金 10%, 短线总仓 ≤ 30%, 其余留中线/现金")
    L.append("- 打板只打有辨识度的龙头/首板, 不碰杂毛; 炸板/不封板 → 次日开盘无条件走")
    L.append("- 亏损 5% 或破当日均价 立即止损, **绝不补仓、绝不扛单**")
    L.append("- 情绪温度 < 30 时**不做**(退潮期打板大概率吃面)")
    L.append("\n⚠️ 短线打板对新手是**高风险负和游戏**(对手是游资+量化+手续费)。仅供参考。")
    return "\n".join(L)


if __name__ == "__main__":
    sent = market_sentiment()
    lad = limit_up_ladder()
    st = strong_pool()
    print(to_markdown(sent, lad, st))

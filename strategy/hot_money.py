"""题材热度 + 龙虎榜游资跟踪模块。

数据(Explore 验证可用):
  - 题材: ak.stock_board_concept_name_ths()  (同花顺, 东财源被限)
  - 龙虎榜个股: ak.stock_lhb_detail_em(start, end)
  - 游资营业部席位: ak.stock_lhb_hyyyb_em(start, end)
  - 机构净买: ak.stock_lhb_jgmmtj_em(start, end)

字段名随 akshare 版本波动, 这里按"包含关键词"做容错。
"""
from __future__ import annotations

import time
from typing import Optional

import akshare as ak
import pandas as pd

from common import setup_logger
from collector.aux_collector import latest_closed_trade_date

log = setup_logger("strategy.hot_money")


def _date_str(date: Optional[str]) -> str:
    return (date or latest_closed_trade_date()).replace("-", "")


def _col(df: pd.DataFrame, *keys: str) -> Optional[str]:
    """按关键词模糊匹配列名, 返回首个命中列。"""
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    for k in keys:
        for c in cols:
            if k in c:
                return c
    return None


def _retry(fn, *a, max_retries=2, sleep=1.0, **k):
    last = None
    for i in range(1, max_retries + 1):
        try:
            df = fn(*a, **k)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            last = e
            time.sleep(sleep * i)
    log.warning(f"{getattr(fn,'__name__','fn')} 失败: {last}")
    return pd.DataFrame()


def concept_heatboard(topn: int = 15) -> pd.DataFrame:
    """题材热度榜: 涨幅 × 上涨家数占比 综合。

    数据源: 东财实时题材(stock_board_concept_spot_em, 数据全) 为主;
            同花顺题材名(stock_board_concept_name_ths, 仅名字) 兜底。
    返回列必含 '板块'(题材名称, 非代码); 有实时数据时含 '热度分'。
    """
    # 1) 东财实时题材(涨跌幅/上涨家数/领涨股/成交额)
    df = _retry(ak.stock_board_concept_spot_em, sleep=1.5, max_retries=3)
    name_c = _col(df, "板块名称", "概念名称", "名称", "name")
    if df is not None and not df.empty and name_c:
        chg_c = _col(df, "涨跌幅", "涨幅")
        up_c = _col(df, "上涨家数")
        dn_c = _col(df, "下跌家数")
        lead_c = _col(df, "领涨股票")
        lead_chg_c = _col(df, "领涨股票-涨跌幅")
        amt_c = _col(df, "成交额")
        out = pd.DataFrame({"板块": df[name_c]})
        if chg_c:
            out["涨跌幅"] = pd.to_numeric(df[chg_c], errors="coerce")
        if up_c and dn_c:
            up = pd.to_numeric(df[up_c], errors="coerce")
            dn = pd.to_numeric(df[dn_c], errors="coerce")
            out["上涨占比"] = (up / (up + dn)).where((up + dn) > 0)
        if lead_c:
            out["领涨股"] = df[lead_c]
        if lead_chg_c:
            out["领涨幅"] = pd.to_numeric(df[lead_chg_c], errors="coerce")
        if amt_c:
            out["成交额"] = pd.to_numeric(df[amt_c], errors="coerce")
        if "涨跌幅" in out:
            ratio = out["上涨占比"] if "上涨占比" in out else 0.5
            out["热度分"] = (out["涨跌幅"].fillna(0) * 0.6 + ratio.fillna(0.5) * 30).round(2)
            out = out.sort_values("热度分", ascending=False)
        return out.head(topn).reset_index(drop=True)

    # 2) 兜底: 同花顺题材名(只有名字, 无实时涨跌)
    df = _retry(ak.stock_board_concept_name_ths, sleep=1.2)
    if df is None or df.empty:
        return pd.DataFrame()
    name_c = _col(df, "name", "板块名称", "名称") or "name"
    out = pd.DataFrame({"板块": df[name_c]})
    out["说明"] = "实时涨跌暂不可用(东财限流), 仅显示题材列表"
    return out.head(topn).reset_index(drop=True)


def dragon_tiger(date: Optional[str] = None, topn: int = 20) -> pd.DataFrame:
    """龙虎榜个股: 净买额 + 上榜原因 + 上榜后表现。"""
    date = _date_str(date)
    df = _retry(ak.stock_lhb_detail_em, start_date=date, end_date=date, sleep=1.2)
    if df.empty:
        return df
    name_c = _col(df, "名称") or df.columns[2]
    code_c = _col(df, "代码") or df.columns[1]
    net_c = _col(df, "龙虎榜净买额", "净买额")
    chg_c = _col(df, "涨跌幅")
    reason_c = _col(df, "上榜原因")
    after1 = _col(df, "上榜后1日")
    after5 = _col(df, "上榜后5日")
    out = pd.DataFrame({"代码": df[code_c], "名称": df[name_c]})
    if chg_c:
        out["当日涨幅"] = pd.to_numeric(df[chg_c], errors="coerce")
    if net_c:
        out["净买额(亿)"] = (pd.to_numeric(df[net_c], errors="coerce") / 1e8).round(2)
    if reason_c:
        out["上榜原因"] = df[reason_c]
    if after1:
        out["上榜后1日"] = pd.to_numeric(df[after1], errors="coerce")
    if after5:
        out["上榜后5日"] = pd.to_numeric(df[after5], errors="coerce")
    if net_c:
        out = out.sort_values("净买额(亿)", ascending=False)
    return out.head(topn).reset_index(drop=True)


def hot_seats(date: Optional[str] = None, topn: int = 10) -> pd.DataFrame:
    """top 游资营业部席位: 按总买卖净额。"""
    date = _date_str(date)
    df = _retry(ak.stock_lhb_hyyyb_em, start_date=date, end_date=date, sleep=1.2)
    if df.empty:
        return df
    name_c = _col(df, "营业部名称", "名称") or df.columns[1]
    net_c = _col(df, "总买卖净额", "净额")
    buy_c = _col(df, "买入总金额")
    sell_c = _col(df, "卖出总金额")
    nbuy_c = _col(df, "买入个股数")
    stocks_c = _col(df, "买入股票")
    out = pd.DataFrame({"营业部": df[name_c]})
    if nbuy_c:
        out["买入股数"] = pd.to_numeric(df[nbuy_c], errors="coerce")
    if buy_c:
        out["买入额(亿)"] = (pd.to_numeric(df[buy_c], errors="coerce") / 1e8).round(2)
    if net_c:
        out["净额(亿)"] = (pd.to_numeric(df[net_c], errors="coerce") / 1e8).round(2)
    if stocks_c:
        out["买入股票"] = df[stocks_c].astype(str).str.slice(0, 40)
    if net_c:
        out = out.sort_values("净额(亿)", ascending=False)
    return out.head(topn).reset_index(drop=True)


def institution_flow(date: Optional[str] = None, topn: int = 15) -> pd.DataFrame:
    """机构净买个股。"""
    date = _date_str(date)
    df = _retry(ak.stock_lhb_jgmmtj_em, start_date=date, end_date=date, sleep=1.2)
    if df.empty:
        return df
    name_c = _col(df, "名称") or df.columns[2]
    code_c = _col(df, "代码") or df.columns[1]
    net_c = _col(df, "机构买入净额", "净额")
    nbuy_c = _col(df, "买方机构数")
    out = pd.DataFrame({"代码": df[code_c], "名称": df[name_c]})
    if nbuy_c:
        out["买方机构数"] = pd.to_numeric(df[nbuy_c], errors="coerce")
    if net_c:
        out["机构净买(亿)"] = (pd.to_numeric(df[net_c], errors="coerce") / 1e8).round(2)
    if net_c:
        out = out.sort_values("机构净买(亿)", ascending=False)
    return out.head(topn).reset_index(drop=True)


def to_markdown(concepts: pd.DataFrame, dt: pd.DataFrame, seats: pd.DataFrame,
                inst: pd.DataFrame, date: str) -> str:
    L = [f"# 🔥 题材热度 + 龙虎榜游资 ({date})\n"]

    L.append("## 题材热度榜")
    if concepts.empty:
        L.append("(同花顺题材接口暂不可用, 稍后重试)")
    else:
        L.append("| 板块 | 涨跌幅 | 上涨占比 | 领涨股 | 热度分 |")
        L.append("|---|---|---|---|---|")
        for _, r in concepts.head(10).iterrows():
            up = f"{r.get('上涨占比', 0):.0%}" if pd.notna(r.get("上涨占比")) else "-"
            L.append(f"| {r['板块']} | {r.get('涨跌幅',0):+.2f}% | {up} | "
                     f"{r.get('领涨股','-')} | {r.get('热度分','-')} |")

    L.append("\n## 龙虎榜个股 (净买额前列)")
    if dt.empty:
        L.append("(今日暂无龙虎榜数据)")
    else:
        L.append("| 代码 | 名称 | 当日涨幅 | 净买额(亿) | 上榜后1日 | 上榜后5日 | 上榜原因 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in dt.head(12).iterrows():
            a1 = f"{r['上榜后1日']:+.1%}" if pd.notna(r.get("上榜后1日")) else "-"
            a5 = f"{r['上榜后5日']:+.1%}" if pd.notna(r.get("上榜后5日")) else "-"
            chg = f"{r.get('当日涨幅',0):+.1f}%" if pd.notna(r.get("当日涨幅")) else "-"
            L.append(f"| {r['代码']} | {r['名称']} | {chg} | {r.get('净买额(亿)','-')} | "
                     f"{a1} | {a5} | {str(r.get('上榜原因','-'))[:20]} |")

    L.append("\n## 游资营业部 (净额前列)")
    if not seats.empty:
        L.append("| 营业部 | 买入股数 | 净额(亿) | 买入股票 |")
        L.append("|---|---|---|---|")
        for _, r in seats.head(8).iterrows():
            L.append(f"| {str(r['营业部'])[:24]} | {r.get('买入股数','-')} | "
                     f"{r.get('净额(亿)','-')} | {r.get('买入股票','-')} |")

    L.append("\n## 机构净买个股")
    if not inst.empty:
        L.append("| 代码 | 名称 | 买方机构数 | 机构净买(亿) |")
        L.append("|---|---|---|---|")
        for _, r in inst.head(8).iterrows():
            L.append(f"| {r['代码']} | {r['名称']} | {r.get('买方机构数','-')} | {r.get('机构净买(亿)','-')} |")

    L.append("\n> 读法: 题材热度看资金主线; 龙虎榜看游资/机构抢哪些票(净买为正=主力进场); "
             "上游资席位跟踪知名游资动向。⚠️ 龙虎榜是滞后信息, 追高风险大, 仅供参考。")
    return "\n".join(L)


if __name__ == "__main__":
    date = latest_closed_trade_date().replace("-", "")
    concepts = concept_heatboard()
    dt = dragon_tiger(date)
    seats = hot_seats(date)
    inst = institution_flow(date)
    print(to_markdown(concepts, dt, seats, inst, date))

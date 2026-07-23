"""资金流 / 题材热度 / 情绪温度计 / 融资余额 采集层 (ETF 策略升级用)。

数据源 (Explore 验证可用):
  - 行业资金流: ak.stock_sector_fund_flow_rank (东财, 中等限流)
  - 题材热度: 复用 strategy.hot_money.concept_heatboard (东财+同花顺兜底)
  - 情绪温度计: 复用 strategy.short_term.market_sentiment (东财涨停池)
  - 融资余额: ak.stock_margin_detail_szse/sse (交易所官方源, 低限流)

设计原则: 所有函数 try/except 包到底, **失败返回空/默认, 永不抛异常**, 上层评分
自动降级为纯动量。双层缓存: 进程内 _cached(ttl) 适配回测/脚本; app 层再叠 @st.cache_data。
"""
from __future__ import annotations

import time
from typing import Optional

import akshare as ak
import pandas as pd

from common import setup_logger
from collector.aux_collector import latest_closed_trade_date, latest_trade_date

log = setup_logger("collector.flow")


def _retry_throttle(fn, *a, max_retries=3, base_sleep=1.0, **k) -> pd.DataFrame:
    """东财源限流退避: Connection/timeout 类 → 长退避(3*2^n); 其他 → 短退避。
    照搬 etf_collector.fetch_etf 的判定。失败返回空 DataFrame。"""
    last = None
    name = getattr(fn, "__name__", "fn")
    for attempt in range(1, max_retries + 1):
        try:
            df = fn(*a, **k)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            last = e
            msg = repr(e).lower()
            throttled = any(x in msg for x in ("connection", "remote", "timeout", "timed out", "aborted", "reset", "json", "decode", "expecting value", "char 0"))
            wait = (3.0 if throttled else base_sleep) * (2 ** (attempt - 1))
            log.debug(f"{name} 第{attempt}次失败({'限流' if throttled else '异常'} {wait:.0f}s): {e}")
            time.sleep(wait)
    log.warning(f"{name} 采集失败: {last}")
    return pd.DataFrame()


def _col(df: pd.DataFrame, *keys: str) -> Optional[str]:
    """按关键词模糊匹配列名(复用 hot_money._col 逻辑, 兼容 akshare 列名变动)。"""
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    for k in keys:
        for c in cols:
            if k in c:
                return c
    return None


# ---- 进程内 TTL 缓存(适配回测/脚本调用, 无 streamlit context) ----
_CACHE: dict = {}


def _cached(key: str, ttl: int, fn, *a, **k):
    """通用 TTL 缓存。key 相同且未过期 → 返回缓存。"""
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    val = fn(*a, **k)
    _CACHE[key] = (now, val)
    return val


def sector_fund_flow(indicator: str = "今日", sector_type: str = "行业资金流",
                     use_cache: bool = True) -> pd.DataFrame:
    """行业资金流排名(东财 stock_sector_fund_flow_rank, 中等限流)。

    返回列: sector/pct_chg/net_amount(元)/net_pct/rank(按净额降序)。失败返回空。
    """
    def _fetch():
        df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
        if df is None or df.empty:
            return pd.DataFrame()
        name_c = _col(df, "名称", "板块") or df.columns[1]
        chg_c = _col(df, "今日涨跌幅", "涨跌幅", "涨幅")
        net_c = _col(df, "今日主力净流入-净额", "主力净流入-净额", "净额")
        pct_c = _col(df, "今日主力净流入-净占比", "净占比")
        out = pd.DataFrame({"sector": df[name_c].astype(str)})
        if chg_c:
            out["pct_chg"] = pd.to_numeric(df[chg_c], errors="coerce")
        if net_c:
            out["net_amount"] = pd.to_numeric(df[net_c], errors="coerce")
        if pct_c:
            out["net_pct"] = pd.to_numeric(df[pct_c], errors="coerce")
        if "net_amount" in out:
            out = out.sort_values("net_amount", ascending=False).reset_index(drop=True)
            out["rank"] = range(1, len(out) + 1)
        return out
    if use_cache:
        return _cached("sector_fund_flow", 300, lambda: _retry_throttle(_fetch, max_retries=3))
    return _retry_throttle(_fetch, max_retries=3)


def concept_heat(topn: int = 60, use_cache: bool = True) -> pd.DataFrame:
    """题材热度(复用 hot_money.concept_heatboard, 多取行供行业匹配)。

    返回列: concept/pct_chg/up_ratio/leading_stock/heat_score。失败返回空。
    """
    def _fetch():
        from strategy.hot_money import concept_heatboard
        df = concept_heatboard(topn=topn)
        if df is None or df.empty or "板块" not in df:
            return pd.DataFrame()
        out = pd.DataFrame({"concept": df["板块"].astype(str)})
        for src, dst in (("涨跌幅", "pct_chg"), ("上涨占比", "up_ratio"),
                         ("领涨股", "leading_stock"), ("热度分", "heat_score")):
            if src in df:
                out[dst] = df[src]
        return out.reset_index(drop=True)
    try:
        if use_cache:
            return _cached("concept_heat", 300, _fetch)
        return _fetch()
    except Exception as e:
        log.warning(f"concept_heat 失败: {e}")
        return pd.DataFrame()


def market_sentiment_snap(date: Optional[str] = None, use_cache: bool = True) -> dict:
    """情绪温度计精简快照(复用 short_term.market_sentiment, 去掉 zt_df 大表)。

    返回 dict: date/n_zt/n_zbgc/n_dt/zha_rate/max_streak/temp/can_play/advice。
    失败返回中性默认(temp=50)。盘前 date 默认取 latest_trade_date(自动昨日)。
    """
    def _fetch():
        from strategy.short_term import market_sentiment
        s = market_sentiment(date)
        return {k: v for k, v in s.items() if k != "zt_df"}
    try:
        if use_cache:
            return _cached("market_sentiment", 600, _fetch)
        return _fetch()
    except Exception as e:
        log.warning(f"market_sentiment_snap 失败: {e}")
        return {"date": (date or latest_trade_date()).replace("-", ""), "temp": 50,
                "n_zt": 0, "n_zbgc": 0, "n_dt": 0, "zha_rate": 0, "max_streak": 0,
                "can_play": "未知", "advice": f"情绪数据获取失败: {e}"}


def margin_balance(date: Optional[str] = None) -> dict:
    """融资余额(交易所官方源 stock_margin_detail_szse/sse, 低限流, T+1)。

    返回 dict: date/sz_balance/sh_balance/total(亿元)。date 默认 latest_closed_trade_date。
    """
    date = date or latest_closed_trade_date()
    d8 = date.replace("-", "")

    def _sz():
        df = _retry_throttle(lambda: ak.stock_margin_detail_szse(date=d8), max_retries=2)
        c = _col(df, "融资余额")
        return pd.to_numeric(df[c], errors="coerce").sum() / 1e8 if (c and not df.empty) else 0.0

    def _sh():
        df = _retry_throttle(lambda: ak.stock_margin_detail_sse(date=d8), max_retries=2)
        c = _col(df, "融资余额")
        return pd.to_numeric(df[c], errors="coerce").sum() / 1e8 if (c and not df.empty) else 0.0

    try:
        sz, sh = _sz(), _sh()
        return {"date": date, "sz_balance": round(sz, 1), "sh_balance": round(sh, 1),
                "total": round(sz + sh, 1)}
    except Exception as e:
        log.warning(f"margin_balance 失败: {e}")
        return {"date": date, "sz_balance": 0.0, "sh_balance": 0.0, "total": 0.0}


if __name__ == "__main__":
    print("=== 行业资金流 top5 ===")
    print(sector_fund_flow().head())
    print("\n=== 题材热度 top5 ===")
    print(concept_heat().head())
    print("\n=== 情绪温度计 ===")
    s = market_sentiment_snap()
    print(f"temp={s.get('temp')} n_zt={s.get('n_zt')} zha_rate={s.get('zha_rate')}")
    print("\n=== 融资余额 ===")
    print(margin_balance())

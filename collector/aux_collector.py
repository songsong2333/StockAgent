"""辅助数据采集: ST标记、交易日历。

akshare 接口偶有变动, 此处对字段做容错并集中维护, 便于后续升级。
"""
from __future__ import annotations

import time
from typing import List, Optional, Set

import akshare as ak
import pandas as pd

from common import setup_logger

log = setup_logger("collector.aux")


def get_st_codes() -> Set[str]:
    """获取当前 ST/*ST 股票代码集合(6位)。"""
    try:
        df = ak.stock_zh_a_st_em()
        # 字段名随版本可能是 '代码' 或 'code'
        code_col = "代码" if "代码" in df.columns else df.columns[1]
        codes = set(df[code_col].astype(str).str.zfill(6))
        log.info(f"获取ST股票 {len(codes)} 只")
        return codes
    except Exception as e:
        log.warning(f"获取ST列表失败: {e}, 返回空集合")
        return set()


def get_trade_dates(start: str = "2010-01-01", end: Optional[str] = None) -> List[str]:
    """获取交易日历, 返回 'YYYY-MM-DD' 字符串列表。"""
    try:
        df = ak.tool_trade_date_hist_sina()
        # 字段通常为 'trade_date'
        col = "trade_date" if "trade_date" in df.columns else df.columns[0]
        dates = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        if end:
            dates = dates[dates <= end]
        dates = dates[dates >= start]
        return sorted(dates.tolist())
    except Exception as e:
        log.error(f"获取交易日历失败: {e}")
        raise


def is_trade_date(date: str) -> bool:
    """判断某日是否为交易日。"""
    try:
        dates = get_trade_dates(start=date, end=date)
        return len(dates) > 0
    except Exception:
        # 降级: 仅按周末判断(节假日可能误判, 但能避免周末跑)
        dt = pd.Timestamp(date)
        return dt.weekday() < 5


def latest_trade_date(today: Optional[str] = None) -> str:
    """获取 <= today 的最近一个交易日(含今天, 即便盘中未结束)。"""
    today = today or pd.Timestamp.now().strftime("%Y-%m-%d")
    dates = get_trade_dates(start="2010-01-01", end=today)
    if not dates:
        return today
    return dates[-1]


def latest_closed_trade_date(now=None, close_hour: int = 16) -> str:
    """最近一个"已收盘"交易日: 当前时间 >= close_hour 才算今天收盘, 否则用前一交易日。

    用于龙虎榜/涨停池等"盘后才有"的数据, 避免盘前取到当天空数据。
    """
    now = now or pd.Timestamp.now()
    today = now.strftime("%Y-%m-%d")
    dates = get_trade_dates(start="2010-01-01", end=today)
    if not dates:
        return today
    # 当天已过收盘点且当天是交易日 -> 用当天; 否则用前一个
    if now.hour >= close_hour and dates[-1] == today:
        return dates[-1]
    # 取倒数第二个(前一交易日); 若没有则退回最后一个
    return dates[-2] if len(dates) >= 2 else dates[-1]

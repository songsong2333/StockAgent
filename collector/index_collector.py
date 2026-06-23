"""指数采集: 沪深300等基准指数, 作为 qlib instrument 一并 dump。

akshare: ak.stock_zh_index_daily(symbol="sh000300")
返回字段: date open high low close volume amount (无 turnover)
存为 raw/<INDEX>.parquet, 与股票同格式, dump 时自动纳入。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import akshare as ak
import pandas as pd

from common import setup_logger, ensure_dir

log = setup_logger("collector.index")

# qlib代码 -> akshare symbol
INDEX_SYMBOLS = {
    "SH000300": "sh000300",   # 沪深300
    "SH000905": "sh000905",   # 中证500
    "SH000001": "sh000001",   # 上证综指
}


def fetch_index(qlib_code: str, start: str, end: str, max_retries: int = 3, sleep: float = 0.3) -> pd.DataFrame:
    """拉取指数日线, 返回标准化 DataFrame。"""
    symbol = INDEX_SYMBOLS.get(qlib_code)
    if not symbol:
        raise ValueError(f"未知指数: {qlib_code}, 支持: {list(INDEX_SYMBOLS)}")
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                    "low": "low", "close": "close",
                                    "volume": "volume", "amount": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            for c in ("open", "high", "low", "close", "volume", "amount"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            import time
            time.sleep(sleep)
            return df
        except Exception as e:
            log.warning(f"{qlib_code} 第{attempt}次失败: {e}")
            import time
            time.sleep(sleep * 2)
    return pd.DataFrame()


def update_index(qlib_code: str, raw_dir: str, start: str, end: str, max_retries: int = 3, sleep: float = 0.3) -> bool:
    """增量更新指数 (与股票增量逻辑一致)。"""
    path = Path(raw_dir) / f"{qlib_code}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        last = old["date"].max().strftime("%Y-%m-%d")
        if last >= end:
            return True
        fetch_start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        old = pd.DataFrame()
        fetch_start = start
    new = fetch_index(qlib_code, fetch_start, end, max_retries, sleep)
    combined = pd.concat([old, new], ignore_index=True) if not new.empty else old
    if combined.empty:
        return False
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    ensure_dir(raw_dir)
    combined.to_parquet(path, index=False)
    return True


def update_benchmark_indices(raw_dir: str, start: str, end: str,
                             codes: List[str] = None) -> int:
    """更新基准指数。默认更新沪深300。"""
    codes = codes or ["SH000300"]
    ok = 0
    for c in codes:
        if update_index(c, raw_dir, start, end):
            ok += 1
            log.info(f"指数 {c} 更新成功")
    return ok

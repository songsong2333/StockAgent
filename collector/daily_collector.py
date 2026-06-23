"""日线行情采集: 前复权 OHLCV, 增量更新, parquet 按股票缓存。

双数据源:
  - 主: ak.stock_zh_a_daily (新浪源, 稳定不易限流)
  - 备: ak.stock_zh_a_hist  (东方财富源, 字段更全但易限流)
统一输出英文列: date open high low close volume amount turnover
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import akshare as ak
import pandas as pd
from tqdm import tqdm

from common import setup_logger, ensure_dir, to_raw_code, load_config

log = setup_logger("collector.daily")

# 东财源中文列 -> 标准英文列
_EM_COL_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
    "涨跌幅": "pct_chg", "涨跌额": "chg", "换手率": "turnover",
}
_STANDARD_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def _raw_path(raw_dir: str, qlib_code: str) -> Path:
    return Path(raw_dir) / f"{qlib_code}.parquet"


def _to_sina_symbol(code: str) -> str:
    """qlib代码/6位代码 -> 新浪源 symbol (sz300308 / sh600000 / bj830799)。"""
    s = str(code)
    if s.startswith(("SH", "SZ", "BJ")):
        return s[0:2].lower() + s[2:]
    s = s.zfill(6)
    prefix = {"6": "sh", "0": "sz", "3": "sz", "8": "bj", "4": "bj"}[s[0]]
    return prefix + s


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名与类型。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_EM_COL_MAP)  # 英文列映射自身, 中文列转英文
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in _STANDARD_COLS if c in df.columns]
    df = df[keep].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def fetch_daily(code: str, start: str, end: str, adjust: str = "qfq",
                max_retries: int = 3, sleep: float = 0.3) -> pd.DataFrame:
    """拉取单只股票日线, 返回标准化 DataFrame。

    code: qlib 代码 (SZ000001) 或 6位代码。先试新浪源, 失败再试东财源。
    """
    sina_sym = _to_sina_symbol(code)
    em_sym = to_raw_code(code) if str(code).startswith(("SH", "SZ", "BJ")) else str(code).zfill(6)
    adjust = adjust or ""

    # --- 新浪源 (主) ---
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina_sym,
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                adjust=adjust,
            )
            df = _normalize(df)
            if not df.empty:
                time.sleep(sleep)
                return df
            break  # 空结果不重试
        except Exception as e:
            log.debug(f"{sina_sym}(新浪) 第{attempt}次失败: {e}")
            time.sleep(sleep * (2 ** attempt))  # 指数退避

    # --- 东财源 (备) ---
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=em_sym, period="daily",
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                adjust=adjust,
            )
            df = _normalize(df)
            if not df.empty:
                time.sleep(sleep)
                return df
            break
        except Exception as e:
            log.debug(f"{em_sym}(东财) 第{attempt}次失败: {e}")
            time.sleep(sleep * (2 ** attempt))

    log.warning(f"{code} 采集失败(新浪+东财 各{max_retries}次)")
    return pd.DataFrame()


def update_one(qlib_code: str, raw_dir: str, start: str, end: str,
               adjust: str, max_retries: int, sleep: float) -> bool:
    """增量更新单只股票: 读取已有 parquet, 仅拉取最后日期之后的新数据。"""
    path = _raw_path(raw_dir, qlib_code)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            last_date = old["date"].max().strftime("%Y-%m-%d")
            # 已是最新则跳过
            if last_date >= end:
                return True
            fetch_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            old = pd.DataFrame()
            fetch_start = start
    else:
        old = pd.DataFrame()
        fetch_start = start

    new = fetch_daily(qlib_code, fetch_start, end, adjust, max_retries, sleep)
    if new.empty and old.empty:
        return False
    combined = pd.concat([old, new], ignore_index=True) if not new.empty else old
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    ensure_dir(raw_dir)
    combined.to_parquet(path, index=False)
    return True


def update_all(cfg: dict, codes: Optional[List[str]] = None, end: Optional[str] = None) -> int:
    """批量增量更新所有股票。返回成功更新数量。"""
    cc = cfg["collector"]
    raw_dir = cfg["paths"]["raw_dir"]
    start = cc["start_date"]
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    adjust = cc.get("adjust", "qfq")

    if codes is None:
        from collector.stock_pool import get_pool_codes
        codes = get_pool_codes(cfg)

    ok = 0
    for code in tqdm(codes, desc="采集日线"):
        try:
            if update_one(code, raw_dir, start, end, adjust, cc["max_retries"], cc["request_sleep"]):
                ok += 1
        except Exception as e:
            log.warning(f"{code} 更新异常: {e}")
    log.info(f"采集完成: {ok}/{len(codes)} 只成功")
    return ok


if __name__ == "__main__":
    cfg = load_config()
    # 单只测试
    update_one("SZ000001", cfg["paths"]["raw_dir"], "2024-01-01", "2024-06-01",
               "qfq", 3, 0.3)

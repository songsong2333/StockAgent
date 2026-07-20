"""场内 ETF 日线采集: akshare fund_etf_hist_em, 前复权, 增量更新, parquet 按ETF代码缓存。

ETF 全程用原始6位代码(不做 SH/SZ 转换), 存独立目录 data/etf/<code>.parquet,
与个股 data/raw/ 解耦, 且不走 qlib(趋势/动量只需 OHLCV)。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import akshare as ak
import pandas as pd
import yaml
from tqdm import tqdm

from common import setup_logger, ensure_dir, load_config, PROJECT_ROOT

log = setup_logger("collector.etf")

# 东财 fund_etf_hist_em 中文列 -> 标准英文(与 daily_collector 对齐)
_EM_COL_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
    "涨跌幅": "pct_chg", "涨跌额": "chg", "换手率": "turnover",
}
_STANDARD_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def _etf_path(etf_dir: Union[str, Path], code: str) -> Path:
    """code 为原始6位字符串, 直接作文件名。"""
    return Path(etf_dir) / f"{code}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名与类型(与 daily_collector._normalize 同逻辑)。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_EM_COL_MAP)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in _STANDARD_COLS if c in df.columns]
    df = df[keep].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def fetch_etf(code: str, start: str, end: str, adjust: str = "qfq",
              max_retries: int = 3, sleep: float = 0.3) -> pd.DataFrame:
    """拉取单只场内 ETF 日线, 返回标准化 DataFrame。code: 原始6位(如 510300)。

    数据源: ak.fund_etf_hist_em(东财)。空结果不重试(可能是代码错/退市)。
    """
    code = str(code).strip().zfill(6)
    adjust = adjust or ""
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.fund_etf_hist_em(
                symbol=code, period="daily",
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                adjust=adjust,
            )
            df = _normalize(df)
            if not df.empty:
                time.sleep(sleep)
                return df
            break  # 空结果不重试(代码错/退市)
        except Exception as e:
            # 限流类(ConnectionError/RemoteDisconnected/超时) → 长退避等东财恢复; 其他异常短退避
            msg = repr(e).lower()
            throttled = any(k in msg for k in ("connection", "remote", "timeout", "timed out", "aborted", "reset"))
            wait = (3.0 if throttled else sleep) * (2 ** (attempt - 1))
            log.debug(f"{code}(ETF) 第{attempt}次失败({'限流,长等待' if throttled else '异常'} {wait:.0f}s): {e}")
            time.sleep(wait)
    log.warning(f"{code} ETF 采集失败({max_retries}次)")
    return pd.DataFrame()


def update_one(code: str, etf_dir: Union[str, Path], start: str, end: str,
               adjust: str, max_retries: int, sleep: float) -> bool:
    """增量更新单只 ETF: 读取已有 parquet, 仅拉最后日期之后的新数据。"""
    code = str(code).strip().zfill(6)
    path = _etf_path(etf_dir, code)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            last_date = pd.to_datetime(old["date"]).max().strftime("%Y-%m-%d")
            if last_date >= end:
                return True
            fetch_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            old = pd.DataFrame()
            fetch_start = start
    else:
        old = pd.DataFrame()
        fetch_start = start

    new = fetch_etf(code, fetch_start, end, adjust, max_retries, sleep)
    if new.empty and old.empty:
        return False
    combined = pd.concat([old, new], ignore_index=True) if not new.empty else old
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    ensure_dir(etf_dir)
    combined.to_parquet(path, index=False)
    return True


def load_etf_pool(cfg: dict) -> dict:
    """读 config/etf_pool.yaml → {benchmark_etf, broad:[{code,name,...}], sector:[...]}。"""
    pool_file = cfg.get("etf", {}).get("pool_file", "config/etf_pool.yaml")
    p = Path(pool_file)
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"ETF 池不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 规范化 code 为 6 位字符串
    for grp in ("broad", "sector"):
        for it in data.get(grp, []) or []:
            it["code"] = str(it["code"]).strip().zfill(6)
    return data


def pool_codes(cfg: dict) -> List[str]:
    """标的池全部代码(benchmark + broad + sector), 去重保序。"""
    pool = load_etf_pool(cfg)
    codes: List[str] = []
    seen = set()
    seq = [pool.get("benchmark_etf")] + [it["code"] for it in pool.get("broad", [])] \
        + [it["code"] for it in pool.get("sector", [])]
    for c in seq:
        c = str(c).strip().zfill(6)
        if c and c not in seen:
            codes.append(c)
            seen.add(c)
    return codes


def update_all(cfg: dict, codes: Optional[List[str]] = None,
               end: Optional[str] = None) -> Tuple[int, List[str]]:
    """批量增量更新所有 ETF。返回 (成功数, 失败代码列表)。"""
    cc = cfg["collector"]
    etf_dir = cfg["paths"].get("etf_dir", "data/etf")
    if not Path(etf_dir).is_absolute():
        etf_dir = PROJECT_ROOT / etf_dir
    start = cc["start_date"]
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    adjust = cc.get("adjust", "qfq")

    if codes is None:
        codes = pool_codes(cfg)

    ok, failed = 0, []
    for code in tqdm(codes, desc="采集ETF"):
        try:
            if update_one(code, etf_dir, start, end, adjust,
                          cc["max_retries"], cc["request_sleep"]):
                ok += 1
            else:
                failed.append(code)
        except Exception as e:
            log.warning(f"{code} 更新异常: {e}")
            failed.append(code)
        time.sleep(cc.get("request_sleep", 0.8))  # 每只间隔, 避免连续请求触发东财限流
    log.info(f"ETF 采集完成: {ok}/{len(codes)} 成功, 失败 {failed}")
    return ok, failed


if __name__ == "__main__":
    cfg = load_config()
    # 单只测试
    df = fetch_etf("510300", "2024-01-01", "2024-06-01")
    print(df.tail() if not df.empty else "空")

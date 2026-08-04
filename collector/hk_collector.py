"""港股日线采集(周期探索平台用)。

⚠️ 东财 stock_hk_hist 在本环境持续 ReadTimeout(33.push2his.eastmoney 限流),
故【默认走新浪 stock_hk_daily】(实测稳定, ~3s, 返回前复权 OHLCV), 东财仅作回退。
新浪源返回完整历史(不带 start/end), 故增量更新=拉全量→合并去重(港股数量少, 可接受)。

code 全程原样 5 位(如 09988), 不 zfill, 存 data/hk/<code>.parquet。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import akshare as ak
import pandas as pd
import yaml

from common import setup_logger, ensure_dir, load_config, PROJECT_ROOT

log = setup_logger("collector.hk")

_DATE_COLS = ("date", "日期")
_STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
# 东财中文列 → 英文(与 etf_collector 一致)
_EM_COL_MAP = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
               "最低": "low", "成交量": "volume", "成交额": "amount"}


def _hk_path(hk_dir: Union[str, Path], code: str) -> Path:
    return Path(hk_dir) / f"{code}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_EM_COL_MAP)
    date_col = next((c for c in _DATE_COLS if c in df.columns), df.columns[0])
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in _STD_COLS if c in df.columns]
    df = df[keep].dropna(subset=["date", "close"]).sort_values("date")
    df = df.drop_duplicates("date").reset_index(drop=True)
    return df


def fetch_hk(code: str, adjust: str = "qfq", max_retries: int = 3,
             sleep: float = 0.5) -> pd.DataFrame:
    """拉取单只港股完整日线(前复权)。主走新浪 stock_hk_daily, 东财回退。

    返回标准化 DataFrame(全量历史, 调用方按需裁剪区间)。
    """
    code = str(code).strip()
    # 主源: 新浪 stock_hk_daily(稳定, 返回完整历史)
    for attempt in range(1, max_retries + 1):
        try:
            df = _normalize(ak.stock_hk_daily(symbol=code, adjust=adjust or ""))
            if not df.empty:
                time.sleep(sleep)
                return df
        except Exception as e:
            log.debug(f"{code} 新浪第{attempt}次失败: {e}")
            time.sleep(sleep * attempt)
    # 回退: 东财 stock_hk_hist(限流, 多半失败但试一次)
    try:
        df = _normalize(ak.stock_hk_hist(symbol=code, period="daily", adjust=adjust or ""))
        if not df.empty:
            log.info(f"{code} 走东财回退成功")
            return df
    except Exception as e:
        log.debug(f"{code} 东财回退失败: {e}")
    log.warning(f"{code} 港股采集失败(新浪/东财均失败)")
    return pd.DataFrame()


def update_one(code: str, hk_dir: Union[str, Path], start: str, end: str,
               adjust: str = "qfq", max_retries: int = 3, sleep: float = 0.5) -> bool:
    """增量更新单只港股(新浪返全量 → 合并去重 → 裁剪到 >=start)。"""
    code = str(code).strip()
    path = _hk_path(hk_dir, code)
    old = pd.DataFrame()
    if path.exists():
        try:
            old = pd.read_parquet(path)
        except Exception:
            old = pd.DataFrame()

    new = fetch_hk(code, adjust=adjust, max_retries=max_retries, sleep=sleep)
    if new.empty and old.empty:
        return False
    combined = pd.concat([old, new], ignore_index=True) if not new.empty else old
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # 裁剪到 >= start(只保留请求的历史起点之后)
    if "date" in combined.columns:
        combined = combined[pd.to_datetime(combined["date"]) >= pd.Timestamp(start)]
    ensure_dir(hk_dir)
    combined.to_parquet(path, index=False)
    return True


def _hk_codes(cfg: dict) -> List[str]:
    """从 config/alt_sources.yaml 的 hk 组取代码清单。"""
    p = Path(cfg.get("cycle", {}).get("alt_file", "config/alt_sources.yaml"))
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return []
    reg = yaml.safe_load(open(p, encoding="utf-8")) or {}
    return [str(it["id"]).strip() for it in reg.get("hk", []) or [] if it.get("id")]


def update_all(cfg: dict, codes: Optional[List[str]] = None,
               end: Optional[str] = None) -> Tuple[int, List[str]]:
    """批量增量更新港股。返回 (成功数, 失败代码)。"""
    hk_dir = cfg["paths"].get("hk_dir", "data/hk")
    hk_dir = hk_dir if Path(hk_dir).is_absolute() else PROJECT_ROOT / hk_dir
    start = cfg["collector"]["start_date"]
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    adjust = cfg["collector"].get("adjust", "qfq")
    sleep = cfg["collector"].get("request_sleep", 0.8)
    retries = cfg["collector"].get("max_retries", 3)

    codes = codes if codes is not None else _hk_codes(cfg)
    ok, failed = 0, []
    for code in codes:
        try:
            if update_one(code, hk_dir, start, end, adjust, retries, sleep):
                ok += 1
            else:
                failed.append(code)
        except Exception as e:
            log.warning(f"{code} 更新异常: {e}")
            failed.append(code)
        time.sleep(sleep)
    log.info(f"港股采集完成: {ok}/{len(codes)} 成功, 失败 {failed}")
    return ok, failed


if __name__ == "__main__":
    cfg = load_config()
    df = fetch_hk("09988")
    print(df.tail() if not df.empty else "空")

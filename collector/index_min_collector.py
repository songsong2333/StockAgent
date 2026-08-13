"""指数 60min K 线采集: 新浪 stock_zh_a_minute(period="60"), 增量合并, parquet 缓存。

「成长进攻 ↔ 红利防守」确定性进出场系统的数据层。
- 新浪源稳定不限流, 单次返回约 2 年 60min 窗口(足够回测)。
- 进攻/大盘标的采指数 60min(symbol 如 sh000905), 存 data/index_min/<symbol>.parquet。
- 同一份 60min 可 resample 出日线, 供日线 conviction 层使用(同源, 无对齐误差)。
- 防守标的红利100(ETF 515180)走东财日线 fund_etf_hist_em, 仅作轮动基准, 见 fetch_defense_daily。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import akshare as ak
import pandas as pd

from common import setup_logger, ensure_dir, PROJECT_ROOT

log = setup_logger("collector.index_min")

_MIN_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _min_path(min_dir: Union[str, Path], symbol: str) -> Path:
    return Path(min_dir) / f"{symbol}.parquet"


def _normalize_min(df: pd.DataFrame) -> pd.DataFrame:
    """stock_zh_a_minute 返回列 day/open/..., 值为字符串; 统一为标准 OHLCV。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in _MIN_COLS if c in df.columns]
    df = df[keep].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def fetch_index_min(symbol: str, period: str = "60",
                    max_retries: int = 3, sleep: float = 1.0) -> pd.DataFrame:
    """拉取指数 60min(新浪 stock_zh_a_minute)。返回标准化 DataFrame。

    新浪一次返回约2年窗口, 不支持 start/end; 增量在 update_one_min 里合并去重。
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_minute(symbol=symbol, period=period)
            df = _normalize_min(df)
            if not df.empty:
                time.sleep(sleep)
                return df
            break
        except Exception as e:
            msg = repr(e).lower()
            throttled = any(k in msg for k in ("connection", "remote", "timeout", "timed out", "aborted", "reset"))
            wait = (3.0 if throttled else sleep) * attempt
            log.debug(f"{symbol} 60min 第{attempt}次失败({'限流' if throttled else '异常'} {wait:.0f}s): {e}")
            time.sleep(wait)
    log.warning(f"{symbol} 60min 采集失败")
    return pd.DataFrame()


def update_one_min(symbol: str, min_dir: Union[str, Path],
                   period: str = "60", max_retries: int = 3, sleep: float = 1.0) -> bool:
    """合并式更新: 新浪返回全窗口, 与本地 parquet 合并去重后落盘。"""
    path = _min_path(min_dir, symbol)
    old = pd.DataFrame()
    if path.exists():
        try:
            old = pd.read_parquet(path)
        except Exception:
            old = pd.DataFrame()
    new = fetch_index_min(symbol, period, max_retries, sleep)
    if new.empty and old.empty:
        return False
    combined = pd.concat([old, new], ignore_index=True) if not new.empty else old
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    ensure_dir(min_dir)
    combined.to_parquet(path, index=False)
    return True


def load_index_min(cfg: dict, symbol: str) -> pd.DataFrame:
    """读取某指数 60min。"""
    min_dir = resolve_min_dir(cfg)
    path = _min_path(min_dir, symbol)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def resolve_min_dir(cfg: dict) -> Path:
    d = cfg.get("paths", {}).get("index_min_dir", "data/index_min")
    p = Path(d)
    return p if p.is_absolute() else PROJECT_ROOT / p


def update_all_min(cfg: dict, symbols: Optional[List[str]] = None) -> Tuple[int, List[str]]:
    """批量更新配置的指数 60min。symbols 缺省取 entry_exit 段的 growth+market。"""
    ee = cfg.get("entry_exit", {})
    if symbols is None:
        symbols = [g["index"] for g in ee.get("growth", [])] + [ee.get("market")]
        symbols = [s for s in symbols if s]
    min_dir = resolve_min_dir(cfg)
    ok, failed = 0, []
    for sym in symbols:
        try:
            if update_one_min(sym, min_dir):
                ok += 1
            else:
                failed.append(sym)
        except Exception as e:
            log.warning(f"{sym} 更新异常: {e}")
            failed.append(sym)
        time.sleep(cfg.get("collector", {}).get("request_sleep", 1.0))
    log.info(f"指数60min 采集完成: {ok}/{len(symbols)} 成功, 失败 {failed}")
    return ok, failed


def resample_daily(min_df: pd.DataFrame) -> pd.DataFrame:
    """60min -> 日线 OHLCV(供日线 conviction 层; 同源无对齐误差)。"""
    if min_df.empty:
        return pd.DataFrame()
    d = min_df.set_index("date")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "amount" in d.columns:
        agg["amount"] = "sum"
    daily = d.resample("1D").agg(agg).dropna(subset=["close"])
    return daily.reset_index()


def fetch_defense_daily(etf_code: str, start: str = None, end: str = None,
                        max_retries: int = 3, sleep: float = 1.5) -> pd.DataFrame:
    """防守 ETF(红利100)日线: 新浪 fund_etf_hist_sina(稳定不限流), 历史6年+。

    东财 fund_etf_hist_em 在本机频繁断连, 故用新浪源。symbol 需交易所前缀(5开头=sh, 1开头=sz)。
    """
    etf_code = str(etf_code).strip().zfill(6)
    prefix = "sh" if etf_code.startswith("5") else "sz"
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.fund_etf_hist_sina(symbol=f"{prefix}{etf_code}")
            if df is not None and not df.empty:
                df = df.rename(columns={"day": "date"}) if "day" in df.columns else df
                df["date"] = pd.to_datetime(df["date"])
                for c in ("open", "high", "low", "close", "volume"):
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[keep].sort_values("date").drop_duplicates("date").reset_index(drop=True)
                if start:
                    df = df[df["date"] >= pd.Timestamp(start)]
                if end:
                    df = df[df["date"] <= pd.Timestamp(end)]
                time.sleep(sleep)
                return df.reset_index(drop=True)
            break
        except Exception as e:
            log.debug(f"{etf_code}(防守)日线 第{attempt}次失败: {e}")
            time.sleep(2.0 * attempt)
    log.warning(f"{etf_code} 防守日线采集失败")
    return pd.DataFrame()


def update_defense(cfg: dict) -> bool:
    """采集并缓存防守 ETF 日线到 index_min 目录。"""
    ee = cfg.get("entry_exit", {})
    code = ee.get("defense", {}).get("etf")
    if not code:
        return False
    df = fetch_defense_daily(code)
    if df.empty:
        return False
    path = resolve_min_dir(cfg) / f"def_{code}.parquet"
    ensure_dir(resolve_min_dir(cfg))
    df.to_parquet(path, index=False)
    log.info(f"防守 {code} 日线已缓存: {len(df)}行 {df['date'].min().date()}~{df['date'].max().date()}")
    return True


def load_defense(cfg: dict) -> pd.DataFrame:
    code = cfg.get("entry_exit", {}).get("defense", {}).get("etf")
    path = resolve_min_dir(cfg) / f"def_{code}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


if __name__ == "__main__":
    from common import load_config
    cfg = load_config()
    ok, failed = update_all_min(cfg)
    print(f"ok={ok} failed={failed}")

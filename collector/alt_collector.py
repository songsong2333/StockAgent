"""周期探索平台 —— 替代数据采集器(可插拔)。

读 config/alt_sources.yaml 注册表, 按 source 分发采集, 统一存 data/store/<id>.parquet
(列 [date, value], 月频)。支持四类 source:
  akshare  调 getattr(ak, func)(), 取 col 列, 按 date_col 解析日期, 重采样到月
  futures  ak.futures_zh_daily_sina(symbol=code), 取 close, 日频→月
  csv      本地 path 或 URL, 用 parser 指定解析器(如 noaa_oni)

港股(source: hk)不在此处理, 由 collector.hk_collector 专管(日频, data/hk)。
加新序列只需在 alt_sources.yaml 追加一条, 不改代码。
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd
import yaml

from common import setup_logger, ensure_dir, load_config, PROJECT_ROOT

log = setup_logger("collector.alt")


# ============== 日期解析(容忍中文/数字各种格式) ==============
def _parse_dates(s: pd.Series) -> pd.Series:
    """把各种日期字符串解析为 Timestamp: 先试 pd.to_datetime, 再正则抽 年-月。"""
    dt = pd.to_datetime(s, errors="coerce", format="mixed")
    if dt.notna().all():
        return dt
    # 回退: 正则 "2024年1月" / "2024年1月份" / "202401"
    def _one(x):
        if pd.isna(x):
            return pd.NaT
        m = re.search(r"(\d{4})\D*(\d{1,2})", str(x))
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1900 <= y <= 2100 and 1 <= mo <= 12:
                return pd.Timestamp(year=y, month=mo, day=1)
        return pd.NaT
    fallback = s.apply(_one)
    return dt.fillna(fallback)


def _to_monthly(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """从 akshare DataFrame 抽 (date, value) 并重采样到月末。"""
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": _parse_dates(df[date_col]) if date_col in df.columns
        else _parse_dates(df.iloc[:, 0]),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    })
    out = out.dropna(subset=["date", "value"])
    out = out.set_index("date").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.resample("ME").last().dropna().reset_index()


# ============== source 适配器 ==============
def _fetch_akshare(item: dict) -> pd.DataFrame:
    """akshare 宏观序列: getattr(ak, func)() → 取 col, 按 date_col 解析, 转月频。"""
    func = getattr(ak, item["func"])
    df = func()
    return _to_monthly(df, item.get("date_col", df.columns[0]), item["col"])


def _fetch_futures(item: dict) -> pd.DataFrame:
    """商品期货连续合约: ak.futures_zh_daily_sina(symbol=code), 取 close → 月频。"""
    df = ak.futures_zh_daily_sina(symbol=item["code"])
    if df is None or df.empty:
        return pd.DataFrame()
    return _to_monthly(df, "date", "close")


def _parse_noaa_oni(url: str) -> pd.DataFrame:
    """NOAA ONI(厄尔尼诺海温异常) 月序列。文件格式: 首行"1950 2026"元信息,
    其后每年一行"年 m1..m12", 末尾文本注脚; 缺测值 -99.0/-99.9。
    解析为 [date(月初), value] 月序列。"""
    raw = pd.read_csv(url, sep=r"\s+", header=None, engine="python", skiprows=1)
    raw.columns = ["year"] + [f"m{i}" for i in range(1, 13)]
    # 仅保留首列为4位年份的行(过滤末尾文本注脚)
    raw = raw[raw["year"].astype(str).str.match(r"^\d{4}$")]
    raw["year"] = raw["year"].astype(int)
    long = raw.melt(id_vars="year", var_name="m", value_name="value")
    long["m"] = long["m"].str[1:].astype(int)
    long["date"] = pd.to_datetime(long["year"].astype(str) + "-" + long["m"].astype(str) + "-01")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long[(long["value"].abs() < 90)].dropna(subset=["value"])  # 去缺测
    return long.sort_values("date")[["date", "value"]].reset_index(drop=True)


def _parse_monthly_csv(path_or_url: str) -> pd.DataFrame:
    """通用月频 CSV: 首列日期, 次列值。"""
    df = pd.read_csv(path_or_url)
    return _to_monthly(df, df.columns[0], df.columns[1])


_PARSERS = {"noaa_oni": _parse_noaa_oni, "monthly_csv": _parse_monthly_csv}


def _fetch_csv(item: dict) -> pd.DataFrame:
    """本地 path 或 URL 的 CSV/数据文件, 用 parser 指定解析器。"""
    src = item.get("url") or item.get("path")
    if not src:
        return pd.DataFrame()
    parser = _PARSERS.get(item.get("parser", "monthly_csv"), _parse_monthly_csv)
    return parser(src)


_DISPATCH = {"akshare": _fetch_akshare, "futures": _fetch_futures, "csv": _fetch_csv}


def fetch_alt(item: dict) -> pd.DataFrame:
    """按 item['source'] 分发采集, 返回 [date, value] 月频 DataFrame。"""
    src = item.get("source")
    fn = _DISPATCH.get(src)
    if fn is None:
        log.warning(f"未知 source={src} (id={item.get('id')})")
        return pd.DataFrame()
    try:
        df = fn(item)
        if df.empty:
            log.warning(f"{item.get('id')}({src}) 返回空")
        return df
    except Exception as e:
        log.warning(f"{item.get('id')}({src}) 采集失败: {e}")
        return pd.DataFrame()


# ============== 落盘 / 批量 ==============
def _store_path(store_dir: Path, sid: str) -> Path:
    return store_dir / f"{sid}.parquet"


def update_one_alt(item: dict, store_dir: Path, start: str, end: str,
                  retries: int = 3, sleep: float = 0.5) -> bool:
    """采集单条替代序列, 裁剪到 [start, end], 存 data/store/<id>.parquet。"""
    sid = str(item["id"]).strip()
    df = pd.DataFrame()
    for a in range(1, retries + 1):
        df = fetch_alt(item)
        if not df.empty:
            break
        time.sleep(sleep * a)
    if df.empty:
        return False
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df.empty:
        return False
    ensure_dir(store_dir)
    df.to_parquet(_store_path(store_dir, sid), index=False)
    return True


def load_alt_registry(cfg: dict) -> dict:
    p = Path(cfg.get("cycle", {}).get("alt_file", "config/alt_sources.yaml"))
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return {}
    return yaml.safe_load(open(p, encoding="utf-8")) or {}


def update_all_alt(cfg: dict, end: Optional[str] = None,
                   categories: Optional[list] = None) -> tuple:
    """批量更新所有替代序列(macro/energy/commodity/climate)。返回 (成功数, 失败id)。"""
    store_dir = cfg["paths"].get("store_dir", "data/store")
    store_dir = store_dir if Path(store_dir).is_absolute() else PROJECT_ROOT / store_dir
    start = cfg["collector"]["start_date"]
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    sleep = cfg["collector"].get("request_sleep", 0.8)
    retries = cfg["collector"].get("max_retries", 3)

    reg = load_alt_registry(cfg)
    cats = categories or [c for c in reg if c != "hk"]
    ok, failed = 0, []
    for cat in cats:
        for item in reg.get(cat, []) or []:
            try:
                if update_one_alt(item, store_dir, start, end, retries, sleep):
                    ok += 1
                    log.info(f"  ✓ {item['id']} ({cat})")
                else:
                    failed.append(item["id"])
            except Exception as e:
                log.warning(f"{item.get('id')} 异常: {e}")
                failed.append(item["id"])
            time.sleep(sleep)
    log.info(f"替代数据采集完成: {ok}/{ok+len(failed)} 成功, 失败 {failed}")
    return ok, failed


if __name__ == "__main__":
    cfg = load_config()
    ok, failed = update_all_alt(cfg)
    print(f"成功 {ok}, 失败 {failed}")

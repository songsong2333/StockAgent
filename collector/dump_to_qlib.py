"""原始 parquet -> qlib bin 格式转换 (自实现, 适配 qlib 0.9.7)。

qlib 0.9.7 已移除官方 dump_bin 脚本, 此处按其 FileFeatureStorage 的二进制格式直写:
  calendars/day.txt          每行一个日期 'YYYY-MM-DD'
  instruments/all.txt        TSV: SYMBOL\\tSTART\\tEND (无表头)
  features/<symbol_lower>/<field>.day.bin
      = np.hstack([start_index, values]).astype('<f4') 小端 float32
      start_index = 该股票首个日期在全局日历中的位置; values 对齐日历, 停牌日填 NaN
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import setup_logger, ensure_dir, load_config, resolve_data_path

log = setup_logger("collector.dump")

QLIB_FIELDS = ["open", "high", "low", "close", "volume", "amount", "turnover"]


def _abs_path(cfg, key: str) -> Path:
    return resolve_data_path(cfg["qlib"]["provider_uri"]).resolve()


def load_all_raw(cfg: dict, codes: Optional[List[str]] = None) -> dict:
    """读取所有股票 parquet, 返回 {qlib_code: DataFrame(date indexed)}。"""
    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"]).resolve()
    files = sorted(raw_dir.glob("*.parquet"))
    if codes:
        wanted = {c.upper() for c in codes}
        files = [f for f in files if f.stem.upper() in wanted]

    data = {}
    cc = cfg["collector"]
    min_bars = cc.get("exclude_new_days", 0)
    for f in tqdm(files, desc="读取原始数据"):
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            log.warning(f"读取 {f.name} 失败: {e}")
            continue
        if df.empty:
            continue
        if min_bars and len(df) < min_bars:
            continue
        df = df[["date"] + [c for c in QLIB_FIELDS if c in df.columns]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()
        # 数值化
        for c in QLIB_FIELDS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        data[f.stem] = df
    return data


def dump(cfg: dict, codes: Optional[List[str]] = None) -> Path:
    """转换原始数据为 qlib bin 格式。返回 qlib_dir。"""
    qlib_dir = ensure_dir(_abs_path(cfg, "qlib"))
    data = load_all_raw(cfg, codes)
    if not data:
        raise ValueError("无可 dump 的数据, 请先运行采集")

    # 1. 全局日历 = 所有股票日期的并集
    all_dates = sorted(set().union(*[set(d.index) for d in data.values()]))
    cal_arr = pd.DatetimeIndex(all_dates)
    cal_dir = ensure_dir(qlib_dir / "calendars")
    np.savetxt(cal_dir / "day.txt", cal_arr.strftime("%Y-%m-%d").to_numpy(), fmt="%s")
    log.info(f"日历 {len(cal_arr)} 天: {cal_arr[0].date()} ~ {cal_arr[-1].date()}")

    # 日期 -> 全局索引
    date_to_idx = {d: i for i, d in enumerate(cal_arr)}

    # 2. instruments/all.txt
    inst_dir = ensure_dir(qlib_dir / "instruments")
    inst_lines = []
    for symbol, df in data.items():
        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")
        inst_lines.append(f"{symbol}\t{start}\t{end}")
    (inst_dir / "all.txt").write_text("\n".join(inst_lines) + "\n", encoding="utf-8")
    log.info(f"股票 {len(data)} 只 -> instruments/all.txt")

    # 3. features/<symbol_lower>/<field>.day.bin
    feat_dir = ensure_dir(qlib_dir / "features")
    fields_present = [f for f in QLIB_FIELDS if any(f in df.columns for df in data.values())]

    for symbol, df in tqdm(data.items(), desc="写 bin"):
        sym_dir = ensure_dir(feat_dir / symbol.lower())
        # 该股票在全局日历中的索引范围
        s_idx = date_to_idx[df.index.min()]
        e_idx = date_to_idx[df.index.max()]
        # 对齐到该范围内的每个日历日(停牌填NaN)
        sub_cal = cal_arr[s_idx:e_idx + 1]
        reindexed = df.reindex(sub_cal)
        for field in fields_present:
            if field not in reindexed.columns:
                continue
            values = reindexed[field].to_numpy(dtype="<f4")
            arr = np.hstack([np.array([s_idx], dtype="<f4"), values])
            arr.astype("<f4").tofile(sym_dir / f"{field}.day.bin")

    log.info(f"dump 完成 -> {qlib_dir}")
    return qlib_dir


if __name__ == "__main__":
    cfg = load_config()
    dump(cfg)

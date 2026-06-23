"""股票池维护: 全A股清单 + 过滤 ST/北交所/科创板/创业板。

输出 qlib 格式代码 (SH/SZ/BJ 前缀) 与 6 位原始代码的映射。
"""
from __future__ import annotations

from typing import List

import akshare as ak
import pandas as pd

from common import setup_logger, to_qlib_code
from collector.aux_collector import get_st_codes

log = setup_logger("collector.pool")


def get_all_a_shares() -> pd.DataFrame:
    """获取全A股代码与名称, 返回 DataFrame[code, name]。"""
    df = ak.stock_info_a_code_name()
    # 字段通常为 'code', 'name'
    df = df.rename(columns=str.lower)
    code_col = "code" if "code" in df.columns else df.columns[0]
    name_col = "name" if "name" in df.columns else df.columns[1]
    out = pd.DataFrame({
        "code": df[code_col].astype(str).str.zfill(6),
        "name": df[name_col].astype(str),
    })
    return out


def get_stock_pool(cfg: dict) -> pd.DataFrame:
    """根据配置过滤得到目标股票池。

    返回 DataFrame[qlib_code, code, name, exchange], 已剔除 ST/北交所等。
    次新股(上市不足N日)在 dump 阶段按数据长度二次过滤。
    """
    cc = cfg["collector"]
    df = get_all_a_shares()

    # 交易所前缀
    df["exchange"] = df["code"].str[0].map({"6": "SH", "0": "SZ", "3": "SZ", "8": "BJ", "4": "BJ"})
    df = df[df["exchange"].notna()].copy()

    # 剔除北交所
    if cc.get("exclude_bj", True):
        df = df[df["exchange"] != "BJ"]
    # 剔除科创板 688
    if cc.get("exclude_kcb", False):
        df = df[~df["code"].str.startswith("688")]
    # 剔除创业板 300/301
    if cc.get("exclude_cyb", False):
        df = df[~df["code"].str.startswith(("300", "301"))]

    # 剔除 ST
    if cc.get("exclude_st", True):
        st = get_st_codes()
        before = len(df)
        df = df[~df["code"].isin(st)]
        log.info(f"剔除ST {before - len(df)} 只")

    df["qlib_code"] = df["code"].apply(to_qlib_code)
    df = df[["qlib_code", "code", "name", "exchange"]].reset_index(drop=True)
    log.info(f"股票池最终 {len(df)} 只 (SH {len(df[df.exchange=='SH'])} / SZ {len(df[df.exchange=='SZ'])})")
    return df


def get_pool_codes(cfg: dict) -> List[str]:
    """返回 qlib 代码列表。"""
    return get_stock_pool(cfg)["qlib_code"].tolist()

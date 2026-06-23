"""因子 IC/IR 分析: 评估单因子或模型预测值的预测能力。

IC (Information Coefficient): 因子值与未来收益的截面相关系数。
IR (Information Ratio): IC 均值 / IC 标准差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import setup_logger

log = setup_logger("strategy.ic")


def calc_ic(pred: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    """计算每日 IC (Spearman 秩相关)。

    pred, label: qlib 预测/标签, MultiIndex (datetime, instrument)。
    返回 DataFrame[index=date, columns=['IC']]。
    """
    # 对齐
    df = pred.copy()
    df.columns = ["pred"]
    lab = label.copy()
    lab.columns = ["label"]
    joined = df.join(lab, how="inner").dropna()

    ic_list = []
    for date, group in joined.groupby(level=0):
        if len(group) < 2:
            continue
        ic = group["pred"].corr(group["label"], method="spearman")
        ic_list.append({"date": date, "IC": ic})
    ic_df = pd.DataFrame(ic_list).set_index("date")
    return ic_df


def ic_summary(ic_df: pd.DataFrame) -> dict:
    """IC 统计摘要。"""
    if ic_df.empty:
        return {}
    ic = ic_df["IC"]
    return {
        "IC均值": round(ic.mean(), 4),
        "IC标准差": round(ic.std(), 4),
        "IR": round(ic.mean() / ic.std(), 4) if ic.std() > 0 else 0.0,
        "IC胜率": round((ic > 0).mean(), 4),
        "IC>0.02占比": round((ic.abs() > 0.02).mean(), 4),
        "样本天数": len(ic),
    }


def analyze(pred: pd.DataFrame, label: pd.DataFrame) -> tuple:
    """完整 IC 分析, 返回 (ic_df, summary)。"""
    ic_df = calc_ic(pred, label)
    summary = ic_summary(ic_df)
    log.info(f"IC分析: {summary}")
    return ic_df, summary

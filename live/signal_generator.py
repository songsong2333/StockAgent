"""信号生成: 用最新数据跑模型打分 -> 目标持仓。

两种模式:
  - 训练并预测: 用 train_and_predict, 取 test 段最后一日打分
  - 加载已训练模型预测最新: 推荐日常用, 避免每日重训
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from common import load_config, setup_logger, to_raw_code
from backtest.qlib_runner import train_and_predict

log = setup_logger("live.signal")


def generate_target_portfolio(cfg: dict, topk: int = None, weight_scheme: str = None) -> pd.DataFrame:
    """生成目标持仓。

    返回 DataFrame[qlib_code, code, name, score, weight]。
    """
    lc = cfg["live"]
    topk = topk or lc["topk"]
    weight_scheme = weight_scheme or lc.get("weight_scheme", "equal")

    _, model, pred = train_and_predict(cfg)

    # pred 可能是 Series 或 DataFrame, 统一成带 score 列的 Series (MultiIndex: datetime, instrument)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("score")

    # 取最新一个交易日的预测值
    if isinstance(pred.index, pd.MultiIndex):
        # 找到日期所在层(名为 datetime 或 level 0)
        date_level = "datetime" if "datetime" in pred.index.names else 0
        last_date = pred.index.get_level_values(date_level).max()
        latest = pred.xs(last_date, level=date_level).copy()
    else:
        last_date = pred.index.max()
        latest = pred.copy()
    # latest 是按 instrument 索引的 Series
    latest = latest.dropna().sort_values(ascending=False)

    # 取 topk (打分最高)
    top = latest.head(topk).to_frame("score").copy()
    top["qlib_code"] = top.index
    top["code"] = top["qlib_code"].apply(to_raw_code)

    # 权重分配
    if weight_scheme == "score":
        # 按打分正向加权(归一化到正数)
        s = top["score"] - top["score"].min() + 1e-6
        top["weight"] = (s / s.sum()).round(4)
    else:
        top["weight"] = round(1.0 / len(top), 4)

    # 附股票名称
    try:
        from collector.stock_pool import get_stock_pool
        names = get_stock_pool(cfg).set_index("qlib_code")["name"]
        top["name"] = top["qlib_code"].map(names).fillna("-")
    except Exception:
        top["name"] = "-"

    top = top[["qlib_code", "code", "name", "score", "weight"]].reset_index(drop=True)
    log.info(f"生成目标持仓 {len(top)} 只 (截至 {last_date})")
    return top


def save_portfolio(portfolio: pd.DataFrame, cfg: dict):
    """保存目标持仓到 cache。"""
    from common import ensure_dir
    path = Path(cfg["paths"]["cache_dir"]) / "target_portfolio.csv"
    ensure_dir(path.parent)
    portfolio.to_csv(path, index=False)
    log.info(f"目标持仓已保存: {path}")
    return path

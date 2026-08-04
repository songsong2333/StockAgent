"""因子封装: 基于 qlib Alpha158, 输出可复用的 handler/dataset 配置。

Alpha158 已内置 158 个量价因子(动量/反转/波动/量价/换手/形态等),
作为多因子选股的冷启动因子库。自定义A股因子可在 custom_factors.py 扩展。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import yaml

from common import load_config, PROJECT_ROOT, to_qlib_code


def load_universe(cfg: dict) -> Union[str, List[str]]:
    """读取股票池。返回 "all" 或 qlib 代码列表。"""
    uni = cfg.get("universe", "all")
    if uni == "all" or not uni:
        return "all"
    path = Path(uni)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    stocks = data.get("stocks", []) if isinstance(data, dict) else data
    return [to_qlib_code(s["code"]) for s in stocks]


def _handler_kwargs(cfg: dict, start_time: str, end_time: str,
                    fit_start: str, fit_end: str) -> dict:
    """构建 Alpha158 handler 的 kwargs(处理器/标签/股票池), 日期由调用方给定。

    回测(build_handler_config)与实盘推理(build_live_dataset_config)共用,
    保证两条链路的因子/归一化/标签口径完全一致。
    """
    m = cfg["model"]
    label = m.get("label", "Ref($close, -2) / Ref($close, -1) - 1")
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start,
        "fit_end_time": fit_end,
        "instruments": load_universe(cfg),   # watchlist 代码列表 或 "all"
        "infer_processors": [
            # 去极值 + 标准化
            {"class": "RobustZScoreNorm", "module_path": "qlib.data.dataset.processor",
             "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "module_path": "qlib.data.dataset.processor",
             "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel", "module_path": "qlib.data.dataset.processor"},
            # 截面排名归一化, 使 label 分布更稳
            {"class": "CSRankNorm", "module_path": "qlib.data.dataset.processor",
             "kwargs": {"fields_group": "label"}},
        ],
        "label": [label],
    }


def build_handler_config(cfg: dict) -> dict:
    """构建 Alpha158 handler 配置(回测用, 日期取自 config 的固定分段)。"""
    m = cfg["model"]
    # 因子预热: start_time 比 train_start 提前 ~6个月, 让滚动因子在训练起点已生效
    warmup_start = (pd.Timestamp(m["train_start"]) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    return _handler_kwargs(cfg, warmup_start, m["test_end"], m["train_start"], m["train_end"])


def build_dataset_config(cfg: dict) -> dict:
    """构建 DatasetH 配置 (handler + 分段)。"""
    m = cfg["model"]
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": build_handler_config(cfg),
            },
            "segments": {
                "train": (m["train_start"], m["train_end"]),
                "valid": (m["valid_start"], m["valid_end"]),
                "test": (m["test_start"], m["test_end"]),
            },
        },
    }


def build_live_dataset_config(cfg: dict, as_of) -> dict:
    """实盘推理数据集: 训练/验证窗滚动到 as_of 之前, test 段末端贴住 as_of。

    与回测的固定 test 段(build_dataset_config)解耦——回测要可复现, 实盘要反映最新数据。
    沿用 config 里训练/验证段的跨度长度, 整体右移使 test 末端 = as_of(最新可用交易日),
    这样模型预测的最新一天就是 as_of, 每日信号随数据更新而前移。

    as_of: 最新可用交易日(一般取 qlib 日历末日, 见 qlib_runner.latest_qlib_date)。
    """
    m = cfg["model"]
    fmt = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
    as_of = pd.Timestamp(as_of)
    # 训练/验证跨度沿用 config, 保证与回测同口径的训练量
    train_len = pd.Timestamp(m["train_end"]) - pd.Timestamp(m["train_start"])
    valid_len = pd.Timestamp(m["valid_end"]) - pd.Timestamp(m["valid_start"])
    test_end = as_of
    test_start = test_end - pd.Timedelta(days=14)     # 近两周窗口, 取末日预测即可
    valid_end = test_start - pd.Timedelta(days=1)
    valid_start = valid_end - valid_len
    train_end = valid_start - pd.Timedelta(days=1)
    train_start = train_end - train_len
    warmup_start = train_start - pd.DateOffset(months=6)   # 因子预热
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": _handler_kwargs(cfg, fmt(warmup_start), fmt(test_end),
                                          fmt(train_start), fmt(train_end)),
            },
            "segments": {
                "train": (fmt(train_start), fmt(train_end)),
                "valid": (fmt(valid_start), fmt(valid_end)),
                "test": (fmt(test_start), fmt(test_end)),
            },
        },
    }


def build_model_config(cfg: dict) -> dict:
    """构建模型配置: lgbm / linear。"""
    cls = cfg["model"].get("model_class", "lgbm")
    if cls == "lgbm":
        return {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "num_leaves": 128,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "colsample_bytree": 0.9,
                "subsample": 0.9,
                "lambda_l1": 0.5,
                "lambda_l2": 0.5,
            },
        }
    elif cls == "linear":
        return {
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {"loss": "mse"},
        }
    raise ValueError(f"未知模型: {cls}")


if __name__ == "__main__":
    import yaml
    print(yaml.safe_dump(build_handler_config(load_config()), allow_unicode=True))

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


def build_handler_config(cfg: dict) -> dict:
    """构建 Alpha158 handler 配置。"""
    m = cfg["model"]
    label = m.get("label", "Ref($close, -2) / Ref($close, -1) - 1")
    # 因子预热: start_time 比 train_start 提前 ~6个月, 让滚动因子在训练起点已生效
    warmup_start = (pd.Timestamp(m["train_start"]) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    return {
        "start_time": warmup_start,
        "end_time": m["test_end"],
        "fit_start_time": m["train_start"],
        "fit_end_time": m["train_end"],
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

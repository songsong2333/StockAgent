"""qlib 回测封装: 训练模型 + 回测, 内置A股规则(T+1/成本/涨跌停)。

使用 qlib.contrib.evaluate.backtest_daily 作为稳定公共API,
通过 kwargs 传入 A股 交易成本与涨跌停限制。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.evaluate import backtest_daily, risk_analysis

from common import load_config, setup_logger
from factor.handler import build_dataset_config, build_model_config

log = setup_logger("backtest.runner")

_INITIALIZED = False


def init_qlib(cfg: dict):
    """初始化 qlib (进程内只初始化一次)。"""
    global _INITIALIZED
    if _INITIALIZED:
        return
    qlib_dir = cfg["qlib"]["provider_uri"]
    p = Path(qlib_dir)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent.parent / p).resolve()
    if not (p / "calendars").exists():
        raise FileNotFoundError(
            f"qlib 数据未就绪: {p}/calendars 不存在, 请先运行 scripts/init_history.py"
        )
    qlib.init(provider_uri=str(p), region=REG_CN if cfg["qlib"]["region"] == "cn" else "us")
    _INITIALIZED = True
    log.info(f"qlib 初始化完成, provider_uri={p}")


def train_and_predict(cfg: dict):
    """训练模型并预测, 返回 (dataset, pred)。"""
    init_qlib(cfg)
    dataset = init_instance_by_config(build_dataset_config(cfg))
    model = init_instance_by_config(build_model_config(cfg))
    log.info("开始训练模型...")
    with R.start(experiment_name="multi_factor"):
        R.log_params(model_class=cfg["model"]["model_class"])
        model.fit(dataset)
        pred = model.predict(dataset)
    log.info(f"预测完成, {len(pred)} 条预测值")
    return dataset, model, pred


def run_backtest(cfg: dict, dataset=None, model=None) -> Tuple[dict, object, object]:
    """跑回测, 返回 (指标字典, report, positions)。

    若不传 dataset/model, 则先训练模型。
    使用 qlib 0.9.7 的 backtest_daily 新签名 (start/end/strategy/exchange_kwargs)。
    """
    if dataset is None or model is None:
        dataset, model, _ = train_and_predict(cfg)

    m = cfg["model"]
    bc = cfg["backtest"]
    # A股成本映射到 qlib: 买入=open_cost(佣金), 卖出=close_cost(佣金+印花税)
    open_cost = bc["commission_rate"]
    close_cost = bc["commission_rate"] + bc.get("stamp_duty", 0.001)
    log.info(f"回测参数: topk={bc['topk']}, n_drop={bc['n_drop']}, "
             f"买入成本={open_cost}, 卖出成本={close_cost}, deal_price={bc['deal_price']}")

    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": (model, dataset),
            "topk": bc["topk"],
            "n_drop": bc["n_drop"],
        },
    }
    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": 0.095,          # 涨跌停9.5%不可成交
        "deal_price": bc["deal_price"],    # 成交价 open/close/vwap
        "open_cost": open_cost,
        "close_cost": close_cost,
        "min_cost": bc.get("min_commission", 5),
    }

    report_normal, positions_normal = backtest_daily(
        start_time=m["test_start"],
        end_time=m["test_end"],
        strategy=strategy,
        account=1e7,                       # 初始资金 1000万
        benchmark=bc.get("benchmark", "SH000300"),  # 基准指数
        exchange_kwargs=exchange_kwargs,
    )

    # risk_analysis 接收日收益 Series, 返回 DataFrame(index=指标, columns=['risk'])
    risk = risk_analysis(report_normal["return"])
    metrics = risk["risk"].to_dict() if "risk" in risk.columns else risk.iloc[:, 0].to_dict()
    log.info(f"回测完成: {metrics}")
    return metrics, report_normal, positions_normal


if __name__ == "__main__":
    cfg = load_config()
    metrics, report, pos = run_backtest(cfg)
    from backtest.report import print_report
    print_report(metrics, report)

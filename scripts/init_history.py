"""首次拉取历史数据并生成 qlib 数据集。

用法:
  # 全市场近5年
  python scripts/init_history.py --years 5
  # 指定股票测试
  python scripts/init_history.py --symbols 000001,600000 --years 1
  # 仅 dump (数据已采集)
  python scripts/init_history.py --skip-collect
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本可从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common import load_config, setup_logger, to_qlib_code
from collector.stock_pool import get_stock_pool
from collector.daily_collector import update_all, update_one
from collector.dump_to_qlib import dump

log = setup_logger("scripts.init_history")


def parse_args():
    p = argparse.ArgumentParser(description="初始化历史数据")
    p.add_argument("--years", type=int, default=5, help="历史数据年数")
    p.add_argument("--symbols", type=str, default="", help="指定6位代码, 逗号分隔(留空=全市场)")
    p.add_argument("--skip-collect", action="store_true", help="跳过采集, 仅dump")
    p.add_argument("--skip-dump", action="store_true", help="跳过dump")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    # 覆盖配置的起始日期, 用命令行指定区间
    cfg["collector"]["start_date"] = start

    if args.symbols:
        codes = [to_qlib_code(s.strip()) for s in args.symbols.split(",") if s.strip()]
        log.info(f"指定股票 {len(codes)} 只: {codes}")
    else:
        pool = get_stock_pool(cfg)
        codes = pool["qlib_code"].tolist()
        log.info(f"全市场股票池 {len(codes)} 只")

    if not args.skip_collect:
        log.info(f"=== 开始采集 {start} ~ {end} ===")
        # 基准指数(回测需要)
        from collector.index_collector import update_benchmark_indices
        bench = cfg["backtest"].get("benchmark", "SH000300")
        update_benchmark_indices(cfg["paths"]["raw_dir"], start, end, [bench])
        if args.symbols:
            for c in codes:
                update_one(c, cfg["paths"]["raw_dir"], start, end,
                           cfg["collector"].get("adjust", "qfq"),
                           cfg["collector"]["max_retries"], cfg["collector"]["request_sleep"])
        else:
            update_all(cfg, codes=codes, end=end)
    else:
        log.info("跳过采集")

    if not args.skip_dump:
        log.info("=== 开始 dump 为 qlib bin ===")
        dump(cfg, codes=codes if args.symbols else None)

    log.info("初始化完成 ✅")


if __name__ == "__main__":
    main()

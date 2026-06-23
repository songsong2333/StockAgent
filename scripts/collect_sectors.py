"""按板块采集龙头股数据。

用法:
  python scripts/collect_sectors.py --start 2025-05-01 --end 2026-06-22
  python scripts/collect_sectors.py --start 2025-05-01 --no-dump   # 只采集不dump
  python scripts/collect_sectors.py --start 2025-05-01 --sectors 电力,PCB  # 指定板块
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from tqdm import tqdm

from common import load_config, setup_logger, to_qlib_code, PROJECT_ROOT
from collector.daily_collector import update_one
from collector.index_collector import update_benchmark_indices
from collector.dump_to_qlib import dump

log = setup_logger("scripts.collect_sectors")

SECTOR_FILE = PROJECT_ROOT / "config" / "sector_stocks.yaml"


def load_sectors(only: list = None) -> dict:
    with open(SECTOR_FILE, "r", encoding="utf-8") as f:
        sectors = yaml.safe_load(f)
    if only:
        sectors = {k: v for k, v in sectors.items() if k in only}
    return sectors


def main():
    p = argparse.ArgumentParser(description="按板块采集龙头股")
    p.add_argument("--start", default="2025-05-01", help="起始日期")
    p.add_argument("--end", default=None, help="结束日期, 默认今天")
    p.add_argument("--sectors", default="", help="指定板块(逗号分隔), 留空=全部")
    p.add_argument("--no-dump", action="store_true", help="不转qlib bin")
    args = p.parse_args()

    import pandas as pd
    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    cc = cfg["collector"]

    only = [s.strip() for s in args.sectors.split(",") if s.strip()] or None
    sectors = load_sectors(only)

    # 统计
    total = sum(len(v) for v in sectors.values())
    log.info(f"=== 板块采集 {args.start} ~ {end}, {len(sectors)}个板块/{total}只股票 ===")
    for k, v in sectors.items():
        log.info(f"  {k}: {len(v)}只")

    # 基准指数
    bench = cfg["backtest"].get("benchmark", "SH000300")
    update_benchmark_indices(raw_dir, args.start, end, [bench])

    # 逐板块逐股票采集
    ok, fail = 0, []
    for sector, stocks in sectors.items():
        log.info(f"--- 采集板块: {sector} ---")
        for s in tqdm(stocks, desc=sector):
            code = to_qlib_code(s["code"])
            try:
                if update_one(code, raw_dir, args.start, end,
                              cc.get("adjust", "qfq"), cc["max_retries"], cc["request_sleep"]):
                    ok += 1
                else:
                    fail.append(f"{s['code']} {s['name']} ({sector})")
            except Exception as e:
                fail.append(f"{s['code']} {s['name']} ({sector}): {e}")
                log.warning(f"{code} 采集异常: {e}")

    log.info(f"采集完成: 成功 {ok}/{total}")
    if fail:
        log.warning(f"失败 {len(fail)} 只: {fail}")

    if not args.no_dump:
        log.info("=== 转换为 qlib bin ===")
        # 只 dump 本次采集的股票 + 基准, 避免混入旧测试数据?
        # 这里 dump 全部 raw(含之前的), 保持数据集完整
        dump(cfg)

    log.info("=== 全部完成 ✅ ===")


if __name__ == "__main__":
    main()

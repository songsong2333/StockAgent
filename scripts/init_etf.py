"""首次拉取 ETF 历史数据(独立于个股 qlib 管线, 不 dump bin)。

用法:
  python scripts/init_etf.py --years 6                       # 全池
  python scripts/init_etf.py --codes 510300,518880 --years 3 # 指定
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本可从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common import load_config, setup_logger
from collector.etf_collector import update_all, pool_codes

log = setup_logger("scripts.init_etf")


def parse_args():
    p = argparse.ArgumentParser(description="初始化 ETF 历史数据(不走qlib)")
    p.add_argument("--years", type=int, default=6, help="历史数据年数")
    p.add_argument("--codes", type=str, default="", help="指定6位ETF代码, 逗号分隔(留空=全池)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    # 覆盖配置的起始日期, 用命令行指定区间
    cfg["collector"]["start_date"] = start

    if args.codes:
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
        log.info(f"指定 ETF {len(codes)} 只: {codes}")
    else:
        codes = pool_codes(cfg)
        log.info(f"ETF 池 {len(codes)} 只: {codes}")

    log.info(f"=== 开始采集 {start} ~ {end} (前复权, 不 dump qlib) ===")
    ok, failed = update_all(cfg, codes=codes, end=end)
    if failed:
        log.warning(f"采集失败的代码(代码错/退市/限流, 请核对 config/etf_pool.yaml): {failed}")
    log.info(f"初始化完成 ✅ 成功 {ok}/{len(codes)}")


if __name__ == "__main__":
    main()

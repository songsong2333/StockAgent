"""每日任务入口: 采集 -> dump -> 生成信号 -> 构建调仓清单 -> 通知。

用法:
  python scripts/run_daily.py             # 完整流程
  python scripts/run_daily.py --dry-run   # 不推送通知
  python scripts/run_daily.py --skip-collect  # 跳过采集, 用现有数据
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common import load_config, setup_logger
from collector.aux_collector import is_trade_date, latest_trade_date
from collector.daily_collector import update_all
from collector.dump_to_qlib import dump
from live.signal_generator import generate_target_portfolio, save_portfolio
from live.portfolio_builder import load_current_holdings, build_rebalance, to_markdown
from live.notify import notify
from backtest.report import save_report

log = setup_logger("scripts.run_daily")


def parse_args():
    p = argparse.ArgumentParser(description="每日任务")
    p.add_argument("--dry-run", action="store_true", help="只生成不推送")
    p.add_argument("--skip-collect", action="store_true", help="跳过采集")
    p.add_argument("--skip-dump", action="store_true", help="跳过dump")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    # 交易日判断
    if not is_trade_date(today):
        log.info(f"{today} 非交易日, 跳过")
        return
    trade_date = latest_trade_date(today)
    log.info(f"=== 每日任务开始, 最近交易日 {trade_date} ===")

    # 1. 采集
    if not args.skip_collect:
        log.info("[1/4] 增量采集日线...")
        from collector.index_collector import update_benchmark_indices
        bench = cfg["backtest"].get("benchmark", "SH000300")
        update_benchmark_indices(cfg["paths"]["raw_dir"], cfg["collector"]["start_date"], trade_date, [bench])
        update_all(cfg, end=trade_date)

    # 2. dump qlib
    if not args.skip_dump:
        log.info("[2/4] 转换为 qlib bin...")
        dump(cfg)

    # 3. 生成目标持仓
    log.info("[3/4] 生成目标持仓...")
    target = generate_target_portfolio(cfg)
    save_portfolio(target, cfg)

    # 4. 调仓清单 + 操作单 + 通知
    log.info("[4/4] 构建调仓清单与操作单...")
    current = load_current_holdings(cfg["live"]["holdings_file"])
    total_capital = current.get("cash", 0.0) + 1e7  # 简化: 现金 + 1000万假设
    orders, summary = build_rebalance(target, current, total_capital)
    md = to_markdown(orders, summary)

    # 操作单 (含买卖价/数量/止损/止盈)
    from live.order_sheet import get_latest_prices, build_order_sheet, to_markdown as order_md
    prices = get_latest_prices(target["qlib_code"].tolist(), cfg)
    sheet = build_order_sheet(target, current, total_capital, prices)
    sheet_md = order_md(sheet, trade_date, total_capital)

    # 保存信号文件
    cache_dir = Path(cfg["paths"]["cache_dir"])
    (cache_dir / "latest_signal.md").write_text(md, encoding="utf-8")
    (cache_dir / "latest_order_sheet.md").write_text(sheet_md, encoding="utf-8")
    log.info(f"调仓清单与操作单已保存: {cache_dir}")

    # 通知 (优先推送操作单, 更可操作)
    if args.dry_run or cfg["live"].get("dry_run", True):
        log.info("dry-run 模式, 不推送通知")
        print(sheet_md)
    else:
        notify(cfg, sheet_md, subject=f"A股量化操作单 {trade_date}")

    log.info("=== 每日任务完成 ===")


if __name__ == "__main__":
    main()

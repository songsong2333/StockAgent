"""成长板进攻信号 CLI: 采集最新指数 60min -> 输出创业板/科创50 的进/持/离/观望。

用法:
  python scripts/run_entry_exit.py               # 采集最新60min + 出信号
  python scripts/run_entry_exit.py --no-collect  # 用现有数据直接出信号(离线)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_config, setup_logger

log = setup_logger("scripts.run_entry_exit")


def parse_args():
    p = argparse.ArgumentParser(description="成长板进攻信号")
    p.add_argument("--no-collect", action="store_true", help="跳过采集, 用现有数据")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    if not args.no_collect:
        from collector.index_min_collector import update_all_min
        log.info("更新指数 60min 数据...")
        ok, failed = update_all_min(cfg)
        if failed:
            log.warning(f"部分指数采集失败: {failed}(用现有缓存出信号)")

    from strategy.entry_exit import generate_live_signals, signals_to_text
    signals = generate_live_signals(cfg)
    print("\n" + signals_to_text(signals) + "\n")


if __name__ == "__main__":
    main()

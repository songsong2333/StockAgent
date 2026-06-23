"""命令行回测入口。

用法:
  python scripts/run_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_config, setup_logger
from backtest.qlib_runner import run_backtest
from backtest.report import save_report, print_report

log = setup_logger("scripts.run_backtest")


def main():
    cfg = load_config()
    log.info("=== 开始回测 ===")
    metrics, report, positions = run_backtest(cfg)
    print_report(metrics, report)
    save_report(metrics, report, Path(cfg["paths"]["cache_dir"]))
    log.info("=== 回测完成 ===")


if __name__ == "__main__":
    main()

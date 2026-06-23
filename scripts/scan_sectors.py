"""运行低位启动板块扫描, 保存报告。

用法:
  python scripts/scan_sectors.py
  python scripts/scan_sectors.py --top 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_config, setup_logger, ensure_dir, PROJECT_ROOT
from strategy.breakout_scanner import scan, to_markdown, build_leader_template

log = setup_logger("scripts.scan_sectors")


def main():
    p = argparse.ArgumentParser(description="低位启动板块扫描")
    p.add_argument("--top", type=int, default=12, help="返回前N个板块")
    args = p.parse_args()
    cfg = load_config()

    log.info("=== 开始扫描 ===")
    tmpl_mean, _ = build_leader_template(cfg)
    res = scan(cfg, top_n=args.top)

    out = res.to_string(index=False)
    print("\n=== 扫描结果 ===")
    print(out)

    cache = ensure_dir(cfg["paths"]["cache_dir"])
    md_path = cache / "sector_breakout_scan.md"
    md_path.write_text(to_markdown(res, tmpl_mean), encoding="utf-8")
    (cache / "sector_breakout_scan.csv").write_text(res.to_csv(index=False), encoding="utf-8")
    log.info(f"报告已保存: {md_path}")


if __name__ == "__main__":
    main()

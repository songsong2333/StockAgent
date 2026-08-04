"""周期探索平台 —— 一键采集全分析宇宙(ETF/行业指数 + 港股 + 替代数据), 拉满历史。

与 init_etf 的区别: 多采港股(data/hk) + 替代数据(data/store: 宏观/能源/商品/气候),
且默认拉长历史(--years 15)以支撑 regime 条件化与多重检验(行业指数/宏观历史 10-17 年)。

用法:
  python scripts/init_cycle.py --years 15            # 全宇宙
  python scripts/init_cycle.py --only etf            # 仅 ETF 池
  python scripts/init_cycle.py --only hk,alt         # 仅港股+替代
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common import load_config, setup_logger

log = setup_logger("scripts.init_cycle")


def parse_args():
    p = argparse.ArgumentParser(description="初始化周期探索平台数据(ETF+港股+替代)")
    p.add_argument("--years", type=int, default=15, help="ETF/指数历史年数(长历史支撑regime)")
    p.add_argument("--only", type=str, default="etf,hk,alt",
                   help="采集范围, 逗号分隔: etf / hk / alt")
    return p.parse_args()


def collect_etf(cfg, start, end):
    from collector.etf_collector import update_all, pool_codes
    codes = pool_codes(cfg)
    log.info(f"[ETF] 池 {len(codes)} 只: {codes}")
    ok, failed = update_all(cfg, codes=codes, end=end)
    if failed:
        log.warning(f"[ETF] 失败(代码错/退市/限流): {failed}")
    return ok, len(codes)


def collect_hk(cfg):
    from collector.hk_collector import update_all
    ok, failed = update_all(cfg)
    if failed:
        log.warning(f"[HK] 失败: {failed}")
    return ok, ok + len(failed)


def collect_alt(cfg):
    from collector.alt_collector import update_all_alt
    ok, failed = update_all_alt(cfg)
    if failed:
        log.warning(f"[ALT] 失败: {failed}")
    return ok, ok + len(failed)


def main():
    args = parse_args()
    cfg = load_config()

    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    scope = [s.strip() for s in args.only.split(",") if s.strip()]
    log.info(f"=== 周期平台数据初始化 {start} ~ {end}, 范围={scope} ===")

    # ETF 用长历史(行业指数代理有 10-17 年), 覆盖 cfg 的 start_date
    if "etf" in scope:
        cfg_etf = dict(cfg)
        cfg_etf["collector"] = dict(cfg["collector"])
        cfg_etf["collector"]["start_date"] = start
        collect_etf(cfg_etf, start, end)
    if "hk" in scope:
        collect_hk(cfg)
    if "alt" in scope:
        collect_alt(cfg)

    # 汇总: universe_panel 实读一遍, 看实际可用序列 + inception
    log.info("=== 汇总: 实际可用序列(按 inception) ===")
    try:
        from collector.store import universe_panel
        panel, names, meta = universe_panel(cfg)
        if panel.empty:
            log.warning("无数据被加载, 检查采集日志。")
            return
        rows = sorted(meta.items(), key=lambda kv: kv[1].get("inception") or "9")
        for sid, m in rows:
            log.info(f"  {sid:10s} {names.get(sid,''):14s} cat={m['category']:8s} "
                     f"kind={m['kind']:5s} n={m['n_obs']:4d} since={m['inception']}")
        log.info(f"共 {panel.shape[1]} 序列 × {panel.shape[0]} 月, "
                 f"区间 {panel.index.min().date()}~{panel.index.max().date()}")
    except Exception as e:
        log.warning(f"汇总失败: {e}")
    log.info("初始化完成 ✅")


if __name__ == "__main__":
    main()

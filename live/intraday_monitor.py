"""盘中盯盘预警 (Pro)。守护进程: 交易时段轮询实时行情, 触发预警推送。

设计:
  - 数据: ak.stock_zh_a_spot_em() 一次取全市场实时, 过滤监控池; 60s 缓存 + 重试。
  - 监控池: watchlist 龙头 + 可选当日涨停股。
  - 规则: 接近涨停(涨>=9%) / 封板(>=9.7%) / 炸板(曾封板现回落到<7%)。
  - 冷静期: 同(code,事件) 10 分钟内不重复推。
  - 风控门槛: 情绪温度<30(退潮)时暂停推送, 防上头。
  - 推送: 复用 live.notify (webhook/邮件) + 写 data/cache/intraday_alerts.log。

运行 (Pro 激活后, 后台或 launchd):
  python -m live.intraday_monitor            # 守护, 仅交易时段
  python -m live.intraday_monitor --test     # 非交易时段强制跑一轮+测试推送
  python -m live.intraday_monitor --interval 30
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List

import akshare as ak
import pandas as pd

from common import load_config, setup_logger, to_qlib_code, PROJECT_ROOT
from collector.aux_collector import latest_closed_trade_date

log = setup_logger("live.intraday")
ALERT_LOG = Path("data/cache/intraday_alerts.log")
COOLDOWN_SEC = 600          # 同事件 10 分钟冷却
NEAR_ZT = 9.0               # 接近涨停 %
LOCK_ZT = 9.7               # 封板 %
BREAKDOWN = 7.0             # 曾封板回落到此 = 炸板


def _watch_codes(cfg: dict) -> List[str]:
    """监控池: watchlist 龙头代码(6位)。"""
    import yaml
    with open(PROJECT_ROOT / "config" / "watchlist.yaml", "r", encoding="utf-8") as f:
        wl = yaml.safe_load(f)["stocks"]
    return [s["code"] for s in wl]


def fetch_spot(codes: List[str]) -> pd.DataFrame:
    """取实时行情, 过滤监控池(带重试, spot 接口偶发超时)。"""
    df = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            log.warning(f"spot 获取第{attempt+1}次失败: {str(e)[:60]}")
            time.sleep(3 * (attempt + 1))
    if df is None or df.empty:
        raise RuntimeError("实时行情多次获取失败(可能非交易时段或被限流)")
    code_col = "代码" if "代码" in df.columns else df.columns[1]
    name_col = "名称" if "名称" in df.columns else df.columns[2]
    chg_col = next((c for c in df.columns if "涨跌幅" in str(c)), None)
    price_col = next((c for c in df.columns if "最新价" in str(c)), None)
    df = df[df[code_col].isin(codes)].copy()
    out = pd.DataFrame({"code": df[code_col], "name": df[name_col]})
    if price_col:
        out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    if chg_col:
        out["pct"] = pd.to_numeric(df[chg_col], errors="coerce")
    return out


def is_market_hours(now: datetime = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 25) <= t <= dtime(15, 5)


def _alert(cfg: dict, code: str, name: str, event: str, pct: float, price: float,
           last_alert: Dict, log_fh) -> bool:
    """发预警(带冷却)。返回是否真的发了。"""
    key = (code, event)
    now = time.time()
    if now - last_alert.get(key, 0) < COOLDOWN_SEC:
        return False
    last_alert[key] = now
    emoji = {"接近涨停": "🟡", "封板": "🔴", "炸板": "🟢"}.get(event, "⚠️")
    line = f"[{datetime.now():%H:%M:%S}] {emoji} {code} {name} {event} 涨幅{pct:+.1f}% 现价{price}"
    log.info(line)
    log_fh.write(line + "\n"); log_fh.flush()
    try:
        from live.notify import notify
        notify(cfg, f"⚡盘中预警\n{line}", subject=f"盘中预警 {code} {event}")
    except Exception as e:
        log.warning(f"推送失败: {e}")
    return True


def run_once(cfg: dict, codes: List[str], state: dict, log_fh, force: bool = False) -> int:
    """跑一轮: 取行情, 检测, 预警。返回触发的预警数。"""
    if not force and not is_market_hours():
        return 0
    # 风控门槛: 情绪退潮暂停推送(但仍记录)
    try:
        from strategy.short_term import market_sentiment
        sent = market_sentiment(latest_closed_trade_date().replace("-", ""))
        if sent["temp"] < 30 and not force:
            log.info(f"情绪温度 {sent['temp']}<30 退潮, 本轮暂停推送")
            return 0
    except Exception as e:
        log.debug(f"情绪读取失败(忽略): {e}")

    try:
        spot = fetch_spot(codes)
    except Exception as e:
        log.warning(f"行情获取失败: {e}")
        return 0
    n = 0
    was_up = state.setdefault("was_limit_up", {})
    for _, r in spot.iterrows():
        code, name, pct, price = r["code"], r["name"], r.get("pct"), r.get("price")
        if pd.isna(pct):
            continue
        pct = float(pct)
        prev_locked = was_up.get(code, False)
        if pct >= LOCK_ZT:
            was_up[code] = True
            if _alert(cfg, code, name, "封板", pct, price, state.setdefault("last", {}), log_fh):
                n += 1
        elif pct >= NEAR_ZT:
            if _alert(cfg, code, name, "接近涨停", pct, price, state.setdefault("last", {}), log_fh):
                n += 1
        elif prev_locked and pct < BREAKDOWN:
            was_up[code] = False
            if _alert(cfg, code, name, "炸板", pct, price, state.setdefault("last", {}), log_fh):
                n += 1
        elif pct < NEAR_ZT:
            was_up[code] = False
    return n


def run_foreground(cfg: dict, interval: int = 60, test: bool = False):
    codes = _watch_codes(cfg)
    log.info(f"盘中预警启动, 监控 {len(codes)} 只, 间隔 {interval}s" + (" (测试模式)" if test else ""))
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    with open(ALERT_LOG, "a", encoding="utf-8") as log_fh:
        if test:
            n = run_once(cfg, codes, state, log_fh, force=True)
            log.info(f"测试轮完成, 触发 {n} 条预警")
            return
        while True:
            try:
                run_once(cfg, codes, state, log_fh)
            except KeyboardInterrupt:
                log.info("收到退出信号, 停止"); break
            except Exception as e:
                log.error(f"轮询异常: {e}")
            time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description="盘中盯盘预警 (Pro)")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--test", action="store_true", help="非交易时段强制跑一轮(测试推送)")
    args = p.parse_args()
    cfg = load_config()
    from paywall import is_pro
    if not is_pro():
        print("⛔ 需要 Pro 激活。用 scripts/gen_license.py 生成码并在 App 内激活。")
        return
    run_foreground(cfg, interval=args.interval, test=args.test)


if __name__ == "__main__":
    main()

"""周期探索平台 —— 统一序列存储与读取抽象。

核心理念: 一切皆时间序列。ETF / 行业指数 / 港股 / 商品 / 宏观 / 气候 统一成
``(id, level_series)`` 宽面板, 后续 seasonality / discovery / regime / attribution
模块只认这个抽象, 不关心数据来源。这是"探索"的地基 —— 任何序列都能扔进同一池子扫描。

数据落位(由各 collector 写入, 本模块只读):
  data/etf/<code>.parquet   ETF + 行业指数(日 OHLCV, 列含 close)
  data/hk/<id>.parquet      港股(日 OHLCV, 列含 close)
  data/store/<id>.parquet   替代数据(月频单值, 列 [date, value])

universe_panel() 把多类序列对齐到同一频率(月/周), 返回 level 面板 + 名称 + 元信息
(category/kind/inception/n)。kind 决定分析层如何做变换: price→用收益(pct_change),
rate/level→用其本身或差分(驱动归因时由调用方显式选择, 这里只如实暴露 level)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from common import setup_logger, load_config, PROJECT_ROOT

log = setup_logger("collector.store")

# 日期列候选名(不同来源命名不一)
_DATE_COLS = ("date", "日期", "统计时间", "TRADE_DATE", "datetime", "time")


def _resolve_dir(cfg: dict, key: str, default: str) -> Path:
    d = cfg.get("paths", {}).get(key, default)
    p = Path(d)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _resample(s: pd.Series, freq: str) -> pd.Series:
    """重采样到目标频率: 日频价取期末最后一值; 已是低频则取期末对齐。

    M → 月末(ME), W → 周五(W-FRI)。价格取 last; 对已是月频的 rate 同样取 last 对齐。
    """
    s = s.dropna()
    if s.empty:
        return s
    idx = pd.to_datetime(s.index, errors="coerce")
    s = pd.Series(s.values, index=idx).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if freq == "M":
        return s.resample("ME").last()
    if freq == "W":
        return s.resample("W-FRI").last()
    return s


def _load_parquet_series(path: Path, value_col: str = "close") -> Optional[pd.Series]:
    """读单只 parquet → Series(索引 date, 值=value_col)。失败/缺列返回 None。"""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.warning(f"读取 {path.name} 失败: {e}")
        return None
    if df is None or df.empty:
        return None
    # 定位日期列
    date_col = next((c for c in _DATE_COLS if c in df.columns), df.columns[0])
    # 定位值列
    col = value_col if value_col in df.columns else next(
        (c for c in ("close", "value", "收盘", "price") if c in df.columns), df.columns[-1])
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(df[date_col], errors="coerce")
    return s.dropna()


def _meta_from(s: pd.Series, category: str) -> dict:
    """从序列算 inception/n_obs。"""
    valid = s.dropna()
    if valid.empty:
        return {"inception": None, "n_obs": 0}
    return {"inception": valid.index.min().strftime("%Y-%m-%d"),
            "n_obs": int(len(valid))}


def _alt_kind(item: dict, category: str) -> str:
    """替代序列的 kind: price(用收益) vs rate(用水平/差分)。

    商品=price; 宏观/气候=rate; 能源里油价=price、用电量同比=rate。
    """
    sid = str(item.get("id", ""))
    if category == "commodity":
        return "price"
    if category == "energy" and "price" in sid:
        return "price"
    return "rate"


def _load_alt_registry(cfg: dict) -> dict:
    """读 config/alt_sources.yaml → {category: [{id,name,source,...}]}。"""
    p = Path(cfg.get("cycle", {}).get("alt_file", "config/alt_sources.yaml"))
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============== 主入口: 对齐多类序列成面板 ==============
def universe_panel(cfg: dict, categories: Optional[list] = None,
                   freq: Optional[str] = None):
    """读取多类序列, 对齐成 level 宽面板。

    返回 (panel, names, meta):
      panel  DataFrame[index=日期(freq对齐), columns=id] = 序列水平值(价格点位/利率/ONI等)
      names  {id: 显示名}
      meta   {id: {category, kind, inception, n_obs}}

    缺失序列(未采集)静默跳过, 不阻塞分析(增量可用)。
    """
    cc = cfg.get("cycle", {})
    categories = categories or cc.get("universe_categories", ["broad", "sector", "seasonal"])
    freq = freq or cc.get("freq", "M")

    series, names, meta = {}, {}, {}
    pool_groups = {"broad", "sector", "seasonal"}

    # 1) ETF 池组(ETF + 行业指数, 皆 data/etf)
    etf_cats = [c for c in categories if c in pool_groups]
    if etf_cats:
        try:
            from collector.etf_collector import load_etf_pool
            pool = load_etf_pool(cfg)
        except Exception as e:
            log.warning(f"读 etf_pool 失败: {e}")
            pool = {}
        etf_dir = _resolve_dir(cfg, "etf_dir", "data/etf")
        for grp in etf_cats:
            for it in pool.get(grp, []) or []:
                code = str(it.get("code", "")).strip()
                if not code:
                    continue
                s = _load_parquet_series(etf_dir / f"{code}.parquet", "close")
                if s is None:
                    continue
                series[code] = _resample(s, freq)
                names[code] = it.get("name", code)
                meta[code] = {"category": grp, "kind": "price", **_meta_from(s, grp)}

    # 2) 港股(data/hk)
    if "hk" in categories:
        hk_dir = _resolve_dir(cfg, "hk_dir", "data/hk")
        alt_names = {it["id"]: it.get("name", it["id"])
                     for it in _load_alt_registry(cfg).get("hk", []) or []}
        for p in sorted(hk_dir.glob("*.parquet")):
            s = _load_parquet_series(p, "close")
            if s is None:
                continue
            sid = p.stem
            series[sid] = _resample(s, freq)
            names[sid] = alt_names.get(sid, sid)
            meta[sid] = {"category": "hk", "kind": "price", **_meta_from(s, "hk")}

    # 3) 替代数据(data/store): macro/energy/commodity/climate
    alt_cats = [c for c in categories if c in ("macro", "energy", "commodity", "climate")]
    if alt_cats:
        reg = _load_alt_registry(cfg)
        store_dir = _resolve_dir(cfg, "store_dir", "data/store")
        for grp in alt_cats:
            for it in reg.get(grp, []) or []:
                sid = str(it.get("id", "")).strip()
                if not sid:
                    continue
                s = _load_parquet_series(store_dir / f"{sid}.parquet", "value")
                if s is None:
                    continue
                series[sid] = _resample(s, freq)
                names[sid] = it.get("name", sid)
                meta[sid] = {"category": grp, "kind": _alt_kind(it, grp), **_meta_from(s, grp)}

    if not series:
        log.warning("universe_panel: 无可用序列(可能尚未采集)。先跑 scripts/init_cycle.py。")
        return pd.DataFrame(), {}, {}

    panel = pd.DataFrame(series).sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    log.info(f"universe_panel: {panel.shape[1]} 序列 × {panel.shape[0]} {freq} 期, "
             f"区间 {panel.index.min().date()}~{panel.index.max().date()}")
    return panel, names, meta


def returns_of(panel: pd.DataFrame, ids: Optional[list] = None) -> pd.DataFrame:
    """价格类面板的收益率(pct_change)。仅对 price 序列有意义; rate 列调用方自负。"""
    sub = panel[ids] if ids else panel
    return sub.pct_change(fill_method=None)


def ids_of_kind(meta: dict, kind: str) -> list:
    """按 kind 过滤序列 id(price/rate)。"""
    return [sid for sid, m in meta.items() if m.get("kind") == kind]


if __name__ == "__main__":
    cfg = load_config()
    panel, names, meta = universe_panel(cfg)
    if panel.empty:
        print("无数据。先跑: python scripts/init_cycle.py")
    else:
        print(f"{panel.shape[1]} 序列, 样例:")
        print(panel.tail(3).round(2))
        print("\ninception 分布:")
        for sid, m in list(meta.items())[:8]:
            print(f"  {sid:10s} {names.get(sid,''):12s} cat={m['category']:8s} "
                  f"kind={m['kind']:5s} n={m['n_obs']:4d} since={m['inception']}")

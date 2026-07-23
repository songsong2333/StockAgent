"""场内宽基/行业 ETF 策略: 趋势择时 + 动量轮动。

两个策略, 共用自写的精确佣金回测引擎(资金+手数模型, 最低5元/笔惩罚):
  - 趋势择时 (宽基+避险池): 双均线 + 防抖(连续确认/带宽过滤/状态保持), 均线上持有、下穿空仓。
  - 动量轮动: 宽基池纯轮动; 行业池叠大盘择时开关(基准ETF均线下空仓避险) + 短线动量加权。

ETF 不走 qlib, 全程原始6位代码, 数据在 data/etf/。与 strategy.rotation 解耦,
仅风格对齐(信号返回 dict + signal_to_markdown; 回测返回含 curve/dates/daily_ret 的 dict,
App 用 metrics_from_returns 缩仓)。回测主循环严格"t-1 收盘信号 → t 日成交", 避未来函数。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from common import setup_logger, load_config, PROJECT_ROOT

log = setup_logger("strategy.etf")


# ============== 数据加载 ==============
def _resolve_etf_dir(cfg: dict) -> Path:
    etf_dir = cfg["paths"].get("etf_dir", "data/etf")
    p = Path(etf_dir)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_etf_panel(cfg: dict, pool: str = "broad"):
    """加载 ETF 收盘/开盘面板 + 收益面板 + 名称映射 + 大盘基准序列。

    pool: "broad" 或 "sector"。返回 (close, open_px, ret, names, bench)。
    面板列=原始6位code, 行=日期(允许含NaN, 上市时间不同)。bench 为基准ETF收盘 Series。
    """
    from collector.etf_collector import load_etf_pool
    data = load_etf_pool(cfg)
    items = data.get(pool, []) or []
    bench_code = str(data.get("benchmark_etf", "510300")).strip().zfill(6)
    codes = [it["code"] for it in items]
    if bench_code not in codes:            # 基准也要加载(即使不在当前 pool)
        codes = [bench_code] + codes
    names = {it["code"]: it.get("name", it["code"]) for it in items}
    names.setdefault(bench_code, "基准ETF")

    etf_dir = _resolve_etf_dir(cfg)
    closes, opens = {}, {}
    for c in codes:
        p = etf_dir / f"{c}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date").sort_index()
        closes[c] = s["close"].astype(float)
        opens[c] = s["open"].astype(float) if "open" in s.columns else s["close"].astype(float)
    close = pd.DataFrame(closes).sort_index().dropna(how="all")
    open_px = pd.DataFrame(opens).sort_index().reindex(index=close.index, columns=close.columns)
    ret = close.pct_change()
    bench = close[bench_code] if bench_code in close.columns else pd.Series(dtype=float)
    return close, open_px, ret, names, bench


# ============== 防抖状态机 ==============
def _debounce(raw: pd.Series, confirm_days: int, band_filter: float,
              ma_short_s: pd.Series, ma_long_s: pd.Series) -> pd.Series:
    """双均线信号防抖: 需连续 confirm_days 日确认 + 均线带宽>=band_filter 才翻转。

    返回 0/1 持仓状态序列(与 raw 同 index, 无未来函数: 第 i 日只用 <=i 的信息)。
    - raw: 原始多空(ma_short>ma_long → 1)
    - bull_trigger: raw 连续 N 日为1 且带宽足够 → 进入多头
    - bear_trigger: raw 连续 N 日为0 且带宽足够 → 退出多头
    - 中间状态保持(状态机), 避免均线缠绕区反复打脸
    """
    bull_trig = (raw.rolling(confirm_days).sum() == confirm_days)
    bear_trig = (raw.rolling(confirm_days).sum() == 0)
    band = (ma_short_s - ma_long_s).abs() / ma_long_s.replace(0, np.nan)
    clear = band >= band_filter
    bull_trig = bull_trig & clear
    bear_trig = bear_trig & clear
    hold = pd.Series(0, index=raw.index, dtype=int)
    state = 0
    v_bull, v_bear, v_malong = bull_trig.values, bear_trig.values, ma_long_s.values
    for i in range(len(raw)):
        if pd.isna(v_malong[i]):          # 长均线未就绪 → 空仓
            state = 0
        elif v_bull[i]:
            state = 1
        elif v_bear[i]:
            state = 0
        hold.iloc[i] = state
    return hold


# ============== 指标 ==============
def _mdd(curve: np.ndarray) -> float:
    if len(curve) < 2:
        return 0.0
    return float((curve / np.maximum.accumulate(curve) - 1).min())


def _sharpe(curve: np.ndarray, periods: int = 252) -> float:
    if len(curve) < 3:
        return 0.0
    r = np.diff(np.log(curve))
    return float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0


def _empty_result() -> dict:
    z = np.array([])
    return {"cum_ret": 0.0, "bench_cum": 0.0, "max_dd": 0.0, "bench_dd": 0.0,
            "sharpe": 0.0, "bench_sharpe": 0.0, "n_days": 0, "curve": z,
            "bench_curve": z, "dates": [], "daily_ret": z, "bench_daily": z,
            "n_trades": 0, "total_commission": 0.0, "turnover_annual": 0.0,
            "trade_log": pd.DataFrame()}


def _summarize(equity: list, dates: list, trade_log: list, bench: pd.Series,
               capital: float) -> dict:
    eq = np.array(equity, dtype=float)
    curve = eq / capital if capital else eq
    daily_ret = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([])
    # 基准 buy&hold, 对齐回测日期
    b = bench.reindex(pd.to_datetime(dates)) if len(bench) else pd.Series(dtype=float)
    b = b.ffill().bfill()
    if len(b) and b.iloc[0] > 0:
        bench_curve = (b / b.iloc[0]).values
    else:
        bench_curve = np.ones(len(curve))
    bench_daily = np.diff(bench_curve) / bench_curve[:-1] if len(bench_curve) > 1 else np.array([])
    # 换手率 = 总成交金额 / 平均持仓市值 / 年数
    n_years = max(len(dates) / 252, 1e-9)
    total_amount = sum(t["amount"] for t in trade_log)
    avg_eq = float(np.mean(eq)) if len(eq) else float(capital)
    turnover = total_amount / avg_eq / n_years if avg_eq > 0 else 0.0
    return {
        "cum_ret": float(curve[-1] - 1) if len(curve) else 0.0,
        "bench_cum": float(bench_curve[-1] - 1) if len(bench_curve) else 0.0,
        "max_dd": _mdd(curve), "bench_dd": _mdd(bench_curve),
        "sharpe": _sharpe(curve), "bench_sharpe": _sharpe(bench_curve),
        "n_days": len(curve), "curve": curve, "bench_curve": bench_curve,
        "dates": dates, "daily_ret": daily_ret, "bench_daily": bench_daily,
        "n_trades": len(trade_log),
        "total_commission": float(sum(t["commission"] for t in trade_log)),
        "turnover_annual": float(turnover),
        "trade_log": pd.DataFrame(trade_log),
    }


# ============== 成交: 先卖后买, 精确佣金 ==============
def _apply_target(holdings: dict, cash: float, target_codes: list, deal_row: pd.Series,
                  total_equity: float, commission_rate: float, min_commission: float,
                  trade_unit: int, date_now, trade_log: list):
    """按 target_codes 调仓: 卖出不在目标的, 等权整手买入新目标。逐笔 max(amount*率, 最低)。"""
    target_set = set(target_codes)
    # 1. 卖出不在 target
    for c in list(holdings.keys()):
        if c in target_set:
            continue
        px = deal_row.get(c)
        if px is None or pd.isna(px) or px <= 0:
            continue                      # 停牌 → 跳过, 持仓保留
        shares = holdings[c]
        amount = shares * px
        comm = max(amount * commission_rate, min_commission)
        cash += amount - comm
        trade_log.append({"date": date_now, "code": c, "side": "卖出",
                          "shares": int(shares), "price": float(px),
                          "amount": float(amount), "commission": float(comm)})
        del holdings[c]
    # 2. 买入新 target (等权, 整手)
    n = max(len(target_codes), 1)
    target_value = total_equity / n
    for c in target_codes:
        if c in holdings:
            continue
        px = deal_row.get(c)
        if px is None or pd.isna(px) or px <= 0:
            continue
        shares = int(target_value / px // trade_unit) * trade_unit
        if shares <= 0:
            continue
        amount = shares * px
        comm = max(amount * commission_rate, min_commission)
        if cash < amount + comm:          # 现金不足, 按可用资金重算
            shares = int((cash - min_commission) / px // trade_unit) * trade_unit
            if shares <= 0:
                continue
            amount = shares * px
            comm = max(amount * commission_rate, min_commission)
        cash -= amount + comm
        holdings[c] = shares
        trade_log.append({"date": date_now, "code": c, "side": "买入",
                          "shares": int(shares), "price": float(px),
                          "amount": float(amount), "commission": float(comm)})
    return holdings, cash


def _cost_params(cfg: dict):
    bc = cfg["backtest"]
    return (bc["commission_rate"], bc["min_commission"],
            bc.get("trade_unit", 100), bc.get("deal_price", "open"))


def _equity(holdings: dict, cash: float, close_row: pd.Series) -> float:
    return cash + sum(sh * close_row[c] for c, sh in holdings.items()
                      if c in close_row and not pd.isna(close_row[c]))


# ============== 策略1: 趋势择时回测 ==============
def backtest_trend(cfg: dict, ma_short: Optional[int] = None, ma_long: Optional[int] = None,
                   confirm_days: Optional[int] = None, band_filter: Optional[float] = None,
                   freq: Optional[int] = None, pool: str = "broad",
                   start: Optional[str] = None, capital: Optional[float] = None) -> dict:
    """宽基+避险池的双均线趋势择时回测(精确佣金)。每只ETF独立择时, 等权持有信号=1的标的。"""
    ec = cfg.get("etf", {}); tc = ec.get("trend", {})
    ma_short = 20 if ma_short is None else ma_short
    ma_long = 60 if ma_long is None else ma_long
    confirm_days = tc.get("confirm_days", 2) if confirm_days is None else confirm_days
    band_filter = tc.get("band_filter", 0.005) if band_filter is None else band_filter
    freq = tc.get("rebalance_freq", 5) if freq is None else freq
    start = start or ec.get("backtest_start", "2019-01-01")
    capital = capital or ec.get("initial_capital", 1_000_000)
    commission_rate, min_commission, trade_unit, deal_price = _cost_params(cfg)

    close, open_px, ret, names, bench = load_etf_panel(cfg, pool)
    codes = list(close.columns)
    close = close[close.index >= start]
    open_px = open_px.reindex(close.index)
    dates = close.index.tolist()
    if len(dates) < ma_long + 5:
        return _empty_result()

    # 预算每只ETF的防抖信号(全序列, 无未来函数)
    signals = {}
    for c in codes:
        ma_s = close[c].rolling(ma_short).mean()
        ma_l = close[c].rolling(ma_long).mean()
        signals[c] = _debounce((ma_s > ma_l).astype(int), confirm_days, band_filter, ma_s, ma_l)

    holdings, cash = {}, float(capital)
    equity_curve, dates_out, trade_log = [], [], []
    deal = open_px if deal_price == "open" else close

    for i in range(1, len(dates)):
        d_prev, d_now = dates[i - 1], dates[i]
        if i % freq == 0:                 # 调仓日: 昨收信号 → 今日成交
            target = [c for c in codes if signals[c].loc[d_prev] == 1
                      and not pd.isna(close[c].loc[d_prev])]
            total_eq = _equity(holdings, cash, close.loc[d_prev])
            holdings, cash = _apply_target(holdings, cash, target, deal.loc[d_now],
                                           total_eq, commission_rate, min_commission,
                                           trade_unit, d_now, trade_log)
        equity_curve.append(_equity(holdings, cash, close.loc[d_now]))
        dates_out.append(d_now)

    return _summarize(equity_curve, dates_out, trade_log, bench, capital)


# ============== 策略2: 动量轮动回测 ==============
def backtest_rotation(cfg: dict, pool: str = "broad", topn: Optional[int] = None,
                      mom_short: Optional[int] = None, mom_long: Optional[int] = None,
                      market_timing: Optional[bool] = None, bench_ma: Optional[int] = None,
                      confirm_days: Optional[int] = None, band_filter: Optional[float] = None,
                      freq: Optional[int] = None, start: Optional[str] = None,
                      capital: Optional[float] = None) -> dict:
    """动量轮动回测(精确佣金)。

    pool="broad": 纯轮动(评分=mom_long/vol60, 趋势过滤)。
    pool="sector": 行业轮动 + 大盘择时开关(benchmark均线下空仓避险); 评分加短线动量横截面zscore。
    """
    ec = cfg.get("etf", {}); rc = ec.get("rotation", {})
    if topn is None:
        topn = rc.get("sector_topn", 3) if pool == "sector" else rc.get("broad_topn", 3)
    mom_short = rc.get("mom_short", 20) if mom_short is None else mom_short
    mom_long = rc.get("mom_long", 60) if mom_long is None else mom_long
    bench_ma = rc.get("bench_ma", 60) if bench_ma is None else bench_ma
    confirm_days = ec.get("trend", {}).get("confirm_days", 2) if confirm_days is None else confirm_days
    band_filter = ec.get("trend", {}).get("band_filter", 0.005) if band_filter is None else band_filter
    freq = rc.get("rebalance_freq", 5) if freq is None else freq
    if market_timing is None:
        market_timing = rc.get("sector_market_timing", True) if pool == "sector" else False
    start = start or ec.get("backtest_start", "2019-01-01")
    capital = capital or ec.get("initial_capital", 1_000_000)
    commission_rate, min_commission, trade_unit, deal_price = _cost_params(cfg)
    bench_code = str(ec.get("benchmark_etf", "510300")).strip().zfill(6)

    close, open_px, ret, names, bench = load_etf_panel(cfg, pool)
    codes = [c for c in close.columns if c != bench_code]
    close = close[close.index >= start]
    open_px = open_px.reindex(close.index)
    ret = ret.reindex(close.index)
    dates = close.index.tolist()
    if len(dates) < max(mom_long, bench_ma) + 5 or not codes:
        return _empty_result()

    # 评分面板(向量化)
    ma20 = close[codes].rolling(20).mean()
    ma60 = close[codes].rolling(60).mean()
    uptrend = ma20 > ma60
    mom_l = close[codes] / close[codes].shift(mom_long) - 1
    mom_s = close[codes] / close[codes].shift(mom_short) - 1
    vol60 = ret[codes].rolling(60).std()
    base = mom_l / vol60
    if pool == "sector":                  # 行业加短线动量横截面zscore
        z = mom_s.sub(mom_s.mean(axis=1), axis=0).div(mom_s.std(axis=1).replace(0, np.nan), axis=0)
        score = 0.6 * base + 0.4 * z
    else:
        score = base

    # 大盘择时开关(sector + market_timing): 基准ETF双均线防抖
    if pool == "sector" and market_timing and bench_code in close.columns:
        bs = close[bench_code]
        bmas = bs.rolling(bench_ma).mean()
        bench_bull = _debounce((bs > bmas).astype(int), confirm_days, band_filter, bs, bmas)
    else:
        bench_bull = pd.Series(1, index=close.index, dtype=int)   # 恒多头(不择时)

    holdings, cash = {}, float(capital)
    equity_curve, dates_out, trade_log = [], [], []
    deal = open_px if deal_price == "open" else close

    for i in range(1, len(dates)):
        d_prev, d_now = dates[i - 1], dates[i]
        if i % freq == 0:
            if bench_bull.loc[d_prev] == 1:
                sc = score.loc[d_prev]
                up = uptrend.loc[d_prev]
                v60 = vol60.loc[d_prev]
                valid = sc.notna() & up.fillna(False) & v60.notna() & (v60 > 0)
                target = list(sc[valid].sort_values(ascending=False).index[:topn])
            else:
                target = []               # 大盘空头 → 空仓避险
            total_eq = _equity(holdings, cash, close.loc[d_prev])
            holdings, cash = _apply_target(holdings, cash, target, deal.loc[d_now],
                                           total_eq, commission_rate, min_commission,
                                           trade_unit, d_now, trade_log)
        equity_curve.append(_equity(holdings, cash, close.loc[d_now]))
        dates_out.append(d_now)

    return _summarize(equity_curve, dates_out, trade_log, bench, capital)


# ============== 最新信号 ==============
def _latest_trend_states(close: pd.DataFrame, codes: list, ma_short: int, ma_long: int,
                         confirm_days: int, band_filter: float):
    """算每只ETF最新一日的(防抖后持仓状态, ma_short, ma_long, 带宽)。"""
    states = {}
    for c in codes:
        s = close[c].dropna()
        if len(s) < ma_long + 5:
            continue
        ma_s = s.rolling(ma_short).mean()
        ma_l = s.rolling(ma_long).mean()
        hold = _debounce((ma_s > ma_l).astype(int), confirm_days, band_filter, ma_s, ma_l)
        ml_last = float(ma_l.iloc[-1])
        states[c] = {
            "hold": int(hold.iloc[-1]),
            "ma_short": round(float(ma_s.iloc[-1]), 3),
            "ma_long": round(ml_last, 3),
            "band%": round(abs(float(ma_s.iloc[-1]) - ml_last) / ml_last * 100, 2) if ml_last else 0.0,
        }
    return states


def single_etf_status(code: str, cfg: dict, ma_short: Optional[int] = None,
                      ma_long: Optional[int] = None, confirm_days: Optional[int] = None,
                      band_filter: Optional[float] = None) -> Optional[dict]:
    """查单只ETF最新趋势状态(用户持仓可能不在 broad 池)。

    返回与 trend_signal candidates 行同构的 dict(含 above_ma20/above_ma60), 或 None(无数据)。
    """
    ec = cfg.get("etf", {}); tc = ec.get("trend", {})
    ma_short = tc.get("ma_short", 20) if ma_short is None else ma_short
    ma_long = tc.get("ma_long", 60) if ma_long is None else ma_long
    confirm_days = tc.get("confirm_days", 2) if confirm_days is None else confirm_days
    band_filter = tc.get("band_filter", 0.005) if band_filter is None else band_filter
    code = str(code).strip().zfill(6)
    close, _, _, names, _ = load_etf_panel(cfg, "broad")
    if code in close.columns:
        s = close[code].dropna()
        name = names.get(code, code)
    else:
        p = _resolve_etf_dir(cfg) / f"{code}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p); df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].astype(float).sort_index()
        try:
            from collector.etf_collector import load_etf_pool
            pool = load_etf_pool(cfg)
            name = next((it["name"] for grp in ("broad", "sector")
                         for it in pool.get(grp, []) or []
                         if str(it["code"]).strip().zfill(6) == code), code)
        except Exception:
            name = code
    if len(s) < ma_long + 5:
        return None
    ma_s = s.rolling(ma_short).mean(); ma_l = s.rolling(ma_long).mean()
    hold = _debounce((ma_s > ma_l).astype(int), confirm_days, band_filter, ma_s, ma_l)
    ms_last = float(ma_s.iloc[-1]); ml_last = float(ma_l.iloc[-1]); px = float(s.iloc[-1])
    return {
        "code": code, "name": name, "close": round(px, 3),
        "hold": int(hold.iloc[-1]), "raw": int(ms_last > ml_last),
        "ma_short": round(ms_last, 3), "ma_long": round(ml_last, 3),
        "band%": round(abs(ms_last - ml_last) / ml_last * 100, 2) if ml_last else 0.0,
        "above_ma20": bool(px > ms_last), "above_ma60": bool(px > ml_last),
    }


# ============== 行业 ETF ↔ 行业/题材映射 (资金流/题材轮动共用) ==============
_SECTOR_KEYWORDS = {
    "半导体": ["半导体", "芯片", "集成电路", "光刻", "封装", "元件"],
    "新能源": ["新能源", "锂电", "光伏", "风电", "储能", "电动车", "电力设备", "电池", "碳中和"],
    "消费": ["消费", "食品饮料", "白酒", "家电", "零售", "免税", "商贸", "纺织"],
    "金融": ["银行", "证券", "保险", "金融", "多元金融"],
    "医药": ["医药", "医疗", "生物", "创新药", "CRO", "中药", "疫苗"],
    "军工": ["军工", "国防", "航天", "兵器", "航空"],
}
# ETF name → sector 反推(池外 ETF 用)
_NAME_SECTOR_HINTS = [
    ("半导体|芯片", "半导体"), ("电池|光伏|新能源|锂电", "新能源"),
    ("消费|食品|白酒|家电", "消费"), ("银行|证券|保险|金融", "金融"),
    ("医药|医疗|生物", "医药"), ("军工|国防|航天", "军工"),
]


def _sector_of(code: str, name: str, cfg: dict) -> str:
    """ETF 的 sector: 优先 pool 的 sector 字段, 否则从 name 反推。"""
    code = str(code).strip().zfill(6)
    try:
        from collector.etf_collector import load_etf_pool
        pool = load_etf_pool(cfg)
        for grp in ("broad", "sector"):
            for it in pool.get(grp, []) or []:
                if str(it.get("code", "")).strip().zfill(6) == code:
                    return it.get("sector") or ""
    except Exception:
        pass
    import re
    for pat, sec in _NAME_SECTOR_HINTS:
        if re.search(pat, str(name)):
            return sec
    return ""


def _match_sector_to_flow(sector: str, flow_df: pd.DataFrame) -> Optional[pd.Series]:
    """ETF sector → 匹配行业资金流行(关键词命中; 多命中取净额最大)。无命中返回 None。"""
    if not sector or flow_df is None or flow_df.empty or "sector" not in flow_df.columns:
        return None
    kws = _SECTOR_KEYWORDS.get(sector, [sector])
    mask = flow_df["sector"].astype(str).str.contains("|".join(kws), na=False, regex=True)
    hit = flow_df[mask]
    if hit.empty:
        return None
    if "net_amount" in hit.columns:
        hit = hit.sort_values("net_amount", ascending=False)
    return hit.iloc[0]


def _match_sector_to_concept(sector: str, heat_df: pd.DataFrame) -> float:
    """ETF sector → 题材热度分(命中题材中最热的)。无命中返回 NaN。"""
    if (not sector or heat_df is None or heat_df.empty
            or "concept" not in heat_df.columns or "heat_score" not in heat_df.columns):
        return np.nan
    kws = _SECTOR_KEYWORDS.get(sector, [sector])
    mask = heat_df["concept"].astype(str).str.contains("|".join(kws), na=False, regex=True)
    hit = heat_df[mask]
    if hit.empty:
        return np.nan
    return float(pd.to_numeric(hit["heat_score"], errors="coerce").max())


def _build_sector_scores(etf_codes: list, names: dict, cfg: dict) -> pd.DataFrame:
    """资金流/题材轮动共用入口: 给一批 ETF code, 返回
    DataFrame(code/name/sector/flow_net/flow_pct/heat_score), 数据失败列 NaN。"""
    from collector.flow_collector import sector_fund_flow, concept_heat
    flow_df = sector_fund_flow()
    heat_df = concept_heat(topn=60)
    rows = []
    for code in etf_codes:
        code = str(code).strip().zfill(6)
        name = names.get(code, code)
        sector = _sector_of(code, name, cfg)
        fr = _match_sector_to_flow(sector, flow_df)
        hs = _match_sector_to_concept(sector, heat_df)
        rows.append({
            "code": code, "name": name, "sector": sector,
            "flow_net": float(fr["net_amount"]) if fr is not None and pd.notna(fr.get("net_amount")) else np.nan,
            "flow_pct": float(fr["net_pct"]) if fr is not None and pd.notna(fr.get("net_pct")) else np.nan,
            "heat_score": hs if not (isinstance(hs, float) and np.isnan(hs)) else np.nan,
        })
    return pd.DataFrame(rows)


def trend_signal(cfg: dict, ma_short: Optional[int] = None, ma_long: Optional[int] = None,
                 confirm_days: Optional[int] = None, band_filter: Optional[float] = None,
                 pool: str = "broad") -> dict:
    ec = cfg.get("etf", {}); tc = ec.get("trend", {})
    ma_short = tc.get("ma_short", 20) if ma_short is None else ma_short
    ma_long = tc.get("ma_long", 60) if ma_long is None else ma_long
    confirm_days = tc.get("confirm_days", 2) if confirm_days is None else confirm_days
    band_filter = tc.get("band_filter", 0.005) if band_filter is None else band_filter
    close, open_px, ret, names, bench = load_etf_panel(cfg, pool)
    codes = list(close.columns)
    last = close.index[-1]
    states = _latest_trend_states(close, codes, ma_short, ma_long, confirm_days, band_filter)
    rows = []
    for c in codes:
        if c not in states:
            continue
        st = states[c]
        rows.append({"code": c, "name": names.get(c, c),
                     "close": round(float(close[c].dropna().iloc[-1]), 3),
                     **st, "raw": int(st["ma_short"] > st["ma_long"])})
    df = pd.DataFrame(rows).sort_values("hold", ascending=False) if rows else pd.DataFrame()
    held = df[df["hold"] == 1] if not df.empty else df
    return {"strategy": "trend", "pool": pool, "date": last.strftime("%Y-%m-%d"),
            "params": {"ma_short": ma_short, "ma_long": ma_long,
                       "confirm_days": confirm_days, "band_filter": band_filter},
            "candidates": df, "top": held}


def rotation_signal(cfg: dict, pool: str = "broad", topn: Optional[int] = None,
                    mom_short: Optional[int] = None, mom_long: Optional[int] = None,
                    market_timing: Optional[bool] = None, bench_ma: Optional[int] = None,
                    confirm_days: Optional[int] = None, band_filter: Optional[float] = None) -> dict:
    ec = cfg.get("etf", {}); rc = ec.get("rotation", {})
    if topn is None:
        topn = rc.get("sector_topn", 3) if pool == "sector" else rc.get("broad_topn", 3)
    mom_short = rc.get("mom_short", 20) if mom_short is None else mom_short
    mom_long = rc.get("mom_long", 60) if mom_long is None else mom_long
    bench_ma = rc.get("bench_ma", 60) if bench_ma is None else bench_ma
    confirm_days = ec.get("trend", {}).get("confirm_days", 2) if confirm_days is None else confirm_days
    band_filter = ec.get("trend", {}).get("band_filter", 0.005) if band_filter is None else band_filter
    if market_timing is None:
        market_timing = rc.get("sector_market_timing", True) if pool == "sector" else False
    bench_code = str(ec.get("benchmark_etf", "510300")).strip().zfill(6)

    close, open_px, ret, names, bench = load_etf_panel(cfg, pool)
    codes = [c for c in close.columns if c != bench_code]
    last = close.index[-1]
    ma20 = close[codes].rolling(20).mean()
    ma60 = close[codes].rolling(60).mean()
    mom_l = close[codes] / close[codes].shift(mom_long) - 1
    mom_s = close[codes] / close[codes].shift(mom_short) - 1
    vol60 = ret[codes].rolling(60).std()
    base = mom_l / vol60
    if pool == "sector":
        z = mom_s.sub(mom_s.mean(axis=1), axis=0).div(mom_s.std(axis=1).replace(0, np.nan), axis=0)
        score = 0.6 * base + 0.4 * z
    else:
        score = base

    # 大盘开关
    market_bull = True
    bench_close = bench_ma_val = None
    if pool == "sector" and market_timing and bench_code in close.columns:
        bs = close[bench_code]
        bmas = bs.rolling(bench_ma).mean()
        bb = _debounce((bs > bmas).astype(int), confirm_days, band_filter, bs, bmas)
        market_bull = bool(bb.iloc[-1] == 1)
        bench_close = round(float(bs.dropna().iloc[-1]), 3)
        bench_ma_val = round(float(bmas.dropna().iloc[-1]), 3)

    rows = []
    for c in codes:
        if pd.isna(score[c].iloc[-1]):
            continue
        rows.append({"code": c, "name": names.get(c, c),
                     "close": round(float(close[c].dropna().iloc[-1]), 3),
                     "mom_long": round(float(mom_l[c].iloc[-1]) * 100, 1),
                     "mom_short": round(float(mom_s[c].iloc[-1]) * 100, 1),
                     "uptrend": bool((ma20[c].iloc[-1] > ma60[c].iloc[-1])),
                     "score": round(float(score[c].iloc[-1]), 3)})
    df = pd.DataFrame(rows)
    cand = df[df["uptrend"]].sort_values("score", ascending=False) if not df.empty else df
    if market_bull:
        top = cand.head(topn).copy()
    else:
        top = pd.DataFrame()              # 空头 → 空仓
    return {"strategy": "rotation", "pool": pool, "date": last.strftime("%Y-%m-%d"),
            "market_bull": market_bull, "bench_close": bench_close, "bench_ma": bench_ma_val,
            "candidates": cand, "top": top}


def rotation_signal_flow(cfg: dict, topn: Optional[int] = None,
                         w_mom: float = 0.4, w_flow: float = 0.3, w_heat: float = 0.3,
                         market_timing: Optional[bool] = None) -> dict:
    """方向①: 资金流+题材热度增强的行业ETF轮动信号(sector池)。

    动量 + 行业资金流净额 + 题材热度, 横截面 rank(pct=True) 加权。东财限流时 flow/heat
    自动降级(权重转移动量), 即退化为纯动量轮动。返回 rotation_signal 同构 + total_score 列
    + weights/data_status。
    """
    ec = cfg.get("etf", {}); rc = ec.get("rotation", {}); rfc = ec.get("rotation_flow", {})
    w_mom = rfc.get("w_mom", w_mom); w_flow = rfc.get("w_flow", w_flow); w_heat = rfc.get("w_heat", w_heat)
    if topn is None:
        topn = rc.get("sector_topn", 3)
    if market_timing is None:
        market_timing = rc.get("sector_market_timing", True)
    base = rotation_signal(cfg, pool="sector", topn=topn, market_timing=market_timing)
    cand = base["candidates"].copy()
    if cand.empty:
        base.update({"strategy": "rotation_flow", "weights": {"w_mom": 1.0, "w_flow": 0, "w_heat": 0},
                     "data_status": {"flow": False, "heat": False}})
        return base
    sc = _build_sector_scores(cand["code"].tolist(), dict(zip(cand["code"], cand["name"])), cfg)
    cand = cand.merge(sc[["code", "flow_net", "heat_score"]], on="code", how="left")
    # 横截面 rank; 数据缺失填0.5中性, 权重转移给可用项(东财限流时退化为纯动量)
    has_flow = bool(cand["flow_net"].notna().any())
    has_heat = bool(cand["heat_score"].notna().any())
    wm, wf, wh = w_mom, (w_flow if has_flow else 0.0), (w_heat if has_heat else 0.0)
    if not has_flow: wm += w_flow
    if not has_heat: wm += w_heat
    cand["flow_rank"] = cand["flow_net"].rank(pct=True).fillna(0.5)
    cand["heat_rank"] = cand["heat_score"].rank(pct=True).fillna(0.5)
    cand["mom_rank"] = cand["score"].rank(pct=True).fillna(0.5)
    s = (wm + wf + wh) or 1.0
    cand["total_score"] = ((wm * cand["mom_rank"] + wf * cand["flow_rank"] + wh * cand["heat_rank"]) / s).round(3)
    cand = cand.sort_values("total_score", ascending=False).reset_index(drop=True)
    top = cand.head(topn).copy() if base["market_bull"] else cand.iloc[0:0].copy()
    base.update({"strategy": "rotation_flow", "candidates": cand, "top": top,
                 "weights": {"w_mom": round(wm / s, 2), "w_flow": round(wf / s, 2), "w_heat": round(wh / s, 2)},
                 "data_status": {"flow": has_flow, "heat": has_heat}})
    return base


def backtest_rotation_flow(cfg: dict, topn: Optional[int] = None,
                           start: Optional[str] = None, capital: Optional[float] = None,
                           freq: Optional[int] = None) -> dict:
    """方向①回测。⚠️ 历史行业资金流/题材热度不可得(akshare 只给当日), 回测只能验证
    动量骨架; flow/heat 增强仅在当前实时信号(rotation_signal_flow)有效。复用 backtest_rotation。"""
    rc = cfg.get("etf", {}).get("rotation", {})
    if topn is None: topn = rc.get("sector_topn", 3)
    if freq is None: freq = rc.get("rebalance_freq", 5)
    return backtest_rotation(cfg, pool="sector", topn=topn, freq=freq, start=start, capital=capital)


# ============== Markdown 输出 ==============
def signal_to_markdown(sig: dict) -> str:
    if sig["strategy"] == "trend":
        return _trend_md(sig)
    return _rotation_md(sig)


def _trend_md(sig: dict) -> str:
    p = sig["params"]
    L = [f"# 📈 ETF趋势择时信号 ({sig['date']})\n"]
    L.append(f"**参数**: 双均线 MA{p['ma_short']}/MA{p['ma_long']}, 连续确认{p['confirm_days']}日, "
             f"带宽过滤{p['band_filter']*100:.1f}% (防抖, 避免均线缠绕区反复交易)\n")
    held = sig["top"]
    if held.empty:
        L.append("> 当前**无标的处于多头确认状态**, 建议空仓观望(均线之下或缠绕中)。\n")
    else:
        L.append("## 本期持有 (防抖后处于多头)")
        L.append("| 代码 | 名称 | 现价 | MA短 | MA长 | 带宽% |")
        L.append("|---|---|---|---|---|---|")
        for _, r in held.iterrows():
            L.append(f"| {r['code']} | {r['name']} | {r['close']} | {r['ma_short']} | "
                     f"{r['ma_long']} | {r['band%']} |")
        L.append(f"\n**等权持有 {len(held)} 只**, 跌破长均线且连续确认后退出。")
    cand = sig["candidates"]
    pending = cand[(cand["hold"] == 0) & (cand["raw"] == 1)] if not cand.empty else cand
    if not pending.empty:
        L.append("\n## 原始多头但未确认 (均线刚金叉/带宽不足, 等连续确认)")
        L.append("| 代码 | 名称 | 现价 | 带宽% |")
        L.append("|---|---|---|---|")
        for _, r in pending.head(6).iterrows():
            L.append(f"| {r['code']} | {r['name']} | {r['close']} | {r['band%']} |")
    return "\n".join(L)


def _rotation_md(sig: dict) -> str:
    pool_name = "宽基+避险" if sig["pool"] == "broad" else "行业/主题"
    L = [f"# 🔄 ETF动量轮动信号 ({sig['date']}, {pool_name}池)\n"]
    if sig["pool"] == "sector":
        if sig["market_bull"]:
            L.append(f"**大盘择时**: 基准ETF {sig['bench_close']} > MA({sig['bench_ma']}) → "
                     "✅ 多头, 允许持仓行业ETF\n")
        else:
            L.append(f"**大盘择时**: 基准ETF {sig['bench_close']} < MA({sig['bench_ma']}) → "
                     "⚠️ 空头, **本期空仓避险**\n")
    top = sig["top"]
    if top.empty:
        L.append("> 本期**空仓**(无符合趋势+动量的标的, 或大盘空头)。\n")
    else:
        L.append(f"## 本期持有 (动量 top{len(top)}, 已过上升趋势过滤)")
        L.append("| 代码 | 名称 | 现价 | 长动量% | 短动量% | 评分 |")
        L.append("|---|---|---|---|---|---|")
        for _, r in top.iterrows():
            L.append(f"| {r['code']} | {r['name']} | {r['close']} | {r['mom_long']} | "
                     f"{r['mom_short']} | {r['score']} |")
        L.append(f"\n**等权持有 {len(top)} 只**, 下期重新评分, 跌出 topN 或转弱则换出。")
    return "\n".join(L)


if __name__ == "__main__":
    cfg = load_config()
    bt = backtest_trend(cfg)
    print(f"趋势择时: 累计 {bt['cum_ret']:+.1%} vs 基准 {bt['bench_cum']:+.1%}, "
          f"回撤 {bt['max_dd']:.1%} vs {bt['bench_dd']:.1%}, "
          f"换手 {bt['turnover_annual']:.1f}x/年, {bt['n_trades']}笔, 佣金{bt['total_commission']:.0f}元")

"""确定性进出场系统: 成长进攻 ↔ 红利防守。

哲学(确定性优先):
- 进场 = 多条件共振(AND)。日线 conviction(大盘闸门+趋势)是硬门槛, 60min 强度/量能/动量
  决定 A/B/C 分级; 仅 A 级建议进场 → 高精度、低频。
- 离场 = 任一拐点预兆(OR)。顶背离/破位/衰竭/死叉/日线转弱/止损, 触发即走 → 保命锁盈。
- 进攻离场 -> 轮动到红利100 防守(组合层, 此处提供信号, 轮动绩效另行归因)。

无未来函数: 信号在 bar i-1(已完成)评估, bar i 开盘执行; 日线 conviction 用前一日已收盘状态。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from common import setup_logger
from collector.index_min_collector import resample_daily

log = setup_logger("strategy.entry_exit")

# 往返成本(ETF: 佣金万2.5双边, 免印花税) ≈ 0.05%
DEFAULT_COST = 0.0005


# ============== 指标(纯 pandas, 无外部 TA 依赖) ==============
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # 边界: 全涨(avg_loss=0)→100, 走平(两者=0)→50; warmup 期(avg为NaN)保持 NaN。
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, dif - dea


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ============== 特征 ==============
def add_min_indicators(min_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """在 60min 上算 MA/RSI/MACD/量能/ATR/近期高点。hh_prev 不含当根(供突破判断, 无未来)。"""
    ee = cfg["entry_exit"]
    d = min_df.copy().reset_index(drop=True)
    d["ma_s"] = d["close"].rolling(ee["ma_short"]).mean()
    d["ma_l"] = d["close"].rolling(ee["ma_long"]).mean()
    d["rsi"] = rsi(d["close"], ee["rsi_period"])
    d["dif"], d["dea"], d["mhist"] = macd(d["close"])
    d["vol_ma"] = d["volume"].rolling(20).mean()
    d["atr"] = atr(d, ee["atr_period"])
    # 突破参照: 前 N 根的最高价(不含当根)
    d["hh_prev"] = d["high"].shift(1).rolling(ee["breakout_lookback"]).max()
    d["bar_range"] = d["high"] - d["low"]
    d["upper_shadow"] = d["high"] - d[["open", "close"]].max(axis=1)
    return d


def daily_uptrend_state(min_df: pd.DataFrame, daily_df: pd.DataFrame,
                        ma_short: int, ma_long: int) -> pd.Series:
    """把日线多头状态映射到每根 60min, 且用**前一交易日**已收盘状态(当日日线未走完, 防未来)。

    多头 = close>MA_long 且 MA_short>MA_long。返回与 min_df 等长的 bool Series。
    """
    dd = daily_df.copy()
    dd["d_ma_s"] = dd["close"].rolling(ma_short).mean()
    dd["d_ma_l"] = dd["close"].rolling(ma_long).mean()
    dd["up"] = (dd["close"] > dd["d_ma_l"]) & (dd["d_ma_s"] > dd["d_ma_l"])
    dd["day"] = pd.to_datetime(dd["date"]).dt.normalize()
    s = dd.set_index("day")["up"].shift(1)          # 前一交易日状态
    bar_day = pd.to_datetime(min_df["date"]).dt.normalize()
    return bar_day.map(s).fillna(False).reset_index(drop=True)


def daily_below_short_state(min_df: pd.DataFrame, daily_df: pd.DataFrame, ma_short: int) -> pd.Series:
    """日线跌破短均线(转弱预兆), 同样取前一日状态。"""
    dd = daily_df.copy()
    dd["d_ma_s"] = dd["close"].rolling(ma_short).mean()
    dd["weak"] = dd["close"] < dd["d_ma_s"]
    dd["day"] = pd.to_datetime(dd["date"]).dt.normalize()
    s = dd.set_index("day")["weak"].shift(1)
    bar_day = pd.to_datetime(min_df["date"]).dt.normalize()
    return bar_day.map(s).fillna(False).reset_index(drop=True)


# ============== 标注信号(供回测与实盘共用) ==============
def annotate_signals(min_df: pd.DataFrame, market_min_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """给 60min 打上 conviction/进场分级/离场标志。逐根只用 <=该根的信息。"""
    ee = cfg["entry_exit"]
    d = add_min_indicators(min_df, cfg)
    daily_self = resample_daily(min_df)
    daily_mkt = resample_daily(market_min_df)
    # 日线 conviction(硬门槛)
    d["c_dtrend"] = daily_uptrend_state(d, daily_self, ee["daily_ma_short"], ee["daily_ma_long"])
    d["c_market"] = daily_uptrend_state(d, daily_mkt, ee["daily_ma_short"], ee["daily_ma_long"])
    d["c_dailyweak"] = daily_below_short_state(d, daily_self, ee["daily_ma_short"])
    # 60min 趋势/强度/量能/动量(分级项)
    d["c_trend"] = (d["close"] > d["ma_l"]) & (d["ma_s"] > d["ma_l"])
    d["c_breakout"] = d["close"] > d["hh_prev"]
    d["c_volume"] = d["volume"] > ee["vol_mult"] * d["vol_ma"]
    # 动量: MACD 多头(dif>dea)且 RSI 不弱。突破时 RSI 天然偏高(中位~71), 上限放宽到 rsi_hi,
    # "过热"不在进场挡(会误杀真突破), 交给离场的衰竭/背离判断。
    d["c_momentum"] = (d["dif"] > d["dea"]) & (d["rsi"] >= ee["rsi_lo"]) & (d["rsi"] <= ee["rsi_hi"])
    # 进场扳机 = 趋势门槛(hard_gate) + 突破("足够强"的事件)。量能/动量是确认项, 用于升级分级。
    d["hard_gate"] = d["c_market"] & d["c_dtrend"] & d["c_trend"]
    d["trigger"] = d["hard_gate"] & d["c_breakout"]
    d["grade"] = np.where(~d["trigger"], "",
                          np.where(d["c_volume"] & d["c_momentum"], "A",
                                   np.where(d["c_volume"] | d["c_momentum"], "B", "C")))
    # 离场预兆(OR)
    lb = ee["breakout_lookback"]
    d["e_div"] = (d["close"] > d["close"].shift(lb)) & (d["rsi"] < d["rsi"].shift(lb) - 5) & \
                 (d["close"] >= d["hh_prev"] * 0.99)          # 价升但动量走弱, 且处高位
    d["e_break"] = (d["close"] < d["ma_s"]) & (d["close"] < d["open"]) & (d["volume"] > d["vol_ma"])
    over = (d["close"] - d["ma_l"]) / d["ma_l"]
    d["e_exhaust"] = (over > ee["overextend_mult"]) & (d["upper_shadow"] > (d["bar_range"] * 0.5)) & d["c_volume"]
    d["e_deadcross"] = (d["ma_s"] < d["ma_l"]) & (d["ma_s"].shift(1) >= d["ma_l"].shift(1))
    d["exit_any"] = d["e_div"] | d["e_break"] | d["e_exhaust"] | d["e_deadcross"] | d["c_dailyweak"]
    return d


# ============== Walk-forward 回测(无未来函数) ==============
def backtest_entry_exit(min_df: pd.DataFrame, market_min_df: pd.DataFrame, cfg: dict,
                        min_grade: str = "B", cost: float = DEFAULT_COST) -> dict:
    """单标的 walk-forward。信号在 bar i-1 评估, bar i 开盘执行。

    min_grade: 允许进场的最低级别 ("A" 仅A / "B" A+B / "C" 全部)。
    返回 {trades, by_grade, 汇总指标}。每笔记录 entry/exit 时间/价/收益/级别/离场原因。
    """
    ee = cfg["entry_exit"]
    d = annotate_signals(min_df, market_min_df, cfg)
    opens = d["open"].values
    lows = d["low"].values
    grades = d["grade"].values
    exits = d["exit_any"].values
    atrs = d["atr"].values
    order = {"A": 0, "B": 1, "C": 2}
    allow = order[min_grade]
    cooldown = ee["cooldown_bars"]
    atr_mult = ee["atr_stop_mult"]

    trades = []
    pos = None               # dict(entry_i, entry_px, stop, grade)
    cooldown_until = -1
    for i in range(1, len(d)):
        sig = i - 1                       # 用已完成 bar sig 做决策
        if pos is None:
            if i <= cooldown_until:
                continue
            g = grades[sig]
            if g and order.get(g, 9) <= allow:
                px = opens[i]
                if not np.isfinite(px) or px <= 0:
                    continue
                stop = px - atr_mult * (atrs[sig] if np.isfinite(atrs[sig]) else px * 0.02)
                pos = {"entry_i": i, "entry_px": px, "entry_date": d["date"].iloc[i],
                       "stop": stop, "grade": g}
        else:
            exit_px, reason = None, None
            # 止损: 当根最低触及止损价
            if lows[i] <= pos["stop"]:
                exit_px, reason = pos["stop"], "stop"
            elif exits[sig]:
                exit_px, reason = opens[i], "signal"
            if exit_px is not None:
                ret = (exit_px - pos["entry_px"]) / pos["entry_px"] - cost
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": d["date"].iloc[i],
                    "grade": pos["grade"], "entry_px": pos["entry_px"], "exit_px": exit_px,
                    "ret": ret, "bars_held": i - pos["entry_i"], "reason": reason,
                })
                cooldown_until = i + cooldown
                pos = None
    # 期末仍持仓: 按最后收盘平仓(标记 open)
    if pos is not None:
        px = d["close"].iloc[-1]
        ret = (px - pos["entry_px"]) / pos["entry_px"] - cost
        trades.append({"entry_date": pos["entry_date"], "exit_date": d["date"].iloc[-1],
                       "grade": pos["grade"], "entry_px": pos["entry_px"], "exit_px": px,
                       "ret": ret, "bars_held": len(d) - pos["entry_i"], "reason": "eod"})

    return {"trades": pd.DataFrame(trades), "annotated": d}


def summarize_trades(trades: pd.DataFrame) -> dict:
    """汇总: 笔数/胜率/平均盈亏/盈亏比/期望/最大单笔亏损。"""
    if trades.empty:
        return {"n": 0, "win_rate": np.nan, "avg_ret": np.nan, "expectancy": np.nan}
    rets = trades["ret"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    return {
        "n": len(rets),
        "win_rate": float((rets > 0).mean()),
        "avg_ret": float(rets.mean()),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff": float(avg_win / avg_loss) if avg_loss > 0 else np.inf,
        "expectancy": float(rets.mean()),
        "max_loss": float(rets.min()),
        "avg_bars": float(trades["bars_held"].mean()),
        "total_ret": float((1 + rets).prod() - 1),
    }


def grade_report(result: dict) -> pd.DataFrame:
    """按确定性分级(A/B/C)分别统计, 验证 A 级是否显著更稳。"""
    trades = result["trades"]
    rows = []
    for g in ["A", "B", "C"]:
        sub = trades[trades["grade"] == g]
        s = summarize_trades(sub)
        s["grade"] = g
        rows.append(s)
    return pd.DataFrame(rows)[["grade", "n", "win_rate", "avg_ret", "payoff",
                               "expectancy", "max_loss", "avg_bars", "total_ret"]]


# ============== v2: 抗whipsaw 趋势跟踪(ER趋势质量过滤 + ATR移动止损) ==============
def efficiency_ratio(close: pd.Series, n: int = 30) -> pd.Series:
    """Kaufman 效率比: 净位移/路径总长 ∈[0,1]。接近1=单边趋势, 接近0=震荡。

    用作"趋势质量"过滤: 只在 ER 高(真趋势)时进场, 从源头砍掉震荡市的假突破 whipsaw。
    """
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def backtest_trend_follow(min_df: pd.DataFrame, market_min_df: pd.DataFrame, cfg: dict,
                          donchian: int = None, er_n: int = None, er_min: float = None,
                          atr_mult: float = None, cost: float = DEFAULT_COST) -> dict:
    """v2 趋势跟踪回测。进场=日线conviction(硬门槛)+Donchian突破+ER趋势质量;
    离场=ATR移动止损(最高收盘回撤 atr_mult×ATR)+日线转弱。让利润奔跑, 确认反转才走。

    参数缺省读 config.entry_exit(donchian/er_n/er_min/atr_trail_mult)。
    无未来: 信号 bar i-1 评估, bar i 开盘执行; 日线用前一日状态。
    """
    ee = cfg["entry_exit"]
    donchian = donchian or ee.get("donchian", 20)
    er_n = er_n or ee.get("er_n", 30)
    er_min = ee.get("er_min", 0.35) if er_min is None else er_min
    atr_mult = atr_mult or ee.get("atr_trail_mult", 4.0)
    d = add_min_indicators(min_df, cfg)
    daily_self = resample_daily(min_df)
    daily_mkt = resample_daily(market_min_df)
    d["c_dtrend"] = daily_uptrend_state(d, daily_self, ee["daily_ma_short"], ee["daily_ma_long"])
    d["c_market"] = daily_uptrend_state(d, daily_mkt, ee["daily_ma_short"], ee["daily_ma_long"])
    d["hard_gate"] = d["c_market"] & d["c_dtrend"]
    d["dc_prev"] = d["high"].shift(1).rolling(donchian).max()
    d["er"] = efficiency_ratio(d["close"], er_n)
    d["entry_sig"] = d["hard_gate"] & (d["close"] > d["dc_prev"]) & (d["er"] > er_min)
    d["dailyweak"] = daily_below_short_state(d, daily_self, ee["daily_ma_short"])

    opens, lows, closes = d["open"].values, d["low"].values, d["close"].values
    atrs = d["atr"].values
    entries, weaks = d["entry_sig"].values, d["dailyweak"].values

    trades, pos = [], None
    in_pos = np.zeros(len(d), dtype=bool)        # 每根 bar 是否持仓(供实盘判断当前状态)
    for i in range(1, len(d)):
        sig = i - 1
        if pos is None:
            if entries[sig]:
                px = opens[i]
                if np.isfinite(px) and px > 0:
                    pos = {"entry_i": i, "entry_px": px, "entry_date": d["date"].iloc[i],
                           "hh_close": px}
        else:
            pos["hh_close"] = max(pos["hh_close"], closes[i - 1])     # 截至上一根的持仓期最高收盘
            atr_now = atrs[sig] if np.isfinite(atrs[sig]) else pos["entry_px"] * 0.02
            trail = pos["hh_close"] - atr_mult * atr_now
            exit_px, reason = None, None
            if lows[i] <= trail:
                exit_px, reason = trail, "trail"
            elif weaks[sig]:
                exit_px, reason = opens[i], "dailyweak"
            if exit_px is not None:
                ret = (exit_px - pos["entry_px"]) / pos["entry_px"] - cost
                trades.append({"entry_date": pos["entry_date"], "exit_date": d["date"].iloc[i],
                               "grade": "TF", "entry_px": pos["entry_px"], "exit_px": exit_px,
                               "ret": ret, "bars_held": i - pos["entry_i"], "reason": reason})
                pos = None
        in_pos[i] = pos is not None
    if pos is not None:
        px = closes[-1]
        trades.append({"entry_date": pos["entry_date"], "exit_date": d["date"].iloc[-1],
                       "grade": "TF", "entry_px": pos["entry_px"], "exit_px": px,
                       "ret": (px - pos["entry_px"]) / pos["entry_px"] - cost,
                       "bars_held": len(d) - pos["entry_i"], "reason": "eod"})
    return {"trades": pd.DataFrame(trades), "annotated": d,
            "in_pos": pd.Series(in_pos, index=d["date"])}


def backtest_daily_trend(self_df: pd.DataFrame, market_df: pd.DataFrame,
                         ma_gate: int = 60, ma_trend: int = 20, donchian: int = 20,
                         er_n: int = 30, er_min: float = 0.30, atr_mult: float = 4.0,
                         cost: float = 0.0005) -> dict:
    """v2 逻辑的日线版(验证回撤保护, 用含熊市的长历史)。

    进场=大盘闸门(market 日线多头, 前一日)+自身趋势(close>MA_gate)+Donchian突破+ER;
    离场=ATR×atr_mult 移动止损。无未来: bar i-1 评估, i 开盘执行; 闸门用前一日。
    返回 {trades, nav(净值Series), in_mkt(持仓bool)}。
    """
    d = self_df.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d["ma_gate"] = d["close"].rolling(ma_gate).mean()
    d["atr"] = atr(d, 14)
    d["dc_prev"] = d["high"].shift(1).rolling(donchian).max()
    d["er"] = efficiency_ratio(d["close"], er_n)
    d["self_trend"] = d["close"] > d["ma_gate"]
    # 大盘闸门(market 日线多头), 对齐到 self 日期并取前一日(无未来)
    m = market_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["m_gate"] = m["close"].rolling(ma_gate).mean()
    m["m_s"] = m["close"].rolling(ma_trend).mean()
    m["m_up"] = (m["close"] > m["m_gate"]) & (m["m_s"] > m["m_gate"])
    gate = m.set_index("date")["m_up"].reindex(d["date"]).shift(1).fillna(False)
    d["c_market"] = gate.values
    d["entry_sig"] = d["c_market"] & d["self_trend"] & (d["close"] > d["dc_prev"]) & (d["er"] > er_min)

    opens, lows, closes = d["open"].values, d["low"].values, d["close"].values
    atrs, entries = d["atr"].values, d["entry_sig"].values
    trades, pos = [], None
    in_mkt = np.zeros(len(d), dtype=bool)
    for i in range(1, len(d)):
        sig = i - 1
        if pos is None:
            if entries[sig]:
                px = opens[i]
                if np.isfinite(px) and px > 0:
                    pos = {"entry_i": i, "entry_px": px, "entry_date": d["date"].iloc[i], "hh": px}
        else:
            in_mkt[i] = True
            pos["hh"] = max(pos["hh"], closes[i - 1])
            atr_now = atrs[sig] if np.isfinite(atrs[sig]) else pos["entry_px"] * 0.02
            trail = pos["hh"] - atr_mult * atr_now
            if lows[i] <= trail:
                ret = (trail - pos["entry_px"]) / pos["entry_px"] - cost
                trades.append({"entry_date": pos["entry_date"], "exit_date": d["date"].iloc[i],
                               "grade": "D", "entry_px": pos["entry_px"], "exit_px": trail,
                               "ret": ret, "bars_held": i - pos["entry_i"], "reason": "trail"})
                pos = None
    if pos is not None:
        px = closes[-1]
        trades.append({"entry_date": pos["entry_date"], "exit_date": d["date"].iloc[-1],
                       "grade": "D", "entry_px": pos["entry_px"], "exit_px": px,
                       "ret": (px - pos["entry_px"]) / pos["entry_px"] - cost,
                       "bars_held": len(d) - pos["entry_i"], "reason": "eod"})
    # 净值(逐日: 持仓吃当日收益, 空仓为0)
    ret = d["close"].pct_change().fillna(0.0)
    pos_flag = pd.Series(in_mkt, index=d["date"]).shift(1).fillna(False)   # 前一日持仓决定今日吃收益
    nav = (1 + ret * pos_flag.values).cumprod()
    return {"trades": pd.DataFrame(trades), "nav": nav, "in_mkt": pos_flag, "dates": d["date"]}


def backtest_rotation(growth_df: pd.DataFrame, market_df: pd.DataFrame,
                      defense_df: pd.DataFrame, defense_filter: bool = True,
                      defense_ma: int = 60, **kw) -> dict:
    """成长↔红利100 轮动: v2 在场内→持成长指数; v2 离场→切入红利100 防守。

    defense_filter=True 时, 红利100 也须自身趋势向上(收盘>MA_defense_ma, 前一日状态)才持有,
    红利走弱则空仓——进一步压回撤(代价是红利下跌期不吃其收益)。
    复用 backtest_daily_trend 的 in_mkt(前一日仓位); 所有状态均取前一日, 无未来函数。
    """
    res = backtest_daily_trend(growth_df, market_df, **kw)
    dates = res["dates"]
    hold_growth = res["in_mkt"]                      # 已 shift(前一日仓位)
    g = growth_df.copy(); g["date"] = pd.to_datetime(g["date"])
    dfn = defense_df.copy(); dfn["date"] = pd.to_datetime(dfn["date"])
    gret = g.set_index("date")["close"].pct_change().reindex(dates).fillna(0.0)
    dret = dfn.set_index("date")["close"].pct_change().reindex(dates).fillna(0.0)
    if defense_filter:
        dfn_up = dfn["close"] > dfn["close"].rolling(defense_ma).mean()
        dfn_up.index = pd.to_datetime(dfn["date"])
        dfn_up = dfn_up.reindex(dates).shift(1).fillna(False)
        def_ret = np.where(dfn_up.values, dret.values, 0.0)
    else:
        def_ret = dret.values
    port_ret = pd.Series(np.where(hold_growth.values, gret.values, def_ret), index=dates)
    nav = (1 + port_ret).cumprod()
    return {"nav": nav, "in_mkt": hold_growth, "dates": dates, "trades": res["trades"]}


def generate_live_signals(cfg: dict) -> list:
    """实盘进攻信号(纯进攻, 无防守): 读最新 60min, 输出每个成长板当前状态。

    每板返回 dict: board/etf/action(进场/持有/观望/刚离场)/close/conviction/er/trail_stop/...
    action 语义: 进场=最新bar触发v2进场(下根开盘杀入); 持有=当前在场内(附移动止损位);
                 观望=空仓无信号; 刚离场=最新bar触发离场。
    """
    from collector.index_min_collector import load_index_min
    ee = cfg["entry_exit"]
    market_sym = ee.get("market")
    market = load_index_min(cfg, market_sym)
    out = []
    for g in ee.get("growth", []):
        sym, name, etf = g["index"], g["name"], g.get("etf", "")
        m = load_index_min(cfg, sym)
        if m.empty or market.empty:
            out.append({"board": name, "etf": etf, "action": "无数据", "close": None})
            continue
        res = backtest_trend_follow(m, market, cfg)
        ann, in_pos, trades = res["annotated"], res["in_pos"], res["trades"]
        last = ann.iloc[-1]
        holding = bool(in_pos.iloc[-1])
        entry_sig = bool(last["entry_sig"])
        exit_sig = bool(last.get("dailyweak", False))
        # 最新一根是否刚离场(最后一笔交易 exit 落在最后bar)
        just_exited = (not trades.empty and trades.iloc[-1]["reason"] != "eod"
                       and pd.Timestamp(trades.iloc[-1]["exit_date"]) == pd.Timestamp(last["date"]))
        if holding:
            action = "持有"
        elif just_exited:
            action = "刚离场"
        elif entry_sig:
            action = "进场"
        else:
            action = "观望"
        # 持仓时的移动止损位(持仓期最高收盘 - atr_mult×ATR)
        trail = None
        if holding and not trades.empty:
            hh = ann.loc[in_pos, "close"].max()
            atr_now = last["atr"] if np.isfinite(last["atr"]) else last["close"] * 0.02
            trail = float(hh - ee.get("atr_trail_mult", 4.0) * atr_now)
        out.append({
            "board": name, "etf": etf, "action": action,
            "close": float(last["close"]), "conviction": bool(last["hard_gate"]),
            "er": float(last["er"]) if np.isfinite(last["er"]) else None,
            "trail_stop": trail, "as_of": pd.Timestamp(last["date"]),
        })
    return out


def signals_to_text(signals: list) -> str:
    """实盘信号转可读文本(CLI/推送用)。"""
    icon = {"进场": "🟢 杀入", "持有": "🔵 持有", "观望": "⚪ 观望", "刚离场": "🔴 刚离场", "无数据": "⚠️ 无数据"}
    lines = ["📡 成长板进攻信号 (v2: 日线强势闸门 + 60min突破 + ER趋势质量 + ATR移动止损)"]
    for s in signals:
        head = f"{icon.get(s['action'], s['action'])} {s['board']}({s['etf']})"
        if s["action"] == "无数据":
            lines.append(head + " — 缺 60min 数据, 先采集"); continue
        detail = f"收盘 {s['close']:.0f} | 日线强势 {'✓' if s['conviction'] else '✗'} | ER {s['er']:.2f}" if s["er"] is not None else ""
        if s["action"] == "持有" and s["trail_stop"]:
            detail += f" | 移动止损 {s['trail_stop']:.0f}(跌破离场)"
        if s["action"] == "进场":
            detail += " | 强势+突破确认, 下根开盘杀入"
        lines.append(f"{head} — {detail} | {s['as_of']:%m-%d %H:%M}")
    lines.append("\n⚠️ 仅进攻信号, 无防守切换; 低频高确定性思路, 非投资建议。")
    return "\n".join(lines)


def perf_summary(nav: pd.Series) -> dict:
    """净值 -> 累计收益/最大回撤/年化。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {"total": np.nan, "max_dd": np.nan, "cagr": np.nan}
    total = float(nav.iloc[-1] - 1)
    max_dd = float((nav / nav.cummax() - 1).min())
    years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if nav.iloc[-1] > 0 else np.nan
    return {"total": total, "max_dd": max_dd, "cagr": cagr}

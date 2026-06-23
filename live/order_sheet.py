"""操作单: 把目标持仓转成可直接下单的买卖清单(含价格/数量/止损/止盈)。

价格基准: 最新交易日收盘价(从 qlib 读取)。
- 买入: 限价 = 收盘 × (1+缓冲), 略高确保成交; 止损 = 买入价×(1-stop); 止盈 = 买入价×(1+tp)
- 卖出: 限价 = 收盘 × (1-缓冲), 略低确保成交
- 数量: 按目标金额 / 买入价, 取整到100股
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from common import setup_logger, to_raw_code, load_config

log = setup_logger("live.order_sheet")


def get_latest_prices(qlib_codes: list, cfg: dict) -> dict:
    """从 qlib 读取最新收盘价。返回 {qlib_code: price}。"""
    import qlib
    from qlib.data import D
    from backtest.qlib_runner import init_qlib
    init_qlib(cfg)
    df = D.features(qlib_codes, ["$close"], start_time="2026-06-01")
    if df.empty:
        return {}
    # 取每只股票最后一根
    last = df.groupby(level=0).tail(1) if isinstance(df.index, pd.MultiIndex) else df.tail(len(qlib_codes))
    prices = {}
    idx_level = 0 if not isinstance(df.index, pd.MultiIndex) else (
        "instrument" if "instrument" in df.index.names else 0)
    for code in qlib_codes:
        try:
            sub = df.xs(code, level=idx_level) if isinstance(df.index, pd.MultiIndex) else df
            if not sub.empty:
                prices[code] = float(sub["$close"].iloc[-1])
        except Exception:
            pass
    return prices


def build_order_sheet(target: pd.DataFrame, current: dict, total_capital: float,
                      prices: dict, stop_loss: float = 0.08, take_profit: float = 0.20,
                      buffer: float = 0.003) -> pd.DataFrame:
    """生成操作单。

    target: 目标持仓 [qlib_code, code, name, score, weight]
    current: {qlib_code: {shares, cost}}
    prices: {qlib_code: 最新收盘价}
    返回 DataFrame: code, name, action, price, shares, amount, stop_loss, take_profit, reason
    """
    target = target.set_index("qlib_code")
    holdings = current.get("holdings", {})
    rows = []
    all_codes = list(set(target.index) | set(holdings.keys()))

    for code in all_codes:
        in_target = code in target.index
        in_hold = code in holdings
        cur_shares = holdings.get(code, {}).get("shares", 0) if in_hold else 0
        px = prices.get(code)
        name = target.loc[code, "name"] if in_target else "-"
        raw_code = target.loc[code, "code"] if in_target else code[2:]

        if not in_target:
            # 不在目标 -> 卖出
            if cur_shares > 0 and px:
                sell_price = round(px * (1 - buffer), 2)
                rows.append({
                    "code": raw_code, "name": name, "action": "卖出",
                    "price": sell_price, "shares": cur_shares,
                    "amount": round(sell_price * cur_shares, 0),
                    "stop_loss": "-", "take_profit": "-",
                    "reason": "退出持仓",
                })
            continue

        weight = float(target.loc[code, "weight"])
        tgt_value = weight * total_capital
        if not px:
            continue
        tgt_shares = int(tgt_value / px / 100) * 100  # 取整到100股
        delta = tgt_shares - cur_shares

        if delta > 0:
            buy_price = round(px * (1 + buffer), 2)
            rows.append({
                "code": raw_code, "name": name, "action": "买入",
                "price": buy_price, "shares": delta,
                "amount": round(buy_price * delta, 0),
                "stop_loss": round(buy_price * (1 - stop_loss), 2),
                "take_profit": round(buy_price * (1 + take_profit), 2),
                "reason": f"目标权重{weight:.0%}, 加仓至{tgt_shares}股",
            })
        elif delta < 0:
            sell_price = round(px * (1 - buffer), 2)
            rows.append({
                "code": raw_code, "name": name, "action": "卖出",
                "price": sell_price, "shares": -delta,
                "amount": round(sell_price * (-delta), 0),
                "stop_loss": "-", "take_profit": "-",
                "reason": f"目标权重{weight:.0%}, 减仓至{tgt_shares}股",
            })
        else:
            rows.append({
                "code": raw_code, "name": name, "action": "持有",
                "price": round(px, 2), "shares": cur_shares,
                "amount": round(px * cur_shares, 0),
                "stop_loss": "-", "take_profit": "-",
                "reason": "已达目标仓位",
            })

    # 买入在前, 卖出在后
    order = {"买入": 0, "卖出": 1, "持有": 2}
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_o"] = df["action"].map(order)
        df = df.sort_values(["_o", "amount"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    return df


def to_markdown(orders: pd.DataFrame, trade_date: str, capital: float) -> str:
    lines = [f"# 📋 操作单 (参考交易日 {trade_date})\n"]
    lines.append(f"总资金: ¥{capital:,.0f} | 价格基准: 最新收盘价 | 买入含±{0.3:.1f}%缓冲 | 止损8% 止盈20%\n")
    buys = orders[orders["action"] == "买入"]
    sells = orders[orders["action"] == "卖出"]
    lines.append(f"**买入 {len(buys)} 笔 | 卖出 {len(sells)} 笔**\n")
    if not orders.empty:
        lines.append("| 代码 | 名称 | 操作 | 委托价 | 数量(股) | 金额(元) | 止损价 | 止盈价 | 说明 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in orders.iterrows():
            lines.append(f"| {r['code']} | {r['name']} | {r['action']} | {r['price']} | "
                         f"{r['shares']} | {r['amount']:,.0f} | {r['stop_loss']} | {r['take_profit']} | {r['reason']} |")
    lines.append("\n**使用说明**: 买入单按「委托价」限价挂单(略高于收盘确保成交); "
                 "买入后按「止损价」挂保护单; 到「止盈价」分批止盈。卖出单按委托价限价卖出。")
    lines.append("\n⚠️ 仅供参考, 不构成投资建议。实际下单请结合盘口与个人风险偏好。")
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = load_config()
    from live.signal_generator import generate_target_portfolio
    from live.portfolio_builder import load_current_holdings
    target = generate_target_portfolio(cfg)
    codes = target["qlib_code"].tolist()
    prices = get_latest_prices(codes, cfg)
    current = load_current_holdings(cfg["live"]["holdings_file"])
    capital = 1_000_000
    orders = build_order_sheet(target, current, capital, prices)
    print(to_markdown(orders, "2026-06-23", capital))

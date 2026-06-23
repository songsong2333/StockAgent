"""调仓清单构建: 当前持仓 vs 目标持仓 -> 买卖清单。

当前持仓文件格式 (YAML):
  cash: 100000          # 现金(元)
  holdings:             # 持仓
    SZ000001:
      shares: 1000       # 股数
      cost: 12.5         # 成本价
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

from common import setup_logger, to_qlib_code

log = setup_logger("live.portfolio")


def load_current_holdings(path: str) -> dict:
    """读取当前持仓 YAML。不存在则返回空持仓。"""
    p = Path(path)
    if not p.exists():
        log.warning(f"当前持仓文件不存在: {p}, 视为空仓")
        return {"cash": 0.0, "holdings": {}}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"cash": 0.0, "holdings": {}}


def build_rebalance(target: pd.DataFrame, current: dict, total_capital: float) -> Tuple[pd.DataFrame, dict]:
    """生成调仓清单。

    target: 目标持仓 DataFrame[qlib_code, weight, ...]
    current: 当前持仓 dict
    total_capital: 总资金(元), 用于计算目标股数

    返回 (orders_df, summary)。
    orders_df: qlib_code, code, name, action(buy/sell/hold), target_value, target_shares,
               current_shares, delta_shares, est_price(占位0,盘前填)
    """
    target = target.set_index("qlib_code")
    holdings = current.get("holdings", {})
    # 用最新价估算(此处留占位, 实盘前由 signal 阶段或人工填)
    target["target_value"] = (target["weight"] * total_capital).round(2)

    rows = []
    all_codes = set(target.index) | set(holdings.keys())
    for code in all_codes:
        in_target = code in target.index
        in_hold = code in holdings
        cur_shares = holdings.get(code, {}).get("shares", 0) if in_hold else 0
        if in_target:
            tv = float(target.loc[code, "target_value"])
            # 目标股数(按100股取整), 需价格; 此处用占位价0, 盘前填入后重算
            tgt_shares = int(tv / 100) * 100 if tv > 0 else 0
            delta = tgt_shares - cur_shares
            if delta > 0:
                action = "buy"
            elif delta < 0:
                action = "sell"
            else:
                action = "hold"
            rows.append({
                "qlib_code": code,
                "code": target.loc[code, "code"],
                "name": target.loc[code, "name"],
                "action": action,
                "target_value": tv,
                "target_shares": tgt_shares,
                "current_shares": cur_shares,
                "delta_shares": delta,
                "est_price": 0.0,
            })
        else:
            # 不在目标, 全部卖出
            rows.append({
                "qlib_code": code,
                "code": code[2:],
                "name": "-",
                "action": "sell" if cur_shares > 0 else "hold",
                "target_value": 0.0,
                "target_shares": 0,
                "current_shares": cur_shares,
                "delta_shares": -cur_shares,
                "est_price": 0.0,
            })

    orders = pd.DataFrame(rows)
    summary = {
        "buy_count": int((orders["action"] == "buy").sum()),
        "sell_count": int((orders["action"] == "sell").sum()),
        "hold_count": int((orders["action"] == "hold").sum()),
        "total_capital": total_capital,
    }
    log.info(f"调仓清单: 买{summary['buy_count']} 卖{summary['sell_count']} 持{summary['hold_count']}")
    return orders, summary


def to_markdown(orders: pd.DataFrame, summary: dict) -> str:
    """调仓清单转 markdown。"""
    lines = ["# 📊 调仓清单\n"]
    lines.append(f"总资金: ¥{summary['total_capital']:,.0f} | "
                 f"买入 {summary['buy_count']} | 卖出 {summary['sell_count']} | 持有 {summary['hold_count']}\n")
    lines.append("\n## 卖出\n")
    sells = orders[orders["action"] == "sell"]
    lines.append("| 代码 | 名称 | 持仓股数 | 目标股数 | 调整股数 |")
    lines.append("|---|---|---|---|---|")
    for _, r in sells.iterrows():
        lines.append(f"| {r['code']} | {r['name']} | {r['current_shares']} | {r['target_shares']} | {r['delta_shares']} |")
    lines.append("\n## 买入\n")
    buys = orders[orders["action"] == "buy"]
    lines.append("| 代码 | 名称 | 目标金额 | 目标股数 | 调整股数 |")
    lines.append("|---|---|---|---|---|")
    for _, r in buys.iterrows():
        lines.append(f"| {r['code']} | {r['name']} | ¥{r['target_value']:,.0f} | {r['target_shares']} | +{r['delta_shares']} |")
    return "\n".join(lines)

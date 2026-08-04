"""实盘调仓逻辑单测(合成数据, 确定性, 无网络/无 qlib)。

覆盖:
- portfolio_builder.build_rebalance: 买/卖/持分类、目标金额、清仓逻辑
- order_sheet.build_order_sheet: 真实股数(金额/价取整到100)、止损止盈、代码前导零
这些是每日会推送给用户的数字, 必须锁死。
"""
import pandas as pd

from live.portfolio_builder import build_rebalance, to_markdown
from live.order_sheet import build_order_sheet


def _target(rows):
    return pd.DataFrame(rows)


# ---------- build_rebalance ----------
def test_rebalance_buy_sell_classification():
    target = _target([
        {"qlib_code": "SZ000001", "code": "000001", "name": "平安银行", "weight": 0.5},
        {"qlib_code": "SH600000", "code": "600000", "name": "浦发银行", "weight": 0.5},
    ])
    current = {"cash": 0.0, "holdings": {
        "SZ000001": {"shares": 100, "cost": 10.0},   # 在目标 -> 加仓(buy)
        "SH601398": {"shares": 200, "cost": 5.0},    # 不在目标 -> 清仓(sell)
    }}
    orders, summary = build_rebalance(target, current, total_capital=100_000)
    o = orders.set_index("qlib_code")
    assert o.loc["SZ000001", "action"] == "buy"
    assert o.loc["SH600000", "action"] == "buy"
    assert o.loc["SH601398", "action"] == "sell"
    assert o.loc["SH601398", "delta_shares"] == -200
    assert summary["buy_count"] == 2
    assert summary["sell_count"] == 1


def test_rebalance_target_value_from_weight():
    target = _target([{"qlib_code": "SZ000001", "code": "000001", "name": "A", "weight": 0.3}])
    orders, _ = build_rebalance(target, {"cash": 0.0, "holdings": {}}, total_capital=200_000)
    # 目标金额 = 权重 × 总资金
    assert abs(orders.iloc[0]["target_value"] - 60_000) < 1e-6


def test_rebalance_markdown_has_sections():
    target = _target([{"qlib_code": "SZ000001", "code": "000001", "name": "A", "weight": 1.0}])
    orders, summary = build_rebalance(target, {"cash": 0.0, "holdings": {}}, total_capital=10_000)
    md = to_markdown(orders, summary)
    assert "买入" in md and "调仓清单" in md


# ---------- build_order_sheet ----------
def test_order_sheet_buy_shares_and_stops():
    target = _target([{"qlib_code": "SZ000001", "code": "000001", "name": "平安银行",
                       "score": 0.5, "weight": 0.2}])
    sheet = build_order_sheet(target, {"cash": 0.0, "holdings": {}},
                              total_capital=1_000_000, prices={"SZ000001": 100.0})
    r = sheet.iloc[0]
    # 目标金额 20万 / 价100 = 2000股(整手)
    assert r["action"] == "买入"
    assert r["shares"] == 2000
    assert r["price"] == 100.3                 # 100 × (1+0.3%缓冲)
    assert r["stop_loss"] == 92.28             # 100.3 × 0.92
    assert r["take_profit"] == 120.36          # 100.3 × 1.20
    assert r["amount"] == 200_600              # 100.3 × 2000
    assert r["code"] == "000001"               # 前导零保留


def test_order_sheet_sell_held_not_in_target():
    target = _target([{"qlib_code": "SZ000001", "code": "000001", "name": "A",
                       "score": 0.5, "weight": 0.2}])
    current = {"cash": 0.0, "holdings": {"SH600000": {"shares": 500, "cost": 10.0}}}
    sheet = build_order_sheet(target, current, total_capital=1_000_000,
                              prices={"SZ000001": 100.0, "SH600000": 20.0})
    sells = sheet[sheet["action"] == "卖出"]
    assert len(sells) == 1
    s = sells.iloc[0]
    assert s["code"] == "600000"
    assert s["shares"] == 500
    assert s["price"] == 19.94                 # 20 × (1-0.3%缓冲)


def test_order_sheet_hold_when_at_target():
    # 已在目标股数 -> 持有(不产生买卖)
    target = _target([{"qlib_code": "SZ000001", "code": "000001", "name": "A",
                       "score": 0.5, "weight": 0.2}])
    # 20万股目标 / 100元 = 2000股
    current = {"cash": 0.0, "holdings": {"SZ000001": {"shares": 2000, "cost": 90.0}}}
    sheet = build_order_sheet(target, current, total_capital=1_000_000,
                              prices={"SZ000001": 100.0})
    assert sheet.iloc[0]["action"] == "持有"

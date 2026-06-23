"""A股量化交易系统 - Mac 桌面应用 (Streamlit)。

启动:
  streamlit run app/app.py
或:
  python app/launcher.py     # 自动打开浏览器

页面:
  概览 / 配置 / 数据采集 / 回测 / 信号调仓
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 app 能导入项目根目录的模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from common import load_config, save_config, to_qlib_code, to_raw_code

st.set_page_config(page_title="A股量化交易系统", page_icon="📈", layout="wide")


# ============== 工具函数 ==============
def get_cfg() -> dict:
    return load_config()


def count_raw_data(cfg) -> int:
    d = Path(cfg["paths"]["raw_dir"])
    if not d.is_absolute():
        d = ROOT / d
    return len(list(d.glob("*.parquet"))) if d.exists() else 0


def qlib_ready(cfg) -> bool:
    d = Path(cfg["qlib"]["provider_uri"])
    if not d.is_absolute():
        d = ROOT / d
    return (d / "calendars").exists()


# ============== 页面: 概览 ==============
def page_overview():
    st.title("📈 A股量化交易系统")
    st.caption("akshare + qlib · 多因子选股 · 信号提醒")

    cfg = get_cfg()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始数据股票数", count_raw_data(cfg))
    c2.metric("qlib数据就绪", "✅" if qlib_ready(cfg) else "❌")
    c3.metric("目标持仓数", cfg["backtest"]["topk"])
    c4.metric("调仓方式", "信号提醒")

    st.divider()
    st.subheader("系统状态")
    status = []
    status.append(("数据源", "akshare (免费)"))
    status.append(("回测框架", "qlib + Alpha158 + LightGBM"))
    status.append(("策略", "多因子选股 TopkDropout"))
    status.append(("原始数据目录", cfg["paths"]["raw_dir"]))
    status.append(("qlib数据目录", cfg["qlib"]["provider_uri"]))
    status.append(("训练区间", f"{cfg['model']['train_start']} ~ {cfg['model']['train_end']}"))
    status.append(("测试区间", f"{cfg['model']['test_start']} ~ {cfg['model']['test_end']}"))
    st.table(pd.DataFrame(status, columns=["项目", "值"]))

    # 最新信号
    sig = ROOT / cfg["paths"]["cache_dir"] / "latest_signal.md"
    if sig.exists():
        st.divider()
        st.subheader("最新调仓信号")
        st.markdown(sig.read_text(encoding="utf-8"))
    else:
        st.info("暂无调仓信号, 请先在「信号调仓」页生成, 或运行 scripts/run_daily.py")


# ============== 页面: 配置 ==============
def page_config():
    st.title("⚙️ 配置")
    cfg = get_cfg()

    with st.form("config_form"):
        st.subheader("数据采集")
        col = st.columns(3)
        cfg["collector"]["adjust"] = col[0].selectbox("复权方式", ["qfq", "hfq", ""], index=0)
        cfg["collector"]["start_date"] = col[1].date_input("历史起始日期", pd.Timestamp(cfg["collector"]["start_date"]))
        cfg["collector"]["request_sleep"] = col[2].number_input("请求间隔(秒)", 0.0, 5.0, float(cfg["collector"]["request_sleep"]))
        cc = st.columns(4)
        cfg["collector"]["exclude_st"] = cc[0].checkbox("剔除ST", value=cfg["collector"]["exclude_st"])
        cfg["collector"]["exclude_new_days"] = cc[1].number_input("新股最少交易日", 0, 1000, int(cfg["collector"]["exclude_new_days"]))
        cfg["collector"]["exclude_kcb"] = cc[2].checkbox("剔除科创板", value=cfg["collector"]["exclude_kcb"])
        cfg["collector"]["exclude_bj"] = cc[3].checkbox("剔除北交所", value=cfg["collector"]["exclude_bj"])

        st.subheader("模型")
        mc = st.columns(4)
        cfg["model"]["model_class"] = mc[0].selectbox("模型", ["lgbm", "linear"], index=0)
        cfg["model"]["train_start"] = str(mc[1].date_input("训练起", pd.Timestamp(cfg["model"]["train_start"])))
        cfg["model"]["train_end"] = str(mc[2].date_input("训练止", pd.Timestamp(cfg["model"]["train_end"])))
        cfg["model"]["test_start"] = str(mc[3].date_input("测试起", pd.Timestamp(cfg["model"]["test_start"])))
        mc2 = st.columns(1)
        cfg["model"]["test_end"] = str(mc2[0].date_input("测试止", pd.Timestamp(cfg["model"]["test_end"])))

        st.subheader("回测")
        bc = st.columns(4)
        cfg["backtest"]["topk"] = bc[0].number_input("持仓数 topk", 1, 200, int(cfg["backtest"]["topk"]))
        cfg["backtest"]["n_drop"] = bc[1].number_input("每次替换 n_drop", 0, 50, int(cfg["backtest"]["n_drop"]))
        cfg["backtest"]["commission_rate"] = bc[2].number_input("佣金率", 0.0, 0.01, float(cfg["backtest"]["commission_rate"]), format="%.5f")
        cfg["backtest"]["stamp_duty"] = bc[3].number_input("印花税(卖)", 0.0, 0.01, float(cfg["backtest"]["stamp_duty"]), format="%.4f")
        bc2 = st.columns(2)
        cfg["backtest"]["deal_price"] = bc2[0].selectbox("成交价", ["open", "close", "vwap"], index=0)
        cfg["backtest"]["t_plus_1"] = bc2[1].checkbox("T+1", value=cfg["backtest"]["t_plus_1"])

        st.subheader("信号调仓")
        lc = st.columns(3)
        cfg["live"]["topk"] = lc[0].number_input("目标持仓数", 1, 200, int(cfg["live"]["topk"]))
        cfg["live"]["weight_scheme"] = lc[1].selectbox("权重方案", ["equal", "score"], index=0)
        cfg["live"]["rebalance_freq"] = lc[2].selectbox("调仓频率", ["daily", "weekly"], index=1)
        cfg["live"]["dry_run"] = st.checkbox("dry-run (只生成不推送)", value=cfg["live"]["dry_run"])

        st.subheader("通知")
        nc = st.columns(2)
        cfg["notify"]["enable"] = nc[0].checkbox("启用通知", value=cfg["notify"]["enable"])
        cfg["notify"]["webhook_type"] = nc[1].selectbox("webhook类型", ["wechat", "dingtalk"], index=0)
        cfg["notify"]["webhook_url"] = st.text_input("webhook URL", cfg["notify"]["webhook_url"])

        submitted = st.form_submit_button("💾 保存配置")
        if submitted:
            # date_input 转 str
            for k in ("train_start", "train_end", "test_start", "test_end"):
                cfg["model"][k] = str(cfg["model"][k])
            cfg["collector"]["start_date"] = str(cfg["collector"]["start_date"])
            save_config(cfg)
            st.success("配置已保存到 config/config.yaml")

    with st.expander("查看完整配置 (YAML)"):
        import yaml
        st.code(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), language="yaml")


# ============== 页面: 数据采集 ==============
def page_data():
    st.title("🗃️ 数据采集")
    cfg = get_cfg()

    st.subheader("股票池")
    if st.button("刷新股票池"):
        st.session_state["pool"] = None
    if "pool" not in st.session_state or not isinstance(st.session_state.get("pool"), pd.DataFrame):
        with st.spinner("获取股票池..."):
            from collector.stock_pool import get_stock_pool
            try:
                st.session_state["pool"] = get_stock_pool(cfg)
            except Exception as e:
                st.error(f"获取股票池失败: {e}")
                st.session_state["pool"] = None
    pool = st.session_state.get("pool")
    if isinstance(pool, pd.DataFrame):
        st.write(f"共 {len(pool)} 只股票")
        st.dataframe(pool, use_container_width=True, height=300)

    st.divider()
    st.subheader("采集历史数据")
    col = st.columns(3)
    years = col[0].number_input("年数", 1, 20, 5)
    symbols = col[1].text_input("指定代码(逗号分隔, 留空=全市场)", "")
    skip_dump = col[2].checkbox("跳过dump", value=False)

    if st.button("🚀 开始采集", type="primary"):
        from collector.daily_collector import update_all, update_one
        from collector.dump_to_qlib import dump
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.DateOffset(years=int(years))).strftime("%Y-%m-%d")
        cfg["collector"]["start_date"] = start

        if symbols.strip():
            codes = [to_qlib_code(s.strip()) for s in symbols.split(",") if s.strip()]
            with st.spinner(f"采集 {len(codes)} 只股票..."):
                for c in codes:
                    update_one(c, cfg["paths"]["raw_dir"], start, end,
                               cfg["collector"].get("adjust", "qfq"),
                               cfg["collector"]["max_retries"], cfg["collector"]["request_sleep"])
        else:
            with st.spinner("采集全市场 (可能需要数小时, 建议后台运行 scripts/init_history.py)..."):
                update_all(cfg, end=end)

        if not skip_dump:
            with st.spinner("转换为 qlib bin 格式..."):
                dump(cfg, codes=[to_qlib_code(s.strip()) for s in symbols.split(",") if s.strip()] if symbols.strip() else None)
        st.success("采集完成 ✅")
        st.rerun()

    st.divider()
    st.subheader("数据状态")
    st.write(f"原始数据: {count_raw_data(cfg)} 只股票")
    st.write(f"qlib 数据就绪: {'✅' if qlib_ready(cfg) else '❌'}")


# ============== 页面: 回测 ==============
def page_backtest():
    st.title("📊 回测")
    cfg = get_cfg()

    if not qlib_ready(cfg):
        st.warning("qlib 数据未就绪, 请先在「数据采集」页采集并 dump 数据。")
        return

    st.write(f"训练: {cfg['model']['train_start']} ~ {cfg['model']['train_end']} | "
             f"测试: {cfg['model']['test_start']} ~ {cfg['model']['test_end']} | "
             f"topk={cfg['backtest']['topk']} n_drop={cfg['backtest']['n_drop']}")

    if st.button("🏃 运行回测", type="primary"):
        with st.spinner("训练模型 + 回测中 (可能需要几分钟)..."):
            try:
                from backtest.qlib_runner import run_backtest
                from backtest.report import save_report
                metrics, report, positions = run_backtest(cfg)
                save_report(metrics, report, Path(cfg["paths"]["cache_dir"]))
                st.session_state["bt_metrics"] = metrics
                st.session_state["bt_report"] = report
                st.session_state["bt_positions"] = positions
                st.success("回测完成 ✅")
            except Exception as e:
                st.error(f"回测失败: {e}")

    metrics = st.session_state.get("bt_metrics")
    report = st.session_state.get("bt_report")
    if metrics:
        st.subheader("关键指标")
        c1, c2, c3, c4 = st.columns(4)
        # qlib 指标键名: annualized_return, information_ratio, max_drawdown
        c1.metric("年化收益", f"{metrics.get('annualized_return', 0):.2%}")
        c2.metric("夏普/IR", f"{metrics.get('information_ratio', 0):.3f}")
        c3.metric("最大回撤", f"{metrics.get('max_drawdown', 0):.2%}")
        c4.metric("年化波动", f"{metrics.get('std', 0):.2%}")

        with st.expander("全部指标"):
            st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})

        # 净值曲线
        if report is not None and not report.empty:
            st.subheader("净值曲线")
            fig = go.Figure()
            r = report.copy()
            r.index = pd.to_datetime(r.index)
            if "return" in r.columns:
                nav = (1 + r["return"]).cumprod()
                fig.add_trace(go.Scatter(x=r.index, y=nav, name="策略", line=dict(color="#e74c3c")))
            if "bench" in r.columns:
                bench = (1 + r["bench"]).cumprod()
                fig.add_trace(go.Scatter(x=r.index, y=bench, name="沪深300", line=dict(color="#3498db")))
            fig.update_layout(height=450, hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ============== 页面: 信号调仓 ==============
def page_signals():
    st.title("📡 信号调仓")
    cfg = get_cfg()

    st.subheader("当前持仓")
    holdings_path = ROOT / cfg["live"]["holdings_file"]
    holdings_path.parent.mkdir(parents=True, exist_ok=True)
    if not holdings_path.exists():
        import yaml
        holdings_path.write_text("cash: 10000000\nholdings: {}\n", encoding="utf-8")
    holdings_text = st.text_area("编辑当前持仓 (YAML)", holdings_path.read_text(encoding="utf-8"), height=150)
    if st.button("💾 保存持仓"):
        holdings_path.write_text(holdings_text, encoding="utf-8")
        st.success("持仓已保存")

    st.divider()
    st.subheader("生成目标持仓与调仓清单")
    total_capital = st.number_input("总资金(元)", value=10_000_000, step=100_000)
    if st.button("🎯 生成信号", type="primary"):
        if not qlib_ready(cfg):
            st.warning("qlib 数据未就绪, 请先采集数据。")
            return
        with st.spinner("跑模型打分生成目标持仓..."):
            try:
                from live.signal_generator import generate_target_portfolio, save_portfolio
                from live.portfolio_builder import load_current_holdings, build_rebalance, to_markdown
                from live.order_sheet import get_latest_prices, build_order_sheet, to_markdown as order_md
                target = generate_target_portfolio(cfg)
                save_portfolio(target, cfg)
                st.session_state["target"] = target

                current = load_current_holdings(cfg["live"]["holdings_file"])
                orders, summary = build_rebalance(target, current, total_capital)
                md = to_markdown(orders, summary)
                (ROOT / cfg["paths"]["cache_dir"]).mkdir(parents=True, exist_ok=True)
                (ROOT / cfg["paths"]["cache_dir"] / "latest_signal.md").write_text(md, encoding="utf-8")
                st.session_state["orders"] = orders
                st.session_state["rebal_md"] = md

                # 操作单 (含买卖价/数量/止损/止盈)
                prices = get_latest_prices(target["qlib_code"].tolist(), cfg)
                sheet = build_order_sheet(target, current, total_capital, prices)
                sheet_md = order_md(sheet, pd.Timestamp.now().strftime("%Y-%m-%d"), total_capital)
                (ROOT / cfg["paths"]["cache_dir"] / "latest_order_sheet.md").write_text(sheet_md, encoding="utf-8")
                st.session_state["order_sheet"] = sheet
                st.session_state["order_sheet_md"] = sheet_md
                st.success("信号生成完成 ✅")
            except Exception as e:
                st.error(f"生成失败: {e}")

    target = st.session_state.get("target")
    if target is not None:
        st.subheader("目标持仓")
        st.dataframe(target, use_container_width=True)

    orders = st.session_state.get("orders")
    if orders is not None:
        st.subheader("调仓清单")
        st.dataframe(orders, use_container_width=True)
        st.markdown(st.session_state.get("rebal_md", ""))

    sheet = st.session_state.get("order_sheet")
    if sheet is not None:
        st.subheader("📋 操作单 (可直接下单)")
        st.caption("委托价=最新收盘价±0.3%缓冲 | 止损8% | 止盈20% | 数量取整到100股")
        st.dataframe(sheet, use_container_width=True, hide_index=True)
        with st.expander("操作单说明"):
            st.markdown(st.session_state.get("order_sheet_md", ""))

    if st.button("📨 推送通知"):
        from live.notify import notify
        md = st.session_state.get("order_sheet_md") or st.session_state.get("rebal_md")
        if md:
            ok = notify(cfg, md)
            st.success("已推送" if ok else "推送未成功 (检查 notify 配置)")
        else:
            st.warning("请先生成信号")


# ============== 主导航 ==============
PAGES = {
    "📈 概览": page_overview,
    "⚙️ 配置": page_config,
    "🗃️ 数据采集": page_data,
    "📊 回测": page_backtest,
    "📡 信号调仓": page_signals,
}

st.sidebar.title("A股量化系统")
choice = st.sidebar.radio("导航", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("akshare · qlib · 多因子选股")
PAGES[choice]()

# 每日任务快捷按钮
st.sidebar.divider()
if st.sidebar.button("▶️ 运行每日任务"):
    st.sidebar.info("请在终端运行: `python scripts/run_daily.py`")

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

# 让 app 能导入项目根目录的模块 (源码模式生效; 打包模式模块已内置, 无害)
_CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_ROOT))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from common import (
    load_config, save_config, to_qlib_code, to_raw_code, PROJECT_ROOT,
)
from strategy.rotation import metrics_from_returns
# 数据/配置路径统一指向: 源码模式=项目根, 打包模式=用户数据目录
ROOT = PROJECT_ROOT

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


@st.cache_data(show_spinner=False, ttl=600)
def _names() -> dict:
    """{6位代码: 名称}。本地优先(自选/龙头池/持仓 yaml, 无网络依赖) + akshare全量兜底。"""
    nm = {}
    # 1) 本地 yaml(可靠) —— 用户关注的股都在这几个文件里
    for f in ("config/watchlist.yaml", "config/sector_leaders.yaml"):
        try:
            for s in _load_yaml_stocks(f):
                if s.get("code"):
                    nm[str(s["code"])] = s.get("name", "")
        except Exception:
            pass
    # 2) 当前持仓
    try:
        cfg = load_config()
        from live.portfolio_builder import load_current_holdings
        for qc, h in (load_current_holdings(cfg["live"]["holdings_file"]).get("holdings", {}) or {}).items():
            nm[qc[2:]] = h.get("name", "")
    except Exception:
        pass
    # 3) akshare 全量(尽力而为, 失败不影响本地)
    try:
        from collector.stock_pool import get_all_a_shares
        df = get_all_a_shares()
        for c, n in zip(df["code"], df["name"]):
            nm.setdefault(str(c), str(n))
    except Exception:
        pass
    return nm


@st.cache_data(show_spinner="建立搜索索引...", ttl=600)
def _search_index() -> pd.DataFrame:
    """全A股搜索索引: [code, name, py(拼音首字母)]。供按代码/名称/拼音搜索。"""
    from pypinyin import lazy_pinyin, Style
    nm = _names()
    codes, names, pys = [], [], []
    for code, name in nm.items():
        try:
            py = "".join(lazy_pinyin(str(name), style=Style.FIRST_LETTER, errors="ignore"))
        except Exception:
            py = ""
        codes.append(code); names.append(str(name)); pys.append(py.lower())
    return pd.DataFrame({"code": codes, "name": names, "py": pys})


def _safe_table(df, empty_msg: str = "暂无数据"):
    """安全渲染表格: 空或异常时显示提示, 而不是抛堆栈。"""
    try:
        if df is None or (hasattr(df, "empty") and df.empty) or (hasattr(df, "shape") and df.shape[0] == 0):
            st.caption(empty_msg)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"{empty_msg}（渲染异常: {e}）")


@st.cache_data(show_spinner=False, ttl=120)
def _live_closes(codes: list) -> dict:
    """取最新收盘价(新浪源, 盘外/周末也返回最近交易日收盘)。{6位代码: 价格}。失败返回{}。"""
    import pandas as _pd
    from collector.daily_collector import fetch_daily
    end = _pd.Timestamp.now().strftime("%Y-%m-%d")
    start = (_pd.Timestamp.now() - _pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    out = {}
    for c in codes:
        try:
            df = fetch_daily(c, start, end, "qfq")
            if df is not None and not df.empty:
                out[str(c).zfill(6)] = float(df.iloc[-1]["close"])
        except Exception:
            pass
    return out


@st.cache_data(show_spinner=False, ttl=600)
def _analyst_info(code: str) -> dict:
    """取个股最新研报: 评级/机构/日期/盈利预测(EPS,PE)。失败返回{}。"""
    import akshare as ak
    df = ak.stock_research_report_em(symbol=str(code).zfill(6))
    if df is None or df.empty:
        return {}
    r = df.iloc[0]
    eps = r.get("2026-盈利预测-收益")
    pe = r.get("2026-盈利预测-市盈率")
    return {
        "rating": r.get("东财评级") or r.get("评级") or "",
        "org": r.get("机构") or "",
        "date": str(r.get("日期", ""))[:10],
        "eps": float(eps) if pd.notna(eps) else None,
        "pe": float(pe) if pd.notna(pe) else None,
        "implied_price": round(float(eps) * float(pe), 2) if pd.notna(eps) and pd.notna(pe) else None,
        "n_reports": int(r.get("近一月个股研报数", 0) or 0),
    }


def _stock_picker(key: str, exclude: set = None):
    """搜索添加股票组件。返回 (code, name) 或 None。

    支持按 6位代码 / 中文名 / 拼音首字母 模糊搜索(如 300308 / 中际 / zjxc)。
    用于把股票加到表格里(不用手敲代码)。任何异常都不抛, 保证页面不崩。
    """
    exclude = exclude or set()
    st.caption("🔍 搜索添加：输代码/名称/拼音(如 300308 / 中际 / zjxc)→下拉选→添加到表格（也可直接在表格里输名称，保存时自动带代码）")
    try:
        idx = _search_index()
    except Exception:
        idx = pd.DataFrame(columns=["code", "name", "py"])
    # ⚠️ Streamlit 不允许在 widget 创建后改它的 key; 用"待清除"标记, 在创建前清空
    clear_flag = f"_clr_pick_{key}"
    if st.session_state.get(clear_flag):
        st.session_state[f"kh_{key}"] = ""
        st.session_state[f"sel_{key}"] = ""
        st.session_state[clear_flag] = False
    kw = st.text_input("搜索", key=f"kh_{key}",
                       placeholder="代码 / 名称 / 拼音", label_visibility="collapsed")
    if not (kw and kw.strip()):
        return None
    k = kw.strip().lower()
    try:
        sub = idx[(idx["code"].astype(str).str.startswith(k)) |
                  (idx["name"].astype(str).str.contains(k, na=False, regex=False)) |
                  (idx["py"].astype(str).str.contains(k, na=False, regex=False))]
        sub = sub[~sub["code"].isin(exclude)].head(20)
    except Exception:
        sub = pd.DataFrame()
    if sub.empty:
        st.caption("无匹配，换个关键词试试（或直接在表格里填名称）")
        return None
    opts = [f"{r.code}　{r.name}" for r in sub.itertuples(index=False)]
    label = f"匹配 {len(sub)} 只" + ("（更多请细化关键词）" if len(sub) >= 20 else "")
    chosen = st.selectbox(label, options=[""] + opts, key=f"sel_{key}")
    col1, _ = st.columns([1, 3])
    if chosen and col1.button("➕ 添加到表格", key=f"add_{key}", type="primary"):
        code = chosen.split()[0]
        name = idx.loc[idx["code"] == code, "name"].iloc[0]
        st.session_state[clear_flag] = True   # 下次运行前清空搜索框
        return (code, name)
    return None


# ============== 页面: 持仓与推荐 ==============
def page_holdings():
    import yaml
    st.title("💼 持仓与推荐")
    st.caption("当前持仓 · 模型推荐 · 调仓缺口 · 操作单(下单价位) · 推送")
    cfg = get_cfg()
    nm = _names()
    holdings_path = ROOT / cfg["live"]["holdings_file"]
    holdings_path.parent.mkdir(parents=True, exist_ok=True)
    if not holdings_path.exists():
        holdings_path.write_text("cash: 0\nholdings: {}\n", encoding="utf-8")
    from live.portfolio_builder import load_current_holdings, build_rebalance
    from live.order_sheet import get_latest_prices, build_order_sheet

    current = load_current_holdings(cfg["live"]["holdings_file"])
    holdings = current.get("holdings", {}) or {}
    cash = float(current.get("cash", 0) or 0)

    # ---------- ① 持仓编辑(表格式, 不暴露YAML) ----------
    st.subheader("📌 我的持仓")
    st.caption("表格里改/加/删行：**代码或名称填一个即可**（保存时自动互查），再填股数和成本→点保存。底部 ＋ 可加行。")
    edit_rows = [{"代码": qc[2:], "名称": nm.get(qc[2:], h.get("name", "-")),
                  "持仓股数": int(h.get("shares", 0)), "成本价": float(h.get("cost", 0) or 0)}
                 for qc, h in holdings.items()]
    edit_df = pd.DataFrame(edit_rows, columns=["代码", "名称", "持仓股数", "成本价"])
    # 每轮从持仓 yaml 重建(不缓存 session_state), 让 data_editor 自己管编辑状态, 新增行才不会丢
    edited = st.data_editor(
        edit_df, num_rows="dynamic", use_container_width=True, key="holdings_editor",
        column_config={"代码": st.column_config.TextColumn(width="small"),
                       "名称": st.column_config.TextColumn(width="medium"),
                       "持仓股数": st.column_config.NumberColumn(min_value=0, step=100),
                       "成本价": st.column_config.NumberColumn(min_value=0.0, step=0.001, format="%.3f")})

    cash_in = st.number_input("现金(元)", value=cash, step=10000.0, key="cash_in")
    if st.button("💾 保存持仓", type="primary", key="save_holdings"):
        name_to_code = {v: k for k, v in nm.items()} if nm else {}  # 名称->代码 反查
        new_hold, seen, unresolved = {}, set(), []
        for _, r in edited.iterrows():
            raw_code = str(r["代码"]).strip()
            name = str(r["名称"]).strip()
            code = None
            if raw_code and raw_code.isdigit():            # 直接填了代码
                code = raw_code.zfill(6)
            elif name and name in name_to_code:            # 只填了名称 → 反查代码
                code = name_to_code[name]
            elif raw_code and raw_code in name_to_code:    # 名称误填到代码列
                code = name_to_code[raw_code]
            if not code:
                if name or raw_code:
                    unresolved.append(name or raw_code)    # 填了内容但识别不出
                continue
            if not (code.isdigit() and len(code) == 6) or code in seen:
                continue
            seen.add(code)
            new_hold[to_qlib_code(code)] = {
                "shares": int(r["持仓股数"] or 0),
                "cost": float(r["成本价"] or 0),
                "name": name or nm.get(code, "")}
        if unresolved:
            st.warning(f"以下未识别出代码（检查名称/代码是否正确）: {unresolved}")
        holdings_path.write_text(
            yaml.safe_dump({"cash": float(cash_in), "holdings": new_hold},
                           allow_unicode=True, sort_keys=False), encoding="utf-8")
        st.success(f"已保存 {len(new_hold)} 只持仓 ✅"); st.rerun()

    # ---------- ② 持仓调仓建议: 现价/市值/浮盈 + 加仓/减仓/止损价位 ----------
    if holdings:
        st.divider()
        st.subheader("📐 持仓调仓建议（现价/浮盈 + 加仓 / 减仓价位）")
        st.caption("基于每只持仓的技术面(均线/支撑/阻力/ATR)：回踩**加仓区间**低吸，到**减仓位**分批止盈，破**止损**离场。"
                   "想看「模型建议买哪些新票」请去 🧪 自研模型 → 推荐持仓。")
        from live.stock_plan import price_levels
        # 先确保每只持仓有足够日线(price_levels 需 ~260 日历史), 缺的自动补采(新浪源)
        import pandas as _pd
        from collector.daily_collector import update_one
        raw_dir = ROOT / cfg["paths"]["raw_dir"]
        ensure_start = (_pd.Timestamp.now() - _pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        ensure_end = _pd.Timestamp.now().strftime("%Y-%m-%d")
        need = []
        for qc in holdings:
            p = raw_dir / f"{qc}.parquet"
            try:
                short = (not p.exists()) or (len(_pd.read_parquet(p)) < 260)
            except Exception:
                short = True
            if short:
                need.append(qc)
        if need:
            with st.spinner(f"首次分析，补采 {len(need)} 只持仓的日线（约 {len(need)*1}0 秒）..."):
                for qc in need:
                    try:
                        update_one(qc, str(raw_dir), ensure_start, ensure_end,
                                   cfg["collector"].get("adjust", "qfq"),
                                   cfg["collector"]["max_retries"], 0.4)
                    except Exception:
                        pass
        # 实时最新价(周末/盘后也是最新收盘价), 取不到则回退 parquet 末值
        with st.spinner("读取实时行情..."):
            try:
                live_px = _live_closes([qc[2:] for qc in holdings])
            except Exception:
                live_px = {}
        advice_rows, missing, total_mv = [], [], 0.0
        for qc, h in holdings.items():
            code = qc[2:]
            shares = int(h.get("shares", 0))
            cost = float(h.get("cost", 0) or 0)
            try:
                p = price_levels(code, cfg, live_px=live_px.get(code))
            except Exception as e:
                missing.append(f"{code}: {type(e).__name__}: {e}")
                continue
            px = live_px.get(code) or p["last_close"]   # 优先实时价
            mv = shares * px
            total_mv += mv
            pnl = (px / cost - 1) if cost else None
            # 机构研报(尽力而为)
            try:
                ai = _analyst_info(code)
            except Exception:
                ai = {}
            sup = p["support"][0] if p["support"] else None
            res = p["resistance"]
            rtouch = p.get("resistance_touches", [])

            def _lvl(vals, touches, i):
                """安全取第 i 个价位(带触及次数); 长度不够返回'-'。"""
                if len(vals) <= i:
                    return "-"
                t = touches[i] if len(touches) > i else 0
                return f"{round(vals[i], 2)}" + (f" ({t}次)" if t else "")

            advice_rows.append({
                "代码": code, "名称": nm.get(code, h.get("name", "-")), "持仓": shares,
                "成本": round(cost, 3), "现价": round(px, 2), "市值": round(mv),
                "浮盈": f"{pnl:+.1%}" if pnl else "-",
                "趋势": p["trend"],
                "距250日高": f"{p['pos_high']:+.1%}",
                "支撑位": round(sup, 2) if sup else "-",
                "加仓区间": f"{p['entry_low']:.2f}~{p['entry_high']:.2f}",
                "压力位1": _lvl(res, rtouch, 0),
                "压力位2": _lvl(res, rtouch, 1),
                "止损": round(p["stop"], 2),
                "机构评级": f"{ai.get('rating','')}{('·'+ai['org']) if ai.get('org') else ''}" or "-",
                "研报估值(26E)": (f"EPS {ai['eps']:.2f}×PE {ai['pe']:.0f}≈{ai['implied_price']}"
                                 if ai.get("implied_price") else "-"),
            })
        if advice_rows:
            st.dataframe(pd.DataFrame(advice_rows), use_container_width=True, hide_index=True,
                         column_config={
                             "止损": st.column_config.NumberColumn(
                                 help="止损价 = 入场区下沿 − 1.5×ATR(14)。\n"
                                      "• ATR(14) = 近14日「平均真实波幅」，即每天正常波动幅度均值。\n"
                                      "• 1.5×ATR = 给正常波动留的余地；跌破说明不是正常震荡、是趋势坏了→离场。\n"
                                      "• 入场下沿：急涨偏离时=MA10/MA20下方；正常时=近端支撑/MA20下方。\n"
                                      "• 注：现相对「假设加仓入场区」算，急涨股的止损离现价较远。"),
                             "加仓区间": st.column_config.TextColumn(
                                 help="回踩低吸区。急涨偏离时=MA10/MA20附近(不追高)；正常时=近端支撑~现价。回到此区间可加仓。"),
                             "压力位1": st.column_config.TextColumn(
                                 help="近30日盘中高点(高于现价)，1.5%内合并为一，括号内=触及次数(越多阻力越实)。到价可分批减仓。"),
                             "压力位2": st.column_config.TextColumn(
                                 help="第二档压力位(更远)。同上算法，触及次数越多越关键。"),
                             "支撑位": st.column_config.TextColumn(
                                 help="近30日盘中低点(低于现价)，1.5%内合并为一。触及次数越多支撑越扎实。"),
                             "研报估值(26E)": st.column_config.TextColumn(
                                 help="估值价 = 2026年预测EPS × 预测PE。\n"
                                      "• EPS = 机构对2026年每股收益的一致/最新预测。\n"
                                      "• PE = 对应2026E EPS的市盈率。\n"
                                      "• 估值价 = EPS×PE，即「按机构预测的盈利和PE，该值多少钱」。\n"
                                      "• 对比现价：估值价>现价=机构隐含看涨空间；<现价=偏贵。"),
                         })
            st.caption(f"股票市值 ¥{total_mv:,.0f} · 现金 ¥{cash:,.0f} · 合计 ¥{total_mv+cash:,.0f}。"
                       "现价=实时最新价；加仓/减仓/止损 基于历史技术面(数据若旧请去「数据采集」更新)。"
                       "💡 列头有 ⓘ 图标的，鼠标悬停可看计算原理。")
            if missing:
                st.caption(f"⚠️ 这些持仓没算出价位(需先采集日线): {missing}")
        else:
            st.info("暂无可分析的持仓——先到「🧪 自研模型 → 数据采集」把这些股的日线采下来。")

        # 推送调仓建议
        if advice_rows and st.button("📨 推送调仓建议到手机", key="push_advice"):
            from live.notify import notify
            md = pd.DataFrame(advice_rows).to_markdown(index=False)
            ok = notify(cfg, f"📐 持仓调仓建议\n\n{md}", subject="A股持仓调仓建议")
            st.success("已推送" if ok else "推送未成功（检查 notify 设置）")


# ============== 页面: 自选股 ==============
def page_watchlist():
    import yaml
    st.title("⭐ 自选股")
    st.caption("你关注的股票池，驱动模型选股/回测/推荐/盯盘。"
               "**表格底部 ＋ 加行**（或选中行删除），代码/名称填一个即可，改完点「💾 保存自选」。"
               "填完一格后记得按 **Tab/回车** 让单元格落定，再点保存。")
    nm = _names()
    path = ROOT / "config" / "watchlist.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("stocks: []\n", encoding="utf-8")
    # 每轮从 yaml 重建表格(不缓存到 session_state), 让 data_editor 自己管编辑状态
    stocks = _load_yaml_stocks("config/watchlist.yaml")
    edit_df = pd.DataFrame([{"代码": s.get("code", ""), "名称": s.get("name", "")} for s in stocks],
                           columns=["代码", "名称"])
    edited = st.data_editor(
        edit_df, num_rows="dynamic", use_container_width=True,
        key="wl_editor", hide_index=True,
        column_config={"代码": st.column_config.TextColumn(width="small", help="6位代码或名称"),
                       "名称": st.column_config.TextColumn(width="large", help="填名称保存时自动带代码")})

    if st.button("💾 保存自选", type="primary", key="save_watchlist"):
        name_to_code = {v: k for k, v in nm.items()} if nm else {}
        new, seen, unresolved = [], set(), []
        for _, r in edited.iterrows():
            raw = str(r["代码"]).strip()
            name = str(r["名称"]).strip()
            if name in ("", "nan"):
                name = ""
            code = None
            if raw and raw.isdigit():
                code = raw.zfill(6)
            elif name and name in name_to_code:
                code = name_to_code[name]
            elif raw and raw in name_to_code:
                code = name_to_code[raw]
            if not code:
                if name or (raw and raw != "nan"):
                    unresolved.append(name or raw)
                continue
            if not (code.isdigit() and len(code) == 6) or code in seen:
                continue
            seen.add(code)
            new.append({"code": code, "name": name or nm.get(code, "")})
        if unresolved:
            st.warning(f"未识别出代码(检查名称/代码): {unresolved}")
        path.write_text(yaml.safe_dump({"stocks": new}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        st.success(f"已保存 {len(new)} 只自选 ✅")
        st.rerun()

    st.caption("改完自选 → 去「🧪 自研模型 → 数据采集」点增量更新 → 再到「💼 持仓与推荐」重新生成推荐。")



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

    # 持仓与推荐速览
    try:
        from live.portfolio_builder import load_current_holdings
        cur = load_current_holdings(cfg["live"]["holdings_file"])
        n_hold = len(cur.get("holdings", {}) or {})
    except Exception:
        n_hold = 0
    target_csv = ROOT / cfg["paths"]["cache_dir"] / "target_portfolio.csv"
    st.info(f"📌 当前持仓 **{n_hold}** 只 · 🎯 推荐就绪: **{'是' if target_csv.exists() else '否(去「💼持仓与推荐」生成)'}**  "
            f"→ 详细看左侧「**💼 持仓与推荐**」页")

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

        st.subheader("ETF策略")
        etfcfg = cfg.setdefault("etf", {})
        ec1 = st.columns(3)
        etfcfg["benchmark_etf"] = ec1[0].text_input("大盘择时基准ETF", str(etfcfg.get("benchmark_etf", "510300")))
        etfcfg["initial_capital"] = ec1[1].number_input("回测初始资金", 10000, 10000000,
                                                        int(etfcfg.get("initial_capital", 1000000)), 10000)
        etfcfg["backtest_start"] = str(ec1[2].date_input("回测起始日",
                                                         pd.Timestamp(str(etfcfg.get("backtest_start", "2019-01-01")))))
        tc = etfcfg.setdefault("trend", {})
        ec2 = st.columns(4)
        tc["ma_short"] = ec2[0].number_input("趋势-短均线", 5, 60, int(tc.get("ma_short", 20)), key="cfg_etf_ts")
        tc["ma_long"] = ec2[1].number_input("趋势-长均线", 20, 250, int(tc.get("ma_long", 60)), key="cfg_etf_tl")
        tc["confirm_days"] = ec2[2].number_input("趋势-确认日数", 1, 5, int(tc.get("confirm_days", 2)), key="cfg_etf_cd")
        tc["band_filter"] = ec2[3].number_input("趋势-带宽%", 0.0, 5.0,
                                                float(tc.get("band_filter", 0.005)) * 100, key="cfg_etf_bf") / 100
        rc = etfcfg.setdefault("rotation", {})
        ec3 = st.columns(4)
        rc["broad_topn"] = ec3[0].number_input("轮动-宽基持仓", 1, 6, int(rc.get("broad_topn", 3)), key="cfg_etf_rbn")
        rc["sector_topn"] = ec3[1].number_input("轮动-行业持仓", 1, 6, int(rc.get("sector_topn", 3)), key="cfg_etf_rsn")
        rc["mom_long"] = ec3[2].number_input("轮动-长动量", 20, 120, int(rc.get("mom_long", 60)), key="cfg_etf_rml")
        rc["mom_short"] = ec3[3].number_input("轮动-短动量", 5, 40, int(rc.get("mom_short", 20)), key="cfg_etf_rms")

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
def _load_yaml_stocks(path: str) -> list:
    """读 watchlist/sector_leaders 的 stocks/leaders 列表。返回 [{code,name,...}]。"""
    import yaml
    p = ROOT / path
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("leaders") or data.get("stocks") or []


def page_data():
    st.title("🗃️ 数据采集")
    cfg = get_cfg()
    nm = _names()

    wl = _load_yaml_stocks("config/watchlist.yaml")
    ld = _load_yaml_stocks("config/sector_leaders.yaml")
    wl_codes = [to_qlib_code(s["code"]) for s in wl]
    ld_codes = [to_qlib_code(s["code"]) for s in ld]

    st.subheader("采集目标（只采你关注的，不拉全市场）")
    c1, c2, c3 = st.columns(3)
    c1.metric("⭐ 自选股", f"{len(wl_codes)} 只")
    c2.metric("🔁 轮动候选池", f"{len(ld_codes)} 只")
    c3.metric("📈 沪深300基准", "1")
    with st.expander(f"⭐ 自选股 ({len(wl_codes)})"):
        st.dataframe(pd.DataFrame([{"代码": s["code"], "名称": s["name"]} for s in wl]),
                     use_container_width=True, hide_index=True)
    with st.expander(f"🔁 轮动候选池 ({len(ld_codes)})"):
        st.dataframe(pd.DataFrame([{"代码": s["code"], "名称": s["name"], "板块": s.get("sector", "")} for s in ld]),
                     use_container_width=True, hide_index=True)

    target = st.radio("采集范围", ["自选+轮动池+基准（推荐）", "仅自选+基准", "仅轮动池+基准"], horizontal=True, key="collect_scope")
    pick = {"自选+轮动池+基准（推荐）": wl_codes + ld_codes,
            "仅自选+基准": wl_codes,
            "仅轮动池+基准": ld_codes}[target]

    st.divider()
    mode = st.radio("采集模式", ["增量更新（补到最新，快）", "回填历史（指定年数）"], horizontal=True, key="collect_mode")
    do_dump = True
    if mode.startswith("增量"):
        years = None
    else:
        years = st.number_input("回填年数", 1, 10, 2)
        do_dump = st.checkbox("采集后转为 qlib bin", value=True)

    if st.button("🚀 开始采集", type="primary"):
        from collector.daily_collector import update_one
        from collector.index_collector import update_benchmark_indices
        from collector.dump_to_qlib import dump
        import time as _t
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        if years:
            start = (pd.Timestamp.now() - pd.DateOffset(years=int(years))).strftime("%Y-%m-%d")
            cfg["collector"]["start_date"] = start
        else:
            start = cfg["collector"].get("start_date", "2024-07-01")
        bench = cfg["backtest"].get("benchmark", "SH000300")
        progress = st.progress(0.0, text="准备采集...")
        ok = 0
        # 基准
        update_benchmark_indices(cfg["paths"]["raw_dir"], start, end, [bench])
        all_codes = pick
        for i, c in enumerate(all_codes):
            try:
                if update_one(c, cfg["paths"]["raw_dir"], start, end,
                              cfg["collector"].get("adjust", "qfq"),
                              cfg["collector"]["max_retries"], cfg["collector"]["request_sleep"]):
                    ok += 1
            except Exception as e:
                st.warning(f"{c} 失败: {e}")
            progress.progress((i + 1) / len(all_codes), text=f"采集 {c} ({i+1}/{len(all_codes)})")
        if do_dump:
            with st.spinner("转换为 qlib bin..."):
                dump(cfg)
        st.success(f"采集完成: {ok}/{len(all_codes)} 只成功 ✅")
        # 不调 st.rerun(): 下方"数据状态"会在本次渲染刷新; rerun 会与导航 widget 撞 key

    st.divider()
    with st.expander("⚙️ 高级：指定代码 / 全市场（慎用）"):
        st.caption("指定代码（逗号分隔）或全市场（5000+只，耗时数小时，一般用不到）。")
        syms = st.text_input("指定代码", "", key="adv_syms")
        full = st.checkbox("全市场（极慢，仅在确有必要时勾选）")
        if st.button("按高级选项采集"):
            from collector.daily_collector import update_all, update_one
            from collector.dump_to_qlib import dump
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = cfg["collector"].get("start_date", "2024-07-01")
            if full:
                with st.spinner("采集全市场（数小时）..."):
                    update_all(cfg, end=end)
            elif syms.strip():
                for c in [to_qlib_code(s.strip()) for s in syms.split(",") if s.strip()]:
                    update_one(c, cfg["paths"]["raw_dir"], start, end,
                               cfg["collector"].get("adjust", "qfq"),
                               cfg["collector"]["max_retries"], cfg["collector"]["request_sleep"])
            dump(cfg)
            st.success("完成 ✅")
            # 不调 st.rerun()(避免与导航 widget 撞 key); 下方数据状态本次渲染即刷新

    st.divider()
    st.subheader("数据状态")
    st.write(f"原始数据: {count_raw_data(cfg)} 只股票")
    st.write(f"qlib 数据就绪: {'✅' if qlib_ready(cfg) else '❌'}")


# ============== 页面: 回测 ==============
def page_backtest():
    st.title("📊 回测")
    cfg = get_cfg()

    if not qlib_ready(cfg):
        st.warning("qlib 数据未就绪, 请先在「🧪 自研模型 → 数据采集」采集并 dump 数据。")
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


# ============== 页面: 轮动策略 ==============
def page_rotation():
    st.title("🔁 轮动策略")
    st.caption("板块龙头轮动 · 大盘择时 · 追高过滤 · 回撤预算控制")
    cfg = get_cfg()
    rc = cfg.get("rotation", {})

    st.subheader("参数调节 (拖动实时看效果)")
    col = st.columns(3)
    topn = col[0].slider("持仓板块数 topN", 2, 8, int(rc.get("topn", 5)))
    target = col[1].slider("🎯 目标最大回撤", 0.05, 0.30, float(rc.get("max_dd_target", 0.15)), 0.01)
    bias = col[2].slider("乖离上限(越低越不追高)", 0.08, 0.30, float(rc.get("bias_max", 0.20)), 0.01)

    universe = rc.get("universe", "config/sector_leaders.yaml")
    # 满仓日收益只算一次(缓存), 滑块拖动时即时缩仓重算
    @st.cache_data(show_spinner="计算满仓回测...", ttl=600)
    def _full_daily(_topn, _bias, _universe):
        from strategy.rotation import backtest_rotation
        bt = backtest_rotation(cfg, topn=_topn, bias_max=_bias, max_per_sector=1, exposure=1.0)
        return bt["daily_ret"], bt["bench_daily"], bt["dates"]

    daily, bench_daily, dates = _full_daily(topn, bias, universe)
    full = metrics_from_returns(daily, 1.0)
    exposure = min(1.0, target / abs(full["max_dd"])) if full["max_dd"] != 0 else 1.0
    scaled = metrics_from_returns(daily, exposure)
    bench = metrics_from_returns(bench_daily, 1.0)

    # 指标卡
    st.subheader("按你的目标回撤 → 自动仓位")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("建议仓位", f"{exposure:.0%}", f"目标回撤 {target:.0%}")
    c2.metric("实际回撤", f"{scaled['max_dd']:.1%}", f"满仓 {full['max_dd']:.1%}")
    c3.metric("累计收益", f"{scaled['cum_ret']:+.1%}", f"满仓 {full['cum_ret']:+.1%}")
    c4.metric("夏普", f"{scaled['sharpe']:.2f}", f"满仓 {full['sharpe']:.2f}")

    # 净值曲线 (满仓 vs 缩仓 vs 基准)
    st.subheader("净值曲线")
    fig = go.Figure()
    dts = pd.to_datetime(dates)
    fig.add_trace(go.Scatter(x=dts, y=full["curve"], name=f"满仓轮动(回撤{full['max_dd']:.0%})",
                             line=dict(color="#bbb", dash="dot")))
    fig.add_trace(go.Scatter(x=dts, y=scaled["curve"], name=f"缩仓 {exposure:.0%}(回撤{scaled['max_dd']:.0%})",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=dts, y=bench["curve"], name=f"沪深300(回撤{bench['max_dd']:.0%})",
                             line=dict(color="#3498db")))
    fig.update_layout(height=420, hovermode="x unified", template="plotly_white", yaxis_title="净值")
    st.plotly_chart(fig, use_container_width=True)

    # 回撤目标权衡表
    with st.expander("📊 回撤目标权衡表"):
        rows = []
        for t in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
            ex = min(1.0, t / abs(full["max_dd"])) if full["max_dd"] else 1.0
            m = metrics_from_returns(daily, ex)
            mark = "👈 当前" if abs(t - target) < 0.005 else ""
            rows.append({"目标回撤": f"{t:.0%}", "仓位": f"{ex:.0%}",
                         "实际回撤": f"{m['max_dd']:.1%}", "累计收益": f"{m['cum_ret']:+.1%}",
                         "夏普": f"{m['sharpe']:.2f}", "": mark})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("仓位越低夏普往往越高 → 满仓偏激进, 适度降杠杆提升风险调整收益。")

    # 当前信号
    st.subheader("本期轮动信号")
    if st.button("🔄 生成最新信号", type="primary"):
        with st.spinner("计算信号..."):
            from strategy.rotation import rotation_signal, signal_to_markdown
            st.session_state["rot_sig"] = rotation_signal(cfg, topn=topn, max_per_sector=1)
    sig = st.session_state.get("rot_sig")
    if sig:
        st.markdown(signal_to_markdown(sig))


# ============== 页面: ETF策略 ==============
_ETF_HOLDINGS_FILE = "data/cache/etf_holdings.yaml"


def _load_etf_holdings() -> dict:
    """加载ETF持仓 {code: {name, shares, cost, last_px}}。不存在返回 {}。"""
    import yaml
    p = ROOT / _ETF_HOLDINGS_FILE
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("holdings", {})


def _save_etf_holdings(holdings: dict):
    import yaml
    p = ROOT / _ETF_HOLDINGS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"holdings": holdings}, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")


@st.cache_data(show_spinner="获取ETF实时行情...", ttl=300)
def _etf_spot_map() -> dict:
    """全市场ETF实时行情 → {code: {name, last_px}}。缓存5分钟(fund_etf_spot_em 不限流,
    返回的是ETF真实元价, 可算真实盈亏)。"""
    import akshare as ak
    df = ak.fund_etf_spot_em()
    m = {}
    for _, r in df.iterrows():
        code = str(r["代码"]).strip().zfill(6)
        px = r.get("最新价")
        m[code] = {"name": str(r.get("名称", code)),
                   "last_px": float(px) if pd.notna(px) else None}
    return m


def _enrich_with_spot(holdings: dict) -> tuple:
    """用ETF实时行情补全/刷新每只持仓的 name + last_px, 并校验代码。
    返回 (holdings, invalid_codes)。spot 拉取失败则保留原值(不阻断)。"""
    try:
        spot = _etf_spot_map()
    except Exception as e:
        st.warning(f"实时行情获取失败({e}), 名称/最新价沿用上次保存值")
        return holdings, []
    invalid = []
    for code, h in holdings.items():
        info = spot.get(code)
        if info is None:
            invalid.append(code)
            continue
        h["name"] = info["name"]
        if info["last_px"] is not None:
            h["last_px"] = info["last_px"]
    return holdings, invalid


# 名称 → 对应宽基指数(新浪源稳定采集); 覆盖各基金公司的沪深300/500/1000/创业板/科创50/上证50
_NAME_INDEX_HINTS = [
    ("沪深300", "sh000300"), ("中证500", "sh000905"), ("中证1000", "sh000852"),
    ("创业板", "sz399006"), ("科创50", "sh000688"), ("上证50", "sh000016"),
]


def _guess_index_by_name(name: str) -> Optional[str]:
    """根据ETF名称猜对应宽基指数代码(用于池外ETF走新浪指数源采集)。"""
    for kw, idx in _NAME_INDEX_HINTS:
        if kw in str(name):
            return idx
    return None


def _ensure_collected(holdings: dict, cfg: dict) -> dict:
    """确保每只持仓ETF有历史数据(parquet); 没有则采集(有指数映射/名称匹配走新浪稳定源, 否则东财)。
    返回 {code: 状态文案}。"""
    from collector.etf_collector import update_one, load_etf_pool
    etf_dir = ROOT / cfg["paths"].get("etf_dir", "data/etf")
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    cc = cfg["collector"]
    adjust = cc.get("adjust", "qfq"); retries = cc.get("max_retries", 3); sleep = cc.get("request_sleep", 0.8)
    idx_map = {}
    try:
        pool = load_etf_pool(cfg)
        for grp in ("broad", "sector"):
            for it in pool.get(grp, []) or []:
                if it.get("index"):
                    idx_map[str(it["code"]).strip().zfill(6)] = it["index"]
    except Exception:
        pass
    spot = {}
    try:
        spot = _etf_spot_map()
    except Exception:
        pass
    results = {}
    for code in holdings:
        if (etf_dir / f"{code}.parquet").exists():
            results[code] = "已有数据"
            continue
        idx = idx_map.get(code) or _guess_index_by_name(spot.get(code, {}).get("name", ""))
        try:
            ok = update_one(code, etf_dir, start, end, adjust, retries, sleep, index_code=idx)
            results[code] = "采集成功" if ok else "采集失败(东财限流/代码无效)"
        except Exception:
            results[code] = "采集异常"
    return results


def _trend_advice(status: Optional[dict], bench_bull: bool) -> tuple:
    """根据单标趋势状态 + 大盘, 生成(动作, 人话原因)。status: single_etf_status 返回。"""
    if not status:
        return ("❓ 未采集", "点上方『保存』按钮自动采集历史数据后出趋势信号")
    hold, raw, band, px = status["hold"], status["raw"], status["band%"], status["close"]
    if hold == 1:
        action = "✅ 持有"
        if status.get("above_ma20", px > status["ma_short"]):
            reason = f"多头排列稳固(带宽{band:.1f}%), 现价站上MA20, 趋势健康, 继续持有"
        else:
            base = "多头基础仍在(MA60之上)" if status.get("above_ma60") else "已破MA60, 趋势转弱"
            reason = f"均线多头但现价跌破MA20, 短期回调; {base}, 持有但警惕"
    elif raw == 1:
        action = "⏳ 观望"
        reason = f"MA20刚上穿MA60但未连续确认(带宽{band:.1f}%偏小), 等站稳再入场, 别追"
    else:
        action = "❌ 卖出"
        reason = f"MA20下穿MA60死叉(带宽{band:.1f}%), 现价跌破均线, 趋势走弱, 建议卖出/不碰"
    if not bench_bull:
        reason += "; ⚠️大盘基准(沪深300)翻空, 系统性风险升温, 偏谨慎"
    return (action, reason)


def _rotation_reason(row) -> str:
    """轮动候选的人话原因。row: rotation_signal 行(mom_long/mom_short/score)。"""
    ml, ms = row.get("mom_long", 0), row.get("mom_short", 0)
    parts = [f"长期{'强势' if ml > 20 else ('上涨' if ml > 0 else '走弱')}({ml:+.0f}%)"]
    if ms > 5:
        parts.append(f"短期加速({ms:+.0f}%)")
    elif ms < -5:
        parts.append(f"短期回调({ms:+.0f}%)")
    parts.append(f"评分{row.get('score', 0):.2f}")
    return ", ".join(parts)


def _show_etf_backtest(bt: dict, daily_key: str, prefix: str, exposure_target: float = 0.15):
    """ETF 回测展示(expander 内调用): 回撤滑块 + 指标卡 + 净值曲线 + 交易明细。"""
    if bt["n_days"] < 2:
        st.warning("ETF 数据不足。请先运行 `python scripts/init_etf.py --years 6`。")
        return
    daily, bench_daily, dates = bt["daily_ret"], bt["bench_daily"], bt["dates"]
    full = metrics_from_returns(daily, 1.0)
    target = st.slider(f"🎯 目标最大回撤 ({prefix})", 0.05, 0.30, exposure_target, 0.01, key=f"{daily_key}_dd")
    exposure = min(1.0, target / abs(full["max_dd"])) if full["max_dd"] != 0 else 1.0
    scaled = metrics_from_returns(daily, exposure)
    bench = metrics_from_returns(bench_daily, 1.0)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("建议仓位", f"{exposure:.0%}", f"目标{target:.0%}")
    c2.metric("累计收益", f"{scaled['cum_ret']:+.1%}", f"满仓{full['cum_ret']:+.1%}")
    c3.metric("最大回撤", f"{scaled['max_dd']:.1%}", f"满仓{full['max_dd']:.1%}")
    c4.metric("夏普", f"{scaled['sharpe']:.2f}", f"满仓{full['sharpe']:.2f}")
    c5.metric("年化换手", f"{bt['turnover_annual']:.1f}x", f"{bt['n_trades']}笔")
    c6.metric("总佣金", f"{bt['total_commission']:.0f}元", f"基准{bt['bench_cum']:+.1%}")
    fig = go.Figure()
    dts = pd.to_datetime(dates)
    fig.add_trace(go.Scatter(x=dts, y=full["curve"], name=f"满仓(回撤{full['max_dd']:.0%})",
                             line=dict(color="#bbb", dash="dot")))
    fig.add_trace(go.Scatter(x=dts, y=scaled["curve"], name=f"缩仓{exposure:.0%}(回撤{scaled['max_dd']:.0%})",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=dts, y=bench["curve"], name=f"基准(回撤{bench['max_dd']:.0%})",
                             line=dict(color="#3498db")))
    fig.update_layout(height=380, hovermode="x unified", template="plotly_white", yaxis_title="净值")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander(f"📋 历史交易明细 ({bt['n_trades']}笔, 总佣金{bt['total_commission']:.0f}元) — 仅回测参考, 非实盘"):
        tl = bt["trade_log"]
        if not tl.empty:
            st.dataframe(tl[["date", "code", "side", "shares", "price", "amount", "commission"]]
                         .rename(columns={"date": "日期", "code": "代码", "side": "方向",
                                          "shares": "股数", "price": "成交价",
                                          "amount": "金额", "commission": "佣金"}),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("无交易记录。")


def _page_trend(cfg):
    from strategy.etf import trend_signal, single_etf_status
    ec = cfg.get("etf", {}); tc = ec.get("trend", {})
    ma_s = int(tc.get("ma_short", 20)); ma_l = int(tc.get("ma_long", 60))
    confirm = int(tc.get("confirm_days", 2)); band = float(tc.get("band_filter", 0.005))
    bench_code = str(ec.get("benchmark_etf", "510300")).strip().zfill(6)
    bench_st = single_etf_status(bench_code, cfg, ma_s, ma_l, confirm, band)
    bench_bull = bool(bench_st and bench_st["hold"] == 1)

    # 1. 我的持仓与操作建议 (决策核心)
    st.subheader("💼 我的持仓与操作建议")
    holdings = _load_etf_holdings()
    holdings, invalid = _enrich_with_spot(holdings)   # 自动补全名称+最新价, 校验代码
    if invalid:
        st.warning(f"⚠️ 以下代码在行情中未找到(可能无效/未上市), 请核对: {invalid}")
    edit_df = pd.DataFrame(
        [{"代码": c, "名称": h.get("name") or c, "数量": h.get("shares", 0),
          "成本": h.get("cost", 0.0), "最新现价": h.get("last_px")}
         for c, h in holdings.items()],
        columns=["代码", "名称", "数量", "成本", "最新现价"])
    edited = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True,
                            key="etf_holdings_editor")
    if st.button("💾 保存(补全名称/最新价 + 采集历史)", key="etf_save_h", type="primary"):
        new_h = {}
        for _, r in edited.iterrows():
            code = str(r.get("代码", "")).strip().zfill(6)
            if not code or code == "000000":
                continue
            new_h[code] = {"name": str(r.get("名称", code)),
                           "shares": int(r.get("数量", 0) or 0),
                           "cost": float(r.get("成本", 0) or 0),
                           "last_px": None}
        new_h, invalid2 = _enrich_with_spot(new_h)   # 补名称+最新价
        with st.spinner("采集持仓ETF历史数据(首次较慢, 宽基走新浪秒级)..."):
            collect_res = _ensure_collected(new_h, cfg)   # 采集历史
        _save_etf_holdings(new_h)
        failed = [c for c, s in collect_res.items() if "失败" in s or "异常" in s]
        msg = []
        if invalid2:
            msg.append(f"代码未找到请核对: {invalid2}")
        if failed:
            msg.append(f"采集失败(趋势信号暂不可用, 多为东财限流): {failed}")
        if msg:
            st.warning("；".join(msg))
        else:
            st.success("持仓已保存: 名称/最新价已补全, 历史数据已采集, 趋势信号就绪")
        st.rerun()

    if holdings:
        rows = []
        for code, h in holdings.items():
            status = single_etf_status(code, cfg, ma_s, ma_l, confirm, band)
            action, reason = _trend_advice(status, bench_bull)
            lpx, cost = h.get("last_px"), h.get("cost")
            pnl = (lpx / cost - 1) if (lpx and cost) else None
            rows.append({"名称": h.get("name", code), "代码": code, "数量": h.get("shares", 0),
                         "成本": cost, "最新现价": lpx if lpx else "—",
                         "浮盈": f"{pnl:+.1%}" if pnl is not None else "无现价",
                         "操作": action, "原因": reason})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("💡 名称/最新价自动从实时行情补全(缓存5分钟); 操作建议基于双均线趋势信号; 浮盈=最新现价/成本-1。")
    else:
        st.info("还没有持仓。点上方表格右下 ➕ 添加(只需填代码/数量/成本), 保存后自动补全名称和最新价。")

    st.divider()

    # 2. 今日趋势信号 (池子全貌, 分组, 用名称)
    st.subheader("📡 今日趋势信号")
    sig = trend_signal(cfg, ma_short=ma_s, ma_long=ma_l, confirm_days=confirm, band_filter=band)
    cand = sig["candidates"]
    groups = [
        ("✅ 多头持有", "均线多头排列, 可持有", cand[cand["hold"] == 1]),
        ("⏳ 待确认", "刚金叉/带宽不足, 别急着追", cand[(cand["hold"] == 0) & (cand["raw"] == 1)]),
        ("❌ 空头观望", "死叉, 回避", cand[(cand["hold"] == 0) & (cand["raw"] == 0)]),
    ]
    for emoji_label, sub, df in groups:
        if df.empty:
            continue
        st.markdown(f"**{emoji_label}** — {sub}")
        for _, r in df.iterrows():
            st.markdown(f"- {r['name']}（{r['code']}）现价 {r['close']}：MA20={r['ma_short']} / MA60={r['ma_long']}, 带宽 {r['band%']}%")

    st.divider()

    # 3. 市场温度
    st.subheader("🌡️ 市场温度")
    if bench_st:
        mood = "大盘多头, 可持仓" if bench_bull else "大盘死叉翻空, 系统性风险升温, 整体偏谨慎"
        st.markdown(f"**沪深300（{bench_code}）**: 现价 {bench_st['close']} vs MA60 {bench_st['ma_long']} → {'✅ 多头' if bench_bull else '❌ 空头'} — {mood}")
    gold_st = single_etf_status("518880", cfg, ma_s, ma_l, confirm, band)
    if gold_st:
        gb = gold_st["hold"] == 1
        gmood = "避险资产多头(避险情绪升温, 对股票ETF偏空)" if gb else "避险资产同步下跌(无港湾, 流动性偏紧)"
        st.markdown(f"**黄金ETF（518880）**: 现价 {gold_st['close']} vs MA60 {gold_st['ma_long']} → {'✅' if gb else '❌'} — {gmood}")

    st.divider()

    # 4. 回测与调参 (折叠)
    with st.expander("🔬 回测与调参（高级, 默认收起）", expanded=False):
        col = st.columns(5)
        ma_s2 = col[0].slider("短均线", 5, 60, ma_s, key="etf_t_ma_s")
        ma_l2 = col[1].slider("长均线", 20, 250, ma_l, key="etf_t_ma_l")
        confirm2 = col[2].slider("确认日数", 1, 5, confirm, key="etf_t_confirm")
        band2 = col[3].slider("带宽过滤%", 0.0, 3.0, band * 100, 0.1, key="etf_t_band") / 100
        freq2 = col[4].select_slider("调仓频率", [1, 5, 10, 20], int(tc.get("rebalance_freq", 5)), key="etf_t_freq")

        @st.cache_data(show_spinner="回测趋势择时...", ttl=600)
        def _bt_trend(ma_s, ma_l, confirm, band, freq):
            from strategy.etf import backtest_trend
            return backtest_trend(cfg, ma_short=ma_s, ma_long=ma_l, confirm_days=confirm,
                                  band_filter=band, freq=freq)
        _show_etf_backtest(_bt_trend(ma_s2, ma_l2, confirm2, band2, freq2), "etf_trend", "趋势择时")


def _page_rotation(cfg):
    from strategy.etf import rotation_signal
    ec = cfg.get("etf", {}); rc = ec.get("rotation", {})
    pool_label = st.radio("标的池", ["broad 宽基+黄金(稳健)", "sector 行业主题(叠大盘择时)"],
                          horizontal=True, key="etf_r_pool")
    pool_key = "broad" if pool_label.startswith("broad") else "sector"
    topn = rc.get("sector_topn", 3) if pool_key == "sector" else rc.get("broad_topn", 3)
    mom_l = int(rc.get("mom_long", 60)); mom_s = int(rc.get("mom_short", 20))
    sig = rotation_signal(cfg, pool=pool_key, topn=int(topn), mom_long=mom_l, mom_short=mom_s)

    # 1. 本期建议持有
    st.subheader("🎯 本期建议持有")
    if pool_key == "sector":
        if sig["market_bull"]:
            st.markdown(f"**大盘多头**（沪深300 {sig['bench_close']} > MA{sig['bench_ma']}）→ 允许持仓行业ETF")
        else:
            st.warning(f"⚠️ **大盘空头**（沪深300 {sig['bench_close']} < MA{sig['bench_ma']}）→ 本期**空仓避险**, 等大盘重回均线上方再入场")
    top = sig["top"]
    if top.empty:
        st.info("本期无符合『上升趋势 + 动量』的标的 → 空仓观望。")
    else:
        rows = [{"名称": r["name"], "代码": r["code"], "现价": r["close"],
                 "为什么选它": _rotation_reason(r)} for _, r in top.iterrows()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"等权持有 {len(top)} 只, 下期重新评分, 跌出 topN 或转弱则换出。")

    # 2. 为什么没选其他
    top_codes = set(top["code"].tolist()) if not top.empty else set()
    excluded = sig["candidates"][~sig["candidates"]["code"].isin(top_codes)]
    with st.expander(f"为什么没选其他（{len(excluded)} 只）"):
        for _, r in excluded.iterrows():
            st.markdown(f"- {r['name']}（{r['code']}）: {_rotation_reason(r)}")

    # 3. 持仓对照
    holdings = _load_etf_holdings()
    if holdings:
        st.subheader("💼 我的持仓对照轮动")
        hr = [{"名称": h.get("name", c), "代码": c,
               "轮动建议": "✅ 在 top, 继续持有" if c in top_codes else "⚠️ 不在 top, 考虑换出/减仓"}
              for c, h in holdings.items()]
        st.dataframe(pd.DataFrame(hr), use_container_width=True, hide_index=True)

    st.divider()

    # 4. 回测 (折叠)
    with st.expander("🔬 回测与调参（高级, 默认收起）", expanded=False):
        col = st.columns(4)
        topn2 = col[0].slider("持仓数", 1, 6, int(topn), key="etf_r_topn")
        mom_l2 = col[1].slider("长动量", 20, 120, mom_l, key="etf_r_moml")
        mom_s2 = col[2].slider("短动量", 5, 40, mom_s, key="etf_r_moms")
        freq2 = col[3].select_slider("调仓频率", [1, 5, 10, 20], int(rc.get("rebalance_freq", 5)), key="etf_r_freq")

        @st.cache_data(show_spinner="回测轮动...", ttl=600)
        def _bt_rot(pool_key, topn, mom_l, mom_s, freq):
            from strategy.etf import backtest_rotation
            return backtest_rotation(cfg, pool=pool_key, topn=topn, mom_long=mom_l, mom_short=mom_s, freq=freq)
        prefix = "宽基轮动" if pool_key == "broad" else "行业轮动"
        _show_etf_backtest(_bt_rot(pool_key, topn2, mom_l2, mom_s2, freq2), "etf_rot", prefix)


def page_etf():
    st.title("📊 ETF策略")
    st.caption("场内宽基 · 给你操作建议(持有/卖出/观望 + 原因), 不只是回测")
    cfg = get_cfg()
    tab_trend, tab_rot = st.tabs(["📈 趋势择时", "🔄 动量轮动"])
    with tab_trend:
        _page_trend(cfg)
    with tab_rot:
        _page_rotation(cfg)


# ============== 页面: 短线博弈 ==============
def page_shortterm():
    st.title("🎰 短线博弈")
    st.caption("情绪温度计 · 打板梯队 · 题材热度 · 龙虎榜游资 · 风控仓位（免费）")

    # ---- stale-while-revalidate: 秒开(用上次数据) + 后台静默刷新 + 仅变化才整体重渲染 ----
    from datetime import timedelta

    @st.cache_data(show_spinner=False, ttl=300)
    def _fetch():
        from strategy.short_term import market_sentiment, limit_up_ladder, strong_pool
        from strategy.hot_money import concept_heatboard, dragon_tiger, hot_seats, institution_flow
        from collector.aux_collector import latest_closed_trade_date
        d = latest_closed_trade_date().replace("-", "")
        return {
            "date": d,
            "sent": market_sentiment(d),
            "ladder": limit_up_ladder(d),
            "strong": strong_pool(d),
            "concepts": concept_heatboard(),
            "dt": dragon_tiger(d),
            "seats": hot_seats(d),
            "inst": institution_flow(d),
        }

    if "st_bundle" not in st.session_state:
        st.session_state["st_bundle"] = None

    def _sig(b):
        """数据指纹: 变了才值得重渲染。"""
        if not b:
            return None
        s = b.get("sent", {}) or {}
        try:
            topc = b["concepts"].iloc[0]["板块"] if (b.get("concepts") is not None and not b["concepts"].empty) else ""
        except Exception:
            topc = ""
        return (s.get("date"), s.get("n_zt"), s.get("max_streak"), int(s.get("temp", 0)), topc)

    ctrl = st.columns([2, 1, 3])
    ctrl[0].caption("🔄 用缓存秒开；点右边立即刷新")
    if ctrl[1].button("🔄 立即刷新", key="st_refresh"):
        _fetch.clear()
        try:
            st.session_state["st_bundle"] = _fetch()
        except Exception as e:
            ctrl[2].error(f"刷新失败: {e}")
    # 注: 不再用 @st.fragment(run_every) 自动刷新——其内部的 st.rerun() 异步触发会与
    # 侧栏导航 widget 撞 key, 导致整个 app 崩溃(StreamlitDuplicateElementKey)。改用手动刷新。
        ctrl[2].caption("⏱️ 后台静默刷新中，数据有变化才更新页面")

    # 优先用 session_state 里的上次数据立即渲染(秒开, 无转圈)
    b = st.session_state["st_bundle"]
    if b is None:
        with st.spinner("首次加载盘面数据…"):  # 仅首次访问会有这一次短暂等待
            try:
                b = _fetch()
                st.session_state["st_bundle"] = b
            except Exception as e:
                st.error(f"盘面数据获取失败(可能被限流): {e}")
                st.info("稍后点「🔄 立即刷新」重试。短线打板对新手是高风险负和游戏, 仅供参考。")
                return

    sent = b["sent"]
    # ---- 情绪温度计大圆盘 ----
    c_gauge, c_info = st.columns([1, 1])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sent["temp"],
        title={"text": "情绪温度"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#e74c3c"},
               "steps": [
                   {"range": [0, 30], "color": "#3ecf8e"},
                   {"range": [30, 50], "color": "#f5c518"},
                   {"range": [50, 75], "color": "#f39c12"},
                   {"range": [75, 100], "color": "#e74c3c"}],
               "threshold": {"line": {"width": 4}, "thickness": 1, "value": sent["temp"]}},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10))
    c_gauge.plotly_chart(fig, use_container_width=True)
    c_info.metric("能否下嘴", sent["can_play"])
    c_info.write(f"涨停 **{sent['n_zt']}** · 跌停 {sent['n_dt']} · 炸板 {sent['n_zbgc']}(率 {sent['zha_rate']:.0%}) · 高度 **{sent['max_streak']}连**")
    c_info.info(sent["advice"])

    # ---- 风控仓位计算器 ----
    st.divider()
    st.subheader("🧮 风控仓位计算器")
    cap = st.number_input("总资金(元)", value=1_000_000, step=100_000, key="st_cap")
    pool_cap = cap * 0.30     # 短线总仓上限
    per_cap = cap * 0.10      # 单只上限
    if sent["temp"] < 30:
        st.warning(f"情绪温度 {sent['temp']}<30, 退潮期打板大概率吃面 → **本期不做, 仓位 0**。")
    else:
        st.write(f"短线总仓上限 ¥{pool_cap:,.0f}（30%）· 单只上限 ¥{per_cap:,.0f}（10%）· 止损 5%")
        strong = b["strong"]
        if "现价" in strong.columns and not strong.empty:
            picks = strong[strong["现价"].notna() & (strong["现价"] > 0)].head(5)
            rows = []
            used = 0.0
            for _, r in picks.iterrows():
                px = float(r["现价"])
                tgt_val = min(per_cap, (pool_cap - used))
                if tgt_val < px * 100:
                    continue
                shares = int(tgt_val / px / 100) * 100
                if shares <= 0:
                    continue
                amt = shares * px
                used += amt
                rows.append({"代码": r["代码"], "名称": r["名称"], "现价": px,
                             "买入股数": shares, "金额": round(amt, 0),
                             "止损价": round(px * 0.95, 2), "涨跌幅": r.get("涨跌幅")})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(f"合计 ¥{sum(r['金额'] for r in rows):,.0f} / 上限 ¥{pool_cap:,.0f}。"
                           "⚠️ 单只≤10%、破板/亏5%即走、不补仓。情绪退潮立即全部清仓。")
            else:
                st.info("无合适标的(价格缺失或资金不足)。")

    # ---- 打板梯队 + 强势股 ----
    st.divider()
    col = st.columns(2)
    with col[0].expander("🎯 打板连板梯队", expanded=True):
        _safe_table(b["ladder"].head(20) if b.get("ladder") is not None else None, "暂无连板数据")
    with col[1].expander("💪 强势股(成交额前列)", expanded=True):
        _safe_table(b.get("strong"), "暂无强势股数据")

    # ---- 题材 + 龙虎榜 (默认展开, 空数据给提示, 渲染异常不抛堆栈) ----
    st.divider()
    col = st.columns(2)
    with col[0].expander("🔥 题材热度（按热度分排序）", expanded=True):
        c = b.get("concepts")
        if c is None or (hasattr(c, "empty") and c.empty):
            st.warning("题材数据暂不可用（东财实时接口被限流）。点上方「🔄 立即刷新」重试，或过几分钟再来。")
        else:
            _safe_table(c, "暂无题材数据")
            if "热度分" not in c.columns:
                st.caption("ℹ️ 实时涨跌暂不可用(东财限流)，当前仅显示题材名称列表。")
    with col[1].expander("🐉 龙虎榜个股(净买前列)", expanded=True):
        _safe_table(b.get("dt"), "今日暂无龙虎榜数据（可能未到披露时间或被限流）")
    col = st.columns(2)
    with col[0].expander("🏦 游资营业部(净额前列)", expanded=True):
        _safe_table(b.get("seats"), "暂无游资席位数据")
    with col[1].expander("🏛️ 机构净买个股", expanded=True):
        _safe_table(b.get("inst"), "暂无机构净买数据")

    st.caption("⚠️ 短线打板对新手是高风险负和游戏(对手是游资+量化+手续费)。数据为盘后/盘中快照, 仅供参考, 不构成投资建议。")


# ============== 页面: 盘中预警 (Pro) ==============
def page_intraday():
    from paywall import locked_page, license_info
    if not locked_page("⚡ 盘中盯盘预警",
                       "实时监控涨停/封板/炸板, 触发即推送(webhook/邮件)。内置冷静期与风控门槛, 防止上头。"):
        return
    st.title("⚡ 盘中盯盘预警")
    info = license_info()
    st.success(f"✅ Pro 已激活 (到期 {info['exp']})")

    from live.intraday_monitor import is_market_hours, _watch_codes, ALERT_LOG
    cfg = get_cfg()
    codes = _watch_codes(cfg)
    c1, c2 = st.columns(2)
    c1.metric("监控池", f"{len(codes)} 只")
    c2.metric("交易时段", "是 ✅" if is_market_hours() else "否（盘外）")

    st.subheader("监控池（watchlist 龙头）")
    st.write("、".join(codes))

    st.subheader("启动守护进程")
    st.code("python -m live.intraday_monitor            # 仅交易时段轮询\n"
            "python -m live.intraday_monitor --interval 30   # 自定义间隔\n"
            "python -m live.intraday_monitor --test     # 立即跑一轮测试推送", language="bash")
    st.caption("建议用 launchd/终端后台常驻。预警写 data/cache/intraday_alerts.log 并推送 webhook/邮件。")

    if st.button("📨 立即测试推送一轮", type="primary"):
        with st.spinner("抓取实时行情并检测..."):
            try:
                from live.intraday_monitor import run_once
                state = {}
                Path(ALERT_LOG).parent.mkdir(parents=True, exist_ok=True)
                with open(ALERT_LOG, "a", encoding="utf-8") as fh:
                    n = run_once(cfg, codes, state, fh, force=True)
                if n:
                    st.success(f"触发 {n} 条预警, 已推送(若 notify 已配置)")
                else:
                    st.info("本轮无预警触发（监控池中无接近涨停/封板/炸板的票，或 notify 未配置）。")
            except Exception as e:
                st.error(f"测试失败: {e}")

    st.subheader("预警日志")
    log_p = Path(ALERT_LOG)
    if log_p.exists():
        lines = log_p.read_text(encoding="utf-8").strip().splitlines()
        st.text("\n".join(lines[-30:]) or "(空)")
    else:
        st.caption("暂无预警记录。")


# ============== 页面: 自研模型 (Pro) ==============
def page_model():
    from paywall import locked_page, license_info
    if not locked_page("🧠 自研模型训练",
                       "自定义标签(收益/方向)+因子组+模型(LightGBM/Ridge), 一键训练评估保存。"
                       "诚实显示命中率, 命中率<55%标注不可靠。"):
        return
    st.title("🧠 自研模型训练")
    info = license_info()
    st.success(f"✅ Pro 已激活 (到期 {info['exp']})")
    st.caption("横截面面板模型: 各股滞后指标 → 未来N日收益/方向。单股短周期信噪比低, 结果仅供参考。")

    cfg = get_cfg()
    col = st.columns(4)
    universe = col[0].selectbox("股票池", ["watchlist", "sector_leaders"], 0)
    label = col[1].selectbox("标签", ["ret(收益回归)", "dir(涨跌方向)"], 0)
    horizon = col[2].slider("预测天数", 3, 20, 5)
    model_type = col[3].selectbox("模型", ["lgbm", "ridge"], 0)
    train_end = st.text_input("训练截止日(此后为测试)", "2025-12-31")
    label_key = "ret" if label.startswith("ret") else "dir"

    if st.button("🚀 开始训练", type="primary"):
        with st.spinner("建面板 + 训练 + 评估..."):
            try:
                from model.trainer import train_pipeline, metrics_to_markdown
                m = train_pipeline(cfg, universe=universe, horizon=horizon,
                                   label=label_key, model_type=model_type, train_end=train_end)
                st.session_state["ml_metrics"] = m
                st.success("训练完成 ✅")
            except Exception as e:
                st.error(f"训练失败: {e}")

    m = st.session_state.get("ml_metrics")
    if m:
        st.markdown(metrics_to_markdown(m))

    st.divider()
    st.subheader(f"已保存模型（最多保留 5 个，按评测结果记录）")
    try:
        from model.trainer import list_models, delete_model
        models = list_models()
        if models:
            rows = []
            for x in models:
                m = x["metrics"]
                rows.append({
                    "文件": x["file"],
                    "股票池": m.get("universe"), "标签": m.get("label"), "模型": m.get("model"),
                    "命中率": f"{m['direction_hit']:.1%}" if m.get("direction_hit") is not None else "-",
                    "IC": m.get("IC"),
                    "topK超额/日": m.get("topk_excess_per_day"),
                    "测试区间": f"{m.get('test_start','')}~{m.get('test_end','')}",
                    "可靠": "✅" if m.get("reliable") else "⚠️",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"共 {len(models)}/5 个。命中率<55% 标⚠️不可靠。新训练超出5个会自动删最旧。")
            # 删除
            with st.expander("🗑 删除某个模型"):
                fn = st.selectbox("选择要删除的", [x["file"] for x in models], key="del_model")
                if st.button("删除", key="do_del_model"):
                    if delete_model(fn):
                        st.success(f"已删除 {fn}"); st.rerun()
                    else:
                        st.error("删除失败")
        else:
            st.caption("暂无已保存模型。训练一个后会出现在这里。")
    except Exception as e:
        st.caption(f"读取模型列表失败: {e}")


# ============== 页面: 推荐持仓 (模型选股输出) ==============
def page_recommend():
    st.subheader("🎯 模型推荐持仓")
    st.caption("多因子模型在自选股里打分，选 top-N 作为目标持仓。生成后到「💼 持仓与推荐」看每只持仓的加仓/减仓建议。")
    cfg = get_cfg()
    nm = _names()
    target_csv = ROOT / cfg["paths"]["cache_dir"] / "target_portfolio.csv"
    cap = st.number_input("参考资金(元)", value=1_000_000, step=100_000, key="rec_cap")
    if target_csv.exists():
        target = pd.read_csv(target_csv)
        try:
            from live.order_sheet import get_latest_prices
            tprices = get_latest_prices(target["qlib_code"].tolist(), cfg)
        except Exception:
            tprices = {}
        rows = []
        for _, r in target.iterrows():
            px = tprices.get(r["qlib_code"])
            val = r["weight"] * cap
            sh = int(val / px / 100) * 100 if px and px > 0 else None
            rows.append({"代码": r["code"], "名称": r.get("name", nm.get(str(r["code"]), "-")),
                         "打分": round(r["score"], 3) if pd.notna(r.get("score")) else None,
                         "权重": f"{r['weight']:.0%}", "现价": px,
                         "建议股数": sh, "建议金额": round(sh * px) if sh and px else None})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"推荐 {len(target)} 只（基于最新一次模型打分）。")
    else:
        st.info("暂未生成推荐，点下方按钮生成。")
    if st.button("🔄 重新生成推荐", type="primary", key="gen_rec"):
        if not qlib_ready(cfg):
            st.warning("qlib 数据未就绪，先在「🗃️ 数据采集」tab 采集并 dump。"); return
        with st.spinner("跑模型打分生成推荐..."):
            try:
                from live.signal_generator import generate_target_portfolio, save_portfolio
                save_portfolio(generate_target_portfolio(cfg), cfg)
                st.success("推荐已生成 ✅"); st.rerun()
            except Exception as e:
                st.error(f"生成失败: {e}")


# ============== 主导航 ==============
def page_model_hub():
    """自研模型一站式: 子导航切换模块(只渲染当前模块, 避免 st.tabs 一次性渲染全部
    导致与侧栏导航 widget 撞 key 的 Streamlit 1.50 bug)。"""
    st.title("🧪 自研模型")
    st.caption("股票池与数据 → 配置 → 回测/推荐 → 轮动 → 训练模型，一站式策略研发。")
    modules = ["⭐ 自选股", "🗃️ 数据采集", "⚙️ 参数配置", "📊 回测",
               "🎯 推荐持仓", "🔁 轮动策略", "🎓 训练模型(Pro)"]
    sub = st.radio("模块", modules, horizontal=True, key="model_sub")
    if sub == "⭐ 自选股":
        page_watchlist()
    elif sub == "🗃️ 数据采集":
        page_data()
    elif sub == "⚙️ 参数配置":
        page_config()
    elif sub == "📊 回测":
        page_backtest()
    elif sub == "🎯 推荐持仓":
        page_recommend()
    elif sub == "🔁 轮动策略":
        page_rotation()
    elif sub == "🎓 训练模型(Pro)":
        page_model()


PAGES = {
    "📈 概览": page_overview,
    "💼 持仓与推荐": page_holdings,
    "🧪 自研模型": page_model_hub,
    "📊 ETF策略": page_etf,
    "🎰 短线博弈": page_shortterm,
    "⚡ 盘中预警 🔒": page_intraday,
}

st.sidebar.title("A股量化系统")
# 用 URL query_params 记住当前页, 刷新/重开后回到原页而非概览
_saved = st.query_params.get("page")
_default = list(PAGES.keys()).index(_saved) if _saved in PAGES else 0
choice = st.sidebar.radio("导航", list(PAGES.keys()), index=_default,
                          label_visibility="collapsed", key="nav_choice")
if st.query_params.get("page") != choice:
    st.query_params["page"] = choice
st.sidebar.divider()
st.sidebar.caption("akshare · qlib · 多因子选股")
PAGES[choice]()

# 每日任务快捷按钮
st.sidebar.divider()
if st.sidebar.button("▶️ 运行每日任务"):
    st.sidebar.info("请在终端运行: `python scripts/run_daily.py`")

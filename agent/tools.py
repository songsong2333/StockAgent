"""Agent 工具注册表 —— 把 strategy/* + 知识库包成 OpenAI tool-calling 格式。

每个 tool = OpenAI schema {type:function, function:{name,description,parameters}} + 一个
python 执行函数 fn(cfg, **args) → JSON 安全的 dict/list。dispatch(cfg, name, args) 分发。

输出刻意紧凑(DataFrame→records, float 取 3 位), 并携带统计诚实的 warning 字段
(N/CI/FDR/相关非因果), 让 LLM 转述时无法丢掉不确定性。
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from common import setup_logger, load_config

log = setup_logger("agent.tools")


# ============== 序列化辅助 ==============
def _json_safe(obj: Any) -> Any:
    """递归转 numpy/pandas/DataFrame → JSON 安全的 python 对象。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, float):
        return None if (obj != obj) else obj       # NaN 检查
    return obj


def _df_records(df: pd.DataFrame, cols=None, n=None, round_ndigits=3):
    """DataFrame → 精简 records(取指定列 + 截 n 行 + 取整)。"""
    if df is None or df.empty:
        return []
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if n:
        df = df.head(n)
    rec = df.to_dict(orient="records")
    out = []
    for r in rec:
        out.append({k: (round(v, round_ndigits) if isinstance(v, (int, float))
                        and not isinstance(v, bool) else v) for k, v in r.items()})
    return _json_safe(out)


def _load_panel(cfg, freq="M"):
    """统一加载入口(各类 tool 共用)。"""
    from collector.store import universe_panel
    return universe_panel(cfg, freq=freq)


# ============== 工具实现 ==============
def t_list_universe(cfg, **_):
    panel, names, meta = _load_panel(cfg)
    rows = []
    for sid in panel.columns:
        m = meta.get(sid, {})
        rows.append({"id": sid, "name": names.get(sid, sid), "category": m.get("category"),
                     "kind": m.get("kind"), "inception": m.get("inception"), "n_obs": m.get("n_obs")})
    return {"n": len(rows), "series": rows,
            "hint": "code/id 即返回的 id 字段(如 159611 电力ETF / 09988 港股 / enso_oni ENSO)。kind=price 用收益分析, kind=rate 作驱动/锚点。"}


def t_get_seasonality(cfg, code="", **_):
    from strategy.seasonality import seasonality_signal, stats_to_markdown
    if not code:
        return {"error": "请提供 code"}
    sig = seasonality_signal(cfg, str(code).strip())
    if "error" in sig:
        return sig
    stats = sig["stats"]
    return {"code": sig["code"], "name": sig["name"],
            "monthly_stats": _df_records(stats.reset_index(), n=12),
            "warning": stats.attrs.get("warning", "")}


def t_detect_event_windows(cfg, code="", **_):
    from strategy.seasonality import seasonality_signal, window_to_markdown
    if not code:
        return {"error": "请提供 code"}
    sig = seasonality_signal(cfg, str(code).strip())
    if "error" in sig:
        return sig
    d = sig["distribution"]
    if d.get("n_years", 0) == 0:
        return {"code": sig["code"], "name": sig["name"], "warning": d.get("warning", "无窗口")}
    return {
        "code": sig["code"], "name": sig["name"],
        "start": d["start"], "end": d["end"], "duration": d["duration"],
        "magnitude": d["magnitude"], "hit_rate": d["hit_rate"], "n_years": d["n_years"],
        "consistent_years": d["n_consistent"],
        "windows": _df_records(d.get("windows"), n=20),
        "warning": d.get("warning", ""),
    }


def t_anchor_study(cfg, target="", anchor="", threshold=0.5, direction="above", pre=4, post=8, **_):
    from strategy.seasonality import anchor_event_study
    if not target or not anchor:
        return {"error": "需提供 target 和 anchor(锚点用 rate 类如 enso_oni/electricity_yoy)"}
    panel, names, _ = _load_panel(cfg)
    if target not in panel.columns:
        return {"error": f"{target} 不在面板"}
    if anchor not in panel.columns:
        return {"error": f"{anchor} 不在面板(可用的 rate 锚点: enso_oni/electricity_yoy/m2_yoy/lpr1y)"}
    res = anchor_event_study(panel[target].dropna(), panel[anchor].dropna(),
                             threshold=float(threshold),
                             direction="above" if direction.startswith("above") else "below",
                             pre=int(pre), post=int(post))
    return {"n_events": res.get("n_events", 0),
            "event_dates": res.get("event_dates", []),
            "curve": _df_records(res.get("curve")) if res.get("curve") is not None else [],
            "warning": res.get("warning", "")}


def t_discover_relations(cfg, target="", **_):
    from strategy.discovery import discover
    if not target:
        return {"error": "请提供 target"}
    panel, names, meta = _load_panel(cfg)
    pids = [c for c in panel.columns if meta.get(c, {}).get("kind") == "price"]
    ret = panel[pids].pct_change(fill_method=None).dropna(how="all")
    if target not in ret.columns:
        return {"error": f"{target} 不在 price 序列中"}
    res = discover(target, ret, meta, names)
    if "error" in res:
        return res
    return {"target": target, "target_name": res.get("target_name", target),
            "top_relations": _df_records(res["table"], cols=[
                "id", "name", "category", "corr", "q", "beta", "r2",
                "best_lag", "leads_target", "stability", "significant"], n=12),
            "warning": res.get("warning", "")}


def t_cluster_universe(cfg, **_):
    from strategy.discovery import cluster_series
    panel, names, meta = _load_panel(cfg)
    pids = [c for c in panel.columns if meta.get(c, {}).get("kind") == "price"]
    ret = panel[pids].pct_change(fill_method=None).dropna(how="all")
    cl = cluster_series(ret, n_clusters=0)
    order = [{"id": c, "name": names.get(c, c)} for c in cl.get("order", [])]
    return {"leaf_order": order, "hint": "聚类排序中相邻=收益相关高(同涨同跌)。用于找'谁和谁一类'。"}


def t_get_regime(cfg, code="", **_):
    from strategy.regime import medium_term_signal
    if not code:
        return {"error": "请提供 code"}
    sig = medium_term_signal(cfg, str(code).strip())
    return sig if "error" in sig else sig


def t_conditional_windows(cfg, code="", dimension="ENSO", **_):
    from strategy.regime import conditional_window_stats, enso_regime_by_year, market_regime_by_year, liquidity_regime_by_year
    if not code:
        return {"error": "请提供 code"}
    panel, names, meta = _load_panel(cfg)
    if code not in panel.columns:
        return {"error": f"{code} 不在面板"}
    dim = (dimension or "ENSO").upper()
    label_map = {}
    if dim.startswith("ENSO") and "enso_oni" in panel.columns:
        ry = enso_regime_by_year(panel["enso_oni"], threshold=cfg["cycle"]["regime"]["enso_threshold"])
        label_map = {1: "厄尔尼诺", -1: "拉尼娜", 0: "中性"}
    elif dim.startswith("市场") or dim.startswith("MARKET"):
        bench = "510300" if "510300" in panel.columns else next((c for c in panel.columns if meta.get(c, {}).get("kind") == "price"), None)
        ry = market_regime_by_year(panel[bench], ma=cfg["cycle"]["regime"]["market_ma"]) if bench else None
        label_map = {1: "牛市", -1: "熊市", 0: "震荡"}
    elif dim.startswith("流动") or dim.startswith("LIQUID") and "m2_yoy" in panel.columns:
        ry = liquidity_regime_by_year(panel["m2_yoy"], lookback=cfg["cycle"]["regime"]["liquidity_lookback"], direction="rise")
        label_map = {1: "宽松", -1: "收紧", 0: "中性"}
    else:
        return {"error": f"dimension={dimension} 无可用数据(需 enso_oni/沪深300/m2_yoy)"}
    if ry is None or ry.empty:
        return {"error": "该 regime 维度无数据"}
    cond = conditional_window_stats(panel[code], ry, method=cfg.get("cycle", {}).get("window", {}).get("method", "peak"))
    if "error" in cond:
        return cond
    out = {}
    for lab, n in cond.get("_regimes", {}).items():
        d = cond.get(lab, {})
        if d.get("n_years", 0) == 0:
            continue
        lname = label_map.get(_safe_int(lab), lab)
        out[lname] = {"n_years": n, "start_median": round(d["start"]["median"], 1),
                      "end_median": round(d["end"]["median"], 1),
                      "magnitude_median": round(d["magnitude"]["median"], 3),
                      "magnitude_ci": [round(d["magnitude"]["ci_lo"], 3), round(d["magnitude"]["ci_hi"], 3)]}
    return {"code": code, "dimension": dim, "by_regime": out, "warning": cond.get("warning", "")}


def _safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def t_query_knowledge(cfg, query="", **_):
    from agent.knowledge import query_laws, law_to_summary
    laws = query_laws(cfg, query, limit=8)
    return {"n": len(laws), "laws": [{"id": l.get("id"), "status": l.get("status"),
            "statement": l.get("statement"), "evidence": l.get("evidence"),
            "related_series": l.get("related_series")} for l in laws],
            "hint": "提议新规律前先查重; 引用已有规律时带 id。"}


def t_propose_law(cfg, statement="", evidence=None, related_series=None, **_):
    from agent.knowledge import propose_law
    if not statement:
        return {"error": "请提供 statement"}
    law = propose_law(cfg, statement, evidence or {}, related_series or [], source="agent")
    return {"ok": True, "id": law["id"], "status": law["status"],
            "note": "已入库为 tentative(待定)。升 confirmed 需过统计门槛或人工确认。"}


def t_update_law_status(cfg, law_id="", status="", reason="", source="agent", **_):
    from agent.knowledge import update_law_status
    if not law_id or not status:
        return {"error": "需提供 law_id 和 status(tentative/confirmed/refuted)"}
    return _json_safe(update_law_status(cfg, law_id, status, reason, source))


# ============== 注册表 ==============
_TOOLS = [
    {"type": "function", "function": {
        "name": "list_universe", "description": "列出分析宇宙里所有可用的序列(ETF/行业指数/港股/商品/宏观/气候), 含代码、类别、起始日、样本量。先调它知道能用哪些 code。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_seasonality", "description": "某标的的月度季节性统计(每月均值/中位/胜率/95%bootstrap CI/样本量)。用于回答'某月某板块历史上表现如何'。",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "序列 id, 如 159611(电力ETF)"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "detect_event_windows", "description": "逐年检测某标的真实主升浪窗口(起止月、持续、幅度、漂移std、交叉验证一致性)。用于回答'某板块的行情一般什么时候启动'。",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "anchor_study", "description": "锚点 event-study: 把各年对齐到某外因(ENSO/用电量/利率)跨阈事件, 看标的是否系统性在锚点后跟动。target=价格类, anchor=rate类(如 enso_oni)。",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "价格类标的 id"},
            "anchor": {"type": "string", "description": "rate 类锚点 id, 如 enso_oni"},
            "threshold": {"type": "number", "description": "事件阈值, 默认 0.5"},
            "direction": {"type": "string", "enum": ["above", "below"], "description": "默认 above"},
            "pre": {"type": "integer"}, "post": {"type": "integer"}},
            "required": ["target", "anchor"]}}},
    {"type": "function", "function": {
        "name": "discover_relations", "description": "对给定 target, 扫描全宇宙找谁最跟随它(相关)、谁领先它(lead-lag), 带 FDR q 值/稳定性。用于'某标的到底跟谁走'。",
        "parameters": {"type": "object", "properties": {"target": {"type": "string"}},
                       "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "cluster_universe", "description": "对全宇宙价格序列做层次聚类, 返回按相关排序的叶子顺序(相邻=同涨同跌)。用于找'谁和谁是一类'。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_regime", "description": "某标的的月级趋势 regime(牛/熊/震荡)+ MA3/6/12 + 多头排列 + 方向建议。回答'现在该不该做这个方向'。",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "conditional_windows", "description": "按 regime 分组对比某标的的窗口分布(如厄尔尼诺年 vs 中性年)。dimension: ENSO/市场/流动性。用于验证'某 regime 下行情是否不同'。",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "dimension": {"type": "string", "description": "ENSO / 市场 / 流动性", "default": "ENSO"}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "query_knowledge", "description": "检索规律知识库(已累积的经济学规律, 含状态/证据)。提议新规律前必先查重; 回答时可引用已有规律。",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "propose_law", "description": "把一个发现提议为规律(默认入库为 tentative 待定)。仅当有充分统计证据(N/CI/FDR)时调用, 不要凭单次小样本就提。",
        "parameters": {"type": "object", "properties": {
            "statement": {"type": "string", "description": "规律的自然语言陈述"},
            "evidence": {"type": "object", "description": "证据 dict, 如 {n_years:14, hit_rate:0.93, source_tool:'detect_windows'}"},
            "related_series": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement"]}}},
    {"type": "function", "function": {
        "name": "update_law_status", "description": "迁移规律状态(tentative/confirmed/refuted)。升 confirmed 需过自动门槛, 否则会被拒(提示需更多样本或人工确认)。",
        "parameters": {"type": "object", "properties": {
            "law_id": {"type": "string"}, "status": {"type": "string", "enum": ["tentative", "confirmed", "refuted"]},
            "reason": {"type": "string"}}, "required": ["law_id", "status"]}}},
]

_FN = {
    "list_universe": t_list_universe, "get_seasonality": t_get_seasonality,
    "detect_event_windows": t_detect_event_windows, "anchor_study": t_anchor_study,
    "discover_relations": t_discover_relations, "cluster_universe": t_cluster_universe,
    "get_regime": t_get_regime, "conditional_windows": t_conditional_windows,
    "query_knowledge": t_query_knowledge, "propose_law": t_propose_law,
    "update_law_status": t_update_law_status,
}


def tools_schema() -> list:
    """OpenAI 格式 tools(给 LLM 的 schema)。"""
    return _TOOLS


def dispatch(cfg: dict, name: str, args: dict) -> str:
    """执行工具, 返回 JSON 字符串(回填 role=tool 的 content)。异常也包成 JSON 不抛。"""
    fn = _FN.get(name)
    if fn is None:
        return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
    try:
        result = fn(cfg, **(args or {}))
        return json.dumps(_json_safe(result), ensure_ascii=False, default=str)
    except Exception as e:
        log.warning(f"工具 {name} 执行异常: {e}")
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


def tool_names() -> list:
    return [t["function"]["name"] for t in _TOOLS]


if __name__ == "__main__":
    cfg = load_config()
    print("工具数:", len(_TOOLS), tool_names())
    # 自检: 派发几个不依赖 LLM 的工具
    print("\n[list_universe]", dispatch(cfg, "list_universe", {})[:200])
    print("\n[query_knowledge ENSO]", dispatch(cfg, "query_knowledge", {"query": "ENSO 电力"})[:300])

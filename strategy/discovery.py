"""周期探索 —— 发现引擎(全市场扫描, 非固定池匹配)。

核心: 不预设关系, 让系统在统一宇宙里【发现】谁和谁相关、谁领先谁、谁聚类。
  - pairwise_corr: 全宇宙两两相关(收益率, 非价格) + p 值 + BH-FDR 校正
  - lead_lag_matrix: ±k 周滞后互相关 → 谁领先谁(判定"跟随")
  - lead_lag_graph: networkx 有向图 + 顶级领先指标(隐藏先行指标)
  - cluster_series: 层次聚类 → 发现"港股X 落在创业板成长簇"
  - discover(target): 给定标的, 排序全宇宙的跟踪/领先关系(含 beta/R²/稳定性/q值)

统计诚实: 相关用收益率; 全宇宙两两扫描≈数千对 → 必走 BH-FDR; 发现榜按稳定性
(滚动相关同号占比)加权, 不只峰值; lead-lag 注明日频噪声大, 主推周/月频。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from common import setup_logger, load_config

log = setup_logger("strategy.discovery")


# ============== 两两相关 + 显著性 + FDR ==============
def corr_pvalue(r: float, n: int) -> float:
    """Pearson 相关系数 → 双侧 p 值(n 为样本量)。"""
    if n <= 2 or abs(r) >= 1:
        return 0.0 if abs(r) >= 1 else np.nan
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(2 * sstats.t.sf(abs(t), df=n - 2))


def pairwise_corr(ret_panel: pd.DataFrame, method: str = "pearson") -> dict:
    """全宇宙两两相关矩阵 + p 值矩阵 + BH-FDR q 值矩阵。

    返回 {corr, p, q, n}: 均为方阵 DataFrame(id×id)。
    """
    corr = ret_panel.corr(method=method)
    ids = list(corr.columns)
    n = int(ret_panel.dropna(how="all").shape[0])
    p = pd.DataFrame(np.nan, index=ids, columns=ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            # 有效样本 = 两列同非空的行数
            nn = ret_panel[[a, b]].dropna().shape[0]
            pv = corr_pvalue(r, nn)
            p.loc[a, b] = pv
            p.loc[b, a] = pv
    # BH-FDR(仅上三角的 N*(N-1)/2 个独立检验)
    try:
        from statsmodels.stats.multitest import multipletests
        upper = p.where(np.triu(np.ones(p.shape, dtype=bool), k=1))
        flat = upper.stack().dropna().values
        if len(flat):
            _, qflat, _, _ = multipletests(flat, method="fdr_bh")
            q = pd.DataFrame(np.nan, index=ids, columns=ids)
            idx = upper.stack().dropna().index
            for (a, b), qv in zip(idx, qflat):
                q.loc[a, b] = qv
                q.loc[b, a] = qv
        else:
            q = p.copy()
    except Exception as e:
        log.warning(f"FDR 校正失败(退化为裸 p): {e}")
        q = p.copy()
    return {"corr": corr, "p": p, "q": q, "n": n}


# ============== Lead-lag(谁领先谁) ==============
def _pair_lead_lag(a: pd.Series, b: pd.Series, max_lag: int) -> tuple:
    """a, b 收益率序列 → (best_lag, best_corr)。lag>0 表示 a 领先 b(a 的过去预测 b 的现在)。

    用 a.shift(L).corr(b): corr(a_{t-L}, b_t); L>0 时 a 先于 b 变动 → a 领先。
    """
    best_lag, best_c = 0, None
    for L in range(-max_lag, max_lag + 1):
        c = a.shift(L).corr(b)
        if pd.notna(c) and (best_c is None or abs(c) > abs(best_c)):
            best_c, best_lag = c, L
    return best_lag, float(best_c) if best_c is not None else 0.0


def lead_lag_matrix(ret_panel: pd.DataFrame, max_lag: int = 8) -> dict:
    """全两两 lead-lag。返回 {lag: 矩阵(a领先b的期数), corr: 峰值相关矩阵}。

    lag[a,b]>0 → a 领先 b(用 a 的当前预测 b 的未来)。对角线 0。
    """
    ids = list(ret_panel.columns)
    L = pd.DataFrame(0, index=ids, columns=ids, dtype=int)
    C = pd.DataFrame(0.0, index=ids, columns=ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            lag, c = _pair_lead_lag(ret_panel[a], ret_panel[b], max_lag)
            # _pair_lead_lag 返回的是 a vs b.shift 的最佳: lag>0 表示 a 领先 b
            L.loc[a, b] = lag
            C.loc[a, b] = c
            # 反向: b vs a
            L.loc[b, a] = -lag
            C.loc[b, a] = c
    return {"lag": L, "corr": C}


def lead_lag_graph(lag_result: dict, meta: dict, names: dict,
                   corr_threshold: float = 0.3) -> dict:
    """建 lead-lag 有向图 + 顶级领先指标(a 领先 b 且 corr>=阈值 → 边 a→b)。

    返回 {edges: [(a,b,lag,corr)], leaders: DataFrame(id, name, leadership_score)}。
    leadership_score = 领先的下游数 × 平均 corr(越高越是隐藏先行指标)。
    """
    import networkx as nx
    L, C = lag_result["lag"], lag_result["corr"]
    G = nx.DiGraph()
    edges = []
    for a in L.index:
        for b in L.columns:
            if a == b:
                continue
            lag = int(L.loc[a, b])
            c = float(C.loc[a, b])
            if lag > 0 and abs(c) >= corr_threshold:     # a 领先 b
                G.add_edge(a, b, lag=lag, corr=c)
                edges.append({"leader": a, "follower": b, "lag": lag, "corr": c,
                              "leader_name": names.get(a, a), "follower_name": names.get(b, b)})
    # 领先分: 下游数 × 平均相关
    rows = []
    for node in G.nodes:
        succ = list(G.successors(node))
        if not succ:
            continue
        avg_c = np.mean([abs(G[node][s]["corr"]) for s in succ])
        rows.append({"id": node, "name": names.get(node, node),
                     "category": meta.get(node, {}).get("category", ""),
                     "n_followers": len(succ), "avg_corr": float(avg_c),
                     "leadership": float(len(succ) * avg_c)})
    leaders = pd.DataFrame(rows).sort_values("leadership", ascending=False) if rows else pd.DataFrame()
    return {"graph": G, "edges": pd.DataFrame(edges), "leaders": leaders}


# ============== 聚类 ==============
def cluster_series(ret_panel: pd.DataFrame, names: Optional[dict] = None,
                   method: str = "correlation", n_clusters: int = 0) -> dict:
    """层次聚类(基于收益率相关距离)。返回 {linkage, labels, order}。

    linkage 供 plotly dendrogram; labels 每 id 的簇号; order 为叶子顺序(供 heatmap 排序)。
    """
    names = names or {}
    panel = ret_panel.dropna(thresh=int(len(ret_panel) * 0.5), axis=1).dropna()
    if panel.shape[1] < 2:
        return {"linkage": None, "labels": {}, "order": list(panel.columns)}
    corr = panel.corr().fillna(0).values
    dist = np.clip(1 - np.abs(corr), 0, 2)
    np.fill_diagonal(dist, 0.0)
    # 保证对称 + 非负
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    order = [panel.columns[i] for i in hierarchy.leaves_list(Z)]
    labels = {}
    if n_clusters and n_clusters > 1:
        cl = hierarchy.fcluster(Z, t=n_clusters, criterion="maxclust")
        labels = {panel.columns[i]: int(cl[i]) for i in range(len(panel.columns))}
    return {"linkage": Z, "labels": labels, "order": order,
            "ids": list(panel.columns)}


# ============== 发现榜(给定 target) ==============
def _rolling_corr_consistency(a: pd.Series, b: pd.Series, window: int = 36) -> float:
    """滚动相关同号占比(稳定性)。无重叠返回 nan。"""
    if len(a) < window + 2:
        return np.nan
    rc = a.rolling(window).corr(b).dropna()
    if rc.empty:
        return np.nan
    overall = np.sign(a.corr(b))
    return float((np.sign(rc) == overall).mean())


def discover(target: str, ret_panel: pd.DataFrame, meta: dict, names: dict,
             max_lag: int = 8, fdr_alpha: float = 0.05) -> dict:
    """给定 target, 排序全宇宙: 谁最跟随它(|corr|)、谁最领先它(lead-lag)。

    返回 {target, table: DataFrame(id,name,category,corr,q,beta,r2,best_lag,leads_target,stability),
          warning}。q<阈值 的标显著; leads_target=True 的可作为先行指标(但相关非因果)。
    """
    if target not in ret_panel.columns:
        return {"target": target, "error": f"{target} 不在面板"}
    pw = pairwise_corr(ret_panel)
    ll = lead_lag_matrix(ret_panel, max_lag=max_lag)
    y = ret_panel[target]
    rows = []
    for sid in ret_panel.columns:
        if sid == target:
            continue
        x = ret_panel[sid]
        pair = x.dropna().align(y.dropna(), join="inner")
        xv, yv = pair[0], pair[1]
        if len(xv) < 5:
            continue
        r = float(y.corr(x))
        beta = float(np.polyfit(xv, yv, 1)[0]) if xv.std() > 0 else np.nan
        ss_res = float(((yv - (beta * xv + (yv.mean() - beta * xv.mean()))) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        lag = int(ll["lag"].loc[sid, target])    # sid vs target: >0 → sid 领先 target
        leads_target = lag > 0
        stab = _rolling_corr_consistency(x, y)
        rows.append({
            "id": sid, "name": names.get(sid, sid),
            "category": meta.get(sid, {}).get("category", ""),
            "corr": r, "q": float(pw["q"].loc[target, sid]),
            "beta": beta, "r2": r2, "best_lag": lag,
            "leads_target": leads_target, "stability": stab,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return {"target": target, "table": table, "warning": "无可比较序列"}
    table["significant"] = table["q"] < fdr_alpha
    # 综合分: |corr| × 稳定性(q 显著的额外加权)
    stab = table["stability"].fillna(0.5)
    table["score"] = (table["corr"].abs() * stab * (1.5 - table["q"].clip(0, 1))).round(3)
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    return {
        "target": target, "target_name": names.get(target, target),
        "table": table,
        "warning": ("按 |corr|×稳定性×(1-q) 综合排序; q<%.2f 标显著(已 BH-FDR 校正)。"
                    "leads_target=True 提示该序列统计上先于 target 变动(预测先后, 非因果)。"
                    "相关用月频收益率。") % fdr_alpha,
    }


if __name__ == "__main__":
    cfg = load_config()
    from collector.store import universe_panel, ids_of_kind
    panel, names, meta = universe_panel(cfg)
    price_ids = [c for c in panel.columns if meta.get(c, {}).get("kind") == "price"]
    ret = panel[price_ids].pct_change().dropna(how="all")
    # 港股 09988 vs 全宇宙: 谁最跟随/谁领先它
    res = discover("09988", ret, meta, names)
    if "error" in res:
        print(res["error"])
    else:
        print(res["warning"])
        print(res["table"].head(8).to_string(index=False))

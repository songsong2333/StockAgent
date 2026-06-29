"""自研 ML 模型训练器 (Pro)。

横截面面板模型: 对每个(日期, 股票)用该股滞后指标预测未来N日收益/方向。
复用 strategy.forecaster.build_ts_dataset 的单股特征工程, 拼成面板训练。
模型: LightGBM(回归) / Ridge。评估: IC / 方向命中率 / top-k 超额。诚实标注命中率<55%不可靠。

⚠️ 单股/小样本短周期预测信噪比低, 模型仅供参考, 切勿重仓依赖。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from common import setup_logger, to_qlib_code, load_config, PROJECT_ROOT
from strategy.forecaster import build_ts_dataset, TS_FEATURES, ridge_fit, ridge_predict

log = setup_logger("model.trainer")
MODEL_DIR = Path("data/cache/models")


def load_universe(cfg: dict, name: str = "watchlist") -> List[str]:
    """读 universe: watchlist / sector_leaders。返回6位代码列表。"""
    import yaml
    mapping = {"watchlist": "config/watchlist.yaml", "sector_leaders": "config/sector_leaders.yaml"}
    path = PROJECT_ROOT / mapping.get(name, mapping["watchlist"])
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data.get("leaders") or data.get("stocks") or []
    return [s["code"] for s in items]


def build_panel(cfg: dict, codes: List[str], horizon: int, start: str = "2024-10-01"):
    """拼横截面面板: 行=(日期,股票), 列=特征+forward收益。"""
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    frames = []
    for code in codes:
        qc = to_qlib_code(code)
        p = raw_dir / f"{qc}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        d, _ = build_ts_dataset(df, horizon)
        d = d[d["date"] >= pd.Timestamp(start)].copy()
        d["code"] = code
        need = TS_FEATURES + ["fwd_ret", "date", "code", "close"]
        d = d[[c for c in need if c in d.columns]].dropna(subset=TS_FEATURES + ["fwd_ret"])
        frames.append(d)
    if not frames:
        raise ValueError("面板为空, 数据不足")
    return pd.concat(frames, ignore_index=True)


def train_pipeline(cfg: dict, universe: str = "watchlist", horizon: int = 5,
                   label: str = "ret", model_type: str = "lgbm",
                   train_end: str = "2025-12-31", topk: int = 3) -> dict:
    """端到端: 建面板 → 训练 → 评估 → 保存。返回指标 + 模型路径。"""
    codes = load_universe(cfg, universe)
    panel = build_panel(cfg, codes, horizon)
    panel = panel.sort_values("date").reset_index(drop=True)
    split = pd.Timestamp(train_end)
    tr = panel[panel["date"] <= split]
    te = panel[panel["date"] > split]
    if len(tr) < 200 or len(te) < 50:
        raise ValueError(f"样本不足: 训练{len(tr)}/测试{len(te)}")

    Xtr = tr[TS_FEATURES].to_numpy(float)
    yret_tr = tr["fwd_ret"].to_numpy(float)
    ytr = yret_tr if label == "ret" else (yret_tr > 0).astype(int)
    Xte = te[TS_FEATURES].to_numpy(float)
    yret_te = te["fwd_ret"].to_numpy(float)

    # 训练
    if model_type == "ridge":
        model = ridge_fit(Xtr, yret_tr, lam=1.0)
        pred_te = ridge_predict(model, Xte)
        pred_kind = "ridge"
    else:
        import lightgbm as lgb
        y_lgb = yret_tr if label == "ret" else ytr
        ds = lgb.Dataset(Xtr, label=y_lgb)
        params = {"objective": "regression" if label == "ret" else "binary",
                  "num_leaves": 16, "learning_rate": 0.05, "feature_fraction": 0.9,
                  "bagging_fraction": 0.9, "bagging_freq": 5, "verbose": -1,
                  "lambda_l1": 0.5, "lambda_l2": 0.5}
        model = lgb.train(params, ds, num_boost_round=200)
        pred_te = model.predict(Xte)
        pred_kind = "lgbm"

    # 评估
    te = te.assign(pred=pred_te, actual=yret_te)
    # IC: 每日 spearman(pred, actual) 均值
    ic = te.groupby("date").apply(
        lambda g: g["pred"].corr(g["actual"], method="spearman")
        if len(g) > 2 else np.nan).mean()
    # 方向命中
    hit = ((te["pred"] > te["pred"].median()) == (te["actual"] > 0)).mean()
    # top-k 超额: 每日 pred 最高 topk 的平均 actual 减 全市场平均 actual
    def topk_excess(g):
        if len(g) <= topk:
            return 0.0
        return g.nlargest(topk, "pred")["actual"].mean() - g["actual"].mean()
    excess = te.groupby("date").apply(topk_excess).mean()
    strat_ret = te.groupby("date").apply(topk_excess).sum()

    reliable = hit >= 0.55
    metrics = {
        "universe": universe, "horizon": horizon, "label": label, "model": model_type,
        "n_train": len(tr), "n_test": len(te), "train_end": train_end,
        "test_start": str(te["date"].min().date()), "test_end": str(te["date"].max().date()),
        "IC": round(float(ic) if pd.notna(ic) else 0, 3),
        "direction_hit": round(float(hit), 3),
        "topk_excess_per_day": round(float(excess), 4),
        "strategy_cum": round(float(strat_ret), 3),
        "reliable": bool(reliable),
    }
    # 保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{universe}_{label}_{model_type}_h{horizon}"
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "kind": pred_kind, "features": TS_FEATURES,
                     "metrics": metrics, "label": label, "horizon": horizon}, f)
    metrics["model_path"] = str(path)
    prune_models(keep=5)   # 最多保留5个, 超出删最旧
    log.info(f"训练完成 {name}: {metrics}")
    return metrics


def prune_models(keep: int = 5):
    """保留最新的 keep 个模型(按修改时间), 删除其余。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pkls = sorted(MODEL_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pkls[keep:]:
        try:
            p.unlink()
            log.info(f"删除旧模型(超出{keep}个上限): {p.name}")
        except Exception:
            pass


def list_models() -> List[dict]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(MODEL_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(p, "rb") as f:
                m = pickle.load(f)
            out.append({"file": p.name, "metrics": m.get("metrics", {}), "path": str(p)})
        except Exception:
            pass
    return out


def delete_model(filename: str) -> bool:
    p = MODEL_DIR / filename
    if p.exists():
        try:
            p.unlink(); return True
        except Exception:
            return False
    return False


def metrics_to_markdown(m: dict) -> str:
    L = [f"## 🧠 自研模型训练结果\n"]
    L.append(f"- universe: **{m['universe']}** | 标签: {m['label']}({m['horizon']}日) | 模型: {m['model']}")
    L.append(f"- 训练 {m['n_train']} / 测试 {m['n_test']} | 测试区间 {m['test_start']} ~ {m['test_end']}")
    L.append(f"- **IC**: {m['IC']} | **方向命中率**: {m['direction_hit']:.1%} | top{m.get('topk',3)}日均超额: {m['topk_excess_per_day']:+.2%}")
    if m["reliable"]:
        L.append("- ✅ 命中率≥55%, 有一定参考价值(仍需结合技术与基本面)。")
    else:
        L.append("- ⚠️ **命中率<55%, 模型对该池预测力弱, 不可单独作为交易依据**(单股短周期信噪比低, 属正常)。")
    L.append(f"- 模型已保存: `{m.get('model_path','')}`")
    L.append("\n> 该模型为横截面面板模型(用各股滞后指标预测未来收益排名), 与选股/信号页的 qlib 模型独立。仅供参考。")
    return "\n".join(L)


if __name__ == "__main__":
    cfg = load_config()
    m = train_pipeline(cfg, universe="watchlist", horizon=5, label="ret", model_type="lgbm")
    print(metrics_to_markdown(m))

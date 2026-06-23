"""单股时序预测: 用该股 2025 年自身历史训练, 预测 2026 未来N日收益。

与多因子横截面模型(选股排名)不同, 这是"自回归时序模型":
  特征 = 该股自己的滞后指标(均线比/动量/波动/RSI/量能/位置);
  标签 = 未来N日收益率 close[t+N]/close[t]-1;
  模型 = Ridge 回归(闭式解, 小样本稳健, 无外部依赖);
  训练 = 2025全年; 测试/预测 = 2026。

注意: 单股时序收益预测信噪比低, R²通常很低(这是市场规律, 不是bug)。
输出要诚实给出置信度与2026回测命中率。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import setup_logger, to_qlib_code, load_config, PROJECT_ROOT

log = setup_logger("strategy.forecaster")

HORIZON = 5  # 预测未来5个交易日

# 时序特征(该股自己的滞后指标)
TS_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "close_over_ma5", "close_over_ma20", "close_over_ma60", "ma5_over_ma20",
    "rsi14", "vol20", "vol_ratio", "atr_ratio", "range_pos60", "dist_ma20",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    down = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_ts_dataset(df: pd.DataFrame, horizon: int = HORIZON):
    """构造时序特征矩阵 + 未来N日收益标签。返回 (增强后的df, 特征列)。"""
    d = df.copy()
    c = d["close"]
    # 滞后收益
    for n in (1, 2, 3, 5, 10, 20):
        d[f"ret_{n}"] = c.pct_change(n)
    # 均线比
    ma5, ma20, ma60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    d["close_over_ma5"] = c / ma5 - 1
    d["close_over_ma20"] = c / ma20 - 1
    d["close_over_ma60"] = c / ma60 - 1
    d["ma5_over_ma20"] = ma5 / ma20 - 1
    # RSI / 波动 / ATR / 量能 / 位置
    d["rsi14"] = _rsi(c, 14)
    ret1 = c.pct_change()
    d["vol20"] = ret1.rolling(20).std()
    d["vol_ratio"] = d["volume"].rolling(20).mean() / d["volume"].rolling(60).mean() - 1
    tr = pd.concat([(d["high"] - d["low"]),
                    (d["high"] - c.shift(1)).abs(),
                    (d["low"] - c.shift(1)).abs()], axis=1).max(axis=1)
    d["atr_ratio"] = tr.rolling(14).mean() / c
    hi60, lo60 = c.rolling(60).max(), c.rolling(60).min()
    d["range_pos60"] = (c - lo60) / (hi60 - lo60)
    d["dist_ma20"] = c / ma20 - 1
    # 标签: 未来 horizon 日收益
    d["fwd_ret"] = c.shift(-horizon) / c - 1
    # 清洗: inf -> nan, 极端值裁剪, 避免数值溢出污染回归
    for f in TS_FEATURES:
        d[f] = d[f].replace([np.inf, -np.inf], np.nan)
        # 用百分位裁剪极端值(仅对有限值)
        finite = d[f].replace([np.inf, -np.inf], np.nan()).dropna()
        if len(finite) > 10:
            lo, hi = finite.quantile(0.001), finite.quantile(0.999)
            d[f] = d[f].clip(lower=lo, upper=hi)
    return d, TS_FEATURES


# ===================== Ridge (闭式解, numpy) =====================
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    """标准化 + Ridge 闭式解。返回 {mu, sigma, w, lam}。"""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    # 增广偏置项
    Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    n = Xa.shape[1]
    reg = lam * np.eye(n)
    reg[-1, -1] = 0.0  # 不正则化偏置
    w = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)
    return {"mu": mu, "sigma": sigma, "w": w, "lam": lam}


def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xs = (X - model["mu"]) / model["sigma"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)  # 缺失填0(=标准化后均值)
    Xs = np.clip(Xs, -10, 10)  # 防溢出
    Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    return Xa @ model["w"]


# ===================== 训练 / 回测 / 预测 =====================
def train_forecast(df: pd.DataFrame, train_start="2025-01-01", train_end="2025-12-31",
                   horizon: int = HORIZON, lam: float = 1.0):
    d, feats = build_ts_dataset(df, horizon)
    d["date"] = pd.to_datetime(d["date"])
    m = (d["date"] >= train_start) & (d["date"] <= train_end)
    need = feats + ["fwd_ret"]
    tr = d[m & d[need].notna().all(axis=1)]
    Xtr = tr[feats].to_numpy(float)
    ytr = tr["fwd_ret"].to_numpy(float)
    model = ridge_fit(Xtr, ytr, lam=lam)
    # 训练集拟合度
    pred_tr = ridge_predict(model, Xtr)
    r2_tr = 1 - np.sum((ytr - pred_tr) ** 2) / np.sum((ytr - ytr.mean()) ** 2) if ytr.std() else 0
    log.info(f"训练: {len(tr)} 样本, R²={r2_tr:.3f}")
    return model, feats, d, {"n_train": len(tr), "r2_train": r2_tr}


def backtest_2026(model, feats, d, test_start="2026-01-01", test_end="2026-06-18"):
    """2026样本外回测: 模型只用2025训练, 对2026每天预测未来N日, 对比实际。"""
    d["date"] = pd.to_datetime(d["date"])
    m = (d["date"] >= test_start) & (d["date"] <= test_end) & d[feats + ["fwd_ret"]].notna().all(axis=1)
    te = d[m]
    if te.empty:
        return {}
    Xte = te[feats].to_numpy(float)
    yte = te["fwd_ret"].to_numpy(float)
    pred = ridge_predict(model, Xte)
    # 方向命中率
    hit = ((pred > 0) == (yte > 0)).mean()
    # 相关性 / MAE
    corr = float(np.corrcoef(pred, yte)[0, 1]) if te.shape[0] > 2 else 0.0
    mae = float(np.mean(np.abs(pred - yte)))
    # 策略: 预测>0则做多该段, 累计实际收益
    strat_ret = np.where(pred > 0, yte, 0).sum()
    # 分组: 预测分位 vs 实际收益(因子单调性)
    return {
        "n_test": len(te), "direction_hit": hit, "corr": corr, "mae": mae,
        "strategy_cum_ret": strat_ret, "buy_hold_ret": yte.sum(),
        "pred_last5_mean": float(np.mean(pred[-5:])) if len(pred) >= 5 else float(np.mean(pred)),
    }


def forecast_latest(code: str, cfg: dict, horizon: int = HORIZON) -> dict:
    """训练(2025) + 预测最新日的未来N日收益 + 2026回测验证。"""
    qc = to_qlib_code(code)
    p = Path(cfg["paths"]["raw_dir"]) / f"{qc}.parquet"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    model, feats, d, train_info = train_forecast(df, lam=1.0)
    bt = backtest_2026(model, feats, d)

    # 最新日预测
    last_row = d.iloc[-1]
    X_last = last_row[feats].to_numpy(float).reshape(1, -1)
    pred_ret = float(ridge_predict(model, X_last)[0])
    last_close = float(last_row["close"])
    last_date = last_row["date"].strftime("%Y-%m-%d")
    target_price = last_close * (1 + pred_ret)
    direction = "看涨" if pred_ret > 0 else "看跌"

    # 置信度: 基于2026方向命中率 + 预测幅度
    hit = bt.get("direction_hit", 0.5)
    vol_fwd = np.nanstd(d["fwd_ret"].dropna().to_numpy())
    mag = abs(pred_ret) / vol_fwd if vol_fwd else 0
    if hit >= 0.58 and mag > 0.3:
        conf = "中高"
    elif hit >= 0.52:
        conf = "中"
    else:
        conf = "低(模型对该股预测力弱)"

    return {
        "code": code, "last_date": last_date, "last_close": last_close,
        "horizon": horizon, "pred_ret": pred_ret, "target_price": target_price,
        "direction": direction, "confidence": conf,
        "train_info": train_info, "backtest": bt,
    }


def to_markdown(r: dict) -> str:
    bt = r["backtest"]
    L = [f"# 🔮 {r['code']} 时序预测 (2025训练→2026预测)\n"]
    L.append(f"**最新**: {r['last_close']} ({r['last_date']}) | **预测未来{r['horizon']}个交易日**\n")
    L.append("## 模型预测")
    L.append(f"- **预测{r['horizon']}日收益**: {r['pred_ret']:+.2%}")
    L.append(f"- **预测目标价**: {r['target_price']:.2f} (现价 {r['last_close']:.2f})")
    L.append(f"- **方向**: {r['direction']}  | **置信度**: {r['confidence']}")
    L.append("\n## 2026样本外回测(验证模型对该股的有效性)")
    if bt:
        L.append(f"- 方向命中率: **{bt['direction_hit']:.1%}** (n={bt['n_test']}) "
                 f"{'✅有效' if bt['direction_hit']>=0.55 else '⚠️接近随机/无效' if bt['direction_hit']>=0.50 else '❌无效'}")
        L.append(f"- 预测-实际相关性: {bt['corr']:+.3f} | MAE: {bt['mae']:.3f}")
        L.append(f"- 跟随预测做多累计: {bt['strategy_cum_ret']:+.2%} | 一直持有累计: {bt['buy_hold_ret']:+.2%}")
    L.append(f"\n## 训练详情")
    L.append(f"- 训练样本: {r['train_info']['n_train']} (2025全年) | 训练R²: {r['train_info']['r2_train']:.3f}")
    L.append(f"- 模型: Ridge回归 + {len(TS_FEATURES)}个自回归特征(均线比/动量/波动/RSI/量能/位置)")
    L.append("\n## 解读")
    if bt.get("direction_hit", 0) < 0.52:
        L.append("- ⚠️ 该模型对此股**预测力弱**(命中率接近随机), 预测值仅供极弱参考, **不要据此交易**。")
        L.append("- 单股短期收益噪声极大, 任何模型都难稳定预测——这是市场规律。")
    else:
        L.append(f"- 模型对此股方向命中率 {bt['direction_hit']:.0%}, 有一定参考价值, 但仍需结合技术面与基本面。")
    L.append("- 这是**时序自回归**(用自己的历史预测自己), 与选股的横截面模型完全不同。")
    L.append("\n⚠️ 仅供参考, 不构成投资建议。短周期股价预测本质极难, 请勿重仓依赖。")
    return "\n".join(L)


if __name__ == "__main__":
    cfg = load_config()
    r = forecast_latest("603629", cfg)
    print(to_markdown(r))

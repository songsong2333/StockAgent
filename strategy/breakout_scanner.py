"""低位启动板块扫描器。

思路:
  1. 龙头启动模板: 从已大涨的龙头股中, 提取它们"启动前夜"的量价特征分布
     (低位 + 盘整收缩 + 均线拐头 + 量能温和 + 紧凑度), 作为"准备爆发"的模板。
  2. 板块扫描: 对申万行业指数计算同一组特征(最新日), 评估与模板的相似度 + 低位度。
  3. 涨价逻辑因子: 大宗品(铜/锡/镍/铝)期货的"低位突破"状态 + 科技细分板块量价启动,
     作为板块启动的领先催化。涨价-板块映射驱动。
  4. 融合打分: 模板相似度 + 低位度 + 启动苗头 + 涨价催化 → 排名"低位待爆发"板块。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import setup_logger, PROJECT_ROOT, load_config

log = setup_logger("strategy.scanner")

# 特征名
FEATURES = [
    "pos_high",       # 距250日高: 越负=越低位
    "range_pos",      # 在250日区间位置: 0=最低, 1=最高
    "vol_contraction",# 20日/60日波动: <1=收缩盘整
    "ma_align",       # (MA20-MA60)/价: >0=短均线在上(拐头)
    "mom20",          # 20日动量
    "vol_ratio",      # 20日/60日量比-1: 温和放量
    "tightness",      # 20日振幅/价: 越小=越紧凑
]

# 涨价大宗品 -> {期货symbol, 对应申万二级板块代码}
COMMODITY_MAP = {
    "铜": {"fut": "cu0", "sectors": ["801055"]},   # 工业金属
    "铝": {"fut": "al0", "sectors": ["801055"]},
    "锡": {"fut": "sn0", "sectors": ["801054"]},   # 小金属
    "镍": {"fut": "ni0", "sectors": ["801054", "801056"]},  # 小金属/能源金属
}

# 重点扫描的申万二级行业(科技+有色细分, 涨价逻辑密集)
KEY_L2 = ["801051", "801053", "801054", "801055", "801056",  # 金属类
          "801081", "801086", "801082", "801083", "801084"]   # 半导体/电子化学品/元件等


# ===================== 特征计算 =====================
def setup_features(close: np.ndarray, volume: Optional[np.ndarray] = None,
                   t: int = -1, lookback: int = 250) -> Optional[Dict[str, float]]:
    """计算 t 时刻(默认最后一根)的启动前夜特征。数据不足返回 None。"""
    n = len(close)
    if t < 0:
        t = n - 1
    if t < lookback or t < 60:
        return None
    c = close
    # 距高位/区间位置
    window = c[t - lookback + 1: t + 1]
    hi, lo = np.max(window), np.min(window)
    pos_high = c[t] / hi - 1
    range_pos = (c[t] - lo) / (hi - lo) if hi > lo else 0.5
    # 波动收缩
    rets = np.diff(np.log(c))
    vol20 = np.std(rets[t - 20:t], ddof=1) if t >= 21 else np.nan
    vol60 = np.std(rets[t - 60:t], ddof=1) if t >= 61 else np.nan
    vol_contraction = vol20 / vol60 if vol60 and not np.isnan(vol60) else 1.0
    # 均线拐头
    ma20 = np.mean(c[t - 19:t + 1])
    ma60 = np.mean(c[t - 59:t + 1])
    ma_align = (ma20 - ma60) / c[t]
    # 动量
    mom20 = c[t] / c[t - 20] - 1
    # 量比
    vol_ratio = 0.0
    if volume is not None and t >= 60:
        v20 = np.mean(volume[t - 19:t + 1])
        v60 = np.mean(volume[t - 59:t + 1])
        vol_ratio = v20 / v60 - 1 if v60 > 0 else 0.0
    # 紧凑度
    w20 = c[t - 19:t + 1]
    tightness = (np.max(w20) - np.min(w20)) / c[t]
    return {
        "pos_high": pos_high, "range_pos": range_pos, "vol_contraction": vol_contraction,
        "ma_align": ma_align, "mom20": mom20, "vol_ratio": vol_ratio, "tightness": tightness,
    }


# ===================== 龙头启动模板 =====================
def build_leader_template(cfg: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """从 watchlist 龙头股提取"启动前夜"特征分布, 返回 (均值, 标准差)。"""
    import yaml
    from pathlib import Path
    from common import to_qlib_code
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    with open(PROJECT_ROOT / "config" / "watchlist.yaml", "r", encoding="utf-8") as f:
        wl = yaml.safe_load(f)["stocks"]

    samples = []
    for s in wl:
        qc = to_qlib_code(s["code"])
        p = raw_dir / f"{qc}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
        close = df["close"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float) if "volume" in df else None
        n = len(close)
        if n < 320:
            continue
        # 找最佳60日前瞻收益的起点 t* (即"启动前夜")
        # forward_60d_return[t] = close[t+60]/close[t]-1, 取最大者对应 t
        if n <= 61:
            continue
        fwd = np.full(n, np.nan)
        for i in range(60, n):
            fwd[i - 60] = close[i] / close[i - 60] - 1
        # 在有前瞻的范围内取 argmax (排除最近60日无前瞻)
        valid_end = n - 60
        if valid_end <= 250:
            continue
        t_star = int(np.nanargmax(fwd[250:valid_end])) + 250
        feat = setup_features(close, vol, t=t_star)
        if feat and all(not np.isnan(v) for v in feat.values()):
            samples.append(feat)
            log.info(f"  {s['name']}({s['code']}) 启动点 t={t_star}/{n} "
                     f"(前瞻60日收益 {fwd[t_star]:.1%})")

    if not samples:
        raise ValueError("无法构建龙头模板, 数据不足")
    df_s = pd.DataFrame(samples)
    log.info(f"龙头模板: {len(samples)} 只样本")
    return df_s.mean().to_dict(), df_s.std().to_dict()


# ===================== 板块指数获取 =====================
def fetch_sector_indices(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """获取申万行业指数日线。返回 {code: DataFrame[date,close,volume]}。"""
    import akshare as ak
    out = {}
    for code in tqdm_bar(codes, "板块指数"):
        try:
            h = ak.index_hist_sw(symbol=code)
            h = h.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume"})
            h["date"] = pd.to_datetime(h["date"])
            h = h[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
            h["close"] = h["close"].astype(float)
            h["volume"] = h["volume"].astype(float)
            out[code] = h
        except Exception as e:
            log.warning(f"板块 {code} 获取失败: {e}")
    return out


def fetch_sector_names() -> Dict[str, str]:
    """{code: 板块名} (一级+二级)。"""
    import akshare as ak
    names = {}
    for fn in [ak.sw_index_first_info, ak.sw_index_second_info]:
        try:
            df = fn()
            for _, r in df.iterrows():
                code = str(r["行业代码"]).replace(".SI", "")
                names[code] = r["行业名称"]
        except Exception as e:
            log.warning(f"板块名获取失败: {e}")
    return names


def fetch_commodities() -> Dict[str, pd.DataFrame]:
    """获取大宗品期货日线。返回 {名称: DataFrame[date,close]}。"""
    import akshare as ak
    import time
    out = {}
    for nm, info in COMMODITY_MAP.items():
        try:
            df = ak.futures_zh_daily_sina(symbol=info["fut"])
            df = df.rename(columns={"date": "date", "close": "close"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
            df["close"] = df["close"].astype(float)
            out[nm] = df
            time.sleep(0.6)
        except Exception as e:
            log.warning(f"大宗品 {nm} 获取失败: {e}")
    return out


def tqdm_bar(items, desc):
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        return list(items)


# ===================== 打分 =====================
def commodity_breakout_score(price: np.ndarray) -> Tuple[float, dict]:
    """大宗品"低位突破"分: 低位+刚突破60日均线+动量转正。返回 (0~1分, 特征)。"""
    f = setup_features(price, None)
    if f is None:
        return 0.0, {}
    ma60 = np.mean(price[-60:])
    ma20 = np.mean(price[-20:])
    # 低位: 距250日高越远越好(限0~1)
    low = float(np.clip(-f["pos_high"] / 0.5, 0, 1))   # 距高50%以上=满分
    # 刚突破: 价在MA60之上但未远离, MA20上穿MA60
    breakout = 1.0 if (price[-1] > ma60 and f["ma_align"] > 0 and f["ma_align"] < 0.15) else 0.3
    # 动量转正
    mom = float(np.clip(f["mom20"] / 0.10, 0, 1))      # 20日涨10%=满分
    score = 0.5 * low + 0.3 * breakout + 0.2 * mom
    return score, f


def score_sector(feat: dict, tmpl_mean: dict, tmpl_std: dict) -> dict:
    """对单个板块最新特征打分。返回各维度分。

    核心原则: "低位待爆发" = 已止跌拐头(MA拐头>=0) + 距高位有空间 + 走势像龙头启动前。
    下跌中(MA拐头为负)的板块是"下跌刀具", 不是待启动, 重罚。
    龙头启动前夜真实特征: pos_high≈-0.16, range_pos≈0.70, ma_align≈+0.108, mom20≈+0.035。
    """
    # 1) 模板相似度: 特征到模板均值的标准化距离(越小越像)
    zs = []
    for k in FEATURES:
        s = tmpl_std.get(k, 1) or 1
        zs.append(abs(feat[k] - tmpl_mean[k]) / s)
    template_fit = 1.0 / (1.0 + np.mean(zs))   # 0~1, 越大越像

    # 2) 上涨空间: 奖励 pos_high 在 -10%~-45%(距高有空间且未崩), -0.20 附近最优
    ph = feat["pos_high"]
    if -0.45 <= ph <= -0.05:
        low_room = float(np.clip(1.0 - abs(ph - (-0.20)) / 0.30, 0, 1))
    else:
        low_room = 0.2

    # 3) 启动苗头: 均线拐头(温和正向) + 盘整收缩 + 动量小正 + 紧凑
    seed = 0.35 * float(np.clip(feat["ma_align"] / 0.05, 0, 1)) \
         + 0.20 * float(np.clip(1 - feat["vol_contraction"], 0, 1)) \
         + 0.25 * float(np.clip(feat["mom20"] / 0.08, 0, 1)) \
         + 0.20 * float(np.clip(0.22 - feat["tightness"], 0, 1) / 0.22)

    # 4) 止跌拐头门槛: MA拐头<0 说明仍在下跌 = 下跌刀具, 不是待启动, 重罚
    basing_gate = 1.0
    if feat["ma_align"] < -0.005:
        basing_gate = 0.15
    elif feat["ma_align"] < 0:
        basing_gate = 0.5
    if feat["range_pos"] > 0.90 and feat["mom20"] > 0.15:  # 已爆发
        basing_gate *= 0.3

    low_room *= basing_gate
    seed *= basing_gate

    return {"template_fit": template_fit, "low_position": low_room,
            "breakout_seed": seed, "feat": feat}


def scan(cfg: dict, top_n: int = 12) -> pd.DataFrame:
    """主扫描。返回板块排名 DataFrame。"""
    import akshare as ak
    log.info("=== 1) 构建龙头启动模板 ===")
    tmpl_mean, tmpl_std = build_leader_template(cfg)
    for k in FEATURES:
        log.info(f"  {k}: 均值={tmpl_mean[k]:.3f} 标准差={tmpl_std[k]:.3f}")

    names = fetch_sector_names()
    # 一级 + 重点二级
    codes_l1 = [c for c in names if c.startswith("801") and len(c) == 6 and c < "801200"]
    codes = sorted(set(codes_l1 + KEY_L2))
    log.info(f"=== 2) 获取 {len(codes)} 个板块指数 ===")
    sectors = fetch_sector_indices(codes)

    log.info("=== 3) 获取大宗品期货(涨价逻辑) ===")
    commodities = fetch_commodities()
    comm_scores = {}
    for nm, df in commodities.items():
        sc, f = commodity_breakout_score(df["close"].to_numpy())
        comm_scores[nm] = (sc, f)
        log.info(f"  {nm} 涨价突破分={sc:.2f} (距高 {f.get('pos_high',0):.1%}, "
                 f"MA拐头 {f.get('ma_align',0):.3f}, mom20 {f.get('mom20',0):.1%})")

    log.info("=== 4) 板块打分 ===")
    rows = []
    for code, df in sectors.items():
        if len(df) < 320:
            continue
        feat = setup_features(df["close"].to_numpy(), df["volume"].to_numpy())
        if feat is None:
            continue
        sc = score_sector(feat, tmpl_mean, tmpl_std)
        # 涨价催化: 该板块关联的大宗品最高突破分
        hike = 0.0
        for nm, info in COMMODITY_MAP.items():
            if code in info["sectors"] and nm in comm_scores:
                hike = max(hike, comm_scores[nm][0])
        # 融合总分
        total = (0.30 * sc["template_fit"] + 0.30 * sc["low_position"]
                 + 0.20 * sc["breakout_seed"] + 0.20 * hike)
        rows.append({
            "代码": code, "板块": names.get(code, code),
            "总分": round(total, 3),
            "模板相似": round(sc["template_fit"], 3),
            "低位度": round(sc["low_position"], 3),
            "启动苗头": round(sc["breakout_seed"], 3),
            "涨价催化": round(hike, 3),
            "距250日高": f"{feat['pos_high']:.1%}",
            "区间位置": f"{feat['range_pos']:.2f}",
            "MA拐头": f"{feat['ma_align']:+.3f}",
            "20日动量": f"{feat['mom20']:+.1%}",
            "波动收缩": f"{feat['vol_contraction']:.2f}",
        })
    res = pd.DataFrame(rows).sort_values("总分", ascending=False).reset_index(drop=True)
    return res.head(top_n)


def to_markdown(res: pd.DataFrame, tmpl_mean: dict = None) -> str:
    lines = ["# 🔍 低位待爆发板块扫描结果\n"]
    lines.append("融合打分 = 0.3×模板相似 + 0.3×低位度 + 0.2×启动苗头 + 0.2×涨价催化\n")
    lines.append("| 排名 | 板块 | 总分 | 模板相似 | 低位度 | 启动苗头 | 涨价催化 | 距250日高 | 区间位置 | MA拐头 | 20日动量 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in res.iterrows():
        lines.append(f"| {i+1} | {r['板块']} | {r['总分']} | {r['模板相似']} | {r['低位度']} | "
                     f"{r['启动苗头']} | {r['涨价催化']} | {r['距250日高']} | {r['区间位置']} | "
                     f"{r['MA拐头']} | {r['20日动量']} |")
    lines.append("\n**读法**: 低位度高=位置低有空间; 模板相似=走势像龙头启动前; "
                 "启动苗头=均线拐头+盘整收缩+温和放量; 涨价催化=关联大宗品(铜锡镍铝)突破。")
    lines.append("\n**策略含义**: 优选 总分高 + 低位度高 + 涨价催化高 的板块 = 低位+涨价驱动+即将启动。")
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = load_config()
    res = scan(cfg)
    print(res.to_string(index=False))

"""板块龙头自动挖掘: 按行业成分股 + 流通市值, 自动选出每个板块的龙头。

算法:
  1. 取行业列表(同花顺行业 stock_board_industry_name_ths, 或用现有 sector_leaders 里的板块名)
  2. 一次取全市场实时行情 stock_zh_a_spot_em → {代码: 流通市值}
  3. 每个行业取成分股 stock_board_industry_cons_em, 按流通市值取 topK = 龙头
  4. 输出 [{code, name, sector}] 可写回 sector_leaders.yaml

注: 东财接口偶发限流/盘外慢, 内置重试与跳过; 挖掘结果供用户在 App 内再审核/编辑。
"""
from __future__ import annotations

import time
from typing import List, Optional

import akshare as ak
import pandas as pd

from common import setup_logger

log = setup_logger("strategy.leader_miner")


def _market_caps() -> dict:
    """{6位代码: 流通市值(元)} 一次取全市场。"""
    df = ak.stock_zh_a_spot_em()
    code_col = "代码" if "代码" in df.columns else df.columns[1]
    cap_col = next((c for c in df.columns if "流通市值" in str(c)), None)
    if cap_col is None:
        cap_col = next((c for c in df.columns if "市值" in str(c)), None)
    out = {}
    for _, r in df.iterrows():
        try:
            out[str(r[code_col]).zfill(6)] = float(r[cap_col])
        except Exception:
            pass
    return out


def _industry_list() -> List[str]:
    """同花顺行业名列表。"""
    df = ak.stock_board_industry_name_ths()
    return df["name"].tolist()


def _constituents(industry: str) -> pd.DataFrame:
    """某行业的成分股(代码/名称)。"""
    df = ak.stock_board_industry_cons_em(symbol=industry)
    code_col = "代码" if "代码" in df.columns else df.columns[1]
    name_col = "名称" if "名称" in df.columns else df.columns[2]
    return pd.DataFrame({"code": df[code_col].astype(str).str.zfill(6), "name": df[name_col]})


def mine(topk: int = 2, sectors: Optional[List[str]] = None,
         sleep: float = 1.0) -> List[dict]:
    """挖掘龙头。sectors=None 用同花顺全行业; 否则只挖指定行业。

    返回 [{code, name, sector}] (已按行业+市值排序)。
    """
    sectors = sectors or _industry_list()
    log.info(f"挖掘 {len(sectors)} 个行业, 每行业 top{topk}")
    caps = _market_caps()
    log.info(f"取得 {len(caps)} 只股票流通市值")
    out = []
    fail = []
    for ind in sectors:
        for attempt in range(2):
            try:
                cons = _constituents(ind)
                cons["cap"] = cons["code"].map(caps)
                cons = cons.dropna(subset=["cap"]).sort_values("cap", ascending=False)
                for _, r in cons.head(topk).iterrows():
                    out.append({"code": r["code"], "name": r["name"], "sector": ind})
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(sleep * 2)
                else:
                    fail.append(ind)
        time.sleep(sleep)
    log.info(f"挖掘完成: {len(out)} 只龙头, {len(fail)} 个行业失败({fail[:5]}...)")
    return out


def to_yaml_rows(leaders: List[dict]) -> List[dict]:
    """整理为 sector_leaders.yaml 的 leaders 格式。"""
    return [{"code": l["code"], "name": l["name"], "sector": l["sector"]} for l in leaders]


if __name__ == "__main__":
    # 默认: 只刷新当前 sector_leaders 里已有的板块
    import yaml
    from pathlib import Path
    from common import PROJECT_ROOT
    p = PROJECT_ROOT / "config" / "sector_leaders.yaml"
    existing = []
    if p.exists():
        existing = [s["sector"] for s in yaml.safe_load(p.read_text(encoding="utf-8")).get("leaders", [])]
    sectors = sorted(set(existing)) or None
    leaders = mine(topk=2, sectors=sectors)
    print(f"挖出 {len(leaders)} 只:")
    for l in leaders[:20]:
        print(f"  {l['sector']}: {l['name']}({l['code']})")

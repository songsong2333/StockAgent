"""回测报告: 指标格式化 + markdown 输出。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from common import setup_logger, ensure_dir

log = setup_logger("backtest.report")


def format_metrics(metrics: dict) -> dict:
    """把 qlib 指标转为易读字典。"""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = round(float(v), 4)
        else:
            out[k] = v
    return out


def _period_summary(report: Optional[pd.DataFrame]) -> Optional[dict]:
    """从净值曲线算真实区间统计: 起止日/交易日数/绝对收益/折合年数。无法计算时返回 None。"""
    if report is None or report.empty or "account" not in report.columns or len(report) < 2:
        return None
    a0 = float(report["account"].iloc[0])
    a1 = float(report["account"].iloc[-1])
    if a0 <= 0:
        return None
    n_days = len(report)
    return {"start": report.index[0], "end": report.index[-1], "a0": a0, "a1": a1,
            "period_ret": a1 / a0 - 1, "n_days": n_days, "years": n_days / 252}


def to_markdown(metrics: dict, report: Optional[pd.DataFrame] = None,
                cfg: Optional[dict] = None) -> str:
    """生成 markdown 报告。

    诚实优先: 先给真实区间绝对收益与设置(股票池/集中度), 短区间时对 annualized_return
    明确标注"年化外推", 避免把小样本、集中持仓的区间收益误读成可持续年化能力。
    cfg 可选: 提供时补充股票池大小与 topk 上下文。
    """
    lines = ["# 回测报告\n"]
    ps = _period_summary(report)

    if cfg is not None:
        try:
            from factor.handler import load_universe
            uni = load_universe(cfg)
            n_uni = "全市场" if uni == "all" else f"{len(uni)} 只"
            topk = cfg.get("backtest", {}).get("topk", "?")
            lines.append(f"> 📌 **设置**: 股票池 {n_uni} · 持仓 topk={topk}(集中持仓, 高波动)\n>")
        except Exception:
            pass
    if ps:
        s = pd.Timestamp(ps["start"]).date()
        e = pd.Timestamp(ps["end"]).date()
        lines.append(f"> 📌 **测试区间**: {s} ~ {e} · {ps['n_days']} 个交易日(≈{ps['years']:.1f} 年)")
        lines.append(f"> 💰 **区间绝对收益**: {ps['a0']:,.0f} → {ps['a1']:,.0f} = **{ps['period_ret']:+.1%}**")
        if ps["years"] < 1.5:
            lines.append("> ⚠️ **诚实提示**: 测试区间较短, 下表 `annualized_return` 是把这段收益**直接年化外推**的数字, "
                         "会显著放大, **不代表可持续的年化能力**。请以区间绝对收益为准, "
                         "并结合集中持仓、小股票池的过拟合风险解读。")
        lines.append("")

    lines.append("## 关键指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    short = ps is not None and ps["years"] < 1.5
    for k, v in metrics.items():
        note = " ⚠️短区间年化外推" if (k == "annualized_return" and short) else ""
        lines.append(f"| {k}{note} | {v} |")

    if report is not None and not report.empty:
        lines.append("\n## 净值曲线(采样)\n")
        lines.append("| 日期 | 账户净值 | 基准 |")
        lines.append("|---|---|---|")
        sample = report.iloc[::max(1, len(report) // 30)]
        for date, row in sample.iterrows():
            lines.append(f"| {date} | {row.get('account', '')} | {row.get('bench', '')} |")
    return "\n".join(lines)


def print_report(metrics: dict, report: Optional[pd.DataFrame] = None, cfg: Optional[dict] = None):
    """打印报告到控制台。"""
    md = to_markdown(metrics, report, cfg)
    print(md)


def save_report(metrics: dict, report: pd.DataFrame, cache_dir: Path,
                name: str = "backtest", cfg: Optional[dict] = None) -> Path:
    """保存报告(markdown + csv净值)到 cache_dir。"""
    ensure_dir(cache_dir)
    md_path = cache_dir / f"{name}_report.md"
    md_path.write_text(to_markdown(metrics, report, cfg), encoding="utf-8")
    if report is not None and not report.empty:
        csv_path = cache_dir / f"{name}_netvalue.csv"
        report.to_csv(csv_path)
    log.info(f"报告已保存: {md_path}")
    return md_path

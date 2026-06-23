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


def to_markdown(metrics: dict, report: Optional[pd.DataFrame] = None) -> str:
    """生成 markdown 报告。"""
    lines = ["# 回测报告\n"]
    lines.append("## 关键指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    if report is not None and not report.empty:
        lines.append("\n## 净值曲线(采样)\n")
        lines.append("| 日期 | 账户净值 | 基准 |")
        lines.append("|---|---|---|")
        sample = report.iloc[::max(1, len(report) // 30)]
        for date, row in sample.iterrows():
            lines.append(f"| {date} | {row.get('account', '')} | {row.get('bench', '')} |")
    return "\n".join(lines)


def print_report(metrics: dict, report: Optional[pd.DataFrame] = None):
    """打印报告到控制台。"""
    md = to_markdown(metrics, report)
    print(md)


def save_report(metrics: dict, report: pd.DataFrame, cache_dir: Path, name: str = "backtest") -> Path:
    """保存报告(markdown + csv净值)到 cache_dir。"""
    ensure_dir(cache_dir)
    md_path = cache_dir / f"{name}_report.md"
    md_path.write_text(to_markdown(metrics, report), encoding="utf-8")
    if report is not None and not report.empty:
        csv_path = cache_dir / f"{name}_netvalue.csv"
        report.to_csv(csv_path)
    log.info(f"报告已保存: {md_path}")
    return md_path

"""通用工具: 配置加载、代码格式转换、日志、交易日历。"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# A股代码 -> qlib 代码 (SH/SZ/BJ 前缀)
# 6开头: 上交所 SH (600/601/603/605/688科创板)
# 0开头: 深交所主板 SZ (000/001/002)
# 3开头: 深交所创业板 SZ (300/301)
# 8/4开头: 北交所 BJ
_EXCHANGE_PREFIX = {
    "6": "SH",
    "0": "SZ",
    "3": "SZ",
    "8": "BJ",
    "4": "BJ",
}


def to_qlib_code(code: str) -> str:
    """6位A股代码 -> qlib代码, 如 '000001' -> 'SZ000001'。"""
    code = str(code).strip().zfill(6)
    if code[0] not in _EXCHANGE_PREFIX:
        raise ValueError(f"无法识别的A股代码: {code}")
    return f"{_EXCHANGE_PREFIX[code[0]]}{code}"


def to_raw_code(qlib_code: str) -> str:
    """qlib代码 -> 6位A股代码, 如 'SZ000001' -> '000001'。"""
    return qlib_code[2:]


def load_config(path: Optional[str] = None) -> dict:
    """加载 YAML 配置, path 默认为 config/config.yaml。"""
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Optional[str] = None):
    """保存配置到 YAML。"""
    import yaml as _yaml
    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        _yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """配置日志: 同时输出到控制台和文件。"""
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

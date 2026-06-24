"""通用工具: 配置加载、代码格式转换、日志、交易日历。"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

import yaml


# ============== 路径机制: 源码模式 / 打包(frozen)模式双语义 ==============
# 打包后代码目录是只读的, config 与数据必须落到用户可写目录。
def _is_frozen() -> bool:
    """是否运行在 PyInstaller 打包环境。"""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _resource_root() -> Path:
    """只读资源根: config 模板等。打包后 = _MEIPASS; 源码 = 项目根。"""
    if _is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _data_root() -> Path:
    """可写、持久化数据根。打包后 = 用户数据目录; 源码 = 项目根(保持兼容)。"""
    if _is_frozen():
        from platformdirs import user_data_dir
        d = Path(user_data_dir("stock_agent", "StockAgent"))
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(__file__).resolve().parent


def resolve_data_path(p: Union[str, Path]) -> Path:
    """把配置里的相对路径解析为绝对路径, 相对 PROJECT_ROOT。"""
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp)


def ensure_user_config() -> None:
    """首次运行: 从打包模板把 config/*.yaml 拷贝到用户 config 目录(不覆盖已有)。"""
    cfg_dir = PROJECT_ROOT / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    tmpl_dir = RESOURCE_ROOT / "config_template"
    if tmpl_dir.exists():
        for f in tmpl_dir.glob("*.yaml"):
            if not (cfg_dir / f.name).exists():
                shutil.copy2(f, cfg_dir / f.name)


PROJECT_ROOT = _data_root()          # 写数据 / 读用户配置 -> 业务逻辑统一沿用此名
RESOURCE_ROOT = _resource_root()     # 读只读模板用
ensure_user_config()                 # 模块导入时确保配置就位
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

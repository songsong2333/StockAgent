# -*- mode: python ; coding: utf-8 -*-
"""A股量化系统 PyInstaller 打包配置 (onedir; Mac 额外产出 .app)。

用法:
  pip install pyinstaller platformdirs pywebview
  pyinstaller build/stock_agent.spec --noconfirm --windowed
  # Mac  -> dist/A股量化系统.app
  # Win  -> dist/A股量化系统/A股量化系统.exe

设计要点见 plan: collect_all 收重依赖, 显式 collect 本项目模块 (app.py 由
streamlit 以脚本形式 exec, 静态分析看不到其 import), Mac 补 libomp。
"""
import os
import sys

from PyInstaller.utils.hooks import (
    collect_all, collect_dynamic_libs, collect_submodules, copy_metadata,
)

APP_NAME = "A股量化系统"
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
# spec 在 build/ 下, 项目根 = SPECPATH(=build/) 的上一级; 所有路径用绝对路径, 避免
# PyInstaller 以 spec 目录解析相对路径导致找不到文件
ROOT = os.path.dirname(os.path.abspath(SPECPATH))
# 关键: 让 spec 求值期间的 collect_submodules 能 import 到本项目包。
# pyinstaller 控制台脚本执行 spec 时 cwd 不在 sys.path 上, 否则 collect_submodules
# 静默失败 -> collector/backtest/live/factor/strategy/scripts 都不会进 bundle
# (表现为打包后 app.py import strategy 报 ModuleNotFoundError)。
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

datas, binaries, hiddenimports = [], [], []


# ---- 重量级依赖: 全量收集 (数据 + 二进制 + 子模块) ----
for pkg in ("streamlit", "plotly", "pyarrow", "qlib", "lightgbm", "akshare", "webview"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ---- 动态库 ----
binaries += collect_dynamic_libs("lightgbm")
if IS_MAC:
    # lib_lightgbm.dylib 依赖 libomp.dylib (homebrew), PyInstaller 不会自动收
    for omp in ("/usr/local/opt/libomp/lib/libomp.dylib",
                "/opt/homebrew/opt/libomp/lib/libomp.dylib"):
        if os.path.exists(omp):
            binaries += [(omp, ".")]

# ---- 包元数据 (版本探测 / streamlit 内部 importlib.metadata) ----
for pkg in ("streamlit", "packaging", "pandas", "numpy", "pyqlib"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# ---- 关键 hidden imports ----
# streamlit 在打包后无 CLI, 靠子线程调 bootstrap.run 启动
hiddenimports += ["streamlit.web.bootstrap"]
# qlib 用 yaml 注册表 + importlib 动态加载, 补几个高频模块保险
hiddenimports += [
    "qlib.contrib.data.handler",
    "qlib.contrib.model.gbdt", "qlib.contrib.model.linear",
    "qlib.contrib.strategy.signal_strategy", "qlib.contrib.evaluate",
    "qlib.data.dataset.processor", "qlib.data.dataset",
]
# 本项目模块: app.py 由 streamlit 以脚本形式 exec, 静态分析看不到其 import,
# 必须显式收集, 否则打包后页面 import collector/backtest/live/factor/strategy 会失败
hiddenimports += ["common", "collector", "backtest", "live", "factor", "strategy", "scripts"]
for pkg in ("collector", "backtest", "live", "factor", "strategy", "scripts"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception as e:
        print(f"[spec] 警告: collect_submodules({pkg}) 失败: {e}")

# ---- pywebview 原生后端 ----
if IS_MAC:
    hiddenimports += ["webview.platforms.cocoa", "webview.platforms.qt"]
elif IS_WIN:
    hiddenimports += [
        "webview.platforms.winforms", "webview.platforms.edgechromium",
        "webview.platforms.mshtml", "clr_loader",
    ]

# ---- 应用资源 ----
# app.py 作为脚本由 streamlit 加载 -> 打进 _MEIPASS/app/ (desktop_app._app_file 据此定位)
datas += [(os.path.join(ROOT, "app", "app.py"), "app")]
# config 模板: 首次运行由 common.ensure_user_config 拷贝到用户数据目录
for f in ("config.yaml", "watchlist.yaml", "sector_leaders.yaml", "sector_stocks.yaml"):
    src = os.path.join(ROOT, "config", f)
    if os.path.exists(src):
        datas += [(src, "config_template")]

# 可选图标
icon = None
if IS_MAC and os.path.exists(os.path.join(ROOT, "build", "app.icns")):
    icon = os.path.join(ROOT, "build", "app.icns")
elif IS_WIN and os.path.exists(os.path.join(ROOT, "build", "app.ico")):
    icon = os.path.join(ROOT, "build", "app.ico")


a = Analysis(
    [os.path.join(ROOT, "app", "desktop_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    icon=icon,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
)

# Mac: 额外生成 .app bundle
if IS_MAC:
    BUNDLE(
        coll,
        name=APP_NAME + ".app",
        bundle_identifier="com.stockagent.desktop",
        icon=icon,
        info_plist={
            "CFBundleDisplayName": APP_NAME,
            "NSHighResolutionCapable": True,
            # 允许 WebKit 加载 http://localhost (pywebview 内嵌 Streamlit)
            "NSAppTransportSecurity": {"NSAllowsArbitraryLoads": True},
        },
    )

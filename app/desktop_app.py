"""A股量化交易系统 - 跨平台桌面入口 (pywebview + Streamlit)。

用原生窗口包裹 Streamlit: 双击启动、独立窗口、无浏览器/地址栏。

  python app/desktop_app.py             # 源码模式运行
  pyinstaller build/stock_agent.spec    # 打包成 .app (Mac) / .exe (Win)

架构 (子进程自重Exec):
  父进程: 选一个空闲端口 -> 派生子进程(自身副本, 带 --serve) -> 等端口就绪 ->
          webview 原生窗口显示该 URL; 关窗后终止子进程。
  子进程 (--serve): 在【自己的主线程】跑 streamlit (bootstrap.run 调用
          signal.signal(SIGTERM) 要求主线程, 不能放工作线程; 且端口必须用
          streamlit.config.set_option 设定, flag_options 的 server.port 不生效)。

父/子同构: 源码模式用 `python 本文件 --serve`, 打包模式用 `exe --serve`。
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List

# 让入口能导入项目根目录的模块 (打包模式模块已内置, 此行无害)
_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

# 子进程服务模式标志 (用项目前缀避免与其他库冲突)
SERVE_FLAG = "--stockagent-serve"


def _app_file() -> str:
    """定位 app.py: 打包模式从 _MEIPASS/app, 源码模式从同级目录。"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / "app" / "app.py")
    return str(Path(__file__).resolve().parent / "app.py")


def _server_cmd(port: int) -> List[str]:
    """构造子进程命令: 打包模式 = [exe, --serve, port]; 源码模式 = [python, 本文件, --serve, port]。"""
    exe = sys.executable
    if getattr(sys, "frozen", False):
        return [exe, SERVE_FLAG, str(port)]
    return [exe, str(Path(__file__).resolve()), SERVE_FLAG, str(port)]


def _free_port() -> int:
    """随机选一个可用端口, 避免多开冲突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(port: int, timeout: float = 45.0) -> bool:
    """轮询直到端口可连接 (Streamlit server 已监听)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _watch_parent() -> None:
    """子进程后台线程: 父进程一旦死亡 (本进程被 init/launchd 收养) 立即自杀,
    避免父进程被强制杀死时成为孤儿进程占用端口。覆盖 kill -9 / 崩溃等所有场景。"""
    import os
    import threading
    parent = os.getppid()

    def _loop():
        while True:
            time.sleep(2)
            try:
                if os.getppid() != parent:  # 父已死, 被收养
                    os._exit(0)
            except Exception:
                os._exit(0)

    threading.Thread(target=_loop, daemon=True).start()


def _run_server(port: int) -> None:
    """子进程入口: 在主线程启动 Streamlit 并阻塞。

    注意: 必须在主线程 (bootstrap.run 内部注册 SIGTERM 处理器);
    端口用 streamlit.config.set_option, flag_options 不生效。
    """
    _watch_parent()
    import os
    import streamlit.config as cfg
    import streamlit.file_util as _fu
    # 修正 frozen 模式下 static 目录定位: streamlit.file_util.get_static_dir() 依赖
    # __file__ 计算 .../streamlit/static, 打包后 __file__ 解析异常会导致 index.html 404
    # (webview 显示 404 而非应用)。强制指向 _MEIPASS/streamlit/static。
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        _static = os.path.join(sys._MEIPASS, "streamlit", "static")
        if os.path.exists(os.path.join(_static, "index.html")):
            _fu.get_static_dir = lambda _s=_static: _s
    cfg.set_option("server.port", port)
    cfg.set_option("server.headless", True)
    cfg.set_option("browser.gatherUsageStats", False)
    cfg.set_option("server.fileWatcherType", "none")
    # 关键: frozen 模式下 streamlit 检测不到正常安装会默认 developmentMode=True,
    # 导致 server.py 不注册静态文件路由 -> / 返回 Tornado 默认 404 (webview 白/404)。
    # 强制 False 让它正常托管打包进来的 index.html / 静态资源。
    cfg.set_option("global.developmentMode", False)
    # 开启脚本健康检查端点 /_stcore/script-health-check, 用于确认 app.py 能跑通
    cfg.set_option("server.scriptHealthCheckEnabled", True)
    from streamlit.web import bootstrap
    bootstrap.run(_app_file(), False, [], {})


def main() -> None:
    # ---- 子进程: 仅跑 Streamlit server ----
    if SERVE_FLAG in sys.argv:
        idx = sys.argv.index(SERVE_FLAG)
        _run_server(int(sys.argv[idx + 1]))
        return

    # ---- 父进程: 拉起子进程 server + 原生窗口 ----
    port = _free_port()
    child = subprocess.Popen(_server_cmd(port))
    url = f"http://localhost:{port}"
    try:
        if not _wait_ready(port):
            raise RuntimeError("Streamlit 服务未在超时内就绪")
        try:
            import webview
            webview.create_window(
                title="A股量化交易系统",
                url=url,
                width=1400,
                height=900,
                min_size=(1000, 700),
            )
            webview.start()  # 主线程阻塞; 关闭窗口后返回
            return  # 窗口关闭, 正常退出
        except Exception as e:
            # 原生窗口不可用 (如 Windows 缺 WebView2 runtime): 回退浏览器
            print(f"[desktop] 原生窗口不可用 ({e}); 回退浏览器: {url}", file=sys.stderr)
            webbrowser.open(url)
            child.wait()  # 阻塞主线程, 让子进程持续服务浏览器
            return
    except Exception as e:
        print(f"[desktop] 启动失败: {e}", file=sys.stderr)
        return
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()


if __name__ == "__main__":
    # Windows 打包模式下, 若依赖库用到 multiprocessing 需要这行保护
    import multiprocessing
    multiprocessing.freeze_support()
    main()

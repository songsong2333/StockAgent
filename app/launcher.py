"""Mac 应用启动器: 启动 Streamlit 服务并自动打开浏览器。

可被 py2app 打包成原生 .app, 也可直接运行:
  python app/launcher.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = str(Path(__file__).resolve().parent / "app.py")
DEFAULT_PORT = 8501


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _open_browser(port: int):
    time.sleep(2.5)
    webbrowser.open(f"http://localhost:{port}")


def main():
    port = int(os.environ.get("STREAMLIT_PORT", DEFAULT_PORT))
    # 若已在运行, 直接打开浏览器
    if _port_in_use(port):
        webbrowser.open(f"http://localhost:{port}")
        print(f"应用已在 http://localhost:{port} 运行, 已为你打开浏览器")
        return

    threading.Thread(target=_open_browser, args=(port,), daemon=True).start()
    print(f"启动 A股量化系统: http://localhost:{port}")
    cmd = [sys.executable, "-m", "streamlit", "run", APP_FILE,
           "--server.port", str(port),
           "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()

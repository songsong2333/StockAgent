"""对话入口 —— CLI REPL 与 app chat 页共用 run()。

history 只存自然语言 [user, assistant] 对(不存 tool 中间态), 节省上下文与 token。
"""
from __future__ import annotations

from common import setup_logger, load_config

log = setup_logger("agent.chat")


def reply(cfg: dict, message: str, history: list | None = None) -> dict:
    """单轮对话(给 app/CLI 共用)。返回 loop.run 的结果 {content, trace, no_key?, truncated?}。"""
    from agent.loop import run
    return run(cfg, message, history=history, purpose="chat")


def cli_repl() -> None:
    """终端交互: python -m agent.chat。"""
    cfg = load_config()
    print("🤖 经济研究助手 (quit 退出 | clear 清历史)")
    print("   可问: '电力ETF现在能买吗' / '厄尔尼诺对煤炭什么影响' / '港股阿里跟谁走'\n")
    history = []
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() == "clear":
            history = []
            print("(历史已清)\n")
            continue
        res = reply(cfg, q, history=history)
        print(f"\n助手: {res['content']}")
        if res.get("trace"):
            print(f"  🔧 调用工具: {', '.join(t['tool'] for t in res['trace'])}")
        if res.get("truncated"):
            print("  ⚠️ 达到最大推理轮数")
        # 只把自然语言对存进 history
        from agent.loop import history_pair
        history += history_pair(q, res["content"])
        print()
    print("再见。")


if __name__ == "__main__":
    cli_repl()

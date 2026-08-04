"""Agent tool-calling 循环单测 —— mock LLM, 无需真实 API key。

用假 chat() 返回可控的 tool_call 序列, 断言 loop: 执行正确工具、解析参数、
max_rounds 截断、无 key 降级。dispatch 走真实 agent.tools.dispatch(端到端)。
"""
import json

import agent.llm as allm
import agent.tools as atools
from agent import loop
from common import load_config


class _FakeFn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args, ensure_ascii=False)


class _FakeTC:
    def __init__(self, name, args, id_="call_1"):
        self.id = id_
        self.type = "function"
        self.function = _FakeFn(name, args)


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or None

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": t.id, "type": "function",
                                "function": {"name": t.function.name,
                                             "arguments": t.function.arguments}}
                               for t in self.tool_calls]
        return d


def _set_chat_sequence(seq, monkeypatch):
    """让 agent.llm.chat 依次返回 seq 中的 FakeMsg; has_key 恒 True。"""
    seq = list(seq)
    monkeypatch.setattr(allm, "has_key", lambda cfg: True)
    monkeypatch.setattr(allm, "chat", lambda *a, **k: seq.pop(0) if seq else _FakeMsg(content="(空)"))


def test_loop_calls_tool_then_terminates(monkeypatch):
    _set_chat_sequence([
        _FakeMsg(tool_calls=[_FakeTC("list_universe", {})]),
        _FakeMsg(content="共24个序列可用"),
    ], monkeypatch)
    cfg = load_config()
    res = loop.run(cfg, "有哪些序列?")
    assert res["content"] == "共24个序列可用"
    assert res["trace"] == [{"tool": "list_universe", "args": {}}]


def test_loop_parses_tool_args(monkeypatch):
    _set_chat_sequence([
        _FakeMsg(tool_calls=[_FakeTC("get_seasonality", {"code": "159611"})]),
        _FakeMsg(content="电力7月均值-0.7%"),
    ], monkeypatch)
    cfg = load_config()
    res = loop.run(cfg, "电力7月怎样")
    assert res["trace"][0]["tool"] == "get_seasonality"
    assert res["trace"][0]["args"] == {"code": "159611"}


def test_loop_max_rounds_truncates(monkeypatch):
    # 永远返回 tool_call → 应在 max_rounds 截断
    _set_chat_sequence([_FakeMsg(tool_calls=[_FakeTC("list_universe", {})])] * 100, monkeypatch)
    # dispatch stub 成快路径(max_rounds 测的是循环计数, 非真实 dispatch)
    monkeypatch.setattr(atools, "dispatch", lambda cfg, name, args: '{"ok":1}')
    cfg = load_config()
    cfg["agent"]["max_rounds"] = 3
    res = loop.run(cfg, "x")
    assert res.get("truncated") is True
    assert len(res["trace"]) == 3


def test_loop_no_key_degrades(monkeypatch):
    monkeypatch.setattr(allm, "has_key", lambda cfg: False)
    cfg = load_config()
    res = loop.run(cfg, "电力行情?")
    assert res.get("no_key") is True
    assert "ZHIPUAI_API_KEY" in res["content"]


def test_loop_history_is_passed_through(monkeypatch):
    captured = {}
    def _capture(cfg, messages, tools=None, purpose="chat", **kw):
        captured["msgs"] = messages
        return _FakeMsg(content="ok")
    monkeypatch.setattr(allm, "has_key", lambda cfg: True)
    monkeypatch.setattr(allm, "chat", _capture)
    cfg = load_config()
    hist = [{"role": "user", "content": "之前问的"},
            {"role": "assistant", "content": "之前答的"}]
    loop.run(cfg, "新问题", history=hist)
    roles = [m["role"] for m in captured["msgs"]]
    assert roles[0] == "system"           # system prompt 在最前
    assert {"role": "user", "content": "之前问的"} in captured["msgs"]
    assert {"role": "user", "content": "新问题"} in captured["msgs"]

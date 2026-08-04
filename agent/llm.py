"""GLM (智谱) 客户端封装 + 模型路由。

tool-calling 是 OpenAI 兼容的: client.chat.completions.create(model, messages, tools,
tool_choice="auto") → response.choices[0].message.tool_calls(OpenAI 格式)。

模型路由: 监控→ glm-4.5-air(便宜), 对话→ glm-4.6(agent 调优)。均 config 可配。
无 api_key 时抛 NoApiKeyError, 上层(chat/monitor)优雅降级。
"""
from __future__ import annotations

import os
from typing import Optional

from common import setup_logger, load_config

log = setup_logger("agent.llm")


class NoApiKeyError(RuntimeError):
    """未配置智谱 API key。"""


_client = None            # 进程级缓存(ZhipuAI 客户端线程安全)


def _api_key(cfg: dict) -> Optional[str]:
    env_name = cfg.get("agent", {}).get("api_key_env", "ZHIPUAI_API_KEY")
    return os.environ.get(env_name)


def get_client(cfg: dict):
    """懒加载 + 缓存 ZhipuAI 客户端。无 key 抛 NoApiKeyError。"""
    global _client
    if _client is not None:
        return _client
    key = _api_key(cfg)
    if not key:
        raise NoApiKeyError(
            f"未设置环境变量 {cfg.get('agent', {}).get('api_key_env', 'ZHIPUAI_API_KEY')}"
            f"(智谱 API key)。前往 open.bigmodel.cn 注册获取后 export 之。")
    from zhipuai import ZhipuAI
    base_url = cfg.get("agent", {}).get("base_url") or None
    _client = ZhipuAI(api_key=key) if not base_url else ZhipuAI(api_key=key, base_url=base_url)
    log.info("GLM 客户端就绪")
    return _client


def has_key(cfg: dict) -> bool:
    return bool(_api_key(cfg))


def model_for(cfg: dict, purpose: str = "chat") -> str:
    """按用途选模型: chat / monitor。"""
    a = cfg.get("agent", {})
    if purpose == "monitor":
        return a.get("model_monitor", "glm-4.5-air")
    return a.get("model_chat", "glm-4.6")


def chat(cfg: dict, messages: list, tools: Optional[list] = None,
         purpose: str = "chat", tool_choice: str = "auto", **kw):
    """调 GLM chat.completions.create(OpenAI 兼容)。无 key 抛 NoApiKeyError。

    返回 response.choices[0].message(含 .content / .tool_calls)。
    """
    client = get_client(cfg)
    model = model_for(cfg, purpose)
    create_kw = dict(model=model, messages=messages, tool_choice=tool_choice)
    if tools:
        create_kw["tools"] = tools
    create_kw.update(kw)
    log.debug(f"GLM 调用 model={model} msgs={len(messages)} tools={len(tools or [])}")
    resp = client.chat.completions.create(**create_kw)
    return resp.choices[0].message

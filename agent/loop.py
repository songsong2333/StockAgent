"""tool-calling 循环 + 诚实 system prompt。

GLM(OpenAI 兼容)标准往返: messages+tools → LLM → 若有 tool_calls 则执行+回填 →
直到无 tool_call → 终答。max_rounds 守卫防失控。

统计诚实(灵魂): system prompt 强制 agent 引 N/CI/FDR、标注相关非因果、小样本降级措辞、
提规律前先 query_knowledge 查重。tools 返回的 warning 字段不可丢。
无 api_key 时优雅降级(返回提示, 不抛)。
"""
from __future__ import annotations

import json
from typing import Optional

from common import setup_logger, load_config

log = setup_logger("agent.loop")

SYSTEM_PROMPT = """你是一个严谨的 A 股/宏观量化研究助手, 服务于一个放弃个股短线、专注 ETF/指数周月级行情的个人投资者。你有一组分析工具(季节性/事件窗口/相关性/lead-lag/regime/锚点event-study)和一个持续累积的"经济学规律"知识库。

【你的核心职责】用工具跑真实数据, 帮用户验证或否定其中期假设(如"电力6-9月有行情""厄尔尼诺影响煤炭""某港股跟随创业板")。

【统计诚实 —— 最高优先级, 不可妥协】
1. 任何结论必须带上样本量 N、置信区间 CI、胜率、FDR q 值等工具返回的统计量。N<5 的结论必须明确标注"样本不足, 仅作假设"。
2. 相关 ≠ 因果。lead-lag/Granger 是"预测先后", 不是"导致"。措辞要准确。
3. 固定月分桶会高估季节性(多重检验); 优先用 detect_event_windows 看窗口分布(起止 std 大=漂移)。
4. 提议规律(propose_law)前, 必须先 query_knowledge 查重; 只有在证据充分(N/CI/FDR 都达标)时才提议, 否则只作"假设"陈述, 不入库。
5. 回答前内部自检: 是否在小样本上过度解读? 是否丢了不确定性? 是否把相关说成了因果? 不达标则降级措辞。

【工具使用】
- 不确定有哪些 code 时先 list_universe。
- 回答尽量基于工具结果, 不要凭空断言。调够工具再回答。
- 工具返回的 warning 字段必须在转述中体现。

【风格】中文, 简洁有结论, 先给一句话判断再给证据+不确定性。不堆术语。"""


def _msg_to_dict(msg) -> dict:
    """把 GLM 返回的 assistant message(含 tool_calls)转成可回填的 dict。"""
    try:
        return msg.model_dump(exclude_none=True)
    except Exception:
        tc = getattr(msg, "tool_calls", None) or []
        return {"role": "assistant", "content": getattr(msg, "content", None),
                "tool_calls": [{"id": t.id, "type": "function",
                                "function": {"name": t.function.name,
                                             "arguments": t.function.arguments}} for t in tc]}


def run(cfg: dict, user_message: str, history: Optional[list] = None,
        purpose: str = "chat") -> dict:
    """跑一轮对话: user_message + history → 调工具 → 终答。

    history: 上一轮的 [user/assistant] 消息列表(只含自然语言, 不含 tool 中间态)。
    返回 {content: 终答文本, trace: [{tool,args}] , no_key/truncated: 标志}。
    """
    from agent.llm import chat, has_key
    from agent.tools import tools_schema, dispatch

    if not has_key(cfg):
        env = cfg.get("agent", {}).get("api_key_env", "ZHIPUAI_API_KEY")
        return {"content": f"⚠️ 未配置 {env}(智谱 API key)。请前往 open.bigmodel.cn 注册获取, "
                f"然后 `export {env}=你的key` 重启。配置后我就能调分析工具回答你了。",
                "trace": [], "no_key": True}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += list(history or [])
    messages.append({"role": "user", "content": user_message})

    schema = tools_schema()
    max_rounds = cfg.get("agent", {}).get("max_rounds", 8)
    trace = []
    last_content = None

    for rnd in range(max_rounds):
        msg = chat(cfg, messages, tools=schema, purpose=purpose)
        last_content = getattr(msg, "content", None)
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            return {"content": last_content, "trace": trace}
        # 有 tool_calls: 回填 assistant + 执行每个工具
        messages.append(_msg_to_dict(msg))
        for tc in tcs:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            log.info(f"round {rnd} tool: {name}({args})")
            result = dispatch(cfg, name, args)
            trace.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # 达到 max_rounds 仍未终止: 用最后一次 content + 提示
    log.warning(f"达到 max_rounds={max_rounds}, 强制终止")
    return {"content": (last_content or "(推理未完成)") + "\n\n⚠️ 达到最大推理轮数, 以上是基于已收集信息的回答。",
            "trace": trace, "truncated": True}


def history_pair(user_message: str, answer: str) -> list:
    """构造可追加到 history 的 [user, assistant] 对(只存自然语言)。"""
    return [{"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    cfg = load_config()
    # 无 key 时验证降级
    res = run(cfg, "你好, 电力ETF现在还能买吗?")
    print("no_key 降级:", res.get("no_key"))
    print(res["content"][:150])

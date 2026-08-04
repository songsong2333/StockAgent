"""经济研究 Agent 子系统 (GLM 大脑 + tool-calling + 规律知识库)。

包结构:
  llm.py       GLM 客户端 + 模型路由
  tools.py     工具注册表(包 strategy/* + 本知识库)
  loop.py      tool-calling 循环 + 诚实 system prompt + 自批驳
  knowledge.py 规律知识库 CRUD
  monitor.py   定时监控(变化扫描 → 摘要 → notify)
  chat.py      对话入口(app/CLI 共用)

agent 的 tools 复用已建好的 strategy/{seasonality,discovery,regime}.py, 不重写分析逻辑。
统计诚实是灵魂: 规律默认 tentative, 升 confirmed 需过门槛或人工。
"""

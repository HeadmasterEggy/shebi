# -*- coding: utf-8 -*-
"""电商评论洞察 Agent。

分层：
  registry.py     工具注册表——pydantic schema、JSON Schema 导出、参数校验与自修复
  tools.py        具体工具实现（本地情感模型、检索、方面统计、爬虫、训练、导出）
  llm.py          LLM provider 抽象（OpenAI 兼容协议，可切 DeepSeek / Qwen / Kimi）
  budget.py       步数 / token / 成本预算，含给单个角色的子预算
  trace.py        全链路执行轨迹（带角色标签）
  react.py        手写 ReAct 执行循环

  embedding.py    SIF 句向量，复用项目自己的 50 维预训练词向量，零 API 成本
  vectorstore.py  评论向量索引 + 字面/语义混合检索（RRF 融合）
  critic.py       确定性引用溯源校验——不调 LLM，产出可复算的无据结论率
  roles.py        Planner / Collector / Analyst / Reporter 的职责与最小工具授权
  supervisor.py   多 Agent 编排：Analyst ⇄ Critic 修订闭环
"""

__all__ = ["registry", "tools", "llm", "budget", "trace", "react",
           "embedding", "vectorstore", "critic", "roles", "supervisor"]

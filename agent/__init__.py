# -*- coding: utf-8 -*-
"""电商评论洞察 Agent。

分层：
  registry.py  工具注册表——pydantic schema、JSON Schema 导出、参数校验与自修复
  tools.py     具体工具实现（本地情感模型、评论检索、统计、爬虫、训练）
  llm.py       LLM provider 抽象（OpenAI 兼容协议，可切 DeepSeek / Qwen / Kimi）
  budget.py    步数 / token / 成本预算
  trace.py     全链路执行轨迹
  react.py     手写 ReAct 执行循环
"""

__all__ = ["registry", "tools", "llm", "budget", "trace", "react"]

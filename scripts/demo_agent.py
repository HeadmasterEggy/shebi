# -*- coding: utf-8 -*-
"""离线演示：用 ScriptedClient 驱动真实工具，跑通一次完整的 ReAct 循环。

剧本刻意包含一次参数越界，用来展示 schema 自修复真的会发生。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tools as _t
from agent.budget import Budget
from agent.llm import LLMResponse, ScriptedClient, ToolCall, Usage
from agent.react import ReActAgent

print("灌入评论库:", _t.bootstrap_store(), "条")

script = [
    # 1) 先检索差评线索 —— 故意把 k 写成 999（超过上限 50）
    LLMResponse(tool_calls=[ToolCall("c1", "search_reviews", {"query": "物流 慢 破损", "k": 999})],
                usage=Usage(1200, 40)),
    # 2) 收到 schema 报错后改对
    LLMResponse(tool_calls=[ToolCall("c1", "search_reviews", {"query": "物流 慢 破损", "k": 5})],
                usage=Usage(1500, 45)),
    # 3) 把检索到的原文批量丢给本地模型判极性
    LLMResponse(tool_calls=[ToolCall("c2", "classify_sentiment", {
        "texts": ["物流太慢了", "包装破损严重", "发货很快包装完好"]})],
                usage=Usage(1800, 60)),
    # 4) 想跑爬虫 —— 有副作用，应被拦下
    LLMResponse(tool_calls=[ToolCall("c3", "scrape_reviews",
                                     {"product_url": "https://item.jd.com/100008348542.html"})],
                usage=Usage(2000, 50)),
    # 5) 给出带引用的答案
    LLMResponse(content="差评集中在物流与包装两类问题上 [review_id: 1, 2]。"
                        "抓取新商品评论需要你授权后才能执行。", usage=Usage(2200, 90)),
]

agent = ReActAgent(ScriptedClient(script, model="gpt-4o-mini"), budget=Budget(max_steps=10))
res = agent.run("这批评论里差评主要集中在什么问题上？")

print("\n" + "=" * 66)
print("状态:", res.status)
print("答案:", res.answer)
print("=" * 66)
import json
print(json.dumps(res.trace.summary(), ensure_ascii=False, indent=2))
print("\n执行轨迹:")
for s in res.trace.steps:
    if s.kind == "llm":
        print(f"  [{s.step}] llm    tokens={s.prompt_tokens}+{s.completion_tokens} ${s.cost_usd:.6f}")
    elif s.kind == "tool":
        flag = "ok " if s.ok else "ERR"
        rep = "  ← 触发 schema 自修复" if s.repaired else ""
        print(f"  [{s.step}] tool   {flag} {s.tool} {json.dumps(s.tool_args, ensure_ascii=False)[:70]}{rep}")
        print(f"          → {(s.observation or '')[:110].splitlines()[0] if s.observation else ''}")
    else:
        print(f"  [{s.step}] {s.kind}")
print("\n轨迹文件:", res.trace.save())

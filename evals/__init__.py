# -*- coding: utf-8 -*-
"""评测。

L5。这一层是「做过 demo」和「会做 agent」的分界线——大部分候选人的 agent
项目到能跑通就停了，说不出自己那套东西到底有多准、多贵、多慢。

    critic_eval.py  Critic 抓幻觉的能力：分类型检出率 + 逐关卡消融
    tools_eval.py   工具层三维对比：本地模型 vs LLM zero-shot 的准确率 / 成本 / 延迟
    tasks.py        agent 任务集
    agent_eval.py   跑任务集，算完成率 / 工具调用正确率 / 引用准确率 / 成本
    report.py       出可直接贴进 README 的 Markdown 表

一条自律：**能离线算的绝不写成需要 API key 才能跑**。critic_eval 和
tools_eval 的本地半边不花一分钱就能复现，这样"无据结论率""准确率持平"
这些话在面试现场当场就能跑给人看。
"""

__all__ = ["critic_eval", "tools_eval", "agent_eval", "tasks", "report"]

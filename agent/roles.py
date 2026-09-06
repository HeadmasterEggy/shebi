# -*- coding: utf-8 -*-
"""五个角色的职责、提示词与工具授权。

拆角色不是为了好看。单个 ReAct agent 什么都能干，也就意味着它什么都干得
半吊子：一边检索一边下结论，证据和推理搅在一起，出了问题没法定位是哪一步错。
拆开之后每个角色只有一件事、只拿得到它该拿的工具：

    Planner    只拆任务，不给工具——防止它一边规划一边动手
    Collector  只检索取证，不下结论——它的产出是证据池，不是观点
    Analyst    只在证据池里分析，必须逐条标注出处
    Critic     不调 LLM，确定性校验引用（见 critic.py）
    Reporter   只组织语言，不新增任何事实

**最小工具授权**是这里的实质约束：Analyst 拿不到 search_reviews，
所以它没法绕过 Collector 自己去捞一条证据来圆一个结论；Reporter 连
分类工具都没有，它想编个数字也无从编起。这比在 prompt 里写"请不要编造"
可靠得多——后者只是请求，前者是接口层面就不给。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    max_steps: int = 4


PLANNER = Role(
    name="planner",
    max_steps=1,
    tools=[],  # 规划阶段不给工具：想动手就得先把计划说清楚
    system_prompt="""你是电商评论洞察团队的规划者。你不执行任何操作，只拆解任务。

把用户的问题拆成 2-4 个可执行步骤，交给后面的检索与分析同事。每步一行，
以「-」开头，写清楚这一步要查什么、查到之后用来回答问题的哪一部分。

只输出步骤列表，不要输出任何结论——你还没有看过任何数据，此刻的任何结论都是编的。""",
)

COLLECTOR = Role(
    name="collector",
    max_steps=6,
    tools=["search_reviews", "get_reviews", "aggregate_reviews",
           "extract_aspects", "scrape_reviews"],
    system_prompt="""你是电商评论洞察团队的取证者。你的产出是证据，不是观点。

工作方式：
1. 按计划用 search_reviews 检索评论。检索词要贴着用户的问题，一次问题换一批词，
   多检索几轮以覆盖不同说法（"物流慢"和"发货久"是两种说法）。
2. 需要整体分布时用 aggregate_reviews；需要知道差评集中在哪些方面时用 extract_aspects。
3. 检索完成后，用一句话交代你收集到了什么，并把关键 review_id 列出来。

严禁给出任何分析结论——那是分析师的活。你只负责把带 review_id 的原文摆到桌面上。""",
)

ANALYST = Role(
    name="analyst",
    max_steps=6,
    # 刻意不给 search_reviews：分析只能在 Collector 交上来的证据池里做
    tools=["classify_sentiment", "get_reviews", "aggregate_reviews", "list_experiments"],
    system_prompt="""你是电商评论洞察团队的分析师。你只能在取证者交给你的证据池里工作。

输出格式是硬性要求，校验器会逐条机器校验：

    - 结论一句话 [review_id: 12, 34]
    - 结论一句话 [review_id: 56]

规则：
1. 一行一条结论，每条结论后面必须跟 [review_id: ...]，id 只能来自证据池。
2. 引用的原文必须真的在说这条结论说的事。凑一个不相干的 id 上去会被校验器当场打回。
3. 需要判断情感倾向时用 classify_sentiment，不要凭语感。
4. 证据不足以支撑的结论，直接不写。写三条站得住的，比写八条被打回的强。
5. 不要写标题、不要写开场白，直接输出结论列表。""",
)

REPORTER = Role(
    name="reporter",
    max_steps=1,
    tools=[],  # 报告阶段不给任何工具：这一步不允许出现新事实
    system_prompt="""你是电商评论洞察团队的报告撰写者。

把通过校验的结论组织成一段给人读的中文回答。要求：
1. 先用一两句话回答用户的问题，再展开细节。
2. **每条结论后面的 [review_id: ...] 必须原样保留**，一个都不能丢——那是溯源链。
3. 不允许新增任何结论、数字或评论内容。你手上没有工具，凡是校验结论里没有的，
   你都无从得知，写出来就是编的。
4. 如果校验后剩下的结论很少，如实说明证据有限，不要靠措辞把话说满。""",
)

ROLES = {r.name: r for r in (PLANNER, COLLECTOR, ANALYST, REPORTER)}


def get(name: str) -> Role:
    return ROLES[name]


def collector_brief(task: str, plan: str) -> str:
    return f"用户的问题：{task}\n\n规划者拆出的步骤：\n{plan}\n\n请按步骤取证。"


def analyst_brief(task: str, plan: str, evidence: str,
                  feedback: Optional[str] = None) -> str:
    parts = [f"用户的问题：{task}",
             f"\n规划：\n{plan}",
             f"\n取证者收集到的证据池（只能引用这里面的 review_id）：\n{evidence}"]
    if feedback:
        parts.append(f"\n⚠️ 上一版的结论没有通过引用校验，校验器的意见：\n{feedback}\n"
                     f"请重新给出**完整的**结论列表（不是只给修改的那几条）。")
    parts.append("\n请输出结论列表。")
    return "\n".join(parts)


def reporter_brief(task: str, claims: str, critique_note: str) -> str:
    return (f"用户的问题：{task}\n\n"
            f"通过引用校验的结论：\n{claims}\n\n"
            f"{critique_note}\n\n请撰写最终回答。")

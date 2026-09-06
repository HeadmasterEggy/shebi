# -*- coding: utf-8 -*-
"""离线演示：Planner → Collector → Analyst ⇄ Critic → Reporter 全流程。

不需要 API key。剧本里的 Analyst 第一版**故意写坏**——编一个不存在的
review_id、给一条结论配一条极性相反的评论、再写一条完全没有出处的结论。
Critic 会把它们逐条挡下来，Analyst 拿着整改意见重写，最终报告里只剩
经得起点开核对的结论。

这个 demo 的意义在于：它把"无据结论率从 X% 降到 Y%"这句话变成了当场能跑
出来的两个数，而不是简历上一个查无实据的百分比。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import store, tools as _t                                 # noqa: E402
from agent.budget import Budget                                      # noqa: E402
from agent.critic import Critic                                      # noqa: E402
from agent.llm import LLMResponse, ScriptedClient, ToolCall, Usage   # noqa: E402
from agent.supervisor import Supervisor                              # noqa: E402

print("灌入评论库:", _t.bootstrap_store(), "条新评论（库内共",
      store.count(), "条）")

TASK = "这批评论里差评主要集中在什么问题上？"

script = [
    # ── Planner ──────────────────────────────────────────────
    LLMResponse(content="- 检索物流、包装、质量相关的负面评论\n"
                        "- 统计各方面的负面占比\n"
                        "- 给出有原文支撑的结论",
                usage=Usage(600, 40)),

    # ── Collector ────────────────────────────────────────────
    LLMResponse(tool_calls=[ToolCall("c1", "search_reviews",
                                     {"query": "物流慢 发货久", "k": 4, "product_id": "demo"})],
                usage=Usage(1400, 40)),
    LLMResponse(tool_calls=[ToolCall("c2", "search_reviews",
                                     {"query": "电池续航 充电", "k": 4, "product_id": "demo"})],
                usage=Usage(1700, 40)),
    LLMResponse(tool_calls=[ToolCall("c3", "extract_aspects", {"product_id": "demo", "min_mentions": 1})],
                usage=Usage(2000, 45)),
    LLMResponse(content="已检索物流与质量两类差评，并跑了一遍方面统计。",
                usage=Usage(2300, 30)),

    # ── Analyst 第一版：四条结论，后三条各踩一个不同的关卡 ────
    LLMResponse(content=(
        "- 差评主要集中在物流配送太慢 [review_id: 1]\n"
        "- 质量做工问题也被频繁提及 [review_id: 999999]\n"
        "- 用户普遍认为价格偏贵\n"
        "- 物流配送其实很快，用户评价积极 [review_id: 1]"),
        usage=Usage(3000, 120)),

    # ── Analyst 第二版：按整改意见重写 ────────────────────────
    LLMResponse(content=(
        "- 差评主要集中在物流配送太慢 [review_id: 1]\n"
        "- 电池续航不足是另一类集中抱怨 [review_id: 3]"),
        usage=Usage(3600, 90)),

    # ── Reporter ─────────────────────────────────────────────
    LLMResponse(content=(
        "这批评论的差评集中在两类问题上。首先是物流配送太慢 [review_id: 1]；"
        "其次是电池续航不足 [review_id: 3]。其余方面证据有限，不下结论。"),
        usage=Usage(1200, 80)),
]


def main() -> int:
    conn = store.get_conn()
    # 让剧本里的 id 1/2/3 指向内容可控的评论，demo 的输出才稳定可读
    seeded = store.fetch(limit=10, source="demo", conn=conn)
    if not seeded:
        store.add_reviews([
            {"source": "demo", "product_id": "demo",
             "text": "物流太慢了，等了一个星期才到，很失望"},
            {"source": "demo", "product_id": "demo",
             "text": "客服回复很及时，问题解决得很快"},
            {"source": "demo", "product_id": "demo",
             "text": "电池续航太差，一天要充三次电"},
            {"source": "demo", "product_id": "demo",
             "text": "外观很漂亮，手感也不错，颜值在线"},
            {"source": "demo", "product_id": "demo",
             "text": "发货速度慢得离谱，客服还一直催我确认收货"},
            {"source": "demo", "product_id": "demo",
             "text": "屏幕色彩鲜艳，看视频很舒服"},
            {"source": "demo", "product_id": "demo",
             "text": "充电特别慢，电量掉得也快，续航是硬伤"},
            {"source": "demo", "product_id": "demo",
             "text": "价格实惠，性价比挺高的，值这个价"},
        ], conn)
        seeded = store.fetch(limit=10, source="demo", conn=conn)
    ids = [r["id"] for r in seeded[:3]]
    remap = dict(zip((1, 2, 3), ids))
    for r in script:
        if r.content:
            for old, new in remap.items():
                r.content = r.content.replace(f"review_id: {old}]", f"review_id: {new}]")

    sup = Supervisor(
        client=ScriptedClient(script, model="gpt-4o-mini"),
        budget=Budget(max_steps=24, max_cost_usd=0.50),
        critic=Critic(conn=conn),
    )
    result = sup.run(TASK)
    board = result.blackboard

    print("\n" + "=" * 70)
    print("最终报告:\n" + (result.answer or "(无)"))
    print("=" * 70)

    print(f"\n证据池: {len(board.evidence)} 条带 review_id 的原文")
    print(f"修订轮数: {board.revisions}")
    for i, c in enumerate(board.critiques, 1):
        print(f"\n── 第 {i} 轮引用校验 ──  "
              f"有据 {c.grounded}/{c.total}，无据率 {c.unsupported_rate}%"
              f"{'  ✓ 通过' if c.passed else '  ✗ 打回'}")
        for v in c.verdicts:
            print(f"   {'✓' if v.grounded else '✗'} 「{v.claim[:32]}」"
                  f"{'  ← ' + '；'.join(v.reasons) if v.reasons else ''}")

    print("\n" + json.dumps(result.trace.summary(), ensure_ascii=False, indent=2))
    path = result.trace.save()
    print(f"\n轨迹文件: {path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())

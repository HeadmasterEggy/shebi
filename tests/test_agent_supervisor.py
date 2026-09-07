# -*- coding: utf-8 -*-
"""多 Agent 编排：角色授权、证据收割、Critic 修订闭环、预算。

全部用 ScriptedClient 离线跑——编排逻辑、预算、修订终止条件这些东西
不该依赖一次真实网络调用才能验证。
"""

import pytest

from agent import roles, store, tools, vectorstore
from agent.budget import Budget, BudgetExceeded
from agent.critic import Critic
from agent.llm import LLMResponse, ScriptedClient, ToolCall, Usage
from agent.supervisor import Supervisor, harvest_evidence
from tests.conftest import needs_w2v


@pytest.fixture
def db(tmp_path, monkeypatch):
    conn = store.connect(str(tmp_path / "reviews.db"))
    store.add_reviews([
        {"source": "t", "product_id": "p", "text": "物流太慢了，等了一个星期才到，很失望"},
        {"source": "t", "product_id": "p", "text": "快递第二天就送到了，速度很快"},
        {"source": "t", "product_id": "p", "text": "客服态度恶劣，退货流程繁琐"},
        {"source": "t", "product_id": "p", "text": "屏幕显示效果不错，色彩很鲜艳"},
    ], conn)
    monkeypatch.setattr(store, "_conn", conn)
    vectorstore.reset()
    yield conn
    vectorstore.reset()


def supervisor(script, db, **kwargs):
    kwargs.setdefault("budget", Budget(max_steps=24, max_cost_usd=1.0))
    return Supervisor(
        client=ScriptedClient(script, model="gpt-4o-mini"),
        critic=Critic(conn=db, check_polarity=False),
        **kwargs,
    )


def CLAIMS(body: str) -> str:
    """分析师的产出格式：推理在标签外，结论在标签内。

    测试走的是和线上一样的契约——用裸列表测编排，等于测了一个线上不存在的分支。
    """
    return f"我先看了一遍证据池。\n<结论>\n{body}\n</结论>"


PLAN = LLMResponse(content="- 检索物流相关差评\n- 给出有据结论", usage=Usage(100, 20))
COLLECT = [
    LLMResponse(tool_calls=[ToolCall("c1", "search_reviews",
                                     {"query": "物流慢", "k": 3, "mode": "lexical"})],
                usage=Usage(200, 20)),
    LLMResponse(content="已检索物流相关评论。", usage=Usage(300, 20)),
]


# ---------------- 角色授权 ----------------
def test_analyst_cannot_reach_the_search_tools():
    """最小工具授权是实质约束：Analyst 想给一个站不住的结论现找一条证据，
    接口层面就做不到。这比在 prompt 里写"请不要编造"可靠。"""
    assert "search_reviews" not in roles.ANALYST.tools
    assert "search_reviews" in roles.COLLECTOR.tools


def test_planner_and_reporter_get_no_tools_at_all():
    """规划阶段不许动手，报告阶段不许出现新事实。"""
    assert roles.PLANNER.tools == []
    assert roles.REPORTER.tools == []


def test_every_role_tool_actually_exists():
    """角色里写错一个工具名，要跑到那一步才炸。在注册表里核一遍便宜得多。"""
    for role in roles.ROLES.values():
        for name in role.tools:
            tools.registry.get(name)      # 不存在会抛 KeyError


def test_collector_is_not_allowed_to_conclude():
    assert "严禁给出任何分析结论" in roles.COLLECTOR.system_prompt


def test_analyst_only_gets_the_polarity_tool():
    """分析师原本还拿着 get_reviews / aggregate_reviews，实测它把这两个当检索
    工具反复重调，单任务烧掉 14 万 token。给它取数工具，就是邀请它重新取一遍。"""
    assert roles.ANALYST.tools == ["classify_sentiment"]
    assert "list_experiments" in roles.COLLECTOR.tools, "取数属于取证阶段"


def test_collector_findings_reach_the_analyst(db):
    """取证者跑出来的统计结果必须转交给分析师。

    这条锁的是一个真实踩过的坑：collector_note 和方面统计当时算完就丢了，
    分析师只拿到评论原文、拿不到任何数字，于是自己重新取一遍——
    token 翻五倍，还经常撞预算上限。
    """
    script = [PLAN,
              LLMResponse(tool_calls=[ToolCall("c1", "extract_aspects",
                                               {"product_id": "p", "min_mentions": 1})],
                          usage=Usage(200, 20)),
              LLMResponse(content="方面统计已跑完。", usage=Usage(300, 20)),
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(400, 40)),
              LLMResponse(content="物流配送慢 [review_id: 1]。", usage=Usage(500, 30))]
    sup = supervisor(script, db)
    result = sup.run("差评集中在哪")

    assert result.blackboard.stats, "extract_aspects 的结果没进黑板"
    # 分析师那一轮的 user 消息里必须同时出现统计结果和取证者的交代
    analyst_prompts = [
        m["content"] for call in sup.client.calls for m in call["messages"]
        if m.get("role") == "user" and "证据池" in (m.get("content") or "")
    ]
    assert analyst_prompts, "没有找到发给分析师的交底"
    brief = analyst_prompts[-1]
    assert "统计结果" in brief
    assert "extract_aspects" in brief
    assert "方面统计已跑完" in brief, "取证者的交代被丢掉了"


# ---------------- 证据收割 ----------------
def test_harvest_picks_up_review_ids_from_any_tool_shape():
    """认"带 review_id 和 text 的字典"这个约定，而不是为每个工具写一份解析。"""
    into = {}
    n = harvest_evidence(
        {"reviews": [{"review_id": 1, "text": "甲"}, {"review_id": 2, "text": "乙"}],
         "nested": {"found": [{"review_id": 3, "text": "丙"}]},
         "noise": {"review_id": 4}},        # 没有 text，不算证据
        into)
    assert n == 3
    assert into == {1: "甲", 2: "乙", 3: "丙"}


def test_evidence_comes_from_tool_output_not_from_what_the_model_says(db):
    """模型在自然语言里吹嘘"我检索到了 999 条"不影响证据池。"""
    script = [PLAN,
              LLMResponse(tool_calls=[ToolCall("c1", "search_reviews",
                                               {"query": "物流慢", "k": 2, "mode": "lexical"})],
                          usage=Usage(200, 20)),
              LLMResponse(content="我检索到了 999 条评论 [review_id: 12345]",
                          usage=Usage(300, 20)),
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(400, 30)),
              LLMResponse(content="物流配送慢 [review_id: 1]", usage=Usage(500, 30))]
    res = supervisor(script, db).run("差评集中在哪")
    assert 12345 not in res.blackboard.evidence
    assert set(res.blackboard.evidence) <= {1, 2, 3, 4}


# ---------------- Critic 修订闭环 ----------------
def test_bad_draft_is_sent_back_and_the_rewrite_is_accepted(db):
    script = [PLAN, *COLLECT,
              # 第一版：一个编造的 id、一条没出处的结论
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 999999]\n"
                                     "- 用户普遍觉得价格偏贵，反复被提到"),
                          usage=Usage(400, 40)),
              # 第二版：改对
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(500, 30)),
              LLMResponse(content="差评集中在物流配送慢 [review_id: 1]。", usage=Usage(600, 30))]
    res = supervisor(script, db).run("差评集中在哪")

    assert res.ok
    assert res.blackboard.revisions == 1, "应当只重做一轮"
    first, last = res.blackboard.critiques[0], res.blackboard.critiques[-1]
    assert first.unsupported_rate == 100.0
    assert last.unsupported_rate == 0.0
    assert "999999" not in (res.answer or "")


def test_revision_stops_at_the_cap_and_ships_what_passed(db):
    """Analyst 改不动就停，把通过校验的部分交出去——这比无限重试更像工程。"""
    bad = LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 999999]\n"
                                     "- 客服服务差 [review_id: 3]"), usage=Usage(400, 40))
    script = [PLAN, *COLLECT, bad, bad, bad,
              LLMResponse(content="客服服务差 [review_id: 3]。", usage=Usage(600, 30))]
    res = supervisor(script, db, max_revisions=2).run("差评集中在哪")

    assert res.blackboard.revisions == 2
    assert len(res.blackboard.critiques) == 3
    assert res.critique.passed is False
    assert "3" in (res.answer or ""), "通过校验的那条应当仍然交付"


def test_only_verified_citations_survive_into_the_report(db):
    """一条结论引了两个 id、其中一个不切题时，交给 Reporter 的版本里
    不该再带着那一个——否则读者点进去看到的是不相干的评论。"""
    script = [PLAN, *COLLECT,
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1, 4]"), usage=Usage(400, 40)),
              LLMResponse(content="物流配送慢 [review_id: 1]。", usage=Usage(600, 30))]
    res = supervisor(script, db).run("差评集中在哪")
    claims = res.blackboard.grounded_claims()
    assert "[review_id: 1]" in claims
    assert "4" not in claims.split("[review_id:")[1]


def test_no_evidence_yields_an_honest_refusal_not_a_confident_answer(db):
    script = [PLAN,
              LLMResponse(content="什么都没检索到。", usage=Usage(200, 20)),
              LLMResponse(content=CLAIMS("- 这个商品口碑很好，用户都很满意"), usage=Usage(400, 40)),
              LLMResponse(content=CLAIMS("- 这个商品口碑很好，用户都很满意"), usage=Usage(450, 40)),
              LLMResponse(content=CLAIMS("- 这个商品口碑很好，用户都很满意"), usage=Usage(500, 40))]
    res = supervisor(script, db).run("口碑如何")
    assert "证据不足" in (res.answer or "")


def test_reporter_dropping_citations_falls_back_to_the_claim_list(db):
    """Reporter 把溯源链弄丢了就不采纳它的版本——宁可给一份朴素但可溯源的列表。"""
    script = [PLAN, *COLLECT,
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(400, 40)),
              LLMResponse(content="这批评论的差评主要集中在物流上。", usage=Usage(600, 30))]
    res = supervisor(script, db).run("差评集中在哪")
    assert "[review_id: 1]" in (res.answer or "")


# ---------------- 轨迹与预算 ----------------
def test_trace_labels_every_step_with_its_role(db):
    script = [PLAN, *COLLECT,
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(400, 40)),
              LLMResponse(content="物流配送慢 [review_id: 1]。", usage=Usage(600, 30))]
    res = supervisor(script, db).run("差评集中在哪")
    summary = res.trace.summary()
    assert summary["roles"] == ["planner", "collector", "analyst", "critic", "reporter"]
    assert summary["unsupported_rate_first"] == 0.0
    assert summary["claims_grounded"] == 1
    assert summary["evidence_pool"] >= 1


def test_role_budgets_roll_up_into_the_global_step_count(db):
    """只记子预算的话全局上限会形同虚设：每个角色都"没超"，加起来早就超了。"""
    script = [PLAN, *COLLECT,
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(400, 40)),
              LLMResponse(content="物流配送慢 [review_id: 1]。", usage=Usage(600, 30))]
    sup = supervisor(script, db)
    res = sup.run("差评集中在哪")
    assert sup.budget.steps == len([s for s in res.trace.steps if s.kind == "llm"])
    assert res.trace.summary()["total_tokens"] > 0


def test_global_step_budget_terminates_the_pipeline(db):
    """步数烧穿时优雅收尾，交回已有的中间结果，而不是抛异常丢掉全部工作。"""
    script = [PLAN] + [COLLECT[0]] * 10
    res = supervisor(script, db, budget=Budget(max_steps=3, max_cost_usd=1.0)).run("差评集中在哪")
    assert res.status == "budget_exceeded"
    assert res.answer and "预算" in res.answer


def test_scoped_budget_caps_a_single_role(db):
    """某个角色在原地打转时，它自己先停，不会把后面角色的额度全吃掉。"""
    parent = Budget(max_steps=50)
    scoped = parent.scoped(2)
    scoped.start_step()
    scoped.start_step()
    with pytest.raises(BudgetExceeded):
        scoped.start_step()
    assert parent.steps == 2, "子预算走过的每一步都要记进总账"


@needs_w2v
def test_end_to_end_with_hybrid_retrieval(db):
    """默认走混合检索的完整一趟，确认 L4 和 L3 接得上。"""
    script = [PLAN,
              LLMResponse(tool_calls=[ToolCall("c1", "search_reviews",
                                               {"query": "物流慢 发货久", "k": 3})],
                          usage=Usage(200, 20)),
              LLMResponse(content="检索完成。", usage=Usage(300, 20)),
              LLMResponse(content=CLAIMS("- 物流配送慢 [review_id: 1]"), usage=Usage(400, 40)),
              LLMResponse(content="差评集中在物流配送慢 [review_id: 1]。", usage=Usage(600, 30))]
    res = supervisor(script, db).run("差评集中在哪")
    assert res.ok and res.critique.passed
    assert 1 in res.blackboard.evidence

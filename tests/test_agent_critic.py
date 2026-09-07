# -*- coding: utf-8 -*-
"""Critic：四道关卡各自都要真的会拦人。

这些用例锁的是「幻觉能不能混进最终报告」这条底线，所以每一关都用一个
只违反那一关的样本来测——某一关退化成恒返回 True 时，只有对应的用例会红。
"""

import pytest

from agent import store
from agent.critic import Critic, extract_ids, parse_claims
from tests.conftest import needs_w2v, needs_weights


@pytest.fixture
def db(tmp_path):
    conn = store.connect(str(tmp_path / "reviews.db"))
    store.add_reviews([
        {"source": "t", "product_id": "p", "text": "物流太慢了，等了一个星期才到，很失望"},
        {"source": "t", "product_id": "p", "text": "快递第二天就送到了，速度很快"},
        {"source": "t", "product_id": "p", "text": "屏幕显示效果不错，色彩很鲜艳"},
        {"source": "t", "product_id": "p", "text": "客服态度恶劣，退货流程繁琐"},
    ], conn)
    return conn


@pytest.fixture
def critic(db):
    """默认体质：关掉需要本地模型的极性关卡，其余关卡照常。"""
    return Critic(conn=db, check_polarity=False)


# ---------------- 解析 ----------------
def test_extract_ids_handles_chinese_punctuation():
    assert extract_ids("结论 [review_id: 1, 2]") == [1, 2]
    assert extract_ids("结论 [review_id：3、4]") == [3, 4]
    assert extract_ids("结论 [review_ids: 5,6]") == [5, 6]
    assert extract_ids("没有引用") == []


def test_parse_claims_ignores_headings_but_keeps_real_claims():
    """标题不该被算进分母——否则无据结论率会被格式噪音稀释。"""
    claims = parse_claims("差评分析：\n- 物流慢是主要问题 [review_id: 1]\n\n- 另一条没有出处的长结论在这里")
    assert [c.text for c in claims] == ["物流慢是主要问题", "另一条没有出处的长结论在这里"]
    assert claims[0].review_ids == [1]
    assert claims[1].review_ids == []


# ---------------- 关卡 1：有没有引用 ----------------
def test_claim_without_citation_is_ungrounded(critic):
    rep = critic.review("- 这个商品的整体口碑相当不错")
    assert rep.total == 1
    assert rep.grounded == 0
    assert "没有任何引用" in rep.verdicts[0].reasons[0]


# ---------------- 关卡 2：id 是否存在 ----------------
def test_fabricated_review_id_is_caught(critic):
    """编一个不存在的 id 是最典型的幻觉，必须当场露馅。"""
    rep = critic.review("- 质量问题被频繁提及 [review_id: 999999]")
    assert rep.grounded == 0
    assert rep.verdicts[0].missing_ids == [999999]
    assert rep.unsupported_rate == 100.0


def test_partially_fabricated_citation_is_still_ungrounded(critic):
    """真 id 混一个假 id，不能因为"有一条是真的"就放行。"""
    rep = critic.review("- 物流配送慢 [review_id: 1, 999999]")
    assert rep.grounded == 0
    assert rep.verdicts[0].missing_ids == [999999]


# ---------------- 关卡 3：是否在证据池内 ----------------
def test_citation_outside_the_evidence_pool_is_rejected(critic):
    rep = critic.review("- 客服态度差 [review_id: 4]", evidence_ids=[1, 2])
    assert rep.grounded == 0
    assert rep.verdicts[0].out_of_scope_ids == [4]


def test_scope_check_can_be_disabled(db):
    rep = Critic(conn=db, check_scope=False, check_polarity=False).review(
        "- 客服态度差 [review_id: 4]", evidence_ids=[1, 2])
    assert rep.grounded == 1


# ---------------- 关卡 4b：方面一致 ----------------
def test_citation_about_a_different_aspect_is_rejected(critic):
    """结论谈物流，却引一条只谈屏幕的评论——这是最难抓的一种幻觉：
    id 真实存在、原文也确实是条评论，但跟结论根本不相干。"""
    rep = critic.review("- 差评集中在物流配送 [review_id: 3]", evidence_ids=[1, 2, 3, 4])
    assert rep.grounded == 0
    assert "并未提及" in "".join(rep.verdicts[0].reasons)


def test_matching_aspect_passes(critic):
    rep = critic.review("- 差评集中在物流配送 [review_id: 1]", evidence_ids=[1, 2, 3, 4])
    assert rep.grounded == 1
    assert rep.verdicts[0].supporting_ids == [1]
    assert rep.passed


# ---------------- 关卡 4a：极性一致（要本地模型） ----------------
@needs_w2v
@needs_weights
def test_polarity_mismatch_is_rejected_by_the_local_model(db):
    """结论说"好评"，被引评论却被本地模型判为消极。

    这一关用的正是毕设自训练的那个模型——它在这里从"被展示的成果"
    变成了"系统内部的质检工序"。
    """
    critic = Critic(conn=db, check_polarity=True)
    rep = critic.review("- 物流配送获得了大量好评 [review_id: 1]", evidence_ids=[1])
    assert rep.grounded == 0
    assert "断言" in "".join(rep.verdicts[0].reasons)


@needs_w2v
@needs_weights
def test_polarity_agreement_passes(db):
    critic = Critic(conn=db, check_polarity=True)
    rep = critic.review("- 物流配送太慢，差评集中 [review_id: 1]", evidence_ids=[1])
    assert rep.grounded == 1


# ---------------- 统计类结论：引用工具产出 ----------------
def test_statistic_claim_may_cite_a_tool_instead_of_a_review(critic):
    """「模型准确率是多少」这类问题，答案来自工具返回值，没有哪条评论能支撑它。

    只认 review_id 的话，agent 查到了正确答案也会被判成"证据不足"——
    首轮实跑 40 条任务里有 4 条栽在这个结构性缺口上。
    """
    rep = critic.review("- 当前默认模型测试集准确率 89.49% [source: list_experiments]",
                        available_sources=["list_experiments"])
    assert rep.grounded == 1 and rep.passed
    assert rep.verdicts[0].sources == ["list_experiments"]


def test_citing_a_tool_that_was_never_called_is_rejected(critic):
    """能引工具，不等于能编工具名——只能引本轮真调用过的。"""
    rep = critic.review("- 模型准确率 99.9% [source: list_experiments]",
                        available_sources=["search_reviews"])
    assert rep.grounded == 0
    assert "没有调用过" in "".join(rep.verdicts[0].reasons)


def test_tool_citation_is_not_accepted_when_no_sources_are_declared(critic):
    """调用方没声明可用工具时，工具引用一律不认——默认保守。"""
    rep = critic.review("- 模型准确率 89.49% [source: list_experiments]")
    assert rep.grounded == 0


# ---------------- 汇总指标 ----------------
def test_unsupported_rate_and_feedback(critic):
    rep = critic.review(
        "- 物流配送慢 [review_id: 1]\n"
        "- 质量有问题 [review_id: 999999]\n"
        "- 一条完全没有出处的结论写在这里",
        evidence_ids=[1, 2, 3, 4])
    assert rep.total == 3 and rep.grounded == 1
    assert rep.unsupported_rate == 66.67
    fb = rep.feedback()
    assert "999999" in fb and "没有任何引用" in fb
    assert "物流配送慢" not in fb, "通过校验的结论不该出现在整改意见里"


def test_empty_report_scores_zero_not_perfect(critic):
    """一条结论都不写不能算满分，否则模型会学会交白卷。"""
    rep = critic.review("")
    assert rep.total == 0
    assert rep.unsupported_rate == 100.0
    assert rep.passed is False


def test_gates_are_reported_for_reproducibility(critic):
    assert "id 存在" in critic.gates()
    assert any("语义" in g for g in critic.gates()) is False, \
        "语义关卡实测判别力不足，默认必须是关闭的"

# -*- coding: utf-8 -*-
"""评测层：对照集构造、指标口径、路径安全。

评测代码自己出错是最难发现的一类问题——它不会崩，只会给出一个看起来
很合理的数字。所以这里测的重点不是"能跑通"，而是**口径是不是对的**：
拒答题有没有被算进完成率、没有期望工具的任务有没有被算进工具正确率、
空结果会不会被当成满分。
"""

import pytest

from agent import store, tools  # noqa: F401 —— import 即完成工具注册
from evals import agent_eval, critic_eval, report
from evals.tasks import TASKS
from tests.conftest import needs_w2v, needs_weights


# ---------------- 任务集本身 ----------------
def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_task_set_contains_refusal_cases():
    """只有正例的评测集等于没做——一个"什么都答"的 agent 能拿满分。"""
    refusals = [t for t in TASKS if t.must_refuse]
    assert len(refusals) >= 8
    assert all(not t.expect_tools for t in refusals), "拒答题不该有期望工具"


def test_answerable_tasks_declare_expected_tools():
    for t in TASKS:
        if not t.must_refuse:
            assert t.expect_tools, f"{t.id} 缺少 expect_tools"


# ---------------- 指标口径 ----------------
def _run(task, **kw):
    r = agent_eval.TaskRun(task=task)
    r.status = kw.get("status", "completed")
    r.answer = kw.get("answer", "有结论 [review_id: 1]")
    r.refused = agent_eval.looks_like_refusal(r.answer)
    r.tools_used = kw.get("tools_used", list(task.expect_tools[:1]))
    return r


def test_refusal_task_passes_only_when_it_actually_refuses():
    refusal = next(t for t in TASKS if t.must_refuse)
    assert _run(refusal, answer="证据不足，评论里没有这个信息").passed is True
    assert _run(refusal, answer="售后电话是 400-123-4567").passed is False


def test_answerable_task_fails_when_it_refuses():
    ask = next(t for t in TASKS if not t.must_refuse)
    assert _run(ask, answer="物流慢是主要问题 [review_id: 1]").passed is True
    assert _run(ask, answer="证据不足，无法回答").passed is False


@pytest.mark.parametrize("answer", [
    "证据不足：没有检索到相关评论",
    "关于成本价，目前无法给出任何估算，证据严重不足",   # 首轮漏判的那条
    "评论中没有提及该信息",
    "查不到相关记录",
])
def test_refusal_detector_covers_the_phrasings_it_missed(answer):
    """关键词匹配会漏。首轮就漏了"证据严重不足"——中间插了两个字，
    恰好不含"证据不足"这个连续子串。"""
    assert agent_eval.looks_like_refusal(answer)


def test_refusal_detector_does_not_fire_on_a_real_answer():
    assert not agent_eval.looks_like_refusal(
        "差评主要集中在物流配送太慢 [review_id: 1, 2]")


def test_tool_accuracy_excludes_tasks_with_no_expected_tools():
    """拒答题没有期望工具，把它算进分母会凭空拉低（或抬高）工具正确率。"""
    refusal = next(t for t in TASKS if t.must_refuse)
    assert _run(refusal).tool_ok is None

    ask = next(t for t in TASKS if not t.must_refuse)
    assert _run(ask, tools_used=[ask.expect_tools[0]]).tool_ok is True
    assert _run(ask, tools_used=["train_model"]).tool_ok is False


def test_aggregate_reports_the_two_pass_rates_separately():
    """完成率和拒答正确率合成一个数的话，"什么都答"和"什么都不答"
    两种坏法会拿到同样的分。"""
    ask = next(t for t in TASKS if not t.must_refuse)
    refusal = next(t for t in TASKS if t.must_refuse)
    summary = agent_eval.aggregate([
        _run(ask),                                        # 答对
        _run(ask, answer="证据不足"),                      # 该答却拒答
        _run(refusal, answer="证据不足，评论里没有"),        # 该拒答且拒答
        _run(refusal, answer="在杭州仓发货"),               # 该拒答却编了
    ])
    assert summary["任务完成率"] == 50.0
    assert summary["拒答正确率"] == 50.0


def test_infrastructure_failures_are_excluded_from_quality_metrics():
    """一次网络抖动让 25 条任务没跑成时，把它们算进完成率的分母，得到的不是
    "agent 只完成了 37%"，而是"评测那天网不好"。两件事必须分开报。"""
    ask = next(t for t in TASKS if not t.must_refuse)
    good = [_run(ask) for _ in range(3)]
    dead = [_run(ask, status="error", answer="") for _ in range(7)]
    summary = agent_eval.aggregate(good + dead)

    assert summary["执行失败"] == 7
    assert summary["执行成功"] == 3
    assert summary["任务完成率"] == 100.0, "质量指标只该在跑成的任务上算"
    assert len(summary["失败任务"]) == 7


def test_cost_and_latency_still_count_failed_runs():
    """失败的任务也真的花了钱、占了时间，成本和延迟按全部任务算。"""
    ask = next(t for t in TASKS if not t.must_refuse)
    a, b = _run(ask), _run(ask, status="error", answer="")
    a.cost_usd, b.cost_usd = 0.02, 0.01
    a.latency_ms, b.latency_ms = 1000.0, 3000.0
    summary = agent_eval.aggregate([a, b])
    assert summary["总成本 USD"] == 0.03
    assert summary["p95 延迟 ms"] == 3000.0


def test_failed_run_never_counts_as_passed():
    ask = next(t for t in TASKS if not t.must_refuse)
    assert _run(ask, status="budget_exceeded").passed is False
    assert _run(ask, answer="").passed is False


# ---------------- Critic 对照集 ----------------
@needs_w2v
def test_case_builder_produces_all_hallucination_kinds():
    if store.count() == 0:
        tools.bootstrap_store(build_index=False)
    cases = critic_eval.build_cases(per_kind=4)
    kinds = {c.kind for c in cases}
    assert kinds == set(critic_eval.KINDS)
    # 负例必须真的是负例，否则整张表都在自我安慰
    assert all(not c.should_be_grounded for c in cases if c.kind != "grounded")


@needs_w2v
def test_fabricated_ids_really_do_not_exist():
    if store.count() == 0:
        tools.bootstrap_store(build_index=False)
    for case in critic_eval.build_cases(per_kind=4):
        if case.kind == "fabricated_id":
            assert store.get(case.review_ids[0]) is None


@needs_w2v
@needs_weights
def test_critic_eval_detects_polarity_hallucination():
    """这是整份评测里唯一非自证的一格，掉下来就该查。"""
    if store.count() == 0:
        tools.bootstrap_store(build_index=False)
    cases = critic_eval.build_cases(per_kind=8)
    result = critic_eval.evaluate(cases)
    assert result.rate("fabricated_id") == 100.0
    assert result.rate("no_citation") == 100.0
    assert result.rate("wrong_polarity") >= 60.0, "极性关卡失效了"


# ---------------- 报告渲染 ----------------
def test_report_renders_missing_values_as_dash_not_zero():
    """没测和测出来是 0，是两件事。渲染成 0 会让人以为跑过了。"""
    md = report.tools_md({"samples": 10, "batch_size": 20, "rows": [
        {"name": "LLM", "accuracy": None, "p50_ms": None, "p95_ms": None,
         "cost_per_1k": 0.0, "unparsed": 0}], "note": None})
    assert "—" in md
    assert "| LLM | — |" in md


def test_agent_report_says_so_when_not_run():
    md = report.agent_md(None)
    assert "未运行" in md and "AGENT_API_KEY" in md

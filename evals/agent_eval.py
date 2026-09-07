# -*- coding: utf-8 -*-
"""Agent 层评测：跑任务集，算四类指标。

指标是分开报的，不合成一个"总分"：

    任务完成率    非拒答题里产出了可用答案的比例
    拒答正确率    拒答题里确实拒答的比例——把这两个混进一个数，
                  一个"什么都答"的 agent 和一个"什么都不答"的 agent
                  会得到一样的分，而它们坏得完全不同
    工具调用正确率 该用的能力用上了没有
    引用准确率    1 − 无据结论率，由 Critic 确定性算出（见 agent/critic.py）

外加成本侧：平均步数 / token / 美元，以及 p50 / p95 延迟。agent 的工程
约束意识就体现在肯不肯报后面这几个数。

要 API key 才能跑。没有 key 时明确报错，不用 ScriptedClient 伪造一份
好看的结果——那样的数字对任何人都没有意义。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.budget import Budget
from agent.llm import LLMClient
from agent.supervisor import Supervisor
from evals.tasks import TASKS, Task

logger = logging.getLogger(__name__)

# 判定"这是一次拒答"的措辞。Reporter 的 prompt 要求证据不足时如实说明，
# Supervisor 在证据为空时也会走 _no_evidence_answer，两条路都落在这些词上。
#
# 关键词匹配是启发式的，会漏。首轮实跑就漏了一条：模型答的是"无法给出任何估算…
# 证据严重不足"，语义上是标准的拒答，但"证据严重不足"里插了两个字，
# 恰好不含"证据不足"这个连续子串，于是被判成"没有拒答"。
# 下面的词表按那次漏判补过；这仍是启发式，边界样例要人工复核。
REFUSAL_MARKERS = ("证据不足", "证据严重不足", "无法回答", "没有相关", "无法从评论",
                   "不下结论", "无法确定", "评论中没有", "没有足够", "不足以",
                   "无法给出", "无法得出", "无法估算", "无从", "查不到", "找不到",
                   "没有披露", "未提及", "没有提及")


@dataclass
class TaskRun:
    task: Task
    status: str = "error"
    answer: str = ""
    refused: bool = False
    tools_used: List[str] = field(default_factory=list)
    unsupported_rate: Optional[float] = None
    claims_total: int = 0
    claims_grounded: int = 0
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace_path: Optional[str] = None
    error: Optional[str] = None

    @property
    def tool_ok(self) -> Optional[bool]:
        """没有期望工具的任务（拒答题）不计入工具正确率。"""
        if not self.task.expect_tools:
            return None
        return bool(set(self.tools_used) & set(self.task.expect_tools))

    @property
    def passed(self) -> bool:
        if self.status != "completed" or not self.answer.strip():
            return False
        return self.refused if self.task.must_refuse else not self.refused

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task.id, "kind": self.task.kind,
            "question": self.task.question, "status": self.status,
            "passed": self.passed, "refused": self.refused,
            "must_refuse": self.task.must_refuse,
            "tools_used": self.tools_used, "tool_ok": self.tool_ok,
            "unsupported_rate": self.unsupported_rate,
            "claims": [self.claims_grounded, self.claims_total],
            "steps": self.steps, "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "answer": self.answer[:300], "error": self.error,
            "trace": (self.trace_path or "").rsplit("/", 1)[-1].removesuffix(".json") or None,
        }


def looks_like_refusal(answer: str) -> bool:
    return any(m in (answer or "") for m in REFUSAL_MARKERS)


# 五个角色跑一趟的真实开销，是实测出来的：单任务约 11 万 token。
# Budget 默认的 6 万上限是给单体 ReAct 循环定的，直接拿来评测多 Agent 流水线
# 会让每条任务都以"预算耗尽"收场——那测的是预算设错了，不是 agent 不行。
DEFAULT_MAX_TOKENS = 200_000


def run_task(task: Task, client: LLMClient, max_steps: int = 24,
             max_cost: float = 0.20,
             max_tokens: int = DEFAULT_MAX_TOKENS) -> TaskRun:
    run = TaskRun(task=task)
    sup = Supervisor(client=client,
                     budget=Budget(max_steps=max_steps, max_tokens=max_tokens,
                                   max_cost_usd=max_cost))
    t0 = time.perf_counter()
    try:
        result = sup.run(task.question)
    except Exception as e:  # noqa: BLE001 —— 单条任务炸掉不该中断整轮评测
        run.error = f"{type(e).__name__}: {e}"
        run.latency_ms = (time.perf_counter() - t0) * 1000
        return run

    run.latency_ms = (time.perf_counter() - t0) * 1000
    # 评测跑出来的轨迹也落盘：出了问题要能在 /traces 里逐步回看，
    # 而不是只看到一个"完成率 40%"然后猜是哪一步坏了
    try:
        run.trace_path = result.trace.save()
    except OSError as e:  # noqa: BLE001 —— 落盘失败不该让这条任务算作失败
        logger.warning("轨迹落盘失败：%s", e)
    run.status = result.status
    run.answer = result.answer or ""
    run.refused = looks_like_refusal(run.answer)
    run.tools_used = sorted({s.tool for s in result.trace.tool_calls
                             if s.tool and s.ok})

    summary = result.trace.summary()
    run.steps = summary.get("steps", 0)
    run.tokens = summary.get("total_tokens", 0)
    run.cost_usd = summary.get("cost_usd", 0.0)

    critique = result.critique
    if critique is not None and critique.total:
        run.unsupported_rate = critique.unsupported_rate
        run.claims_total = critique.total
        run.claims_grounded = critique.grounded
    return run


def aggregate(runs: List[TaskRun]) -> Dict[str, Any]:
    """汇总。

    **基础设施失败和质量分开算。** 一次网络抖动让 25 条任务没跑成时，把它们
    算进完成率的分母，得到的不是"agent 只完成了 37%"，而是"评测那天网不好"——
    两件事混进一个数字，指标就废了。所以质量类指标只在真正执行完的任务上算，
    失败数单独报；成本和延迟按全部任务算，因为失败的那些也真的花了钱、占了时间。
    """
    failed = [r for r in runs if r.status == "error" or r.error]
    done = [r for r in runs if r not in failed]

    answerable = [r for r in done if not r.task.must_refuse]
    refusals = [r for r in done if r.task.must_refuse]
    with_tools = [r for r in done if r.tool_ok is not None]
    with_claims = [r for r in done if r.unsupported_rate is not None]
    lat = sorted(r.latency_ms for r in runs)

    def pct(rows, pred):
        return round(sum(1 for r in rows if pred(r)) / len(rows) * 100, 2) if rows else None

    def p(q):
        return round(lat[min(len(lat) - 1, int(round(q * (len(lat) - 1))))], 1) if lat else None

    grounded = sum(r.claims_grounded for r in with_claims)
    total_claims = sum(r.claims_total for r in with_claims)

    return {
        "tasks": len(runs),
        "执行成功": len(done),
        "执行失败": len(failed),
        "失败任务": [r.task.id for r in failed],
        "任务完成率": pct(answerable, lambda r: r.passed),
        "拒答正确率": pct(refusals, lambda r: r.passed),
        "工具调用正确率": pct(with_tools, lambda r: r.tool_ok),
        "引用准确率": (round(grounded / total_claims * 100, 2) if total_claims else None),
        "结论数": [grounded, total_claims],
        "平均步数": round(sum(r.steps for r in runs) / len(runs), 2) if runs else None,
        "平均 token": round(sum(r.tokens for r in runs) / len(runs)) if runs else None,
        "总成本 USD": round(sum(r.cost_usd for r in runs), 4),
        "单任务成本 USD": (round(sum(r.cost_usd for r in runs) / len(runs), 5)
                           if runs else None),
        "p50 延迟 ms": p(0.50),
        "p95 延迟 ms": p(0.95),
    }


def run(limit: Optional[int] = None, kind: Optional[str] = None,
        client: Optional[LLMClient] = None, max_steps: int = 24,
        max_cost: float = 0.20,
        max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    from agent import tools as _tools

    tasks = [t for t in TASKS if kind is None or t.kind == kind]
    if limit:
        tasks = tasks[:limit]

    if client is None:
        from agent.llm import build_client
        client = build_client()          # 没有 key 时在这里抛，不静默降级

    _tools.bootstrap_store()
    runs = [run_task(t, client, max_steps, max_cost, max_tokens) for t in tasks]
    return {"summary": aggregate(runs), "runs": [r.to_dict() for r in runs],
            "model": getattr(client, "model", "unknown")}

# -*- coding: utf-8 -*-
"""Supervisor 编排：Planner → Collector → Analyst ⇄ Critic → Reporter。

L2 那个单体 ReAct 循环能回答问题，但它同时在做四件事——拆任务、找证据、
下结论、组织语言——出了错没法说清是哪一环坏了。这里把四件事拆成四个角色，
中间插一道确定性的 Critic 闸门：

    Planner ──► Collector ──► Analyst ──► Critic ──┬─(不通过)─► Analyst 重做
                    │                              │
                 证据池                          (通过)
                    │                              ▼
                    └──────────► 只有被引用过的证据 ──► Reporter

三个值得讲的设计：

1. **证据池是从工具返回值里收割的，不是从模型的话里解析的。**
   Collector 每次工具调用成功，回调就把带 review_id 的原文存进黑板。
   于是"证据池"是客观发生过的检索结果，模型说过什么不影响它。

2. **Analyst 拿不到检索工具。** 它只能在证据池里引用。想给一个站不住的
   结论现找一条证据，接口层面就做不到。

3. **Critic 不调 LLM。** 见 critic.py。它给出的 unsupported_rate 是算出来的，
   所以"引入 Critic 后无据结论率从 X% 降到 Y%"这句话可以当场复现给面试官看。

修订是有上限的：Analyst 改不动就停，把通过校验的部分交出去，并在报告里
说明证据有限。这比无限重试更像工程。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent import roles as role_defs
from agent.budget import Budget, BudgetExceeded
from agent.critic import Critic, CritiqueReport, extract_ids
from agent.llm import LLMClient
from agent.react import ConfirmFn, ReActAgent, deny_all
from agent.registry import ToolRegistry, ToolResult
from agent.registry import registry as default_registry
from agent.trace import RunTrace, StepRecord

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
# 交给 Analyst 的证据池条数上限：再多就是在烧 prompt token
MAX_EVIDENCE = 40


@dataclass
class Blackboard:
    """五个角色之间的共享黑板。"""
    task: str
    plan: str = ""
    evidence: Dict[int, str] = field(default_factory=dict)
    collector_note: str = ""
    findings: str = ""
    report: str = ""
    revisions: int = 0
    critiques: List[CritiqueReport] = field(default_factory=list)

    @property
    def critique(self) -> Optional[CritiqueReport]:
        return self.critiques[-1] if self.critiques else None

    def evidence_block(self, limit: int = MAX_EVIDENCE) -> str:
        if not self.evidence:
            return "（证据池为空——取证者没有检索到任何评论）"
        items = list(self.evidence.items())[:limit]
        return "\n".join(f"[review_id: {rid}] {text}" for rid, text in items)

    def grounded_claims(self) -> str:
        """通过校验的结论，且只保留**通过校验的那几个** review_id。

        一条结论引了三个 id、其中一个不切题时，交给 Reporter 的版本里不该
        再带着那一个——否则读者顺着它点进去，看到的是一条不相干的评论。
        """
        c = self.critique
        if c is None:
            return self.findings
        lines = [f"- {v.claim} [review_id: {', '.join(str(i) for i in v.supporting_ids)}]"
                 for v in c.verdicts if v.grounded]
        return "\n".join(lines)


@dataclass
class SupervisorResult:
    answer: Optional[str]
    trace: RunTrace
    status: str
    blackboard: Blackboard

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @property
    def critique(self) -> Optional[CritiqueReport]:
        return self.blackboard.critique


def harvest_evidence(payload: Any, into: Dict[int, str]) -> int:
    """从任意工具返回值里递归捞出 {review_id: text}。

    工具的返回结构各不相同（search_reviews 是 reviews 列表，get_reviews 是
    found 列表，将来还会有别的），与其为每个工具写一份解析，不如认准
    "带 review_id 和 text 的字典"这个约定——这正是工具层第 2 条硬规则的用途。
    """
    found = 0
    if isinstance(payload, dict):
        rid, text = payload.get("review_id"), payload.get("text")
        if isinstance(rid, int) and isinstance(text, str) and text:
            if rid not in into:
                into[rid] = text
                found += 1
        for v in payload.values():
            found += harvest_evidence(v, into)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            found += harvest_evidence(v, into)
    return found


class Supervisor:
    def __init__(self, client: LLMClient, registry: Optional[ToolRegistry] = None,
                 budget: Optional[Budget] = None, confirm: ConfirmFn = deny_all,
                 critic: Optional[Critic] = None, max_revisions: int = MAX_REVISIONS):
        self.client = client
        self.registry = registry or default_registry
        # 五个角色跑一趟，步数天然比单体循环多，默认给宽一点
        self.budget = budget or Budget(max_steps=24, max_cost_usd=0.50)
        self.confirm = confirm
        self.critic = critic or Critic()
        self.max_revisions = max_revisions

    # ------------------------------------------------------------------
    def run(self, task: str) -> SupervisorResult:
        trace = RunTrace(task=task, model=getattr(self.client, "model", "unknown"))
        board = Blackboard(task=task)
        status = "completed"

        try:
            board.plan = self._planner(task, trace)
            board.collector_note = self._collector(board, trace)
            self._analyst_critic_loop(board, trace)
            board.report = self._reporter(board, trace)
        except BudgetExceeded as e:
            status = "budget_exceeded"
            trace.add(StepRecord(step=len(trace.steps) + 1, kind="error",
                                 started_at=time.time(), error=str(e)))
            board.report = board.report or self._fallback_answer(board, str(e))
        except Exception as e:  # noqa: BLE001
            logger.exception("supervisor 执行异常")
            status = "error"
            trace.add(StepRecord(step=len(trace.steps) + 1, kind="error",
                                 started_at=time.time(),
                                 error=f"{type(e).__name__}: {e}"))

        trace.status = status
        trace.answer = board.report
        trace.finished_at = time.time()
        trace.budget = self.budget.snapshot()
        trace.extra = self._extra(board)
        return SupervisorResult(answer=board.report, trace=trace,
                                status=status, blackboard=board)

    # ------------------------------------------------------------------
    @staticmethod
    def _guard(result, role_name: str):
        """角色跑完后的闸门。

        角色自己的步数用完不算事故，流水线带着已有成果继续；总预算烧穿
        则必须整体收尾——否则每个角色都"各自超支"，加起来把钱烧光还照跑不误。
        """
        if result.budget_scope == "global":
            raise BudgetExceeded(f"{role_name} 阶段触及全局预算上限", "global")
        return result

    def _agent(self, role: role_defs.Role, **kwargs) -> ReActAgent:
        return ReActAgent(
            client=self.client,
            registry=self.registry,
            budget=self.budget.scoped(role.max_steps),
            system_prompt=role.system_prompt,
            confirm=self.confirm,
            allowed_tools=role.tools or [],
            **kwargs,
        )

    def _planner(self, task: str, trace: RunTrace) -> str:
        role = role_defs.PLANNER
        result = self._guard(self._agent(role).run(task, trace=trace, role=role.name),
                             role.name)
        return (result.answer or "").strip() or "- 检索与问题相关的评论\n- 分析并给出有据结论"

    def _collector(self, board: Blackboard, trace: RunTrace) -> str:
        role = role_defs.COLLECTOR

        def harvest(name: str, args: Dict[str, Any], result: ToolResult) -> None:
            harvest_evidence(result.content, board.evidence)

        agent = self._agent(role, on_tool_result=harvest)
        result = self._guard(
            agent.run(role_defs.collector_brief(board.task, board.plan),
                      trace=trace, role=role.name), role.name)
        return (result.answer or "").strip()

    def _analyst_critic_loop(self, board: Blackboard, trace: RunTrace) -> None:
        role = role_defs.ANALYST
        feedback: Optional[str] = None

        for attempt in range(self.max_revisions + 1):
            board.revisions = attempt
            brief = role_defs.analyst_brief(board.task, board.plan,
                                            board.evidence_block(), feedback)
            result = self._guard(self._agent(role).run(brief, trace=trace, role=role.name),
                                 role.name)
            board.findings = (result.answer or "").strip()

            critique = self.critic.review(board.findings, evidence_ids=board.evidence.keys())
            board.critiques.append(critique)
            self._record_critique(trace, critique, attempt)

            if critique.passed:
                return
            feedback = critique.feedback()
            # 预算快见底就别再来一轮了，把已通过的部分交出去
            if self.budget.remaining_steps <= role_defs.REPORTER.max_steps:
                logger.info("预算不足以再修订一轮，提前收敛")
                return

    def _record_critique(self, trace: RunTrace, critique: CritiqueReport,
                         attempt: int) -> None:
        summary = (f"第 {attempt + 1} 轮校验：{critique.grounded}/{critique.total} 条结论有据，"
                   f"无据率 {critique.unsupported_rate}%"
                   f"{'，通过' if critique.passed else '，打回重做'}")
        trace.add(StepRecord(
            step=len(trace.steps) + 1, kind="critic", started_at=time.time(),
            role="critic", ok=critique.passed, thought=summary,
            observation=json.dumps(critique.to_dict(), ensure_ascii=False)[:4000],
        ))

    def _reporter(self, board: Blackboard, trace: RunTrace) -> str:
        role = role_defs.REPORTER
        claims = board.grounded_claims()
        if not claims.strip():
            return self._no_evidence_answer(board)

        note = self._critique_note(board)
        result = self._guard(self._agent(role).run(
            role_defs.reporter_brief(board.task, claims, note),
            trace=trace, role=role.name), role.name)
        report = (result.answer or "").strip()

        # Reporter 把溯源链弄丢了就不采纳它的版本——宁可给一份朴素但可溯源的列表
        if report and extract_ids(report):
            return report
        logger.warning("Reporter 输出丢失了引用标注，回退到结论列表")
        return claims

    def _critique_note(self, board: Blackboard) -> str:
        c = board.critique
        if c is None:
            return ""
        dropped = c.total - c.grounded
        if not dropped:
            return f"全部 {c.total} 条结论均通过引用校验。"
        return (f"共 {c.total} 条结论，其中 {dropped} 条未通过引用校验已被剔除，"
                f"下面只剩通过的部分。撰写时请如实反映证据的覆盖范围。")

    def _no_evidence_answer(self, board: Blackboard) -> str:
        if not board.evidence:
            return ("证据不足：没有检索到与该问题相关的评论，无法给出有据结论。"
                    "可以换一组检索词，或先采集该商品的评论。")
        return (f"证据不足：检索到 {len(board.evidence)} 条评论，"
                f"但没有任何一条结论通过引用校验，因此不输出结论。")

    def _fallback_answer(self, board: Blackboard, reason: str) -> str:
        claims = board.grounded_claims().strip()
        if claims:
            return (f"未能在预算内走完全流程（{reason}），"
                    f"以下是已通过引用校验的部分结论：\n{claims}")
        return f"未能在预算内完成任务（{reason}），且尚无通过校验的结论。"

    def _extra(self, board: Blackboard) -> Dict[str, Any]:
        """挂到 trace 上的编排层指标。评测脚本直接读这几个数。"""
        if not board.critiques:
            return {"evidence_pool": len(board.evidence), "revisions": board.revisions}
        first, last = board.critiques[0], board.critiques[-1]
        return {
            "evidence_pool": len(board.evidence),
            "revisions": board.revisions,
            "citation_passed": last.passed,
            "claims_total": last.total,
            "claims_grounded": last.grounded,
            # 这两个数就是"引入 Critic 前后"的对照：第一版 vs 定稿
            "unsupported_rate_first": first.unsupported_rate,
            "unsupported_rate_final": last.unsupported_rate,
            "critique": last.to_dict(),
        }

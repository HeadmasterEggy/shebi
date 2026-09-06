# -*- coding: utf-8 -*-
"""手写的 ReAct 执行循环。

刻意不套框架。这个循环只有一件事：

    模型思考 → 发起工具调用 → 执行 → 把结果写回对话 → 再思考 → … → 给出答案

围绕它的四个工程约束才是真正值钱的部分：

  1. **预算**：步数 / token / 成本任一触顶就优雅收尾，交回已有的中间结果
  2. **schema 自修复**：参数校验失败时把 pydantic 的报错连同 JSON Schema 回灌，
     让模型自己改；每个工具调用最多修 N 次，修复成功率进 trace
  3. **人工确认**：有副作用的工具（爬虫、训练）默认拦下来，由调用方放行
  4. **全链路 trace**：每一步都落盘，可复盘、可评测、可在前端画时间线

理解透了再换 LangGraph，而不是反过来。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.budget import Budget, BudgetExceeded
from agent.llm import LLMClient, LLMResponse, ToolCall, estimate_cost
from agent.registry import ToolRegistry, ToolResult
from agent.registry import registry as default_registry
from agent.trace import RunTrace, StepRecord

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是电商评论洞察助手。你的工作是回答关于商品评论的问题，并且每条结论都要有原文支撑。

工作原则：
1. 判断文本情感一律用 classify_sentiment 工具（本地模型，几乎零成本），不要自己凭语感下结论。
2. 需要批量处理时，一次把所有文本传进去，不要逐条调用。
3. 任何写进答案的结论，都要能引用到 search_reviews / get_reviews 返回的 review_id。
   引用格式：在结论后面写 [review_id: 12, 34]。
4. 拿不到证据就明说「证据不足」，不要编造评论内容或数字。
5. 信息足够时立刻给出最终答案，不要多余的工具调用。

最终答案用中文，简洁、有据。"""


@dataclass
class AgentResult:
    answer: Optional[str]
    trace: RunTrace
    status: str

    @property
    def ok(self) -> bool:
        return self.status == "completed"


# 需要人工确认时的回调：返回 True 放行，False 拒绝
ConfirmFn = Callable[[str, Dict[str, Any]], bool]


def deny_all(tool_name: str, args: Dict[str, Any]) -> bool:
    """默认策略：不放行任何有副作用的工具。"""
    return False


class ReActAgent:
    def __init__(self, client: LLMClient, registry: Optional[ToolRegistry] = None,
                 budget: Optional[Budget] = None, system_prompt: str = SYSTEM_PROMPT,
                 confirm: ConfirmFn = deny_all, max_repairs_per_call: int = 2,
                 allowed_tools: Optional[List[str]] = None):
        self.client = client
        self.registry = registry or default_registry
        self.budget = budget or Budget()
        self.system_prompt = system_prompt
        self.confirm = confirm
        self.max_repairs_per_call = max_repairs_per_call
        self.allowed_tools = allowed_tools

    # ------------------------------------------------------------------
    def run(self, task: str) -> AgentResult:
        trace = RunTrace(task=task, model=getattr(self.client, "model", "unknown"))
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        tools = self.registry.schemas(self.allowed_tools)
        # 每个 tool_call_id 已经修复过几次，防止在同一个错误上死循环
        repairs: Dict[str, int] = {}
        status, answer = "error", None

        try:
            while True:
                self.budget.start_step()
                response = self._call_llm(messages, tools, trace)

                if not response.tool_calls:
                    answer = response.content or ""
                    status = "completed"
                    trace.add(StepRecord(step=len(trace.steps) + 1, kind="final",
                                         started_at=time.time(), thought=answer))
                    break

                messages.append(_assistant_message(response))
                for call in response.tool_calls:
                    observation, repaired = self._execute(call, repairs, trace)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": observation,
                    })
                    if repaired:
                        repairs[call.id] = repairs.get(call.id, 0) + 1

                self.budget.check()

        except BudgetExceeded as e:
            status = "budget_exceeded"
            trace.add(StepRecord(step=len(trace.steps) + 1, kind="error",
                                 started_at=time.time(), error=str(e)))
            answer = self._wrap_up(messages, trace, str(e))
        except Exception as e:  # noqa: BLE001
            logger.exception("agent 执行异常")
            status = "error"
            trace.add(StepRecord(step=len(trace.steps) + 1, kind="error",
                                 started_at=time.time(), error=f"{type(e).__name__}: {e}"))

        trace.status = status
        trace.answer = answer
        trace.finished_at = time.time()
        trace.budget = self.budget.snapshot()
        return AgentResult(answer=answer, trace=trace, status=status)

    # ------------------------------------------------------------------
    def _call_llm(self, messages, tools, trace: RunTrace) -> LLMResponse:
        t0 = time.perf_counter()
        response = self.client.chat(messages, tools=tools)
        latency = (time.perf_counter() - t0) * 1000

        model = getattr(self.client, "model", "unknown")
        self.budget.add_usage(model, response.usage)
        trace.add(StepRecord(
            step=len(trace.steps) + 1, kind="llm", started_at=time.time(),
            latency_ms=round(latency, 2), thought=response.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=round(estimate_cost(model, response.usage), 6),
        ))
        return response

    def _execute(self, call: ToolCall, repairs: Dict[str, int],
                 trace: RunTrace) -> tuple[str, bool]:
        """执行一次工具调用，返回 (写回对话的 observation, 本次是否触发自修复)。"""
        # 人工确认闸门
        try:
            tool = self.registry.get(call.name)
        except KeyError:
            tool = None

        if tool is not None and tool.requires_confirmation:
            if not self.confirm(call.name, call.arguments):
                obs = (f"工具 {call.name} 有副作用，未获得人工确认，本次不执行。"
                       "请改用无副作用的工具，或在答案中说明这一步需要用户授权。")
                trace.add(StepRecord(step=len(trace.steps) + 1, kind="tool",
                                     started_at=time.time(), tool=call.name,
                                     tool_args=call.arguments, ok=False,
                                     error="denied_by_confirmation", observation=obs))
                return obs, False

        result: ToolResult = self.registry.invoke(call.name, call.arguments)

        # 参数校验失败 → 回灌错误让模型自修复
        repaired = False
        if not result.ok and result.repair_hint:
            used = repairs.get(call.id, 0)
            if used < self.max_repairs_per_call:
                repaired = True
            else:
                result = ToolResult(
                    ok=False,
                    error=result.error,
                    content=None,
                    latency_ms=result.latency_ms,
                )

        observation = result.to_observation()
        trace.add(StepRecord(
            step=len(trace.steps) + 1, kind="tool", started_at=time.time(),
            latency_ms=round(result.latency_ms, 2), tool=call.name,
            tool_args=call.arguments, ok=result.ok,
            observation=observation[:2000], error=result.error, repaired=repaired,
        ))
        return observation, repaired

    def _wrap_up(self, messages, trace: RunTrace, reason: str) -> str:
        """预算耗尽时，把已经拿到的东西整理成一句可用的交代，而不是直接抛错。"""
        gathered = [s for s in trace.tool_calls if s.ok]
        if not gathered:
            return f"未能完成任务（{reason}），且没有拿到任何可用的中间结果。"
        used = ", ".join(sorted({s.tool for s in gathered if s.tool}))
        return (f"未能在预算内完成任务（{reason}）。"
                f"已完成的工具调用：{used}。请缩小问题范围或提高预算后重试。")


def _assistant_message(response: LLMResponse) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [{
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
        } for c in response.tool_calls],
    }

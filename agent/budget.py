# -*- coding: utf-8 -*-
"""执行预算。

一个没有预算的 agent 循环，遇到模型反复调用同一个工具时会一直烧钱直到超时。
这里把三个上限做成显式对象：步数、token、成本。任何一个触顶就优雅终止，
把已经拿到的中间结果交回去，而不是抛异常丢掉全部工作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent.llm import Usage, estimate_cost


class BudgetExceeded(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    max_steps: int = 12
    max_tokens: Optional[int] = 60_000
    max_cost_usd: Optional[float] = 0.50

    steps: int = field(default=0, init=False)
    usage: Usage = field(default_factory=Usage, init=False)
    cost_usd: float = field(default=0.0, init=False)

    def start_step(self) -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"达到步数上限 {self.max_steps}")
        self.steps += 1

    def add_usage(self, model: str, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.cost_usd += estimate_cost(model, usage)

    def check(self) -> None:
        """在一步结束后检查：超了就在下一步开始前停下。"""
        if self.max_tokens is not None and self.usage.total > self.max_tokens:
            raise BudgetExceeded(f"达到 token 上限 {self.max_tokens}（已用 {self.usage.total}）")
        if self.max_cost_usd is not None and self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"达到成本上限 ${self.max_cost_usd}（已花 ${self.cost_usd:.4f}）")

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps)

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total,
            "cost_usd": round(self.cost_usd, 6),
        }

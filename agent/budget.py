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
    """超支。

    scope 区分两种性质完全不同的超支：

      "role"   某个角色自己的步数用完了。它该停下，但流水线继续——
               取证者查了六轮还没查够，也该让分析师拿已有的证据往下走。
      "global" 整条流水线的总预算烧穿了。这时候必须整体收尾，
               再往下跑只是继续烧钱。

    不做这个区分的话，全局超支会被每个角色各自吞掉，流水线照跑不误。
    """

    def __init__(self, reason: str, scope: str = "global"):
        super().__init__(reason)
        self.reason = reason
        self.scope = scope


@dataclass
class Budget:
    max_steps: int = 12
    max_tokens: Optional[int] = 60_000
    max_cost_usd: Optional[float] = 0.50
    scope: str = "global"

    steps: int = field(default=0, init=False)
    usage: Usage = field(default_factory=Usage, init=False)
    cost_usd: float = field(default=0.0, init=False)

    def start_step(self) -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"达到步数上限 {self.max_steps}", self.scope)
        self.steps += 1

    def add_usage(self, model: str, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.cost_usd += estimate_cost(model, usage)

    def check(self) -> None:
        """在一步结束后检查：超了就在下一步开始前停下。"""
        if self.max_tokens is not None and self.usage.total > self.max_tokens:
            raise BudgetExceeded(
                f"达到 token 上限 {self.max_tokens}（已用 {self.usage.total}）", self.scope)
        if self.max_cost_usd is not None and self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"达到成本上限 ${self.max_cost_usd}（已花 ${self.cost_usd:.4f}）", self.scope)

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps)

    def scoped(self, max_steps: int) -> "ScopedBudget":
        """派生一个子预算给某个角色用。"""
        return ScopedBudget(self, max_steps)

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total,
            "cost_usd": round(self.cost_usd, 6),
        }


class ScopedBudget(Budget):
    """单个角色的预算视图。

    多 Agent 下有两种超支，得分开管：一个角色自己在原地打转，和整条流水线
    总共走了多少步、烧了多少钱。所以子预算给角色一个自己的步数上限，同时
    把每一步、每一个 token 都记进总账，并在每步之后拿总账再查一次——
    某个角色把全局预算烧穿时它当场就停，而不是等编排层在阶段之间才发现。

    两边都记，是因为漏掉任何一边都会出事：只记子预算，全局上限形同虚设
    （每个角色都"没超"，加起来早就超了）；只记总账，一个角色在原地打转时
    会把后面角色的额度全吃掉。
    """

    def __init__(self, parent: Budget, max_steps: int):
        super().__init__(max_steps=max_steps, max_tokens=None, max_cost_usd=None,
                         scope="role")
        self.parent = parent

    def start_step(self) -> None:
        super().start_step()      # 先查角色自己的步数上限
        self.parent.start_step()  # 再把这一步记进总账，全局上限同样生效

    def add_usage(self, model: str, usage: Usage) -> None:
        super().add_usage(model, usage)
        self.parent.add_usage(model, usage)

    def check(self) -> None:
        super().check()
        self.parent.check()

# -*- coding: utf-8 -*-
"""执行轨迹。

每一步都落盘：模型想了什么、调了哪个工具、传了什么参数、拿到什么、
花了多少 token、多少毫秒。三个用途：

  1. 前端画「Agent 执行轨迹」时间线
  2. 出问题时能复盘到具体某一步，而不是只看到最终答案不对
  3. 评测时统计工具调用正确率、schema 修复成功率、平均步数与成本

一行一个 JSON，方便直接 pandas 读。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import Config

TRACE_DIR = os.path.join(Config.runtime_dir, "traces")


@dataclass
class StepRecord:
    step: int
    kind: str                      # llm | tool | final | error | critic
    started_at: float
    # 多 Agent 编排下由哪个角色产生：planner / collector / analyst / critic / reporter
    role: Optional[str] = None
    latency_ms: float = 0.0
    thought: Optional[str] = None
    tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    ok: Optional[bool] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    repaired: bool = False         # 这一步是否是一次 schema 自修复
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str = ""
    model: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    status: str = "running"        # running | completed | budget_exceeded | error
    answer: Optional[str] = None
    steps: List[StepRecord] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    # 编排层产物：引用校验报告、修订轮数等，评测直接读这里
    extra: Dict[str, Any] = field(default_factory=dict)

    def add(self, record: StepRecord) -> StepRecord:
        self.steps.append(record)
        return record

    # ---- 派生指标：评测直接用这些 ----
    @property
    def tool_calls(self) -> List[StepRecord]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def failed_tool_calls(self) -> List[StepRecord]:
        return [s for s in self.tool_calls if s.ok is False]

    @property
    def repair_attempts(self) -> int:
        return sum(1 for s in self.steps if s.repaired)

    @property
    def roles_used(self) -> List[str]:
        seen = [s.role for s in self.steps if s.role]
        return list(dict.fromkeys(seen))

    def steps_by_role(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.steps:
            if s.role:
                out[s.role] = out.get(s.role, 0) + 1
        return out

    def summary(self) -> Dict[str, Any]:
        calls = self.tool_calls
        # 注意 key 不要和 budget.snapshot() 撞车：budget 里的 steps 是循环轮数，
        # 这里的 records 是落盘的轨迹条数（一轮里可能有多条工具记录）。
        return {
            "run_id": self.run_id,
            "status": self.status,
            "records": len(self.steps),
            "tool_calls": len(calls),
            "failed_tool_calls": len(self.failed_tool_calls),
            "repair_attempts": self.repair_attempts,
            "tools_used": sorted({s.tool for s in calls if s.tool}),
            **({"roles": self.roles_used, "steps_by_role": self.steps_by_role()}
               if self.roles_used else {}),
            **self.extra,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 3),
            **self.budget,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["summary"] = self.summary()
        return d

    def save(self, directory: str = TRACE_DIR) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        return path


def load_trace(run_id: str, directory: str = TRACE_DIR) -> Dict[str, Any]:
    with open(os.path.join(directory, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def list_traces(directory: str = TRACE_DIR) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))

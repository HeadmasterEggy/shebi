# -*- coding: utf-8 -*-
"""ReAct 循环：预算、schema 自修复、人工确认、终止条件。

全部用 ScriptedClient 离线跑——这些逻辑不该依赖一次真实的网络调用才能验证。
"""

import pytest
from pydantic import BaseModel, Field

from agent.budget import Budget
from agent.llm import LLMResponse, ScriptedClient, ToolCall, Usage
from agent.react import ReActAgent
from agent.registry import ToolRegistry


class Args(BaseModel):
    n: int = Field(..., ge=1, le=10)


@pytest.fixture
def reg():
    r = ToolRegistry()
    calls = []

    @r.register(name="double", description="翻倍", args_model=Args)
    def _double(n: int):
        calls.append(n)
        return {"value": n * 2}

    @r.register(name="danger", description="有副作用", args_model=Args,
                requires_confirmation=True)
    def _danger(n: int):
        calls.append(("danger", n))
        return {"done": True}

    r.observed_calls = calls
    return r


def _tool_call(name, args, cid="c1"):
    return LLMResponse(tool_calls=[ToolCall(id=cid, name=name, arguments=args)],
                       usage=Usage(100, 20))


def _final(text):
    return LLMResponse(content=text, usage=Usage(80, 30))


def test_happy_path_calls_tool_then_answers(reg):
    client = ScriptedClient([_tool_call("double", {"n": 3}), _final("结果是 6")])
    res = ReActAgent(client, registry=reg).run("把 3 翻倍")
    assert res.ok and res.answer == "结果是 6"
    assert reg.observed_calls == [3]
    assert [s.kind for s in res.trace.steps] == ["llm", "tool", "llm", "final"]


def test_schema_repair_feeds_error_back_and_succeeds(reg):
    """第一次参数越界，把报错回灌后模型改对——这条路径必须走通并被记录。"""
    client = ScriptedClient([
        _tool_call("double", {"n": 99}),          # 越界
        _tool_call("double", {"n": 5}, cid="c1"),  # 修复后重试
        _final("结果是 10"),
    ])
    res = ReActAgent(client, registry=reg).run("翻倍")
    assert res.ok
    assert reg.observed_calls == [5]
    assert res.trace.repair_attempts == 1
    # 修复提示必须真的进了回灌给模型的对话历史
    last_messages = client.calls[-1]["messages"]
    assert any("JSON Schema" in str(m.get("content", "")) for m in last_messages)


def test_repair_is_capped_to_avoid_infinite_loop(reg):
    client = ScriptedClient([_tool_call("double", {"n": 99}, cid="c1") for _ in range(6)]
                            + [_final("放弃")])
    agent = ReActAgent(client, registry=reg, budget=Budget(max_steps=8),
                       max_repairs_per_call=2)
    res = agent.run("翻倍")
    assert res.trace.repair_attempts <= 2


def test_step_budget_terminates_gracefully(reg):
    """步数耗尽要交回已有结果，而不是抛异常丢掉全部工作。"""
    client = ScriptedClient([_tool_call("double", {"n": 2}, cid=f"c{i}") for i in range(20)])
    res = ReActAgent(client, registry=reg, budget=Budget(max_steps=4)).run("一直翻倍")
    assert res.status == "budget_exceeded"
    assert res.answer and "预算" in res.answer
    assert "double" in res.answer          # 已完成的工具调用要交代出来
    assert res.trace.budget["steps"] == 4


def test_cost_budget_terminates(reg):
    client = ScriptedClient(
        [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="double", arguments={"n": 2})],
                     usage=Usage(1_000_000, 1_000_000)) for i in range(10)],
        model="gpt-4o",
    )
    res = ReActAgent(client, registry=reg,
                     budget=Budget(max_steps=50, max_cost_usd=1.0)).run("烧钱")
    assert res.status == "budget_exceeded"
    assert res.trace.budget["cost_usd"] > 1.0


def test_token_budget_terminates(reg):
    client = ScriptedClient(
        [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="double", arguments={"n": 2})],
                     usage=Usage(30_000, 10_000)) for i in range(10)])
    res = ReActAgent(client, registry=reg,
                     budget=Budget(max_steps=50, max_tokens=50_000,
                                   max_cost_usd=None)).run("烧 token")
    assert res.status == "budget_exceeded"


def test_side_effect_tool_is_blocked_by_default(reg):
    client = ScriptedClient([_tool_call("danger", {"n": 1}), _final("已说明需要授权")])
    res = ReActAgent(client, registry=reg).run("跑那个危险的")
    assert res.ok
    assert reg.observed_calls == []          # 实现根本没被调用
    tool_step = res.trace.tool_calls[0]
    assert tool_step.ok is False and tool_step.error == "denied_by_confirmation"


def test_side_effect_tool_runs_when_confirmed(reg):
    client = ScriptedClient([_tool_call("danger", {"n": 1}), _final("done")])
    res = ReActAgent(client, registry=reg, confirm=lambda n, a: True).run("跑")
    assert res.ok and reg.observed_calls == [("danger", 1)]


def test_unknown_tool_does_not_crash_the_loop(reg):
    client = ScriptedClient([_tool_call("no_such_tool", {}), _final("换个工具")])
    res = ReActAgent(client, registry=reg).run("调个不存在的")
    assert res.ok
    assert res.trace.tool_calls[0].ok is False


def test_trace_records_tokens_and_cost(reg):
    client = ScriptedClient([_tool_call("double", {"n": 1}), _final("ok")], model="gpt-4o-mini")
    res = ReActAgent(client, registry=reg).run("t")
    b = res.trace.budget
    assert b["total_tokens"] == 100 + 20 + 80 + 30
    assert b["cost_usd"] > 0


def test_trace_roundtrips_to_disk(reg, tmp_path):
    client = ScriptedClient([_tool_call("double", {"n": 1}), _final("ok")])
    res = ReActAgent(client, registry=reg).run("t")
    path = res.trace.save(str(tmp_path))
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary"]["tool_calls"] == 1
    assert data["summary"]["records"] == 4  # llm, tool, llm, final
    assert data["summary"]["tools_used"] == ["double"]


def test_allowed_tools_restricts_exposed_schemas(reg):
    client = ScriptedClient([_final("ok")])
    ReActAgent(client, registry=reg, allowed_tools=["double"]).run("t")
    exposed = {t["function"]["name"] for t in client.calls[0]["tools"]}
    assert exposed == {"double"}

# -*- coding: utf-8 -*-
"""工具注册表：schema 导出、参数校验、修复提示。"""

import pytest
from pydantic import BaseModel, Field

from agent.registry import ToolError, ToolRegistry


class Args(BaseModel):
    n: int = Field(..., ge=1, le=10, description="一个 1..10 的整数")
    tag: str = "default"


@pytest.fixture
def reg():
    r = ToolRegistry()

    @r.register(name="double", description="把 n 翻倍", args_model=Args)
    def _double(n: int, tag: str = "default"):
        return {"value": n * 2, "tag": tag}

    @r.register(name="boom", description="总是失败", args_model=Args,
                requires_confirmation=True)
    def _boom(n: int, tag: str = "default"):
        raise ToolError("外部服务不可用", retryable=True)

    return r


def test_json_schema_is_function_calling_shaped(reg):
    schema = reg.get("double").json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "double"
    params = schema["function"]["parameters"]
    assert params["properties"]["n"]["maximum"] == 10
    assert "n" in params["required"]


def test_valid_call_returns_content(reg):
    res = reg.invoke("double", {"n": 4})
    assert res.ok and res.content["value"] == 8


def test_invalid_arg_returns_repair_hint_not_exception(reg):
    """参数越界不该抛异常，而要给出能回灌给模型的修复提示。"""
    res = reg.invoke("double", {"n": 99})
    assert res.ok is False
    assert "n" in res.error
    assert "JSON Schema" in res.repair_hint
    assert "less than or equal to 10" in res.error or "10" in res.error


def test_missing_required_arg_is_reported(reg):
    res = reg.invoke("double", {})
    assert res.ok is False and "n" in res.error


def test_unknown_tool_lists_available_tools(reg):
    res = reg.invoke("nope", {})
    assert res.ok is False
    assert "double" in res.repair_hint and "boom" in res.repair_hint


def test_tool_exception_is_captured_not_raised(reg):
    res = reg.invoke("boom", {"n": 1})
    assert res.ok is False and "外部服务不可用" in res.error


def test_signature_mismatch_is_caught_at_registration():
    """注册时就该发现实现与参数模型对不上，而不是运行到一半才炸。"""
    r = ToolRegistry()
    with pytest.raises(TypeError):
        @r.register(name="bad", description="签名不匹配", args_model=Args)
        def _bad(wrong_name: int):
            return wrong_name


def test_duplicate_registration_is_rejected(reg):
    with pytest.raises(ValueError):
        @reg.register(name="double", description="重名", args_model=Args)
        def _dup(n: int, tag: str = "default"):
            return n


def test_observation_is_string_for_both_outcomes(reg):
    assert isinstance(reg.invoke("double", {"n": 2}).to_observation(), str)
    assert isinstance(reg.invoke("double", {"n": 0}).to_observation(), str)


def test_confirmation_flag_is_exposed(reg):
    assert reg.get("boom").requires_confirmation is True
    assert reg.get("double").requires_confirmation is False

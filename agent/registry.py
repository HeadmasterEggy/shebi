# -*- coding: utf-8 -*-
"""工具注册表。

一个工具 = 一个 pydantic 参数模型 + 一个可调用实现。注册时自动导出
JSON Schema 供 function calling 使用，调用时强制校验参数。

校验失败不是终点：错误信息会被整理成一段模型能读懂的文本回灌给它，
让它自己改参数重试（schema 自修复）。修复成功率是一个值得记录的指标。
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """工具执行失败。区分「可重试」与「不可重试」，交给循环决定要不要再试。"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ToolResult:
    ok: bool
    content: Any = None
    error: Optional[str] = None
    # 参数校验失败时，这里放能直接回灌给模型的修复提示
    repair_hint: Optional[str] = None
    latency_ms: float = 0.0

    def to_observation(self) -> str:
        """转成写进对话历史的 observation 文本。"""
        if self.ok:
            return _render(self.content)
        if self.repair_hint:
            return f"参数校验失败：{self.error}\n{self.repair_hint}"
        return f"工具执行失败：{self.error}"


def _render(value: Any, limit: int = 4000) -> str:
    import json

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) > limit:
        text = text[:limit] + f"\n…（已截断，完整长度 {len(text)} 字符）"
    return text


@dataclass
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    func: Callable[..., Any]
    # 有副作用的工具（发外部请求、占用 GPU、写磁盘）需要人工确认
    requires_confirmation: bool = False
    tags: List[str] = field(default_factory=list)

    def json_schema(self) -> Dict[str, Any]:
        """OpenAI function calling 格式的工具描述。"""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def invoke(self, raw_args: Dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            args = self.args_model(**(raw_args or {}))
        except ValidationError as e:
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(
                ok=False,
                error=_format_validation_error(e),
                repair_hint=self._repair_hint(),
                latency_ms=elapsed,
            )
        except TypeError as e:
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(ok=False, error=str(e),
                              repair_hint=self._repair_hint(), latency_ms=elapsed)

        try:
            out = self.func(**args.model_dump())
        except ToolError as e:
            return ToolResult(ok=False, error=str(e),
                              latency_ms=(time.perf_counter() - started) * 1000)
        except Exception as e:  # noqa: BLE001 - 工具异常不应打断整个 agent
            logger.exception("工具 %s 执行异常", self.name)
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}",
                              latency_ms=(time.perf_counter() - started) * 1000)

        return ToolResult(ok=True, content=out,
                          latency_ms=(time.perf_counter() - started) * 1000)

    def _repair_hint(self) -> str:
        import json

        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ("请按下面的 JSON Schema 重新给出参数，只输出参数对象：\n"
                + json.dumps(schema, ensure_ascii=False))


def _format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(根)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, args_model: Type[BaseModel],
                 requires_confirmation: bool = False, tags: Optional[List[str]] = None):
        """装饰器：把一个函数注册成工具。"""

        def deco(func: Callable[..., Any]):
            if name in self._tools:
                raise ValueError(f"工具重名: {name}")
            _check_signature(func, args_model, name)
            self._tools[name] = Tool(
                name=name, description=description, args_model=args_model,
                func=func, requires_confirmation=requires_confirmation,
                tags=tags or [],
            )
            return func

        return deco

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def all(self) -> List[Tool]:
        return [self._tools[n] for n in self.names()]

    def schemas(self, only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = self.all() if only is None else [self.get(n) for n in only]
        return [t.json_schema() for t in tools]

    def invoke(self, name: str, raw_args: Dict[str, Any]) -> ToolResult:
        try:
            tool = self.get(name)
        except KeyError:
            return ToolResult(
                ok=False,
                error=f"不存在的工具 '{name}'",
                repair_hint="可用工具：" + ", ".join(self.names()),
            )
        return tool.invoke(raw_args)


def _check_signature(func: Callable[..., Any], args_model: Type[BaseModel], name: str):
    """注册时就校验函数签名与参数模型一致，避免运行到一半才发现对不上。"""
    params = set(inspect.signature(func).parameters)
    fields = set(args_model.model_fields)
    missing = fields - params
    if missing:
        raise TypeError(f"工具 {name} 的实现缺少参数: {sorted(missing)}")


registry = ToolRegistry()

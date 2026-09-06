# -*- coding: utf-8 -*-
"""LLM provider 抽象。

只依赖 OpenAI 的 chat completions 协议，因此同一份代码可以在
OpenAI / Anthropic 兼容端点 / DeepSeek / Qwen(百炼) / Kimi 之间切换——
改的是 base_url 和 model 两个环境变量，不是代码。

  开发调试  →  便宜的国内模型
  面试演示  →  效果最稳的模型
  断网 / CI →  ScriptedClient，不发任何网络请求

价格表用于把 token 折算成成本，让预算控制能以「钱」为单位。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    raw: Any = None


# 每百万 token 的美元单价，用于成本预算。未知模型按 0 计。
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o":            {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":       {"in": 0.15, "out": 0.60},
    "deepseek-chat":     {"in": 0.27, "out": 1.10},
    "qwen-plus":         {"in": 0.11, "out": 0.28},
    "qwen-max":          {"in": 0.34, "out": 1.37},
}


def estimate_cost(model: str, usage: Usage) -> float:
    price = PRICING.get(model)
    if not price:
        for key, p in PRICING.items():
            if model.startswith(key):
                price = p
                break
    if not price:
        return 0.0
    return (usage.prompt_tokens * price["in"] + usage.completion_tokens * price["out"]) / 1_000_000


class LLMClient:
    """provider 接口。实现只需要一个 chat 方法。"""

    model: str = "unknown"

    def chat(self, messages: Sequence[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             temperature: float = 0.0) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """任何兼容 OpenAI chat completions 协议的端点。"""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 60.0):
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("AGENT_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "缺少 API key。设置 AGENT_API_KEY（可配合 AGENT_BASE_URL / AGENT_MODEL "
                "切换到 DeepSeek、Qwen 等 OpenAI 兼容端点），或改用 ScriptedClient 离线运行。"
            )

    def chat(self, messages, tools=None, temperature: float = 0.0) -> LLMResponse:
        import urllib.error
        import urllib.request

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM 请求失败 {e.code}: {e.read().decode('utf-8')[:400]}") from e

        return self._parse(body)

    def _parse(self, body: Dict[str, Any]) -> LLMResponse:
        choice = body["choices"][0]
        msg = choice["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                # 参数不是合法 JSON 也不丢弃：交给工具层校验，让模型自修复
                args = {"__malformed__": raw_args}
            calls.append(ToolCall(id=tc.get("id") or f"call_{i}",
                                  name=fn.get("name", ""), arguments=args))

        u = body.get("usage") or {}
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=calls,
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=body,
        )


class ScriptedClient(LLMClient):
    """按剧本回放的假 provider。

    用于离线测试执行循环本身——预算、schema 自修复、终止条件这些逻辑
    不该依赖一次真实的网络调用才能验证。
    """

    def __init__(self, script: Sequence[LLMResponse], model: str = "scripted"):
        self.script = list(script)
        self.model = model
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, tools=None, temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.script:
            return LLMResponse(content="（剧本已用尽）", usage=Usage(1, 1))
        return self.script.pop(0)


def build_client() -> LLMClient:
    """按环境变量组装 provider；没有 key 时明确报错，不静默降级。"""
    return OpenAICompatibleClient()

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
import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class _Transient(Exception):
    """值得重试的失败。只在本模块内部流转，对外一律是 RuntimeError。"""


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


# 值得重试的 HTTP 状态：限流与网关抖动。
# 400/401/402/403 不在其中——参数错、没授权、余额不足，重试一百次也是同样结果，
# 只会把错误延迟三倍才报出来。
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatibleClient(LLMClient):
    """任何兼容 OpenAI chat completions 协议的端点。

    **带重试。** 这一条是评测跑出来的：一次瞬时 TLS 握手失败（网络抖动，
    十几秒后自行恢复）让 40 条任务里的 25 条全军覆没——每条任务八到十次调用，
    任意一次断掉整条流水线就废了。一个要连续发几百次网络请求的 agent，
    没有重试等于把成功率押在网络一次都不抖上。

    退避带随机抖动：多条任务同时撞上限流时，固定退避会让它们在同一时刻
    一起重试，第二次照样一起被限流。
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 60.0,
                 max_retries: int = 3, backoff_base: float = 1.0,
                 sleep=None):
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("AGENT_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep or time.sleep
        if not self.api_key:
            raise RuntimeError(
                "缺少 API key。设置 AGENT_API_KEY（可配合 AGENT_BASE_URL / AGENT_MODEL "
                "切换到 DeepSeek、Qwen 等 OpenAI 兼容端点），或改用 ScriptedClient 离线运行。"
            )

    def chat(self, messages, tools=None, temperature: float = 0.0) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._parse(self._post(payload))
            except _Transient as e:
                last = e
                if attempt == self.max_retries:
                    break
                delay = self.backoff_base * (2 ** attempt) * (0.5 + random.random())
                logger.warning("LLM 请求失败（%s），%.1fs 后第 %d 次重试：%s",
                               type(e.__cause__ or e).__name__, delay, attempt + 1, e)
                self._sleep(delay)

        raise RuntimeError(
            f"LLM 请求连续失败 {self.max_retries + 1} 次：{last}") from last

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发一次请求。可重试的失败统一包成 _Transient，其余原样抛出。"""
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = f"LLM 请求失败 {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            if e.code in RETRYABLE_STATUS:
                raise _Transient(detail) from e
            # 参数错、没授权、余额不足——重试没有意义，立刻报出来
            raise RuntimeError(detail) from e
        except urllib.error.URLError as e:
            # 连不上、DNS 解析失败、TLS 握手失败：这类基本都是瞬时的
            raise _Transient(f"网络错误：{e.reason}") from e
        except (TimeoutError, socket.timeout) as e:
            raise _Transient("请求超时") from e

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

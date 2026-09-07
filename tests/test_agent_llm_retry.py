# -*- coding: utf-8 -*-
"""LLM 调用的重试策略。

这一层是评测跑出来的：一次瞬时 TLS 握手失败让 40 条任务里的 25 条全军覆没。
每条任务八到十次调用，任意一次断掉整条流水线就废了——一个要连续发几百次
网络请求的 agent，没有重试等于把成功率押在网络一次都不抖上。

打桩打在 `urllib.request.urlopen` 这一层而不是 `_post`：错误分类（哪些算瞬时、
哪些是永久性）正是在 `_post` 里做的，绕过它就等于没测这部分。
"""

import io
import json
import urllib.error
import urllib.request

import pytest

from agent.llm import RETRYABLE_STATUS, OpenAICompatibleClient

BODY = {
    "choices": [{"message": {"content": "好"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


class _Resp(io.BytesIO):
    """urlopen 返回值的最小替身：一个能当上下文管理器用的字节流。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return urllib.error.HTTPError("https://x/v1/chat/completions", code, "err", {},
                                  io.BytesIO(b'{"error":"boom"}'))


class FakeNet:
    """按剧本决定每一次 urlopen 的结果，并记录被调了几次。"""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        out = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(out, Exception):
            raise out
        return _Resp(json.dumps(BODY).encode("utf-8"))


@pytest.fixture
def client():
    return OpenAICompatibleClient(model="test-model", api_key="sk-test",
                                  backoff_base=0.0, sleep=lambda _: None)


def ask(client):
    return client.chat([{"role": "user", "content": "hi"}])


# ---------------- 该重试的 ----------------
def test_transient_tls_failure_is_retried(monkeypatch, client):
    """就是这个场景干掉了 25 条任务：TLS 握手瞬时失败，十几秒后自行恢复。"""
    net = FakeNet(urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED]"), "ok")
    monkeypatch.setattr(urllib.request, "urlopen", net)

    assert ask(client).content == "好"
    assert net.calls == 2, "第一次失败后应当重试一次就成功"


def test_timeout_is_retried(monkeypatch, client):
    net = FakeNet(TimeoutError(), TimeoutError(), "ok")
    monkeypatch.setattr(urllib.request, "urlopen", net)
    assert ask(client).content == "好"
    assert net.calls == 3


@pytest.mark.parametrize("code", sorted(RETRYABLE_STATUS))
def test_retryable_status_codes_are_retried(monkeypatch, client, code):
    """限流与网关抖动值得再试一次。"""
    net = FakeNet(http_error(code), "ok")
    monkeypatch.setattr(urllib.request, "urlopen", net)
    assert ask(client).content == "好"
    assert net.calls == 2


def test_retries_are_capped_and_the_error_is_surfaced(monkeypatch):
    c = OpenAICompatibleClient(model="m", api_key="sk-test", max_retries=2,
                               backoff_base=0.0, sleep=lambda _: None)
    net = FakeNet(*[urllib.error.URLError("boom")] * 5)
    monkeypatch.setattr(urllib.request, "urlopen", net)

    with pytest.raises(RuntimeError, match="连续失败 3 次"):
        ask(c)
    assert net.calls == 3, "重试必须有上限，不能一直试下去"


# ---------------- 不该重试的 ----------------
@pytest.mark.parametrize("code,why", [
    (400, "参数错"), (401, "没授权"), (402, "余额不足"), (403, "被禁"), (404, "路径错"),
])
def test_permanent_errors_are_not_retried(monkeypatch, client, code, why):
    """重试一百次也是同样结果，只会把错误延迟三倍才报出来。"""
    net = FakeNet(*[http_error(code)] * 5)
    monkeypatch.setattr(urllib.request, "urlopen", net)

    with pytest.raises(RuntimeError) as e:
        ask(client)
    assert net.calls == 1, f"{why}（HTTP {code}）不该重试"
    assert "连续失败" not in str(e.value), "永久性错误应当原样报出，不套重试话术"
    assert str(code) in str(e.value)


def test_backoff_grows_and_is_jittered(monkeypatch):
    """固定退避会让同时撞上限流的任务在同一时刻一起重试，第二次照样一起被限流。"""
    delays = []
    c = OpenAICompatibleClient(model="m", api_key="sk-test", max_retries=3,
                               backoff_base=1.0, sleep=delays.append)
    monkeypatch.setattr(urllib.request, "urlopen",
                        FakeNet(*[urllib.error.URLError("x")] * 3, "ok"))
    ask(c)

    assert len(delays) == 3
    assert delays[0] < delays[-1], "退避应当递增"
    # 抖动系数落在 [0.5, 1.5)，所以第 n 次退避在 base·2ⁿ 的这个区间内
    for i, d in enumerate(delays):
        assert 2 ** i * 0.5 <= d < 2 ** i * 1.5

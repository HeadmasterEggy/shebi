# -*- coding: utf-8 -*-
"""工具层：批量语义、溯源 id、副作用标记、本地模型接线。"""

import pytest

from agent import store
from agent.registry import registry
from agent.tools import MAX_BATCH  # noqa: F401 —— import 即完成工具注册
from config import Config
from tests.conftest import needs_w2v, needs_weights



@pytest.fixture
def db(tmp_path):
    conn = store.connect(str(tmp_path / "reviews.db"))
    store.add_reviews([
        {"source": "test", "product_id": "p1", "text": "物流很慢，等了一个星期才到", "gold_label": 0},
        {"source": "test", "product_id": "p1", "text": "屏幕有划痕，客服态度也差", "gold_label": 0},
        {"source": "test", "product_id": "p1", "text": "手机质量很好，用起来非常流畅", "gold_label": 1},
        {"source": "test", "product_id": "p2", "text": "包装完好，发货速度快", "gold_label": 1},
    ], conn)
    return conn


# ---------------- 注册表本身 ----------------
def test_all_tools_expose_valid_schemas():
    for tool in registry.all():
        schema = tool.json_schema()
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["description"], f"{tool.name} 缺少描述"
        assert "properties" in schema["function"]["parameters"]


def test_side_effect_tools_require_confirmation():
    """发外部请求、占算力的工具必须显式标记，不能默默执行。"""
    for name in ("scrape_reviews", "train_model"):
        assert registry.get(name).requires_confirmation, f"{name} 应当需要人工确认"


def test_read_only_tools_do_not_require_confirmation():
    for name in ("classify_sentiment", "search_reviews", "get_reviews",
                 "aggregate_reviews", "list_experiments"):
        assert registry.get(name).requires_confirmation is False


def test_classify_accepts_a_batch_not_a_single_string():
    """批量优先：texts 是数组。让模型一次传 200 条，而不是循环调 200 次。"""
    params = registry.get("classify_sentiment").json_schema()["function"]["parameters"]
    assert params["properties"]["texts"]["type"] == "array"
    assert params["properties"]["texts"]["maxItems"] == MAX_BATCH


def test_classify_rejects_empty_batch():
    res = registry.invoke("classify_sentiment", {"texts": []})
    assert res.ok is False and res.repair_hint


def test_classify_rejects_unknown_model_name():
    res = registry.invoke("classify_sentiment", {"texts": ["还行"], "model": "gpt5"})
    assert res.ok is False and res.repair_hint


# ---------------- 检索与溯源 ----------------
def test_search_returns_review_ids(db, monkeypatch):
    monkeypatch.setattr(store, "_conn", db)
    res = registry.invoke("search_reviews", {"query": "物流 慢", "k": 3})
    assert res.ok
    reviews = res.content["reviews"]
    assert reviews, "应当检索到内容"
    assert all("review_id" in r for r in reviews), "每条结果必须带 review_id 供溯源"
    assert "物流" in reviews[0]["text"]


def test_search_can_scope_to_one_product(db, monkeypatch):
    monkeypatch.setattr(store, "_conn", db)
    res = registry.invoke("search_reviews", {"query": "包装 发货", "k": 5, "product_id": "p2"})
    assert res.ok
    assert all(r["review_id"] == 4 for r in res.content["reviews"])


def test_get_reviews_reports_missing_ids(db, monkeypatch):
    """Critic 校验引用时，编造出来的 id 必须被明确标记为不存在。"""
    monkeypatch.setattr(store, "_conn", db)
    res = registry.invoke("get_reviews", {"review_ids": [1, 99999]})
    assert res.ok
    assert [r["review_id"] for r in res.content["found"]] == [1]
    assert res.content["missing"] == [99999]


def test_search_with_no_match_returns_empty_not_error(db, monkeypatch):
    monkeypatch.setattr(store, "_conn", db)
    res = registry.invoke("search_reviews", {"query": "螺旋桨核聚变", "k": 3})
    assert res.ok and res.content["reviews"] == []


# ---------------- 副作用工具 ----------------
def test_scrape_rejects_non_url():
    res = registry.invoke("scrape_reviews", {"product_url": "不是链接"})
    assert res.ok is False and res.repair_hint


def test_train_rejects_out_of_range_epochs():
    res = registry.invoke("train_model", {"model": "cnn", "epochs": 9999})
    assert res.ok is False and res.repair_hint


# ---------------- 接本地模型 ----------------
@needs_w2v
@needs_weights
def test_classify_uses_the_local_model_and_gets_polarity_right():
    res = registry.invoke("classify_sentiment", {
        "texts": ["这个手机质量很好，用起来非常流畅",
                  "物流太慢了，包装还破损了，很失望"],
    })
    assert res.ok
    labels = [r["sentiment"] for r in res.content["results"]]
    assert labels == ["积极", "消极"], f"本地模型判定错误: {labels}"
    assert res.content["model"] == Config.default_model


@needs_w2v
@needs_weights
def test_classify_is_batched_in_one_forward_pass():
    """一次传 32 条应当明显快于 32 次单条调用的量级。"""
    import time
    texts = ["物流很慢，包装也破了"] * 32
    t0 = time.perf_counter()
    res = registry.invoke("classify_sentiment", {"texts": texts})
    elapsed = time.perf_counter() - t0
    assert res.ok and res.content["count"] == 32
    assert elapsed < 2.0, f"批量 32 条耗时 {elapsed:.2f}s，可能退化成了逐条推理"


@needs_w2v
@needs_weights
def test_aggregate_reports_distribution(db, monkeypatch):
    monkeypatch.setattr(store, "_conn", db)
    res = registry.invoke("aggregate_reviews", {"product_id": "p1", "sample_limit": 10})
    assert res.ok
    c = res.content
    assert c["analyzed"] == 3
    assert c["positive"] + c["negative"] == 3
    assert 0 <= c["positive_rate"] <= 100


def test_aggregate_on_empty_selection_is_a_clear_error(db, monkeypatch):
    monkeypatch.setattr(store, "_conn", db)
    res = registry.invoke("aggregate_reviews", {"product_id": "不存在的商品"})
    assert res.ok is False
    assert "评论库" in res.error

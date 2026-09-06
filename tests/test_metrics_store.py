# -*- coding: utf-8 -*-
"""界面上的指标必须来自真实评测，读不到就是 None —— 绝不回落到写死的数字。

原实现在读取失败时返回 {"accuracy": 0.85, "f1_score": 0.84, "recall": 0.83}，
于是前端长期展示一组从未测过的数据。
"""

import json

import pytest

import metrics_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "model_metrics.json"
    monkeypatch.setattr(metrics_store, "METRICS_PATH", str(path))
    return path


def test_returns_none_when_no_record(store):
    assert metrics_store.load_model_metrics("cnn") is None


def test_returns_none_for_unevaluated_model(store):
    metrics_store.save_model_metrics("cnn", 0.9, 0.9, 0.9, sample_count=100)
    assert metrics_store.load_model_metrics("bilstm") is None


def test_never_returns_the_old_hardcoded_placeholder(store):
    """0.85 / 0.84 / 0.83 是旧代码编造的那组数字，任何路径都不该再出现。"""
    for model in ("cnn", "lstm", "bilstm", "lstm_attention", "bilstm_attention"):
        got = metrics_store.load_model_metrics(model)
        assert got is None or (got["accuracy"], got["f1_score"], got["recall"]) != (0.85, 0.84, 0.83)


def test_roundtrip_keeps_values_and_provenance(store):
    metrics_store.save_model_metrics("cnn", 0.8949, 0.8948, 0.8949, sample_count=6335)
    got = metrics_store.load_model_metrics("cnn")
    assert got["accuracy"] == pytest.approx(0.8949)
    assert got["sampleCount"] == 6335
    assert got["dataset"] == "test"
    assert got["evaluatedAt"]  # 必须带评测时间，指标不能来路不明


def test_models_are_stored_separately(store):
    """原实现从全局 metrics_log.csv 取 iloc[-1]，拿到的可能是别的模型的结果。"""
    metrics_store.save_model_metrics("cnn", 0.90, 0.90, 0.90)
    metrics_store.save_model_metrics("lstm", 0.70, 0.70, 0.70)
    assert metrics_store.load_model_metrics("cnn")["accuracy"] == pytest.approx(0.90)
    assert metrics_store.load_model_metrics("lstm")["accuracy"] == pytest.approx(0.70)


def test_corrupt_file_degrades_to_none(store):
    store.write_text("{ not json", encoding="utf-8")
    assert metrics_store.load_model_metrics("cnn") is None


def test_file_is_human_readable(store):
    metrics_store.save_model_metrics("cnn", 0.5, 0.5, 0.5)
    assert isinstance(json.loads(store.read_text(encoding="utf-8")), dict)

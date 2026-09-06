# -*- coding: utf-8 -*-
"""端到端：登录后调用 /api/analyze，负面文本必须判成负面。

这是最贴近用户实际体验的一层回归。修复前，线上对
「物流太慢了，包装还破损了，很失望」返回的是「积极 99.997%」。
"""

import os
import time

import pytest

from config import Config
from tests.conftest import needs_weights, needs_w2v

pytestmark = [needs_w2v, needs_weights]

NEGATIVE = "物流太慢了，包装还破损了，很失望"
POSITIVE = "这个手机质量很好，用起来非常流畅"


@pytest.fixture(scope="module")
def client():
    os.environ["SHEBI_ALLOW_DEV_SECRET"] = "1"
    import app as app_module

    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    with app_module.app.app_context():
        from models import db, User
        db.create_all()
        if not User.query.filter_by(username="pytest_user").first():
            db.session.add(User(username="pytest_user", email="pytest@example.invalid",
                                password="pytest-password", is_admin=True))
            db.session.commit()

    with app_module.app.test_client() as c:
        c.post("/auth/login", data={"username": "pytest_user",
                                    "password": "pytest-password"},
               follow_redirects=True)
        yield c


def _analyze(client, text, model_type=None):
    payload = {"text": text}
    if model_type:
        payload["model_type"] = model_type
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    return resp.get_json()


def test_negative_text_is_classified_negative(client):
    data = _analyze(client, NEGATIVE)
    assert data["sentences"][0]["sentiment"] == "消极", data["sentences"][0]


def test_positive_text_is_classified_positive(client):
    data = _analyze(client, POSITIVE)
    assert data["sentences"][0]["sentiment"] == "积极", data["sentences"][0]


def test_mixed_input_does_not_collapse_to_one_label(client):
    """两句一正一负放在一起，模型不能全判成同一个标签。"""
    data = _analyze(client, f"{POSITIVE}\n{NEGATIVE}")
    labels = [s["sentiment"] for s in data["sentences"]]
    assert len(set(labels)) == 2, f"预测坍缩到单一标签: {labels}"


def test_metrics_are_real_or_absent_never_fabricated(client):
    """指标要么带着评测出处，要么是 null —— 不能是旧代码那组编造的 0.85/0.84/0.83。"""
    metrics = _analyze(client, POSITIVE)["modelMetrics"]
    if metrics is None:
        return
    assert (metrics["accuracy"], metrics["f1_score"], metrics["recall"]) != (0.85, 0.84, 0.83)
    assert metrics["evaluatedAt"], "指标必须带评测时间"
    assert metrics["sampleCount"], "指标必须带样本量"


def test_warm_requests_are_fast(client):
    """预热后的请求必须是"只做前向推理"的量级。

    改造前每个请求都重建词表(遍历 5 万行训练集)、重载 12MB 词向量、重新反序列化模型，
    实测 17.4s。阈值放在 2s 是为了容忍 CI 机器负载，真实值应在几十毫秒。
    """
    _analyze(client, POSITIVE)  # 预热：把冷启动排除在计时之外
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        _analyze(client, POSITIVE)
        samples.append(time.perf_counter() - t0)
    best = min(samples)
    assert best < 2.0, f"最快一次仍耗时 {best:.2f}s，推理资源缓存可能失效了"


def test_unknown_model_type_is_rejected(client):
    resp = client.post("/api/analyze", json={"text": POSITIVE, "model_type": "gpt5"})
    assert resp.status_code == 400


def test_empty_text_is_rejected(client):
    resp = client.post("/api/analyze", json={"text": ""})
    assert resp.status_code == 400


def test_no_temp_prediction_files_left_behind(client):
    """每个请求写自己的临时文件，用完必须删掉。"""
    _analyze(client, POSITIVE)
    leftovers = [f for f in os.listdir(Config.runtime_dir) if f.startswith("pre_")]
    assert not leftovers, f"残留临时文件: {leftovers}"

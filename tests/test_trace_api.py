# -*- coding: utf-8 -*-
"""轨迹查看接口。

重点是那个把 run_id 拼进文件路径的地方：`runtime/traces/<run_id>.json`。
不校验的话 `../../instance/users` 就能读到仓库里任何一个 .json 文件——
这个项目刚因为把用户库提交进公开仓库栽过一次，同一类错误不该再犯第二遍。
"""

import os

import pytest

from tests.conftest import needs_w2v

pytestmark = needs_w2v

TRAVERSAL = [
    "../../config",
    "..%2f..%2fconfig",
    "abc/../../x",
    "运行",
    "ABCDEF",          # 大写：文件名是小写十六进制，大写一律拒绝
    "a" * 40,          # 超长
    "",
]


@pytest.fixture(scope="module")
def client():
    os.environ["SHEBI_ALLOW_DEV_SECRET"] = "1"
    import app as app_module

    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    with app_module.app.app_context():
        from models import db, User
        db.create_all()
        if not User.query.filter_by(username="pytest_trace").first():
            db.session.add(User(username="pytest_trace", email="trace@example.invalid",
                                password="pytest-password", is_admin=True))
            db.session.commit()

    with app_module.app.test_client() as c:
        c.post("/auth/login", data={"username": "pytest_trace",
                                    "password": "pytest-password"},
               follow_redirects=True)
        yield c


def test_trace_list_returns_json(client):
    resp = client.get("/api/traces")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data["traces"], list)
    assert data["count"] == len(data["traces"])


def test_trace_page_renders(client):
    resp = client.get("/traces")
    assert resp.status_code == 200
    assert b"traces.js" in resp.data


@pytest.mark.parametrize("run_id", TRAVERSAL)
def test_malformed_run_id_is_rejected_before_touching_the_filesystem(client, run_id):
    resp = client.get(f"/api/traces/{run_id}")
    assert resp.status_code in (400, 404), \
        f"run_id={run_id!r} 返回了 {resp.status_code}"
    if resp.status_code == 400:
        assert "非法" in resp.get_json()["error"]


def test_unknown_but_wellformed_run_id_is_404(client):
    resp = client.get("/api/traces/" + "0" * 12)
    assert resp.status_code == 404


def test_saved_trace_is_readable_end_to_end(client, tmp_path, monkeypatch):
    """存进去的轨迹要能原样读回来——列表页和详情页共用这一条通路。"""
    import agent.trace as trace_mod
    from agent.trace import RunTrace, StepRecord

    monkeypatch.setattr(trace_mod, "TRACE_DIR", str(tmp_path))
    t = RunTrace(task="差评集中在哪", model="scripted")
    t.add(StepRecord(step=1, kind="llm", started_at=0.0, role="planner",
                     thought="先检索"))
    t.status, t.answer, t.finished_at = "completed", "物流慢 [review_id: 1]", 1.0
    t.save(str(tmp_path))

    resp = client.get(f"/api/traces/{t.run_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["task"] == "差评集中在哪"
    assert data["steps"][0]["role"] == "planner"

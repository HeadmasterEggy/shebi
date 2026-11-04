# -*- coding: utf-8 -*-
"""Web 接口的输入校验、权限与边界行为。不依赖模型权重。"""

import json
import os
import uuid

import pytest
import torch

import training_control as tc
from config import Config

PASSWORD = "pytest-password-1"


@pytest.fixture(scope="module")
def app_module():
    os.environ["SHEBI_ALLOW_DEV_SECRET"] = "1"
    import app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module


@pytest.fixture(scope="module")
def users(app_module):
    from models import db, User
    tag = uuid.uuid4().hex[:8]
    names = {"admin": f"adm_{tag}", "user": f"usr_{tag}", "admin2": f"adm2_{tag}"}
    with app_module.app.app_context():
        db.create_all()
        for key, name in names.items():
            db.session.add(User(username=name, email=f"{name}@example.invalid",
                                password=PASSWORD, is_admin=key.startswith("admin")))
        db.session.commit()
        ids = {k: User.query.filter_by(username=v).first().id for k, v in names.items()}
    yield {k: {"name": names[k], "id": ids[k]} for k in names}
    with app_module.app.app_context():
        for name in names.values():
            User.query.filter(User.username.like(f"%{tag}%")).delete(synchronize_session=False)
        db.session.commit()


def _login(app_module, username):
    c = app_module.app.test_client()
    resp = c.post("/auth/login", data={"username": username, "password": PASSWORD})
    assert resp.status_code == 302
    return c


@pytest.fixture
def admin(app_module, users):
    return _login(app_module, users["admin"]["name"])


@pytest.fixture
def user(app_module, users):
    return _login(app_module, users["user"]["name"])


# ---------------------------------------------------------------------------
# 鉴权：API 返回 JSON 错误码，而不是 302 到一整页 HTML
# ---------------------------------------------------------------------------
def test_anonymous_api_gets_401_json(app_module):
    resp = app_module.app.test_client().get("/api/models")
    assert resp.status_code == 401
    assert resp.get_json()["error"]


def test_anonymous_page_still_redirects_to_login(app_module):
    resp = app_module.app.test_client().get("/traces")
    assert resp.status_code == 302
    assert "/auth/login" in resp.headers["Location"]


@pytest.mark.parametrize("method,path", [
    ("post", "/api/train/start"),
    ("post", "/api/train/stop"),
    ("post", "/api/train/pause"),
    ("get", "/api/admin/users"),
    ("put", "/api/admin/update_user/1"),
])
def test_non_admin_gets_403_json(user, method, path):
    resp = getattr(user, method)(path, json={})
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "需要管理员权限"


@pytest.mark.parametrize("target", [
    "https://evil.example/x", "//evil.example/x", "/\\evil.example", "\\\\evil.example",
])
def test_login_rejects_open_redirect(app_module, users, target):
    c = app_module.app.test_client()
    resp = c.post(f"/auth/login?next={target}",
                  data={"username": users["user"]["name"], "password": PASSWORD})
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/", resp.headers["Location"]


def test_login_keeps_safe_next(app_module, users):
    c = app_module.app.test_client()
    resp = c.post("/auth/login?next=/traces",
                  data={"username": users["user"]["name"], "password": PASSWORD})
    assert resp.headers["Location"] == "/traces"


# ---------------------------------------------------------------------------
# /api/analyze：输入校验与结果对齐
# ---------------------------------------------------------------------------
class _LengthModel(torch.nn.Module):
    """非零 token 多于 2 个判正面，否则判负面。用来让不同句子拿到不同结果。"""

    def forward(self, x):
        n = (x != 0).sum(dim=1).float()
        return torch.stack([torch.full_like(n, 2.5), n], dim=1)


@pytest.fixture
def fake_model(app_module, monkeypatch):
    monkeypatch.setattr(app_module.engine, "get_model", lambda model_type=None: _LengthModel())


@pytest.mark.parametrize("payload", [None, [], "text", {"text": 123}, {"text": "好", "model_type": 1}])
def test_analyze_rejects_malformed_body(user, payload):
    resp = user.post("/api/analyze", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400


def test_analyze_rejects_oversized_input(user, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_ANALYZE_CHARS", 10)
    assert user.post("/api/analyze", json={"text": "好" * 11}).status_code == 413


def test_analyze_all_punctuation_is_400_not_garbage(user, fake_model):
    resp = user.post("/api/analyze", json={"text": "！！！\n。。。"})
    assert resp.status_code == 400
    assert resp.get_json()["skipped"] == ["！！！", "。。。"]


def test_analyze_results_stay_aligned_when_a_sentence_is_dropped(user, fake_model):
    """清洗后为空的句子被跳过时，后面的结果不能整体错位一格。

    修复前：「！！！」拿到了「质量很差」的预测结果，「质量很差」又拿到了下一句的。
    """
    long_positive = "这个手机质量很好，屏幕清晰，运行流畅，续航持久，物流迅速"
    resp = user.post("/api/analyze", json={"text": f"！！！\n质量差\n{long_positive}"})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    texts = [s["text"] for s in data["sentences"]]
    assert texts == ["质量差", long_positive]
    assert [s["sentiment"] for s in data["sentences"]] == ["消极", "积极"]
    assert data["skipped"] == ["！！！"]


def test_analyze_error_does_not_leak_internals(user, app_module, monkeypatch):
    def boom(model_type=None):
        raise RuntimeError("/secret/path/model.pkl is corrupt")
    monkeypatch.setattr(app_module.engine, "get_model", boom)
    resp = user.post("/api/analyze", json={"text": "质量很好"})
    assert resp.status_code == 500
    assert "/secret/path" not in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# 训练调度
# ---------------------------------------------------------------------------
class _FakeManager:
    def __init__(self, running=False):
        self.running = running
        self.started = 0
        self.terminated = 0

    def is_running(self):
        return self.running

    def has_handle(self):
        return True

    def returncode(self):
        return None if self.running else 1

    def start(self, args=None, script=None):
        self.args = args
        self.started += 1
        self.running = True
        return 4242

    def terminate(self, timeout=10):
        self.terminated += 1
        self.running = False
        return True


@pytest.fixture
def train_env(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "params_path", str(tmp_path / "params.json"))
    monkeypatch.setattr(Config, "progress_path", str(tmp_path / "progress.json"))
    monkeypatch.setattr(tc, "CONTROL_PATH", str(tmp_path / "control.json"))
    monkeypatch.setattr(tc, "LOG_PATH", str(tmp_path / "training.log"))
    mgr = _FakeManager()
    monkeypatch.setattr(app_module, "train_manager", mgr)
    return mgr, tmp_path


@pytest.mark.parametrize("payload", [
    {"model_type": "gpt5"},
    {"model_type": "cnn", "epochs": 0},
    {"model_type": "cnn", "epochs": "many"},
    {"model_type": "cnn", "batch_size": True},
    {"model_type": "lstm", "num_layers": 99},
    {"model_type": "cnn", "dropout": 1.5},
    {"model_type": "cnn", "optimizer": "lbfgs"},
    {"model_type": "cnn", "kernel_sizes": []},
    {"model_type": "cnn", "kernel_sizes": [0, 3]},
    {"model_type": "cnn", "kernel_sizes": "2,3"},
])
def test_train_start_validates_params(admin, train_env, payload):
    mgr, _ = train_env
    resp = admin.post("/api/train/start", json=payload)
    assert resp.status_code == 400, resp.get_json()
    assert mgr.started == 0


def test_train_start_writes_params_and_starts_once(admin, train_env):
    mgr, tmp = train_env
    resp = admin.post("/api/train/start", json={"model_type": "LSTM", "epochs": 3, "hidden_dim": 64})
    assert resp.status_code == 200, resp.get_json()
    params = json.loads((tmp / "params.json").read_text())
    assert params["model_type"] == "lstm" and params["epochs"] == 3 and params["hidden_dim"] == 64
    assert json.loads((tmp / "progress.json").read_text())["status"] == "initializing"

    assert mgr.args == ["--params", Config.params_path], "超参文件必须显式传给 main.py"

    again = admin.post("/api/train/start", json={"model_type": "cnn"})
    assert again.status_code == 409
    assert mgr.started == 1


def test_pause_toggles_via_control_file(admin, train_env):
    mgr, _ = train_env
    assert admin.post("/api/train/pause").status_code == 404   # 没在跑
    mgr.running = True
    assert admin.post("/api/train/pause").get_json() == {"status": "paused"}
    assert tc.read_command() == "pause"
    assert admin.post("/api/train/pause").get_json() == {"status": "resumed"}
    assert tc.read_command() == "resume"


def test_stop_is_graceful_by_default_and_force_terminates(admin, train_env):
    mgr, tmp = train_env
    assert admin.post("/api/train/stop").status_code == 404    # 原实现永远走到这里
    mgr.running = True
    assert admin.post("/api/train/stop").get_json() == {"status": "stopping"}
    assert tc.read_command() == "stop"
    assert mgr.terminated == 0

    resp = admin.post("/api/train/stop", json={"force": True})
    assert resp.get_json() == {"status": "stopped"}
    assert mgr.terminated == 1
    assert json.loads((tmp / "progress.json").read_text())["status"] == "stopped"


def test_progress_marks_crashed_process_as_failed(user, train_env):
    """训练进程被 kill / 崩溃后，进度不能一直停在 running。"""
    mgr, tmp = train_env
    tc.atomic_write_json(Config.progress_path, {"status": "running", "current_epoch": 2})
    data = user.get("/api/train/progress").get_json()
    assert data["status"] == "failed"
    assert "返回码" in data["error"]
    assert data["running"] is False


def test_progress_includes_log_tail(user, train_env):
    _, tmp = train_env
    (tmp / "training.log").write_text("epoch 1\nepoch 2\n", encoding="utf-8")
    data = user.get("/api/train/progress").get_json()
    assert data["recent_logs"][-1] == "epoch 2"


def test_save_model_name_cannot_escape_directory(admin, train_env, app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "model_dir", str(tmp_path / "model"))
    monkeypatch.setattr(app_module, "SAVED_MODELS_DIR", str(tmp_path / "saved"))
    os.makedirs(Config.model_dir)
    (tmp_path / "model" / "cnn_model_best.pkl").write_bytes(b"weights")
    tc.atomic_write_json(Config.params_path, {"model_type": "cnn"})

    resp = admin.post("/api/train/save_model", json={"name": "../../escaped"})
    assert resp.status_code == 200
    assert resp.get_json()["filename"] == "escaped.pkl"
    assert (tmp_path / "saved" / "escaped.pkl").read_bytes() == b"weights"
    assert not (tmp_path.parent / "escaped.pkl").exists()


# ---------------------------------------------------------------------------
# 用户管理
# ---------------------------------------------------------------------------
def test_partial_admin_update_does_not_demote_or_clear_email(admin, users, app_module):
    """只改密码时，原实现把 is_admin 默认成 False、email 改成 None。"""
    target = users["admin2"]
    resp = admin.put(f"/api/admin/update_user/{target['id']}", json={"password": "another-pass-1"})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()["user"]
    assert body["is_admin"] is True
    assert body["email"] == f"{target['name']}@example.invalid"


def test_admin_cannot_demote_self(admin, users):
    resp = admin.put(f"/api/admin/user/{users['admin']['id']}", json={"is_admin": False})
    assert resp.status_code == 400


def test_admin_update_validates_email(admin, users):
    resp = admin.put(f"/api/admin/update_user/{users['user']['id']}", json={"email": "not-an-email"})
    assert resp.status_code == 400


def test_create_user_validates_fields(admin):
    assert admin.post("/api/admin/create_user", json={
        "username": "x", "email": "x@example.invalid", "password": "long-enough-1"}).status_code == 400
    assert admin.post("/api/admin/create_user", json={
        "username": "okname", "email": "ok@example.invalid", "password": "short"}).status_code == 400
    assert admin.post("/api/admin/create_user", data="null",
                      content_type="application/json").status_code == 400


def test_password_change_keeps_email_and_requires_current(user, users, app_module):
    resp = user.post("/api/profile/update", json={"new_password": "brand-new-pass"})
    assert resp.status_code == 400

    resp = user.post("/api/profile/update", json={
        "current_password": PASSWORD, "new_password": "brand-new-pass"})
    assert resp.status_code == 200
    profile = user.get("/api/profile").get_json()
    assert profile["email"] == f"{users['user']['name']}@example.invalid"

    # 改回去，别影响同模块其他用例
    user.post("/api/profile/update", json={"current_password": "brand-new-pass", "new_password": PASSWORD})


def test_register_rejects_weak_password(app_module):
    c = app_module.app.test_client()
    name = f"reg_{uuid.uuid4().hex[:6]}"
    c.post("/auth/register", data={"username": name, "email": f"{name}@example.invalid", "password": "123"})
    from models import User
    with app_module.app.app_context():
        assert User.query.filter_by(username=name).first() is None


# ---------------------------------------------------------------------------
# 用户设置
# ---------------------------------------------------------------------------
@pytest.fixture
def settings_dir(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "USER_SETTINGS_DIR", str(tmp_path / "settings"))
    return tmp_path / "settings"


def test_settings_roundtrip_and_reset(user, settings_dir):
    assert user.get("/api/settings").get_json()["theme"] == "light"
    assert user.post("/api/settings/update", json={"theme": "dark"}).status_code == 200
    assert user.get("/api/settings").get_json()["theme"] == "dark"
    assert user.post("/api/settings/reset").status_code == 200
    assert user.get("/api/settings").get_json()["theme"] == "light"


@pytest.mark.parametrize("payload", [
    ["not", "a", "dict"],
    {"theme": {"nested": "object"}},
    {"../evil": 1},
    {"theme": "x" * 5000},
])
def test_settings_rejects_bad_payload(user, settings_dir, payload):
    resp = user.post("/api/settings/update", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 跨域与响应头
# ---------------------------------------------------------------------------
def test_no_wildcard_cors(user):
    """原实现 CORS(app) 对任意来源放行全部接口。"""
    resp = user.get("/api/models", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in resp.headers


def test_security_headers_present(app_module):
    resp = app_module.app.test_client().get("/auth/login")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"


def test_session_cookie_is_httponly_and_samesite(app_module, users):
    c = app_module.app.test_client()
    resp = c.post("/auth/login", data={"username": users["user"]["name"], "password": PASSWORD})
    cookie = resp.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


# ---------------------------------------------------------------------------
# 登录限流 / 健康检查 / 爬虫已移除
# ---------------------------------------------------------------------------
def test_login_is_throttled_after_repeated_failures(app_module, users, monkeypatch):
    from auth import LoginThrottle
    monkeypatch.setattr("auth.login_throttle", LoginThrottle(max_failures=3, window_s=600))
    c = app_module.app.test_client()
    name = users["user"]["name"]
    for _ in range(3):
        assert c.post("/auth/login", data={"username": name, "password": "wrong"}).status_code == 302
    # 第 4 次即便密码正确也被拒
    resp = c.post("/auth/login", data={"username": name, "password": PASSWORD})
    assert resp.status_code == 429


def test_successful_login_resets_failure_count():
    from auth import LoginThrottle
    t = LoginThrottle(max_failures=2, window_s=600)
    t.failure("k")
    t.reset("k")
    t.failure("k")
    assert t.retry_after("k") == 0
    t.failure("k")
    assert t.retry_after("k") > 0


def test_healthz_needs_no_login(app_module):
    resp = app_module.app.test_client().get("/healthz")
    assert resp.status_code == 200 and resp.get_json()["status"] == "ok"


def test_scraper_endpoint_is_gone(user):
    assert user.post("/api/scrape/product", json={"url": "https://item.jd.com/1.html"}).status_code == 404


def test_analyze_returns_503_when_model_untrained(user, app_module, monkeypatch):
    from utils import ModelNotAvailable

    def missing(model_type=None):
        raise ModelNotAvailable("lstm 模型尚未训练")
    monkeypatch.setattr(app_module.engine, "get_model", missing)
    resp = user.post("/api/analyze", json={"text": "质量很好", "model_type": "lstm"})
    assert resp.status_code == 503
    assert "尚未训练" in resp.get_json()["error"]


def test_train_start_passes_cnn_kernel_sizes_and_optimizer(admin, train_env):
    """前端提交的 kernel_sizes 与优化器原来被后端丢弃，选什么都按 (2,3,4) + Adam 训练。"""
    _, tmp = train_env
    resp = admin.post("/api/train/start", json={"model_type": "cnn", "kernel_sizes": [5, 2, 2],
                                                "optimizer": "SGD", "num_filters": 64})
    assert resp.status_code == 200, resp.get_json()
    params = json.loads((tmp / "params.json").read_text())
    assert params["filter_sizes"] == [2, 5]
    assert params["optimizer"] == "sgd" and params["num_filters"] == 64


def test_server_and_trainer_agree_on_optimizers(app_module):
    import main
    assert tuple(app_module.TRAIN_OPTIMIZERS) == tuple(main.OPTIMIZERS)


# ---------------------------------------------------------------------------
# 请求体大小
# ---------------------------------------------------------------------------
def test_request_body_size_is_capped(app_module):
    """原来没有 MAX_CONTENT_LENGTH：匿名请求也会把任意大的请求体整个读进内存。"""
    limit = app_module.app.config["MAX_CONTENT_LENGTH"]
    assert limit and limit <= 10 * 1024 * 1024
    c = app_module.app.test_client()
    resp = c.post("/auth/login", data=b"username=" + b"x" * (limit + 1),
                  content_type="application/x-www-form-urlencoded")
    assert resp.status_code == 413


def test_oversized_api_request_gets_json_413(user, app_module):
    limit = app_module.app.config["MAX_CONTENT_LENGTH"]
    resp = user.post("/api/analyze", data=b'{"text":"' + b"a" * limit + b'"}',
                     content_type="application/json")
    assert resp.status_code == 413
    assert "请求体过大" in resp.get_json()["error"]


def test_max_length_text_still_fits_under_body_cap(user, app_module, monkeypatch):
    """上限内最长的分析文本（全是三字节的汉字）必须能通过请求体上限，到达文本长度校验。"""
    monkeypatch.setattr(app_module.engine, "get_model",
                        lambda model_type=None: (_ for _ in ()).throw(RuntimeError("stop here")))
    text = "好" * app_module.MAX_ANALYZE_CHARS
    resp = user.post("/api/analyze", json={"text": text})
    assert resp.status_code != 413, "合法长度的文本被请求体上限误拦"


def test_traces_are_admin_only(user, admin):
    """注册是开放的：普通用户不能读到运营者的 agent 运行轨迹（问题、工具参数、检索到的评论原文）。"""
    assert user.get("/api/traces").status_code == 403
    assert user.get("/api/traces/abcdef123456").status_code == 403
    assert user.get("/traces").status_code == 302
    assert admin.get("/api/traces").status_code == 200

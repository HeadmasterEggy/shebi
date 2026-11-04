# -*- coding: utf-8 -*-
"""改密码必须作废此前签发的会话与"记住我" Cookie。

原来 Flask-Login 只在会话里存用户 id：账号被盗后改密码，对方已登录的会话与
365 天有效的"记住我" Cookie 照样能用。
"""

import os
import uuid

import pytest

PW, NEW_PW = "original-pass-1", "brand-new-pass-2"


@pytest.fixture(scope="module")
def flask_app():
    os.environ["SHEBI_ALLOW_DEV_SECRET"] = "1"
    import app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture
def account(flask_app):
    from models import db, User
    names = {"user": f"sr_{uuid.uuid4().hex[:8]}", "admin": f"sra_{uuid.uuid4().hex[:8]}"}
    with flask_app.app_context():
        db.create_all()
        db.session.add(User(names["user"], f"{names['user']}@example.invalid", PW))
        db.session.add(User(names["admin"], f"{names['admin']}@example.invalid", PW, is_admin=True))
        db.session.commit()
        ids = {k: User.query.filter_by(username=v).first().id for k, v in names.items()}
    yield {k: {"name": names[k], "id": ids[k]} for k in names}
    with flask_app.app_context():
        for v in names.values():
            User.query.filter_by(username=v).delete()
        db.session.commit()


def _login(flask_app, name, password=PW, remember=False):
    c = flask_app.test_client()
    data = {"username": name, "password": password}
    if remember:
        data["remember"] = "on"
    assert c.post("/auth/login", data=data).status_code == 302
    return c


def _logged_in(client):
    return client.get("/api/profile").status_code == 200


def test_password_change_revokes_other_sessions(flask_app, account):
    name = account["user"]["name"]
    other, me = _login(flask_app, name), _login(flask_app, name)
    assert _logged_in(other)

    resp = me.post("/api/profile/update", json={"current_password": PW, "new_password": NEW_PW})
    assert resp.status_code == 200
    assert not _logged_in(other), "改密码后其他会话仍然有效"
    assert _logged_in(me), "改密码的那个会话应当被重新签发、保持登录"


def test_password_change_revokes_remember_me_cookie(flask_app, account):
    name = account["user"]["name"]
    stolen = _login(flask_app, name, remember=True)
    stolen.delete_cookie("session")            # 只剩"记住我" Cookie
    assert _logged_in(stolen), "前提：记住我 Cookie 本身可以恢复登录"
    stolen.delete_cookie("session")

    me = _login(flask_app, name)
    me.post("/api/profile/update", json={"current_password": PW, "new_password": NEW_PW})
    assert not _logged_in(stolen), "改密码后，旧的记住我 Cookie 仍能登录"


def test_admin_reset_revokes_target_sessions_but_not_own(flask_app, account):
    victim = _login(flask_app, account["user"]["name"])
    admin = _login(flask_app, account["admin"]["name"])
    assert admin.put(f"/api/admin/update_user/{account['user']['id']}",
                     json={"password": NEW_PW}).status_code == 200
    assert not _logged_in(victim)
    assert _logged_in(_login(flask_app, account["user"]["name"], NEW_PW))

    assert admin.put(f"/api/admin/update_user/{account['admin']['id']}",
                     json={"password": NEW_PW}).status_code == 200
    assert _logged_in(admin), "管理员改自己的密码后不应被踢下线"


def test_legacy_id_only_session_is_rejected(flask_app, account):
    """旧格式（只存 id）的会话无法证明密码未变，一律要求重新登录。"""
    from auth import load_user
    with flask_app.app_context():
        assert load_user(str(account["user"]["id"])) is None
        assert load_user(f"{account['user']['id']}:deadbeefdeadbeef") is None
        assert load_user("not-a-number:abc") is None

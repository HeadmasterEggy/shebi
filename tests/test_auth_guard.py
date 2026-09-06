# -*- coding: utf-8 -*-
"""未登录访问受保护接口必须被挡下。

单独成一个模块：test_api_analyze.py 里有一个模块级的已登录 client fixture，
它活跃期间 Flask-Login 会在应用上下文里留下当前用户，套一个新的 test_client
也拿不到干净的匿名状态，测出来的结果没有意义。
"""

import os

import pytest

from tests.conftest import needs_w2v

pytestmark = needs_w2v

PROTECTED = [
    ("/api/analyze", "post"),
    ("/api/models", "get"),
    ("/api/user", "get"),
    ("/api/admin/users", "get"),
]


@pytest.fixture(scope="module")
def anon_client():
    os.environ["SHEBI_ALLOW_DEV_SECRET"] = "1"
    import app as app_module
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


@pytest.mark.parametrize("path,method", PROTECTED)
def test_protected_endpoints_reject_anonymous(anon_client, path, method):
    resp = getattr(anon_client, method)(path, json={"text": "很好"})
    assert resp.status_code in (301, 302, 401, 403), \
        f"{method.upper()} {path} 对未登录用户返回了 {resp.status_code}"

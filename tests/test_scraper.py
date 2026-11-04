# -*- coding: utf-8 -*-
"""爬虫：URL 白名单、解析、去重入库、并发与鉴权。全程不启动浏览器。"""

import json
import os
import threading

import pytest

import scraper_api as sc
from agent import store


def _payload(*texts, floor="flo_11103"):
    return {"result": {"floors": [{"floorId": floor, "data": [
        {"commentInfo": {"commentData": t, "commentScore": "5", "commentDate": "2026-01-01",
                         "userNickName": "u"}} for t in texts]}]}}


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url,pid", [
    ("https://item.jd.com/100008348542.html", "100008348542"),
    ("http://item.jd.com/100008348542.html?spm=x#comment", "100008348542"),
    ("https://item.m.jd.com/product/100008348542.html", "100008348542"),
    ("100008348542", "100008348542"),
])
def test_accepts_jd_product_urls(url, pid):
    assert sc.extract_product_id(url) == pid


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/1234.html",   # 云元数据
    "http://127.0.0.1:5003/1234.html",
    "https://item.jd.com.evil.example/1234.html",           # 后缀伪装
    "https://evil.example/item.jd.com/1234.html",
    "https://user@item.jd.com/1234.html",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "https://item.jd.com/../admin.html",
    "",
    None,
    12345,
])
def test_rejects_non_jd_urls(url):
    """原实现把用户给的 URL 原样交给浏览器打开，且接口不需要登录——匿名 SSRF。"""
    assert sc.extract_product_id(url) is None


def test_browser_only_ever_opens_canonical_url():
    assert sc.product_url("100008348542") == "https://item.jd.com/100008348542.html"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def test_parse_extracts_dedupes_and_normalises_whitespace():
    payloads = [
        json.dumps(_payload("质量很好\n物流快", "一般")),
        _payload("一般", "包装破损"),                 # 重复的「一般」只保留一次
        _payload("别的楼层", floor="flo_other"),
        "not json", b"\xff", None, 42, {"result": None},
    ]
    comments = sc.parse_comment_payloads(payloads)
    assert [c["text"] for c in comments] == ["质量很好 物流快", "一般", "包装破损"]
    assert comments[0]["score"] == 5 and comments[0]["date"] == "2026-01-01"


def test_parse_tolerates_bad_score():
    p = _payload("好")
    p["result"]["floors"][0]["data"][0]["commentInfo"]["commentScore"] = "五星"
    assert sc.parse_comment_payloads([p])[0]["score"] is None


# ---------------------------------------------------------------------------
# 抓取编排
# ---------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SCRAPE_DIR", str(tmp_path / "scraped"))
    conn = store.connect(str(tmp_path / "reviews.db"))
    monkeypatch.setattr(store, "_conn", conn)
    return conn


def test_scrape_saves_and_ingests_without_duplicates(env):
    fetcher = lambda pid, rounds: [_payload("屏幕清晰", "续航差")]
    first = sc.scrape("1234567", fetcher=fetcher, use_cache=False)
    assert first["count"] == 2 and len(first["new_review_ids"]) == 2
    assert store.count(env, source="jd:1234567") == 2
    assert os.path.exists(os.path.join(sc.SCRAPE_DIR, "1234567.csv"))

    fetcher2 = lambda pid, rounds: [_payload("续航差", "发货快")]
    second = sc.scrape("1234567", fetcher=fetcher2, use_cache=False)
    assert len(second["new_review_ids"]) == 1, "重复抓取同一商品不应产生重复行"
    assert store.count(env, source="jd:1234567") == 3


def test_scrape_uses_fresh_cache(env):
    calls = []

    def fetcher(pid, rounds):
        calls.append(pid)
        return [_payload("好用")]

    sc.scrape("7654321", fetcher=fetcher)
    again = sc.scrape("7654321", fetcher=fetcher)
    assert again["cached"] is True and again["count"] == 1
    assert calls == ["7654321"]


def test_only_one_scrape_at_a_time(env):
    started, release = threading.Event(), threading.Event()

    def slow(pid, rounds):
        started.set()
        release.wait(5)
        return []

    t = threading.Thread(target=sc.scrape, args=("1111111",), kwargs={"fetcher": slow, "use_cache": False})
    t.start()
    started.wait(5)
    try:
        with pytest.raises(sc.ScraperBusy):
            sc.scrape("2222222", fetcher=slow, use_cache=False)
    finally:
        release.set()
        t.join()


def test_lock_is_released_when_fetch_fails(env):
    def boom(pid, rounds):
        raise RuntimeError("chrome crashed")

    with pytest.raises(RuntimeError):
        sc.scrape("3333333", fetcher=boom, use_cache=False)
    assert sc.scrape("3333333", fetcher=lambda p, r: [], use_cache=False)["count"] == 0


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flask_app():
    os.environ["SHEBI_ALLOW_DEV_SECRET"] = "1"
    import app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module.app


def test_endpoint_requires_login(flask_app):
    resp = flask_app.test_client().post("/api/scrape/product",
                                        json={"url": "https://item.jd.com/100008348542.html"})
    assert resp.status_code == 401


def test_endpoint_rejects_foreign_url_before_touching_browser(flask_app, monkeypatch):
    import uuid
    from models import db, User

    def never(*a, **k):
        raise AssertionError("不该走到抓取这一步")

    monkeypatch.setattr(sc, "scrape", never)
    name = f"scr_{uuid.uuid4().hex[:8]}"
    with flask_app.app_context():
        db.create_all()
        db.session.add(User(username=name, email=f"{name}@example.invalid", password="pytest-password-1"))
        db.session.commit()
    c = flask_app.test_client()
    c.post("/auth/login", data={"username": name, "password": "pytest-password-1"})
    try:
        resp = c.post("/api/scrape/product", json={"url": "http://169.254.169.254/1234.html"})
        assert resp.status_code == 400
    finally:
        with flask_app.app_context():
            User.query.filter_by(username=name).delete()
            db.session.commit()

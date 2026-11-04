# -*- coding: utf-8 -*-
"""京东商品评论爬虫。

原实现的问题：

- ``/api/scrape/product`` **不需要登录**，并且把用户给的 URL 原样交给一个真实的
  Chromium 去打开——任何匿名访问者都能让服务器的浏览器去访问任意地址
  （内网管理后台、云厂商元数据接口 169.254.169.254 ……），典型的 SSRF。
- 浏览器在异常路径上不关闭，每失败一次泄漏一组 Chrome 进程；也没有并发限制。
- 模块导入时就在当前工作目录下建 ``csv/ txt/ cache/`` 三个目录、并调用
  ``logging.basicConfig`` 改写全局日志配置。
- 抓下来的评论只写 CSV，agent 的评论库根本看不到，``scrape_reviews`` 工具只能报错。

现在拆成三层：URL 规范化 → 浏览器抓取（唯一有副作用、需要 DrissionPage 的部分）
→ 纯函数解析。抓取结果写入 ``runtime/`` 并去重入库，供 agent 检索。
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request
from flask_login import login_required

from config import Config
from training_control import atomic_write_json, read_json

logger = logging.getLogger(__name__)

scraper_bp = Blueprint('scraper', __name__, url_prefix='/api/scrape')

SCRAPE_DIR = os.path.join(Config.runtime_dir, "scraped")
CACHE_TTL_S = 86400
COMMENT_API = "https://api.m.jd.com/client.action"
COMMENT_FLOOR_ID = "flo_11103"

# 允许的商品页域名。只接受京东商品页，并且真正交给浏览器的 URL 是用商品 ID
# 重新拼出来的，用户传入的 URL 本身永远不会被打开。
_ALLOWED_HOSTS = {"item.jd.com", "item.m.jd.com", "npcitem.jd.hk", "item.jd.hk"}
_PID_RE = re.compile(r"^\d{4,20}$")

_scrape_lock = threading.Lock()


class ScrapeError(Exception):
    """可以原样展示给用户的抓取错误。"""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ScraperBusy(ScrapeError):
    def __init__(self):
        super().__init__("已有抓取任务在运行，请稍后再试", status=409)


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------
def extract_product_id(url: str) -> Optional[str]:
    """从京东商品链接（或纯数字商品 ID）中取出商品 ID；不是京东商品页返回 None。"""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if _PID_RE.match(url):
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        return None
    if (parsed.hostname or "").lower() not in _ALLOWED_HOSTS:
        return None
    match = re.fullmatch(r"/(?:product/)?(\d{4,20})\.html", parsed.path or "")
    return match.group(1) if match else None


def product_url(pid: str) -> str:
    return f"https://item.jd.com/{pid}.html"


# ---------------------------------------------------------------------------
# 解析（纯函数，不依赖浏览器）
# ---------------------------------------------------------------------------
def parse_comment_payloads(payloads: Iterable[Any]) -> List[Dict[str, Any]]:
    """从评论接口的响应体里取出评论，按文本去重、保持出现顺序。"""
    comments, seen = [], set()
    for data in payloads:
        if isinstance(data, (str, bytes)):
            try:
                data = json.loads(data)
            except ValueError:
                continue
        if not isinstance(data, dict):
            continue
        floors = (data.get("result") or {}).get("floors") or []
        for floor in floors:
            if not isinstance(floor, dict) or floor.get("floorId") != COMMENT_FLOOR_ID:
                continue
            for item in floor.get("data") or []:
                info = (item or {}).get("commentInfo") or {}
                text = " ".join(str(info.get("commentData") or "").split())
                if not text or text in seen:
                    continue
                seen.add(text)
                try:
                    score = int(info.get("commentScore"))
                except (TypeError, ValueError):
                    score = None
                comments.append({
                    "text": text,
                    "date": info.get("commentDate") or "",
                    "score": score,
                    "nickname": info.get("userNickName") or "",
                })
    return comments


# ---------------------------------------------------------------------------
# 浏览器抓取
# ---------------------------------------------------------------------------
def fetch_comment_payloads(pid: str, rounds: int = 8) -> List[Any]:
    """打开商品页、点开评价区并滚动，收集评论接口的响应体。"""
    try:
        from DrissionPage import ChromiumPage
    except ImportError as e:  # 依赖缺失不应该让整个 Web 服务起不来
        raise ScrapeError("服务器未安装 DrissionPage，无法抓取", status=503) from e

    page = None
    payloads = []
    try:
        page = ChromiumPage()
        page.listen.start(COMMENT_API)
        page.get(product_url(pid))

        for selector in ("#comment", "text=商品评价", "text=全部评价"):
            try:
                btn = page.ele(selector, timeout=2)
                if btn:
                    btn.click()
                    time.sleep(2)
            except Exception as e:  # 页面改版时按钮可能不存在，继续滚动即可
                logger.debug("点击 %s 失败: %s", selector, e)

        page.listen.wait(count=5, timeout=8)
        for _ in range(rounds):
            cards = page.eles(".jdc-pc-rate-card") or page.eles(".comment-item")
            if cards:
                cards[-1].hover()
                cards[-1].scroll(800)
                time.sleep(2)
            resp = page.listen.wait(timeout=3)
            body = getattr(getattr(resp, "response", None), "body", None)
            if body:
                payloads.append(body)
    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:
                logger.warning("关闭浏览器失败", exc_info=True)
    return payloads


# ---------------------------------------------------------------------------
# 落盘与入库
# ---------------------------------------------------------------------------
def _cache_path(pid):
    return os.path.join(SCRAPE_DIR, f"{pid}.json")


def load_cached(pid: str, ttl: float = CACHE_TTL_S) -> Optional[List[Dict[str, Any]]]:
    data = read_json(_cache_path(pid))
    if not isinstance(data, dict) or time.time() - data.get("timestamp", 0) >= ttl:
        return None
    comments = data.get("comments")
    return comments if isinstance(comments, list) else None


def save_results(pid: str, comments: List[Dict[str, Any]]) -> None:
    """JSON 作缓存与原始记录；CSV 供人看、可直接拖进 Excel。"""
    os.makedirs(SCRAPE_DIR, exist_ok=True)
    atomic_write_json(_cache_path(pid), {"timestamp": time.time(), "comments": comments})

    import csv
    with open(os.path.join(SCRAPE_DIR, f"{pid}.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["commentDate", "commentData", "commentScore", "userNickName"])
        for c in comments:
            writer.writerow([c.get("date", ""), c["text"], c.get("score", ""), c.get("nickname", "")])


def ingest(pid: str, comments: List[Dict[str, Any]], conn=None) -> List[int]:
    """去重后写入 agent 评论库，返回新写入的 review_id。重复抓取同一商品不会产生重复行。"""
    from agent import store

    conn = conn or store.get_conn()
    source = f"jd:{pid}"
    existing = {row[0] for row in conn.execute("SELECT text FROM reviews WHERE source=?", (source,))}
    rows = [
        {"source": source, "product_id": pid, "text": c["text"],
         "score": c.get("score"), "created_at": c.get("date") or None}
        for c in comments if c["text"] not in existing
    ]
    return store.add_reviews(rows, conn=conn)


def scrape(pid: str, rounds: int = 8, use_cache: bool = True, fetcher=None) -> Dict[str, Any]:
    """抓取一个商品的评论。同一时刻只允许一个抓取任务（每个任务占一整个浏览器）。"""
    if not _PID_RE.match(pid or ""):
        raise ScrapeError("商品 ID 不合法")

    if use_cache:
        cached = load_cached(pid)
        if cached is not None:
            return {"product_id": pid, "comments": cached, "cached": True,
                    "count": len(cached), "new_review_ids": []}

    if not _scrape_lock.acquire(blocking=False):
        raise ScraperBusy()
    try:
        payloads = (fetcher or fetch_comment_payloads)(pid, rounds)
    finally:
        _scrape_lock.release()

    comments = parse_comment_payloads(payloads)
    new_ids = []
    if comments:
        save_results(pid, comments)
        new_ids = ingest(pid, comments)
    logger.info("商品 %s：抓到 %d 条评论，新入库 %d 条", pid, len(comments), len(new_ids))
    return {"product_id": pid, "comments": comments, "cached": False,
            "count": len(comments), "new_review_ids": new_ids}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@scraper_bp.route('/product', methods=['POST'])
@login_required
def scrape_product():
    """抓取商品评论。只接受京东商品链接。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    pid = extract_product_id(data.get('url'))
    if not pid:
        return jsonify({'error': '请输入京东商品链接，例如 https://item.jd.com/100008348542.html'}), 400

    try:
        result = scrape(pid)
    except ScrapeError as e:
        return jsonify({'error': str(e)}), e.status
    except Exception:
        logger.exception('抓取商品 %s 失败', pid)
        return jsonify({'error': '抓取失败，详见服务端日志'}), 502

    return jsonify({
        'product_id': pid,
        'comments': [{'text': c['text']} for c in result['comments']],
        'count': result['count'],
        'cached': result['cached'],
        'new_reviews': len(result['new_review_ids']),
    })

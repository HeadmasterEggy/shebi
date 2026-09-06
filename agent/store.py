# -*- coding: utf-8 -*-
"""评论存储。

Agent 的结论必须能追溯到具体某一条评论，所以每条评论都要有稳定 id。
底层用 SQLite，检索先走轻量的词重合打分——够 agent 用，也不引入向量库依赖；
后续接 embedding 时只需替换 search() 的实现，接口不变。
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional

from config import Config

DB_PATH = os.path.join(Config.runtime_dir, "reviews.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,           -- dataset / jd:<商品id>
    product_id  TEXT,
    text        TEXT NOT NULL,
    tokens      TEXT,                    -- 空格分词，供检索打分
    score       INTEGER,                 -- 原始星级，没有则为空
    gold_label  INTEGER,                 -- 数据集自带标签：0 消极 / 1 积极
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_source ON reviews(source);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
"""

_lock = threading.Lock()


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = connect()
    return _conn


def db_path(conn: Optional[sqlite3.Connection] = None) -> str:
    """这个连接实际连的是哪个文件。

    向量索引必须和它索引的那个库绑定——测试用临时库、生产用 runtime/reviews.db，
    索引要是认不出自己索的是谁，就会拿 A 库的 id 去 B 库里查原文。
    """
    conn = conn or get_conn()
    row = conn.execute("PRAGMA database_list").fetchone()
    return (row[2] if row else "") or ":memory:"


def count(conn: Optional[sqlite3.Connection] = None, source: Optional[str] = None) -> int:
    conn = conn or get_conn()
    if source:
        return conn.execute("SELECT COUNT(*) FROM reviews WHERE source=?", (source,)).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]


def add_reviews(rows: Iterable[Dict[str, Any]], conn: Optional[sqlite3.Connection] = None) -> List[int]:
    """写入评论，返回新增的 id 列表。"""
    conn = conn or get_conn()
    ids = []
    with _lock:
        cur = conn.cursor()
        for r in rows:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            cur.execute(
                "INSERT INTO reviews(source, product_id, text, tokens, score, gold_label, created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (r.get("source", "unknown"), r.get("product_id"), text,
                 r.get("tokens") or _tokenize(text), r.get("score"),
                 r.get("gold_label"), r.get("created_at")),
            )
            ids.append(cur.lastrowid)
        conn.commit()
    return ids


def get(review_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    conn = conn or get_conn()
    row = conn.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
    return dict(row) if row else None


def fetch(limit: int = 100, source: Optional[str] = None, product_id: Optional[str] = None,
          conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    conn = conn or get_conn()
    sql, params = "SELECT * FROM reviews WHERE 1=1", []
    if source:
        sql += " AND source=?"
        params.append(source)
    if product_id:
        sql += " AND product_id=?"
        params.append(product_id)
    sql += " ORDER BY id LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params)]


_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z]+|\d+")


def _tokenize(text: str) -> str:
    """轻量分词：中文按字、英文数字按串。

    检索打分用不着 jieba 的精度，按字反而对短查询更鲁棒；
    真正的语义检索留给后续的 embedding 版本。
    """
    return " ".join(_TOKEN_RE.findall(text))


def search(query: str, k: int = 5, product_id: Optional[str] = None,
           conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """按查询词与评论的词重合度打分，返回 top-k（含 id，便于溯源）。"""
    conn = conn or get_conn()
    q_tokens = set(_tokenize(query).split())
    if not q_tokens:
        return []

    sql, params = "SELECT id, text, tokens, score, gold_label, source, product_id FROM reviews", []
    if product_id:
        sql += " WHERE product_id=?"
        params.append(product_id)

    scored = []
    for row in conn.execute(sql, params):
        tokens = set((row["tokens"] or "").split())
        if not tokens:
            continue
        overlap = len(q_tokens & tokens)
        if not overlap:
            continue
        # 对长评论做一点长度惩罚，避免"什么词都包含"的长文霸榜
        rel = overlap / (len(q_tokens) ** 0.5 * (len(tokens) ** 0.35 + 1e-9))
        scored.append((rel, dict(row)))

    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    out = []
    for rel, row in scored[:k]:
        row.pop("tokens", None)
        row["relevance"] = round(rel, 4)
        out.append(row)
    return out


def seed_from_dataset(path: Optional[str] = None, limit: int = 2000,
                      conn: Optional[sqlite3.Connection] = None) -> int:
    """用数据集里的真实电商评论把库填上，让检索与统计工具离线可用。

    数据集是已分词的「标签 空格分词文本」格式，这里还原成可读文本。
    """
    conn = conn or get_conn()
    path = path or Config.test_path
    source = "dataset"
    if count(conn, source) > 0:
        return 0

    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label = int(parts[0])
            text = "".join(parts[1:])
            rows.append({"source": source, "text": text, "gold_label": label,
                         "product_id": "dataset"})
    return len(add_reviews(rows, conn))

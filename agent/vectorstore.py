# -*- coding: utf-8 -*-
"""评论向量索引与混合检索。

L4 的检索层。规模是几千条评论、50 维向量——numpy 暴力算要 1ms 量级，
上 FAISS / Milvus 纯属给简历凑名词，还要解释为什么引入一个跑不满的依赖。

**这里有一个被实测推翻的设计。** 最初的版本是常规做法：把每条评论压成一个
句向量，查询也压成一个，比余弦。跑起来发现排序是错的——查「续航很差」，
排第一的是「运行很流畅，玩游戏一点都不卡」（0.74），真正该命中的
「一天要充三次电，出门必须带充电宝」掉到第 5。原因是 50 维词向量做均值
池化时，一句话里所有词平权，内容词被功能词摊平了。

改成词级 MaxSim（ColBERT 的思路）：查询里的每个词去评论里找最像的那个词，
再按 SIF 权重加权平均。

    score(q, d) = Σ_i  w_i · max_j cos(q_i, d_j)  /  Σ_i w_i

同一批样本上，「续航很差」的目标从第 5 名回到第 1 名，其余查询不变。

**关于下限，有一件必须说清楚的事：MaxSim 的绝对分随语料规模饱和。**
10 条评论的小库上，相关查询最低 0.550、无关查询（"红烧肉的家常做法"）
最高 0.466，看起来 0.50 是条干净的分界线；换到 2000 条的真实库上，同样
那句"红烧肉的家常做法"能拿到 0.804——因为语料一大，任何一个词都能在
某条评论里找到一个凑合的最近邻。

试过用"查询里权重最高的那个词有没有被命中"当闸门（相关查询全是 1.000，
无关查询 0.79 / 0.66，分得很开），但那等于要求核心词字面出现，会把
"续航"→"一天要充三次电"这类释义召回一起挡掉——而那正是这一层存在的理由。

所以结论是：**语义召回不承担主题过滤的职责**，下限只用来挡退化匹配
（全部词都不在词表里的查询会得到零向量，一条都返回不了）。
保证结论有据的是 Critic 那道确定性闸门，不是检索的相似度阈值。
把这两件事混在一个数字上，两边都做不好。

代价是索引要存词级向量而不是句向量：2000 条评论约 3 万个词、50 维、
float32，6MB 上下，可以接受。到十万条以上（约 300MB）就该改成
"先便宜召回若干条、再对候选做 MaxSim 重排"的两段式，接口不用变。

**混合检索**：词重合认字不认意思（查「续航差」召不回「一天要充三次电」），
向量认意思不认字（查具体型号、错别字时不如字面匹配）。两路用 RRF
（Reciprocal Rank Fusion）融合：score = Σ 1/(K + rank)。RRF 只看名次不看分数，
所以不需要把两路量纲不同的分数归一化——这是它比加权求和省心的地方。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent import store
from agent.embedding import encoder as default_encoder
from config import Config

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = os.path.join(Config.runtime_dir, "review_vectors.npz")

# RRF 的平滑常数，原论文取 60：名次越靠后贡献越小，但不至于断崖
RRF_K = 60
# MaxSim 下限。只用来挡退化匹配，不是主题过滤器——理由见模块 docstring。
MIN_SIMILARITY = 0.50


@dataclass
class VectorIndex:
    """词级向量索引，CSR 式布局。

    tokens 是所有评论的词向量首尾相接拼成的一张大表，offsets 记录每条评论
    占用的区间。这样查询时只做一次矩阵乘法，再按区间取最大值，
    不用为每条评论单独开一个数组。
    """
    ids: np.ndarray                 # (n_docs,)
    tokens: np.ndarray              # (n_tokens, dim)，已 L2 归一化
    offsets: np.ndarray             # (n_docs + 1,)

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.tokens.shape[1]) if self.tokens.size else 0

    def search(self, q_vecs: np.ndarray, q_weights: np.ndarray, k: int,
               allowed: Optional[np.ndarray] = None,
               min_similarity: float = MIN_SIMILARITY) -> List[Tuple[int, float]]:
        """MaxSim 打分，返回 [(review_id, score), ...] 按分降序。"""
        if self.size == 0 or q_vecs.shape[0] == 0:
            return []

        # (n_tokens, q) 的词对词相似度，再按评论区间取每个查询词的最大值
        sims = self.tokens @ q_vecs.T
        best = np.maximum.reduceat(sims, self.offsets[:-1], axis=0)   # (n_docs, q)
        total = float(q_weights.sum())
        if total <= 1e-8:
            return []
        scores = (best * q_weights).sum(axis=1) / total

        if allowed is not None:
            scores = np.where(np.isin(self.ids, allowed), scores, -1.0)

        k = min(k, self.size)
        top = np.argpartition(-scores, k - 1)[:k] if k < self.size else np.arange(self.size)
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top
                if scores[i] >= min_similarity]


# 按库文件分开缓存。同一个进程里同时开着生产库和测试库是常态，
# 一个全局索引会让 A 库的 review_id 被拿去 B 库里查原文。
_indexes: Dict[str, VectorIndex] = {}
_lock = threading.RLock()


def index_path_for(db: str) -> Optional[str]:
    """索引文件就放在它所索引的库旁边，同名不同后缀，一眼能看出配对关系。"""
    if not db or db == ":memory:":
        return None            # 内存库不落盘，进程结束就该消失
    if os.path.abspath(db) == os.path.abspath(store.DB_PATH):
        return DEFAULT_INDEX_PATH
    return os.path.splitext(db)[0] + ".vectors.npz"


def save(index: VectorIndex, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, ids=index.ids, tokens=index.tokens, offsets=index.offsets)
    return path


def load(path: Optional[str]) -> Optional[VectorIndex]:
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return VectorIndex(ids=data["ids"], tokens=data["tokens"], offsets=data["offsets"])
    except (OSError, ValueError, KeyError):
        logger.warning("向量索引损坏或格式过时，将重建：%s", path)
        return None


def _encode_docs(rows: List[Dict[str, Any]], enc) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把若干评论编码成 (ids, tokens, lengths)。

    一个词都不在词表里的评论会被整条丢掉：它在 MaxSim 下和任何查询的分数
    都是 0，留在索引里只是白占一个区间，还会让 reduceat 遇到空段。
    """
    ids, mats, lengths = [], [], []
    for row in rows:
        vecs, _ = enc.weighted_vectors(row["text"])
        if vecs.shape[0] == 0:
            continue
        ids.append(row["id"])
        mats.append(vecs)
        lengths.append(vecs.shape[0])
    if not ids:
        return (np.zeros(0, dtype=np.int64),
                np.zeros((0, enc.dim), dtype=np.float32),
                np.zeros(0, dtype=np.int64))
    return (np.array(ids, dtype=np.int64),
            np.vstack(mats).astype(np.float32),
            np.array(lengths, dtype=np.int64))


def _offsets(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def build(conn: Optional[sqlite3.Connection] = None, encoder=None,
          path: Optional[str] = None) -> VectorIndex:
    """全量重建：编码所有评论的词向量，落盘。"""
    enc = encoder or default_encoder
    conn = conn or store.get_conn()
    if path is None:
        path = index_path_for(store.db_path(conn))

    rows = [dict(r) for r in conn.execute("SELECT id, text FROM reviews ORDER BY id")]
    if rows:
        logger.info("正在编码 %d 条评论……", len(rows))
    ids, tokens, lengths = _encode_docs(rows, enc)
    index = VectorIndex(ids=ids, tokens=tokens, offsets=_offsets(lengths))
    if path:
        save(index, path)
    return index


def get_index(conn: Optional[sqlite3.Connection] = None, encoder=None) -> VectorIndex:
    """取该库的索引；没有就建，库里有新评论就增量补上。"""
    enc = encoder or default_encoder
    conn = conn or store.get_conn()
    db = store.db_path(conn)
    path = index_path_for(db)

    with _lock:
        index = _indexes.get(db)
        if index is None:
            index = load(path) or build(conn, enc, path)
            _indexes[db] = index
        _sync(index, conn, enc, path)
        return index


def _sync(index: VectorIndex, conn: sqlite3.Connection, enc,
          path: Optional[str]) -> None:
    """增量：只编码索引里还没有的评论 id。"""
    known = set(index.ids.tolist())
    rows = [dict(r) for r in conn.execute("SELECT id, text FROM reviews ORDER BY id")
            if r["id"] not in known]
    if not rows:
        return

    logger.info("增量编码 %d 条新评论", len(rows))
    ids, tokens, lengths = _encode_docs(rows, enc)
    if ids.size == 0:
        return
    index.ids = np.concatenate([index.ids, ids])
    index.tokens = np.vstack([index.tokens, tokens]) if index.tokens.size else tokens
    index.offsets = _offsets(np.concatenate([np.diff(index.offsets), lengths]))
    if path:
        save(index, path)


def reset(db: Optional[str] = None) -> None:
    """丢掉进程内缓存的索引。测试与「换了一个库」时用。"""
    with _lock:
        if db is None:
            _indexes.clear()
        else:
            _indexes.pop(db, None)


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def semantic_search(query: str, k: int = 5, product_id: Optional[str] = None,
                    conn: Optional[sqlite3.Connection] = None, encoder=None,
                    min_similarity: float = MIN_SIMILARITY) -> List[Dict[str, Any]]:
    enc = encoder or default_encoder
    conn = conn or store.get_conn()
    index = get_index(conn, enc)
    if index.size == 0:
        return []

    allowed = None
    if product_id is not None:
        allowed = np.array(
            [r["id"] for r in conn.execute(
                "SELECT id FROM reviews WHERE product_id=?", (product_id,))],
            dtype=np.int64)
        if allowed.size == 0:
            return []

    q_vecs, q_weights = enc.weighted_vectors(query)
    hits = index.search(q_vecs, q_weights, k=k, allowed=allowed,
                        min_similarity=min_similarity)
    return [_row(conn, rid, similarity=round(score, 4)) for rid, score in hits]


def hybrid_search(query: str, k: int = 5, product_id: Optional[str] = None,
                  conn: Optional[sqlite3.Connection] = None, encoder=None,
                  min_similarity: float = MIN_SIMILARITY,
                  depth_factor: int = 3) -> List[Dict[str, Any]]:
    """词重合 + 向量两路召回，RRF 融合。

    两路各取 k*depth_factor 条再融合：只取 k 条的话，一个文档在某一路排在
    第 k+1 名就完全失去了被融合抬上来的机会。
    """
    conn = conn or store.get_conn()
    depth = max(k * depth_factor, k)

    lexical = store.search(query, k=depth, product_id=product_id, conn=conn)
    semantic = semantic_search(query, k=depth, product_id=product_id, conn=conn,
                               encoder=encoder, min_similarity=min_similarity)

    scores: Dict[int, float] = {}
    detail: Dict[int, Dict[str, Any]] = {}
    for rank, row in enumerate(lexical):
        rid = row["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
        detail.setdefault(rid, {})["relevance"] = row.get("relevance")
        detail[rid]["lexical_rank"] = rank + 1
    for rank, row in enumerate(semantic):
        rid = row["review_id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
        detail.setdefault(rid, {})["similarity"] = row.get("similarity")
        detail[rid]["semantic_rank"] = rank + 1

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    out = []
    for rid, score in ranked:
        row = _row(conn, rid)
        row.update(detail.get(rid, {}))
        row["rrf_score"] = round(score, 6)
        out.append(row)
    return out


def _row(conn: sqlite3.Connection, review_id: int, **extra) -> Dict[str, Any]:
    r = conn.execute(
        "SELECT id, text, score, gold_label, source, product_id FROM reviews WHERE id=?",
        (review_id,)).fetchone()
    row = dict(r) if r else {"id": review_id, "text": ""}
    row["review_id"] = row.pop("id")
    row.update(extra)
    return row

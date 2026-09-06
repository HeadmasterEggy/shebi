# -*- coding: utf-8 -*-
"""L4 检索层：句向量、向量索引、混合检索。

这一层的价值主张是「查『续航差』要能召回『一天要充三次电』」——
没有字面重合的同义说法。所以核心用例就直接锁这件事。
"""

import numpy as np
import pytest

from agent import store, vectorstore
from agent.embedding import SifEncoder
from tests.conftest import needs_w2v

pytestmark = needs_w2v


REVIEWS = [
    "物流太慢了，等了一个星期才到，很失望",
    "快递第二天就送到了，速度很快",
    "一天要充三次电，出门必须带充电宝",
    "屏幕显示效果不错，色彩很鲜艳",
    "客服态度恶劣，退货流程繁琐",
    "包装盒子压扁了，里面倒是没事",
    "做工精细，材质摸起来很扎实",
    "价格便宜，性价比很高，值这个价",
    "运行很流畅，玩游戏一点都不卡",
    "外观漂亮，颜值在线，同事都问我在哪买的",
]


@pytest.fixture
def db(tmp_path):
    conn = store.connect(str(tmp_path / "reviews.db"))
    store.add_reviews([{"source": "t", "product_id": "p", "text": t} for t in REVIEWS], conn)
    vectorstore.reset()
    yield conn
    vectorstore.reset()


@pytest.fixture
def enc():
    return SifEncoder()


# ---------------- 编码器 ----------------
def test_encoded_vectors_are_l2_normalised(enc):
    vecs = enc.encode(REVIEWS)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_out_of_vocabulary_text_encodes_to_zero_not_noise(enc):
    """全是生僻词时给零向量：它和任何东西的余弦都是 0，会被相似度下限挡掉，
    而不是变成一个指向随机方向的、看起来跟某条评论很像的向量。"""
    vec = enc.encode(["𠮷𠮷𠮷"])[0]
    assert np.allclose(vec, 0.0)


def test_support_scores_relevant_evidence_higher(enc):
    hit = enc.support("物流配送太慢", "物流太慢了，等了一个星期才到，很失望")
    miss = enc.support("物流配送太慢", "屏幕显示效果不错，色彩很鲜艳")
    assert hit > miss


def test_token_maxsim_beats_sentence_cosine_on_paraphrase(enc):
    """这条用例锁的是一个实测结论：句向量在释义上会排错，词级 MaxSim 不会。

    句向量把「续航很差」判得最像「运行很流畅」——均值池化把内容词摊平了。
    改回句向量索引的话，这条会立刻红。
    """
    query = "续航很差"
    target = "一天要充三次电，出门必须带充电宝"
    distractor = "运行很流畅，玩游戏一点都不卡"

    assert enc.similarity(query, distractor) > enc.similarity(query, target), \
        "句向量基线的行为变了，实测结论需要重新标定"
    assert enc.support(query, target) > enc.support(query, distractor)


# ---------------- 索引 ----------------
def test_index_is_built_and_persisted(db, tmp_path):
    index = vectorstore.get_index(db)
    assert index.size == len(REVIEWS)
    # CSR 布局：offsets 比文档数多一个，且末位等于词向量总数
    assert index.offsets.shape[0] == index.size + 1
    assert index.offsets[-1] == index.tokens.shape[0]
    assert (tmp_path / "reviews.vectors.npz").exists(), "索引应落在它所索引的库旁边"


def test_index_is_keyed_by_database_file(tmp_path):
    """两个库不能共用一个索引——否则会拿 A 库的 review_id 去 B 库里查原文。"""
    vectorstore.reset()
    a = store.connect(str(tmp_path / "a.db"))
    b = store.connect(str(tmp_path / "b.db"))
    store.add_reviews([{"source": "t", "text": t} for t in REVIEWS], a)
    store.add_reviews([{"source": "t", "text": "只有一条评论"}], b)
    assert vectorstore.get_index(a).size == len(REVIEWS)
    assert vectorstore.get_index(b).size == 1
    vectorstore.reset()


def test_new_reviews_are_encoded_incrementally(db):
    first = vectorstore.get_index(db).size
    store.add_reviews([{"source": "t", "product_id": "p", "text": "发货速度慢得离谱"}], db)
    assert vectorstore.get_index(db).size == first + 1


# ---------------- 检索 ----------------
def test_semantic_search_finds_paraphrase_without_shared_characters(db):
    """这就是引入向量检索的全部理由。

    「续航」和「一天要充三次电」没有任何字面重合，词重合检索召不回，
    语义检索能，而且排第一。
    """
    query = "续航很差"
    target = "一天要充三次电，出门必须带充电宝"

    lexical = [r["text"] for r in store.search(query, k=3, conn=db)]
    semantic = [r["text"] for r in vectorstore.semantic_search(query, k=3, conn=db)]

    assert target not in lexical, "如果词重合都能召回，这个用例就失去意义了"
    assert semantic and semantic[0] == target


def test_out_of_vocabulary_query_returns_nothing(db):
    """这是召回层唯一的硬保证：查询里一个词都不在词表里时返回空，
    而不是给一堆碰巧向量方向接近的评论。

    注意它**不是**主题过滤器——MaxSim 的绝对分随语料规模饱和，
    "红烧肉的家常做法"在两千条评论上照样能拿到 0.8。挡住无据结论的是
    Critic，不是这里的阈值。
    """
    assert vectorstore.semantic_search("𠮷𠮷𠮷𠮷", k=3, conn=db) == []


def test_hybrid_search_returns_traceable_rows(db):
    hits = vectorstore.hybrid_search("物流太慢", k=3, conn=db)
    assert hits
    assert all("review_id" in h and "text" in h for h in hits)
    assert all("rrf_score" in h for h in hits)


def test_hybrid_recalls_both_literal_and_semantic_matches(db):
    """RRF 融合的意义：两路各自的漏网之鱼互相补上。"""
    hits = {h["text"] for h in vectorstore.hybrid_search("续航差 物流慢", k=6, conn=db)}
    assert "物流太慢了，等了一个星期才到，很失望" in hits    # 字面命中
    assert "一天要充三次电，出门必须带充电宝" in hits          # 语义命中


def test_search_can_be_scoped_to_one_product(tmp_path):
    vectorstore.reset()
    conn = store.connect(str(tmp_path / "scoped.db"))
    store.add_reviews([{"source": "t", "product_id": "p1", "text": t} for t in REVIEWS], conn)
    store.add_reviews([{"source": "t", "product_id": "p2", "text": "电池不耐用，半天就没电"}], conn)
    hits = vectorstore.semantic_search("续航差", k=5, product_id="p2", conn=conn)
    assert [h["text"] for h in hits] == ["电池不耐用，半天就没电"]
    vectorstore.reset()

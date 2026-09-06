# -*- coding: utf-8 -*-
"""句向量编码器。

不调任何 embedding API——直接复用毕设自己那份 50 维预训练词向量
（word2vec/wiki_word2vec_50.bin），按 SIF 加权平均成句向量：

    v(s) = 1/|s| · Σ  a / (a + p(w)) · v(w)      然后减去语料第一主成分

SIF（Arora et al., 2017《A Simple but Tough-to-Beat Baseline》）挑的正是
这个项目的处境：有词向量、没有句向量模型、也不想为每条评论付一次 API 费。
高频词（"的""了""这个"）的权重被 a/(a+p(w)) 压低，效果显著优于朴素平均。

但**句向量这条路实测是输的**，这里记下来免得后人再走一遍：用它做检索时，
查「续航很差」排第一的是「运行很流畅，玩游戏一点都不卡」（0.74），真正
该命中的「一天要充三次电」掉到第 5；用它做引用校验时，196 条带标签对照集
上的判别准确率只有 59%，约等于抛硬币。根因是均值池化——一句话里所有词
平权，内容词被功能词摊平，而 SIF 权重衡量的是"在电商评论语料里罕见"，
分析师写的书面语（"存在""用户""对此"）在评论语料里恰恰罕见，权重反被顶高。

所以真正在用的是 support()：词级加权 MaxSim。同一批样本上「续航很差」
回到第 1 名。encode()/similarity() 保留下来，作为 scripts/calibrate_critic.py
里那个被比下去的基线——结论要可复现，基线就不能删。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

FREQ_PATH = os.path.join(Config.runtime_dir, "word_freq.json")

# SIF 论文的推荐区间是 1e-3 ~ 1e-4，取 1e-3
SIF_A = 1e-3


def _load_word_freq(path: Optional[str] = None) -> Dict[str, float]:
    """词频 p(w)，从训练集统计。

    train.txt 是「标签 空格分词文本」格式，本来就是分好词的，
    统计词频不需要再过一次 jieba。结果缓存到 runtime/，只算一次。
    """
    if os.path.exists(FREQ_PATH):
        try:
            with open(FREQ_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("词频缓存损坏，重新统计：%s", FREQ_PATH)

    counter: Counter = Counter()
    with open(path or Config.train_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                counter.update(parts[1:])

    total = sum(counter.values()) or 1
    freq = {w: c / total for w, c in counter.items()}
    os.makedirs(Config.runtime_dir, exist_ok=True)
    with open(FREQ_PATH, "w", encoding="utf-8") as f:
        json.dump(freq, f, ensure_ascii=False)
    logger.info("词频统计完成：%d 个词", len(freq))
    return freq


class SifEncoder:
    """把中文文本编码成 L2 归一化的句向量。

    线程安全：词表、词向量、词频都是懒加载 + 双检锁，和 inference.engine 一致。
    """

    def __init__(self, a: float = SIF_A):
        self.a = a
        self._lock = threading.RLock()
        self._freq: Optional[Dict[str, float]] = None
        self._stopwords: Optional[set] = None

    # ---------- 依赖的资源 ----------
    @property
    def word2id(self) -> Dict[str, int]:
        from inference import engine
        return engine.word2id

    @property
    def vectors(self) -> np.ndarray:
        from inference import engine
        return engine.w2vec.numpy()

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def freq(self) -> Dict[str, float]:
        if self._freq is None:
            with self._lock:
                if self._freq is None:
                    self._freq = _load_word_freq()
        return self._freq

    @property
    def stopwords(self) -> set:
        if self._stopwords is None:
            with self._lock:
                if self._stopwords is None:
                    with open(Config.stopword_path, encoding="utf-8") as f:
                        self._stopwords = {line.strip() for line in f if line.strip()}
        return self._stopwords

    # ---------- 编码 ----------
    def _tokens(self, text: str) -> List[str]:
        import jieba
        stop = self.stopwords
        return [w for w in jieba.cut(text or "", cut_all=False)
                if w.strip() and w not in stop]

    def raw_encode(self, texts: Sequence[str]) -> np.ndarray:
        """SIF 加权均值，未去主成分、未归一化。拟合主成分时用这个。"""
        w2id, vecs, freq = self.word2id, self.vectors, self.freq
        out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)

        for i, text in enumerate(texts):
            acc = np.zeros(vecs.shape[1], dtype=np.float32)
            n = 0
            for w in self._tokens(text):
                idx = w2id.get(w)
                # pad_idx 是占位符，不携带语义，别把它算进句向量
                if idx is None or idx == Config.pad_idx:
                    continue
                acc += (self.a / (self.a + freq.get(w, 0.0))) * vecs[idx]
                n += 1
            if n:
                out[i] = acc / n
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """完整编码：SIF → 去主成分 → L2 归一化。

        全部词都不在词表里的文本会得到零向量——这是有意的：它和任何东西的
        余弦都是 0，检索时自然被相似度下限挡掉，而不是变成一个随机方向。
        """
        return self.postprocess(self.raw_encode(texts))

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        mat = np.asarray(raw, dtype=np.float32).copy()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        np.divide(mat, norms, out=mat, where=norms > 1e-8)
        mat[norms.reshape(-1) <= 1e-8] = 0.0
        return mat

    def similarity(self, a: str, b: str) -> float:
        """两段文本的整句余弦相似度。

        **这是基线，不是主力**——见模块 docstring 里的实测。留着是为了让
        scripts/calibrate_critic.py 能把它和 support() 摆在一起比。
        """
        vecs = self.encode([a, b])
        return float(np.dot(vecs[0], vecs[1]))

    # ------------------------------------------------------------------
    def weighted_vectors(self, text: str):
        """返回 (词向量矩阵, SIF 权重)，都只含在词表里的词。"""
        w2id, vecs, freq = self.word2id, self.vectors, self.freq
        rows, weights = [], []
        for w in self._tokens(text):
            idx = w2id.get(w)
            if idx is None or idx == Config.pad_idx:
                continue
            rows.append(vecs[idx])
            weights.append(self.a / (self.a + freq.get(w, 0.0)))
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32), np.zeros(0, dtype=np.float32)
        mat = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        np.divide(mat, norms, out=mat, where=norms > 1e-8)
        return mat, np.asarray(weights, dtype=np.float32)

    def support(self, claim: str, evidence: str) -> float:
        """结论被原文支持的程度，[0, 1]。Critic 的语义关卡用这个。

        不是整句余弦——那个指标在这里实测只有 56% 的判别准确率，因为 50 维
        词向量做均值池化时，一句话里的模板词（"方面""存在""问题"）和内容词
        权重相同，长句子的语义被摊平了。

        改成按词的加权 MaxSim（ColBERT 的思路）：结论里的每个词去原文里找
        最像的那个词，再按 SIF 权重加权平均。

            support = Σ  w_i · max_j cos(c_i, e_j)  /  Σ w_i

        SIF 权重 a/(a+p(w)) 在这里第二次发挥作用：高频模板词权重被压到很低，
        "物流""续航""客服"这类低频内容词说了算。这正是要的效果——判断一条
        引用切不切题，看的就是内容词对不对得上。
        """
        c_vecs, c_w = self.weighted_vectors(claim)
        e_vecs, _ = self.weighted_vectors(evidence)
        if c_vecs.shape[0] == 0 or e_vecs.shape[0] == 0:
            return 0.0
        # (claim_len, evidence_len) 的相似度矩阵，取每行最大
        best = (c_vecs @ e_vecs.T).max(axis=1)
        total = float(c_w.sum())
        if total <= 1e-8:
            return 0.0
        return float(np.clip((c_w * best).sum() / total, 0.0, 1.0))


encoder = SifEncoder()

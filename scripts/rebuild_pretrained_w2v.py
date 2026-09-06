"""
从仓库内已有的 word2id.txt + word_vec.txt 重建 gensim 二进制词向量文件。

背景：word2vec/wiki_word2vec_50.bin 由 Git LFS 托管，在没有 LFS 访问权限的
环境里 clone 下来只是一个 133 字节的指针文件，导致 build_word2vec() 报错。
本脚本用仓库里已有的 60723x50 语料词向量重建一个等价的 .bin，使项目可离线跑通。
"""
import os
import numpy as np
from gensim.models import KeyedVectors

W2ID = "word2vec/word2id.txt"
VECS = "word2vec/word_vec.txt"
OUT = "word2vec/wiki_word2vec_50.bin"

words = []
with open(W2ID, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            words.append(parts[0])

vecs = np.loadtxt(VECS, dtype=np.float32)
print(f"words={len(words)} vecs={vecs.shape}")
n = min(len(words), vecs.shape[0])

kv = KeyedVectors(vector_size=vecs.shape[1])
kv.add_vectors(words[:n], vecs[:n])
kv.save_word2vec_format(OUT, binary=True)
print(f"saved -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定 Critic 的语义支持阈值。

critic.SUPPORT_THRESHOLD 决定「这条引用算不算支持这条结论」。拍一个数字上去
是没法向面试官交代的，所以这里用评论库本身造一个带标签的对照集，把阈值算出来：

  正例：结论「<方面>方面存在问题」 ←→ 一条真的在抱怨这个方面的评论
  负例：同一条结论            ←→ 一条抱怨另一个方面的评论

方面归属来自 tools.ASPECT_LEXICON 的词典匹配，是确定性的，所以这个标签集
不依赖任何模型的判断，可复算。输出各阈值下的准确率，取最优。

    python scripts/calibrate_critic.py
"""

from __future__ import annotations

import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import store, tools                       # noqa: E402
from agent.embedding import encoder                  # noqa: E402

CLAIM_TEMPLATE = "{aspect}方面存在问题，用户对此不满"
SEED = 20260906


def build_pairs(limit: int = 600, per_aspect: int = 12):
    """造对照集：(结论, 评论原文, 是否真的支持)。"""
    rows = store.fetch(limit=limit)
    by_aspect: dict[str, list[str]] = {}
    for row in rows:
        for aspect, words in tools.ASPECT_LEXICON.items():
            if any(w in row["text"] for w in words):
                by_aspect.setdefault(aspect, []).append(row["text"])

    # 只留下确实只谈一个方面的评论，避免一条评论同时是正例和负例
    unique: dict[str, list[str]] = {}
    for aspect, texts in by_aspect.items():
        for t in texts:
            hit = [a for a, ws in tools.ASPECT_LEXICON.items() if any(w in t for w in ws)]
            if len(hit) == 1:
                unique.setdefault(aspect, []).append(t)

    rng = random.Random(SEED)
    aspects = [a for a, t in unique.items() if len(t) >= 3]
    pairs = []
    for aspect in aspects:
        claim = CLAIM_TEMPLATE.format(aspect=aspect)
        others = [a for a in aspects if a != aspect]
        for text in unique[aspect][:per_aspect]:
            pairs.append((claim, text, True))
        for _ in range(min(per_aspect, len(unique[aspect]))):
            other = rng.choice(others)
            pairs.append((claim, rng.choice(unique[other]), False))
    return pairs


def main() -> int:
    logging.disable(logging.INFO)
    if store.count() == 0:
        tools.bootstrap_store(build_index=False)

    pairs = build_pairs()
    if not pairs:
        print("评论库里凑不出对照集，先跑 tools.bootstrap_store()")
        return 1

    print(f"对照集：{sum(1 for _, _, l in pairs if l)} 正 / "
          f"{sum(1 for _, _, l in pairs if not l)} 负\n")

    best_overall = None
    for label, fn in (("整句余弦 similarity()", encoder.similarity),
                      ("加权 MaxSim support()", encoder.support)):
        sims = [(fn(claim, text), lab) for claim, text, lab in pairs]
        pos = [s for s, l in sims if l]
        neg = [s for s, l in sims if not l]
        print(f"── {label} ──")
        print(f"  正例均值 {sum(pos)/len(pos):.3f}   负例均值 {sum(neg)/len(neg):.3f}")
        print("  阈值      准确率   正例召回   负例拒绝")
        best = (0.0, 0.0)
        for i in range(20, 91, 5):
            th = i / 100
            tp = sum(1 for s in pos if s >= th)
            tn = sum(1 for s in neg if s < th)
            acc = (tp + tn) / len(sims)
            if acc > best[1]:
                best = (th, acc)
            print(f"   {th:.2f}    {acc*100:6.2f}%   {tp/len(pos)*100:6.2f}%   "
                  f"{tn/len(neg)*100:6.2f}%")
        print(f"  最优 {best[0]:.2f} → 准确率 {best[1]*100:.2f}%\n")
        if best_overall is None or best[1] > best_overall[2]:
            best_overall = (label, best[0], best[1])

    print(f"胜出：{best_overall[0]}，阈值 {best_overall[1]:.2f}，"
          f"准确率 {best_overall[2]*100:.2f}%")
    print("Critic 偏保守：漏掉一条真引用只是让 Analyst 多改一轮，"
          "放过一条假引用则是幻觉直接进报告。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

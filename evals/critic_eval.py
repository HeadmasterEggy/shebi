# -*- coding: utf-8 -*-
"""Critic 有效性评测：它到底能抓住几种幻觉？

简历上写「引入 Critic 后无据结论率从 X% 降到 Y%」，面试官下一句一定是
「那 Critic 本身准不准」。这个脚本就是用来回答那一句的。

用真实评论库构造带标签的对照集，四类幻觉各造一批，每一类对应 Critic 的
一道关卡：

    fabricated_id   引用一个不存在的 review_id      → 关卡 2
    no_citation     结论没有任何出处                → 关卡 1
    wrong_polarity  断言差评，却引一条正面评论      → 关卡 4（本地模型判极性）
    wrong_aspect    谈物流，却引一条只谈屏幕的评论  → 关卡 5（方面词典）

**必须先说清楚哪些数字不算成绩：**

- fabricated_id / no_citation 是结构性的，查库和正则必然 100% 命中。
  它们出现在表里是为了证明关卡接线正确，不是能力证明。
- wrong_aspect 的构造方式和方面关卡用的是同一份词典（agent/aspects.py），
  所以检出率高是**自证**，只能说明关卡按设计工作，不能当泛化能力。
  真实的分析师不会照着词典写结论。
- **只有 wrong_polarity 是真测量**：负例用数据集自带的 gold_label 构造，
  关卡用的是本地模型的预测（测试集 89.49%），两边独立，检出率反映的是
  这条关卡在真实误判下的表现。误报率同理。

把这四行分开报，而不是合成一个漂亮的总分——合成之后那个数字没有意义。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent import store, tools
from agent.aspects import ASPECT_LEXICON
from agent.critic import Critic

SEED = 20260907

# 结论模板。刻意写几种不同措辞，避免整个对照集被一句话的用词带偏。
NEG_TEMPLATES = [
    "{aspect}是差评集中的问题",
    "用户对{aspect}普遍不满",
    "{aspect}方面被反复吐槽",
]
POS_TEMPLATES = [
    "{aspect}获得了不少好评",
    "用户对{aspect}比较满意",
    "{aspect}方面的反馈是正面的",
]

KINDS = ("grounded", "fabricated_id", "no_citation", "wrong_polarity", "wrong_aspect")


@dataclass
class Case:
    kind: str
    claim: str
    review_ids: List[int]
    should_be_grounded: bool

    def as_report_line(self) -> str:
        if not self.review_ids:
            return f"- {self.claim}"
        ids = ", ".join(str(i) for i in self.review_ids)
        return f"- {self.claim} [review_id: {ids}]"


@dataclass
class GateResult:
    name: str
    per_kind: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def rate(self, kind: str) -> Optional[float]:
        hit, total = self.per_kind.get(kind, (0, 0))
        return None if not total else round(hit / total * 100, 2)


def _single_aspect_pool(limit: int = 2000) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """按 (方面, gold_label) 分桶，只收只谈一个方面的评论。

    只留单方面的评论，是为了让"换一条别的方面的评论"这个负例真的只错在方面上，
    而不是同时错好几处——一条样本同时踩两道关卡的话，就分不清是哪一关抓住的。
    """
    pool: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for row in store.fetch(limit=limit):
        hit = [a for a, words in ASPECT_LEXICON.items()
               if any(w in row["text"] for w in words)]
        if len(hit) != 1 or row["gold_label"] is None:
            continue
        pool.setdefault(hit[0], {}).setdefault(int(row["gold_label"]), []).append(row)
    return pool


def build_cases(per_kind: int = 40, limit: int = 2000) -> List[Case]:
    rng = random.Random(SEED)
    pool = _single_aspect_pool(limit)
    # 正负两种极性都有存货的方面才能用来造极性错配的负例
    usable = [a for a, by_label in pool.items()
              if len(by_label.get(0, [])) >= 2 and len(by_label.get(1, [])) >= 2]
    if len(usable) < 2:
        return []

    cases: List[Case] = []

    def neg_claim(aspect):
        return rng.choice(NEG_TEMPLATES).format(aspect=aspect)

    def pos_claim(aspect):
        return rng.choice(POS_TEMPLATES).format(aspect=aspect)

    for i in range(per_kind):
        aspect = usable[i % len(usable)]
        others = [a for a in usable if a != aspect]

        # 有据：差评结论 + 一条同方面的真实负面评论
        row = rng.choice(pool[aspect][0])
        cases.append(Case("grounded", neg_claim(aspect), [row["id"]], True))

        # 编造 id
        cases.append(Case("fabricated_id", neg_claim(aspect), [900_000 + i], False))

        # 无引用
        cases.append(Case("no_citation", neg_claim(aspect), [], False))

        # 极性错配：说差评，却引一条数据集标注为正面的同方面评论
        pos_row = rng.choice(pool[aspect][1])
        cases.append(Case("wrong_polarity", neg_claim(aspect), [pos_row["id"]], False))

        # 方面错配：说这个方面，却引一条只谈别的方面、但极性一致的评论
        other = rng.choice(others)
        other_row = rng.choice(pool[other][0])
        cases.append(Case("wrong_aspect", neg_claim(aspect), [other_row["id"]], False))

    # 正面结论也造一批，避免整套评测只在"差评"这一种措辞上成立
    for i in range(per_kind // 2):
        aspect = usable[i % len(usable)]
        row = rng.choice(pool[aspect][1])
        cases.append(Case("grounded", pos_claim(aspect), [row["id"]], True))
        neg_row = rng.choice(pool[aspect][0])
        cases.append(Case("wrong_polarity", pos_claim(aspect), [neg_row["id"]], False))

    return cases


def evaluate(cases: List[Case], critic: Optional[Critic] = None,
             label: str = "全部关卡") -> GateResult:
    """跑一遍 Critic，按幻觉类型统计检出率。

    每条 case 单独交给 Critic：合成一份大报告的话，一条结论的判定会被
    同一批被引评论的极性批处理顺序影响，分不清是哪条出的问题。
    """
    critic = critic or Critic()
    result = GateResult(name=label)
    # 证据池放宽到所有真实 id，让"越出证据池"这道关卡不参与——这里要测的是
    # 内容层面的幻觉，不是范围控制
    evidence = [c.review_ids[0] for c in cases if c.review_ids]

    for case in cases:
        rep = critic.review(case.as_report_line(), evidence_ids=evidence)
        grounded = bool(rep.verdicts) and rep.verdicts[0].grounded
        correct = grounded == case.should_be_grounded
        hit, total = result.per_kind.get(case.kind, (0, 0))
        result.per_kind[case.kind] = (hit + int(correct), total + 1)
    return result


def ablations(cases: List[Case]) -> List[GateResult]:
    """逐关卡关掉，看检出率掉多少。

    消融是这份评测里唯一能说明"每道关卡各自贡献了什么"的部分。
    只报一个总分的话，没人知道拿掉哪一关会塌。
    """
    return [
        evaluate(cases, Critic(check_polarity=True, check_aspect=True), "全部关卡"),
        evaluate(cases, Critic(check_polarity=False, check_aspect=True), "关掉极性关卡"),
        evaluate(cases, Critic(check_polarity=True, check_aspect=False), "关掉方面关卡"),
        evaluate(cases, Critic(check_polarity=False, check_aspect=False), "只剩结构性关卡"),
    ]


def run(per_kind: int = 40) -> Dict[str, Any]:
    if store.count() == 0:
        tools.bootstrap_store(build_index=False)
    cases = build_cases(per_kind=per_kind)
    if not cases:
        return {"cases": 0, "error": "评论库里凑不出对照集"}
    results = ablations(cases)
    return {
        "cases": len(cases),
        "per_kind_counts": {k: sum(1 for c in cases if c.kind == k) for k in KINDS},
        "results": [{"name": r.name, **{k: r.rate(k) for k in KINDS}} for r in results],
    }

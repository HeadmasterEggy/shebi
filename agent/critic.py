# -*- coding: utf-8 -*-
"""Critic：引用溯源校验。

Planner / Collector / Analyst 是多 Agent 的标配，真正拉开差距的是这一层。

**关键设计：校验是确定性的，不是再叫一次大模型去"判断"。**
让 LLM 去审 LLM，审查者自己也会幻觉，每审一次还要付钱、加延迟，最后拿到
一个说不清怎么来的分数。这里每一关都可复算：

  1. 结论有没有引用          —— 正则解析 [review_id: 12, 34]
  2. 引用的 id 是否真实存在  —— 查 SQLite，编出来的 id 当场露馅
  3. 是否在检索到的证据池内  —— 防止模型把别处见过的 id 抄过来
  4. 引用是否真的支持结论    —— 极性一致 + 方面一致（详见下）

第 4 关踩过一个坑，值得记下来：最初用的是句向量余弦，拿评论库造了 196 条
带标签的对照集一标定，判别准确率只有 **59%**——基本等于抛硬币。原因是 SIF
权重衡量的是"在电商评论语料里罕见"，而分析师写的书面语（"存在""用户"
"对此"）在评论语料里恰恰罕见，权重被顶高，把"物流""续航"这些真正的内容词
淹掉了。标定脚本 scripts/calibrate_critic.py 可以复现这个结论。

所以第 4 关重做成两条不需要调阈值的确定性子关卡：

  4a **极性一致**：结论断言"差评"，被引评论就必须被本地模型判为消极。
      这一关用的正是毕设自训练的那个模型（测试集 89.49%）——它在这里
      从"被展示的成果"变成了"系统内部的质检工序"。
  4b **方面一致**：结论提到的方面（物流/续航/客服…）必须在被引评论里出现。
      方面词典是确定性的，见 agent/aspects.py。

语义余弦作为可选的第三条子关卡保留（semantic_threshold），默认关闭，
理由就是上面那个 59%。把一个自己都没测过的检查摆在流程里，比没有更糟。

产出 unsupported_rate。这个数字就是"引入 Critic 前后无据结论率从 X% 降到 Y%"
里的 X 和 Y——它是算出来的，面试时能当场复现。
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from agent import store
from agent.aspects import asserted_polarity, aspects_of

logger = logging.getLogger(__name__)

# 匹配 [review_id: 12, 34] / [review_id：12] / [review_ids: 12,34]
CITATION_RE = re.compile(r"\[\s*review_ids?\s*[:：]\s*([0-9,，、\s]+?)\s*\]", re.I)

# 语义关卡的默认阈值。None = 关闭，理由见模块 docstring 里的标定结果。
DEFAULT_SEMANTIC_THRESHOLD: Optional[float] = None


@dataclass
class Claim:
    """一条结论及它声称的出处。"""
    text: str
    review_ids: List[int] = field(default_factory=list)
    index: int = 0


@dataclass
class ClaimVerdict:
    claim: str
    review_ids: List[int]
    grounded: bool
    reasons: List[str] = field(default_factory=list)
    supporting_ids: List[int] = field(default_factory=list)
    missing_ids: List[int] = field(default_factory=list)
    out_of_scope_ids: List[int] = field(default_factory=list)
    rejected: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "review_ids": self.review_ids,
            "grounded": self.grounded,
            "reasons": self.reasons,
            "supporting_ids": self.supporting_ids,
            "missing_ids": self.missing_ids,
            "out_of_scope_ids": self.out_of_scope_ids,
            "rejected": {str(k): v for k, v in self.rejected.items()},
        }


@dataclass
class CritiqueReport:
    verdicts: List[ClaimVerdict]
    gates: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def grounded(self) -> int:
        return sum(1 for v in self.verdicts if v.grounded)

    @property
    def unsupported_rate(self) -> float:
        """无据结论占比（%）。一条结论都没产出时算 100——交白卷不是满分。"""
        if not self.verdicts:
            return 100.0
        return round((self.total - self.grounded) / self.total * 100, 2)

    @property
    def passed(self) -> bool:
        return bool(self.verdicts) and self.grounded == self.total

    def failures(self) -> List[ClaimVerdict]:
        return [v for v in self.verdicts if not v.grounded]

    def feedback(self) -> str:
        """打回给 Analyst 的整改意见。要具体到哪条结论、错在哪。"""
        bad = self.failures()
        if not self.verdicts:
            return ("上一版没有产出任何带引用的结论。请为每条结论补上 "
                    "[review_id: x, y] 形式的出处，出处必须来自证据池。")
        if not bad:
            return ""
        lines = [f"以下 {len(bad)} 条结论未通过引用校验，请逐条修正后重新给出完整结论列表："]
        for v in bad:
            lines.append(f"  - 「{v.claim[:60]}」：{'；'.join(v.reasons)}")
        lines.append("修正方式：换成真正支持该结论的 review_id，或直接删掉这条结论。"
                     "不要保留没有出处的结论。")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total,
            "grounded_claims": self.grounded,
            "unsupported_rate": self.unsupported_rate,
            "passed": self.passed,
            "gates": self.gates,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# ---------------------------------------------------------------------------
def parse_claims(text: str) -> List[Claim]:
    """把一段报告拆成「结论 + 引用」。

    按行拆：一行一条结论是 prompt 里明确要求的格式。标题、空行、纯过渡句
    不计入分母——否则"无据结论率"会被格式噪音稀释成一个好看但没意义的数。
    """
    claims: List[Claim] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("-*•").strip()
        if not line:
            continue
        ids = extract_ids(line)
        body = CITATION_RE.sub("", line).strip(" 。;；,，")
        if not body:
            continue
        # 无引用的行只有看起来像结论（够长、不是标题）才计入
        if not ids and (len(body) < 8 or body.endswith(("：", ":"))):
            continue
        claims.append(Claim(text=body, review_ids=ids, index=len(claims) + 1))
    return claims


def extract_ids(text: str) -> List[int]:
    ids: List[int] = []
    for chunk in CITATION_RE.findall(text or ""):
        for part in re.split(r"[,，、\s]+", chunk):
            if part.isdigit():
                rid = int(part)
                if rid not in ids:
                    ids.append(rid)
    return ids


class Critic:
    """确定性引用校验器。

    evidence_ids 是 Collector 实际检索到的 id 集合。传了就启用第 3 关；
    不传则跳过，只校验存在性与支持性。

    三个开关都可以单独关掉，方便做消融——"关掉极性关卡后无据结论率变化多少"
    是一个能直接写进报告的数字。
    """

    def __init__(self, conn: Optional[sqlite3.Connection] = None,
                 check_scope: bool = True, check_polarity: bool = True,
                 check_aspect: bool = True,
                 semantic_threshold: Optional[float] = DEFAULT_SEMANTIC_THRESHOLD,
                 model: Optional[str] = None, encoder=None):
        self.conn = conn
        self.check_scope = check_scope
        self.check_polarity = check_polarity
        self.check_aspect = check_aspect
        self.semantic_threshold = semantic_threshold
        self.model = model
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            from agent.embedding import encoder as default_encoder
            self._encoder = default_encoder
        return self._encoder

    def gates(self) -> List[str]:
        names = ["有引用", "id 存在"]
        if self.check_scope:
            names.append("在证据池内")
        if self.check_polarity:
            names.append("极性一致")
        if self.check_aspect:
            names.append("方面一致")
        if self.semantic_threshold is not None:
            names.append(f"语义相似 ≥{self.semantic_threshold}")
        return names

    # ------------------------------------------------------------------
    def review(self, report: str,
               evidence_ids: Optional[Iterable[int]] = None) -> CritiqueReport:
        claims = parse_claims(report)
        scope: Optional[Set[int]] = set(evidence_ids) if evidence_ids is not None else None

        # 先把所有被引评论的原文和极性一次性算完：极性判断要过本地模型，
        # 按结论逐条调用就退化成了工具层第 1 条硬规则明令禁止的那种用法。
        cited = [rid for c in claims for rid in c.review_ids]
        texts = self._fetch_texts(cited)
        polarity = self._classify(texts) if self.check_polarity else {}

        return CritiqueReport(
            verdicts=[self._judge(c, scope, texts, polarity) for c in claims],
            gates=self.gates(),
        )

    # ------------------------------------------------------------------
    def _fetch_texts(self, review_ids: Iterable[int]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for rid in dict.fromkeys(review_ids):
            row = store.get(rid, conn=self.conn)
            if row is not None:
                out[rid] = row["text"]
        return out

    def _classify(self, texts: Dict[int, str]) -> Dict[int, str]:
        """本地模型给每条被引评论判个极性。一次前向，不逐条。"""
        if not texts:
            return {}
        try:
            from agent.tools import classify_sentiment
            ids = list(texts)
            out = classify_sentiment([texts[i] for i in ids], model=self.model)
            return {rid: r["sentiment"] for rid, r in zip(ids, out["results"])}
        except Exception as e:  # noqa: BLE001 —— 没有权重时不能让校验整体瘫掉
            logger.warning("本地模型不可用，跳过极性关卡：%s", e)
            return {}

    def _judge(self, claim: Claim, scope: Optional[Set[int]],
               texts: Dict[int, str], polarity: Dict[int, str]) -> ClaimVerdict:
        verdict = ClaimVerdict(claim=claim.text, review_ids=list(claim.review_ids),
                               grounded=False)

        # 关卡 1：有没有引用
        if not claim.review_ids:
            verdict.reasons.append("没有任何引用")
            return verdict

        # 关卡 2：引用的评论是否真实存在
        verdict.missing_ids = [rid for rid in claim.review_ids if rid not in texts]
        if verdict.missing_ids:
            verdict.reasons.append(f"引用了不存在的 review_id {verdict.missing_ids}")

        candidates = [rid for rid in claim.review_ids if rid in texts]

        # 关卡 3：是否越出了检索到的证据池
        if scope is not None and self.check_scope:
            verdict.out_of_scope_ids = [rid for rid in candidates if rid not in scope]
            if verdict.out_of_scope_ids:
                verdict.reasons.append(
                    f"引用了未经检索的 review_id {verdict.out_of_scope_ids}")
            candidates = [rid for rid in candidates if rid not in verdict.out_of_scope_ids]

        # 关卡 4：引用是否真的支持这条结论
        for rid in candidates:
            why = self._rejects(claim.text, texts[rid], polarity.get(rid))
            if why:
                verdict.rejected[rid] = why
            else:
                verdict.supporting_ids.append(rid)

        if candidates and not verdict.supporting_ids:
            why = "；".join(sorted(set(verdict.rejected.values())))
            verdict.reasons.append(f"引用的原文不支持这条结论（{why}）")

        verdict.grounded = bool(verdict.supporting_ids) and not verdict.missing_ids
        return verdict

    def _rejects(self, claim: str, evidence: str,
                 evidence_polarity: Optional[str]) -> Optional[str]:
        """这条引用为什么不算支持；返回 None 表示通过。"""
        # 4a 极性一致
        if self.check_polarity and evidence_polarity:
            want = asserted_polarity(claim)
            if want and want != evidence_polarity:
                return f"结论断言{want}，但被引评论被判为{evidence_polarity}"

        # 4b 方面一致
        if self.check_aspect:
            claim_aspects = aspects_of(claim)
            if claim_aspects and not (claim_aspects & aspects_of(evidence)):
                return (f"结论谈的是{'、'.join(sorted(claim_aspects))}，"
                        f"被引评论并未提及")

        # 4c 语义相似（默认关闭，见模块 docstring）
        if self.semantic_threshold is not None:
            try:
                score = self.encoder.support(claim, evidence)
            except Exception as e:  # noqa: BLE001
                logger.warning("句向量不可用，跳过语义关卡：%s", e)
                return None
            if score < self.semantic_threshold:
                return f"语义相似度 {score:.2f} 低于阈值 {self.semantic_threshold}"
        return None

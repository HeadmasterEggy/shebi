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

# 分析师用它把结论列表和推理过程隔开。有这个块就只读块内，
# 没有则退回"只认列表项"的宽松模式（见 parse_claims）。
CLAIM_BLOCK_RE = re.compile(r"<结论>(.*?)</结论>", re.S)

# 列表项前缀：- * • 或 "1." "1、"
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.、)])\s+")

# 引用工具产出：[source: list_experiments]
#
# 这一条是评测逼出来的。原来只认 review_id，等于假设"每条结论都是关于评论内容的"。
# 可"当前模型准确率是多少""一共有多少条评论"这类问题，答案来自工具返回值，
# 根本没有评论可引——40 条任务里有 4 条因此结构性地无法通过校验：
# agent 明明查到了正确答案，却被判成"证据不足"。
SOURCE_RE = re.compile(r"\[\s*source\s*[:：]\s*([A-Za-z_,，、\s]+?)\s*\]")

# 语义关卡的默认阈值。None = 关闭，理由见模块 docstring 里的标定结果。
DEFAULT_SEMANTIC_THRESHOLD: Optional[float] = None


@dataclass
class Claim:
    """一条结论及它声称的出处。

    出处有两种：评论原文（review_ids）和工具产出（sources）。
    统计类结论只能是后者——没有哪一条评论能"支撑"住"库里共 2000 条评论"。
    """
    text: str
    review_ids: List[int] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
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
    sources: List[str] = field(default_factory=list)

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
            **({"sources": self.sources} if self.sources else {}),
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
    """把分析师的产出拆成「结论 + 引用」。

    **这个函数踩过一个把整份评测报废的坑，值得写下来。** 最初的版本是
    "每一行只要够长就算一条结论"，实测跑下来，模型在结论列表前面还写了
    一大段推理和证据罗列，698 行的产出被数成 477 条结论、其中带引用的
    一条都没有——"无据结论率 100%"于是成了"模型话多"的度量，跟幻觉毫无关系。
    整轮评测的引用准确率因此全部作废。

    现在的口径分两级：

      1. 产出里有 <结论>…</结论> 块时，**只读块内**。这是和分析师约定的
         机器可校验格式，推理过程爱写多少写多少，不进分母。
      2. 没有这个块时退回宽松模式，但**只认列表项**（- * • 或 "1."）。
         散文段落一律不算结论——它们本来就不是结论。

    宁可漏掉一条没按格式写的结论（分析师会收到整改意见重来），
    也不要把叙述文字算成无据结论：后者会让这个指标彻底失去意义。
    """
    block = CLAIM_BLOCK_RE.search(text or "")
    body_text = block.group(1) if block else (text or "")
    strict = block is None      # 没有结论块时才需要靠列表项前缀筛选

    claims: List[Claim] = []
    for raw_line in body_text.splitlines():
        if strict and not BULLET_RE.match(raw_line):
            continue
        line = BULLET_RE.sub("", raw_line).strip()
        if not line:
            continue
        ids = extract_ids(line)
        sources = extract_sources(line)
        content = SOURCE_RE.sub("", CITATION_RE.sub("", line)).strip(" 。;；,，")
        if not content:
            continue
        # 标题、"负面：" 这类小节名不是结论
        if not ids and not sources and (len(content) < 8
                                        or content.endswith(("：", ":"))):
            continue
        claims.append(Claim(text=content, review_ids=ids, sources=sources,
                            index=len(claims) + 1))
    return claims


def extract_claim_block(text: str) -> Optional[str]:
    """取出 <结论>…</结论> 的内容；没有这个块返回 None。

    编排层用它判断分析师有没有按格式交作业。没按格式就直接打回，
    而不是去解析那一大段推理——把叙述文字当成结论校验，指标就废了。
    """
    m = CLAIM_BLOCK_RE.search(text or "")
    return m.group(1) if m else None


def extract_sources(text: str) -> List[str]:
    """取出 [source: xxx] 里声明的工具名。"""
    names: List[str] = []
    for chunk in SOURCE_RE.findall(text or ""):
        for part in re.split(r"[,，、\s]+", chunk):
            if part and part not in names:
                names.append(part)
    return names


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
               evidence_ids: Optional[Iterable[int]] = None,
               available_sources: Optional[Iterable[str]] = None) -> CritiqueReport:
        """校验一份结论列表。

        available_sources 是这一轮里真正调用成功过的工具名。传了它，统计类
        结论就可以用 [source: 工具名] 作为出处——但只能引真调过的工具，
        编一个没调过的名字照样过不了。不传则不接受工具引用。
        """
        claims = parse_claims(report)
        scope: Optional[Set[int]] = set(evidence_ids) if evidence_ids is not None else None
        sources: Set[str] = set(available_sources or ())

        # 先把所有被引评论的原文和极性一次性算完：极性判断要过本地模型，
        # 按结论逐条调用就退化成了工具层第 1 条硬规则明令禁止的那种用法。
        cited = [rid for c in claims for rid in c.review_ids]
        texts = self._fetch_texts(cited)
        polarity = self._classify(texts) if self.check_polarity else {}

        return CritiqueReport(
            verdicts=[self._judge(c, scope, texts, polarity, sources) for c in claims],
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
               texts: Dict[int, str], polarity: Dict[int, str],
               sources: Set[str]) -> ClaimVerdict:
        verdict = ClaimVerdict(claim=claim.text, review_ids=list(claim.review_ids),
                               grounded=False)
        verdict.sources = list(claim.sources)

        # 统计类结论：出处是工具产出，不是某条评论。
        # 只校验"这个工具这一轮真的调过"——编一个没调过的工具名同样过不了。
        if claim.sources:
            bad = [s for s in claim.sources if s not in sources]
            if bad:
                verdict.reasons.append(f"引用了本轮没有调用过的工具 {bad}")
                return verdict
            verdict.grounded = True
            return verdict

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

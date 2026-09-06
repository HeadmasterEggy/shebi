# -*- coding: utf-8 -*-
"""Agent 任务集。

40 条任务，覆盖六类意图。造这个集合时守两条规矩：

1. **必须有拒答题。** 12 条 refusal 任务问的是评论库里根本没有的信息
   （售后电话、发货仓库地址、销量排名）。一个只会答"是"的 agent 在
   只有正例的评测集上能拿满分——那样的评测集等于没做。
2. **expect_tools 记的是"应当用到的能力"，不是"标准答案调用序列"。**
   一条问方面分布的题，用 extract_aspects 或先 search 再 classify 都算对；
   把评测钉死在某一条路径上，改进 prompt 反而会让分数下降。

引用准确率不在这里定义——它由 Critic 在运行时算出来（见 agent/critic.py），
这里只负责给出任务和意图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Task:
    id: str
    question: str
    kind: str
    expect_tools: List[str] = field(default_factory=list)
    must_refuse: bool = False


def _t(i, q, kind, tools=(), refuse=False):
    return Task(id=f"T{i:02d}", question=q, kind=kind,
                expect_tools=list(tools), must_refuse=refuse)


ASPECT = ("extract_aspects", "aggregate_reviews", "search_reviews")
SEARCH = ("search_reviews", "get_reviews")
AGG = ("aggregate_reviews", "extract_aspects")

TASKS: List[Task] = [
    # ---- 方面归因：agent 最常被问的一类 ----
    _t(1,  "这批评论里差评主要集中在什么问题上？", "aspect", ASPECT),
    _t(2,  "用户最不满意的三个方面是什么？", "aspect", ASPECT),
    _t(3,  "物流方面的负面反馈多不多？", "aspect", ASPECT),
    _t(4,  "关于包装的抱怨具体是什么内容？", "aspect", SEARCH),
    _t(5,  "客服服务被投诉的点在哪里？", "aspect", SEARCH),
    _t(6,  "有没有人提到屏幕有问题？具体怎么说的？", "aspect", SEARCH),
    _t(7,  "续航方面用户怎么评价？", "aspect", SEARCH),
    _t(8,  "质量做工的负面率大概是多少？", "aspect", ASPECT),

    # ---- 检索取证：结论必须挂得上原文 ----
    _t(9,  "找几条抱怨发货慢的评论，把原文给我", "retrieval", SEARCH),
    _t(10, "有没有评论提到退货流程麻烦？", "retrieval", SEARCH),
    _t(11, "把提到电池的评论列出来", "retrieval", SEARCH),
    _t(12, "有人说包装破损吗？", "retrieval", SEARCH),
    _t(13, "找出提到性价比的评论", "retrieval", SEARCH),
    _t(14, "有没有夸外观好看的评论？", "retrieval", SEARCH),
    _t(15, "关于运行速度的评价是正面还是负面？", "retrieval", SEARCH),
    _t(16, "把最典型的一条差评找出来", "retrieval", SEARCH),

    # ---- 整体统计 ----
    _t(17, "这批评论整体口碑怎么样？", "aggregate", AGG),
    _t(18, "好评和差评的比例大概是多少？", "aggregate", AGG),
    _t(19, "负面评论占比超过一半了吗？", "aggregate", AGG),
    _t(20, "一共有多少条评论？", "aggregate", AGG),

    # ---- 对比推理：需要两次取证再比 ----
    _t(21, "物流和客服，哪个方面的投诉更多？", "compare", ASPECT),
    _t(22, "续航和屏幕，用户更在意哪个？", "compare", ASPECT),
    _t(23, "价格方面的评价比质量方面更正面吗？", "compare", ASPECT),
    _t(24, "正面评论里被夸得最多的是什么？", "compare", ASPECT),

    # ---- 模型元信息：考的是它知不知道去查 list_experiments ----
    _t(25, "现在用的情感模型准确率是多少？", "meta", ("list_experiments",)),
    _t(26, "cnn 和 bilstm 哪个在测试集上表现更好？", "meta", ("list_experiments",)),
    _t(27, "本地模型单条推理要多久？", "meta", ("list_experiments",)),
    _t(28, "有哪些模型可以用？", "meta", ("list_experiments",)),

    # ---- 拒答：库里根本没有的信息，答出来就是编 ----
    _t(29, "这个商品的售后电话是多少？", "refusal", (), True),
    _t(30, "商家的发货仓库在哪个城市？", "refusal", (), True),
    _t(31, "这款商品上个月卖了多少台？", "refusal", (), True),
    _t(32, "评论里提到的那位客服叫什么名字？", "refusal", (), True),
    _t(33, "这个商品在同类里排第几？", "refusal", (), True),
    _t(34, "厂家的保修政策是几年？", "refusal", (), True),
    _t(35, "买家的平均年龄是多少？", "refusal", (), True),
    _t(36, "这批评论是哪一年产生的？", "refusal", (), True),
    _t(37, "竞品的口碑比这个好吗？", "refusal", (), True),
    _t(38, "这个商品的成本价大概多少？", "refusal", (), True),
    _t(39, "差评用户后来退货成功了吗？", "refusal", (), True),
    _t(40, "评论里有多少是刷的？", "refusal", (), True),
]


def by_kind(kind: str) -> List[Task]:
    return [t for t in TASKS if t.kind == kind]


KINDS = sorted({t.kind for t in TASKS})

# 中文电商评论情感分析系统

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HeadmasterEggy/shebi)

基于 PyTorch 的中文文本情感分析平台，支持 TextCNN / LSTM / BiLSTM / LSTM+Attention /
BiLSTM+Attention 五种模型，含京东评论爬虫、训练调度、日志分析与 Web 服务。

> **分支说明**
> - `main` —— 当前开发分支。在毕设基础上做了一轮工程审计与修复（见下方「本轮修复」），
>   并外挂了一层 Agent（工具层 / ReAct 循环 / 多 Agent 编排 / 评论检索）。
> - `thesis` —— 毕业设计完成时的原样快照，不含任何后续改造，需要对照时切过去即可。

---

## 快速开始

```bash
git clone https://github.com/HeadmasterEggy/shebi.git
cd shebi

python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 预训练词向量托管在 Git LFS。没有 LFS 配额时 clone 到的是 133 字节指针文件，
# 用仓库内已有的 word_vec.txt 重建一份等价的二进制词向量：
python scripts/rebuild_pretrained_w2v.py

# 创建管理员账户（凭据只从环境变量读取，源码里不再写死）
ADMIN_USERNAME=admin ADMIN_EMAIL=you@example.com ADMIN_PASSWORD=<你的密码> \
  python init_db.py

# 训练至少一个模型（默认模型见 config.py）
python main.py --model bilstm --epochs 6 --batch-size 128 --learning-rate 1e-3

# 在测试集上评测，产出界面展示所需的真实指标
python evaluate.py

# 启动服务
SECRET_KEY=$(openssl rand -hex 32) python app.py
# 访问 http://localhost:5003
```

### Docker

```bash
docker compose up --build
```

---

## 本轮修复

原实现存在若干缺陷，其中一个直接导致线上推理结果全错。

### 1. TextCNN 在推理模式下坍缩（致命）

`cnn_model.py` 用了 `nn.BatchNorm1d`。BatchNorm 在 `train()` 下用当前 batch 统计量、
在 `eval()` 下切换到 running stats；而嵌入层参与训练（`freeze=False`）使特征分布
持续漂移，running stats 始终追不上。结果是 `model.eval()` 下特征被压平，
分类头退化为常数分类器：

```
同一份权重，仅切换归一化层的模式：
  eval()      acc 49.87%   预测分布 [0, 6334]     ← 全部判为正类
  batch stats acc 76.07%   预测分布 [3185, 3149]
```

49.87% 恰好等于验证集正类占比（3159/6334）。线上 `/api/analyze` 永远走 `eval()`，
于是任何差评都被判成「积极」——实测「物流太慢了，包装还破损了，很失望」
返回**积极 99.997%**。

**修复**：改用 `nn.LayerNorm`。它在样本内部归一化，不维护 running stats，
train / eval 行为一致，且不受推理时 batch 大小影响（线上单条请求 batch=1，
BatchNorm 在此本就不适用）。

效果（dropout 0.5, weight_decay 1e-4, lr 1e-3）：

| epoch | 修复前 val_acc | 修复后 val_acc |
|------:|--------------:|--------------:|
| 1     | 49.87%        | **87.42%**    |
| 5     | 50.05%        | **89.77%**    |
| 100   | 71.23%        | —（6 轮即收敛）|

回归测试见 `tests/test_model_architecture.py`。

### 2. 单次推理 17.4 秒

`/api/analyze` 每收到一个请求都重新走一遍：重建词表（遍历 train.txt 50674 行 +
val.txt 6334 行并回写磁盘）→ 把三个数据集全部转成索引矩阵 → 用 gensim 载入 12MB
二进制词向量并拼 60723×50 矩阵 → `torch.load()` 反序列化模型。这些与用户输入无关。

**修复**：抽出 `inference.py`，把词表 / 词向量 / 模型收成带可重入锁的进程内单例，
启动时预热。词表改为直接读 `word2id.txt` 而非全量重建。

```
改造前  17.4s / 请求
改造后  首次 82ms，之后 ~10ms
```

### 3. 界面展示的指标是编造的

原实现从全局 `metrics_log.csv` 取 `iloc[-1]`（里面混着所有模型的结果），
读不到时回落到写死的 `accuracy 0.85 / f1 0.84 / recall 0.83`。

**修复**：新增 `evaluate.py` 在测试集上评测，按模型分开写入
`runtime/model_metrics.json`；读不到就返回 `null`，由前端显示「未评测」。

### 4. 硬编码开发机绝对路径

`main.py` 把 `/Users/joey/PycharmProjects/shebi/config/progress.json` 写死在源码里，
换任何机器运行训练都抛 `FileNotFoundError`。

**修复**：所有路径以 `config.py` 所在目录为基准解析；运行时文件统一放 `runtime/`。

### 5. 敏感数据入库

`instance/users.db` 被提交进公开仓库（6 个真实账号 + 邮箱 + 密码哈希），
`SECRET_KEY` 默认 `dev_key_for_testing`，管理员 `admin/admin123` 写死在源码里。

**修复**：`instance/` 与 `*.db` 加入 `.gitignore`；`SECRET_KEY` 必须由环境变量提供；
管理员账户改由 `init_db.py` 依据环境变量创建。
**历史清理步骤见 [SECURITY_CLEANUP.md](SECURITY_CLEANUP.md)——仅从索引移除不够。**

### 6. 其他

- `padding_idx` 原为 `vocab_size - 1`，把词表末尾一个真实词当成了填充符；改为 `_PAD_` 的真实索引 0
- `Config.vocab_size` 长期停留在 54848，真实词表 60723；模型构建改为以词向量矩阵行数为准
- `Config.model_name` 并不存在，`app.py` 走到那一行必抛 `AttributeError`
- 日志格式笔误 `%(levellevel)s`，每条日志都触发一次 logging 内部异常
- `data/pre.txt` 被所有请求共享写入，并发时互相覆盖；改为每请求独立临时文件
- 补齐缺失的 `requirements.txt`（README 一直让人 `pip install -r`，但文件不存在）

---

## Agent 层

把这套情感分析能力包成工具，交给一个手写的 ReAct 循环来编排。
本地自训练的模型是它手里最便宜的那把工具（单条 0.52ms，几乎零成本）。

```bash
# 列出全部工具及其 JSON Schema
python -m agent --tools

# 离线跑通（不需要 API key，用剧本驱动真实工具）
python scripts/demo_agent.py         # 单体 ReAct 循环
python scripts/demo_supervisor.py    # 多 Agent 编排 + Critic 修订闭环

# 接真实模型
AGENT_API_KEY=sk-...  python -m agent "这批评论里差评主要集中在什么问题上？"
AGENT_API_KEY=sk-...  python -m agent --multi "这批评论里差评主要集中在什么问题上？"

# 回看执行轨迹
python -m agent --traces
python -m agent --trace <run_id>
```

### provider 切换

只依赖 OpenAI 的 chat completions 协议，换端点不换代码：

```bash
# OpenAI
AGENT_API_KEY=sk-...            AGENT_MODEL=gpt-4o-mini
# DeepSeek
AGENT_API_KEY=sk-...            AGENT_MODEL=deepseek-chat   AGENT_BASE_URL=https://api.deepseek.com/v1
# Qwen（阿里云百炼）
AGENT_API_KEY=sk-...            AGENT_MODEL=qwen-plus       AGENT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 工具

| Tool | 说明 | 副作用 |
|---|---|---|
| `classify_sentiment` | 本地模型批量判情感，一次最多 512 条 | — |
| `search_reviews` | 检索评论，字面 / 语义 / 混合三种模式，返回带 `review_id` 的原文 | — |
| `get_reviews` | 按 id 取原文，供引用溯源校验 | — |
| `aggregate_reviews` | 整体统计：总量、情感分布、正负比 | — |
| `extract_aspects` | 方面级统计：十类方面的提及量、负面率与代表性 id | — |
| `export_report` | 导出 Markdown 报告，引用自动展开成原文附录 | — |
| `list_experiments` | 各模型在测试集上的真实指标 | — |
| `scrape_reviews` | 抓取京东商品评论 | 需人工确认 |
| `train_model` | 按超参训练模型 | 需人工确认 |

三条设计规则：**批量优先**（`texts` 是数组，不是单条）、**可溯源**（返回评论一律带 id）、
**副作用显式标记**（发网络请求 / 占算力的工具默认拦下，由调用方放行）。

### 循环里的工程约束

- **预算**：步数 / token / 成本三个上限，任一触顶就优雅收尾，把已拿到的中间结果交回去
- **schema 自修复**：参数校验失败时，把 pydantic 报错连同 JSON Schema 回灌给模型让它自己改；
  每个调用最多修 2 次，修复次数进 trace
- **全链路 trace**：每步落盘（模型输出、工具、入参、observation、token、延迟、成本），
  写进 `runtime/traces/<run_id>.json`

先手写这一层，理解透了再考虑换 LangGraph——不是反过来。

### 检索：为什么句向量这条路被推翻了

评论入向量库，用的是项目自己那份 50 维预训练词向量做 SIF 加权，不调任何
embedding API。第一版按常规做法把每条评论压成一个句向量比余弦，**实测是错的**：

| 查询「续航很差」 | 排第一 | 目标「一天要充三次电」 |
|---|---|---|
| 句向量余弦 | 运行很流畅，玩游戏一点都不卡（0.74） | 第 5 名 |
| 词级 MaxSim | 一天要充三次电（0.61） | **第 1 名** |

根因是均值池化把内容词摊平了。改成词级加权 MaxSim（ColBERT 思路）后目标回到第一。
索引因此存词级向量而非句向量：2000 条评论约 2.9 万个词向量、5.8MB，可接受；
到十万条以上就该改成"先召回再重排"的两段式，接口不用变。

检索默认走混合模式，词重合与向量两路召回后用 RRF 融合——RRF 只看名次不看分数，
不需要把两路量纲不同的分数归一化。

一句必须说清的话：**语义召回不承担主题过滤的职责**。MaxSim 的绝对分随语料规模
饱和，"红烧肉的家常做法"在两千条评论上照样能拿到 0.8。挡住无据结论的是下面的
Critic，不是检索的相似度阈值。

### 多 Agent 编排与 Critic

```
Planner ──► Collector ──► Analyst ──► Critic ──┬─(不通过)─► Analyst 重做
                │                              │
             证据池                          (通过)
                │                              ▼
                └────────► 只有通过校验的引用 ──► Reporter
```

三个设计点：

1. **证据池从工具返回值里收割，不从模型的话里解析。** Collector 每次工具调用成功，
   回调就把带 `review_id` 的原文存进黑板。模型在自然语言里吹嘘"我检索到了 999 条"
   不影响证据池。
2. **最小工具授权。** Analyst 拿不到 `search_reviews`，想给站不住的结论现找一条证据，
   接口层面就做不到；Planner 和 Reporter 一个工具都没有。这比在 prompt 里写
   "请不要编造"可靠。
3. **Critic 不调 LLM。** 让 LLM 审 LLM，审查者自己也会幻觉，还要付钱。这里五道关卡
   全部可复算：有引用 → id 存在 → 在证据池内 → 极性一致 → 方面一致。

其中极性关卡用的正是毕设自训练的那个模型——它在这里从"被展示的成果"变成了
"系统内部的质检工序"：结论断言"差评"，被引评论就必须被本地模型判为消极。

**这一关也踩过坑。** 最初用句向量余弦判断"引用切不切题"，拿评论库造了 196 条带标签
对照集一标定，准确率只有 **59%**，约等于抛硬币——SIF 权重衡量的是"在电商评论语料里
罕见"，而分析师写的书面语（"存在""用户""对此"）在评论语料里恰恰罕见，权重反被顶高。
所以语义关卡被降级成可选项、默认关闭，改用上面两条确定性关卡。标定可复现：

```bash
python scripts/calibrate_critic.py
```

产出 `unsupported_rate`。`scripts/demo_supervisor.py` 跑一趟能看到它从
**75% 降到 0%**——第一版四条结论里三条分别踩中"编造 id""无引用""极性相反"，
重写后两条全部通过。这两个数是算出来的，不是简历上一个查无实据的百分比。

### 评测

```bash
python -m evals                    # 不需要 API key 的部分：Critic + 工具层本地半边
python -m evals tools --llm        # 工具层三维对比，含 LLM（需 AGENT_API_KEY）
python -m evals agent --limit 5    # 40 条 agent 任务集（需 AGENT_API_KEY）
python -m evals --md report.md     # 结果写成 Markdown
```

**工具层：本地模型 vs LLM zero-shot**（500 条 test.txt 抽样，同一套标签）

|  | 准确率 | p50 延迟 | p95 延迟 | 成本 / 千条 |
|---|---|---|---|---|
| 本地 TextCNN | 87.2% | **0.34 ms** | **0.91 ms** | **≈ $0** |
| deepseek-chat zero-shot | **90.4%** | 47.62 ms | 60.50 ms | $0.0108 |

**这个结果和立项时的预判是反的，照实记下来。** 当初以为会是「准确率接近、成本低
两三个数量级」，实际是 **LLM 准确率高 3.2 个点**，而本地模型赢在另外两个维度：
延迟快 140 倍、边际成本为零。

所以保留自训练模型的论证不是「它更准」，而是**它在 agent 循环里的位置不同**：

| 场景 | 本地 TextCNN | deepseek-chat |
|---|---|---|
| agent 一次判 200 条评论 | 68 ms | 9.5 s + 网络往返 |
| 批处理 10 万条 | 34 秒 · $0 | 79 分钟 · $1.08 |

一个 agent 在一轮推理里要给几百条评论定极性时，47ms/条是走不通的。
这才是「把自训练模型封装成工具」的技术理由——**不是它更准，是它在这个位置上更合适**。

口径说明（会被追问）：本地模型逐条测延迟，LLM 走批量（每批 20 条）再摊到每条，
这对 LLM 有利但是真实用法；本地模型在这批数据的训练集上训过、LLM 没见过，
这对 LLM 不利。500 条抽样的标准误约 1.3 个点，3.2 个点的差距大体可信，
但要下死结论应当跑满 6335 条。

**Critic：抓幻觉的能力与代价**（240 条对照集，四类幻觉各对应一道关卡）

| 配置 | 真有据<br>不该误伤 | 编造 id | 无引用 | 极性错配 | 方面错配 |
|---|---|---|---|---|---|
| 全部关卡 | 86.67% | 100% | 100% | **85.0%** | 100% |
| 关掉极性关卡 | 100% | 100% | 100% | 0% | 100% |
| 关掉方面关卡 | 86.67% | 100% | 100% | 85.0% | 2.5% |
| 只剩结构性关卡 | 100% | 100% | 100% | 0% | 0% |

哪些数字算成绩，哪些不算：

- `编造 id` / `无引用` 是结构性的，查库和正则必然全中——在表里是为了证明关卡接线正确。
- `方面错配` 的负例是用方面关卡自己那份词典构造的，**属于自证**。
- **只有 `极性错配` 是真测量**：负例用数据集 gold_label 构造，关卡用本地模型的预测，两边独立。
- `真有据` 那一列是误伤率的反面。极性关卡换来 85% 的检出，代价是 **13.3% 的正常结论被错误打回**、
  要多改一轮。这个取舍是划算的（打回只是多花一轮，放过则是幻觉进报告），但得说出来。

**Agent 任务集**：40 条，六类意图，其中 **12 条是拒答题**——问的是评论库里根本没有的信息
（售后电话、发货仓库、销量排名）。只有正例的评测集等于没做：一个"什么都答"的 agent
在上面能拿满分。完成率和拒答正确率分开报，否则"什么都答"和"什么都不答"两种坏法会同分。

首轮实跑（deepseek-chat，40 条全跑完，共 $0.90）的结果**只有一半可用**，如实记下来：

| 指标 | 结果 | 可信度 |
|---|---|---|
| 工具调用正确率 | 100% | ✅ 可用 |
| 平均步数 | 10.75 | ✅ 可用 |
| 平均 token / 任务 | 65,009 | ✅ 可用 |
| 单任务成本 | $0.0224 | ✅ 可用 |
| p50 / p95 延迟 | 36.2 s / 110.8 s | ✅ 可用 |
| 任务完成率 | ~~75%~~ | ❌ 作废，见下 |
| 拒答正确率 | ~~75%~~ | ❌ 作废 |
| 引用准确率 | ~~37.9%~~ | ❌ 作废 |

**为什么作废**：当时 `parse_claims` 是"每一行只要够长就算一条结论"。模型在结论列表前
写了一大段推理和证据罗列，一条任务 698 行的产出被数成 **477 条结论、带引用的一条都没有**。
于是"无据结论率"变成了"模型话多"的度量，跟幻觉毫无关系；连带着 Supervisor 触发
"证据不足"兜底，把完成率压低、把拒答正确率抬高。**三个指标全部下游于同一个解析 bug。**

已修：分析师的产出格式改成机器可校验的契约——推理写在标签外，结论放进 `<结论></结论>` 块，
没给块就直接打回，不去解析散文。同一份真实产出重新解析，477 条降为 0 条（正确：那三版
都没有结论块，本来就不该算作已提交的结论）。

同一轮还暴露了第二个 bug：账户余额耗尽（HTTP 402）时，角色的 ReAct 循环把异常兜住、
把自己标成 error 后**正常返回**，Supervisor 照跑不误——40 条任务全部显示 "completed"。
现在角色硬失败会中断整条流水线并如实标记。

> 这两个 bug 都是评测跑出来的，不是读代码读出来的。**这就是做 L5 的理由**：
> 不评测的话，这套东西会一直"看起来能跑"。重跑需要充值。

### 可观测：Agent 执行轨迹页

`/traces`（需登录）。左栏列出每一次执行，右栏画成时间线：每一步是哪个角色、
调了什么工具、传了什么参数、拿回什么、花了多少 token 和多少钱。Critic 那一步
会展开成逐条结论的 ✓/✗ 与打回理由。

轨迹本来就一直在往 `runtime/traces/` 落盘，这一页只是让它能被人读。

> 页面里的每一段文本都当不可信内容处理——`thought` 是模型生成的，`observation`
> 里有爬来的评论原文。前端全程用 `textContent` 组装 DOM，一处 `innerHTML` 都没有；
> 后端对 `run_id` 做十六进制白名单校验，否则 `../` 就能读到仓库里任何 JSON 文件。

---

## 测试

```bash
SHEBI_ALLOW_DEV_SECRET=1 python -m pytest
```

152 个用例，覆盖：归一化层的 train/eval 一致性、eval 下不坍缩、padding_idx、
预训练词向量不被模型加载改写、路径可移植性、指标不可编造、推理缓存与死锁、
端到端情感判定与鉴权；agent 层的 schema 校验与自修复、角色/全局两级预算终止、
副作用拦截、trace 落盘；检索层的词级 MaxSim 排序、索引按库隔离、增量编码；
Critic 五道关卡各自的拦截行为与修订闭环；
以及评测层的指标口径（拒答题不计入完成率、无期望工具的任务不计入工具正确率）
与轨迹接口的路径穿越防护。
缺少模型权重时相关用例自动跳过。

---

## 常用命令

```bash
python main.py --model {cnn,lstm,bilstm,lstm_attention,bilstm_attention} \
               --epochs 10 --batch-size 128 --learning-rate 1e-3

python evaluate.py                 # 评测全部已有权重的模型
python evaluate.py --model cnn     # 只评测一个
python evaluate.py --table         # 只打印已有结果

python compare_logs.py             # 训练日志分析与可视化
```

---

## 目录结构

```
shebi/
├── app.py                # Flask 后端与 REST API
├── inference.py          # 推理资源单例（词表 / 词向量 / 模型）
├── evaluate.py           # 测试集评测，产出真实指标
├── metrics_store.py      # 按模型分开保存的评测结果
├── main.py               # 训练主程序
├── utils.py              # 模型构建与加载
├── cnn_model.py          # TextCNN
├── lstm_model.py         # LSTM / BiLSTM / +Attention
├── data_Process.py       # 分词、词表、序列化
├── compare_logs.py       # 训练日志分析与可视化
├── scraper_api.py        # 京东评论爬虫 API
├── models.py / auth.py   # 用户模型与认证
├── config.py             # 全局配置（路径以项目根为基准）
├── agent/
│   ├── registry.py       # 工具注册表：pydantic schema、校验与自修复
│   ├── tools.py          # 9 个工具
│   ├── store.py          # SQLite 评论库（带 review_id 供溯源）
│   ├── embedding.py      # SIF 句向量 / 词级 MaxSim，复用项目自己的词向量
│   ├── vectorstore.py    # 词级向量索引 + RRF 混合检索
│   ├── aspects.py        # 方面词典与极性词典（工具与 Critic 共用）
│   ├── critic.py         # 确定性引用溯源校验
│   ├── roles.py          # 五个角色的职责与最小工具授权
│   ├── supervisor.py     # 多 Agent 编排与修订闭环
│   ├── react.py          # 手写 ReAct 循环
│   ├── budget.py         # 角色 / 全局两级预算
│   ├── llm.py            # provider 抽象
│   └── trace.py          # 全链路轨迹
├── evals/
│   ├── critic_eval.py    # Critic 抓幻觉的能力：分类型检出率 + 逐关卡消融
│   ├── tools_eval.py     # 工具层三维对比：准确率 / 成本 / 延迟
│   ├── tasks.py          # 40 条 agent 任务集（含 12 条拒答题）
│   ├── agent_eval.py     # 跑任务集，算完成率 / 工具正确率 / 引用准确率
│   └── report.py         # 出可直接贴进 README 的 Markdown
├── scripts/              # 运维脚本
├── tests/                # pytest 用例
└── static/ templates/    # 前端
```

## 技术栈

Flask · PyTorch · SQLAlchemy · gensim · jieba · scikit-learn · Bootstrap 5 · ECharts · SQLite

## 许可证

MIT License

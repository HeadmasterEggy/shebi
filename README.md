# 中文电商评论情感分析系统

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HeadmasterEggy/shebi)

基于 PyTorch 的中文文本情感分析平台，支持 TextCNN / LSTM / BiLSTM / LSTM+Attention /
BiLSTM+Attention 五种模型，含京东评论爬虫、训练调度、日志分析与 Web 服务。

> **分支说明**：`毕设完成分支` 在原毕设基础上做了一轮工程审计与修复，
> 详见下方「本轮修复」与 [SECURITY_CLEANUP.md](SECURITY_CLEANUP.md)。

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

# 离线跑通一次完整循环（不需要 API key，用剧本驱动真实工具）
python scripts/demo_agent.py

# 接真实模型
AGENT_API_KEY=sk-...  python -m agent "这批评论里差评主要集中在什么问题上？"

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
| `search_reviews` | 关键词检索评论，返回带 `review_id` 的原文 | — |
| `get_reviews` | 按 id 取原文，供引用溯源校验 | — |
| `aggregate_reviews` | 整体统计：总量、情感分布、正负比 | — |
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

---

## 测试

```bash
SHEBI_ALLOW_DEV_SECRET=1 python -m pytest
```

74 个用例，覆盖：归一化层的 train/eval 一致性、eval 下不坍缩、padding_idx、
路径可移植性、指标不可编造、推理缓存与死锁、端到端情感判定与鉴权，
以及 agent 层的 schema 校验与自修复、三种预算终止、副作用拦截、trace 落盘。
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
├── scripts/              # 运维脚本
├── tests/                # pytest 用例
└── static/ templates/    # 前端
```

## 技术栈

Flask · PyTorch · SQLAlchemy · gensim · jieba · scikit-learn · Bootstrap 5 · ECharts · SQLite

## 许可证

MIT License

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

## 第二轮修复：Web 层与训练调度

第一轮修的是模型和推理；这一轮把 Web 层和训练调度从头审了一遍，下面每一条都有对应的回归测试
（`tests/test_api_hardening.py`、`tests/test_training_control.py`）。

### 训练调度：暂停是假的，停止永远 404，跑久了会卡死

| 现象 | 根因 | 修复 |
|---|---|---|
| 训练跑一会儿就不动了 | 子进程 `stdout=PIPE` 但没人读，tqdm 写满 64KB 管道缓冲后阻塞在 `write()` | 输出重定向到 `runtime/training.log`，进度接口读它的末尾 |
| 暂停按钮无效 | 只改了 progress.json 的 status，`main.py` 从不读 | 新增控制文件 `runtime/train_control.json`，训练循环在 batch 之间响应暂停 / 继续 / 停止 |
| 停止接口永远 404 | 去取 `app.config['TRAINING_PROCESS']`，这个键从没人写过 | `training_control.TrainingProcessManager` 持有进程句柄；默认优雅停止，`{"force": true}` 强杀 |
| 进程崩溃后界面一直"训练中" | `/api/train/progress` 注册了两次，带退出检测的那个永远路由不到 | 合并为一个，进程退出而状态未到终态时按退出码改成 completed / failed |
| 线上用的是 last 而不是 best | 训练结束后把最后一轮权重写回 `*_model_best.pkl` | 删掉这次覆盖，best 只在验证集准确率创新高时保存 |
| 训练完线上还是旧模型 | 推理单例加载一次永久缓存 | 按权重文件 (mtime, size) 指纹检测变化，自动重载 |
| 普通用户能启动训练 | `/api/train/start` 只要求登录 | 改为管理员；参数做类型与范围校验；同一时刻只允许一个训练进程 |
| 进度接口偶发"格式错误" | Web 与训练进程同时读写 progress.json | 所有 JSON 一律先写临时文件再 `os.replace` |
| `val_loss` 15.1 对着 `train_loss` 0.35 | 验证损失返回的是全部 batch 之和 | 改为平均每 batch |

### `/api/analyze`：结果与原文错位

清洗后为空的句子（纯标点、表情、数字、停用词）会被丢掉，但原实现按下标把原文回填到结果上——
输入「！！！\n质量很差」，「质量很差」的判定被贴到了「！！！」头上，后面每一句都错位一格。
现在逐句记住原文下标，被跳过的句子放进 `skipped` 字段；全部为空时返回 400。

顺带：推理不再经过临时文件中转（原来模型加载失败时临时文件不会被删），
请求体做类型校验，文本 / 句子数有上限，500 不再把异常原文回给客户端。

### 鉴权与账号

- `/api/*` 未登录返回 **401 JSON**、非管理员返回 **403 JSON**。原先一律 302 到 HTML 页面，
  前端 `fetch` 跟随重定向后 `response.json()` 抛 "Unexpected token <"
- 登录的 `next` 参数只接受站内相对路径（`//evil.com`、`/\evil.com` 曾能形成开放重定向）
- **改一次密码丢一次邮箱**：改密码的请求不带 email，旧代码 `email != current_user.email` 成立，
  把邮箱写成了 None。管理员接口同理：只改密码时 `is_admin` 默认成 False，顺手把对方降级
- 不能取消自己的管理员权限，也不能降级最后一个管理员
- 注册 / 建号校验用户名、邮箱格式与密码长度（≥ 8）
- 另存模型的文件名经 `secure_filename`，`../../app` 不再能写到仓库任意位置
- 用户设置只接受标量值、限制键名与大小，落盘到 `runtime/user_settings/`

### 删掉的代码

| 删除 | 理由 |
|---|---|
| `eval.py` | 只有 `val_accuracy` 被用到（已挪进 `main.py` 并修正）；其余与 `evaluate.py` 重复，且每次验证都往 `metrics_log.csv` 追加——正是早期界面编造指标的数据源 |
| `web.py` | 交互式 `input()` 爬虫脚本，与 `scraper_api.py` 重复 |
| `/api/info`、`/auth/api/user` | 调试用的端点清单（内容早已过时）与重复路由 |
| `torch.serialization.add_safe_globals(...)` | 加载走 `weights_only=False`，这段从未生效 |
| `User.is_administrator / check_password / generate_password_hash` | 与 `is_admin / verify_password / set_password` 重复 |

---

## 第三轮修复：爬虫、前端 XSS 与跨域

### 爬虫：匿名 SSRF

`/api/scrape/product` **不需要登录**，并把请求里的 URL 原样交给服务器上的 Chromium 打开。
任何匿名访问者都能让服务器的浏览器去访问任意地址——内网服务、`http://169.254.169.254`
云元数据接口、本机管理端口。

现在：

- 接口要求登录；同一时刻只允许一个抓取任务（每个任务占一整个浏览器），忙时返回 409
- 只接受京东商品页域名，并且**交给浏览器的 URL 由商品 ID 重新拼出**，用户传入的字符串本身永远不会被打开
- 浏览器在 `finally` 里关闭（原来每次异常泄漏一组 Chrome 进程）
- 解析拆成纯函数 `parse_comment_payloads`，可以脱离浏览器测试；去重、清理空白
- 结果写入 `runtime/scraped/`（原来在当前工作目录下建 `csv/ txt/ cache/`），并**去重入库**到 agent 评论库——
  `scrape_reviews` 工具从"永远报错"变成真能用，抓完即可 `search_reviews(product_id=...)`
- `DrissionPage` 改为延迟导入：没装它，Web 服务照样能起，只是抓取接口返回 503

### 前端：存储型 XSS

逐句分析结果用 `innerHTML` 拼接 `sentence.text`，而这段文本可能来自**爬取的第三方评论**。
一条内容为 `<img src=x onerror=...>` 的评论，就能在查看结果的用户会话里执行脚本。用 Playwright
在真实浏览器里对照验证过：旧代码注入成功，新代码只显示字面文本。

同类问题一并处理：用户管理表格里的用户名 / 邮箱（还被拼进了 `onclick='...'` 属性）、词频标签与词云
提示框、历史记录的标题与预览、文件名、训练日志、toast。统一走 `shared.js` 里的 `escapeHtml`；
训练日志改用 `textContent`。

### 跨域与 Cookie

- 删掉全局 `CORS(app)`：前端与 API 同源，根本不需要跨域，原配置对任意来源放行全部接口。
  `Flask-Cors` 依赖随之移除
- 会话 Cookie 设为 `HttpOnly` + `SameSite=Lax`，跨站表单 POST 不再携带登录态
- 补 `X-Content-Type-Options: nosniff`、`X-Frame-Options: SAMEORIGIN`、`Referrer-Policy`

### 训练入口统一

- agent 的 `train_model` 原来自己 `Popen` 一个训练进程，绕过了单实例约束——可以和 Web 端的训练
  同时跑、同时覆盖同一份权重。现在与 Web 端共用 `training_control.manager`
- `main.py` 原来无条件读取 `runtime/params.json` 作为默认值，命令行直接训练会悄悄继承 Web 端上一次
  的超参。现在只有显式传 `--params` 时才读

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
| `scrape_reviews` | 抓取京东商品评论并去重入库 | 需人工确认 |
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

**工具层：本地模型 vs LLM zero-shot**（**全量 6335 条** test.txt，同一套标签）

|  | 准确率 | p50 延迟 | p95 延迟 | 成本 / 千条 |
|---|---|---|---|---|
| 本地 TextCNN | 89.47% | **0.43 ms** | **1.07 ms** | **≈ $0** |
| deepseek-chat zero-shot | 90.36% | 61.38 ms | 78.64 ms | $0.011 |

准确率**差 0.89 个百分点**，延迟差 **143 倍**，成本差到没法比。这就是把自训练模型
封装成 agent 工具的技术论证：

| 场景 | 本地 TextCNN | deepseek-chat |
|---|---|---|
| agent 一轮判 200 条评论 | 86 ms | 12.3 s + 网络往返 |
| 批处理 10 万条 | 43 秒 · $0 | 102 分钟 · $1.10 |

一个 agent 在一轮推理里要给几百条评论定极性时，61 ms/条是走不通的。

> **抽样噪声的教训，记在这儿。** 先跑的 500 条抽样给出的是「LLM 高 3.2 个点」，
> 据此一度把论证改成了「本地模型输了准确率、赢在延迟和成本」。跑满 6335 条之后
> 差距收敛到 0.89 个点——**那 3.2 个点大半是抽样噪声**（500 条时单边标准误约 1.3 点）。
> 结论性的数字不要拿抽样估计去写。顺带一个交叉验证：本表的 89.47% 与
> `evaluate.py` 独立跑出的 89.49% 相差一个样本，两条推理通路对得上。

口径说明（会被追问）：本地模型逐条测延迟，LLM 走批量（每批 20 条）再摊到每条，
这对 LLM 有利但是真实用法；本地模型在这批数据的训练集上训过、LLM 没见过，
这对 LLM 不利。所以结论只能说「在这个领域的数据上本地模型不输」，
**不能**推广成「小模型优于大模型」。

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

实跑（deepseek-chat，40 条全部执行成功，$0.88）：

| 指标 | 结果 | 说明 |
|---|---|---|
| 引用准确率 | **94.0%** | 100 条结论中 94 条有据，由 Critic 确定性判定 |
| 任务完成率 | 85.71% | 非拒答题（28 条）里产出可用答案的比例 |
| 拒答正确率 | 75.0% | 12 条拒答题 |
| 工具调用正确率 | 100% | |
| 平均步数 / token | 10.12 / 65,925 | |
| 单任务成本 | $0.0221 | 全集 $0.8849 |
| p50 / p95 延迟 | 39.6 s / 97.7 s | |

分类别看，`aspect` `retrieval` `compare` 三类 20 条全过，失败集中在两处，
**两处都是评测或设计本身的问题，不是模型不行**：

- **`meta` 只过了 1/4**，`aggregate` 掉了 1 条。同一个根因：引用机制只认
  `review_id`，等于假设"每条结论都是在说评论里的内容"。而"当前模型准确率是多少"
  "一共有多少条评论"这类问题，答案来自工具返回值，**没有哪一条评论能支撑它**——
  agent 明明查到了正确答案，却被判成"证据不足"。已修：新增 `[source: 工具名]`
  这第二种出处，且只能引本轮真正调用过的工具，编一个没调过的名字照样过不了。
- **`refusal` 掉了 3 条**，其中 2 条是**题目出错了**：问"评论里提到的那位客服叫什么名字"
  和"厂家保修政策是几年"，评论里恰好真有（`review_id` 1466 写了客服名，多条写了
  保修一年），agent 带着引用答对了却被判成"该拒答却没拒答"。已换成评论里确实不存在的
  信息。第 3 条是拒答检测漏判：模型答的是"证据**严重**不足"，中间插了两个字，
  不含"证据不足"这个连续子串。词表已补。

> 这三处都是**评测跑出来的**，不是读代码读出来的。修完需要重跑才能拿到新数字，
> 上表是修复前的实测值。

### 三轮实跑，两轮作废

评测本身踩的坑值得记下来，这比一张漂亮的表更能说明问题：

| 轮次 | 结果 | 根因 |
|---|---|---|
| 第一轮 | 三个质量指标全废 | `parse_claims` 把散文当结论：698 行产出被数成 477 条"无据结论" |
| 第二轮 | 25/40 任务失败 | 一次瞬时 TLS 抖动。**LLM 调用当时没有重试** |
| 第三轮 | 40/40 干净 | 上表数据 |

第二轮还暴露了一个更隐蔽的问题：余额耗尽时，角色的 ReAct 循环把异常兜住、
把自己标成 error 后**正常返回**，Supervisor 照跑不误——40 条任务全部显示 `completed`。
现在角色硬失败会中断整条流水线并如实标记，第二轮那 25 条因此才被正确记成失败
（而不是伪装成完成）。

补齐的三项工程能力，每一项都是被评测逼出来的：

- **LLM 调用重试**：指数退避 + 随机抖动，区分该重与不该重（429/5xx/网络错误重试，
  400/401/402/403 立刻报出——余额不足重试三次只会把同一个错误延迟三倍）
- **角色硬失败中断流水线**，不再被静默吞掉
- **评测口径分离**：基础设施失败不进质量指标的分母，否则得到的不是
  "agent 只完成了 37%"，而是"评测那天网不好"；成本和延迟仍按全部任务算，
  失败的那些也真的花了钱

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

264 个用例，覆盖：归一化层的 train/eval 一致性、eval 下不坍缩、padding_idx、
预训练词向量不被模型加载改写、路径可移植性、指标不可编造、推理缓存与死锁、
端到端情感判定与鉴权；agent 层的 schema 校验与自修复、角色/全局两级预算终止、
副作用拦截、trace 落盘；检索层的词级 MaxSim 排序、索引按库隔离、增量编码；
Critic 五道关卡各自的拦截行为与修订闭环；
以及评测层的指标口径（拒答题不计入完成率、无期望工具的任务不计入工具正确率）
与轨迹接口的路径穿越防护；Web 层的参数校验、权限码、开放重定向、结果对齐，
训练控制通道的暂停 / 停止 / 大输出不死锁 / 单实例约束。
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
├── inference.py          # 推理资源单例（词表 / 词向量 / 模型），权重变化自动重载
├── training_control.py   # Web ↔ 训练进程控制通道（暂停 / 停止 / 日志 / 单实例）
├── evaluate.py           # 测试集评测，产出真实指标
├── metrics_store.py      # 按模型分开保存的评测结果
├── main.py               # 训练主程序
├── utils.py              # 模型构建与加载
├── cnn_model.py          # TextCNN
├── lstm_model.py         # LSTM / BiLSTM / +Attention
├── data_Process.py       # 分词、词表、序列化
├── compare_logs.py       # 训练日志分析与可视化
├── scraper_api.py        # 京东评论爬虫：URL 白名单 / 抓取 / 解析 / 入库
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

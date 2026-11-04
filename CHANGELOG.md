# 工程记录

毕设完成后（快照见 `thesis` 分支）在 `main` 上做的逐轮审计与修复。每一轮都只写**查出来的问题、
根因和修复方式**，数字均为实测；对应的回归测试在 `tests/` 下。

| 轮次 | 主题 | PR |
|---|---|---|
| 一 | TextCNN 推理坍缩、单次推理 17s、编造的指标、写死的路径、敏感数据入库 | — |
| 二 | 训练调度（卡死 / 假暂停 / 停止 404）、分析结果错位、鉴权与账号 | #7 |
| 三 | 爬虫匿名 SSRF、前端存储型 XSS、全局 CORS、训练入口统一 | #8 |
| 四 | 删除爬虫、随机模型冒充结果、编造的混淆矩阵、容器部署 | #9 |
| 五 | 依赖漏洞清零（29 个）、权重加载禁用任意反序列化、CI 漏洞审计 | #10 |
| 六 | 撤销删除使用公开密码、盲猜接口、无人读取的派生数据 | #11 |
| 七 | 剩余前端注入点、README 拆分 | #12 |
| 八 | 训练表单的结构参数：被忽略的、以及训练出来却加载不了的 | #13 |
| 九 | 请求体大小不设上限 | #14 |
| 十 | 训练日志分析脚本的路径与编码；不再提交生成的图表 | #15 |
| 十一 | 改密码不会让已登录的会话失效 | #16 |
| 十二 | 上传 GBK 编码的文本文件变成乱码 | #17 |
| 十三 | 任何注册用户都能读到全部 agent 运行轨迹 | #18 |
| 十四 | 反向代理之后可定向锁死任意账号；HTTPS 下 Cookie 未标 Secure | #19 |
| 十五 | 导出 CSV 的公式注入；混淆矩阵从未被导出 | #20 |
| 十六 | 注册不设限：可批量注册账号、消耗 CPU、枚举用户名 | #21 |

---

## 第一轮：模型与推理

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

> 爬虫模块已在第四轮整体删除，下面「爬虫」一节仅作历史记录。

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

## 第四轮：删掉爬虫、不再给假结果、能正经部署

### 删除爬虫

项目的核心是本地情感模型与基于它的分析，评论来源是用户粘贴 / 上传的文本和自带的数据集，
不需要现场去电商网站抓。爬虫带来的只有负担：一个要在服务器上驱动真实 Chromium 的接口
（上一轮刚修完它的 SSRF）、一个约 2MB 的浏览器自动化依赖和三个从未被导入的 HTTP/HTML 解析库。

删除：`scraper_api.py`、`static/js/webscraper.js`、前端的商品链接输入面板与样式、agent 的
`scrape_reviews` 工具、`DrissionPage` / `requests` / `beautifulsoup4` / `lxml` 四个依赖。
`scripts/demo_agent.py` 里演示"副作用工具被拦下"的那一步改用 `train_model`。

### 不再用随机模型冒充结果

权重文件缺失或加载失败时，`initialize_model` 原来只打一条警告，然后返回**随机初始化的模型**——
接口照常返回看起来像样的置信度。刚 clone、Git LFS 只拉到 133 字节指针文件时，五个模型全是这样。
现在抛 `ModelNotAvailable`，`/api/analyze` 返回 503 并提示去训练，agent 工具返回明确的错误。

这一改还揪出一条测试：`test_collector_findings_reach_the_analyst` 在没有权重的 CI 上
其实一直跑在随机模型上。它测的是黑板交接，现在把分类换成了固定桩。

### 混淆矩阵是编的

界面上的混淆矩阵用「本次输入里各类句子数 × 测试集准确率」现算——用户输入没有真实标签，
四个格子是编出来的，坐标轴还接反了。现在 `evaluate.py` 记录测试集上的真实混淆矩阵，
界面只画它并标明「测试集混淆矩阵（N 条）」；没评测过就隐藏图表，指标卡显示「未评测」
（原来显示一个光秃秃的 `%`）。

### 部署

- **Dockerfile 的健康检查一直是坏的**：探测的 `/api/info` 需要登录，第二轮又把它删了。
  新增无需登录的 `/healthz`
- `python app.py` 是 Flask 开发服务器；容器改用 gunicorn（`wsgi.py`，单 worker 多线程——
  训练进程管理器与模型缓存是进程内状态）
- `docker-compose.yml` 原来默认 `SHEBI_ALLOW_DEV_SECRET=1`，不设密钥也能起，会话签名用的是
  源码里的公开字符串，等于谁都能伪造登录态。现在必须提供 `SECRET_KEY`
- `vercel.json` 关闭 Vercel 自动部署，原因见「快速开始 → Docker」

### 其他

- 登录限流：同一 IP + 用户名 10 分钟内失败 5 次后暂时拒绝（429）
- 停用词改用集合、清洗正则改为模块级预编译：2000 条评论清洗 + 分词 1.40s → 0.66s
- `data_Process.py` 不再在导入时调用 `logging.basicConfig` 改写全局日志配置

---

## 第五轮：依赖漏洞清零

GitHub 在默认分支上报了 21 个 Dependabot 告警（1 个严重、2 个高危）。本地用 `pip-audit`
对实际安装的环境复核，共 **29 个已知漏洞**：

| 包 | 原版本 | 漏洞数 | 处理 |
|---|---|---|---|
| torch | 2.4.1 | 24 | 升级到 2.15.0 |
| Werkzeug | 3.0.3 | 11 | 升级到 3.1.9 |
| Flask | 3.0.3 | 2 | 升级到 3.1.3 |
| pytest | 8.3.3 | 2 | 升级到 9.2.0 |
| Flask-Cors / requests / lxml | — | 4 | 第三、四轮已删除这三个依赖 |

升级后复核：**No known vulnerabilities found**。只升有漏洞的包，其余版本不动，
在新版本上重新训练、评测、跑全量测试均通过。

顺带两处：

- **权重加载不再做任意 pickle 反序列化**。原来 `torch.load(..., weights_only=False)`，
  换掉一个 `.pkl` 就能在服务进程里执行代码。`main.py` 一直只保存 `state_dict`
  （仓库里的权重也是在这之后提交的），改为 `weights_only=True`；有回归测试往权重文件里
  塞一个会写文件的 pickle，断言加载失败且文件没被写出
- **CI 加依赖漏洞审计**：有已知漏洞就变红。CPU 版 torch 的版本号带 `+cpu`，PyPI 上查不到会被
  `pip-audit` 跳过——而 torch 恰恰是漏洞最多的那个，所以按基础版本号单独审计一次。
  静态检查也从只查根目录扩到 `agent/ evals/ scripts/ tests/`

---

## 第六轮：用户管理里的公开密码与盲猜接口

### 撤销删除后，账号密码是仓库里公开的字符串

管理后台的"撤销删除"会把被删用户重新建出来，密码写死为 `tempPassword123`——这个字符串就在
`admin.js` 里，任何读过源码的人都能登录被恢复的账号。现在恢复时用浏览器的 CSPRNG
（`crypto.getRandomValues`）生成 16 位随机密码，只向管理员显示一次，并明说原密码无法还原
（哈希已随账号删除）。浏览器实测：随机密码能登录，`tempPassword123` 不能。

### 删除 / 修改用户时向"可能的"地址盲猜

编辑、删除、修改三个操作都会依次向三四个"可能的" API 地址发请求（其中多数从来不存在），
全部失败后再补发一次。读操作只是每次留下几条 404；**删除被正常拒绝时**（如"不能删除最后一个
管理员"），`DELETE` 会被换着地址重发、最后再补发一次。现在每个操作只请求它唯一的真实接口，
浏览器实测编辑 / 删除 / 撤销各一个请求，服务端零 404。

### 删除无人读取的派生文件

`word2vec/{train,val,test}_{data,label}.txt` 共约 12MB，是 `data_Process.py` 一次性导出的
索引矩阵，没有任何代码读取——训练和评测都直接从 `data/*.txt` 现算。删除文件及导出代码。
`data/data.xlsx`、`data/trained.txt` 与 `data/dataProcess/*.ipynb` 是数据集从原始表格到
train / val / test 划分的来源，保留以便复现。

---

## 第七轮：收尾

- 个人资料页的活动时间线只转义了标题；时间、描述以及 CSS 类名（来自 `localStorage`，可被改写）
  现在一并转义，类名走白名单
- 导入设置文件时，文件里的值被直接拼进 CSS 选择器，格式不对就抛 `DOMException`、整个导入中断；
  改用 `CSS.escape`
- README 中逐轮的修复记录（约 280 行）移到本文件，README 回到"怎么用"

---

## 第八轮：训练表单里的结构参数

训练页面让用户选卷积核数量、卷积核尺寸、优化器、LSTM 隐藏维度与层数，实际情况是：

| 参数 | 原来 |
|---|---|
| 优化器 | 后端照收不误，`main.py` 一律用 Adam |
| 卷积核尺寸 | 前端以 `kernel_sizes` 提交，后端不认这个字段，一律 (2, 3, 4) |
| 卷积核数量 | `main.py` 解析了 `--num-filters`，但没传给模型，一律 128 |
| LSTM 隐藏维度 / 层数 | 训练时生效；**推理时按 Config 默认值建模型**，权重装不进去 |

最后一条最糟：网页上把 hidden_dim 改成 64 训练出来的权重，`load_state_dict` 报 size mismatch，
这个模型从此加载不了；而且异常是裸 `RuntimeError`，接口只能回一个笼统的 500。

修复：

- 决定权重形状的结构参数随权重写进 `*_model_best.arch.json`，推理端据此重建同样结构的模型；
  没有这个文件的旧权重按 Config 默认值加载，行为不变
- 权重与结构描述都原子写入（先结构、后权重；推理引擎按权重指纹重载），Web 进程不会读到写了一半的文件
- 结构对不上时抛 `ModelNotAvailable`（503 + 提示重新训练），不再是 500
- 优化器（adam / sgd / rmsprop）与卷积核尺寸真正传到训练里；前后端的优化器列表由测试锁定一致

实测：命令行以 `--num-filters 32 --filter-sizes 3,5 --optimizer sgd` 训练一轮，日志里是两个
32 通道、核宽 3 / 5 的卷积层；`evaluate.py` 与推理引擎都按记录的结构加载成功。

---

## 第九轮：请求体大小不设上限

`app.py` 没有设置 `MAX_CONTENT_LENGTH`，Flask 会把任意大的请求体整个读进内存再解析。
`/api/analyze` 的 5 万字上限要等 JSON 解析完才检查，而 `/auth/login` 连登录都不需要——
实测一个匿名的 20MB 表单 POST 被完整接收并解析。

现在请求体上限默认 2MB（`SHEBI_MAX_REQUEST_BYTES` 可调），超限返回 413；`/api/*` 返回
JSON 错误，前端照常显示错误信息而不是解析 HTML 失败。上限内最长的分析文本
（5 万个三字节汉字，约 150KB）有测试保证不会被误拦。

---

## 第十轮：训练日志分析脚本

- `compare_logs.py` 写死相对路径 `log` / `model_comparison_results`，不在项目根目录下运行就
  `FileNotFoundError: 'log'`（实测）。改为以项目根为基准，并支持 `--log-dir` / `--output-dir`
- HTML 报告用平台默认编码写入，Windows（GBK）上中文乱码；改为显式 UTF-8
- 显式使用 matplotlib 的 Agg 后端，无显示器的环境也能跑
- `model_comparison_results/`（73 张图 + 汇总表 + HTML，约 26MB）是这个脚本的输出，
  可由 `log/*.csv` 随时重新生成，从仓库移除并加入 `.gitignore`。另外注意：其中 TextCNN 的
  曲线来自第一轮修复 BatchNorm 坍缩**之前**的训练日志：前几轮验证集准确率恒为 49.87%
  （恰好是正类占比），各组超参的最佳值散落在 69%–89%，反映的是那个缺陷而不是模型水平，
  放在仓库里只会误导

---

## 第十一轮：改密码不会让已登录的会话失效

Flask-Login 默认只在会话和"记住我" Cookie 里存用户 id。账号被盗之后改密码，对方已登录的
会话、以及有效期 365 天的"记住我" Cookie 照样能用。实测：两个客户端同时登录，一个改密码，
另一个访问 `/api/profile` 仍返回 200。

修复：`User.get_id()` 返回 `"id:密码哈希指纹"`，`load_user` 用 `hmac.compare_digest` 校验指纹，
对不上即视为未登录。于是：

- 自己改密码：其他会话与"记住我" Cookie 全部失效，当前会话按新指纹重新签发，不用重新登录
- 管理员重置某人的密码：对方所有会话失效；管理员改自己的密码不会把自己踢下线
- 旧格式（只有 id）的会话无法证明密码未变，升级后要求重新登录一次

浏览器实测：改密码后改密码的一方仍为 200、另一个会话变为 401，新密码可正常登录。

---

## 第十二轮：上传 GBK 编码的文本文件变成乱码

文件上传固定用 `readAsText(file, 'UTF-8')` 读取。旧版 Windows 记事本和很多导出工具存的是
GBK，读进来每个汉字都变成 `�`，界面不报错，模型照样对一串替换字符给出"分析结果"。

现在按字节读入后自行解码：先以 `TextDecoder('utf-8', {fatal: true})` 严格解码（非法字节序列
直接失败而不是静默替换），失败再按 GB18030（GBK 的超集）解码，并在文件信息里注明用了哪种编码。

浏览器实测同一段文字的三种文件：UTF-8、带 BOM 的 UTF-8、GBK。旧代码读 GBK 文件得到
`�����ܲ����̫��`；新代码三者均得到原文「质量很差，物流太慢」。

---

## 第十三轮：任何注册用户都能读到全部 agent 运行轨迹

注册是开放的（`/auth/register` 不需要邀请），而 `/traces`、`/api/traces`、`/api/traces/<id>`
只要求登录。于是任何人注册一个账号，就能读到运营者每一次 agent 运行的完整轨迹：提问内容、
每一步的工具调用参数、检索回来的评论原文、token 与花费。

轨迹页是运维观测工具，不是面向普通用户的功能，改为仅管理员可访问；普通用户的导航栏不再显示入口。
浏览器实测：普通用户页面上无该入口、接口返回 403；管理员不受影响。

---

## 第十四轮：部署在反向代理之后

- **定向锁死任意账号**：部署在 nginx / 云负载均衡之后时，`request.remote_addr` 永远是代理的地址，
  而登录限流（第四轮）按 (IP, 用户名) 计数——攻击者对某个用户名连错 5 次，所有人用正确密码登录
  这个账号都得到 429。实测复现。新增 `SHEBI_TRUSTED_PROXIES=N`：显式声明代理层数后才用
  `ProxyFix` 从 `X-Forwarded-*` 还原真实客户端 IP；默认 0，不信任这些头，否则谁都能伪造
  `X-Forwarded-For` 绕过限流（同样有测试）
- **HTTPS 下 Cookie 未标 Secure**：新增 `SHEBI_SECURE_COOKIES=1`，会话与"记住我" Cookie 只经 HTTPS 发送
- 两个开关写进 `docker-compose.yml` 与 README 的部署说明

---

## 第十五轮：导出结果

- **CSV 公式注入**：句子原文只转义了引号就写进 CSV。原文可能来自别人提供的文件，以 `=` `+` `-` `@`
  开头的单元格会被 Excel / WPS 当成公式执行（`=HYPERLINK(...)`、`=cmd|...`），导出后一打开即触发。
  现在这类单元格前加单引号，按文本显示
- **混淆矩阵从未被导出**：导出代码读 `modelMetrics.confusion_matrix`，而接口自第四轮起给的是
  `confusionMatrix`；行列标签也接反了（第 0 行是真实消极）。现在按实际字段与方向导出，并注明是测试集、
  样本量多少

浏览器内对真实的导出函数实测：`=HYPERLINK(...)` 导出为 `'=HYPERLINK(...)`，普通引号照常转义，
生成的 CSV 经 Python `csv` 解析列数正确；混淆矩阵导出为「真实消极 2612 / 542，真实积极 276 / 2905」。

---

## 第十六轮：注册不设限

`/auth/register` 无需登录、不设任何频率限制。实测单个 IP 2.9 秒注册 20 个账号；每次注册还要做一次
约 100ms 的密码哈希（PBKDF2 60 万轮，故意设计得慢），批量注册可以把 CPU 占满。"用户名已被使用"的
提示也能被拿来枚举已存在的用户名。

- 注册按 IP 限流，默认每小时 5 次（`SHEBI_REGISTER_PER_HOUR` 可调），每次提交都计数、不论成败
- `SHEBI_ALLOW_REGISTRATION=0` 可关闭自助注册，此时登录页不再显示注册入口，账号只由管理员创建
- 与第十四轮配合：部署在代理之后时需设 `SHEBI_TRUSTED_PROXIES`，否则所有人共享代理 IP 的配额


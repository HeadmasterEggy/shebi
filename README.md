# Shebi 项目

Shebi 是一个基于 Python 的情感分析与文本分类平台，支持多种深度学习模型（如 TextCNN、LSTM、BiLSTM、带注意力机制的 LSTM 等），并集成了京东商品评论爬取、可视化训练日志分析、用户管理和 Web API 服务等功能。适用于中文文本的情感分析、数据挖掘和自动化文本处理场景。

## 主要功能

- 支持多种深度学习文本分类模型（TextCNN、LSTM、BiLSTM、Attention 等）
- 提供 RESTful API，支持文本情感分析、模型训练、用户管理等
- 集成京东商品评论爬虫，自动采集评论数据
- 训练过程可视化与日志对比分析工具
- 支持用户注册、登录、权限管理（管理员/普通用户）
- 支持模型参数调优与训练进度实时监控
- 结果可视化与 HTML 报告自动生成

## 特性

- 简单易用，接口友好
- 高度可扩展，支持自定义模型与数据
- 支持多用户与权限管理
- 支持中文分词与停用词过滤
- 训练日志与模型性能对比分析
- 支持模型保存与加载

## 安装

```bash
git clone https://github.com/yourusername/shebi.git
cd shebi
pip install -r requirements.txt
```

## 使用方法

### 启动 Web 服务

```bash
python app.py
```

默认端口为 5003，可通过浏览器访问 http://localhost:5003

### 训练模型

```bash
python main.py --model cnn --epochs 10 --batch-size 128
```
或通过 Web 前端/接口发起训练任务。

### 爬取京东商品评论

通过 Web 前端或 API `/api/scrape/product` 提交商品链接，自动采集评论数据。

### 日志分析与可视化

```bash
python compare_logs.py
```
生成训练日志对比图和 HTML 报告，结果保存在 `model_comparison_results/` 目录。

## 目录结构

```
shebi/
├── main.py
├── app.py
├── compare_logs.py
├── ...
├── README.md
└── requirements.txt
```

## 贡献

欢迎提交 issue 和 pull request！

## 许可证

MIT License

# Shebi 情感分析系统

Shebi 是一个基于 Python 的中文文本情感分析平台，支持多种深度学习模型（TextCNN、LSTM、BiLSTM、Attention 等），集成了京东商品评论爬取、训练日志分析、用户管理和 Web API 服务。适用于中文文本情感分析、数据挖掘和自动化文本处理。

## 主要功能

- 多模型支持：TextCNN、LSTM、BiLSTM、Attention 等
- RESTful API：文本情感分析、模型训练、用户管理
- 评论爬虫：自动采集京东商品评论
- 训练日志分析与可视化
- 用户注册、登录、权限管理（管理员/普通用户）
- 支持模型参数调优与训练进度监控
- 结果可视化与 HTML 报告自动生成

## 技术栈

- 后端：Flask、PyTorch、SQLAlchemy、Jieba
- 前端：Bootstrap 5、ECharts、JavaScript
- 数据库：SQLite

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
浏览器访问 http://localhost:5003

### 训练模型

```bash
python main.py --model cnn --epochs 10 --batch-size 128
```

### 爬取京东商品评论

通过 Web 前端或 API `/api/scrape/product` 提交商品链接。

### 日志分析与可视化

```bash
python compare_logs.py
```
结果保存在 `model_comparison_results/` 目录。

## 目录结构

```
shebi/
├── app.py                # 主Web后端
├── main.py               # 训练主程序
├── utils.py              # 模型工具函数
├── compare_logs.py       # 日志分析与可视化
├── scraper_api.py        # 评论爬虫API
├── web.py                # 命令行爬虫脚本
├── config.py             # 配置文件
├── cnn_model.py          # TextCNN模型
├── lstm_model.py         # LSTM/BiLSTM模型
├── data_Process.py       # 数据处理与分词
├── models.py             # 用户与数据库模型
├── requirements.txt
└── static/               # 前端静态资源
```

## 许可证

MIT License

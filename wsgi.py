# -*- coding: utf-8 -*-
"""生产环境入口：gunicorn wsgi:app

`python app.py` 用的是 Flask 自带的开发服务器，单线程、不适合对外服务。

只能开 1 个 worker：训练进程管理器与模型缓存都是进程内状态，多 worker 时
"同一时刻只有一个训练任务"的约束会失效，每个 worker 也各自加载一份模型。
并发靠 --threads。
"""

from app import _warmup, app

__all__ = ["app"]

_warmup()

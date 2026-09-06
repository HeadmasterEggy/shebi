FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，让代码改动不会让依赖层缓存失效
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# 预训练词向量走 Git LFS，clone 时常常只拿到指针文件；
# 仓库内的 word_vec.txt 足以重建一份等价的二进制词向量。
RUN python scripts/rebuild_pretrained_w2v.py || \
    echo "跳过词向量重建（文件可能已存在）"

EXPOSE 5003
ENV HOST=0.0.0.0 PORT=5003

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:5003/api/info')" || exit 1

CMD ["python", "app.py"]

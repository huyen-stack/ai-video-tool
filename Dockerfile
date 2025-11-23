# 1. 换车：使用 Python 3.11 (更强的环境，强制刷新缓存)
FROM python:3.11-slim

WORKDIR /app

# 2. 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# 3. 【核武器】强制指定使用官方源下载，防止 Zeabur 内部源抽风
RUN pip install --upgrade pip
RUN pip install google-generative-ai -i https://pypi.org/simple
RUN pip install streamlit opencv-python-headless Pillow numpy -i https://pypi.org/simple

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

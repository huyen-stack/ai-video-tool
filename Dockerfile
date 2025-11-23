# 1. 使用 Python 3.10
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统基础库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制文件
COPY . .

# 5. 【关键修改】直接在命令里安装，不读取 requirements.txt 文件
# 这样可以避开所有文件格式、编码、隐藏字符的问题
RUN pip install --upgrade pip
RUN pip install streamlit google-generative-ai opencv-python-headless Pillow numpy

# 6. 开放端口
EXPOSE 8501

# 7. 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

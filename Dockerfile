# 1. 指定使用 Python 3.10
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 【修正】安装 OpenCV 需要的系统库 (这里改好了!)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制你的代码到服务器
COPY . .

# 5. 安装 Python 依赖库
RUN pip install --no-cache-dir -r requirements.txt

# 6. 告诉服务器我们要用 8501 端口
EXPOSE 8501

# 7. 启动软件的命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

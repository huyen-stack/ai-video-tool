# 1. 指定使用 Python 3.10 (稳定版，完美支持 Google AI)
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 强制安装 OpenCV 需要的系统“零件” (解决系统库缺失问题)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
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

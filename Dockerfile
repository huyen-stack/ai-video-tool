# 1. 使用完整版 Python 3.10 (比 slim 版更强壮，包含更多工具)
FROM python:3.10

# 2. 设置工作目录
WORKDIR /app

# 3. 安装 OpenCV 需要的系统库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制你的代码
COPY . .

# 5. 【关键修改】先强制升级 pip (安装工具)，再安装依赖库
# 这一步能解决 "No matching distribution" 的报错
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6. 开放端口
EXPOSE 8501

# 7. 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

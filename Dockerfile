FROM python:3.10-slim

WORKDIR /app

# 1. 安装系统库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# 2. 升级 pip
RUN pip install --upgrade pip

# 3. 【决胜一招】使用清华大学镜像源安装
# 这通常能解决腾讯云节点连不上官方库的问题
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    google-generative-ai \
    streamlit \
    opencv-python-headless \
    Pillow \
    numpy

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

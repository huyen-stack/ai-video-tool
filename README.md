# AI Video Tool (免 SDK / 纯 HTTP)

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署（Streamlit Cloud / Zeabur）
建议设置环境变量：
- ZAI_API_KEY
- ZAI_BASE_URL（默认 https://open.bigmodel.cn/api/paas/v4）
- ZAI_TEXT_MODEL（默认 glm-4.5）
- ZAI_VISION_MODEL（默认 glm-4.6v）

import streamlit as st
import google.generativeai as genai
import tempfile
import os
import cv2
from PIL import Image
import concurrent.futures

# ========================
# 工具函数
# ========================

def extract_6_keyframes(video_path: str):
    """从视频中等间隔抽取 6 张关键帧，返回 PIL.Image 列表。"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    # 按 7 等分取 6 个位置
    frame_indices = [int(total_frames * (i + 1) / 7) for i in range(6)]
    images = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))
        else:
            # 占位灰图，避免后面出错
            images.append(Image.new("RGB", (200, 200), color="gray"))

    cap.release()
    return images


def analyze_single_image(img: Image.Image, model):
    """调用 Gemini 对单张图片做一句话分镜描述。"""
    try:
        prompt = "请用一句中文简述画面内容（例如人物动作）和景别（特写/中景/全景）。"
        # Gemini 多模态：文字 + 图片
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"分析失败：{e}"


def analyze_images_concurrently(images, model):
    """并发分析多张图片，提升速度。"""
    if not images:
        return []

    descriptions = [""] * len(images)
    status = st.empty()
    status.info("⚡ 正在启动多线程 AI 分析，请稍候...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(images)) as executor:
        future_to_index = {
            executor.submit(analyze_single_image, img, model): i
            for i, img in enumerate(images)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                descriptions[i] = future.result()
            except Exception as e:
                descriptions[i] = f"分析失败：{e}"

    status.empty()
    return descriptions


# ========================
# Streamlit 页面配置
# ========================

st.set_page_config(
    page_title="AI 极速分镜助手",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI 视频分镜分析（六格极速版）")
st.markdown("上传视频，自动截取 6 张关键帧，并并发调用 Gemini 做一句话分镜解说。")

# ========================
# 侧边栏：API Key
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 API Key")
    api_key = st.text_input("输入 Google API Key", type="password", help="粘贴以 AIza 开头的密钥")
    st.markdown("---")
    if not api_key:
        st.warning("🔴 等待输入 Key")
    else:
        st.success("🟢 Key 已就绪")

    st.markdown("### 📝 使用说明")
    st.markdown("1. 粘贴 Key\n2. 上传视频\n3. 点击按钮开始分析")

# ========================
# 初始化 Gemini 模型
# ========================

model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        # 模型名称可以按需改成 gemini-1.5-pro 等
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Key 格式不对或初始化失败：{e}")
        model = None

# ========================
# 主流程：上传视频 + 分镜分析
# ========================

uploaded_file = st.file_uploader(
    "📂 第二步：拖入视频文件（建议 < 50MB）",
    type=["mp4", "mov", "m4v", "avi"]
)

if uploaded_file and st.button("🚀 开始极速分析"):
    if not api_key or model is None:
        st.error("请先输入有效的 Google API Key。")
    else:
        # 1. 保存上传的视频到临时文件
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.info("⏳ 正在从视频中提取 6 张关键帧...")
        images = extract_6_keyframes(tmp_path)

        # 删除临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if not images:
            st.error("❌ 无法从视频中读取帧，请检查视频文件是否损坏。")
        else:
            st.success("✅ 已成功提取 6 张关键帧！")

            # 2. 展示帧图预览
            st.subheader("🖼 截图预览（六宫格）")
            cols = st.columns(3)
            for i, img in enumerate(images):
                with cols[i % 3]:
                    st.image(img, caption=f"第 {i + 1} 张关键帧", use_column_width=True)

            # 3. 调用 Gemini 做分镜解说
            st.subheader("🧠 AI 分镜解说结果")
            with st.spinner("正在调用 Gemini 进行图像理解..."):
                descriptions = analyze_images_concurrently(images, model)

            # 4. 图文对应输出（方便复制到你的视频脚本系统）
            for i, desc in enumerate(descriptions):
                st.markdown(f"**第 {i + 1} 张：** {desc}")

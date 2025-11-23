import streamlit as st
import google.generativeai as genai
import tempfile
import os
import cv2
from PIL import Image
import concurrent.futures


# ========================
# 全局配置
# ========================

# 你可以在这里统一切换模型：
#   - "gemini-flash-latest"
#   - "gemini-2.5-flash-lite"
#   - "gemini-2.5-flash"
GEMINI_MODEL_NAME = "gemini-flash-latest"


# ========================
# 工具函数
# ========================

def extract_6_keyframes(video_path: str):
    """
    从视频中等间隔抽取 6 张关键帧，返回 PIL.Image 列表。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # 按 7 等分取 6 个位置（1/7, 2/7, ... 6/7）
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


def _extract_text_from_response(resp) -> str:
    """
    兼容不同版本 SDK 的 Gemini 响应解析：
    1. 优先用 resp.text
    2. 再从 candidates[].content.parts[].text 里把文本拼出来
    3. 实在不行就把 resp 转成字符串返回，方便调试
    """
    # ① 先试试 resp.text
    text = getattr(resp, "text", None)
    if text and isinstance(text, str) and text.strip():
        return text.strip()

    # ② 尝试走 candidates -> content.parts
    try:
        texts = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)
        if texts:
            return " ".join(texts).strip()
    except Exception:
        pass

    # ③ 兜底：直接转字符串，至少能看到模型返回的大概结构
    try:
        return str(resp)
    except Exception:
        return ""


def analyze_single_image(img: Image.Image, model):
    """
    调用 Gemini 对单张图片做一句话分镜描述。
    """
    try:
        prompt = (
            "你现在是短视频分镜师，请用一句简短中文描述画面："
            "包含【景别：特写/中景/全景】+【主体是谁在做什么】。"
        )
        resp = model.generate_content([prompt, img])
        text = _extract_text_from_response(resp)
        if not text:
            return "分析失败：模型未返回文本内容"
        return text
    except Exception as e:
        return f"分析失败：{e}"


def analyze_images_concurrently(images, model):
    """
    并发分析多张图片，加速整体速度。
    """
    if not images:
        return []

    descriptions = [""] * len(images)
    status = st.empty()
    status.info("⚡ 正在启动多线程 AI 分析，请稍候...")

    max_workers = min(len(images), 6)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    layout="wide",
)

st.title("🎬 AI 视频分镜分析（六格极速版）")
st.markdown(
    "上传一个短视频，我会自动从中截取 **6 张关键帧**，"
    "并用 Gemini 并发生成 **一句话分镜描述**。"
)


# ========================
# 侧边栏：API Key 输入
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 API Key")
    api_key = st.text_input(
        "输入 Google API Key",
        type="password",
        help="粘贴你的 Gemini API Key（通常以 AIza 开头）",
    )

    st.markdown("---")
    if not api_key:
        st.warning("🔴 还没有 Key，先去 https://ai.google.dev/ 申请一个")
    else:
        st.success("🟢 Key 已就绪")

    st.markdown("### 📝 使用步骤")
    st.markdown("1. 在上面输入 API Key\n2. 在主界面上传视频\n3. 点击“开始极速分析”")


# ========================
# 初始化 Gemini 模型
# ========================

model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"❌ 初始化 Gemini 模型失败：{e}")
        model = None


# ========================
# 主流程：上传视频 + 截帧 + AI 分析
# ========================

uploaded_file = st.file_uploader(
    "📂 第二步：拖入视频文件（建议 < 50MB）",
    type=["mp4", "mov", "m4v", "avi", "mpeg"],
)

if uploaded_file and st.button("🚀 开始极速分析"):
    if not api_key or model is None:
        st.error("请先在左侧输入有效的 Google API Key。")
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
            st.error("❌ 无法从视频中读取帧，请检查视频文件是否损坏或格式异常。")
        else:
            st.success("✅ 已成功提取 6 张关键帧！")

            # 2. 展示六宫格截图
            st.subheader("🖼 截图预览（六宫格）")
            cols = st.columns(3)
            for i, img in enumerate(images):
                with cols[i % 3]:
                    st.image(img, caption=f"第 {i + 1} 张关键帧", use_column_width=True)

            # 3. 调用 Gemini 做分镜解说
            st.subheader("🧠 AI 分镜解说结果")
            with st.spinner("正在调用 Gemini 进行图像理解与描述..."):
                descriptions = analyze_images_concurrently(images, model)

            # 4. 图文对应输出，方便复制
            for i, desc in enumerate(descriptions):
                st.markdown(f"**第 {i + 1} 张：** {desc}")

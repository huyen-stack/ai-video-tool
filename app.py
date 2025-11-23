import streamlit as st
import google.generative_ai as genai
import tempfile
import os
import cv2
from PIL import Image
import concurrent.futures

# --- 页面配置 ---
st.set_page_config(page_title="AI 极速分镜助手", page_icon="🎬", layout="wide")

st.title("🎬 AI 视频分镜分析 (六格极速版)")
st.markdown("上传视频，自动截取 6 张关键帧，并并发调用 AI 进行解说。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔑 第一步")
    api_key = st.text_input("输入 Google API Key", type="password", help="粘贴以 AIza 开头的密钥")
    st.markdown("---")
    if not api_key:
        st.warning("🔴 等待输入 Key")
    else:
        st.success("🟢 Key 已就绪")
    st.markdown("### 📝 使用说明")
    st.markdown("1. 粘贴 Key\n2. 上传视频\n3. 点击开始")

# --- 核心功能 1: 截取图片 ---
def extract_6_keyframes(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return []
    
    frame_indices = [int(total_frames * (i + 1) / 7) for i in range(6)]
    images = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))
        else:
            images.append(Image.new('RGB', (200, 200), color='gray'))
    cap.release()
    return images

# --- 核心功能 2: 并发 AI 分析 ---
def analyze_single_image(img, model):
    try:
        prompt = "请用一句中文简述画面内容（例如人物动作）和景别（特写/全景）。"
        response = model.generate_content([prompt, img])
        return response.text
    except:
        return "分析失败"

def analyze_images_concurrently(images, model):
    descriptions = [""] * 6
    status = st.empty()
    status.info("⚡️ 正在启动 6 个 AI 线程同时分析，请稍候...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_index = {executor.submit(analyze_single_image, img, model): i for i, img in enumerate(images)}
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            descriptions[i] = future.result()
            
    status.empty()
    return descriptions

# --- 主程序逻辑 ---
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        st.error("Key 格式不对")

    uploaded_file = st.file_uploader("📂 第二步：拖入视频 (建议 < 50MB)", type=['mp4', 'mov'])

    if uploaded_file and st.button("🚀 开始极速分析"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        with st.spinner('正在截取关键帧...'):
            keyframes = extract_6_keyframes(video_path)
            
        if keyframes:
            descriptions = analyze_images_concurrently(keyframes, model)
            st.success("✅ 分析完成！")
            st.divider()
            
            cols = st.columns(6)
            for i, col in enumerate(cols):
                with col:
                    st.image(keyframes[i], use_column_width=True, caption=f"镜头 {i+1}")
                    st.info(descriptions[i])
                    
        os.remove(video_path)

elif not api_key:
    st.info("👈 请先在左侧输入 Key")

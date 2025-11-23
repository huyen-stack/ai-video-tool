import streamlit as st
import google.generativeai as genai
import tempfile
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures
import json

# ========================
# 全局配置
# ========================

# 可按需替换：
#   "gemini-flash-latest"
#   "gemini-2.5-flash-lite"
#   "gemini-2.5-flash"
GEMINI_MODEL_NAME = "gemini-flash-latest"

# 展示时的图片与色卡宽度
DISPLAY_IMAGE_WIDTH = 320
PALETTE_WIDTH = 320
PALETTE_HEIGHT = 26

# ========================
# 页面 / 全局样式
# ========================

st.set_page_config(
    page_title="AI 自动关键帧分镜助手 Pro",
    page_icon="🎬",
    layout="wide",
)

# 简单全局 CSS，让页面更像一个 Landing Page
st.markdown(
    """
    <style>
    /* 主体背景 & 字体微调 */
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    /* 让 markdown 里的文字颜色更柔和 */
    .stMarkdown, .stText {
        color: #e5e7eb;
    }
    /* code 区块字体稍小一点 */
    .stCode {
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 顶部 Hero 区块
st.markdown(
    """
    <div style="
        padding: 18px 24px;
        border-radius: 18px;
        margin-bottom: 16px;
        background: radial-gradient(circle at top left, #38bdf8 0, #0f172a 45%, #020617 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
    ">
      <h1 style="margin: 0 0 8px 0; color: #e5e7eb; font-size: 1.6rem;">
        🎬 AI 自动关键帧分镜助手 Pro
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        一键解析整条视频，自动抽取关键帧，生成
        <b>分镜脚本 / 主色调色卡 / 剧情大纲 / 10 秒广告旁白</b>，
        做剪辑和广告策划时直接当「AI 导演助理」用。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================
# 工具函数：根据时长自动抽关键帧
# ========================

def extract_keyframes_dynamic(
    video_path: str,
    min_frames: int = 6,
    max_frames: int = 30,
    base_fps: float = 1.5,
):
    """
    根据视频时长自动抽取关键帧：
    - 按时长估算目标帧数：ideal_n = duration * base_fps
    - 在 [min_frames, max_frames] 范围内截取
    - 均匀抽帧（后续可再叠加更复杂的“镜头切换检测”）
    返回 PIL.Image 列表。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 25.0  # 给个默认值

    if total_frames <= 0:
        cap.release()
        return []

    duration = total_frames / fps  # 秒
    ideal_n = int(duration * base_fps)
    target_n = max(min_frames, ideal_n)
    target_n = min(target_n, max_frames, total_frames)

    if target_n <= 0:
        cap.release()
        return []

    # 均匀抽帧：在 [0, total_frames) 上取 target_n 个点
    step = total_frames / float(target_n)
    frame_indices = [int(i * step) for i in range(target_n)]

    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))
        else:
            images.append(Image.new("RGB", (200, 200), color="gray"))

    cap.release()
    return images

# ========================
# 主色调色卡相关
# ========================

def get_color_palette(pil_img: Image.Image, num_colors: int = 5):
    """
    使用 KMeans 聚类提取图片主色调，返回 [(R,G,B), ...]。
    """
    img = pil_img.resize((120, 120))
    arr = np.array(img)
    data = arr.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv2.kmeans(
        data,
        num_colors,
        None,
        criteria,
        10,
        flags,
    )
    centers = centers.astype(int)
    colors = [tuple(map(int, c)) for c in centers]
    return colors


def make_palette_image(colors, width: int = PALETTE_WIDTH, height: int = PALETTE_HEIGHT):
    """
    把一组 RGB 颜色画成一条水平色卡条。
    """
    if not colors:
        return Image.new("RGB", (width, height), color="gray")

    bar = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(bar)
    n = len(colors)
    band_width = max(width // n, 1)

    for i, color in enumerate(colors):
        x0 = i * band_width
        x1 = width if i == n - 1 else (i + 1) * band_width
        draw.rectangle([x0, 0, x1, height], fill=color)

    return bar


def rgb_to_hex(rgb_tuple):
    r, g, b = rgb_tuple
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

# ========================
# 解析 Gemini 返回
# ========================

def _extract_text_from_response(resp) -> str:
    """
    兼容不同版本 SDK 的 Gemini 响应解析：
    1. 优先用 resp.text
    2. 再从 candidates[].content.parts[].text 里把文本拼出来
    3. 实在不行就把 resp 转成字符串返回
    """
    text = getattr(resp, "text", None)
    if text and isinstance(text, str) and text.strip():
        return text.strip()

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

    try:
        return str(resp)
    except Exception:
        return ""

# ========================
# 单帧分析：视听语言 + 语义
# ========================

def analyze_single_image(img: Image.Image, model):
    """
    调用 Gemini 对单张图片做专业级分镜分析：
    景别 / 运镜 / 角度 / 构图 / 色彩与光影 / 内容 / 情绪 / 标签
    """
    try:
        prompt = (
            "你现在是专业的电影分镜师 + 摄影指导 + 短视频运营顾问。"
            "请分析这张画面，并严格使用下面的中文模板输出，简洁但专业：\n\n"
            "【景别】（远景/全景/中景/近景/特写 等）\n"
            "【运镜】（推/拉/摇/移/跟/升降/固定镜头；如无法判断就写“静止镜头”）\n"
            "【拍摄角度】（俯拍/仰拍/平视/上帝视角 等）\n"
            "【构图】（例如：三分法/中心构图/对称构图/前景-主体-背景 等）\n"
            "【色彩与光影】（画面色调：偏暖/偏冷/中性；明暗：高调/低调；可简单描述主色）\n"
            "【画面内容】（一句话描述谁在做什么）\n"
            "【情绪氛围】（例如：轻松、甜蜜、治愈、紧张、压抑、酷炫 等）\n"
            "【关键词标签】（用 #标签 形式给出 3-8 个，例如：#夜景 #自拍 #都市 #暖色调）\n\n"
            "只输出以上 8 行内容，不要加解释或小标题。"
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
    status.info(f"⚡ 正在并发分析 {len(images)} 个关键帧，请稍候...")

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
# 整体视频层面的总结
# ========================

def analyze_overall_video(frame_descriptions, model):
    """
    基于若干关键帧的分析结果，对整段视频做：
    - 剧情大纲
    - 整体视听风格
    - 话题标签
    - 商业与合规风险
    """
    n = len(frame_descriptions)
    joined = "\n\n".join(
        f"第 {i + 1} 帧：\n{desc}" for i, desc in enumerate(frame_descriptions)
    )

    prompt = f"""
你现在是资深视频导演 + 剪辑师 + 短视频运营专家 + 内容合规审核员。
下面是从一段视频中抽取的 {n} 个关键帧的详细说明，请你基于这些说明，对整段视频做整体分析。

=== 关键帧说明开始 ===
{joined}
=== 关键帧说明结束 ===

请严格按下面结构输出中文分析：

【剧情大纲】
用 2-4 句概括这段视频的大致内容/人物关系/发生场景。

【整体视听风格】
从节奏快慢、镜头感、色彩气质（暖/冷/日常/梦幻）、情绪氛围等角度总结整体风格。

【适合的话题标签】
用 #标签 形式给出 5-10 个，适合抖音/小红书/视频号等平台，例如：
#城市夜景 #治愈自拍 #氛围感美女

【商业与合规风险】
从“血腥/暴力/色情/政治/品牌商标”等维度，简单评估：
整体风险级别：低 / 中 / 高
并用 2-3 句话说明需要注意的点（例如：服装暴露程度、未成年人形象、是否有明显品牌 Logo 等）。

请直接输出以上 4 个小节，不要添加额外说明。
"""
    try:
        resp = model.generate_content(prompt)
        return _extract_text_from_response(resp)
    except Exception as e:
        return f"整体分析失败：{e}"

# ========================
# 10 秒广告旁白脚本生成
# ========================

def generate_ad_script(frame_descriptions, model):
    """
    基于若干关键帧的分析，生成一条 10 秒左右的中文广告旁白脚本。
    """
    n = len(frame_descriptions)
    joined = "\n\n".join(
        f"第 {i + 1} 帧：\n{desc}" for i, desc in enumerate(frame_descriptions)
    )

    prompt = f"""
你是一名资深广告导演 + 文案。
我有一个由 {n} 个画面组成的竖版短视频，时长大约 8-12 秒。
下面是每个画面的专业分析，请你基于这些信息，写一条适合配合这些画面播放的中文广告旁白脚本。

=== 关键帧分析 ===
{joined}
=== 关键帧分析结束 ===

要求：
1. 旁白总时长控制在 8-12 秒左右（正常语速），文本 35-70 字即可。
2. 风格与画面调性匹配（如果是氛围感自拍，就偏情绪/生活方式；如果是产品展示，就多讲卖点）。
3. 用自然口语化中文，不要出现“画面中”“镜头里”这类字眼，直接对观众说话。
4. 如果画面看起来像个人生活 vlog，可以弱化“购买号召”，更偏向情绪感染。
5. 如果画面中有明显产品或品牌（如饮料、零食、护肤品等），可以适当加入温柔的“种草话术”。

请严格按照下面格式输出：

【10秒广告旁白脚本】
（在这里写完整的一段旁白，不要拆成多行，不要标注镜头编号）

不要输出其他任何内容。
"""
    try:
        resp = model.generate_content(prompt)
        return _extract_text_from_response(resp)
    except Exception as e:
        return f"广告文案生成失败：{e}"

# ========================
# 侧边栏：API Key 输入
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 Gemini API Key")
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
    st.markdown("1. 在上面输入 API Key\n2. 上传视频\n3. 点击下方按钮一键解析")

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
# 主流程：上传视频 + 抽帧 + 分析 + 布局展示
# ========================

uploaded_file = st.file_uploader(
    "📂 第二步：拖入视频文件（建议 < 50MB）",
    type=["mp4", "mov", "m4v", "avi", "mpeg"],
)

if uploaded_file and st.button("🚀 一键解析整条视频"):
    if not api_key or model is None:
        st.error("请先在左侧输入有效的 Google API Key。")
    else:
        # 1. 保存上传的视频到临时文件
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.info("⏳ 正在根据视频时长自动抽取关键帧...")
        images = extract_keyframes_dynamic(tmp_path)

        # 删除临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if not images:
            st.error("❌ 无法从视频中读取帧，请检查视频文件是否损坏或格式异常。")
        else:
            st.success(f"✅ 已成功抽取 {len(images)} 个关键帧！")

            # 2. 主色调计算
            frame_palettes = []
            for img in images:
                try:
                    palette_colors = get_color_palette(img, num_colors=5)
                except Exception:
                    palette_colors = []
                frame_palettes.append(palette_colors)

            # 3. 调用 Gemini 做逐帧分析
            with st.spinner("🧠 正在分析每一帧的景别、运镜、构图、情绪与标签..."):
                frame_descriptions = analyze_images_concurrently(images, model)

            # 4. 整体总结 & 广告文案
            with st.spinner("📚 正在生成整段视频的剧情大纲与话题标签..."):
                overall = analyze_overall_video(frame_descriptions, model)
            with st.spinner("🎤 正在生成 10 秒广告旁白脚本..."):
                ad_script = generate_ad_script(frame_descriptions, model)

            # 5. Tabs 布局：像网站那样分区展示
            tab_frames, tab_story, tab_json = st.tabs(
                ["🎞 关键帧 & 逐帧分析", "📚 剧情总结 & 广告旁白", "📦 JSON 导出"]
            )

            # --- Tab1：逐帧卡片布局 ---
            with tab_frames:
                st.markdown(
                    f"共抽取 **{len(images)}** 个关键帧。每一帧下方的文字为可复制分镜分析。"
                )
                st.markdown("---")

                for i, (img, desc, palette) in enumerate(
                    zip(images, frame_descriptions, frame_palettes)
                ):
                    with st.container():
                        st.markdown(
                            f"#### 🎬 关键帧 {i + 1}",
                        )
                        c1, c2 = st.columns([1.2, 2])

                        with c1:
                            st.image(
                                img,
                                caption=f"第 {i + 1} 帧画面",
                                width=DISPLAY_IMAGE_WIDTH,
                            )
                            palette_img = make_palette_image(palette)
                            st.image(
                                palette_img,
                                caption="主色调色卡",
                                width=PALETTE_WIDTH,
                            )
                            if palette:
                                hex_list = ", ".join(rgb_to_hex(c) for c in palette)
                                st.caption(f"主色 HEX：{hex_list}")

                        with c2:
                            st.markdown("**分镜分析（可复制）：**")
                            st.code(desc, language="markdown")

                        st.markdown("---")

            # --- Tab2：整体分析 + 广告文案 ---
            with tab_story:
                st.markdown("### 📚 整体剧情与视听风格总结")
                st.code(overall, language="markdown")

                st.markdown("### 🎤 10 秒广告旁白脚本")
                st.code(ad_script, language="markdown")

            # --- Tab3：JSON 导出 ---
            with tab_json:
                st.markdown("### 📦 导出 JSON 分析结果")

                export_frames = []
                for i, (desc, palette) in enumerate(
                    zip(frame_descriptions, frame_palettes)
                ):
                    export_frames.append(
                        {
                            "index": i + 1,
                            "analysis": desc,
                            "palette_rgb": [list(c) for c in (palette or [])],
                            "palette_hex": [rgb_to_hex(c) for c in (palette or [])],
                        }
                    )

                export_data = {
                    "meta": {
                        "model": GEMINI_MODEL_NAME,
                        "frame_count": len(images),
                    },
                    "frames": export_frames,
                    "overall_analysis": overall,
                    "ad_script_10s": ad_script,
                }

                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

                st.download_button(
                    label="⬇️ 下载 JSON 分析文件",
                    data=json_str,
                    file_name="video_analysis.json",
                    mime="application/json",
                )

                with st.expander("🔍 预览部分 JSON 内容"):
                    st.code(json_str[:3000] + ("\n...\n" if len(json_str) > 3000 else ""), language="json")

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
    page_title="AI 自动关键帧分镜 & Midjourney 提示词助手",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    .stMarkdown, .stText {
        color: #e5e7eb;
    }
    .stCode {
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 顶部 Hero
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
        🎬 AI 自动关键帧分镜助手 Pro · Midjourney 提示词版
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        上传一个视频，自动抽取关键帧，生成
        <b>结构化 JSON + Midjourney 提示词 + 分镜解读 + 剧情大纲 + 10 秒广告旁白</b>，
        直接当「AI 导演 + MJ 提示词工程师」使用。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# 抽关键帧（根据时长自动决定数量）
# ========================

def extract_keyframes_dynamic(
    video_path: str,
    min_frames: int = 6,
    max_frames: int = 30,
    base_fps: float = 0.8,
):
    """
    根据视频时长自动抽取关键帧：
    - 估算目标帧数: ideal_n = duration * base_fps
    - 限制在 [min_frames, max_frames]
    - 均匀抽帧
    返回 PIL.Image 列表。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        return [], 0.0

    duration = total_frames / fps  # 秒
    ideal_n = int(duration * base_fps)
    target_n = max(min_frames, ideal_n)
    target_n = min(target_n, max_frames, total_frames)

    if target_n <= 0:
        cap.release()
        return [], duration

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
    return images, duration


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
    兼容不同版本 SDK 的 Gemini 响应解析。
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
# 单帧分析：生成结构化 JSON + MJ 提示词
# ========================

def analyze_single_image(img: Image.Image, model, index: int):
    """
    输出一个结构化 dict：
    {
      "index": index,
      "scene_description_zh": ...,
      "tags_zh": [...],
      "camera": {
        "shot_type_zh": ...,
        "angle_zh": ...,
        "movement_zh": ...,
        "composition_zh": ...
      },
      "color_and_light_zh": ...,
      "mood_zh": ...,
      "midjourney_prompt": ...,
      "midjourney_negative_prompt": ...
    }
    """
    try:
        prompt = f"""
你现在是专业的电影分镜师 + 提示词工程师。
请仔细分析给你的这一帧画面，并输出一个 JSON 对象，用于在 Midjourney 中复现该画面。

必须使用下面这些 key（英文），value 多为中文说明，Midjourney 相关字段为英文：

{{
  "index": 整数，当前帧序号，固定为 {index},
  "scene_description_zh": "一句完整的中文句子，描述画面中是谁在什么场景下做什么",
  "tags_zh": ["#短中文标签1", "#标签2", "..."],
  "camera": {{
    "shot_type_zh": "景别，例如：远景 / 全景 / 中景 / 近景 / 特写",
    "angle_zh": "拍摄角度，例如：俯拍 / 仰拍 / 平视 / 上帝视角",
    "movement_zh": "运镜方式，例如：静止镜头 / 轻微推镜 / 跟拍 等",
    "composition_zh": "构图方式，例如：三分法构图 / 中心构图 / 对称构图 / 前景-主体-背景 等"
  }},
  "color_and_light_zh": "用一两句中文描述画面的色调和光线，例如：整体偏暖，高调柔和逆光，粉色和米白为主色",
  "mood_zh": "用中文概括情绪氛围，例如：亲切、甜美、带货分享氛围",
  "midjourney_prompt": "一行英文 Midjourney v6 提示词，用逗号分隔短语，尽量精确描述人物外观、姿态、场景、道具、光线、色调、构图和氛围，适合 9:16 竖版，结尾加 --ar 9:16 --v 6.0 --style raw",
  "midjourney_negative_prompt": "一行英文负面提示词，例如：text, subtitle, watermark, extra fingers, deformed hands, distorted face, low resolution, blurry, cartoon, anime, painting"
}}

要求：
1. 只输出 JSON，不要任何解释或额外文字。
2. 所有字符串使用双引号，不要使用单引号。
3. JSON 中不能有注释，不能有多余的逗号。
4. midjourney_prompt 必须是英文，适合直接粘贴给 Midjourney v6。
"""
        resp = model.generate_content([prompt, img])
        text = _extract_text_from_response(resp)
        if not text:
            raise ValueError("模型未返回文本")

        # 尝试从文本中截取 JSON 子串
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        json_str = text[start : end + 1]
        info = json.loads(json_str)

        # 确保 index 存在且正确
        info["index"] = index

        # 填充默认结构，避免后续 KeyError
        info.setdefault("scene_description_zh", "")
        info.setdefault("tags_zh", [])
        info.setdefault("camera", {})
        info.setdefault("color_and_light_zh", "")
        info.setdefault("mood_zh", "")
        info.setdefault("midjourney_prompt", "")
        info.setdefault("midjourney_negative_prompt", "")
        cam = info["camera"]
        cam.setdefault("shot_type_zh", "")
        cam.setdefault("angle_zh", "")
        cam.setdefault("movement_zh", "")
        cam.setdefault("composition_zh", "")

        return info

    except Exception as e:
        # 解析失败时返回一个占位结构
        return {
            "index": index,
            "scene_description_zh": f"（AI 分析失败：{e}）",
            "tags_zh": [],
            "camera": {
                "shot_type_zh": "",
                "angle_zh": "",
                "movement_zh": "",
                "composition_zh": "",
            },
            "color_and_light_zh": "",
            "mood_zh": "",
            "midjourney_prompt": "",
            "midjourney_negative_prompt": "",
        }


def analyze_images_concurrently(images, model, max_ai_frames: int):
    """
    并发分析多张图片，加速整体速度。
    只对前 max_ai_frames 帧做 AI 调用，其余帧用占位说明。
    返回：长度等于 images 的列表，每个元素是上面定义的 dict。
    """
    n = len(images)
    if n == 0:
        return []

    use_n = min(max_ai_frames, n)
    results = [None] * n

    status = st.empty()
    status.info(f"⚡ 正在对前 {use_n} 帧进行 AI 分析（共 {n} 帧），其余帧保留截图与色卡。")

    # 先对需要分析的帧并发调用
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(use_n, 6)) as executor:
        future_to_index = {
            executor.submit(analyze_single_image, images[i], model, i + 1): i
            for i in range(use_n)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {
                    "index": i + 1,
                    "scene_description_zh": f"（AI 分析失败：{e}）",
                    "tags_zh": [],
                    "camera": {
                        "shot_type_zh": "",
                        "angle_zh": "",
                        "movement_zh": "",
                        "composition_zh": "",
                    },
                    "color_and_light_zh": "",
                    "mood_zh": "",
                    "midjourney_prompt": "",
                    "midjourney_negative_prompt": "",
                }

    # 对未分析的帧填充占位
    for i in range(use_n, n):
        results[i] = {
            "index": i + 1,
            "scene_description_zh": "（本帧未做 AI 分析，用于节省当前 API 配额，但仍可用于视觉参考和色卡。）",
            "tags_zh": [],
            "camera": {
                "shot_type_zh": "",
                "angle_zh": "",
                "movement_zh": "",
                "composition_zh": "",
            },
            "color_and_light_zh": "",
            "mood_zh": "",
            "midjourney_prompt": "",
            "midjourney_negative_prompt": "",
        }

    status.empty()
    return results


# ========================
# 整体视频层面的总结
# ========================

def analyze_overall_video(frame_infos, model):
    """
    使用已有的帧级信息，生成整段视频的剧情大纲等。
    """
    described = [
        info
        for info in frame_infos
        if info.get("scene_description_zh")
        and "未做 AI 分析" not in info["scene_description_zh"]
        and "AI 分析失败" not in info["scene_description_zh"]
    ]
    if not described:
        return "（暂未获取到有效的帧级分析，无法生成整体剧情大纲。）"

    parts = []
    for info in described:
        idx = info["index"]
        cam = info.get("camera", {})
        tags = info.get("tags_zh", [])
        part = (
            f"第 {idx} 帧：{info.get('scene_description_zh', '')}\n"
            f"景别：{cam.get('shot_type_zh', '')}；角度：{cam.get('angle_zh', '')}；运镜：{cam.get('movement_zh', '')}；构图：{cam.get('composition_zh', '')}\n"
            f"色彩与光影：{info.get('color_and_light_zh', '')}\n"
            f"情绪氛围：{info.get('mood_zh', '')}\n"
            f"标签：{'、'.join(tags)}"
        )
        parts.append(part)

    joined = "\n\n".join(parts)

    prompt = f"""
你现在是资深视频导演 + 剪辑师 + 短视频运营专家 + 内容合规审核员。
下面是从一段视频中抽取的若干关键帧的详细说明，请你基于这些说明，对整段视频做整体分析。

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

def generate_ad_script(frame_infos, model):
    """
    基于若干关键帧的分析，生成一条 10 秒左右的中文广告旁白脚本。
    """
    described = [
        info
        for info in frame_infos
        if info.get("scene_description_zh")
        and "未做 AI 分析" not in info["scene_description_zh"]
        and "AI 分析失败" not in info["scene_description_zh"]
    ]
    if not described:
        return "（暂未获取到有效的帧级分析，无法生成广告旁白脚本。）"

    parts = []
    for info in described:
        idx = info["index"]
        tags = info.get("tags_zh", [])
        parts.append(
            f"第 {idx} 帧：{info.get('scene_description_zh', '')}；标签：{'、'.join(tags)}"
        )
    joined = "\n".join(parts)

    prompt = f"""
你是一名资深广告导演 + 文案。
我有一个由若干画面组成的竖版短视频，时长大约 8-12 秒。
下面是每个画面的简要说明，请你基于这些信息，写一条适合配合这些画面播放的中文广告旁白脚本。

=== 关键帧概览 ===
{joined}
=== 关键帧概览结束 ===

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
# 侧边栏：API Key & 参数设置
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 Gemini API Key")
    api_key = st.text_input(
        "输入 Google API Key",
        type="password",
        help="粘贴你的 Gemini API Key（通常以 AIza 开头）",
    )

    st.markdown("---")
    max_ai_frames = st.slider(
        "本次最多做 AI 分析的帧数（消耗配额）",
        min_value=4,
        max_value=20,
        value=10,
        step=1,
    )
    st.caption("建议：10 秒视频 6~10 帧即可；超出部分仍会显示截图和色卡，但不调 AI。")

    if not api_key:
        st.warning("🔴 还没有 Key，先去 https://ai.google.dev/ 申请一个")
    else:
        st.success("🟢 Key 已就绪")


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
        images, duration = extract_keyframes_dynamic(tmp_path)

        # 删除临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if not images:
            st.error("❌ 无法从视频中读取帧，请检查视频文件是否损坏或格式异常。")
        else:
            st.success(
                f"✅ 已成功抽取 {len(images)} 个关键帧（视频约 {duration:.1f} 秒，当前最多对 {max_ai_frames} 帧做 AI 分析）。"
            )

            # 记录每帧的主色调
            frame_palettes = []
            for img in images:
                try:
                    palette_colors = get_color_palette(img, num_colors=5)
                except Exception:
                    palette_colors = []
                frame_palettes.append(palette_colors)

            # 3. 调用 Gemini 做逐帧分析（结构化 JSON + MJ 提示词）
            with st.spinner("🧠 正在为关键帧生成结构化分析 + Midjourney 提示词..."):
                frame_infos = analyze_images_concurrently(
                    images, model, max_ai_frames=max_ai_frames
                )

            # 4. 整体总结 & 广告文案
            with st.spinner("📚 正在生成整段视频的剧情大纲与话题标签..."):
                overall = analyze_overall_video(frame_infos, model)
            with st.spinner("🎤 正在生成 10 秒广告旁白脚本..."):
                ad_script = generate_ad_script(frame_infos, model)

            # 5. Tabs 布局：像网站那样分区展示
            tab_frames, tab_story, tab_json = st.tabs(
                ["🎞 关键帧 & MJ 提示词", "📚 剧情总结 & 广告旁白", "📦 JSON 导出"]
            )

            # --- Tab1：逐帧卡片布局 ---
            with tab_frames:
                st.markdown(
                    f"共抽取 **{len(images)}** 个关键帧，其中前 **{min(len(images), max_ai_frames)}** 帧做了 AI 分析和 Midjourney 提示词生成。"
                )
                st.markdown("---")

                for i, (img, info, palette) in enumerate(
                    zip(images, frame_infos, frame_palettes)
                ):
                    with st.container():
                        st.markdown(f"### 📘 关键帧 {i + 1}")

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
                            # 用 JSON 拼出一个“8 行分镜分析”
                            cam = info.get("camera", {})
                            tags = info.get("tags_zh", [])
                            analysis_text = "\n".join(
                                [
                                    f"【景别】{cam.get('shot_type_zh', '')}",
                                    f"【运镜】{cam.get('movement_zh', '')}",
                                    f"【拍摄角度】{cam.get('angle_zh', '')}",
                                    f"【构图】{cam.get('composition_zh', '')}",
                                    f"【色彩与光影】{info.get('color_and_light_zh', '')}",
                                    f"【画面内容】{info.get('scene_description_zh', '')}",
                                    f"【情绪氛围】{info.get('mood_zh', '')}",
                                    f"【关键词标签】{' '.join(tags)}",
                                ]
                            ).strip()

                            st.markdown("**分镜分析（可复制）：**")
                            st.code(
                                analysis_text
                                or "（暂无分镜分析，可能未做 AI 分析）",
                                language="markdown",
                            )

                            st.markdown("**Midjourney 提示词（可复制）：**")
                            st.code(
                                info.get("midjourney_prompt")
                                or "（暂无 Midjourney 提示词，可能未做 AI 分析）",
                                language="markdown",
                            )

                            st.markdown("**Midjourney 负面提示词（可选）：**")
                            st.code(
                                info.get("midjourney_negative_prompt") or "",
                                language="markdown",
                            )

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
                for info, palette in zip(frame_infos, frame_palettes):
                    export_frames.append(
                        {
                            "index": info.get("index"),
                            "scene_description_zh": info.get(
                                "scene_description_zh", ""
                            ),
                            "tags_zh": info.get("tags_zh", []),
                            "camera": info.get("camera", {}),
                            "color_and_light_zh": info.get(
                                "color_and_light_zh", ""
                            ),
                            "mood_zh": info.get("mood_zh", ""),
                            "midjourney_prompt": info.get("midjourney_prompt", ""),
                            "midjourney_negative_prompt": info.get(
                                "midjourney_negative_prompt", ""
                            ),
                            "palette_rgb": [list(c) for c in (palette or [])],
                            "palette_hex": [rgb_to_hex(c) for c in (palette or [])],
                        }
                    )

                export_data = {
                    "meta": {
                        "model": GEMINI_MODEL_NAME,
                        "frame_count": len(images),
                        "max_ai_frames_this_run": max_ai_frames,
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
                    preview = json_str[:3000] + (
                        "\n...\n" if len(json_str) > 3000 else ""
                    )
                    st.code(preview, language="json")

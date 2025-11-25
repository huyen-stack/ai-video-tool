import streamlit as st
import google.generativeai as genai
import tempfile
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures
import json
from datetime import datetime
import yt_dlp  # 抖音/B站/TikTok/YouTube 下载
from typing import Optional, Tuple, List, Dict, Any

# ========================
# 全局配置
# ========================

GEMINI_MODEL_NAME = "gemini-flash-latest"  # 可换成 gemini-2.5-flash-lite 等

# 免费版典型速率：每分钟 10 次 generateContent
FREE_TIER_RPM_LIMIT = 10

DISPLAY_IMAGE_WIDTH = 320
PALETTE_WIDTH = 320
PALETTE_HEIGHT = 26

# 初始化会话状态：API Key + 历史记录
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "analysis_history" not in st.session_state:
    # 每条元素结构：
    # {
    #   "id": "run_1",
    #   "created_at": "...",
    #   "meta": {...},
    #   "data": {... 完整 export_data ...}
    # }
    st.session_state["analysis_history"] = []


# ========================
# 页面 / 全局样式
# ========================

st.set_page_config(
    page_title="AI 自动关键帧分镜 & 视频提示词助手",
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
        🎬 AI 自动关键帧分镜助手 Pro · SORA/VEO 视频提示词 + 时间区间 + 历史记录
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        上传视频或输入抖音/B站/TikTok/YouTube 链接，设置分析时间区间，自动抽取关键帧，生成
        <b>结构化 JSON + Midjourney 提示词 + SORA/VEO 英文视频提示词 + 分镜解读 + 剧情大纲 + 10 秒广告旁白 + 时间轴分镜脚本</b>，
        并在当前会话中保存多条分析记录，方便对比与下载。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# 抽关键帧（支持时间区间）
# ========================

def extract_keyframes_dynamic(
    video_path: str,
    min_frames: int = 6,
    max_frames: int = 30,
    base_fps: float = 0.8,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[List[Image.Image], float, Tuple[float, float]]:
    """
    根据视频时长自动抽取关键帧，仅在 [start_sec, end_sec] 范围内。
    返回：
      images: 抽到的 PIL.Image 列表
      duration: 整条视频总时长（秒）
      used_range: (start_used, end_used) 实际生效的分析时间范围（秒）
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        return [], 0.0, (0.0, 0.0)

    duration = total_frames / fps  # 整条视频时长

    # 规范化时间范围
    if start_sec is None or start_sec < 0:
        start_sec = 0.0
    if end_sec is None or end_sec <= start_sec or end_sec > duration:
        end_sec = duration

    start_frame = int(start_sec * fps)
    end_frame_excl = min(total_frames, int(end_sec * fps))
    segment_frames = end_frame_excl - start_frame

    # 如果区间非法，退回整段
    if segment_frames <= 0:
        start_sec = 0.0
        end_sec = duration
        start_frame = 0
        end_frame_excl = total_frames
        segment_frames = total_frames

    segment_duration = segment_frames / fps

    ideal_n = int(segment_duration * base_fps)
    target_n = max(min_frames, ideal_n)
    target_n = min(target_n, max_frames, segment_frames)

    if target_n <= 0:
        cap.release()
        return [], duration, (start_sec, end_sec)

    step = segment_frames / float(target_n)
    frame_indices = [start_frame + int(i * step) for i in range(target_n)]

    images: List[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))
        else:
            images.append(Image.new("RGB", (200, 200), color="gray"))

    cap.release()
    return images, duration, (start_sec, end_sec)


# ========================
# 从链接下载视频
# ========================

def download_video_from_url(url: str) -> str:
    """使用 yt-dlp 从给定 URL 下载视频到临时文件，返回路径。"""
    if not url:
        raise ValueError("视频链接为空")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    ydl_opts = {
        "format": "mp4/bestvideo+bestaudio/best",
        "outtmpl": tmp_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return tmp_path


# ========================
# 主色调色卡相关
# ========================

def get_color_palette(pil_img: Image.Image, num_colors: int = 5):
    img_small = pil_img.resize((120, 120))
    arr = np.array(img_small)
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
# 单帧分析：结构化 JSON + MJ 提示词 + 视频提示词
# ========================

def analyze_single_image(img: Image.Image, model, index: int) -> Dict[str, Any]:
    """
    对单帧做全面分析：
    - 中文分镜（景别/机位/光线/情绪/标签）
    - 人物服饰/表情/动作/道具
    - 场景细节 / 科技道具 / 动作趋势
    - 物理 & 环境细节（风雨雪、头发衣物反应、受力形变、结构破坏、碎片飞散等）
    - Midjourney 提示词
    - SORA/VEO 用英文视频提示词 video_prompt_en
    """
    try:
        prompt = f"""
你现在是电影导演 + 摄影指导 + 服化道总监 + 提示词工程师。
请仔细分析给你的这一帧画面，并输出一个 JSON 对象，用于：
1）人类导演阅读分镜
2）Midjourney 生成分镜图
3）SORA/VEO 等视频模型生成对应镜头

必须使用下面这些 key（英文），value 大部分为中文说明，英文提示词字段为英文：

{{
  "index": 整数，当前帧序号，固定为 {index},

  "scene_description_zh": (
    "用 1～3 句完整中文，把当前画面描述得尽量具体，必须同时包含："
    "1）主要人物的身份 + 大致年龄 + 性别 + 外观特征（尤其是服装颜色/款式、是否有披风、头饰等）；"
    "2）此刻正在做的动作以及动作方向和速度感（例如：从画面右上高速滑向左下、抬手推开门、低头看手机），"
    "如果画面明显处在一个极限动作过程之中（例如：摩托车刚从山坡飞出准备落在飞机机翼上，人物即将抛弃载具跳到机翼上），"
    "要用一句完整的话把“从哪里飞来/飞向哪里”的运动路径说出来，而不是只写“人物在空中”或“在飞机旁边”；"
    "3）所在的场景类型与空间结构（例如：临海山路、峡谷、城市高楼之间的空中轨道），分清前景/中景/背景里各有什么物体（悬崖、海面、树木、房屋、车流等），以及地形特征（蜿蜒、陡坡、悬空平台等）；"
    "4）镜头相对人物的位置和视角（例如：第一人称视角、紧贴背后的追随视角、肩膀后视角、从侧后方略仰拍、俯视俯冲等），以及镜头大致沿着什么方向运动。"
    "禁止只写“某人站在某处”或“人物在路上移动”这种模糊句子，必须写到让美术或 3D 场景师能按文字搭场景的程度；"
    "忽略画面中的 App UI 图标/文字/界面元素，不要把抖音/快手按钮写进描述。"
  ),

  "tags_zh": ["#短中文标签1", "#标签2", "..."],

  "camera": {{
    "shot_type_zh": "景别，例如：远景 / 全景 / 中景 / 中近景 / 近景 / 特写",
    "shot_type": "英文景别，例如：wide shot, full shot, medium shot, medium close-up, close-up",
    "angle_zh": "拍摄角度，例如：俯拍 / 仰拍 / 平视 / 上帝视角 / 侧拍等",
    "angle": "英文角度描述，例如：eye-level, low angle, high angle, top-down",
    "movement_zh": "运镜方式，例如：静止镜头 / 缓慢推近 / 手持跟拍 / 横移 / 甩镜 等",
    "movement": "英文运镜描述，例如：slow dolly-in, handheld tracking from left to right",
    "composition_zh": "构图方式，例如：三分法构图 / 中心构图 / 对称构图 / 前景-主体-背景 等",
    "composition": "英文构图描述，例如：rule-of-thirds, subject on right third, strong foreground elements"
  }},

  "color_and_light_zh": "用 1～2 句中文描述画面的整体色调和光线（色温/对比/主光源方向及是否有逆光/轮廓光等）",
  "mood_zh": "用中文概括情绪氛围（紧张/温暖/梦幻/冷峻/商业感等）",

  "characters": [
    {{
      "role_zh": "人物身份，例如：女主 / 男主 / 科学家 / 侦探 / 厨师",
      "gender_zh": "性别，例如：女性 / 男性 / 不明显",
      "age_look_zh": "年龄观感，例如：20多岁 / 中年",
      "body_type_zh": "体型，例如：偏瘦 / 健壮",
      "clothing_zh": "服装风格与颜色，例如：白色科技感紧身衣，带蓝色发光纹路",
      "hair_zh": "发型与发色，例如：短发，黑色或银色挑染",
      "expression_zh": "面部表情，例如：专注、愤怒、惊讶、轻松微笑",
      "pose_body_zh": "身体姿态，例如：前倾操作控制台 / 半蹲准备起跳 / 身体后仰正在刹车",
      "props_zh": "人物手上或身上明显可见的道具，例如：手枪、光剑、平底锅、平板电脑、飞行扫帚"
    }}
  ],

  "character_action_detail_zh": (
    "专门描述人物动作的细节，用 1～3 句完整中文，从“头部→上肢→躯干→下肢”的顺序写。"
    "必须说清：1）此刻身体重心位置（例如：前倾、后仰、蹲下、腾空、贴在某物表面）；"
    "2）双手/手指在做什么动作（例如：右手死死抓住机翼边缘，左手撑住摩托车车把，手指吃力绷紧）；"
    "3）双腿/脚的姿态和指向（例如：双腿夹紧油箱，脚尖朝外伸直，右脚刚离开山坡边缘）；"
    "4）与道具或环境的具体接触点（例如：膝盖抵在机翼表面、脚跟踩在栏杆上）。"
    "禁止写成“人物奔跑/跳跃”这种笼统描述，必须写到读者能在脑中看到具体肢体姿态。"
  ),

  "face_expression_detail_zh": (
    "专门描述面部表情与眼神变化，用 1～3 句中文写清："
    "1）眉毛、眼睛、嘴角、下颌等肌肉的状态（紧绷/放松/抽动等）；"
    "2）眼睛的细节：眼色、瞳孔大小、是否充血/泛光、是否有泪光；"
    "3）如果有风、雨、拳头打在脸上等外力，脸部是否出现形变、抖动、回弹。"
  ),

  "cloth_hair_reaction_zh": (
    "用 1～3 句中文写清人物头发和衣服对风/动作/爆炸/惯性的反应。"
    "例如：长发被大风向后吹起、刘海贴在脸上、外套被风鼓起又被压回、下摆在奔跑时有延迟甩动感。"
  ),

  "environment_detail_zh": (
    "用 2～4 句中文，按照 前景 / 中景 / 背景 的层次，尽可能具体地描述场景环境。必须写出："
    "1）空间类型（室内/室外、厨房/街道/仓库/办公室/宇宙飞船舱室等）；"
    "2）前景靠近镜头的物体和质感（例如桌面、栏杆、玻璃、光屏，写清颜色/材质/是否虚化）；"
    "3）中景主体周围的环境结构（墙面、柜子、机器、车辆、人群等）；"
    "4）背景中可识别的建筑/山体/城市天际线/窗外景色；"
    "5）地面和顶部的感觉（例如：水泥地、木地板、带油污的瓷砖地、裸露管线的天花板）。"
    "不要只写“在一个房间里”，必须写到能让美术师按文字搭景的程度。"
    "如画面以天空/云层/海面等为主，也要写出云层形态、海浪走势、远处地貌等。"
  ),

  "weather_force_detail_zh": (
    "如果画面中有风、雨、雪、爆炸冲击波、车辆高速行驶带来的气流等，请用 1～3 句中文具体写出："
    "1）这些环境力来自哪个方向（例如：从左侧斜上方吹来、从人物身后涌来）；"
    "2）它们如何作用在人物和环境上（例如：雨点抽打在脸上和衣服上、风把水面吹起浪花）；"
    "3）是否造成明显晃动、震动或其他反馈。若画面没有明显环境力，可写“无明显风雨或环境冲击”。"
  ),

  "props_and_tech_detail_zh": (
    "用 1～3 句中文，列出画面中最重要的 3～8 个道具/科技元素，并说明它们的外观、位置和状态。"
    "例如：“左前景是一个银色笔记本电脑，屏幕发出冷蓝色光；人物右手边有一台黑色咖啡机，机身有水渍反光；"
    "背景墙上挂着两幅抽象画；天空中有两架无人机从左向右飞过；画面右侧是一块半透明蓝色全息屏幕，悬浮在空中，显示数据图表”。"
  ),

  "physics_reaction_detail_zh": (
    "用 1～3 句中文，专门描述“受力”与“形变→回弹”的过程。"
    "例如：拳头砸在人物脸上导致脸颊被明显挤压、下巴偏移，随后肌肉抖动并缓慢回弹；"
    "或者车辆撞上护栏时车头被压扁、车身抖动、人物身体被安全带拉扯后弹回。"
  ),

  "structure_damage_detail_zh": (
    "用 1～3 句中文描述物体或结构本身的损坏情况，例如：车头金属折叠、挡风玻璃开裂、建筑墙体塌陷、机翼断裂等，"
    "写清“哪一部分”因撞击或爆炸产生了怎样的形变或破损。"
  ),

  "debris_motion_detail_zh": (
    "用 1～3 句中文描述碎片/玻璃渣/零件/石块等飞散的轨迹和状态："
    "例如：挡风玻璃碎片向前方和右上方抛洒，形成扇形弧线，随后落在地面；"
    "建筑碎石向下坠落时扬起灰尘。若画面无明显碎片飞散，可写“无明显碎片飞散”。"
  ),

  "motion_detail_zh": (
    "用 1～3 句中文，从“上一瞬间→当前瞬间→下一瞬间”的顺序，描述这一镜头所属动作片段。"
    "如果能从画面推断出大致动作，请写清：上一瞬间人物大概在做什么，当前瞬间画面定格在什么状态，下一瞬间极有可能发生什么。"
  ),

  "fx_detail_zh": (
    "如果画面中有火花、烟雾、尘土、能量波、魔法特效、碎裂粒子等，请用 1～2 句中文描述它们的位置、颜色和运动方向。"
  ),

  "lighting_color_detail_zh": (
    "在 color_and_light_zh 的基础上，进一步用 1～2 句中文精细描述光源数量、方向、色温差异、"
    "是否有闪电/爆光/频闪等。"
  ),

  "audio_cue_detail_zh": (
    "用 1～3 句中文描述这一帧所在片段的声音设计："
    "包括环境声（风声、雨声、车辆噪音、机器运转）、人物台词、特效声（打击声、爆炸声）以及 BGM 的情绪和节奏点。"
  ),

  "edit_rhythm_detail_zh": (
    "用 1～2 句中文描述剪辑节奏和时间处理，例如：正常速度、轻微慢动作、大幅慢动作、突然加速、甩镜转场、闪白转场等。"
  ),

  "midjourney_prompt": "一行英文 Midjourney v6 提示词，适合生成这一帧的静态分镜图",
  "midjourney_negative_prompt": "一行英文负面提示词，例如：text, subtitle, watermark, extra fingers, deformed hands, distorted face, low resolution, blurry, cartoon, anime, painting",

  "video_prompt_en": (
    "一段英文视频提示词，适合给 SORA/VEO 使用。用 3-5 句描述：人物外观、当前动作、"
    "运镜方式（要说明是追随视角/第一人称/侧向跟拍等以及镜头移动方向）、"
    "环境地形（例如 winding coastal mountain road, steep cliff, ocean on the left），以及光线与氛围。"
    "最后一句写清这是一段几秒钟的镜头，例如：'4 second shot, vertical 9:16, 24fps, cinematic, highly detailed.'"
  )
}}

要求：
1. 只输出一个 JSON 对象，不要任何解释或额外文字。
2. 所有字符串必须使用双引号，不要使用单引号。
3. JSON 中不能有注释，不能有多余的逗号。
"""
        resp = model.generate_content([prompt, img])
        text = _extract_text_from_response(resp)
        if not text:
            raise ValueError("模型未返回文本")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        json_str = text[start : end + 1]
        info = json.loads(json_str)

        info["index"] = index
        info.setdefault("scene_description_zh", "")
        info.setdefault("tags_zh", [])
        info.setdefault("camera", {})
        info.setdefault("color_and_light_zh", "")
        info.setdefault("mood_zh", "")
        info.setdefault("characters", [])
        info.setdefault("character_action_detail_zh", "")
        info.setdefault("face_expression_detail_zh", "")
        info.setdefault("cloth_hair_reaction_zh", "")
        info.setdefault("environment_detail_zh", "")
        info.setdefault("weather_force_detail_zh", "")
        info.setdefault("props_and_tech_detail_zh", "")
        info.setdefault("physics_reaction_detail_zh", "")
        info.setdefault("structure_damage_detail_zh", "")
        info.setdefault("debris_motion_detail_zh", "")
        info.setdefault("motion_detail_zh", "")
        info.setdefault("fx_detail_zh", "")
        info.setdefault("lighting_color_detail_zh", "")
        info.setdefault("audio_cue_detail_zh", "")
        info.setdefault("edit_rhythm_detail_zh", "")
        info.setdefault("midjourney_prompt", "")
        info.setdefault("midjourney_negative_prompt", "")
        info.setdefault("video_prompt_en", "")

        cam = info["camera"]
        cam.setdefault("shot_type_zh", "")
        cam.setdefault("shot_type", "")
        cam.setdefault("angle_zh", "")
        cam.setdefault("angle", "")
        cam.setdefault("movement_zh", "")
        cam.setdefault("movement", "")
        cam.setdefault("composition_zh", "")
        cam.setdefault("composition", "")

        return info

    except Exception as e:
        return {
            "index": index,
            "scene_description_zh": f"（AI 分析失败：{e}）",
            "tags_zh": [],
            "camera": {
                "shot_type_zh": "",
                "shot_type": "",
                "angle_zh": "",
                "angle": "",
                "movement_zh": "",
                "movement": "",
                "composition_zh": "",
                "composition": "",
            },
            "color_and_light_zh": "",
            "mood_zh": "",
            "characters": [],
            "character_action_detail_zh": "",
            "face_expression_detail_zh": "",
            "cloth_hair_reaction_zh": "",
            "environment_detail_zh": "",
            "weather_force_detail_zh": "",
            "props_and_tech_detail_zh": "",
            "physics_reaction_detail_zh": "",
            "structure_damage_detail_zh": "",
            "debris_motion_detail_zh": "",
            "motion_detail_zh": "",
            "fx_detail_zh": "",
            "lighting_color_detail_zh": "",
            "audio_cue_detail_zh": "",
            "edit_rhythm_detail_zh": "",
            "midjourney_prompt": "",
            "midjourney_negative_prompt": "",
            "video_prompt_en": "",
        }


def analyze_images_concurrently(
    images: List[Image.Image], model, max_ai_frames: int
) -> List[Dict[str, Any]]:
    """
    并发分析多张图片。
    只对前 max_ai_frames 帧做 AI 调用，其余帧用占位说明。
    """
    n = len(images)
    if n == 0:
        return []

    use_n = min(max_ai_frames, n)
    results: List[Dict[str, Any]] = [None] * n  # type: ignore

    status = st.empty()
    status.info(f"⚡ 正在对前 {use_n} 帧进行 AI 分析（共 {n} 帧），其余帧保留截图与色卡。")

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
                        "shot_type": "",
                        "angle_zh": "",
                        "angle": "",
                        "movement_zh": "",
                        "movement": "",
                        "composition_zh": "",
                        "composition": "",
                    },
                    "color_and_light_zh": "",
                    "mood_zh": "",
                    "characters": [],
                    "character_action_detail_zh": "",
                    "face_expression_detail_zh": "",
                    "cloth_hair_reaction_zh": "",
                    "environment_detail_zh": "",
                    "weather_force_detail_zh": "",
                    "props_and_tech_detail_zh": "",
                    "physics_reaction_detail_zh": "",
                    "structure_damage_detail_zh": "",
                    "debris_motion_detail_zh": "",
                    "motion_detail_zh": "",
                    "fx_detail_zh": "",
                    "lighting_color_detail_zh": "",
                    "audio_cue_detail_zh": "",
                    "edit_rhythm_detail_zh": "",
                    "midjourney_prompt": "",
                    "midjourney_negative_prompt": "",
                    "video_prompt_en": "",
                }

    for i in range(use_n, n):
        results[i] = {
            "index": i + 1,
            "scene_description_zh": "（本帧未做 AI 分析，用于节省当前 API 配额，但仍可用于视觉参考和色卡。）",
            "tags_zh": [],
            "camera": {
                "shot_type_zh": "",
                "shot_type": "",
                "angle_zh": "",
                "angle": "",
                "movement_zh": "",
                "movement": "",
                "composition_zh": "",
                "composition": "",
            },
            "color_and_light_zh": "",
            "mood_zh": "",
            "characters": [],
            "character_action_detail_zh": "",
            "face_expression_detail_zh": "",
            "cloth_hair_reaction_zh": "",
            "environment_detail_zh": "",
            "weather_force_detail_zh": "",
            "props_and_tech_detail_zh": "",
            "physics_reaction_detail_zh": "",
            "structure_damage_detail_zh": "",
            "debris_motion_detail_zh": "",
            "motion_detail_zh": "",
            "fx_detail_zh": "",
            "lighting_color_detail_zh": "",
            "audio_cue_detail_zh": "",
            "edit_rhythm_detail_zh": "",
            "midjourney_prompt": "",
            "midjourney_negative_prompt": "",
            "video_prompt_en": "",
        }

    status.empty()
    return results


# ========================
# 整体视频层面的总结
# ========================

def analyze_overall_video(frame_infos: List[Dict[str, Any]], model) -> str:
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

=== 帧级说明开始 ===
{joined}
=== 帧级说明结束 ===

请严格按下面结构输出中文分析：

【剧情大纲】
用 2-4 句概括这段视频的大致内容/人物关系/发生场景。

【整体视听风格】
从节奏快慢、镜头感、色彩气质（暖/冷/日常/梦幻）、情绪氛围等角度总结整体风格。

【适合的话题标签】
用 #标签 形式给出 5-10 个，适合抖音/小红书/视频号等平台。

【商业与合规风险】
从“血腥/暴力/色情/政治/品牌商标”等维度，简单评估：
整体风险级别：低 / 中 / 高
并用 2-3 句话说明需要注意的点。

请直接输出以上 4 个小节，不要添加额外说明。
"""
    try:
        resp = model.generate_content(prompt)
        return _extract_text_from_response(resp)
    except Exception as e:
        msg = str(e)
        if "quota" in msg or "You exceeded your current quota" in msg:
            return "整体分析失败：当前 Gemini 免费额度的每分钟调用次数已用完，请稍等几十秒或减少本次分析帧数后重试。"
        return f"整体分析失败：{msg}"


# ========================
# 10 秒广告旁白脚本生成
# ========================

def generate_ad_script(frame_infos: List[Dict[str, Any]], model) -> str:
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
2. 风格与画面调性匹配。
3. 用自然口语化中文，不要出现“画面中”“镜头里”字眼。

请严格按照下面格式输出：

【10秒广告旁白脚本】
（在这里写完整的一段旁白）

不要输出其他任何内容。
"""
    try:
        resp = model.generate_content(prompt)
        return _extract_text_from_response(resp)
    except Exception as e:
        msg = str(e)
        if "quota" in msg or "You exceeded your current quota" in msg:
            return "广告文案生成失败：当前 Gemini 免费额度的每分钟调用次数已用完，请稍等几十秒或减少本次分析帧数后重试。"
        return f"广告文案生成失败：{msg}"


# ========================
# 时间轴分镜脚本生成（纯拼接版）
# ========================

def generate_timeline_shotlist(
    frame_infos: List[Dict[str, Any]],
    used_range: Tuple[float, float],
) -> str:
    """
    不再调用 AI，总结 = 把每帧分析到的所有内容结构化搬到时间轴里。

    时间轴分段规则：
    - 从 0 开始计时，以分析区间长度 / 帧数 均分每段时长
    - 第 i 帧对应时间段 [t_i, t_{i+1}]，尽量保证段落连续、不重叠
    """
    n = len(frame_infos)
    if n == 0:
        return "（暂无关键帧，无法生成时间轴分镜脚本。）"

    start_used, end_used = used_range
    total_len = max(0.1, end_used - start_used)
    seg = total_len / n

    lines: List[str] = []

    for i, info in enumerate(frame_infos):
        t0 = i * seg
        t1 = (i + 1) * seg
        if i == n - 1:
            t1 = total_len

        shot_id = f"S{i+1:02d}"

        cam = info.get("camera", {}) or {}
        tags = info.get("tags_zh", []) or []

        scene = (info.get("scene_description_zh") or "").strip()
        char_act = (info.get("character_action_detail_zh") or "").strip()
        env = (info.get("environment_detail_zh") or "").strip()
        props = (info.get("props_and_tech_detail_zh") or "").strip()
        motion = (info.get("motion_detail_zh") or "").strip()
        mood = (info.get("mood_zh") or "").strip()

        shot_type = cam.get("shot_type_zh", "")
        angle = cam.get("angle_zh", "")
        movement = cam.get("movement_zh", "")
        composition = cam.get("composition_zh", "")

        face = (info.get("face_expression_detail_zh") or "").strip()
        cloth_hair = (info.get("cloth_hair_reaction_zh") or "").strip()
        weather = (info.get("weather_force_detail_zh") or "").strip()
        physics = (info.get("physics_reaction_detail_zh") or "").strip()
        structure_damage = (info.get("structure_damage_detail_zh") or "").strip()
        debris_motion = (info.get("debris_motion_detail_zh") or "").strip()
        fx = (info.get("fx_detail_zh") or "").strip()
        lighting = (info.get("lighting_color_detail_zh") or "").strip()
        audio = (info.get("audio_cue_detail_zh") or "").strip()
        edit = (info.get("edit_rhythm_detail_zh") or "").strip()

        block_lines: List[str] = []
        block_lines.append(f"【{shot_id} | {t0:.1f}-{t1:.1f} 秒】")

        if scene:
            block_lines.append(f"画面内容：{scene}")
        if char_act:
            block_lines.append(f"人物动作：{char_act}")
        if face:
            block_lines.append(f"面部与眼神：{face}")
        if cloth_hair:
            block_lines.append(f"服装与头发：{cloth_hair}")

        if env:
            block_lines.append(f"场景与空间：{env}")
        if weather:
            block_lines.append(f"天气与环境力：{weather}")

        if props:
            block_lines.append(f"道具与科技：{props}")
        if structure_damage:
            block_lines.append(f"结构损坏：{structure_damage}")
        if debris_motion:
            block_lines.append(f"碎片与飞散轨迹：{debris_motion}")
        if physics:
            block_lines.append(f"受力与物理反馈：{physics}")

        if fx:
            block_lines.append(f"特效与粒子：{fx}")
        if lighting:
            block_lines.append(f"光线与色彩：{lighting}")

        cam_desc_parts = []
        if shot_type:
            cam_desc_parts.append(f"景别：{shot_type}")
        if angle:
            cam_desc_parts.append(f"角度：{angle}")
        if movement:
            cam_desc_parts.append(f"运镜：{movement}")
        if composition:
            cam_desc_parts.append(f"构图：{composition}")
        if cam_desc_parts:
            block_lines.append("机位与运动：" + "；".join(cam_desc_parts))

        if mood:
            block_lines.append(f"情绪氛围：{mood}")
        if motion:
            block_lines.append(f"动作趋势：{motion}")

        if audio:
            block_lines.append(f"声音与节奏：{audio}")
        if edit:
            block_lines.append(f"剪辑与节奏：{edit}")

        if tags:
            block_lines.append("标签：" + " ".join(tags))

        lines.append("\n".join(block_lines))

    return "\n\n".join(lines)


# ========================
# 侧边栏：API Key & 参数设置
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 Gemini API Key")
    api_key = st.text_input(
        "输入 Google API Key",
        type="password",
        value=st.session_state["api_key"],
        help="粘贴你的 Gemini API Key（通常以 AIza 开头）",
    )
    st.session_state["api_key"] = api_key

    st.markdown("---")
    max_ai_frames = st.slider(
        "本次最多做 AI 分析的帧数（消耗配额）",
        min_value=4,
        max_value=20,
        value=10,
        step=1,
    )
    st.caption("建议：10 秒视频 6~10 帧即可；超出部分仍会显示截图和色卡，但不调 AI。")

    st.markdown("---")
    st.markdown("⏱ 分析时间范围（单位：秒）")
    start_sec = st.number_input(
        "从第几秒开始（含）", min_value=0.0, value=0.0, step=0.5,
        help="精确到 0.5 秒；默认 0 表示从头开始"
    )
    end_sec = st.number_input(
        "到第几秒结束（0 或 ≤开始秒 表示直到结尾）",
        min_value=0.0, value=0.0, step=0.5,
        help="例如：只分析 3~8 秒，就填 3 和 8；填 0 或不大于开始秒则分析到结尾"
    )

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
# 主流程：上传/链接 选择 + 抽帧 + 分析 + 布局展示
# ========================

source_mode = st.radio(
    "📥 选择视频来源",
    ["上传本地文件", "输入网络视频链接（抖音 / B站 / TikTok / YouTube）"],
    index=0,
)

video_url: Optional[str] = None
uploaded_file = None

if source_mode == "上传本地文件":
    uploaded_file = st.file_uploader(
        "📂 上传视频文件（建议 < 50MB）",
        type=["mp4", "mov", "m4v", "avi", "mpeg"],
    )
else:
    video_url = st.text_input(
        "🔗 输入视频链接",
        placeholder="例如：https://v.douyin.com/xxxxxx 或 https://www.douyin.com/video/xxxxxxxxx",
    )

if st.button("🚀 一键解析整条视频"):
    if not api_key or model is None:
        st.error("请先在左侧输入有效的 Google API Key。")
    else:
        tmp_path: Optional[str] = None
        source_label = ""
        source_type = ""

        try:
            # 1. 准备视频路径
            if source_mode == "上传本地文件":
                source_type = "upload"
                if not uploaded_file:
                    st.error("请先上传一个视频文件。")
                    st.stop()
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                source_label = uploaded_file.name
            else:
                source_type = "url"
                if not video_url:
                    st.error("请输入一个有效的视频链接。")
                    st.stop()
                st.info("🌐 正在从网络下载视频，请稍候...")
                tmp_path = download_video_from_url(video_url)
                source_label = video_url

            if not tmp_path:
                st.error("视频路径异常，请重试。")
            else:
                # 2. 抽帧（带时间区间）
                st.info("⏳ 正在根据指定时间区间自动抽取关键帧...")
                images, duration, used_range = extract_keyframes_dynamic(
                    tmp_path,
                    start_sec=start_sec,
                    end_sec=end_sec if end_sec > 0 else None,
                )
                start_used, end_used = used_range

                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

                if not images:
                    st.error("❌ 无法从视频中读取帧，请检查视频是否损坏或格式异常。")
                    st.stop()

                st.success(
                    f"✅ 已成功抽取 {len(images)} 个关键帧（视频总长约 {duration:.1f} 秒，"
                    f"本次分析区间：{start_used:.1f}–{end_used:.1f} 秒）。"
                )

                # 3. 主色调
                frame_palettes: List[List[Tuple[int, int, int]]] = []
                for img in images:
                    try:
                        palette_colors = get_color_palette(img, num_colors=5)
                    except Exception:
                        palette_colors = []
                    frame_palettes.append(palette_colors)

                # ⭐ 控制本次 AI 调用总数不超过免费 10 次
                overhead_calls = 2  # 整体 + 广告文案
                max_ai_frames_safe = max(
                    1,
                    min(max_ai_frames, FREE_TIER_RPM_LIMIT - overhead_calls),
                )
                if max_ai_frames_safe < max_ai_frames:
                    st.info(
                        f"为避免触发免费额度限制，本次只对 **前 {max_ai_frames_safe} 帧** 做 AI 分析 "
                        f"（侧边栏设置为 {max_ai_frames} 帧）。"
                    )

                # 4. 帧级分析
                with st.spinner("🧠 正在为关键帧生成结构化分析 + MJ 提示词 + 视频提示词..."):
                    frame_infos = analyze_images_concurrently(
                        images, model, max_ai_frames=max_ai_frames_safe
                    )

                # 5. 整体分析 + 广告文案 + 时间轴分镜（时间轴为纯拼接）
                with st.spinner("📚 正在生成整段视频的剧情大纲与话题标签..."):
                    overall = analyze_overall_video(frame_infos, model)
                with st.spinner("🎤 正在生成 10 秒广告旁白脚本..."):
                    ad_script = generate_ad_script(frame_infos, model)
                with st.spinner("🎬 正在生成时间轴分镜脚本（纯拼接版）..."):
                    timeline_shotlist = generate_timeline_shotlist(
                        frame_infos, used_range=used_range
                    )

                # 6. 组装 export_data + 写入历史记录
                export_frames = []
                for info, palette in zip(frame_infos, frame_palettes):
                    export_frames.append(
                        {
                            "index": info.get("index"),
                            "scene_description_zh": info.get("scene_description_zh", ""),
                            "tags_zh": info.get("tags_zh", []),
                            "camera": info.get("camera", {}),
                            "color_and_light_zh": info.get("color_and_light_zh", ""),
                            "mood_zh": info.get("mood_zh", ""),
                            "characters": info.get("characters", []),
                            "character_action_detail_zh": info.get("character_action_detail_zh", ""),
                            "face_expression_detail_zh": info.get("face_expression_detail_zh", ""),
                            "cloth_hair_reaction_zh": info.get("cloth_hair_reaction_zh", ""),
                            "environment_detail_zh": info.get("environment_detail_zh", ""),
                            "weather_force_detail_zh": info.get("weather_force_detail_zh", ""),
                            "props_and_tech_detail_zh": info.get("props_and_tech_detail_zh", ""),
                            "physics_reaction_detail_zh": info.get("physics_reaction_detail_zh", ""),
                            "structure_damage_detail_zh": info.get("structure_damage_detail_zh", ""),
                            "debris_motion_detail_zh": info.get("debris_motion_detail_zh", ""),
                            "motion_detail_zh": info.get("motion_detail_zh", ""),
                            "fx_detail_zh": info.get("fx_detail_zh", ""),
                            "lighting_color_detail_zh": info.get("lighting_color_detail_zh", ""),
                            "audio_cue_detail_zh": info.get("audio_cue_detail_zh", ""),
                            "edit_rhythm_detail_zh": info.get("edit_rhythm_detail_zh", ""),
                            "midjourney_prompt": info.get("midjourney_prompt", ""),
                            "midjourney_negative_prompt": info.get("midjourney_negative_prompt", ""),
                            "video_prompt_en": info.get("video_prompt_en", ""),
                            "palette_rgb": [list(c) for c in (palette or [])],
                            "palette_hex": [rgb_to_hex(c) for c in (palette or [])],
                        }
                    )

                export_data = {
                    "meta": {
                        "model": GEMINI_MODEL_NAME,
                        "frame_count": len(images),
                        "max_ai_frames_this_run": max_ai_frames_safe,
                        "duration_sec_est": duration,
                        "start_sec_used": start_used,
                        "end_sec_used": end_used,
                        "source_type": source_type,
                        "source_label": source_label,
                    },
                    "frames": export_frames,
                    "overall_analysis": overall,
                    "ad_script_10s": ad_script,
                    "timeline_shotlist_zh": timeline_shotlist,
                }

                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

                history = st.session_state["analysis_history"]
                run_id = f"run_{len(history) + 1}"
                history.append(
                    {
                        "id": run_id,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "meta": export_data["meta"],
                        "data": export_data,
                    }
                )
                st.session_state["analysis_history"] = history

                # 7. Tabs 展示
                tab_frames, tab_story, tab_json, tab_history = st.tabs(
                    [
                        "🎞 关键帧 & 提示词",
                        "📚 剧情总结 & 广告旁白 & 时间轴分镜",
                        "📦 JSON 导出（本次）",
                        "🕘 历史记录（本会话）",
                    ]
                )

                # --- Tab1：逐帧卡片 ---
                with tab_frames:
                    st.markdown(
                        f"共抽取 **{len(images)}** 个关键帧，其中前 **{min(len(images), max_ai_frames_safe)}** 帧做了 AI 分析。"
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
                                    hex_list = ", ".join(
                                        rgb_to_hex(c) for c in palette
                                    )
                                    st.caption(f"主色 HEX：{hex_list}")

                            with c2:
                                cam = info.get("camera", {})
                                tags = info.get("tags_zh", [])
                                analysis_lines = [
                                    f"【景别】{cam.get('shot_type_zh', '')}",
                                    f"【运镜】{cam.get('movement_zh', '')}",
                                    f"【拍摄角度】{cam.get('angle_zh', '')}",
                                    f"【构图】{cam.get('composition_zh', '')}",
                                    f"【色彩与光影】{info.get('color_and_light_zh', '')}",
                                    f"【画面内容】{info.get('scene_description_zh', '')}",
                                    f"【情绪氛围】{info.get('mood_zh', '')}",
                                    f"【关键词标签】{' '.join(tags)}",
                                ]
                                analysis_text = "\n".join(analysis_lines).strip()

                                st.markdown("**分镜分析（可复制）：**")
                                st.code(
                                    analysis_text
                                    or "（暂无分镜分析，可能未做 AI 分析）",
                                    language="markdown",
                                )

                                st.markdown("**人物动作细节（可复制）：**")
                                st.code(
                                    info.get("character_action_detail_zh")
                                    or "（暂无动作细节，可能未做 AI 分析）",
                                    language="markdown",
                                )

                                st.markdown("**场景细节（可复制）：**")
                                scene_detail = info.get("environment_detail_zh", "")
                                props_detail = info.get("props_and_tech_detail_zh", "")
                                scene_text = (scene_detail + "\n\n道具与科技元素：" + props_detail).strip()
                                st.code(
                                    scene_text or "（暂无场景细节，可能未做 AI 分析）",
                                    language="markdown",
                                )

                                # 高级物理/环境细节组合展示
                                advanced_detail = []
                                if info.get("face_expression_detail_zh"):
                                    advanced_detail.append("【面部与眼神】" + info["face_expression_detail_zh"])
                                if info.get("cloth_hair_reaction_zh"):
                                    advanced_detail.append("【服装与头发】" + info["cloth_hair_reaction_zh"])
                                if info.get("weather_force_detail_zh"):
                                    advanced_detail.append("【天气与环境力】" + info["weather_force_detail_zh"])
                                if info.get("physics_reaction_detail_zh"):
                                    advanced_detail.append("【受力与物理反馈】" + info["physics_reaction_detail_zh"])
                                if info.get("structure_damage_detail_zh"):
                                    advanced_detail.append("【结构损坏】" + info["structure_damage_detail_zh"])
                                if info.get("debris_motion_detail_zh"):
                                    advanced_detail.append("【碎片飞散】" + info["debris_motion_detail_zh"])
                                if info.get("fx_detail_zh"):
                                    advanced_detail.append("【特效与粒子】" + info["fx_detail_zh"])
                                if info.get("lighting_color_detail_zh"):
                                    advanced_detail.append("【光线细节】" + info["lighting_color_detail_zh"])
                                if info.get("audio_cue_detail_zh"):
                                    advanced_detail.append("【声音与节奏】" + info["audio_cue_detail_zh"])
                                if info.get("edit_rhythm_detail_zh"):
                                    advanced_detail.append("【剪辑节奏】" + info["edit_rhythm_detail_zh"])

                                if advanced_detail:
                                    st.markdown("**高级物理 / 环境细节（可复制）：**")
                                    st.code(
                                        "\n".join(advanced_detail),
                                        language="markdown",
                                    )

                                st.markdown("**SORA / VEO 视频提示词（英文，可复制）：**")
                                st.code(
                                    info.get("video_prompt_en") or "（暂无视频提示词）",
                                    language="markdown",
                                )

                                st.markdown("**Midjourney 静帧提示词（可选）：**")
                                st.code(
                                    info.get("midjourney_prompt")
                                    or "（暂无 Midjourney 提示词）",
                                    language="markdown",
                                )

                            st.markdown("---")

                # --- Tab2：整体分析 + 广告文案 + 时间轴分镜 ---
                with tab_story:
                    st.markdown("### 📚 整体剧情与视听风格总结")
                    st.code(overall, language="markdown")

                    st.markdown("### 🎤 10 秒广告旁白脚本")
                    st.code(ad_script, language="markdown")

                    st.markdown("### 🎬 时间轴分镜脚本（可复制）")
                    st.code(timeline_shotlist, language="markdown")

                # --- Tab3：本次 JSON 导出 ---
                with tab_json:
                    st.markdown("### 📦 下载本次分析的 JSON 文件")
                    st.download_button(
                        label="⬇️ 下载本次 video_analysis.json",
                        data=json_str,
                        file_name="video_analysis.json",
                        mime="application/json",
                    )

                    with st.expander("🔍 预览部分 JSON 内容"):
                        preview = json_str[:3000] + (
                            "\n...\n" if len(json_str) > 3000 else ""
                        )
                        st.code(preview, language="json")

                # --- Tab4：历史记录（当前会话） ---
                with tab_history:
                    st.markdown("### 🕘 当前会话历史记录（刷新页面会清空）")

                    history = st.session_state.get("analysis_history", [])
                    if not history:
                        st.info("当前会话还没有任何历史记录。")
                    else:
                        options = [
                            f"{len(history) - i}. {h['created_at']} | {h['meta'].get('source_label','')} | "
                            f"{h['meta'].get('frame_count',0)} 帧 | 区间 {h['meta'].get('start_sec_used',0):.1f}-{h['meta'].get('end_sec_used',0):.1f}s"
                            for i, h in enumerate(reversed(history))
                        ]
                        idx_display = st.selectbox(
                            "选择一条历史记录查看",
                            options=list(range(len(history))),
                            format_func=lambda i: options[i],
                        )
                        real_index = len(history) - 1 - idx_display
                        selected = history[real_index]

                        st.markdown(
                            f"**ID：** `{selected['id']}`  \n"
                            f"**时间：** {selected['created_at']}  \n"
                            f"**来源类型：** {selected['meta'].get('source_type','')}  \n"
                            f"**来源标识：** {selected['meta'].get('source_label','')}  \n"
                            f"**分析区间：** {selected['meta'].get('start_sec_used',0):.1f}–{selected['meta'].get('end_sec_used',0):.1f} 秒  \n"
                            f"**帧数：** {selected['meta'].get('frame_count',0)}  \n"
                            f"**模型：** {selected['meta'].get('model','')}"
                        )

                        hist_json = json.dumps(
                            selected["data"], ensure_ascii=False, indent=2
                        )
                        st.download_button(
                            label="⬇️ 下载该历史记录 JSON",
                            data=hist_json,
                            file_name=f"video_analysis_{selected['id']}.json",
                            mime="application/json",
                        )

                        frames = selected["data"].get("frames", [])
                        if frames:
                            st.markdown("#### 部分帧预览（中文场景 + 英文视频提示词）")
                            for f in frames[:3]:
                                st.markdown(f"**第 {f.get('index')} 帧：**")
                                st.write(f.get("scene_description_zh", ""))
                                vp = f.get("video_prompt_en", "")
                                if vp:
                                    st.code(vp, language="markdown")
                                st.markdown("---")

        except Exception as e:
            st.error(f"下载或解析视频时发生错误：{e}")

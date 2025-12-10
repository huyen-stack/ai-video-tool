import os
import io
import json
import time
import base64
import threading
import tempfile
import concurrent.futures
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yt_dlp

# =========================================================
# Z.ai / 智谱（Zhipu）HTTP API 基础配置
# =========================================================
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/chat/completions"  # 官方示例接口 :contentReference[oaicite:2]{index=2}

DEFAULT_TEXT_MODEL = "glm-4.5"     # 纯文本（你也可在侧边栏改）
DEFAULT_VISION_MODEL = "glm-4.6v"  # 视觉模型，支持 image_url :contentReference[oaicite:3]{index=3}

# =========================================================
# 简易限流器：避免免费额度 / RPM 触发
# =========================================================
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.lock = threading.Lock()
        self.calls: List[float] = []

    def acquire(self):
        with self.lock:
            now = time.time()
            window = 60.0
            self.calls = [t for t in self.calls if now - t < window]

            if len(self.calls) >= self.rpm:
                sleep_s = window - (now - self.calls[0]) + 0.05
                sleep_s = max(0.0, sleep_s)
            else:
                sleep_s = 0.0

        if sleep_s > 0:
            time.sleep(sleep_s)

        with self.lock:
            self.calls.append(time.time())


def zai_chat_completions(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    thinking_type: str = "disabled",
    timeout: int = 120,
    rate_limiter: Optional[RateLimiter] = None,
) -> str:
    """
    Z.ai Chat Completions HTTP 调用。
    - Authorization: Bearer <api_key> :contentReference[oaicite:4]{index=4}
    - Vision: messages.content 支持 image_url / text 混合 :contentReference[oaicite:5]{index=5}
    """
    if rate_limiter:
        rate_limiter.acquire()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "thinking": {"type": thinking_type},  # 文档示例中存在 thinking.type :contentReference[oaicite:6]{index=6}
    }

    resp = requests.post(ZAI_BASE_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")

    data = resp.json()
    # 兼容常见结构：choices[0].message.content
    try:
        return (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)[:2000]


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    从模型输出里提取 JSON 对象（容错：可能带额外文字/代码块）。
    """
    if not text:
        raise ValueError("空响应")
    # 去掉 ```json ``` 包裹
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未检测到 JSON 对象")
    obj_str = cleaned[start:end + 1]
    return json.loads(obj_str)


def pil_to_data_url(img: Image.Image, quality: int = 92) -> str:
    """
    PIL -> data:image/jpeg;base64,...
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# =========================================================
# 功能 A：分镜 JSON 生成（原 Gemini 版本 => 改 Z.ai）
# =========================================================
def build_storyboard_prompt(brand: str, product: str, duration_sec: int, style: str) -> str:
    return f"""
你是一位资深短视频导演和广告文案，擅长为抖音 / 小红书 / 视频号设计高转化竖版广告。

请为下面的产品设计一个时长约 {duration_sec} 秒的竖版短视频广告分镜，包含每个镜头的文案和用于 AI 出图的英文提示词。

品牌：{brand}
产品：{product}
整体风格：{style}

要求：
1. 输出必须是标准 JSON（不要任何多余解释、注释或 Markdown），顶层结构：
{{
  "brand": "...",
  "product": "...",
  "duration_sec": {duration_sec},
  "style": "...",
  "scenes": [
    {{
      "id": "S01",
      "time_range": "0.0-2.0",
      "shot_desc": "中文，描述画面，适合给导演看的分镜描述",
      "camera": "中文，镜头机位与运动（如：手持中景推近、航拍俯视摇镜等）",
      "action": "中文，人物动作与关键行为",
      "mood": "中文，情绪氛围（如温馨、紧张、治愈、烟火气）",
      "voiceover": "中文旁白/口播文案，口语化、有销售力，适合配音直接念",
      "image_prompt_en": "英文提示词，用于生成这一镜头的 AI 静帧画面，包含人物、环境、光线、镜头、画质等细节"
    }}
  ]
}}

2. 注意：
- time_range 从 0.0 秒开始，后一镜头的开始时间紧接前一镜头结束时间，总时长控制在 {duration_sec} 秒左右。
- voiceover 尽量自然口语化，像一个真实主播在讲，而不是新闻播音腔。
- image_prompt_en 要尽量详细、摄影感强（close-up / medium / wide / 9:16 / cinematic lighting / 8k 等）。
""".strip()


def generate_storyboard_zai(
    api_key: str,
    model: str,
    brand: str,
    product: str,
    duration_sec: int,
    style: str,
    limiter: RateLimiter,
) -> Dict[str, Any]:
    prompt = build_storyboard_prompt(brand, product, duration_sec, style)
    text = zai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2500,
        thinking_type="disabled",
        rate_limiter=limiter,
    )
    return extract_json_object(text)


def extract_voiceover(data: Dict[str, Any]) -> str:
    scenes = data.get("scenes", []) or []
    lines = []
    for s in scenes:
        sid = s.get("id", "")
        t = s.get("time_range", "")
        vo = s.get("voiceover", "")
        if vo:
            lines.append(f"[{sid} | {t}] {vo}")
    return "\n".join(lines).strip()


# =========================================================
# 功能 B：视频抽帧 + 多帧视觉分析（原 Gemini => 改 Z.ai Vision）
# =========================================================
DISPLAY_IMAGE_WIDTH = 320
PALETTE_WIDTH = 320
PALETTE_HEIGHT = 26


def extract_keyframes_dynamic(
    video_path: str,
    min_frames: int = 6,
    max_frames: int = 30,
    base_fps: float = 0.8,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[List[Image.Image], float, Tuple[float, float]]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        return [], 0.0, (0.0, 0.0)

    duration = total_frames / fps

    if start_sec is None or start_sec < 0:
        start_sec = 0.0
    if end_sec is None or end_sec <= start_sec or end_sec > duration:
        end_sec = duration

    start_frame = int(start_sec * fps)
    end_frame_excl = min(total_frames, int(end_sec * fps))
    segment_frames = end_frame_excl - start_frame

    if segment_frames <= 0:
        start_sec, end_sec = 0.0, duration
        start_frame, end_frame_excl = 0, total_frames
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        else:
            images.append(Image.new("RGB", (200, 200), color="gray"))

    cap.release()
    return images, duration, (float(start_sec), float(end_sec))


def download_video_from_url(url: str) -> str:
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


def get_color_palette(pil_img: Image.Image, num_colors: int = 5):
    img_small = pil_img.resize((120, 120))
    arr = np.array(img_small)
    data = arr.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, flags)
    centers = centers.astype(int)
    return [tuple(map(int, c)) for c in centers]


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


def analyze_single_image_zai(
    api_key: str,
    vision_model: str,
    index: int,
    img: Image.Image,
    limiter: RateLimiter,
) -> Dict[str, Any]:
    """
    用 Z.ai 视觉模型（如 glm-4.6v）做单帧分析。
    按官方格式：messages[0].content = [ {type:image_url,...}, {type:text,...} ] :contentReference[oaicite:7]{index=7}
    """
    prompt = f"""
你现在是电影导演 + 摄影指导 + 服化道总监 + 提示词工程师。
请仔细分析给你的这一帧画面，并输出一个 JSON 对象（只输出 JSON，不要任何解释）。

必须使用下面这些 key（英文），value 大部分为中文说明，英文提示词字段为英文：

{{
  "index": {index},
  "scene_description_zh": "1-3句中文，极具体描述人物+动作路径+场景前中后景+机位视角；忽略UI文字",
  "tags_zh": ["#标签1","#标签2"],
  "camera": {{
    "shot_type_zh": "远景/全景/中景/近景/特写",
    "shot_type": "wide/full/medium/close-up",
    "angle_zh": "俯拍/仰拍/平视/侧拍",
    "angle": "high/low/eye-level",
    "movement_zh": "推近/跟拍/横移/手持/甩镜",
    "movement": "dolly-in/handheld tracking/pan",
    "composition_zh": "三分法/中心/对称/前景-主体-背景",
    "composition": "rule-of-thirds/center/symmetry"
  }},
  "color_and_light_zh": "1-2句色调/光线/主光方向/轮廓光",
  "mood_zh": "情绪氛围",
  "characters": [
    {{
      "role_zh": "身份",
      "gender_zh": "性别",
      "age_look_zh": "年龄观感",
      "body_type_zh": "体型",
      "clothing_zh": "服装颜色款式",
      "hair_zh": "发型发色",
      "expression_zh": "表情",
      "pose_body_zh": "姿态",
      "props_zh": "道具"
    }}
  ],
  "character_action_detail_zh": "1-3句，头->手->躯干->腿，写清接触点",
  "face_expression_detail_zh": "1-3句，眉眼嘴下颌、眼神细节、外力形变回弹",
  "cloth_hair_reaction_zh": "1-3句，风/惯性对头发衣服影响",
  "environment_detail_zh": "2-4句，前景/中景/背景，材质与空间结构",
  "weather_force_detail_zh": "风雨雪/气流/冲击波方向与反馈（无则写无）",
  "props_and_tech_detail_zh": "1-3句，列出关键道具与位置状态",
  "physics_reaction_detail_zh": "受力/形变/回弹过程（无则写无）",
  "structure_damage_detail_zh": "结构损坏（无则写无）",
  "debris_motion_detail_zh": "碎片飞散轨迹（无则写无）",
  "motion_detail_zh": "上一瞬间->当前->下一瞬间动作推断",
  "fx_detail_zh": "火花烟尘粒子（无则写无）",
  "lighting_color_detail_zh": "更精细光源数量方向色温差",
  "audio_cue_detail_zh": "环境声+特效声+BGM节奏点",
  "edit_rhythm_detail_zh": "剪辑节奏/慢动作/闪白等",
  "midjourney_prompt": "一行英文MJ提示词",
  "midjourney_negative_prompt": "一行英文负面词",
  "video_prompt_en": "3-5句英文视频提示词，最后一句写：'4 second shot, vertical 9:16, 24fps, cinematic, highly detailed.'"
}}
""".strip()

    try:
        data_url = pil_to_data_url(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = zai_chat_completions(
            api_key=api_key,
            model=vision_model,
            messages=messages,
            temperature=0.4,
            max_tokens=2600,
            thinking_type="disabled",
            rate_limiter=limiter,
        )
        info = extract_json_object(text)

        # 保底字段
        info["index"] = index
        info.setdefault("tags_zh", [])
        info.setdefault("camera", {})
        cam = info["camera"]
        for k in ["shot_type_zh","shot_type","angle_zh","angle","movement_zh","movement","composition_zh","composition"]:
            cam.setdefault(k, "")

        for k in [
            "scene_description_zh","color_and_light_zh","mood_zh","characters",
            "character_action_detail_zh","face_expression_detail_zh","cloth_hair_reaction_zh",
            "environment_detail_zh","weather_force_detail_zh","props_and_tech_detail_zh",
            "physics_reaction_detail_zh","structure_damage_detail_zh","debris_motion_detail_zh",
            "motion_detail_zh","fx_detail_zh","lighting_color_detail_zh","audio_cue_detail_zh",
            "edit_rhythm_detail_zh","midjourney_prompt","midjourney_negative_prompt","video_prompt_en"
        ]:
            info.setdefault(k, "" if k != "characters" else [])

        return info

    except Exception as e:
        return {
            "index": index,
            "scene_description_zh": f"（AI 分析失败：{e}）",
            "tags_zh": [],
            "camera": {
                "shot_type_zh": "", "shot_type": "",
                "angle_zh": "", "angle": "",
                "movement_zh": "", "movement": "",
                "composition_zh": "", "composition": "",
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


def analyze_images_concurrently_zai(
    api_key: str,
    vision_model: str,
    images: List[Image.Image],
    max_ai_frames: int,
    limiter: RateLimiter,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    n = len(images)
    if n == 0:
        return []
    use_n = min(max_ai_frames, n)
    results: List[Dict[str, Any]] = [None] * n  # type: ignore

    status = st.empty()
    status.info(f"正在对前 {use_n} 帧进行 AI 分析（共 {n} 帧）…")

    max_workers = max(1, min(int(max_workers), 8, use_n))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(analyze_single_image_zai, api_key, vision_model, i + 1, images[i], limiter): i
            for i in range(use_n)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    for i in range(use_n, n):
        results[i] = {
            "index": i + 1,
            "scene_description_zh": "（本帧未做 AI 分析，用于节省配额，但仍可用于视觉参考和色卡。）",
            "tags_zh": [],
            "camera": {"shot_type_zh":"","shot_type":"","angle_zh":"","angle":"","movement_zh":"","movement":"","composition_zh":"","composition":""},
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


def analyze_overall_video_zai(frame_infos: List[Dict[str, Any]], api_key: str, text_model: str, limiter: RateLimiter) -> str:
    described = [
        info for info in frame_infos
        if info.get("scene_description_zh")
        and "未做 AI 分析" not in info["scene_description_zh"]
        and "AI 分析失败" not in info["scene_description_zh"]
    ]
    if not described:
        return "（暂未获取到有效的帧级分析，无法生成整体剧情大纲。）"

    parts = []
    for info in described:
        cam = info.get("camera", {}) or {}
        tags = info.get("tags_zh", []) or []
        parts.append(
            f"第 {info['index']} 帧：{info.get('scene_description_zh','')}\n"
            f"景别：{cam.get('shot_type_zh','')}；角度：{cam.get('angle_zh','')}；运镜：{cam.get('movement_zh','')}；构图：{cam.get('composition_zh','')}\n"
            f"色彩与光影：{info.get('color_and_light_zh','')}\n"
            f"情绪氛围：{info.get('mood_zh','')}\n"
            f"标签：{'、'.join(tags)}"
        )
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
用 #标签 形式给出 5-10 个。

【商业与合规风险】
整体风险级别：低 / 中 / 高
并用 2-3 句话说明需要注意的点。

只输出以上 4 个小节，不要添加额外说明。
""".strip()

    return zai_chat_completions(
        api_key=api_key,
        model=text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1800,
        thinking_type="disabled",
        rate_limiter=limiter,
    )


def generate_ad_script_zai(frame_infos: List[Dict[str, Any]], api_key: str, text_model: str, limiter: RateLimiter) -> str:
    described = [
        info for info in frame_infos
        if info.get("scene_description_zh")
        and "未做 AI 分析" not in info["scene_description_zh"]
        and "AI 分析失败" not in info["scene_description_zh"]
    ]
    if not described:
        return "（暂未获取到有效的帧级分析，无法生成广告旁白脚本。）"

    joined = "\n".join([f"第 {i['index']} 帧：{i.get('scene_description_zh','')}" for i in described])

    prompt = f"""
你是一名资深广告导演 + 文案。
我有一个由若干画面组成的竖版短视频，时长大约 8-12 秒。
下面是每个画面的简要说明，请你基于这些信息，写一条适合配合这些画面播放的中文广告旁白脚本。

=== 关键帧概览 ===
{joined}
=== 关键帧概览结束 ===

要求：
1) 旁白总时长控制在 8-12 秒左右（正常语速），文本 35-70 字；
2) 自然口语化中文，不要出现“画面中”“镜头里”；
3) 与画面调性匹配。

按下面格式输出：

【10秒广告旁白脚本】
（完整一段旁白）

不要输出其他任何内容。
""".strip()

    return zai_chat_completions(
        api_key=api_key,
        model=text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600,
        thinking_type="disabled",
        rate_limiter=limiter,
    )


def generate_timeline_shotlist(frame_infos: List[Dict[str, Any]], used_range: Tuple[float, float]) -> str:
    n = len(frame_infos)
    if n == 0:
        return "（暂无关键帧，无法生成时间轴分镜脚本。）"

    start_used, end_used = used_range
    total_len = max(0.1, end_used - start_used)
    seg = total_len / n

    lines: List[str] = []
    for i, info in enumerate(frame_infos):
        t0 = i * seg
        t1 = total_len if i == n - 1 else (i + 1) * seg
        shot_id = f"S{i+1:02d}"

        cam = info.get("camera", {}) or {}
        tags = info.get("tags_zh", []) or []

        def g(k: str) -> str:
            v = info.get(k, "")
            return (v or "").strip() if isinstance(v, str) else ""

        block = [f"【{shot_id} | {t0:.1f}-{t1:.1f} 秒】"]
        if g("scene_description_zh"): block.append(f"画面内容：{g('scene_description_zh')}")
        if g("character_action_detail_zh"): block.append(f"人物动作：{g('character_action_detail_zh')}")
        if g("face_expression_detail_zh"): block.append(f"面部与眼神：{g('face_expression_detail_zh')}")
        if g("cloth_hair_reaction_zh"): block.append(f"服装与头发：{g('cloth_hair_reaction_zh')}")
        if g("environment_detail_zh"): block.append(f"场景与空间：{g('environment_detail_zh')}")
        if g("weather_force_detail_zh"): block.append(f"天气与环境力：{g('weather_force_detail_zh')}")
        if g("props_and_tech_detail_zh"): block.append(f"道具与科技：{g('props_and_tech_detail_zh')}")
        if g("structure_damage_detail_zh"): block.append(f"结构损坏：{g('structure_damage_detail_zh')}")
        if g("debris_motion_detail_zh"): block.append(f"碎片与飞散轨迹：{g('debris_motion_detail_zh')}")
        if g("physics_reaction_detail_zh"): block.append(f"受力与物理反馈：{g('physics_reaction_detail_zh')}")
        if g("fx_detail_zh"): block.append(f"特效与粒子：{g('fx_detail_zh')}")
        if g("lighting_color_detail_zh"): block.append(f"光线与色彩：{g('lighting_color_detail_zh')}")

        cam_desc = []
        if cam.get("shot_type_zh"): cam_desc.append(f"景别：{cam.get('shot_type_zh')}")
        if cam.get("angle_zh"): cam_desc.append(f"角度：{cam.get('angle_zh')}")
        if cam.get("movement_zh"): cam_desc.append(f"运镜：{cam.get('movement_zh')}")
        if cam.get("composition_zh"): cam_desc.append(f"构图：{cam.get('composition_zh')}")
        if cam_desc: block.append("机位与运动：" + "；".join(cam_desc))

        if g("mood_zh"): block.append(f"情绪氛围：{g('mood_zh')}")
        if g("motion_detail_zh"): block.append(f"动作趋势：{g('motion_detail_zh')}")
        if g("audio_cue_detail_zh"): block.append(f"声音与节奏：{g('audio_cue_detail_zh')}")
        if g("edit_rhythm_detail_zh"): block.append(f"剪辑与节奏：{g('edit_rhythm_detail_zh')}")
        if tags: block.append("标签：" + " ".join(tags))

        lines.append("\n".join(block))

    return "\n\n".join(lines)


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="智谱/GLM 分镜 & 视频分析工具", page_icon="🎬", layout="wide")

# Session State
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []

st.markdown(
    """
    <style>
    .main { background-color: #0f172a; color: #e5e7eb; }
    .stMarkdown, .stText, label, p, div { color: #e5e7eb !important; }
    .stCode { font-size: 0.85rem !important; }
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
        🎬 智谱/GLM：分镜生成 + 视频关键帧分析（免SDK版）
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        统一走 Z.ai HTTP API：分镜JSON生成 / 视频抽帧 / 视觉分析 / 时间轴脚本 / 历史记录与下载。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🔑 API 配置（Z.ai / 智谱）")
    api_key_env = os.getenv("ZAI_API_KEY", "").strip()
    api_key = st.text_input("ZAI_API_KEY（优先用环境变量）", type="password", value=api_key_env)

    st.markdown("---")
    text_model = st.text_input("文本模型（用于：分镜/总结/广告文案）", value=DEFAULT_TEXT_MODEL)
    vision_model = st.text_input("视觉模型（用于：帧分析）", value=DEFAULT_VISION_MODEL)

    st.markdown("---")
    rpm = st.slider("每分钟最大调用次数（限流）", 1, 60, 10, 1)
    limiter = RateLimiter(rpm=rpm)

    st.markdown("---")
    max_ai_frames = st.slider("本次最多做 AI 分析的帧数", 1, 20, 10, 1)
    max_workers = st.slider("并发线程数（建议 2-4）", 1, 8, 3, 1)

    st.markdown("---")
    st.markdown("⏱ 分析时间范围（秒）")
    start_sec = st.number_input("从第几秒开始（含）", min_value=0.0, value=0.0, step=0.5)
    end_sec = st.number_input("到第几秒结束（0 或 ≤开始秒 表示直到结尾）", min_value=0.0, value=0.0, step=0.5)

    if not api_key:
        st.warning("请设置环境变量 ZAI_API_KEY，或在此粘贴 Key。")
    else:
        st.success("Key 已就绪。")


tab_storyboard, tab_video = st.tabs(["🧩 分镜 JSON 生成（文本）", "🎞 视频关键帧分析（视觉）"])

# -------------------------------
# Tab 1：分镜 JSON 生成
# -------------------------------
with tab_storyboard:
    st.subheader("🧩 分镜 + 口播文案 + 英文出图提示词（JSON）")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("品牌（必填）", value="邵警秘卤")
        product = st.text_input("产品（必填）", value="卤鸭脖+卤鸭翅 夜宵套餐")
        duration_sec = st.number_input("视频时长（秒）", min_value=5, max_value=120, value=15, step=1)
    with col2:
        style = st.text_area("整体风格（中文描述）", value="烟火气、夜宵档、真实街边风格，有点幽默，适合抖音", height=110)

    if st.button("✨ 生成分镜 & 文案（走智谱/GLM）", type="primary", key="btn_story"):
        if not api_key:
            st.error("请先配置 ZAI_API_KEY")
        elif not brand or not product:
            st.error("请先填写品牌和产品")
        else:
            with st.spinner("正在调用智谱/GLM生成分镜…"):
                try:
                    data = generate_storyboard_zai(api_key, text_model, brand, product, int(duration_sec), style, limiter)
                except Exception as e:
                    st.error(f"生成失败：{e}")
                else:
                    st.success("生成完成！")
                    st.subheader("📜 分镜 JSON")
                    st.json(data)

                    voice_script = extract_voiceover(data)
                    st.subheader("🎙 旁白脚本")
                    st.text_area("可复制给配音用", value=voice_script, height=220)

                    st.download_button(
                        "下载 storyboard.json",
                        data=json.dumps(data, ensure_ascii=False, indent=2),
                        file_name="storyboard.json",
                        mime="application/json",
                    )
                    st.download_button(
                        "下载 voiceover_script.txt",
                        data=voice_script,
                        file_name="voiceover_script.txt",
                        mime="text/plain",
                    )

# -------------------------------
# Tab 2：视频关键帧分析
# -------------------------------
with tab_video:
    st.subheader("🎞 上传/链接 → 抽关键帧 → 视觉分析 → 时间轴脚本 → JSON/历史记录")

    source_mode = st.radio(
        "📥 选择视频来源",
        ["上传本地文件", "输入网络视频链接（抖音 / B站 / TikTok / YouTube）"],
        index=0,
        horizontal=True,
    )
    video_url: Optional[str] = None
    uploaded_file = None

    if source_mode == "上传本地文件":
        uploaded_file = st.file_uploader("📂 上传视频文件（建议 < 50MB）", type=["mp4", "mov", "m4v", "avi", "mpeg"])
    else:
        video_url = st.text_input("🔗 输入视频链接", placeholder="例如：https://v.douyin.com/xxxxxx 或 https://www.youtube.com/watch?v=...")

    if st.button("🚀 一键解析整条视频（走智谱/GLM视觉）", type="primary", key="btn_video"):
        if not api_key:
            st.error("请先配置 ZAI_API_KEY")
            st.stop()

        tmp_path: Optional[str] = None
        source_label = ""
        source_type = ""

        try:
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
                st.info("🌐 正在从网络下载视频，请稍候…")
                tmp_path = download_video_from_url(video_url)
                source_label = video_url

            st.info("⏳ 正在抽取关键帧…")
            images, duration, used_range = extract_keyframes_dynamic(
                tmp_path,
                start_sec=float(start_sec),
                end_sec=float(end_sec) if end_sec and end_sec > 0 else None,
            )
            start_used, end_used = used_range

            try:
                if tmp_path:
                    os.remove(tmp_path)
            except OSError:
                pass

            if not images:
                st.error("无法从视频中读取帧，请检查格式或文件是否损坏。")
                st.stop()

            st.success(f"✅ 抽取 {len(images)} 帧（视频总长约 {duration:.1f}s；分析区间 {start_used:.1f}-{end_used:.1f}s）")

            # 色卡
            frame_palettes: List[List[Tuple[int, int, int]]] = []
            for img in images:
                try:
                    frame_palettes.append(get_color_palette(img, num_colors=5))
                except Exception:
                    frame_palettes.append([])

            # 帧分析（视觉）
            with st.spinner("🧠 正在调用视觉模型分析关键帧…"):
                frame_infos = analyze_images_concurrently_zai(
                    api_key=api_key,
                    vision_model=vision_model,
                    images=images,
                    frame_infos = analyze_images_concurrently_zai(
    api_key=api_key,
    vision_model=vision_model,
    images=images,
    max_ai_frames=int(max_ai_frames),
    limiter=limiter,
    max_workers=int(max_workers),
)
                    ,
                    limiter=limiter,
                    max_workers=int(max_workers),
                )

            # 整体总结 + 广告旁白 + 时间轴（文本）
            with st.spinner("📚 整体剧情总结…"):
                overall = analyze_overall_video_zai(frame_infos, api_key, text_model, limiter)
            with st.spinner("🎤 10秒广告旁白…"):
                ad_script = generate_ad_script_zai(frame_infos, api_key, text_model, limiter)
            with st.spinner("🎬 时间轴分镜脚本…"):
                timeline_shotlist = generate_timeline_shotlist(frame_infos, used_range=used_range)

            export_frames = []
            for info, palette in zip(frame_infos, frame_palettes):
                export_frames.append({
                    **info,
                    "palette_rgb": [list(c) for c in (palette or [])],
                    "palette_hex": [rgb_to_hex(c) for c in (palette or [])],
                })

            export_data = {
                "meta": {
                    "text_model": text_model,
                    "vision_model": vision_model,
                    "frame_count": len(images),
                    "max_ai_frames_this_run": int(max_ai_frames),
                    "duration_sec_est": float(duration),
                    "start_sec_used": float(start_used),
                    "end_sec_used": float(end_used),
                    "source_type": source_type,
                    "source_label": source_label,
                    "api_base_url": ZAI_BASE_URL,
                },
                "frames": export_frames,
                "overall_analysis": overall,
                "ad_script_10s": ad_script,
                "timeline_shotlist_zh": timeline_shotlist,
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

            # 历史记录
            history = st.session_state["analysis_history"]
            run_id = f"run_{len(history) + 1}"
            history.append({
                "id": run_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "meta": export_data["meta"],
                "data": export_data,
            })
            st.session_state["analysis_history"] = history

            tab_frames, tab_story, tab_json, tab_history = st.tabs(
                ["🎞 关键帧 & 提示词", "📚 总结 & 广告 & 时间轴", "📦 JSON 导出（本次）", "🕘 历史记录"]
            )

            with tab_frames:
                st.markdown(f"共 **{len(images)}** 帧；其中前 **{min(len(images), int(max_ai_frames))}** 帧做了 AI 视觉分析。")
                st.markdown("---")

                for i, (img, info, palette) in enumerate(zip(images, frame_infos, frame_palettes)):
                    st.markdown(f"### 📘 关键帧 {i+1}")
                    c1, c2 = st.columns([1.2, 2])

                    with c1:
                        st.image(img, caption=f"第 {i+1} 帧画面", width=DISPLAY_IMAGE_WIDTH)
                        palette_img = make_palette_image(palette)
                        st.image(palette_img, caption="主色调色卡", width=PALETTE_WIDTH)
                        if palette:
                            st.caption("主色 HEX：" + ", ".join(rgb_to_hex(c) for c in palette))

                    with c2:
                        cam = info.get("camera", {}) or {}
                        tags = info.get("tags_zh", []) or []
                        analysis_text = "\n".join([
                            f"【景别】{cam.get('shot_type_zh','')}",
                            f"【运镜】{cam.get('movement_zh','')}",
                            f"【拍摄角度】{cam.get('angle_zh','')}",
                            f"【构图】{cam.get('composition_zh','')}",
                            f"【色彩与光影】{info.get('color_and_light_zh','')}",
                            f"【画面内容】{info.get('scene_description_zh','')}",
                            f"【情绪氛围】{info.get('mood_zh','')}",
                            f"【关键词标签】{' '.join(tags)}",
                        ]).strip()

                        st.markdown("**分镜分析（可复制）：**")
                        st.code(analysis_text or "（暂无）", language="markdown")

                        st.markdown("**人物动作细节（可复制）：**")
                        st.code(info.get("character_action_detail_zh") or "（暂无）", language="markdown")

                        st.markdown("**场景细节（可复制）：**")
                        scene_detail = (info.get("environment_detail_zh") or "").strip()
                        props_detail = (info.get("props_and_tech_detail_zh") or "").strip()
                        st.code((scene_detail + "\n\n道具与科技元素：" + props_detail).strip() or "（暂无）", language="markdown")

                        advanced = []
                        for title, key in [
                            ("面部与眼神", "face_expression_detail_zh"),
                            ("服装与头发", "cloth_hair_reaction_zh"),
                            ("天气与环境力", "weather_force_detail_zh"),
                            ("受力与物理反馈", "physics_reaction_detail_zh"),
                            ("结构损坏", "structure_damage_detail_zh"),
                            ("碎片飞散", "debris_motion_detail_zh"),
                            ("特效与粒子", "fx_detail_zh"),
                            ("光线细节", "lighting_color_detail_zh"),
                            ("声音与节奏", "audio_cue_detail_zh"),
                            ("剪辑节奏", "edit_rhythm_detail_zh"),
                        ]:
                            v = (info.get(key) or "").strip()
                            if v:
                                advanced.append(f"{v}")
                        if advanced:
                            st.markdown("**高级物理 / 环境细节（可复制）：**")
                            st.code("\n".join(advanced), language="markdown")

                        st.markdown("**SORA / VEO 视频提示词（英文）：**")
                        st.code(info.get("video_prompt_en") or "（暂无）", language="markdown")

                        st.markdown("**Midjourney 静帧提示词：**")
                        st.code(info.get("midjourney_prompt") or "（暂无）", language="markdown")

                    st.markdown("---")

            with tab_story:
                st.markdown("### 📚 整体剧情与视听风格总结")
                st.code(overall, language="markdown")
                st.markdown("### 🎤 10 秒广告旁白脚本")
                st.code(ad_script, language="markdown")
                st.markdown("### 🎬 时间轴分镜脚本")
                st.code(timeline_shotlist, language="markdown")

            with tab_json:
                st.download_button("⬇️ 下载本次 video_analysis.json", data=json_str, file_name="video_analysis.json", mime="application/json")
                with st.expander("🔍 预览 JSON（前 3000 字符）"):
                    st.code(json_str[:3000] + ("\n...\n" if len(json_str) > 3000 else ""), language="json")

            with tab_history:
                history = st.session_state.get("analysis_history", [])
                if not history:
                    st.info("当前会话还没有历史记录。")
                else:
                    options = [
                        f"{len(history)-i}. {h['created_at']} | {h['meta'].get('source_label','')} | "
                        f"{h['meta'].get('frame_count',0)} 帧 | 区间 {h['meta'].get('start_sec_used',0):.1f}-{h['meta'].get('end_sec_used',0):.1f}s"
                        for i, h in enumerate(reversed(history))
                    ]
                    idx_display = st.selectbox("选择一条历史记录查看", options=list(range(len(history))), format_func=lambda i: options[i])
                    real_index = len(history) - 1 - idx_display
                    selected = history[real_index]

                    st.markdown(
                        f"**ID：** `{selected['id']}`  \n"
                        f"**时间：** {selected['created_at']}  \n"
                        f"**来源类型：** {selected['meta'].get('source_type','')}  \n"
                        f"**来源标识：** {selected['meta'].get('source_label','')}  \n"
                        f"**分析区间：** {selected['meta'].get('start_sec_used',0):.1f}–{selected['meta'].get('end_sec_used',0):.1f} 秒  \n"
                        f"**帧数：** {selected['meta'].get('frame_count',0)}  \n"
                        f"**文本模型：** {selected['meta'].get('text_model','')}  \n"
                        f"**视觉模型：** {selected['meta'].get('vision_model','')}"
                    )

                    hist_json = json.dumps(selected["data"], ensure_ascii=False, indent=2)
                    st.download_button("⬇️ 下载该历史记录 JSON", data=hist_json, file_name=f"video_analysis_{selected['id']}.json", mime="application/json")

        except Exception as e:
            st.error(f"下载或解析视频时发生错误：{e}")

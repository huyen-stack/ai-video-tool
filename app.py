import os
import json
import time
import base64
import tempfile
import threading
import concurrent.futures
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yt_dlp


# =========================
# Z.ai / 智谱 HTTP Client（不使用 SDK）
# =========================

DEFAULT_ZAI_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class RateLimiter:
    """简单的按分钟限流：同一窗口内最多 N 次请求。"""
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.lock = threading.Lock()
        self.ts: List[float] = []

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # 只保留 60 秒窗口内的时间戳
                self.ts = [t for t in self.ts if now - t < 60.0]
                if len(self.ts) < self.rpm:
                    self.ts.append(now)
                    return
                # 需要等待
                wait = 60.0 - (now - self.ts[0]) + 0.02
            time.sleep(max(0.05, wait))


def _clean_key(k: str) -> str:
    # 避免 “Illegal header value” 之类问题：去掉换行/空格
    return (k or "").strip().replace("\n", "").replace("\r", "")


def zai_chat_completion(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    endpoint: str = DEFAULT_ZAI_ENDPOINT,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> str:
    api_key = _clean_key(api_key)
    if not api_key:
        raise RuntimeError("缺少 ZAI_API_KEY（智谱 API Key）")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    # 常见结构：choices[0].message.content
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"无法解析返回结构：{json.dumps(data, ensure_ascii=False)[:2000]}")

    # content 可能是字符串，也可能是数组
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # 兼容多段内容
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts).strip()

    return str(content).strip()


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """从模型输出中截取第一个 { 到最后一个 }，并解析 JSON。"""
    if not text:
        raise ValueError("空返回")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未检测到 JSON 对象")
    js = text[start : end + 1]
    return json.loads(js)


def _pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
    img.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# =========================
# 视频处理：下载/抽帧/色卡
# =========================

DISPLAY_IMAGE_WIDTH = 320
PALETTE_WIDTH = 320
PALETTE_HEIGHT = 26


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


def extract_keyframes_dynamic(
    video_path: str,
    min_frames: int = 6,
    max_frames: int = 30,
    base_fps: float = 0.8,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[List[Image.Image], float, Tuple[float, float]]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-2:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        return [], 0.0, (0.0, 0.0)

    duration = total_frames / fps

    if start_sec is None or start_sec < 0:
        start_sec = 0.0
    if end_sec is None or end_sec <= start_sec or end_sec > duration:
        end_sec = duration

    start_frame = max(0, int(start_sec * fps))
    end_frame_excl = min(total_frames, int(end_sec * fps))
    segment_frames = end_frame_excl - start_frame

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        else:
            images.append(Image.new("RGB", (320, 320), color="gray"))

    cap.release()
    return images, duration, (float(start_sec), float(end_sec))


def get_color_palette(pil_img: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    img_small = pil_img.resize((120, 120))
    arr = np.array(img_small)
    data = arr.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, _, centers = cv2.kmeans(data, num_colors, None, criteria, 10, flags)
    centers = centers.astype(int)
    return [tuple(map(int, c)) for c in centers]


def make_palette_image(colors: List[Tuple[int, int, int]], width: int = PALETTE_WIDTH, height: int = PALETTE_HEIGHT) -> Image.Image:
    if not colors:
        return Image.new("RGB", (width, height), color="gray")

    bar = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(bar)
    n = len(colors)
    band_w = max(width // n, 1)

    for i, color in enumerate(colors):
        x0 = i * band_w
        x1 = width if i == n - 1 else (i + 1) * band_w
        draw.rectangle([x0, 0, x1, height], fill=color)
    return bar


def rgb_to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
    r, g, b = rgb_tuple
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


# =========================
# 业务：分镜 JSON 生成（文本）
# =========================

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
      "shot_desc": "中文，描述画面，适合给导演看的分镜描述（具体、可搭景）",
      "camera": "中文，镜头机位与运动（如：手持中景推近、航拍俯视摇镜等）",
      "action": "中文，人物动作与关键行为（明确姿态、接触点、运动方向）",
      "mood": "中文，情绪氛围（如温馨、紧张、治愈、烟火气）",
      "voiceover": "中文旁白/口播文案，口语化、有销售力",
      "image_prompt_en": "英文提示词，用于生成该镜头静帧：人物、环境、光线、镜头、画质、9:16、cinematic"
    }}
  ]
}}

2. time_range 从 0.0 秒开始，后一镜头紧接前一镜头结束，总时长控制在 {duration_sec} 秒左右。
3. voiceover 像真实主播口播，不要新闻播音腔。
4. 只输出 JSON。
""".strip()


def generate_storyboard_zai(api_key: str, endpoint: str, model_text: str, brand: str, product: str, duration_sec: int, style: str) -> Dict[str, Any]:
    prompt = build_storyboard_prompt(brand, product, duration_sec, style)
    messages = [
        {"role": "system", "content": "你只输出严格 JSON，不要任何解释。"},
        {"role": "user", "content": prompt},
    ]
    text = zai_chat_completion(
        api_key=api_key,
        endpoint=endpoint,
        model=model_text,
        messages=messages,
        temperature=0.2,
        max_tokens=2500,
        timeout=120,
    )
    return _extract_json_from_text(text)


def extract_voiceover(data: Dict[str, Any]) -> str:
    scenes = data.get("scenes", []) or []
    lines = []
    for s in scenes:
        sid = s.get("id", "")
        tr = s.get("time_range", "")
        vo = s.get("voiceover", "")
        if vo:
            lines.append(f"[{sid} | {tr}] {vo}")
    return "\n".join(lines).strip()


# =========================
# 业务：单帧视觉分析（输出 JSON）
# =========================

def build_frame_prompt(index: int) -> str:
    # 为减少 token/失败率，这里保留你需要的核心字段结构，且仍然很“细”。
    # 你如果还要更长版本，可以把这里扩写。
    return f"""
你现在是电影导演 + 摄影指导 + 服化道总监 + 提示词工程师。
请分析给你的这一帧画面，并输出一个 JSON 对象（只输出 JSON）。

必须包含下列字段（英文 key），中文字段写中文，英文提示词字段写英文：

{{
  "index": {index},
  "scene_description_zh": "1-3 句中文：人物身份/外观/服装 + 正在做的具体动作路径 + 场景结构(前中后景) + 机位视角与运动。忽略任何 UI/字幕。",
  "tags_zh": ["#标签1","#标签2"],
  "camera": {{
    "shot_type_zh": "远景/全景/中景/近景/特写",
    "angle_zh": "俯拍/仰拍/平视/侧拍等",
    "movement_zh": "推近/跟拍/横移/手持/甩镜等",
    "composition_zh": "三分法/中心/对称/前中后景层次等"
  }},
  "color_and_light_zh": "色温/对比/主光方向/轮廓光等",
  "mood_zh": "紧张/温暖/冷峻/商业感等",
  "character_action_detail_zh": "按 头部→上肢→躯干→下肢 描述动作细节，重心、接触点、受力。",
  "face_expression_detail_zh": "眉眼口下颌肌肉状态，眼神与情绪，是否有外力形变与回弹。",
  "cloth_hair_reaction_zh": "头发衣物对风/惯性/动作的反应。",
  "environment_detail_zh": "前景/中景/背景空间与材质细节。",
  "weather_force_detail_zh": "风雨雪/气流/冲击波方向与作用；无则写无明显。",
  "props_and_tech_detail_zh": "3-8 个关键道具/科技元素的外观位置状态。",
  "physics_reaction_detail_zh": "形变→回弹/拉扯/震动等物理反馈；无则写无明显。",
  "structure_damage_detail_zh": "结构损坏与部位细节；无则写无明显。",
  "debris_motion_detail_zh": "碎片飞散轨迹；无则写无明显。",
  "motion_detail_zh": "上一瞬间→当前→下一瞬间动作趋势。",
  "fx_detail_zh": "烟尘/火花/能量/粒子；无则写无明显。",
  "lighting_color_detail_zh": "更精细的光源数量/方向/色温差/爆光频闪等。",
  "audio_cue_detail_zh": "环境声/特效声/台词/BGM 情绪节奏点。",
  "edit_rhythm_detail_zh": "剪辑节奏：正常/慢动作/闪白/甩镜转场等。",
  "midjourney_prompt": "一行英文 MJ v6 提示词（9:16, cinematic, ultra detailed, no text）",
  "midjourney_negative_prompt": "text, subtitle, watermark, extra fingers, deformed hands, distorted face, low resolution, blurry",
  "video_prompt_en": "3-5 句英文视频提示词，说明人物外观、动作、运镜方向、地形环境、光线氛围，末句：'4 second shot, vertical 9:16, 24fps, cinematic, highly detailed.'"
}}

要求：只输出 JSON；所有字符串必须双引号；不得有注释/多余逗号。
""".strip()


def analyze_single_image_zai(
    api_key: str,
    endpoint: str,
    vision_model: str,
    img: Image.Image,
    index: int,
    limiter: RateLimiter,
) -> Dict[str, Any]:
    try:
        limiter.acquire()

        data_url = _pil_to_data_url(img, fmt="JPEG", quality=92)
        prompt = build_frame_prompt(index)

        # 多模态消息：text + image_url
        messages = [
            {"role": "system", "content": "你只输出严格 JSON，不要解释。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        text = zai_chat_completion(
            api_key=api_key,
            endpoint=endpoint,
            model=vision_model,
            messages=messages,
            temperature=0.2,
            max_tokens=2600,
            timeout=180,
        )
        info = _extract_json_from_text(text)
        info["index"] = index
        return info
    except Exception as e:
        return {
            "index": index,
            "scene_description_zh": f"（AI 分析失败：{e}）",
            "tags_zh": [],
            "camera": {"shot_type_zh": "", "angle_zh": "", "movement_zh": "", "composition_zh": ""},
            "color_and_light_zh": "",
            "mood_zh": "",
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
    endpoint: str,
    vision_model: str,
    images: List[Image.Image],
    max_ai_frames: int,
    limiter: RateLimiter,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    n = len(images)
    if n == 0:
        return []

    use_n = min(int(max_ai_frames), n)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    status = st.empty()
    status.info(f"正在对前 {use_n} 帧调用智谱视觉模型，其余帧仅保留截图与色卡。")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(use_n, int(max_workers)))) as ex:
        fut_map = {
            ex.submit(analyze_single_image_zai, api_key, endpoint, vision_model, images[i], i + 1, limiter): i
            for i in range(use_n)
        }
        for fut in concurrent.futures.as_completed(fut_map):
            i = fut_map[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"index": i + 1, "scene_description_zh": f"（AI 分析失败：{e}）"}

    for i in range(use_n, n):
        results[i] = {
            "index": i + 1,
            "scene_description_zh": "（本帧未做 AI 分析，用于节省配额，但仍可用于视觉参考和色卡。）",
            "tags_zh": [],
            "camera": {"shot_type_zh": "", "angle_zh": "", "movement_zh": "", "composition_zh": ""},
            "color_and_light_zh": "",
            "mood_zh": "",
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
    return [r for r in results if r is not None]


# =========================
# 整体总结 / 广告旁白 / 时间轴脚本
# =========================

def analyze_overall_video_zai(api_key: str, endpoint: str, model_text: str, frame_infos: List[Dict[str, Any]]) -> str:
    described = [
        f"第{f.get('index')}帧：{f.get('scene_description_zh','')}\n标签：{' '.join(f.get('tags_zh',[]) or [])}"
        for f in frame_infos
        if f.get("scene_description_zh") and "未做 AI 分析" not in f["scene_description_zh"] and "AI 分析失败" not in f["scene_description_zh"]
    ]
    if not described:
        return "（暂未获取到有效帧级分析，无法生成整体总结。）"

    joined = "\n\n".join(described)
    prompt = f"""
你是资深短视频导演 + 剪辑师 + 运营专家。
基于以下关键帧描述，总结整条视频：

===关键帧描述===
{joined}
===结束===

按结构输出（中文）：
【剧情大纲】2-4 句
【整体视听风格】2-4 句
【适合的话题标签】5-10 个 #标签
【商业与合规风险】风险：低/中/高 + 2-3 句说明
只输出上述四段，不要额外解释。
""".strip()

    messages = [{"role": "user", "content": prompt}]
    return zai_chat_completion(api_key, model_text, messages, endpoint=endpoint, temperature=0.2, max_tokens=1200, timeout=120)


def generate_ad_script_zai(api_key: str, endpoint: str, model_text: str, frame_infos: List[Dict[str, Any]]) -> str:
    described = [
        f"第{f.get('index')}帧：{f.get('scene_description_zh','')}"
        for f in frame_infos
        if f.get("scene_description_zh") and "未做 AI 分析" not in f["scene_description_zh"] and "AI 分析失败" not in f["scene_description_zh"]
    ]
    if not described:
        return "（暂未获取到有效帧级分析，无法生成广告旁白。）"
    joined = "\n".join(described)
    prompt = f"""
你是一名资深广告导演 + 文案。
下面是竖版短视频关键帧概览，请写一段 8-12 秒中文口播旁白（35-70 字），口语化、有销售力，不要写“画面中/镜头里”。

===关键帧概览===
{joined}
===结束===

格式：
【10秒广告旁白脚本】
（旁白）
""".strip()
    messages = [{"role": "user", "content": prompt}]
    return zai_chat_completion(api_key, model_text, messages, endpoint=endpoint, temperature=0.4, max_tokens=600, timeout=120)


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
        t1 = (i + 1) * seg
        if i == n - 1:
            t1 = total_len

        sid = f"S{i+1:02d}"
        cam = info.get("camera", {}) or {}
        tags = info.get("tags_zh", []) or []

        def _g(k: str) -> str:
            return (info.get(k) or "").strip()

        block = [f"【{sid} | {t0:.1f}-{t1:.1f} 秒】"]
        if _g("scene_description_zh"):
            block.append(f"画面内容：{_g('scene_description_zh')}")
        if _g("character_action_detail_zh"):
            block.append(f"人物动作：{_g('character_action_detail_zh')}")
        if _g("face_expression_detail_zh"):
            block.append(f"面部与眼神：{_g('face_expression_detail_zh')}")
        if _g("cloth_hair_reaction_zh"):
            block.append(f"服装与头发：{_g('cloth_hair_reaction_zh')}")
        if _g("environment_detail_zh"):
            block.append(f"场景与空间：{_g('environment_detail_zh')}")
        if _g("weather_force_detail_zh"):
            block.append(f"天气与环境力：{_g('weather_force_detail_zh')}")
        if _g("props_and_tech_detail_zh"):
            block.append(f"道具与科技：{_g('props_and_tech_detail_zh')}")
        if _g("structure_damage_detail_zh"):
            block.append(f"结构损坏：{_g('structure_damage_detail_zh')}")
        if _g("debris_motion_detail_zh"):
            block.append(f"碎片与飞散轨迹：{_g('debris_motion_detail_zh')}")
        if _g("physics_reaction_detail_zh"):
            block.append(f"受力与物理反馈：{_g('physics_reaction_detail_zh')}")
        if _g("fx_detail_zh"):
            block.append(f"特效与粒子：{_g('fx_detail_zh')}")
        if _g("lighting_color_detail_zh"):
            block.append(f"光线与色彩：{_g('lighting_color_detail_zh')}")

        cam_desc = []
        if cam.get("shot_type_zh"):
            cam_desc.append(f"景别：{cam.get('shot_type_zh')}")
        if cam.get("angle_zh"):
            cam_desc.append(f"角度：{cam.get('angle_zh')}")
        if cam.get("movement_zh"):
            cam_desc.append(f"运镜：{cam.get('movement_zh')}")
        if cam.get("composition_zh"):
            cam_desc.append(f"构图：{cam.get('composition_zh')}")
        if cam_desc:
            block.append("机位与运动：" + "；".join(cam_desc))

        if _g("mood_zh"):
            block.append(f"情绪氛围：{_g('mood_zh')}")
        if _g("motion_detail_zh"):
            block.append(f"动作趋势：{_g('motion_detail_zh')}")
        if _g("audio_cue_detail_zh"):
            block.append(f"声音与节奏：{_g('audio_cue_detail_zh')}")
        if _g("edit_rhythm_detail_zh"):
            block.append(f"剪辑与节奏：{_g('edit_rhythm_detail_zh')}")

        if tags:
            block.append("标签：" + " ".join(tags))

        lines.append("\n".join(block))

    return "\n\n".join(lines)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="智谱/GLM：分镜生成 + 视频关键帧分析（HTTP版）", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []  # 本会话历史

st.markdown(
    """
    <div style="
        padding: 16px 18px;
        border-radius: 16px;
        margin-bottom: 14px;
        background: linear-gradient(90deg, #0b5cff 0%, #0a1a3a 60%, #050812 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
    ">
      <div style="font-size: 20px; font-weight: 700; color: white;">
        智谱/GLM：分镜 JSON 生成 + 视频关键帧分析（免SDK纯HTTP）
      </div>
      <div style="margin-top: 6px; color: rgba(255,255,255,0.85); font-size: 13px;">
        解决你遇到的依赖冲突（zhipuai/sniffio/pyjwt），统一走 Z.ai HTTP API。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("API 配置（Z.ai / 智谱）")
    api_key = st.text_input("ZAI_API_KEY（建议放环境变量）", type="password", value=os.getenv("ZAI_API_KEY", ""))
    endpoint = st.text_input("Z.ai Endpoint", value=DEFAULT_ZAI_ENDPOINT)

    st.markdown("---")
    st.caption("模型名可自定义；你截图里用的是 glm-4.5 / glm-4.6v，也可以直接填。")
    model_text = st.text_input("文本模型（分镜/总结/广告）", value="glm-4.5")
    model_vision = st.text_input("视觉模型（帧分析）", value="glm-4v-plus")

    st.markdown("---")
    rpm = st.slider("每分钟最大请求数（限流）", min_value=1, max_value=60, value=10, step=1)
    max_ai_frames = st.slider("本次最多做 AI 分析的帧数", min_value=1, max_value=30, value=10, step=1)
    max_workers = st.slider("并发线程数（建议 2-4）", min_value=1, max_value=8, value=3, step=1)

    st.markdown("---")
    st.caption("分析时间范围（秒）")
    start_sec = st.number_input("从第几秒开始（含）", min_value=0.0, value=0.0, step=0.5)
    end_sec = st.number_input("到第几秒结束（0 表示到结尾）", min_value=0.0, value=0.0, step=0.5)

    if not api_key:
        st.warning("请先填写智谱 API Key（或在部署平台设置环境变量 ZAI_API_KEY）。")
    else:
        st.success("Key 已填写。")

limiter = RateLimiter(rpm=rpm)

tab_storyboard, tab_video, tab_history = st.tabs(["分镜 JSON 生成（文本）", "视频关键帧分析（视觉）", "历史记录（本会话）"])


# ---------- Tab1：分镜 ----------
with tab_storyboard:
    c1, c2 = st.columns([1, 1])
    with c1:
        brand = st.text_input("品牌", value="邵警秘卤")
        product = st.text_input("产品", value="卤鸭脖+卤鸭翅 夜宵套餐")
        duration_sec = st.number_input("时长（秒）", min_value=5, max_value=120, value=15, step=1)
    with c2:
        style = st.text_area("整体风格（中文）", value="烟火气、夜宵档、真实街边风格、有点幽默", height=100)

    if st.button("生成分镜 JSON（走智谱/GLM）", type="primary", key="btn_story"):
        if not api_key:
            st.error("请先填写 ZAI_API_KEY。")
        else:
            with st.spinner("正在调用智谱生成分镜..."):
                try:
                    data = generate_storyboard_zai(api_key, endpoint, model_text, brand, product, int(duration_sec), style)
                except Exception as e:
                    st.error(f"生成失败：{e}")
                else:
                    st.success("生成完成")
                    st.subheader("分镜 JSON")
                    st.json(data)

                    vo = extract_voiceover(data)
                    st.subheader("旁白脚本（可复制）")
                    st.text_area("voiceover", value=vo, height=220)

                    st.download_button(
                        "下载 storyboard.json",
                        data=json.dumps(data, ensure_ascii=False, indent=2),
                        file_name="storyboard.json",
                        mime="application/json",
                    )
                    st.download_button(
                        "下载 voiceover_script.txt",
                        data=vo,
                        file_name="voiceover_script.txt",
                        mime="text/plain",
                    )


# ---------- Tab2：视频 ----------
with tab_video:
    source_mode = st.radio("视频来源", ["上传本地文件", "输入网络视频链接（抖音/B站/TikTok/YouTube）"], index=0)
    video_url: Optional[str] = None
    uploaded_file = None

    if source_mode == "上传本地文件":
        uploaded_file = st.file_uploader("上传视频文件（建议 < 50MB）", type=["mp4", "mov", "m4v", "avi", "mpeg"])
    else:
        video_url = st.text_input("输入视频链接", placeholder="https://v.douyin.com/xxxxxx 或 https://www.youtube.com/watch?v=...")

    if st.button("一键解析整条视频（走智谱/GLM视觉）", type="primary", key="btn_video"):
        if not api_key:
            st.error("请先填写 ZAI_API_KEY。")
            st.stop()

        tmp_path: Optional[str] = None
        source_label = ""
        source_type = ""

        try:
            # 1) 准备视频
            if source_mode == "上传本地文件":
                source_type = "upload"
                if not uploaded_file:
                    st.error("请先上传视频文件。")
                    st.stop()
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                source_label = uploaded_file.name
            else:
                source_type = "url"
                if not video_url:
                    st.error("请输入有效链接。")
                    st.stop()
                st.info("正在下载网络视频...")
                tmp_path = download_video_from_url(video_url)
                source_label = video_url

            if not tmp_path:
                st.error("视频路径异常。")
                st.stop()

            # 2) 抽帧
            st.info("正在抽取关键帧...")
            images, duration, used_range = extract_keyframes_dynamic(
                tmp_path,
                start_sec=float(start_sec),
                end_sec=(float(end_sec) if float(end_sec) > 0 else None),
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            if not images:
                st.error("无法读取帧，视频可能损坏或格式不支持。")
                st.stop()

            st.success(f"抽取 {len(images)} 帧完成（总长约 {duration:.1f}s，分析区间 {used_range[0]:.1f}-{used_range[1]:.1f}s）")

            # 3) 色卡
            palettes: List[List[Tuple[int, int, int]]] = []
            for im in images:
                try:
                    palettes.append(get_color_palette(im, 5))
                except Exception:
                    palettes.append([])

            # 4) 帧级 AI 分析
            with st.spinner("正在调用智谱视觉模型分析关键帧..."):
                frame_infos = analyze_images_concurrently_zai(
                    api_key=api_key,
                    endpoint=endpoint,
                    vision_model=model_vision,
                    images=images,
                    max_ai_frames=int(max_ai_frames),
                    limiter=limiter,
                    max_workers=int(max_workers),
                )

            # 5) 整体总结 + 广告旁白 + 时间轴
            with st.spinner("生成整体总结..."):
                overall = analyze_overall_video_zai(api_key, endpoint, model_text, frame_infos)
            with st.spinner("生成 10 秒广告旁白..."):
                ad_script = generate_ad_script_zai(api_key, endpoint, model_text, frame_infos)
            timeline = generate_timeline_shotlist(frame_infos, used_range=used_range)

            # 6) 导出数据
            export_frames = []
            for info, pal in zip(frame_infos, palettes):
                export_frames.append({
                    **info,
                    "palette_rgb": [list(c) for c in (pal or [])],
                    "palette_hex": [rgb_to_hex(c) for c in (pal or [])],
                })

            export_data = {
                "meta": {
                    "endpoint": endpoint,
                    "text_model": model_text,
                    "vision_model": model_vision,
                    "frame_count": len(images),
                    "max_ai_frames_this_run": int(max_ai_frames),
                    "duration_sec_est": float(duration),
                    "start_sec_used": float(used_range[0]),
                    "end_sec_used": float(used_range[1]),
                    "source_type": source_type,
                    "source_label": source_label,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "frames": export_frames,
                "overall_analysis": overall,
                "ad_script_10s": ad_script,
                "timeline_shotlist_zh": timeline,
            }

            # 7) 写入历史
            history = st.session_state["analysis_history"]
            run_id = f"run_{len(history) + 1}"
            history.append({"id": run_id, "created_at": export_data["meta"]["created_at"], "data": export_data})
            st.session_state["analysis_history"] = history

            # 8) 展示
            st.markdown("---")
            st.subheader("关键帧与提示词")

            for i, (img, info, pal) in enumerate(zip(images, frame_infos, palettes)):
                st.markdown(f"### 关键帧 {i+1}")
                c1, c2 = st.columns([1.2, 2])
                with c1:
                    st.image(img, width=DISPLAY_IMAGE_WIDTH)
                    pal_img = make_palette_image(pal)
                    st.image(pal_img, width=PALETTE_WIDTH)
                    if pal:
                        st.caption("HEX: " + ", ".join(rgb_to_hex(c) for c in pal))
                with c2:
                    cam = info.get("camera", {}) or {}
                    tags = info.get("tags_zh", []) or []
                    st.code(
                        "\n".join([
                            f"【景别】{cam.get('shot_type_zh','')}",
                            f"【运镜】{cam.get('movement_zh','')}",
                            f"【角度】{cam.get('angle_zh','')}",
                            f"【构图】{cam.get('composition_zh','')}",
                            f"【色彩与光影】{info.get('color_and_light_zh','')}",
                            f"【画面内容】{info.get('scene_description_zh','')}",
                            f"【情绪氛围】{info.get('mood_zh','')}",
                            f"【标签】{' '.join(tags)}",
                        ]),
                        language="markdown",
                    )
                    st.markdown("**人物动作细节**")
                    st.code(info.get("character_action_detail_zh","") or "（无）", language="markdown")
                    st.markdown("**场景细节**")
                    st.code(info.get("environment_detail_zh","") or "（无）", language="markdown")
                    st.markdown("**SORA/VEO 视频提示词（英文）**")
                    st.code(info.get("video_prompt_en","") or "（无）", language="markdown")

                st.markdown("---")

            st.subheader("整体总结 / 广告旁白 / 时间轴分镜")
            st.code(overall, language="markdown")
            st.code(ad_script, language="markdown")
            st.code(timeline, language="markdown")

            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button("下载本次 video_analysis.json", data=json_str, file_name="video_analysis.json", mime="application/json")

        except Exception as e:
            st.error(f"解析失败：{e}")


# ---------- Tab3：历史 ----------
with tab_history:
    st.subheader("历史记录（刷新页面会清空）")
    history = st.session_state.get("analysis_history", [])
    if not history:
        st.info("暂无历史记录。")
    else:
        options = [
            f"{i+1}. {h['created_at']} | {h['data']['meta'].get('source_label','')}"
            for i, h in enumerate(history)
        ]
        idx = st.selectbox("选择一条记录", list(range(len(history))), format_func=lambda i: options[i])
        selected = history[idx]["data"]
        st.json(selected["meta"])
        st.download_button(
            "下载该历史记录 JSON",
            data=json.dumps(selected, ensure_ascii=False, indent=2),
            file_name=f"video_analysis_{history[idx]['id']}.json",
            mime="application/json",
        )

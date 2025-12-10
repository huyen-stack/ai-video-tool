import os
import json
import time
import base64
import hmac
import hashlib
import tempfile
import concurrent.futures
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yt_dlp  # 抖音/B站/TikTok/YouTube 下载


# ========================
# 全局配置（智谱）
# ========================

# BigModel ChatCompletions API（v4）
DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 视觉模型：用于「prompt + image」分析（建议用 4V / Vision 类模型名）
DEFAULT_VISION_MODEL = "glm-4v"

# 文本模型：用于「整体总结/广告文案」（可用同一个，也可分开）
DEFAULT_TEXT_MODEL = "glm-4.6"

# 你原先的免费 RPM 限制是 Gemini 的；智谱的配额因账号而异。
# 这里保留一个“自我节流”的参数，避免并发把接口打爆：
DEFAULT_MAX_RPM = 30
DEFAULT_MAX_CONCURRENT = 2

DISPLAY_IMAGE_WIDTH = 320
PALETTE_WIDTH = 320
PALETTE_HEIGHT = 26


# ========================
# Session State
# ========================
if "zhipu_api_key" not in st.session_state:
    st.session_state["zhipu_api_key"] = os.getenv("ZHIPU_API_KEY", "")
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []


# ========================
# JWT（可选）生成：不依赖 pyjwt
# 某些智谱 key 是 id.secret 形式，若直接 Bearer 不行可用 JWT 模式
# ========================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def make_jwt_from_id_secret(api_key: str, exp_seconds: int = 60) -> str:
    if "." not in api_key:
        raise ValueError("JWT 模式需要 api_key 为 {id}.{secret} 格式。")
    kid, secret = api_key.split(".", 1)

    header = {"alg": "HS256", "sign_type": "SIGN"}
    now_ms = int(time.time() * 1000)
    payload = {
        "api_key": kid,
        "exp": now_ms + exp_seconds * 1000,
        "timestamp": now_ms,
    }

    header_b64 = _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def build_auth_header(raw_key: str, auth_mode: str) -> str:
    raw_key = (raw_key or "").strip()
    if not raw_key:
        raise ValueError("请先填写 ZHIPU_API_KEY（智谱 API Key）。")

    if auth_mode == "直接 API Key（推荐）":
        return f"Bearer {raw_key}"

    if auth_mode == "JWT（id.secret）":
        token = make_jwt_from_id_secret(raw_key)
        return f"Bearer {token}"

    return f"Bearer {raw_key}"


# ========================
# 智谱调用（文本 / 图文）
# ========================
_last_call_ts = 0.0
_call_lock = concurrent.futures.thread.Lock() if hasattr(concurrent.futures, "thread") else None
_semaphore = None  # runtime set


def _throttle(max_rpm: int):
    """简单节流：按 max_rpm 控制最小间隔。"""
    global _last_call_ts
    if max_rpm <= 0:
        return
    min_interval = 60.0 / float(max_rpm)
    now = time.time()
    wait = (_last_call_ts + min_interval) - now
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()


def _extract_content_from_bigmodel(resp_json: Dict[str, Any]) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False)


def bigmodel_text(
    base_url: str,
    api_key: str,
    auth_mode: str,
    model: str,
    prompt: str,
    max_rpm: int,
    timeout_sec: int = 120,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    global _semaphore
    if _semaphore is None:
        _semaphore = concurrent.futures.thread.Semaphore(DEFAULT_MAX_CONCURRENT) if hasattr(concurrent.futures, "thread") else None

    auth = build_auth_header(api_key, auth_mode)
    headers = {"Authorization": auth, "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

    # 兜底：无 semaphore 环境也能跑
    if _semaphore is not None:
        with _semaphore:
            _throttle(max_rpm)
            r = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)
    else:
        _throttle(max_rpm)
        r = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)

    if r.status_code != 200:
        try:
            raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(r.json(), ensure_ascii=False)}")
        except Exception:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

    return _extract_content_from_bigmodel(r.json())


def bigmodel_vision(
    base_url: str,
    api_key: str,
    auth_mode: str,
    model: str,
    prompt: str,
    img: Image.Image,
    max_rpm: int,
    timeout_sec: int = 180,
    temperature: float = 0.4,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """图文多模态：把 PIL 图片编码为 data URL 传入。"""
    global _semaphore
    if _semaphore is None:
        _semaphore = concurrent.futures.thread.Semaphore(DEFAULT_MAX_CONCURRENT) if hasattr(concurrent.futures, "thread") else None

    auth = build_auth_header(api_key, auth_mode)
    headers = {"Authorization": auth, "Content-Type": "application/json"}

    # PIL -> PNG base64
    buf = tempfile.SpooledTemporaryFile()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    # BigModel v4 多模态消息（常见格式：content 为数组，含 text + image_url）
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

    if _semaphore is not None:
        with _semaphore:
            _throttle(max_rpm)
            r = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)
    else:
        _throttle(max_rpm)
        r = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)

    if r.status_code != 200:
        try:
            raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(r.json(), ensure_ascii=False)}")
        except Exception:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

    return _extract_content_from_bigmodel(r.json())


# ========================
# 页面 / 全局样式
# ========================
st.set_page_config(
    page_title="AI 自动关键帧分镜 & 视频提示词助手（智谱版）",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #0f172a; color: #e5e7eb; }
    .stMarkdown, .stText { color: #e5e7eb; }
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
        🎬 AI 自动关键帧分镜助手 Pro ·（智谱 BigModel 版）
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
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, flags)
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
# 单帧分析：结构化 JSON + MJ 提示词 + 视频提示词（智谱图文）
# ========================
def analyze_single_image(
    img: Image.Image,
    index: int,
    zhipu_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        prompt = f"""
你现在是电影导演 + 摄影指导 + 服化道总监 + 提示词工程师。
请仔细分析给你的这一帧画面，并输出一个 JSON 对象，用于：
1）人类导演阅读分镜
2）Midjourney 生成分镜图
3）SORA/VEO 等视频模型生成对应镜头

必须使用下面这些 key（英文），value 大部分为中文说明，英文提示词字段为英文：

{{
  "index": {index},
  "scene_description_zh": "用 1～3 句完整中文，把当前画面描述得尽量具体（忽略 UI 元素）",
  "tags_zh": ["#短中文标签1", "#标签2"],
  "camera": {{
    "shot_type_zh": "远景/全景/中景/近景/特写",
    "shot_type": "wide shot/full shot/medium shot/close-up",
    "angle_zh": "俯拍/仰拍/平视/侧拍等",
    "angle": "high angle/low angle/eye-level",
    "movement_zh": "推近/跟拍/横移/甩镜等",
    "movement": "slow dolly-in/handheld tracking/pan",
    "composition_zh": "三分法/中心/对称/前景-主体-背景等",
    "composition": "rule-of-thirds/centered/symmetry"
  }},
  "color_and_light_zh": "1-2 句中文描述色调与光线",
  "mood_zh": "中文情绪氛围",
  "characters": [
    {{
      "role_zh": "人物身份",
      "gender_zh": "女性/男性/不明显",
      "age_look_zh": "年龄观感",
      "body_type_zh": "体型",
      "clothing_zh": "服装风格与颜色",
      "hair_zh": "发型与发色",
      "expression_zh": "表情",
      "pose_body_zh": "姿态",
      "props_zh": "道具"
    }}
  ],
  "character_action_detail_zh": "动作细节（头→手→躯干→腿，写具体接触点与方向/速度）",
  "face_expression_detail_zh": "面部与眼神细节（含外力形变回弹若有）",
  "cloth_hair_reaction_zh": "头发与衣物对风/惯性的反应",
  "environment_detail_zh": "前景/中景/背景的空间结构与材质",
  "weather_force_detail_zh": "风雨雪/冲击波等环境力细节（无则写无明显）",
  "props_and_tech_detail_zh": "关键道具/科技元素（位置/外观/状态）",
  "physics_reaction_detail_zh": "受力与形变回弹过程",
  "structure_damage_detail_zh": "结构损坏（哪部分怎样破损）",
  "debris_motion_detail_zh": "碎片飞散轨迹（无则写无明显）",
  "motion_detail_zh": "上一瞬间→当前→下一瞬间动作推断",
  "fx_detail_zh": "烟尘/火花/能量粒子等（无可空）",
  "lighting_color_detail_zh": "更精细光源方向/色温差/频闪等",
  "audio_cue_detail_zh": "声音设计（环境声/特效声/BGM）",
  "edit_rhythm_detail_zh": "剪辑节奏（慢动作/闪白/甩镜等）",
  "midjourney_prompt": "一行英文 MJ v6 提示词",
  "midjourney_negative_prompt": "一行英文负面提示词",
  "video_prompt_en": "3-5 句英文视频提示词，最后一句写 4 second shot, vertical 9:16, 24fps, cinematic, highly detailed."
}}

要求：
1) 只输出一个 JSON 对象，不要任何解释或额外文字；
2) 全部双引号；无注释；无多余逗号。
""".strip()

        text = bigmodel_vision(
            base_url=zhipu_cfg["base_url"],
            api_key=zhipu_cfg["api_key"],
            auth_mode=zhipu_cfg["auth_mode"],
            model=zhipu_cfg["vision_model"],
            prompt=prompt,
            img=img,
            max_rpm=zhipu_cfg["max_rpm"],
            temperature=0.35,
            max_tokens=4096,
        )

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        info = json.loads(text[start : end + 1])

        # 兜底字段
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
    images: List[Image.Image],
    max_ai_frames: int,
    zhipu_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n = len(images)
    if n == 0:
        return []

    use_n = min(max_ai_frames, n)
    results: List[Dict[str, Any]] = [None] * n  # type: ignore

    status = st.empty()
    status.info(f"⚡ 正在对前 {use_n} 帧进行 AI 分析（共 {n} 帧），其余帧保留截图与色卡。")

    # 并发不要开太大，避免触发限流；这里最多 4
    workers = min(use_n, 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(analyze_single_image, images[i], i + 1, zhipu_cfg): i
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
            "scene_description_zh": "（本帧未做 AI 分析，用于节省当前配额，但仍可用于视觉参考和色卡。）",
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
# 整体视频层面的总结（智谱文本）
# ========================
def analyze_overall_video(frame_infos: List[Dict[str, Any]], zhipu_cfg: Dict[str, Any]) -> str:
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
整体风险级别：低 / 中 / 高
并用 2-3 句话说明需要注意的点。

只输出以上 4 个小节，不要添加额外说明。
""".strip()

    try:
        return bigmodel_text(
            base_url=zhipu_cfg["base_url"],
            api_key=zhipu_cfg["api_key"],
            auth_mode=zhipu_cfg["auth_mode"],
            model=zhipu_cfg["text_model"],
            prompt=prompt,
            max_rpm=zhipu_cfg["max_rpm"],
            temperature=0.5,
            max_tokens=2048,
        )
    except Exception as e:
        return f"整体分析失败：{e}"


# ========================
# 10 秒广告旁白脚本生成（智谱文本）
# ========================
def generate_ad_script(frame_infos: List[Dict[str, Any]], zhipu_cfg: Dict[str, Any]) -> str:
    described = [
        info for info in frame_infos
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
        parts.append(f"第 {idx} 帧：{info.get('scene_description_zh', '')}；标签：{'、'.join(tags)}")
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
""".strip()

    try:
        return bigmodel_text(
            base_url=zhipu_cfg["base_url"],
            api_key=zhipu_cfg["api_key"],
            auth_mode=zhipu_cfg["auth_mode"],
            model=zhipu_cfg["text_model"],
            prompt=prompt,
            max_rpm=zhipu_cfg["max_rpm"],
            temperature=0.7,
            max_tokens=1024,
        )
    except Exception as e:
        return f"广告文案生成失败：{e}"


# ========================
# 时间轴分镜脚本生成（纯拼接，不调用 AI）
# ========================
def generate_timeline_shotlist(
    frame_infos: List[Dict[str, Any]],
    used_range: Tuple[float, float],
) -> str:
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

        def _s(k): return (info.get(k) or "").strip()

        scene = _s("scene_description_zh")
        char_act = _s("character_action_detail_zh")
        env = _s("environment_detail_zh")
        props = _s("props_and_tech_detail_zh")
        motion = _s("motion_detail_zh")
        mood = _s("mood_zh")

        face = _s("face_expression_detail_zh")
        cloth_hair = _s("cloth_hair_reaction_zh")
        weather = _s("weather_force_detail_zh")
        physics = _s("physics_reaction_detail_zh")
        structure_damage = _s("structure_damage_detail_zh")
        debris_motion = _s("debris_motion_detail_zh")
        fx = _s("fx_detail_zh")
        lighting = _s("lighting_color_detail_zh")
        audio = _s("audio_cue_detail_zh")
        edit = _s("edit_rhythm_detail_zh")

        shot_type = cam.get("shot_type_zh", "")
        angle = cam.get("angle_zh", "")
        movement = cam.get("movement_zh", "")
        composition = cam.get("composition_zh", "")

        block_lines: List[str] = []
        block_lines.append(f"【{shot_id} | {t0:.1f}-{t1:.1f} 秒】")

        if scene: block_lines.append(f"画面内容：{scene}")
        if char_act: block_lines.append(f"人物动作：{char_act}")
        if face: block_lines.append(f"面部与眼神：{face}")
        if cloth_hair: block_lines.append(f"服装与头发：{cloth_hair}")

        if env: block_lines.append(f"场景与空间：{env}")
        if weather: block_lines.append(f"天气与环境力：{weather}")

        if props: block_lines.append(f"道具与科技：{props}")
        if structure_damage: block_lines.append(f"结构损坏：{structure_damage}")
        if debris_motion: block_lines.append(f"碎片与飞散轨迹：{debris_motion}")
        if physics: block_lines.append(f"受力与物理反馈：{physics}")

        if fx: block_lines.append(f"特效与粒子：{fx}")
        if lighting: block_lines.append(f"光线与色彩：{lighting}")

        cam_desc_parts = []
        if shot_type: cam_desc_parts.append(f"景别：{shot_type}")
        if angle: cam_desc_parts.append(f"角度：{angle}")
        if movement: cam_desc_parts.append(f"运镜：{movement}")
        if composition: cam_desc_parts.append(f"构图：{composition}")
        if cam_desc_parts:
            block_lines.append("机位与运动：" + "；".join(cam_desc_parts))

        if mood: block_lines.append(f"情绪氛围：{mood}")
        if motion: block_lines.append(f"动作趋势：{motion}")

        if audio: block_lines.append(f"声音与节奏：{audio}")
        if edit: block_lines.append(f"剪辑与节奏：{edit}")

        if tags: block_lines.append("标签：" + " ".join(tags))
        lines.append("\n".join(block_lines))

    return "\n\n".join(lines)


# ========================
# 侧边栏：智谱配置
# ========================
with st.sidebar:
    st.header("🔑 第一步：配置智谱 BigModel")

    zhipu_key = st.text_input(
        "ZHIPU_API_KEY",
        type="password",
        value=st.session_state["zhipu_api_key"],
        help="建议在部署环境变量里配置 ZHIPU_API_KEY",
    )
    st.session_state["zhipu_api_key"] = zhipu_key

    auth_mode = st.selectbox("鉴权方式", ["直接 API Key（推荐）", "JWT（id.secret）"], index=0)
    base_url = st.text_input("智谱接口地址", value=DEFAULT_ZHIPU_BASE_URL)
    vision_model = st.text_input("视觉模型（图+文）", value=DEFAULT_VISION_MODEL)
    text_model = st.text_input("文本模型（总结/文案）", value=DEFAULT_TEXT_MODEL)

    st.markdown("---")
    max_rpm = st.slider("最大请求速率（自我节流 RPM）", 1, 120, DEFAULT_MAX_RPM, 1)
    max_concurrent = st.slider("最大并发（建议 1-3）", 1, 6, DEFAULT_MAX_CONCURRENT, 1)

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

    if not zhipu_key:
        st.warning("🔴 还没有 Key：请去智谱开放平台创建 API Key，并配置到环境变量 ZHIPU_API_KEY")
    else:
        st.success("🟢 Key 已就绪")


# 把并发控制同步到 semaphore
try:
    import threading
    _semaphore = threading.Semaphore(int(max_concurrent))
except Exception:
    _semaphore = None

zhipu_cfg = {
    "api_key": zhipu_key,
    "auth_mode": auth_mode,
    "base_url": (base_url or "").strip(),
    "vision_model": (vision_model or "").strip(),
    "text_model": (text_model or "").strip(),
    "max_rpm": int(max_rpm),
}


# ========================
# 主流程：上传/链接 选择 + 抽帧 + 分析 + 展示
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
    if not zhipu_cfg["api_key"]:
        st.error("请先在左侧输入有效的智谱 API Key。")
    else:
        tmp_path: Optional[str] = None
        source_label = ""
        source_type = ""

        try:
            # 1) 准备视频路径
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
                st.stop()

            # 2) 抽帧（带时间区间）
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

            # 3) 主色调
            frame_palettes: List[List[Tuple[int, int, int]]] = []
            for img in images:
                try:
                    palette_colors = get_color_palette(img, num_colors=5)
                except Exception:
                    palette_colors = []
                frame_palettes.append(palette_colors)

            # 4) 帧级分析
            with st.spinner("🧠 正在为关键帧生成结构化分析 + MJ 提示词 + 视频提示词..."):
                frame_infos = analyze_images_concurrently(
                    images, max_ai_frames=max_ai_frames, zhipu_cfg=zhipu_cfg
                )

            # 5) 整体分析 + 广告文案 + 时间轴分镜（时间轴为纯拼接）
            with st.spinner("📚 正在生成整段视频的剧情大纲与话题标签..."):
                overall = analyze_overall_video(frame_infos, zhipu_cfg)
            with st.spinner("🎤 正在生成 10 秒广告旁白脚本..."):
                ad_script = generate_ad_script(frame_infos, zhipu_cfg)
            with st.spinner("🎬 正在生成时间轴分镜脚本（纯拼接版）..."):
                timeline_shotlist = generate_timeline_shotlist(frame_infos, used_range=used_range)

            # 6) 组装 export_data + 写入历史记录
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
                    "provider": "zhipu_bigmodel",
                    "vision_model": zhipu_cfg["vision_model"],
                    "text_model": zhipu_cfg["text_model"],
                    "frame_count": len(images),
                    "max_ai_frames_this_run": min(max_ai_frames, len(images)),
                    "duration_sec_est": duration,
                    "start_sec_used": start_used,
                    "end_sec_used": end_used,
                    "source_type": source_type,
                    "source_label": source_label,
                    "base_url": zhipu_cfg["base_url"],
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

            # 7) Tabs 展示
            tab_frames, tab_story, tab_json, tab_history = st.tabs(
                ["🎞 关键帧 & 提示词", "📚 剧情总结 & 广告旁白 & 时间轴分镜", "📦 JSON 导出（本次）", "🕘 历史记录（本会话）"]
            )

            with tab_frames:
                st.markdown(f"共抽取 **{len(images)}** 个关键帧，其中前 **{min(len(images), max_ai_frames)}** 帧做了 AI 分析。")
                st.markdown("---")

                for i, (img, info, palette) in enumerate(zip(images, frame_infos, frame_palettes)):
                    with st.container():
                        st.markdown(f"### 📘 关键帧 {i + 1}")
                        c1, c2 = st.columns([1.2, 2])

                        with c1:
                            st.image(img, caption=f"第 {i + 1} 帧画面", width=DISPLAY_IMAGE_WIDTH)
                            palette_img = make_palette_image(palette)
                            st.image(palette_img, caption="主色调色卡", width=PALETTE_WIDTH)
                            if palette:
                                st.caption("主色 HEX：" + ", ".join(rgb_to_hex(c) for c in palette))

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
                            st.markdown("**分镜分析（可复制）：**")
                            st.code("\n".join([x for x in analysis_lines if x.strip()]), language="markdown")

                            st.markdown("**人物动作细节（可复制）：**")
                            st.code(info.get("character_action_detail_zh") or "（暂无动作细节）", language="markdown")

                            st.markdown("**场景细节（可复制）：**")
                            scene_detail = info.get("environment_detail_zh", "")
                            props_detail = info.get("props_and_tech_detail_zh", "")
                            scene_text = (scene_detail + "\n\n道具与科技元素：" + props_detail).strip()
                            st.code(scene_text or "（暂无场景细节）", language="markdown")

                            advanced = []
                            for k, title in [
                                ("face_expression_detail_zh", "面部与眼神"),
                                ("cloth_hair_reaction_zh", "服装与头发"),
                                ("weather_force_detail_zh", "天气与环境力"),
                                ("physics_reaction_detail_zh", "受力与物理反馈"),
                                ("structure_damage_detail_zh", "结构损坏"),
                                ("debris_motion_detail_zh", "碎片飞散"),
                                ("fx_detail_zh", "特效与粒子"),
                                ("lighting_color_detail_zh", "光线细节"),
                                ("audio_cue_detail_zh", "声音与节奏"),
                                ("edit_rhythm_detail_zh", "剪辑节奏"),
                            ]:
                                v = info.get(k) or ""
                                if v.strip():
                                    advanced.append(f"{v.strip()}")
                            if advanced:
                                st.markdown("**高级物理 / 环境细节（可复制）：**")
                                st.code("\n".join(advanced), language="markdown")

                            st.markdown("**SORA / VEO 视频提示词（英文，可复制）：**")
                            st.code(info.get("video_prompt_en") or "（暂无视频提示词）", language="markdown")

                            st.markdown("**Midjourney 静帧提示词（可选）：**")
                            st.code(info.get("midjourney_prompt") or "（暂无 Midjourney 提示词）", language="markdown")

                        st.markdown("---")

            with tab_story:
                st.markdown("### 📚 整体剧情与视听风格总结")
                st.code(overall, language="markdown")
                st.markdown("### 🎤 10 秒广告旁白脚本")
                st.code(ad_script, language="markdown")
                st.markdown("### 🎬 时间轴分镜脚本（可复制）")
                st.code(timeline_shotlist, language="markdown")

            with tab_json:
                st.markdown("### 📦 下载本次分析的 JSON 文件")
                st.download_button(
                    label="⬇️ 下载本次 video_analysis.json",
                    data=json_str,
                    file_name="video_analysis.json",
                    mime="application/json",
                )
                with st.expander("🔍 预览部分 JSON 内容"):
                    preview = json_str[:3000] + ("\n...\n" if len(json_str) > 3000 else "")
                    st.code(preview, language="json")

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
                        f"**视觉模型：** {selected['meta'].get('vision_model','')}  \n"
                        f"**文本模型：** {selected['meta'].get('text_model','')}"
                    )

                    hist_json = json.dumps(selected["data"], ensure_ascii=False, indent=2)
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

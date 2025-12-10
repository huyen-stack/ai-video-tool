
import base64
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st

# Optional deps (guarded)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:
    YoutubeDL = None  # type: ignore

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class RateLimiter:
    """Simple per-minute limiter. acquire() blocks until allowed."""
    rpm: int
    _lock: threading.Lock = threading.Lock()
    _window_start_ms: int = 0
    _count: int = 0

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = _now_ms()
                if self._window_start_ms == 0:
                    self._window_start_ms = now
                    self._count = 0
                elapsed = now - self._window_start_ms
                if elapsed >= 60_000:
                    self._window_start_ms = now
                    self._count = 0
                if self._count < self.rpm:
                    self._count += 1
                    return
                wait_ms = max(0, 60_000 - elapsed)
            time.sleep(min(1.0, wait_ms / 1000.0))


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def b64_jpeg_from_rgb(rgb: np.ndarray, quality: int = 85) -> str:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless 未安装，无法编码图片。")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败。")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def resize_max_width(rgb: np.ndarray, max_w: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w <= max_w:
        return rgb
    scale = max_w / float(w)
    new_w = max_w
    new_h = max(1, int(round(h * scale)))
    if cv2 is None:
        ys = (np.linspace(0, h - 1, new_h)).astype(int)
        xs = (np.linspace(0, w - 1, new_w)).astype(int)
        return rgb[ys][:, xs]
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def download_video_to_temp(url: str) -> str:
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp 未安装，无法下载链接视频。")
    tmp_dir = tempfile.mkdtemp(prefix="video_dl_")
    outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
    ydl_opts = {"outtmpl": outtmpl, "quiet": True, "noplaylist": True, "format": "mp4/best"}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
    return filepath


def extract_frames(
    video_path: str,
    start_s: float,
    end_s: float,
    target_frames: int,
    max_w: int,
    jpeg_quality: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless 未安装，无法抽帧。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0:
        fps = 25.0

    duration_s = (total_frames / fps) if total_frames > 0 else 0.0

    start_s = max(0.0, float(start_s))
    end_s = float(end_s)
    if end_s <= 0.0 or end_s > duration_s:
        end_s = duration_s if duration_s > 0 else max(start_s, 0.0)
    if end_s < start_s:
        start_s, end_s = end_s, start_s

    start_idx = int(math.floor(start_s * fps))
    end_idx = int(math.floor(end_s * fps))
    start_idx = max(0, min(start_idx, max(0, total_frames - 1)))
    end_idx = max(0, min(end_idx, max(0, total_frames - 1)))
    if end_idx < start_idx:
        end_idx = start_idx

    span = end_idx - start_idx + 1
    if span <= 0:
        raise RuntimeError("时间范围无有效帧。")

    n = max(1, int(target_frames))
    n = min(n, span)

    if n == 1:
        sample_idxs = [start_idx + span // 2]
    else:
        sample_idxs = [start_idx + int(round(i * (span - 1) / (n - 1))) for i in range(n)]
        sample_idxs = sorted(set(sample_idxs))
        while len(sample_idxs) < n and sample_idxs[-1] < end_idx:
            sample_idxs.append(sample_idxs[-1] + 1)

    frames: List[Dict[str, Any]] = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_max_width(rgb, max_w=max_w)
        b64 = b64_jpeg_from_rgb(rgb, quality=jpeg_quality)
        ts = float(idx) / float(fps)
        frames.append({"frame_index": int(idx), "timestamp_s": round(ts, 3), "image_b64": b64})

    cap.release()

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3),
        "range": {"start_s": round(start_s, 3), "end_s": round(end_s, 3)},
        "extracted": len(frames),
    }
    return frames, meta


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    timeout_s: int = 120,
) -> Tuple[Optional[str], Dict[str, Any]]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    url = base_url.strip()
    if not url.endswith("/chat/completions"):
        if url.endswith("/v1"):
            url = url + "/chat/completions"
        elif url.endswith("/v1/"):
            url = url + "chat/completions"
        else:
            url = url.rstrip("/") + "/v1/chat/completions"

    payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": float(temperature)}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        raw = resp.text
        try:
            data = resp.json()
        except Exception:
            data = {"raw": raw}
        if resp.status_code >= 400:
            return None, {"status": resp.status_code, "url": url, "response": data}

        content = None
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    content = msg.get("content")
        return content, {"status": resp.status_code, "url": url, "response": data}
    except Exception as e:
        return None, {"status": -1, "url": url, "error": repr(e)}


def vision_analyze_frame(
    base_url: str,
    api_key: str,
    model_vision: str,
    frame: Dict[str, Any],
    limiter: RateLimiter,
    prompt: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    limiter.acquire()
    b64 = frame["image_b64"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ]

    content, debug = call_chat_completion(base_url, api_key, model_vision, messages, temperature=temperature, timeout_s=120)

    if content is None and isinstance(debug, dict) and int(debug.get("status", 0)) >= 400:
        # fallback: send base64 in text (truncated)
        limiter.acquire()
        messages2 = [
            {
                "role": "user",
                "content": prompt
                + "\n\n下面是一张 JPEG 图片 base64（不含 data: 前缀），请基于图片内容回答：\n"
                + b64[:8000],
            }
        ]
        content2, debug2 = call_chat_completion(base_url, api_key, model_vision, messages2, temperature=temperature, timeout_s=120)
        if content2 is not None:
            content = content2
            debug = {"fallback_used": True, **debug2}

    return {
        "frame_index": frame["frame_index"],
        "timestamp_s": frame["timestamp_s"],
        "analysis_text": content,
        "debug": debug if content is None else None,
    }


def build_frame_prompt(detail_level: str) -> str:
    base = (
        "你是一位资深短视频分镜分析师。请根据图片内容，输出严格 JSON（不要任何多余解释）。\n"
        "字段：scene, shot, camera, lighting_color, subjects, action, emotion, props, environment, tags(数组).\n"
        "要求：中文为主，尽量具体，避免“可能/大概”。\n"
    )
    if detail_level == "极简":
        base += "尽量简短，每个字段 1-2 句话。\n"
    elif detail_level == "详细":
        base += "尽量详细，action/emotion/environment/camera 要写清楚细节。\n"
    else:
        base += "适中详细。\n"
    base += "输出示例：{\"scene\":\"...\",\"shot\":\"...\"}\n"
    return base


def try_parse_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    obj = safe_json_loads(text.strip())
    if isinstance(obj, dict):
        return obj
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        obj2 = safe_json_loads(candidate)
        if isinstance(obj2, dict):
            return obj2
    return None


def normalize_frame_result(res: Dict[str, Any]) -> Dict[str, Any]:
    parsed = try_parse_json_from_text(res.get("analysis_text"))
    if parsed:
        return {"frame_index": res["frame_index"], "timestamp_s": res["timestamp_s"], **parsed}
    return res


st.set_page_config(page_title="AI 视频关键帧分析 (免SDK)", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:1.2rem;padding-bottom:3rem;}
.card{border:1px solid #eee;border-radius:14px;padding:14px;background:#fff;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("AI 视频关键帧分析（免 SDK / 纯 HTTP）")

with st.sidebar:
    st.subheader("API 配置")
    api_key = st.text_input("ZAI_API_KEY（或环境变量）", value=os.getenv("ZAI_API_KEY", ""), type="password")
    base_url = st.text_input("Base URL", value=os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"))
    model_text = st.text_input("文本模型", value=os.getenv("ZAI_TEXT_MODEL", "glm-4.5"))
    model_vision = st.text_input("视觉模型", value=os.getenv("ZAI_VISION_MODEL", "glm-4.6v"))

    st.divider()
    st.subheader("性能/限流")
    rpm = st.slider("每分钟最大调用次数", 0, 120, 30, 1)
    workers = st.slider("并发线程数", 1, 8, 3, 1)
    max_frames = st.slider("最多分析帧数", 1, 60, 24, 1)

    st.divider()
    st.subheader("抽帧参数")
    max_w = st.slider("图片最大宽度", 320, 1280, 768, 32)
    jpeg_quality = st.slider("JPEG 质量", 50, 95, 85, 1)
    detail_level = st.selectbox("分析详细程度", ["适中", "详细", "极简"], index=0)

tab1, tab2 = st.tabs(["视频关键帧分析（视觉）", "分镜 JSON 生成（文本）"])

with tab1:
    st.markdown("#### 1) 选择视频来源")
    src = st.radio("视频来源", ["上传本地文件", "输入视频链接"], horizontal=True)

    video_path: Optional[str] = None

    if src == "上传本地文件":
        up = st.file_uploader("上传视频文件", type=["mp4", "mov", "m4v", "avi", "mkv"])
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read())
            tmp.flush()
            tmp.close()
            video_path = tmp.name
    else:
        url = st.text_input("粘贴视频链接")
        if url and st.button("下载链接视频到临时文件"):
            with st.spinner("正在下载..."):
                try:
                    video_path = download_video_to_temp(url)
                    st.success("下载完成")
                except Exception as e:
                    st.error(str(e))

    st.markdown("#### 2) 时间范围（秒）")
    c1, c2 = st.columns(2)
    with c1:
        start_s = st.number_input("开始", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    with c2:
        end_s = st.number_input("结束（0=到结尾）", min_value=0.0, value=0.0, step=0.1, format="%.2f")

    run = st.button("一键抽关键帧并分析", type="primary", disabled=(video_path is None))

    if run:
        if video_path is None:
            st.error("请先上传或下载视频。")
        with st.spinner("抽取关键帧中..."):
            try:
                frames, meta = extract_frames(video_path, float(start_s), float(end_s), int(max_frames), int(max_w), int(jpeg_quality))
            except Exception as e:
                st.error(f"抽帧失败：{e}")
                frames, meta = [], {}

        if meta:
            st.success(f"抽帧完成：{meta.get('extracted')} 帧（总时长 {meta.get('duration_s')}s）")

        if not api_key:
            st.warning("未填写 API Key：仅展示抽帧结果，不做模型分析。")
            st.download_button("下载抽帧 JSON（含 base64）", data=json.dumps({"meta": meta, "frames": frames}, ensure_ascii=False), file_name="frames_raw.json")
        else:
            limiter = RateLimiter(rpm=int(rpm))
            prompt = build_frame_prompt(detail_level=detail_level)

            prog = st.progress(0.0, text="分析中...")
            results: List[Dict[str, Any]] = []
            errors = 0

            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                futs = [
                    ex.submit(vision_analyze_frame, base_url, api_key, model_vision, f, limiter, prompt, 0.2)
                    for f in frames
                ]
                total = len(futs)
                done = 0
                for fut in as_completed(futs):
                    res = fut.result()
                    if res.get("analysis_text") is None:
                        errors += 1
                    results.append(res)
                    done += 1
                    prog.progress(done / max(1, total), text=f"分析中... {done}/{total}（失败 {errors}）")

            prog.empty()
            results = sorted(results, key=lambda x: float(x.get("timestamp_s", 0.0)))
            normalized = [normalize_frame_result(r) for r in results]
            out = {"meta": meta, "frames": normalized}

            st.markdown("#### 3) 结果预览")
            for i, r in enumerate(normalized[: min(20, len(normalized))], start=1):
                with st.expander(f"关键帧 {i} | t={r.get('timestamp_s')}s | idx={r.get('frame_index')}", expanded=(i <= 2)):
                    b64 = None
                    for f in frames:
                        if f["frame_index"] == r.get("frame_index"):
                            b64 = f["image_b64"]
                            break
                    if b64:
                        st.image(base64.b64decode(b64), caption=f"t={r.get('timestamp_s')}s", width=360)
                    st.json(r)

            st.download_button("下载分析结果 JSON", data=json.dumps(out, ensure_ascii=False), file_name="video_keyframes_analysis.json")
            if errors > 0:
                st.warning(f"有 {errors} 帧分析失败。请展开失败帧查看 debug。")

with tab2:
    st.markdown("#### 输入产品/风格 → 生成短视频分镜 JSON（纯文本模型）")
    brand = st.text_input("品牌", value="（可不填）")
    product = st.text_input("产品", value="（可不填）")
    style = st.text_input("整体风格", value="写实、自然光、生活化、电影感")
    duration = st.slider("目标时长（秒）", 5, 60, 15, 1)

    if st.button("生成分镜 JSON", disabled=(not api_key)):
        prompt = f"""
你是一位资深短视频导演与广告文案，请输出严格 JSON（不要任何解释/Markdown）。
生成一个竖版短视频分镜脚本，时长约 {duration} 秒，适配抖音/小红书。
品牌：{brand}
产品：{product}
整体风格：{style}

输出 JSON 顶层字段：brand, product, style, duration_sec, shots
shots: 数组，每个元素包含:
- id, start_s, end_s
- visual（画面内容：人物动作、表情、服装、环境影响）
- camera（机位与运动）
- audio（旁白/对白/环境声/音乐提示）
- text_on_screen（屏幕文字）
- prompt_en（用于 Sora/Veo 的英文提示词）
镜头数 6-10 个，start/end 连贯覆盖全片。
"""
        messages = [{"role": "user", "content": prompt.strip()}]
        with st.spinner("生成中..."):
            content, debug = call_chat_completion(base_url, api_key, model_text, messages, temperature=0.4, timeout_s=120)

        if content is None:
            st.error("生成失败（请看 debug）")
            st.json(debug)
        else:
            parsed = safe_json_loads(content.strip())
            if isinstance(parsed, dict):
                st.success("生成成功（已解析为 JSON）")
                st.json(parsed)
                st.download_button("下载分镜 JSON", data=json.dumps(parsed, ensure_ascii=False), file_name="storyboard.json")
            else:
                st.warning("模型返回不是严格 JSON（已原样展示）")
                st.code(content)
                st.download_button("下载原始文本", data=content, file_name="storyboard_raw.txt")

    if not api_key:
        st.info("请先在左侧填写 API Key（或设置环境变量 ZAI_API_KEY）。")

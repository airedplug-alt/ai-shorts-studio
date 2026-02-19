#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SHORTS STUDIO PRO v1.0.0
===========================
© chally.choi — All Rights Reserved

Commercial-grade YouTube Shorts auto-generation studio
with ComfyUI AI video generation engine.

Target Environment: Windows 11 WSL2 + Ubuntu + RTX 3090
Key Features:
  - ComfyUI + FramePack/Wan2.1 AI video generation
  - Dual scenario generation with scene descriptions
  - Korean -> English prompt auto-translation
  - Platform-specific output format presets
  - Customizable subtitle styles with Auto mode
  - Real-time progress with ETA estimation
  - ComfyUI real-time log streaming in UI
  - One-click start (ComfyUI + Flask unified)
  - Video deletion support
  - Pure FFmpeg pipeline with GPU acceleration
  - edge-tts async with dedicated event loop
  - Ollama LLM local AI
  - Real-time WebSocket UI
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import psutil
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, abort, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from loguru import logger

# ═══════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════
load_dotenv()
BASE_DIR = Path(__file__).parent.resolve()
APP_VERSION = "1.0.0"

# Loguru
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(
    sys.stdout,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)
logger.add(
    LOG_DIR / "app.log",
    rotation="50 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)

# Flask
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)

(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "data" / "shorts.db"

app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", os.urandom(32).hex()),
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{DB_PATH}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True, "pool_recycle": 3600},
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,
    JSON_ENSURE_ASCII=False,
)

CORS(app, origins="*")
sio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    manage_session=False,
)
db = SQLAlchemy(app)
executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("THREAD_POOL_SIZE", "4")),
    thread_name_prefix="shorts-worker",
)


# ═══════════════════════════════════════════════════
#  DB MODELS
# ═══════════════════════════════════════════════════
class Job(db.Model):
    __tablename__ = "jobs"
    id = db.Column(db.String(64), primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    topics = db.Column(db.Text, default="[]")
    options = db.Column(db.Text, default="{}")
    source_urls = db.Column(db.Text, default="[]")
    topic_mode = db.Column(db.String(20), default="auto")
    status = db.Column(db.String(20), default="pending", index=True)
    progress = db.Column(db.Integer, default=0)
    current_step = db.Column(db.String(200))
    message = db.Column(db.String(500))
    output_files = db.Column(db.Text, default="[]")
    error_message = db.Column(db.Text)
    llm_model = db.Column(db.String(100))
    scenario_data = db.Column(db.Text, default="{}")
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at = db.Column(db.DateTime)
    duration_sec = db.Column(db.Float)

    def as_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "topics": _safe_json(self.topics),
            "options": _safe_json(self.options),
            "source_urls": _safe_json(self.source_urls),
            "topic_mode": self.topic_mode,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "message": self.message,
            "output_files": _safe_json(self.output_files),
            "error_message": self.error_message,
            "llm_model": self.llm_model,
            "scenario_data": _safe_json(self.scenario_data),
            "created_at": _fmt_dt(self.created_at),
            "completed_at": _fmt_dt(self.completed_at),
            "duration_sec": self.duration_sec,
        }


class Script(db.Model):
    __tablename__ = "scripts"
    id = db.Column(db.String(64), primary_key=True)
    job_id = db.Column(
        db.String(64), db.ForeignKey("jobs.id", ondelete="CASCADE")
    )
    topic = db.Column(db.String(500))
    content = db.Column(db.Text)
    edited = db.Column(db.Text)
    voice = db.Column(db.String(80))
    source_type = db.Column(db.String(20), default="ai")
    scenes = db.Column(db.Text, default="[]")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def as_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "topic": self.topic,
            "content": self.edited or self.content,
            "voice": self.voice,
            "source_type": self.source_type,
            "scenes": _safe_json(self.scenes),
        }


def _safe_json(v):
    if not v:
        return [] if v == "[]" or v is None else {}
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return v


def _fmt_dt(dt):
    return dt.isoformat() if dt else None


with app.app_context():
    db.create_all()
    logger.info(f"DB initialized: {DB_PATH}")


# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════
class Config:
    OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
    MUSIC_DIR = BASE_DIR / os.getenv("MUSIC_DIR", "music")
    FONTS_DIR = BASE_DIR / os.getenv("FONTS_DIR", "fonts")
    CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", "cache")
    COMFYUI_DIR = BASE_DIR / "comfyui"

    VIDEO = {
        "width": int(os.getenv("DEFAULT_WIDTH", "1080")),
        "height": int(os.getenv("DEFAULT_HEIGHT", "1920")),
        "fps": int(os.getenv("DEFAULT_FPS", "30")),
        "crf": int(os.getenv("DEFAULT_CRF", "23")),
        "codec": os.getenv("DEFAULT_CODEC", "libx264"),
        "audio_br": os.getenv("DEFAULT_AUDIO_BITRATE", "192k"),
    }

    LLM = {
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "exaone3.5:32b"),
        "fallback": os.getenv("OLLAMA_FALLBACK_MODEL", "mistral:7b"),
        "timeout": int(os.getenv("OLLAMA_TIMEOUT", "120")),
        "temp": float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
        "max_tok": int(os.getenv("OLLAMA_MAX_TOKENS", "2000")),
        "ctx": int(os.getenv("OLLAMA_CONTEXT_SIZE", "8192")),
    }

    COMFYUI = {
        "host": os.getenv("COMFYUI_HOST", "http://127.0.0.1:8188"),
        "timeout": int(os.getenv("COMFYUI_TIMEOUT", "600")),
    }

    MAX_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))

    # Platform output presets
    OUTPUT_PRESETS = {
        "youtube_shorts": {
            "name": "YouTube Shorts", "icon": "🎬",
            "width": 1080, "height": 1920, "fps": 30, "ratio": "9:16",
            "max_duration": 60, "codec": "libx264",
        },
        "instagram_reels": {
            "name": "Instagram Reels", "icon": "📱",
            "width": 1080, "height": 1920, "fps": 30, "ratio": "9:16",
            "max_duration": 90, "codec": "libx264",
        },
        "tiktok": {
            "name": "TikTok", "icon": "🎵",
            "width": 1080, "height": 1920, "fps": 30, "ratio": "9:16",
            "max_duration": 60, "codec": "libx264",
        },
        "x_twitter": {
            "name": "X (Twitter)", "icon": "🐦",
            "width": 1080, "height": 1080, "fps": 30, "ratio": "1:1",
            "max_duration": 140, "codec": "libx264",
        },
        "youtube_standard": {
            "name": "YouTube 일반", "icon": "📺",
            "width": 1920, "height": 1080, "fps": 30, "ratio": "16:9",
            "max_duration": 0, "codec": "libx264",
        },
        "instagram_feed": {
            "name": "Instagram 피드", "icon": "📸",
            "width": 1080, "height": 1350, "fps": 30, "ratio": "4:5",
            "max_duration": 60, "codec": "libx264",
        },
        "custom": {
            "name": "커스텀", "icon": "⚙️",
            "width": 1080, "height": 1920, "fps": 30, "ratio": "custom",
            "max_duration": 0, "codec": "libx264",
        },
    }

    # Subtitle style presets
    SUBTITLE_STYLES = {
        "classic": {
            "name": "클래식", "desc": "하단 검정 배경 흰 글씨 (뉴스 스타일)",
            "position": "bottom", "bg": "rgba(0,0,0,0.8)", "color": "#FFFFFF",
            "font_weight": "bold", "font_size": 36,
        },
        "youtube_shorts": {
            "name": "유튜브 쇼츠", "desc": "가운데 큰 글씨, 단어 강조 (노란 하이라이트)",
            "position": "center", "bg": "none", "color": "#FFFFFF",
            "highlight": "#FFD700", "font_weight": "black", "font_size": 48,
        },
        "cinematic": {
            "name": "시네마틱", "desc": "하단 얇은 폰트, 반투명 배경",
            "position": "bottom", "bg": "rgba(0,0,0,0.4)", "color": "#FFFFFF",
            "font_weight": "light", "font_size": 32,
        },
        "pop": {
            "name": "팝", "desc": "컬러풀, 그림자, 두꺼운 폰트",
            "position": "center", "bg": "none", "color": "#FF6B6B",
            "shadow": True, "font_weight": "black", "font_size": 44,
        },
        "minimal": {
            "name": "미니멀", "desc": "작은 글씨, 배경 없음",
            "position": "bottom", "bg": "none", "color": "#CCCCCC",
            "font_weight": "normal", "font_size": 28,
        },
        "custom": {
            "name": "커스텀", "desc": "폰트/색상/배경 직접 설정",
            "position": "bottom", "bg": "rgba(0,0,0,0.6)", "color": "#FFFFFF",
            "font_weight": "bold", "font_size": 36,
        },
    }

    # Subtitle color options
    SUBTITLE_COLORS = {
        "auto": {"name": "Auto (가독성 자동 최적화)", "icon": "🤖"},
        "white": {"name": "흰색", "icon": "⬜", "hex": "#FFFFFF"},
        "black": {"name": "검정", "icon": "⬛", "hex": "#000000"},
        "yellow": {"name": "노란색", "icon": "🟨", "hex": "#FFD700"},
        "custom": {"name": "커스텀", "icon": "🎨"},
    }

    # Quality presets
    QUALITY_PRESETS = {
        "balance": {
            "name": "밸런스 (추천)",
            "desc": "480p 생성 → AI 업스케일 1080p",
            "gen_width": 480, "gen_height": 832,
            "upscale": True, "icon": "🚀",
            "est_time_per_scene": 240,  # 4 min
        },
        "ultra": {
            "name": "최고품질",
            "desc": "720p 직접 생성",
            "gen_width": 720, "gen_height": 1280,
            "upscale": False, "icon": "💎",
            "est_time_per_scene": 600,  # 10 min
        },
    }

    # BGM mood categories
    BGM_MOODS = {
        "emotional": {"name": "감성/드라마", "icon": "🎭", "keywords": ["piano", "acoustic", "emotional"]},
        "tension": {"name": "긴장/스릴", "icon": "🔥", "keywords": ["electronic", "pulse", "dark"]},
        "humor": {"name": "유머/밝음", "icon": "😄", "keywords": ["upbeat", "pop", "fun"]},
        "info": {"name": "정보/교육", "icon": "🧠", "keywords": ["lofi", "ambient", "calm"]},
        "tech": {"name": "테크/미래", "icon": "🚀", "keywords": ["synthwave", "edm", "futuristic"]},
        "healing": {"name": "힐링/자연", "icon": "🌿", "keywords": ["nature", "acoustic", "soft"]},
        "celebration": {"name": "축하/이벤트", "icon": "🎉", "keywords": ["bigbeat", "fanfare", "party"]},
        "none": {"name": "BGM 없음", "icon": "🔇", "keywords": []},
    }

    @classmethod
    def init_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.MUSIC_DIR, cls.FONTS_DIR, cls.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.init_dirs()


# ═══════════════════════════════════════════════════
#  GPU DETECTOR — WSL2 / Native Linux / Fallback
# ═══════════════════════════════════════════════════
class GPUDetector:
    _cache: Optional[dict] = None
    _lock = threading.Lock()

    @classmethod
    def detect(cls) -> dict:
        with cls._lock:
            if cls._cache is not None:
                return cls._cache
            cls._cache = cls._do_detect()
            return cls._cache

    @classmethod
    def refresh(cls) -> dict:
        with cls._lock:
            cls._cache = None
        return cls.detect()

    @classmethod
    def _do_detect(cls) -> dict:
        info = {
            "available": False, "name": "N/A",
            "vram_total_mb": 0, "vram_used_mb": 0, "vram_free_mb": 0,
            "driver_version": "N/A", "cuda_version": "N/A",
            "nvenc_supported": False, "utilization": 0, "temperature": 0,
            "power_draw": 0, "power_limit": 0, "fan_speed": 0,
            "pci_bus": "N/A", "compute_capability": "N/A",
            "wsl2": cls._is_wsl2(), "multi_gpu": [],
        }
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.used,memory.free,"
                 "driver_version,utilization.gpu,temperature.gpu,"
                 "power.draw,power.limit,fan.speed,pci.bus_id,compute_cap",
                 "--format=csv,noheader,nounits"],
                timeout=5, stderr=subprocess.DEVNULL, text=True,
            ).strip()
            gpus = []
            for line in out.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 12:
                    continue

                def _safe_int(v):
                    try:
                        return int(float(v))
                    except (ValueError, TypeError):
                        return 0

                def _safe_float(v):
                    try:
                        return round(float(v), 1)
                    except (ValueError, TypeError):
                        return 0.0

                gpu = {
                    "name": parts[0],
                    "vram_total_mb": _safe_int(parts[1]),
                    "vram_used_mb": _safe_int(parts[2]),
                    "vram_free_mb": _safe_int(parts[3]),
                    "driver_version": parts[4],
                    "utilization": _safe_int(parts[5]),
                    "temperature": _safe_int(parts[6]),
                    "power_draw": _safe_float(parts[7]),
                    "power_limit": _safe_float(parts[8]),
                    "fan_speed": _safe_int(parts[9]),
                    "pci_bus": parts[10],
                    "compute_capability": parts[11],
                }
                gpus.append(gpu)
            if gpus:
                primary = gpus[0]
                info.update(available=True, **{k: v for k, v in primary.items()})
                info["multi_gpu"] = gpus if len(gpus) > 1 else []
            try:
                cuda_out = subprocess.check_output(
                    ["nvidia-smi"], timeout=5, stderr=subprocess.DEVNULL, text=True
                )
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", cuda_out)
                if m:
                    info["cuda_version"] = m.group(1)
            except Exception:
                pass
            try:
                enc_check = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=5,
                )
                info["nvenc_supported"] = "h264_nvenc" in enc_check.stdout
            except Exception:
                pass
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        except Exception as e:
            logger.debug(f"GPU detect error: {e}")
        return info

    @staticmethod
    def _is_wsl2() -> bool:
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except Exception:
            return False


gpu_info = GPUDetector.detect()
logger.info(
    f"GPU: {gpu_info['name']} | VRAM: {gpu_info['vram_total_mb']}MB | "
    f"NVENC: {gpu_info['nvenc_supported']} | WSL2: {gpu_info['wsl2']}"
)


# ═══════════════════════════════════════════════════
#  COMFYUI ENGINE — AI Video Generation
# ═══════════════════════════════════════════════════

def _comfy_log(level: str, msg: str):
    """Emit a ComfyUI log to connected WebSocket clients."""
    try:
        sio.emit("comfyui_log", {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": msg,
        })
    except Exception:
        pass
class ComfyUIEngine:
    """
    ComfyUI API integration for AI video generation.
    Supports FramePack and Wan2.1 workflows.
    Falls back to PIL+FFmpeg slideshow if ComfyUI is not available.

    v7.0.1 Fixes:
      - Dynamic checkpoint detection (no hardcoded model names)
      - Random positive seed (ComfyUI requires seed >= 0)
      - Removed 'note' keys from workflow JSON (fixes node_replace_manager TypeError)
      - Output file download logic for generate_video
      - Better error logging for prompt validation failures
    """

    def __init__(self):
        self._available: Optional[bool] = None
        self._lock = threading.Lock()
        self._checkpoints_cache: Optional[list] = None
        self._object_info_cache: Optional[dict] = None

    # ── Server connectivity ──────────────────────────

    def check(self) -> bool:
        """Check if ComfyUI server is running."""
        if self._available is not None:
            return self._available
        try:
            r = httpx.get(f"{cfg.COMFYUI['host']}/system_stats", timeout=5)
            self._available = r.status_code == 200
            if self._available:
                logger.info("ComfyUI server detected")
        except Exception:
            self._available = False
            logger.debug("ComfyUI not available - will use fallback rendering")
        return self._available

    def reset_cache(self):
        self._available = None
        self._checkpoints_cache = None
        self._object_info_cache = None

    def get_status(self) -> dict:
        """Get ComfyUI server status including available checkpoints."""
        try:
            r = httpx.get(f"{cfg.COMFYUI['host']}/system_stats", timeout=5)
            if r.status_code == 200:
                data = r.json()
                ckpts = self._get_checkpoints()
                return {
                    "available": True,
                    "vram_total": data.get("devices", [{}])[0].get("vram_total", 0),
                    "vram_free": data.get("devices", [{}])[0].get("vram_free", 0),
                    "models_loaded": len(data.get("models", [])),
                    "checkpoints": ckpts[:10] if ckpts else [],
                    "checkpoint_count": len(ckpts) if ckpts else 0,
                }
        except Exception:
            pass
        return {"available": False}

    # ── Model discovery ──────────────────────────────

    def _get_object_info(self) -> dict:
        """Fetch ComfyUI object_info (all node types and their options)."""
        if self._object_info_cache is not None:
            return self._object_info_cache
        try:
            r = httpx.get(f"{cfg.COMFYUI['host']}/object_info", timeout=15)
            if r.status_code == 200:
                self._object_info_cache = r.json()
                return self._object_info_cache
        except Exception as e:
            logger.warning(f"Failed to fetch object_info: {e}")
        return {}

    def _get_checkpoints(self) -> list:
        """Get list of available checkpoint model names from ComfyUI."""
        if self._checkpoints_cache is not None:
            return self._checkpoints_cache
        try:
            obj_info = self._get_object_info()
            loader = obj_info.get("CheckpointLoaderSimple", {})
            input_req = loader.get("input", {}).get("required", {})
            ckpt_list = input_req.get("ckpt_name", [[]])[0]
            if isinstance(ckpt_list, list) and ckpt_list:
                self._checkpoints_cache = ckpt_list
                logger.info(f"ComfyUI checkpoints found: {ckpt_list}")
                return ckpt_list
        except Exception as e:
            logger.warning(f"Failed to get checkpoints: {e}")
        return []

    def _pick_checkpoint(self) -> Optional[str]:
        """Pick the best available checkpoint for text-to-image generation."""
        ckpts = self._get_checkpoints()
        if not ckpts:
            logger.error("No checkpoints found in ComfyUI!")
            return None

        # Priority order for SDXL / SD1.5 checkpoints
        preferred_keywords = [
            "sdxl", "sd_xl", "dreamshaperxl", "juggernautxl", "realvisxl",
            "sd3", "flux", "playground",
            "dreamshaper", "realisticvision", "revanimated",
            "sd_1", "sd1", "v1-5", "v1.5",
        ]
        for keyword in preferred_keywords:
            for ckpt in ckpts:
                if keyword.lower() in ckpt.lower():
                    logger.info(f"Selected checkpoint: {ckpt}")
                    return ckpt

        # Fallback: just use the first available checkpoint
        logger.info(f"Using first available checkpoint: {ckpts[0]}")
        return ckpts[0]

    def _get_available_nodes(self) -> set:
        """Get set of available node class_types from ComfyUI."""
        obj_info = self._get_object_info()
        return set(obj_info.keys())

    def _has_node(self, class_type: str) -> bool:
        """Check if a specific node type is available."""
        return class_type in self._get_available_nodes()

    @staticmethod
    def _random_seed() -> int:
        """Generate a random seed value that ComfyUI accepts (0 ~ 2^53)."""
        return random.randint(0, 2**53 - 1)

    # ── Image generation ─────────────────────────────

    def generate_image(
        self, prompt: str, width: int, height: int,
        output_path: Path, progress_cb=None
    ) -> bool:
        """
        Generate a start frame image using ComfyUI.
        Falls back to PIL gradient if ComfyUI unavailable or no checkpoints.
        """
        if not self.check():
            return self._fallback_image(prompt, width, height, output_path)

        try:
            workflow = self._build_t2i_workflow(prompt, width, height)
            if workflow is None:
                logger.warning("Cannot build t2i workflow (no checkpoint) - using fallback")
                return self._fallback_image(prompt, width, height, output_path)

            result = self._queue_prompt_and_wait_image(workflow)
            if result:
                # Download generated image from ComfyUI
                for node_id, node_out in result.items():
                    images = node_out.get("images", [])
                    for img_info in images:
                        img_url = (
                            f"{cfg.COMFYUI['host']}/view?"
                            f"filename={img_info['filename']}"
                            f"&subfolder={img_info.get('subfolder', '')}"
                            f"&type={img_info.get('type', 'output')}"
                        )
                        r = httpx.get(img_url, timeout=30)
                        if r.status_code == 200 and len(r.content) > 1000:
                            output_path.write_bytes(r.content)
                            logger.info(f"Image saved: {output_path} ({len(r.content)} bytes)")
                            _comfy_log("SUCCESS", f"✅ 이미지 저장: {len(r.content)//1024}KB")
                            return True

            logger.warning("ComfyUI returned no images - using fallback")
            _comfy_log("WARN", "⚠️ ComfyUI 이미지 없음 → 폴백 이미지 사용")
        except Exception as e:
            logger.warning(f"ComfyUI image gen failed: {e}")

        return self._fallback_image(prompt, width, height, output_path)

    # ── Video generation ─────────────────────────────

    def generate_video(
        self, image_path: Path, prompt: str,
        num_frames: int, width: int, height: int,
        output_path: Path, model: str = "framepack",
        progress_cb=None,
    ) -> bool:
        """
        Generate video from image + prompt using ComfyUI.
        Falls back to static video if ComfyUI unavailable.
        """
        if not self.check():
            return self._fallback_video(image_path, output_path, num_frames)

        try:
            # Upload image to ComfyUI first
            uploaded_name = self._upload_image(image_path)
            if not uploaded_name:
                logger.warning("Failed to upload image to ComfyUI - using fallback")
                return self._fallback_video(image_path, output_path, num_frames)

            if model == "wan2.1":
                workflow = self._build_wan_i2v_workflow(
                    uploaded_name, prompt, num_frames, width, height
                )
            else:
                workflow = self._build_framepack_workflow(
                    uploaded_name, prompt, num_frames, width, height
                )

            if workflow is None:
                logger.warning("Cannot build video workflow - using fallback")
                _comfy_log("WARN", "⚠️ AI 비디오 노드 없음 → Ken Burns 효과로 대체")
                return self._fallback_video(image_path, output_path, num_frames)

            result = self._queue_prompt_and_wait_video(workflow, output_path, progress_cb)
            if result:
                return True

        except Exception as e:
            logger.warning(f"ComfyUI video gen failed: {e}")

        return self._fallback_video(image_path, output_path, num_frames)

    # ── Image upload ─────────────────────────────────

    def _upload_image(self, image_path: Path) -> Optional[str]:
        """Upload a local image to ComfyUI /upload/image endpoint."""
        try:
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/png")}
                r = httpx.post(
                    f"{cfg.COMFYUI['host']}/upload/image",
                    files=files,
                    timeout=30,
                )
            if r.status_code == 200:
                data = r.json()
                uploaded_name = data.get("name", image_path.name)
                logger.info(f"Image uploaded to ComfyUI: {uploaded_name}")
                return uploaded_name
        except Exception as e:
            logger.warning(f"Image upload to ComfyUI failed: {e}")
        return None

    # ── Prompt queueing & waiting ────────────────────

    def _queue_prompt(self, workflow: dict) -> Optional[dict]:
        """Queue a workflow prompt to ComfyUI and return response."""
        try:
            client_id = uuid.uuid4().hex
            payload = {"prompt": workflow, "client_id": client_id}
            r = httpx.post(
                f"{cfg.COMFYUI['host']}/prompt",
                json=payload,
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            else:
                # Log validation errors
                try:
                    err = r.json()
                    logger.error(f"ComfyUI prompt rejected ({r.status_code}): {json.dumps(err, indent=2, ensure_ascii=False)[:500]}")
                except Exception:
                    logger.error(f"ComfyUI prompt rejected ({r.status_code}): {r.text[:300]}")
        except Exception as e:
            logger.error(f"ComfyUI queue error: {e}")
        return None

    def _queue_prompt_and_wait_image(self, workflow: dict) -> Optional[dict]:
        """Queue prompt and wait for image generation completion."""
        resp = self._queue_prompt(workflow)
        if not resp:
            return None

        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            return None

        start_time = time.monotonic()
        timeout = min(cfg.COMFYUI["timeout"], 180)  # Max 3 min for images
        while time.monotonic() - start_time < timeout:
            time.sleep(2)
            try:
                hr = httpx.get(
                    f"{cfg.COMFYUI['host']}/history/{prompt_id}",
                    timeout=5,
                )
                if hr.status_code == 200:
                    history = hr.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed", False) or status.get("status_str") == "success":
                            return history[prompt_id].get("outputs", {})
                        # Also check if outputs exist even without explicit completion flag
                        outputs = history[prompt_id].get("outputs", {})
                        if outputs:
                            return outputs
            except Exception:
                pass

        logger.warning(f"ComfyUI image generation timeout after {timeout}s")
        return None

    def _queue_prompt_and_wait_video(
        self, workflow: dict, output_path: Path, progress_cb=None
    ) -> bool:
        """Queue prompt and wait for video generation, then download output."""
        resp = self._queue_prompt(workflow)
        if not resp:
            return False

        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            return False

        start_time = time.monotonic()
        timeout = cfg.COMFYUI["timeout"]
        while time.monotonic() - start_time < timeout:
            time.sleep(3)
            try:
                hr = httpx.get(
                    f"{cfg.COMFYUI['host']}/history/{prompt_id}",
                    timeout=5,
                )
                if hr.status_code == 200:
                    history = hr.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        if outputs:
                            # Try to find and download video/gif output
                            for node_id, node_out in outputs.items():
                                # Check for video outputs (gifs or videos)
                                for key in ["gifs", "videos", "images"]:
                                    items = node_out.get(key, [])
                                    for item in items:
                                        fname = item.get("filename", "")
                                        subfolder = item.get("subfolder", "")
                                        ftype = item.get("type", "output")
                                        if fname:
                                            dl_url = (
                                                f"{cfg.COMFYUI['host']}/view?"
                                                f"filename={fname}"
                                                f"&subfolder={subfolder}"
                                                f"&type={ftype}"
                                            )
                                            dr = httpx.get(dl_url, timeout=120)
                                            if dr.status_code == 200 and len(dr.content) > 5000:
                                                output_path.write_bytes(dr.content)
                                                logger.info(f"Video saved: {output_path} ({len(dr.content)} bytes)")
                                                _comfy_log("SUCCESS", f"✅ 비디오 저장: {len(dr.content)//1024//1024}MB")
                                                return True
                            # If we got outputs but couldn't download, still mark as done
                            logger.warning("ComfyUI outputs found but no downloadable video")
                            return False
            except Exception:
                pass

            elapsed = time.monotonic() - start_time
            if progress_cb:
                pct = min(int(elapsed / timeout * 100), 95)
                progress_cb(pct, f"AI 영상 생성 중... ({int(elapsed)}초)")

        logger.warning(f"ComfyUI video timeout after {timeout}s")
        return False

    # ── Workflow builders ────────────────────────────

    def _build_t2i_workflow(self, prompt: str, w: int, h: int) -> Optional[dict]:
        """Build text-to-image workflow using the best available checkpoint."""
        ckpt_name = self._pick_checkpoint()
        if not ckpt_name:
            return None

        seed = self._random_seed()
        logger.info(f"T2I workflow: checkpoint={ckpt_name}, seed={seed}, size={w}x{h}")
        _comfy_log("INFO", f"🎨 이미지 생성 시작: {ckpt_name} ({w}x{h})")

        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt_name},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "blurry, low quality, distorted, watermark, text, logo",
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": w, "height": h, "batch_size": 1},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": 25,
                    "cfg": 7.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"images": ["6", 0], "filename_prefix": "aishorts_gen"},
            },
        }

    def _build_framepack_workflow(
        self, image_name: str, prompt: str, frames: int, w: int, h: int
    ) -> Optional[dict]:
        """Build FramePack image-to-video workflow."""
        seed = self._random_seed()
        logger.info(f"FramePack workflow: image={image_name}, seed={seed}, frames={frames}")
        _comfy_log("INFO", f"🎬 FramePack 비디오 생성: {frames}프레임")

        # Check if FramePackSampler node exists
        if not self._has_node("FramePackSampler"):
            logger.warning("FramePackSampler node not available in ComfyUI")
            return None

        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "FramePackSampler",
                "inputs": {
                    "image": ["1", 0],
                    "prompt": prompt,
                    "num_frames": frames,
                    "width": w,
                    "height": h,
                    "steps": 25,
                    "cfg": 7.0,
                    "seed": seed,
                    "use_teacache": True,
                },
            },
        }

    def _build_wan_i2v_workflow(
        self, image_name: str, prompt: str, frames: int, w: int, h: int
    ) -> Optional[dict]:
        """Build Wan2.1 image-to-video workflow."""
        seed = self._random_seed()
        logger.info(f"Wan2.1 workflow: image={image_name}, seed={seed}, frames={frames}")
        _comfy_log("INFO", f"🎬 Wan2.1 비디오 생성: {frames}프레임")

        # Check if WanVideoSampler node exists
        if not self._has_node("WanVideoSampler"):
            logger.warning("WanVideoSampler node not available in ComfyUI")
            return None

        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "image": ["1", 0],
                    "prompt": prompt,
                    "num_frames": frames,
                    "width": w,
                    "height": h,
                    "steps": 30,
                    "cfg": 6.0,
                    "seed": seed,
                },
            },
        }

    # ── Fallback renderers ───────────────────────────

    def _fallback_image(self, prompt: str, w: int, h: int, output_path: Path) -> bool:
        """Generate gradient background with PIL when ComfyUI unavailable."""
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (w, h), (13, 17, 23))
            draw = ImageDraw.Draw(img)
            # Gradient background
            for y in range(h):
                ratio = y / h
                r = int(13 + (35 - 13) * ratio)
                g = int(17 + (55 - 17) * ratio)
                b = int(23 + (80 - 23) * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))
            # Soft glow
            cx, cy = w // 2, h // 2
            for radius in range(350, 0, -15):
                alpha_val = max(0, int(20 * (1 - radius / 350)))
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=(
                        int(alpha_val * 0.3),
                        min(255, 100 + alpha_val * 3),
                        min(255, 180 + alpha_val * 2),
                    ),
                )
            img.save(str(output_path), "PNG")
            logger.info(f"Fallback image generated: {output_path}")
            _comfy_log("INFO", "📷 폴백 그라디언트 이미지 생성됨")
            return True
        except Exception as e:
            logger.error(f"Fallback image failed: {e}")
            return False

    def _fallback_video(self, image_path: Path, output_path: Path, num_frames: int) -> bool:
        """Create Ken Burns effect video from image using FFmpeg (zoom+pan)."""
        try:
            duration = max(num_frames / 30.0, 2.0)
            # Ken Burns: slow zoom-in with subtle pan for cinematic feel
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-loop", "1", "-i", str(image_path),
                "-vf", (
                    f"scale=8000:-1,"
                    f"zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':"
                    f"y='ih/2-(ih/zoom/2)':d={int(duration * 30)}:s=1080x1920:fps=30"
                ),
                "-c:v", "libx264", "-t", str(duration),
                "-pix_fmt", "yuv420p", "-preset", "fast",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                _comfy_log("INFO", f"📹 Ken Burns 효과 비디오 생성 ({duration:.1f}초)")
                return True
            # Fallback to simple static if zoompan fails
            cmd_simple = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-loop", "1", "-i", str(image_path),
                "-c:v", "libx264", "-t", str(duration),
                "-pix_fmt", "yuv420p", "-preset", "fast",
                str(output_path),
            ]
            subprocess.run(cmd_simple, capture_output=True, timeout=60)
            return output_path.exists() and output_path.stat().st_size > 1000
        except Exception as e:
            logger.error(f"Fallback video failed: {e}")
            return False


comfyui = ComfyUIEngine()


# ═══════════════════════════════════════════════════
#  COMFYUI PROCESS MANAGER — Unified start/stop
# ═══════════════════════════════════════════════════
class ComfyUIProcessManager:
    """
    Manages ComfyUI as a subprocess, captures logs in real-time,
    and streams them to the UI via WebSocket.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._logs: list[dict] = []
        self._max_logs = 500
        self._log_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def find_comfyui_path(self) -> Optional[Path]:
        """Auto-detect ComfyUI installation path."""
        candidates = [
            BASE_DIR / "comfyui" / "ComfyUI",
            BASE_DIR / "ComfyUI",
            Path.home() / "ComfyUI",
            BASE_DIR.parent / "ComfyUI",
        ]
        for p in candidates:
            if (p / "main.py").exists():
                return p
        return None

    def is_running(self) -> bool:
        if self._process and self._process.poll() is None:
            return True
        # Also check if external ComfyUI is running
        try:
            r = httpx.get(f"{cfg.COMFYUI['host']}/system_stats", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def start(self) -> dict:
        """Start ComfyUI subprocess."""
        if self.is_running():
            self._add_log("INFO", "ComfyUI가 이미 실행 중입니다")
            return {"success": True, "message": "Already running"}

        comfyui_path = self.find_comfyui_path()
        if not comfyui_path:
            self._add_log("ERROR", "ComfyUI 설치 경로를 찾을 수 없습니다")
            return {"success": False, "message": "ComfyUI not found"}

        venv_python = comfyui_path / "venv" / "bin" / "python"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        main_py = str(comfyui_path / "main.py")

        try:
            self._add_log("INFO", f"ComfyUI 시작 중... ({comfyui_path})")
            self._process = subprocess.Popen(
                [python_cmd, main_py, "--listen", "127.0.0.1", "--port", "8188"],
                cwd=str(comfyui_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self._thread = threading.Thread(target=self._read_output, daemon=True)
            self._thread.start()

            # Wait for server to be ready
            self._add_log("INFO", "ComfyUI 서버 대기 중...")
            for i in range(60):
                time.sleep(2)
                try:
                    r = httpx.get(f"{cfg.COMFYUI['host']}/system_stats", timeout=2)
                    if r.status_code == 200:
                        self._add_log("SUCCESS", "✅ ComfyUI 서버 준비 완료!")
                        comfyui.reset_cache()
                        return {"success": True, "message": "ComfyUI started"}
                except Exception:
                    pass

            self._add_log("WARN", "ComfyUI 서버 시작 대기 시간 초과 (계속 로딩 중일 수 있음)")
            return {"success": True, "message": "Started but not yet responding"}

        except Exception as e:
            self._add_log("ERROR", f"ComfyUI 시작 실패: {e}")
            return {"success": False, "message": str(e)}

    def stop(self) -> dict:
        if self._process and self._process.poll() is None:
            self._add_log("INFO", "ComfyUI 종료 중...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._add_log("INFO", "ComfyUI 종료됨")
            comfyui.reset_cache()
            return {"success": True, "message": "Stopped"}
        return {"success": True, "message": "Not running"}

    def _read_output(self):
        """Read ComfyUI stdout/stderr and store logs."""
        try:
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    level = "INFO"
                    if "error" in line.lower() or "traceback" in line.lower():
                        level = "ERROR"
                    elif "warning" in line.lower():
                        level = "WARN"
                    elif "loaded" in line.lower() or "success" in line.lower():
                        level = "SUCCESS"
                    self._add_log(level, line)
        except Exception:
            pass

    def _add_log(self, level: str, message: str):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        with self._log_lock:
            self._logs.append(entry)
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs:]
        # Emit to connected clients via WebSocket
        try:
            sio.emit("comfyui_log", entry)
        except Exception:
            pass

    def get_logs(self, n: int = 50) -> list:
        with self._log_lock:
            return self._logs[-n:]


comfyui_proc = ComfyUIProcessManager()
class ScenarioEngine:
    """
    Generates dual scenarios with detailed scene descriptions.
    Pipeline: Topic -> 2 Scenarios (different styles) -> Scene breakdown -> English prompts
    """

    SCENARIO_STYLES = {
        "emotional": {"name": "감성 다큐", "desc": "진지하고 감성적인 스토리텔링", "icon": "🎭"},
        "humor": {"name": "유머 예능", "desc": "가볍고 재미있는 예능 스타일", "icon": "😄"},
        "news": {"name": "뉴스 브리핑", "desc": "전문적인 뉴스 앵커 스타일", "icon": "📰"},
        "story": {"name": "스토리텔링", "desc": "주인공이 있는 서사 구조", "icon": "📖"},
        "info": {"name": "정보 전달", "desc": "팩트 중심 빠른 정보 전달", "icon": "🧠"},
        "cinematic": {"name": "시네마틱", "desc": "영화 같은 영상미", "icon": "🎬"},
    }

    SCENE_TEMPLATE = """당신은 유튜브 숏츠 전문 영상 감독입니다.

아래 주제와 스타일로 {num_scenes}개의 장면(Scene)으로 구성된 영상 시나리오를 작성하세요.
전체 영상 길이는 약 {duration}초입니다.

주제: {topic}
스타일: {style_name} - {style_desc}

각 장면은 반드시 아래 형식으로 작성하세요:

===SCENE 1===
[나레이션] (이 장면의 나레이션 대본, 한국어, 자연스러운 구어체)
[구도] (풀샷/미디엄샷/클로즈업/조감도/POV 등)
[카메라] (고정/팬/틸트/줌인/줌아웃/트래킹 등)
[조명] (자연광/스튜디오/네온/석양/야경 등 구체적으로)
[행동] (등장인물의 구체적 행동/동작/표정)
[배경] (장소, 소품, 환경 구체 묘사)
[분위기] (전체적인 감정/톤)

===SCENE 2===
...

규칙:
1. 첫 장면은 강렬한 후킹으로 시작
2. 마지막 장면은 구독/좋아요 유도
3. 장면 전환이 자연스럽게 이어질 것
4. 각 장면의 나레이션은 2~3문장
5. {style_instruction}"""

    TRANSLATE_PROMPT = """You are a professional AI video prompt engineer.
Translate the following Korean scene description into an English video generation prompt.
The prompt should be highly detailed and optimized for AI video generation models (FramePack/Wan2.1).

Korean scene description:
- Composition: {composition}
- Camera: {camera}
- Lighting: {lighting}
- Action: {action}
- Background: {background}
- Mood: {mood}

Generate a single English paragraph prompt. Include:
1. Main subject and action description
2. Camera movement and angle
3. Lighting and color grading details
4. Environment and atmosphere
5. Quality keywords: "cinematic, 4K, high quality, detailed"

Output ONLY the English prompt text, nothing else."""

    def __init__(self, ai_engine):
        self.ai = ai_engine

    def generate_dual_scenarios(
        self, topic: str, num_scenes: int = 10, duration: int = 40,
        style_a: str = "emotional", style_b: str = "humor",
    ) -> dict:
        """Generate two scenarios with different styles."""
        result = {
            "topic": topic,
            "scenario_a": None,
            "scenario_b": None,
            "style_a": style_a,
            "style_b": style_b,
        }

        # Generate scenario A
        logger.info(f"Generating scenario A ({style_a}) for: {topic}")
        result["scenario_a"] = self._generate_scenario(
            topic, num_scenes, duration, style_a
        )

        # Generate scenario B
        logger.info(f"Generating scenario B ({style_b}) for: {topic}")
        result["scenario_b"] = self._generate_scenario(
            topic, num_scenes, duration, style_b
        )

        return result

    def _generate_scenario(
        self, topic: str, num_scenes: int, duration: int, style: str
    ) -> dict:
        """Generate a single scenario with scene descriptions."""
        style_info = self.SCENARIO_STYLES.get(style, self.SCENARIO_STYLES["emotional"])
        style_instructions = {
            "emotional": "감성적이고 진지한 톤, 시청자의 감정을 자극",
            "humor": "유머러스하고 밈 감성, MZ세대 말투, 웃음 유발",
            "news": "전문적이고 신뢰감 있는 톤, 팩트 기반",
            "story": "주인공 시점의 서사 구조, 기승전결",
            "info": "핵심 정보를 빠르게 전달, 숫자와 데이터 활용",
            "cinematic": "영화 같은 시각적 연출, 드라마틱한 전개",
        }

        prompt = self.SCENE_TEMPLATE.format(
            topic=topic,
            num_scenes=num_scenes,
            duration=duration,
            style_name=style_info["name"],
            style_desc=style_info["desc"],
            style_instruction=style_instructions.get(style, "자연스럽게 진행"),
        )

        # Try LLM generation
        script_text = None
        model_used = "template"

        if self.ai._check():
            try:
                r = httpx.post(
                    f"{cfg.LLM['host']}/api/generate",
                    json={
                        "model": cfg.LLM["model"],
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": cfg.LLM["temp"],
                            "num_predict": cfg.LLM["max_tok"],
                            "num_ctx": cfg.LLM["ctx"],
                        },
                    },
                    timeout=cfg.LLM["timeout"],
                )
                r.raise_for_status()
                script_text = r.json().get("response", "").strip()
                model_used = cfg.LLM["model"]
                if len(script_text) < 50:
                    script_text = None
            except Exception as e:
                logger.warning(f"LLM scenario gen failed: {e}")

        # Parse scenes from LLM output or use fallback
        if script_text:
            scenes = self._parse_scenes(script_text, num_scenes)
        else:
            scenes = self._fallback_scenes(topic, num_scenes, style)

        # Build full narration from scene narrations
        full_narration = " ".join(s.get("narration", "") for s in scenes if s.get("narration"))

        return {
            "style": style,
            "style_name": style_info["name"],
            "style_icon": style_info["icon"],
            "narration": full_narration,
            "scenes": scenes,
            "model_used": model_used,
            "scene_count": len(scenes),
        }

    def _parse_scenes(self, text: str, expected: int) -> list[dict]:
        """Parse scene descriptions from LLM output."""
        scenes = []
        # Split by ===SCENE pattern
        parts = re.split(r'===\s*SCENE\s*\d+\s*===', text, flags=re.IGNORECASE)

        for i, part in enumerate(parts):
            if not part.strip():
                continue

            scene = {
                "scene_num": len(scenes) + 1,
                "narration": "",
                "composition": "",
                "camera": "",
                "lighting": "",
                "action": "",
                "background": "",
                "mood": "",
                "en_prompt": "",
            }

            # Extract fields
            patterns = {
                "narration": r'\[나레이션\]\s*(.+?)(?=\[|$)',
                "composition": r'\[구도\]\s*(.+?)(?=\[|$)',
                "camera": r'\[카메라\]\s*(.+?)(?=\[|$)',
                "lighting": r'\[조명\]\s*(.+?)(?=\[|$)',
                "action": r'\[행동\]\s*(.+?)(?=\[|$)',
                "background": r'\[배경\]\s*(.+?)(?=\[|$)',
                "mood": r'\[분위기\]\s*(.+?)(?=\[|$)',
            }

            for key, pattern in patterns.items():
                m = re.search(pattern, part, re.DOTALL)
                if m:
                    scene[key] = m.group(1).strip()

            # If no structured fields found, use the whole text as narration
            if not scene["narration"] and part.strip():
                scene["narration"] = part.strip()[:200]

            if scene["narration"]:
                scenes.append(scene)

        # Pad to expected count if needed
        while len(scenes) < expected:
            idx = len(scenes) + 1
            scenes.append({
                "scene_num": idx,
                "narration": f"장면 {idx} 내용이 여기에 들어갑니다.",
                "composition": "미디엄 샷",
                "camera": "고정",
                "lighting": "자연광",
                "action": "등장인물이 활동 중",
                "background": "일상적인 배경",
                "mood": "밝고 긍정적",
                "en_prompt": "",
            })

        return scenes[:expected]

    def _fallback_scenes(self, topic: str, num_scenes: int, style: str) -> list[dict]:
        """Generate template-based scenes when LLM is unavailable."""
        templates = [
            {"narration": f"{topic}에 대해 알고 계셨나요? 지금 바로 핵심을 알려드립니다!",
             "composition": "클로즈업", "camera": "줌인", "lighting": "드라마틱 조명",
             "action": "호기심 어린 표정", "background": "모던한 스튜디오", "mood": "궁금증 유발"},
            {"narration": "먼저 가장 중요한 포인트부터 살펴볼까요?",
             "composition": "미디엄 샷", "camera": "슬로우 팬", "lighting": "따뜻한 자연광",
             "action": "설명하는 제스처", "background": "밝은 공간", "mood": "친근하고 기대감"},
        ]

        scenes = []
        for i in range(num_scenes):
            t = templates[i % len(templates)]
            scenes.append({
                "scene_num": i + 1,
                "narration": t["narration"] if i < 2 else f"{topic}의 핵심 포인트 {i}입니다.",
                "composition": t["composition"],
                "camera": t["camera"],
                "lighting": t["lighting"],
                "action": t["action"],
                "background": t["background"],
                "mood": t["mood"],
                "en_prompt": "",
            })

        # Customize last scene
        if scenes:
            scenes[-1]["narration"] = f"이상 {topic}에 대해 알아봤습니다! 구독과 좋아요 부탁드려요!"
            scenes[-1]["mood"] = "마무리, 감사"

        return scenes

    def translate_scene_to_prompt(self, scene: dict) -> str:
        """Translate Korean scene description to English video prompt."""
        if not self.ai._check():
            return self._fallback_translate(scene)

        prompt = self.TRANSLATE_PROMPT.format(
            composition=scene.get("composition", "medium shot"),
            camera=scene.get("camera", "static"),
            lighting=scene.get("lighting", "natural light"),
            action=scene.get("action", "person speaking"),
            background=scene.get("background", "modern studio"),
            mood=scene.get("mood", "positive"),
        )

        try:
            r = httpx.post(
                f"{cfg.LLM['host']}/api/generate",
                json={
                    "model": cfg.LLM["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 300,
                        "num_ctx": cfg.LLM["ctx"],
                    },
                },
                timeout=30,
            )
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            if len(text) > 20:
                return text
        except Exception as e:
            logger.warning(f"Prompt translation failed: {e}")

        return self._fallback_translate(scene)

    def _fallback_translate(self, scene: dict) -> str:
        """Simple keyword-based English prompt generation."""
        parts = []
        comp = scene.get("composition", "")
        if "클로즈업" in comp:
            parts.append("close-up shot")
        elif "풀샷" in comp:
            parts.append("full shot")
        elif "조감도" in comp:
            parts.append("aerial view")
        else:
            parts.append("medium shot")

        camera = scene.get("camera", "")
        if "줌인" in camera:
            parts.append("slow zoom in")
        elif "팬" in camera:
            parts.append("smooth panning")
        elif "트래킹" in camera:
            parts.append("tracking shot")

        lighting = scene.get("lighting", "")
        if "자연광" in lighting:
            parts.append("natural lighting")
        elif "네온" in lighting:
            parts.append("neon lighting, cyberpunk atmosphere")
        elif "석양" in lighting:
            parts.append("golden hour sunset lighting")

        action = scene.get("action", "person in the scene")
        bg = scene.get("background", "modern environment")
        mood = scene.get("mood", "positive")

        parts.append(f"showing {action}")
        parts.append(f"in {bg}")
        parts.append(f"{mood} atmosphere")
        parts.append("cinematic quality, 4K, detailed, high quality")

        return ", ".join(parts)


# ═══════════════════════════════════════════════════
#  AI ENGINE (Ollama)
# ═══════════════════════════════════════════════════
class AIEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._available: Optional[bool] = None

    def _check(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = httpx.get(f"{cfg.LLM['host']}/api/tags", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def reset_cache(self):
        self._available = None

    def generate(self, topic: str) -> tuple[str, str]:
        import random
        FALLBACK_SCRIPTS = [
            "{topic}에 대해 알고 계셨나요? 지금 바로 핵심을 알려드립니다! 전문가들도 주목하는 이 내용, 끝까지 봐주시고 구독과 좋아요 꼭 눌러주세요!",
            "오늘 꼭 알아야 할 {topic} 완벽 정리! 최신 트렌드와 핵심 정보를 빠르게 전달해드립니다. 놓치지 마시고 구독 버튼도 눌러주세요!",
        ]
        if not self._check():
            return random.choice(FALLBACK_SCRIPTS).format(topic=topic), "template"
        try:
            prompt = f"""당신은 대한민국 유튜브 숏츠 전문 스크립트 작가입니다.
아래 주제로 40초 분량(120~160자)의 대본을 작성하세요.
규칙:
1. 첫 문장은 시청자를 사로잡는 강렬한 후킹
2. 자연스러운 한국어 구어체 (존댓말)
3. 핵심 정보 2~3가지 간결하게
4. 마지막은 구독/좋아요 유도
5. 대본 텍스트만 출력

주제: {topic}"""
            r = httpx.post(
                f"{cfg.LLM['host']}/api/generate",
                json={
                    "model": cfg.LLM["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": cfg.LLM["temp"],
                        "num_predict": cfg.LLM["max_tok"],
                        "num_ctx": cfg.LLM["ctx"],
                    },
                },
                timeout=cfg.LLM["timeout"],
            )
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            if len(text) > 20:
                return text, cfg.LLM["model"]
        except Exception as e:
            logger.warning(f"LLM gen failed: {e}")
        return random.choice(FALLBACK_SCRIPTS).format(topic=topic), "template"

    def list_models(self) -> list[dict]:
        try:
            r = httpx.get(f"{cfg.LLM['host']}/api/tags", timeout=5)
            return [
                {"name": m["name"], "size": m.get("size", 0)}
                for m in r.json().get("models", [])
            ]
        except Exception:
            return []


ai = AIEngine()
scenario_engine = ScenarioEngine(ai)


# ═══════════════════════════════════════════════════
#  TTS ENGINE (edge-tts with dedicated async loop)
# ═══════════════════════════════════════════════════
class TTSEngine:
    VOICES = {
        "ko-KR-SunHiNeural": "선희 (여성, 자연스러운)",
        "ko-KR-InJoonNeural": "인준 (남성, 신뢰감)",
        "ko-KR-HyunsuNeural": "현수 (남성, 차분한)",
        "ko-KR-YuJinNeural": "유진 (여성, 활발한)",
        "ko-KR-BongJinNeural": "봉진 (남성, 따뜻한)",
        "en-US-JennyNeural": "Jenny (English, Female)",
        "en-US-GuyNeural": "Guy (English, Male)",
        "ja-JP-NanamiNeural": "Nanami (日本語, 女性)",
        "zh-CN-XiaoxiaoNeural": "Xiaoxiao (中文, 女性)",
    }

    RATES = {
        "very_slow": "-30%", "slow": "-15%",
        "normal": "+0%", "fast": "+15%", "very_fast": "+30%",
    }

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="tts-loop"
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def synthesize(
        self, text: str, voice: str, out_path: Path,
        rate: str = "+0%", volume: str = "+0%"
    ) -> bool:
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._synth(text, voice, out_path, rate, volume), self._loop
            )
            future.result(timeout=60)
            ok = out_path.exists() and out_path.stat().st_size > 500
            if ok:
                logger.info(f"TTS done: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
            return ok
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    @staticmethod
    async def _synth(text: str, voice: str, out_path: Path, rate: str, volume: str):
        import edge_tts
        comm = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
        await comm.save(str(out_path))


tts_engine = TTSEngine()


# ═══════════════════════════════════════════════════
#  VIDEO ENGINE v7 — ComfyUI + FFmpeg Pipeline
# ═══════════════════════════════════════════════════
class VideoEngine:
    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/mnt/c/Windows/Fonts/malgunbd.ttf",
        "/mnt/c/Windows/Fonts/NanumGothicBold.ttf",
    ]

    def __init__(self):
        self._font_path = self._find_font()
        self._cpu_count = os.cpu_count() or 4
        logger.info(f"Video engine font: {self._font_path}")
        logger.info(f"Video engine CPU threads: {self._cpu_count}")

    def _find_font(self) -> Optional[str]:
        for f in self.FONT_CANDIDATES:
            if Path(f).exists():
                return f
        try:
            out = subprocess.check_output(
                ["fc-list", ":lang=ko", "--format=%{file}\n"],
                text=True, timeout=5,
            )
            for line in out.strip().split("\n"):
                p = line.strip()
                if p and Path(p).exists():
                    return p
        except Exception:
            pass
        return None

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", str(audio_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data["format"]["duration"])
        except Exception:
            pass
        return 40.0

    def _get_encoder_settings(self, options: dict) -> dict:
        use_gpu = options.get("use_gpu", True) and gpu_info["nvenc_supported"]
        crf = options.get("crf", cfg.VIDEO["crf"])
        if use_gpu:
            return {
                "codec": "h264_nvenc",
                "extra": ["-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-gpu", "0"],
                "is_gpu": True,
            }
        else:
            return {
                "codec": "libx264",
                "extra": ["-preset", "fast", "-crf", str(crf), "-threads", str(self._cpu_count)],
                "is_gpu": False,
            }

    def _pick_bgm(self, mood: str) -> Optional[str]:
        if mood == "none":
            return None
        mood_dir = cfg.MUSIC_DIR / mood
        files = list(mood_dir.glob("*.mp3")) if mood_dir.exists() else []
        if not files:
            files = list(cfg.MUSIC_DIR.rglob("*.mp3"))
        return str(files[0]) if files else None

    def _generate_text_overlay(
        self, w: int, h: int, text: str, out_path: Path,
        subtitle_settings: dict
    ):
        """Generate subtitle overlay with PIL."""
        from PIL import Image, ImageDraw, ImageFont

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        def load_font(size: int) -> ImageFont.FreeTypeFont:
            if self._font_path:
                try:
                    return ImageFont.truetype(self._font_path, size)
                except Exception:
                    pass
            return ImageFont.load_default()

        font_size = subtitle_settings.get("font_size", 36)
        position = subtitle_settings.get("position", "bottom")
        color_mode = subtitle_settings.get("color_mode", "auto")
        color_hex = subtitle_settings.get("color", "#FFFFFF")
        style = subtitle_settings.get("style", "classic")

        font = load_font(font_size)
        lines = textwrap.wrap(text[:200], width=max(10, int(w / font_size * 1.2)))[:5]

        if not lines:
            lines = [text[:30]]

        # Calculate position
        line_height = font_size + 10
        total_height = len(lines) * line_height

        if position == "top":
            start_y = int(h * 0.08)
        elif position == "center":
            start_y = (h - total_height) // 2
        else:  # bottom
            start_y = int(h * 0.78) - total_height

        # Determine text color
        if color_mode == "auto":
            text_color = (255, 255, 255, 255)
            stroke_color = (0, 0, 0, 200)
        else:
            r_val = int(color_hex[1:3], 16)
            g_val = int(color_hex[3:5], 16)
            b_val = int(color_hex[5:7], 16)
            text_color = (r_val, g_val, b_val, 255)
            stroke_color = (0, 0, 0, 200) if (r_val + g_val + b_val) > 384 else (255, 255, 255, 200)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            tx = (w - tw) // 2
            ty = start_y + i * line_height

            # Background box for classic/cinematic styles
            if style in ("classic", "cinematic", "custom"):
                pad = 12
                bg_alpha = 180 if style == "classic" else 100
                draw.rounded_rectangle(
                    [tx - pad, ty - 6, tx + tw + pad, ty + font_size + 6],
                    radius=8, fill=(0, 0, 0, bg_alpha),
                )

            # Text shadow for pop style
            if style == "pop":
                draw.text((tx + 3, ty + 3), line, font=font, fill=(0, 0, 0, 150))

            # Stroke/outline
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                draw.text((tx + dx, ty + dy), line, font=font, fill=stroke_color)

            # Main text
            draw.text((tx, ty), line, font=font, fill=text_color)

        overlay.save(str(out_path), "PNG")

    def create_from_scenes(
        self, job_id: str, scenes: list[dict], options: dict,
        progress_cb=None, eta_cb=None,
    ) -> Optional[str]:
        """
        Create video from scene descriptions using ComfyUI or fallback.
        This is the v7.0 main pipeline.
        """
        out_dir = cfg.OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        topic = options.get("topic", "AI Shorts")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic[:25]).strip("_") or "shorts"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{ts}_{safe}.mp4"

        # Output format settings
        preset_id = options.get("output_preset", "youtube_shorts")
        preset = cfg.OUTPUT_PRESETS.get(preset_id, cfg.OUTPUT_PRESETS["youtube_shorts"])
        w = options.get("width", preset["width"])
        h = options.get("height", preset["height"])
        fps = options.get("fps", preset["fps"])

        # Quality settings
        quality = options.get("quality", "balance")
        quality_preset = cfg.QUALITY_PRESETS.get(quality, cfg.QUALITY_PRESETS["balance"])
        gen_w = quality_preset["gen_width"]
        gen_h = quality_preset["gen_height"]

        # Subtitle settings
        subtitle_settings = options.get("subtitle", {
            "style": "classic", "position": "bottom",
            "font_size": 36, "color_mode": "auto", "color": "#FFFFFF",
        })

        encoder = self._get_encoder_settings(options)
        total_scenes = len(scenes)
        scene_times = []  # Track time per scene for ETA

        logger.info(
            f"Creating video: {w}x{h}@{fps}fps | {total_scenes} scenes | "
            f"Quality: {quality} | Encoder: {encoder['codec']}"
        )

        scene_clips = []
        for idx, scene in enumerate(scenes):
            scene_start = time.monotonic()
            scene_dir = out_dir / f"scene_{idx:02d}"
            scene_dir.mkdir(exist_ok=True)

            if progress_cb:
                pct = int(idx / total_scenes * 80) + 5
                progress_cb(pct, f"Scene {idx + 1}/{total_scenes} 생성 중")

            # 1. Translate to English prompt
            en_prompt = scene.get("en_prompt", "")
            if not en_prompt:
                en_prompt = scenario_engine.translate_scene_to_prompt(scene)
                scene["en_prompt"] = en_prompt

            logger.info(f"Scene {idx + 1} prompt: {en_prompt[:80]}...")

            # 2. Generate start frame image
            img_path = scene_dir / "start_frame.png"
            comfyui.generate_image(en_prompt, gen_w, gen_h, img_path)

            # 3. Generate AI video from image (or fallback to static)
            scene_duration = options.get("scene_duration", 4.0)
            num_frames = int(scene_duration * fps)
            video_path = scene_dir / "scene_raw.mp4"

            ai_model = options.get("ai_model", "framepack")
            comfyui.generate_video(
                img_path, en_prompt, num_frames, gen_w, gen_h,
                video_path, model=ai_model,
            )

            if not video_path.exists():
                # Create from static image
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                    "-loop", "1", "-i", str(img_path),
                    "-c:v", "libx264", "-t", str(scene_duration),
                    "-pix_fmt", "yuv420p", "-vf", f"scale={gen_w}:{gen_h}",
                    str(video_path),
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)

            # 4. Add subtitle overlay to scene
            narration = scene.get("narration", "")
            if narration:
                sub_path = scene_dir / "subtitle.png"
                self._generate_text_overlay(gen_w, gen_h, narration, sub_path, subtitle_settings)

                scene_with_sub = scene_dir / "scene_sub.mp4"
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                    "-i", str(video_path),
                    "-i", str(sub_path),
                    "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an",
                    str(scene_with_sub),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and scene_with_sub.exists():
                    video_path = scene_with_sub

            scene_clips.append(str(video_path))

            # Track ETA
            scene_time = time.monotonic() - scene_start
            scene_times.append(scene_time)
            if eta_cb and len(scene_times) > 0:
                avg_time = sum(scene_times) / len(scene_times)
                remaining = (total_scenes - idx - 1) * avg_time + 30  # +30s for post-processing
                eta_cb(remaining, avg_time, idx + 1, total_scenes)

        if not scene_clips:
            logger.error("No scene clips generated")
            return None

        # 5. Concatenate all scenes
        if progress_cb:
            progress_cb(85, "장면 연결 중")

        concat_file = out_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for clip in scene_clips:
                f.write(f"file '{clip}'\n")

        concat_output = out_dir / "concat.mp4"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(concat_output),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # 6. Upscale if needed
        if quality_preset.get("upscale") and concat_output.exists():
            if progress_cb:
                progress_cb(88, f"업스케일 {gen_w}p → {w}p")
            upscaled = out_dir / "upscaled.mp4"
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(concat_output),
                "-vf", f"scale={w}:{h}:flags=lanczos",
                "-c:v", encoder["codec"], *encoder["extra"],
                "-pix_fmt", "yuv420p",
                str(upscaled),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and upscaled.exists():
                concat_output = upscaled

        # 7. Generate and merge TTS audio
        if progress_cb:
            progress_cb(90, "TTS 음성 합성 중")

        full_narration = " ".join(s.get("narration", "") for s in scenes if s.get("narration"))
        voice = options.get("voice", "ko-KR-SunHiNeural")
        tts_rate = TTSEngine.RATES.get(options.get("tts_speed", "normal"), "+0%")
        audio_path = out_dir / "narration.mp3"

        tts_ok = tts_engine.synthesize(full_narration, voice, audio_path, rate=tts_rate)

        # 8. Mix audio + BGM + video
        if progress_cb:
            progress_cb(93, "오디오 믹싱")

        bgm_mood = options.get("bgm_mood", "none")
        bgm_path = self._pick_bgm(bgm_mood)

        has_tts = tts_ok and audio_path.exists()
        has_bgm = bgm_path and Path(bgm_path).exists()

        # Build final FFmpeg command
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        cmd += ["-i", str(concat_output)]

        input_idx = 1
        if has_tts:
            cmd += ["-i", str(audio_path)]
            tts_idx = input_idx
            input_idx += 1
        if has_bgm:
            cmd += ["-i", str(bgm_path)]
            bgm_idx = input_idx
            input_idx += 1

        if has_tts and has_bgm:
            af = (
                f"[{tts_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[tts];"
                f"[{bgm_idx}:a]aloop=loop=-1:size=2e+09,aformat=sample_rates=44100:channel_layouts=stereo,"
                f"volume=0.15[bgm];"
                f"[tts][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            cmd += ["-filter_complex", af]
            cmd += ["-map", "0:v", "-map", "[aout]"]
        elif has_tts:
            cmd += ["-map", "0:v", "-map", f"{tts_idx}:a"]
        elif has_bgm:
            af = f"[{bgm_idx}:a]aloop=loop=-1:size=2e+09,volume=0.3[bgm]"
            cmd += ["-filter_complex", af]
            cmd += ["-map", "0:v", "-map", "[bgm]"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
            cmd += ["-map", "0:v", "-map", f"{input_idx}:a"]

        cmd += [
            "-c:v", "copy" if not quality_preset.get("upscale") else encoder["codec"],
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-movflags", "+faststart", "-shortest",
            str(out_file),
        ]

        if progress_cb:
            progress_cb(95, "최종 렌더링")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode != 0:
            logger.error(f"Final render error: {proc.stderr[-500:]}")
            # Fallback: just copy video without audio
            shutil.copy2(str(concat_output), str(out_file))

        # Cleanup temp files
        try:
            for f in [concat_file, concat_output]:
                if f.exists():
                    f.unlink(missing_ok=True)
        except Exception:
            pass

        if out_file.exists() and out_file.stat().st_size > 10_000:
            sz = out_file.stat().st_size / 1024 / 1024
            logger.success(f"Video complete: {out_file.name} ({sz:.1f}MB)")
            return str(out_file)

        logger.error("Output file missing or too small")
        return None

    def create_legacy(
        self, job_id: str, topic: str, script: str,
        audio_path: Optional[Path], options: dict,
        progress_cb=None,
    ) -> Optional[str]:
        """Legacy v6 style video creation (slideshow mode)."""
        out_dir = cfg.OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic[:25]).strip("_") or "shorts"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{ts}_{safe}.mp4"

        preset_id = options.get("output_preset", "youtube_shorts")
        preset = cfg.OUTPUT_PRESETS.get(preset_id, cfg.OUTPUT_PRESETS["youtube_shorts"])
        w = options.get("width", preset["width"])
        h = options.get("height", preset["height"])
        fps = options.get("fps", preset["fps"])

        encoder = self._get_encoder_settings(options)

        try:
            if progress_cb:
                progress_cb(10, "배경 생성 중")

            bg_path = out_dir / "bg.png"
            comfyui._fallback_image("gradient background", w, h, bg_path)

            if progress_cb:
                progress_cb(20, "텍스트 오버레이 생성")

            overlay_path = out_dir / "overlay.png"
            subtitle_settings = options.get("subtitle", {
                "style": "classic", "position": "bottom",
                "font_size": 36, "color_mode": "auto",
            })
            self._generate_text_overlay(w, h, script[:150], overlay_path, subtitle_settings)

            has_audio = audio_path and audio_path.exists() and audio_path.stat().st_size > 500
            duration = self._get_audio_duration(str(audio_path)) + 1.5 if has_audio else 40.0
            duration = min(max(duration, 10.0), 180.0)

            if progress_cb:
                progress_cb(40, "렌더링 중")

            # Build command
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
            cmd += ["-loop", "1", "-i", str(bg_path), "-t", str(duration)]
            cmd += ["-loop", "1", "-i", str(overlay_path), "-t", str(duration)]

            input_idx = 2
            if has_audio:
                cmd += ["-i", str(audio_path)]
                tts_idx = input_idx
                input_idx += 1

            bgm_mood = options.get("bgm_mood", "none")
            bgm_path_str = self._pick_bgm(bgm_mood)
            has_bgm = bgm_path_str and Path(bgm_path_str).exists()
            if has_bgm:
                cmd += ["-i", str(bgm_path_str)]
                bgm_idx = input_idx
                input_idx += 1

            vf = f"[0:v][1:v]overlay=0:0:format=auto,fps={fps}"

            if has_audio and has_bgm:
                af = (
                    f"[{tts_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[tts];"
                    f"[{bgm_idx}:a]aloop=loop=-1:size=2e+09,aformat=sample_rates=44100:channel_layouts=stereo,"
                    f"volume=0.15[bgm];"
                    f"[tts][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                )
                cmd += ["-filter_complex", f"{vf}[vout];{af}"]
                cmd += ["-map", "[vout]", "-map", "[aout]"]
            elif has_audio:
                cmd += ["-filter_complex", f"{vf}[vout]"]
                cmd += ["-map", "[vout]", "-map", f"{tts_idx}:a"]
            else:
                cmd += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}"]
                cmd += ["-filter_complex", f"{vf}[vout]"]
                cmd += ["-map", "[vout]", "-map", f"{input_idx}:a"]

            cmd += ["-c:v", encoder["codec"]] + encoder["extra"]
            cmd += [
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-ar", "44100", "-movflags", "+faststart",
                "-shortest", "-t", str(duration), str(out_file),
            ]

            if progress_cb:
                progress_cb(60, f"인코딩 ({encoder['codec']})")

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if proc.returncode != 0:
                logger.error(f"Render error: {proc.stderr[-500:]}")
                return None

            for tmp in [bg_path, overlay_path]:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

            if out_file.exists() and out_file.stat().st_size > 10_000:
                sz = out_file.stat().st_size / 1024 / 1024
                logger.success(f"Legacy render: {out_file.name} ({sz:.1f}MB)")
                return str(out_file)

        except Exception as e:
            logger.exception(f"Legacy video failed: {e}")
        return None


vid_engine = VideoEngine()


# ═══════════════════════════════════════════════════
#  JOB MANAGER with ETA
# ═══════════════════════════════════════════════════
class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.active: dict[str, threading.Thread] = {}

    def create(self, title, topics, options, source_urls=None, topic_mode="auto") -> str:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        with app.app_context():
            job = Job(
                id=job_id, title=title,
                topics=json.dumps(topics, ensure_ascii=False),
                options=json.dumps(options, ensure_ascii=False),
                source_urls=json.dumps(source_urls or [], ensure_ascii=False),
                topic_mode=topic_mode, llm_model=cfg.LLM["model"],
            )
            db.session.add(job)
            db.session.commit()
        logger.info(f"Job created: {job_id}")
        return job_id

    def update(self, job_id: str, **kw):
        try:
            with app.app_context():
                job = db.session.get(Job, job_id)
                if not job:
                    return
                for k, v in kw.items():
                    if k in ("topics", "options", "output_files", "source_urls", "scenario_data"):
                        setattr(job, k, json.dumps(v, ensure_ascii=False))
                    elif k == "error_message":
                        job.error_message = str(v)[:1000]
                        job.status = "failed"
                    else:
                        setattr(job, k, v)
                job.updated_at = datetime.now(timezone.utc)
                db.session.commit()
                sio.emit("job_update", job.as_dict())
        except Exception as e:
            logger.error(f"job.update [{job_id}]: {e}")

    def get(self, job_id: str) -> Optional[dict]:
        with app.app_context():
            job = db.session.get(Job, job_id)
            return job.as_dict() if job else None

    def recent(self, limit=30) -> list[dict]:
        with app.app_context():
            rows = Job.query.order_by(Job.created_at.desc()).limit(limit).all()
            return [r.as_dict() for r in rows]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its output files."""
        try:
            with app.app_context():
                job = db.session.get(Job, job_id)
                if not job:
                    return False

                # Delete output directory
                out_dir = cfg.OUTPUT_DIR / job_id
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                    logger.info(f"Deleted output: {out_dir}")

                # Delete scripts
                Script.query.filter_by(job_id=job_id).delete()
                db.session.delete(job)
                db.session.commit()
                logger.info(f"Job deleted: {job_id}")
                return True
        except Exception as e:
            logger.error(f"Delete failed [{job_id}]: {e}")
            return False

    def run_v7(self, job_id: str, topic: str, options: dict):
        """v7.0 pipeline: scenario -> scenes -> AI video -> compose."""
        t_start = time.monotonic()

        def prog(pct: int, step: str):
            self.update(job_id, progress=min(pct, 99), current_step=step, message=step)
            sio.emit("pipeline_step", {
                "job_id": job_id, "step": step, "progress": pct,
            })

        def eta(remaining: float, avg_time: float, done: int, total: int):
            sio.emit("eta_update", {
                "job_id": job_id,
                "remaining_seconds": int(remaining),
                "avg_scene_time": round(avg_time, 1),
                "scenes_done": done,
                "scenes_total": total,
                "eta_formatted": _format_eta(remaining),
                "estimated_completion": (
                    datetime.now() + __import__("datetime").timedelta(seconds=remaining)
                ).strftime("%H:%M"),
            })

        try:
            # Step 1: Generate dual scenarios
            prog(5, "시나리오 생성 중 (2개 스타일)")
            num_scenes = options.get("num_scenes", 10)
            duration = options.get("duration", 40)
            style_a = options.get("style_a", "emotional")
            style_b = options.get("style_b", "humor")

            scenarios = scenario_engine.generate_dual_scenarios(
                topic, num_scenes=num_scenes, duration=duration,
                style_a=style_a, style_b=style_b,
            )

            self.update(job_id, scenario_data=scenarios, llm_model=scenarios["scenario_a"]["model_used"])

            # Emit scenarios for user selection
            sio.emit("scenarios_ready", {
                "job_id": job_id,
                "scenarios": scenarios,
            })
            prog(15, "시나리오 생성 완료 — 선택 대기 중")

            # Wait for user selection (with timeout)
            selected = options.get("auto_select", "a")  # Auto-select A for now
            selected_scenario = scenarios[f"scenario_{selected}"]
            scenes = selected_scenario["scenes"]
            full_narration = selected_scenario["narration"]

            # Step 2: Translate scenes to English prompts
            prog(20, "영문 프롬프트 변환 중")
            for i, scene in enumerate(scenes):
                if not scene.get("en_prompt"):
                    scene["en_prompt"] = scenario_engine.translate_scene_to_prompt(scene)
                if (i + 1) % 3 == 0:
                    prog(20 + int(i / len(scenes) * 10), f"프롬프트 변환 {i + 1}/{len(scenes)}")

            # Step 3: Generate video from scenes
            prog(30, "AI 영상 생성 시작")
            options["topic"] = topic

            out_file = vid_engine.create_from_scenes(
                job_id, scenes, options,
                progress_cb=lambda p, s: prog(30 + int(p * 0.6), s),
                eta_cb=eta,
            )

            if out_file:
                sz = round(Path(out_file).stat().st_size / 1024 / 1024, 1)
                rel = str(Path(out_file).relative_to(BASE_DIR))
                outputs = [{
                    "path": rel, "abs_path": out_file,
                    "topic": topic, "size_mb": sz,
                    "model": selected_scenario["model_used"],
                    "scenes": len(scenes),
                    "quality": options.get("quality", "balance"),
                    "preset": options.get("output_preset", "youtube_shorts"),
                }]

                duration_sec = round(time.monotonic() - t_start, 1)
                self.update(
                    job_id, status="completed", progress=100,
                    current_step="완료", output_files=outputs,
                    message=f"완료 ({duration_sec}초)",
                    completed_at=datetime.now(timezone.utc),
                    duration_sec=duration_sec,
                )
                sio.emit("job_completed", {
                    "job_id": job_id, "outputs": outputs, "duration": duration_sec,
                })
                logger.success(f"Job complete: {job_id} | {sz}MB | {duration_sec}s")
            else:
                self.update(job_id, error_message="영상 렌더링 실패")

        except Exception as e:
            logger.exception(f"Pipeline v7 error: {e}")
            self.update(job_id, error_message=str(e))

    def run_legacy(self, job_id: str, topics: list[str], options: dict):
        """Legacy v6 pipeline for quick generation."""
        t_start = time.monotonic()
        outputs = []

        self.update(job_id, status="running", message=f"{len(topics)}개 처리 시작")

        for idx, topic in enumerate(topics):
            def prog(lp: int, step: str):
                overall = int(idx / len(topics) * 100) + int(lp / len(topics))
                self.update(job_id, progress=min(overall, 99), current_step=step,
                            message=f"[{idx + 1}/{len(topics)}] {step}")

            try:
                prog(10, "대본 생성")
                script, model_used = ai.generate(topic)
                self.update(job_id, llm_model=model_used)

                voice = options.get("voice", "ko-KR-SunHiNeural")
                tts_rate = TTSEngine.RATES.get(options.get("tts_speed", "normal"), "+0%")

                with app.app_context():
                    s = Script(
                        id=uuid.uuid4().hex, job_id=job_id,
                        topic=topic, content=script, voice=voice, source_type="ai",
                    )
                    db.session.add(s)
                    db.session.commit()

                prog(30, "TTS 합성")
                audio_dir = cfg.OUTPUT_DIR / job_id
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"tts_{idx:02d}.mp3"
                tts_ok = tts_engine.synthesize(script, voice, audio_path, rate=tts_rate)

                prog(50, "영상 렌더링")
                out_file = vid_engine.create_legacy(
                    job_id, topic, script,
                    audio_path if tts_ok else None, options,
                    progress_cb=lambda p, s: prog(50 + int(p * 0.45), s),
                )

                if out_file:
                    sz = round(Path(out_file).stat().st_size / 1024 / 1024, 1)
                    rel = str(Path(out_file).relative_to(BASE_DIR))
                    outputs.append({"path": rel, "abs_path": out_file, "topic": topic, "size_mb": sz})

            except Exception as e:
                logger.exception(f"Legacy pipeline error: {e}")
                continue

        duration_sec = round(time.monotonic() - t_start, 1)
        self.update(
            job_id, status="completed", progress=100,
            current_step="완료", output_files=outputs,
            message=f"{len(outputs)}/{len(topics)}개 완료 ({duration_sec}초)",
            completed_at=datetime.now(timezone.utc), duration_sec=duration_sec,
        )
        sio.emit("job_completed", {"job_id": job_id, "outputs": outputs, "duration": duration_sec})


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}초"
    elif seconds < 3600:
        return f"{int(seconds // 60)}분 {int(seconds % 60)}초"
    else:
        return f"{int(seconds // 3600)}시간 {int((seconds % 3600) // 60)}분"


jm = JobManager()


# ═══════════════════════════════════════════════════
#  SAMPLE TRENDS & SYSTEM METRICS
# ═══════════════════════════════════════════════════
SAMPLE_TRENDS = [
    {"rank": 1, "title": "ChatGPT o3-mini 실제 사용 후기", "views": "892K", "age": "1일 전", "category": "AI/기술", "hot": True, "icon": "🤖"},
    {"rank": 2, "title": "RTX 5090 vs 4090 성능 비교", "views": "654K", "age": "2일 전", "category": "하드웨어", "hot": True, "icon": "💻"},
    {"rank": 3, "title": "2026 상반기 투자 핫 종목 TOP5", "views": "521K", "age": "1일 전", "category": "재테크", "hot": False, "icon": "💰"},
    {"rank": 4, "title": "집에서 만드는 완벽한 달고나 커피", "views": "478K", "age": "3일 전", "category": "음식", "hot": False, "icon": "☕"},
    {"rank": 5, "title": "삼성 Galaxy S26 카메라 충격 리뷰", "views": "341K", "age": "2일 전", "category": "스마트폰", "hot": False, "icon": "📱"},
    {"rank": 6, "title": "애플 Vision Pro 2 vs 메타 Quest 4", "views": "287K", "age": "1일 전", "category": "XR", "hot": True, "icon": "🥽"},
    {"rank": 7, "title": "웹개발자 연봉 현실 2026", "views": "195K", "age": "4일 전", "category": "IT", "hot": False, "icon": "👨‍💻"},
    {"rank": 8, "title": "다이어트 없이 복부지방 빼는 법", "views": "762K", "age": "2일 전", "category": "건강", "hot": True, "icon": "💪"},
]


def get_system_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE_DIR))
    gpu: dict = {
        "util": 0, "mem_used": 0, "mem_total": 0,
        "temp": 0, "power": 0, "available": False,
        "name": "N/A", "vram_free": 0,
    }
    try:
        raw = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name,memory.free",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL, text=True,
        ).strip().split(",")
        gpu = {
            "util": int(float(raw[0].strip())) if raw[0].strip() not in ("[Not Supported]", "N/A", "") else 0,
            "mem_used": int(raw[1].strip()),
            "mem_total": int(raw[2].strip()),
            "temp": int(float(raw[3].strip())) if raw[3].strip() not in ("[Not Supported]", "N/A", "") else 0,
            "power": round(float(raw[4].strip()), 1) if raw[4].strip() not in ("[Not Supported]", "N/A", "") else 0,
            "available": True,
            "name": raw[5].strip(),
            "vram_free": int(raw[6].strip()),
        }
    except Exception:
        pass
    return {
        "cpu": round(cpu, 1),
        "cpu_freq_mhz": round(cpu_freq.current, 0) if cpu_freq else 0,
        "cpu_cores": psutil.cpu_count(logical=False) or 0,
        "cpu_threads": psutil.cpu_count(logical=True) or 0,
        "ram_pct": round(mem.percent, 1),
        "ram_used_gb": round(mem.used / 1024**3, 1),
        "ram_total_gb": round(mem.total / 1024**3, 1),
        "disk_pct": round(disk.percent, 1),
        "disk_used_gb": round(disk.used / 1024**3, 1),
        "disk_total_gb": round(disk.total / 1024**3, 1),
        "gpu": gpu,
        "workers": len(jm.active),
        "wsl2": gpu_info.get("wsl2", False),
        "comfyui": comfyui.check(),
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(BASE_DIR / "templates", "studio.html")


# ── v7.0 AI Video Generation ──
@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = data.get("topic", "")
        topics = data.get("topics", [])
        options = data.get("options", {})
        mode = data.get("mode", "v7")  # v7 or legacy

        if not topic and not topics:
            return jsonify({"error": "주제를 입력해주세요"}), 400
        if len(jm.active) >= cfg.MAX_JOBS:
            return jsonify({"error": f"동시 작업 한도 초과 (최대 {cfg.MAX_JOBS}개)"}), 429

        if mode == "v7" and topic:
            title = f"AI영상 {datetime.now().strftime('%H:%M')}"
            job_id = jm.create(title, [topic], options, topic_mode="ai_video")

            def _run():
                with jm._lock:
                    jm.active[job_id] = threading.current_thread()
                try:
                    jm.run_v7(job_id, topic, options)
                finally:
                    jm.active.pop(job_id, None)

            t = threading.Thread(target=_run, daemon=True, name=f"job-{job_id}")
            t.start()
            return jsonify({"job_id": job_id, "topic": topic, "mode": "v7", "status": "started"}), 202
        else:
            # Legacy mode
            if not topics:
                topics = [topic] if topic else []
            if not topics:
                return jsonify({"error": "주제를 입력해주세요"}), 400

            title = f"Shorts {datetime.now().strftime('%H:%M')} ({len(topics)}개)"
            job_id = jm.create(title, topics, options, topic_mode="legacy")

            def _run():
                with jm._lock:
                    jm.active[job_id] = threading.current_thread()
                try:
                    jm.run_legacy(job_id, topics, options)
                finally:
                    jm.active.pop(job_id, None)

            t = threading.Thread(target=_run, daemon=True, name=f"job-{job_id}")
            t.start()
            return jsonify({"job_id": job_id, "topics": topics, "mode": "legacy", "status": "started"}), 202

    except Exception as e:
        logger.exception("api_generate exception")
        return jsonify({"error": str(e)}), 500


# ── Scenario Generation ──
@app.route("/api/scenarios", methods=["POST"])
def api_scenarios():
    """Generate dual scenarios for comparison."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = data.get("topic", "")
        num_scenes = data.get("num_scenes", 10)
        duration = data.get("duration", 40)
        style_a = data.get("style_a", "emotional")
        style_b = data.get("style_b", "humor")

        if not topic:
            return jsonify({"error": "주제를 입력해주세요"}), 400

        scenarios = scenario_engine.generate_dual_scenarios(
            topic, num_scenes=num_scenes, duration=duration,
            style_a=style_a, style_b=style_b,
        )
        return jsonify({"success": True, "scenarios": scenarios})

    except Exception as e:
        logger.exception("Scenario generation error")
        return jsonify({"error": str(e)}), 500


# ── Job Management ──
@app.route("/api/jobs")
def api_jobs():
    return jsonify({"jobs": jm.recent(int(request.args.get("limit", 30)))})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = jm.get(job_id)
    return jsonify(job) if job else (jsonify({"error": "not found"}), 404)


@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def api_cancel(job_id):
    jm.update(job_id, status="cancelled", message="사용자 취소")
    return jsonify({"success": True})


@app.route("/api/job/<job_id>/delete", methods=["DELETE"])
def api_delete(job_id):
    """Delete a job and its files."""
    success = jm.delete_job(job_id)
    if success:
        return jsonify({"success": True, "message": f"작업 {job_id} 삭제 완료"})
    return jsonify({"error": "삭제 실패"}), 404


@app.route("/api/jobs/delete-all", methods=["DELETE"])
def api_delete_all():
    """Delete all completed jobs."""
    try:
        with app.app_context():
            jobs = Job.query.filter(Job.status.in_(["completed", "failed", "cancelled"])).all()
            count = 0
            for job in jobs:
                if jm.delete_job(job.id):
                    count += 1
        return jsonify({"success": True, "deleted": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Config & Presets ──
@app.route("/api/output-presets")
def api_output_presets():
    return jsonify({"presets": cfg.OUTPUT_PRESETS})


@app.route("/api/subtitle-styles")
def api_subtitle_styles():
    return jsonify({
        "styles": cfg.SUBTITLE_STYLES,
        "colors": cfg.SUBTITLE_COLORS,
    })


@app.route("/api/quality-presets")
def api_quality_presets():
    return jsonify({"presets": cfg.QUALITY_PRESETS})


@app.route("/api/scenario-styles")
def api_scenario_styles():
    return jsonify({"styles": ScenarioEngine.SCENARIO_STYLES})


@app.route("/api/bgm-moods")
def api_bgm_moods():
    return jsonify({"moods": cfg.BGM_MOODS})


# ── System ──
@app.route("/api/system")
def api_system():
    return jsonify(get_system_metrics())


@app.route("/api/gpu")
def api_gpu():
    return jsonify(GPUDetector.refresh())


@app.route("/api/comfyui/status")
def api_comfyui_status():
    comfyui.reset_cache()
    status = comfyui.get_status()
    # Add ComfyUI process manager status
    status["process_running"] = comfyui_proc.is_running() if comfyui_proc else False
    return jsonify(status)


@app.route("/api/comfyui/logs")
def api_comfyui_logs():
    """Get recent ComfyUI logs."""
    lines = int(request.args.get("lines", 50))
    logs = comfyui_proc.get_logs(lines) if comfyui_proc else []
    return jsonify({"logs": logs})


@app.route("/api/comfyui/start", methods=["POST"])
def api_comfyui_start():
    """Start ComfyUI server."""
    def _start():
        result = comfyui_proc.start()
        sio.emit("comfyui_status", result)
    threading.Thread(target=_start, daemon=True).start()
    return jsonify({"status": "starting"})


@app.route("/api/comfyui/stop", methods=["POST"])
def api_comfyui_stop():
    """Stop ComfyUI server."""
    result = comfyui_proc.stop()
    return jsonify(result)


@app.route("/api/trends")
def api_trends():
    return jsonify({"trends": SAMPLE_TRENDS, "source": "sample"})


@app.route("/api/ollama/models")
def api_ollama_models():
    return jsonify({
        "models": ai.list_models(),
        "current": cfg.LLM["model"],
        "available": ai._check(),
    })


@app.route("/api/ollama/switch", methods=["POST"])
def api_ollama_switch():
    data = request.get_json(force=True, silent=True) or {}
    model = data.get("model", "")
    if not model:
        return jsonify({"error": "모델명 필요"}), 400
    cfg.LLM["model"] = model
    ai.reset_cache()
    return jsonify({"success": True, "model": model})


@app.route("/api/voices")
def api_voices():
    return jsonify({
        "voices": [
            {"id": k, "name": v, "lang": k.split("-")[0].upper()}
            for k, v in TTSEngine.VOICES.items()
        ]
    })


@app.route("/api/music")
def api_music():
    files = []
    for f in sorted(cfg.MUSIC_DIR.rglob("*.mp3")):
        files.append({
            "id": f.stem, "name": f.stem.replace("_", " ").title(),
            "genre": f.parent.name,
            "size_mb": round(f.stat().st_size / 1024**2, 1),
        })
    return jsonify({"music": files})


# ── File Access ──
@app.route("/api/download/<path:filename>")
def api_download(filename):
    from urllib.parse import unquote
    filename = unquote(filename)
    fp = BASE_DIR / filename
    resolved = fp.resolve()
    if not resolved.is_file() or not str(resolved).startswith(str(BASE_DIR.resolve())):
        abort(404)
    return send_file(resolved, as_attachment=True)


@app.route("/api/preview/<path:filename>")
def api_preview(filename):
    from urllib.parse import unquote
    filename = unquote(filename)
    fp = BASE_DIR / filename
    resolved = fp.resolve()
    if not resolved.is_file() or not str(resolved).startswith(str(BASE_DIR.resolve())):
        abort(404)
    ct = "video/mp4"
    if resolved.suffix.lower() == ".mp3":
        ct = "audio/mpeg"
    elif resolved.suffix.lower() == ".webm":
        ct = "video/webm"
    return send_file(resolved, mimetype=ct)


@app.route("/api/version")
def api_version():
    return jsonify({
        "app_version": APP_VERSION,
        "python": sys.version.split(" ")[0],
        "gpu": gpu_info,
        "ollama_available": ai._check(),
        "comfyui_available": comfyui.check(),
        "wsl2": gpu_info.get("wsl2", False),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "healthy",
        "version": APP_VERSION,
        "ollama": ai._check(),
        "comfyui": comfyui.check(),
        "active_jobs": len(jm.active),
        "gpu": gpu_info.get("name", "N/A"),
        "nvenc": gpu_info.get("nvenc_supported", False),
        "wsl2": gpu_info.get("wsl2", False),
        "timestamp": datetime.now().isoformat(),
    })


# ── Error Handlers ──
@app.errorhandler(404)
def err404(e):
    return jsonify({"error": "Not Found"}), 404


@app.errorhandler(500)
def err500(e):
    logger.error(f"500: {e}")
    return jsonify({"error": "Internal Server Error"}), 500


# SocketIO events
@sio.on("connect")
def on_connect():
    emit("connected", {
        "version": APP_VERSION,
        "ollama": ai._check(),
        "comfyui": comfyui.check(),
        "gpu": gpu_info.get("name", "N/A"),
        "nvenc": gpu_info.get("nvenc_supported", False),
    })


@sio.on("disconnect")
def on_disconnect():
    logger.debug(f"Client disconnected: {request.sid}")


@sio.on("ping_metrics")
def on_ping():
    emit("metrics", get_system_metrics())


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"AI Shorts Studio PRO v{APP_VERSION} -> http://0.0.0.0:{port}")
    logger.info(f"   LLM:     {cfg.LLM['model']}")
    logger.info(f"   GPU:     {gpu_info['name']} ({gpu_info['vram_total_mb']}MB VRAM)")
    logger.info(f"   NVENC:   {gpu_info['nvenc_supported']}")
    logger.info(f"   WSL2:    {gpu_info['wsl2']}")
    logger.info(f"   ComfyUI: {comfyui.check()}")
    logger.info(f"   DB:      {DB_PATH}")
    sio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )

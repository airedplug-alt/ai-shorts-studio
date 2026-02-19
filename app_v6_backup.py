#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SHORTS STUDIO PRO v6.0
===========================
Commercial-grade YouTube Shorts auto-generation studio.

Target Environment: Windows 11 WSL2 + Ubuntu
Key Features:
  - URL analysis -> scenario reconstruction/reinterpretation
  - GPU info display (card type, VRAM, driver, CUDA)
  - NVENC/CUDA GPU acceleration + CPU max performance
  - Pure FFmpeg pipeline (no MoviePy)
  - edge-tts async with dedicated event loop
  - Ollama LLM local AI
  - Real-time WebSocket UI
"""
from __future__ import annotations

import asyncio
import json
import os
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
APP_VERSION = "6.0.1"

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
    manage_session=False,  # Fix Flask 3.1+ session compatibility
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
    current_step = db.Column(db.String(80))
    message = db.Column(db.String(500))
    output_files = db.Column(db.Text, default="[]")
    error_message = db.Column(db.Text)
    llm_model = db.Column(db.String(100))
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
    source_type = db.Column(db.String(20), default="ai")  # ai, url_analysis, template
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def as_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "topic": self.topic,
            "content": self.edited or self.content,
            "voice": self.voice,
            "source_type": self.source_type,
        }


def _safe_json(v):
    if not v:
        return []
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
        "max_tok": int(os.getenv("OLLAMA_MAX_TOKENS", "500")),
        "ctx": int(os.getenv("OLLAMA_CONTEXT_SIZE", "4096")),
    }

    MAX_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))

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
    """Detects GPU info: card name, VRAM, driver, CUDA, NVENC support."""

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
            "available": False,
            "name": "N/A",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_free_mb": 0,
            "driver_version": "N/A",
            "cuda_version": "N/A",
            "nvenc_supported": False,
            "utilization": 0,
            "temperature": 0,
            "power_draw": 0,
            "power_limit": 0,
            "fan_speed": 0,
            "pci_bus": "N/A",
            "compute_capability": "N/A",
            "wsl2": cls._is_wsl2(),
            "multi_gpu": [],
        }

        try:
            # nvidia-smi full query
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu="
                    "name,"
                    "memory.total,memory.used,memory.free,"
                    "driver_version,"
                    "utilization.gpu,"
                    "temperature.gpu,"
                    "power.draw,power.limit,"
                    "fan.speed,"
                    "pci.bus_id,"
                    "compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

            gpus = []
            for line in out.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 12:
                    continue
                gpu = {
                    "name": parts[0],
                    "vram_total_mb": int(float(parts[1])),
                    "vram_used_mb": int(float(parts[2])),
                    "vram_free_mb": int(float(parts[3])),
                    "driver_version": parts[4],
                    "utilization": int(float(parts[5])) if parts[5].strip() not in ("[Not Supported]", "N/A", "") else 0,
                    "temperature": int(float(parts[6])) if parts[6].strip() not in ("[Not Supported]", "N/A", "") else 0,
                    "power_draw": round(float(parts[7]), 1) if parts[7].strip() not in ("[Not Supported]", "N/A", "") else 0,
                    "power_limit": round(float(parts[8]), 1) if parts[8].strip() not in ("[Not Supported]", "N/A", "") else 0,
                    "fan_speed": int(float(parts[9])) if parts[9].strip() not in ("[Not Supported]", "N/A", "") else 0,
                    "pci_bus": parts[10],
                    "compute_capability": parts[11],
                }
                gpus.append(gpu)

            if gpus:
                primary = gpus[0]
                info.update(
                    available=True,
                    name=primary["name"],
                    vram_total_mb=primary["vram_total_mb"],
                    vram_used_mb=primary["vram_used_mb"],
                    vram_free_mb=primary["vram_free_mb"],
                    driver_version=primary["driver_version"],
                    utilization=primary["utilization"],
                    temperature=primary["temperature"],
                    power_draw=primary["power_draw"],
                    power_limit=primary["power_limit"],
                    fan_speed=primary["fan_speed"],
                    pci_bus=primary["pci_bus"],
                    compute_capability=primary["compute_capability"],
                    multi_gpu=gpus if len(gpus) > 1 else [],
                )

            # CUDA version
            try:
                cuda_out = subprocess.check_output(
                    ["nvidia-smi"], timeout=5, stderr=subprocess.DEVNULL, text=True
                )
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", cuda_out)
                if m:
                    info["cuda_version"] = m.group(1)
            except Exception:
                pass

            # NVENC support check
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
#  URL ANALYZER — yt-dlp based video analysis
# ═══════════════════════════════════════════════════
class URLAnalyzer:
    """
    Analyzes video URLs (YouTube, etc.) using yt-dlp.
    Extracts metadata, subtitles, description.
    Sends to LLM for scenario reconstruction/reinterpretation.
    """

    REWRITE_STYLES = {
        "expand": {
            "name": "확장 재구성",
            "desc": "원본의 핵심을 유지하면서 살을 덧붙여 풍성하게",
            "prompt": "원본 내용을 기반으로, 빠진 배경 정보와 추가 설명을 넣어 더 풍성한 60초 숏츠 대본으로 확장해주세요.",
        },
        "compress": {
            "name": "압축 핵심",
            "desc": "불필요한 내용 삭제, 핵심만 간결하게",
            "prompt": "원본에서 핵심 정보만 추출하여 30초 분량의 임팩트 있는 숏츠 대본으로 압축해주세요.",
        },
        "humor": {
            "name": "유머 스타일",
            "desc": "재미있고 밈 감성으로 재해석",
            "prompt": "원본 내용을 유머러스하고 밈 감성으로 재해석하여 웃긴 숏츠 대본을 만들어주세요. MZ세대 말투로.",
        },
        "news": {
            "name": "뉴스 앵커",
            "desc": "전문 뉴스 앵커 스타일로 재구성",
            "prompt": "원본 내용을 전문 뉴스 앵커가 브리핑하는 스타일로 재구성해주세요. 신뢰감과 전문성을 살려주세요.",
        },
        "story": {
            "name": "스토리텔링",
            "desc": "이야기하듯 내러티브 형식으로",
            "prompt": "원본 내용을 마치 친구에게 이야기하듯 내러티브 형식의 숏츠 대본으로 만들어주세요. 기승전결 구조로.",
        },
        "debate": {
            "name": "토론/반박",
            "desc": "다른 시각에서 분석하고 반박하는 스타일",
            "prompt": "원본 내용을 비판적 시각에서 분석하고, 다른 관점에서 반박하는 토론형 숏츠 대본을 만들어주세요.",
        },
    }

    @staticmethod
    def extract_metadata(url: str) -> dict:
        """Extract video metadata using yt-dlp."""
        result = {
            "success": False,
            "url": url,
            "title": "",
            "description": "",
            "duration": 0,
            "view_count": 0,
            "like_count": 0,
            "channel": "",
            "upload_date": "",
            "tags": [],
            "categories": [],
            "subtitles_text": "",
            "thumbnail": "",
            "language": "",
            "error": None,
        }

        try:
            # Check if yt-dlp is available
            yt_dlp_cmd = shutil.which("yt-dlp")
            if not yt_dlp_cmd:
                result["error"] = "yt-dlp가 설치되지 않았습니다. install.sh를 실행해주세요."
                return result

            # Extract metadata (no download)
            cmd = [
                yt_dlp_cmd,
                "--dump-json",
                "--no-download",
                "--no-playlist",
                "--socket-timeout", "15",
                url,
            ]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if proc.returncode != 0:
                result["error"] = f"yt-dlp 오류: {proc.stderr[:300]}"
                return result

            data = json.loads(proc.stdout)
            result.update(
                success=True,
                title=data.get("title", ""),
                description=data.get("description", "")[:2000],
                duration=data.get("duration", 0),
                view_count=data.get("view_count", 0),
                like_count=data.get("like_count", 0),
                channel=data.get("channel", data.get("uploader", "")),
                upload_date=data.get("upload_date", ""),
                tags=data.get("tags", [])[:20],
                categories=data.get("categories", []),
                thumbnail=data.get("thumbnail", ""),
                language=data.get("language", ""),
            )

            # Try to get subtitles/captions
            subtitles = data.get("subtitles", {})
            auto_captions = data.get("automatic_captions", {})

            sub_text = ""
            # Prefer Korean subs, then English, then auto
            for lang_pref in ["ko", "ko-KR", "en", "en-US"]:
                if lang_pref in subtitles:
                    sub_text = URLAnalyzer._fetch_subtitle(url, lang_pref, yt_dlp_cmd)
                    break
                if lang_pref in auto_captions:
                    sub_text = URLAnalyzer._fetch_subtitle(url, lang_pref, yt_dlp_cmd, auto=True)
                    break

            result["subtitles_text"] = sub_text[:3000]

        except subprocess.TimeoutExpired:
            result["error"] = "URL 분석 시간 초과 (30초)"
        except json.JSONDecodeError:
            result["error"] = "메타데이터 파싱 실패"
        except Exception as e:
            result["error"] = f"분석 오류: {str(e)[:200]}"

        return result

    @staticmethod
    def _fetch_subtitle(url: str, lang: str, yt_dlp_cmd: str, auto: bool = False) -> str:
        """Fetch subtitle text for a video."""
        try:
            tmp_dir = cfg.CACHE_DIR / "subs"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = tmp_dir / f"sub_{uuid.uuid4().hex[:8]}"

            cmd = [
                yt_dlp_cmd,
                "--skip-download",
                "--write-auto-sub" if auto else "--write-sub",
                "--sub-lang", lang,
                "--sub-format", "vtt",
                "--convert-subs", "srt",
                "-o", str(tmp_file),
                "--no-playlist",
                "--socket-timeout", "10",
                url,
            ]
            subprocess.run(cmd, capture_output=True, timeout=20)

            # Find the subtitle file
            for ext in [".ko.srt", ".ko-KR.srt", ".en.srt", ".en-US.srt", ".srt"]:
                srt_path = Path(str(tmp_file) + ext)
                if srt_path.exists():
                    text = srt_path.read_text(encoding="utf-8", errors="ignore")
                    # Clean SRT formatting
                    text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d+ --> \d{2}:\d{2}:\d{2},\d+\n', '', text)
                    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                    text = re.sub(r'\n{2,}', '\n', text).strip()
                    # Cleanup
                    try:
                        srt_path.unlink()
                    except Exception:
                        pass
                    return text[:3000]
        except Exception as e:
            logger.debug(f"Subtitle fetch failed: {e}")
        return ""

    @staticmethod
    def reconstruct_scenario(
        metadata: dict,
        style: str = "expand",
        ai_engine=None,
    ) -> tuple[str, str]:
        """
        Reconstruct/reinterpret scenario from URL analysis.
        Returns (script, model_used).
        """
        style_info = URLAnalyzer.REWRITE_STYLES.get(style, URLAnalyzer.REWRITE_STYLES["expand"])

        # Build context from metadata
        title = metadata.get("title", "알 수 없음")
        description = metadata.get("description", "")[:500]
        subtitles = metadata.get("subtitles_text", "")[:1000]
        tags = ", ".join(metadata.get("tags", [])[:10])
        channel = metadata.get("channel", "")
        duration = metadata.get("duration", 0)

        prompt = f"""당신은 유튜브 숏츠 전문 시나리오 재구성 작가입니다.

원본 영상 정보:
- 제목: {title}
- 채널: {channel}
- 길이: {duration}초
- 태그: {tags}
- 설명: {description[:300]}
{f'- 자막 내용: {subtitles[:600]}' if subtitles else ''}

스타일 지시: {style_info['prompt']}

규칙:
1. 60초 분량(150~200자) 숏츠 대본
2. 첫 문장은 강렬한 후킹
3. 자연스러운 한국어 구어체 (존댓말)
4. 핵심 정보 2~3가지 간결하게
5. 마지막은 구독/좋아요 유도
6. 대본 텍스트만 출력 (제목/설명/마크다운 없이)
7. 원본과 차별화된 새로운 관점으로 재해석"""

        if ai_engine and ai_engine._check():
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
                text = r.json().get("response", "").strip()
                if len(text) > 20:
                    return text, cfg.LLM["model"]
            except Exception as e:
                logger.warning(f"LLM reconstruction failed: {e}")

        # Fallback: template-based reconstruction
        fallback = (
            f"'{title}' 영상을 분석했습니다! "
            f"핵심 포인트를 정리하면, {description[:100] if description else '흥미로운 내용이 가득합니다'}. "
            f"이 영상에서 놓치면 안 되는 정보, 지금 바로 확인하세요! "
            f"구독과 좋아요 부탁드립니다!"
        )
        return fallback, "template"


url_analyzer = URLAnalyzer()


# ═══════════════════════════════════════════════════
#  AI ENGINE (Ollama)
# ═══════════════════════════════════════════════════
SCRIPT_PROMPT = """당신은 대한민국 유튜브 숏츠 전문 스크립트 작가입니다.
아래 주제로 60초 분량(150~200자)의 대본을 작성하세요.

규칙:
1. 첫 문장은 시청자를 즉시 사로잡는 강렬한 후킹 문장
2. 자연스러운 한국어 구어체 (존댓말)
3. 핵심 정보 2~3가지 간결하게
4. 마지막은 구독/좋아요/댓글 유도
5. 대본 텍스트만 출력 (제목, 설명, 마크다운 없이)

주제: {topic}"""

FALLBACK_SCRIPTS = [
    "{topic}에 대해 알고 계셨나요? 지금 바로 핵심을 알려드립니다! 전문가들도 주목하는 이 내용, 끝까지 봐주시고 구독과 좋아요 꼭 눌러주세요!",
    "오늘 꼭 알아야 할 {topic} 완벽 정리! 최신 트렌드와 핵심 정보를 빠르게 전달해드립니다. 놓치지 마시고 구독 버튼도 눌러주세요!",
    "{topic}, 아직도 모르셨나요? 이 영상 하나로 모든 것을 이해하실 수 있습니다. 좋아요와 구독으로 응원해 주세요!",
]


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

    def _call(self, topic: str, model: str) -> Optional[str]:
        try:
            r = httpx.post(
                f"{cfg.LLM['host']}/api/generate",
                json={
                    "model": model,
                    "prompt": SCRIPT_PROMPT.format(topic=topic),
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
            return text if len(text) > 20 else None
        except Exception as e:
            logger.warning(f"Ollama [{model}] failed: {e}")
            return None

    def generate(self, topic: str) -> tuple[str, str]:
        import random
        if not self._check():
            logger.warning("Ollama not connected -> using template")
            return random.choice(FALLBACK_SCRIPTS).format(topic=topic), "template"
        script = self._call(topic, cfg.LLM["model"])
        if script:
            return script, cfg.LLM["model"]
        script = self._call(topic, cfg.LLM["fallback"])
        if script:
            return script, cfg.LLM["fallback"]
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
        "very_slow": "-30%",
        "slow": "-15%",
        "normal": "+0%",
        "fast": "+15%",
        "very_fast": "+30%",
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
                logger.info(
                    f"TTS done: {out_path.name} ({out_path.stat().st_size // 1024}KB)"
                )
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
#  VIDEO ENGINE v6 — Pure FFmpeg + GPU Acceleration
# ═══════════════════════════════════════════════════
class VideoEngine:
    """
    Pure FFmpeg video engine with GPU acceleration support.
    - Auto-detects NVENC and uses GPU encoding when available
    - Falls back to libx264 CPU encoding
    - Multi-threaded FFmpeg for max CPU performance
    """

    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # WSL2 / Windows fonts
        "/mnt/c/Windows/Fonts/malgunbd.ttf",  # Malgun Gothic Bold
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

    def _pick_bgm(self, genre: str) -> Optional[str]:
        if genre == "none":
            return None
        genre_dir = cfg.MUSIC_DIR / genre
        files = list(genre_dir.glob("*.mp3")) if genre_dir.exists() else []
        if not files:
            files = list(cfg.MUSIC_DIR.rglob("*.mp3"))
        return str(files[0]) if files else None

    def _generate_background(self, w: int, h: int, out_path: Path, style: str = "gradient"):
        """Generate background image with PIL."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), (13, 17, 23))
        draw = ImageDraw.Draw(img)

        if style == "gradient":
            for y in range(h):
                ratio = y / h
                r = int(13 + (25 - 13) * ratio)
                g = int(17 + (35 - 17) * ratio)
                b = int(23 + (60 - 23) * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))

            # Center glow
            cx, cy = w // 2, h // 2
            for radius in range(350, 0, -15):
                alpha_val = max(0, int(20 * (1 - radius / 350)))
                glow_r = int(0 + alpha_val * 0.3)
                glow_g = int(100 + alpha_val * 3)
                glow_b = int(180 + alpha_val * 2)
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=(glow_r, min(255, glow_g), min(255, glow_b)),
                )
        elif style == "warm":
            for y in range(h):
                ratio = y / h
                r = int(30 + (60 - 30) * ratio)
                g = int(15 + (25 - 15) * ratio)
                b = int(15 + (20 - 15) * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))
        elif style == "cool":
            for y in range(h):
                ratio = y / h
                r = int(10 + (15 - 10) * ratio)
                g = int(15 + (30 - 15) * ratio)
                b = int(30 + (70 - 30) * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))

        img.save(str(out_path), "PNG")

    def _generate_overlay(
        self, w: int, h: int, topic: str, script: str, out_path: Path,
        subtitle_style: str = "default"
    ):
        """Generate RGBA text overlay with PIL."""
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

        # Title
        title_font = load_font(48)
        lines = textwrap.wrap(topic, width=14)
        if not lines:
            lines = [topic[:20]]

        box_y = h // 2 - 240
        for i, line in enumerate(lines[:3]):
            bbox = draw.textbbox((0, 0), line, font=title_font)
            tw = bbox[2] - bbox[0]
            tx = (w - tw) // 2
            ty = box_y + i * 70
            pad = 18
            draw.rounded_rectangle(
                [tx - pad, ty - pad, tx + tw + pad, ty + 54 + pad],
                radius=12, fill=(0, 0, 0, 180),
            )
            draw.text((tx + 2, ty + 2), line, font=title_font, fill=(0, 0, 0, 200))
            draw.text((tx, ty), line, font=title_font, fill=(255, 255, 255, 255))

        # Subtitle
        sub_font = load_font(36)
        sub_lines = textwrap.wrap(script[:150], width=18)[:4]
        sub_y = int(h * 0.65)

        for i, line in enumerate(sub_lines):
            bbox = draw.textbbox((0, 0), line, font=sub_font)
            tw = bbox[2] - bbox[0]
            tx = (w - tw) // 2
            ty = sub_y + i * 55
            pad = 12
            draw.rounded_rectangle(
                [tx - pad, ty - 6, tx + tw + pad, ty + 40 + 6],
                radius=8, fill=(0, 0, 0, 160),
            )
            draw.text((tx + 1, ty + 1), line, font=sub_font, fill=(0, 0, 0, 180))
            draw.text((tx, ty), line, font=sub_font, fill=(255, 240, 80, 255))

        # Watermark
        wm_font = load_font(20)
        wm_text = "AI SHORTS STUDIO PRO"
        bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (w - tw - 24, h - 55), wm_text, font=wm_font, fill=(255, 255, 255, 80)
        )

        # Bottom gradient
        for y in range(h - 200, h):
            alpha = int(200 * (y - (h - 200)) / 200)
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, min(200, alpha)))

        overlay.save(str(out_path), "PNG")

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", str(audio_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data["format"]["duration"])
        except Exception as e:
            logger.warning(f"ffprobe duration failed: {e}")
        return 60.0

    def _get_encoder_settings(self, options: dict) -> dict:
        """Determine encoder settings based on GPU availability and user preference."""
        use_gpu = options.get("use_gpu", True) and gpu_info["nvenc_supported"]
        crf = options.get("crf", cfg.VIDEO["crf"])
        preset = options.get("preset", "fast")

        if use_gpu:
            # NVENC GPU encoding
            return {
                "codec": "h264_nvenc",
                "extra": [
                    "-preset", "p4",  # p1(fastest) ~ p7(slowest/best)
                    "-rc", "vbr",
                    "-cq", str(crf),
                    "-b:v", "0",
                    "-gpu", "0",
                ],
                "is_gpu": True,
            }
        else:
            # CPU encoding with maximum performance
            threads = self._cpu_count
            return {
                "codec": "libx264",
                "extra": [
                    "-preset", preset,
                    "-crf", str(crf),
                    "-threads", str(threads),
                ],
                "is_gpu": False,
            }

    def create(
        self,
        job_id: str,
        topic: str,
        script: str,
        audio_path: Optional[Path],
        options: dict,
        progress_cb=None,
    ) -> Optional[str]:
        """Create video using pure FFmpeg pipeline with GPU acceleration."""
        out_dir = cfg.OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        safe = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in topic[:25]).strip("_")
            or "shorts"
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{ts}_{safe}.mp4"

        w = options.get("width", cfg.VIDEO["width"])
        h = options.get("height", cfg.VIDEO["height"])
        fps = options.get("fps", cfg.VIDEO["fps"])
        bg_style = options.get("bg_style", "gradient")

        encoder = self._get_encoder_settings(options)
        logger.info(
            f"Rendering: {w}x{h}@{fps}fps | "
            f"Encoder: {encoder['codec']} | GPU: {encoder['is_gpu']}"
        )

        try:
            if progress_cb:
                progress_cb(10, "배경 생성 중")
            bg_path = out_dir / "bg.png"
            self._generate_background(w, h, bg_path, bg_style)

            if progress_cb:
                progress_cb(20, "텍스트 오버레이 생성")
            overlay_path = out_dir / "overlay.png"
            self._generate_overlay(w, h, topic, script, overlay_path)

            has_audio = (
                audio_path and audio_path.exists() and audio_path.stat().st_size > 500
            )
            if has_audio:
                tts_duration = self._get_audio_duration(str(audio_path))
                duration = min(max(tts_duration + 1.5, 10.0), 180.0)
                logger.info(f"TTS: {tts_duration:.1f}s -> video: {duration:.1f}s")
            else:
                duration = 30.0

            if progress_cb:
                progress_cb(40, "FFmpeg 렌더링 시작")

            bgm_path = self._pick_bgm(options.get("bgm", "none"))
            cmd = self._build_ffmpeg_cmd(
                bg_path=bg_path,
                overlay_path=overlay_path,
                audio_path=audio_path if has_audio else None,
                bgm_path=bgm_path,
                output_path=out_file,
                duration=duration,
                w=w, h=h, fps=fps,
                encoder=encoder,
            )

            logger.debug(f"FFmpeg cmd: {' '.join(cmd)}")

            if progress_cb:
                progress_cb(55, f"인코딩 중 ({encoder['codec']})")

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if proc.returncode != 0:
                logger.error(f"FFmpeg error:\n{proc.stderr[-1000:]}")
                # Fallback to simple CPU
                return self._simple_fallback(
                    out_dir, out_file, bg_path,
                    audio_path if has_audio else None,
                    duration, w, h, fps, options.get("crf", cfg.VIDEO["crf"])
                )

            if progress_cb:
                progress_cb(95, "렌더링 완료")

            if out_file.exists() and out_file.stat().st_size > 10_000:
                sz = out_file.stat().st_size / 1024 / 1024
                logger.success(f"Render complete: {out_file.name} ({sz:.1f}MB)")
                for tmp in [bg_path, overlay_path]:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                return str(out_file)
            else:
                logger.error("Output file missing or too small")
                return None

        except Exception as e:
            logger.exception(f"Video creation failed: {e}")
            return None

    def _build_ffmpeg_cmd(
        self,
        bg_path: Path,
        overlay_path: Path,
        audio_path: Optional[Path],
        bgm_path: Optional[str],
        output_path: Path,
        duration: float,
        w: int, h: int, fps: int,
        encoder: dict,
    ) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]

        # Inputs
        cmd += ["-loop", "1", "-i", str(bg_path), "-t", str(duration)]
        cmd += ["-loop", "1", "-i", str(overlay_path), "-t", str(duration)]

        input_idx = 2
        has_tts = audio_path is not None
        has_bgm = bgm_path is not None and Path(bgm_path).exists()

        if has_tts:
            cmd += ["-i", str(audio_path)]
            tts_idx = input_idx
            input_idx += 1

        if has_bgm:
            cmd += ["-i", str(bgm_path)]
            bgm_idx = input_idx
            input_idx += 1

        # Video filter
        vf = f"[0:v][1:v]overlay=0:0:format=auto,fps={fps}"

        # Audio filter
        if has_tts and has_bgm:
            af = (
                f"[{tts_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[tts];"
                f"[{bgm_idx}:a]aloop=loop=-1:size=2e+09,aformat=sample_rates=44100:channel_layouts=stereo,"
                f"volume=0.15[bgm];"
                f"[tts][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            cmd += ["-filter_complex", f"{vf}[vout];{af}"]
            cmd += ["-map", "[vout]", "-map", "[aout]"]
        elif has_tts:
            cmd += ["-filter_complex", f"{vf}[vout]"]
            cmd += ["-map", "[vout]", "-map", f"{tts_idx}:a"]
        elif has_bgm:
            af = (
                f"[{bgm_idx}:a]aloop=loop=-1:size=2e+09,"
                f"volume=0.3[bgm]"
            )
            cmd += ["-filter_complex", f"{vf}[vout];{af}"]
            cmd += ["-map", "[vout]", "-map", "[bgm]"]
        else:
            cmd += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}"]
            cmd += ["-filter_complex", f"{vf}[vout]"]
            cmd += ["-map", "[vout]", "-map", f"{input_idx}:a"]

        # Encoder
        cmd += ["-c:v", encoder["codec"]]
        cmd += encoder["extra"]
        cmd += [
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-movflags", "+faststart",
            "-shortest",
            "-t", str(duration),
            str(output_path),
        ]
        return cmd

    def _simple_fallback(
        self, out_dir: Path, out_file: Path, bg_path: Path,
        audio_path: Optional[Path], duration: float,
        w: int, h: int, fps: int, crf: int
    ) -> Optional[str]:
        logger.info("Using simple FFmpeg fallback (CPU)")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-loop", "1", "-i", str(bg_path),
        ]
        if audio_path:
            cmd += ["-i", str(audio_path)]
            cmd += ["-map", "0:v", "-map", "1:a"]
        else:
            cmd += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo"]
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-threads", str(self._cpu_count),
            "-shortest", "-t", str(duration),
            str(out_file),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 10_000:
                sz = out_file.stat().st_size / 1024 / 1024
                logger.success(f"Fallback render: {out_file.name} ({sz:.1f}MB)")
                return str(out_file)
        except Exception as e:
            logger.error(f"Fallback render failed: {e}")
        return None


vid_engine = VideoEngine()


# ═══════════════════════════════════════════════════
#  PIPELINE & JOB MANAGER
# ═══════════════════════════════════════════════════
PIPELINE_STEPS = [
    "트렌드 수집",
    "AI 대본 생성",
    "TTS 음성 합성",
    "BGM 믹싱",
    "자막 생성",
    "영상 렌더링",
    "후처리",
    "완료",
]


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.active: dict[str, threading.Thread] = {}

    def create(
        self, title, topics, options, source_urls=None, topic_mode="auto"
    ) -> str:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        with app.app_context():
            job = Job(
                id=job_id,
                title=title,
                topics=json.dumps(topics, ensure_ascii=False),
                options=json.dumps(options, ensure_ascii=False),
                source_urls=json.dumps(source_urls or [], ensure_ascii=False),
                topic_mode=topic_mode,
                llm_model=cfg.LLM["model"],
            )
            db.session.add(job)
            db.session.commit()
        logger.info(f"Job created: {job_id} | {len(topics)} topics")
        return job_id

    def update(self, job_id: str, **kw):
        try:
            with app.app_context():
                job = db.session.get(Job, job_id)
                if not job:
                    return
                for k, v in kw.items():
                    if k in ("topics", "options", "output_files", "source_urls"):
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

    def run(self, job_id: str, topics: list[str], options: dict):
        t_start = time.monotonic()
        total = len(topics)
        outputs = []

        self.update(job_id, status="running", message=f"{total}개 처리 시작")

        for idx, topic in enumerate(topics):
            base_pct = int(idx / total * 100)

            def prog(lp: int, step: str):
                overall = base_pct + int(lp / total)
                self.update(
                    job_id,
                    progress=min(overall, 99),
                    current_step=step,
                    message=f"[{idx + 1}/{total}] {step}",
                )
                sio.emit(
                    "pipeline_step",
                    {
                        "job_id": job_id,
                        "step": step,
                        "progress": overall,
                        "topic_idx": idx,
                    },
                )

            try:
                prog(5, PIPELINE_STEPS[0])

                # Check if this is a URL-analysis job
                url_mode = options.get("url_mode", False)
                rewrite_style = options.get("rewrite_style", "expand")

                if url_mode and options.get("url_metadata"):
                    # URL analysis mode -> reconstruct scenario
                    prog(15, "URL 분석 시나리오 재구성")
                    script, model_used = url_analyzer.reconstruct_scenario(
                        options["url_metadata"],
                        style=rewrite_style,
                        ai_engine=ai,
                    )
                    source_type = "url_analysis"
                else:
                    # Normal AI script generation
                    prog(15, PIPELINE_STEPS[1])
                    script, model_used = ai.generate(topic)
                    source_type = "ai" if model_used != "template" else "template"

                self.update(job_id, llm_model=model_used)
                logger.info(f"Script done ({model_used}): {len(script)} chars")

                voice = options.get("voice", "ko-KR-SunHiNeural")
                tts_rate = TTSEngine.RATES.get(
                    options.get("tts_speed", "normal"), "+0%"
                )

                with app.app_context():
                    s = Script(
                        id=uuid.uuid4().hex,
                        job_id=job_id,
                        topic=topic,
                        content=script,
                        voice=voice,
                        source_type=source_type,
                    )
                    db.session.add(s)
                    db.session.commit()
                    sio.emit(
                        "script_ready",
                        {
                            "job_id": job_id,
                            "topic": topic,
                            "script": script,
                            "script_id": s.id,
                            "model": model_used,
                            "source_type": source_type,
                        },
                    )

                # TTS
                prog(35, PIPELINE_STEPS[2])
                audio_dir = cfg.OUTPUT_DIR / job_id
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"tts_{idx:02d}.mp3"

                tts_ok = tts_engine.synthesize(script, voice, audio_path, rate=tts_rate)
                if not tts_ok:
                    logger.warning("TTS failed -> no audio")
                    audio_path = None

                prog(45, PIPELINE_STEPS[3])
                prog(55, PIPELINE_STEPS[5])

                out_file = vid_engine.create(
                    job_id, topic, script, audio_path, options,
                    progress_cb=lambda p, s: prog(55 + int(p * 0.4), s),
                )

                if out_file:
                    sz = round(Path(out_file).stat().st_size / 1024 / 1024, 1)
                    rel = str(Path(out_file).relative_to(BASE_DIR))
                    outputs.append({
                        "path": rel,
                        "abs_path": out_file,
                        "topic": topic,
                        "size_mb": sz,
                        "model": model_used,
                    })
                    logger.success(f"Done [{idx + 1}/{total}]: {Path(out_file).name} ({sz}MB)")
                else:
                    logger.error(f"Render failed: {topic}")
                    sio.emit("job_error", {"job_id": job_id, "message": f"렌더링 실패: {topic[:30]}"})

            except Exception as e:
                logger.exception(f"Pipeline error: {e}")
                sio.emit("job_error", {"job_id": job_id, "message": str(e)})
                continue

        duration_sec = round(time.monotonic() - t_start, 1)
        self.update(
            job_id,
            status="completed",
            progress=100,
            current_step="완료",
            message=f"{len(outputs)}/{total}개 완료 ({duration_sec}초)",
            output_files=outputs,
            completed_at=datetime.now(timezone.utc),
            duration_sec=duration_sec,
        )
        sio.emit("job_completed", {
            "job_id": job_id, "outputs": outputs, "duration": duration_sec,
        })
        logger.success(f"Job complete: {job_id} | {len(outputs)} videos | {duration_sec}s")


jm = JobManager()


# ═══════════════════════════════════════════════════
#  SAMPLE TRENDS
# ═══════════════════════════════════════════════════
SAMPLE_TRENDS = [
    {"rank": 1, "title": "ChatGPT o3-mini 실제 사용 후기", "views": "892K", "age": "1일 전", "category": "AI/기술", "hot": True, "icon": "🤖"},
    {"rank": 2, "title": "RTX 5090 vs 4090 성능 비교 벤치마크", "views": "654K", "age": "2일 전", "category": "하드웨어", "hot": True, "icon": "💻"},
    {"rank": 3, "title": "2026 상반기 투자 핫 종목 TOP5", "views": "521K", "age": "1일 전", "category": "재테크", "hot": False, "icon": "💰"},
    {"rank": 4, "title": "집에서 만드는 완벽한 달고나 커피", "views": "478K", "age": "3일 전", "category": "음식", "hot": False, "icon": "☕"},
    {"rank": 5, "title": "삼성 Galaxy S26 카메라 충격 리뷰", "views": "341K", "age": "2일 전", "category": "스마트폰", "hot": False, "icon": "📱"},
    {"rank": 6, "title": "애플 Vision Pro 2 vs 메타 Quest 4", "views": "287K", "age": "1일 전", "category": "XR", "hot": True, "icon": "🥽"},
    {"rank": 7, "title": "웹개발자 연봉 현실 2026", "views": "195K", "age": "4일 전", "category": "IT", "hot": False, "icon": "👨‍💻"},
    {"rank": 8, "title": "다이어트 없이 복부지방 빼는 법", "views": "762K", "age": "2일 전", "category": "건강", "hot": True, "icon": "💪"},
]


# ═══════════════════════════════════════════════════
#  SYSTEM METRICS (Enhanced for WSL2)
# ═══════════════════════════════════════════════════
def get_system_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE_DIR))

    # Real-time GPU metrics
    gpu: dict = {
        "util": 0, "mem_used": 0, "mem_total": 0,
        "temp": 0, "power": 0, "available": False,
        "name": "N/A", "vram_free": 0,
    }
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=3, stderr=subprocess.DEVNULL, text=True,
        ).strip().split(",")
        gpu = {
            "util": int(raw[0].strip()) if raw[0].strip() not in ("[Not Supported]", "N/A", "") else 0,
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
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(BASE_DIR / "templates", "studio.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        data = request.get_json(force=True, silent=True) or {}
        topics = data.get("topics", [])
        options = data.get("options", {})
        source_urls = data.get("source_urls", [])
        topic_mode = data.get("topic_mode", "auto")

        if not topics:
            return jsonify({"error": "주제를 1개 이상 선택해주세요"}), 400
        if len(topics) > 10:
            return jsonify({"error": "주제는 최대 10개"}), 400
        if len(jm.active) >= cfg.MAX_JOBS:
            return jsonify({"error": f"동시 작업 한도 초과 (최대 {cfg.MAX_JOBS}개)"}), 429

        title = f"Shorts {datetime.now().strftime('%H:%M')} ({len(topics)}개)"
        job_id = jm.create(title, topics, options, source_urls, topic_mode)

        def _run():
            with jm._lock:
                jm.active[job_id] = threading.current_thread()
            try:
                jm.run(job_id, topics, options)
            finally:
                jm.active.pop(job_id, None)

        t = threading.Thread(target=_run, daemon=True, name=f"job-{job_id}")
        t.start()
        return jsonify({"job_id": job_id, "topics": topics, "status": "started"}), 202

    except Exception as e:
        logger.exception("api_generate exception")
        return jsonify({"error": str(e)}), 500


# ── URL Analysis API ──
@app.route("/api/analyze-url", methods=["POST"])
def api_analyze_url():
    """Analyze a video URL and return metadata + rewrite options."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "URL을 입력해주세요"}), 400

        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return jsonify({"error": "유효한 URL을 입력해주세요"}), 400

        logger.info(f"Analyzing URL: {url}")
        metadata = url_analyzer.extract_metadata(url)

        if not metadata["success"]:
            return jsonify({"error": metadata.get("error", "분석 실패")}), 400

        return jsonify({
            "success": True,
            "metadata": metadata,
            "rewrite_styles": {
                k: {"name": v["name"], "desc": v["desc"]}
                for k, v in URLAnalyzer.REWRITE_STYLES.items()
            },
        })

    except Exception as e:
        logger.exception("URL analysis error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-from-url", methods=["POST"])
def api_generate_from_url():
    """Generate shorts from a URL analysis with selected rewrite style."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        metadata = data.get("metadata", {})
        style = data.get("style", "expand")
        options = data.get("options", {})

        if not metadata.get("title"):
            return jsonify({"error": "URL 분석 데이터가 없습니다. 먼저 URL을 분석해주세요."}), 400

        # Inject URL metadata into options
        options["url_mode"] = True
        options["url_metadata"] = metadata
        options["rewrite_style"] = style

        topic = f"[{URLAnalyzer.REWRITE_STYLES.get(style, {}).get('name', style)}] {metadata['title'][:50]}"
        topics = [topic]

        if len(jm.active) >= cfg.MAX_JOBS:
            return jsonify({"error": f"동시 작업 한도 초과 (최대 {cfg.MAX_JOBS}개)"}), 429

        title = f"URL재해석 {datetime.now().strftime('%H:%M')}"
        job_id = jm.create(title, topics, options, [metadata.get("url", "")], "url")

        def _run():
            with jm._lock:
                jm.active[job_id] = threading.current_thread()
            try:
                jm.run(job_id, topics, options)
            finally:
                jm.active.pop(job_id, None)

        t = threading.Thread(target=_run, daemon=True, name=f"job-{job_id}")
        t.start()
        return jsonify({"job_id": job_id, "topic": topic, "style": style, "status": "started"}), 202

    except Exception as e:
        logger.exception("generate-from-url exception")
        return jsonify({"error": str(e)}), 500


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


@app.route("/api/job/<job_id>/scripts")
def api_scripts(job_id):
    with app.app_context():
        scripts = Script.query.filter_by(job_id=job_id).all()
        return jsonify({"scripts": [s.as_dict() for s in scripts]})


@app.route("/api/script/<script_id>", methods=["PUT"])
def api_update_script(script_id):
    data = request.get_json(force=True, silent=True) or {}
    with app.app_context():
        s = db.session.get(Script, script_id)
        if not s:
            return jsonify({"error": "not found"}), 404
        s.edited = data.get("content", s.content)
        db.session.commit()
    return jsonify({"success": True})


@app.route("/api/system")
def api_system():
    return jsonify(get_system_metrics())


@app.route("/api/gpu")
def api_gpu():
    """Detailed GPU information endpoint."""
    return jsonify(GPUDetector.refresh())


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
            "id": f.stem,
            "name": f.stem.replace("_", " ").title(),
            "genre": f.parent.name,
            "size_mb": round(f.stat().st_size / 1024**2, 1),
        })
    return jsonify({"music": files})


@app.route("/api/presets")
def api_presets():
    return jsonify({
        "presets": [
            {"id": "4k_ultra", "name": "4K Ultra", "w": 2160, "h": 3840, "fps": 60, "crf": 16, "codec": "libx264"},
            {"id": "hd_high", "name": "HD High", "w": 1080, "h": 1920, "fps": 30, "crf": 18, "codec": "libx264"},
            {"id": "hd_standard", "name": "HD Standard", "w": 1080, "h": 1920, "fps": 30, "crf": 23, "codec": "libx264"},
            {"id": "yt_shorts", "name": "YouTube Shorts", "w": 1080, "h": 1920, "fps": 30, "crf": 23, "codec": "libx264"},
            {"id": "fast_draft", "name": "Fast Draft", "w": 720, "h": 1280, "fps": 24, "crf": 28, "codec": "libx264"},
            {"id": "tiktok", "name": "TikTok", "w": 1080, "h": 1920, "fps": 30, "crf": 22, "codec": "libx264"},
        ]
    })


@app.route("/api/download/<path:filename>")
def api_download(filename):
    from urllib.parse import unquote
    filename = unquote(filename)
    fp = BASE_DIR / filename
    resolved = fp.resolve()
    if not resolved.is_file() or not str(resolved).startswith(str(BASE_DIR.resolve())):
        logger.warning(f"Download 404: {filename} -> {resolved}")
        abort(404)
    return send_file(resolved, as_attachment=True)


@app.route("/api/preview/<path:filename>")
def api_preview(filename):
    from urllib.parse import unquote
    filename = unquote(filename)
    fp = BASE_DIR / filename
    resolved = fp.resolve()
    if not resolved.is_file() or not str(resolved).startswith(str(BASE_DIR.resolve())):
        logger.warning(f"Preview 404: {filename} -> {resolved}")
        abort(404)
    # Determine content type
    ct = "video/mp4"
    if resolved.suffix.lower() == ".mp3":
        ct = "audio/mpeg"
    elif resolved.suffix.lower() == ".webm":
        ct = "video/webm"
    return send_file(resolved, mimetype=ct)


@app.route("/api/version")
def api_version():
    import importlib.metadata as im
    pkgs = ["flask", "edge-tts", "Pillow", "flask-socketio", "httpx", "psutil", "yt-dlp"]
    info = {}
    for p in pkgs:
        try:
            info[p] = im.version(p)
        except Exception:
            info[p] = "N/A"

    ffmpeg_ver = "N/A"
    try:
        ffmpeg_ver = subprocess.check_output(
            ["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True
        ).split("\n")[0]
    except Exception:
        pass

    return jsonify({
        "app_version": APP_VERSION,
        "python": sys.version.split(" ")[0],
        "ffmpeg": ffmpeg_ver,
        "gpu": gpu_info,
        "ollama_models": ai.list_models(),
        "packages": info,
        "wsl2": gpu_info.get("wsl2", False),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "healthy",
        "version": APP_VERSION,
        "ollama": ai._check(),
        "active_jobs": len(jm.active),
        "gpu": gpu_info.get("name", "N/A"),
        "nvenc": gpu_info.get("nvenc_supported", False),
        "wsl2": gpu_info.get("wsl2", False),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/rewrite-styles")
def api_rewrite_styles():
    """List available URL rewrite styles."""
    return jsonify({
        "styles": {
            k: {"name": v["name"], "desc": v["desc"]}
            for k, v in URLAnalyzer.REWRITE_STYLES.items()
        }
    })


# Error handlers
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
    logger.info(f"   LLM:   {cfg.LLM['model']}")
    logger.info(f"   GPU:   {gpu_info['name']} ({gpu_info['vram_total_mb']}MB VRAM)")
    logger.info(f"   NVENC: {gpu_info['nvenc_supported']}")
    logger.info(f"   WSL2:  {gpu_info['wsl2']}")
    logger.info(f"   DB:    {DB_PATH}")
    sio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )

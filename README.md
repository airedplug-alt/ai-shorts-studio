# AI SHORTS STUDIO PRO v1.0.0

© chally.choi — All Rights Reserved

Commercial-grade YouTube Shorts auto-generation studio with ComfyUI AI video generation engine.

## Overview
- **Name**: AI Shorts Studio PRO
- **Version**: 1.0.0
- **Author**: chally.choi
- **Target**: Windows 11 WSL2 + Ubuntu + RTX 3090 (24GB VRAM)
- **License**: Private

## Key Features

### AI Video Generation Engine
- **ComfyUI Integration**: FramePack + Wan2.1 14B AI video generation
- **Dynamic Checkpoint Detection**: Auto-discovers installed models
- **Dual Scenario**: 2 different style scenarios generated for comparison
- **Scene Breakdown**: 6-axis detail (composition, camera, lighting, action, background, mood)
- **Korean -> English Prompt**: Auto-translation via Ollama LLM
- **Quality Options**: 480p + AI upscale or 720p direct

### Unified Launch
- **Single start.sh**: ComfyUI + Flask launched together
- **Auto ComfyUI detection**: Finds and starts ComfyUI automatically
- **Ctrl+C cleanup**: All processes stopped cleanly

### Real-time Monitoring
- **Progress Panel**: Scene-by-scene progress with ETA
- **ComfyUI Log Console**: Live log streaming in UI right panel
- **System Monitor**: GPU/CPU/VRAM/Temperature tracking
- **WebSocket**: Real-time updates via Socket.IO

### Output Format Presets
| Platform | Resolution | Ratio | Max Duration |
|---|---|---|---|
| YouTube Shorts | 1080x1920 | 9:16 | 60s |
| Instagram Reels | 1080x1920 | 9:16 | 90s |
| TikTok | 1080x1920 | 9:16 | 60s |
| X (Twitter) | 1080x1080 | 1:1 | 140s |
| YouTube Standard | 1920x1080 | 16:9 | Unlimited |
| Instagram Feed | 1080x1350 | 4:5 | 60s |

## Quick Start

### Installation
```bash
git clone https://github.com/airedplug-alt/ai-shorts-studio.git
cd ai-shorts-studio
./install.sh           # Base install (Python, FFmpeg, dependencies)
./install.sh comfyui   # Install ComfyUI + FramePack + Wan2.1 (~50GB)
```

### Checkpoint Model (Required for AI images)
```bash
cd comfyui/ComfyUI/models/checkpoints/
wget -O sd_xl_base_1.0.safetensors \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
```

### Run
```bash
./start.sh    # Starts ComfyUI + Flask in one command
# Open http://localhost:5000
# Ctrl+C to stop everything
```

## Architecture
```
ai-shorts-studio/
├── app.py              # Flask + ComfyUI engine (main application)
├── start.sh            # Unified launcher (ComfyUI + Flask)
├── install.sh          # Auto-installer
├── templates/
│   └── studio.html     # Full UI (single-page app)
├── static/             # Static assets
├── comfyui/ComfyUI/    # ComfyUI installation
├── output/             # Generated videos
├── music/              # BGM files
├── fonts/              # Korean fonts
├── logs/               # Application logs
└── data/               # SQLite database
```

## API Endpoints
| Method | Path | Description |
|---|---|---|
| GET | / | Web UI |
| POST | /api/generate | Start video generation |
| POST | /api/scenarios | Generate dual scenarios |
| GET | /api/jobs | List jobs |
| GET | /api/comfyui/status | ComfyUI status + checkpoints |
| GET | /api/comfyui/logs | ComfyUI log history |
| GET | /api/system | System metrics |
| GET | /api/health | Health check |

## Requirements
- Python 3.9+
- FFmpeg with NVENC support (recommended)
- NVIDIA GPU with 12GB+ VRAM (recommended: 24GB)
- Ollama for AI script generation (optional)
- ComfyUI for AI video generation (optional, fallback available)

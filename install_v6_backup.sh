#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# AI SHORTS STUDIO PRO v6.0 - 자동 설치 스크립트
# ═══════════════════════════════════════════════════════════════
# 지원 환경: Ubuntu 20.04+ / WSL2 Ubuntu / Debian 11+
# 자동 감지: WSL2, GPU, NVENC, Python, FFmpeg, 한글 폰트
# 에러 핸들링: 각 단계별 진단 + 자동 복구 시도 + 상세 안내
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# ── Config ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/install.log"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON_MIN="3.9"
REQUIRED_SPACE_MB=2000  # 2GB minimum

# ── Logging ──
log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
ok()  { echo -e "${GREEN}  ✅ $1${NC}" | tee -a "$LOG_FILE"; }
warn(){ echo -e "${YELLOW}  ⚠️  $1${NC}" | tee -a "$LOG_FILE"; }
err() { echo -e "${RED}  ❌ $1${NC}" | tee -a "$LOG_FILE"; }
info(){ echo -e "${PURPLE}  ℹ️  $1${NC}" | tee -a "$LOG_FILE"; }
line(){ echo -e "${CYAN}$(printf '═%.0s' {1..60})${NC}"; }

# ── Error handler ──
trap 'on_error $? $LINENO' ERR

on_error() {
    local exit_code=$1
    local line_no=$2
    err "설치 중 오류 발생 (종료코드: $exit_code, 줄: $line_no)"
    echo ""
    echo -e "${RED}${BOLD}🔧 문제 해결 방법:${NC}"
    echo -e "  1. 로그 파일 확인: ${CYAN}cat $LOG_FILE${NC}"
    echo -e "  2. 시스템 업데이트: ${CYAN}sudo apt update && sudo apt upgrade -y${NC}"
    echo -e "  3. 재시도: ${CYAN}./install.sh${NC}"
    echo -e "  4. 클린 재설치: ${CYAN}rm -rf venv && ./install.sh${NC}"
    echo ""
    echo -e "  문제가 계속되면 아래 정보를 포함하여 이슈를 등록해주세요:"
    echo -e "  - OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2)"
    echo -e "  - Python: $(python3 --version 2>/dev/null || echo 'not found')"
    echo -e "  - 로그: $LOG_FILE"
    exit $exit_code
}

# ═══════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   AI SHORTS STUDIO PRO v6.0 INSTALLER     ║"
echo "  ║   YouTube Shorts 자동 생성 스튜디오        ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${NC}"
echo "" > "$LOG_FILE"
log "설치 시작: $(date)"

# ═══════════════════════════════════════════
# STEP 1: 환경 감지
# ═══════════════════════════════════════════
line
log "${BOLD}[1/8] 환경 감지${NC}"

# WSL2 감지
IS_WSL2=false
if grep -qi "microsoft" /proc/version 2>/dev/null; then
    IS_WSL2=true
    ok "WSL2 환경 감지됨"
else
    info "네이티브 Linux 환경"
fi

# OS 정보
if [ -f /etc/os-release ]; then
    . /etc/os-release
    ok "OS: $PRETTY_NAME"
else
    warn "OS 정보를 확인할 수 없습니다"
fi

# 아키텍처
ARCH=$(uname -m)
ok "아키텍처: $ARCH"

# 디스크 공간 확인
AVAIL_MB=$(df -BM "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/M//')
if [ "${AVAIL_MB:-0}" -lt "$REQUIRED_SPACE_MB" ]; then
    err "디스크 공간 부족: ${AVAIL_MB}MB (최소 ${REQUIRED_SPACE_MB}MB 필요)"
    echo -e "  ${YELLOW}해결: 불필요한 파일 삭제 후 재시도${NC}"
    exit 1
fi
ok "디스크 공간: ${AVAIL_MB}MB 사용 가능"

# ═══════════════════════════════════════════
# STEP 2: 시스템 패키지
# ═══════════════════════════════════════════
line
log "${BOLD}[2/8] 시스템 패키지 설치${NC}"

# sudo 확인
if ! command -v sudo &>/dev/null; then
    err "sudo가 설치되어 있지 않습니다"
    info "root 사용자라면: apt install sudo"
    exit 1
fi

# APT 업데이트
log "  패키지 목록 업데이트 중..."
if sudo apt-get update -qq >> "$LOG_FILE" 2>&1; then
    ok "패키지 목록 업데이트 완료"
else
    warn "패키지 목록 업데이트 실패 - 계속 진행합니다"
    info "수동 실행: sudo apt update"
fi

# 필수 패키지 목록
PACKAGES=(
    "python3"
    "python3-pip"
    "python3-venv"
    "python3-dev"
    "ffmpeg"
    "fonts-nanum"
    "fonts-noto-cjk"
    "build-essential"
    "curl"
    "git"
)

log "  필수 패키지 설치 중..."
FAILED_PKGS=()
for pkg in "${PACKAGES[@]}"; do
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        continue  # Already installed
    fi
    if sudo apt-get install -y -qq "$pkg" >> "$LOG_FILE" 2>&1; then
        ok "$pkg 설치됨"
    else
        FAILED_PKGS+=("$pkg")
        warn "$pkg 설치 실패"
    fi
done

if [ ${#FAILED_PKGS[@]} -gt 0 ]; then
    warn "일부 패키지 설치 실패: ${FAILED_PKGS[*]}"
    info "수동 설치: sudo apt install ${FAILED_PKGS[*]}"
    info "계속 진행합니다..."
fi

# ═══════════════════════════════════════════
# STEP 3: Python 버전 확인
# ═══════════════════════════════════════════
line
log "${BOLD}[3/8] Python 확인${NC}"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$($cmd --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        min_minor=$(echo "$PYTHON_MIN" | cut -d. -f2)

        if [ "$major" -ge 3 ] && [ "$minor" -ge "$min_minor" ]; then
            PYTHON_CMD="$cmd"
            ok "Python $ver ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    err "Python $PYTHON_MIN 이상이 필요합니다"
    echo ""
    echo -e "  ${YELLOW}${BOLD}해결 방법:${NC}"
    echo -e "  ${CYAN}sudo apt install python3.11 python3.11-venv python3.11-dev${NC}"
    echo -e "  또는"
    echo -e "  ${CYAN}sudo add-apt-repository ppa:deadsnakes/ppa${NC}"
    echo -e "  ${CYAN}sudo apt update && sudo apt install python3.11${NC}"
    exit 1
fi

# pip 확인
if ! $PYTHON_CMD -m pip --version &>/dev/null; then
    warn "pip가 없습니다. 설치 시도..."
    if sudo apt-get install -y -qq python3-pip >> "$LOG_FILE" 2>&1; then
        ok "pip 설치 완료"
    else
        err "pip 설치 실패"
        info "수동 설치: curl https://bootstrap.pypa.io/get-pip.py | python3"
        exit 1
    fi
fi

# ═══════════════════════════════════════════
# STEP 4: Python 가상 환경
# ═══════════════════════════════════════════
line
log "${BOLD}[4/8] Python 가상 환경 설정${NC}"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    ok "기존 가상 환경 사용: $VENV_DIR"
else
    log "  가상 환경 생성 중..."
    if $PYTHON_CMD -m venv "$VENV_DIR" >> "$LOG_FILE" 2>&1; then
        ok "가상 환경 생성 완료"
    else
        err "가상 환경 생성 실패"
        echo ""
        echo -e "  ${YELLOW}해결 방법:${NC}"
        echo -e "  ${CYAN}sudo apt install python3-venv${NC}"
        echo -e "  ${CYAN}rm -rf $VENV_DIR && ./install.sh${NC}"
        exit 1
    fi
fi

# 활성화
source "$VENV_DIR/bin/activate"
ok "가상 환경 활성화됨: $(python --version)"

# pip 업그레이드
log "  pip 업그레이드 중..."
python -m pip install --upgrade pip setuptools wheel >> "$LOG_FILE" 2>&1 && ok "pip 업그레이드 완료" || warn "pip 업그레이드 실패 (계속)"

# ═══════════════════════════════════════════
# STEP 5: Python 패키지 설치
# ═══════════════════════════════════════════
line
log "${BOLD}[5/8] Python 패키지 설치${NC}"

# requirements.txt 생성
cat > "${SCRIPT_DIR}/requirements.txt" << 'REQEOF'
# AI Shorts Studio PRO v6.0 Dependencies
flask>=3.0.0
flask-socketio>=5.3.0
flask-cors>=4.0.0
flask-sqlalchemy>=3.1.0
simple-websocket>=1.0.0

# AI & TTS
edge-tts>=6.1.0
httpx>=0.25.0

# Image Processing
Pillow>=10.0.0

# System Monitoring
psutil>=5.9.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0

# Video Analysis
yt-dlp>=2024.1.0
REQEOF

ok "requirements.txt 생성됨"

log "  패키지 설치 중... (1~3분 소요)"
if pip install -r "${SCRIPT_DIR}/requirements.txt" >> "$LOG_FILE" 2>&1; then
    ok "모든 Python 패키지 설치 완료"
else
    err "일부 패키지 설치 실패"
    warn "개별 설치를 시도합니다..."

    # 개별 설치 fallback
    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [ -z "$line" ] && continue
        pkg_name=$(echo "$line" | cut -d'>' -f1 | cut -d'=' -f1)
        if pip install "$line" >> "$LOG_FILE" 2>&1; then
            ok "$pkg_name"
        else
            err "$pkg_name 설치 실패"
            info "수동 설치: pip install $line"
        fi
    done < "${SCRIPT_DIR}/requirements.txt"
fi

# 설치 검증
log "  패키지 설치 검증 중..."
VERIFY_PKGS=("flask" "flask_socketio" "edge_tts" "PIL" "psutil" "httpx" "loguru")
ALL_OK=true
for pkg in "${VERIFY_PKGS[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        ok "$pkg ✓"
    else
        err "$pkg 로드 실패"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    warn "일부 패키지 검증 실패"
    info "수동 설치 후 재시도: pip install -r requirements.txt"
fi

# ═══════════════════════════════════════════
# STEP 6: FFmpeg 확인
# ═══════════════════════════════════════════
line
log "${BOLD}[6/8] FFmpeg 확인${NC}"

if command -v ffmpeg &>/dev/null; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1)
    ok "$FFMPEG_VER"

    # NVENC 지원 확인
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then
        ok "NVENC (h264_nvenc) 지원 ✓"
    else
        info "NVENC 미지원 → CPU 인코딩 사용"
        if [ "$IS_WSL2" = true ]; then
            echo ""
            echo -e "  ${YELLOW}${BOLD}WSL2에서 NVENC 사용하기:${NC}"
            echo -e "  1. Windows에 최신 NVIDIA 드라이버 설치"
            echo -e "  2. WSL2용 CUDA Toolkit 설치:"
            echo -e "     ${CYAN}https://developer.nvidia.com/cuda-wsl${NC}"
            echo -e "  3. FFmpeg NVENC 빌드 또는:"
            echo -e "     ${CYAN}sudo apt install ffmpeg${NC} (일부 버전 지원)"
        fi
    fi

    # ffprobe 확인
    if command -v ffprobe &>/dev/null; then
        ok "ffprobe ✓"
    else
        warn "ffprobe 미발견"
        info "sudo apt install ffmpeg 로 재설치"
    fi
else
    err "FFmpeg가 설치되지 않았습니다"
    echo ""
    echo -e "  ${CYAN}sudo apt install ffmpeg${NC}"
    echo ""
    # 자동 설치 시도
    log "  FFmpeg 자동 설치 시도..."
    if sudo apt-get install -y ffmpeg >> "$LOG_FILE" 2>&1; then
        ok "FFmpeg 설치 완료"
    else
        err "FFmpeg 자동 설치 실패"
        info "수동 설치: sudo apt install ffmpeg"
    fi
fi

# ═══════════════════════════════════════════
# STEP 7: GPU / NVIDIA 감지
# ═══════════════════════════════════════════
line
log "${BOLD}[7/8] GPU 감지${NC}"

GPU_DETECTED=false
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        GPU_DETECTED=true
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_VRAM=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        GPU_DRIVER=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
        ok "GPU: $GPU_NAME"
        ok "VRAM: $GPU_VRAM"
        ok "Driver: $GPU_DRIVER"

        # CUDA 확인
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $NF}' || echo "N/A")
        ok "CUDA: $CUDA_VER"
    else
        warn "nvidia-smi 실행 실패"
    fi
else
    info "NVIDIA GPU 미감지"
    info "CPU 인코딩 모드로 작동합니다"
    echo ""
    if [ "$IS_WSL2" = true ]; then
        echo -e "  ${YELLOW}WSL2에서 GPU 사용하기:${NC}"
        echo -e "  1. Windows에 최신 NVIDIA 드라이버 설치 (Game Ready 또는 Studio)"
        echo -e "  2. WSL2 재시작: ${CYAN}wsl --shutdown${NC} 후 다시 실행"
        echo -e "  3. 확인: ${CYAN}nvidia-smi${NC}"
    else
        echo -e "  ${YELLOW}GPU 사용하기:${NC}"
        echo -e "  1. NVIDIA 드라이버 설치: ${CYAN}sudo apt install nvidia-driver-xxx${NC}"
        echo -e "  2. 재부팅 후 확인: ${CYAN}nvidia-smi${NC}"
    fi
fi

# ═══════════════════════════════════════════
# STEP 8: 한글 폰트 & 디렉토리 설정
# ═══════════════════════════════════════════
line
log "${BOLD}[8/8] 폰트 & 디렉토리 설정${NC}"

# 한글 폰트 확인
FONT_FOUND=false
for font_path in \
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" \
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" \
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"; do
    if [ -f "$font_path" ]; then
        ok "한글 폰트: $font_path"
        FONT_FOUND=true
        break
    fi
done

if [ "$FONT_FOUND" = false ]; then
    warn "한글 폰트 미발견"
    log "  한글 폰트 설치 시도..."
    if sudo apt-get install -y fonts-nanum fonts-noto-cjk >> "$LOG_FILE" 2>&1; then
        ok "한글 폰트 설치 완료"
        # 폰트 캐시 갱신
        fc-cache -f -v >> "$LOG_FILE" 2>&1 || true
    else
        warn "폰트 자동 설치 실패"
        info "수동 설치: sudo apt install fonts-nanum fonts-noto-cjk"
    fi
fi

# 디렉토리 생성
DIRS=("output" "music" "fonts" "cache" "data" "logs" "static" "templates")
for dir in "${DIRS[@]}"; do
    mkdir -p "${SCRIPT_DIR}/${dir}"
done
ok "작업 디렉토리 생성 완료"

# .env 파일 생성 (없을 경우)
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    cat > "${SCRIPT_DIR}/.env" << 'ENVEOF'
# AI Shorts Studio PRO v6.0 Configuration
PORT=5000
LOG_LEVEL=INFO
SECRET_KEY=ai-shorts-studio-v6-secret-change-me

# Directories
OUTPUT_DIR=output
MUSIC_DIR=music
FONTS_DIR=fonts
CACHE_DIR=cache

# Video defaults
DEFAULT_WIDTH=1080
DEFAULT_HEIGHT=1920
DEFAULT_FPS=30
DEFAULT_CRF=23
DEFAULT_CODEC=libx264
DEFAULT_AUDIO_BITRATE=192k

# Ollama (로컬 LLM)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=exaone3.5:32b
OLLAMA_FALLBACK_MODEL=mistral:7b
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.8
OLLAMA_MAX_TOKENS=500

# Concurrency
MAX_CONCURRENT_JOBS=4
THREAD_POOL_SIZE=4
ENVEOF
    ok ".env 설정 파일 생성됨"
else
    ok ".env 설정 파일 존재"
fi

# 실행 스크립트 생성
cat > "${SCRIPT_DIR}/start.sh" << 'STARTEOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"
cd "$SCRIPT_DIR"

echo ""
echo "🎬 AI Shorts Studio PRO v6.0 시작 중..."
echo "   URL: http://localhost:${PORT:-5000}"
echo "   로그: logs/app.log"
echo "   종료: Ctrl+C"
echo ""

python app.py
STARTEOF
chmod +x "${SCRIPT_DIR}/start.sh"
ok "start.sh 실행 스크립트 생성됨"

# ═══════════════════════════════════════════
# 설치 완료 보고서
# ═══════════════════════════════════════════
echo ""
line
echo ""
echo -e "${GREEN}${BOLD}  🎉 AI SHORTS STUDIO PRO v6.0 설치 완료!${NC}"
echo ""
line
echo ""
echo -e "  ${BOLD}환경 정보:${NC}"
echo -e "  ├─ OS:      ${CYAN}${PRETTY_NAME:-Unknown}${NC}"
echo -e "  ├─ WSL2:    ${CYAN}${IS_WSL2}${NC}"
echo -e "  ├─ Python:  ${CYAN}$(python --version 2>&1)${NC}"
echo -e "  ├─ FFmpeg:  ${CYAN}$(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)${NC}"
if [ "$GPU_DETECTED" = true ]; then
echo -e "  ├─ GPU:     ${GREEN}${GPU_NAME} (${GPU_VRAM})${NC}"
echo -e "  └─ CUDA:    ${GREEN}${CUDA_VER}${NC}"
else
echo -e "  └─ GPU:     ${YELLOW}미감지 (CPU 모드)${NC}"
fi
echo ""
echo -e "  ${BOLD}실행 방법:${NC}"
echo -e "  ${CYAN}./start.sh${NC}"
echo -e "  또는"
echo -e "  ${CYAN}source venv/bin/activate && python app.py${NC}"
echo ""
echo -e "  ${BOLD}브라우저 접속:${NC}"
echo -e "  ${CYAN}http://localhost:5000${NC}"
echo ""

if [ "$IS_WSL2" = true ]; then
echo -e "  ${BOLD}WSL2 팁:${NC}"
echo -e "  ├─ Windows 브라우저에서 접속: ${CYAN}http://localhost:5000${NC}"
echo -e "  ├─ GPU 확인: ${CYAN}nvidia-smi${NC}"
echo -e "  └─ WSL 재시작: ${CYAN}wsl --shutdown${NC} (PowerShell)"
echo ""
fi

echo -e "  ${BOLD}선택사항 (AI 대본 생성):${NC}"
echo -e "  ${CYAN}curl -fsSL https://ollama.com/install.sh | sh${NC}"
echo -e "  ${CYAN}ollama pull exaone3.5:32b${NC}  (한국어 특화 모델)"
echo -e "  ${CYAN}ollama pull mistral:7b${NC}     (가벼운 범용 모델)"
echo ""
line
log "설치 완료: $(date)"

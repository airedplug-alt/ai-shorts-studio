#!/bin/bash
# ═══════════════════════════════════════════
#  AI SHORTS STUDIO PRO v1.0.0 — Unified Launcher
#  © chally.choi
#  ComfyUI + Flask 한방 실행
# ═══════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Config
FLASK_PORT=5000
COMFYUI_PORT=8188
COMFYUI_PID=""
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Banner ─────────────────────────────────
echo -e ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${PURPLE}${BOLD}AI SHORTS STUDIO PRO${NC} ${CYAN}v1.0.0${NC}                     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ${GREEN}© chally.choi${NC}                                   ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                  ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ComfyUI + Flask ${YELLOW}통합 실행${NC}                       ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Cleanup function ──────────────────────
cleanup() {
    echo ""
    echo -e "${YELLOW}🔴 종료 중...${NC}"
    if [ -n "$COMFYUI_PID" ] && kill -0 "$COMFYUI_PID" 2>/dev/null; then
        echo -e "   ComfyUI 종료 (PID: $COMFYUI_PID)..."
        kill "$COMFYUI_PID" 2>/dev/null
        wait "$COMFYUI_PID" 2>/dev/null
    fi
    # Kill any remaining child processes
    jobs -p | xargs -r kill 2>/dev/null
    echo -e "${GREEN}✅ 모든 프로세스 종료 완료${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ── Activate Python venv ──────────────────
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${GREEN}🐍 Python 가상환경 활성화${NC}"
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo -e "${YELLOW}⚠️  venv 없음 — 시스템 Python 사용${NC}"
fi

# ── Find & Start ComfyUI ──────────────────
COMFYUI_DIR=""
for candidate in \
    "$SCRIPT_DIR/comfyui/ComfyUI" \
    "$SCRIPT_DIR/ComfyUI" \
    "$HOME/ComfyUI" \
    "$(dirname "$SCRIPT_DIR")/ComfyUI"; do
    if [ -f "$candidate/main.py" ]; then
        COMFYUI_DIR="$candidate"
        break
    fi
done

if [ -n "$COMFYUI_DIR" ]; then
    echo -e "${CYAN}🎨 ComfyUI 발견: ${COMFYUI_DIR}${NC}"

    # Check if ComfyUI already running
    if curl -s "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ComfyUI 이미 실행 중 (포트 $COMFYUI_PORT)${NC}"
    else
        echo -e "${YELLOW}🚀 ComfyUI 서버 시작 중...${NC}"

        # Determine Python for ComfyUI
        COMFYUI_PYTHON="python3"
        if [ -f "$COMFYUI_DIR/venv/bin/python" ]; then
            COMFYUI_PYTHON="$COMFYUI_DIR/venv/bin/python"
        fi

        # Start ComfyUI in background
        cd "$COMFYUI_DIR"
        $COMFYUI_PYTHON main.py --listen 127.0.0.1 --port $COMFYUI_PORT \
            > "$LOG_DIR/comfyui.log" 2>&1 &
        COMFYUI_PID=$!
        cd "$SCRIPT_DIR"

        echo -e "   PID: $COMFYUI_PID | 로그: $LOG_DIR/comfyui.log"

        # Wait for ComfyUI to be ready (max 120s)
        echo -ne "   서버 대기 중"
        for i in $(seq 1 60); do
            if curl -s "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
                echo ""
                echo -e "${GREEN}   ✅ ComfyUI 준비 완료!${NC}"
                break
            fi
            # Check if process died
            if ! kill -0 "$COMFYUI_PID" 2>/dev/null; then
                echo ""
                echo -e "${RED}   ❌ ComfyUI 시작 실패 — 로그 확인: $LOG_DIR/comfyui.log${NC}"
                COMFYUI_PID=""
                break
            fi
            echo -ne "."
            sleep 2
        done
    fi
else
    echo -e "${YELLOW}⚠️  ComfyUI 미설치 — AI 영상은 폴백(슬라이드쇼) 모드로 생성됩니다${NC}"
    echo -e "   설치하려면: ${CYAN}./install.sh comfyui${NC}"
fi

# ── Start Flask App ───────────────────────
echo ""
echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 AI Shorts Studio PRO v1.0.0 시작${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
echo -e ""
echo -e "   ${CYAN}🌐 웹 UI:${NC}      ${BOLD}http://localhost:$FLASK_PORT${NC}"
if [ -n "$COMFYUI_DIR" ]; then
    echo -e "   ${CYAN}🎨 ComfyUI:${NC}    http://127.0.0.1:$COMFYUI_PORT"
fi
echo -e "   ${CYAN}📁 출력 폴더:${NC}  $SCRIPT_DIR/output/"
echo -e "   ${CYAN}📋 로그:${NC}       $LOG_DIR/"
echo -e ""
echo -e "   ${YELLOW}Ctrl+C로 모든 서버 종료${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
echo ""

# Run Flask (foreground — Ctrl+C stops everything)
python app.py

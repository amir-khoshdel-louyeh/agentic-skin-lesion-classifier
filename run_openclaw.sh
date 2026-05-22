#!/bin/bash

# Term colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Checking system prerequisites...${NC}"

# 1. Check Ollama Service
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✅ Ollama service is running.${NC}"
else
    echo -e "${YELLOW}⚠️ Ollama service is not running. Starting Ollama...${NC}"
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# 2. Activate Python Virtual Environment
if [ -d ".venv" ]; then
    echo -e "${GREEN}📦 Activating Python virtual environment (.venv)...${NC}"
    source .venv/bin/activate
else
    echo -e "${RED}❌ Error: .venv folder not found! Make sure you are in the project root directory.${NC}"
    exit 1
fi

# 3. Check and Start OpenClaw Gateway
echo -e "${YELLOW}📡 Checking OpenClaw Gateway status...${NC}"
if ss -tuln | grep -q ":18789 "; then
    echo -e "${GREEN}✅ OpenClaw Gateway is active on port 18789.${NC}"
else
    echo -e "${YELLOW}⚠️ Gateway is offline. Launching gateway background process...${NC}"
    openclaw gateway --listen 127.0.0.1:18789 > /dev/null 2>&1 &
    sleep 4
fi

# 4. Launch Crestodian Agent
echo -e "${GREEN}🚀 All systems ready! Launching Crestodian...${NC}"
echo "------------------------------------------------"
openclaw crestodian
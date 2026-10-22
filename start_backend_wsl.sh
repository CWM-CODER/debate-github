#!/bin/bash
# Clash of Minds — WSL Startup Script
# Run this inside Ubuntu/WSL terminal

echo "🧠 Clash of Minds — Starting Backend..."

cd /mnt/d/Debate

# Create venv in Linux home directory to avoid Windows NTFS permission issues
VENV_PATH="$HOME/.debate_venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "[1/3] Creating Python virtual environment..."
    sudo apt install python3-full -y
    python3 -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    echo "[2/3] Installing dependencies..."
    pip install -r backend/requirements.txt
else
    source $VENV_PATH/bin/activate
fi

echo "[3/3] Starting FastAPI backend on port 8000..."
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

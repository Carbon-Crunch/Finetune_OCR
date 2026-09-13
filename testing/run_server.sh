#!/usr/bin/env bash
set -e

# Directory where this script lives
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " Starting Industrial OCR Model Server"
echo "============================================"

# Ensure uv is installed and in PATH
if ! command -v uv &> /dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "[*] Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment with uv..."
    if [ -x "/home/zeus/miniconda3/envs/cloudspace/bin/python" ]; then
        uv venv .venv --system-site-packages --python /home/zeus/miniconda3/envs/cloudspace/bin/python
    else
        uv venv .venv --python 3.12
    fi
fi

# Activate virtualenv
source .venv/bin/activate

# Install / verify required dependencies
echo "[*] Checking required dependencies..."
uv pip install \
    "numpy<2" \
    "torchvision" \
    "transformers>=4.48.0,<5.0" \
    "peft>=0.18.0" \
    "accelerate" \
    "bitsandbytes" \
    "qwen-vl-utils" \
    "fastapi" \
    "uvicorn" \
    "python-multipart" \
    "pillow" \
    "requests" \
    "gdown"


# Ensure adapter weights are available
if [ ! -d "$PROJECT_DIR/saves_v1" ]; then
    if [ -f "$PROJECT_DIR/saves_v1.zip" ]; then
        echo "[*] Extracting saves_v1.zip..."
        unzip -q "$PROJECT_DIR/saves_v1.zip" -d "$PROJECT_DIR"
    else
        echo "[*] saves_v1 adapter not found. Downloading from Google Drive..."
        gdown --fuzzy "https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link" -O "$PROJECT_DIR/saves_v1.zip"
        unzip -q "$PROJECT_DIR/saves_v1.zip" -d "$PROJECT_DIR"
    fi
fi

# Ensure adapter symlink exists
mkdir -p "$PROJECT_DIR/saves/Qwen2.5-VL-7B-Instruct/lora"
if [ -d "$PROJECT_DIR/saves_v1" ] && [ ! -e "$PROJECT_DIR/saves/Qwen2.5-VL-7B-Instruct/lora/industrial_ocr_v1" ]; then
    ln -sfn "$PROJECT_DIR/saves_v1" "$PROJECT_DIR/saves/Qwen2.5-VL-7B-Instruct/lora/industrial_ocr_v1"
fi

# Default environment variables
export ADAPTER_PATH="${ADAPTER_PATH:-$PROJECT_DIR/saves_v1}"
export BASE_MODEL="${BASE_MODEL:-unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

echo "[*] Base Model:   $BASE_MODEL"
echo "[*] Adapter Path: $ADAPTER_PATH"
echo "[*] Server URL:   http://$HOST:$PORT"
echo "============================================"

# Launch the FastAPI server
exec python server.py

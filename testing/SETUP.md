# Industrial Document OCR Server Setup Guide

FastAPI web server and REST API for document orientation detection and structured field extraction using **Qwen2.5-VL-7B-Instruct** with fine-tuned LoRA weights.

---

## Architecture Overview

1. **Base Model**: `unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit`
   - 4-bit BitsAndBytes quantization.
   - Fits easily on GPUs with 15-16 GB VRAM (Tesla T4, V100, RTX 3090/4090, A10G), consuming ~7.5 GB VRAM during inference.
2. **LoRA Adapter**: [`saves_v1/`](https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link)
   - Fine-tuned adapter for industrial documents.
   - [Download `saves_v1.zip` from Google Drive](https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link).
3. **Pipeline**:
   - **Step 1**: Orientation detection (determines 0°, 90°, 180°, or 270° clockwise rotation needed).
   - **Step 2**: Image rotation correction.
   - **Step 3**: Structured field extraction as clean JSON.
4. **Server**: FastAPI + Uvicorn serving both a web UI and a `/api/ocr` endpoint.

---

## Quick Start (One Command)

To set up the environment (if not already created) and launch the server:

```bash
./run_server.sh
```

The script automatically:
1. Installs or locates `uv`.
2. Creates the Python virtual environment (`.venv`) using the system CUDA PyTorch runtime.
3. Installs all required packages in seconds.
4. Sets up the adapter symlink path.
5. Starts the server on `http://0.0.0.0:8000`.

---

## Manual Setup with `uv`

If you prefer to configure the environment step-by-step:

### 1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Create the Virtual Environment
Create a virtual environment that reuses the machine's CUDA-enabled PyTorch:
```bash
uv venv .venv --system-site-packages --python /home/zeus/miniconda3/envs/cloudspace/bin/python
source .venv/bin/activate
```

> **Note**: If running outside this studio, create a standard Python 3.10-3.12 venv and install PyTorch with CUDA:
> ```bash
> uv venv .venv --python 3.12
> source .venv/bin/activate
> uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 3. Install Dependencies
```bash
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
```

### 4. Download Model Weights (`saves_v1.zip`)
If `saves_v1/` is not already present, download the LoRA adapter weights:
- **Direct Link**: [Download `saves_v1.zip` from Google Drive](https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link)
- **CLI Download using `gdown`**:
```bash
gdown --fuzzy "https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link" -O saves_v1.zip
unzip -q saves_v1.zip
```

### 5. Verify Adapter Paths
Ensure the adapter directory exists:
```bash
mkdir -p saves/Qwen2.5-VL-7B-Instruct/lora
ln -sfn "$(pwd)/saves_v1" "$(pwd)/saves/Qwen2.5-VL-7B-Instruct/lora/industrial_ocr_v1"
```

### 6. Run the Server
```bash
python server.py
```
Or directly with Uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BASE_MODEL` | `unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit` | Hugging Face base vision-language model |
| `ADAPTER_PATH` | `./saves_v1` | Path to the trained LoRA adapter weights |
| `HOST` | `0.0.0.0` | Host address to bind the server |
| `PORT` | `8000` | Port to bind the server |

---

## API Usage

### 1. Web Interface
Open `http://localhost:8000/` in your browser. You can drag and drop any document image to inspect orientation prediction and extracted fields.

### 2. cURL
```bash
curl -X POST http://localhost:8000/api/ocr \
  -F "file=@/path/to/document.png"
```

### 3. Python Request
```python
import requests

url = "http://localhost:8000/api/ocr"
with open("document.png", "rb") as f:
    files = {"file": ("document.png", f, "image/png")}
    response = requests.post(url, files=files)
    print(response.json())
```

### Sample API Response
```json
{
  "orientation": {
    "predicted_degree": 0,
    "raw": "{\"degree_needed\":0,\"rotation_needed\":false}"
  },
  "extraction": {
    "rotation_needed": false,
    "degree": 0,
    "data_type": "Invoice",
    "data_category": "Commercial Invoice",
    "extracted_fields": {
      "invoice_number": "INV-2026-001",
      "total_amount": {
        "value": 1500.5,
        "unit": "USD"
      }
    }
  }
}
```

---

## Process Management

- **Check if server is running**:
  ```bash
  curl http://127.0.0.1:8000/
  ```
- **Check GPU Memory**:
  ```bash
  nvidia-smi
  ```
- **Stop server**:
  ```bash
  pkill -f "python server.py"
  ```

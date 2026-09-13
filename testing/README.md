# Testing & Inference Module: Industrial Document OCR Server

FastAPI web application and REST API service for running inference with **Qwen2.5-VL-7B-Instruct** and fine-tuned LoRA adapters (`saves_v1`).

---

## 🚀 Overview

The testing module provides a high-throughput, low-latency document processing service with:
1. **Orientation Auto-Detection**: Classifies page rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
2. **Dynamic Rotation Correction**: Automatically rotates the document image in-memory using PIL before passing it to field extraction.
3. **Structured Field Extraction**: Generates clean, strictly formatted JSON adhering to domain-specific boilerplates (energy invoices, meter readings, coal reports, boiler logs, etc.).
4. **Interactive OpenAPI /docs**: Fully documented FastAPI Swagger UI and ReDoc with Pydantic response models and "Try it out" document upload testing.

---

## 📂 File Layout

```
testing/
├── README.md             # This file (overview, API guide, and quick start)
├── SETUP.md              # Detailed setup, virtualenv, and process management guide
├── server.py             # Core FastAPI application with web UI and REST endpoints
├── run_server.sh         # Auto-setup and launch script
├── pyproject.toml        # Dependencies and metadata
├── uv.lock               # Pinned package versions
├── saves_v1/             # Trained LoRA adapter directory (extracted from saves_v1.zip)
├── saves_v1.zip          # Adapter zip archive (~320 MB) [External Google Drive download]
├── Batch3/               # Directory of sample test documents (PDFs and JPG scans)
└── Batch3.zip            # Sample evaluation batch archive
```

---

## ⚡ Quick Start

### One-Command Launch
```bash
./run_server.sh
```

`run_server.sh` handles everything automatically:
- Checks/installs `uv` package manager.
- Creates `.venv` with CUDA PyTorch runtime.
- Installs dependencies from `pyproject.toml`.
- Checks for `saves_v1/`; if missing, automatically unzips or downloads `saves_v1.zip` from Google Drive.
- Creates adapter symlinks under `saves/Qwen2.5-VL-7B-Instruct/lora/industrial_ocr_v1`.
- Boots the server on `http://0.0.0.0:8000`.

---

## Model Weights Setup

The model uses base weights from Hugging Face (`unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit`) and a fine-tuned LoRA adapter:

- **Adapter Directory**: `testing/saves_v1/`
- **Google Drive Link**: [Download `saves_v1.zip`](https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link)
- **Manual CLI Download**:
  ```bash
  pip install gdown
  gdown --fuzzy "https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link" -O saves_v1.zip
  unzip -q saves_v1.zip
  ```

---

## API Reference

### 1. Interactive Web UI (`/`)
Navigate to `http://localhost:8000/` in your browser. Drag and drop any document (JPEG, PNG, or PDF) to visually inspect orientation prediction and extracted JSON data side-by-side. Direct links to Swagger Docs and Health status are integrated into the header.

### 2. Interactive Swagger UI & OpenAPI (`/docs` & `/redoc`)
Navigate to `http://localhost:8000/docs` or `http://localhost:8000/redoc`:
- **Swagger UI (`/docs`)**: Interactive OpenAPI specification with Pydantic response models and the "Try it out" file upload interface.
- **ReDoc (`/redoc`)**: Comprehensive, clean technical API documentation.

### 3. `POST /api/ocr`
Upload an image or document for orientation detection and structured extraction.

- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (binary file upload)
- **Supported Formats**: PNG, JPG, JPEG, PDF

#### Request Example (cURL)
```bash
curl -X POST http://localhost:8000/api/ocr \
  -F "file=@Batch3/120.pdf"
```

#### Request Example (Python)
```python
import requests

url = "http://localhost:8000/api/ocr"
with open("Batch3/124.JPG", "rb") as f:
    response = requests.post(url, files={"file": ("124.JPG", f, "image/jpeg")})
    print(response.json())
```

#### Response Example
```json
{
  "orientation": {
    "predicted_degree": 90,
    "raw": "{\"degree_needed\": 90, \"rotation_needed\": true}"
  },
  "extraction": {
    "rotation_needed": true,
    "degree": 90,
    "data_type": "Energy",
    "data_category": "Electricity Bill",
    "extracted_fields": {
      "consumer_number": "100234891",
      "bill_date": "2024-03-15",
      "total_units_consumed": 4520,
      "tariff_type": "Industrial HT-1"
    }
  }
}
```

### 4. `GET /health`
Returns the operational health and loaded model status:
```json
{
  "status": "healthy",
  "base_model": "unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit",
  "adapter_path": "./saves_v1",
  "device": "cuda"
}
```

---

## Testing with `Batch3/`

The `Batch3/` directory contains sample real-world documents for testing:
- **PDF Documents**: `120.pdf`, `121.pdf`, `122.pdf`, `129.pdf`, `141.pdf`, `148.pdf` (multi-page invoices, monthly summaries, meter logs).
- **Image Scans**: `124.JPG`, `125.JPG`, `134.JPG`, `136.JPG`, `140.JPG` (handwritten power logs, skewed bills, rotated production sheets).

To test a batch document using the API:
```bash
# Test orientation correction on a rotated invoice
curl -X POST http://localhost:8000/api/ocr -F "file=@Batch3/124.JPG"

# Test extraction on a PDF report
curl -X POST http://localhost:8000/api/ocr -F "file=@Batch3/120.pdf"
```

---

## ⚙️ Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `BASE_MODEL` | `unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit` | Hugging Face base vision-language model checkpoint |
| `ADAPTER_PATH` | `./saves_v1` | Path to fine-tuned LoRA adapter weights |
| `HOST` | `0.0.0.0` | Host interface to bind the server |
| `PORT` | `8000` | Port to bind the server |

To run on a custom port or specify a custom adapter path:
```bash
PORT=8080 ADAPTER_PATH="/path/to/custom_lora" python server.py
```

For complete step-by-step installation instructions, see [SETUP.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/testing/SETUP.md).

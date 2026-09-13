# Industrial Document OCR: Fine-Tuning & Inference Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--VL--7B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20FastAPI-green.svg)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

End-to-end vision-language pipeline for **industrial document orientation detection** and **structured JSON field extraction** using **Qwen2.5-VL-7B-Instruct** fine-tuned with LoRA/QLoRA via **Unsloth**.

---

## Key Capabilities

1. **Orientation Auto-Correction (Task A)**:
   - Accurately classifies document rotation into $0^\circ, 90^\circ, 180^\circ,$ or $270^\circ$ clockwise.
   - Automatically corrects scan orientation prior to extraction.
2. **Structured Field Extraction (Task B)**:
   - Extracts domain-specific industrial fields (Invoices, Electric Bills, Meter Readings, Fuel Reports, Process Log Books) into clean, schema-conforming JSON.
   - Handles multi-page PDFs, handwritten power logs, tabular process logs, and low-contrast scanned receipts.
3. **Optimized Inference**:
   - 4-bit quantized base model + LoRA adapter runs in **~7.5 GB VRAM**, making it deployable on consumer and enterprise GPUs (RTX 3090/4090, T4, V100, A10G, A100).
4. **Interactive Web UI & REST API**:
   - Ready-to-use FastAPI backend with drag-and-drop document inspection web interface.

---

## Repository Structure

```
new_finetune_ocr/
├── README.md                  # Project root overview and entry point (this file)
├── .gitignore                 # Configured for ML checkpoints, weights, datasets, & venvs
│
├── testing/                   # Inference & Serving Subsystem
│   ├── README.md              # Inference server guide & API reference
│   ├── SETUP.md               # Step-by-step installation & deployment guide
│   ├── server.py              # FastAPI server (Web UI + POST /api/ocr endpoint)
│   ├── run_server.sh          # One-click launch script with auto venv & dependency setup
│   ├── pyproject.toml         # Inference dependencies and uv configuration
│   ├── uv.lock                # Deterministic lockfile for testing environment
│   ├── Batch3/                # Sample evaluation documents (PDFs and JPG scans) [gitignored]
│   └── saves_v1/              # Trained LoRA adapter weights [gitignored]
│
└── training/                  # Model Fine-Tuning & Offline Pipeline Subsystem
    ├── README.md              # Offline training & evaluation harness documentation
    ├── architecture-fine-tuning.md # End-to-end model and data architecture spec
    ├── llm_eval_architecture.md    # LLM evaluation and metrics calculation spec
    ├── training.py            # Unsloth fine-tuning pipeline with QLoRA & augmentations
    ├── evaluate.py            # Comprehensive evaluation script (metrics, HTML reports)
    ├── batch_processor.py     # Concurrent document processor for dataset generation
    ├── check_gtr_status.py    # Health check for ground-truth JSON files
    ├── update_gtr_orientation.py # Multi-provider rotation annotation script (Gemini/Groq)
    ├── pyproject.toml         # Training dependencies specification
    ├── requirements-train.txt # Pip requirements for training & evaluation
    ├── uv.lock                # Deterministic lockfile for training environment
    │
    ├── configs/               # Configuration & Boilerplates
    │   ├── README.md          # Config architecture and schema registry guide
    │   ├── config.yaml        # Preprocessing, handwriting, and file limits config
    │   ├── generic_fallback.json # Generic fallback JSON schema
    │   └── boilerplates/      # Category schemas (Energy, Fuel, Production)
    │       ├── registry.json  # Schema registry and document classification rules
    │       ├── energy/        # Electricity bills, meter readings, invoices
    │       ├── fuel/          # Coal consignment, fuel lab tests, gas bills
    │       └── production/    # Boiler logs, process log books, turbine logs
    │
    └── dataset/               # Dataset Hub
        ├── README.md          # Dataset guide, schemas, and download links
        ├── all_gtr/           # Ground-truth JSON annotations (~7.5 MB) [Tracked in Git]
        └── all_file/          # Raw images & PDFs (~1.5 GB) [External on Google Drive]
```

---

## ⚡ Quick Start

### 1. Run the Inference Server (`testing/`)

To launch the FastAPI OCR server with interactive UI:

```bash
cd testing
./run_server.sh
```

- Web UI: Open `http://localhost:8000` in your browser.
- REST API: `POST http://localhost:8000/api/ocr` with `multipart/form-data` (`file=@document.png`).
- Refer to [testing/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/testing/README.md) for full endpoint specifications.

### 2. Fine-Tune the Model (`training/`)

To fine-tune `Qwen2.5-VL-7B-Instruct` on industrial documents:

```bash
cd training

# Install dependencies
uv pip install -r requirements-train.txt

# Run training with default parameters (or override via environment variables)
python training.py
```

- Refer to [training/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/README.md) for training configurations, synthetic augmentations, and evaluation reporting.

---

## 🔗 External Assets & Downloads

| Asset | Size | Storage | Description / Download Link |
| :--- | :---: | :---: | :--- |
| **Trained LoRA Weights (`saves_v1.zip`)** | ~320 MB | Google Drive | [Download `saves_v1.zip`](https://drive.google.com/file/d/1CQBz2TJhN72Du3z-I7FU9kcKMXKm4Tv8/view?usp=drive_link) |
| **Raw Training Documents (`all_file`)** | ~1.5 GB | Google Drive | [Download Dataset Archive](https://drive.google.com/file/d/1E6HpBXeC9uujbay31xqAhqaZL_TXLXKS/view?usp=drive_link) |
| **Ground Truth Annotations (`all_gtr`)** | ~7.5 MB | Git Repository | Included directly under [training/dataset/all_gtr/](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/dataset/all_gtr) |

---


## Documentation Index

- **[testing/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/testing/README.md)**: Inference server guide, API documentation, and UI instructions.
- **[testing/SETUP.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/testing/SETUP.md)**: Detailed deployment, virtualenv setup, and process management.
- **[training/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/README.md)**: Training, data augmentation, hyperparameter overrides, and evaluation.
- **[training/configs/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/configs/README.md)**: Boilerplates, category registries, and extraction rules.
- **[training/dataset/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/dataset/README.md)**: Dataset schema, download instructions, and file organization.

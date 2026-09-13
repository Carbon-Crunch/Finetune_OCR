# Training & Offline Pipeline: Industrial Document OCR

Offline fine-tuning and evaluation suite for **Qwen2.5-VL-7B-Instruct** using **Unsloth** (QLoRA) on custom industrial document image-to-JSON datasets.

---

## 📂 Subsystem Layout

```
training/
├── README.md                      # This training & offline pipeline guide
├── architecture-fine-tuning.md     # In-depth architectural & augmentation spec
├── llm_eval_architecture.md        # Comprehensive evaluation metric methodology
├── training.py                    # Main Unsloth QLoRA training script
├── evaluate.py                    # Multi-task evaluation harness (Accuracy, F1, HTML reports)
├── batch_processor.py             # Concurrent document processor for dataset generation
├── check_gtr_status.py            # Ground-truth JSON verification & health checker
├── update_gtr_orientation.py      # LLM-based orientation auto-labeler (Groq/Gemini key pool)
├── requirements-train.txt         # Pip dependency requirements
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Pinned dependencies lockfile
│
├── configs/                       # Configuration schemas & boilerplates
│   ├── README.md                  # Config architecture guide
│   ├── config.yaml                # Preprocessing and handwriting detection config
│   ├── generic_fallback.json     # Default fallback schema
│   └── boilerplates/              # Domain-specific JSON templates
│       ├── registry.json          # Document type & category schema registry
│       ├── energy/                # Electricity bills, meter readings, invoices
│       ├── fuel/                  # Coal consignments, fuel lab tests, gas bills
│       └── production/            # Boiler logs, process log books, turbine logs
│
└── dataset/                       # Industrial document dataset
    ├── README.md                  # Dataset specifications and download guide
    ├── all_gtr/                   # Ground-truth JSON annotations [Tracked in Git]
    └── all_file/                  # Raw images and PDFs (~1.5 GB) [Google Drive]
```

---

## 🚀 Environment Setup

We recommend executing training on **Lightning AI Studio**, a cloud instance, or a dedicated workstation with an **NVIDIA L4 (24GB)** or **A100 (40GB/80GB)** GPU.

```bash
# Verify Python version >= 3.10
python3 --version

# Install dependencies using uv:
uv pip install -r requirements-train.txt

# Or using standard pip:
pip install -r requirements-train.txt
```

---

## 📦 Dataset Acquisition

The dataset consists of two components:
1. **Ground Truth Extractions (`dataset/all_gtr/`)**: Lightweight JSON files tracked directly in this repository.
2. **Raw Scanned Documents (`dataset/all_file/`)**: ~1.5 GB of images and PDFs, hosted externally on Google Drive.

### Downloading Raw Documents (`all_file/`)
- **Direct Link**: [Download Dataset from Google Drive](https://drive.google.com/file/d/1E6HpBXeC9uujbay31xqAhqaZL_TXLXKS/view?usp=drive_link)
- **CLI Download**:
```bash
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/1E6HpBXeC9uujbay31xqAhqaZL_TXLXKS/view?usp=drive_link" -O dataset.zip
unzip -q dataset.zip
```

Verify your dataset directory structure:
- `dataset/all_file/` — contains raw images and PDFs (e.g., `1000.jpg`, `120.pdf`).
- `dataset/all_gtr/` — contains ground-truth JSON files (e.g., `gtr_1000.json`, `gtr_120.json`).

---

## ⚡ Model Fine-Tuning (`training.py`)

Fine-tune `Qwen2.5-VL-7B-Instruct` with Unsloth 4-bit QLoRA and synthetic augmentation:

```bash
python training.py
```

### Key Training Features
- **4-Bit NF4 Quantization**: Minimizes VRAM usage without loss of quality.
- **Dual-Task Joint Training**: Concurrently optimizes for orientation angle ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and JSON structured extraction.
- **Synthetic Rotation Augmentation**: Generates synthetic $90^\circ, 180^\circ, 270^\circ$ rotated versions of clean documents on-the-fly.
- **Photometric Augmentations**: Introduces brightness, contrast, sharpness, blur, and JPEG compression noise to improve scanned document robustness.

### Hyperparameter Overrides via Environment Variables

| Variable | Default | Description |
| :--- | :---: | :--- |
| `EPOCHS` | `3` | Number of training epochs |
| `MAX_SAMPLES` | `0` | Max training samples (`0` uses all data; set to e.g. `50` for smoke testing) |
| `PER_DEVICE_BATCH` | `1` | Batch size per GPU device |
| `GRAD_ACCUM` | `8` | Gradient accumulation steps (Effective batch size = `PER_DEVICE_BATCH * GRAD_ACCUM`) |
| `LEARNING_RATE` | `1e-4` | Peak learning rate with cosine decay |
| `LORA_RANK` | `32` | LoRA adapter rank ($r$) |
| `ROTATION_AUG_PER_DOC` | `4` | Synthetic rotation samples generated per document |
| `PHOTO_AUG_ENABLED` | `1` | Enable/disable photometric augmentations (`1` or `0`) |
| `PHOTO_AUG_PROB` | `0.5` | Probability per photometric transform |
| `MERGE_WEIGHTS` | `0` | Set to `1` to export merged standalone weights after training |

#### Practical Training Examples:
```bash
# Smoke test on 50 samples for 1 epoch
MAX_SAMPLES=50 EPOCHS=1 python training.py

# Full production training on A100 (40GB) with weight merging enabled
PER_DEVICE_BATCH=2 GRAD_ACCUM=4 ROTATION_AUG_PER_DOC=4 MERGE_WEIGHTS=1 python training.py
```

---

## 📊 Comprehensive Evaluation (`evaluate.py`)

Evaluate the fine-tuned checkpoint against test documents across both tasks:

```bash
python evaluate.py

# Subset evaluation for fast verification
MAX_EVAL_SAMPLES=100 python evaluate.py
```

### What `evaluate.py` Computes:
1. **Orientation Classification**:
   - Accuracy, Macro/Weighted Precision, Recall, and F1 score.
   - 4x4 Confusion Matrix across $[0^\circ, 90^\circ, 180^\circ, 270^\circ]$.
2. **Structured JSON Extraction**:
   - Field-level Exact Match (EM) accuracy.
   - Micro, Macro, and Weighted F1 metrics.
   - Breakdown of top hardest fields vs best performing fields.
3. **Generated Artifacts**:
   - `saves/Qwen2.5-VL-7B/lora/industrial_ocr_v1/eval_report/report.html` (interactive HTML report with Plotly charts).
   - `saves/Qwen2.5-VL-7B/lora/industrial_ocr_v1/eval_report/metrics_summary.json` (aggregate scores).
   - `saves/Qwen2.5-VL-7B/lora/industrial_ocr_v1/eval_report/per_doc_results.json` (document-level predictions).

---

## 🛠️ Dataset Management & Utility Scripts

### 1. Verification of Ground Truth (`check_gtr_status.py`)
Scans `all_gtr/` and verifies schema completeness, missing files, and unpopulated fields:
```bash
python check_gtr_status.py
```

### 2. Auto-Updating Orientation Labels (`update_gtr_orientation.py`)
Uses multi-provider pooled vision APIs (Groq + Gemini) to detect and populate missing rotation annotations in `all_gtr/*.json`:
```bash
# Pool multiple API keys to avoid rate limits
python update_gtr_orientation.py
```

### 3. Batch Document Extraction Processor (`batch_processor.py`)
Processes raw documents directly from `all_file/` with combined extraction and rotation logic:
```bash
# Process all files with 4 concurrent workers
python batch_processor.py --all --workers 4

# Process single file
python batch_processor.py --file 514.jpg
```

---

## 📦 Deployment & Export

### Standalone Inference (LoRA Adapter)
The trained LoRA adapter (~320 MB) is saved to `saves/Qwen2.5-VL-7B/lora/industrial_ocr_v1/` and can be copied directly to `testing/saves_v1/`.

### High-Throughput Serving via vLLM
If serving with vLLM in production:
1. Run training with `MERGE_WEIGHTS=1`.
2. Launch vLLM with the merged model directory:
   ```bash
   vllm serve ./saves/Qwen2.5-VL-7B/merged/industrial_ocr_v1/ --limit-mm-per-prompt image=1
   ```

For configuration schemas and boilerplates, see [configs/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/configs/README.md).  
For dataset schema details, see [dataset/README.md](file:///home/ubuntu/Anurag/Coding/Projects/carbon_crunch/new_finetune_ocr/training/dataset/README.md).

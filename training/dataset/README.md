# Industrial Document OCR Dataset

This folder hosts the dataset used to train and evaluate **Qwen2.5-VL-7B-Instruct** for document orientation classification and structured field extraction.

---

## 📂 Structure Overview

```
dataset/
├── README.md                  # This file
├── all_gtr/                   # Ground truth JSON annotations (~7.5 MB) [Tracked in Git]
│   ├── gtr_1.json
│   ├── gtr_2.json
│   └── ... (1,000+ files)
│
└── all_file/                  # Raw document files (~1.5 GB) [Ignored in Git, hosted on Google Drive]
    ├── 1.pdf
    ├── 2.jpg
    └── ... (1,000+ files)
```

---

## ⬇️ Downloading Raw Documents (`all_file/`)

Due to GitHub repository size guidelines, raw image and PDF documents (`all_file/`, ~1.5 GB) are stored externally.

### 1. Direct Browser Download
Download the dataset archive from Google Drive:
- **Direct Link**: [Download Dataset Archive (`dataset.zip`)](https://drive.google.com/file/d/1E6HpBXeC9uujbay31xqAhqaZL_TXLXKS/view?usp=drive_link)

### 2. Command-Line Download via `gdown`
Run inside the `training/` directory:
```bash
# Install gdown if needed
pip install gdown

# Download the zip file
gdown --fuzzy "https://drive.google.com/file/d/1E6HpBXeC9uujbay31xqAhqaZL_TXLXKS/view?usp=drive_link" -O dataset.zip

# Unpack directly into dataset/all_file/
unzip -q dataset.zip
```

Once extracted, verify that `all_file/` contains `.jpg`, `.jpeg`, `.png`, and `.pdf` files corresponding to the file IDs in `all_gtr/`.

---

## 📄 Ground Truth Annotation Schema (`all_gtr/*.json`)

Every document in `all_file/<id>.<ext>` has a matching ground truth JSON file in `all_gtr/gtr_<id>.json`.

### Schema Format:
```json
{
  "rotation_needed": true,
  "degree": 90,
  "data_type": "Energy",
  "data_category": "Electricity Bill",
  "extraction_rules": "Extract consumer information, meter readings, and billed units. Ignore fine print and tax breakdown.",
  "extracted_fields": {
    "consumer_name": {
      "value": "Acme Industrial Manufacturing Ltd"
    },
    "consumer_number": {
      "value": "100234891"
    },
    "billing_period": {
      "value": "01/03/2024 - 31/03/2024"
    },
    "units_consumed": {
      "value": "4520",
      "unit": "kWh"
    },
    "total_payable": {
      "value": "54240.00",
      "unit": "INR"
    }
  }
}
```

### Key Fields:
- **`rotation_needed`** (`bool`): `true` if the raw image is rotated ($90^\circ, 180^\circ,$ or $270^\circ$ clockwise); `false` if it is upright ($0^\circ$).
- **`degree`** (`int`): Degree of clockwise rotation required to restore the document to normal orientation ($0, 90, 180, 270$).
- **`data_type`** (`string`): Top-level industrial domain (`Energy`, `Fuel`, `Production & Operational`).
- **`data_category`** (`string`): Document category matching `configs/boilerplates/registry.json`.
- **`extraction_rules`** (`string`): Prompt instructions given to vision-language models for extraction.
- **`extracted_fields`** (`object`): Structured key-value dictionary containing scalar values, units, or tabular arrays (`entries`).

---

## 🛠️ Dataset Management Scripts

### Verifying Ground Truth Integrity
To inspect coverage and check for missing fields across `all_gtr/`:
```bash
# Run from training/
python check_gtr_status.py
```

### Auto-Populating Orientation Labels
If you add new documents without rotation annotations, run:
```bash
# Run from training/
python update_gtr_orientation.py
```
This script queries vision LLMs with key pooling and rate limit rotation to determine the required orientation angle and update `all_gtr/*.json`.

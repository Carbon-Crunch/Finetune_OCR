"""
=============================================================================
  Industrial Document OCR — Qwen2.5-VL-7B Fine-Tuning Script
=============================================================================

Overview
--------
This script fine-tunes `Qwen/Qwen2.5-VL-7B-Instruct` using Unsloth's
fast QLoRA implementation for two tasks:

  Task A – Orientation Classification  (Hybrid Augmentation)
      Input:  document image (original or synthetically rotated)
      Output: JSON {"degree_needed": <0|90|180|270>, "rotation_needed": <bool>}
      Labels: sourced from manually verified GTR `degree` field.
              Each real doc generates 4 training examples by applying
              additional θ∈{0,90,180,270}° clockwise rotations.
              New label = (D - θ + 360) % 360  where D = GTR degree.

  Task B – Structured JSON Extraction  (Photometric Augmentation)
      Input:  document image (brightness/contrast/blur jitter) +
              boilerplate schema from configs/
      Output: pure JSON matching the generic_fallback / category-specific
              boilerplate fields (invoice_date, supplier_name, entries[], …)
              Output always includes `data_type` and `data_category`.

Dataset Layout
--------------
  dataset/
    all_file/      raw source files  (*.jpg, *.jpeg, *.png, *.pdf)
    all_gtr/       ground-truth JSON (gtr_<id>.json — must include `degree`)

  configs/
    generic_fallback.json          reference field schema
    boilerplates/registry.json     category → boilerplate mapping
    boilerplates/{energy,fuel,production}/*.json

  final_consolidated-old.json      fallback data_type/category mapping by file_name

The script:
  1. Builds valid image-GTR pairs from all_gtr/ + all_file/.
  2. Converts multi-page PDFs → JPEG images on-the-fly (first page only).
  3. Generates extraction conversations (Task B, photometric augmentation).
  4. Generates 4× rotation conversations per document (Task A, synthetic rotation
     + photometric augmentation) using the manually verified GTR degree label.
  5. Shuffles and mixes both task types into a single HuggingFace Dataset.
  6. Loads Qwen2.5-VL-7B-Instruct via Unsloth (4-bit QLoRA).
  7. Trains with SFTTrainer / Unsloth's fast-path trainer.
  8. Runs a post-training orientation accuracy evaluation on eval split.
  9. Saves the LoRA adapter to `saves/Qwen2.5-VL-7B/lora/industrial_ocr_v1/`.

Usage (on a 24 GB GPU — L4 / A10G / A100)
------------------------------------------
  pip install unsloth[cu124-ampere-torch250] qwen-vl-utils pdf2image
  python training.py

  # Optional overrides via env vars:
  MAX_SAMPLES=500  EPOCHS=1  ROTATION_AUG_PER_DOC=2  python training.py
============================================================================="""

import gc
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Disable Unsloth torch.compile kernels — avoids cuDNN init failures on some hosts.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch

# Work around broken cuDNN init in some cloud GPU environments (falls back to native conv).
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    # Also disable cuDNN backend for scaled_dot_product_attention
    torch.backends.cuda.enable_cudnn_sdp(False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — all relative to this file's directory so the script runs from any CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "dataset"
ALL_FILE_DIR = DATASET_ROOT / "all_file"
ALL_GTR_DIR = DATASET_ROOT / "all_gtr"
CONFIGS_DIR = REPO_ROOT / "configs"
GENERIC_FALLBACK_PATH = CONFIGS_DIR / "generic_fallback.json"
REGISTRY_PATH = CONFIGS_DIR / "boilerplates" / "registry.json"
BOILERPLATES_DIR = CONFIGS_DIR / "boilerplates"
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-VL-7B-Instruct")
_model_basename = MODEL_NAME.split("/")[-1]
OUTPUT_DIR = REPO_ROOT / "saves" / _model_basename / "lora" / "industrial_ocr_v1"
CACHE_DIR = REPO_ROOT / ".cache" / "training"
CONVERTED_IMG_DIR = CACHE_DIR / "converted_images"

# ---------------------------------------------------------------------------
# Training hyper-parameters (can be overridden via environment variables)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))   # 0 = use all
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", "0.95"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "100"))
SEED = int(os.environ.get("SEED", "42"))

# ---------------------------------------------------------------------------
# Augmentation hyper-parameters
# ---------------------------------------------------------------------------
# Number of extra synthetic rotations generated per document for Task A.
# E.g. 4 means θ∈{0,90,180,270} → 4 rotation examples per real document.
# Set to 1 to only train on the real (unrotated) image with its GTR degree.
ROTATION_AUG_PER_DOC = int(os.environ.get("ROTATION_AUG_PER_DOC", "4"))

# Apply photometric augmentation (brightness/contrast/blur) to images.
# Set to 0 to disable (useful for debugging).
PHOTO_AUG_ENABLED = os.environ.get("PHOTO_AUG_ENABLED", "1").strip().lower() not in ("0", "false", "no")

# Probability that any single photometric augmentation is applied (per image).
PHOTO_AUG_PROB = float(os.environ.get("PHOTO_AUG_PROB", "0.5"))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = (
    "<image>\n"
    "You are an industrial document OCR expert. "
    "Extract ALL structured fields from this document as a single, compact JSON object. "
    "Use the field names exactly as shown in the schema. "
    "The output JSON MUST always include 'data_type' and 'data_category' fields indicating which document classification was used. "
    "Do NOT include markdown code fences, explanatory text, or preamble. "
    "Output ONLY the JSON object."
)

ORIENTATION_PROMPT = (
    "<image>\n"
    "Examine this document image. "
    "What clockwise rotation in degrees is needed to make the text correctly upright and readable? "
    "Choose from: 0, 90, 180, or 270 degrees only. "
    "Reply with a compact JSON object in this exact format: "
    '{"degree_needed": <number>, "rotation_needed": <true|false>}'
    " — no markdown, no extra text."
)


# ---------------------------------------------------------------------------
# Helper: PDF -> image conversion
# ---------------------------------------------------------------------------
def pdf_to_image(pdf_path: Path, output_dir: Path) -> Optional[Path]:
    """Convert the first page of a PDF to JPEG and return the image path."""
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        logger.warning(
            "pdf2image not installed. Skipping PDF '%s'. "
            "Install it with: pip install pdf2image",
            pdf_path.name,
        )
        return None

    img_path = output_dir / f"{pdf_path.stem}.jpg"
    if img_path.exists():
        return img_path

    try:
        pages = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=200)
        if pages:
            pages[0].save(str(img_path), "JPEG", quality=90)
            return img_path
    except Exception as exc:
        logger.warning("Could not convert '%s': %s", pdf_path.name, exc)
    return None


def resolve_image_path(doc_id: str, output_dir: Path) -> Optional[Path]:
    """
    Given a document ID (without extension), locate the corresponding image file.
    Supports .jpg, .jpeg, .png natively; converts .pdf on-the-fly.
    """
    image_exts = [".jpg", ".jpeg", ".png"]
    pdf_exts = [".pdf", ".PDF"]

    for ext in image_exts:
        candidate = ALL_FILE_DIR / f"{doc_id}{ext}"
        if candidate.exists():
            return candidate

    for ext in pdf_exts:
        candidate = ALL_FILE_DIR / f"{doc_id}{ext}"
        if candidate.exists():
            return pdf_to_image(candidate, output_dir)

    return None


# ---------------------------------------------------------------------------
# Helper: load boilerplate schema hint
# ---------------------------------------------------------------------------
def _get_hint_for_category(
    data_type: str, data_category: str, registry: Dict[str, Any]
) -> str:
    """Extract top-level schema keys for a single category from registry."""
    if not (data_type and data_category):
        return ""
    type_data = registry.get("data_types", {}).get(data_type, {})
    cat_data = type_data.get("categories", {}).get(data_category, {})
    boilerplate_rel = cat_data.get("boilerplate", "")
    if boilerplate_rel:
        bp_path = BOILERPLATES_DIR / boilerplate_rel
        if bp_path.exists():
            try:
                bp = json.loads(bp_path.read_text(encoding="utf-8"))
                top_keys = list(bp.get("extracted_fields", bp).keys())
                return f"{data_category} ({', '.join(top_keys)})"
            except Exception:
                pass
    return ""


_CONSOLIDATED_LOOKUP: Optional[Dict[str, Tuple[str, str]]] = None


def get_consolidated_lookup() -> Dict[str, Tuple[str, str]]:
    """Load and cache final_consolidated-old.json mapping doc_id/file_name -> (data_type, data_category)."""
    global _CONSOLIDATED_LOOKUP
    if _CONSOLIDATED_LOOKUP is not None:
        return _CONSOLIDATED_LOOKUP

    _CONSOLIDATED_LOOKUP = {}
    consolidated_path = REPO_ROOT / "final_consolidated-old.json"
    if consolidated_path.exists():
        try:
            data = json.loads(consolidated_path.read_text(encoding="utf-8"))
            for item in data.get("consolidated_data", []):
                fn = str(item.get("file_name", "")).strip()
                dt = item.get("data_type", "")
                dc = item.get("data_category", "")
                if fn and fn != "N/A":
                    _CONSOLIDATED_LOOKUP[fn] = (dt, dc)
        except Exception as exc:
            logger.warning("Could not load final_consolidated-old.json: %s", exc)
    return _CONSOLIDATED_LOOKUP


def load_boilerplate_hint(
    gtr_data: Any, doc_id: str = ""
) -> Tuple[str, str, str]:
    """
    Return a tuple of (schema_hint, resolved_data_type, resolved_data_category).
    Lookup order:
    1. Multi-template extractions in gtr_data.
    2. Single-template data_type / data_category directly inside gtr_data.
    3. Check final_consolidated-old.json using doc_id (file_name) before going to generic fallback.
    4. Generic fallback (`generic_fallback.json`).
    """
    # Some GTR files are JSON arrays — unwrap to dict or return generic fallback
    if isinstance(gtr_data, list):
        if len(gtr_data) == 1 and isinstance(gtr_data[0], dict):
            gtr_data = gtr_data[0]
        else:
            return "", "Generic", "Fallback"

    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

        # 1. Handle multi-template extractions (where 2-3 boilerplates are added together)
        if gtr_data.get("multi_template_extraction") or isinstance(
            gtr_data.get("extractions"), list
        ):
            extractions = gtr_data.get("extractions", [])
            hints = []
            dt_list, dc_list = [], []
            for item in extractions:
                dt = item.get("data_type", "")
                dc = item.get("data_category", "")
                if dt and dt not in dt_list:
                    dt_list.append(dt)
                if dc and dc not in dc_list:
                    dc_list.append(dc)
                hint = _get_hint_for_category(dt, dc, registry)
                if hint and hint not in hints:
                    hints.append(hint)
            if hints:
                return (
                    "Multi-template extraction schemas -> " + " | ".join(hints),
                    ", ".join(dt_list) if dt_list else gtr_data.get("data_type", "Multi-Template"),
                    ", ".join(dc_list) if dc_list else gtr_data.get("data_category", "Multi-Template"),
                )

        # 2. Handle single-template extractions
        data_type = str(gtr_data.get("data_type", "")).strip()
        data_category = str(gtr_data.get("data_category", "")).strip()
        if data_type and data_type != "N/A" and data_category and data_category != "N/A":
            hint = _get_hint_for_category(data_type, data_category, registry)
            if hint:
                return f"Fields: {hint}", data_type, data_category
    except Exception:
        pass

    # 3. Check final_consolidated-old.json using file_name / doc_id before generic fallback
    if doc_id:
        lookup = get_consolidated_lookup()
        if str(doc_id) in lookup:
            cons_dt, cons_dc = lookup[str(doc_id)]
            if cons_dt and cons_dt != "N/A" and cons_dc and cons_dc != "N/A":
                try:
                    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                    hint = _get_hint_for_category(cons_dt, cons_dc, registry)
                    if hint:
                        return f"Fields: {hint}", cons_dt, cons_dc
                except Exception:
                    pass
                # Even if no specific boilerplate hint is found for cons_dt/cons_dc, return them as resolved type/category
                try:
                    fallback = json.loads(GENERIC_FALLBACK_PATH.read_text(encoding="utf-8"))
                    top_keys = list(fallback.get("extracted_fields", fallback).keys())
                    return f"Fields: {', '.join(top_keys)}", cons_dt, cons_dc
                except Exception:
                    return "", cons_dt, cons_dc

    # 4. Generic fallback keys
    data_type = str(gtr_data.get("data_type", "")).strip()
    data_category = str(gtr_data.get("data_category", "")).strip()
    try:
        fallback = json.loads(GENERIC_FALLBACK_PATH.read_text(encoding="utf-8"))
        top_keys = list(fallback.get("extracted_fields", fallback).keys())
        return (
            f"Fields: {', '.join(top_keys)}",
            data_type if data_type and data_type != "N/A" else "Generic",
            data_category if data_category and data_category != "N/A" else "Fallback",
        )
    except Exception:
        return "", "Generic", "Fallback"


# ---------------------------------------------------------------------------
# Image Augmentation Helpers
# ---------------------------------------------------------------------------
def augment_image_photometric(img: Any) -> Any:
    """
    Apply random photometric augmentation to a PIL Image.
    Safe for both extraction and rotation tasks (does not alter geometry).
    Operations applied independently with probability PHOTO_AUG_PROB:
      - Brightness jitter  ±30%
      - Contrast jitter    ±30%
      - Sharpness jitter   ±50%
      - Gaussian blur      (radius 0–1.5)
      - JPEG re-encoding   (quality 60–95, simulates scan degradation)
    """
    if not PHOTO_AUG_ENABLED:
        return img

    import random
    from PIL import ImageEnhance, ImageFilter  # type: ignore
    from io import BytesIO  # type: ignore

    rng = random.Random()  # use default seed (varies each call)

    if rng.random() < PHOTO_AUG_PROB:
        factor = rng.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if rng.random() < PHOTO_AUG_PROB:
        factor = rng.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if rng.random() < PHOTO_AUG_PROB:
        factor = rng.uniform(0.5, 1.5)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    if rng.random() < PHOTO_AUG_PROB * 0.5:   # blur less frequently
        radius = rng.uniform(0.3, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if rng.random() < PHOTO_AUG_PROB * 0.4:   # JPEG degradation occasionally
        quality = rng.randint(60, 95)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        from PIL import Image as _PIL  # type: ignore
        img = _PIL.open(buf).convert("RGB")

    return img


def rotate_image_clockwise(img: Any, degrees: int) -> Any:
    """
    Rotate a PIL Image clockwise by `degrees` (0, 90, 180, 270).
    PIL.Image.rotate() is counter-clockwise, so we negate.
    expand=True ensures the full rotated image is preserved (no cropping).
    """
    if degrees == 0:
        return img
    # PIL rotate is CCW, so clockwise = rotate(360 - degrees)
    return img.rotate(360 - degrees, expand=True)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_raw_pairs() -> List[Dict[str, Any]]:
    """
    Scan all_gtr/ for gtr_<id>.json files, find the matching image,
    and build a list of dicts: {id, image_path, gtr_json, gtr_data}.

    Returns only pairs where a valid image could be resolved.
    """
    CONVERTED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    pairs: List[Dict[str, Any]] = []
    gtr_files = sorted(ALL_GTR_DIR.glob("gtr_*.json"))
    logger.info("Found %d ground-truth JSON files in %s", len(gtr_files), ALL_GTR_DIR)

    for gtr_file in gtr_files:
        # Extract numeric ID from filename: gtr_514.json -> "514"
        match = re.match(r"gtr_(\d+)\.json", gtr_file.name)
        if not match:
            continue
        doc_id = match.group(1)

        # Load and validate ground-truth JSON
        try:
            raw_text = gtr_file.read_text(encoding="utf-8")
            gtr_data = json.loads(raw_text)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping %s — JSON error: %s", gtr_file.name, exc)
            continue

        # Unwrap single-element list GTR files; skip multi-element lists
        if isinstance(gtr_data, list):
            if len(gtr_data) == 1 and isinstance(gtr_data[0], dict):
                gtr_data = gtr_data[0]
                raw_text = json.dumps(gtr_data, ensure_ascii=False)
            else:
                logger.debug("Skipping %s — top-level JSON array", gtr_file.name)
                continue

        # Skip trivially empty GTR files (e.g. just "{}" or very small)
        if not gtr_data or len(raw_text.strip()) < 10:
            logger.debug("Skipping %s — empty / trivial GTR", gtr_file.name)
            continue

        # Resolve the source image
        img_path = resolve_image_path(doc_id, CONVERTED_IMG_DIR)
        if img_path is None:
            logger.debug("No image found for doc_id=%s — skipping", doc_id)
            continue

        pairs.append(
            {
                "id": doc_id,
                "image_path": str(img_path),
                "gtr_json": raw_text.strip(),
                "gtr_data": gtr_data,
            }
        )

    logger.info("Built %d valid image-GTR pairs", len(pairs))
    return pairs


def format_extraction_conversation(pair: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task B — JSON Extraction.
    Converts a raw pair into a Qwen2.5-VL conversation for structured field extraction.
    Photometric augmentation is stored as a flag; actual PIL transform happens in
    formatting_func at batch-load time so augmentation varies each epoch.

    Returns a dict with keys: "id", "images", "messages", "task", "aug_rotate_deg".
    """
    schema_hint, resolved_dt, resolved_dc = load_boilerplate_hint(
        pair["gtr_data"], pair["id"]
    )
    user_content = EXTRACTION_PROMPT
    if schema_hint:
        user_content = user_content.rstrip(".") + f"\n{schema_hint}."

    # Ensure the target ground-truth JSON explicitly includes data_type and data_category
    target_data = dict(pair["gtr_data"]) if isinstance(pair["gtr_data"], dict) else pair["gtr_data"]
    if isinstance(target_data, dict):
        if not target_data.get("data_type") or target_data.get("data_type") == "N/A":
            target_data["data_type"] = resolved_dt
        if not target_data.get("data_category") or target_data.get("data_category") == "N/A":
            target_data["data_category"] = resolved_dc

    # Compact the ground-truth JSON so the target tokens are minimal
    try:
        compact_answer = json.dumps(
            target_data, ensure_ascii=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        compact_answer = pair["gtr_json"]

    return {
        "id": f"doc_{pair['id']}_extraction",
        "images": [pair["image_path"]],
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": compact_answer},
        ],
        "task": "extraction",
        "aug_rotate_deg": 0,       # no geometric rotation for extraction task
    }


def build_rotation_conversations(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Task A — Orientation Classification (Hybrid Augmentation).

    For each document, the GTR `degree` field holds the manually verified
    clockwise correction angle (0, 90, 180, 270) to make the image upright.

    We generate ROTATION_AUG_PER_DOC training examples by applying extra
    synthetic clockwise rotations θ∈{0,90,180,270} to the original image.
    The new corrective label becomes: (D - θ + 360) % 360.

    The target `assistant` response is a structured JSON for accuracy checking:
        {"degree_needed": <int>, "rotation_needed": <bool>}

    Returns a list of conversation dicts (up to 4 per document).
    """
    gtr_degree = int(pair["gtr_data"].get("degree", 0))
    # Clamp to valid values; default 0 if not set
    if gtr_degree not in (0, 90, 180, 270):
        gtr_degree = 0

    # All possible extra synthetic rotations (clockwise)
    all_thetas = [0, 90, 180, 270]
    # Cap to ROTATION_AUG_PER_DOC (always include θ=0 = real label first)
    thetas = all_thetas[:max(1, min(ROTATION_AUG_PER_DOC, 4))]

    conversations = []
    for theta in thetas:
        # New corrective label after applying θ° additional clockwise rotation
        new_degree = (gtr_degree - theta + 360) % 360
        rotation_needed = new_degree != 0

        # Structured JSON answer for accuracy checking
        answer = json.dumps(
            {"degree_needed": new_degree, "rotation_needed": rotation_needed},
            separators=(",", ":")
        )

        conversations.append({
            "id": f"doc_{pair['id']}_rot{theta:03d}",
            "images": [pair["image_path"]],
            "messages": [
                {"role": "user", "content": ORIENTATION_PROMPT},
                {"role": "assistant", "content": answer},
            ],
            "task": "rotation",
            "aug_rotate_deg": theta,    # extra CW degrees to apply at load time
            "gtr_degree": gtr_degree,   # original GTR label (for eval tracing)
        })

    return conversations


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1 - Import Unsloth (must happen before transformers)
    # -----------------------------------------------------------------------
    logger.info("Importing Unsloth ...")
    try:
        from unsloth import FastVisionModel  # type: ignore
    except ImportError as exc:
        logger.error(
            "Unsloth is not installed.\n"
            "Install it with:\n"
            "  pip install 'unsloth[cu124-ampere-torch250]'\n"
            "or visit https://github.com/unslothai/unsloth for your CUDA version.\n"
            "Error: %s",
            exc,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2 - Build dataset (Hybrid: Extraction + Rotation with Augmentation)
    # -----------------------------------------------------------------------
    logger.info("Building dataset from %s ...", DATASET_ROOT)
    raw_pairs = build_raw_pairs()

    if not raw_pairs:
        logger.error(
            "No valid image-GTR pairs found! "
            "Ensure dataset/all_file/ and dataset/all_gtr/ are populated."
        )
        sys.exit(1)

    # Optional sample cap (useful for quick smoke-tests)
    if MAX_SAMPLES and MAX_SAMPLES < len(raw_pairs):
        logger.info("Capping dataset at %d samples (MAX_SAMPLES env var)", MAX_SAMPLES)
        import random
        random.seed(SEED)
        raw_pairs = random.sample(raw_pairs, MAX_SAMPLES)

    # --- Task B: Extraction conversations (1 per document) ---
    extraction_convs = [format_extraction_conversation(p) for p in raw_pairs]

    # --- Task A: Rotation conversations (ROTATION_AUG_PER_DOC per document) ---
    rotation_convs: List[Dict[str, Any]] = []
    for p in raw_pairs:
        rotation_convs.extend(build_rotation_conversations(p))

    # Count label distribution for rotation task
    from collections import Counter
    rot_label_counts = Counter(
        json.loads(c["messages"][-1]["content"]).get("degree_needed", -1)
        for c in rotation_convs
    )
    logger.info(
        "Rotation label distribution — 0°: %d  |  90°: %d  |  180°: %d  |  270°: %d",
        rot_label_counts.get(0, 0),
        rot_label_counts.get(90, 0),
        rot_label_counts.get(180, 0),
        rot_label_counts.get(270, 0),
    )

    # Merge and shuffle all conversations so both tasks interleave each batch
    import random as _random
    all_convs = extraction_convs + rotation_convs
    _random.seed(SEED)
    _random.shuffle(all_convs)

    logger.info(
        "Total conversations: %d  (Extraction: %d  |  Rotation: %d  |  Rotation aug/doc: %d)",
        len(all_convs),
        len(extraction_convs),
        len(rotation_convs),
        ROTATION_AUG_PER_DOC,
    )

    # Train / eval split (preserve task ratio across both splits)
    split_idx = int(len(all_convs) * TRAIN_SPLIT)
    train_convs = all_convs[:split_idx]
    eval_convs  = all_convs[split_idx:] if split_idx < len(all_convs) else []
    eval_rotation_convs = [c for c in eval_convs if c.get("task") == "rotation"]

    logger.info(
        "Train: %d  |  Eval: %d  (split=%.0f%%)  |  Eval rotation examples: %d",
        len(train_convs),
        len(eval_convs),
        TRAIN_SPLIT * 100,
        len(eval_rotation_convs),
    )

    # Convert to HuggingFace Dataset
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        sys.exit(1)

    # Strip internal metadata fields (task, aug_rotate_deg, gtr_degree) before
    # passing to HuggingFace Dataset — they are only needed locally above.
    def _strip_meta(c: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in c.items() if k not in ("task", "aug_rotate_deg", "gtr_degree")}

    train_dataset = Dataset.from_list([_strip_meta(c) for c in train_convs])
    eval_dataset  = Dataset.from_list([_strip_meta(c) for c in eval_convs]) if eval_convs else None

    # -----------------------------------------------------------------------
    # Step 3 - Load model + processor via Unsloth
    # -----------------------------------------------------------------------
    logger.info("Loading model '%s' via Unsloth (4-bit QLoRA) ...", MODEL_NAME)

    model, processor = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,           # QLoRA: 4-bit base weights
        use_gradient_checkpointing="unsloth",  # Unsloth's patch (longer sequences)
    )

    # -----------------------------------------------------------------------
    # Step 4 - Attach LoRA adapters
    # -----------------------------------------------------------------------
    logger.info(
        "Attaching LoRA adapters  rank=%d  alpha=%d  dropout=%.2f ...",
        LORA_RANK,
        LORA_ALPHA,
        LORA_DROPOUT,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,        # fine-tune the visual encoder layers
        finetune_language_layers=True,      # fine-tune the language decoder layers
        finetune_attention_modules=True,    # q, k, v, o projections
        finetune_mlp_modules=True,          # gate, up, down projections
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        random_state=SEED,
        use_rslora=False,                   # standard LoRA scaling
    )

    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Step 5 - Per-example augmentation (runs each epoch via the collator)
    # -----------------------------------------------------------------------
    def augment_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and augment images for one training example.

        UnslothVisionDataCollator calls this once per sample before tokenisation,
        so photometric jitter varies across epochs.
        """
        from PIL import Image as _PIL_Image  # type: ignore

        example = dict(example)
        example_id = example["id"]
        image_paths = example["images"]
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        is_rotation_task = "_rot" in example_id
        aug_cw_deg = 0
        if is_rotation_task:
            try:
                aug_cw_deg = int(example_id.rsplit("_rot", 1)[-1])
            except ValueError:
                aug_cw_deg = 0

        pil_images: List[Any] = []
        for img_p in image_paths:
            try:
                img = _PIL_Image.open(img_p).convert("RGB")
                if aug_cw_deg != 0:
                    img = rotate_image_clockwise(img, aug_cw_deg)
                img = augment_image_photometric(img)
                pil_images.append(img)
            except Exception as exc:
                logger.warning("Cannot open/augment image %s: %s", img_p, exc)
                pil_images.append(_PIL_Image.new("RGB", (64, 64), color=(128, 128, 128)))

        example["images"] = pil_images

        # Convert "<image>\n..." string content to multi-part list format
        # required by Qwen2.5-VL processor for proper vision token injection.
        messages = example.get("messages", [])
        new_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str) and "<image>" in content:
                parts = content.split("<image>", 1)
                content_list = []
                if parts[0].strip():
                    content_list.append({"type": "text", "text": parts[0]})
                content_list.append({"type": "image"})
                if parts[1].strip():
                    content_list.append({"type": "text", "text": parts[1].lstrip("\n")})
                new_messages.append({"role": msg["role"], "content": content_list})
            else:
                new_messages.append(msg)
        example["messages"] = new_messages

        return example

    # -----------------------------------------------------------------------
    # Step 6 - Data collator
    # -----------------------------------------------------------------------
    try:
        from unsloth.trainer import UnslothVisionDataCollator  # type: ignore

        data_collator = UnslothVisionDataCollator(
            model,
            processor,
            max_seq_length=MAX_SEQ_LEN,
            formatting_func=augment_example,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user",
            response_part="<|im_start|>assistant",
        )
        logger.info("Using UnslothVisionDataCollator")
    except ImportError:
        # Fallback: use TRL DataCollatorForCompletionOnlyLM
        try:
            from trl import DataCollatorForCompletionOnlyLM  # type: ignore

            response_template = "<|im_start|>assistant"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template,
                tokenizer=processor.tokenizer,
            )
            logger.info("Using TRL DataCollatorForCompletionOnlyLM as fallback")
        except ImportError:
            from transformers import DataCollatorWithPadding  # type: ignore

            data_collator = DataCollatorWithPadding(processor.tokenizer)
            logger.info("Using DataCollatorWithPadding as last-resort fallback")

    # -----------------------------------------------------------------------
    # Step 7 - Training arguments
    # -----------------------------------------------------------------------
    logger.info("Configuring training arguments ...")
    try:
        from trl import SFTConfig  # type: ignore
    except ImportError:
        logger.error("trl not installed. Run: pip install trl")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,                          # L4 / A10G / A100 support bfloat16
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",             # 8-bit Adam saves ~2x optimiser memory
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=SAVE_STEPS if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        report_to="none",                   # set to "wandb" if you use W&B
        remove_unused_columns=False,        # keep images/messages columns
        dataloader_num_workers=0,
        seed=SEED,
        overwrite_output_dir=True,
        # Required for Unsloth vision fine-tuning
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=MAX_SEQ_LEN,
    )

    # -----------------------------------------------------------------------
    # Step 8 - Initialise SFTTrainer (Unsloth-patched)
    # -----------------------------------------------------------------------
    logger.info("Initialising SFTTrainer ...")
    try:
        from trl import SFTTrainer  # type: ignore
    except ImportError:
        logger.error("trl not installed. Run: pip install trl")
        sys.exit(1)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # -----------------------------------------------------------------------
    # Step 9 - Train!
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting fine-tuning ...")
    logger.info("  Model         : %s", MODEL_NAME)
    logger.info("  LoRA rank     : %d", LORA_RANK)
    logger.info("  LoRA alpha    : %d", LORA_ALPHA)
    logger.info("  Epochs        : %d", EPOCHS)
    logger.info(
        "  Batch size    : %d (x%d accum = eff. %d)",
        PER_DEVICE_BATCH,
        GRAD_ACCUM,
        PER_DEVICE_BATCH * GRAD_ACCUM,
    )
    logger.info("  Learning rate : %.2e", LEARNING_RATE)
    logger.info("  Max seq len   : %d", MAX_SEQ_LEN)
    logger.info("  Train samples : %d  (Extraction: ~%d  |  Rotation: ~%d)",
                 len(train_dataset),
                 len(train_dataset) // (1 + ROTATION_AUG_PER_DOC),
                 len(train_dataset) * ROTATION_AUG_PER_DOC // (1 + ROTATION_AUG_PER_DOC))
    logger.info("  Rotation aug  : %d synthetic rotations per document", ROTATION_AUG_PER_DOC)
    logger.info("  Photo aug     : %s (prob=%.2f)", "ON" if PHOTO_AUG_ENABLED else "OFF", PHOTO_AUG_PROB)
    logger.info("  Output dir    : %s", OUTPUT_DIR)
    logger.info("=" * 60)

    train_result = trainer.train()

    # -----------------------------------------------------------------------
    # Step 10 - Save LoRA adapter + processor
    # -----------------------------------------------------------------------
    logger.info("Training complete. Saving LoRA adapter ...")
    trainer.save_model(str(OUTPUT_DIR))  # saves adapter_config.json + adapter weights
    processor.save_pretrained(str(OUTPUT_DIR))

    # Also save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("LoRA adapter saved to: %s", OUTPUT_DIR)
    logger.info(
        "Run `python evaluate.py` to compute full accuracy, F1, and generate charts."
    )

    # -----------------------------------------------------------------------
    # Step 11 - (Optional) Merge adapter into base weights
    # -----------------------------------------------------------------------
    merge_env = os.environ.get("MERGE_WEIGHTS", "0")
    if merge_env.strip().lower() in ("1", "true", "yes"):
        merged_output = (
            REPO_ROOT / "saves" / _model_basename / "merged" / "industrial_ocr_v1"
        )
        merged_output.mkdir(parents=True, exist_ok=True)
        logger.info("Merging LoRA weights into base model -> %s", merged_output)
        try:
            # Unsloth helper — saves in 16-bit for direct vLLM loading
            model.save_pretrained_merged(
                str(merged_output),
                processor,
                save_method="merged_16bit",
            )
            logger.info("Merged model saved to: %s", merged_output)
        except AttributeError:
            # Fallback: PEFT merge
            logger.info("Falling back to PEFT merge ...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(merged_output))
            processor.save_pretrained(str(merged_output))
            logger.info("Merged model saved to: %s", merged_output)
            del merged_model
            gc.collect()

    # -----------------------------------------------------------------------
    # Finished
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Fine-tuning complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Deploy the LoRA adapter (~150 MB) from:")
    logger.info("       %s", OUTPUT_DIR)
    logger.info("")
    logger.info("  2. In root/model.py, load the fine-tuned model:")
    logger.info("       from peft import PeftModel")
    logger.info(
        "       base = Qwen2_5_VLForConditionalGeneration.from_pretrained("
    )
    logger.info(
        "           'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype=torch.float16,"
    )
    logger.info("           device_map='auto')")
    logger.info("       model = PeftModel.from_pretrained(base, '%s')", OUTPUT_DIR)
    logger.info(
        "       model = model.merge_and_unload()  # zero inference overhead"
    )
    logger.info("")
    logger.info(
        "  3. For batch / high-throughput inference, set MERGE_WEIGHTS=1"
    )
    logger.info("     and serve the merged model with vLLM:")
    logger.info(
        "       vllm serve %s/merged/industrial_ocr_v1 --limit-mm-per-prompt image=1",
        REPO_ROOT / "saves" / _model_basename,
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

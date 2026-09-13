#!/usr/bin/env python3
"""
=============================================================================
  GTR Orientation & Rotation Updater Script (`update_gtr_orientation.py`)
=============================================================================

Overview
--------
This script updates all Ground Truth (GTR) JSON files (`gtr_<id>.json`) in `all_gtr/`
with exact rotation orientation labels (`rotation_needed` and `degree`) based on the
corresponding image or PDF in `all_file/`.

If a document (e.g., multi-page PDF) contains multiple pages/images, this script
evaluates each page individually and updates the GTR file with both top-level
orientation (for Page 1) and per-page orientation records (`page_orientations`).

Handles both JSON object root (`{...}`) and JSON array root (`[...]`) structures.

Skip Already Processed Logic
----------------------------
By default, this script skips any GTR file that ALREADY has `rotation_needed` and `degree`
written (`Skipped (already updated: degree=...)`). This means you can stop/resume the
script anytime (`Ctrl+C`), and it will never repeat work unless `--force` is passed!

Unified Multi-Provider Pooling (Groq + Gemini) & Automatic Rate-Limit Handling
------------------------------------------------------------------------------
To run at maximum throughput and handle API rate limits across free-tier keys:
1. `UnifiedKeyPool`: Automatically pools both **Groq API keys** (`30 RPM`) and **Gemini API keys** (`5 RPM/key`).
2. `Groq Vision Support`: Uses `meta-llama/llama-4-scout-17b-16e-instruct` via OpenAI-compatible vision endpoints.
3. `Gemini Auto-Routing`: Automatically detects if a key requires `gemini-2.0-flash` vs `gemini-2.5-flash` on 404 NOT_FOUND.
4. `Quota Backoff & Cooldowns`: When any key returns a 429 quota error, that exact key is put on cooldown and
   the script rotates to the next available key across providers.
5. `Automatic Sleep`: If all keys reach quota limits simultaneously, worker threads wait automatically
   for the cooldown window to reset before resuming.

Output Schema Added to GTR JSON:
--------------------------------
For single-page / image documents:
{
  "rotation_needed": false,  # true if degree != 0
  "degree": 0,               # 0, 90, 180, or 270 (degrees clockwise to upright)
  ... (existing fields preserved exactly)
}

For multi-page PDFs / multi-image documents:
{
  "rotation_needed": false,
  "degree": 0,
  "page_orientations": [
    {
      "page_number": 1,
      "image_index": 1,
      "rotation_needed": false,
      "degree": 0
    },
    {
      "page_number": 2,
      "image_index": 2,
      "rotation_needed": true,
      "degree": 90
    }
  ],
  ...
}

Usage:
------
  # Run across all files using hybrid mode with 6 workers across Groq & Gemini keys:
  ./.venv/bin/python3 update_gtr_orientation.py --workers 6

  # Run using ONLY Groq:
  ./.venv/bin/python3 update_gtr_orientation.py --provider groq --workers 4

  # Pass custom API keys (Groq 'gsk_...' and/or Gemini 'AIza...'):
  ./.venv/bin/python3 update_gtr_orientation.py --api-keys gsk_MY_GROQ_KEY AIzaMyGeminiKey
=============================================================================
"""

import os
import sys
import glob
import json
import argparse
import logging
import re
import time
import base64
import threading
import requests
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing optional image/PDF libraries
# ---------------------------------------------------------------------------
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        pypdf = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


# ---------------------------------------------------------------------------
# Unified Key Pool (Groq + Gemini) with Automatic Cooldown & Routing
# ---------------------------------------------------------------------------
class KeyPoolEntry:
    def __init__(self, key: str, provider: str, default_model: str):
        self.key = key
        self.provider = provider  # 'groq' or 'gemini'
        self.model = default_model
        self.cooldown_until = 0.0
        self.client = None


class UnifiedKeyPool:
    def __init__(self, api_keys: List[str], provider_filter: str = "auto", groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct", gemini_model: str = "gemini-2.5-flash"):
        self.entries: List[KeyPoolEntry] = []
        self.lock = threading.Lock()
        self.index = 0

        # Deduplicate keys while keeping order
        seen = set()
        cleaned_keys = []
        for k in api_keys:
            ck = k.strip().strip('"\'')
            if ck and ck not in seen:
                seen.add(ck)
                cleaned_keys.append(ck)

        # Classify and initialize keys
        for k in cleaned_keys:
            if k.startswith("gsk_"):
                if provider_filter in ["auto", "groq"]:
                    entry = KeyPoolEntry(k, "groq", groq_model)
                    self.entries.append(entry)
            else:
                if provider_filter in ["auto", "gemini"]:
                    entry = KeyPoolEntry(k, "gemini", gemini_model)
                    try:
                        from google import genai
                        entry.client = genai.Client(api_key=k)
                        self.entries.append(entry)
                    except Exception as exc:
                        logger.warning("Could not init Gemini client for key %s...: %s", k[:10], exc)

        if not self.entries:
            raise RuntimeError(f"No valid API keys active for provider mode '{provider_filter}'!")

        groq_count = sum(1 for e in self.entries if e.provider == "groq")
        gemini_count = sum(1 for e in self.entries if e.provider == "gemini")
        logger.info("Initialized UnifiedKeyPool with %d total keys (Groq: %d, Gemini: %d).", len(self.entries), groq_count, gemini_count)

    def get_entry(self) -> KeyPoolEntry:
        """Get the next available KeyPoolEntry. Blocks if all keys are on cooldown."""
        while True:
            with self.lock:
                now = time.time()
                available = [e for e in self.entries if e.cooldown_until <= now]
                if available:
                    self.index = (self.index + 1) % len(available)
                    return available[self.index]

                # All keys on cooldown
                soonest = min(self.entries, key=lambda e: e.cooldown_until)
                wait_time = max(1.0, soonest.cooldown_until - now)

            logger.info("All %d API keys reached rate limit quota. Waiting %.1f seconds for next available quota window...", len(self.entries), wait_time)
            time.sleep(wait_time)

    def report_rate_limit(self, entry: KeyPoolEntry, retry_delay: float = 15.0):
        """Mark an entry as rate-limited for `retry_delay` seconds."""
        with self.lock:
            target_time = time.time() + retry_delay
            entry.cooldown_until = max(entry.cooldown_until, target_time)
            logger.warning("[%s] API key %s... hit 429 limit. Rotating key & putting on %.1f s cooldown.", entry.provider.upper(), entry.key[:10], retry_delay)

    def switch_model_for_entry(self, entry: KeyPoolEntry, new_model: str):
        """Switch preferred model on 404 NOT_FOUND."""
        with self.lock:
            if entry.model != new_model:
                logger.info("[%s] Auto-routing key %s... from %s to %s.", entry.provider.upper(), entry.key[:10], entry.model, new_model)
                entry.model = new_model


# ---------------------------------------------------------------------------
# Helper: Resolve corresponding file path for a doc_id
# ---------------------------------------------------------------------------
def resolve_file_path(doc_id: str, file_dir: Path) -> Optional[Path]:
    """Find the corresponding image or PDF file for a given numeric doc_id."""
    extensions = [".pdf", ".PDF", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    for ext in extensions:
        candidate = file_dir / f"{doc_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Method 1: Metadata Orientation Extraction
# ---------------------------------------------------------------------------
def get_metadata_orientation(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract orientation degree (0, 90, 180, 270) per page/image using file metadata.
    Returns a list of dicts: [{"page_number": 1, "image_index": 1, "degree": 0}, ...]
    """
    ext = file_path.suffix.lower()
    results: List[Dict[str, Any]] = []

    if ext == ".pdf":
        if not pypdf:
            return [{"page_number": 1, "image_index": 1, "degree": 0}]
        try:
            reader = pypdf.PdfReader(str(file_path))
            for i, page in enumerate(reader.pages):
                rot = page.get("/Rotate", 0)
                try:
                    rot_int = int(rot) % 360
                except (ValueError, TypeError):
                    rot_int = 0
                results.append({
                    "page_number": i + 1,
                    "image_index": i + 1,
                    "degree": rot_int
                })
            return results if results else [{"page_number": 1, "image_index": 1, "degree": 0}]
        except Exception as exc:
            logger.debug("Error reading PDF metadata for %s: %s", file_path.name, exc)
            return [{"page_number": 1, "image_index": 1, "degree": 0}]

    elif ext in [".jpg", ".jpeg", ".png"]:
        if not Image:
            return [{"page_number": 1, "image_index": 1, "degree": 0}]
        try:
            with Image.open(file_path) as img:
                exif = img.getexif()
                orient = exif.get(0x0112, 1) if exif else 1
                # Map EXIF orientation to clockwise degree needed to make upright
                if orient in [3, 4]:
                    deg = 180
                elif orient in [6, 7]:
                    deg = 270  # Image is 90 CW rotated, need 270 CW to correct
                elif orient in [5, 8]:
                    deg = 90   # Image is 270 CW rotated, need 90 CW to correct
                else:
                    deg = 0
                return [{"page_number": 1, "image_index": 1, "degree": deg}]
        except Exception as exc:
            logger.debug("Error reading EXIF metadata for %s: %s", file_path.name, exc)
            return [{"page_number": 1, "image_index": 1, "degree": 0}]

    return [{"page_number": 1, "image_index": 1, "degree": 0}]


# ---------------------------------------------------------------------------
# Method 2: Vision Orientation Extraction (Groq + Gemini with Pool Retry)
# ---------------------------------------------------------------------------
def get_vision_orientation(key_pool: UnifiedKeyPool, file_path: Path) -> List[Dict[str, Any]]:
    """
    Use Vision AI across Groq and Gemini pools to classify required clockwise rotation degree.
    If file_path is a multi-page PDF, converts each page to a PIL Image and checks each page.
    Automatically catches 429 quota errors and 404 model errors, routing keys cleanly.
    """
    if not Image:
        raise ImportError("PIL (Pillow) is required for Vision orientation checks.")

    ext = file_path.suffix.lower()
    pil_images: List[Any] = []

    if ext == ".pdf":
        if not convert_from_path:
            logger.warning("pdf2image not installed, checking metadata for PDF %s instead", file_path.name)
            return get_metadata_orientation(file_path)
        try:
            pil_images = convert_from_path(str(file_path), dpi=150)
        except Exception as exc:
            logger.warning("Could not convert PDF %s to images: %s. Using metadata.", file_path.name, exc)
            return get_metadata_orientation(file_path)
    else:
        try:
            pil_images = [Image.open(file_path).convert("RGB")]
        except Exception as exc:
            logger.warning("Could not open image %s: %s. Defaulting to 0.", file_path.name, exc)
            return [{"page_number": 1, "image_index": 1, "degree": 0}]

    if not pil_images:
        return [{"page_number": 1, "image_index": 1, "degree": 0}]

    results: List[Dict[str, Any]] = []
    prompt = (
        "Examine this document image carefully. "
        "What clockwise rotation in degrees (choose exactly from: 0, 90, 180, 270) is needed "
        "to make the text correctly upright and readable? "
        "Reply with ONLY the number (0, 90, 180, or 270) and nothing else."
    )

    for i, img in enumerate(pil_images):
        deg = 0
        max_retries = 8
        for attempt in range(max_retries):
            entry = key_pool.get_entry()
            try:
                if entry.provider == "groq":
                    # Convert PIL image to base64 JPEG
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    headers = {
                        "Authorization": f"Bearer {entry.key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": entry.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                                ]
                            }
                        ]
                    }
                    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
                    if resp.status_code == 200:
                        raw_text = resp.json()["choices"][0]["message"]["content"].strip()
                    elif resp.status_code == 429:
                        raise Exception(f"429 Too Many Requests: {resp.text}")
                    else:
                        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
                else:
                    # Gemini
                    response = entry.client.models.generate_content(model=entry.model, contents=[img, prompt])
                    raw_text = (response.text or "").strip()

                matches = re.findall(r"\b(0|90|180|270)\b", raw_text)
                if matches:
                    deg = int(matches[0])
                else:
                    try:
                        val = int(re.sub(r"\D", "", raw_text))
                        if val in [0, 90, 180, 270]:
                            deg = val
                    except ValueError:
                        deg = 0
                break  # Success!
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str and ("not available" in err_str or "NOT_FOUND" in err_str):
                    if entry.provider == "gemini":
                        fallback_model = "gemini-2.0-flash" if entry.model != "gemini-2.0-flash" else "gemini-1.5-flash"
                        key_pool.switch_model_for_entry(entry, fallback_model)
                    continue
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str or "Too Many Requests" in err_str:
                    delay_match = re.search(r"Please retry in ([0-9.]+)s", err_str)
                    if not delay_match:
                        delay_match = re.search(r"retryDelay': '([0-9.]+)s'", err_str)
                    delay = float(delay_match.group(1)) + 2.0 if delay_match else 15.0
                    key_pool.report_rate_limit(entry, retry_delay=delay)
                    if attempt == max_retries - 1:
                        logger.warning("Max retries exceeded on page %d of %s after quota errors. Falling back to metadata.", i + 1, file_path.name)
                        meta_res = get_metadata_orientation(file_path)
                        deg = meta_res[i]["degree"] if i < len(meta_res) else 0
                else:
                    logger.warning("Non-quota error calling %s on %s (page %d): %s", entry.provider.upper(), file_path.name, i + 1, exc)
                    meta_res = get_metadata_orientation(file_path)
                    deg = meta_res[i]["degree"] if i < len(meta_res) else 0
                    break

        results.append({
            "page_number": i + 1,
            "image_index": i + 1,
            "degree": deg
        })

    return results


# ---------------------------------------------------------------------------
# Method 3: Hybrid Orientation Extraction
# ---------------------------------------------------------------------------
def get_hybrid_orientation(key_pool: Optional[UnifiedKeyPool], file_path: Path) -> List[Dict[str, Any]]:
    """
    Check metadata first. If metadata says any page needs rotation (!= 0), trust metadata.
    If metadata says all pages are 0 and key_pool is available, visually verify using AI pool.
    """
    meta_results = get_metadata_orientation(file_path)
    if any(item["degree"] != 0 for item in meta_results):
        return meta_results

    if key_pool is not None:
        return get_vision_orientation(key_pool, file_path)

    return meta_results


# ---------------------------------------------------------------------------
# Single File Processor with Skip Logic (`if one gtr is done... then skip it`)
# ---------------------------------------------------------------------------
def process_single_gtr(gtr_path: Path, file_dir: Path, method: str, key_pool: Optional[UnifiedKeyPool], force: bool) -> Tuple[bool, str]:
    """
    Process a single GTR JSON file and update its orientation fields.
    Handles dict ({...}) and list ([...]) root objects.
    SKIPS any file where `rotation_needed` and `degree` are already present (unless force=True).
    """
    match = re.match(r"gtr_(\d+)\.json", gtr_path.name)
    if not match:
        return False, f"Skipped invalid filename {gtr_path.name}"

    doc_id = match.group(1)
    source_file = resolve_file_path(doc_id, file_dir)
    if not source_file:
        return False, f"Source file not found for doc_id {doc_id} ({gtr_path.name})"

    try:
        raw_text = gtr_path.read_text(encoding="utf-8")
        gtr_data = json.loads(raw_text)
    except Exception as exc:
        return False, f"Failed reading/parsing {gtr_path.name}: {exc}"

    # =========================================================================
    # SKIP LOGIC: If already updated with rotation labels, immediately skip!
    # =========================================================================
    if isinstance(gtr_data, dict):
        if not force and "rotation_needed" in gtr_data and "degree" in gtr_data:
            return True, f"Skipped (already updated: degree={gtr_data['degree']})"
    elif isinstance(gtr_data, list) and gtr_data and isinstance(gtr_data[0], dict):
        if not force and "rotation_needed" in gtr_data[0] and "degree" in gtr_data[0]:
            return True, f"Skipped (already updated: degree={gtr_data[0]['degree']})"

    # Determine per-page orientations
    if method == "gemini" or method == "groq" or method == "vision":
        if key_pool is None:
            return False, "Key pool not initialized (check API keys)"
        orientations = get_vision_orientation(key_pool, source_file)
    elif method == "metadata":
        orientations = get_metadata_orientation(source_file)
    else:
        # hybrid
        orientations = get_hybrid_orientation(key_pool, source_file)

    # Format page orientations with rotation_needed flag
    page_records = []
    for item in orientations:
        deg = item["degree"]
        page_records.append({
            "page_number": item["page_number"],
            "image_index": item["image_index"],
            "rotation_needed": bool(deg != 0),
            "degree": deg
        })

    # Top-level orientation reflects Page 1 (for single-page script compatibility)
    primary_deg = page_records[0]["degree"] if page_records else 0
    primary_needed = bool(primary_deg != 0)

    # Update gtr_data cleanly supporting dict or list roots
    if isinstance(gtr_data, dict):
        updated_gtr = {
            "rotation_needed": primary_needed,
            "degree": primary_deg
        }
        if len(page_records) > 1:
            updated_gtr["page_orientations"] = page_records
        for k, v in gtr_data.items():
            if k not in ["rotation_needed", "degree", "page_orientations"]:
                updated_gtr[k] = v
    elif isinstance(gtr_data, list):
        updated_list = []
        for item in gtr_data:
            if isinstance(item, dict):
                new_item = {
                    "rotation_needed": primary_needed,
                    "degree": primary_deg
                }
                if len(page_records) > 1:
                    new_item["page_orientations"] = page_records
                for k, v in item.items():
                    if k not in ["rotation_needed", "degree", "page_orientations"]:
                        new_item[k] = v
                updated_list.append(new_item)
            else:
                updated_list.append(item)
        updated_gtr = updated_list
    else:
        return False, f"Unsupported JSON root structure type: {type(gtr_data)}"

    # Write back formatted JSON
    try:
        gtr_path.write_text(json.dumps(updated_gtr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True, f"Updated {gtr_path.name} (degree={primary_deg}, pages={len(page_records)})"
    except Exception as exc:
        return False, f"Failed writing {gtr_path.name}: {exc}"


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Update GTR JSON files with orientation & rotation labels using Unified Key Pool.")
    parser.add_argument("--gtr-dir", type=str, default="./all_gtr", help="Directory containing gtr_*.json files")
    parser.add_argument("--file-dir", type=str, default="./all_file", help="Directory containing raw document images/PDFs")
    parser.add_argument("--method", type=str, choices=["hybrid", "metadata", "vision"], default="hybrid", help="Orientation detection method")
    parser.add_argument("--provider", type=str, choices=["auto", "groq", "gemini"], default="auto", help="Provider pool selection")
    parser.add_argument("--groq-model", type=str, default="meta-llama/llama-4-scout-17b-16e-instruct", help="Groq Vision model name")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash", help="Gemini Vision model name")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already updated GTR files")
    parser.add_argument("--api-keys", nargs="+", default=[], help="List of Groq (gsk_...) or Gemini (AIza...) API keys")
    args = parser.parse_args()

    gtr_dir = Path(args.gtr_dir).resolve()
    file_dir = Path(args.file_dir).resolve()

    if not gtr_dir.exists():
        logger.error("GTR directory does not exist: %s", gtr_dir)
        sys.exit(1)
    if not file_dir.exists():
        logger.error("File directory does not exist: %s", file_dir)
        sys.exit(1)

    # Gather API keys from arguments, env var, or .env
    pool_keys = list(args.api_keys)
    if os.environ.get("GROQ_API_KEY"):
        pool_keys.append(os.environ["GROQ_API_KEY"])
    if os.environ.get("GEMINI_API_KEYS"):
        pool_keys.extend(os.environ["GEMINI_API_KEYS"].split(","))
    if os.environ.get("GEMINI_API_KEY"):
        pool_keys.append(os.environ["GEMINI_API_KEY"])

    # Check local .env files
    env_paths = [Path(__file__).parent / ".env", Path.cwd() / ".env"]
    for p in env_paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY=") or line.startswith("GROQ_API_KEY="):
                    pool_keys.append(line.split("=", 1)[1].strip().strip('"\''))

    if not pool_keys and args.method in ["vision", "hybrid"]:
        logger.warning(
            "No API keys found. Please set GEMINI_API_KEY / GROQ_API_KEY in environment or pass --api-keys."
        )

    key_pool = None
    if args.method in ["vision", "hybrid"]:
        try:
            key_pool = UnifiedKeyPool(pool_keys, provider_filter=args.provider, groq_model=args.groq_model, gemini_model=args.gemini_model)
        except Exception as exc:
            if args.method == "vision":
                logger.error("Failed to initialize UnifiedKeyPool: %s", exc)
                sys.exit(1)
            else:
                logger.warning("Could not initialize UnifiedKeyPool (%s). Hybrid will fallback to metadata only.", exc)

    gtr_files = sorted(gtr_dir.glob("gtr_*.json"))
    if args.limit > 0:
        gtr_files = gtr_files[:args.limit]

    logger.info("Starting processing on %d GTR files across %s using method '%s' (%d workers)...", len(gtr_files), gtr_dir, args.method, args.workers)

    success_count = 0
    failure_count = 0
    skipped_count = 0

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_single_gtr, gtr_path, file_dir, args.method, key_pool, args.force): gtr_path
                for gtr_path in gtr_files
            }
            for i, future in enumerate(as_completed(future_to_file), 1):
                gtr_path = future_to_file[future]
                try:
                    success, msg = future.result()
                    if success:
                        if "Skipped" in msg:
                            skipped_count += 1
                            logger.debug("[%d/%d] %s", i, len(gtr_files), msg)
                        else:
                            success_count += 1
                            logger.info("[%d/%d] %s", i, len(gtr_files), msg)
                    else:
                        failure_count += 1
                        logger.warning("[%d/%d] %s", i, len(gtr_files), msg)
                except Exception as exc:
                    failure_count += 1
                    logger.error("[%d/%d] Exception on %s: %s", i, len(gtr_files), gtr_path.name, exc)
                if i % 100 == 0 or i == len(gtr_files):
                    logger.info("Progress: %d / %d processed (Updated: %d, Skipped: %d, Failed: %d)", i, len(gtr_files), success_count, skipped_count, failure_count)
    else:
        for i, gtr_path in enumerate(gtr_files, 1):
            success, msg = process_single_gtr(gtr_path, file_dir, args.method, key_pool, args.force)
            if success:
                if "Skipped" in msg:
                    skipped_count += 1
                    logger.debug("[%d/%d] %s", i, len(gtr_files), msg)
                else:
                    success_count += 1
                    logger.info("[%d/%d] %s", i, len(gtr_files), msg)
            else:
                failure_count += 1
                logger.warning("[%d/%d] %s", i, len(gtr_files), msg)
            if i % 100 == 0 or i == len(gtr_files):
                logger.info("Progress: %d / %d processed (Updated: %d, Skipped: %d, Failed: %d)", i, len(gtr_files), success_count, skipped_count, failure_count)

    logger.info("Completed! Updated: %d  |  Skipped (Already Processed): %d  |  Failed: %d", success_count, skipped_count, failure_count)


if __name__ == "__main__":
    main()

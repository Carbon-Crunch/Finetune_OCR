#!/usr/bin/env python3
"""
=============================================================================
  Unified Extraction & Rotation Pipeline (`batch_processor.py`)
=============================================================================

Overview
--------
This script processes document images/PDFs directly from `all_file/` (`--file-dir`)
and outputs or updates structured Ground Truth (GTR) JSON files in `all_gtr/` (`--gtr-dir`).
It has been upgraded to ignore the old folder-based batch system (`all_batches_zip/Batch_X`)
and operate cleanly on single files, multiple specific files, or the entire `all_file/` dataset.

Combined Extraction + Orientation Rotation in ONE API Call
----------------------------------------------------------
Whenever a file is processed through the LLM, the prompt mandates that the model
extract BOTH the structured data fields (matching the Classifier Registry/Boilerplates)
AND perform the clockwise rotation classification (`rotation_needed`, `degree`, and `page_orientations`)
within the EXACT SAME single API response!

Output JSON Structure (`gtr_<doc_id>.json`):
--------------------------------------------
{
  "rotation_needed": false,
  "degree": 0,
  "page_orientations": [
    {
      "page_number": 1,
      "image_index": 1,
      "rotation_needed": false,
      "degree": 0
    }
  ],
  ... (Extracted boilerplate fields or multi_template_extraction object)
}

Multi-Key & Multi-Provider API Pool
-----------------------------------
Embedded with automatic multi-key rotation (all 5 user keys + .env keys) to handle
free-tier API rate limits (`429 RESOURCE_EXHAUSTED`) and model auto-routing (`404 NOT_FOUND`).

Usage Examples:
---------------
  # 1. Process a single file by ID or filename:
  ./.venv/bin/python3 batch_processor.py 1000.jpg
  ./.venv/bin/python3 batch_processor.py 1000

  # 2. Process multiple specific files:
  ./.venv/bin/python3 batch_processor.py 1000 1001.pdf 1002.jpg

  # 3. Process ALL files in all_file/ (with 4 concurrent workers):
  ./.venv/bin/python3 batch_processor.py --all --workers 4

  # 4. Process all remaining files (limiting to 50):
  ./.venv/bin/python3 batch_processor.py --all --limit 50 --workers 4
=============================================================================
"""

import os
import re
import json
import glob
import sys
import time
import argparse
import logging
import threading
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

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ---------------------------------------------------------------------------
# Multi-Key Pool for Gemini API with Automatic 429 Cooldown & Model Auto-Routing
# ---------------------------------------------------------------------------
class GeminiKeyPool:
    def __init__(self, api_keys: List[str], default_model: str = "gemini-3.1-flash-lite"):
        seen = set()
        self.api_keys = []
        for k in api_keys:
            cleaned = k.strip().strip('"\'')
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                self.api_keys.append(cleaned)

        self.clients: Dict[str, Any] = {}
        self.key_cooldowns: Dict[str, float] = {k: 0.0 for k in self.api_keys}
        self.key_models: Dict[str, str] = {k: default_model for k in self.api_keys}
        self.lock = threading.Lock()
        self.index = 0

        if not self.api_keys:
            raise RuntimeError("No API keys provided to pool!")

        if not genai:
            raise ImportError("google-genai SDK not installed.")

        for k in self.api_keys:
            try:
                self.clients[k] = genai.Client(api_key=k)
            except Exception as exc:
                logger.warning("Could not init client for key %s...: %s", k[:10], exc)

        if not self.clients:
            raise RuntimeError("No valid API clients could be initialized!")

        logger.info("Initialized GeminiKeyPool with %d active API keys.", len(self.clients))

    def get_client(self) -> Tuple[Any, str, str]:
        """Get the next available (client, key, model_to_use). Blocks if all keys are on cooldown."""
        while True:
            with self.lock:
                now = time.time()
                available = [k for k in self.api_keys if k in self.clients and self.key_cooldowns[k] <= now]
                if available:
                    self.index = (self.index + 1) % len(available)
                    chosen = available[self.index]
                    return self.clients[chosen], chosen, self.key_models[chosen]

                soonest = min((k for k in self.api_keys if k in self.clients), key=lambda k: self.key_cooldowns[k])
                wait_time = max(1.0, self.key_cooldowns[soonest] - now)

            logger.info("All %d API keys reached rate limit quota. Waiting %.1f seconds for quota window...", len(self.clients), wait_time)
            time.sleep(wait_time)

    def report_rate_limit(self, key: str, retry_delay: float = 15.0):
        with self.lock:
            target_time = time.time() + retry_delay
            self.key_cooldowns[key] = max(self.key_cooldowns.get(key, 0.0), target_time)
            logger.warning("API key %s... hit 429 limit. Rotating key & putting on %.1f s cooldown.", key[:10], retry_delay)

    def switch_model_for_key(self, key: str, new_model: str):
        with self.lock:
            if self.key_models.get(key) != new_model:
                logger.info("Auto-routing key %s... from %s to %s.", key[:10], self.key_models.get(key), new_model)
                self.key_models[key] = new_model


# ---------------------------------------------------------------------------
# Config Resolution & Registry Loading
# ---------------------------------------------------------------------------
REGISTRY_PATH = None
for r_path in ["./configs/boilerplates/registry.json", "./configs/registry.json", "./configs_1/configs/registry.json", "configs/boilerplates/registry.json"]:
    if os.path.exists(r_path):
        REGISTRY_PATH = r_path
        break

if not REGISTRY_PATH:
    matches = glob.glob("**/registry.json", recursive=True)
    if matches:
        REGISTRY_PATH = matches[0]
    else:
        logger.warning("Registry file 'registry.json' not found in current workspace.")

FALLBACK_PATH = None
if REGISTRY_PATH:
    for f_path in ["./configs/generic_fallback.json", "./configs_1/configs/generic_fallback.json", os.path.join(os.path.dirname(REGISTRY_PATH), "generic_fallback.json"), os.path.join(os.path.dirname(os.path.dirname(REGISTRY_PATH)), "generic_fallback.json")]:
        if os.path.exists(f_path):
            FALLBACK_PATH = f_path
            break

BOILERPLATES_DIR = None
if REGISTRY_PATH:
    for b_dir in ["./configs/boilerplates", "./configs_1/configs/boilerplates", os.path.dirname(REGISTRY_PATH)]:
        if os.path.exists(b_dir) and os.path.isdir(b_dir):
            BOILERPLATES_DIR = b_dir
            break

REGISTRY = {}
if REGISTRY_PATH and os.path.exists(REGISTRY_PATH):
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            REGISTRY = json.load(f)
    except Exception as e:
        logger.error("Could not load registry: %s", e)

FALLBACK_JSON = None
if FALLBACK_PATH and os.path.exists(FALLBACK_PATH):
    try:
        with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
            FALLBACK_JSON = json.load(f)
    except Exception as e:
        logger.error("Could not load fallback JSON: %s", e)

# Load final_consolidated.json lookup
CONSOLIDATED_LOOKUP = {}
for root_json_path in ["final_consolidated.json", "final_consolidated-old.json", "consolidated.json", "batch_json/consolidated.json"]:
    if os.path.exists(root_json_path):
        try:
            with open(root_json_path, "r", encoding="utf-8") as f:
                cons_data = json.load(f)
                for entry in cons_data.get("consolidated_data", []):
                    fn = entry.get("file_name")
                    if fn is not None:
                        CONSOLIDATED_LOOKUP[str(fn)] = entry
                        try:
                            CONSOLIDATED_LOOKUP[int(fn)] = entry
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logger.warning("Could not load %s: %s", root_json_path, e)

# Build a catalog of unique (data_type, data_category) pairs
SIMILAR_EXAMPLES_CATALOG = {}
for item in CONSOLIDATED_LOOKUP.values():
    if isinstance(item, dict):
        dt = item.get("data_type")
        dc = item.get("data_category")
        if dt and dc and dt in REGISTRY.get("data_types", {}):
            cat_dict = REGISTRY["data_types"][dt].get("categories", {})
            if dc in cat_dict:
                bp = cat_dict[dc].get("boilerplate")
                rules = cat_dict[dc].get("extraction_rules")
                key = f"{dt} -> {dc} [Template: {bp}]"
                if key not in SIMILAR_EXAMPLES_CATALOG:
                    SIMILAR_EXAMPLES_CATALOG[key] = {
                        "data_type": dt,
                        "data_category": dc,
                        "boilerplate": bp,
                        "extraction_rules": rules
                    }


def load_boilerplate_template(boilerplate_relative_path: str) -> Optional[Dict[str, Any]]:
    """Maps and loads the correct JSON template from configs_1/configs/boilerplates/"""
    if not BOILERPLATES_DIR:
        return None
    clean_relative_path = boilerplate_relative_path.lower().strip()
    full_path = os.path.join(BOILERPLATES_DIR, clean_relative_path)
    if not full_path.endswith(".json"):
        full_path += ".json"

    if not os.path.exists(full_path):
        filename = os.path.basename(full_path)
        search_pattern = os.path.join(BOILERPLATES_DIR, "**", filename)
        matches = glob.glob(search_pattern, recursive=True)
        if matches:
            full_path = matches[0]
        else:
            return None

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Error reading boilerplate %s: %s", full_path, exc)
        return None


# Pre-load available boilerplates for prompt context
AVAILABLE_BOILERPLATES = {}
for data_type, content in REGISTRY.get("data_types", {}).items():
    for cat_name, cat_details in content.get("categories", {}).items():
        b_path = cat_details.get("boilerplate")
        if b_path:
            b_json = load_boilerplate_template(b_path)
            if b_json:
                AVAILABLE_BOILERPLATES[b_path] = b_json


# ---------------------------------------------------------------------------
# Single File Processor: Combined Extraction + Rotation in 1 API Call
# ---------------------------------------------------------------------------
def process_single_file(file_path: Path, gtr_dir: Path, key_pool: GeminiKeyPool, force: bool) -> Tuple[bool, str]:
    """Process a single raw file from all_file/ and write gtr_<doc_id>.json directly into all_gtr/."""
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    base_name = file_path.stem
    output_json_path = gtr_dir / f"gtr_{base_name}.json"

    # Skip if output already exists and force is false
    if output_json_path.exists() and not force:
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # Verify if it already has both rotation_needed/degree AND some extraction data
            if isinstance(existing_data, dict) and "rotation_needed" in existing_data and "degree" in existing_data:
                return True, f"Skipped {file_path.name} (gtr_{base_name}.json already complete with rotation labels)"
        except Exception:
            pass  # If corrupted or incomplete, re-run

    consolidated_entry = CONSOLIDATED_LOOKUP.get(base_name)
    if not consolidated_entry:
        try:
            consolidated_entry = CONSOLIDATED_LOOKUP.get(int(base_name))
        except ValueError:
            pass

    consolidated_context = ""
    if consolidated_entry:
        consolidated_context = f"""
VERIFIED PRIOR KNOWLEDGE FROM CONSOLIDATED MAPPING (`final_consolidated.json` for file_name {base_name}):
The file '{file_path.name}' is present in `final_consolidated.json` with the following keys:
{json.dumps(consolidated_entry, indent=2)}

CRITICAL RE-VERIFICATION & OVERRIDE RULE (POTENTIAL WRONG IDENTIFICATION):
Note that the `data_type` and `data_category` in `final_consolidated.json` CAN SOMETIMES BE WRONGLY IDENTIFIED!
- Visually inspect the attached document (`{file_path.name}`) and verify its actual contents against the Classifier Registry.
- If `final_consolidated.json` is correct, use its `data_type` and `data_category` and select the matching boilerplate template.
- IF WRONG OR MISCLASSIFIED, OVERRIDE IT immediately by matching the visual evidence.
"""
    else:
        consolidated_context = """
(Note: No existing entry in `final_consolidated.json` for this file number. Classify and map by visually inspecting the document against the Classifier Registry.)
"""

    similar_catalog_section = f"""
CATALOG OF KNOWN SIMILAR DOCUMENT EXAMPLES (`final_consolidated.json` Reference):
{json.dumps(SIMILAR_EXAMPLES_CATALOG, indent=2)}
"""

    # Combined Prompt requesting BOTH rotation orientation AND structured data
    prompt = f"""
You are an advanced structured data pipeline agent. You must process the attached document strictly following these 4 steps:
{consolidated_context}
{similar_catalog_section}

Step 1: Classify this document based on the Classifier Registry below:
{json.dumps(REGISTRY, indent=2)}

Step 2: Choose the exact JSON template(s) from `available_boilerplates` options:
{json.dumps(AVAILABLE_BOILERPLATES, indent=2)}

Step 3: Orientation & Clockwise Rotation Analysis (MANDATORY ROOT FIELDS)
Examine the document's visual text orientation across all pages/images.
You MUST include the following rotation orientation fields directly at the ROOT / TOP LEVEL of your output JSON object:
  "rotation_needed": <true if degree != 0 else false>,
  "degree": <choose exactly one integer: 0, 90, 180, or 270 degrees clockwise required to make text upright>,
  "page_orientations": [
    {{
      "page_number": 1,
      "image_index": 1,
      "rotation_needed": <true/false>,
      "degree": <0, 90, 180, or 270>
    }}
  ]

Step 4: Populate the selected template(s) and combine with rotation orientation
- STRICT RULE: Do not change the JSON structure or key names of the chosen template(s). Replace only "null" values.
- If the document fits into exactly 1 category, output a single JSON object with the rotation fields at the top, followed by all template keys:
  {{
    "rotation_needed": false,
    "degree": 0,
    "page_orientations": [ {{"page_number": 1, "image_index": 1, "rotation_needed": false, "degree": 0}} ],
    "data_type": "<Classified Data Type>",
    "data_category": "<Classified Category>",
    ... (template fields)
  }}
- IF THE SINGLE FILE CONTAINS 2 OR MORE DIFFERENT DATA TYPES AND CATEGORIES, output:
  {{
    "rotation_needed": false,
    "degree": 0,
    "page_orientations": [ {{"page_number": 1, "image_index": 1, "rotation_needed": false, "degree": 0}} ],
    "multi_template_extraction": true,
    "document_id": "{base_name}",
    "extractions": [
      {{
        "data_type": "<First Data Type>",
        "data_category": "<First Category>",
        "boilerplate_used": "<Path 1>",
        "data": {{ ... }}
      }}
    ]
  }}
- Output ONLY valid JSON. Do not wrap in markdown code blocks or backticks.
"""

    max_retries = 6
    for attempt in range(max_retries):
        client, key_used, target_model = key_pool.get_client()
        uploaded_file = None
        try:
            uploaded_file = client.files.upload(file=str(file_path))
            response = client.models.generate_content(
                model=target_model,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            raw_text = (response.text or "").strip()
            # Clean possible markdown wrapping just in case
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            result_json = json.loads(raw_text.strip())

            # Ensure top-level rotation properties exist cleanly
            if isinstance(result_json, dict):
                if "degree" not in result_json:
                    result_json["degree"] = 0
                if "rotation_needed" not in result_json:
                    result_json["rotation_needed"] = bool(result_json["degree"] != 0)
                if "page_orientations" not in result_json:
                    result_json["page_orientations"] = [
                        {"page_number": 1, "image_index": 1, "rotation_needed": result_json["rotation_needed"], "degree": result_json["degree"]}
                    ]

            # Save to output_json_path
            gtr_dir.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as out_f:
                json.dump(result_json, out_f, ensure_ascii=False, indent=2)

            return True, f"Processed {file_path.name} -> gtr_{base_name}.json (degree={result_json.get('degree', 0)})"

        except Exception as exc:
            err_str = str(exc)
            if "404" in err_str and ("not available" in err_str or "NOT_FOUND" in err_str):
                fallback_model = "gemini-2.0-flash" if target_model != "gemini-2.0-flash" else "gemini-1.5-flash"
                key_pool.switch_model_for_key(key_used, fallback_model)
                continue
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                delay_match = re.search(r"Please retry in ([0-9.]+)s", err_str)
                if not delay_match:
                    delay_match = re.search(r"retryDelay': '([0-9.]+)s'", err_str)
                delay = float(delay_match.group(1)) + 2.0 if delay_match else 15.0
                key_pool.report_rate_limit(key_used, retry_delay=delay)
                if attempt == max_retries - 1:
                    return False, f"Failed processing {file_path.name} after quota retries: {exc}"
            else:
                if attempt == max_retries - 1:
                    return False, f"Error processing {file_path.name}: {exc}"
        finally:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

    return False, f"Exceeded max retries on {file_path.name}"


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Process document files from all_file/ with combined extraction & rotation logic.")
    parser.add_argument("files", nargs="*", help="Specific file IDs (e.g. 1000 or 1000.jpg) to process")
    parser.add_argument("--all", action="store_true", help="Process ALL files inside all_file/")
    parser.add_argument("--file-dir", type=str, default="./all_file", help="Source directory containing raw documents/images")
    parser.add_argument("--gtr-dir", type=str, default="./all_gtr", help="Target directory where gtr_*.json files will be written")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Default Gemini Vision model to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads when processing multiple files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force re-processing of files that already exist in all_gtr/")
    parser.add_argument("--api-keys", nargs="+", default=[], help="Additional Gemini API keys for pooling")
    args = parser.parse_args()

    file_dir = Path(args.file_dir).resolve()
    gtr_dir = Path(args.gtr_dir).resolve()

    if not file_dir.exists():
        logger.error("Source directory does not exist: %s", file_dir)
        sys.exit(1)
    gtr_dir.mkdir(parents=True, exist_ok=True)

    # Gather API keys
    pool_keys = list(args.api_keys)
    if os.environ.get("GEMINI_API_KEYS"):
        pool_keys.extend(os.environ["GEMINI_API_KEYS"].split(","))
    if os.environ.get("GEMINI_API_KEY"):
        pool_keys.append(os.environ["GEMINI_API_KEY"])

    env_paths = [Path(__file__).parent / ".env", Path.cwd() / ".env"]
    for p in env_paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    pool_keys.append(line.split("=", 1)[1].strip().strip('"\''))

    if not pool_keys:
        logger.warning("No API keys found. Please set GEMINI_API_KEY or pass --api-keys.")

    key_pool = GeminiKeyPool(pool_keys, default_model=args.model)

    # Resolve target files to process
    files_to_process: List[Path] = []
    if args.files:
        for item in args.files:
            # Check if user passed ID like '1000' or filename '1000.jpg'
            cand = file_dir / item
            if cand.exists():
                files_to_process.append(cand)
            else:
                # Try finding matching extension in file_dir
                matches = list(file_dir.glob(f"{item}.*"))
                if matches:
                    files_to_process.append(matches[0])
                else:
                    logger.warning("Could not find file in %s matching '%s'", file_dir, item)
    elif args.all or len(sys.argv) == 1:
        # Default mode if no args given: process all files in all_file/
        exts = ["*.pdf", "*.jpg", "*.jpeg", "*.png", "*.PDF", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in exts:
            files_to_process.extend(file_dir.glob(ext))
        files_to_process = sorted(list(set(files_to_process)), key=lambda p: p.stem)
    else:
        parser.print_help()
        sys.exit(0)

    if args.limit > 0:
        files_to_process = files_to_process[:args.limit]

    logger.info("Starting processing on %d files from %s -> %s (%d workers)...", len(files_to_process), file_dir, gtr_dir, args.workers)

    success_count = 0
    failure_count = 0
    skipped_count = 0

    if args.workers > 1 and len(files_to_process) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_single_file, fp, gtr_dir, key_pool, args.force): fp
                for fp in files_to_process
            }
            for i, future in enumerate(as_completed(future_to_file), 1):
                fp = future_to_file[future]
                try:
                    success, msg = future.result()
                    if success:
                        if "Skipped" in msg:
                            skipped_count += 1
                            logger.debug("[%d/%d] %s", i, len(files_to_process), msg)
                        else:
                            success_count += 1
                            logger.info("[%d/%d] %s", i, len(files_to_process), msg)
                    else:
                        failure_count += 1
                        logger.warning("[%d/%d] %s", i, len(files_to_process), msg)
                except Exception as exc:
                    failure_count += 1
                    logger.error("[%d/%d] Exception on %s: %s", i, len(files_to_process), fp.name, exc)
                if i % 20 == 0 or i == len(files_to_process):
                    logger.info("Progress: %d / %d files (Updated: %d, Skipped: %d, Failed: %d)", i, len(files_to_process), success_count, skipped_count, failure_count)
    else:
        for i, fp in enumerate(files_to_process, 1):
            success, msg = process_single_file(fp, gtr_dir, key_pool, args.force)
            if success:
                if "Skipped" in msg:
                    skipped_count += 1
                    logger.debug("[%d/%d] %s", i, len(files_to_process), msg)
                else:
                    success_count += 1
                    logger.info("[%d/%d] %s", i, len(files_to_process), msg)
            else:
                failure_count += 1
                logger.warning("[%d/%d] %s", i, len(files_to_process), msg)
            if i % 20 == 0 or i == len(files_to_process):
                logger.info("Progress: %d / %d files (Updated: %d, Skipped: %d, Failed: %d)", i, len(files_to_process), success_count, skipped_count, failure_count)

    logger.info("Batch execution finished! Updated: %d  |  Skipped: %d  |  Failed: %d", success_count, skipped_count, failure_count)


if __name__ == "__main__":
    main()

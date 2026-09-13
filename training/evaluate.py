#!/usr/bin/env python3
"""
=============================================================================
  Industrial Document OCR — Model Evaluation Script (`evaluate.py`)
=============================================================================

Overview
--------
This script evaluates the fine-tuned Qwen2.5-VL-7B LoRA model on the
ground-truth dataset stored in `dataset/all_gtr/`.

For every document it:
  1. Runs inference (both extraction and orientation tasks).
  2. Compares each output field against the GTR JSON value.
  3. Computes the following metrics:

  Task A – Orientation Classification
    - Accuracy, Precision, Recall, F1 (macro & weighted)
    - Confusion matrix (4-class: 0/90/180/270)

  Task B – JSON Extraction (field-by-field)
    - Per-field Exact Match accuracy
    - Overall micro / macro / weighted Precision, Recall, F1
    - Top-10 hardest fields (most errors)
    - Per data_type/data_category breakdown

  4. Generates a rich standalone HTML report with Plotly charts saved to:
       saves/Qwen2.5-VL-7B/lora/industrial_ocr_v1/eval_report/report.html

Usage
-----
  python evaluate.py

  # Optional overrides:
  LORA_ADAPTER_DIR=path/to/adapter  MAX_EVAL_SAMPLES=100  python evaluate.py
=============================================================================
"""

import gc
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT        = Path(__file__).resolve().parent
DATASET_ROOT     = REPO_ROOT / "dataset"
ALL_FILE_DIR     = DATASET_ROOT / "all_file"
ALL_GTR_DIR      = DATASET_ROOT / "all_gtr"
CONFIGS_DIR      = REPO_ROOT / "configs"
CONVERTED_DIR    = REPO_ROOT / ".cache" / "training" / "converted_images"

BASE_MODEL         = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-VL-7B-Instruct")
_model_basename    = BASE_MODEL.split("/")[-1]

DEFAULT_ADAPTER  = (
    REPO_ROOT / "saves" / _model_basename / "lora" / "industrial_ocr_v1"
)
REPORT_DIR = DEFAULT_ADAPTER / "eval_report"

# ---------------------------------------------------------------------------
# Config (overridable via env vars)
# ---------------------------------------------------------------------------
ADAPTER_DIR        = Path(os.environ.get("LORA_ADAPTER_DIR", str(DEFAULT_ADAPTER)))
MAX_EVAL_SAMPLES   = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))   # 0 = all
SEED               = int(os.environ.get("SEED", "42"))
MAX_NEW_TOKENS_EXT = int(os.environ.get("MAX_NEW_TOKENS_EXT", "512"))
MAX_NEW_TOKENS_ROT = int(os.environ.get("MAX_NEW_TOKENS_ROT", "40"))

ORIENTATION_PROMPT_TEXT = (
    "Examine this document image. "
    "What clockwise rotation in degrees is needed to make the text correctly upright and readable? "
    "Choose from: 0, 90, 180, or 270 degrees only. "
    'Reply with a compact JSON object in this exact format: '
    '{"degree_needed": <number>, "rotation_needed": <true|false>}'
    " - no markdown, no extra text."
)

EXTRACTION_PROMPT_TEXT = (
    "You are an industrial document OCR expert. "
    "Extract ALL structured fields from this document as a single, compact JSON object. "
    "Use the field names exactly as shown in the schema. "
    "The output JSON MUST always include 'data_type' and 'data_category' fields indicating which document classification was used. "
    "Do NOT include markdown code fences, explanatory text, or preamble. "
    "Output ONLY the JSON object."
)


def _build_messages(prompt_text: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


# ---------------------------------------------------------------------------
# Helper: resolve image path
# ---------------------------------------------------------------------------
def resolve_image(doc_id: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = ALL_FILE_DIR / f"{doc_id}{ext}"
        if p.exists():
            return p
    for ext in [".pdf", ".PDF"]:
        p = ALL_FILE_DIR / f"{doc_id}{ext}"
        if p.exists():
            try:
                from pdf2image import convert_from_path  # type: ignore
                out = CONVERTED_DIR / f"{doc_id}.jpg"
                if out.exists():
                    return out
                CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
                pages = convert_from_path(str(p), first_page=1, last_page=1, dpi=200)
                if pages:
                    pages[0].save(str(out), "JPEG", quality=90)
                    return out
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Helper: rotate PIL image clockwise
# ---------------------------------------------------------------------------
def rotate_cw(img: Any, degrees: int) -> Any:
    if degrees == 0:
        return img
    return img.rotate(360 - degrees, expand=True)


# ---------------------------------------------------------------------------
# Field comparison helpers
# ---------------------------------------------------------------------------
def _normalise(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, int):
        return str(v)
    return str(v).strip().lower()


def _flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
    result: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            result.update(_flatten(v, full_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            result.update(_flatten(item, f"{prefix}.{i}"))
    else:
        result[prefix] = _normalise(obj)
    return result


def compare_jsons(
    predicted: Any, ground_truth: Any
) -> Tuple[Dict[str, bool], int, int]:
    gtr_flat  = _flatten(ground_truth)
    pred_flat = _flatten(predicted) if predicted is not None else {}
    SKIP_KEYS = {"rotation_needed", "degree", "extraction_rules", "page_orientations"}
    field_matches: Dict[str, bool] = {}
    for key, gtr_val in gtr_flat.items():
        if any(seg in SKIP_KEYS for seg in key.split(".")):
            continue
        pred_val = pred_flat.get(key, "")
        field_matches[key] = (pred_val == gtr_val)
    n_correct = sum(field_matches.values())
    n_total   = len(field_matches)
    return field_matches, n_correct, n_total


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_orientation_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, Any]:
    classes = [0, 90, 180, 270]
    confusion = {a: {p: 0 for p in classes} for a in classes}
    for t, p in zip(y_true, y_pred):
        if t in confusion and p in confusion.get(t, {}):
            confusion[t][p] += 1
    per_class: Dict[int, Dict[str, float]] = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[a][c] for a in classes if a != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for t in y_true if t == c)}
    n = len(y_true)
    accuracy    = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    macro_f1    = sum(m["f1"]        for m in per_class.values()) / 4
    macro_p     = sum(m["precision"] for m in per_class.values()) / 4
    macro_r     = sum(m["recall"]    for m in per_class.values()) / 4
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_class.values()) / n if n else 0.0
    )
    return {
        "accuracy":         accuracy,
        "macro_precision":  macro_p,
        "macro_recall":     macro_r,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "per_class":        per_class,
        "confusion":        confusion,
        "n_samples":        n,
        "n_correct":        sum(1 for t, p in zip(y_true, y_pred) if t == p),
    }


def compute_extraction_metrics(
    all_field_results: List[Dict[str, bool]]
) -> Dict[str, Any]:
    field_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "total": 0})
    total_correct = 0
    total_fields  = 0
    for doc_result in all_field_results:
        for field, is_correct in doc_result.items():
            base_key = re.sub(r"\.\d+\.", ".N.", field)
            base_key = re.sub(r"\.\d+$", ".N", base_key)
            field_stats[base_key]["total"] += 1
            if is_correct:
                field_stats[base_key]["tp"] += 1
                total_correct += 1
            total_fields += 1
    micro_f1   = total_correct / total_fields if total_fields else 0.0
    field_f1s  = [v["tp"] / v["total"] for v in field_stats.values() if v["total"]]
    macro_f1   = sum(field_f1s) / len(field_f1s) if field_f1s else 0.0
    weighted_f1 = (
        sum((v["tp"] / v["total"]) * v["total"] for v in field_stats.values() if v["total"])
        / total_fields if total_fields else 0.0
    )
    per_field_accuracy = {
        k: {"accuracy": v["tp"] / v["total"], "correct": v["tp"], "total": v["total"]}
        for k, v in field_stats.items() if v["total"] > 0
    }
    return {
        "overall_field_accuracy": micro_f1,
        "micro_f1":     micro_f1,
        "macro_f1":     macro_f1,
        "weighted_f1":  weighted_f1,
        "total_correct": total_correct,
        "total_fields":  total_fields,
        "per_field":     per_field_accuracy,
    }


# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------
def _json_safe(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _json_safe(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_json_safe(i) for i in v]
    return str(v)


def generate_html_report(
    orient_metrics: Dict[str, Any],
    extract_metrics: Dict[str, Any],
    per_doc_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    classes   = [0, 90, 180, 270]
    conf      = orient_metrics["confusion"]
    z         = [[conf[a][p] for p in classes] for a in classes]
    z_text    = [[str(conf[a][p]) for p in classes] for a in classes]

    pf = extract_metrics["per_field"]
    sorted_fields = sorted(pf.items(), key=lambda x: x[1]["accuracy"])
    worst_20 = sorted_fields[:20]
    best_20  = sorted_fields[-20:][::-1]

    pc = orient_metrics["per_class"]
    rot_classes   = [f"{c}deg" for c in classes]
    rot_precision = [round(pc[c]["precision"] * 100, 1) for c in classes]
    rot_recall    = [round(pc[c]["recall"]    * 100, 1) for c in classes]
    rot_f1        = [round(pc[c]["f1"]        * 100, 1) for c in classes]

    doc_acc = [round(d["field_accuracy"] * 100, 1) for d in per_doc_results if "field_accuracy" in d]

    per_field_rows = ""
    for k, v in sorted(extract_metrics["per_field"].items(), key=lambda x: x[1]["accuracy"]):
        acc = v["accuracy"]
        badge_cls = "badge-ok" if acc >= 0.85 else ("badge-warn" if acc >= 0.5 else "badge-err")
        badge_lbl = "Good" if acc >= 0.85 else ("Fair" if acc >= 0.5 else "Poor")
        per_field_rows += (
            f'<tr><td>{k}</td><td>{v["correct"]}</td><td>{v["total"]}</td>'
            f'<td>{round(acc*100,1)}%</td>'
            f'<td><span class="{badge_cls}">{badge_lbl}</span></td></tr>\n'
        )

    per_doc_rows = ""
    for d in per_doc_results:
        rot_ok = d.get("rot_correct", False)
        per_doc_rows += (
            f'<tr><td>{d["id"]}</td><td>{d.get("data_category","?")}</td>'
            f'<td>{d.get("pred_degree","?")}</td><td>{d.get("gtr_degree","?")}</td>'
            f'<td><span class="{"badge-ok" if rot_ok else "badge-err"}">{"OK" if rot_ok else "Fail"}</span></td>'
            f'<td>{round(d.get("field_accuracy",0)*100,1)}%</td>'
            f'<td>{d.get("fields_correct",0)}/{d.get("fields_total",0)}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR Model Evaluation Report</title>
<script src="{PLOTLY_CDN}"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;padding:24px}}
h1{{font-size:2rem;font-weight:700;color:#7dd3fc;margin-bottom:4px}}
h2{{font-size:1.1rem;font-weight:600;color:#94a3b8;margin:28px 0 12px;letter-spacing:.05em;text-transform:uppercase}}
.subtitle{{color:#64748b;font-size:.9rem;margin-bottom:32px}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:32px}}
.metric-card{{background:#1e2535;border:1px solid #2d3748;border-radius:12px;padding:18px 14px;text-align:center}}
.metric-card .value{{font-size:1.9rem;font-weight:800;color:#7dd3fc}}
.metric-card .label{{font-size:.73rem;color:#64748b;margin-top:4px;text-transform:uppercase;letter-spacing:.05em}}
.chart-box{{background:#1e2535;border:1px solid #2d3748;border-radius:12px;padding:18px;margin-bottom:20px}}
.chart-title{{font-size:.95rem;font-weight:600;color:#cbd5e1;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:#1a2235;color:#94a3b8;text-align:left;padding:8px 10px;border-bottom:1px solid #2d3748}}
td{{padding:6px 10px;border-bottom:1px solid #1a2235;color:#cbd5e1}}
tr:hover td{{background:#1a2235}}
.badge-ok{{background:#166534;color:#86efac;border-radius:4px;padding:2px 7px;font-size:.72rem}}
.badge-warn{{background:#713f12;color:#fde68a;border-radius:4px;padding:2px 7px;font-size:.72rem}}
.badge-err{{background:#7f1d1d;color:#fca5a5;border-radius:4px;padding:2px 7px;font-size:.72rem}}
.divider{{height:1px;background:#2d3748;margin:28px 0}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:800px){{.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>OCR Model Evaluation Report</h1>
<p class="subtitle">{_model_basename} LoRA &middot; Adapter: {ADAPTER_DIR.name} &middot; Samples: {len(per_doc_results)}</p>

<h2>Overall Metrics</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="value">{round(orient_metrics['accuracy']*100,1)}%</div><div class="label">Orientation Accuracy</div></div>
  <div class="metric-card"><div class="value">{round(orient_metrics['macro_f1']*100,1)}%</div><div class="label">Orient Macro F1</div></div>
  <div class="metric-card"><div class="value">{round(orient_metrics['weighted_f1']*100,1)}%</div><div class="label">Orient Weighted F1</div></div>
  <div class="metric-card"><div class="value">{round(orient_metrics['macro_precision']*100,1)}%</div><div class="label">Orient Macro Prec</div></div>
  <div class="metric-card"><div class="value">{round(orient_metrics['macro_recall']*100,1)}%</div><div class="label">Orient Macro Recall</div></div>
  <div class="metric-card"><div class="value">{round(extract_metrics['overall_field_accuracy']*100,1)}%</div><div class="label">Field Exact Match</div></div>
  <div class="metric-card"><div class="value">{round(extract_metrics['macro_f1']*100,1)}%</div><div class="label">Extract Macro F1</div></div>
  <div class="metric-card"><div class="value">{round(extract_metrics['weighted_f1']*100,1)}%</div><div class="label">Extract Weighted F1</div></div>
  <div class="metric-card"><div class="value">{extract_metrics['total_correct']}</div><div class="label">Fields Correct</div></div>
  <div class="metric-card"><div class="value">{extract_metrics['total_fields']}</div><div class="label">Total Fields Eval</div></div>
</div>

<div class="divider"></div>
<h2>Task A &mdash; Orientation Classification</h2>
<div class="two-col">
  <div class="chart-box"><div class="chart-title">Confusion Matrix</div><div id="confMatrix"></div></div>
  <div class="chart-box"><div class="chart-title">Per-Class Precision / Recall / F1</div><div id="perClassBar"></div></div>
</div>

<div class="divider"></div>
<h2>Task B &mdash; JSON Extraction Field Accuracy</h2>
<div class="two-col">
  <div class="chart-box"><div class="chart-title">Worst 20 Fields</div><div id="worstFields"></div></div>
  <div class="chart-box"><div class="chart-title">Best 20 Fields</div><div id="bestFields"></div></div>
</div>
<div class="chart-box"><div class="chart-title">Per-Document Field Accuracy Distribution</div><div id="docAccDist"></div></div>

<div class="divider"></div>
<h2>All Fields &mdash; Detailed Accuracy</h2>
<div class="chart-box">
<table><thead><tr><th>Field</th><th>Correct</th><th>Total</th><th>Accuracy</th><th>Status</th></tr></thead>
<tbody>{per_field_rows}</tbody></table>
</div>

<div class="divider"></div>
<h2>Per-Document Results</h2>
<div class="chart-box">
<table><thead><tr><th>Doc ID</th><th>Category</th><th>Pred Deg</th><th>GTR Deg</th><th>Rotation</th><th>Field Acc</th><th>Correct/Total</th></tr></thead>
<tbody>{per_doc_rows}</tbody></table>
</div>

<script>
const DARK={{paper_bgcolor:'#1e2535',plot_bgcolor:'#1e2535',font:{{color:'#cbd5e1',family:'Segoe UI,system-ui,sans-serif'}}}};

Plotly.newPlot('confMatrix',[{{
  type:'heatmap',z:{json.dumps(z)},
  x:['0','90','180','270'],y:['0','90','180','270'],
  text:{json.dumps(z_text)},texttemplate:'%{{text}}',
  colorscale:'Blues',showscale:true
}}],{{...DARK,xaxis:{{title:'Predicted'}},yaxis:{{title:'Actual',autorange:'reversed'}},margin:{{t:10,b:60,l:60,r:20}},height:300}});

Plotly.newPlot('perClassBar',[
  {{name:'Precision',type:'bar',x:{json.dumps(rot_classes)},y:{json.dumps(rot_precision)},marker:{{color:'#38bdf8'}}}},
  {{name:'Recall',   type:'bar',x:{json.dumps(rot_classes)},y:{json.dumps(rot_recall)},   marker:{{color:'#818cf8'}}}},
  {{name:'F1',       type:'bar',x:{json.dumps(rot_classes)},y:{json.dumps(rot_f1)},       marker:{{color:'#34d399'}}}}
],{{...DARK,barmode:'group',yaxis:{{title:'% Score',range:[0,105]}},xaxis:{{title:'Class'}},legend:{{orientation:'h',y:-0.28}},margin:{{t:10,b:80,l:50,r:20}},height:300}});

Plotly.newPlot('worstFields',[{{
  type:'bar',orientation:'h',
  x:{json.dumps([round(v[1]["accuracy"]*100,1) for v in worst_20])},
  y:{json.dumps([v[0] for v in worst_20])},
  marker:{{color:'#f87171'}}
}}],{{...DARK,xaxis:{{title:'Accuracy %',range:[0,105]}},margin:{{t:10,b:40,l:230,r:20}},height:420}});

Plotly.newPlot('bestFields',[{{
  type:'bar',orientation:'h',
  x:{json.dumps([round(v[1]["accuracy"]*100,1) for v in best_20])},
  y:{json.dumps([v[0] for v in best_20])},
  marker:{{color:'#34d399'}}
}}],{{...DARK,xaxis:{{title:'Accuracy %',range:[0,105]}},margin:{{t:10,b:40,l:230,r:20}},height:420}});

Plotly.newPlot('docAccDist',[{{
  type:'histogram',x:{json.dumps(doc_acc)},nbinsx:20,
  marker:{{color:'#7dd3fc',opacity:0.8}},name:'Documents'
}}],{{...DARK,xaxis:{{title:'Field Accuracy %',range:[0,105]}},yaxis:{{title:'Documents'}},bargap:0.05,margin:{{t:10,b:60,l:60,r:20}},height:260}});
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved -> %s", output_path)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("OCR Model Evaluation")
    logger.info("  Adapter  : %s", ADAPTER_DIR)
    logger.info("  Base     : %s", BASE_MODEL)
    logger.info("=" * 60)

    # Step 1 - Load model
    logger.info("Loading model + LoRA adapter ...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # type: ignore
        from peft import PeftModel  # type: ignore
        import torch  # type: ignore

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model = model.merge_and_unload()
        model.eval()
        processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR))
        device = next(model.parameters()).device
        logger.info("Model loaded on device: %s", device)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        sys.exit(1)

    # Step 2 - Collect eval documents
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    gtr_files = sorted(ALL_GTR_DIR.glob("gtr_*.json"))
    logger.info("Found %d GTR files", len(gtr_files))

    eval_docs: List[Dict[str, Any]] = []
    for gf in gtr_files:
        m = re.match(r"gtr_(\d+)\.json", gf.name)
        if not m:
            continue
        doc_id = m.group(1)
        try:
            gtr_data = json.loads(gf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not gtr_data:
            continue
        img_path = resolve_image(doc_id)
        if img_path is None:
            continue
        eval_docs.append({"id": doc_id, "img_path": img_path, "gtr_data": gtr_data})

    if MAX_EVAL_SAMPLES and MAX_EVAL_SAMPLES < len(eval_docs):
        import random
        random.seed(SEED)
        eval_docs = random.sample(eval_docs, MAX_EVAL_SAMPLES)

    logger.info("Evaluating %d documents ...", len(eval_docs))

    # Step 3 - Inference + comparison
    from PIL import Image as _PIL  # type: ignore
    import torch  # type: ignore

    per_doc_results: List[Dict[str, Any]] = []
    orientation_y_true: List[int] = []
    orientation_y_pred: List[int] = []
    all_field_results: List[Dict[str, bool]] = []

    for i, doc in enumerate(eval_docs):
        doc_id   = doc["id"]
        img_path = doc["img_path"]
        gtr_data = doc["gtr_data"]

        logger.info("[%d/%d] Evaluating doc %s ...", i + 1, len(eval_docs), doc_id)

        result: Dict[str, Any] = {
            "id":            doc_id,
            "data_type":     gtr_data.get("data_type", ""),
            "data_category": gtr_data.get("data_category", ""),
            "gtr_degree":    int(gtr_data.get("degree", 0)),
        }

        try:
            img = _PIL.open(str(img_path)).convert("RGB")

            # --- Task A: Orientation ---
            gtr_degree = int(gtr_data.get("degree", 0))
            orientation_y_true.append(gtr_degree)

            rot_messages = _build_messages(ORIENTATION_PROMPT_TEXT)
            rot_text = processor.apply_chat_template(
                rot_messages, tokenize=False, add_generation_prompt=True
            )
            rot_inputs = processor(text=[rot_text], images=[img], return_tensors="pt", padding=True, truncation=False).to(device)
            with torch.no_grad():
                rot_out = model.generate(
                    **rot_inputs, max_new_tokens=MAX_NEW_TOKENS_ROT,
                    do_sample=False, repetition_penalty=1.1,
                )
            rot_raw = processor.tokenizer.decode(
                rot_out[0][rot_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            pred_degree = -1
            try:
                pred_json = json.loads(rot_raw)
                pred_degree = int(pred_json.get("degree_needed", -1))
            except Exception:
                nums = re.findall(r"\b(0|90|180|270)\b", rot_raw)
                if nums:
                    pred_degree = int(nums[0])
            if pred_degree not in (0, 90, 180, 270):
                pred_degree = 0

            orientation_y_pred.append(pred_degree)
            result["pred_degree"] = pred_degree
            result["rot_correct"] = (pred_degree == gtr_degree)

            img_for_ext = rotate_cw(img, pred_degree) if pred_degree != 0 else img

            # --- Task B: Extraction ---
            ext_messages = _build_messages(EXTRACTION_PROMPT_TEXT)
            ext_text = processor.apply_chat_template(
                ext_messages, tokenize=False, add_generation_prompt=True
            )
            ext_inputs = processor(text=[ext_text], images=[img_for_ext], return_tensors="pt", padding=True, truncation=False).to(device)
            with torch.no_grad():
                ext_out = model.generate(
                    **ext_inputs, max_new_tokens=MAX_NEW_TOKENS_EXT,
                    do_sample=False, repetition_penalty=1.1,
                )
            ext_raw = processor.tokenizer.decode(
                ext_out[0][ext_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            pred_extraction = None
            try:
                pred_extraction = json.loads(ext_raw)
            except Exception:
                m = re.search(r"\{.*\}", ext_raw, re.DOTALL)
                if m:
                    try:
                        pred_extraction = json.loads(m.group())
                    except Exception:
                        pass

            field_matches, n_correct, n_total = compare_jsons(pred_extraction, gtr_data)
            all_field_results.append(field_matches)

            result["fields_correct"] = n_correct
            result["fields_total"]   = n_total
            result["field_accuracy"] = n_correct / n_total if n_total else 0.0

            logger.info(
                "  Rotation: GTR=%d Pred=%d [%s]  |  Extraction: %d/%d fields (%.1f%%)",
                gtr_degree, pred_degree,
                "OK" if result["rot_correct"] else "FAIL",
                n_correct, n_total, result["field_accuracy"] * 100,
            )

        except Exception as exc:
            logger.warning("  Error on doc %s: %s", doc_id, exc)
            result["error"] = str(exc)

        per_doc_results.append(result)
        if 'ext_inputs' in dir():
            del ext_inputs
        if 'rot_inputs' in dir():
            del rot_inputs
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()

    # Step 4 - Compute metrics
    logger.info("Computing metrics ...")
    orient_metrics  = compute_orientation_metrics(orientation_y_true, orientation_y_pred)
    extract_metrics = compute_extraction_metrics(all_field_results)

    logger.info("=" * 60)
    logger.info("ORIENTATION  accuracy=%.1f%%  macro_f1=%.1f%%  weighted_f1=%.1f%%",
                orient_metrics["accuracy"]*100, orient_metrics["macro_f1"]*100,
                orient_metrics["weighted_f1"]*100)
    for c in [0, 90, 180, 270]:
        m = orient_metrics["per_class"][c]
        logger.info("  %3ddeg  P=%.1f%%  R=%.1f%%  F1=%.1f%%  support=%d",
                    c, m["precision"]*100, m["recall"]*100, m["f1"]*100, m["support"])
    logger.info("EXTRACTION  field_acc=%.1f%%  macro_f1=%.1f%%  weighted_f1=%.1f%%",
                extract_metrics["overall_field_accuracy"]*100,
                extract_metrics["macro_f1"]*100, extract_metrics["weighted_f1"]*100)
    for fname, fstat in sorted(extract_metrics["per_field"].items(), key=lambda x: x[1]["accuracy"])[:10]:
        logger.info("  HARD  %-45s  %.1f%%", fname, fstat["accuracy"]*100)
    logger.info("=" * 60)

    # Step 5 - Save JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics_summary.json").write_text(
        json.dumps({"orientation": _json_safe(orient_metrics),
                    "extraction": _json_safe(extract_metrics),
                    "n_documents": len(per_doc_results)}, indent=2),
        encoding="utf-8"
    )
    (REPORT_DIR / "per_doc_results.json").write_text(
        json.dumps([_json_safe(d) for d in per_doc_results], indent=2),
        encoding="utf-8"
    )

    # Step 6 - Generate HTML report
    generate_html_report(
        orient_metrics=orient_metrics,
        extract_metrics=extract_metrics,
        per_doc_results=per_doc_results,
        output_path=REPORT_DIR / "report.html",
    )
    logger.info("Open report: %s", REPORT_DIR / "report.html")


if __name__ == "__main__":
    main()

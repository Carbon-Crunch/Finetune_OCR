"""
Industrial OCR Model Server
FastAPI + Qwen2.5-VL-7B with LoRA adapter
"""

import io
import json
import logging
import re
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_ADAPTER = (
    PROJECT_ROOT / "saves_v1"
    if (PROJECT_ROOT / "saves_v1").exists()
    else (
        Path("/teamspace/studios/this_studio/saves_v1")
        if Path("/teamspace/studios/this_studio/saves_v1").exists()
        else PROJECT_ROOT / "saves/Qwen2.5-VL-7B-Instruct/lora/industrial_ocr_v1"
    )
)
ADAPTER_PATH = Path(os.environ.get("ADAPTER_PATH", str(DEFAULT_ADAPTER)))
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    if model is None:
        load_model()
    yield


app = FastAPI(title="Industrial OCR", lifespan=lifespan)

model = None
processor = None



def load_model():
    global model, processor
    log.info("Loading model...")
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    model.eval()
    min_pixels = int(os.environ.get("MIN_PIXELS", 256 * 28 * 28))
    max_pixels = int(os.environ.get("MAX_PIXELS", 1280 * 28 * 28))
    processor = AutoProcessor.from_pretrained(
        str(ADAPTER_PATH),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    log.info(f"Model loaded on {DEVICE} with min_pixels={min_pixels}, max_pixels={max_pixels}")


def rotate_cw(img: Image.Image, degrees: int) -> Image.Image:
    if degrees == 90:
        return img.transpose(Image.Transpose.ROTATE_270)
    elif degrees == 180:
        return img.transpose(Image.Transpose.ROTATE_180)
    elif degrees == 270:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


def run_inference(img: Image.Image, prompt_text: str, max_tokens: int = 1024) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True, truncation=False).to(DEVICE)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
        res = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res


ORIENTATION_PROMPT = (
    "Examine this document image. "
    "What clockwise rotation in degrees is needed to make the text correctly upright and readable? "
    "Choose from: 0, 90, 180, or 270 degrees only. "
    'Reply with a compact JSON object: {"degree_needed": <number>, "rotation_needed": <true|false>}'
    " - no markdown, no extra text."
)

EXTRACTION_PROMPT = (
    "You are an industrial document OCR expert. "
    "Extract ALL structured fields from this document as a single, compact JSON object. "
    "Use the field names exactly as shown in the schema. "
    "The output JSON MUST always include 'data_type' and 'data_category' fields. "
    "Do NOT include markdown code fences, explanatory text, or preamble. "
    "Output ONLY the JSON object."
)


def extract_json(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    idx = s.find("{")
    if idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[idx:])
            return obj
        except Exception:
            pass
    return None


@app.post("/api/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Step 1: Detect orientation
    rot_raw = run_inference(img, ORIENTATION_PROMPT, max_tokens=40)
    pred_degree = 0
    pred_json = extract_json(rot_raw)
    if isinstance(pred_json, dict) and "degree_needed" in pred_json:
        try:
            pred_degree = int(pred_json["degree_needed"])
        except (ValueError, TypeError):
            pass
    else:
        nums = re.findall(r"\b(0|90|180|270)\b", rot_raw)
        if nums:
            pred_degree = int(nums[0])
    if pred_degree not in (0, 90, 180, 270):
        pred_degree = 0

    # Step 2: Rotate and extract
    img_corrected = rotate_cw(img, pred_degree) if pred_degree != 0 else img
    ext_raw = run_inference(img_corrected, EXTRACTION_PROMPT, max_tokens=2048)

    extraction = extract_json(ext_raw)

    return {
        "orientation": {"predicted_degree": pred_degree, "raw": rot_raw},
        "extraction": extraction if extraction else {"raw": ext_raw, "parse_error": True},
    }



@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Industrial OCR</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e1e4e8; min-height: 100vh; }
.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
h1 { font-size: 1.8rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #58a6ff, #a371f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { color: #8b949e; margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
@media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem; }
.upload-area { border: 2px dashed #30363d; border-radius: 8px; padding: 3rem 1.5rem; text-align: center; cursor: pointer; transition: all 0.2s; }
.upload-area:hover, .upload-area.dragover { border-color: #58a6ff; background: #1c2128; }
.upload-area input { display: none; }
.upload-area p { color: #8b949e; margin-top: 0.5rem; }
.preview { margin-top: 1rem; text-align: center; }
.preview img { max-width: 100%; max-height: 400px; border-radius: 8px; border: 1px solid #30363d; }
.btn { display: inline-block; padding: 0.75rem 1.5rem; background: #238636; color: #fff; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; margin-top: 1rem; transition: background 0.2s; }
.btn:hover { background: #2ea043; }
.btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
.result { margin-top: 1rem; }
.result-section { margin-bottom: 1rem; }
.result-section h3 { font-size: 0.85rem; text-transform: uppercase; color: #8b949e; margin-bottom: 0.5rem; }
.json-box { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; max-height: 500px; overflow-y: auto; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
.badge-ok { background: #1b4332; color: #6ee7b7; }
.spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #30363d; border-top-color: #58a6ff; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 0.5rem; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }
.status { margin-top: 1rem; color: #8b949e; }
</style>
</head>
<body>
<div class="container">
  <h1>Industrial OCR</h1>
  <p class="subtitle">Qwen2.5-VL-7B + LoRA &mdash; orientation detection &amp; field extraction</p>
  <div class="grid">
    <div class="panel">
      <div class="upload-area" id="dropzone">
        <svg width="48" height="48" fill="none" stroke="#58a6ff" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 16V4m0 0L8 8m4-4l4 4M4 17v2a1 1 0 001 1h14a1 1 0 001-1v-2"/></svg>
        <p>Drop a document image here or click to upload</p>
        <input type="file" id="fileInput" accept="image/*">
      </div>
      <div class="preview" id="preview"></div>
      <button class="btn" id="runBtn" disabled>Run OCR</button>
      <div class="status" id="status"></div>
    </div>
    <div class="panel">
      <div class="result" id="result">
        <p style="color:#484f58">Results will appear here after processing.</p>
      </div>
    </div>
  </div>
</div>
<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const runBtn = document.getElementById('runBtn');
const status = document.getElementById('status');
const result = document.getElementById('result');
let selectedFile = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="preview">`;
  runBtn.disabled = false;
  result.innerHTML = '<p style="color:#484f58">Ready. Click "Run OCR" to process.</p>';
}

runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Processing... (this may take 30-60s)';
  result.innerHTML = '';
  const fd = new FormData();
  fd.append('file', selectedFile);
  try {
    const res = await fetch('/api/ocr', { method: 'POST', body: fd });
    const data = await res.json();
    let html = '';
    html += '<div class="result-section"><h3>Orientation</h3>';
    html += `<span class="badge badge-ok">${data.orientation.predicted_degree}&deg; rotation needed</span>`;
    html += '</div>';
    html += '<div class="result-section"><h3>Extracted Fields</h3>';
    html += `<div class="json-box">${syntaxHighlight(JSON.stringify(data.extraction, null, 2))}</div>`;
    html += '</div>';
    result.innerHTML = html;
  } catch (e) {
    result.innerHTML = `<p style="color:#f85149">Error: ${e.message}</p>`;
  }
  status.innerHTML = '';
  runBtn.disabled = false;
});

function syntaxHighlight(json) {
  return json.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"([^"]+)":/g, '<span style="color:#79c0ff">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span style="color:#a5d6ff">"$1"</span>')
    .replace(/: (\\d+)/g, ': <span style="color:#d2a8ff">$1</span>')
    .replace(/: (true|false|null)/g, ': <span style="color:#7ee787">$1</span>');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)

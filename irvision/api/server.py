"""FastAPI backend for the web frontend.

Run locally:
    uvicorn irvision.api.server:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /api/health                  model status
    GET  /api/examples                built-in demo images (from demo/)
    POST /api/process/example/{id}    run the pipeline on a demo image
    POST /api/process                 run the pipeline on an uploaded image (+ optional reference)
    GET  /api/report                  test-set results written by the evaluation scripts

Environment variables
    IRVISION_ALLOWED_ORIGINS   comma-separated CORS origins (default: *)
    IRVISION_MAX_SIDE          largest accepted image side in px (default: 2048)
    IRVISION_MAX_UPLOAD_MB     largest accepted upload (default: 25)
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from irvision import __version__
from irvision.api.payload import build_payload, data_url_writer, gray_u8, load_report, rgb_u8
from irvision.inference.io import read_image
from irvision.inference.pipeline import IRVisionPipeline
from irvision.utils.config import PROJECT_ROOT, load_config

DEMO_DIR = PROJECT_ROOT / "demo"
MAX_SIDE = int(os.getenv("IRVISION_MAX_SIDE", "2048"))
MAX_UPLOAD_BYTES = int(os.getenv("IRVISION_MAX_UPLOAD_MB", "25")) * 1024 * 1024

EXAMPLE_INFO = {
    "hyderabad_farmland": ("Farmland", "Irrigated fields around Hyderabad: 92 % cropland"),
    "hyderabad_lakes": ("Lakes", "Lakes and farmland: water is preserved best"),
    "hyderabad_city": ("City centre", "Dense urban area: the known weak spot"),
}

app = FastAPI(title="IRVision API", version=__version__,
              description="Thermal-infrared satellite image enhancement, colorization and semantic validation.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("IRVISION_ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
_lock = threading.Lock()   # one inference at a time: models are shared and GPU memory is small


@lru_cache(maxsize=1)
def get_pipeline() -> IRVisionPipeline:
    return IRVisionPipeline(load_config())


def list_examples() -> dict[str, dict]:
    examples = {}
    for ir_path in sorted(DEMO_DIR.glob("*_ir.png")):
        example_id = ir_path.name[: -len("_ir.png")]
        ref_path = DEMO_DIR / f"{example_id}_truecolor.png"
        if ref_path.exists():
            title, description = EXAMPLE_INFO.get(example_id, (example_id.replace("_", " ").title(), ""))
            examples[example_id] = {"id": example_id, "title": title, "description": description,
                                    "ir": ir_path, "reference": ref_path}
    return examples


def _run(image, nodata, reference, super_resolution: bool, detection: bool, source: dict) -> dict:
    if image.shape[0] > MAX_SIDE or image.shape[1] > MAX_SIDE:
        raise HTTPException(422, f"Image is {image.shape[1]}x{image.shape[0]} px; the limit is {MAX_SIDE} px per side.")
    pipeline = get_pipeline()
    with _lock:
        try:
            result = pipeline.process_image(image, reference, nodata,
                                            super_resolution=super_resolution, detection=detection)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    ref_display = None
    if reference is not None and "psnr" in result.metrics:
        ref_display = IRVisionPipeline.prepare_reference(reference, (3, *result.normalized.shape), [])
    return build_payload(result, data_url_writer, ref_display, super_resolution, detection, source)


@app.get("/api/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {"status": "ok", "version": __version__, "device": pipeline.device,
            "colorizer": pipeline.colorizer_info.get("name"), "colorizer_fallback": pipeline.colorizer_info.get("fallback"),
            "segmenter": bool(pipeline.segmenter)}


@app.get("/api/examples")
def examples() -> list[dict]:
    out = []
    for ex in list_examples().values():
        ref, _ = read_image(ex["reference"])
        thumb = rgb_u8(np.moveaxis(ref.astype(np.float32) / 255.0, -1, 0))[::4, ::4]
        ir, _ = read_image(ex["ir"])
        ir_f = ir.astype(np.float32)
        lo, hi = np.percentile(ir_f[ir_f > 0], [2, 98]) if (ir_f > 0).any() else (0, 1)
        ir_thumb = gray_u8((ir_f - lo) / max(hi - lo, 1e-6))[::4, ::4]
        out.append({"id": ex["id"], "title": ex["title"], "description": ex["description"],
                    "thumbnail": data_url_writer("thumb", thumb), "ir_thumbnail": data_url_writer("ir", ir_thumb)})
    return out


@app.post("/api/process/example/{example_id}")
def process_example(example_id: str, super_resolution: bool = False, detection: bool = False) -> dict:
    ex = list_examples().get(example_id)
    if ex is None:
        raise HTTPException(404, f"Unknown example '{example_id}'")
    image, nodata = read_image(ex["ir"])
    reference, _ = read_image(ex["reference"])
    return _run(image, nodata, reference, super_resolution, detection,
                {"kind": "example", "id": example_id, "title": ex["title"]})


@app.post("/api/process")
async def process_upload(
    image: UploadFile = File(..., description="Single-band thermal image: GeoTIFF, PNG or JPEG"),
    reference: UploadFile | None = File(None, description="Optional true-colour reference image"),
    super_resolution: bool = Form(False),
    detection: bool = Form(False),
) -> dict:
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        ir, nodata = read_image(data, image.filename)
        ref = None
        if reference is not None and reference.filename:
            ref, _ = read_image(await reference.read(), reference.filename)
            if ref.dtype == np.uint16:
                ref = ref.astype(np.float32) / 65535.0
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _run(ir, nodata, ref, super_resolution, detection, {"kind": "upload", "title": image.filename})


@app.get("/api/report")
def report() -> dict:
    return load_report(load_config()["paths"]["results_dir"])

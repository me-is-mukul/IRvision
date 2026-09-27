"""Turn pipeline results into what the web frontend displays.

Images are rendered for display (IR as grayscale, RGB with the display gamma,
land-cover maps in their palette, detections drawn on the colorized image) and
handed to an ``ImageWriter`` that returns a reference the browser can load:
a ``data:`` URL for the live API, or a file URL for the static demo export.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from irvision.inference.pipeline import PipelineResult
from irvision.semantic.landcover import CLASSES, PALETTE, colorize_labels
from irvision.utils.visualization import draw_detections

MAX_DISPLAY_SIDE = 1024

# pipeline stages in execution order, with display labels and a one-line description
STAGES = [
    ("validation", "Validation", "Single band, size limits, no-data mask"),
    ("normalization", "Normalization", "Robust 2–98 % percentile stretch"),
    ("enhancement", "Enhancement", "CLAHE local contrast (16-bit)"),
    ("super_resolution", "Super-resolution", "EDSR ×2 upscaling (optional)"),
    ("colorization", "Colorization", "U-Net: thermal → RGB"),
    ("segmentation", "Land-cover map", "DeepLabV3 on the colorized image"),
    ("detection", "Object detection", "YOLOv8-OBB, aerial classes (optional)"),
    ("metrics", "Quality metrics", "PSNR and SSIM vs. true colour"),
    ("semantic_validation", "Semantic check", "Land cover: colorized vs. true"),
    ("detection_validation", "Detection check", "Objects: colorized vs. true"),
]

ImageWriter = Callable[[str, np.ndarray], str]   # (name, HxW or HxWx3 uint8) -> URL


LOSSLESS = {"semantic", "reference_semantic", "thumb_labels"}   # label maps: exact colours


def encode_image(name: str, image: np.ndarray) -> tuple[bytes, str]:
    """Photo-like images as WebP (small), label maps as PNG (exact). Returns ``(bytes, extension)``."""
    img = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ext, params = (".png", []) if name in LOSSLESS else (".webp", [cv2.IMWRITE_WEBP_QUALITY, 88])
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"{ext} encoding failed")
    return buf.tobytes(), ext


def data_url_writer(name: str, image: np.ndarray) -> str:
    """Encode an image as a ``data:`` URL (used by the live API)."""
    data, ext = encode_image(name, image)
    return f"data:image/{ext[1:]};base64," + base64.b64encode(data).decode("ascii")


def file_writer(directory: Path, url_prefix: str) -> ImageWriter:
    """Write image files to ``directory`` and return their URLs (used by the static export)."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, image: np.ndarray) -> str:
        data, ext = encode_image(name, image)
        (directory / f"{name}{ext}").write_bytes(data)
        return f"{url_prefix}/{name}{ext}"

    return write


def _fit(image: np.ndarray, nearest: bool = False, max_side: int = MAX_DISPLAY_SIDE) -> np.ndarray:
    """Downscale for display if larger than ``max_side``; ``nearest`` keeps label colours exact."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interp)


def gray_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(np.nan_to_num(image), 0, 1) * 255).round().astype(np.uint8)


# Fixed "natural colour" display rendering, identical for predictions and ground truth.
# Per-channel stretch from the 1st..99th percentiles of all true-colour training data
# (reflectance/0.3 scale), then gamma and a mild saturation boost. Display only: every
# metric is computed on the unmodified data.
DISPLAY_LO = np.array([0.0, 0.03, 0.0], np.float32)
DISPLAY_HI = np.array([0.65, 0.52, 0.36], np.float32)
DISPLAY_SAT = 1.25


def rgb_u8(rgb_chw: np.ndarray, gamma: float = 1.3) -> np.ndarray:
    img = np.nan_to_num(np.moveaxis(rgb_chw, 0, -1)).astype(np.float32)
    img = np.clip((img - DISPLAY_LO) / (DISPLAY_HI - DISPLAY_LO), 0, 1) ** (1.0 / gamma)
    lum = img.mean(axis=-1, keepdims=True)
    img = np.clip(lum + DISPLAY_SAT * (img - lum), 0, 1)
    return (img * 255).round().astype(np.uint8)


def stage_list(result: PipelineResult, super_resolution: bool, detection: bool) -> list[dict]:
    """Every pipeline stage in order with its measured time; skipped stages are marked."""
    times = result.metrics["time_ms"]
    out = []
    for key, label, description in STAGES:
        if key in times:
            status = "done"
        elif (key == "super_resolution" and not super_resolution) or (key == "detection" and not detection):
            status = "off"
        else:
            status = "skipped"
        out.append({"key": key, "label": label, "description": description, "status": status,
                    "ms": round(times.get(key, 0.0), 1)})
    return out


def build_payload(result: PipelineResult, write: ImageWriter, reference_rgb: np.ndarray | None = None,
                  super_resolution: bool = False, detection: bool = False, source: dict | None = None) -> dict:
    """JSON-ready description of one pipeline run."""
    m = result.metrics
    colorized = rgb_u8(result.colorized)
    images = {
        "input": write("input", _fit(gray_u8(result.normalized))),
        "enhanced": write("enhanced", _fit(gray_u8(result.enhanced))),
        "colorized": write("colorized", _fit(colorized)),
    }
    if result.detections:
        drawn = draw_detections(colorized.astype(np.float32) / 255.0, result.detections)
        images["detections"] = write("detections", _fit((drawn * 255).round().astype(np.uint8)))
    if result.super_resolved is not None:
        images["super_resolved"] = write("super_resolved", _fit(gray_u8(result.super_resolved)))
    if reference_rgb is not None and "psnr" in m:
        ref = np.asarray(reference_rgb, dtype=np.float32)
        if ref.ndim == 3 and ref.shape[0] == 3:
            images["reference"] = write("reference", _fit(rgb_u8(np.nan_to_num(ref))))
    if result.semantic_map is not None:
        images["semantic"] = write("semantic", _fit(colorize_labels(result.semantic_map), nearest=True))
    if result.reference_semantic_map is not None:
        images["reference_semantic"] = write("reference_semantic", _fit(colorize_labels(result.reference_semantic_map), nearest=True))

    metrics: dict = {"total_ms": round(m["time_ms"]["total"], 1)}
    if "psnr" in m:
        metrics["psnr"] = round(float(m["psnr"]), 2)
        metrics["ssim"] = round(float(m["ssim"]), 4)
    if "semantic" in m:
        sem = m["semantic"]
        metrics["semantic"] = {"agreement": round(sem["pixel_agreement"], 4), "miou": round(sem["miou"], 4),
                               "mean_dice": round(sem["mean_dice"], 4), "iou": sem["iou"]}
    if "detection" in m:
        metrics["detection"] = m["detection"]

    return {
        "source": source or {},
        "shape": m["input_shape"],
        "output_shape": m["output_shape"],
        "scale": result.scale,
        "colorizer": result.colorizer,
        "valid_fraction": round(m["valid_fraction"], 4),
        "stages": stage_list(result, super_resolution, detection),
        "images": images,
        "metrics": metrics,
        "detections": result.detections,
        "reference_detections": result.reference_detections,
        "legend": [{"name": n, "color": "#%02x%02x%02x" % tuple(int(c) for c in col)} for n, col in zip(CLASSES, PALETTE)],
        "warnings": result.warnings,
    }


def load_report(results_dir: Path) -> dict:
    """Condensed test-set results for the web report (only what the evaluation scripts wrote)."""
    report: dict = {}

    def read(name: str) -> dict | None:
        path = results_dir / f"{name}_metrics.json"
        return json.loads(path.read_text()) if path.exists() else None

    names = {"mean_color": "Average colour", "lut": "Lookup table", "unet": "IRVision U-Net"}
    if model := read("model"):
        report["colorization"] = {
            "patches": model["num_patches"],
            "rows": [{"method": names.get(r["method"], r["method"]), "all": [r["PSNR (dB)"], r["SSIM"]],
                      "holdout": next(([h["PSNR (dB)"], h["SSIM"]] for h in model["holdout_table"]
                                       if h["method"] == r["method"]), None)}
                     for r in model["table"]],
            "end_to_end": model.get("end_to_end_holdout", []),
        }
    if sem := read("semantic"):
        allr = sem["results"]["all"]
        report["semantic"] = {
            "ceiling": round(allr["true_vs_worldcover"]["miou"], 3),
            "rows": [{"method": names[m], "miou": round(allr[f"{m}_vs_true"]["miou"], 3),
                      "agreement": round(allr[f"{m}_vs_true"]["pixel_agreement"], 3)}
                     for m in ("unet", "lut", "mean_color") if f"{m}_vs_true" in allr],
            "per_class": allr["unet_vs_true"]["iou"] if "unet_vs_true" in allr else {},
        }
    if adv := read("advanced"):
        sr = adv["super_resolution"]["results"]["all"]
        det = adv["detection"]["results"]["all"]
        report["advanced"] = {
            "super_resolution": [{"method": "U-Net", **sr["unet"]}, {"method": "EDSR ×2 + U-Net", **sr["unet_sr"]}],
            "detection": {"true_rgb": adv["detection"]["classes_on_true_rgb"],
                          "unet_rgb": adv["detection"]["classes_on_unet_rgb"],
                          "matched": det["unet"]["tp"]},
        }
    return report

"""Evaluate the advanced optional stages on the test split: super-resolution and detection.

1. Super-resolution ablation. For every test patch:
       without SR:  U-Net(CLAHE IR)
       with SR:     U-Net(EDSR x2(CLAHE IR)), resized back to 30 m for scoring
   Scored with PSNR / SSIM and semantic consistency (segmenter on colorized vs true RGB).
2. Detection consistency. YOLOv8n-OBB (DOTA) on the true RGB and on each colorized RGB;
   detections on the true RGB are the reference (no object ground truth exists).

Output: outputs/results/advanced_metrics.json

Usage:
    python scripts/evaluate_advanced.py
    python scripts/evaluate_advanced.py --max-patches 50      # quick run
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from datetime import datetime

import cv2
import numpy as np

from irvision.detection.detector import AerialDetector
from irvision.detection.metrics import match_counts, summarize_counts
from irvision.evaluation.metrics import psnr, ssim
from irvision.models.baseline import LUTColorizer
from irvision.models.colorizer import UNetColorizer
from irvision.models.super_resolution import SuperResolver
from irvision.semantic.metrics import confusion_matrix, summarize
from irvision.semantic.segmenter import LandCoverSegmenter
from irvision.utils.config import get_device, load_config, resolve_path
from irvision.utils.log import get_logger

log = get_logger("evaluate_advanced")


def down(rgb: np.ndarray, shape) -> np.ndarray:
    out = cv2.resize(np.moveaxis(rgb, 0, -1), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.clip(np.moveaxis(out, -1, 0), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, opt, ev = cfg["paths"], cfg["optional"], cfg["evaluation"]
    device = get_device(cfg)
    unet = UNetColorizer.from_checkpoint(resolve_path(cfg["inference"]["checkpoint"]), device)
    lut = LUTColorizer.load(paths["models_dir"] / "baseline_lut.npy")
    seg = LandCoverSegmenter.from_checkpoint(resolve_path(cfg["semantic"]["checkpoint"]), device)
    sr = SuperResolver.load(opt["super_resolution"]["model"], opt["super_resolution"]["scale"], device)
    dcfg = opt["detection"]
    det = AerialDetector.load(resolve_path(dcfg["model_path"]), device, confidence=dcfg["confidence"],
                              image_size=dcfg["image_size"], brightness_gamma=dcfg["brightness_gamma"],
                              tile_size=dcfg["tile_size"])
    held = [json.loads(p.read_text())["scene_id"] for p in sorted(paths["processed_dir"].glob("*_report.json"))
            if json.loads(p.read_text()).get("holdout")]

    files = sorted(paths["test_dir"].glob("*.npz"))[: args.max_patches]
    k = 5
    subsets = ("all", "holdout")
    sr_scores = {s: {m: {"psnr": [], "ssim": [], "cm": np.zeros((k, k), np.int64)} for m in ("unet", "unet_sr")}
                 for s in subsets}
    det_counts = {s: {m: collections.Counter() for m in ("unet", "lut")} for s in subsets}
    classes_true, classes_unet = collections.Counter(), collections.Counter()
    t_sr = []
    for f in files:
        with np.load(f) as d:
            ir, rgb, valid = d[ev["input_key"]][0].astype(np.float32), d["rgb"].astype(np.float32), d["valid"]
            ir_abs = d["ir_abs"][0].astype(np.float32) if unet.needs_abs else None
        in_held = any(f.name.startswith(h) for h in held)
        seg_true = seg.predict(rgb, valid)

        pred = unet.colorize(ir, ir_abs)
        t0 = time.perf_counter()
        ir_sr = sr.upscale(ir)
        t_sr.append((time.perf_counter() - t0) * 1000)
        abs_sr = None if ir_abs is None else cv2.resize(ir_abs, ir_sr.shape[::-1], interpolation=cv2.INTER_CUBIC)
        pred_sr = down(unet.colorize(ir_sr, abs_sr), ir.shape)

        dets_true, dets_unet, dets_lut = det.detect(rgb), det.detect(pred), det.detect(lut.colorize(ir))
        classes_true.update(d["class_name"] for d in dets_true)
        classes_unet.update(d["class_name"] for d in dets_unet)

        for s in subsets:
            if s == "holdout" and not in_held:
                continue
            for m, p in (("unet", pred), ("unet_sr", pred_sr)):
                sr_scores[s][m]["psnr"].append(psnr(p, rgb, valid))
                sr_scores[s][m]["ssim"].append(ssim(p, rgb, valid))
                sr_scores[s][m]["cm"] += confusion_matrix(seg.predict(p, valid), seg_true)
            det_counts[s]["unet"].update(match_counts(dets_unet, dets_true, dcfg["match_iou"]))
            det_counts[s]["lut"].update(match_counts(dets_lut, dets_true, dcfg["match_iou"]))

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "num_patches": len(files), "holdout_scenes": held, "device": device,
        "super_resolution": {
            "model": opt["super_resolution"]["model"], "scale": opt["super_resolution"]["scale"],
            "mean_ms_per_256px_patch": round(float(np.mean(t_sr)), 2),
            "results": {s: {m: {"psnr": round(float(np.mean(v["psnr"])), 3), "ssim": round(float(np.mean(v["ssim"])), 4),
                                "semantic_miou": round(summarize(v["cm"])["miou"], 4),
                                "semantic_agreement": round(summarize(v["cm"])["pixel_agreement"], 4),
                                "patches": len(v["psnr"])}
                            for m, v in sr_scores[s].items()} for s in subsets},
        },
        "detection": {
            "model": dcfg["model_path"], "confidence": dcfg["confidence"], "match_iou": dcfg["match_iou"],
            "reference": "detections on the true RGB (no object ground truth exists)",
            "classes_on_true_rgb": dict(classes_true), "classes_on_unet_rgb": dict(classes_unet),
            "results": {s: {m: summarize_counts({"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                                                 "n_pred": c["n_pred"], "n_ref": c["n_ref"]})
                            for m, c in det_counts[s].items()} for s in subsets},
        },
    }
    out = paths["results_dir"] / "advanced_metrics.json"
    out.write_text(json.dumps(report, indent=2))

    for s in subsets:
        log.info("== %s test patches", s)
        for m, r in report["super_resolution"]["results"][s].items():
            log.info("  %-8s PSNR %.2f SSIM %.4f semantic mIoU %.4f agreement %.1f%%", m, r["psnr"], r["ssim"],
                     r["semantic_miou"], 100 * r["semantic_agreement"])
        for m, r in report["detection"]["results"][s].items():
            log.info("  detection %-5s pred %d ref %d tp %d precision %s recall %s", m, r["n_pred"], r["n_ref"], r["tp"],
                     r["precision"], r["recall"])
    log.info("Detected classes on true RGB: %s | on U-Net RGB: %s", dict(classes_true), dict(classes_unet))
    log.info("Results: %s", out)


if __name__ == "__main__":
    main()

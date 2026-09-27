"""Phase 7: semantic validation of the colorization.

The land-cover segmenter (trained on *true* RGB) is run on the true RGB and on
each method's colorized RGB, for every test patch. Reported (IoU/mIoU/Dice/pixel
agreement, confusion matrices pooled over patches):

  * consistency  : segmentation(colorized) vs segmentation(true RGB)
                   -> "does the colorized image mean the same thing?"
  * vs WorldCover: segmentation(image) vs ESA WorldCover labels
                   -> the true-RGB row is the ceiling (segmenter quality)

Outputs:
    outputs/results/semantic_metrics.json
    outputs/results/semantic_comparison.png

Usage:
    python scripts/evaluate_semantic.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from irvision.models.baseline import LUTColorizer, MeanColorColorizer
from irvision.models.colorizer import UNetColorizer
from irvision.semantic.landcover import CLASSES, IGNORE, colorize_labels
from irvision.semantic.metrics import confusion_matrix, summarize
from irvision.semantic.segmenter import LandCoverSegmenter
from irvision.utils.config import get_device, load_config, resolve_path
from irvision.utils.log import get_logger
from irvision.utils.visualization import save_comparison_grid

log = get_logger("evaluate_semantic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", default=None, help="colorizer checkpoint (default: config inference.checkpoint)")
    parser.add_argument("--out-name", default="semantic", help="output prefix: <out-name>_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, ev = cfg["paths"], cfg["evaluation"]
    split = args.split or ev["split"]
    device = get_device(cfg)
    seg_ckpt, unet_ckpt = resolve_path(cfg["semantic"]["checkpoint"]), resolve_path(args.checkpoint or cfg["inference"]["checkpoint"])
    for p in (seg_ckpt, unet_ckpt, paths["models_dir"] / "baseline_lut.npy"):
        if not p.exists():
            raise SystemExit(f"Missing {p}. See README (train_segmenter.py / train.py / evaluate_baseline.py).")

    seg = LandCoverSegmenter.from_checkpoint(seg_ckpt, device)
    methods = {
        "unet": UNetColorizer.from_checkpoint(unet_ckpt, device),
        "lut": LUTColorizer.load(paths["models_dir"] / "baseline_lut.npy"),
        "mean_color": MeanColorColorizer(np.load(paths["models_dir"] / "baseline_mean_color.npy")),
    }
    held = [json.loads(p.read_text())["scene_id"] for p in sorted(paths["processed_dir"].glob("*_report.json"))
            if json.loads(p.read_text()).get("holdout")]

    k = len(CLASSES)
    subsets = ("all", "holdout")
    cms = {s: {key: np.zeros((k, k), np.int64) for key in
               ["true_vs_worldcover", *(f"{m}_vs_worldcover" for m in methods), *(f"{m}_vs_true" for m in methods)]}
           for s in subsets}
    files = sorted(paths[f"{split}_dir"].glob("*.npz"))
    figure_rows, picks = [], set(np.linspace(0, len(files) - 1, 4).astype(int).tolist())
    for i, f in enumerate(files):
        with np.load(f) as d:
            ir, rgb, valid = d[ev["input_key"]][0].astype(np.float32), d["rgb"].astype(np.float32), d["valid"]
            worldcover = np.where(valid, d["landcover"][0], IGNORE).astype(np.uint8)   # stored (1, S, S)
            ir_abs = d["ir_abs"][0].astype(np.float32)
        seg_true = seg.predict(rgb, valid)
        preds = {m: col.colorize(ir, ir_abs) if getattr(col, "needs_abs", False) else col.colorize(ir)
                 for m, col in methods.items()}
        segs = {m: seg.predict(p, valid) for m, p in preds.items()}
        in_holdout = any(f.name.startswith(h) for h in held)
        for s in subsets:
            if s == "holdout" and not in_holdout:
                continue
            c = cms[s]
            c["true_vs_worldcover"] += confusion_matrix(seg_true, worldcover)
            for m in methods:
                c[f"{m}_vs_worldcover"] += confusion_matrix(segs[m], worldcover)
                c[f"{m}_vs_true"] += confusion_matrix(segs[m], seg_true)
        if i in picks:
            to_chw = lambda lab: np.moveaxis(colorize_labels(lab), -1, 0).astype(np.float32) / 255.0  # noqa: E731
            figure_rows.append([rgb, to_chw(seg_true), preds["unet"], to_chw(segs["unet"]), to_chw(worldcover)])

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "split": split, "num_patches": len(files), "holdout_scenes": held,
        "segmenter": str(seg_ckpt), "segmenter_val": seg.meta.get("val", {}).get("miou"),
        "colorizer": str(unet_ckpt), "classes": CLASSES,
        "notes": "Confusion matrices pooled over patches; invalid (cloud/no-data) pixels excluded. "
                 "'X_vs_true' = segmentation of colorized image X vs segmentation of the true RGB.",
        "results": {s: {key: summarize(cm) for key, cm in cms[s].items()} for s in subsets},
    }
    out = paths["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.out_name}_metrics.json").write_text(json.dumps(report, indent=2))
    save_comparison_grid(out / f"{args.out_name}_comparison.png", figure_rows,
                         ["True RGB", "Seg(true RGB)", "U-Net RGB", "Seg(U-Net RGB)", "WorldCover labels"])

    for s in subsets:
        log.info("== %s test patches", s)
        for key, r in report["results"][s].items():
            log.info("  %-24s mIoU %.4f  Dice %.4f  agreement %.4f", key, r["miou"], r["mean_dice"], r["pixel_agreement"])
    log.info("Results: %s", out / f"{args.out_name}_metrics.json")


if __name__ == "__main__":
    main()

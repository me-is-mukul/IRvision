"""Phase 5/6: evaluate the trained U-Net against the baselines on the test split.

All methods are scored with the same function (evaluation.evaluate_colorizer)
on the same patches. Results are split into:
  * all test patches
  * held-out scene(s) only: whole scenes never seen in training (the honest number)
Also runs the complete ``process_image`` pipeline on each full held-out scene
(end-to-end check on an unseen image, PLAN.md §20).

Needs: a checkpoint (default: config inference.checkpoint) and the fitted
baselines from scripts/evaluate_baseline.py.

Outputs:
    outputs/results/model_metrics.json       tables shown in the app's "Model report"
    outputs/results/model_per_patch.csv
    outputs/results/model_comparison.png

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint outputs/models/unet_l1_ssim/best.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from irvision.evaluation.evaluate import evaluate_colorizer, iter_patches, time_colorizer
from irvision.inference.pipeline import IRVisionPipeline, load_example_scene
from irvision.models.baseline import LUTColorizer, MeanColorColorizer
from irvision.models.colorizer import UNetColorizer
from irvision.utils.config import get_device, load_config, resolve_path
from irvision.utils.log import get_logger
from irvision.utils.visualization import save_comparison_grid

log = get_logger("evaluate_model")


def holdout_scenes(processed_dir) -> list[str]:
    return [json.loads(p.read_text())["scene_id"] for p in sorted(processed_dir.glob("*_report.json"))
            if json.loads(p.read_text()).get("holdout")]


def table(per_patch: pd.DataFrame, timing: dict) -> list[dict]:
    rows = []
    for name, g in per_patch.groupby("method", sort=False):
        rows.append({"method": name, "PSNR (dB)": round(g["psnr"].mean(), 2), "PSNR std": round(g["psnr"].std(), 2),
                     "SSIM": round(g["ssim"].mean(), 4), "SSIM std": round(g["ssim"].std(), 4),
                     "ms / 256px": round(timing.get(name, float("nan")), 2), "patches": len(g)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None, help="default: config inference.checkpoint")
    parser.add_argument("--split", default=None)
    parser.add_argument("--out-name", default="model", help="output prefix: <out-name>_metrics.json etc.")
    parser.add_argument("--tta", action="store_true", help="test-time augmentation (8 flips/rotations)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, ev = cfg["paths"], cfg["evaluation"]
    split = args.split or ev["split"]
    eval_dir = paths[f"{split}_dir"]
    device = get_device(cfg)
    ckpt = resolve_path(args.checkpoint or cfg["inference"]["checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first: python scripts/train.py --run-name ...")
    for f in ("baseline_lut.npy", "baseline_mean_color.npy"):
        if not (paths["models_dir"] / f).exists():
            raise SystemExit(f"Missing {f}. Run scripts/evaluate_baseline.py first.")

    unet = UNetColorizer.from_checkpoint(ckpt, device, tile_size=cfg["inference"]["tile_size"],
                                         tile_overlap=cfg["inference"]["tile_overlap"], tta=args.tta)
    lut = LUTColorizer.load(paths["models_dir"] / "baseline_lut.npy")
    mean_color = MeanColorColorizer(np.load(paths["models_dir"] / "baseline_mean_color.npy"))
    colorizers = [mean_color, lut, unet]
    log.info("Checkpoint %s (epoch %s, val %s)", ckpt, unet.meta.get("epoch"), unet.meta.get("val_metrics"))

    per_patch = pd.concat([evaluate_colorizer(c, eval_dir, ev["input_key"]) for c in colorizers], ignore_index=True)
    held = holdout_scenes(paths["processed_dir"])
    per_patch["holdout"] = per_patch["scene_id"].isin(held)

    # UNetColorizer returns numpy, so GPU work is finished (synchronized) inside each timed call
    timing = {c.name: time_colorizer(c, repeats=ev["timing_repeats"])["mean_ms"] for c in colorizers}

    # end-to-end: full pipeline on each whole held-out scene
    pipeline = IRVisionPipeline(cfg, colorizer=unet, device=device)
    end_to_end = []
    for sid in held:
        ir, ref, _ = load_example_scene(paths["raw_dir"] / sid, cfg)
        res = pipeline.process_image(ir, reference_rgb=ref)
        end_to_end.append({"scene_id": sid, "shape": list(ir.shape), "psnr": round(res.metrics["psnr"], 2),
                           "ssim": round(res.metrics["ssim"], 4),
                           "time_ms": {k: round(v, 1) for k, v in res.metrics["time_ms"].items()},
                           "warnings": res.warnings})
        log.info("End-to-end %s %s: PSNR %.2f SSIM %.4f in %.0f ms", sid, ir.shape, res.metrics["psnr"],
                 res.metrics["ssim"], res.metrics["time_ms"]["total"])

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(ckpt), "checkpoint_epoch": unet.meta.get("epoch"),
        "checkpoint_val_metrics": unet.meta.get("val_metrics"),
        "split": split, "num_patches": int(per_patch["patch"].nunique()),
        "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "metric_notes": "PSNR/SSIM on [0,1] RGB, masked to valid pixels, mean over 256px patches. "
                        f"Timing: one 256x256 image on {device}. Baselines fitted on train only.",
        "table": table(per_patch, timing),
        "holdout_scenes": held,
        "holdout_table": table(per_patch[per_patch["holdout"]], timing) if held else [],
        "stripe_table": table(per_patch[~per_patch["holdout"]], timing),
        "end_to_end_holdout": end_to_end,
    }
    out = paths["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.out_name}_metrics.json").write_text(json.dumps(report, indent=2))
    per_patch.to_csv(out / f"{args.out_name}_per_patch.csv", index=False)

    samples = list(iter_patches(eval_dir, ev["input_key"]))
    picks = np.linspace(0, len(samples) - 1, ev["num_figure_samples"] + 1).astype(int)
    rows, titles = [], []
    for i in picks:
        name, ir, rgb, _ = samples[i]
        rows.append([ir, lut.colorize(ir), unet.colorize(ir), rgb])
        titles.append(name.split("_")[2] + " " + "_".join(name.split("_")[-2:]))
    save_comparison_grid(out / f"{args.out_name}_comparison.png", rows,
                         ["IR + CLAHE (input)", "LUT baseline", "U-Net", "Ground truth RGB"], titles)

    log.info("=" * 72)
    for label, key in (("ALL TEST PATCHES", "table"), ("HELD-OUT SCENE ONLY", "holdout_table"), ("TEST STRIPES ONLY", "stripe_table")):
        log.info("%s", label)
        for r in report[key]:
            log.info("  %-12s PSNR %6.2f  SSIM %.4f  %7.2f ms  (%d patches)", r["method"], r["PSNR (dB)"], r["SSIM"],
                     r["ms / 256px"], r["patches"])
    log.info("Results: %s", out / f"{args.out_name}_metrics.json")


if __name__ == "__main__":
    main()

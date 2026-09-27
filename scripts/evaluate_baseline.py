"""Phase 4: fit and evaluate the non-learned baselines (mean colour, gray, pseudo-colour, LUT).

Steps:
    1. fit the mean-colour and LUT baselines on the *train* split (never on test)
    2. compute masked PSNR / SSIM of every baseline on the evaluation split
    3. time CLAHE and each colorizer on a 256x256 image (CPU)
    4. write results + a visual comparison

Outputs:
    outputs/models/baseline_lut.npy               fitted lookup table
    outputs/models/baseline_mean_color.npy        fitted mean RGB
    outputs/results/baseline_metrics.json         summary (the numbers to quote)
    outputs/results/baseline_per_patch.csv        every patch, every method
    outputs/results/baseline_comparison.png       IR | baselines | ground truth

Usage:
    python scripts/evaluate_baseline.py
    python scripts/evaluate_baseline.py --split val
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd

from irvision.evaluation.evaluate import evaluate_colorizer, iter_patches, summarize, time_colorizer
from irvision.evaluation.metrics import time_function
from irvision.models.baseline import ColormapColorizer, GrayColorizer, LUTColorizer, MeanColorColorizer
from irvision.preprocessing.enhancement import apply_clahe
from irvision.utils.config import load_config
from irvision.utils.log import get_logger
from irvision.utils.visualization import save_comparison_grid

log = get_logger("evaluate_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--split", default=None, help="split to evaluate on (default: config evaluation.split)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, ev, bl, clahe_cfg = cfg["paths"], cfg["evaluation"], cfg["baseline"], cfg["preprocessing"]["clahe"]
    split = args.split or ev["split"]
    eval_dir = paths[f"{split}_dir"]

    log.info("Fitting mean-colour and LUT baselines on the train split ...")
    train = [(ir, rgb, v) for _, ir, rgb, v in iter_patches(paths["train_dir"], ev["input_key"])]
    mean_color = MeanColorColorizer.fit(train)
    lut = LUTColorizer.fit(train, bins=bl["lut_bins"])
    del train
    lut.save(paths["models_dir"] / "baseline_lut.npy")
    np.save(paths["models_dir"] / "baseline_mean_color.npy", mean_color.color)

    colorizers = [mean_color, GrayColorizer(), ColormapColorizer(bl["colormap"]), lut]

    log.info("Evaluating on the %s split (%s) ...", split, eval_dir)
    per_patch = pd.concat([evaluate_colorizer(c, eval_dir, ev["input_key"]) for c in colorizers], ignore_index=True)

    timing = {c.name: time_colorizer(c, repeats=ev["timing_repeats"]) for c in colorizers}
    sample_ir = next(iter_patches(eval_dir, "ir"))[1]
    timing["clahe"] = time_function(
        lambda: apply_clahe(sample_ir, clahe_cfg["clip_limit"], clahe_cfg["tile_grid_size"]), repeats=ev["timing_repeats"]
    )

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "num_patches": int(per_patch["patch"].nunique()),
        "metric_notes": "PSNR/SSIM on [0,1] RGB, masked to valid pixels; timing = CPU, one 256x256 image",
        "machine": f"{platform.processor()} / {platform.python_version()}",
        "methods": {},
    }
    for name, g in per_patch.groupby("method", sort=False):
        summary["methods"][name] = {
            "psnr_mean": g["psnr"].mean(), "psnr_std": g["psnr"].std(),
            "ssim_mean": g["ssim"].mean(), "ssim_std": g["ssim"].std(),
            "time_ms": timing[name]["mean_ms"],
            "per_scene": g.groupby("scene_id")[["psnr", "ssim"]].mean().round(4).to_dict(orient="index"),
        }
    summary["clahe_time_ms"] = timing["clahe"]["mean_ms"]

    results_dir = paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2))
    per_patch.to_csv(results_dir / "baseline_per_patch.csv", index=False)

    # visual comparison: evenly spaced samples across the split
    samples = list(iter_patches(eval_dir, ev["input_key"]))
    picks = np.linspace(0, len(samples) - 1, ev["num_figure_samples"]).astype(int)
    rows, row_titles = [], []
    for i in picks:
        name, ir, rgb, valid = samples[i]
        rows.append([ir, *(c.colorize(ir) for c in colorizers), rgb])
        row_titles.append(name.split("_")[2] + " " + "_".join(name.split("_")[-2:]))
    fig = save_comparison_grid(
        results_dir / "baseline_comparison.png", rows,
        ["IR + CLAHE (input)", *(c.name for c in colorizers), "Ground truth RGB"], row_titles,
    )

    log.info("=" * 70)
    log.info("%s split, %d patches (masked metrics)", split, summary["num_patches"])
    log.info("%-18s %8s %8s %10s", "method", "PSNR dB", "SSIM", "time ms")
    for name, m in summary["methods"].items():
        log.info("%-18s %8.2f %8.4f %10.3f", name, m["psnr_mean"], m["ssim_mean"], m["time_ms"])
    log.info("%-18s %8s %8s %10.3f", "clahe (256px)", "", "", summary["clahe_time_ms"])
    log.info("Results: %s", results_dir / "baseline_metrics.json")
    log.info("Figure:  %s", fig)


if __name__ == "__main__":
    main()

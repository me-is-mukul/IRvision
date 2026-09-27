"""Evaluate any colorizer on a folder of patches.

A *colorizer* is any object with ``name`` and ``colorize(ir (H,W)) -> (3,H,W)``
(see irvision/models/baseline.py). The U-Net will be evaluated with the same
function so all numbers are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from irvision.evaluation.metrics import psnr, ssim, time_function


def iter_patches(split_dir: str | Path, input_key: str = "ir_clahe"):
    """Yield ``(name, ir (H,W), rgb (3,H,W), valid (H,W))`` as float32 from patch files."""
    files = sorted(Path(split_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz patches in {split_dir}. Run scripts/prepare_dataset.py first.")
    for f in files:
        with np.load(f) as d:
            yield f.stem, d[input_key][0].astype(np.float32), d["rgb"].astype(np.float32), d["valid"]


def scene_of(patch_name: str) -> str:
    """``LC09_..._T1_r00000_c00128`` -> ``LC09_..._T1``."""
    return patch_name.rsplit("_", 2)[0]


def evaluate_colorizer(colorizer, split_dir: str | Path, input_key: str = "ir_clahe") -> pd.DataFrame:
    """Per-patch PSNR/SSIM (masked to valid pixels). One row per patch."""
    rows = []
    for name, ir, rgb, valid in iter_patches(split_dir, input_key):
        pred = colorizer.colorize(ir)
        rows.append({
            "method": colorizer.name,
            "patch": name,
            "scene_id": scene_of(name),
            "psnr": psnr(pred, rgb, valid),
            "ssim": ssim(pred, rgb, valid),
        })
    return pd.DataFrame(rows)


def time_colorizer(colorizer, size: int = 256, repeats: int = 50) -> dict[str, float]:
    """Inference time for one ``size`` x ``size`` image."""
    ir = np.random.default_rng(0).random((size, size), dtype=np.float32)
    return time_function(lambda: colorizer.colorize(ir), repeats=repeats)


def summarize(per_patch: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of PSNR/SSIM per method (all scenes pooled)."""
    return per_patch.groupby("method")[["psnr", "ssim"]].agg(["mean", "std"]).round(4)

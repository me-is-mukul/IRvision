"""Turn downloaded Landsat scenes into train/val/test patches.

For each scene folder in data/raw/:
    1. load + co-register all bands onto one grid, mask fill/cloud/shadow
    2. normalize IR (Band 10) and RGB (B4/B3/B2) to [0, 1]
    3. verify IR<->RGB alignment (phase correlation + positive control)
    4. CLAHE-enhance the IR
    5. cut 256x256 patches, drop cloudy ones, assign spatial split
    6. write patches, a manifest, a per-scene report and check figures

Outputs:
    data/{train,val,test}/<scene_id>_r<row>_c<col>.npz
    data/processed/<scene_id>.npz            full-scene arrays (float16)
    data/processed/<scene_id>_report.json    stats, alignment result, patch counts
    data/processed/manifest.csv              one row per patch (all scenes)
    outputs/dataset/<scene_id>_overview.png  look at this before training!
    outputs/dataset/<scene_id>_patches.png

Usage:
    python scripts/prepare_dataset.py                    # all scenes in data/raw
    python scripts/prepare_dataset.py --scene LC09_...   # one scene
    python scripts/prepare_dataset.py --force            # keep scenes that fail alignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from irvision.preprocessing.alignment import read_on_grid, verify_alignment
from irvision.preprocessing.enhancement import apply_clahe
from irvision.preprocessing.landsat import load_scene_from_config
from irvision.preprocessing.normalization import normalize
from irvision.preprocessing.patches import SPLITS, extract_patches, save_patch
from irvision.semantic.landcover import IGNORE
from irvision.utils.config import PROJECT_ROOT, load_config
from irvision.utils.log import get_logger
from irvision.utils.visualization import save_patch_grid, save_scene_overview

log = get_logger("prepare_dataset")


def load_landcover(scene_dir: Path, scene_id: str, grid) -> np.ndarray | None:
    """Land-cover labels on the scene grid if scripts/download_landcover.py was run, else None."""
    path = scene_dir / f"{scene_id}_LANDCOVER.TIF"
    if not path.exists():
        return None
    labels, _ = read_on_grid(path, grid, resampling="nearest", fill_value=IGNORE)
    return labels.astype(np.uint8)


def is_holdout(scene_dir: Path, scene_id: str, split_cfg: dict) -> bool:
    """A scene is held out if listed in `patches.split.test_scenes` or downloaded from a `holdout: true` AOI."""
    if scene_id in split_cfg.get("test_scenes", []):
        return True
    meta = scene_dir / "metadata.json"
    return meta.exists() and bool(json.loads(meta.read_text()).get("holdout", False))


def process_scene(scene_dir: Path, cfg: dict, force: bool) -> tuple[dict, list[dict]]:
    paths, pre, pcfg, acfg = cfg["paths"], cfg["preprocessing"], cfg["patches"], cfg["alignment"]
    scene = load_scene_from_config(scene_dir, cfg)
    sid = scene.scene_id

    ir, ir_stats = normalize(scene.ir_kelvin, pre["ir_normalization"], scene.valid)
    rgb, rgb_stats = normalize(scene.rgb_reflectance, pre["rgb_normalization"], scene.valid)
    ir_abs, _ = normalize(scene.ir_kelvin, pre["ir_abs_normalization"], scene.valid)
    log.info("  IR  %s: %.1f K .. %.1f K", ir_stats["method"], ir_stats["vmin"], ir_stats["vmax"])
    log.info("  RGB %s: %.3f .. %.3f reflectance", rgb_stats["method"], rgb_stats["vmin"], rgb_stats["vmax"])

    alignment = verify_alignment(
        ir, rgb, scene.valid,
        max_shift_px=acfg["max_shift_px"],
        min_response=acfg["min_response"],
        control_shift=tuple(acfg["positive_control_shift"]),
    )
    log.info(
        "  alignment: %s, shift (%.2f, %.2f) px, response %.3f, control %s -> (%.2f, %.2f)",
        alignment.status.upper(), *alignment.shift_px, alignment.response,
        alignment.control_expected, *alignment.control_measured,
    )

    clahe_cfg = pre["clahe"]
    ir_clahe = apply_clahe(ir, clahe_cfg["clip_limit"], clahe_cfg["tile_grid_size"], scene.valid)

    fig_dir = paths["outputs_dir"] / "dataset"
    overview = save_scene_overview(
        fig_dir / f"{sid}_overview.png", sid, ir, ir_clahe, rgb, scene.valid,
        alignment_text=f"alignment {alignment.status}: {alignment.message}",
    )
    log.info("  overview figure: %s", overview)

    report = {
        "scene_id": sid,
        "scene_dir": str(scene_dir),
        "shape": list(scene.valid.shape),
        "crs": str(scene.grid.crs),
        "resampled_bands": scene.resampled_bands,
        "valid_fraction": scene.valid_fraction,
        "ir_normalization": ir_stats,
        "rgb_normalization": rgb_stats,
        "alignment": alignment.to_dict(),
        "patches": {s: 0 for s in SPLITS},
    }

    if alignment.status == "fail" and not force:
        log.error("  %s FAILED alignment (%s): no patches written. Use --force to override.", sid, alignment.message)
        report["skipped"] = "alignment failed"
        return report, []
    if alignment.status == "inconclusive":
        log.warning("  alignment inconclusive (%s), check %s by eye", alignment.message, overview.name)

    np.savez_compressed(
        paths["processed_dir"] / f"{sid}.npz",
        ir=ir.astype(np.float16), ir_clahe=ir_clahe.astype(np.float16),
        rgb=rgb.astype(np.float16), valid=scene.valid,
    )

    extra = {"ir_abs": ir_abs}
    landcover = load_landcover(scene_dir, sid, scene.grid)
    if landcover is not None:
        extra["landcover"] = landcover   # labels for semantic validation (scripts/download_landcover.py)
    split_cfg = pcfg["split"]
    holdout = is_holdout(scene_dir, sid, split_cfg)
    report["holdout"] = holdout
    if holdout:
        log.info("  held-out scene: every patch goes to the test split")
    rows, samples = [], []
    for patch in extract_patches(
        ir, ir_clahe, rgb, scene.valid,
        size=pcfg["size"], stride=pcfg["stride"],
        max_invalid_fraction=pcfg["max_invalid_fraction"],
        fractions=split_cfg["fractions"], force_split="test" if holdout else None,
        extra=extra,
    ):
        name = f"{sid}_r{patch.row:05d}_c{patch.col:05d}"
        path = paths[f"{patch.split}_dir"] / f"{name}.npz"
        save_patch(path, patch)
        report["patches"][patch.split] += 1
        rows.append({
            "scene_id": sid, "patch": name, "split": patch.split, "row": patch.row, "col": patch.col,
            "invalid_fraction": round(patch.invalid_fraction, 4), "path": path.relative_to(PROJECT_ROOT).as_posix(),
        })
        if sum(t.startswith(patch.split) for *_, t in samples) < 3 and len(rows) % 5 == 1:  # a few per split
            samples.append((patch.ir_clahe, patch.rgb, f"{patch.split} r{patch.row} c{patch.col}"))

    log.info("  patches: %s", report["patches"])
    if samples:
        grid = save_patch_grid(fig_dir / f"{sid}_patches.png", *zip(*samples))
        log.info("  patch figure: %s", grid)
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--scene", action="append", help="scene folder name(s) in data/raw (default: all)")
    parser.add_argument("--force", action="store_true", help="write patches even if alignment fails")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    for key in ("processed_dir", "train_dir", "val_dir", "test_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(p for p in paths["raw_dir"].iterdir() if p.is_dir())
    if args.scene:
        scene_dirs = [p for p in scene_dirs if p.name in args.scene]
    if not scene_dirs:
        raise SystemExit(f"No scene folders in {paths['raw_dir']}. Run scripts/download_landsat.py first.")

    manifest_path = paths["processed_dir"] / "manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    reports = []
    for scene_dir in scene_dirs:
        sid = scene_dir.name
        # remove stale outputs of this scene so re-runs are idempotent
        for split in SPLITS:
            for old in paths[f"{split}_dir"].glob(f"{sid}_r*.npz"):
                old.unlink()
        if not manifest.empty:
            manifest = manifest[manifest["scene_id"] != sid]

        report, rows = process_scene(scene_dir, cfg, args.force)
        (paths["processed_dir"] / f"{report['scene_id']}_report.json").write_text(json.dumps(report, indent=2))
        manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
        reports.append(report)

    manifest.to_csv(manifest_path, index=False)
    log.info("=" * 70)
    for r in reports:
        log.info("%s  alignment=%-12s valid=%5.1f%%  patches=%s", r["scene_id"], r["alignment"]["status"],
                 100 * r["valid_fraction"], r["patches"])
    if not manifest.empty:
        log.info("Manifest totals: %s", manifest["split"].value_counts().to_dict())
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

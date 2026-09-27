"""Export ready-to-upload demo images from the held-out scene into demo/.

For three 512 px crops of the held-out city (chosen automatically from the
land-cover map: most water, most crops, most built-up) it writes:

    demo/<name>_ir.png          16-bit PNG, surface temperature in centi-Kelvin (upload as IR)
    demo/<name>_truecolor.png   8-bit RGB, same normalization as the model target (upload as reference)
    demo/<name>_ir.tif          (first crop only) float32 GeoTIFF in Kelvin, georeferenced

Then it runs every pair through the same code path as an app upload
(io.read_image -> process_image) and writes the measured results to
demo/results.json, so the demo numbers are real.

Usage:
    python scripts/export_demo.py
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import rasterio

from irvision.inference.io import read_image
from irvision.inference.pipeline import IRVisionPipeline, load_example_scene
from irvision.preprocessing.alignment import GridSpec
from irvision.semantic.landcover import CLASSES, IGNORE
from irvision.utils.config import PROJECT_ROOT, load_config
from irvision.utils.log import get_logger

log = get_logger("export_demo")

CROP = 512
# demo name -> land-cover class index whose share is maximized
TARGETS = {"hyderabad_lakes": CLASSES.index("water"),
           "hyderabad_farmland": CLASSES.index("low vegetation / crops"),
           "hyderabad_city": CLASSES.index("built-up")}


def pick_crops(landcover: np.ndarray, stride: int = 128) -> dict[str, tuple[int, int]]:
    """Top-left (row, col) of a non-overlapping crop per target, maximizing that class's share."""
    h, w = landcover.shape
    chosen: dict[str, tuple[int, int]] = {}
    for name, cls in TARGETS.items():
        best, best_score = None, -1.0
        for r in range(0, h - CROP + 1, stride):
            for c in range(0, w - CROP + 1, stride):
                if any(abs(r - r2) < CROP and abs(c - c2) < CROP for r2, c2 in chosen.values()):
                    continue
                win = landcover[r : r + CROP, c : c + CROP]
                known = win != IGNORE
                if known.mean() < 0.95:
                    continue
                score = float((win[known] == cls).mean())
                if cls == CLASSES.index("water"):
                    score = min(score, 0.35)  # lakes *with* land around them, not open water only
                if score > best_score:
                    best, best_score = (r, c), score
        chosen[name] = best
        log.info("%-20s crop at %s, %s share %.1f%%", name, best, CLASSES[cls], 100 * best_score)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = cfg["paths"]["raw_dir"]
    held = [d for d in sorted(raw.iterdir())
            if (d / "metadata.json").exists() and json.loads((d / "metadata.json").read_text()).get("holdout")]
    if not held:
        raise SystemExit("No held-out scene in data/raw (see config: aois with holdout: true).")
    scene_dir = held[0]
    sid = scene_dir.name
    out = PROJECT_ROOT / "demo"
    out.mkdir(exist_ok=True)

    ir_k, rgb, valid = load_example_scene(scene_dir, cfg)
    with rasterio.open(scene_dir / f"{sid}_LANDCOVER.TIF") as src:
        landcover = src.read(1)
    crops = pick_crops(landcover)
    grid = GridSpec.from_file(next(scene_dir.glob("*_SR_B4.TIF")))

    pipeline = IRVisionPipeline(cfg)
    rows = []
    for i, (name, (r, c)) in enumerate(crops.items()):
        win = np.s_[r : r + CROP, c : c + CROP]
        ir = np.nan_to_num(ir_k[win], nan=0.0)                         # 0 = no data (rare: clouds)
        ir_png = np.clip(np.round(ir * 100), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / f"{name}_ir.png"), ir_png)
        truecolor = (np.clip(np.nan_to_num(np.moveaxis(rgb[(slice(None), *win)], 0, -1)), 0, 1) * 255).round()
        cv2.imwrite(str(out / f"{name}_truecolor.png"), cv2.cvtColor(truecolor.astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i == 0:
            transform = rasterio.windows.transform(rasterio.windows.Window(c, r, CROP, CROP), grid.transform)
            with rasterio.open(out / f"{name}_ir.tif", "w", driver="GTiff", width=CROP, height=CROP, count=1,
                               dtype="float32", crs=grid.crs, transform=transform, nodata=np.nan) as dst:
                dst.write(ir_k[win].astype(np.float32), 1)

        # verify through the upload code path
        image, nodata = read_image((out / f"{name}_ir.png").read_bytes(), f"{name}_ir.png")
        reference, _ = read_image((out / f"{name}_truecolor.png").read_bytes(), f"{name}_truecolor.png")
        res = pipeline.process_image(image, reference, nodata)
        sem = res.metrics.get("semantic", {})
        rows.append({"name": name, "crop": f"rows {r}-{r + CROP}, cols {c}-{c + CROP}",
                     "psnr": res.metrics["psnr"], "ssim": res.metrics["ssim"],
                     "agreement": sem.get("pixel_agreement"), "miou": sem.get("miou"),
                     "time_ms": res.metrics["time_ms"]["total"], "warnings": res.warnings})
        log.info("%-20s PSNR %.2f SSIM %.3f semantic agreement %s", name, res.metrics["psnr"], res.metrics["ssim"],
                 f"{sem['pixel_agreement']:.1%}" if sem else "n/a")

    write_results(out, sid, rows, pipeline.colorizer_info.get("checkpoint", ""))
    log.info("Demo files in %s", out)


def write_results(out, sid: str, rows: list[dict], checkpoint: str) -> None:
    """Measured results for the demo files (quoted in README → Evaluation & results)."""
    (out / "results.json").write_text(json.dumps({"scene_id": sid, "checkpoint": checkpoint,
                                                  "crop_size": CROP, "demos": rows}, indent=2))


if __name__ == "__main__":
    main()

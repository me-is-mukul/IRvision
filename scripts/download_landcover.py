"""Fetch ESA WorldCover 2021 land cover for every scene, on the scene's own pixel grid.

These are the *labels* for semantic validation (PLAN.md Phase 7). WorldCover is
10 m in EPSG:4326; it is reprojected onto the scene's 30 m UTM grid with
*mode* resampling (majority class) and merged into 5 classes
(irvision/semantic/landcover.py).

Output: data/raw/<scene_id>/<scene_id>_LANDCOVER.TIF  (uint8 class index, 255 = no data)
Then re-run scripts/prepare_dataset.py so patches include a `landcover` array.

Usage:
    python scripts/download_landcover.py                 # all scenes in data/raw
    python scripts/download_landcover.py --scene LC09_...
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from irvision.preprocessing.alignment import GridSpec
from irvision.preprocessing.landsat import find_band_files
from irvision.semantic.landcover import CLASSES, IGNORE, remap_worldcover
from irvision.utils.config import load_config
from irvision.utils.log import get_logger

log = get_logger("download_landcover")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/esa-worldcover"
GDAL_HTTP_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
                 "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2"}


def worldcover_items(bbox_lonlat: tuple[float, float, float, float]) -> list[dict]:
    """WorldCover 2021 (v200) tiles intersecting the bbox."""
    body = {"collections": ["esa-worldcover"], "bbox": list(bbox_lonlat), "limit": 20,
            "query": {"esa_worldcover:product_version": {"eq": "2.0.0"}}}
    resp = requests.post(f"{STAC}/search", json=body, timeout=60)
    resp.raise_for_status()
    items = resp.json()["features"]
    if not items:
        raise RuntimeError(f"No WorldCover 2021 tiles for bbox {bbox_lonlat}")
    return items


def landcover_on_grid(grid: GridSpec, token: str) -> np.ndarray:
    """WorldCover codes reprojected (mode) onto ``grid``; 0 where no tile covers a pixel."""
    left, bottom = grid.transform * (0, grid.height)
    right, top = grid.transform * (grid.width, 0)
    bbox = transform_bounds(grid.crs, "EPSG:4326", left, bottom, right, top)
    codes = np.zeros(grid.shape, np.uint8)
    with rasterio.Env(**GDAL_HTTP_ENV):
        for item in worldcover_items(bbox):
            part = np.zeros(grid.shape, np.uint8)
            with rasterio.open(f"{item['assets']['map']['href']}?{token}") as src:
                reproject(source=rasterio.band(src, 1), destination=part, src_transform=src.transform,
                          src_crs=src.crs, src_nodata=0, dst_transform=grid.transform, dst_crs=grid.crs,
                          dst_nodata=0, resampling=Resampling.mode)
            codes = np.where(codes == 0, part, codes)   # tiles don't overlap; fill the gaps
            log.info("  tile %s", item["id"])
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--scene", action="append", help="scene folder name(s) (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    token = requests.get(TOKEN_URL, timeout=60).json()["token"]
    scene_dirs = sorted(p for p in cfg["paths"]["raw_dir"].iterdir() if p.is_dir())
    if args.scene:
        scene_dirs = [p for p in scene_dirs if p.name in args.scene]

    for scene_dir in scene_dirs:
        sid = scene_dir.name
        out = scene_dir / f"{sid}_LANDCOVER.TIF"
        if out.exists():
            log.info("%s: land cover exists, skipping", sid)
            continue
        ref = find_band_files(scene_dir, {"red": cfg["dataset"]["bands"]["red"]})["red"]
        grid = GridSpec.from_file(ref)
        log.info("%s: fetching WorldCover", sid)
        labels = remap_worldcover(landcover_on_grid(grid, token))
        with rasterio.open(out, "w", driver="GTiff", width=grid.width, height=grid.height, count=1, dtype="uint8",
                           crs=grid.crs, transform=grid.transform, nodata=IGNORE, compress="deflate") as dst:
            dst.write(labels, 1)
        counts = np.bincount(labels[labels != IGNORE], minlength=len(CLASSES)) / max((labels != IGNORE).sum(), 1)
        log.info("  class shares: %s", {c: round(float(v), 3) for c, v in zip(CLASSES, counts)})
    log.info("Done. Next: python scripts/prepare_dataset.py (adds `landcover` to patches)")


if __name__ == "__main__":
    main()

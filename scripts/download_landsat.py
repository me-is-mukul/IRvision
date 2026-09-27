"""Download Landsat 8/9 Collection-2 Level-2 scenes from Microsoft Planetary Computer.

No account is needed. Only a square crop around each area of interest (AOI)
is downloaded: the files are cloud-optimized GeoTIFFs, so rasterio reads just
that window over HTTP instead of the full ~1 GB scene.

Output (one folder per scene, USGS file naming):
    data/raw/<scene_id>/<scene_id>_SR_B4.TIF   (+ SR_B3, SR_B2, ST_B10, QA_PIXEL)
    data/raw/<scene_id>/metadata.json          (provenance: STAC properties, window)

Usage:
    python scripts/download_landsat.py                      # every AOI in config
    python scripts/download_landsat.py --aoi bengaluru      # a single AOI
    python scripts/download_landsat.py --crop-size 0        # full scenes (large!)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window, from_bounds

from irvision.utils.config import load_config
from irvision.utils.log import get_logger

log = get_logger("download_landsat")

# Planetary Computer asset key for each band name used in config.dataset.bands
PC_ASSETS = {"red": "red", "green": "green", "blue": "blue", "ir": "lwir11", "qa": "qa_pixel"}

GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",   # don't list the remote folder
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def search_scene(dl_cfg: dict, point: list[float], datetime: str) -> dict:
    """Return the least-cloudy STAC item covering ``point`` in the date range."""
    body = {
        "collections": [dl_cfg["collection"]],
        "intersects": {"type": "Point", "coordinates": point},
        "datetime": datetime,
        "query": {
            "eo:cloud_cover": {"lt": dl_cfg["max_cloud_cover"]},
            "platform": {"in": dl_cfg["platforms"]},
        },
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
        "limit": 1,
    }
    resp = requests.post(f"{dl_cfg['stac_url']}/search", json=body, timeout=60)
    resp.raise_for_status()
    items = resp.json()["features"]
    if not items:
        raise RuntimeError(f"No scene found at {point} for {datetime} with cloud < {dl_cfg['max_cloud_cover']}%")
    return items[0]


def get_sas_token(token_url: str) -> str:
    resp = requests.get(token_url, timeout=60)
    resp.raise_for_status()
    return resp.json()["token"]


def crop_window(
    src: rasterio.io.DatasetReader, point: list[float], size: int | None, max_fill: float = 0.01, factor: int = 16
) -> Window:
    """Square ``size`` px window as close as possible to ``point`` (lon/lat) with almost no fill.

    Landsat rasters are tilted footprints padded with fill (DN 0). If the point
    lies near the footprint edge, a window centred on it would be partly empty,
    so we search a ``factor``x downsampled copy (cheap: read from the COG
    overviews) for the nearest window whose fill fraction is <= ``max_fill``.
    Falls back to the point-centred window if none qualifies.
    """
    if not size:
        return Window(0, 0, src.width, src.height)
    size = min(size, src.width, src.height)
    xs, ys = transform_coords("EPSG:4326", src.crs, [point[0]], [point[1]])
    row, col = src.index(xs[0], ys[0])
    centred = (
        min(max(row - size // 2, 0), src.height - size),
        min(max(col - size // 2, 0), src.width - size),
    )

    small_h, small_w, s = src.height // factor, src.width // factor, size // factor
    fill = (src.read(1, out_shape=(small_h, small_w)) == 0).astype(np.float64)
    integral = np.pad(fill.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    fill_frac = (integral[s:, s:] - integral[:-s, s:] - integral[s:, :-s] + integral[:-s, :-s]) / (s * s)
    rows, cols = np.nonzero(fill_frac <= max_fill)  # candidate top-lefts (downsampled)
    if rows.size == 0:
        log.warning("  no fill-free %dpx window found; using the point-centred crop", size)
        return Window(centred[1], centred[0], size, size)

    target_r, target_c = centred[0] / factor, centred[1] / factor
    best = np.argmin((rows - target_r) ** 2 + (cols - target_c) ** 2)
    row_off = min(int(rows[best]) * factor, src.height - size)
    col_off = min(int(cols[best]) * factor, src.width - size)
    moved_km = math.hypot(row_off - centred[0], col_off - centred[1]) * abs(src.res[0]) / 1000
    if moved_km > 0:
        log.info("  crop moved %.1f km from the AOI point to avoid no-data pixels", moved_km)
    return Window(col_off, row_off, size, size)


def download_scene(item: dict, aoi: dict, crop_size: int | None, bands: dict, token: str, out_root: Path) -> Path:
    point = aoi["point"]
    scene_id = item["id"]
    out_dir = out_root / scene_id
    expected = [out_dir / f"{scene_id}_{suffix}.TIF" for suffix in bands.values()]
    if all(p.exists() for p in expected):
        log.info("%s already downloaded, skipping", scene_id)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(**GDAL_HTTP_ENV):
        ref_href = f"{item['assets'][PC_ASSETS['red']]['href']}?{token}"
        with rasterio.open(ref_href) as ref:
            window = crop_window(ref, point, crop_size)
            bounds = rasterio.windows.bounds(window, ref.transform)

        for name, suffix in bands.items():
            href = f"{item['assets'][PC_ASSETS[name]]['href']}?{token}"
            dst_path = out_dir / f"{scene_id}_{suffix}.TIF"
            with rasterio.open(href) as src:
                # same geographic bounds for every band, even if a band's grid differs
                win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                data = src.read(1, window=win)
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff", width=data.shape[1], height=data.shape[0],
                    transform=src.window_transform(win), compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256,
                )
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(data, 1)
            log.info("  %-5s -> %s (%dx%d)", name, dst_path.name, data.shape[1], data.shape[0])

    metadata = {
        "scene_id": scene_id,
        "source": "Microsoft Planetary Computer, collection landsat-c2-l2",
        "aoi": aoi["name"],
        "holdout": bool(aoi.get("holdout", False)),   # prepare_dataset.py sends holdout scenes to test
        "properties": item["properties"],
        "aoi_point_lonlat": point,
        "crop_size_px": crop_size,
        "window_bounds": list(bounds),
        "asset_hrefs": {name: item["assets"][PC_ASSETS[name]]["href"] for name in bands},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aoi", action="append", help="AOI name(s) from config (default: all)")
    parser.add_argument("--crop-size", type=int, default=None, help="crop size in px; 0 = full scene")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dl_cfg = cfg["dataset"]["download"]
    crop_size = dl_cfg["crop_size"] if args.crop_size is None else (args.crop_size or None)
    aois = [a for a in dl_cfg["aois"] if not args.aoi or a["name"] in args.aoi]
    if not aois:
        raise SystemExit(f"No AOI matches {args.aoi}; available: {[a['name'] for a in dl_cfg['aois']]}")

    token = get_sas_token(dl_cfg["token_url"])
    for aoi in aois:
        item = search_scene(dl_cfg, aoi["point"], aoi["datetime"])
        props = item["properties"]
        log.info("AOI %s -> %s (%s, cloud %.2f%%)", aoi["name"], item["id"], props["datetime"][:10], props["eo:cloud_cover"])
        download_scene(item, aoi, crop_size, cfg["dataset"]["bands"], token, cfg["paths"]["raw_dir"])
    log.info("Done. Next: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()

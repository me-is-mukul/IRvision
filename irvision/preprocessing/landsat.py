"""Load a Landsat 8/9 Collection-2 Level-2 scene into aligned numpy arrays.

A *scene directory* contains one GeoTIFF per band, named with the USGS
suffixes (``<scene_id>_SR_B4.TIF``, ``<scene_id>_ST_B10.TIF``, ...). This is the
layout produced by ``scripts/download_landsat.py`` and by USGS EarthExplorer.

Physical units (USGS C2 L2 scale factors):
    surface reflectance = DN * 2.75e-5 - 0.2
    surface temperature = DN * 0.00341802 + 149.0   [Kelvin]
DN == 0 is fill (no data) for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from irvision.preprocessing.alignment import GridSpec, read_on_grid
from irvision.preprocessing.masks import qa_invalid_mask
from irvision.utils.log import get_logger

log = get_logger(__name__)

SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
ST_SCALE, ST_OFFSET = 0.00341802, 149.0

DEFAULT_BANDS = {"red": "SR_B4", "green": "SR_B3", "blue": "SR_B2", "ir": "ST_B10", "qa": "QA_PIXEL"}
DEFAULT_MASK_FLAGS = ("fill", "dilated_cloud", "cloud", "cloud_shadow")


@dataclass
class LandsatScene:
    """One scene with every band on the same pixel grid."""

    scene_id: str
    ir_kelvin: np.ndarray        # (H, W) float32, NaN where invalid
    rgb_reflectance: np.ndarray  # (3, H, W) float32 in R, G, B order, NaN where invalid
    qa: np.ndarray               # (H, W) uint16 raw QA_PIXEL
    valid: np.ndarray            # (H, W) bool — usable pixels (no fill/cloud/shadow)
    grid: GridSpec
    resampled_bands: list[str] = field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


def find_band_files(scene_dir: str | Path, bands: dict[str, str] | None = None) -> dict[str, Path]:
    """Map band names (``red``, ``ir``, ...) to files in ``scene_dir`` by suffix."""
    scene_dir = Path(scene_dir)
    bands = bands or DEFAULT_BANDS
    tifs = [p for p in scene_dir.iterdir() if p.suffix.upper() in (".TIF", ".TIFF")]
    found = {}
    for name, suffix in bands.items():
        matches = [p for p in tifs if p.stem.upper().endswith(f"_{suffix.upper()}")]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '*_{suffix}.TIF' for band '{name}' in {scene_dir}, found {len(matches)}"
            )
        found[name] = matches[0]
    return found


def load_scene(
    scene_dir: str | Path,
    bands: dict[str, str] | None = None,
    reference_band: str = "red",
    resampling: str = "bilinear",
    mask_flags: tuple[str, ...] | list[str] = DEFAULT_MASK_FLAGS,
) -> LandsatScene:
    """Load, co-register and mask a scene.

    Every band is placed on the grid of ``reference_band`` (reprojected and
    resampled if its CRS/transform/shape differ). QA uses nearest-neighbour
    resampling so bit flags are never interpolated.
    """
    bands = bands or DEFAULT_BANDS
    files = find_band_files(scene_dir, bands)
    grid = GridSpec.from_file(files[reference_band])
    scene_id = files[reference_band].stem[: -len(bands[reference_band]) - 1]
    log.info("Loading scene %s (%dx%d, %s)", scene_id, grid.width, grid.height, grid.crs)

    raw, resampled = {}, []
    for name, path in files.items():
        is_qa = name == "qa"
        raw[name], was_resampled = read_on_grid(
            path, grid, resampling="nearest" if is_qa else resampling, fill_value=1 if is_qa else 0
        )
        if was_resampled:
            resampled.append(name)
            log.info("  band %-5s resampled onto the %s grid", name, reference_band)

    fill = np.zeros(grid.shape, dtype=bool)
    for name in ("red", "green", "blue", "ir"):
        fill |= raw[name] == 0
    valid = ~fill & ~qa_invalid_mask(raw["qa"], mask_flags)

    ir = raw["ir"].astype(np.float32) * ST_SCALE + ST_OFFSET
    rgb = np.stack([raw[b] for b in ("red", "green", "blue")]).astype(np.float32) * SR_SCALE + SR_OFFSET
    ir[~valid] = np.nan
    rgb[:, ~valid] = np.nan

    scene = LandsatScene(scene_id, ir, rgb, raw["qa"].astype(np.uint16), valid, grid, resampled)
    log.info("  valid pixels: %.1f%%", 100 * scene.valid_fraction)
    return scene


def load_scene_from_config(scene_dir: str | Path, cfg: dict) -> LandsatScene:
    """``load_scene`` with parameters taken from the ``dataset`` config section."""
    ds = cfg["dataset"]
    return load_scene(
        scene_dir,
        bands=ds["bands"],
        reference_band=ds["reference_band"],
        resampling=ds["resampling"],
        mask_flags=ds["mask_flags"],
    )

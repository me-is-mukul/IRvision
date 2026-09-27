"""The single high-level inference API (PLAN.md §12).

    from irvision.inference.pipeline import process_image
    result = process_image(ir_array)                       # any (H, W) IR image
    result = process_image(ir_array, reference_rgb=rgb)    # + PSNR/SSIM vs ground truth

Stages (each timed, in ``result.metrics["time_ms"]``):
    validation -> normalization -> CLAHE -> [super-resolution x2, EDSR] -> colorization (U-Net)
    -> [semantic segmentation, DeepLabV3] -> [object detection, YOLOv8-OBB] -> metrics

Semantic segmentation (land-cover map of the colorized image; with a reference
also of the true RGB, plus mIoU / Dice / agreement between the two maps) runs when
``optional.segmentation.enabled`` and a segmenter checkpoint exists.

Optional stages ([...]) are switched in ``config.optional``. They can never break
the core pipeline: if one is enabled but unavailable or failing, it is skipped
and a warning is added to ``result.warnings``.

The colorizer is the trained U-Net (``inference.checkpoint``). If the checkpoint
is missing, the pipeline falls back to the LUT baseline, then to pseudo-colour,
and says so in ``result.warnings``/``pipeline.colorizer_info``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from irvision.detection.metrics import compare_detections
from irvision.evaluation.metrics import psnr, ssim
from irvision.inference.io import to_single_channel
from irvision.models.baseline import ColormapColorizer, LUTColorizer
from irvision.preprocessing.enhancement import apply_clahe
from irvision.preprocessing.normalization import normalize
from irvision.semantic.metrics import compare_maps
from irvision.utils.config import get_device, load_config, resolve_path
from irvision.utils.log import get_logger

log = get_logger(__name__)

# optional stage -> the phase that implements it
@dataclass
class PipelineResult:
    input: np.ndarray                   # (H, W) float32, values as received
    enhanced: np.ndarray                # (H, W) normalized + CLAHE
    colorized: np.ndarray               # (3, H*s, W*s) RGB in [0, 1]; s = scale (1 without super-resolution)
    valid: np.ndarray                   # (H, W) bool, False = no data
    normalized: np.ndarray              # (H, W) normalized, before CLAHE
    scale: int = 1                      # output / input size (2 with super-resolution)
    super_resolved: np.ndarray | None = None        # (H*s, W*s) super-resolved IR: the colorizer's input
    semantic_map: np.ndarray | None = None          # (H, W) land-cover classes of the colorized image
    reference_semantic_map: np.ndarray | None = None  # same for the reference RGB (if given)
    detections: list | None = None                  # objects found in `colorized` (output pixel coords)
    reference_detections: list | None = None        # same for the reference RGB (if given)
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    colorizer: str = ""


def load_default_colorizer(cfg: dict, device: str) -> tuple[object, dict]:
    """U-Net checkpoint -> LUT baseline -> pseudo-colour. Returns ``(colorizer, info)``."""
    inf = cfg["inference"]
    ckpt = resolve_path(inf["checkpoint"])
    if ckpt.exists():
        from irvision.models.colorizer import UNetColorizer

        col = UNetColorizer.from_checkpoint(ckpt, device, tile_size=inf["tile_size"], tile_overlap=inf["tile_overlap"])
        return col, {"name": "unet", "checkpoint": str(ckpt), "epoch": col.meta.get("epoch"),
                     "val_metrics": col.meta.get("val_metrics"), "fallback": False}

    lut_path = cfg["paths"]["models_dir"] / "baseline_lut.npy"
    if lut_path.exists():
        log.warning("No U-Net checkpoint at %s; falling back to the LUT baseline", ckpt)
        return LUTColorizer.load(lut_path), {"name": "lut", "checkpoint": str(lut_path), "fallback": True,
                                             "reason": f"U-Net checkpoint not found: {ckpt}"}

    log.warning("No U-Net checkpoint or LUT baseline found; falling back to pseudo-colour")
    return ColormapColorizer(cfg["baseline"]["colormap"]), {
        "name": "colormap", "fallback": True, "reason": "no trained model or fitted baseline found"}


def load_default_segmenter(cfg: dict, device: str) -> tuple[object | None, dict]:
    """Land-cover segmenter if segmentation is enabled and its checkpoint exists."""
    if not cfg.get("optional", {}).get("segmentation", {}).get("enabled", False):
        return None, {"loaded": False, "reason": "disabled in config (optional.segmentation.enabled)"}
    ckpt = resolve_path(cfg["semantic"]["checkpoint"])
    if not ckpt.exists():
        log.warning("Segmentation enabled but no checkpoint at %s; semantic map will be skipped", ckpt)
        return None, {"loaded": False, "reason": f"segmenter checkpoint not found: {ckpt} (run scripts/train_segmenter.py)"}
    try:
        from irvision.semantic.segmenter import LandCoverSegmenter

        seg = LandCoverSegmenter.from_checkpoint(ckpt, device)
    except Exception as exc:  # optional component: never break the core pipeline (PLAN §18)
        log.exception("Could not load segmenter")
        return None, {"loaded": False, "reason": f"segmenter failed to load: {exc}"}
    return seg, {"loaded": True, "checkpoint": str(ckpt), "val": seg.meta.get("val", {}).get("miou")}


class IRVisionPipeline:
    def __init__(self, cfg: dict | None = None, colorizer=None, device: str | None = None, segmenter=None):
        self.cfg = cfg or load_config()
        self.device = device or get_device(self.cfg)
        if colorizer is None:
            self.colorizer, self.colorizer_info = load_default_colorizer(self.cfg, self.device)
        else:
            self.colorizer, self.colorizer_info = colorizer, {"name": colorizer.name, "fallback": False}
        if segmenter is not None:
            self.segmenter, self.segmenter_info = segmenter, {"loaded": True}
        else:
            self.segmenter, self.segmenter_info = load_default_segmenter(self.cfg, self.device)
        self._optional_models: dict[str, tuple[object | None, str]] = {}   # stage -> (model, failure reason)

    # -- public API ---------------------------------------------------------
    def process_image(
        self,
        image: np.ndarray,
        reference_rgb: np.ndarray | None = None,
        nodata_mask: np.ndarray | None = None,
        super_resolution: bool | None = None,
        detection: bool | None = None,
    ) -> PipelineResult:
        """Run the full pipeline on one IR image.

        ``image``: (H, W) any numeric dtype/scale (Kelvin, DN, uint8, ...), or
        (H, W, 3) which is converted to grayscale with a warning.
        ``reference_rgb``: optional ground truth, (3, H, W) or (H, W, 3), float
        in [0, 1] or uint8. Enables PSNR/SSIM, semantic and detection consistency.
        ``nodata_mask``: optional (H, W) bool, True = no data (e.g. from ``io.read_image``).
        ``super_resolution`` / ``detection``: override the config switches for this call.
        """
        warnings: list[str] = []
        times: dict[str, float] = {}
        pre = self.cfg["preprocessing"]

        with _timer(times, "validation"):
            ir, valid = self._validate(image, nodata_mask, warnings)
        with _timer(times, "normalization"):
            normalized, norm_stats = normalize(ir, pre["ir_normalization"], valid)
        with _timer(times, "enhancement"):
            enhanced = apply_clahe(normalized, pre["clahe"]["clip_limit"], pre["clahe"]["tile_grid_size"], valid)

        # [optional] super-resolution of the IR before colorization (PLAN order)
        model_input, out_valid, scale, super_resolved = enhanced, valid, 1, None
        if self._use("super_resolution", super_resolution):
            sr = self._optional_model("super_resolution", warnings)
            if sr is not None:
                try:
                    with _timer(times, "super_resolution"):
                        super_resolved = sr.upscale(enhanced)
                    model_input, scale = super_resolved, sr.scale
                    out_valid = cv2.resize(valid.astype(np.uint8), super_resolved.shape[::-1],
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                except Exception as exc:  # optional stage must never break the core result
                    log.exception("Super-resolution failed")
                    warnings.append(f"Super-resolution failed and was skipped: {exc}")

        with _timer(times, "colorization"):
            colorized = np.clip(self.colorizer.colorize(model_input), 0, 1).astype(np.float32)
            colorized[:, ~out_valid] = 0.0
        # metrics and segmentation work at the input resolution (the segmenter is trained at 30 m)
        native = colorized if scale == 1 else _resize_rgb(colorized, ir.shape)

        semantic_map = None
        if self._stage_enabled("segmentation"):
            if self.segmenter is None:
                warnings.append(f"Semantic segmentation skipped: {self.segmenter_info['reason']}")
            else:
                try:
                    with _timer(times, "segmentation"):
                        semantic_map = self.segmenter.predict(native, valid)
                except Exception as exc:
                    log.exception("Segmentation failed")
                    warnings.append(f"Semantic segmentation failed and was skipped: {exc}")

        detections, detector = None, None
        if self._use("detection", detection):
            detector = self._optional_model("detection", warnings)
            if detector is not None:
                try:
                    with _timer(times, "detection"):
                        detections = detector.detect(colorized)
                except Exception as exc:
                    log.exception("Detection failed")
                    warnings.append(f"Object detection failed and was skipped: {exc}")
                    detector = None

        metrics: dict = {"input_shape": list(ir.shape), "output_shape": list(colorized.shape[1:]),
                         "valid_fraction": float(valid.mean()), "normalization": norm_stats}
        reference_semantic_map = reference_detections = None
        ref = self._prepare_reference(reference_rgb, native.shape, warnings) if reference_rgb is not None else None
        if ref is not None:
            mask = valid & np.all(np.isfinite(ref), axis=0)
            ref = np.nan_to_num(ref)
            with _timer(times, "metrics"):
                metrics["psnr"] = psnr(native, ref, mask)
                metrics["ssim"] = ssim(native, ref, mask)
            if semantic_map is not None:
                # semantic validation: does the colorized image mean the same as the real one?
                try:
                    with _timer(times, "semantic_validation"):
                        reference_semantic_map = self.segmenter.predict(ref, mask)
                        metrics["semantic"] = compare_maps(semantic_map, reference_semantic_map)
                except Exception as exc:
                    log.exception("Semantic validation failed")
                    warnings.append(f"Semantic validation failed and was skipped: {exc}")
            if detections is not None and detector is not None:
                # detection consistency: same objects found in the colorized and the true image?
                try:
                    with _timer(times, "detection_validation"):
                        ref_out = ref if scale == 1 else _resize_rgb(ref, colorized.shape[1:], cv2.INTER_CUBIC)
                        reference_detections = detector.detect(ref_out)
                        metrics["detection"] = compare_detections(
                            detections, reference_detections, self.cfg["optional"]["detection"]["match_iou"])
                except Exception as exc:
                    log.exception("Detection validation failed")
                    warnings.append(f"Detection validation failed and was skipped: {exc}")
        times["total"] = sum(times.values())
        metrics["time_ms"] = times

        if self.colorizer_info.get("fallback"):
            warnings.append(f"Using fallback colorizer '{self.colorizer_info['name']}': {self.colorizer_info['reason']}")
        return PipelineResult(
            input=ir, enhanced=enhanced, colorized=colorized, valid=valid, normalized=normalized, scale=scale,
            super_resolved=super_resolved, semantic_map=semantic_map, reference_semantic_map=reference_semantic_map,
            detections=detections, reference_detections=reference_detections,
            metrics=metrics, warnings=warnings, colorizer=self.colorizer_info["name"],
        )

    # -- optional models (loaded on first use, never fatal) ----------------------
    def _use(self, stage: str, override: bool | None) -> bool:
        return self._stage_enabled(stage) if override is None else bool(override)

    def _optional_model(self, stage: str, warnings: list[str]):
        """Load the super-resolution or detection model once. On failure: warn and return None."""
        if stage in self._optional_models:
            model, reason = self._optional_models[stage]
        else:
            ocfg = self.cfg["optional"][stage]
            try:
                if stage == "super_resolution":
                    from irvision.models.super_resolution import SuperResolver

                    model = SuperResolver.load(ocfg["model"], ocfg["scale"], self.device)
                else:
                    from irvision.detection.detector import AerialDetector

                    model = AerialDetector.load(
                        resolve_path(ocfg["model_path"]), self.device, confidence=ocfg["confidence"],
                        image_size=ocfg["image_size"], brightness_gamma=ocfg["brightness_gamma"],
                        tile_size=ocfg["tile_size"])
                reason = ""
            except ModuleNotFoundError as exc:
                log.warning("Could not load %s model: %s", stage, exc)
                model, reason = None, (f"{exc}. Install requirements.txt, and run with the project's "
                                       f"Python: .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py")
            except Exception as exc:  # optional component: never break the core pipeline (PLAN §18)
                log.exception("Could not load %s model", stage)
                model, reason = None, str(exc)
            self._optional_models[stage] = (model, reason)
        if model is None:
            warnings.append(f"{stage.replace('_', ' ').capitalize()} skipped: {reason}")
        return model

    # -- stages ---------------------------------------------------------------
    def _validate(self, image, nodata_mask, warnings) -> tuple[np.ndarray, np.ndarray]:
        inf = self.cfg["inference"]
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        arr, warning = to_single_channel(arr)
        if warning:
            warnings.append(warning)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D IR image, got shape {arr.shape}")
        h, w = arr.shape
        if min(h, w) < inf["min_size"] or max(h, w) > inf["max_size"]:
            raise ValueError(f"Image size {w}x{h} outside the allowed range {inf['min_size']}..{inf['max_size']} px per side")

        ir = arr.astype(np.float32)
        valid = np.isfinite(ir)
        if nodata_mask is not None:
            if nodata_mask.shape != ir.shape:
                raise ValueError(f"nodata_mask shape {nodata_mask.shape} != image shape {ir.shape}")
            valid &= ~nodata_mask
        if valid.sum() < 16:
            raise ValueError("Image has no valid pixels")
        if float(np.ptp(ir[valid])) == 0:
            warnings.append("Image is constant (no contrast); the result will be uniform.")
        return np.where(valid, ir, 0.0).astype(np.float32), valid

    @staticmethod
    def _prepare_reference(reference_rgb, shape, warnings) -> np.ndarray | None:
        """Reference RGB as (3, H, W) float in [0, 1], or None (with a warning) if unusable."""
        ref = np.asarray(reference_rgb)
        if ref.ndim == 3 and ref.shape[-1] in (3, 4) and ref.shape[0] not in (3, 4):
            ref = np.moveaxis(ref[..., :3], -1, 0)
        ref = ref.astype(np.float32) / 255.0 if np.issubdtype(ref.dtype, np.integer) else ref.astype(np.float32)
        if ref.shape != shape:
            warnings.append(f"Reference RGB shape {ref.shape} does not match output {shape}; metrics skipped.")
            return None
        return ref

    def _stage_enabled(self, stage: str) -> bool:
        return bool(self.cfg.get("optional", {}).get(stage, {}).get("enabled", False))


def _resize_rgb(rgb: np.ndarray, shape: tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """(3, h, w) -> (3, *shape)."""
    out = cv2.resize(np.moveaxis(rgb, 0, -1), (shape[1], shape[0]), interpolation=interpolation)
    return np.clip(np.moveaxis(out, -1, 0), 0, 1).astype(np.float32)


@contextmanager
def _timer(times: dict, key: str):
    start = time.perf_counter()
    yield
    times[key] = (time.perf_counter() - start) * 1000


@lru_cache(maxsize=1)
def _default_pipeline() -> IRVisionPipeline:
    return IRVisionPipeline()


def process_image(image: np.ndarray, reference_rgb: np.ndarray | None = None,
                  nodata_mask: np.ndarray | None = None) -> PipelineResult:
    """Convenience wrapper around a cached default ``IRVisionPipeline``."""
    return _default_pipeline().process_image(image, reference_rgb, nodata_mask)


def load_example_scene(scene_dir: str | Path, cfg: dict | None = None):
    """Load a Landsat scene as pipeline input + ground truth for demos/evaluation.

    Returns ``(ir_kelvin (H,W) with NaN = invalid, reference_rgb (3,H,W) in [0,1], valid (H,W))``.
    """
    from irvision.preprocessing.landsat import load_scene_from_config

    cfg = cfg or load_config()
    scene = load_scene_from_config(scene_dir, cfg)
    rgb, _ = normalize(scene.rgb_reflectance, cfg["preprocessing"]["rgb_normalization"], scene.valid)
    rgb[:, ~scene.valid] = np.nan
    return scene.ir_kelvin, rgb, scene.valid

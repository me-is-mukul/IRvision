# IRVision: Architecture

How the code is organized, how data flows through it, and where to change what.
Usage, results and decisions are in [README.md](README.md).

---

## 1. The big picture

```text
OFFLINE (scripts/)
  Planetary Computer ─► download_landsat, download_landcover ─► prepare_dataset ─► data/{train,val,test}/*.npz
                                                                                       │
     evaluate_baseline ─► LUT + mean-colour baselines  ◄───────────────────────────────┤
     train             ─► U-Net checkpoint (L1 + SSIM [+ VGG19 perceptual])  ◄──────────┤
     train_segmenter   ─► land-cover segmenter checkpoint  ◄───────────────────────────┤
     evaluate_model / evaluate_semantic / evaluate_advanced ─► outputs/results/*.json ◄┘
  download_pretrained ─► EDSR (super-resolution), YOLOv8-OBB (detection), VGG19 (loss)

ONLINE (irvision.inference.pipeline.process_image)
  IR image ─► validate ─► normalize ─► CLAHE ─► [super-resolution] ─► colorize ─► segment ─► [detect] ─► metrics
  (+ optional                                        EDSR x2          U-Net      DeepLabV3   YOLOv8-OBB
   reference)                                                                       PSNR / SSIM, semantic
                                                                                    and detection consistency
                         ▲
              app/streamlit_app.py (UI only), evaluation scripts, tests
```

Three rules hold the design together:

1. **One inference entry point.** The app, the evaluation scripts and the tests all go through
   `process_image` / `IRVisionPipeline`, so they can never disagree.
2. **One interface per role.** Every colorizer has `colorize(ir) -> rgb`. Every metric function
   takes `(pred, target, valid)`. New models plug in without new evaluation code.
3. **One config.** `config/config.yaml` holds every parameter. Code reads it via `load_config()`.

---

## 2. Package layout

```text
irvision/
├── preprocessing/        raw rasters → clean, aligned, normalized arrays
│   ├── landsat.py        find band files, load a scene onto one grid, convert to K / reflectance, mask
│   ├── alignment.py      GridSpec, read_on_grid (reproject/resample), verify_alignment (phase corr.)
│   ├── masks.py          QA_PIXEL bit decoding → invalid mask
│   ├── normalization.py  percentile / range scaling to [0, 1] (returns the stats used)
│   ├── enhancement.py    16-bit CLAHE with no-data handling
│   └── patches.py        patch windows, spatial stripe split, Patch, save_patch
├── models/
│   ├── baseline.py       MeanColor / Gray / Colormap / LUT colorizers (no learning)
│   ├── unet.py           U-Net network, build_unet, load_checkpoint
│   ├── colorizer.py      UNetColorizer: U-Net behind the colorizer interface (pad + tile)
│   └── super_resolution.py  SuperResolver: pre-trained EDSR x2 (tiled, scale-aware)
├── training/
│   ├── dataset.py        PatchDataset (stacks input channels, RAM cache)
│   ├── augment.py        RandomFlipRotate (same transform for input, target, mask)
│   ├── losses.py         masked L1, masked SSIM, VGG19 perceptual loss, ColorizationLoss
│   └── trainer.py        train(): loop, validation, checkpointing, history/curves
├── semantic/             semantic validation
│   ├── landcover.py      5 classes, WorldCover → class mapping, palette
│   ├── segmenter.py      DeepLabV3-MobileNetV3, LandCoverSegmenter.predict
│   ├── metrics.py        confusion matrix → IoU / mIoU / Dice / agreement
│   └── train.py          segmenter dataset + training loop
├── detection/            object detection (optional stage)
│   ├── detector.py       AerialDetector: YOLOv8n-OBB (DOTA), tiled 4x-upscaled inference
│   └── metrics.py        box IoU, greedy matching, precision / recall / F1
├── evaluation/
│   ├── metrics.py        masked PSNR, masked SSIM, time_function
│   └── evaluate.py       evaluate_colorizer (any colorizer, per-patch table), iter_patches
├── inference/
│   ├── pipeline.py       IRVisionPipeline, process_image, PipelineResult, fallbacks
│   └── io.py             read_image (GeoTIFF/PNG/JPEG + no-data), to_png_bytes
└── utils/
    ├── config.py         load_config (absolute paths), get_device (warns on CPU)
    ├── log.py            get_logger
    ├── tiling.py         tiled_apply: any-size (and up-scaling) inference with feathered blending
    └── visualization.py  check figures (scene overview, patch grid, comparison grid)

scripts/                  thin CLIs: parse args → call irvision → write outputs
app/streamlit_app.py      dashboard (no processing logic)
config/config.yaml        all parameters
tests/                    pytest, synthetic data only
demo/                     ready-to-upload demo images (scripts/export_demo.py)
data/, outputs/           generated (not versioned)
```

**Dependency direction** (arrows = "may import"): `utils` ← `preprocessing` ← `models` / `semantic`
← `training` / `evaluation` ← `inference` ← `scripts` / `app`. Lower layers never import upper ones.

---

## 3. Offline data flow

```text
scripts/download_landsat.py
   STAC search (least cloud) → crop window moved inside the valid footprint
   → windowed COG reads → data/raw/<scene>/<scene>_{SR_B4,SR_B3,SR_B2,ST_B10,QA_PIXEL}.TIF + metadata.json

scripts/download_landcover.py
   WorldCover tiles → reproject (mode) onto the scene grid → 5 classes → <scene>_LANDCOVER.TIF

scripts/prepare_dataset.py            (per scene)
   landsat.load_scene        all bands on the red-band grid, K / reflectance, QA mask
   normalization.normalize   IR percentile stretch, RGB fixed range, IR absolute range
   alignment.verify_alignment   shift must be ≤ 1 px AND an injected shift must be recovered
   enhancement.apply_clahe
   patches.extract_patches   stripe split (or all-test if held out), invalid-fraction filter
   → data/{train,val,test}/*.npz, data/processed/{<scene>.npz, <scene>_report.json, manifest.csv}
   → outputs/dataset/<scene>_{overview,patches}.png
```

Training and evaluation read only the patch files. That keeps them fast (RAM-cached, ~0.25 GB) and
independent of GDAL.

---

## 4. Online flow: `IRVisionPipeline.process_image`

```text
input (H,W any dtype) [+ reference RGB] [+ nodata mask]
 │
 ├─ _validate            to single channel (warn if colour), size 32..8192, valid = finite & ~nodata
 ├─ normalize            config.preprocessing.ir_normalization (percentile)
 ├─ apply_clahe          16-bit, no-data filled then re-masked
 ├─ [super_resolution]   optional (off by default, D-020): EDSR x2 → output is 2H x 2W
 ├─ colorizer.colorize   UNetColorizer: pad to /16 → tiles of 512 (overlap 64) → blend
 │                       with SR: resized back to H x W ("native") for scoring and segmentation
 ├─ segmenter.predict    optional (on); DeepLabV3 on the native colorized RGB, tiles of 1024
 ├─ [detection]          optional (off by default, D-019): YOLOv8-OBB on 256 px tiles → 1024
 └─ metrics (only with a reference)
       psnr / ssim on valid pixels (native resolution)
       segment the reference too → compare_maps → metrics["semantic"]
       detect on the reference too → compare_detections → metrics["detection"]
 │
 ▼
PipelineResult(input, normalized, enhanced, colorized, valid, scale, super_resolved,
               semantic_map, reference_semantic_map, detections, reference_detections,
               metrics{..., time_ms per stage}, warnings, colorizer)
```

**Fallbacks** (built in `IRVisionPipeline.__init__`):

| Component | Chain | Reported in |
|---|---|---|
| Colorizer | U-Net checkpoint → fitted LUT → pseudo-colour | `result.warnings`, `pipeline.colorizer_info` |
| Segmenter | checkpoint → skipped (no semantic map) | `result.warnings`, `pipeline.segmenter_info` |
| Super-resolution / detector | loaded on first use; load failure → stage skipped (remembered) | `result.warnings` |
| Optional stage at runtime | exception → stage skipped, core result kept | `result.warnings` |

`process_image(..., super_resolution=True/False, detection=True/False)` overrides the config
switches per call (the app's sidebar toggles use this).

Only invalid *input* raises (`ValueError`). The app shows that as a message.

---

## 5. Key interfaces

```python
# Colorizer: baselines and U-Net alike (models/baseline.py, models/colorizer.py)
class Colorizer(Protocol):
    name: str
    def colorize(self, ir: np.ndarray) -> np.ndarray: ...    # (H,W) [0,1] -> (3,H,W) float32 [0,1]

# Segmenter (semantic/segmenter.py)
LandCoverSegmenter.predict(rgb: (3,H,W) [0,1], valid: (H,W) bool | None) -> (H,W) uint8   # 255 = ignore

# Metrics: masks are always (H,W) bool, True = count this pixel
psnr(pred, target, valid) -> float          ssim(pred, target, valid) -> float
compare_maps(pred_labels, target_labels, valid) -> {miou, mean_dice, pixel_agreement, iou{}, dice{}}

# Super-resolution (models/super_resolution.py)
SuperResolver.upscale(ir: (H,W) [0,1]) -> (H*s, W*s) float32

# Detection (detection/)
AerialDetector.detect(rgb: (3,H,W) [0,1]) -> [{class_name, confidence, box[x1,y1,x2,y2], polygon[4x2]}]
compare_detections(pred, ref, iou_threshold) -> {tp, fp, fn, n_pred, n_ref, precision, recall, f1}

# Any-size inference (utils/tiling.py)
tiled_apply(fn: (C,h,w)->(K,h*s,w*s), image: (C,H,W), out_channels=K, tile_size, overlap, scale=s) -> (K,H*s,W*s)
```

**Checkpoint contents.** The U-Net checkpoint (`outputs/models/<run>/best.pt`) holds `state_dict`,
`model_cfg`, `inputs`, `preprocessing`, `loss_weights`, `epoch` and `val_metrics`. The segmenter
checkpoint holds `state_dict`, `classes`, `epoch` and `val`. Both are self-describing, so loading
needs no config.

---

## 6. Where to change what

| I want to… | Change | Then run |
|---|---|---|
| add a training city | `config.dataset.download.aois` | download_landsat → download_landcover → prepare_dataset |
| hold out another city | add `holdout: true` to its AOI | prepare_dataset |
| try a new loss | `irvision/training/losses.py` + `config.training.loss_weights` (e.g. `--perceptual 0.1`) | train → evaluate_model → evaluate_semantic |
| switch SR / detection on by default | `config.optional.super_resolution.enabled` / `detection.enabled` | evaluate_advanced |
| use another detector / SR model | `config.optional.detection.model_path` / `super_resolution.model` | evaluate_advanced |
| try a new colorization model | a class with `name` + `colorize()`. Use it in `IRVisionPipeline(colorizer=...)` | evaluate via `evaluate_colorizer` |
| deploy another checkpoint | `config.inference.checkpoint` | restart the app |
| change land-cover classes | `irvision/semantic/landcover.py` | download_landcover → prepare_dataset → train_segmenter |
| add an optional pipeline stage | a hook in `IRVisionPipeline.process_image` + a switch in `config.optional` | add tests in `tests/test_pipeline.py` |
| change the UI | `app/streamlit_app.py` (UI only; call the pipeline) | `pytest tests/test_app.py` |

---

## 7. Generated files (not versioned)

| Path | Written by | Contents |
|---|---|---|
| `data/raw/<scene>/` | download scripts | band GeoTIFFs, land cover, `metadata.json` (provenance, `holdout` flag) |
| `data/processed/` | prepare_dataset | full-scene arrays, per-scene reports, `manifest.csv` |
| `data/{train,val,test}/` | prepare_dataset | 256 px patch `.npz` files |
| `outputs/dataset/` | prepare_dataset | overview and patch check figures |
| `outputs/models/` | evaluate_baseline, train, train_segmenter | LUT, mean colour, checkpoints, histories, curves |
| `outputs/pretrained/` | download_pretrained | YOLOv8n-OBB weights (EDSR and VGG19 live in the HF / torch caches) |
| `outputs/results/` | evaluate_* scripts | metric JSON/CSV and comparison figures (the app's Model report reads these) |

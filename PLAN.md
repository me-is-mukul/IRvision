# IRVision — Implementation Plan

You are the primary developer responsible for building this project.

Your job is to take this plan and implement the project incrementally, keeping the repository runnable at every stage.

Do not attempt to implement everything at once.

---

# 1. Project Goal

Build **IRVision**, an AI-powered system that takes a single-channel infrared/thermal satellite image and produces:

```text
IR Satellite Image
        ↓
IR Enhancement
        ↓
Optional Super Resolution
        ↓
IR → RGB Colorization
        ↓
Semantic Validation
        ↓
Optional Object Detection
        ↓
Metrics + Visualization
```

The final product should be a Streamlit application demonstrating the complete pipeline.

The core objective is:

> Enhance and colorize infrared satellite imagery while preserving structural and semantic information useful for downstream interpretation.

Do NOT claim that the generated RGB image is the physically exact RGB ground truth. It is a learned/plausible RGB reconstruction.

---

# 2. Development Philosophy

Build from the inside out.

Priority:

```text
1. Dataset
2. Preprocessing
3. Baseline
4. Core colorization model
5. Evaluation
6. Semantic validation
7. Super-resolution
8. Object detection
9. Streamlit UI
10. Final polish
```

Never build the UI before the inference pipeline works.

Never add advanced models before the basic pipeline works.

Never allow an optional component to break the core system.

---

# 3. MVP

The minimum working product is:

```text
IR image
   ↓
Normalization
   ↓
CLAHE enhancement
   ↓
U-Net colorization
   ↓
RGB output
   ↓
PSNR + SSIM + inference time
   ↓
Streamlit visualization
```

This MVP must work before any advanced feature is attempted.

---

# 4. Advanced Features

After the MVP works, implement these in order:

### A. Super Resolution

Use a pre-trained model such as:

* ESPCN
* EDSR
* SwinIR

Do not train the SR model from scratch.

### B. Semantic Segmentation

Use a pre-trained model such as:

* SegFormer
* DeepLabV3+

Use it to compare semantic interpretation between generated RGB and ground-truth RGB.

Calculate:

* IoU
* mIoU
* Dice

### C. Object Detection

Use a pre-trained YOLO model.

Compare downstream detection performance where the dataset supports it.

### D. Perceptual Loss

Use a frozen pre-trained VGG19 feature extractor.

Only add this after the basic colorization model works.

---

# 5. Dataset

Primary target:

**Landsat 8/9**

Use:

```text
Band 2 → Blue
Band 3 → Green
Band 4 → Red
Band 10 → Thermal IR
```

Construct:

```text
Input:
Band 10

Target:
Band 4 + Band 3 + Band 2
```

The data pipeline must handle:

* GeoTIFF loading
* CRS
* reprojection
* resampling
* co-registration
* invalid pixels
* cloud filtering where possible
* normalization
* patch extraction

Default patch size:

```text
256 × 256
```

Before training, visually verify that IR and RGB are spatially aligned.

---

# 6. Repository Structure

Create:

```text
irvision/
│
├── app/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── inference/
├── utils/
├── notebooks/
├── scripts/
├── tests/
│
├── requirements.txt
├── README.md
├── PLAN.md
└── CHANGELOG.md
```

Use clean modular Python code.

---

# 7. Technology Stack

Use:

```text
Python
PyTorch
Torchvision
OpenCV
Rasterio
GDAL
NumPy
Pandas
scikit-image
Albumentations
Matplotlib
Streamlit
tqdm
```

Optional:

```text
Ultralytics
segmentation-models-pytorch
```

---

# 8. Models

## Model We Train

Primary model:

```text
U-Net
```

Input:

```text
1 channel IR
```

Output:

```text
3 channel RGB
```

Experiment later with Lab color space.

---

## Pre-trained Models

Do NOT train these from scratch:

```text
ESPCN / EDSR / SwinIR
    → Super Resolution

SegFormer / DeepLabV3+
    → Semantic Segmentation

YOLO
    → Object Detection

VGG19
    → Perceptual Loss
```

All external models must be wrapped behind modular interfaces so they can be enabled/disabled.

---

# 9. Baseline

Before training the neural network, implement:

```text
IR
 ↓
CLAHE
 ↓
Pseudo-color
```

Record:

```text
PSNR
SSIM
Inference time
```

The learned model should be compared against this baseline.

---

# 10. Colorization Model

Start with:

```text
U-Net + L1 loss
```

Then experiment with:

```text
U-Net
+
L1
+
SSIM loss
```

Then optionally:

```text
L1
+
SSIM
+
Perceptual loss
```

Do not start with GANs.

Pix2Pix/CycleGAN may be explored only if the baseline architecture is already stable.

---

# 11. Evaluation

Required:

```text
PSNR
SSIM
Inference Time
```

Optional:

```text
FID
mIoU
Dice
mAP
Precision
Recall
```

Never fabricate metrics.

All displayed metrics must come from actual evaluation runs.

---

# 12. Inference Pipeline

Create a single high-level API:

```python
process_image(image)
```

It should execute:

```text
Input
 ↓
Validation
 ↓
Normalization
 ↓
CLAHE
 ↓
Optional SR
 ↓
Colorization
 ↓
Optional Semantic Segmentation
 ↓
Optional Object Detection
 ↓
Metrics
```

Return a structured result containing:

```text
input
enhanced
colorized
semantic_map
detections
metrics
```

---

# 13. Streamlit Application

Create a clean dashboard.

Required:

```text
Upload IR image
        ↓
Process
        ↓
Original IR
Enhanced IR
Colorized RGB
Semantic Map
Metrics
```

Optional:

```text
Object detections
```

The application must show:

```text
PSNR
SSIM
Inference Time
```

and semantic/detection metrics when available.

Cache model loading so models aren't reloaded on every inference.

---

# 14. Configuration

Do not hard-code training parameters.

Use:

```text
config/config.yaml
```

Configuration should include:

```text
dataset
patch size
batch size
epochs
learning rate
model
loss weights
device
paths
optional model switches
```

---

# 15. Testing

Implement tests for:

```text
Dataset loader
Normalization
CLAHE
Patch generation
U-Net forward pass
Metrics
Inference pipeline
```

At minimum, verify:

```python
dummy_input = torch.randn(1, 1, 256, 256)
output = model(dummy_input)

assert output.shape == (1, 3, 256, 256)
```

---

# 16. Development Phases

## Phase 1 — Setup

Create repository, dependencies, configuration and logging.

## Phase 2 — Dataset

Load one real Landsat scene.

Extract IR and RGB.

Verify alignment.

Generate patches.

## Phase 3 — Preprocessing

Implement normalization and CLAHE.

## Phase 4 — Baseline

Implement pseudo-color baseline and metrics.

## Phase 5 — Core AI

Implement U-Net and training pipeline.

## Phase 6 — Evaluation

Implement PSNR, SSIM and inference timing.

## Phase 7 — Semantic Layer

Add pre-trained segmentation model and semantic consistency.

## Phase 8 — Advanced Enhancement

Add super-resolution.

## Phase 9 — Downstream Task

Add YOLO where appropriate.

## Phase 10 — Product

Build Streamlit dashboard.

## Phase 11 — Finalization

Testing, optimization, documentation and demo preparation.

---

# 17. Hackathon Priority

Use this priority system.

### P0 — Mandatory

```text
Dataset
Preprocessing
U-Net
IR → RGB
```

### P1 — Mandatory

```text
PSNR
SSIM
Inference time
Streamlit
```

### P2 — High Value

```text
Semantic segmentation
Semantic consistency
```

### P3 — Optional

```text
Super-resolution
YOLO
FID
```

### P4 — Stretch

```text
GAN
Advanced attention
Foundation models
Multimodal analysis
```

If time is running out, stop adding features and polish P0/P1.

---

# 18. Fallback Strategy

The project must remain functional if advanced components fail.

If SR fails:

```text
IR → CLAHE → U-Net → RGB
```

If segmentation fails:

```text
IR → CLAHE → U-Net → RGB → PSNR/SSIM
```

If YOLO fails:

Continue without detection.

If the Landsat training pipeline cannot be completed in time:

Use a suitable paired thermal/RGB dataset to validate the colorization architecture, while keeping the Landsat preprocessing/evaluation pipeline as the target satellite-domain component.

Clearly document which dataset was used for which purpose.

---

# 19. Documentation Requirement

The repository must contain:

```text
README.md
PLAN.md
CHANGELOG.md
```

The README should explain:

* Project
* Problem
* Architecture
* Installation
* Dataset
* Training
* Inference
* Evaluation
* Usage
* Results
* Limitations
* Future work

---

# 20. Definition of Done

The MVP is complete when:

```text
Real IR input
      ↓
Preprocessing
      ↓
U-Net
      ↓
RGB output
      ↓
PSNR / SSIM
      ↓
Streamlit
```

works end-to-end on an unseen image.

The advanced version is complete when:

```text
IR
 ↓
Enhancement
 ↓
SR
 ↓
Colorization
 ↓
Semantic Validation
 ↓
Object Detection
 ↓
Metrics
 ↓
Dashboard
```

works end-to-end.

---

# 21. IMPORTANT DEVELOPMENT RULES FOR CLAUDE

1. Do not implement the entire project in one step.

2. Work phase-by-phase.

3. After completing each phase, run tests or a real verification.

4. Never assume a component works without testing it.

5. Never fabricate results.

6. Never silently ignore errors.

7. Keep the application runnable after every major change.

8. Prefer simple working implementations over unnecessarily complex architectures.

9. Keep external/pre-trained models modular.

10. Do not download huge datasets unless necessary.

11. Optimize for a 24–48 hour hackathon.

12. If a feature is consuming too much time, implement the fallback and move on.

13. Maintain clean Git commits.

14. Update documentation as development progresses.

15. Before making architectural changes, check whether they conflict with this plan.

---

# 22. First Task

Start ONLY with Phase 1 and Phase 2.

Your first objective is:

```text
Repository
    ↓
Environment
    ↓
Landsat loader
    ↓
IR + RGB extraction
    ↓
Alignment verification
    ↓
Normalization
    ↓
CLAHE
    ↓
256×256 patches
    ↓
Visualization
```

Do NOT implement the colorization model yet.

Once this pipeline works, continue to the next phase.

---

# 23. Completion Protocol

At the end of every development phase:

1. Run relevant tests.
2. Verify the output manually where applicable.
3. Update `CHANGELOG.md`.
4. Update `README.md` if functionality changed.
5. Record important decisions.
6. Record known issues.
7. Commit the changes.
8. Only then proceed to the next phase.

```

### Extra command for Claude

After giving Claude `PLAN.md`, give it this instruction:

:::writing{variant="chat_message" id="80641" title="Claude Documentation Command"}
From this point onward, treat `PLAN.md` as the source of truth for the project.

Implement the project phase-by-phase and do not skip verification.

IMPORTANT: Document everything you do.

For every meaningful action, maintain `CHANGELOG.md` with:

- Date/time
- Phase
- What was implemented
- Files created
- Files modified
- Dependencies added
- Commands executed
- Tests executed
- Test results
- Model/dataset decisions
- Problems encountered
- How each problem was solved
- Current project status
- Next planned step

Also update `README.md` whenever the project's functionality, architecture, setup process, usage, or evaluation procedure changes.

After every major phase, provide me with a concise progress report containing:

1. Completed
2. Verified
3. Failed/issues
4. Decisions made
5. Files changed
6. Next step

Do not claim something is complete unless you have actually tested it.

Do not fabricate metrics, model performance, dataset results, or successful execution.

If something fails, document the failure and fix it or use the fallback strategy defined in `PLAN.md`.

Keep the repository runnable at all times.

Before starting a new phase, inspect the current implementation and `CHANGELOG.md` so that you continue from the actual project state rather than assuming previous work succeeded.

Start now with Phase 1 and Phase 2 only. Do not proceed to model training until the dataset loading, alignment, preprocessing, and patch-generation pipeline has been successfully verified.
```
